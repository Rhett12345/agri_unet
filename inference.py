"""
inference.py
============
Full-disk inference for precipitation classification.

For each AGRI scene file:
  1. Reads raw AGRI BT + geolocation (lat, lon, VZA, SZA).
  2. Slices into overlapping patches (cfg.PATCH_SIZE, cfg.PATCH_OVERLAP).
  3. Runs model in batch mode.
  4. Reassembles predictions via Gaussian-weighted averaging.
  5. Saves outputs as a compressed .npz file.

Output .npz keys:
    latitude, longitude
    precip_class  : integer class map (H, W)   0-3
    precip_prob   : (H, W, 4)     softmax probabilities
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

import config as cfg
from fusion_io import read_agri_scene
from dataset import NormStats
from model import build_model

log = logging.getLogger(__name__)


def _build_region_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    构建区域 mask（True=在训练区域内）。
    读取 fusion_config 中的 REGION_LAT/LON 参数。
    """
    try:
        import fusion_config as fc
        lat_min = float(getattr(fc, "REGION_LAT_MIN", -90))
        lat_max = float(getattr(fc, "REGION_LAT_MAX", 90))
        lon_min = float(getattr(fc, "REGION_LON_MIN", -180))
        lon_max = float(getattr(fc, "REGION_LON_MAX", 180))
    except ImportError:
        return np.ones(lat.shape, dtype=bool)

    # If any bound is at the global extreme, treat as no region filter
    if lat_min <= -89 and lat_max >= 89 and lon_min <= -179 and lon_max >= 179:
        return np.ones(lat.shape, dtype=bool)

    log.info("Inference region: lat=[%.1f, %.1f] lon=[%.1f, %.1f]", lat_min, lat_max, lon_min, lon_max)
    mask = (np.isfinite(lat) & np.isfinite(lon)
            & (lat >= lat_min) & (lat <= lat_max)
            & (lon >= lon_min) & (lon <= lon_max))
    return mask


def _gaussian_weight_map(ph: int, pw: int) -> np.ndarray:
    sigma_h, sigma_w = ph / 4.0, pw / 4.0
    yy = np.arange(ph) - ph / 2.0
    xx = np.arange(pw) - pw / 2.0
    xx, yy = np.meshgrid(xx, yy)
    w = np.exp(-(xx ** 2 / (2 * sigma_w ** 2) + yy ** 2 / (2 * sigma_h ** 2)))
    return w.astype(np.float32)


def _extract_patches(arr: np.ndarray, ph: int, pw: int, stride_h: int, stride_w: int):
    H, W = arr.shape[:2]
    for i in range(0, H - ph + 1, stride_h):
        for j in range(0, W - pw + 1, stride_w):
            yield arr[i:i + ph, j:j + pw, :], i, j


def _stitch(pred_sum: np.ndarray, weight_sum: np.ndarray) -> np.ndarray:
    wt = weight_sum[..., np.newaxis] if pred_sum.ndim == 3 else weight_sum
    return np.where(wt > 0, pred_sum / np.maximum(wt, 1e-8), np.nan)


def run_inference(agri_file: Path,
                  stats: NormStats,
                  checkpoint: Optional[Path] = None,
                  out_dir: Optional[Path] = None,
                  batch_size: int = 64) -> Path:
    """Produce full-disk precipitation classification for one AGRI scene."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = checkpoint or cfg.CHECKPOINT_BEST
    out_dir = out_dir or cfg.RETRIEVAL_DIR

    model = build_model().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    log.info("Loaded model from %s", checkpoint)

    # ── Read AGRI ──
    if agri_file.suffix.lower() in (".npz",):
        d = np.load(agri_file, allow_pickle=True)
        BT  = d["BT_converted"] if "BT_converted" in d else d["BT"]
        lat = d["latitude"]; lon = d["longitude"]
    else:
        agri = read_agri_scene(agri_file)
        if agri is None:
            raise RuntimeError(f"Failed to read {agri_file}")
        BT  = agri["BT"]
        lat = agri["lat"]
        lon = agri["lon"]

    H, W = BT.shape[:2]

    # ── Normalise BT ──
    BT_norm = (BT - stats.agri_mean) / (stats.agri_std + 1e-8)

    # ── Geo: only lat, lon (2 channels) ──
    geo_full = np.stack([lat / 90.0, lon / 180.0], axis=-1).astype(np.float32)
    geo_full = np.nan_to_num(geo_full, nan=0.0)

    # ── Region mask ──
    region_mask = _build_region_mask(lat, lon)
    region_active = not bool(np.all(region_mask))

    # ── Patch geometry ──
    ph, pw   = cfg.PATCH_SIZE
    overlap_h, overlap_w = cfg.PATCH_OVERLAP
    stride_h = max(1, ph - overlap_h)
    stride_w = max(1, pw - overlap_w)
    wmap     = _gaussian_weight_map(ph, pw)   # (ph, pw)

    # ── Accumulation buffers ──
    C = cfg.NUM_CLASSES
    prob_sum    = np.zeros((H, W, C), dtype=np.float32)
    weight_sum  = np.zeros((H, W), dtype=np.float32)

    x_buf, positions_buf = [], []

    def _flush():
        if not x_buf:
            return
        x = torch.from_numpy(np.stack(x_buf, axis=0)).to(device)  # (B, C+2, ph, pw)

        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                logits = model(x)                                    # (B, 4)

        probs = F.softmax(logits, dim=1).cpu().numpy()               # (B, 4)

        for b, (si, sj) in enumerate(positions_buf):
            # Broadcast patch-level prediction to whole patch with Gaussian weight
            for c in range(C):
                prob_sum[si:si+ph, sj:sj+pw, c] += probs[b, c] * wmap
            weight_sum[si:si+ph, sj:sj+pw] += wmap

        x_buf.clear()
        positions_buf.clear()

    for bt_patch, si, sj in _extract_patches(BT_norm, ph, pw, stride_h, stride_w):
        nan_ratio = np.isnan(bt_patch).mean()
        if nan_ratio > 0.8:
            continue
        if region_active:
            if not region_mask[si:si+ph, sj:sj+pw].any():
                continue
        geo_patch = geo_full[si:si+ph, sj:sj+pw, :]
        bt_filled = np.where(np.isnan(bt_patch), 0.0, bt_patch)
        # Concat BT + geo → (ph, pw, C+2) → (C+2, ph, pw)
        x_patch = np.concatenate([bt_filled, geo_patch], axis=-1).transpose(2, 0, 1)
        x_patch = np.ascontiguousarray(x_patch)
        x_buf.append(x_patch)
        positions_buf.append((si, sj))
        if len(x_buf) >= batch_size:
            _flush()
    _flush()

    # ── Stitch ──
    prob_map = _stitch(prob_sum, weight_sum)   # (H, W, C)

    if region_active:
        prob_map[~region_mask] = np.nan

    class_map = np.full(prob_map.shape[:2], -1, dtype=np.int16)
    valid_mask = np.isfinite(prob_map).any(axis=-1)
    if valid_mask.any():
        class_map[valid_mask] = np.nanargmax(prob_map[valid_mask], axis=-1).astype(np.int16)

    # ── Save ──
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = agri_file.stem
    out_path = out_dir / f"{stem}_precip.npz"

    np.savez_compressed(
        out_path,
        latitude=lat,
        longitude=lon,
        precip_class=class_map,
        precip_prob=prob_map.astype(np.float32),
    )
    log.info("Saved retrieval → %s", out_path)
    return out_path


def main():
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Full-disk AGRI precipitation classification")
    parser.add_argument("--agri_file", default=None)
    parser.add_argument("--agri_dir",  default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out_dir",    default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    stats = NormStats.load(cfg.STATS_FILE)
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    out_d = Path(args.out_dir) if args.out_dir else None

    if args.agri_dir:
        agri_dir = Path(args.agri_dir)
        agri_files = sorted(
            list(agri_dir.glob("*.HDF")) + list(agri_dir.glob("*.hdf"))
        )
        agri_files = [f for f in agri_files if "_FDI-_" in f.name]
        log.info("Batch inference on %d files", len(agri_files))
        for f in agri_files:
            try:
                run_inference(f, stats, ckpt, out_d)
            except Exception as exc:
                log.error("Failed for %s: %s", f.name, exc)
    elif args.agri_file:
        run_inference(Path(args.agri_file), stats=stats, checkpoint=ckpt, out_dir=out_d)
    else:
        parser.error("Either --agri_file or --agri_dir required")


if __name__ == "__main__":
    main()
