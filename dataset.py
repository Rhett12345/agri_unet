"""
dataset.py
==========
PyTorch Dataset for AGRI → GPM precipitation classification.

Key design decisions
--------------------
- **Lazy loading**: __init__ builds a patch index. Each __getitem__ opens the
  HDF5 file, reads the required patch, and closes immediately. Memory is O(1)
  regardless of dataset size.
- **NormStats** is pre-computed once and loaded from disk.
- Each sample: X=(7,11,11) AGRI patch, Y=scalar label 0-3
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config as cfg

log = logging.getLogger(__name__)


def _split_dates_for_mode(mode: str):
    date_map = {
        "train": getattr(cfg, "TRAIN_DATES", []),
        "val": getattr(cfg, "VAL_DATES", []),
        "test": getattr(cfg, "TEST_DATES", []),
    }
    return set(date_map.get(mode, []) or [])


def _filter_h5_files_by_dates(h5_files: List[Path], mode: str) -> List[Path]:
    dates = _split_dates_for_mode(mode)
    if not dates:
        return h5_files
    filtered = [p for p in h5_files if any(part in dates for part in p.parts)]
    log.info("Using %d/%d %s files after date filter",
             len(filtered), len(h5_files), mode)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation statistics I/O
# ─────────────────────────────────────────────────────────────────────────────

class NormStats:
    """Container for per-channel AGRI BT mean/std."""

    def __init__(self, agri_mean: np.ndarray, agri_std: np.ndarray):
        self.agri_mean = agri_mean.astype(np.float32)
        self.agri_std  = agri_std.astype(np.float32)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, agri_mean=self.agri_mean, agri_std=self.agri_std)
        log.info("Saved normalisation stats → %s", path)

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        d = np.load(path)
        return cls(d["agri_mean"], d["agri_std"])


# ─────────────────────────────────────────────────────────────────────────────
# Stats: per-file worker
# ─────────────────────────────────────────────────────────────────────────────

def _stats_worker(h5_path: str) -> Optional[dict]:
    import h5py
    import numpy as np
    try:
        with h5py.File(h5_path, "r") as f:
            if "Samples" not in f or "agri" not in f["Samples"]:
                return None
            bt = f["Samples/agri"][()].astype(np.float64)  # (N, C, H, W)
            n_agri = bt.shape[1]
            if n_agri != cfg.AGRI_CHANNELS:
                return None
            flat_bt = bt.transpose(0, 2, 3, 1).reshape(-1, n_agri)
    except Exception:
        return None

    valid = np.isfinite(flat_bt).all(axis=1) & (flat_bt != 0.0).any(axis=1)
    n_bt = int(valid.sum())
    if n_bt == 0:
        return None
    bt_valid = flat_bt[valid]
    return {
        "n": n_bt,
        "sum_bt": bt_valid.sum(axis=0),
        "sumsq_bt": (bt_valid ** 2).sum(axis=0),
    }


def compute_and_save_stats(
    paired_dir: Path,
    out_path: Path = cfg.STATS_FILE,
    n_workers: int = min(8, os.cpu_count() or 1),
) -> "NormStats":
    """Compute normalisation statistics from all paired HDF5 files."""
    log.info("Computing normalisation statistics from %s (workers=%d)", paired_dir, n_workers)

    h5_files = _filter_h5_files_by_dates(sorted(paired_dir.rglob("*.h5")), "train")
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {paired_dir}")

    n_agri = cfg.AGRI_CHANNELS
    total_n = 0
    sum_bt   = np.zeros(n_agri, dtype=np.float64)
    sumsq_bt = np.zeros(n_agri, dtype=np.float64)

    paths = [str(p) for p in h5_files]
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_stats_worker, p): p for p in paths}
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            if result is None:
                continue
            total_n += result["n"]
            sum_bt   += result["sum_bt"]
            sumsq_bt += result["sumsq_bt"]
            if done % 10 == 0 or done == len(paths):
                log.info("  %d / %d files processed", done, len(paths))

    if total_n < 2:
        raise RuntimeError(f"Not enough valid pixels: {total_n}")

    agri_mean = (sum_bt / total_n).astype(np.float32)
    agri_var  = (sumsq_bt - (sum_bt ** 2) / total_n) / (total_n - 1)
    agri_std  = np.sqrt(np.maximum(agri_var, 1e-12)).astype(np.float32)

    stats = NormStats(agri_mean=agri_mean, agri_std=agri_std)
    stats.save(out_path)
    log.info("Stats computed across %d files (valid px=%d)", len(h5_files), total_n)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Patch index builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_patch_index(h5_files: List[Path], mode: str) -> List[Tuple[Path, int]]:
    """Scan all HDF5 files and return list of (file_path, sample_idx) tuples."""
    index: List[Tuple[Path, int]] = []
    for h5f in h5_files:
        try:
            with h5py.File(h5f, "r") as f:
                if "Samples" not in f or "agri" not in f["Samples"]:
                    continue
                n = int(f["Samples/agri"].shape[0])
                for s in range(n):
                    index.append((h5f, s))
        except Exception:
            continue
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PrecipDataset(Dataset):
    """
    PyTorch Dataset for AGRI → GPM precipitation classification.

    Each item:
        agri  : FloatTensor (7, H, W) – z-score normalised BT
        geo   : FloatTensor (4, H, W) – [lat/90, lon/180, VZA/90, SZA/90]
        label : LongTensor (H, W) – integer class label 0-3 (broadcast)
    """

    def __init__(self,
                 paired_dir: Path,
                 stats: NormStats,
                 patch_size: Tuple[int, int] = cfg.PATCH_SIZE,
                 mode: str = "train"):
        self.stats      = stats
        self.patch_size = patch_size
        self.mode       = mode
        self.ph, self.pw = patch_size

        h5_files = _filter_h5_files_by_dates(sorted(paired_dir.rglob("*.h5")), mode)
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found in {paired_dir}")

        log.info("Building patch index from %d files (mode=%s)...", len(h5_files), mode)
        self._index = _build_patch_index(h5_files, mode)
        log.info("Dataset ready – %d samples (from %d files, mode=%s)",
                 len(self._index), len(h5_files), mode)

        self._warned_files = set()
        # Per-worker file handle cache: avoid open/close on every __getitem__
        self._fh_cache: dict = {}   # {Path: h5py.File}

    def __len__(self) -> int:
        return len(self._index)

    def _get_fh(self, h5f: Path) -> h5py.File:
        """Return cached file handle, opening if needed."""
        fh = self._fh_cache.get(h5f)
        if fh is None or not fh.id.valid:
            # Evict oldest if cache grew too large
            if len(self._fh_cache) >= 12:
                oldest = next(iter(self._fh_cache))
                try:
                    self._fh_cache[oldest].close()
                except Exception:
                    pass
                del self._fh_cache[oldest]
            fh = h5py.File(h5f, "r")
            self._fh_cache[h5f] = fh
        return fh

    def __getitem__(self, idx: int):
        h5f, s_idx = self._index[idx]
        ph, pw = self.ph, self.pw

        for attempt in range(10):
            try:
                fh = self._get_fh(h5f)
                samples = fh["Samples"]
                agri_patch = samples["agri"][s_idx].astype(np.float32)   # (7, H, W)
                geo_patch  = samples["geo"][s_idx].astype(np.float32)    # (2, H, W): lat, lon
                label_val  = int(samples["label"][s_idx])                 # scalar 0-3
                break
            except Exception:
                try:
                    del self._fh_cache[h5f]
                except Exception:
                    pass
                if attempt == 0 and h5f.name not in self._warned_files:
                    log.warning("Read error at %s [%d]", h5f.name, s_idx)
                    self._warned_files.add(h5f)
                if attempt == 9:
                    log.error("All read retries exhausted, returning zero sample")
                    return (
                        torch.zeros(cfg.AGRI_CHANNELS + cfg.GEO_CHANNELS, ph, pw),
                        torch.tensor(-100, dtype=torch.long),
                    )
                h5f, s_idx = self._index[np.random.randint(0, len(self._index))]

        # ── BT normalisation ──
        agri_norm = (agri_patch - self.stats.agri_mean[:, None, None]) / \
                     (self.stats.agri_std[:, None, None] + 1e-8)
        agri_norm = np.nan_to_num(agri_norm, nan=0.0)

        # ── Geo normalisation: lat/90, lon/180 ──
        geo_norm = np.stack([
            geo_patch[0] / 90.0,
            geo_patch[1] / 180.0,
        ], axis=0).astype(np.float32)
        geo_norm = np.nan_to_num(geo_norm, nan=0.0)

        # ── Train augmentations (applied to BT + geo together) ──
        if self.mode == "train":
            agri_norm = agri_norm + np.random.randn(*agri_norm.shape).astype(np.float32) * 0.02

            if np.random.rand() < 0.5:
                agri_norm = np.flip(agri_norm, axis=2).copy()
                geo_norm  = np.flip(geo_norm,  axis=2).copy()
            if np.random.rand() < 0.5:
                agri_norm = np.flip(agri_norm, axis=1).copy()
                geo_norm  = np.flip(geo_norm,  axis=1).copy()

            k = np.random.randint(0, 4)
            if k:
                agri_norm = np.rot90(agri_norm, k=k, axes=(1, 2)).copy()
                geo_norm  = np.rot90(geo_norm,  k=k, axes=(1, 2)).copy()

        # ── Concat BT + geo → single tensor (C+2, H, W) ──
        x = np.concatenate([agri_norm, geo_norm], axis=0)   # (9, H, W)
        x = torch.from_numpy(x.copy())
        lbl = torch.tensor(label_val, dtype=torch.long)

        return x, lbl


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def build_test_dataloader(stats: NormStats):
    """Return only the test DataLoader."""
    test_ds = PrecipDataset(cfg.PAIRED_TEST_DIR, stats, mode="test")
    return DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                      pin_memory=True, num_workers=cfg.NUM_WORKERS,
                      persistent_workers=True, prefetch_factor=4)
