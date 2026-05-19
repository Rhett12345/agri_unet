"""
tools/validate_agri_l2.py
==========================
Compare model inference (.npz) against official FY4A AGRI L2 CLP/CTH products.

For each matched (inference .npz, L2 CLP .NC, L2 CTH .NC) triplet:
  1. Read model CLP_pred / CTH_pred from .npz
  2. Read official L2 CLP / CTH, remap to 3-class system
  3. Pixel-by-pixel comparison on the shared 2748x2748 grid
  4. Compute OA, per-class accuracy, confusion matrix, CTH RMSE/MAE/Bias/R
  5. Generate publication-quality figures

Usage:
  # Single scene
  python tools/validate_agri_l2.py --npz_dir /path/to/retrieval/ --day 20190503

  # All scenes in a day, save summary CSV
  python tools/validate_agri_l2.py --npz_dir /path/to/retrieval/ --day 20190503 --summary
"""

import argparse, logging, sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import config as cfg
from fusion_io import _extract_timestamp_from_filename, _find_matching_l2_file

# Suppress fontTools subsetting noise when saving SVG/PDF
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# matplotlib publication defaults
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

OUT_DIR = cfg.ROOT / "eval" / "agri_l2_validation"


def _classify_npz(npz_name: str) -> str:
    """从文件名判别数据来源: 'fy4a' / 'fy4b_direct' / 'fy4b_b2a'。"""
    if "FY4B" in npz_name or "FY4B" in npz_name:
        if "converted" in npz_name.lower():
            return "fy4b_b2a"
        return "fy4b_direct"
    return "fy4a"


def save_figure(fig, stem: str, npz_name: str, day: str):
    category = _classify_npz(npz_name)
    fig_dir = OUT_DIR / category / day
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("svg", None), ("pdf", None), ("png", 300)]:
        path = fig_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")


# ---------------------------------------------------------------------------
# L2 label reading (same as fusion_io but standalone for validation)
# ---------------------------------------------------------------------------

AGRI_L2_CLP_PHASE_MAP = cfg.AGRI_L2_CLP_PHASE_MAP  # {0:0, 1:1, 2:1, 3:2, 4:2}
CTH_VMIN, CTH_VMAX = cfg.AGRI_L2_CTH_VALID_RANGE   # (1, 20000)
CTH_FILL = cfg.AGRI_L2_CTH_FILL_VALUE               # -999


def read_l2_clp(nc_path: Path) -> Optional[np.ndarray]:
    import netCDF4 as nc
    ds = nc.Dataset(str(nc_path), "r")
    var = ds.variables["CLP"]
    var.set_auto_mask(False)
    raw = np.asarray(var[:], dtype=np.int16)
    ds.close()
    clp = np.full(raw.shape, np.nan, dtype=np.float32)
    for src, dst in AGRI_L2_CLP_PHASE_MAP.items():
        clp[raw == src] = float(dst)
    return clp


def read_l2_cth(nc_path: Path) -> Optional[np.ndarray]:
    import netCDF4 as nc
    ds = nc.Dataset(str(nc_path), "r")
    var = ds.variables["CTH"]
    var.set_auto_mask(False)
    raw = np.asarray(var[:], dtype=np.float32)
    ds.close()
    raw[(raw <= 0) | (raw >= 65500) | ~np.isfinite(raw)] = np.nan
    raw[raw < CTH_VMIN] = np.nan
    raw[raw > CTH_VMAX] = np.nan
    return raw


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(clp_pred: np.ndarray, clp_true: np.ndarray,
                    cth_pred: np.ndarray, cth_true: np.ndarray) -> dict:
    """Compute CLP classification + CTH regression metrics."""
    valid_clp = np.isfinite(clp_pred) & np.isfinite(clp_true)
    n_clp = int(valid_clp.sum())

    if n_clp == 0:
        return {"n_clp": 0, "n_cth": 0}

    pred = clp_pred[valid_clp].astype(np.int32)
    true = clp_true[valid_clp].astype(np.int32)

    oa = float((pred == true).mean()) * 100.0

    classes = sorted(set(pred) | set(true))
    per_class = {}
    for c in classes:
        mask = true == c
        if mask.any():
            per_class[f"cls{c}_acc"] = float((pred[mask] == c).mean()) * 100.0
        else:
            per_class[f"cls{c}_acc"] = -1.0

    accs = [v for v in per_class.values() if v >= 0]
    macro_acc = float(np.mean(accs)) if accs else 0.0

    # Confusion matrix
    cm = np.zeros((3, 3), dtype=np.int64)
    for ti in range(3):
        for pi in range(3):
            cm[ti, pi] = int(((true == ti) & (pred == pi)).sum())

    # CTH
    valid_cth = (np.isfinite(cth_pred) & np.isfinite(cth_true) &
                 np.isfinite(clp_pred) & np.isfinite(clp_true) &
                 (clp_pred > 0) & (clp_true > 0))
    n_cth = int(valid_cth.sum())

    cth_r = cth_rmse = cth_mae = cth_bias = 0.0
    if n_cth > 10:
        p = cth_pred[valid_cth]
        t = cth_true[valid_cth]
        cth_rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        cth_mae  = float(np.mean(np.abs(p - t)))
        cth_bias = float(np.mean(p - t))
        pm, tm = p - p.mean(), t - t.mean()
        denom = np.sqrt((pm ** 2).sum() * (tm ** 2).sum())
        cth_r = float((pm * tm).sum() / max(denom, 1e-12))

    return {
        "n_clp": n_clp, "n_cth": n_cth,
        "oa": oa, "macro_acc": macro_acc,
        "confusion_matrix": cm,
        "cth_rmse": cth_rmse, "cth_mae": cth_mae,
        "cth_bias": cth_bias, "cth_r": cth_r,
        **per_class,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm: np.ndarray, class_names: list, scene_id: str):
    """Normalised confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1) * 100

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100, aspect="equal")

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_norm[i,j]:.1f}%\n({cm[i,j]:,})",
                    ha="center", va="center", fontsize=6, color="white" if cm_norm[i, j] > 50 else "black")

    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(class_names, fontsize=6.5)
    ax.set_yticklabels(class_names, fontsize=6.5)
    ax.set_xlabel("Model prediction", fontsize=7)
    ax.set_ylabel("AGRI L2 reference", fontsize=7)
    ax.set_title(f"CLP confusion matrix\n{scene_id}", fontsize=7, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Row %", fontsize=6)
    fig.tight_layout()
    return fig


def _plot_cth_scatter(cth_pred: np.ndarray, cth_true: np.ndarray,
                      metrics: dict, scene_id: str, max_samples: int = 50_000):
    """Density scatter of predicted vs true CTH."""
    valid = (np.isfinite(cth_pred) & np.isfinite(cth_true))
    p, t = cth_pred[valid], cth_true[valid]
    if len(p) > max_samples:
        idx = np.random.choice(len(p), max_samples, replace=False)
        p, t = p[idx], t[idx]

    fig, ax = plt.subplots(figsize=(3.0, 2.8))

    # 2D histogram
    bins = np.linspace(0, 18000, 120)
    h, xe, ye = np.histogram2d(t, p, bins=[bins, bins])
    h = h.T
    # log-density
    h_log = np.log10(h.clip(min=1))
    im = ax.pcolormesh(xe, ye, h_log, cmap="YlOrRd", rasterized=True, shading="auto")

    ax.plot([0, 18000], [0, 18000], "k--", lw=0.7, alpha=0.5)

    r, rmse, bias = metrics["cth_r"], metrics["cth_rmse"], metrics["cth_bias"]
    ax.text(0.03, 0.97,
            f"$R$ = {r:.4f}\nRMSE = {rmse:.0f} m\nMAE = {metrics['cth_mae']:.0f} m\n"
            f"Bias = {bias:+.0f} m\n$n$ = {metrics['n_cth']:,}",
            transform=ax.transAxes, fontsize=6, va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))

    ax.set_xlabel("AGRI L2 CTH (m)", fontsize=7)
    ax.set_ylabel("Model CTH (m)", fontsize=7)
    ax.set_title(f"CTH validation\n{scene_id}", fontsize=7, fontweight="bold")
    ax.set_xlim(0, 18000); ax.set_ylim(0, 18000)
    ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("log$_{10}$(count)", fontsize=6)
    fig.tight_layout()
    return fig


def _plot_spatial_error(clp_pred: np.ndarray, clp_true: np.ndarray,
                        lat: np.ndarray, lon: np.ndarray, scene_id: str,
                        sub_lon: float = 104.7):
    """Spatial map of CLP agreement / disagreement."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    valid = np.isfinite(clp_pred) & np.isfinite(clp_true)
    agree = valid & (clp_pred == clp_true)
    disagree = valid & (clp_pred != clp_true)

    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.PlateCarree(central_longitude=sub_lon)

    fig, ax = plt.subplots(figsize=(6, 5.5), subplot_kw={"projection": map_crs})
    ax.add_feature(cfeature.COASTLINE, lw=0.4, alpha=0.5, zorder=2)
    ax.set_extent([sub_lon - 85, sub_lon + 85, -85, 85], crs=data_crs)

    # Subsample for plotting
    step = 12
    s = slice(None, None, step), slice(None, None, step)

    ax.scatter(lon[agree][::10], lat[agree][::10], s=0.15, alpha=0.25,
               color="royalblue", rasterized=True, zorder=1,
               transform=data_crs, label="Agreement")
    ax.scatter(lon[disagree][::10], lat[disagree][::10], s=0.3, alpha=0.6,
               color="crimson", rasterized=True, zorder=2,
               transform=data_crs, label="Disagreement")

    gl = ax.gridlines(draw_labels=True, alpha=0.3, linestyle="--", linewidth=0.4)
    gl.top_labels = False; gl.right_labels = False

    oa = (agree.sum() / max(valid.sum(), 1)) * 100
    ax.set_title(f"CLP spatial agreement — {scene_id}\nOA = {oa:.1f}%  (blue=agree, red=disagree)",
                 fontsize=7, fontweight="bold")
    ax.legend(loc="lower left", fontsize=6, markerscale=6)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------

def validate_one(npz_path: Path, clp_nc: Path, cth_nc: Path,
                 out_dir: Path, scene_id: str = "") -> dict:
    """Run validation for one triplet and save figures。兼容 FY-4A / FY-4B。"""
    npz_name = npz_path.name
    day = scene_id[:8]

    # Read model predictions
    data = np.load(npz_path)
    clp_pred = data["CLP_pred"].astype(np.float32)
    cth_pred = data["CTH_pred"].astype(np.float32)
    lat = data.get("latitude", np.zeros_like(clp_pred))
    lon = data.get("longitude", np.zeros_like(clp_pred))
    data.close()

    # 检测卫星确定星下点经度
    sat = _detect_satellite(npz_name)
    sub_lon = 133.0 if sat == "B" else 104.7

    # Read L2 truth
    clp_true = read_l2_clp(clp_nc)
    cth_true = read_l2_cth(cth_nc)

    if clp_true is None or cth_true is None:
        raise RuntimeError(f"Failed to read L2 data for {scene_id}")

    assert clp_pred.shape == clp_true.shape, (
        f"Shape mismatch: pred={clp_pred.shape} true={clp_true.shape}")

    metrics = compute_metrics(clp_pred, clp_true, cth_pred, cth_true)
    metrics["scene_id"] = scene_id

    log.info("  %s  CLP OA=%5.2f%%  CTH R=%.4f  RMSE=%.0f m",
             scene_id, metrics["oa"], metrics["cth_r"], metrics["cth_rmse"])

    # Generate figures
    if out_dir:
        stem = scene_id.replace(" ", "_")

        cm = metrics.get("confusion_matrix")
        if cm is not None:
            fig_cm = _plot_confusion_matrix(cm, cfg.CLP_CLASS_NAMES, scene_id)
            save_figure(fig_cm, f"{stem}_confusion", npz_name, day)
            plt.close(fig_cm)

        if metrics["n_cth"] > 10:
            fig_scatter = _plot_cth_scatter(cth_pred, cth_true, metrics, scene_id)
            save_figure(fig_scatter, f"{stem}_cth_scatter", npz_name, day)
            plt.close(fig_scatter)

        fig_spatial = _plot_spatial_error(clp_pred, clp_true, lat, lon, scene_id, sub_lon)
        save_figure(fig_spatial, f"{stem}_spatial", npz_name, day)
        plt.close(fig_spatial)

    return metrics


def _detect_satellite(filename: str) -> str:
    """从文件名检测卫星: 'A' (FY-4A) 或 'B' (FY-4B)。"""
    if "FY4B" in filename:
        return "B"
    return "A"


def _find_l2_for_npz(npz_name: str, ts: str, product: str) -> Optional[Path]:
    """根据 npz 文件名找到对应 L2 NC 文件。兼容 FY-4A / FY-4B。"""
    sat = _detect_satellite(npz_name)
    date_str = ts[:8]
    if sat == "B":
        l2_root = Path("/data/Data_yuq/FY4B")
        disk_id = "1330E"
        sat_id = "FY4B"
    else:
        l2_root = cfg.FY4A_L2_ROOT
        disk_id = "1047E"
        sat_id = "FY4A"

    l2_dir = l2_root / product / date_str
    if not l2_dir.is_dir():
        return None

    # 匹配逻辑：找同 timestamp 的 L2 文件
    for f in sorted(l2_dir.iterdir()):
        if not f.name.endswith(".NC"):
            continue
        f_ts = _extract_timestamp_from_filename(f.name)
        if f_ts == ts:
            return f
    return None


def find_npz_l2_pairs(npz_dir: Path, day: str) -> list:
    """Pair model .npz files with L2 .NC files by timestamp。兼容 FY-4A / FY-4B。"""
    npz_files = sorted(npz_dir.rglob(f"*{day}*.npz"))
    pairs = []
    for npz_path in npz_files:
        ts = _extract_timestamp_from_filename(npz_path.name)
        if ts is None:
            continue
        clp_nc = _find_l2_for_npz(npz_path.name, ts, "CLP")
        cth_nc = _find_l2_for_npz(npz_path.name, ts, "CTH")
        if clp_nc and cth_nc:
            scene_id = f"{ts[:8]}_{ts[8:]}"
            pairs.append((npz_path, clp_nc, cth_nc, scene_id))
    return pairs


def main():
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Validate model against AGRI L2")
    parser.add_argument("--npz_dir", required=True, help="Directory with inference .npz files")
    parser.add_argument("--day", required=True, help="Date YYYYMMDD to validate")
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--summary", action="store_true", help="Write summary CSV")
    args = parser.parse_args()

    pairs = find_npz_l2_pairs(Path(args.npz_dir), args.day)
    log.info("Found %d (npz, L2) pairs for day %s", len(pairs), args.day)

    if not pairs:
        log.error("No pairs found. Check --npz_dir and --day.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    all_metrics = []
    for npz_p, clp_nc, cth_nc, sid in pairs:
        try:
            m = validate_one(npz_p, clp_nc, cth_nc, out_dir, sid)
            all_metrics.append(m)
        except Exception as exc:
            log.error("Failed %s: %s", sid, exc)

    if all_metrics and args.summary:
        import pandas as pd
        rows = []
        for m in all_metrics:
            row = {k: v for k, v in m.items() if k != "confusion_matrix"}
            rows.append(row)
        df = pd.DataFrame(rows)
        csv_path = out_dir / f"summary_{args.day}.csv"
        df.to_csv(csv_path, index=False)

        # Print aggregate
        valid = df[df["n_clp"] > 0]
        if len(valid):
            log.info("=== Day %s aggregate (n=%d) ===", args.day, len(valid))
            log.info("CLP OA:  %.2f ± %.2f %%", valid["oa"].mean(), valid["oa"].std())
            log.info("CLP Macro: %.2f ± %.2f %%", valid["macro_acc"].mean(), valid["macro_acc"].std())
            cth_valid = valid[valid["n_cth"] > 10]
            if len(cth_valid):
                log.info("CTH RMSE: %.0f ± %.0f m", cth_valid["cth_rmse"].mean(), cth_valid["cth_rmse"].std())
                log.info("CTH R:    %.4f ± %.4f", cth_valid["cth_r"].mean(), cth_valid["cth_r"].std())


if __name__ == "__main__":
    main()
