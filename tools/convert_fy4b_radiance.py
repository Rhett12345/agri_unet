"""
convert_fy4b_radiance.py
========================
FY-4B AGRI 辐射值 → FY-4A 等效辐射值转换工具。

转换系数基于两星交叉定标得到的辐射值线性/多项式关系。
IR 通道通过 Planck 函数在 BT ↔ Radiance 之间转换后再应用系数。

用法:
  python tools/convert_fy4b_radiance.py --fdi_dir /data/Data_yuq/FY4B/FDI/20230501 \
      --out_dir /data/Data_yuq/FY4B/FDI_converted/20230501

输出: 与输入同名的 .npz 文件，包含 converted_BT (H,W,6), lat, lon, VZA, SZA。
     可直接用 run_inference 读取（需配合 dataset 中的 NormStats）。

对比方式:
  1. 直接外推: python main.py --stages infer --agri_dir .../FY4B/FDI/20230501 ...
  2. 转换外推: 先跑本脚本转换，再用转换后的 npz 文件推理
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from fusion_io import (
    read_agri_scene, _paired_geo_file, _read_geo,
    _h5_read_first, _h5_read_first_or, _lut_calibrate,
)

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# 物理常数 + 通道波长
# ═══════════════════════════════════════════════════════════════════
_H_PLANCK = 6.62607015e-34   # J·s
_C_LIGHT  = 2.99792458e8     # m/s
_K_BOLTZ  = 1.380649e-23     # J/K
_C1 = 2.0 * _H_PLANCK * _C_LIGHT ** 2   # 2hc²
_C2 = _H_PLANCK * _C_LIGHT / _K_BOLTZ    # hc/k

# 通道中心波长 (μm) — FY-4A / FY-4B
_WAVELENGTH_A = {9: 6.25, 10: 7.10, 11: 8.50, 12: 10.70, 13: 12.00, 14: 13.50}
_WAVELENGTH_B = {9: 6.25, 10: 6.95, 12: 8.55, 13: 10.80, 14: 12.00, 15: 13.30}


def bt_to_radiance(bt_k: np.ndarray, wl_um: float) -> np.ndarray:
    """BT (K) → 光谱辐亮度 (W·m⁻²·sr⁻¹·μm⁻¹)，Planck 公式。"""
    wl_m = wl_um * 1e-6
    valid = np.isfinite(bt_k) & (bt_k > 0)
    rad = np.full_like(bt_k, np.nan, dtype=np.float64)
    if valid.any():
        t = bt_k[valid].astype(np.float64)
        rad[valid] = _C1 / (wl_m ** 5) / (np.exp(_C2 / (wl_m * t)) - 1.0) * 1e-6
    return rad.astype(np.float32)


def radiance_to_bt(rad: np.ndarray, wl_um: float) -> np.ndarray:
    """光谱辐亮度 → BT (K)，Planck 反函数。"""
    wl_m = wl_um * 1e-6
    valid = np.isfinite(rad) & (rad > 0)
    bt = np.full_like(rad, np.nan, dtype=np.float64)
    if valid.any():
        r = rad[valid].astype(np.float64) * 1e6  # → W·m⁻²·sr⁻¹·m⁻¹
        bt[valid] = _C2 / (wl_m * np.log(_C1 / (wl_m ** 5 * r) + 1.0))
    return bt.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# 转换系数
# ═══════════════════════════════════════════════════════════════════

def load_b2a_coeffs(csv_path: str) -> Dict[int, dict]:
    """加载 FY-4B → FY-4A 转换系数，含 Planck 订正 S_t/Q_t/S_r/Q_r。

    Returns
    -------
    dict key=物理通道号, value=dict with:
        model_type, c1, c2, intercept,
        S_t, Q_t, S_r, Q_r  (default 1,0,1,0)
    """
    coeffs = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["Direction"] != "B2A":
                continue
            src_ch = int(row["Source_Ch"].split("ch")[1])

            def _float(col, default=0.0):
                v = row.get(col, "").strip()
                return float(v) if v else default

            coeffs[src_ch] = {
                "model_type": row.get("Model", "linear").strip(),
                "c1": _float("Coeff_1"),
                "c2": _float("Coeff_2") if row.get("Coeff_2", "").strip() else None,
                "intercept": _float("Intercept"),
                "S_t": _float("S_t", 1.0),   # Planck订正: 目标(B)侧
                "Q_t": _float("Q_t", 0.0),
                "S_r": _float("S_r", 1.0),   # Planck订正: 参考(A)侧
                "Q_r": _float("Q_r", 0.0),
            }
    return coeffs


def convert_channel(val: np.ndarray, ch_b: int, wl_b: Optional[float],
                    wl_a: Optional[float], coeff: dict) -> np.ndarray:
    """
    单通道转换（含 Planck 单波长订正）：

    BT_B → S_t/Q_t 订正 → Planck → Rad_B → 系数转换 → Rad_A
         → 逆Planck → BT_A_raw → S_r/Q_r 订正 → BT_A

    订正公式（来自 sensitivity.py）：
      T_planck = S_t · T_true + Q_t    (源通道 Planck 偏置修正)
      T_true   = S_r · T_planck + Q_r  (目标通道 Planck 偏置修正)
    """
    if ch_b not in coeff:
        log.warning("No coeff for B%02d, pass-through", ch_b)
        return val

    c = coeff[ch_b]
    S_t, Q_t = c["S_t"], c["Q_t"]
    S_r, Q_r = c["S_r"], c["Q_r"]
    valid = np.isfinite(val)

    if wl_b is not None and wl_a is not None:
        # ① 源通道 BT 订正 → Planck → 辐亮度
        bt_planck = S_t * val + Q_t
        rad_b = bt_to_radiance(bt_planck, wl_b)

        # ② 系数转换
        rad_a = np.full_like(rad_b, np.nan, dtype=np.float32)
        v = rad_b[valid]
        if c["c2"] is not None:
            rad_a[valid] = c["c2"] * v ** 2 + c["c1"] * v + c["intercept"]
        else:
            rad_a[valid] = c["c1"] * v + c["intercept"]
        rad_a[valid][rad_a[valid] <= 0] = np.nan

        # ③ 逆 Planck → 目标通道 BT 订正
        bt_raw = radiance_to_bt(rad_a, wl_a)
        return S_r * bt_raw + Q_r
    else:
        out = np.full_like(val, np.nan, dtype=np.float32)
        v = val[valid]
        if c["c2"] is not None:
            out[valid] = c["c2"] * v ** 2 + c["c1"] * v + c["intercept"]
        else:
            out[valid] = c["c1"] * v + c["intercept"]
        return out


# ═══════════════════════════════════════════════════════════════════
# 主转换
# ═══════════════════════════════════════════════════════════════════

def convert_one_scene(fdi_path: Path, out_dir: Path, coeffs: dict) -> Optional[Path]:
    """
    转换单个 FY-4B FDI 场景 → FY-4A 等效 BT .npz。

    通过 Planck 函数做 BT ↔ 辐亮度转换，在辐亮度空间应用 B2A 系数。
    """
    # 读原始场景获取 BT + 地理信息
    log.info("Reading: %s", fdi_path.name)
    scene = read_agri_scene(fdi_path)
    if scene is None:
        log.error("Failed to read %s", fdi_path)
        return None

    bt_b = scene["BT"]          # (H, W, 6)  FY-4B BT [K]
    lat = scene["lat"]
    lon = scene["lon"]
    vza = scene["VZA"]
    sza = scene["SZA"]

    # B 通道索引 [8,9,11,12,13,14] → 物理通道 [9,10,12,13,14,15]
    b_indices = cfg._AGRI_BT_CHANNEL_INDICES_B
    # A 通道索引 [8,9,10,11,12,13] → 物理通道 [9,10,11,12,13,14]
    a_indices = cfg._AGRI_BT_CHANNEL_INDICES_A

    H, W, C = bt_b.shape
    bt_a = np.full_like(bt_b, np.nan, dtype=np.float32)

    for ci in range(C):
        ch_b = b_indices[ci] + 1
        ch_a = a_indices[ci] + 1
        wl_b = _WAVELENGTH_B.get(ch_b)
        wl_a = _WAVELENGTH_A.get(ch_a)
        log.info("  B%02d(%.2fμm) → A%02d(%.2fμm)", ch_b, wl_b or 0, ch_a, wl_a or 0)
        bt_a[:, :, ci] = convert_channel(bt_b[:, :, ci], ch_b, wl_b, wl_a, coeffs)

    # 保存
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = fdi_path.stem
    out_path = out_dir / f"{stem}_converted.npz"
    np.savez_compressed(
        out_path,
        BT_converted=bt_a,
        BT_original=bt_b,
        latitude=lat, longitude=lon,
        VZA=vza, SZA=sza,
    )
    log.info("Saved: %s", out_path)
    return out_path


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="FY-4B FDI → FY-4A 等效 BT 转换")
    parser.add_argument("--fdi_dir", required=True,
                        help="FY-4B FDI HDF5 目录")
    parser.add_argument("--out_dir", required=True,
                        help="输出 .npz 目录")
    parser.add_argument("--coeff_csv", default=None,
                        help="转换系数 CSV (默认: 项目根 transfer_coeff_fy4a_fy4b_v1.csv)")
    args = parser.parse_args()

    csv_path = args.coeff_csv or str(
        Path(__file__).resolve().parent.parent / "transfer_coeff_fy4a_fy4b_v1.csv")
    coeffs = load_b2a_coeffs(csv_path)
    log.info("Loaded %d B2A coefficients from %s", len(coeffs), csv_path)

    fdi_dir = Path(args.fdi_dir)
    fdi_files = sorted(list(fdi_dir.glob("*.HDF")) + list(fdi_dir.glob("*.hdf")))
    fdi_files = [f for f in fdi_files if "_FDI-_" in f.name]

    out_dir = Path(args.out_dir)
    n_ok = 0
    for f in fdi_files:
        result = convert_one_scene(f, out_dir, coeffs)
        if result:
            n_ok += 1

    log.info("Done: %d/%d converted", n_ok, len(fdi_files))


if __name__ == "__main__":
    main()
