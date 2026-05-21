"""
fusion_core.py
==============
AGRI → GPM 空间匹配引擎。

核心功能：
  1. 对每个 GPM 0.1° 格点，在 AGRI 经纬度网格中找到最近邻行列
  2. 以该行列为中心提取 11×11 AGRI patch
  3. 按 GPM 降水率分配分类标签 (0-3)
  4. 夜间检测：可见光通道置零

设计原则
--------
- 使用 scipy cKDTree 进行球面最近邻搜索
- 全分辨率 AGRI 采样以保证精度
- 向量化批量处理
- 避免 GDAL 等重型 GIS 依赖
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from scipy.spatial import cKDTree

import config as cfg
import fusion_config as fc

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 球面坐标转换
# ---------------------------------------------------------------------------

def latlon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """(lat, lon) 度 -> 单位球面 XYZ，用于 KD-tree 弦长距离。"""
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    return np.column_stack([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r),
    ])


def km_to_chord(km: float) -> float:
    """地球表面距离 (km) -> 单位球弦长。"""
    return 2.0 * np.sin(km / (2.0 * 6371.0))


# ---------------------------------------------------------------------------
# AGRI 有效圆盘掩膜
# ---------------------------------------------------------------------------

def compute_tight_disk_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    margin_deg: float = 5.0,
    sub_lon: float = 105.0,
) -> np.ndarray:
    """
    计算 AGRI 全圆盘缩紧后的有效像元 mask。
    以星下点 (0°N, sub_lon°E) 为圆心，剔除距圆盘边界 margin_deg 度以内的边缘像元。
    """
    valid = np.isfinite(lat) & np.isfinite(lon)
    if not valid.any():
        return valid

    sub_lat = 0.0
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    c_lat_r = np.deg2rad(sub_lat)
    c_lon_r = np.deg2rad(sub_lon)

    dlat = lat_r - c_lat_r
    dlon = lon_r - c_lon_r
    a = np.sin(dlat * 0.5) ** 2 + np.cos(c_lat_r) * np.cos(lat_r) * np.sin(dlon * 0.5) ** 2
    dist_deg = np.rad2deg(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))

    max_dist = float(dist_deg[valid].max())
    threshold = max(0.0, max_dist - margin_deg)

    return valid & (dist_deg <= threshold)


# ---------------------------------------------------------------------------
# 夜间判断
# ---------------------------------------------------------------------------

def is_nighttime_sza(sza: np.ndarray) -> bool:
    """判断整景是否为夜间（SZA 中位数 > 85°）。"""
    valid = np.isfinite(sza)
    if not valid.any():
        return False
    return bool(np.nanmedian(sza[valid]) > 85.0)


# ---------------------------------------------------------------------------
# 主匹配引擎
# ---------------------------------------------------------------------------

def match_agri_to_gpm(
    agri: dict,
    gpm_precip: np.ndarray,
    gpm_lat: np.ndarray,
    gpm_lon: np.ndarray,
    gpm_quality: Optional[np.ndarray] = None,
    dt_min: float = 0.0,
    step: int = 1,
    max_samples: int = 0,
    min_precip_quality: float = 0.0,
    region_lat_min: float = -90.0,
    region_lat_max: float = 90.0,
    region_lon_min: float = -180.0,
    region_lon_max: float = 180.0,
) -> List[dict]:
    """
    将 AGRI 数据匹配到 GPM 格点，为每个有效 GPM 格点生成一个样本。

    Parameters
    ----------
    agri : dict with lat, lon, VZA, SZA, BT (all 2748×2748)
    gpm_precip : (N_lat, N_lon) GPM 降水率
    gpm_lat : (N_lat,) GPM 纬度轴
    gpm_lon : (N_lon,) GPM 经度轴
    gpm_quality : optional (N_lat, N_lon) GPM 质量
    dt_min : AGRI-GPM 时间差 (分钟)
    step : 格点采样步长 (1=全采样)
    max_samples : 最大样本数 (0=不限制)
    min_precip_quality : 最低 GPM 质量

    Returns
    -------
    List of dict with agri, geo, label, precip, gpm_lat, gpm_lon, dt_min
    """
    ph, pw = cfg.PATCH_SIZE
    ph_half = ph // 2

    agri_lat = agri["lat"]
    agri_lon = agri["lon"]
    agri_bt  = agri["BT"]
    agri_vza = agri["VZA"]
    agri_sza = agri["SZA"]
    H_a, W_a, C = agri_bt.shape

    # ── 构建 AGRI 全分辨率 (lat, lon) → X,Y,Z 表 ──
    # 采样策略：每隔 sample_k 个像素取一点用于 KD-tree
    # sample_k=2 → ~1.9M 点，精度 ±1 pixel
    sample_k = 2
    rows = np.arange(0, H_a, sample_k, dtype=np.int32)
    cols = np.arange(0, W_a, sample_k, dtype=np.int32)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    valid_agri = np.isfinite(agri_lat[rr, cc]) & np.isfinite(agri_lon[rr, cc])

    agri_xyz = latlon_to_xyz(
        agri_lat[rr, cc][valid_agri],
        agri_lon[rr, cc][valid_agri],
    )
    tree = cKDTree(agri_xyz)

    # 行列映射
    agri_rows = rr[valid_agri]
    agri_cols = cc[valid_agri]

    # ── 判断是否为夜间 ──
    night = is_nighttime_sza(agri_sza)
    if night:
        log.debug("Nighttime scene detected, will zero visible channels")

    # ── 构建 GPM 查询点 ──
    N_lat = len(gpm_lat)
    N_lon = len(gpm_lon)

    # 确定 AGRI 覆盖范围
    full_valid = np.isfinite(agri_lat) & np.isfinite(agri_lon)
    agri_lat_min = float(np.nanmin(agri_lat[full_valid]))
    agri_lat_max = float(np.nanmax(agri_lat[full_valid]))
    agri_lon_min = float(np.nanmin(agri_lon[full_valid]))
    agri_lon_max = float(np.nanmax(agri_lon[full_valid]))

    # 收集查询点
    query_pts = []
    query_idx = []   # [(li, lj), ...]
    for li in range(0, N_lat, step):
        plat = gpm_lat[li]
        if plat < agri_lat_min - 1.0 or plat > agri_lat_max + 1.0:
            continue
        if plat < region_lat_min or plat > region_lat_max:
            continue
        for lj in range(0, N_lon, step):
            plon = gpm_lon[lj]
            if plon < agri_lon_min - 1.0 or plon > agri_lon_max + 1.0:
                continue
            if plon < region_lon_min or plon > region_lon_max:
                continue
            pval = gpm_precip[li, lj]
            if not np.isfinite(pval) or pval < -9000:
                continue
            if gpm_quality is not None and min_precip_quality > 0:
                qval = gpm_quality[li, lj]
                if np.isfinite(qval) and qval < min_precip_quality:
                    continue
            query_pts.append((plat, plon))
            query_idx.append((li, lj, pval))

    if not query_pts:
        return []

    query_arr = np.array(query_pts, dtype=np.float64)
    query_xyz = latlon_to_xyz(query_arr[:, 0], query_arr[:, 1])

    # KD-tree 搜索半径：~20 km
    chord_limit = km_to_chord(20.0)
    dist, nn_idx = tree.query(query_xyz, k=1, distance_upper_bound=chord_limit)

    # ── 生成样本 ──
    samples = []
    for qi in range(len(query_idx)):
        if nn_idx[qi] >= len(agri_rows):
            continue

        center_row = int(agri_rows[nn_idx[qi]])
        center_col = int(agri_cols[nn_idx[qi]])

        # 提取 11×11 patch
        r_start = center_row - ph_half
        c_start = center_col - ph_half
        r_end = r_start + ph
        c_end = c_start + pw

        if r_start < 0 or c_start < 0 or r_end > H_a or c_end > W_a:
            continue

        # 直接切片
        patch_bt = agri_bt[r_start:r_end, c_start:c_end, :].copy()           # (11, 11, 7)
        patch_lat = agri_lat[r_start:r_end, c_start:c_end].copy()
        patch_lon = agri_lon[r_start:r_end, c_start:c_end].copy()

        # 有效性检查
        valid_frac = np.isfinite(patch_bt).mean()
        if valid_frac < 0.5:
            continue

        # NaN → 0
        patch_bt = np.where(np.isfinite(patch_bt), patch_bt, 0.0).astype(np.float32)
        patch_lat = np.where(np.isfinite(patch_lat), patch_lat, 0.0).astype(np.float32)
        patch_lon = np.where(np.isfinite(patch_lon), patch_lon, 0.0).astype(np.float32)

        # 夜间：可见光通道置零（用场景级 SZA 判断）
        if night:
            for vi in cfg.VIS_CHANNEL_INDICES:
                patch_bt[:, :, vi] = 0.0

        # Transpose BT to (C, H, W)
        patch_bt = patch_bt.transpose(2, 0, 1)   # (7, 11, 11)

        # Geo: only lat, lon (2 channels)
        geo = np.stack([patch_lat, patch_lon], axis=0).astype(np.float32)

        li, lj, pval = query_idx[qi]
        label = cfg.precip_to_class(float(pval))

        samples.append({
            "agri": patch_bt,
            "geo": geo,
            "label": label,
            "precip": float(pval),
            "gpm_lat": float(gpm_lat[li]),
            "gpm_lon": float(gpm_lon[lj]),
            "dt_min": float(dt_min),
        })

        if max_samples > 0 and len(samples) >= max_samples:
            break

    return samples
