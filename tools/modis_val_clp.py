"""
MODIS CLP 验证：在 MODIS 完全落入 AGRI disk 的前提下，
尝试多种策略提升 CLP 二值评估指标。
用法: conda run -n cloudunet python tools/modis_val_clp.py
"""
import sys, glob, logging
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
import fusion_config as fc
from fusion_io import (
    read_agri_scene, read_myd06, read_modis_geo_quick,
    find_matching_modis, find_matching_myd03,
    apply_quality_filter, _find_matching_l2_file,
)
from fusion_core import (
    aggregate_modis_to_agri, check_modis_in_agri_disk,
    latlon_to_xyz, km_to_chord,
)

MIN_DISK_DIST_KM = 10.0
MIN_VALID_PIXELS = 5000
SEARCH_RADIUS_KM = 2.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def find_agri_scenes(day):
    day_dir = cfg.AGRI_ROOT / day
    if not day_dir.exists():
        return []
    files = sorted(glob.glob(str(day_dir / "**/*_FDI-_*.HDF"), recursive=True))
    scenes = []
    for f in files:
        fname = Path(f).name
        idx = fname.find(day)
        if idx < 0:
            continue
        ts = fname[idx:idx + 14]
        try:
            dt = datetime.strptime(ts, "%Y%m%d%H%M%S")
            scenes.append((Path(f), ts, dt))
        except ValueError:
            pass
    return sorted(scenes, key=lambda x: x[2])


def read_l2_clp(fdi_path):
    l2_clp_nc = _find_matching_l2_file(fdi_path, "CLP")
    if l2_clp_nc is None:
        return None
    import netCDF4 as nc
    ds = nc.Dataset(str(l2_clp_nc), "r")
    v = ds.variables["CLP"]
    v.set_auto_mask(False)
    raw = np.asarray(v[:], dtype=np.int16)
    ds.close()
    l2_clp = np.full(raw.shape, np.nan, dtype=np.float32)
    for src, dst in cfg.AGRI_L2_CLP_PHASE_MAP.items():
        l2_clp[raw == src] = float(dst)
    return l2_clp


def cm_to_binary(cm_1km):
    """0,1=cloud(1), 2,3=clear(0)"""
    out = np.full_like(cm_1km, np.nan, dtype=np.float32)
    valid = np.isfinite(cm_1km)
    out[valid & (cm_1km <= 1)] = 1.0
    out[valid & (cm_1km >= 2)] = 0.0
    return out


def cm_to_binary_confident(cm_1km):
    """0=cloud(1), 3=clear(0), 1,2=NaN"""
    out = np.full_like(cm_1km, np.nan, dtype=np.float32)
    out[cm_1km == 0] = 1.0
    out[cm_1km == 3] = 0.0
    return out


def cm_strict_clear(cm_1km):
    """只取 Confident Clear (3) 为 clear(0)，其余全部 NaN。
    用于策略: 只对最确定的晴空做评估。"""
    out = np.full_like(cm_1km, np.nan, dtype=np.float32)
    out[cm_1km == 3] = 0.0
    return out


def cm_strict_cloud(cm_1km):
    """只取 Confident Cloudy (0) 为 cloud(1)，其余全部 NaN。"""
    out = np.full_like(cm_1km, np.nan, dtype=np.float32)
    out[cm_1km == 0] = 1.0
    return out


def project_cm(agri_lat, agri_lon, cm_granules, radius_km, time_max_min=None):
    """最近邻投影 MODIS CM 到 AGRI 网格，返回 dict of {name: grid}。"""
    H, W = agri_lat.shape
    N = H * W
    a_lat_f = agri_lat.ravel()
    a_lon_f = agri_lon.ravel()
    valid = np.isfinite(a_lat_f) & np.isfinite(a_lon_f)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        empty = np.full((H, W), np.nan)
        return {k: empty for k in ["cm_all", "cm_conf", "strict_clear", "strict_cloud"]}

    a_xyz = latlon_to_xyz(a_lat_f[valid_idx], a_lon_f[valid_idx])
    chord = km_to_chord(radius_km)
    t_max = time_max_min if time_max_min is not None else fc.TIME_LOW_Q_MIN

    bufs = {k: [[] for _ in range(len(valid_idx))]
            for k in ["cm_all", "cm_conf", "strict_clear", "strict_cloud", "dt"]}

    for g in cm_granules:
        lat_f = g["lat"].ravel()
        lon_f = g["lon"].ravel()
        geo_ok = np.isfinite(lat_f) & np.isfinite(lon_f)

        if g["scan_time"] is not None:
            dt_f = np.abs(g["scan_time"].ravel()[:len(lat_f)]).astype(np.float32)
        else:
            dt_f = np.full(len(lat_f), 0.0, np.float32)

        keep = geo_ok & np.isfinite(g["cm_all"].ravel()) & (dt_f <= t_max)
        if keep.sum() == 0:
            continue

        idx_k = np.where(keep)[0]
        m_xyz = latlon_to_xyz(lat_f[idx_k], lon_f[idx_k])
        tree = cKDTree(m_xyz)
        dist, nn = tree.query(a_xyz, k=1, distance_upper_bound=chord, workers=1)

        for ka in range(len(valid_idx)):
            ii = nn[ka]
            if ii >= len(m_xyz):
                continue
            for key in ["cm_all", "cm_conf", "strict_clear", "strict_cloud"]:
                bufs[key][ka].append(float(g[key].ravel()[idx_k[ii]]))
            bufs["dt"][ka].append(float(dt_f[idx_k[ii]]))

    outs = {}
    for key in ["cm_all", "cm_conf", "strict_clear", "strict_cloud"]:
        out = np.full(N, np.nan, np.float32)
        for ka, agri_idx in enumerate(valid_idx):
            if bufs[key][ka]:
                dt_v = np.array(bufs["dt"][ka])
                best = int(np.argmin(dt_v))
                out[agri_idx] = bufs[key][ka][best]
        outs[key] = out.reshape(H, W)

    return outs


def match_scene(fdi_path, agri_dt, day, time_max_min=None):
    scene = read_agri_scene(fdi_path)
    if scene is None:
        return None, "read_agri_scene failed"
    a_lat, a_lon = scene["lat"], scene["lon"]

    l2_clp = read_l2_clp(fdi_path)
    if l2_clp is None:
        return None, "no L2 CLP file"

    day_dir = cfg.MODIS_ROOT / day
    if not day_dir.exists():
        return None, "no MODIS dir"
    dt_obj = datetime.strptime(day, "%Y%m%d")
    jday = dt_obj.timetuple().tm_yday
    modis_files = [Path(f) for f in sorted(glob.glob(str(day_dir / f"*A{dt_obj.year}{jday:03d}*.hdf")))]
    myd03_files = [Path(f) for f in sorted(glob.glob(str(cfg.MYD03_ROOT / day / "*.hdf")))]

    matched = find_matching_modis(agri_dt, modis_files)
    if not matched:
        return None, "no MODIS within time window"

    cfg.MODIS_FILTER_WEAK_QUALITY = True
    cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_CLP = (0, 3)
    cfg.MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_REG = (0,)

    modis_list, cm_granules = [], []
    for mf in matched:
        myd03 = find_matching_myd03(mf, myd03_files)
        geo = read_modis_geo_quick(mf, myd03_file=myd03)
        if geo is None:
            continue
        m = read_myd06(mf, agri_dt=agri_dt, myd03_file=myd03, geo_cache=geo)
        if m is None:
            continue
        modis_list.append(m)
        cm_raw = m.get("CM_1km")
        if cm_raw is None:
            continue
        lat_1 = m.get("lat_1km")
        lon_1 = m.get("lon_1km")
        if lat_1 is None or lon_1 is None:
            continue
        cm_granules.append({
            "cm_all": cm_to_binary(cm_raw),
            "cm_conf": cm_to_binary_confident(cm_raw),
            "strict_clear": cm_strict_clear(cm_raw),
            "strict_cloud": cm_strict_cloud(cm_raw),
            "lat": lat_1, "lon": lon_1,
            "scan_time": m.get("scan_time_1km"),
        })

    if not modis_list or not cm_granules:
        return None, "no valid MODIS data"

    g0 = cm_granules[0]
    if not check_modis_in_agri_disk(g0["lat"], g0["lon"], a_lat, a_lon, max_dist_km=MIN_DISK_DIST_KM):
        return None, "MODIS not fully inside AGRI disk"

    # Pipeline CLP
    vza = np.zeros_like(a_lat)
    sza = np.zeros_like(a_lat)
    labels = aggregate_modis_to_agri(a_lat, a_lon, modis_list)
    if labels is None:
        return None, "aggregation failed"
    labels = apply_quality_filter({"VZA": vza, "SZA": sza}, labels)

    l2_bin = (l2_clp > 0).astype(np.float32)
    l2_bin[np.isnan(l2_clp)] = np.nan
    pipe_bin = (labels["CLP"] > 0).astype(np.float32)
    pipe_bin[np.isnan(labels["CLP"])] = np.nan

    cm_grids = project_cm(a_lat, a_lon, cm_granules, SEARCH_RADIUS_KM, time_max_min)

    return {
        "l2_bin": l2_bin, "pipe_bin": pipe_bin,
        "cm_grids": cm_grids,
        "lat": a_lat, "lon": a_lon,
        "modis_lat": g0["lat"], "modis_lon": g0["lon"],
    }, None


def binary_metrics(ref, pred, min_n=MIN_VALID_PIXELS):
    v = np.isfinite(ref) & np.isfinite(pred)
    n = int(v.sum())
    if n < min_n:
        return None
    r, p = ref[v].astype(int), pred[v].astype(int)
    oa = float((r == p).mean()) * 100
    both_c = int(((r == 1) & (p == 1)).sum())
    both_cl = int(((r == 0) & (p == 0)).sum())
    rc_pc = int(((r == 1) & (p == 0)).sum())
    rcl_pc = int(((r == 0) & (p == 1)).sum())
    return {"n": n, "oa": oa, "both_cloud": both_c, "both_clear": both_cl,
            "ref_cloud_pred_clear": rc_pc, "ref_clear_pred_cloud": rcl_pc}


def main():
    day = "20190505"
    scenes = find_agri_scenes(day)

    # ================================================================
    # 策略 0: 基线 (CM-conf, time<=7.5min)
    # 策略 1: 缩时间窗 (CM-conf, time<=3min)
    # 策略 2: 只对 MODIS 确定晴空评估 (strict_clear vs L2)
    # 策略 3: 只对 MODIS 确定云评估 (strict_cloud vs L2)
    # 策略 4: 只用 MODIS 云量 > 60% 的场景
    # 策略 5: 只用 MODIS 云量 > L2 云量的场景
    # ================================================================

    strategies = [
        ("0:CM-conf-5min", dict(time_max_min=5.0)),
        ("1:CM-conf-3min", dict(time_max_min=3.0)),
        ("2:strict-clear", dict(time_max_min=5.0)),
        ("3:strict-cloud", dict(time_max_min=5.0)),
    ]

    all_scene_results = []

    for si, (fdi, ts, agri_dt) in enumerate(scenes):
        log.info("Scene %d/%d: %s", si + 1, len(scenes), ts)

        # 基线匹配 (5min)
        data, err = match_scene(fdi, agri_dt, day, time_max_min=5.0)
        if data is None:
            log.info("  SKIP: %s", err)
            continue

        # 缩时间窗匹配 (3min)
        data_3min, _ = match_scene(fdi, agri_dt, day, time_max_min=3.0)

        l2 = data["l2_bin"]

        # 云量
        l2_valid = np.isfinite(l2)
        l2_cfrac = float(l2[l2_valid].mean()) * 100 if l2_valid.any() else 0

        cm_conf = data["cm_grids"]["cm_conf"]
        cm_valid = np.isfinite(cm_conf)
        cm_cfrac = float(cm_conf[cm_valid].mean()) * 100 if cm_valid.any() else 0

        scene_info = {"scene": ts, "l2_cfrac": l2_cfrac, "cm_cfrac": cm_cfrac}

        # 策略 0: CM-conf, 7.5min
        m0 = binary_metrics(l2, cm_conf)
        scene_info["s0"] = m0

        # 策略 1: CM-conf, 3min
        if data_3min is not None:
            cm_conf_3 = data_3min["cm_grids"]["cm_conf"]
            m1 = binary_metrics(l2, cm_conf_3)
        else:
            m1 = None
        scene_info["s1"] = m1

        # 策略 2: strict clear (L2 clear vs MODIS confident clear)
        # ref = L2, pred = strict_clear (只有 MODIS 确定晴空=0, 其余=NaN)
        # 这样只评估 MODIS 确定晴空的像元
        strict_clear = data["cm_grids"]["strict_clear"]
        m2 = binary_metrics(l2, strict_clear)
        scene_info["s2"] = m2

        # 策略 3: strict cloud (L2 cloud vs MODIS confident cloud)
        strict_cloud = data["cm_grids"]["strict_cloud"]
        m3 = binary_metrics(l2, strict_cloud)
        scene_info["s3"] = m3

        all_scene_results.append(scene_info)

        # 打印
        def _fmt(m):
            if m is None:
                return "N/A"
            return f"OA={m['oa']:.1f}% N={m['n']}"

        log.info("  L2_cfrac=%.1f%% CM_cfrac=%.1f%%", l2_cfrac, cm_cfrac)
        log.info("  S0(7.5min): %s", _fmt(m0))
        log.info("  S1(3min):   %s", _fmt(m1))
        log.info("  S2(strict-clear): %s", _fmt(m2))
        log.info("  S3(strict-cloud): %s", _fmt(m3))

    if not all_scene_results:
        log.info("No valid scenes")
        return

    # ================================================================
    # 汇总
    # ================================================================
    log.info("")
    log.info("=" * 100)
    log.info("SUMMARY (day=%s, MODIS fully inside AGRI disk)", day)
    log.info("=" * 100)

    # 策略 4: 只用 MODIS 云量 > 60% 的场景
    # 策略 5: 只用 MODIS 云量 > L2 云量的场景

    def _pool(results, filter_fn=None, key="s0"):
        filtered = [r for r in results if filter_fn is None or filter_fn(r)]
        metrics = [r[key] for r in filtered if r[key] is not None]
        if not metrics:
            return None, 0
        total_n = sum(m["n"] for m in metrics)
        total_bc = sum(m["both_cloud"] for m in metrics)
        total_bcl = sum(m["both_clear"] for m in metrics)
        total_rcpc = sum(m["ref_cloud_pred_clear"] for m in metrics)
        total_rclpc = sum(m["ref_clear_pred_cloud"] for m in metrics)
        pooled_oa = (total_bc + total_bcl) / total_n * 100 if total_n > 0 else 0
        oa_vals = [m["oa"] for m in metrics]
        return {
            "pooled_oa": pooled_oa, "n_scenes": len(metrics), "total_n": total_n,
            "mean_oa": np.mean(oa_vals), "std_oa": np.std(oa_vals),
            "both_cloud": total_bc, "both_clear": total_bcl,
            "rc_pc": total_rcpc, "rcl_pc": total_rclpc,
        }, len(metrics)

    log.info("")
    log.info("%-30s  %8s  %8s  %8s  %10s  %6s",
             "Strategy", "PooledOA", "MeanOA", "StdOA", "TotalN", "nScene")
    log.info("-" * 90)

    # 策略 0: 全场景
    r, n = _pool(all_scene_results, key="s0")
    if r:
        log.info("%-30s  %7.1f%%  %7.1f%%  %7.1f%%  %10d  %6d",
                 "0:CM-conf-5min (all)", r["pooled_oa"], r["mean_oa"], r["std_oa"], r["total_n"], r["n_scenes"])

    # 策略 1: 缩时间窗
    r, n = _pool(all_scene_results, key="s1")
    if r:
        log.info("%-30s  %7.1f%%  %7.1f%%  %7.1f%%  %10d  %6d",
                 "1:CM-conf-3min (all)", r["pooled_oa"], r["mean_oa"], r["std_oa"], r["total_n"], r["n_scenes"])

    # 策略 2: strict clear
    r, n = _pool(all_scene_results, key="s2")
    if r:
        log.info("%-30s  %7.1f%%  %7.1f%%  %7.1f%%  %10d  %6d",
                 "2:strict-clear (all)", r["pooled_oa"], r["mean_oa"], r["std_oa"], r["total_n"], r["n_scenes"])

    # 策略 3: strict cloud
    r, n = _pool(all_scene_results, key="s3")
    if r:
        log.info("%-30s  %7.1f%%  %7.1f%%  %7.1f%%  %10d  %6d",
                 "3:strict-cloud (all)", r["pooled_oa"], r["mean_oa"], r["std_oa"], r["total_n"], r["n_scenes"])

    log.info("-" * 90)

    # 策略 4: MODIS 云量 > 60%
    r, n = _pool(all_scene_results, filter_fn=lambda r: r["cm_cfrac"] > 60, key="s0")
    if r:
        log.info("%-30s  %7.1f%%  %7.1f%%  %7.1f%%  %10d  %6d",
                 "4:CM_cfrac>60% (5min)", r["pooled_oa"], r["mean_oa"], r["std_oa"], r["total_n"], r["n_scenes"])

    # 策略 5: MODIS 云量 > L2 云量
    r, n = _pool(all_scene_results, filter_fn=lambda r: r["cm_cfrac"] > r["l2_cfrac"], key="s0")
    if r:
        log.info("%-30s  %7.1f%%  %7.1f%%  %7.1f%%  %10d  %6d",
                 "5:CM>L2_cfrac (5min)", r["pooled_oa"], r["mean_oa"], r["std_oa"], r["total_n"], r["n_scenes"])

    # 策略 6: MODIS 云量 > L2 云量 + 3min
    r, n = _pool(all_scene_results, filter_fn=lambda r: r["cm_cfrac"] > r["l2_cfrac"], key="s1")
    if r:
        log.info("%-30s  %7.1f%%  %7.1f%%  %7.1f%%  %10d  %6d",
                 "6:CM>L2_cfrac (3min)", r["pooled_oa"], r["mean_oa"], r["std_oa"], r["total_n"], r["n_scenes"])

    # 策略 7: MODIS 云量 > 60% + strict clear
    r, n = _pool(all_scene_results, filter_fn=lambda r: r["cm_cfrac"] > 60, key="s2")
    if r:
        log.info("%-30s  %7.1f%%  %7.1f%%  %7.1f%%  %10d  %6d",
                 "7:CM>60%+strict-clear", r["pooled_oa"], r["mean_oa"], r["std_oa"], r["total_n"], r["n_scenes"])

    # 逐场景明细
    log.info("")
    log.info("Per-scene detail:")
    log.info("%-20s  %8s  %8s  %8s  %8s  %8s  %8s  %8s",
             "Scene", "L2_cfrac", "CM_cfrac", "S0_OA", "S1_OA", "S2_OA", "S3_OA", "Status")
    for r in all_scene_results:
        def _oa(key):
            m = r.get(key)
            return f"{m['oa']:.1f}%" if m else "N/A"
        status = "GOOD" if r["cm_cfrac"] > r["l2_cfrac"] else "BIAS"
        log.info("%-20s  %7.1f%%  %7.1f%%  %8s  %8s  %8s  %8s  %8s",
                 r["scene"], r["l2_cfrac"], r["cm_cfrac"],
                 _oa("s0"), _oa("s1"), _oa("s2"), _oa("s3"), status)


if __name__ == "__main__":
    main()
