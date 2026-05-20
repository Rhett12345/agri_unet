"""
fusion_config.py
================
GPM → AGRI 融合配置。
与 config.py 分离，以便在不改动主 config 的前提下调整融合参数。
所有参数均可被环境变量覆盖（见末尾）。
"""

import os
import multiprocessing as _mp

# ─────────────────────────────────────────────────────────────────────────────
# 时间匹配（分钟）
# ─────────────────────────────────────────────────────────────────────────────
TIME_MAX_MIN = float(os.environ.get("FUSION_TIME_MAX_MIN", "15.0"))
# GPM 半小时文件与 AGRI 景的最大时间差

# ─────────────────────────────────────────────────────────────────────────────
# 空间匹配
# ─────────────────────────────────────────────────────────────────────────────
GPM_GRID_RES_DEG = float(os.environ.get("FUSION_GPM_GRID_RES_DEG", "0.1"))

# AGRI 全圆盘边缘收缩度数：避免边缘像元插值伪影
AGRI_DISK_MARGIN_DEG = float(os.environ.get("FUSION_AGRI_DISK_MARGIN_DEG", "5.0"))
AGRI_SUB_LON = float(os.environ.get("FUSION_AGRI_SUB_LON", "104.7"))

# ─────────────────────────────────────────────────────────────────────────────
# Patch 采样
# ─────────────────────────────────────────────────────────────────────────────
PATCH_HALF = int(os.environ.get("FUSION_PATCH_HALF", "5"))  # half-size for 11×11

# 质量控制：precipitationQualityIndex 最低阈值
MIN_PRECIP_QUALITY = float(os.environ.get("FUSION_MIN_PRECIP_QUALITY", "0.0"))

# GPM 格点采样步长（每隔 N 个格点采样一个，1=全采样）
GPM_SAMPLE_STEP = int(os.environ.get("FUSION_GPM_SAMPLE_STEP", "1"))

# 每景最多采样数（0=不限制）
MAX_SAMPLES_PER_SCENE = int(os.environ.get("FUSION_MAX_SAMPLES_PER_SCENE", "0"))

# ─────────────────────────────────────────────────────────────────────────────
# 多进程
# ─────────────────────────────────────────────────────────────────────────────
N_FUSION_WORKERS = int(os.environ.get("FUSION_N_WORKERS", str(max(1, (_mp.cpu_count() or 4) - 1))))

# ─────────────────────────────────────────────────────────────────────────────
# 调试 / 日志
# ─────────────────────────────────────────────────────────────────────────────
FUSION_LOG_PIXEL_STATS = os.environ.get("FUSION_LOG_PIXEL_STATS", "1") == "1"

_qc_diag_raw = os.environ.get(
    "ENABLE_QC_DIAGNOSTICS",
    os.environ.get("FUSION_ENABLE_QC_DIAGNOSTICS", "0"),
)
ENABLE_QC_DIAGNOSTICS = _qc_diag_raw.strip().lower() in {"1", "true", "yes", "on"}
QC_DIAGNOSTICS_DIR = os.environ.get("FUSION_QC_DIAGNOSTICS_DIR", "runs/qc_diagnostics_gpm")
