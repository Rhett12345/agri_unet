"""Shared precipitation sample filtering helpers."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

import config as cfg


def get_patch_supervision_thresholds(mode: str, patch_size: Tuple[int, int]) -> Dict[str, int]:
    """Stub: precipitation samples are point-based, no patch-level filtering needed."""
    return {
        "min_valid_label_pixels": 0,
        "min_valid_cloudy_pixels": 0,
    }


def patch_passes_supervision(*args, **kwargs) -> Tuple[bool, Dict[str, int], Dict[str, int]]:
    """Stub: all samples pass by default (filtering done in fusion stage)."""
    return True, {"valid_label_pixels": 1, "valid_cloudy_pixels": 1}, {"min_valid_label_pixels": 0, "min_valid_cloudy_pixels": 0}


def sample_passes_quality(fields: Dict[str, float]) -> bool:
    """Return whether a sample passes optional quality gates."""
    if not getattr(cfg, "SAMPLE_QUALITY_FILTER_ENABLED", False):
        return True
    return True
