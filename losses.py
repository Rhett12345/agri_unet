"""
losses.py
=========
Loss function builder with class weights.

Usage:
    from losses import build_loss
    loss_fn = build_loss(class_weights_tensor)
"""

import torch
import torch.nn as nn


def build_loss(class_weights: torch.Tensor = None) -> nn.Module:
    """
    Build CrossEntropyLoss with optional class weights.

    Parameters
    ----------
    class_weights : (C,) Tensor or None
    """
    if class_weights is not None:
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    return nn.CrossEntropyLoss(ignore_index=-100)
