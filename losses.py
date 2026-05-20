"""
losses.py
=========
Loss functions for precipitation classification.

- WeightedCrossEntropyLoss: standard CE with class weights
- FocalLoss: CE with (1-p)^γ modulation for class imbalance

Usage:
    from losses import build_loss
    loss_fn = build_loss("weighted_ce", class_weights, device)
    loss_fn = build_loss("focal", class_weights, device, gamma=2.0, alpha=None)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss with class weights, ignoring NaN targets."""

    def __init__(self, weight: torch.Tensor, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W)
        targets: (B, H, W) with integer labels, NaN where ignore
        """
        # Flatten to (B*H*W, C) and (B*H*W,)
        B, C, H, W = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1).long()

        valid = torch.isfinite(targets_flat.float()) & (targets_flat >= 0)
        if not valid.any():
            return logits.sum() * 0.0

        logits_valid = logits_flat[valid]
        targets_valid = targets_flat[valid].clamp(0, C - 1)

        return F.cross_entropy(
            logits_valid, targets_valid,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Parameters
    ----------
    gamma : float, default=2.0
        Focusing parameter. Higher gamma → more focus on hard examples.
    alpha : Tensor or None
        Class weights. If None, no class weighting.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
    ):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1).long()

        valid = torch.isfinite(targets_flat.float()) & (targets_flat >= 0)
        if not valid.any():
            return logits.sum() * 0.0

        logits_valid = logits_flat[valid]
        targets_valid = targets_flat[valid].clamp(0, C - 1)

        ce_loss = F.cross_entropy(logits_valid, targets_valid, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = (1.0 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets_valid]
            focal = alpha_t * focal

        return focal.mean()


def build_loss(
    loss_type: str,
    class_weights: torch.Tensor = None,
    device: torch.device = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """
    Build loss function by name.

    Parameters
    ----------
    loss_type : "weighted_ce" or "focal"
    class_weights : (C,) Tensor or None
    device : torch.device
    gamma : Focal Loss gamma
    label_smoothing : label smoothing for CE

    Returns
    -------
    nn.Module loss function
    """
    if class_weights is not None and device is not None:
        class_weights = class_weights.to(device)

    loss_type = loss_type.lower()
    if loss_type == "weighted_ce":
        return WeightedCrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    elif loss_type == "focal":
        return FocalLoss(gamma=gamma, alpha=class_weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'weighted_ce' or 'focal'.")
