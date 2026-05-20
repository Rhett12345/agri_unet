"""
train.py
========
Training loop for AGRI → GPM precipitation classification.

Features
--------
- Mixed-precision (AMP) training via torch.cuda.amp
- Gradient clipping
- ReduceLROnPlateau scheduler
- Weighted CrossEntropy (with Focal Loss support)
- Multi-checkpoint saving (best loss, best OA, best F1_class3)
- Per-epoch CSV log

Usage:
    python train.py
"""

import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.amp import GradScaler, autocast

import config as cfg
from dataset import NormStats, build_dataloaders
from losses import build_loss
from model import build_model

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def _seed_everything(seed: int = cfg.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _batch_metrics(logits, labels, device):
    """
    logits: (B, C, H, W)
    labels: (B, H, W) with integer class labels
    Returns dict with per-class accuracy and counts for F1.
    """
    preds = logits.argmax(dim=1)
    valid = (labels >= 0) & (labels < cfg.NUM_CLASSES)

    C = cfg.NUM_CLASSES
    tp = torch.zeros(C, dtype=torch.int64)
    fp = torch.zeros(C, dtype=torch.int64)
    fn = torch.zeros(C, dtype=torch.int64)

    if valid.any():
        for c in range(C):
            tp[c] = ((preds[valid] == c) & (labels[valid] == c)).sum().item()
            fp[c] = ((preds[valid] == c) & (labels[valid] != c)).sum().item()
            fn[c] = ((preds[valid] != c) & (labels[valid] == c)).sum().item()

    correct = int((preds[valid] == labels[valid]).sum().item())
    total = int(valid.sum().item())
    oa = (correct / total * 100.0) if total > 0 else 0.0

    return {
        "oa": oa,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_valid": total,
    }


def _compute_f1(tp_sum, fp_sum, fn_sum):
    """Compute per-class and macro F1 from accumulated counts."""
    C = len(tp_sum)
    f1_per_class = []
    for c in range(C):
        denom = float(tp_sum[c] + 0.5 * (fp_sum[c] + fn_sum[c]))
        if denom > 0:
            f1_per_class.append(float(tp_sum[c]) / denom * 100.0)
        else:
            f1_per_class.append(0.0)
    return f1_per_class


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metric_value(metrics, monitor: str):
    monitor = (monitor or "val_loss").lower()
    if monitor in {"val_loss", "loss"}:
        return float(metrics.get("loss", 1e9))
    if monitor in {"val_oa", "oa"}:
        return float(metrics.get("oa", 0.0))
    if monitor == "val_f1_class3":
        return float(metrics.get("f1_class3", 0.0))
    if monitor == "val_macro_f1":
        return float(metrics.get("macro_f1", 0.0))
    raise ValueError(f"Unsupported monitor: {monitor}")


def _monitor_mode(monitor: str) -> str:
    return "min" if (monitor or "").lower() in {"val_loss", "loss"} else "max"


def _is_better(candidate, current, monitor: str) -> bool:
    if current is None:
        return True
    cv = _metric_value(candidate, monitor)
    cv_cur = _metric_value(current, monitor)
    if _monitor_mode(monitor) == "min":
        return cv < cv_cur
    return cv > cv_cur


# ─────────────────────────────────────────────────────────────────────────────
# Epoch runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    C = cfg.NUM_CLASSES
    totals = {"loss": 0.0, "n": 0}
    tp_sum = torch.zeros(C, dtype=torch.int64)
    fp_sum = torch.zeros(C, dtype=torch.int64)
    fn_sum = torch.zeros(C, dtype=torch.int64)
    n_correct = 0
    n_total = 0

    total_batches = len(loader)
    log_interval = max(1, total_batches // 10)

    for batch_idx, (agri, geo, labels) in enumerate(loader):
        agri   = agri.to(device)
        geo    = geo.to(device)
        labels = labels.to(device)

        with autocast(device.type if scaler else "cpu", enabled=(scaler is not None)):
            logits = model(agri, geo=geo)
            loss = loss_fn(logits, labels)

        if training:
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()

        m = _batch_metrics(logits, labels, device)
        B = agri.shape[0]
        totals["loss"] += loss.item() * B
        totals["n"]   += B
        n_correct     += m["oa"] / 100.0 * m["n_valid"]
        n_total       += m["n_valid"]
        tp_sum += m["tp"]
        fp_sum += m["fp"]
        fn_sum += m["fn"]

        if (batch_idx + 1) % log_interval == 0 or batch_idx == total_batches - 1:
            cur_n = max(totals["n"], 1)
            f1 = _compute_f1(tp_sum, fp_sum, fn_sum)
            tag = "train" if training else "val"
            log.info("  %s %d/%d | loss=%.4f | OA=%.1f%% | F1_c3=%.1f%%",
                     tag, batch_idx + 1, total_batches,
                     totals["loss"] / cur_n,
                     (n_correct / max(n_total, 1)) * 100.0,
                     f1[3])

    N = max(totals["n"], 1)
    f1_per = _compute_f1(tp_sum, fp_sum, fn_sum)
    macro_f1 = float(np.mean(f1_per)) if f1_per else 0.0

    result = {
        "loss": totals["loss"] / N,
        "oa": (n_correct / max(n_total, 1)) * 100.0,
        "macro_f1": macro_f1,
    }
    for c in range(C):
        result[f"f1_class{c}"] = f1_per[c]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(stats: NormStats):
    _seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on %s", device)

    train_dl, val_dl, _ = build_dataloaders(stats)

    log.info("train samples = %d", len(train_dl.dataset))
    log.info("val samples   = %d", len(val_dl.dataset))
    log.info("train iters/epoch = %d", len(train_dl))
    log.info("val iters/epoch   = %d", len(val_dl))

    model     = build_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.LR_FACTOR,
        patience=cfg.LR_PATIENCE, min_lr=cfg.MIN_LR,
    )
    scaler = GradScaler(device.type) if torch.cuda.is_available() else None

    # ── Compute class distribution for loss weights ──
    log.info("Computing class distribution from training data...")
    class_counts = torch.zeros(cfg.NUM_CLASSES, dtype=torch.float32)
    file_samples = defaultdict(list)
    for idx in train_dl.dataset._index:
        h5f, s_idx = idx
        file_samples[h5f].append(s_idx)

    import h5py
    n_scanned = 0
    total_samples = len(train_dl.dataset)
    for h5f, s_list in file_samples.items():
        try:
            with h5py.File(h5f, "r") as f:
                for s_idx in s_list:
                    lbl = int(f["Samples/label"][s_idx])
                    if 0 <= lbl < cfg.NUM_CLASSES:
                        class_counts[lbl] += 1.0
                    n_scanned += 1
        except Exception:
            n_scanned += len(s_list)
    log.info("Class distribution: %d/%d samples scanned", n_scanned, total_samples)

    total = class_counts.sum()
    pcts = [(class_counts[c] / total * 100).item() if total > 0 else 0.0
            for c in range(cfg.NUM_CLASSES)]
    log.info("Precip class distribution: %s",
             " | ".join(f"{cfg.PRECIP_CLASS_NAMES[c]}={pcts[c]:.1f}%" for c in range(cfg.NUM_CLASSES)))

    # Class weights: inverse frequency, capped at 10
    class_weights = torch.ones(cfg.NUM_CLASSES, dtype=torch.float32)
    valid_classes = class_counts > 0
    if valid_classes.any():
        freq = class_counts[valid_classes] / class_counts[valid_classes].sum().clamp_min(1.0)
        raw = 1.0 / freq.clamp_min(1e-6)
        class_weights[valid_classes] = raw.clamp_max(10.0)
    log.info("Class weights: %s",
             " | ".join(f"c{c}={class_weights[c].item():.2f}" for c in range(cfg.NUM_CLASSES)))

    # Build loss
    loss_type = getattr(cfg, "LOSS_TYPE", "weighted_ce")
    loss_fn = build_loss(loss_type, class_weights, device)
    log.info("Loss: %s", loss_type)

    # ── Training state ──
    monitor = getattr(cfg, "CHECKPOINT_MONITOR", "val_f1_class3")
    best_selected = None
    best_loss = None
    best_oa = None
    best_f1_c3 = None
    epochs_no_best = 0
    log_rows = []

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        train_m = _run_epoch(model, train_dl, loss_fn, device,
                             optimizer=optimizer, scaler=scaler)
        val_m   = _run_epoch(model, val_dl,   loss_fn, device)

        # Scheduler step on val macro F1
        scheduler.step(val_m["macro_f1"])
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        f1_str = " ".join(
            f"c{c}={val_m.get(f'f1_class{c}', 0):.1f}"
            for c in range(cfg.NUM_CLASSES)
        )
        log.info(
            "Epoch %3d/%d | TrainLoss=%.4f | ValLoss=%.4f | OA=%.2f%% | "
            "MacroF1=%.2f%% | LR=%.2e | %s | %.1fs",
            epoch, cfg.NUM_EPOCHS,
            train_m["loss"], val_m["loss"], val_m["oa"],
            val_m["macro_f1"], lr_now, f1_str, elapsed,
        )

        # ── Checkpoints ──
        if _is_better(val_m, best_loss, "val_loss"):
            best_loss = dict(val_m)
            torch.save(model.state_dict(), cfg.CHECKPOINT_BEST_LOSS)
            if monitor == "val_loss":
                torch.save(model.state_dict(), cfg.CHECKPOINT_BEST)

        if _is_better(val_m, best_oa, "val_oa"):
            best_oa = dict(val_m)
            torch.save(model.state_dict(), cfg.CHECKPOINT_BEST_OA)
            if monitor == "val_oa":
                torch.save(model.state_dict(), cfg.CHECKPOINT_BEST)

        if _is_better(val_m, best_f1_c3, "val_f1_class3"):
            best_f1_c3 = dict(val_m)
            torch.save(model.state_dict(), cfg.CHECKPOINT_BEST_F1_C3)
            if monitor == "val_f1_class3":
                torch.save(model.state_dict(), cfg.CHECKPOINT_BEST)

        is_best = _is_better(val_m, best_selected, monitor)
        if is_best:
            best_selected = dict(val_m)
            epochs_no_best = 0
            torch.save(model.state_dict(), cfg.CHECKPOINT_BEST)
            log.info("  New best %s: %.6f (OA=%.2f%% MacroF1=%.2f%% F1_c3=%.2f%%) → saved %s",
                     monitor, _metric_value(val_m, monitor), val_m["oa"],
                     val_m["macro_f1"], val_m.get("f1_class3", 0), cfg.CHECKPOINT_BEST.name)
        else:
            epochs_no_best += 1

        torch.save(model.state_dict(), cfg.CHECKPOINT_LAST)

        # ── CSV log ──
        row = dict(epoch=epoch, lr=lr_now,
                   **{f"train_{k}": v for k, v in train_m.items()},
                   **{f"val_{k}": v for k, v in val_m.items()})
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(cfg.LOG_DIR / "train_log.csv", index=False)

        # ── Early stopping ──
        if epochs_no_best >= cfg.EARLY_STOP_PATIENCE:
            log.info("Early stopping at epoch %d", epoch)
            break

    if best_selected is not None:
        log.info("Training complete. Best %s: %.6f, OA: %.2f%%, MacroF1: %.2f%%, F1_c3: %.2f%%",
                 monitor, _metric_value(best_selected, monitor),
                 best_selected["oa"], best_selected["macro_f1"],
                 best_selected.get("f1_class3", 0))
