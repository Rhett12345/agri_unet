"""
train.py
========
Training loop for AGRI → GPM precipitation classification (CNN+Transformer).

Features
--------
- Per-epoch random 20% subsample with class-balanced WeightedRandomSampler
- Mixed-precision (AMP) training
- Hardcoded class loss weights [1, 3, 5, 10]
- Multi-checkpoint saving
- Per-epoch CSV log
"""

import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

import config as cfg
from dataset import NormStats, PrecipDataset, _filter_h5_files_by_dates
from model import build_model

log = logging.getLogger(__name__)


def _seed_everything(seed: int = cfg.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Build dataloaders with class-balanced subsampling (train only)
# ─────────────────────────────────────────────────────────────────────────────

def _build_balanced_train_dl(stats: NormStats):
    """Build train DataLoader with WeightedRandomSampler.

    Scans all HDF5 labels once (merged count + label storage), computes
    per-class weights, and caches sample_weights to avoid re-scanning.
    """
    train_ds = PrecipDataset(cfg.PAIRED_TRAIN_DIR, stats, mode="train")
    total = len(train_ds)
    log.info("train samples (total) = %d", total)

    cache_path = cfg.STATS_DIR / "sample_weights_cache.npz"

    # ── Try cache first ──
    if cache_path.exists():
        c = np.load(cache_path)
        if int(c["n_samples"]) == total:
            sample_weights = c["sample_weights"]
            class_counts = c["class_counts"]
            log.info("Loaded cached sample weights (%d samples)", total)
        else:
            log.info("Cache stale (n_samples mismatch), re-scanning...")
            cache_path.unlink(missing_ok=True)
            sample_weights = None
    else:
        sample_weights = None

    # ── Scan (if needed) ──
    if sample_weights is None:
        log.info("Scanning labels (single pass, merged count + weight)...")

        # Group sample indices by file
        file_samples = defaultdict(list)
        for h5f, s_idx in train_ds._index:
            file_samples[h5f].append(s_idx)

        # Single pass: count classes + store labels
        class_counts = np.zeros(cfg.NUM_CLASSES, dtype=np.int64)
        all_labels = np.full(total, -1, dtype=np.int8)
        idx = 0
        n_scanned = 0
        for h5f, s_list in file_samples.items():
            try:
                with h5py.File(h5f, "r") as f:
                    labels_arr = f["Samples/label"][()]
                    for s_idx in s_list:
                        lbl = int(labels_arr[s_idx])
                        if 0 <= lbl < cfg.NUM_CLASSES:
                            class_counts[lbl] += 1
                        all_labels[idx] = lbl
                        idx += 1
                        n_scanned += 1
            except Exception:
                idx += len(s_list)
                n_scanned += len(s_list)
                continue
            if n_scanned % 5000000 == 0 or n_scanned == total:
                log.info("  scanned %d/%d", n_scanned, total)

        log.info("Class counts: %s", dict(enumerate(class_counts.tolist())))
        pcts = class_counts / class_counts.sum() * 100
        log.info("Class %%: %s",
                 " | ".join(f"{cfg.PRECIP_CLASS_NAMES[c]}={pcts[c]:.1f}%" for c in range(cfg.NUM_CLASSES)))

        # Compute per-class sampling weights
        target = np.array(cfg.TARGET_RATIOS)
        actual = class_counts / class_counts.sum().clip(1e-9)
        w_per_class = target / actual.clip(1e-9)

        # Build sample_weights from stored labels (no file I/O)
        log.info("Building sample weights from cached labels...")
        sample_weights = np.zeros(total, dtype=np.float32)
        for i in range(total):
            lbl = all_labels[i]
            if 0 <= lbl < cfg.NUM_CLASSES:
                sample_weights[i] = w_per_class[lbl]

        sample_weights = np.maximum(sample_weights, 0.0)
        if sample_weights.sum() == 0:
            sample_weights[:] = 1.0

        # Cache to disk
        cfg.STATS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path,
                           sample_weights=sample_weights,
                           class_counts=class_counts,
                           n_samples=total)
        log.info("Cached sample weights → %s", cache_path)

    else:
        # Recompute w_per_class from cached counts for logging
        target = np.array(cfg.TARGET_RATIOS)
        actual = class_counts / class_counts.sum().clip(1e-9)
        w_per_class = target / actual.clip(1e-9)

    # Num samples per epoch = 20% of total
    n_epoch = max(1, int(total * cfg.SUBSAMPLE_FRAC))
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=n_epoch,
        replacement=True,
    )
    log.info("Sampler: %d samples/epoch (%.0f%%), class weights=%s",
             n_epoch, cfg.SUBSAMPLE_FRAC * 100,
             [f"{w:.2f}" for w in w_per_class])

    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                          sampler=sampler,
                          pin_memory=True, num_workers=cfg.NUM_WORKERS,
                          persistent_workers=True, prefetch_factor=4)
    return train_dl


def _build_val_test_dls(stats: NormStats):
    """Build val/test DataLoaders (no subsampling)."""
    val_ds  = PrecipDataset(cfg.PAIRED_VAL_DIR, stats, mode="val")
    test_ds = PrecipDataset(cfg.PAIRED_TEST_DIR, stats, mode="test")

    common = dict(pin_memory=True, num_workers=cfg.NUM_WORKERS,
                  persistent_workers=True, prefetch_factor=4)
    val_dl  = DataLoader(val_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, **common)
    test_dl = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, **common)

    log.info("val samples = %d  test samples = %d", len(val_ds), len(test_ds))
    return val_dl, test_dl


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _batch_metrics(logits, labels, device):
    """logits: (B, C), labels: (B,)"""
    preds = logits.argmax(dim=1)
    valid = (labels >= 0) & (labels < cfg.NUM_CLASSES)

    C = cfg.NUM_CLASSES
    tp = np.zeros(C, dtype=np.int64)
    fp = np.zeros(C, dtype=np.int64)
    fn = np.zeros(C, dtype=np.int64)

    if valid.any():
        p = preds[valid].cpu().numpy()
        t = labels[valid].cpu().numpy()
        correct = int((p == t).sum())
        total = int(valid.sum())
        for c in range(C):
            tp[c] = int(((p == c) & (t == c)).sum())
            fp[c] = int(((p == c) & (t != c)).sum())
            fn[c] = int(((p != c) & (t == c)).sum())
    else:
        correct = 0
        total = 0
    oa = (correct / total * 100.0) if total > 0 else 0.0
    return {"oa": oa, "tp": tp, "fp": fp, "fn": fn, "n_valid": total}


def _compute_f1(tp, fp, fn):
    C = len(tp)
    f1 = []
    for c in range(C):
        denom = float(tp[c] + 0.5 * (fp[c] + fn[c]))
        f1.append(float(tp[c]) / denom * 100.0 if denom > 0 else 0.0)
    return f1


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
    return 0.0


def _monitor_mode(monitor: str) -> str:
    return "min" if (monitor or "").lower() in {"val_loss", "loss"} else "max"


def _is_better(candidate, current, monitor: str) -> bool:
    if current is None:
        return True
    cv = _metric_value(candidate, monitor)
    cv_cur = _metric_value(current, monitor)
    return cv < cv_cur if _monitor_mode(monitor) == "min" else cv > cv_cur


# ─────────────────────────────────────────────────────────────────────────────
# Epoch runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    C = cfg.NUM_CLASSES
    totals = {"loss": 0.0, "n": 0}
    tp_sum = np.zeros(C, dtype=np.int64)
    fp_sum = np.zeros(C, dtype=np.int64)
    fn_sum = np.zeros(C, dtype=np.int64)
    n_correct = 0
    n_total = 0

    total_batches = len(loader)
    log_interval = max(1, total_batches // 10)

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with autocast(device.type if scaler else "cpu", enabled=(scaler is not None)):
            logits = model(x)
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
        B = x.shape[0]
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
    macro_f1 = float(np.mean(f1_per))

    result = {
        "loss": totals["loss"] / N,
        "oa": (n_correct / max(n_total, 1)) * 100.0,
        "macro_f1": macro_f1,
    }
    for c in range(C):
        result[f"f1_class{c}"] = f1_per[c]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(stats: NormStats):
    _seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on %s", device)

    train_dl = _build_balanced_train_dl(stats)
    val_dl, test_dl = _build_val_test_dls(stats)

    log.info("train iters/epoch = %d", len(train_dl))
    log.info("val iters/epoch   = %d", len(val_dl))

    model     = build_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.LR_FACTOR,
        patience=cfg.LR_PATIENCE, min_lr=cfg.MIN_LR,
    )
    scaler = GradScaler(device.type) if torch.cuda.is_available() else None

    class_weights = torch.tensor(cfg.CLASS_WEIGHTS, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    log.info("Loss weights: %s", cfg.CLASS_WEIGHTS)

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

        # Rebuild train sampler each epoch (different 20% subset)
        if epoch > 1:
            del train_dl
            train_dl = _build_balanced_train_dl(stats)

        train_m = _run_epoch(model, train_dl, loss_fn, device,
                             optimizer=optimizer, scaler=scaler)
        val_m   = _run_epoch(model, val_dl,   loss_fn, device)

        scheduler.step(val_m["f1_class3"])
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        f1_str = " ".join(f"c{c}={val_m.get(f'f1_class{c}', 0):.1f}" for c in range(cfg.NUM_CLASSES))
        log.info(
            "Epoch %3d/%d | TrainLoss=%.4f | ValLoss=%.4f | OA=%.2f%% | "
            "MacroF1=%.2f%% | %s | %.1fs",
            epoch, cfg.NUM_EPOCHS,
            train_m["loss"], val_m["loss"], val_m["oa"],
            val_m["macro_f1"], f1_str, elapsed,
        )

        # Checkpoints
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
            log.info("  New best %s: %.6f OA=%.2f%% F1_c3=%.1f%% → %s",
                     monitor, _metric_value(val_m, monitor), val_m["oa"],
                     val_m.get("f1_class3", 0), cfg.CHECKPOINT_BEST.name)
        else:
            epochs_no_best += 1

        torch.save(model.state_dict(), cfg.CHECKPOINT_LAST)

        row = dict(epoch=epoch, lr=lr_now,
                   **{f"train_{k}": v for k, v in train_m.items()},
                   **{f"val_{k}": v for k, v in val_m.items()})
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(cfg.LOG_DIR / "train_log.csv", index=False)

        if epochs_no_best >= cfg.EARLY_STOP_PATIENCE:
            log.info("Early stopping at epoch %d", epoch)
            break

    log.info("Training complete. Best %s OA:%.2f%% MacroF1:%.2f%% F1_c3:%.1f%%",
             monitor,
             best_selected["oa"] if best_selected else 0,
             best_selected["macro_f1"] if best_selected else 0,
             best_selected.get("f1_class3", 0) if best_selected else 0)
