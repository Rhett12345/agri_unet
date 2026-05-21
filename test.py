"""
test.py
=======
Evaluation of precipitation classification model on the held-out test set.

Metrics reported
----------------
  OA              : Overall Accuracy
  F1 per class    : F1_class0 ~ F1_class3
  HSS             : Heidke Skill Score
  ETS             : Equitable Threat Score
  Confusion matrix: numerical + image
  Classification report: precision, recall, f1, support

Outputs saved to cfg.EVAL_OUTPUT_DIR:
  - metrics_summary.csv
  - confusion_matrix.{svg,pdf,png}
  - classification_report.csv

Usage:
    python test.py [--checkpoint <path>]
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report

import config as cfg
from dataset import NormStats, build_test_dataloader
from model import build_model

log = logging.getLogger(__name__)

CLASS_NAMES = list(cfg.PRECIP_CLASS_NAMES)

# ── Style ──
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
})

C_BLUE   = "#0F4D92"
C_GREEN  = "#2E9E44"
C_RED    = "#E53935"
C_TEAL   = "#42949E"
C_ORANGE = "#E8871D"
C_NEUTRAL = "#767676"
C_LIGHT  = "#CFCECE"


# ─────────────────────────────────────────────────────────────────────────────
# Skill scores
# ─────────────────────────────────────────────────────────────────────────────

def _compute_hss(cm: np.ndarray) -> float:
    """
    Heidke Skill Score.
    HSS = (Σ_correct - Σ_expected) / (N - Σ_expected)
    where expected = sum over classes of (N_i_obs * N_i_fcst) / N
    """
    N = cm.sum()
    if N == 0:
        return 0.0
    correct = cm.diagonal().sum()
    expected = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / N
    denom = N - expected
    if denom == 0:
        return 0.0
    return (correct - expected) / denom


def _compute_ets(cm: np.ndarray) -> float:
    """
    Equitable Threat Score (Gilbert Skill Score).
    ETS = (correct - expected) / (N + Σ_hits_per_class - expected)
    where expected = sum over classes of (N_i_obs * N_i_fcst) / N
    """
    N = cm.sum()
    if N == 0:
        return 0.0
    correct = cm.diagonal().sum()
    expected = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / N
    hits_per_class = cm.diagonal().sum()
    denom = N - expected
    if denom == 0:
        return 0.0
    return (correct - expected) / (N + correct - expected)


def _compute_f1_per_class(cm: np.ndarray) -> np.ndarray:
    """Per-class F1 from confusion matrix."""
    n_cls = cm.shape[0]
    f1 = np.zeros(n_cls)
    for c in range(n_cls):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + 0.5 * (fp + fn)
        if denom > 0:
            f1[c] = tp / denom * 100.0
    return f1


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _save_pub(fig, base_path):
    base = str(base_path).replace(".png", "")
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    log.info("Saved → %s.{svg,pdf,png}", base)


def _plot_confusion_matrix(cm: np.ndarray, out_path: Path):
    n_cls = cm.shape[0]
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(1e-9) * 100  # row-normalized

    fig, ax = plt.subplots(figsize=(90 / 25.4, 75 / 25.4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100, aspect="auto")

    for i in range(n_cls):
        for j in range(n_cls):
            pct = cm_norm[i, j]
            cnt = cm[i, j]
            color = "white" if pct > 50 else "black"
            ax.text(j, i - 0.15, f"{cnt:,}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")
            ax.text(j, i + 0.18, f"({pct:.1f}%)", ha="center", va="center",
                    fontsize=5.5, color=color)

    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels(CLASS_NAMES, fontsize=6)
    ax.set_yticklabels(CLASS_NAMES, fontsize=6)
    ax.set_xlabel("Predicted", fontsize=6.5)
    ax.set_ylabel("True", fontsize=6.5)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row %", fontsize=5.5)

    _save_pub(fig, out_path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def collect_test_predictions(
    stats: NormStats,
    checkpoint: Path,
    test_dl=None,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    log.info("Loaded checkpoint %s", checkpoint)
    model.eval()

    if test_dl is None:
        test_dl = build_test_dataloader(stats)

    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, labels in test_dl:
            x = x.to(device)
            labels = labels.to(device)

            logits = model(x)              # (B, 4)
            preds = logits.argmax(dim=1)   # (B,)

            valid = (labels >= 0) & (labels < cfg.NUM_CLASSES)
            if valid.any():
                all_true.append(labels[valid].cpu().numpy())
                all_pred.append(preds[valid].cpu().numpy())

    return {
        "y_true": np.concatenate(all_true) if all_true else np.array([], dtype=np.int64),
        "y_pred": np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64),
    }


def evaluate(stats: NormStats, checkpoint: Optional[Path] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Evaluating on %s", device)

    checkpoint = checkpoint or cfg.CHECKPOINT_BEST
    try:
        arrays = collect_test_predictions(stats, checkpoint, device=device)
    except FileNotFoundError:
        log.error("Checkpoint not found: %s", checkpoint)
        return

    y_true = arrays["y_true"]
    y_pred = arrays["y_pred"]

    if len(y_true) == 0:
        log.warning("No valid predictions found")
        return

    # ── Confusion matrix ──
    cm = confusion_matrix(y_true, y_pred, labels=list(range(cfg.NUM_CLASSES)))
    n_cls = cm.shape[0]

    # ── OA ──
    oa = float((y_true == y_pred).mean() * 100.0)

    # ── Per-class F1 ──
    f1_per = _compute_f1_per_class(cm)

    # ── HSS ──
    hss = _compute_hss(cm)

    # ── ETS ──
    ets = _compute_ets(cm)

    # ── Classification report ──
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                    labels=list(range(n_cls)), zero_division=0,
                                    output_dict=True)

    # ── Print summary ──
    log.info("─" * 60)
    log.info("Overall Accuracy (OA): %.2f%%", oa)
    log.info("Heidke Skill Score (HSS): %.4f", hss)
    log.info("Equitable Threat Score (ETS): %.4f", ets)
    log.info("─" * 60)
    for c, name in enumerate(CLASS_NAMES):
        sup = int(cm[c].sum())
        log.info("  %-14s  F1=%.2f%%  Prec=%.2f%%  Rec=%.2f%%  Support=%d",
                 name,
                 f1_per[c],
                 report[name]["precision"] * 100 if name in report else 0,
                 report[name]["recall"] * 100 if name in report else 0,
                 sup)
    log.info("─" * 60)
    log.info("Macro F1: %.2f%%", float(np.mean(f1_per)))
    log.info("─" * 60)

    # ── Save outputs ──
    cfg.EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        {"metric": "OA",          "value": oa,    "unit": "%"},
        {"metric": "HSS",         "value": hss,   "unit": "score"},
        {"metric": "ETS",         "value": ets,   "unit": "score"},
        {"metric": "Macro_F1",    "value": float(np.mean(f1_per)), "unit": "%"},
        {"metric": "Total_samples","value": len(y_true), "unit": "count"},
    ]
    for c, name in enumerate(CLASS_NAMES):
        rows.append({"metric": f"F1_{name.replace(' ','_')}", "value": f1_per[c], "unit": "%"})
        rows.append({"metric": f"Precision_{name.replace(' ','_')}",
                     "value": report[name]["precision"] * 100 if name in report else 0, "unit": "%"})
        rows.append({"metric": f"Recall_{name.replace(' ','_')}",
                     "value": report[name]["recall"] * 100 if name in report else 0, "unit": "%"})
        rows.append({"metric": f"Support_{name.replace(' ','_')}",
                     "value": int(cm[c].sum()), "unit": "count"})

    pd.DataFrame(rows).to_csv(cfg.EVAL_OUTPUT_DIR / "metrics_summary.csv", index=False)

    # ── Classification report CSV ──
    report_rows = []
    for c, name in enumerate(CLASS_NAMES):
        if name in report:
            report_rows.append({
                "class": name,
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1-score": report[name]["f1-score"],
                "support": int(report[name]["support"]),
            })
    pd.DataFrame(report_rows).to_csv(cfg.EVAL_OUTPUT_DIR / "classification_report.csv", index=False)

    _plot_confusion_matrix(cm, cfg.EVAL_OUTPUT_DIR / "confusion_matrix")

    log.info("Evaluation complete – results in %s", cfg.EVAL_OUTPUT_DIR)


def main():
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )
    parser = argparse.ArgumentParser(description="Evaluate Precipitation Classification Model")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    stats = NormStats.load(cfg.STATS_FILE)
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    evaluate(stats, ckpt)


if __name__ == "__main__":
    main()
