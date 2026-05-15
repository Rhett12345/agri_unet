"""
AGRI L2 CLP vs MODIS 验证可视化 — 双方案合并为一张 2×2 图。

用法:
    conda run -n cloudunet python tools/plot_validation_figures.py
输出:
    tools/validation_report.{svg,pdf,tiff,png}
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# ═══════════════════════════════════════════════════════════════════════
# Nature 风格设置
# ═══════════════════════════════════════════════════════════════════════
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
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 3, "ytick.major.size": 3,
})

# ═══════════════════════════════════════════════════════════════════════
# 色板
# ═══════════════════════════════════════════════════════════════════════
C_BLUE    = "#0F4D92"
C_GREEN   = "#2E9E44"
C_RED     = "#E53935"
C_TEAL    = "#42949E"
C_ORANGE  = "#E8871D"
C_NEUTRAL = "#767676"
C_LIGHT   = "#CFCECE"
C_BG      = "#F5F5F5"

# ═══════════════════════════════════════════════════════════════════════
# 数据 — 改这里刷新全图
# ═══════════════════════════════════════════════════════════════════════

SCENES = [
    {"ts": "03:00", "l2_cf": 69.2, "cm_cf": 92.2, "oa": 88.1, "n": 203862},
    {"ts": "06:00", "l2_cf": 71.1, "cm_cf": 90.1, "oa": 86.7, "n":  97755},
    {"ts": "06:15", "l2_cf": 70.9, "cm_cf": 47.2, "oa": 47.1, "n": 446605},
    {"ts": "08:00", "l2_cf": 69.0, "cm_cf": 63.2, "oa": 50.3, "n": 364524},
    {"ts": "15:15", "l2_cf": 64.1, "cm_cf": 70.6, "oa": 64.6, "n": 156425},
    {"ts": "17:00", "l2_cf": 83.3, "cm_cf": 68.5, "oa": 56.3, "n": 393832},
    {"ts": "22:00", "l2_cf": 65.5, "cm_cf": 46.2, "oa": 41.2, "n": 209806},
]

# Pooled 混淆矩阵: [[L2=Cloud & MODIS=Cloud,  L2=Cloud & MODIS=Clear],
#                   [L2=Clear & MODIS=Cloud,  L2=Clear & MODIS=Clear]]
CM = np.array([[543008, 538363],
               [210412, 136200]])

# ── 衍生指标 ──
tp, fp = CM[0, 0], CM[0, 1]
fn, tn = CM[1, 0], CM[1, 1]
CDR = tp / (tp + fn) * 100
FAR = fp / (fp + tn) * 100
OA  = (tp + tn) / CM.sum() * 100

# ── 分层 ──
HIGH_CF = [s for s in SCENES if s["cm_cf"] > 60]
LOW_CF  = [s for s in SCENES if s["cm_cf"] <= 60]

def _pooled_oa(ss):
    total_n = sum(s["n"] for s in ss)
    if total_n == 0:
        return 0.0
    return sum(int(s["n"] * s["oa"] / 100) for s in ss) / total_n * 100

OA_HIGH = _pooled_oa(HIGH_CF)
OA_LOW  = _pooled_oa(LOW_CF)


# ═══════════════════════════════════════════════════════════════════════
# 主图
# ═══════════════════════════════════════════════════════════════════════

def main():
    fig = plt.figure(figsize=(170 / 25.4, 130 / 25.4))
    gs = GridSpec(2, 2, wspace=0.35, hspace=0.45,
                  left=0.09, right=0.97, top=0.93, bottom=0.08)

    # ── a: CDR & FAR 堆叠水平条 (方案一核心) ──
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_cdr_far(ax_a)
    _label(ax_a, "a")

    # ── b: 混淆矩阵热力图 (方案一支撑) ──
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_cm(ax_b)
    _label(ax_b, "b")

    # ── c: 分层 OA (方案二核心) ──
    ax_c = fig.add_subplot(gs[1, 0])
    _draw_stratified_oa(ax_c)
    _label(ax_c, "c")

    # ── d: L2 vs MODIS 云量散点 + 逐场景 OA (方案二支撑) ──
    ax_d = fig.add_subplot(gs[1, 1])
    _draw_cf_scatter(ax_d)
    _label(ax_d, "d")

    out = Path(__file__).resolve().parent / "validation_report"
    for fmt, kw in [("svg", {}), ("pdf", {}),
                     ("tiff", {"dpi": 600}), ("png", {"dpi": 600})]:
        fig.savefig(f"{out}.{fmt}", bbox_inches="tight", **kw)
    print(f"Saved: {out}.{{svg,pdf,tiff,png}}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 子图
# ═══════════════════════════════════════════════════════════════════════

def _label(ax, s):
    ax.set_title(s, fontweight="bold", fontsize=8, loc="left", pad=4)


def _draw_cdr_far(ax):
    """CDR & FAR 堆叠水平条"""
    labels = ["CDR", "FAR"]
    vals   = [CDR, FAR]
    colors = [C_GREEN, C_RED]
    descs  = [
        "MODIS Cloudy → L2 Cloud",
        "MODIS Clear → L2 Cloud",
    ]
    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=colors, height=0.45, edgecolor="none")
    ax.barh(y, [100 - v for v in vals], left=vals,
            color=[C_LIGHT]*2, height=0.45, edgecolor="none", alpha=0.5)

    for bar, val, desc in zip(bars, vals, descs):
        ax.text(val - 2, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", ha="right", va="center",
                fontsize=7.5, fontweight="bold", color="white")
        ax.text(102, bar.get_y() + bar.get_height()/2,
                desc, ha="left", va="center", fontsize=5.5, color=C_NEUTRAL,
                style="italic")
    ax.set_xlim(0, 175)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7, fontweight="bold")
    ax.set_xlabel("Rate (%)", fontsize=6.5)
    ax.axvline(50, color=C_LIGHT, ls="--", lw=0.5, zorder=0)
    ax.invert_yaxis()


def _draw_cm(ax):
    """混淆矩阵热力图"""
    cm_pct = CM / CM.sum() * 100
    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=55, aspect="auto")

    for i in range(2):
        for j in range(2):
            c = "white" if cm_pct[i, j] > 30 else "black"
            ax.text(j, i, f"{CM[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=6.5, color=c, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MODIS Cloud", "MODIS Clear"], fontsize=6)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["L2 Cloud", "L2 Clear"], fontsize=6)
    ax.tick_params(length=0)

    # 角标
    for xy, txt, clr in [
        ((0, -0.38), "Hit\n(TP)", C_GREEN),
        ((1, -0.38), "False Alarm\n(FP)", C_RED),
        ((0, 1.38), "Miss\n(FN)", C_ORANGE),
        ((1, 1.38), "Correct Rej.\n(TN)", C_TEAL),
    ]:
        ax.text(xy[0], xy[1], txt, ha="center", va="center",
                fontsize=5, color=clr, fontweight="bold")


def _draw_stratified_oa(ax):
    """分层 pooled OA 柱状图"""
    strata = [
        ("Overall", OA, C_BLUE),
        ("MODIS CF\n> 60%", OA_HIGH, C_TEAL),
        ("MODIS CF\n≤ 60%", OA_LOW, C_ORANGE),
    ]
    x = np.arange(len(strata))
    vals   = [v for _, v, _ in strata]
    colors = [c for _, _, c in strata]
    labels = [l for l, _, _ in strata]

    bars = ax.bar(x, vals, color=colors, width=0.5, edgecolor="none")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f"{val:.1f}%", ha="center", fontsize=7, fontweight="bold")
    # 标注场景数
    ax.text(1, OA_HIGH + 5, f"n={len(HIGH_CF)}", ha="center", fontsize=5.5, color=C_TEAL)
    ax.text(2, OA_LOW + 5, f"n={len(LOW_CF)}", ha="center", fontsize=5.5, color=C_ORANGE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Pooled OA (%)", fontsize=6.5)
    ax.set_ylim(0, 105)
    ax.axhline(50, color=C_LIGHT, ls="--", lw=0.5, zorder=0)
    # Δ 标注
    ax.annotate("", xy=(2, OA_LOW), xytext=(1, OA_HIGH),
                arrowprops=dict(arrowstyle="<->", color=C_NEUTRAL, lw=0.8))
    ax.text(1.5, (OA_HIGH + OA_LOW)/2, f"Δ={OA_HIGH - OA_LOW:.1f} pp",
            ha="center", va="center", fontsize=5.5, color=C_NEUTRAL,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))


def _draw_cf_scatter(ax):
    """L2 vs MODIS 云量散点 (颜色=OA, 大小=N)"""
    l2_cf = [s["l2_cf"] for s in SCENES]
    cm_cf = [s["cm_cf"] for s in SCENES]
    oa_v  = [s["oa"] for s in SCENES]
    sizes = [max(30, s["n"] / 8000) for s in SCENES]

    # y=x 线 + 区域
    ax.plot([35, 100], [35, 100], ls="--", color=C_LIGHT, lw=0.7, zorder=0)
    ax.fill_between([35, 100], [35, 100], 100, alpha=0.04, color=C_RED, zorder=0)
    ax.fill_between([35, 100], 0, [35, 100], alpha=0.04, color=C_GREEN, zorder=0)

    sc = ax.scatter(l2_cf, cm_cf, c=oa_v, cmap="RdYlGn", vmin=35, vmax=95,
                    s=sizes, edgecolors="white", linewidths=0.5, zorder=2)
    for s in SCENES:
        ax.annotate(s["ts"], (s["l2_cf"], s["cm_cf"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=5, color=C_NEUTRAL)

    ax.set_xlabel("AGRI L2 Cloud Fraction (%)", fontsize=6.5)
    ax.set_ylabel("MODIS CM Cloud Fraction (%)", fontsize=6.5)
    ax.set_xlim(40, 96)
    ax.set_ylim(40, 96)
    ax.set_aspect("equal")

    cb = ax.figure.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("OA (%)", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)

    ax.text(85, 50, "L2 over-estimates", fontsize=5, color=C_RED, alpha=0.55,
            ha="center", style="italic")
    ax.text(50, 85, "L2 under-estimates", fontsize=5, color=C_GREEN, alpha=0.55,
            ha="center", style="italic")

    # High/Low CF 标记
    for s in SCENES:
        marker = "*" if s["cm_cf"] > 60 else ""
        if marker:
            ax.annotate(marker, (s["l2_cf"], s["cm_cf"] - 4),
                       fontsize=9, color=C_TEAL, ha="center")


if __name__ == "__main__":
    main()
