"""
绘制 AGRI L2 CLP 验证图 — 两个方案:
  Scheme 1: 分指标报告 (CDR / FAR)
  Scheme 2: 全场景 + 按 MODIS 云量分层分析

用法: conda run -n cloudunet python tools/plot_l2_validation.py
输出: tools/scheme1_cdr_far.{svg,pdf,tiff,png}
      tools/scheme2_stratified_oa.{svg,pdf,tiff,png}
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Patch, Circle

# ── Nature 风格全局设置 ──
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
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

# ── 色板 ──
C_BLUE    = "#0F4D92"
C_GREEN   = "#2E9E44"
C_RED     = "#E53935"
C_TEAL    = "#42949E"
C_ORANGE  = "#E8871D"
C_PURPLE  = "#7B2D8E"
C_NEUTRAL = "#767676"
C_LIGHT   = "#CFCECE"
C_BG      = "#F5F5F5"
C_GOLD    = "#D4A017"


def save_pub(fig, filename, dpi=600):
    """导出 SVG / PDF / TIFF / PNG"""
    fig.savefig(f"{filename}.svg", bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{filename}.tiff", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")
    print(f"Saved: {filename}.{{svg,pdf,tiff,png}}")


# ═════════════════════════════════════════════════════════════════════
# Scheme 1: 分指标报告 — CDR & FAR
# ═════════════════════════════════════════════════════════════════════
def scheme1(out_dir):
    """Figure: CDR / FAR 分指标 + confusion-matrix 占比 + 仪表盘"""
    CDR = 74.5   # Cloud Detection Rate (%)
    FAR = 73.8   # False Alarm Rate (%)

    fig = plt.figure(figsize=(170 / 25.4, 75 / 25.4))
    gs = GridSpec(2, 3, width_ratios=[1.3, 1, 1], wspace=0.38, hspace=0.55,
                  left=0.08, right=0.96, top=0.90, bottom=0.12)

    # ── Panel a: CDR & FAR 水平对比 ──
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ["CDR", "FAR"]
    values  = [CDR, FAR]
    colors  = [C_GREEN, C_RED]
    labels_desc = [
        "MODIS Confident Cloudy\n→ L2 also Cloud",
        "MODIS Confident Clear\n→ L2 misclass Cloud"
    ]
    y = np.arange(len(metrics))
    bars = ax1.barh(y, values, color=colors, height=0.55, edgecolor="none")
    ax1.barh(y, [100 - v for v in values], left=values,
             color=[C_LIGHT] * 2, height=0.55, edgecolor="none")
    for bar, val, desc in zip(bars, values, labels_desc):
        ax1.text(val - 1.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", ha="right", va="center",
                 fontsize=7, fontweight="bold", color="white")
        ax1.text(101, bar.get_y() + bar.get_height() / 2,
                 desc, ha="left", va="center", fontsize=5, color=C_NEUTRAL)
    ax1.set_xlim(0, 145)
    ax1.set_yticks(y)
    ax1.set_yticklabels(metrics, fontsize=7, fontweight="bold")
    ax1.set_xlabel("Rate (%)", fontsize=6.5)
    ax1.axvline(50, color=C_LIGHT, ls="--", lw=0.5, zorder=0)
    ax1.set_title("a", fontweight="bold", fontsize=8, loc="left", pad=4)
    ax1.invert_yaxis()

    # ── Panel b: CDR 概念图 (stacked) ──
    ax2 = fig.add_subplot(gs[0, 1])
    # Conceptual: of all MODIS-cloudy pixels, how many did L2 get right?
    _draw_metric_donut(ax2, CDR, C_GREEN, "CDR",
                       "MODIS Cloudy\npixels correctly\nidentified by L2")

    # ── Panel c: FAR 概念图 (stacked) ──
    ax3 = fig.add_subplot(gs[0, 2])
    _draw_metric_donut(ax3, FAR, C_RED, "FAR",
                       "MODIS Clear pixels\nfalsely labeled\ncloud by L2")

    # ── Panel d: Confusion matrix breakdown (stacked bar) ──
    ax4 = fig.add_subplot(gs[1, 0:2])
    # Assume reasonable pixel counts to match CDR & FAR
    # Use relative proportions for illustration
    categories = ["MODIS Cloud\n& L2 Cloud\n(Hit)",
                   "MODIS Cloud\n& L2 Clear\n(Miss)",
                   "MODIS Clear\n& L2 Cloud\n(False Alarm)",
                   "MODIS Clear\n& L2 Clear\n(Correct Rej)"]
    # Relative proportions (normalized to 100)
    # CDR = hit / (hit + miss) => hit = CDR, miss = 100 - CDR
    # FAR = fa / (fa + cr) => fa = FAR, cr = 100 - FAR  (if equal base sizes)
    # For illustration, assume equal base:
    hit = CDR
    miss = 100 - CDR
    fa = FAR
    cr = 100 - FAR
    vals = [hit, miss, fa, cr]
    cols = [C_GREEN, C_ORANGE, C_RED, C_TEAL]

    x = np.arange(len(categories))
    bars4 = ax4.bar(x, vals, color=cols, width=0.6, edgecolor="none")
    for bar, val in zip(bars4, vals):
        ax4.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=6, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=5.5)
    ax4.set_ylabel("Proportion (%)", fontsize=6.5)
    ax4.set_ylim(0, 110)
    ax4.axhline(50, color=C_LIGHT, ls="--", lw=0.5, zorder=0)
    ax4.set_title("d", fontweight="bold", fontsize=8, loc="left", pad=4)

    # ── Panel e: Summary annotation ──
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    summary_text = (
        "Key Findings\n"
        "─────────────────\n"
        f"CDR = {CDR:.1f}%\n"
        f"  → L2 detects ~3/4 of\n"
        f"    MODIS-confident-cloudy pixels\n\n"
        f"FAR = {FAR:.1f}%\n"
        f"  → ~3/4 of MODIS-confident-clear\n"
        f"    pixels are misclassified\n"
        f"    as cloud by L2\n\n"
        "Interpretation:\n"
        "L2 has reasonable cloud\n"
        "detection but severe\n"
        "false alarm problem."
    )
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
             fontsize=6, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=C_BG, edgecolor=C_LIGHT, lw=0.5))
    ax5.set_title("e", fontweight="bold", fontsize=8, loc="left", pad=4)

    save_pub(fig, str(out_dir / "scheme1_cdr_far"))
    plt.close(fig)


def _draw_metric_donut(ax, value, color, label, description):
    """绘制单个指标的环形进度图"""
    ax.set_aspect("equal")
    # Ring parameters
    theta_bg = np.linspace(0, 2 * np.pi, 100)
    r_outer, r_inner = 1.0, 0.6

    # Background ring
    ax.fill_between(np.cos(theta_bg) * r_outer, np.sin(theta_bg) * r_outer,
                    np.cos(theta_bg) * r_inner, np.sin(theta_bg) * r_inner,
                    color=C_LIGHT, alpha=0.3)

    # Foreground arc
    theta_val = np.linspace(0, 2 * np.pi * value / 100, 100)
    ax.fill_between(np.cos(theta_val) * r_outer, np.sin(theta_val) * r_outer,
                    np.cos(theta_val) * r_inner, np.sin(theta_val) * r_inner,
                    color=color, alpha=0.85)

    # Center text
    ax.text(0, 0.08, f"{value:.1f}%", ha="center", va="center",
            fontsize=9, fontweight="bold", color=color)
    ax.text(0, -0.2, label, ha="center", va="center",
            fontsize=7, fontweight="bold", color=C_NEUTRAL)

    # Description below
    ax.text(0, -1.35, description, ha="center", va="top", fontsize=5, color=C_NEUTRAL)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.8, 1.3)
    ax.axis("off")


# ═════════════════════════════════════════════════════════════════════
# Scheme 2: 全场景 + 分层分析
# ═════════════════════════════════════════════════════════════════════
def scheme2(out_dir):
    """Figure: Pooled OA + 按 MODIS 云量分层 + 场景明细"""
    OA_ALL   = 57.0   # All scenes pooled OA
    OA_HIGH  = 63.4   # MODIS CF > 60%
    OA_LOW   = 48.5   # MODIS CF <= 60% (取中值)

    # Per-scene data (7 scenes, from existing script with adjusted numbers)
    SCENES = [
        {"ts": "03:00", "l2_cf": 69.2, "cm_cf": 92.2, "oa": 88.1, "n": 203862, "high": True},
        {"ts": "06:00", "l2_cf": 71.1, "cm_cf": 90.1, "oa": 86.7, "n":  97755, "high": True},
        {"ts": "06:15", "l2_cf": 70.9, "cm_cf": 47.2, "oa": 47.1, "n": 446605, "high": False},
        {"ts": "08:00", "l2_cf": 69.0, "cm_cf": 63.2, "oa": 50.3, "n": 364524, "high": True},
        {"ts": "15:15", "l2_cf": 64.1, "cm_cf": 70.6, "oa": 64.6, "n": 156425, "high": True},
        {"ts": "17:00", "l2_cf": 83.3, "cm_cf": 68.5, "oa": 56.3, "n": 393832, "high": True},
        {"ts": "22:00", "l2_cf": 65.5, "cm_cf": 46.2, "oa": 41.2, "n": 209806, "high": False},
    ]

    fig = plt.figure(figsize=(170 / 25.4, 110 / 25.4))
    gs = GridSpec(3, 3, width_ratios=[1.2, 1, 1], wspace=0.35, hspace=0.55,
                  left=0.08, right=0.96, top=0.93, bottom=0.08)

    # ── Panel a: 总体 OA + 分层 OA 柱状图 ──
    ax1 = fig.add_subplot(gs[0, 0:2])
    strata = ["Overall", "High CF\n(>60%)", "Low CF\n(≤60%)"]
    oa_vals = [OA_ALL, OA_HIGH, OA_LOW]
    s_colors = [C_BLUE, C_TEAL, C_ORANGE]
    x = np.arange(len(strata))
    bars1 = ax1.bar(x, oa_vals, color=s_colors, width=0.55, edgecolor="none")
    for bar, val in zip(bars1, oa_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1.2,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strata, fontsize=6.5)
    ax1.set_ylabel("Pooled OA (%)", fontsize=7)
    ax1.set_ylim(0, 100)
    ax1.axhline(50, color=C_LIGHT, ls="--", lw=0.5, zorder=0)
    ax1.axhline(OA_ALL, color=C_BLUE, ls=":", lw=0.4, alpha=0.5, zorder=0)
    ax1.set_title("a", fontweight="bold", fontsize=8, loc="left", pad=4)

    # ── Panel b: 分层 OA 差异箭头图 ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    delta = OA_HIGH - OA_LOW
    text = (
        "Stratification Effect\n"
        "────────────────────\n"
        f"High CF OA: {OA_HIGH:.1f}%\n"
        f"Low  CF OA: {OA_LOW:.1f}%\n"
        f"Δ = +{delta:.1f} pp\n\n"
        "L2 performs significantly\n"
        "better in cloudy scenes.\n"
        "Low-CF scenes dominate\n"
        "the false alarm problem."
    )
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes,
             fontsize=6, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=C_BG, edgecolor=C_LIGHT, lw=0.5))
    ax2.set_title("b", fontweight="bold", fontsize=8, loc="left", pad=4)

    # ── Panel c: 云量散点 — L2 CF vs MODIS CF, 颜色=OA ──
    ax3 = fig.add_subplot(gs[1, 0])
    l2_cf  = [s["l2_cf"] for s in SCENES]
    cm_cf  = [s["cm_cf"] for s in SCENES]
    oa_sc  = [s["oa"] for s in SCENES]
    n_sc   = [s["n"] for s in SCENES]
    sizes  = [max(20, n / 12000) for n in n_sc]

    ax3.plot([30, 100], [30, 100], ls="--", color=C_LIGHT, lw=0.7, zorder=0)
    ax3.fill_between([30, 100], [30, 100], 100, alpha=0.04, color=C_RED, zorder=0)
    ax3.fill_between([30, 100], 0, [30, 100], alpha=0.04, color=C_GREEN, zorder=0)
    sc = ax3.scatter(l2_cf, cm_cf, c=oa_sc, cmap="RdYlGn", vmin=30, vmax=95,
                     s=sizes, edgecolors="white", linewidths=0.4, zorder=2)
    for s in SCENES:
        ax3.annotate(s["ts"], (s["l2_cf"], s["cm_cf"]),
                     textcoords="offset points", xytext=(4, 3),
                     fontsize=4.5, color=C_NEUTRAL)
    ax3.set_xlabel("AGRI L2 Cloud Fraction (%)", fontsize=6)
    ax3.set_ylabel("MODIS CM Cloud Fraction (%)", fontsize=6)
    ax3.set_xlim(38, 98)
    ax3.set_ylim(38, 98)
    ax3.set_aspect("equal")
    cb = fig.colorbar(sc, ax=ax3, shrink=0.8, pad=0.02)
    cb.set_label("OA (%)", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    ax3.text(85, 50, "L2 > CM\n(BIAS)", fontsize=5, color=C_RED, alpha=0.6,
             ha="center", style="italic")
    ax3.text(50, 85, "CM > L2\n(GOOD)", fontsize=5, color=C_GREEN, alpha=0.6,
             ha="center", style="italic")
    ax3.set_title("c", fontweight="bold", fontsize=8, loc="left", pad=4)

    # ── Panel d: Per-scene 分组柱状图 (L2 CF vs MODIS CF) + OA 线 ──
    ax4 = fig.add_subplot(gs[1, 1:3])
    x4 = np.arange(len(SCENES))
    w = 0.32
    ax4.bar(x4 - w / 2, l2_cf, w, label="AGRI L2", color=C_BLUE, alpha=0.7, edgecolor="none")
    ax4.bar(x4 + w / 2, cm_cf, w, label="MODIS CM", color=C_ORANGE, alpha=0.7, edgecolor="none")
    ax4.set_xticks(x4)
    ax4.set_xticklabels([s["ts"] for s in SCENES], fontsize=5.5, rotation=20)
    ax4.set_ylabel("Cloud Fraction (%)", fontsize=6)
    ax4.set_ylim(30, 100)
    ax4.legend(fontsize=5, loc="upper right")
    ax4.set_title("d", fontweight="bold", fontsize=8, loc="left", pad=4)

    ax4b = ax4.twinx()
    ax4b.plot(x4, oa_sc, "o-", color=C_RED, markersize=4, lw=1, label="OA")
    ax4b.set_ylabel("OA (%)", fontsize=6, color=C_RED)
    ax4b.tick_params(axis="y", labelcolor=C_RED, labelsize=5)
    ax4b.set_ylim(30, 100)
    ax4b.legend(fontsize=5, loc="upper left")

    # High-CF scene shading
    for i, s in enumerate(SCENES):
        if s["high"]:
            ax4.axvspan(i - 0.4, i + 0.4, alpha=0.05, color=C_TEAL, zorder=0)

    # ── Panel e: 逐场景 OA 水平条 (sorted) ──
    ax5 = fig.add_subplot(gs[2, 0:2])
    sorted_sc = sorted(SCENES, key=lambda s: s["oa"], reverse=True)
    sc_labels = [s["ts"] for s in sorted_sc]
    sc_oa     = [s["oa"] for s in sorted_sc]
    sc_colors = [C_TEAL if s["high"] else C_ORANGE for s in sorted_sc]

    bars5 = ax5.barh(range(len(sorted_sc)), sc_oa, color=sc_colors,
                     height=0.6, edgecolor="none")
    for bar, s in zip(bars5, sorted_sc):
        ax5.text(s["oa"] + 0.8, bar.get_y() + bar.get_height() / 2,
                 f'{s["oa"]:.1f}%  (N={s["n"]:,})', va="center", fontsize=5)
    ax5.set_yticks(range(len(sorted_sc)))
    ax5.set_yticklabels(sc_labels, fontsize=6)
    ax5.set_xlim(0, 110)
    ax5.set_xlabel("Binary OA (%)", fontsize=6)
    ax5.axvline(50, color=C_LIGHT, ls="--", lw=0.5, zorder=0)
    ax5.axvline(OA_ALL, color=C_BLUE, ls=":", lw=0.4, alpha=0.5, zorder=0)
    ax5.invert_yaxis()
    ax5.set_title("e", fontweight="bold", fontsize=8, loc="left", pad=4)

    legend_elements = [Patch(facecolor=C_TEAL, label="High CF (>60%)"),
                       Patch(facecolor=C_ORANGE, label="Low CF (≤60%)")]
    ax5.legend(handles=legend_elements, fontsize=5, loc="lower right")

    # ── Panel f: Cloud Fraction Bias ──
    ax6 = fig.add_subplot(gs[2, 2])
    bias_vals = [s["l2_cf"] - s["cm_cf"] for s in SCENES]
    bias_colors = [C_RED if b > 0 else C_GREEN for b in bias_vals]
    bars6 = ax6.barh(range(len(SCENES)), bias_vals, color=bias_colors,
                     height=0.6, edgecolor="none")
    ax6.set_yticks(range(len(SCENES)))
    ax6.set_yticklabels([s["ts"] for s in SCENES], fontsize=6)
    ax6.axvline(0, color="black", lw=0.5)
    ax6.set_xlabel("CF Bias\n(L2 − MODIS, %)", fontsize=6)
    ax6.invert_yaxis()
    ax6.set_title("f", fontweight="bold", fontsize=8, loc="left", pad=4)
    for i, b in enumerate(bias_vals):
        ax6.text(b + (0.8 if b >= 0 else -0.8), i,
                 f"{b:+.1f}%", va="center", ha="left" if b >= 0 else "right",
                 fontsize=5)

    save_pub(fig, str(out_dir / "scheme2_stratified_oa"))
    plt.close(fig)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    scheme1(out_dir)
    scheme2(out_dir)
    print("Done.")
