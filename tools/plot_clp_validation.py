"""
绘制 AGRI L2 vs MODIS CLP 验证图（Nature 风格）。
Figure 1: 分指标报告 (CDR / FAR / Confusion Matrix)
Figure 2: 分层分析 (Cloud Fraction 对比 / 按云量分层 OA / 逐场景明细)

用法: conda run -n cloudunet python tools/plot_clp_validation.py
输出: tools/fig1_cdr_far.{svg,pdf,tiff}, tools/fig2_stratified.{svg,pdf,tiff}
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

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
C_BLUE   = "#0F4D92"
C_GREEN  = "#2E9E44"
C_RED    = "#E53935"
C_TEAL   = "#42949E"
C_ORANGE = "#E8871D"
C_NEUTRAL = "#767676"
C_LIGHT  = "#CFCECE"
C_BG     = "#F5F5F5"


# ── 验证数据 ──
SCENES = [
    {"ts": "03:00", "l2_cf": 69.2, "cm_cf": 92.2, "oa": 88.1, "n": 203862},
    {"ts": "06:00", "l2_cf": 71.1, "cm_cf": 90.1, "oa": 86.7, "n":  97755},
    {"ts": "06:15", "l2_cf": 70.9, "cm_cf": 47.2, "oa": 47.1, "n": 446605},
    {"ts": "08:00", "l2_cf": 69.0, "cm_cf": 63.2, "oa": 50.3, "n": 364524},
    {"ts": "15:15", "l2_cf": 64.1, "cm_cf": 70.6, "oa": 64.6, "n": 156425},
    {"ts": "17:00", "l2_cf": 83.3, "cm_cf": 68.5, "oa": 56.3, "n": 393832},
    {"ts": "22:00", "l2_cf": 65.5, "cm_cf": 46.2, "oa": 41.2, "n": 209806},
]

# Pooled confusion matrix
CM = np.array([[543008, 538363],
               [210412, 136200]])

# Derived metrics
CDR = CM[0, 0] / (CM[0, 0] + CM[1, 0]) * 100   # 72.1%
FAR = CM[0, 1] / (CM[0, 1] + CM[1, 1]) * 100   # 79.8%
OA  = (CM[0, 0] + CM[1, 1]) / CM.sum() * 100    # 47.6%
PREC = CM[0, 0] / (CM[0, 0] + CM[0, 1]) * 100   # 50.2%


def save_pub(fig, filename, dpi=600):
    fig.savefig(f"{filename}.svg", bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{filename}.tiff", dpi=dpi, bbox_inches="tight")
    print(f"Saved: {filename}.svg/.pdf/.tiff")


# ═══════════════════════════════════════════════════════════════════
# Figure 1: 分指标报告 (CDR / FAR)
# ═══════════════════════════════════════════════════════════════════
def figure1(out_dir):
    fig = plt.figure(figsize=(170 / 25.4, 60 / 25.4))  # 170mm wide, 60mm tall
    gs = GridSpec(1, 3, width_ratios=[1, 1.2, 1.3], wspace=0.4,
                  left=0.08, right=0.97, top=0.88, bottom=0.18)

    # ── Panel a: CDR & FAR bar chart ──
    ax1 = fig.add_subplot(gs[0])
    metrics = ["CDR", "FAR"]
    values = [CDR, FAR]
    colors = [C_GREEN, C_RED]
    bars = ax1.bar(metrics, values, color=colors, width=0.55, edgecolor="none")
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=6, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Rate (%)", fontsize=6.5)
    ax1.set_title("a", fontweight="bold", fontsize=8, loc="left", pad=4)
    ax1.axhline(50, color=C_LIGHT, ls="--", lw=0.5, zorder=0)
    ax1.tick_params(axis="x", labelsize=6.5)
    ax1.tick_params(axis="y", labelsize=6)

    # ── Panel b: Confusion matrix heatmap ──
    ax2 = fig.add_subplot(gs[1])
    cm_pct = CM / CM.sum() * 100
    im = ax2.imshow(cm_pct, cmap="Blues", vmin=0, vmax=50, aspect="auto")
    labels = [["{:,}\n({:.1f}%)".format(CM[i, j], cm_pct[i, j])
               for j in range(2)] for i in range(2)]
    for i in range(2):
        for j in range(2):
            color = "white" if cm_pct[i, j] > 30 else "black"
            ax2.text(j, i, labels[i][j], ha="center", va="center",
                     fontsize=5.5, color=color)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["MODIS\nCloud", "MODIS\nClear"], fontsize=5.5)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["L2 Cloud", "L2 Clear"], fontsize=5.5)
    ax2.set_title("b", fontweight="bold", fontsize=8, loc="left", pad=4)
    ax2.tick_params(length=0)

    # ── Panel c: Summary metrics ──
    ax3 = fig.add_subplot(gs[2])
    summary_labels = ["OA", "Precision", "CDR", "FAR"]
    summary_vals = [OA, PREC, CDR, FAR]
    summary_colors = [C_BLUE, C_TEAL, C_GREEN, C_RED]
    y_pos = np.arange(len(summary_labels))
    bars3 = ax3.barh(y_pos, summary_vals, color=summary_colors,
                     height=0.55, edgecolor="none")
    for bar, val in zip(bars3, summary_vals):
        ax3.text(val + 1.2, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", ha="left", va="center", fontsize=5.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(summary_labels, fontsize=6)
    ax3.set_xlim(0, 100)
    ax3.set_xlabel("Rate (%)", fontsize=6.5)
    ax3.axvline(50, color=C_LIGHT, ls="--", lw=0.5, zorder=0)
    ax3.set_title("c", fontweight="bold", fontsize=8, loc="left", pad=4)
    ax3.invert_yaxis()

    save_pub(fig, str(out_dir / "fig1_cdr_far"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Figure 2: 分层分析
# ═══════════════════════════════════════════════════════════════════
def figure2(out_dir):
    fig = plt.figure(figsize=(170 / 25.4, 100 / 25.4))  # 170mm x 100mm
    gs = GridSpec(2, 3, wspace=0.35, hspace=0.45,
                  left=0.08, right=0.97, top=0.92, bottom=0.1)

    # ── Panel a: L2 vs MODIS cloud fraction scatter ──
    ax1 = fig.add_subplot(gs[0, 0])
    l2_cf = [s["l2_cf"] for s in SCENES]
    cm_cf = [s["cm_cf"] for s in SCENES]
    oa_vals = [s["oa"] for s in SCENES]
    n_vals = [s["n"] for s in SCENES]
    sizes = [max(15, n / 15000) for n in n_vals]

    # Identity line
    ax1.plot([30, 100], [30, 100], ls="--", color=C_LIGHT, lw=0.7, zorder=0)
    ax1.fill_between([30, 100], [30, 100], 100, alpha=0.04, color=C_RED, zorder=0)
    ax1.fill_between([30, 100], 0, [30, 100], alpha=0.04, color=C_GREEN, zorder=0)

    sc = ax1.scatter(l2_cf, cm_cf, c=oa_vals, cmap="RdYlGn", vmin=30, vmax=95,
                     s=sizes, edgecolors="white", linewidths=0.4, zorder=2)
    for s in SCENES:
        ax1.annotate(s["ts"], (s["l2_cf"], s["cm_cf"]),
                     textcoords="offset points", xytext=(4, 3),
                     fontsize=4.5, color=C_NEUTRAL)
    ax1.set_xlabel("AGRI L2 Cloud Fraction (%)", fontsize=6)
    ax1.set_ylabel("MODIS CM Cloud Fraction (%)", fontsize=6)
    ax1.set_xlim(38, 98)
    ax1.set_ylim(38, 98)
    ax1.set_aspect("equal")
    ax1.set_title("a", fontweight="bold", fontsize=8, loc="left", pad=4)
    cb = fig.colorbar(sc, ax=ax1, shrink=0.75, pad=0.02)
    cb.set_label("OA (%)", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    # Region labels
    ax1.text(85, 50, "L2 > CM\n(BIAS)", fontsize=5, color=C_RED, alpha=0.6,
             ha="center", style="italic")
    ax1.text(50, 85, "CM > L2\n(GOOD)", fontsize=5, color=C_GREEN, alpha=0.6,
             ha="center", style="italic")

    # ── Panel b: Per-scene OA with cloud fraction overlay ──
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(SCENES))
    width = 0.35
    bars_l2 = ax2.bar(x - width / 2, l2_cf, width, label="AGRI L2",
                      color=C_BLUE, alpha=0.7, edgecolor="none")
    bars_cm = ax2.bar(x + width / 2, cm_cf, width, label="MODIS CM",
                      color=C_ORANGE, alpha=0.7, edgecolor="none")
    ax2.set_xticks(x)
    ax2.set_xticklabels([s["ts"] for s in SCENES], fontsize=5, rotation=30)
    ax2.set_ylabel("Cloud Fraction (%)", fontsize=6)
    ax2.set_ylim(30, 100)
    ax2.legend(fontsize=5, loc="upper right")
    ax2.set_title("b", fontweight="bold", fontsize=8, loc="left", pad=4)

    # OA line overlay
    ax2b = ax2.twinx()
    ax2b.plot(x, oa_vals, "o-", color=C_RED, markersize=4, lw=1, label="OA")
    ax2b.set_ylabel("OA (%)", fontsize=6, color=C_RED)
    ax2b.tick_params(axis="y", labelcolor=C_RED, labelsize=5)
    ax2b.set_ylim(30, 100)
    ax2b.legend(fontsize=5, loc="upper left")

    # ── Panel c: Stratified OA ──
    ax3 = fig.add_subplot(gs[0, 2])
    # Classify scenes
    good = [s for s in SCENES if s["cm_cf"] > s["l2_cf"]]
    bias = [s for s in SCENES if s["cm_cf"] <= s["l2_cf"]]

    # High/low cloud fraction
    high_cf = [s for s in SCENES if s["cm_cf"] > 60]
    low_cf = [s for s in SCENES if s["cm_cf"] <= 60]

    # Pooled OA for each stratum
    def pooled_oa(ss):
        total_n = sum(s["n"] for s in ss)
        # Approximate: use OA to back-calculate agreement pixels
        total_agree = sum(int(s["n"] * s["oa"] / 100) for s in ss)
        return total_agree / total_n * 100 if total_n > 0 else 0

    strata = ["All\n(n=7)", "CM>L2\n(n=3)", "L2≥CM\n(n=4)",
              "CM>60%\n(n=5)", "CM≤60%\n(n=2)"]
    strata_oa = [pooled_oa(SCENES), pooled_oa(good), pooled_oa(bias),
                 pooled_oa(high_cf), pooled_oa(low_cf)]
    strata_colors = [C_BLUE, C_GREEN, C_RED, C_TEAL, C_ORANGE]

    bars3 = ax3.bar(strata, strata_oa, color=strata_colors,
                    width=0.6, edgecolor="none")
    for bar, val in zip(bars3, strata_oa):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=5.5, fontweight="bold")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Pooled OA (%)", fontsize=6)
    ax3.axhline(50, color=C_LIGHT, ls="--", lw=0.5, zorder=0)
    ax3.tick_params(axis="x", labelsize=5.5)
    ax3.tick_params(axis="y", labelsize=5)
    ax3.set_title("c", fontweight="bold", fontsize=8, loc="left", pad=4)

    # ── Panel d: Per-scene OA bar (sorted) ──
    ax4 = fig.add_subplot(gs[1, 0:2])
    sorted_scenes = sorted(SCENES, key=lambda s: s["oa"], reverse=True)
    scene_labels = [s["ts"] for s in sorted_scenes]
    scene_oa = [s["oa"] for s in sorted_scenes]
    scene_colors = [C_GREEN if s["cm_cf"] > s["l2_cf"] else C_RED for s in sorted_scenes]

    bars4 = ax4.barh(range(len(sorted_scenes)), scene_oa, color=scene_colors,
                     height=0.6, edgecolor="none")
    for i, (bar, s) in enumerate(zip(bars4, sorted_scenes)):
        ax4.text(s["oa"] + 0.8, i, f'{s["oa"]:.1f}%  (N={s["n"]:,})',
                 va="center", fontsize=5)
    ax4.set_yticks(range(len(sorted_scenes)))
    ax4.set_yticklabels(scene_labels, fontsize=6)
    ax4.set_xlim(0, 105)
    ax4.set_xlabel("Binary OA (%)", fontsize=6)
    ax4.axvline(50, color=C_LIGHT, ls="--", lw=0.5, zorder=0)
    ax4.invert_yaxis()
    ax4.set_title("d", fontweight="bold", fontsize=8, loc="left", pad=4)

    # Legend for d
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=C_GREEN, label="CM > L2 (GOOD)"),
                       Patch(facecolor=C_RED, label="L2 ≥ CM (BIAS)")]
    ax4.legend(handles=legend_elements, fontsize=5, loc="lower right")

    # ── Panel e: Cloud fraction bias ──
    ax5 = fig.add_subplot(gs[1, 2])
    bias_vals = [s["l2_cf"] - s["cm_cf"] for s in SCENES]
    bias_colors = [C_RED if b > 0 else C_GREEN for b in bias_vals]
    bars5 = ax5.barh(range(len(SCENES)), bias_vals, color=bias_colors,
                     height=0.6, edgecolor="none")
    ax5.set_yticks(range(len(SCENES)))
    ax5.set_yticklabels([s["ts"] for s in SCENES], fontsize=6)
    ax5.axvline(0, color="black", lw=0.5)
    ax5.set_xlabel("Cloud Fraction Bias\n(L2 - MODIS, %)", fontsize=6)
    ax5.invert_yaxis()
    ax5.set_title("e", fontweight="bold", fontsize=8, loc="left", pad=4)
    for i, b in enumerate(bias_vals):
        ax5.text(b + (0.5 if b >= 0 else -0.5), i,
                 f"{b:+.1f}%", va="center", ha="left" if b >= 0 else "right",
                 fontsize=5)

    save_pub(fig, str(out_dir / "fig2_stratified"))
    plt.close(fig)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    figure1(out_dir)
    figure2(out_dir)
    print("Done.")
