#!/usr/bin/env python3
"""
Prototype Vertical Timeline for Figure 1.
Explores top-to-bottom clinical flow (0h -> 52h) to compare with horizontal timeline.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Styling configuration
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

REPO_ROOT = Path("/home/alireza/Sepsis-Onset-Timing-Paradox")
FIG_DIR = REPO_ROOT / "reports" / "figures"

def generate_vertical_figure1():
    # Width 15 inches, height 8.5 inches for clean side-by-side vertical timeline + bar chart
    fig = plt.figure(figsize=(15.0, 8.5), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.32, 
                          left=0.06, right=0.96, top=0.92, bottom=0.12)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ---------------- PANEL A: VERTICAL TIMELINE ----------------
    ax1.set_title("A. Vertical Clinical Timeline & Evaluation Clocks", fontweight="bold", loc="left", fontsize=13.5, pad=16)
    
    # Invert Y so time flows TOP to BOTTOM: 0h at top, 56h at bottom
    ax1.set_xlim(-1.0, 11.5)
    ax1.set_ylim(56, -3)  # top is -3, bottom is 56
    ax1.axis("off")

    x_axis = 1.8  # Main vertical time axis position

    # Main vertical timeline axis
    ax1.plot([x_axis, x_axis], [-0.5, 54], color="#334155", lw=3.0, zorder=2)
    for h in [0, 10, 20, 30, 40, 50]:
        ax1.plot([x_axis - 0.25, x_axis + 0.25], [h, h], color="#334155", lw=2.0, zorder=3)
        ax1.text(x_axis - 0.45, h, f"{h}h", ha="right", va="center", fontsize=11, fontweight="bold", color="#334155")

    # Surveillance shaded band [19, 30] vertically
    ax1.axhspan(19, 30, xmin=0.18, xmax=0.96, facecolor="#EEF2FF", edgecolor="#C7D2FE", lw=1.5, zorder=0)

    def draw_badge(ax, x, y, letter, bg_color="#0F172A", text_color="white", fontsize=11):
        ax.text(x, y, letter, ha="center", va="center", fontsize=fontsize, fontweight="bold",
                color=text_color, zorder=12,
                bbox=dict(boxstyle="circle,pad=0.30", facecolor=bg_color, edgecolor="white", lw=1.4))

    # a: Admission (t=0)
    ax1.plot(x_axis, 0, "o", color="#0F172A", markersize=9, zorder=5)
    draw_badge(ax1, x_axis - 1.2, 0, "a", bg_color="#0F172A")
    ax1.plot([x_axis - 0.9, x_axis], [0, 0], color="#0F172A", lw=1.6, linestyle=":", zorder=4)

    # b: Surveillance window [19, 30] - vertical double bracket
    x_bracket = x_axis + 0.9
    ax1.annotate("", xy=(x_bracket, 19), xytext=(x_bracket, 30),
                arrowprops=dict(arrowstyle="<->", color="#4338CA", lw=2.2))
    draw_badge(ax1, x_bracket, 16.5, "b", bg_color="#4338CA")
    ax1.text(x_bracket + 0.5, 16.5, "Surveillance (19–30h)", ha="left", va="center", fontsize=10.5, fontweight="bold", color="#4338CA")

    # c: Sepsis-3 onset (t=28) - Red dot & badge on timeline
    ax1.plot(x_axis, 28, "o", color="#DC2626", markersize=10, zorder=5)
    draw_badge(ax1, x_axis - 1.2, 28, "c", bg_color="#DC2626")
    ax1.plot([x_axis - 0.9, x_axis], [28, 28], color="#DC2626", lw=1.8, linestyle=":", zorder=4)
    ax1.plot([x_axis, 11.0], [28, 28], color="#DC2626", lw=1.6, linestyle=":", zorder=2)

    # d: Suspicion anchor (t=52) - Blue dot & badge
    ax1.plot(x_axis, 52, "o", color="#2563EB", markersize=10, zorder=5)
    draw_badge(ax1, x_axis - 1.2, 52, "d", bg_color="#2563EB")
    ax1.plot([x_axis - 0.9, x_axis], [52, 52], color="#2563EB", lw=1.8, linestyle=":", zorder=4)
    ax1.plot([x_axis, 11.0], [52, 52], color="#2563EB", lw=1.6, linestyle=":", zorder=2)

    # e: Retrospective Lookback lead gap (28h to 52h)
    x_gap = x_axis + 1.2
    ax1.annotate("", xy=(x_gap, 28), xytext=(x_gap, 52),
                arrowprops=dict(arrowstyle="<->", color="#334155", lw=2.2))
    draw_badge(ax1, x_gap + 0.6, 40, "e", bg_color="#334155")
    ax1.text(x_gap + 1.2, 40, "Lead Gap\nGi ≈ 24h", ha="left", va="center", fontsize=10, fontweight="bold", color="#334155")

    # --- VERTICAL SWIMLANES (Tracks across X) ---
    track_f = 6.2
    track_g = 8.4
    track_h = 10.3

    # Header labels for tracks
    draw_badge(ax1, track_f, 10, "f", bg_color="#EA580C")
    ax1.text(track_f, 6.5, "Prediction\nHorizon", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#9A3412")
    # f-bar: [22, 28]
    ax1.add_patch(patches.Rectangle((track_f - 0.75, 22), 1.5, 6.0, 
                                   facecolor="#FED7AA", edgecolor="#EA580C", lw=2.0, zorder=3))
    ax1.text(track_f, 24.0, "[22h,", ha="center", va="center", fontsize=11, fontweight="bold", color="#7C2D12")
    ax1.text(track_f, 26.0, "28h)", ha="center", va="center", fontsize=11, fontweight="bold", color="#7C2D12")

    draw_badge(ax1, track_g, 10, "g", bg_color="#16A34A")
    ax1.text(track_g, 6.5, "Anchor-Eligible\nDecisions", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#14532D")
    # g-bar: [22, 28]
    ax1.add_patch(patches.Rectangle((track_g - 0.75, 22), 1.5, 6.0, 
                                   facecolor="#BBF7D0", edgecolor="#16A34A", lw=2.0, zorder=3))
    ax1.text(track_g, 24.0, "t + δ", ha="center", va="center", fontsize=11, fontweight="bold", color="#064E3B")
    ax1.text(track_g, 26.0, "≤ C", ha="center", va="center", fontsize=11, fontweight="bold", color="#064E3B")

    draw_badge(ax1, track_h, 44, "h", bg_color="#64748B")
    ax1.text(track_h, 40.5, "Post-Anchor\nCensored", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#475569")
    # h-bar: [52, 55.5]
    ax1.add_patch(patches.Rectangle((track_h - 0.75, 52), 1.5, 3.5, 
                                   facecolor="#F1F5F9", edgecolor="#64748B", linestyle="--", lw=1.8, zorder=3))
    ax1.text(track_h, 53.75, "t > C", ha="center", va="center", fontsize=11, fontweight="bold", color="#1E293B")


    # ---------------- PANEL B: BAR CHART ----------------
    ax2.set_title("B. Pre-Treatment Opportunity by Anchor", fontweight="bold", loc="left", fontsize=13.5, pad=16)

    categories = [
        "Broad Historical\n($T_{\\mathrm{Rx,start}}$)",
        "MetaVision Earliest Rx\n($T_{\\mathrm{Rx,start}}$)",
        "MetaVision Primary\n($T_{\\mathrm{susp}}$)"
    ]

    definite = [0.30, 0.0, 99.34]
    indet = [32.64, 31.13, 0.66]
    inelig = [67.06, 68.87, 0.0]

    y_pos = np.arange(len(categories))
    bar_h = 0.44

    ax2.barh(y_pos, definite, height=bar_h, color="#16A34A", edgecolor="#0F172A", lw=1.1, label="Definite ($O^{\\mathrm{def}}=1$)")
    ax2.barh(y_pos, indet, height=bar_h, left=definite, color="#F59E0B", edgecolor="#0F172A", lw=1.1, label="Indeterminate ($O^{\\mathrm{indet}}=1$)")
    ax2.barh(y_pos, inelig, height=bar_h, left=np.array(definite) + np.array(indet), color="#94A3B8", edgecolor="#0F172A", lw=1.1, label="Ineligible ($O=0$)")

    ax2.text(50, 2, "99.3% Definite (150/151)", ha="center", va="center", color="white", fontweight="bold", fontsize=10.5)
    ax2.text(15.5, 1, "31.1% Indet (47)", ha="center", va="center", color="#0F172A", fontweight="bold", fontsize=9.8)
    ax2.text(65, 1, "68.9% Ineligible (104)", ha="center", va="center", color="white", fontweight="bold", fontsize=9.8)
    ax2.text(16.5, 0, "32.6% Indet (110)", ha="center", va="center", color="#0F172A", fontweight="bold", fontsize=9.8)
    ax2.text(66, 0, "67.1% Ineligible (226)", ha="center", va="center", color="white", fontweight="bold", fontsize=9.8)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(categories, fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Within-Covered Stays ($V = 1$) Proportion (%)", fontsize=11.5, fontweight="bold", labelpad=8)
    ax2.set_xlim(0, 100)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3, 
               frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9.8)

    out_png = FIG_DIR / "Figure1_Vertical_Prototype.png"
    out_pdf = FIG_DIR / "Figure1_Vertical_Prototype.pdf"
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Generated vertical prototype at {out_png}")

if __name__ == "__main__":
    generate_vertical_figure1()
