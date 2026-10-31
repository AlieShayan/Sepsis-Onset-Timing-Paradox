#!/usr/bin/env python3
"""generate_jamia_figures.py

Generates publication-quality figures for JAMIA Research and Applications:
  - Figure 1: Operational Clocks, Actionable Pretreatment Decision Windows, and Interval Uncertainty Waterfall
  - Figure 2: Operational Alert-Budget Performance Frontier (Emitted Alert Burden vs PPV and NNE)
  - Figure 3: Cumulative Population Coverage Decomposition (S -> V -> O -> D)
"""
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path("/home/alireza/Sepsis-Onset-Timing-Paradox")
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})


def generate_figure1_clocks_and_waterfall():
    fig = plt.figure(figsize=(17.2, 8.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.36, 1.0], wspace=0.38, left=0.035, right=0.975, top=0.93, bottom=0.07)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.set_title("A. Operational Clocks & Anchor-Eligible Decision Boundary", fontweight="bold", pad=18, fontsize=13)
    ax1.set_xlim(-15, 57)
    ax1.set_ylim(-1.65, 9.2)
    ax1.axis("off")

    t_admit = 0.0
    t_surv_start = 19.0
    t_surv_end = 30.0
    t_s3 = 28.0
    t_susp = 52.2
    t_h_start = t_s3 - 6.0

    y_axis = 6.6

    # Main Time Axis
    ax1.plot([-2, 55], [y_axis, y_axis], color="#64748B", lw=2.2, zorder=1)
    for h in [0, 10, 20, 30, 40, 50]:
        ax1.plot([h, h], [y_axis - 0.12, y_axis + 0.12], color="#64748B", lw=1.5, zorder=2)
        ax1.text(h, y_axis - 0.35, f"{h}h", ha="center", va="top", fontsize=9.0, color="#64748B")

    # Shaded background band for Surveillance Window [19h, 30h]
    ax1.axvspan(t_surv_start, t_surv_end, ymin=0.46, ymax=0.92, facecolor="#EEF2FF", alpha=0.6, zorder=0)

    # Vertical guideline stems (cleared neatly below callout texts)
    ax1.plot([t_admit, t_admit], [y_axis, y_axis + 1.20], color="#0F172A", lw=1.5, linestyle=":", zorder=2)
    ax1.plot([t_s3, t_s3], [1.8, y_axis + 1.20], color="#DC2626", lw=1.5, linestyle=":", zorder=2)
    ax1.plot([t_susp, t_susp], [0.75, y_axis + 1.20], color="#2563EB", lw=1.5, linestyle=":", zorder=2)

    # Markers
    ax1.plot(t_admit, y_axis, "o", color="#0F172A", markersize=8.5, zorder=5)
    ax1.plot(t_s3, y_axis, "o", color="#DC2626", markersize=9.5, zorder=5)
    ax1.plot(t_susp, y_axis, "o", color="#2563EB", markersize=9.5, zorder=5)

    # Top Callouts - shifted slightly downward to preserve ~2% height clearance below the main panel title
    ax1.text(t_admit, y_axis + 1.30, "ICU Admission\n($T_{\\mathrm{admit}},\\; t=0$)", 
             ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#0F172A")

    surv_center = (t_surv_start + t_surv_end) / 2
    ax1.annotate("", xy=(t_surv_start, y_axis + 0.48), xytext=(t_surv_end, y_axis + 0.48),
                 arrowprops=dict(arrowstyle="<->", color="#4338CA", lw=1.5))
    ax1.text(surv_center, y_axis + 0.60, "Surveillance Window\nHours 19–30 (12 Decision Bins)", 
             ha="center", va="bottom", fontsize=8.4, fontweight="bold", color="#3730A3")

    # Sepsis-3 Onset (t=28h) - shifted downward (~2% panel height) for generous clearance below title
    ax1.text(t_s3, y_axis + 1.30, "Sepsis-3 Onset ($T_{\\mathrm{S3}}$)\n[Illustrative: 28.0h; covered med: 28.0h]", 
             ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#DC2626")

    ax1.text(t_susp, y_axis + 1.30, "Suspicion Anchor ($T_{\\mathrm{susp}}$)\n[Illustrative: 52.2h; cohort med: 50.7h]", 
             ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#1D4ED8")

    # Operational Search-Window Lead Gap Callout
    gap_y = 5.40
    gap_center = (t_s3 + t_susp) / 2
    ax1.annotate("", xy=(t_s3, gap_y), xytext=(t_susp, gap_y),
                 arrowprops=dict(arrowstyle="<->", color="#334155", lw=1.8))
    gap_box = (
        "First-Bin Lookback Identity (all: 274/408, 67.2%; covered: 100/151, 66.2%):\n"
        "Search-Window Lead Gap: $G_i = T_{\\mathrm{susp},i} - T_{\\mathrm{S3},i} = 24h + h\\,\\mathrm{frac}(x_i) < 25h$\n"
        "Coordinate: $x_i = (T_{\\mathrm{susp},i} - 24h - T_{\\mathrm{admit},i})/h$ with bin width $h = 1\\mathrm{h}$\n"
        "Full cohort ($N=408$): median +24.2h (IQR: 22.8–24.6h; empirical range: -11.2h to +24.99h)\n"
        "Illustrative paired trajectory ($52.2 - 28.0 = 24.2\\mathrm{h} \\neq$ marginal medians diff: $50.7 - 28.0 = 22.7\\mathrm{h}$)"
    )
    ax1.text(gap_center, gap_y - 0.20, gap_box, ha="center", va="top", fontsize=8.1, color="#0F172A",
             bbox=dict(boxstyle="round,pad=0.48", facecolor="#F8FAFC", edgecolor="#94A3B8", lw=0.9))

    # Swimlanes
    lane1_y = 3.85
    ax1.text(t_h_start - 1.5, lane1_y, "Anticipation Horizon ($H=6\\mathrm{h}$):", ha="right", va="center", 
             fontsize=8.8, fontweight="bold", color="#C2410C")
    ax1.add_patch(patches.Rectangle((t_h_start, lane1_y - 0.26), 6.0, 0.52, 
                                    facecolor="#FED7AA", edgecolor="#EA580C", lw=1.3, zorder=3))
    ax1.text(t_h_start + 3.0, lane1_y, "$t \\in [22\\mathrm{h}, 28\\mathrm{h})$", 
             ha="center", va="center", fontsize=8.5, fontweight="bold", color="#9A3412")
    ax1.text(t_s3 + 1.5, lane1_y, "$Y_i(t; 6\\mathrm{h}) = \\mathbf{1}\\{t < T_{\\mathrm{S3},i} \\leq t + 6\\mathrm{h}\\}$", 
             ha="left", va="center", fontsize=8.8, fontweight="bold", color="#EA580C")

    lane2_y = 2.55
    ax1.text(t_h_start - 1.5, lane2_y, "Anchor-Eligible Decisions:", ha="right", va="center", 
             fontsize=8.8, fontweight="bold", color="#15803D")
    ax1.add_patch(patches.Rectangle((t_h_start, lane2_y - 0.26), 6.0, 0.52, 
                                    facecolor="#BBF7D0", edgecolor="#16A34A", lw=1.3, zorder=3))
    ax1.text(t_h_start + 3.0, lane2_y, "$t + \\delta \\leq C_i$", 
             ha="center", va="center", fontsize=8.5, fontweight="bold", color="#14532D")
    ax1.text(t_s3 + 1.5, lane2_y, "$\\mathcal{A}_i^{(C)}(\\delta) = \\{t \\in \\mathcal{T}_i : 0 < T_{\\mathrm{S3},i}-t \\leq 6\\mathrm{h}, \\; t+\\delta \\leq C_i\\}$  ($O_i=1, C_i=T_{\\mathrm{susp}}$)", 
             ha="left", va="center", fontsize=8.4, color="#15803D", fontweight="bold")

    lane3_y = 1.25
    ax1.text(t_susp - 1.5, lane3_y, "Post-Anchor Period:", ha="right", va="center", 
             fontsize=8.5, color="#475569")
    ax1.add_patch(patches.Rectangle((t_susp, lane3_y - 0.26), 4.5, 0.52, 
                                    facecolor="#F1F5F9", edgecolor="#64748B", linestyle="--", lw=1.2, zorder=3))
    ax1.text(t_susp + 2.25, lane3_y, "$t > C_i$", 
             ha="center", va="center", fontsize=8.5, color="#475569")
    ax1.text(t_susp + 2.25, lane3_y - 0.40, "[Conceptual period outside surveillance stream; censored in $\\mathcal{R}_0^{\\mathrm{hybrid}}$ & $\\mathcal{R}_2^{\\mathrm{hybrid}}$]", 
             ha="center", va="top", fontsize=7.6, color="#475569")

    # Mathematics Card - positioned nicely to fill down to the bottom
    formula_text = (
        "MATHEMATICAL RISK SET FORMULATION (§2.3):\n"
        "• Evaluated Base Set ($\\mathcal{R}$):       $\\mathcal{R} = \\{(i,t) : i \\in \\mathcal{C}, \\; t \\in \\mathcal{T}_i, \\; t < T_{\\mathrm{S3},i}\\}$   (97,700 windows across 8,254 stays)\n"
        "• Anticipation Label ($H=6\\mathrm{h}$):    $Y_i(t; 6\\mathrm{h}) = \\mathbf{1}\\{t < T_{\\mathrm{S3}, i} \\leq t + 6\\mathrm{h}\\}$   (678 pos / 97,022 neg; prevalence $\\pi = 0.694\\%$)\n"
        "• Anchor-Restricted Risk Sets:  $\\mathcal{R}_\\delta^{(C)} = \\{(i,t) \\in \\mathcal{R} : t + \\delta \\leq C_i\\}$\n"
        "    — Exploratory Hybrid Analysis ($C_i = T_{\\mathrm{susp}}$ for cases, $T_{\\mathrm{Rx,start}}$ for controls):  $\\mathcal{R}_0^{\\mathrm{hybrid}} = 70,192$;  $\\mathcal{R}_2^{\\mathrm{hybrid}} = 70,129$ windows\n"
        "    — Uniform Earliest-Rx Analysis ($C_i = T_{\\mathrm{Rx,start}}$ for all stays):  $\\mathcal{R}_0^{\\mathrm{Rx}} = 65,926$;  $\\mathcal{R}_2^{\\mathrm{Rx}} = 65,877$ windows ($N_+=0$, non-estimable)"
    )
    ax1.text(-14.5, -0.05, formula_text, fontsize=8.9, color="#0F172A", family="monospace", va="top",
             linespacing=1.40,
             bbox=dict(boxstyle="round,pad=0.65", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1.2))

    # Panel B
    ax2.set_title("B. Population Opportunity Attrition", fontweight="bold", pad=16, fontsize=13)
    categories = [
        "Total Incident Sepsis",
        "Surveillance Covered\n($V=1$)",
        "Eligible under\nmidnight assignment",
        "Indeterminate\nOpportunity",
    ]
    m3_vals = [100.0, 9.4, 0.03, 3.07]
    m4_vals = [100.0, 37.0, 36.8, 0.2]
    m3_labels = ["100.0% (3,581/3,581)", "9.4% (337/3,581)", "0.03% (1/3,581)", "3.1% (110/3,581)"]
    m4_labels = ["100.0% (408/408)", "37.0% (151/408)", "36.8% (150/408)", "0.2% (1/408)"]

    y = np.arange(len(categories))
    height = 0.35
    rects1 = ax2.barh(y + height/2, m3_vals, height, 
                      label="Broad Historical MIMIC-III (Earliest Rx Proxy: $T_{\\mathrm{Rx,start}},\\; N=3,581$)", 
                      color="#93C5FD", edgecolor="#1D4ED8", hatch="//", lw=1.1)
    rects2 = ax2.barh(y - height/2, m4_vals, height, 
                      label="MIMIC-III MetaVision (Primary Suspicion Anchor: $T_{\\mathrm{susp}},\\; N=408$)", 
                      color="#1E40AF", edgecolor="#0F172A", lw=1.1)

    ax2.set_xlabel("Proportion of Incident Sepsis Episodes (%)", fontsize=10.5, fontweight="bold", labelpad=8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(categories, fontsize=9.2)
    ax2.set_xlim(0, 132)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#CBD5E1", fontsize=8.2)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    for r, text_lbl in zip(rects1, m3_labels):
        w = r.get_width()
        ax2.text(w + 1.8, r.get_y() + r.get_height()/2, text_lbl, va="center", fontsize=8.2, color="#1D4ED8", fontweight="bold")

    for r, text_lbl in zip(rects2, m4_labels):
        w = r.get_width()
        ax2.text(w + 1.8, r.get_y() + r.get_height()/2, text_lbl, va="center", fontsize=8.2, color="#0F172A", fontweight="bold")


    out_png = FIG_DIR / "Figure1_Clocks_and_Opportunity_Waterfall.png"
    out_pdf = FIG_DIR / "Figure1_Clocks_and_Opportunity_Waterfall.pdf"
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()

    # Also sync to manuscript latex figures dir
    latex_fig_dir = PROJECT_ROOT / "manuscript" / "latex" / "figures"
    if latex_fig_dir.exists():
        import shutil
        shutil.copy(out_pdf, latex_fig_dir / "Figure1_Clocks_and_Opportunity_Waterfall.pdf")
        shutil.copy(out_png, latex_fig_dir / "Figure1_Clocks_and_Opportunity_Waterfall.png")
        logger.info("Synchronized Figure 1 to %s", latex_fig_dir)


def generate_figure2_operational_frontiers():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    t3_path = TABLE_DIR / "Table3_Alert_Budget_Refractory_Workload.csv"
    if not t3_path.exists():
        logger.warning("Table 3 not found, skipping Figure 2 generation.")
        return

    df_t3 = pd.read_csv(t3_path)

    # Differentiated markers and hollow styles to make coincident points 100% distinguishable
    model_display = {
        "BiMamba SSM (Proposed)": {
            "label": "BiMamba SSM", "color": "#2563EB", "ls": "-", "marker": "s", "ms": 7.2, 
            "mfc": "#2563EB", "mec": "#1D4ED8", "mew": 1.2, "zorder": 4
        },
        "Multi-Model Stacking Ensemble": {
            "label": "Stacking Ensemble", "color": "#DC2626", "ls": "--", "marker": "o", "ms": 11.0, 
            "mfc": "none", "mec": "#DC2626", "mew": 2.2, "zorder": 5
        },
        "Tabular GBDT (LightGBM)": {
            "label": "LightGBM", "color": "#16A34A", "ls": "-.", "marker": "^", "ms": 11.5, 
            "mfc": "none", "mec": "#16A34A", "mew": 2.0, "zorder": 6
        },
        "Tabular GBDT (XGBoost)": {
            "label": "XGBoost", "color": "#9333EA", "ls": (0, (4, 1.5, 1, 1.5)), "marker": "v", "ms": 9.5, 
            "mfc": "none", "mec": "#9333EA", "mew": 2.0, "zorder": 5
        },
        "Random Forest": {
            "label": "Random Forest", "color": "#D97706", "ls": ":", "marker": "d", "ms": 6.8, 
            "mfc": "#D97706", "mec": "#92400E", "mew": 1.2, "zorder": 4
        },
    }

    budgets = [5.01, 10.00, 20.00]

    # --- Left Panel: Episode PPV vs Budget ---
    ax1.set_title("A. Episode-Level PPV vs. Emitted Alert Budget", fontweight="bold", pad=12)
    for mname, cfg in model_display.items():
        sub = df_t3[df_t3["Model Architecture"] == mname]
        if len(sub) == 0:
            continue
        ppv_vals = [float(s.split(" ")[0]) for s in sub["Episode PPV % [95% CI]"]]
        ax1.plot(budgets, ppv_vals, label=cfg["label"], color=cfg["color"], linestyle=cfg["ls"], 
                 marker=cfg["marker"], markersize=cfg["ms"], markerfacecolor=cfg["mfc"], 
                 markeredgecolor=cfg["mec"], markeredgewidth=cfg["mew"], lw=2.2, zorder=cfg["zorder"])

    # Coincident point callout for BiMamba = LightGBM at budget 10
    ax1.annotate("BiMamba = LightGBM (1.72%)", xy=(10.0, 1.72), xytext=(10.0, 1.22),
                 arrowprops=dict(arrowstyle="->", color="#0F172A", lw=0.9),
                 fontsize=8.0, fontweight="bold", color="#0F172A", ha="center",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor="#94A3B8", lw=0.8))

    ax1.set_xlabel("Emitted Alert Budget (episodes / 100 surveillance patient-days)", fontsize=10.5, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Episode-Level PPV (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_xticks(budgets)
    ax1.set_xticklabels(["5", "10", "20"])
    ax1.set_ylim(0, 3.5)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9.0)

    # --- Right Panel: 12h Refractory Workload (NNE) ---
    ax2.set_title("B. Evaluation Workload (NNE) per Detected Sepsis Patient", fontweight="bold", pad=12)
    for mname, cfg in model_display.items():
        sub = df_t3[df_t3["Model Architecture"] == mname]
        if len(sub) == 0:
            continue
        nne_vals = []
        for s in sub["NNE per Detected Patient [95% CI]"]:
            val_str = s.split(" ")[0]
            nne_vals.append(np.nan if val_str == "Undefined" else float(val_str))
        ax2.plot(budgets, nne_vals, label=cfg["label"], color=cfg["color"], linestyle=cfg["ls"], 
                 marker=cfg["marker"], markersize=cfg["ms"], markerfacecolor=cfg["mfc"], 
                 markeredgecolor=cfg["mec"], markeredgewidth=cfg["mew"], lw=2.2, zorder=cfg["zorder"])

    # Coincident point callout for BiMamba = LightGBM at budget 10
    ax2.annotate("BiMamba = LightGBM (58.1)", xy=(10.0, 58.1), xytext=(10.0, 75.0),
                 arrowprops=dict(arrowstyle="->", color="#0F172A", lw=0.9),
                 fontsize=8.0, fontweight="bold", color="#0F172A", ha="center",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor="#94A3B8", lw=0.8))

    ax2.set_xlabel("Emitted Alert Budget (episodes / 100 surveillance patient-days)", fontsize=10.5, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Emitted Alert Episodes per Detected Patient (NNE)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_xticks(budgets)
    ax2.set_xticklabels(["5", "10", "20"])
    ax2.set_ylim(20, 115)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9.0)

    plt.tight_layout()
    out_png = FIG_DIR / "Figure2_Operational_Alert_Budget_Frontier.png"
    out_pdf = FIG_DIR / "Figure2_Operational_Alert_Budget_Frontier.pdf"
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()

    # Sync to manuscript latex figures dir
    latex_fig_dir = PROJECT_ROOT / "manuscript" / "latex" / "figures"
    if latex_fig_dir.exists():
        import shutil
        shutil.copy(out_pdf, latex_fig_dir / "Figure2_Operational_Alert_Budget_Frontier.pdf")
        shutil.copy(out_png, latex_fig_dir / "Figure2_Operational_Alert_Budget_Frontier.png")
        logger.info("Synchronized Figure 2 to %s", latex_fig_dir)

    logger.info("Saved Figure 2 to %s and %s", out_png, out_pdf)


def generate_figure3_decomposition():
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_title("Cumulative Population Coverage and Anchor-Conditioned Detection", fontweight="bold", pad=16, fontsize=13)

    stages = [
        "1. Total Incident Sepsis\n[N=408 / 408]",
        "2. Surveillance Covered\n[N=151 / 408]",
        "3. Suspicion-Anchor Eligible\n[N=150 / 408]",
        "4. Model-Alert Policy Detection\n[N=7 / 408]",
    ]
    values = [100.0, 37.01, 36.76, 1.72]
    colors = ["#334155", "#0284C7", "#16A34A", "#DC2626"]

    bars = ax.bar(stages, values, color=colors, width=0.48, edgecolor="#0F172A", lw=1.2, zorder=3)
    ax.set_ylabel("Cumulative Proportion of Sepsis Episodes (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylim(0, 118)
    ax.set_xlim(-0.6, 3.75)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for bar, val in zip(bars, values):
        h = bar.get_height()
        y_pos = h + 2.4 if h > 5 else h + 2.0
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{val:.1f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#0F172A")

    # Annotations with off-center arrows that never collide with percentage labels
    ax.annotate(
        "Surveillance Truncation\nTransition: 151/408 = 37.0%\n(257 stays develop sepsis after h36)",
        xy=(1.22, 37.0), xytext=(1.50, 75),
        arrowprops=dict(arrowstyle="->", color="#0284C7", lw=1.8),
        fontsize=9.0, color="#0369A1", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F0F9FF", edgecolor="#BAE6FD", lw=1.0)
    )

    ax.annotate(
        "Suspicion-Anchor Eligibility\nTransition: 150/151 = 99.3%\n[149/151 = 98.7% for 2h margin]",
        xy=(2.22, 36.8), xytext=(2.45, 56),
        arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.8),
        fontsize=9.0, color="#15803D", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F0FDF4", edgecolor="#BBF7D0", lw=1.0)
    )

    ax.annotate(
        "Model-Alert Policy Detection\nTransition: 7/150 = 4.7%\n(10 ep/100 pt-days budget & 12h suppression)",
        xy=(3.22, 2.0), xytext=(3.15, 24),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.8),
        fontsize=9.0, color="#B91C1C", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF2F2", edgecolor="#FECACA", lw=1.0)
    )

    plt.tight_layout()
    out_png = FIG_DIR / "Figure3_Three_Stage_Decomposition.png"
    out_pdf = FIG_DIR / "Figure3_Three_Stage_Decomposition.pdf"
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()

    # Sync to manuscript latex figures dir
    latex_fig_dir = PROJECT_ROOT / "manuscript" / "latex" / "figures"
    if latex_fig_dir.exists():
        import shutil
        shutil.copy(out_pdf, latex_fig_dir / "Figure3_Three_Stage_Decomposition.pdf")
        shutil.copy(out_png, latex_fig_dir / "Figure3_Three_Stage_Decomposition.png")
        logger.info("Synchronized Figure 3 to %s", latex_fig_dir)

    logger.info("Saved Figure 3 to %s and %s", out_png, out_pdf)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    generate_figure1_clocks_and_waterfall()
    generate_figure2_operational_frontiers()
    generate_figure3_decomposition()
    logger.info("All figures generated successfully!")


if __name__ == "__main__":
    main()
