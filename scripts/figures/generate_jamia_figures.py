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
    fig = plt.figure(figsize=(16.8, 8.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.34, 1.0], wspace=0.36, left=0.035, right=0.975, top=0.93, bottom=0.07)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ==============================================================================
    # Panel A: Operational Clocks & Actionable Decision Boundary
    # ==============================================================================
    ax1.set_title("A. Operational Clocks & Actionable Decision Boundary", fontweight="bold", pad=16, fontsize=12.5)
    ax1.set_xlim(-15, 56)
    ax1.set_ylim(-4.1, 8.2)
    ax1.axis("off")

    # Anchor coordinates in ICU hours
    t_admit = 0.0
    t_surv_start = 19.0
    t_surv_end = 30.0
    t_s3 = 28.0      # representative covered Sepsis-3 onset
    t_susp = 51.0    # representative suspicion anchor
    t_h_start = t_s3 - 6.0  # 22.0h

    y_axis = 5.4

    # 1. Main Time Axis Line & Axis Ticks
    ax1.plot([-2, 54], [y_axis, y_axis], color="#64748B", lw=2.2, zorder=1)

    # Axis ticks along timeline
    for h in [0, 10, 20, 30, 40, 50]:
        ax1.plot([h, h], [y_axis - 0.12, y_axis + 0.12], color="#64748B", lw=1.5, zorder=2)
        ax1.text(h, y_axis - 0.35, f"{h}h", ha="center", va="top", fontsize=8.5, color="#64748B")

    # Shaded background band for Surveillance Window [19h, 30h]
    ax1.axvspan(t_surv_start, t_surv_end, ymin=0.36, ymax=0.88, facecolor="#EEF2FF", alpha=0.6, zorder=0)

    # Vertical guideline stems
    ax1.plot([t_admit, t_admit], [y_axis, y_axis + 1.25], color="#0F172A", lw=1.5, linestyle=":", zorder=2)
    ax1.plot([t_s3, t_s3], [1.1, y_axis + 1.95], color="#DC2626", lw=1.5, linestyle=":", zorder=2)
    ax1.plot([t_susp, t_susp], [0.2, y_axis + 1.25], color="#2563EB", lw=1.5, linestyle=":", zorder=2)

    # Markers on the axis
    ax1.plot(t_admit, y_axis, "o", color="#0F172A", markersize=8.5, zorder=5)
    ax1.plot(t_s3, y_axis, "o", color="#DC2626", markersize=9.5, zorder=5)
    ax1.plot(t_susp, y_axis, "o", color="#2563EB", markersize=9.5, zorder=5)

    # --- Top Callouts (Staggered vertically and separated horizontally) ---
    # ICU Admission (t=0)
    ax1.text(t_admit, y_axis + 1.35, "ICU Admission\n($T_{\\mathrm{admit}},\\; t=0$)", 
             ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#0F172A")

    # Surveillance Protocol (bracket slightly above timeline)
    surv_center = (t_surv_start + t_surv_end) / 2
    ax1.annotate("", xy=(t_surv_start, y_axis + 0.52), xytext=(t_surv_end, y_axis + 0.52),
                 arrowprops=dict(arrowstyle="<->", color="#4338CA", lw=1.5))
    ax1.text(surv_center, y_axis + 0.68, "Surveillance Window\nHours 19–30 (12 Decision Bins)", 
             ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#3730A3")

    # Sepsis-3 Onset (t=28h) - High tier (y = y_axis + 2.05)
    ax1.text(t_s3, y_axis + 2.05, "Sepsis-3 Onset ($T_{\\mathrm{S3}}$)\n[Acute $\\Delta\\mathrm{SOFA}\\geq 2$; med 28.0h]", 
             ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#DC2626")

    # Suspicion Anchor (t=51h) - Staggered below Sepsis-3 Onset
    ax1.text(t_susp, y_axis + 1.35, "Suspicion Anchor ($T_{\\mathrm{susp}}$)\n[Culture & Abx Pair; med 50.7h]", 
             ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#1D4ED8")

    # 2. Operational Search-Window Lead Gap (Double Arrow between Sepsis-3 and Suspicion)
    gap_y = 4.35
    gap_center = (t_s3 + t_susp) / 2
    ax1.annotate("", xy=(t_s3, gap_y), xytext=(t_susp, gap_y),
                 arrowprops=dict(arrowstyle="<->", color="#334155", lw=1.8))
    gap_box = (
        "Search-Window Lead Gap: $G_i = T_{\\mathrm{susp},i} - T_{\\mathrm{S3},i}$\n"
        "Lookback begins at $[T_{\\mathrm{susp}}-24\\mathrm{h}] \\rightarrow G_i < 25\\mathrm{h}$ (67.2%)\n"
        "(Cohort Median: +24.2h; IQR: 22.8–24.6h; max: 24.99h)"
    )
    ax1.text(gap_center, gap_y - 0.20, gap_box, ha="center", va="top", fontsize=8.0, color="#0F172A",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#F8FAFC", edgecolor="#94A3B8", lw=0.9))

    # 3. Horizontal Swimlanes for Horizons and Actionability
    # Lane 1: Anticipation Horizon (H = 6h lookback)
    lane1_y = 2.7
    ax1.text(t_h_start - 1.5, lane1_y, "Anticipation Horizon ($H=6\\mathrm{h}$):", ha="right", va="center", 
             fontsize=8.5, fontweight="bold", color="#C2410C")
    ax1.add_patch(patches.Rectangle((t_h_start, lane1_y - 0.24), 6.0, 0.48, 
                                    facecolor="#FED7AA", edgecolor="#EA580C", lw=1.3, zorder=3))
    ax1.text(t_h_start + 3.0, lane1_y, "$t \\in [22\\mathrm{h}, 28\\mathrm{h})$", 
             ha="center", va="center", fontsize=8.3, fontweight="bold", color="#9A3412")
    ax1.text(t_s3 + 1.5, lane1_y, "$Y_i(t; 6\\mathrm{h}) = 1$", 
             ha="left", va="center", fontsize=8.8, fontweight="bold", color="#EA580C")

    # Lane 2: Actionable Pre-Proxy Surveillance Decisions
    lane2_y = 1.6
    ax1.text(t_h_start - 1.5, lane2_y, "Actionable Decisions ($\\mathcal{A}_i$):", ha="right", va="center", 
             fontsize=8.5, fontweight="bold", color="#15803D")
    ax1.add_patch(patches.Rectangle((t_h_start, lane2_y - 0.24), 6.0, 0.48, 
                                    facecolor="#BBF7D0", edgecolor="#16A34A", lw=1.3, zorder=3))
    ax1.text(t_h_start + 3.0, lane2_y, "$t + \\delta \\leq C_i$", 
             ha="center", va="center", fontsize=8.3, fontweight="bold", color="#14532D")
    ax1.text(t_s3 + 1.5, lane2_y, "Pre-suspicion opportunity ($O_i=1$)", 
             ha="left", va="center", fontsize=8.5, color="#15803D", fontweight="bold")

    # Lane 3: Post-Anchor Censoring Region
    lane3_y = 0.5
    ax1.text(t_susp - 1.5, lane3_y, "Post-Anchor Period:", ha="right", va="center", 
             fontsize=8.2, color="#475569")
    ax1.add_patch(patches.Rectangle((t_susp, lane3_y - 0.24), 4.5, 0.48, 
                                    facecolor="#F1F5F9", edgecolor="#64748B", linestyle="--", lw=1.2, zorder=3))
    ax1.text(t_susp + 2.25, lane3_y, "$t > C_i$", 
             ha="center", va="center", fontsize=8.2, color="#475569")
    ax1.text(t_susp + 2.25, lane3_y - 0.40, "[Censored in $\\mathcal{R}_0$ & $\\mathcal{R}_2$]", 
             ha="center", va="top", fontsize=7.6, color="#475569")

    # 4. Expanded & Generously Spaced Mathematical Card at bottom
    formula_text = (
        "MATHEMATICAL RISK SET FORMULATION (§2.3)\n"
        "• Evaluated Base Set ($\\mathcal{R}$):\n"
        "    $\\mathcal{R} = \\{(i,t) : i \\in \\mathcal{C},\\; t \\in \\{19\\dots 30\\}\\mathrm{h},\\; t < T_{\\mathrm{S3}, i}\\}$  "
        "(97,700 decision windows across 8,254 stays; 4,070.8 pt-days)\n"
        "• Anticipation Horizon ($H=6\\mathrm{h}$):\n"
        "    $Y_i(t; 6\\mathrm{h}) = \\mathbf{1}\\{t < T_{\\mathrm{S3}, i} \\leq t + 6\\mathrm{h}\\}$  "
        "(678 positive / 97,022 negative windows; base prevalence $\\pi = 0.694\\%$)\n"
        "• Exploratory Hybrid Sets (Outcome-Stratified Restriction):\n"
        "    $\\mathcal{R}_\\delta^{\\mathrm{hybrid}} = \\{(i,t) \\in \\mathcal{R} : t + \\delta \\leq C_i\\}$  with $C_i = T_{\\mathrm{susp}}$ for cases and $T_{\\mathrm{Rx,start}}$ for controls\n"
        "    $\\rightarrow \\mathcal{R}_0^{\\mathrm{hybrid}}$: 70,192 windows (674 pos / 69,518 neg)  |  $\\mathcal{R}_2^{\\mathrm{hybrid}}$: 70,129 windows (663 pos / 69,466 neg)\n"
        "• Uniform Earliest-Prescription Sets (Prospective Operational Benchmark):\n"
        "    $\\mathcal{R}_\\delta^{\\mathrm{Rx}} = \\{(i,t) \\in \\mathcal{R} : t + \\delta \\leq T_{\\mathrm{Rx,start},i}\\}$  "
        "applied identically across all patients ($T_{\\mathrm{Rx,start}} \\leq 3.83\\mathrm{h}$ for cases)\n"
        "    $\\rightarrow \\mathcal{R}_0^{\\mathrm{Rx}}$: 65,926 control windows (2,746.9 pt-days; $N_+=0$; discrimination mathematically non-estimable)\n"
        "    $\\rightarrow \\mathcal{R}_2^{\\mathrm{Rx}}$: 65,877 control windows (2,744.9 pt-days; $N_+=0$; discrimination mathematically non-estimable)"
    )
    ax1.text(-14.5, -0.3, formula_text, fontsize=7.9, color="#0F172A", family="monospace", va="top",
             linespacing=1.26,
             bbox=dict(boxstyle="round,pad=0.55", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1.2))

    # ==============================================================================
    # Panel B: Population Opportunity Attrition
    # ==============================================================================
    ax2.set_title("B. Population Opportunity Attrition", fontweight="bold", pad=16, fontsize=12.5)

    categories = [
        "Total Incident Sepsis",
        "Surveillance Covered\n($V=1$)",
        "Eligible under\nmidnight assignment",
        "Indeterminate\nOpportunity",
    ]
    # Data from Table 4:
    m3_vals = [100.0, 9.4, 0.03, 3.07]
    m4_vals = [100.0, 37.0, 36.8, 0.2]

    y = np.arange(len(categories))
    height = 0.35

    rects1 = ax2.barh(y + height/2, m3_vals, height, label="Broad Historical MIMIC-III (Earliest Rx: $T_{\\mathrm{Rx,start}}$)", 
                      color="#93C5FD", edgecolor="#1D4ED8", hatch="//", lw=1.1)
    rects2 = ax2.barh(y - height/2, m4_vals, height, label="MIMIC-III MetaVision (Primary Benchmark: $T_{\\mathrm{susp}}$)", 
                      color="#1E40AF", edgecolor="#0F172A", lw=1.1)

    ax2.set_xlabel("Proportion of Incident Sepsis Episodes (%)", fontsize=10.5, fontweight="bold", labelpad=8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(categories, fontsize=9.2)
    ax2.set_xlim(0, 118)

    # Legend placed cleanly in top right corner
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#CBD5E1", fontsize=8.4)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    # Value annotations with clean spacing
    for r in rects1:
        w = r.get_width()
        text_val = f"{w:.1f}%" if w >= 0.1 else f"{w:.2f}%"
        ax2.text(w + 1.8, r.get_y() + r.get_height()/2, text_val, va="center", fontsize=8.2, color="#1D4ED8", fontweight="bold")

    for r in rects2:
        w = r.get_width()
        ax2.text(w + 1.8, r.get_y() + r.get_height()/2, f"{w:.1f}%", va="center", fontsize=8.2, color="#0F172A", fontweight="bold")

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    t3_path = TABLE_DIR / "Table3_Alert_Budget_Refractory_Workload.csv"
    if not t3_path.exists():
        logger.warning("Table 3 not found, skipping Figure 2 generation.")
        return

    df_t3 = pd.read_csv(t3_path)

    model_display = {
        "BiMamba SSM (Proposed)": {"label": "BiMamba SSM", "color": "#2563EB", "ls": "-", "marker": "s"},
        "Multi-Model Stacking Ensemble": {"label": "Stacking Ensemble", "color": "#DC2626", "ls": "--", "marker": "o"},
        "Tabular GBDT (LightGBM)": {"label": "LightGBM", "color": "#16A34A", "ls": "-.", "marker": "^"},
        "Tabular GBDT (XGBoost)": {"label": "XGBoost", "color": "#9333EA", "ls": "-.", "marker": "v"},
        "Random Forest": {"label": "Random Forest", "color": "#D97706", "ls": ":", "marker": "d"},
    }

    budgets = [5.01, 10.00, 20.00]

    # --- Left Panel: Episode PPV vs Budget ---
    ax1.set_title("A. Episode-Level PPV vs. Emitted Alert Budget", fontweight="bold")
    for mname, cfg in model_display.items():
        sub = df_t3[df_t3["Model Architecture"] == mname]
        if len(sub) == 0:
            continue
        ppv_vals = []
        for s in sub["Episode PPV % [95% CI]"]:
            val_str = s.split(" ")[0]
            ppv_vals.append(float(val_str))
        ax1.plot(budgets, ppv_vals, label=cfg["label"], color=cfg["color"], linestyle=cfg["ls"], marker=cfg["marker"], lw=2.2, ms=7)

    ax1.set_xlabel("Emitted Alert Budget (episodes / 100 surveillance patient-days)")
    ax1.set_ylabel("Episode-Level PPV (%)")
    ax1.set_xticks(budgets)
    ax1.set_ylim(0, 3.5)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # --- Right Panel: 12h Refractory Workload (NNE) ---
    ax2.set_title("B. Evaluation Workload (NNE) per Detected Sepsis Patient", fontweight="bold")
    for mname, cfg in model_display.items():
        sub = df_t3[df_t3["Model Architecture"] == mname]
        if len(sub) == 0:
            continue
        nne_vals = []
        for s in sub["NNE per Detected Patient [95% CI]"]:
            val_str = s.split(" ")[0]
            if val_str == "Undefined":
                nne_vals.append(np.nan)
            else:
                nne_vals.append(float(val_str))
        ax2.plot(budgets, nne_vals, label=cfg["label"], color=cfg["color"], linestyle=cfg["ls"], marker=cfg["marker"], lw=2.2, ms=7)

    ax2.set_xlabel("Emitted Alert Budget (episodes / 100 surveillance patient-days)")
    ax2.set_ylabel("Emitted Alert Episodes per Detected Patient (NNE)")
    ax2.set_xticks(budgets)
    ax2.set_ylim(20, 115)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    out_png = FIG_DIR / "Figure2_Operational_Alert_Budget_Frontier.png"
    out_pdf = FIG_DIR / "Figure2_Operational_Alert_Budget_Frontier.pdf"
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()
    logger.info("Saved Figure 2 to %s and %s", out_png, out_pdf)


def generate_figure3_decomposition():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title("Cumulative Population Coverage & Actionability Decomposition", fontweight="bold", pad=14)

    stages = [
        "1. Total Incident Sepsis\n100.0%\n[N=408 / 408]",
        "2. Surveillance Covered\n37.0%\n[N=151 / 408]",
        "3. Pretreatment Eligible\n36.8%\n[N=150 / 408]",
        "4. Timely Detected\n1.7%\n[N=7 / 408]",
    ]
    values = [100.0, 37.01, 36.76, 1.72]
    colors = ["#334155", "#0284C7", "#16A34A", "#DC2626"]

    bars = ax.bar(stages, values, color=colors, width=0.5, edgecolor="#0F172A", lw=1.2)
    ax.set_ylabel("Cumulative Proportion of Sepsis Episodes (%)")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2.5, f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

    # Annotations for conditional transitions
    ax.annotate(
        "Surveillance Truncation\nTransition: 151/408 = 37.0%\n(257 stays develop sepsis after h36)",
        xy=(1, 37.0), xytext=(1.35, 75),
        arrowprops=dict(arrowstyle="->", color="#0284C7", lw=1.8),
        fontsize=9, color="#0369A1", ha="center"
    )
    ax.annotate(
        "Pretreatment Actionability\nTransition: 150/151 = 99.3%\n[149/151 = 98.7% for 2h buffer]",
        xy=(2, 36.8), xytext=(2.35, 55),
        arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.8),
        fontsize=9, color="#15803D", ha="center"
    )
    ax.annotate(
        "Operational Alert Budget\nTransition: 7/150 = 4.7%\n(BiMamba SSM at 10 emitted episodes/100 pt-days)",
        xy=(3, 1.72), xytext=(3.0, 25),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.8),
        fontsize=9, color="#B91C1C", ha="center"
    )

    plt.tight_layout()
    out_png = FIG_DIR / "Figure3_Three_Stage_Decomposition.png"
    out_pdf = FIG_DIR / "Figure3_Three_Stage_Decomposition.pdf"
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()
    logger.info("Saved Figure 3 to %s and %s", out_png, out_pdf)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    generate_figure1_clocks_and_waterfall()
    generate_figure2_operational_frontiers()
    generate_figure3_decomposition()
    logger.info("All figures generated successfully!")


if __name__ == "__main__":
    main()
