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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.18, 1]})

    # --- Panel A: Operational Clocks & Actionable Pretreatment Decision Window ---
    ax1.set_title("A. Operational Clocks & Actionable Decision Boundary", fontweight="bold", pad=12)
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-0.8, 6.5)
    ax1.axis("off")

    # Draw ICU timeline bar
    ax1.add_patch(patches.Rectangle((0, 4.5), 10, 0.4, color="#E2E8F0", zorder=1))
    ax1.text(0, 5.1, "ICU Admission\n($T_{\\mathrm{admit}}$)", ha="center", fontsize=9.5, fontweight="bold")
    ax1.text(2.5, 5.1, "Surveillance Start\n($t=19\\mathrm{h}$, $b=18\\mathrm{h}$ context)", ha="center", fontsize=8.5, color="#1E293B")
    ax1.text(5.5, 5.1, "Antimicrobials ($T_{\\mathrm{treatment}}$)\n[First Confirmed Dose]", ha="center", fontsize=9, color="#2563EB", fontweight="bold")
    ax1.text(8.5, 5.1, "Operational Sepsis-3 ($T_{\\mathrm{S3}}$)\n[Culture + Abx + $\\Delta$SOFA$\\geq$2]", ha="center", fontsize=9, color="#DC2626", fontweight="bold")

    # Markers on timeline
    ax1.plot([0, 2.5, 5.5, 8.5], [4.7, 4.7, 4.7, 4.7], "o", color="#0F172A", markersize=8, zorder=2)
    ax1.plot(5.5, 4.7, "o", color="#2563EB", markersize=9, zorder=3)
    ax1.plot(8.5, 4.7, "o", color="#DC2626", markersize=9, zorder=3)

    # Diagnostic lead gap G = T_S3 - T_treatment = +24.2h
    ax1.annotate(
        "", xy=(5.5, 4.3), xytext=(8.5, 4.3),
        arrowprops=dict(arrowstyle="<->", color="#475569", lw=1.5)
    )
    ax1.text(7.0, 3.85, "Diagnostic Lead Gap\n$G = T_{\\mathrm{S3}} - T_{\\mathrm{treatment}} = +24.2\\mathrm{h}$ (median)",
             ha="center", fontsize=8.5, color="#334155", fontweight="bold")

    # Lookback horizon H=6h before Sepsis-3
    ax1.add_patch(patches.Rectangle((6.0, 2.7), 2.5, 0.5, facecolor="#FED7AA", alpha=0.8, zorder=1))
    ax1.text(7.25, 2.95, "Anticipation Horizon $Y_i(t; H)=\\mathbf{1}\\{t < T_{\\mathrm{S3}} \\leq t+6\\mathrm{h}\\}$", ha="center", va="center", fontsize=8.0, color="#C2410C", fontweight="bold")

    # Actionable Decision Window A_i (intersection of lookback and pretreatment)
    ax1.add_patch(patches.Rectangle((2.5, 1.6), 3.0, 0.5, facecolor="#BBF7D0", alpha=0.9, zorder=1))
    ax1.text(4.0, 1.85, "Pretreatment Decision Window $\\mathcal{A}_i(\\delta) = \\{t \\in \\mathcal{T}_i : 0 < T_{\\mathrm{S3},i}-t \\leq H, t+\\delta \\leq T_{\\mathrm{treatment},i}\\}$", ha="center", va="center", fontsize=7.5, color="#15803D", fontweight="bold")

    # Post-treatment observation period
    ax1.add_patch(patches.Rectangle((5.5, 0.5), 3.0, 0.5, facecolor="#E2E8F0", alpha=0.9, edgecolor="#64748B", linestyle="--", zorder=1))
    ax1.text(7.0, 0.75, "Post-treatment observation period ($t > T_{\\mathrm{treatment}}$)\n[Retained in $\\mathcal{R}$; censored in $\\mathcal{R}_0$ and $\\mathcal{R}_2$]", ha="center", va="center", fontsize=7.5, color="#334155")

    # Mathematical formulation box adhering to Section 2.3
    formula_text = (
        "Mathematical Formalization (§2.3):\n"
        "• Evaluated Risk Set:  $\\mathcal{R} = \\{(i,t) : i \\in \\mathcal{C}, t \\in \\mathcal{T}_i, t < T_{\\mathrm{S3}, i}\\}$ (97,700 windows across 8,254 stays)\n"
        "• Anticipation Label:  $Y_i(t; H) = \\mathbf{1}\\{t < T_{\\mathrm{S3}, i} \\leq t + H\\}$ ($H = 6\\mathrm{h}$; 678 pos / 97,022 neg, $\\pi = 0.694\\%$)\n"
        "• Pretreatment Sets:   $\\mathcal{R}_\\delta = \\{(i,t) \\in \\mathcal{R} : t + \\delta \\leq T_{\\mathrm{treatment}, i}\\}$ ($\\mathcal{R}_0$: 70,192; $\\mathcal{R}_2$: 70,129 windows)"
    )
    ax1.text(0.0, -0.65, formula_text, fontsize=8.2, color="#0F172A", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1))

    # --- Panel B: Opportunity Waterfall with Interval Bounds ---
    ax2.set_title("B. Population Opportunity Attrition", fontweight="bold", pad=12)

    categories = [
        "Total Incident Sepsis",
        "Surveillance Covered (V=1)",
        "Definitely Actionable (O=1)",
        "Indeterminate Opportunity",
    ]
    # Data from Table 4:
    m3_vals = [100.0, 9.4, 0.03, 3.07]
    m4_vals = [100.0, 37.0, 36.8, 0.0]

    y = np.arange(len(categories))
    height = 0.35

    rects1 = ax2.barh(y + height/2, m3_vals, height, label="MIMIC-III (Date Proxy with Interval Bounds)", color="#93C5FD", edgecolor="#1D4ED8", hatch="//")
    rects2 = ax2.barh(y - height/2, m4_vals, height, label="MIMIC-IV (Hourly Confirmed eMAR)", color="#1E40AF", edgecolor="#0F172A")

    ax2.set_xlabel("Proportion of Incident Sepsis Episodes (%)")
    ax2.set_yticks(y)
    ax2.set_yticklabels(categories)
    ax2.set_xlim(0, 115)
    ax2.legend(loc="center right", frameon=True)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    for r in rects1:
        w = r.get_width()
        ax2.text(w + 1.5, r.get_y() + r.get_height()/2, f"{w:.1f}%", va="center", fontsize=8.5, color="#1D4ED8")
    for r in rects2:
        w = r.get_width()
        ax2.text(w + 1.5, r.get_y() + r.get_height()/2, f"{w:.1f}%", va="center", fontsize=8.5, color="#0F172A", fontweight="bold")

    plt.tight_layout()
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

    logger.info("Saved Figure 1 to %s and %s", out_png, out_pdf)
    plt.close()
    logger.info("Saved Figure 1 to %s and %s", out_png, out_pdf)


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
