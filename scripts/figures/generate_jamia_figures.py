#!/usr/bin/env python3
"""generate_jamia_figures.py

Generates publication-quality figures for JAMIA Research and Applications:
  - Figure 1: Operational Clocks, Actionable Pretreatment Windows, and Interval Uncertainty Waterfall
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.15, 1]})

    # --- Panel A: Operational Clocks & Actionable Pretreatment Decision Window ---
    ax1.set_title("A. Operational Clocks & Actionable Decision Boundary", fontweight="bold", pad=12)
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-0.5, 6.5)
    ax1.axis("off")

    # Draw ICU timeline bar
    ax1.add_patch(patches.Rectangle((0, 4.5), 10, 0.4, color="#E2E8F0", zorder=1))
    ax1.text(0, 5.1, "ICU Admission\n($T_{\\mathrm{admit}}$)", ha="center", fontsize=9.5, fontweight="bold")
    ax1.text(2.5, 5.1, "Surveillance Start\n($t=19\\mathrm{h}$, $b=18\\mathrm{h}$ context)", ha="center", fontsize=8.5, color="#1E293B")
    ax1.text(5.5, 5.1, "Antibiotics ($T_{\\mathrm{treatment}}$)\n[First Systemic IV]", ha="center", fontsize=9, color="#2563EB", fontweight="bold")
    ax1.text(8.5, 5.1, "Operational Sepsis-3 ($T_{\\mathrm{S3}}$)\n[Culture + Abx + $\\Delta$SOFA$\\geq$2]", ha="center", fontsize=9, color="#DC2626", fontweight="bold")

    # Markers on timeline
    ax1.plot([0, 2.5, 5.5, 8.5], [4.7, 4.7, 4.7, 4.7], "o", color="#0F172A", markersize=8, zorder=2)
    ax1.plot(5.5, 4.7, "o", color="#2563EB", markersize=9, zorder=3)
    ax1.plot(8.5, 4.7, "o", color="#DC2626", markersize=9, zorder=3)

    # Diagnostic lead gap G = T_S3 - T_treatment = +24.2h
    ax1.annotate(
        "", xy=(8.5, 4.3), xytext=(5.5, 4.3),
        arrowprops=dict(arrowstyle="<->", color="#0F172A", lw=2)
    )
    ax1.text(7.0, 3.9, "Diagnostic Lead Gap\n($G = T_{\\mathrm{S3}} - T_{\\mathrm{treatment}} = +24.2\\mathrm{h}$)", ha="center", fontsize=9, color="#0F172A", fontweight="bold")

    # Actionable Decision Window: t in [T_S3 - H, min(T_S3, T_treatment - delta)]
    ax1.annotate(
        "", xy=(5.0, 3.3), xytext=(2.5, 3.3),
        arrowprops=dict(arrowstyle="<->", color="#16A34A", lw=2.2)
    )
    ax1.text(3.75, 2.8, "Actionable Pretreatment Window\n($t + \\delta \\leq T_{\\mathrm{treatment}}$, with $\\delta$ response lead buffer)", ha="center", fontsize=8.5, color="#15803D", fontweight="bold")

    # Evidence tiers
    ax1.text(0, 2.2, "Evidence Hierarchy for Clinical Treatment Timing:", fontsize=10.5, fontweight="bold", color="#0F172A")
    tiers = [
        ("Tier 1 (MIMIC-IV eMAR):", "Confirmed administration with minute-level timestamps (Primary Actionability Benchmark)", "#16A34A"),
        ("Tier 2 (MIMIC-IV POE):", "Computerized order entry and prescription start time (Order-based lead time)", "#0284C7"),
        ("Tier 3 (MIMIC-III Rx):", "Calendar-date proxy [00:00, 24:00) with interval censoring (Bounded Sensitivity Proxy)", "#D97706"),
    ]
    y_pos = 1.65
    for title, desc, col in tiers:
        ax1.add_patch(patches.Rectangle((0, y_pos - 0.1), 0.25, 0.3, color=col))
        ax1.text(0.4, y_pos + 0.05, title, fontsize=9, fontweight="bold", color=col)
        ax1.text(0.4, y_pos - 0.2, desc, fontsize=8, color="#475569")
        y_pos -= 0.65

    # --- Panel B: Bounded Opportunity Waterfall ---
    ax2.set_title("B. Opportunity Attrition & Timestamp Uncertainty", fontweight="bold", pad=12)
    categories = [
        "Total Incident Sepsis",
        "Surveillance Covered (V=1)",
        "Definitely Actionable (O=1)",
        "Indeterminate Opportunity",
    ]
    # Data from Table 4:
    # MIMIC-III (N=3581): Total=100.0%, Covered=9.4%, DefElig=0.03%, Indet=3.07%
    m3_vals = [100.0, 9.4, 0.03, 3.07]
    # MIMIC-IV (N=408): Total=100.0%, Covered=37.0%, DefElig=36.8%, Indet=0.0%
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
    logger.info("Saved Figure 1 to %s and %s", out_png, out_pdf)


def generate_figure2_operational_frontiers():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    budgets = [5.01, 10.00, 20.00]
    models = {
        "BiMamba SSM": {"color": "#2563EB", "ls": "-", "marker": "s"},
        "Convex Ensemble": {"color": "#DC2626", "ls": "--", "marker": "o"},
        "GBDT (LightGBM)": {"color": "#16A34A", "ls": "-.", "marker": "^"},
        "Random Forest": {"color": "#D97706", "ls": ":", "marker": "d"},
    }

    # Episode PPV % from Table 3
    ppv_data = {
        "BiMamba SSM": [1.96, 1.72, 1.23],
        "Convex Ensemble": [2.94, 1.47, 0.98],
        "GBDT (LightGBM)": [2.94, 1.72, 1.23],
        "Random Forest": [1.47, 1.47, 1.11],
    }

    # NNE Patient Workload from Table 3
    nne_data = {
        "BiMamba SSM": [51.0, 58.1, 81.4],
        "Convex Ensemble": [34.0, 67.8, 101.8],
        "GBDT (LightGBM)": [34.0, 58.1, 81.4],
        "Random Forest": [68.0, 67.8, 90.4],
    }

    # --- Left Panel: Episode PPV vs Budget ---
    ax1.set_title("A. Episode-Level PPV vs. Emitted Alert Budget", fontweight="bold")
    for name, cfg in models.items():
        ax1.plot(budgets, ppv_data[name], label=name, color=cfg["color"], linestyle=cfg["ls"], marker=cfg["marker"], lw=2.2, ms=7)

    ax1.set_xlabel("Emitted Alert Budget (episodes / 100 surveillance patient-days)")
    ax1.set_ylabel("Episode-Level PPV (%)")
    ax1.set_xticks(budgets)
    ax1.set_ylim(0, 4.0)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # --- Right Panel: 12h Refractory Workload (NNE) ---
    ax2.set_title("B. Evaluation Workload (NNE) per Detected Sepsis Patient", fontweight="bold")
    for name, cfg in models.items():
        ax2.plot(budgets, nne_data[name], label=name, color=cfg["color"], linestyle=cfg["ls"], marker=cfg["marker"], lw=2.2, ms=7)

    ax2.set_xlabel("Emitted Alert Budget (episodes / 100 surveillance patient-days)")
    ax2.set_ylabel("Emitted Alert Episodes per Detected Patient (NNE)")
    ax2.set_xticks(budgets)
    ax2.set_ylim(20, 120)
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
    # Cumulative coverage percentages
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
        "Pretreatment Actionability\nTransition: 150/151 = 99.3%\n(Alert opportunity precedes antibiotics)",
        xy=(2, 36.8), xytext=(2.35, 55),
        arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.8),
        fontsize=9, color="#15803D", ha="center"
    )
    ax.annotate(
        "Operational Alert Budget\nTransition: 7/150 = 4.7%\n(BiMamba SSM at 10 emitted episodes/100 pt-days)",
        xy=(3, 1.7), xytext=(3.0, 25),
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
