#!/usr/bin/env python3
"""generate_jamia_figures.py

Generates publication-quality figures for JAMIA Research and Applications:
  - Figure 1: Operational Clocks, Timestamp Precision Hierarchy, and Opportunity Waterfall
  - Figure 2: Operational Alert-Budget Frontier (Raw vs 12h Refractory Workload)
  - Figure 3: Three-Stage Coverage & Actionability Decomposition (S -> V -> O -> D)
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

# Set style
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.1, 1]})

    # --- Panel A: Operational Clocks & Measurement Hierarchy ---
    ax1.set_title("A. Operational Clocks & Timestamp Precision Hierarchy", fontweight="bold", pad=12)
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-0.5, 6.5)
    ax1.axis("off")

    # Draw ICU timeline bar
    ax1.add_patch(patches.Rectangle((0, 4.5), 10, 0.4, color="#E2E8F0", zorder=1))
    ax1.text(0, 5.1, "ICU Admission ($T_{\\mathrm{admit}}$)", ha="center", fontsize=10, fontweight="bold")
    ax1.text(2.5, 5.1, "Surveillance Start ($t=19\\mathrm{h}$)\n[$b=18\\mathrm{h}$ History Context]", ha="center", fontsize=9, color="#1E293B")
    ax1.text(6.0, 5.1, "Sepsis-3 Onset ($T_{\\mathrm{S3}}$)\n[Acute SOFA $\\Delta \\geq 2$]", ha="center", fontsize=9, color="#DC2626", fontweight="bold")
    ax1.text(8.5, 5.1, "Antibiotics ($T_{\\mathrm{treat}}$)\n[Systemic IV]", ha="center", fontsize=9, color="#2563EB", fontweight="bold")

    # Markers on timeline
    ax1.plot([0, 2.5, 6.0, 8.5], [4.7, 4.7, 4.7, 4.7], "o", color="#0F172A", markersize=8, zorder=2)
    ax1.plot(6.0, 4.7, "o", color="#DC2626", markersize=9, zorder=3)
    ax1.plot(8.5, 4.7, "o", color="#2563EB", markersize=9, zorder=3)

    # Prediction window
    ax1.annotate(
        "", xy=(6.0, 4.3), xytext=(4.0, 4.3),
        arrowprops=dict(arrowstyle="<->", color="#DC2626", lw=2)
    )
    ax1.text(5.0, 3.9, "Prediction Horizon ($H=6\\mathrm{h}$)", ha="center", fontsize=9, color="#DC2626")

    # Lead time buffer
    ax1.annotate(
        "", xy=(8.5, 4.3), xytext=(6.0, 4.3),
        arrowprops=dict(arrowstyle="<->", color="#2563EB", lw=2)
    )
    ax1.text(7.25, 3.9, "Lead Gap ($G = +24.2\\mathrm{h}$)", ha="center", fontsize=9, color="#2563EB")

    # Evidence tiers
    ax1.text(0, 2.8, "Evidence Hierarchy for Clinical Treatment Timing:", fontsize=11, fontweight="bold", color="#0F172A")
    tiers = [
        ("Tier 1 (MIMIC-IV eMAR):", "Confirmed administration with minute-level timestamps (Primary Actionability Benchmark)", "#16A34A"),
        ("Tier 2 (MIMIC-IV POE):", "Timestamped computerized order entry (Order-based lead time)", "#0284C7"),
        ("Tier 3 (MIMIC-III Rx):", "Calendar-date proxy [00:00, 24:00) with interval censoring (Bounded Sensitivity Proxy)", "#D97706"),
    ]
    y_pos = 2.1
    for title, desc, col in tiers:
        ax1.add_patch(patches.Rectangle((0, y_pos - 0.1), 0.25, 0.35, color=col))
        ax1.text(0.4, y_pos + 0.05, title, fontsize=9.5, fontweight="bold", color=col)
        ax1.text(0.4, y_pos - 0.22, desc, fontsize=8.5, color="#475569")
        y_pos -= 0.75

    # --- Panel B: Bounded Opportunity Waterfall ---
    ax2.set_title("B. Opportunity Attrition & Interval Uncertainty", fontweight="bold", pad=12)
    categories = [
        "Total Incident Sepsis",
        "Pre-ICU / Same-Day",
        "Definitely In-ICU",
        "Covered in Day 2 [19–30h]",
        "Pretreatment Valid (O=1)",
    ]
    # Counts
    # MIMIC-III (N=3581)
    m3_vals = [100.0, 77.4, 22.6, 37.0, 36.8]  # % of sepsis
    # MIMIC-IV (N=1596)
    m4_vals = [100.0, 16.4, 83.6, 38.2, 38.1]

    y = np.arange(len(categories))
    height = 0.35

    rects1 = ax2.barh(y + height/2, m3_vals, height, label="MIMIC-III (Interval Bounded)", color="#93C5FD", edgecolor="#1D4ED8", hatch="//")
    rects2 = ax2.barh(y - height/2, m4_vals, height, label="MIMIC-IV (Hourly Confirmed)", color="#1E40AF", edgecolor="#0F172A")

    ax2.set_xlabel("Proportion of Sepsis Episodes (%)")
    ax2.set_yticks(y)
    ax2.set_yticklabels(categories)
    ax2.set_xlim(0, 115)
    ax2.legend(loc="lower right", frameon=True)
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

    budgets = [5, 10, 20]
    models = {
        "★ Champion (Ensemble)": {"color": "#DC2626", "ls": "-", "marker": "o"},
        "BiMamba SSM": {"color": "#2563EB", "ls": "--", "marker": "s"},
        "Tabular GBDT (LightGBM)": {"color": "#16A34A", "ls": "-.", "marker": "^"},
        "Random Forest": {"color": "#D97706", "ls": ":", "marker": "d"},
    }

    # Data from Table 3
    # Window PPV %
    ppv_data = {
        "★ Champion (Ensemble)": [2.94, 2.95, 2.33],
        "BiMamba SSM": [4.41, 2.95, 1.97],
        "Tabular GBDT (LightGBM)": [3.43, 2.46, 1.60],
        "Random Forest": [2.45, 1.72, 1.47],
    }

    # Population Sensitivity % (out of 408 septic stays)
    sens_pop_data = {
        "★ Champion (Ensemble)": [0.74, 1.47, 1.23],
        "BiMamba SSM": [1.23, 1.23, 1.47],
        "Tabular GBDT (LightGBM)": [1.23, 1.47, 2.21],
        "Random Forest": [0.74, 0.74, 1.47],
    }

    # NNE Patient Workload
    nne_data = {
        "★ Champion (Ensemble)": [38.0, 38.5, 84.4],
        "BiMamba SSM": [23.2, 45.4, 65.2],
        "Tabular GBDT (LightGBM)": [28.4, 44.0, 58.4],
        "Random Forest": [40.0, 74.3, 71.8],
    }

    # --- Left Panel: Window PPV vs Budget ---
    ax1.set_title("A. Window-Level Sepsis-3 PPV vs. Alert Budget", fontweight="bold")
    for name, cfg in models.items():
        ax1.plot(budgets, ppv_data[name], label=name, color=cfg["color"], linestyle=cfg["ls"], marker=cfg["marker"], lw=2.2, ms=7)

    ax1.set_xlabel("Alert Budget (alerts / 100 eligible surveillance patient-days)")
    ax1.set_ylabel("Window Sepsis-3 PPV (%)")
    ax1.set_xticks(budgets)
    ax1.set_ylim(0, 6)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # Annotate collapse
    ax1.axhspan(0, 3.0, color="#FEE2E2", alpha=0.4, label="Sub-3% PPV Zone")
    ax1.text(12, 0.5, "Severe False-Alarm Fatigue (PPV < 3%)", color="#B91C1C", fontsize=9, fontweight="bold")

    # --- Right Panel: 12h Refractory Workload (NNE) ---
    ax2.set_title("B. Evaluation Workload (NNE) per Unique Detected Sepsis Patient", fontweight="bold")
    for name, cfg in models.items():
        ax2.plot(budgets, nne_data[name], label=name, color=cfg["color"], linestyle=cfg["ls"], marker=cfg["marker"], lw=2.2, ms=7)

    ax2.set_xlabel("Alert Budget (alerts / 100 eligible surveillance patient-days)")
    ax2.set_ylabel("Refractory Alert Episodes per Detected Patient (NNE)")
    ax2.set_xticks(budgets)
    ax2.set_ylim(15, 100)
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
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Three-Stage Population Coverage & Actionability Decomposition", fontweight="bold", pad=14)

    stages = [
        "Total Incident Sepsis\nP(S=1)\n[N=408]",
        "Surveillance Coverage\nP(V=1 | S=1)\n[N=151, 37.0%]",
        "Pretreatment Valid\nP(O=1 | V=1, S=1)\n[N=150, 99.3%]",
        "Model Timely Alert\nP(D=1 | O=1, V=1, S=1)\n[N=6, 4.0%]",
        "Population Yield\nP(D=1 | S=1)\n[N=6, 1.5%]",
    ]
    values = [100.0, 37.01, 36.76, 1.47, 1.47]
    colors = ["#334155", "#0284C7", "#16A34A", "#DC2626", "#7C3AED"]

    bars = ax.bar(stages, values, color=colors, width=0.55, edgecolor="#0F172A", lw=1.2)
    ax.set_ylabel("Population Proportion (%)")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2.5, f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

    # Add explanatory annotations
    ax.annotate(
        "Surveillance Truncation\n(Day 2 hours 19–30 only;\n257 cases develop sepsis later)",
        xy=(1, 37), xytext=(1.4, 75),
        arrowprops=dict(arrowstyle="->", color="#0284C7", lw=1.8),
        fontsize=9, color="#0369A1", ha="center"
    )
    ax.annotate(
        "Operational Alert Budget\nConstraint (10 alerts/100 pt-days;\n6 unique patients detected)",
        xy=(3.5, 2), xytext=(3.5, 35),
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
