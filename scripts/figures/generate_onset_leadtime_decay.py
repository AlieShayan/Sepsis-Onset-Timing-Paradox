#!/usr/bin/env python3
"""generate_onset_leadtime_decay.py

Generates Figure 2 for the Onset Timing Paradox paper:
"Figure 2: Lead-Time Decay Profiles and the Delayed Chart Mirroring Paradox
Across Deep Sequence, State-Space, and Fair Transformer Models."

Displays:
  - Panel A: Lead-time decay curves under Canonical Sepsis-3 (T_sepsis3).
  - Panel B: Lead-time decay curves under Clinical Treatment Onset (T_treatment).
  - Panel C: Delayed Chart Mirroring Rate (% of model alarms firing post-treatment).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path("/home/alireza/Sepsis-Onset-Timing-Paradox")
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.prediction_loader import load_harmonized_benchmark_dataset, MODEL_NAME_MAP
from src.metrics.leadtime_decay import (
    evaluate_horizon_performance,
    compute_delayed_chart_mirroring_rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VAULT_ASSETS = Path("/home/alireza/Personalized-Sepsis-Dynamics-Vault/5-assets/figures")


def plot_figure_2(
    df_eval: pd.DataFrame,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    VAULT_ASSETS.mkdir(parents=True, exist_ok=True)

    # Models selected for clear visual comparison
    target_models = [
        "Proposed Champion (Ensemble+T-Smooth)",
        "BiMamba_SSM",
        "Transformer_FairMasked",
        "Clinical Transformer",
        "Standard Deep Bi-LSTM",
        "Tabular GBDT (XGBoost)",
    ]

    model_colors = {
        "Proposed Champion (Ensemble+T-Smooth)": "#D97706",   # Deep Amber / Gold
        "BiMamba_SSM": "#2563EB",                             # Vibrant Blue
        "Transformer_FairMasked": "#0D9488",                  # Emerald Teal
        "Clinical Transformer": "#7C3AED",                    # Purple
        "Standard Deep Bi-LSTM": "#64748B",                   # Slate
        "Tabular GBDT (XGBoost)": "#DC2626",                  # Red / Coral
    }

    model_markers = {
        "Proposed Champion (Ensemble+T-Smooth)": "o",
        "BiMamba_SSM": "s",
        "Transformer_FairMasked": "^",
        "Clinical Transformer": "v",
        "Standard Deep Bi-LSTM": "D",
        "Tabular GBDT (XGBoost)": "P",
    }

    horizons = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0]

    logger.info("Evaluating horizons under Canonical Sepsis-3...")
    perf_sep3 = evaluate_horizon_performance(df_eval, target_models, "T_SEPSIS3_H_ICU", horizons)

    logger.info("Evaluating horizons under Clinical Treatment Onset...")
    perf_treat = evaluate_horizon_performance(df_eval, target_models, "T_TREATMENT_H_ICU", horizons)

    logger.info("Computing Delayed Chart Mirroring Rates...")
    mirror_df = compute_delayed_chart_mirroring_rate(df_eval, target_models)

    # Save performance tables to reports
    perf_sep3.to_csv(out_dir.parent / "tables" / "leadtime_decay_sepsis3.csv", index=False)
    perf_treat.to_csv(out_dir.parent / "tables" / "leadtime_decay_treatment.csv", index=False)
    mirror_df.to_csv(out_dir.parent / "tables" / "delayed_chart_mirroring_rates.csv", index=False)

    # Style configuration
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "axes.edgecolor": "#334155",
        "axes.linewidth": 1.1,
    })

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)

    # ─────────────────────────────────────────────────────────────
    # Panel A: Canonical Sepsis-3 Horizon Decay
    # ─────────────────────────────────────────────────────────────
    ax1 = axes[0]
    for m in target_models:
        sub = perf_sep3[perf_sep3["model"] == m].sort_values("horizon_h")
        label = MODEL_NAME_MAP.get(m, m)
        ax1.plot(
            sub["horizon_h"], sub["auprc"] * 100,
            marker=model_markers[m], markersize=7, linewidth=2.4,
            color=model_colors[m], label=label
        )

    ax1.set_title("A. Canonical Sepsis-3 ($T_{\\mathrm{sepsis3}}$)", weight="bold", pad=12)
    ax1.set_xlabel("Warning Horizon Ahead of Sepsis-3 [Hours]")
    ax1.set_ylabel("Window-Level AUPRC (%)")
    ax1.set_xticks(horizons)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.92)

    # ─────────────────────────────────────────────────────────────
    # Panel B: Clinical Treatment Onset Collapse
    # ─────────────────────────────────────────────────────────────
    ax2 = axes[1]
    for m in target_models:
        sub = perf_treat[perf_treat["model"] == m].sort_values("horizon_h")
        label = MODEL_NAME_MAP.get(m, m)
        ax2.plot(
            sub["horizon_h"], sub["auprc"] * 100,
            marker=model_markers[m], markersize=7, linewidth=2.4,
            color=model_colors[m], label=label
        )

    ax2.set_title("B. Bedside Treatment Onset ($T_{\\mathrm{treatment}}$)", weight="bold", pad=12)
    ax2.set_xlabel("Actionable Warning Horizon Ahead of Intervention [Hours]")
    ax2.set_ylabel("Actionable Window AUPRC (%)")
    ax2.set_xticks(horizons)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.92)

    # Annotate collapse
    ax2.text(
        0.5, 0.75,
        "Severe Collapse of Actionable Lead Time\n(Clinicians Pre-Empt Retrospective Labels)",
        transform=ax2.transAxes, fontsize=10.5, weight="bold", color="#B45309",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF3C7", edgecolor="#F59E0B", alpha=0.9),
        ha="center"
    )

    # ─────────────────────────────────────────────────────────────
    # Panel C: Delayed Chart Mirroring Rates (Horizontal Barplot)
    # ─────────────────────────────────────────────────────────────
    ax3 = axes[2]
    mirror_plot_df = mirror_df.sort_values("delayed_chart_mirroring_rate", ascending=True).copy()
    display_labels = [MODEL_NAME_MAP.get(m, m) for m in mirror_plot_df["model"]]
    rates = mirror_plot_df["delayed_chart_mirroring_rate"] * 100

    bars = ax3.barh(display_labels, rates, color="#EF4444", alpha=0.75, edgecolor="#991B1B", height=0.6)

    # Annotate bars
    for bar in bars:
        w = bar.get_width()
        ax3.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=10, weight="bold")

    ax3.set_title("C. Post-Treatment Delayed Chart Mirroring", weight="bold", pad=12)
    ax3.set_xlabel("Alarms Issued AFTER Antibiotics Initiated (%)")
    ax3.set_xlim(0, 105)
    ax3.axvline(50, color="#94A3B8", linestyle="--", linewidth=1.2)
    ax3.grid(True, axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()

    png_path = out_dir / "Figure2_Lead_Time_Decay_Profiles.png"
    pdf_path = out_dir / "Figure2_Lead_Time_Decay_Profiles.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    logger.info("Saved Figure 2 to %s and %s", png_path, pdf_path)

    # Copy to Vault
    vault_target = VAULT_ASSETS / "onset_fig2_decay_profile.png"
    fig.savefig(vault_target, dpi=300, bbox_inches="tight")
    logger.info("Copied Figure 2 to Vault: %s", vault_target)

    plt.close(fig)


def main():
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    logger.info("Loading unified evaluation dataset...")
    df_eval = load_harmonized_benchmark_dataset()

    plot_figure_2(df_eval, fig_dir)
    logger.info("Figure 2 generation completed successfully!")


if __name__ == "__main__":
    main()
