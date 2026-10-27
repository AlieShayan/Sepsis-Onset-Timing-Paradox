#!/usr/bin/env python3
"""generate_onset_ambiguity_distribution.py

Generates Figure 1 for the Onset Timing Paradox paper:
"Figure 1: Distribution of Temporal Gaps Between Retrospective Sepsis-3 Consensus,
Clinical Treatment Onset, and Organ Dysfunction."

Displays:
  - Panel A: Distribution (KDE + Histogram) of delta(T_sepsis3 - T_treatment)
    for MIMIC-III (MetaVision) and MIMIC-IV (Epic Modern).
  - Panel B: Cumulative Distribution Function (CDF) showing proportion of patients
    with clinical antimicrobial intervention initiated hours before Sepsis-3 onset.
"""
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path("/home/alireza/Sepsis-Onset-Timing-Paradox")
VAULT_ASSETS = Path("/home/alireza/Personalized-Sepsis-Dynamics-Vault/5-assets/figures")


def plot_figure_1(
    df_iii: pd.DataFrame,
    df_iv: pd.DataFrame,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    VAULT_ASSETS.mkdir(parents=True, exist_ok=True)

    # Filter MetaVision for clean electronic monitoring comparison
    if "DBSOURCE" in df_iii.columns:
        df_mv = df_iii[df_iii["DBSOURCE"] == "metavision"].copy()
    else:
        df_mv = df_iii.copy()

    # Style configuration
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
        "axes.edgecolor": "#334155",
        "axes.linewidth": 1.1,
    })

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Colors
    c_iii = "#2563EB"  # Deep Royal Blue (MetaVision)
    c_iv = "#0D9488"   # Teal (MIMIC-IV)
    c_line = "#DC2626" # Crimson alert line

    # Focus on clinical window: [-24h, +48h]
    clip_min, clip_max = -24.0, 48.0
    mv_gaps = df_mv["delta_treatment_gap_h"].clip(clip_min, clip_max).dropna()
    iv_gaps = df_iv["delta_treatment_gap_h"].clip(clip_min, clip_max).dropna()

    mv_med = df_mv["delta_treatment_gap_h"].median()
    iv_med = df_iv["delta_treatment_gap_h"].median()

    # ─────────────────────────────────────────────────────────────
    # Panel A: Density and Histogram of Temporal Gap
    # ─────────────────────────────────────────────────────────────
    ax1 = axes[0]
    bins = np.linspace(clip_min, clip_max, 45)

    sns.histplot(
        mv_gaps, bins=bins, kde=True, color=c_iii, alpha=0.35,
        line_kws={"linewidth": 2.2}, label=f"MIMIC-III MetaVision (Median: +{mv_med:.1f}h)",
        ax=ax1, stat="density"
    )
    sns.histplot(
        iv_gaps, bins=bins, kde=True, color=c_iv, alpha=0.35,
        line_kws={"linewidth": 2.2}, label=f"MIMIC-IV Modern Epic (Median: +{iv_med:.1f}h)",
        ax=ax1, stat="density"
    )

    ax1.axvline(0, color=c_line, linestyle="--", linewidth=1.8, label="Simultaneous Onset (Δt = 0)")
    ax1.axvspan(0, 24, color="#FEF3C7", alpha=0.4, label="Delayed Chart Mirroring Zone (5–24h)")

    ax1.set_title("A. Distribution of Lead Gap: $T_{\\mathrm{sepsis3}} - T_{\\mathrm{treatment}}$", weight="bold", pad=12)
    ax1.set_xlabel("Hours Clinician Began IV Antibiotics Prior to Sepsis-3 [Hours]")
    ax1.set_ylabel("Probability Density")
    ax1.set_xlim(-24, 48)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.92)

    # Annotate Delayed Chart Mirroring
    ax1.annotate(
        "Bedside Clinician\nTreated FIRST\n(Pre-Sepsis3 Intervention)",
        xy=(8, ax1.get_ylim()[1] * 0.45),
        xytext=(16, ax1.get_ylim()[1] * 0.60),
        arrowprops=dict(facecolor="#B45309", arrowstyle="->", lw=1.5),
        fontsize=9.5, weight="bold", color="#B45309", ha="center"
    )

    # ─────────────────────────────────────────────────────────────
    # Panel B: Cumulative Percentage Treated Ahead of Sepsis-3
    # ─────────────────────────────────────────────────────────────
    ax2 = axes[1]
    eval_hours = np.linspace(-24, 48, 200)
    mv_cdf = np.array([(mv_gaps >= h).mean() * 100 for h in eval_hours])
    iv_cdf = np.array([(iv_gaps >= h).mean() * 100 for h in eval_hours])

    ax2.plot(eval_hours, mv_cdf, color=c_iii, linewidth=2.8, label="MIMIC-III MetaVision")
    ax2.plot(eval_hours, iv_cdf, color=c_iv, linewidth=2.8, label="MIMIC-IV Modern Epic")

    # Specific thresholds
    pct_mv_0 = (df_mv["delta_treatment_gap_h"] >= 0).mean() * 100
    pct_iv_0 = (df_iv["delta_treatment_gap_h"] >= 0).mean() * 100

    pct_mv_6 = (df_mv["delta_treatment_gap_h"] >= 6).mean() * 100
    pct_iv_6 = (df_iv["delta_treatment_gap_h"] >= 6).mean() * 100

    ax2.axvline(0, color=c_line, linestyle="--", linewidth=1.5)
    ax2.axvline(6, color="#D97706", linestyle=":", linewidth=1.5, label="Standard 6h ML Warning Horizon")

    ax2.scatter([0, 0], [pct_mv_0, pct_iv_0], color=[c_iii, c_iv], s=70, zorder=5)
    ax2.scatter([6, 6], [pct_mv_6, pct_iv_6], color=[c_iii, c_iv], s=70, zorder=5)

    ax2.annotate(f"{pct_mv_0:.1f}%", xy=(0, pct_mv_0), xytext=(-8, pct_mv_0 + 3), fontsize=10, weight="bold", color=c_iii)
    ax2.annotate(f"{pct_iv_0:.1f}%", xy=(0, pct_iv_0), xytext=(-8, pct_iv_0 - 5), fontsize=10, weight="bold", color=c_iv)

    ax2.annotate(f"{pct_mv_6:.1f}%", xy=(6, pct_mv_6), xytext=(9, pct_mv_6 + 2), fontsize=10, weight="bold", color=c_iii)
    ax2.annotate(f"{pct_iv_6:.1f}%", xy=(6, pct_iv_6), xytext=(9, pct_iv_6 - 5), fontsize=10, weight="bold", color=c_iv)

    ax2.set_title("B. Proportion of Patients Treated Prior to Sepsis-3", weight="bold", pad=12)
    ax2.set_xlabel("Anticipation Threshold Ahead of Sepsis-3 [Hours]")
    ax2.set_ylabel("Cumulative Percentage of Sepsis Patients (%)")
    ax2.set_xlim(-24, 48)
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.92)

    plt.tight_layout()

    # Save PNG and PDF
    png_path = out_dir / "Figure1_Onset_Timing_Distribution.png"
    pdf_path = out_dir / "Figure1_Onset_Timing_Distribution.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    logger.info("Saved Figure 1 to %s and %s", png_path, pdf_path)

    # Copy to Vault
    vault_target = VAULT_ASSETS / "onset_fig1_temporal_gap.png"
    fig.savefig(vault_target, dpi=300, bbox_inches="tight")
    logger.info("Copied Figure 1 to Vault: %s", vault_target)

    plt.close(fig)


def main():
    tables_dir = REPO_ROOT / "reports" / "tables"
    fig_dir = REPO_ROOT / "reports" / "figures"

    p_iii = tables_dir / "tri_onset_labels_mimiciii.csv"
    p_iv = tables_dir / "tri_onset_labels_mimiciv.csv"

    if not p_iii.exists() or not p_iv.exists():
        logger.error("Missing label files in %s. Run extract_tri_onset_labels.py first.", tables_dir)
        return

    logger.info("Loading labels for Figure 1...")
    df_iii = pd.read_csv(p_iii)
    df_iv = pd.read_csv(p_iv)

    plot_figure_1(df_iii, df_iv, fig_dir)
    logger.info("Figure 1 generation completed successfully!")


if __name__ == "__main__":
    main()
