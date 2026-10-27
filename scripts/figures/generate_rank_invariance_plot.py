#!/usr/bin/env python3
"""generate_rank_invariance_plot.py

Generates Figure 3 for the Onset Timing Paradox paper:
"Figure 3: Architectural Rank Invariance Across Onset Definitions and Evaluation Protocols."

Displays:
  - Panel A: Bump chart / Slopegraph tracing model performance ranking across evaluation protocols.
  - Panel B: Pairwise Spearman (rho) and Kendall (tau) rank correlation matrix across the 13 architectures.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/alireza/Sepsis-Onset-Timing-Paradox")
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, kendalltau

from src.data.prediction_loader import load_harmonized_benchmark_dataset, MODEL_NAME_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VAULT_ASSETS = Path("/home/alireza/Personalized-Sepsis-Dynamics-Vault/5-assets/figures")


def plot_figure_3(
    df_eval: pd.DataFrame,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    VAULT_ASSETS.mkdir(parents=True, exist_ok=True)

    # 13 Models in standard evaluation
    all_models = [
        "Proposed Champion (Ensemble+T-Smooth)",
        "BiMamba_SSM",
        "SE-DualGated Mamba-LSTM",
        "Hierarchical Mamba-LSTM",
        "Transformer_FairMasked",
        "Clinical Transformer",
        "Multimodal Hybrid (BiLSTM+MLP)",
        "Standard Deep Bi-LSTM",
        "Standard Deep Bi-GRU",
        "Dilated TCN",
        "Tabular GBDT (XGBoost)",
        "Tabular GBDT (LightGBM)",
        "Random Forest",
        "Logistic Regression (L2)",
    ]
    models_present = [m for m in all_models if m in df_eval.columns]

    # Evaluate models under Sepsis-3 (H=6h), Sepsis-3 (H=2h), and Physiological Shock SOFA (H=6h)
    eval_protocols = {
        "Canonical Sepsis-3 (H=6h)": ("T_SEPSIS3_H_ICU", 6.0),
        "Early Window Sepsis-3 (H=2h)": ("T_SEPSIS3_H_ICU", 2.0),
        "Acute Organ Shock SOFA (H=6h)": ("T_SOFA_H_ICU", 6.0),
    }

    rank_data = {}
    metric_data = {}

    w_end = df_eval["ABS_END_H"].values

    for proto_name, (onset_col, H) in eval_protocols.items():
        t_on = df_eval[onset_col].values
        is_sep = df_eval[onset_col].notna().values

        pos_mask = is_sep & (w_end <= t_on) & (w_end >= t_on - H)
        valid_mask = (~is_sep) | (is_sep & (w_end <= t_on))

        y_true = np.zeros(len(df_eval), dtype=int)
        y_true[pos_mask] = 1

        sub_y = y_true[valid_mask]
        auprc_scores = {}
        for m in models_present:
            sub_p = df_eval[m].values[valid_mask]
            from sklearn.metrics import average_precision_score
            auprc_scores[m] = average_precision_score(sub_y, sub_p)

        metric_data[proto_name] = auprc_scores
        # Rank: 1 is best (highest AUPRC)
        s = pd.Series(auprc_scores).rank(ascending=False, method="min")
        rank_data[proto_name] = s

    df_ranks = pd.DataFrame(rank_data)
    df_metrics = pd.DataFrame(metric_data)

    # Save summary table
    summary_table = df_metrics.copy()
    summary_table.index = [MODEL_NAME_MAP.get(m, m) for m in summary_table.index]
    summary_table.to_csv(out_dir.parent / "tables" / "Table1_Onset_Paradox_Benchmark.csv")

    # Compute correlation matrix across protocols
    proto_names = list(eval_protocols.keys())
    spearman_mat = np.zeros((len(proto_names), len(proto_names)))
    kendall_mat = np.zeros((len(proto_names), len(proto_names)))

    for i, p1 in enumerate(proto_names):
        for j, p2 in enumerate(proto_names):
            s_rho, _ = spearmanr(df_metrics[p1], df_metrics[p2])
            k_tau, _ = kendalltau(df_metrics[p1], df_metrics[p2])
            spearman_mat[i, j] = s_rho
            kendall_mat[i, j] = k_tau

    # Style configuration
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.titlesize": 15,
        "axes.edgecolor": "#334155",
        "axes.linewidth": 1.1,
    })

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300, gridspec_kw={"width_ratios": [1.4, 1.0]})

    # ─────────────────────────────────────────────────────────────
    # Panel A: Bump Chart of Architectural Ranks
    # ─────────────────────────────────────────────────────────────
    ax1 = axes[0]
    palette = sns.color_palette("tab20", len(models_present))

    for idx, m in enumerate(models_present):
        y_ranks = df_ranks.loc[m].values
        disp_name = MODEL_NAME_MAP.get(m, m)
        is_highlight = "Champion" in disp_name or "BiMamba" in disp_name or "Fair" in disp_name

        lw = 3.2 if is_highlight else 1.6
        alpha = 1.0 if is_highlight else 0.55
        color = "#D97706" if "Champion" in disp_name else ("#2563EB" if "BiMamba" in disp_name else ("#0D9488" if "Fair" in disp_name else palette[idx]))

        ax1.plot(proto_names, y_ranks, marker="o", markersize=8 if is_highlight else 5,
                 linewidth=lw, alpha=alpha, color=color, label=disp_name)

        # Label at rightmost point
        ax1.text(len(proto_names) - 0.95, y_ranks[-1], f"  {disp_name} (#{int(y_ranks[-1])})",
                 va="center", ha="left", fontsize=9.5, weight="bold" if is_highlight else "normal",
                 color=color)

    ax1.set_title("A. Architectural Rank Invariance Across Protocols", weight="bold", pad=12)
    ax1.set_ylabel("Rank Position (1 = Top Performing Model)")
    ax1.set_ylim(len(models_present) + 0.8, 0.5)
    ax1.set_yticks(range(1, len(models_present) + 1))
    ax1.set_xlim(-0.1, len(proto_names) + 0.5)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    # ─────────────────────────────────────────────────────────────
    # Panel B: Correlation Heatmap
    # ─────────────────────────────────────────────────────────────
    ax2 = axes[1]
    short_proto_names = ["Sepsis-3\n(H=6h)", "Sepsis-3\n(H=2h)", "SOFA Shock\n(H=6h)"]
    df_corr = pd.DataFrame(spearman_mat, index=short_proto_names, columns=short_proto_names)

    sns.heatmap(
        df_corr, annot=True, fmt=".3f", cmap="Blues", vmin=0.85, vmax=1.0,
        cbar_kws={"label": "Spearman Rank Correlation (ρ)"}, ax=ax2,
        annot_kws={"size": 13, "weight": "bold"}, square=True, linewidths=1.2, linecolor="white"
    )

    ax2.set_title("B. Invariance Stability Matrix (Spearman $\\rho$)", weight="bold", pad=12)

    plt.tight_layout()

    png_path = out_dir / "Figure3_Architectural_Rank_Invariance.png"
    pdf_path = out_dir / "Figure3_Architectural_Rank_Invariance.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    logger.info("Saved Figure 3 to %s and %s", png_path, pdf_path)

    # Copy to Vault
    vault_target = VAULT_ASSETS / "onset_fig3_rank_invariance.png"
    fig.savefig(vault_target, dpi=300, bbox_inches="tight")
    logger.info("Copied Figure 3 to Vault: %s", vault_target)

    plt.close(fig)


def main():
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    logger.info("Loading unified evaluation dataset...")
    df_eval = load_harmonized_benchmark_dataset()

    plot_figure_3(df_eval, fig_dir)
    logger.info("Figure 3 generation completed successfully!")


if __name__ == "__main__":
    main()
