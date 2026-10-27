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

    # Models in harmonized benchmark
    all_models = [
        "Proposed Champion (Ensemble+T-Smooth)",
        "BiMamba_SSM",
        "Transformer_FairMasked",
        "Random Forest",
        "Tabular GBDT (XGBoost)",
        "Tabular GBDT (LightGBM)",
        "Logistic Regression (L2)",
        "Clinical Transformer",
    ]
    models_present = [m for m in all_models if m in df_eval.columns]

    # Evaluate models under multi-horizon Sepsis-3 anticipation protocols
    eval_protocols = {
        "Early Horizon (H=2h)": ("T_SEPSIS3_H_ICU", 2.0),
        "Canonical Sepsis-3 (H=6h)": ("T_SEPSIS3_H_ICU", 6.0),
        "Extended Horizon (H=12h)": ("T_SEPSIS3_H_ICU", 12.0),
    }

    from sklearn.metrics import roc_auc_score, average_precision_score

    rank_data = {}
    auroc_data = {}
    auprc_data = {}

    w_end = df_eval["ABS_END_H"].values

    for proto_name, (onset_col, H) in eval_protocols.items():
        t_on = df_eval[onset_col].values
        is_sep = df_eval[onset_col].notna().values

        pos_mask = is_sep & (w_end <= t_on) & (w_end >= t_on - H)
        valid_mask = (~is_sep) | (is_sep & (w_end <= t_on))

        y_true = np.zeros(len(df_eval), dtype=int)
        y_true[pos_mask] = 1

        sub_y = y_true[valid_mask]
        proto_aurocs = {}
        proto_auprcs = {}
        for m in models_present:
            sub_p = df_eval[m].values[valid_mask]
            proto_aurocs[m] = roc_auc_score(sub_y, sub_p)
            proto_auprcs[m] = average_precision_score(sub_y, sub_p)

        auroc_data[proto_name] = proto_aurocs
        auprc_data[proto_name] = proto_auprcs

        # Rank based on AUROC (1 is top performing model)
        s = pd.Series(proto_aurocs).rank(ascending=False, method="min")
        rank_data[proto_name] = s

    df_ranks = pd.DataFrame(rank_data)
    df_aurocs = pd.DataFrame(auroc_data)
    df_auprcs = pd.DataFrame(auprc_data)

    # Save comprehensive summary table
    table_rows = []
    for m in models_present:
        disp_name = MODEL_NAME_MAP.get(m, m)
        table_rows.append({
            "Model": disp_name,
            "Early H=2h (AUROC)": f"{df_aurocs.loc[m, 'Early Horizon (H=2h)']:.4f}",
            "Early H=2h (AUPRC %)": f"{df_auprcs.loc[m, 'Early Horizon (H=2h)'] * 100:.3f}%",
            "Canonical H=6h (AUROC)": f"{df_aurocs.loc[m, 'Canonical Sepsis-3 (H=6h)']:.4f}",
            "Canonical H=6h (AUPRC %)": f"{df_auprcs.loc[m, 'Canonical Sepsis-3 (H=6h)'] * 100:.3f}%",
            "Extended H=12h (AUROC)": f"{df_aurocs.loc[m, 'Extended Horizon (H=12h)']:.4f}",
            "Extended H=12h (AUPRC %)": f"{df_auprcs.loc[m, 'Extended Horizon (H=12h)'] * 100:.3f}%",
            "Rank (H=6h)": int(df_ranks.loc[m, 'Canonical Sepsis-3 (H=6h)']),
        })
    df_summary = pd.DataFrame(table_rows).sort_values("Rank (H=6h)")
    table_path = out_dir.parent / "tables" / "Table1_Onset_Paradox_Benchmark.csv"
    df_summary.to_csv(table_path, index=False)
    logger.info("Saved Table 1 summary to %s", table_path)

    # Compute correlation matrix across protocols using AUROC metrics
    proto_names = list(eval_protocols.keys())
    spearman_mat = np.zeros((len(proto_names), len(proto_names)))
    kendall_mat = np.zeros((len(proto_names), len(proto_names)))

    for i, p1 in enumerate(proto_names):
        for j, p2 in enumerate(proto_names):
            s_rho, _ = spearmanr(df_aurocs[p1], df_aurocs[p2])
            k_tau, _ = kendalltau(df_aurocs[p1], df_aurocs[p2])
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

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300, gridspec_kw={"width_ratios": [1.45, 0.95]})

    # ─────────────────────────────────────────────────────────────
    # Panel A: Bump Chart of Architectural Ranks
    # ─────────────────────────────────────────────────────────────
    ax1 = axes[0]
    
    model_styles = {
        "Proposed Champion (Ensemble+T-Smooth)": {"color": "#D97706", "lw": 3.4, "marker": "o", "ms": 9},
        "BiMamba_SSM": {"color": "#2563EB", "lw": 3.4, "marker": "s", "ms": 9},
        "Random Forest": {"color": "#059669", "lw": 2.4, "marker": "D", "ms": 7},
        "Transformer_FairMasked": {"color": "#0D9488", "lw": 3.0, "marker": "^", "ms": 8},
        "Tabular GBDT (XGBoost)": {"color": "#DC2626", "lw": 2.0, "marker": "P", "ms": 6},
        "Tabular GBDT (LightGBM)": {"color": "#EA580C", "lw": 2.0, "marker": "v", "ms": 6},
        "Logistic Regression (L2)": {"color": "#64748B", "lw": 2.0, "marker": "X", "ms": 6},
        "Clinical Transformer": {"color": "#7C3AED", "lw": 2.2, "marker": "<", "ms": 7},
    }

    proto_labels = ["Early Horizon\n(H=2h)", "Canonical Sepsis-3\n(H=6h)", "Extended Horizon\n(H=12h)"]

    for m in models_present:
        y_ranks = df_ranks.loc[m].values
        disp_name = MODEL_NAME_MAP.get(m, m)
        style = model_styles.get(m, {"color": "#94A3B8", "lw": 1.6, "marker": "o", "ms": 5})
        is_highlight = "Champion" in disp_name or "BiMamba" in disp_name or "Fair" in disp_name

        ax1.plot(proto_labels, y_ranks, marker=style["marker"], markersize=style["ms"],
                 linewidth=style["lw"], alpha=1.0 if is_highlight else 0.8,
                 color=style["color"], label=disp_name)

        # Label at rightmost point
        final_rank = int(y_ranks[-1])
        final_auroc = df_aurocs.loc[m, "Extended Horizon (H=12h)"]
        ax1.text(len(proto_labels) - 0.95, y_ranks[-1], f"  {disp_name} (#{final_rank}, {final_auroc:.3f})",
                 va="center", ha="left", fontsize=9.5, weight="bold" if is_highlight else "normal",
                 color=style["color"])

    ax1.set_title("A. Architectural Rank Invariance Across Prediction Horizons", weight="bold", pad=12)
    ax1.set_ylabel("Rank Position (1 = Highest AUROC)")
    ax1.set_ylim(len(models_present) + 0.6, 0.5)
    ax1.set_yticks(range(1, len(models_present) + 1))
    ax1.set_xlim(-0.15, len(proto_labels) + 0.8)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    # ─────────────────────────────────────────────────────────────
    # Panel B: Correlation Heatmap
    # ─────────────────────────────────────────────────────────────
    ax2 = axes[1]
    df_corr = pd.DataFrame(spearman_mat, index=proto_labels, columns=proto_labels)

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
