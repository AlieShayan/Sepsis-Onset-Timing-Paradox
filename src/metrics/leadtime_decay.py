"""src/metrics/leadtime_decay.py

Evaluation utilities for analyzing lead-time decay, operational actionable windows,
delayed chart mirroring, and architectural rank invariance across tri-onset definitions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_horizon_performance(
    df: pd.DataFrame,
    model_cols: list[str],
    onset_col: str,
    horizons: list[float] = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0],
) -> pd.DataFrame:
    """Evaluate AUPRC and AUROC across horizons for a specific onset definition."""
    results = []
    w_end = df["ABS_END_H"].values
    t_on = df[onset_col].values
    is_sep = df[onset_col].notna().values

    for H in horizons:
        # Window-level labeling relative to target onset:
        # Positive: within H hours of target onset
        pos_mask = is_sep & (w_end <= t_on) & (w_end >= t_on - H)
        # Valid evaluation windows: exclude post-onset windows (prevent outcome leakage)
        valid_mask = (~is_sep) | (is_sep & (w_end <= t_on))

        y_true = np.zeros(len(df), dtype=int)
        y_true[pos_mask] = 1

        sub_y = y_true[valid_mask]
        n_pos = sub_y.sum()
        prev = n_pos / len(sub_y) if len(sub_y) > 0 else 0.0

        for m in model_cols:
            if m not in df.columns:
                continue
            sub_p = df[m].values[valid_mask]
            if n_pos > 0 and len(np.unique(sub_y)) > 1:
                auprc = average_precision_score(sub_y, sub_p)
                auroc = roc_auc_score(sub_y, sub_p)
            else:
                auprc = 0.0
                auroc = np.nan

            results.append({
                "onset_definition": onset_col,
                "horizon_h": H,
                "model": m,
                "n_eval_windows": len(sub_y),
                "n_pos_windows": int(n_pos),
                "prevalence": prev,
                "auprc": auprc,
                "auroc": auroc,
            })

    return pd.DataFrame(results)


def compute_delayed_chart_mirroring_rate(
    df: pd.DataFrame,
    model_cols: list[str],
    horizon_h: float = 6.0,
) -> pd.DataFrame:
    """Computes actionable lead-time degradation and chart mirroring penalty across architectures.

    Quantifies the sharp performance collapse when moving from retrospective consensus
    (T_sepsis3) to bedside actionable intervention (T_treatment).
    """
    records = []
    perf_sep3 = evaluate_horizon_performance(df, model_cols, "T_SEPSIS3_H_ICU", [horizon_h])
    perf_treat = evaluate_horizon_performance(df, model_cols, "T_TREATMENT_H_ICU", [horizon_h])

    s_sep3 = perf_sep3[perf_sep3["horizon_h"] == horizon_h].set_index("model")
    s_treat = perf_treat[perf_treat["horizon_h"] == horizon_h].set_index("model")

    for m in model_cols:
        if m not in s_sep3.index or m not in s_treat.index:
            continue
        auprc_sep3 = float(s_sep3.loc[m, "auprc"])
        auprc_treat = float(s_treat.loc[m, "auprc"])
        auroc_sep3 = float(s_sep3.loc[m, "auroc"])
        auroc_treat = float(s_treat.loc[m, "auroc"])

        # Relative retention and degradation penalty
        retention = (auprc_treat / auprc_sep3) if auprc_sep3 > 0 else 0.0
        degradation_penalty = (1.0 - retention) * 100.0

        records.append({
            "model": m,
            "horizon_h": horizon_h,
            "auprc_sepsis3": auprc_sep3,
            "auprc_treatment": auprc_treat,
            "auroc_sepsis3": auroc_sep3,
            "auroc_treatment": auroc_treat,
            "actionable_retention_rate": float(retention * 100.0),
            "delayed_chart_mirroring_rate": float(degradation_penalty),
        })

    return pd.DataFrame(records)



def compute_rank_invariance(
    df_metrics: pd.DataFrame,
    horizon: float = 6.0,
    metric: str = "auprc",
) -> tuple[pd.DataFrame, float, float]:
    """Computes Spearman and Kendall rank correlations between onset definitions."""
    sub = df_metrics[df_metrics["horizon_h"] == horizon].copy()
    pivot = sub.pivot(index="model", columns="onset_definition", values=metric)

    cols = pivot.columns.tolist()
    if len(cols) >= 2:
        c1, c2 = cols[0], cols[1]
        spearman_rho, _ = spearmanr(pivot[c1], pivot[c2])
        kendall_tau, _ = kendalltau(pivot[c1], pivot[c2])
    else:
        spearman_rho, kendall_tau = 1.0, 1.0

    return pivot, spearman_rho, kendall_tau
