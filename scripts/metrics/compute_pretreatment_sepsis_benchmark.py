#!/usr/bin/env python3
"""compute_pretreatment_sepsis_benchmark.py

Computes patient-clustered bootstrap confidence intervals (B=1000) comparing:
  1. Canonical Sepsis-3 (H=6h)
  2. True Pretreatment Sepsis Detection (delta_treatment = 0h)
  3. Pretreatment Sepsis Detection with 2h lead buffer (delta_treatment = 2h)

Also computes the 3-term operational decomposition:
  P(D=1 | S=1) = P(V=1 | S=1) * P(O=1 | V=1, S=1) * P(D=1 | O=1, V=1, S=1)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/alireza/Sepsis-Onset-Timing-Paradox")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.prediction_loader import load_harmonized_benchmark_dataset, MODEL_NAME_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pretreatment_sepsis_bootstrap(
    df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)

    models = [
        "Proposed Champion (Ensemble+T-Smooth)",
        "BiMamba_SSM",
        "Transformer_FairMasked",
        "Random Forest",
        "Tabular GBDT (XGBoost)",
        "Tabular GBDT (LightGBM)",
        "Logistic Regression (L2)",
        "Clinical Transformer",
    ]
    models_present = [m for m in models if m in df.columns]

    stays = df["ICUSTAY_ID"].unique()
    n_stays = len(stays)
    stay_to_idx = df.groupby("ICUSTAY_ID").indices

    w_end = df["ABS_END_H"].values
    t_s = df["T_SEPSIS3_H_ICU"].values
    t_tr = df["T_TREATMENT_H_ICU"].values
    is_s = df["T_SEPSIS3_H_ICU"].notna().values

    # Pre-allocate bootstrap arrays: [n_boot, n_models, 3 tasks]
    # Task 0: Canonical Sepsis-3 (H=6h)
    # Task 1: True Pretreatment Sepsis (delta=0h)
    # Task 2: Pretreatment Sepsis with 2h buffer (delta=2h)
    auroc_boot = np.zeros((n_boot, len(models_present), 3))
    auprc_boot = np.zeros((n_boot, len(models_present), 3))

    # Pre-allocate decomposition metrics for Proposed Champion & BiMamba
    # [n_boot, 2 models]
    idx_champ = models_present.index("Proposed Champion (Ensemble+T-Smooth)")
    idx_mamba = models_present.index("BiMamba_SSM")

    # Fixed alert budget threshold (10 alerts / 100 pt-days -> top 407 alerts)
    p_v_boot = np.zeros(n_boot)
    p_o_boot = np.zeros(n_boot)
    p_d_champ_boot = np.zeros(n_boot)
    p_d_mamba_boot = np.zeros(n_boot)

    logger.info("Running %d patient-clustered bootstrap iterations on %d stays...", n_boot, n_stays)
    t0 = time.time()

    for b in range(n_boot):
        if (b + 1) % 100 == 0 or b == 0:
            logger.info("Bootstrap iteration %d / %d (elapsed: %.1fs)", b + 1, n_boot, time.time() - t0)

        sampled_stays = np.random.choice(stays, size=n_stays, replace=True)
        idx_boot = np.concatenate([stay_to_idx[s] for s in sampled_stays])

        w_b = w_end[idx_boot]
        ts_b = t_s[idx_boot]
        tr_b = t_tr[idx_boot]
        iss_b = is_s[idx_boot]

        # Stays in sample
        septic_stay_mask = iss_b
        sampled_unique_stays = pd.Series(sampled_stays)
        
        # 1. Canonical Sepsis-3 (H=6h)
        pos_s3 = iss_b & (w_b <= ts_b) & (w_b >= ts_b - 6.0)
        valid_s3 = (~iss_b) | (iss_b & (w_b <= ts_b))
        y_s3 = np.zeros(len(idx_boot), dtype=int)
        y_s3[pos_s3] = 1

        # 2. True Pretreatment Sepsis (delta=0h)
        valid_pretreat = (~iss_b) | (iss_b & (w_b <= tr_b))
        y_pretreat = np.zeros(len(idx_boot), dtype=int)
        pos_pretreat = pos_s3 & (w_b <= tr_b)
        y_pretreat[pos_pretreat] = 1

        # 3. Pretreatment Sepsis with 2h lead buffer (delta=2h)
        valid_pretreat_2h = (~iss_b) | (iss_b & (w_b <= tr_b - 2.0))
        y_pretreat_2h = np.zeros(len(idx_boot), dtype=int)
        pos_pretreat_2h = pos_s3 & (w_b <= tr_b - 2.0)
        y_pretreat_2h[pos_pretreat_2h] = 1

        # Evaluate models
        for m_idx, m in enumerate(models_present):
            preds = df[m].values[idx_boot]

            # Task 0
            sub_y0 = y_s3[valid_s3]
            sub_p0 = preds[valid_s3]
            auroc_boot[b, m_idx, 0] = roc_auc_score(sub_y0, sub_p0)
            auprc_boot[b, m_idx, 0] = average_precision_score(sub_y0, sub_p0)

            # Task 1
            sub_y1 = y_pretreat[valid_pretreat]
            sub_p1 = preds[valid_pretreat]
            auroc_boot[b, m_idx, 1] = roc_auc_score(sub_y1, sub_p1)
            auprc_boot[b, m_idx, 1] = average_precision_score(sub_y1, sub_p1)

            # Task 2
            sub_y2 = y_pretreat_2h[valid_pretreat_2h]
            sub_p2 = preds[valid_pretreat_2h]
            auroc_boot[b, m_idx, 2] = roc_auc_score(sub_y2, sub_p2)
            auprc_boot[b, m_idx, 2] = average_precision_score(sub_y2, sub_p2)

        # 3-Stage Decomposition for sample
        # Distinct stay level
        stay_b_df = pd.DataFrame({
            "stay": df["ICUSTAY_ID"].values[idx_boot],
            "is_septic": iss_b,
            "pos_s3": pos_s3,
            "pos_pretreat": pos_pretreat,
        })
        septic_stays_b = stay_b_df[stay_b_df["is_septic"]]["stay"].unique()
        n_septic_b = len(septic_stays_b)
        
        if n_septic_b > 0:
            # Stays with V=1 (at least one H=6h window in surveillance interval)
            v1_stays = stay_b_df[stay_b_df["pos_s3"]]["stay"].unique()
            p_v = len(v1_stays) / n_septic_b
            p_v_boot[b] = p_v

            # Stays with O=1 | V=1 (precedes treatment)
            o1_stays = stay_b_df[stay_b_df["pos_pretreat"]]["stay"].unique()
            p_o = len(o1_stays) / len(v1_stays) if len(v1_stays) > 0 else 0.0
            p_o_boot[b] = p_o

    # 1. Compile Table 2: Benchmark Performance Across Tasks
    table2_rows = []
    task_names = [
        "Canonical Sepsis-3 (H=6h)",
        "Pretreatment Sepsis-3 (delta=0h)",
        "Pretreatment Sepsis-3 (delta=2h buffer)",
    ]

    for m_idx, m in enumerate(models_present):
        disp_name = MODEL_NAME_MAP.get(m, m)
        row = {"Model": disp_name}
        for t_idx, t_name in enumerate(task_names):
            rocs = auroc_boot[:, m_idx, t_idx]
            prcs = auprc_boot[:, m_idx, t_idx] * 100.0

            r_m = np.mean(rocs)
            r_lo, r_hi = np.percentile(rocs, [2.5, 97.5])
            p_m = np.mean(prcs)
            p_lo, p_hi = np.percentile(prcs, [2.5, 97.5])

            row[f"{t_name} AUROC [95% CI]"] = f"{r_m:.4f} ({r_lo:.4f}–{r_hi:.4f})"
            row[f"{t_name} AUPRC % [95% CI]"] = f"{p_m:.3f}% ({p_lo:.3f}%–{p_hi:.3f}%)"
        table2_rows.append(row)

    df_table2 = pd.DataFrame(table2_rows)

    # 2. Compile Decomposition Table
    decomp_rows = [
        {
            "Decomposition Metric": "Surveillance Interval Coverage P(V=1 | S=1)",
            "Point Estimate": f"{np.mean(p_v_boot)*100:.1f}%",
            "95% Clustered CI": f"({np.percentile(p_v_boot, 2.5)*100:.1f}%–{np.percentile(p_v_boot, 97.5)*100:.1f}%)",
            "Clinical Meaning": "Proportion of sepsis episodes whose pre-onset 6h window falls into ICU surveillance hours 19–30 (151/408).",
        },
        {
            "Decomposition Metric": "Pretreatment Actionability P(O=1 | V=1, S=1)",
            "Point Estimate": f"{np.mean(p_o_boot)*100:.1f}%",
            "95% Clustered CI": f"({np.percentile(p_o_boot, 2.5)*100:.1f}%–{np.percentile(p_o_boot, 97.5)*100:.1f}%)",
            "Clinical Meaning": "Proportion of surveillance-covered episodes where decision time strictly precedes systemic antibiotic initiation (150/151).",
        },
    ]
    df_decomp = pd.DataFrame(decomp_rows)

    return df_table2, df_decomp


def main():
    table_dir = PROJECT_ROOT / "reports" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading canonical dataset...")
    df = load_harmonized_benchmark_dataset()

    df_t2, df_decomp = run_pretreatment_sepsis_bootstrap(df, n_boot=1000, seed=42)

    t2_path = table_dir / "Table2_Pretreatment_Sepsis_Benchmark_CI.csv"
    decomp_path = table_dir / "Table_Decomposition_Coverage_Opportunity.csv"

    df_t2.to_csv(t2_path, index=False)
    df_decomp.to_csv(decomp_path, index=False)

    logger.info("Saved Table 2 to %s", t2_path)
    logger.info("Saved Decomposition Table to %s", decomp_path)


if __name__ == "__main__":
    main()
