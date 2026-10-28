#!/usr/bin/env python3
"""compute_pretreatment_sepsis_benchmark.py

Computes patient-clustered bootstrap confidence intervals (B=1000) comparing:
  1. Canonical Sepsis-3 (H=6h) on full surveillance risk set (N=97,700 windows, 678 positives)
  2. True Pretreatment Sepsis Detection (delta=0h) on risk set R_0 excluding post-treatment windows (N=70,192 windows, 674 positives)
  3. Pretreatment Sepsis Detection with 2h buffer (delta=2h) on risk set R_2 (N=70,129 windows, 663 positives)

Also computes paired differences (Delta AUROC = R_0 - Canonical) with 95% CIs.
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
) -> pd.DataFrame:
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
    has_tr = df["T_TREATMENT_H_ICU"].notna().values

    # Pre-allocate bootstrap arrays: [n_boot, n_models, 3 tasks]
    auroc_boot = np.zeros((n_boot, len(models_present), 3))
    auprc_boot = np.zeros((n_boot, len(models_present), 3))
    delta_auroc_boot = np.zeros((n_boot, len(models_present)))

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
        hastr_b = has_tr[idx_boot]

        # 1. Canonical Sepsis-3 (H=6h) on full stream
        pos_s3 = iss_b & (w_b <= ts_b) & (w_b >= ts_b - 6.0)
        valid_s3 = (~iss_b) | (iss_b & (w_b <= ts_b))
        y_s3 = np.zeros(len(idx_boot), dtype=int)
        y_s3[pos_s3] = 1

        # 2. Pretreatment Risk Set R_0 (w_b <= tr_b)
        r0_mask = (~hastr_b) | (hastr_b & (w_b <= tr_b))
        y_r0 = y_s3[r0_mask]

        # 3. Pretreatment Risk Set R_2 (w_b + 2h <= tr_b)
        r2_mask = (~hastr_b) | (hastr_b & (w_b <= tr_b - 2.0))
        y_r2 = y_s3[r2_mask]

        # Evaluate models
        for m_idx, m in enumerate(models_present):
            preds = df[m].values[idx_boot]

            # Task 0 (Canonical)
            sub_y0 = y_s3[valid_s3]
            sub_p0 = preds[valid_s3]
            roc0 = roc_auc_score(sub_y0, sub_p0)
            prc0 = average_precision_score(sub_y0, sub_p0)
            auroc_boot[b, m_idx, 0] = roc0
            auprc_boot[b, m_idx, 0] = prc0

            # Task 1 (Pretreatment Risk Set R_0)
            sub_p1 = preds[r0_mask]
            roc1 = roc_auc_score(y_r0, sub_p1)
            prc1 = average_precision_score(y_r0, sub_p1)
            auroc_boot[b, m_idx, 1] = roc1
            auprc_boot[b, m_idx, 1] = prc1

            # Task 2 (Pretreatment Risk Set R_2)
            sub_p2 = preds[r2_mask]
            roc2 = roc_auc_score(y_r2, sub_p2)
            prc2 = average_precision_score(y_r2, sub_p2)
            auroc_boot[b, m_idx, 2] = roc2
            auprc_boot[b, m_idx, 2] = prc2

            # Paired difference Delta AUROC = R_0 - Canonical
            delta_auroc_boot[b, m_idx] = roc1 - roc0

    # 1. Point estimates on original held-out test dataset
    valid_s3_pt = (~is_s) | (is_s & (w_end <= t_s))
    pos_s3_pt = is_s & (w_end <= t_s) & (w_end >= t_s - 6.0)
    y_s3_pt = np.zeros(len(df), dtype=int)
    y_s3_pt[pos_s3_pt] = 1

    r0_mask_pt = (~has_tr) | (has_tr & (w_end <= t_tr))
    y_r0_pt = y_s3_pt[r0_mask_pt]

    r2_mask_pt = (~has_tr) | (has_tr & (w_end <= t_tr - 2.0))
    y_r2_pt = y_s3_pt[r2_mask_pt]

    pt_estimates = {}
    for m_idx, m in enumerate(models_present):
        preds_pt = df[m].values
        # Task 0
        roc0_pt = roc_auc_score(y_s3_pt[valid_s3_pt], preds_pt[valid_s3_pt])
        prc0_pt = average_precision_score(y_s3_pt[valid_s3_pt], preds_pt[valid_s3_pt]) * 100.0
        # Task 1
        roc1_pt = roc_auc_score(y_r0_pt, preds_pt[r0_mask_pt])
        prc1_pt = average_precision_score(y_r0_pt, preds_pt[r0_mask_pt]) * 100.0
        # Task 2
        roc2_pt = roc_auc_score(y_r2_pt, preds_pt[r2_mask_pt])
        prc2_pt = average_precision_score(y_r2_pt, preds_pt[r2_mask_pt]) * 100.0

        delta_pt = roc1_pt - roc0_pt
        pt_estimates[m] = {
            0: (roc0_pt, prc0_pt),
            1: (roc1_pt, prc1_pt),
            2: (roc2_pt, prc2_pt),
            "delta": delta_pt,
        }

    # 2. Compile Table 2 with original held-out point estimates and bootstrap percentile CIs
    table2_rows = []
    task_names = [
        "Canonical Sepsis-3 (Full Risk Set)",
        "Pretreatment Sepsis (Risk Set R0, delta=0h)",
        "Pretreatment Sepsis (Risk Set R2, delta=2h)",
    ]

    for m_idx, m in enumerate(models_present):
        disp_name = MODEL_NAME_MAP.get(m, m).replace("★ ", "")
        if disp_name == "Proposed Champion (Ensemble)":
            disp_name = "Multi-Model Stacking Ensemble"
        row = {"Model Architecture": disp_name}
        for t_idx, t_name in enumerate(task_names):
            rocs = auroc_boot[:, m_idx, t_idx]
            prcs = auprc_boot[:, m_idx, t_idx] * 100.0

            r_pt, p_pt = pt_estimates[m][t_idx]
            r_lo, r_hi = np.percentile(rocs, [2.5, 97.5])
            p_lo, p_hi = np.percentile(prcs, [2.5, 97.5])

            row[f"{t_name} AUROC [95% CI]"] = f"{r_pt:.4f} ({r_lo:.4f}–{r_hi:.4f})"
            row[f"{t_name} AUPRC % [95% CI]"] = f"{p_pt:.3f}% ({p_lo:.3f}%–{p_hi:.3f}%)"

        # Paired difference
        deltas = delta_auroc_boot[:, m_idx]
        d_pt = pt_estimates[m]["delta"]
        d_lo, d_hi = np.percentile(deltas, [2.5, 97.5])
        row["Paired Delta AUROC (R0 - Canonical) [95% CI]"] = f"{d_pt:+.4f} ({d_lo:+.4f} to {d_hi:+.4f})"

        table2_rows.append(row)

    return pd.DataFrame(table2_rows)


def main():
    table_dir = PROJECT_ROOT / "reports" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading canonical dataset...")
    df = load_harmonized_benchmark_dataset()

    df_t2 = run_pretreatment_sepsis_bootstrap(df, n_boot=1000, seed=42)
    t2_path = table_dir / "Table2_Pretreatment_Sepsis_Benchmark_CI.csv"
    df_t2.to_csv(t2_path, index=False)
    logger.info("Saved Table 2 to %s", t2_path)
    print(df_t2.to_string())


if __name__ == "__main__":
    main()
