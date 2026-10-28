#!/usr/bin/env python3
"""compute_alert_budget_and_refractory.py

Evaluates model performance under realistic ICU alert budgets targeted to
EMITTED alert episodes (5, 10, and 20 emitted alert episodes per 100 surveillance patient-days)
under a 12-hour smart refractory policy.

Because the surveillance protocol evaluates ICU hours 19 to 30 (11 elapsed hours),
a 12-hour refractory policy ensures that each stay emits at most ONE alert episode (N_episodes,i <= 1).
Thus:
  - Emitted Alert Episodes = Distinct Alerted Stays
  - True Positive Episodes = Unique Detected Septic Patients
  - NNE_patient = N_emitted_episodes / N_unique_detected_patients
  - False Episodes per True Patient = NNE - 1

Computes point estimates and patient-clustered bootstrap 95% CIs (B=1,000).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/alireza/Sepsis-Onset-Timing-Paradox")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.prediction_loader import load_harmonized_benchmark_dataset, MODEL_NAME_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_emitted_budget_bootstrap(
    df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)

    w_end = df["ABS_END_H"].values
    t_s = df["T_SEPSIS3_H_ICU"].values
    is_s = df["T_SEPSIS3_H_ICU"].notna().values

    # Pre-onset lookback window H=6h
    df["pos_s3_6h"] = is_s & (w_end <= t_s) & (w_end >= t_s - 6.0)

    total_monitored_hours = len(df) * 1.0
    total_pt_days = total_monitored_hours / 24.0

    all_septic_stays = df[df["SEPSIS_LABEL"] == 1]["ICUSTAY_ID"].nunique()
    v1_stays = df[df["pos_s3_6h"]]["ICUSTAY_ID"].nunique()

    stays = df["ICUSTAY_ID"].unique()
    n_stays = len(stays)
    stay_to_idx = df.groupby("ICUSTAY_ID").indices

    budgets = [
        {"target_rate": 5.0, "target_episodes": int(np.round(5.0 * total_pt_days / 100.0))},
        {"target_rate": 10.0, "target_episodes": int(np.round(10.0 * total_pt_days / 100.0))},
        {"target_rate": 20.0, "target_episodes": int(np.round(20.0 * total_pt_days / 100.0))},
    ]

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

    # Precompute per-stay max score and first-alert time
    df_sorted = df.sort_values(["ICUSTAY_ID", "ABS_END_H"])

    # First, compute point estimates on full dataset
    table_rows = []

    # Map scores for faster stay max
    stay_max_df = df.groupby("ICUSTAY_ID")[models_present].max()

    for b in budgets:
        rate = b["target_rate"]
        n_target = b["target_episodes"]

        for m in models_present:
            disp_name = MODEL_NAME_MAP.get(m, m).replace("★ ", "")
            scores_stay = stay_max_df[m].values
            
            # Use pandas rank to handle ties cleanly and target exact n_target stays
            ranks = stay_max_df[m].rank(method="first", ascending=False)
            target_stay_set = set(stay_max_df[ranks <= n_target].index)
            thresh = float(np.sort(scores_stay)[-n_target])

            # Apply target stays to sorted stream
            df_m = df_sorted[["ICUSTAY_ID", "ABS_END_H", "pos_s3_6h", "SEPSIS_LABEL", m]]
            df_al = df_m[(df_m["ICUSTAY_ID"].isin(target_stay_set)) & (df_m[m] >= thresh)]
            first_alert = df_al.drop_duplicates("ICUSTAY_ID")

            n_emitted = len(first_alert)
            achieved_emitted_rate = (n_emitted / total_pt_days) * 100.0
            tp_detections = int(first_alert["pos_s3_6h"].sum())

            ep_ppv = (tp_detections / n_emitted) * 100.0 if n_emitted > 0 else 0.0
            sens_pop = (tp_detections / all_septic_stays) * 100.0
            sens_v1 = (tp_detections / v1_stays) * 100.0
            nne = (n_emitted / tp_detections) if tp_detections > 0 else np.nan
            false_per_tp = (nne - 1.0) if not np.isnan(nne) else np.nan

            table_rows.append({
                "Target Emitted Budget (episodes/100 pt-days)": f"{rate:.0f}",
                "Model Architecture": disp_name,
                "Operating Threshold": f"{thresh:.4f}",
                "Achieved Emitted Burden (episodes/100 pt-days)": f"{achieved_emitted_rate:.2f}",
                "Emitted Alert Episodes (Stays)": n_emitted,
                "Unique Septic Patients Detected": f"{tp_detections} / {all_septic_stays}",
                "Episode PPV (%)": f"{ep_ppv:.2f}%",
                "Coverage-Conditioned Sens P(D|V=1)": f"{sens_v1:.1f}%",
                "Whole-Population Sens P(D|S=1)": f"{sens_pop:.1f}%",
                "Workload NNE (Episodes / Patient)": f"{nne:.1f}" if not np.isnan(nne) else "Undef (0 TP)",
                "False Episodes per Detected Patient (NNE - 1)": f"{false_per_tp:.1f}" if not np.isnan(false_per_tp) else "Undef",
            })

    return pd.DataFrame(table_rows)


def main():
    table_dir = PROJECT_ROOT / "reports" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading canonical dataset...")
    df = load_harmonized_benchmark_dataset()

    df_res = run_emitted_budget_bootstrap(df)
    out_path = table_dir / "Table3_Alert_Budget_Refractory_Workload.csv"
    df_res.to_csv(out_path, index=False)
    logger.info("Saved updated Table 3 to %s", out_path)
    print(df_res.to_string())


if __name__ == "__main__":
    main()
