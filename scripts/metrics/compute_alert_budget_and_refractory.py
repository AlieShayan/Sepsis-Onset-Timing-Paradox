#!/usr/bin/env python3
"""compute_alert_budget_and_refractory.py

Evaluates model performance under realistic ICU alert budgets (5, 10, 20 alerts / 100 pt-days)
and simulates a 12-hour smart refractory policy (alert suppression) to mitigate alert fatigue.

Calculates:
  - Achieved Alert Burden (alerts / 100 eligible surveillance patient-days)
  - Window-level Sepsis-3 PPV (%)
  - Refractory Alert Episodes
  - Unique Sepsis Episodes Detected (Patient-level Sensitivity)
  - Workload per unique detected patient (NNE_patient = Episodes / Detected Patients)
  - False alarm episodes per true patient detection (NNE - 1)
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


def simulate_refractory_policy(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    refractory_hours: float = 12.0,
) -> tuple[int, int, int]:
    """Simulates alert refractory suppression on sorted ICU stays."""
    df_sorted = df[["ICUSTAY_ID", "ABS_END_H", "pos_s3_6h", score_col]].sort_values(
        ["ICUSTAY_ID", "ABS_END_H"]
    )
    raw_alerts = df_sorted[score_col].values >= threshold

    refractory_alerts = []
    last_alert_time = {}

    for stay, t, is_al in zip(
        df_sorted["ICUSTAY_ID"].values, df_sorted["ABS_END_H"].values, raw_alerts
    ):
        if is_al:
            if stay not in last_alert_time or (t - last_alert_time[stay] >= refractory_hours):
                refractory_alerts.append(True)
                last_alert_time[stay] = t
            else:
                refractory_alerts.append(False)
        else:
            refractory_alerts.append(False)

    df_sorted["refractory_alert"] = refractory_alerts

    n_episodes = int(df_sorted["refractory_alert"].sum())
    # True positive episodes (alerted during pre-onset window)
    tp_mask = df_sorted["refractory_alert"] & df_sorted["pos_s3_6h"]
    n_tp_episodes = int(tp_mask.sum())
    unique_detected_stays = df_sorted[tp_mask]["ICUSTAY_ID"].nunique()

    return n_episodes, n_tp_episodes, unique_detected_stays


def evaluate_operational_frontiers(df: pd.DataFrame) -> pd.DataFrame:
    w_end = df["ABS_END_H"].values
    t_s = df["T_SEPSIS3_H_ICU"].values
    is_s = df["T_SEPSIS3_H_ICU"].notna().values

    # Pre-onset lookback window H=6h
    df["pos_s3_6h"] = is_s & (w_end <= t_s) & (w_end >= t_s - 6.0)

    total_monitored_hours = len(df) * 1.0
    total_pt_days = total_monitored_hours / 24.0

    all_septic_stays = df[df["SEPSIS_LABEL"] == 1]["ICUSTAY_ID"].nunique()
    v1_stays = df[df["pos_s3_6h"]]["ICUSTAY_ID"].nunique()

    logger.info("Monitored hours: %.1f (%.2f patient-days)", total_monitored_hours, total_pt_days)
    logger.info("Total Septic Stays: %d | Surveillance-Covered Stays (V=1): %d", all_septic_stays, v1_stays)

    budgets = [
        {"target": "5 alerts / 100 pt-days", "n_target": int(np.round(5.0 * total_pt_days / 100.0))},
        {"target": "10 alerts / 100 pt-days", "n_target": int(np.round(10.0 * total_pt_days / 100.0))},
        {"target": "20 alerts / 100 pt-days", "n_target": int(np.round(20.0 * total_pt_days / 100.0))},
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

    rows = []
    for b in budgets:
        target_name = b["target"]
        n_target = b["n_target"]

        for m in models_present:
            disp_name = MODEL_NAME_MAP.get(m, m)
            scores = df[m].values
            thresh = float(np.sort(scores)[-n_target])

            raw_alert_mask = scores >= thresh
            n_raw_alerts = int(raw_alert_mask.sum())
            achieved_burden = (n_raw_alerts / total_pt_days) * 100.0

            win_tp = int((raw_alert_mask & df["pos_s3_6h"].values).sum())
            win_ppv = (win_tp / n_raw_alerts) * 100.0 if n_raw_alerts > 0 else 0.0

            # 12h refractory simulation
            n_episodes, n_tp_episodes, unique_stays = simulate_refractory_policy(
                df, m, thresh, refractory_hours=12.0
            )

            sens_pop = (unique_stays / all_septic_stays) * 100.0
            sens_v1 = (unique_stays / v1_stays) * 100.0

            nne_patient = (n_episodes / unique_stays) if unique_stays > 0 else np.nan
            false_alarms_per_detection = (nne_patient - 1.0) if not np.isnan(nne_patient) else np.nan

            rows.append({
                "Target Budget": target_name,
                "Model": disp_name,
                "Operating Threshold": f"{thresh:.4f}",
                "Achieved Burden (alerts/100 pt-days)": f"{achieved_burden:.2f}",
                "Raw Alert Windows": n_raw_alerts,
                "Window Sepsis-3 PPV (%)": f"{win_ppv:.2f}%",
                "12h Refractory Episodes": n_episodes,
                "Unique Patients Detected": f"{unique_stays} / {all_septic_stays}",
                "Coverage-Conditioned Sens P(D|V=1)": f"{sens_v1:.1f}%",
                "Population Sens P(D|S=1)": f"{sens_pop:.1f}%",
                "Workload NNE (Episodes / Patient)": f"{nne_patient:.1f}" if not np.isnan(nne_patient) else "Undef (0 TP)",
                "False Episodes per True Patient (NNE - 1)": f"{false_alarms_per_detection:.1f}" if not np.isnan(false_alarms_per_detection) else "Undef",
            })

    return pd.DataFrame(rows)


def main():
    table_dir = PROJECT_ROOT / "reports" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading canonical dataset...")
    df = load_harmonized_benchmark_dataset()

    df_res = evaluate_operational_frontiers(df)
    out_path = table_dir / "Table3_Alert_Budget_Refractory_Workload.csv"
    df_res.to_csv(out_path, index=False)
    logger.info("Saved Table 3 to %s", out_path)


if __name__ == "__main__":
    main()
