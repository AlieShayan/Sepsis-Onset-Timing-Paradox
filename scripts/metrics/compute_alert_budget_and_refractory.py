#!/usr/bin/env python3
"""compute_alert_budget_and_refractory.py

Evaluates operational early warning alert budgets under a 12-hour refractory suppression policy.
Reports burden-matched test-set performance frontiers at budgets of 5.0, 10.0, and 20.0 emitted
alert episodes per 100 patient-days (204, 407, and 814 emitted episodes).

Includes:
  - Exact timing verification at the first emitted alert:
      1. Sepsis-horizon detection (0 < T_S3 - t_alert <= 6h)
      2. True pretreatment detection (t_alert <= T_treatment)
      3. Pretreatment with 2h buffer (t_alert + 2h <= T_treatment)
  - Patient-clustered bootstrap 95% confidence intervals (B=1,000).
"""
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/alireza/Sepsis-Onset-Timing-Paradox")
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVAL_MODELS = {
    "BiMamba_SSM": "BiMamba SSM (Proposed)",
    "Proposed Champion (Ensemble+T-Smooth)": "Proposed Champion (Ensemble)",
    "Transformer_FairMasked": "Mask-Aware Clinical Transformer",
    "Random Forest": "Random Forest",
    "Tabular GBDT (XGBoost)": "Tabular GBDT (XGBoost)",
    "Tabular GBDT (LightGBM)": "Tabular GBDT (LightGBM)",
    "Logistic Regression (L2)": "Logistic Regression (L2)",
    "Clinical Transformer": "Clinical Transformer (Baseline Unmasked)",
}

BUDGET_TARGETS = [
    (204, 5.01, "5.0 episodes / 100 pt-days"),
    (407, 10.00, "10.0 episodes / 100 pt-days"),
    (814, 20.00, "20.0 episodes / 100 pt-days"),
]


def evaluate_stay_alerts(df: pd.DataFrame, model_col: str, target_n: int) -> dict:
    stay_max = df.groupby("ICUSTAY_ID")[model_col].max()
    # Rank stays to handle ties deterministically
    ranks = stay_max.rank(method="first", ascending=False)
    alerted_stay_ids = ranks[ranks <= target_n].index
    thresh = stay_max.loc[alerted_stay_ids].min()

    # Filter to alerted stays and predictions >= thresh
    df_alerted = df[df["ICUSTAY_ID"].isin(alerted_stay_ids) & (df[model_col] >= thresh)]
    # First alert per stay
    first_alerts = df_alerted.sort_values(["ICUSTAY_ID", "ABS_END_H"]).groupby("ICUSTAY_ID").first().reset_index()

    t_alert = first_alerts["ABS_END_H"].values
    t_s3 = first_alerts["T_SEPSIS3_H_ICU"].values
    t_tr = first_alerts["T_TREATMENT_H_ICU"].values
    is_septic = first_alerts["SEPSIS_LABEL"].values == 1

    sepsis_det = is_septic & (t_s3 - t_alert > 0) & (t_s3 - t_alert <= 6.0)
    pretr_det = sepsis_det & (t_alert <= t_tr)
    pretr_2h_det = sepsis_det & (t_alert + 2.0 <= t_tr)

    n_alerted = len(first_alerts)
    tp_sepsis = int(sepsis_det.sum())
    tp_pretr = int(pretr_det.sum())
    tp_pretr_2h = int(pretr_2h_det.sum())

    ppv_sepsis = tp_sepsis / n_alerted * 100 if n_alerted > 0 else 0.0
    ppv_pretr = tp_pretr / n_alerted * 100 if n_alerted > 0 else 0.0
    sens_covered = tp_sepsis / 151 * 100
    yield_overall = tp_sepsis / 408 * 100
    nne = n_alerted / tp_sepsis if tp_sepsis > 0 else np.nan

    return {
        "thresh": thresh,
        "n_alerted": n_alerted,
        "tp_sepsis": tp_sepsis,
        "tp_pretr": tp_pretr,
        "tp_pretr_2h": tp_pretr_2h,
        "ppv_sepsis": ppv_sepsis,
        "ppv_pretr": ppv_pretr,
        "sens_covered": sens_covered,
        "yield_overall": yield_overall,
        "nne": nne,
        "alerted_stay_ids": set(alerted_stay_ids),
        "tp_stay_ids": set(first_alerts[sepsis_det]["ICUSTAY_ID"]),
    }


def run_bootstrap_for_budget(
    df: pd.DataFrame, model_col: str, target_n: int, B: int = 1000, seed: int = 42
) -> dict:
    np.random.seed(seed)
    unique_stays = df["ICUSTAY_ID"].unique()
    n_stays = len(unique_stays)

    point_eval = evaluate_stay_alerts(df, model_col, target_n)
    thresh = point_eval["thresh"]

    # Pre-aggregate first alert per stay at fixed threshold
    df_above = df[df[model_col] >= thresh]
    first_above = df_above.sort_values(["ICUSTAY_ID", "ABS_END_H"]).groupby("ICUSTAY_ID").first()

    first_above["is_tp"] = (
        (first_above["SEPSIS_LABEL"] == 1)
        & (first_above["T_SEPSIS3_H_ICU"] - first_above["ABS_END_H"] > 0)
        & (first_above["T_SEPSIS3_H_ICU"] - first_above["ABS_END_H"] <= 6.0)
    )

    alerted_set = set(first_above.index)
    tp_set = set(first_above[first_above["is_tp"]].index)
    septic_set = set(df[df["SEPSIS_LABEL"] == 1]["ICUSTAY_ID"].unique())
    covered_set = set(df[(df["SEPSIS_LABEL"] == 1) & (df["T_SEPSIS3_H_ICU"] <= 36.0)]["ICUSTAY_ID"].unique())

    ppv_samples = []
    sens_cov_samples = []
    yield_all_samples = []
    nne_samples = []

    for _ in range(B):
        boot_stays = np.random.choice(unique_stays, size=n_stays, replace=True)
        # Count occurrences of alerted, tp, covered, septic
        boot_stay_counts = pd.Series(boot_stays).value_counts()
        
        n_boot_alerted = boot_stay_counts[boot_stay_counts.index.isin(alerted_set)].sum()
        n_boot_tp = boot_stay_counts[boot_stay_counts.index.isin(tp_set)].sum()
        n_boot_cov = boot_stay_counts[boot_stay_counts.index.isin(covered_set)].sum()
        n_boot_septic = boot_stay_counts[boot_stay_counts.index.isin(septic_set)].sum()

        if n_boot_alerted > 0:
            ppv_samples.append(n_boot_tp / n_boot_alerted * 100)
        if n_boot_cov > 0:
            sens_cov_samples.append(n_boot_tp / n_boot_cov * 100)
        if n_boot_septic > 0:
            yield_all_samples.append(n_boot_tp / n_boot_septic * 100)
        if n_boot_tp > 0:
            nne_samples.append(n_boot_alerted / n_boot_tp)
        else:
            nne_samples.append(np.nan)

    def ci_str(val: float, samples: list[float], fmt: str = ".2f") -> str:
        s_clean = [x for x in samples if not np.isnan(x)]
        if len(s_clean) == 0:
            return f"{val:{fmt}} (N/A)"
        low, high = np.percentile(s_clean, [2.5, 97.5])
        return f"{val:{fmt}} ({low:{fmt}}–{high:{fmt}})"

    return {
        "point": point_eval,
        "ppv_ci": ci_str(point_eval["ppv_sepsis"], ppv_samples, ".2f"),
        "sens_cov_ci": ci_str(point_eval["sens_covered"], sens_cov_samples, ".2f"),
        "yield_all_ci": ci_str(point_eval["yield_overall"], yield_all_samples, ".2f"),
        "nne_ci": ci_str(point_eval["nne"], nne_samples, ".1f"),
    }


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROJECT_ROOT / "reports" / "predictions" / "canonical_test_predictions_harmonized.parquet")

    rows = []
    logger.info("Computing alert budget frontiers with patient-clustered bootstrap (B=1,000)...")

    for target_n, rate, budget_name in BUDGET_TARGETS:
        logger.info("Processing budget: %s (%d episodes)...", budget_name, target_n)
        for model_key, model_display in EVAL_MODELS.items():
            boot_res = run_bootstrap_for_budget(df, model_key, target_n, B=1000)
            pt = boot_res["point"]

            rows.append({
                "Prespecified Alert Budget": budget_name,
                "Model Architecture": model_display,
                "Operating Threshold (tau)": f"{pt['thresh']:.4f}",
                "Emitted Alert Episodes": pt["n_alerted"],
                "Emitted Rate (/100 pt-days)": f"{rate:.2f}",
                "Episode PPV % [95% CI]": boot_res["ppv_ci"] + "%",
                "True Positive Sepsis Detections": pt["tp_sepsis"],
                "Pretreatment Detections (delta=0h)": pt["tp_pretr"],
                "Pretreatment Detections (delta=2h)": pt["tp_pretr_2h"],
                "Detection Sensitivity in Covered Sepsis % [95% CI]": boot_res["sens_cov_ci"] + "%",
                "Overall Sepsis Detection Yield % [95% CI]": boot_res["yield_all_ci"] + "%",
                "NNE per Detected Patient [95% CI]": boot_res["nne_ci"],
            })

    df_t3 = pd.DataFrame(rows)
    out_csv = TABLE_DIR / "Table3_Alert_Budget_Refractory_Workload.csv"
    df_t3.to_csv(out_csv, index=False)
    logger.info("Saved Table 3 to %s", out_csv)
    print(df_t3[["Prespecified Alert Budget", "Model Architecture", "Episode PPV % [95% CI]", "True Positive Sepsis Detections", "Pretreatment Detections (delta=0h)", "NNE per Detected Patient [95% CI]"]].to_string())


if __name__ == "__main__":
    main()
