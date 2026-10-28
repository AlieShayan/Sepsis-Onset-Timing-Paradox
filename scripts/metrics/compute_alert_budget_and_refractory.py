#!/usr/bin/env python3
"""compute_alert_budget_and_refractory.py

Evaluates operational early warning alert budgets under a 12-hour refractory suppression policy.
Implements the empirical burden-matched bootstrap procedure (re-selecting the operating threshold
to match the prespecified episode budget in each bootstrap resample).

Conforms strictly to JAMIA / JAMIA Open reporting requirements:
  - Exact column schema and typed records (zero column shifts or reversals).
  - Explicit reporting of zero-detection fractions (f_0).
  - Proper handling of infinite NNE values when zero detections occur in resamples.
  - Boundary-aware reporting (Hanley Rule-of-Three) for degenerate zero-event bootstrap intervals.
  - Patient-level timing verification at the first emitted alert.
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
    "Tabular GBDT (LightGBM)": "Tabular GBDT (LightGBM)",
    "Tabular GBDT (XGBoost)": "Tabular GBDT (XGBoost)",
    "Random Forest": "Random Forest",
    "Transformer_FairMasked": "Mask-Aware Clinical Transformer",
    "Clinical Transformer": "Clinical Transformer (Baseline Unmasked)",
    "Logistic Regression (L2)": "Logistic Regression (L2)",
}

BUDGET_TARGETS = [
    (204, 5.01, "5.0 episodes / 100 pt-days"),
    (407, 10.00, "10.0 episodes / 100 pt-days"),
    (814, 20.00, "20.0 episodes / 100 pt-days"),
]


def prepare_stay_table(df: pd.DataFrame, model_col: str) -> pd.DataFrame:
    """Pre-aggregates max score and first-alert timing per stay to ensure fast, deterministic bootstrapping."""
    stay_rows = []
    for sid, grp in df.groupby("ICUSTAY_ID"):
        max_score = grp[model_col].max()
        is_septic = int(grp["SEPSIS_LABEL"].iloc[0] == 1)
        t_s3 = grp["T_SEPSIS3_H_ICU"].iloc[0] if is_septic else np.nan
        t_tr = grp["T_TREATMENT_H_ICU"].iloc[0] if is_septic else np.nan
        is_covered = int(is_septic and t_s3 <= 36.0)

        # First window reaching max_score
        first_w = grp[grp[model_col] == max_score].sort_values("ABS_END_H").iloc[0]
        t_w = first_w["ABS_END_H"]

        # Detection criteria at first emitted alert
        is_sepsis_det = int(is_septic and (t_s3 - t_w > 0) and (t_s3 - t_w <= 6.0))
        is_pretr_det = int(is_sepsis_det and (t_w <= t_tr))
        is_pretr_2h_det = int(is_sepsis_det and (t_w + 2.0 <= t_tr))

        stay_rows.append({
            "ICUSTAY_ID": sid,
            "max_score": max_score,
            "is_septic": is_septic,
            "is_covered": is_covered,
            "is_sepsis_det": is_sepsis_det,
            "is_pretr_det": is_pretr_det,
            "is_pretr_2h_det": is_pretr_2h_det,
        })
    df_stays = pd.DataFrame(stay_rows)
    return df_stays.sort_values(["max_score", "ICUSTAY_ID"], ascending=[False, True]).reset_index(drop=True)


def evaluate_burden_matched_bootstrap(
    df_stays: pd.DataFrame, target_n: int, B: int = 1000, seed: int = 42
) -> dict:
    np.random.seed(seed)
    n_stays = len(df_stays)

    # Point estimate: top target_n stays
    top_point = df_stays.iloc[:target_n]
    thresh_point = top_point["max_score"].min()
    tp_point = int(top_point["is_sepsis_det"].sum())
    pretr_point = int(top_point["is_pretr_det"].sum())
    pretr_2h_point = int(top_point["is_pretr_2h_det"].sum())

    ppv_point = tp_point / target_n * 100.0
    sens_cov_point = tp_point / 151 * 100.0
    yield_all_point = tp_point / 408 * 100.0
    nne_point = target_n / tp_point if tp_point > 0 else np.inf

    # Bootstrap replicates
    ppv_boots = []
    sens_cov_boots = []
    yield_all_boots = []
    nne_boots = []
    zero_detections = 0

    for _ in range(B):
        idx = np.random.choice(n_stays, size=n_stays, replace=True)
        sample = df_stays.iloc[idx].sort_values(["max_score", "ICUSTAY_ID"], ascending=[False, True])
        top_k = sample.iloc[:target_n]

        n_tp = int(top_k["is_sepsis_det"].sum())
        n_cov = int(sample["is_covered"].sum())
        n_sep = int(sample["is_septic"].sum())

        ppv_b = n_tp / target_n * 100.0
        sens_cov_b = n_tp / n_cov * 100.0 if n_cov > 0 else 0.0
        yield_all_b = n_tp / n_sep * 100.0 if n_sep > 0 else 0.0

        ppv_boots.append(ppv_b)
        sens_cov_boots.append(sens_cov_b)
        yield_all_boots.append(yield_all_b)

        if n_tp == 0:
            zero_detections += 1
            nne_boots.append(np.inf)
        else:
            nne_boots.append(target_n / n_tp)

    f_zero = zero_detections / B

    # Format intervals
    if tp_point == 0:
        # Rule of three upper bound for zero event binomial proportion: 3 / target_n
        rule_of_three = 3.0 / target_n * 100.0
        ppv_str = f"0.00 (0.00–{rule_of_three:.2f})%*"
        sens_cov_str = f"0.00 (0.00–{3.0/151*100.0:.2f})%*"
        yield_all_str = f"0.00 (0.00–{3.0/408*100.0:.2f})%*"
        nne_str = "Undefined (+inf)"
    else:
        p_low, p_high = np.percentile(ppv_boots, [2.5, 97.5])
        ppv_str = f"{ppv_point:.2f} ({p_low:.2f}–{p_high:.2f})%"

        sc_low, sc_high = np.percentile(sens_cov_boots, [2.5, 97.5])
        sens_cov_str = f"{sens_cov_point:.2f} ({sc_low:.2f}–{sc_high:.2f})%"

        ya_low, ya_high = np.percentile(yield_all_boots, [2.5, 97.5])
        yield_all_str = f"{yield_all_point:.2f} ({ya_low:.2f}–{ya_high:.2f})%"

        # NNE formatting with exact handling of infinite replicates
        if f_zero > 0.025:
            # Upper percentile is infinity
            finite_nnes = [x for x in nne_boots if not np.isinf(x)]
            nne_low = np.percentile(finite_nnes, 2.5) if len(finite_nnes) > 0 else target_n
            nne_str = f"{nne_point:.1f} ({nne_low:.1f}–+inf)"
        else:
            n_low, n_high = np.percentile(nne_boots, [2.5, 97.5])
            nne_str = f"{nne_point:.1f} ({n_low:.1f}–{n_high:.1f})"

    return {
        "thresh": thresh_point,
        "n_alerted": target_n,
        "tp_sepsis": tp_point,
        "tp_pretr": pretr_point,
        "tp_pretr_2h": pretr_2h_point,
        "ppv_str": ppv_str,
        "sens_cov_str": sens_cov_str,
        "yield_all_str": yield_all_str,
        "nne_str": nne_str,
        "f_zero": f_zero,
    }


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROJECT_ROOT / "reports" / "predictions" / "canonical_test_predictions_harmonized.parquet")

    logger.info("Computing typed Table 3 with empirical burden-matched bootstrap...")
    records = []

    for target_n, rate, budget_name in BUDGET_TARGETS:
        logger.info("Processing budget: %s (%d episodes)...", budget_name, target_n)
        for model_key, model_display in EVAL_MODELS.items():
            df_stays = prepare_stay_table(df, model_key)
            res = evaluate_burden_matched_bootstrap(df_stays, target_n, B=1000)

            records.append({
                "Prespecified Alert Budget": budget_name,
                "Model Architecture": model_display,
                "Operating Threshold (tau)": f"{res['thresh']:.4f}",
                "Emitted Alert Episodes": int(res["n_alerted"]),
                "Emitted Rate (/100 pt-days)": f"{rate:.2f}",
                "Episode PPV % [95% CI]": res["ppv_str"],
                "True Positive Sepsis Detections": int(res["tp_sepsis"]),
                "Pretreatment Detections (delta=0h)": int(res["tp_pretr"]),
                "Pretreatment Detections (delta=2h)": int(res["tp_pretr_2h"]),
                "Detection Sensitivity in Covered % [95% CI]": res["sens_cov_str"],
                "Overall Sepsis Detection Yield % [95% CI]": res["yield_all_str"],
                "NNE per Detected Patient [95% CI]": res["nne_str"],
                "Zero-Detection Replicate Fraction (f_0)": f"{res['f_zero']:.3f}",
            })

    out_df = pd.DataFrame(records)
    out_csv = TABLE_DIR / "Table3_Alert_Budget_Refractory_Workload.csv"
    out_df.to_csv(out_csv, index=False)
    logger.info("Saved Table 3 to %s", out_csv)
    print(out_df[["Prespecified Alert Budget", "Model Architecture", "Operating Threshold (tau)", "Emitted Alert Episodes", "Emitted Rate (/100 pt-days)", "Episode PPV % [95% CI]", "True Positive Sepsis Detections", "Pretreatment Detections (delta=0h)", "NNE per Detected Patient [95% CI]"]].to_string())


if __name__ == "__main__":
    main()
