#!/usr/bin/env python3
"""generate_table1_cohort_progression_ledger.py

Generates Table 1: Cohort Progression & Denominator Accounting Ledger.
Traces every single ICU admission from the base test cohort through surveillance eligibility,
sepsis onset timing, prediction lookback windows, and treatment opportunity.
Conforms strictly to JAMIA / JAMIA Open reporting requirements.
"""
from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path("/home/alireza/Sepsis-Onset-Timing-Paradox")
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROJECT_ROOT / "reports" / "predictions" / "canonical_test_predictions_harmonized.parquet"
    df = pd.read_parquet(parquet_path)
    df_tri = pd.read_csv(PROJECT_ROOT / "reports" / "tables" / "tri_onset_labels_mimiciii.csv")

    total_stays = df["ICUSTAY_ID"].nunique()
    total_windows = len(df)
    total_hours = total_windows * 1.0
    total_pt_days = total_hours / 24.0

    septic_mask = df["SEPSIS_LABEL"] == 1
    septic_stays = df[septic_mask]["ICUSTAY_ID"].nunique()
    control_stays = total_stays - septic_stays

    # Onset timing among septic stays
    df_septic_stays = df[septic_mask].drop_duplicates("ICUSTAY_ID")
    t_s = df_septic_stays["T_SEPSIS3_H_ICU"]

    # Incident Sepsis (> 0h)
    incident_sepsis_stays = (t_s > 0).sum()

    # Early Incident (<= 36h) vs Late Incident (> 36h)
    # Surveillance is hours 19-30. With H=6h lookback, [T_s3 - 6, T_s3] intersects [19, 30] iff T_s3 in [20, 36].
    covered_sepsis_stays = ((t_s >= 20.0) & (t_s <= 36.0)).sum()
    late_sepsis_stays = (t_s > 36.0).sum()

    # Canonical Risk Set
    n_pos_canon = (df["LABEL"] == 1).sum()
    n_neg_canon = (df["LABEL"] == 0).sum()
    prev_canon = n_pos_canon / total_windows * 100

    # Risk Set R0^hybrid (delta = 0h): w_end <= T_TREATMENT_H_ICU (T_susp for cases, T_Rx,start for controls)
    mask_r0 = df["T_TREATMENT_H_ICU"].isna() | (df["ABS_END_H"] <= df["T_TREATMENT_H_ICU"])
    df_r0 = df[mask_r0]
    stays_r0 = df_r0["ICUSTAY_ID"].nunique()
    n_pos_r0 = (df_r0["LABEL"] == 1).sum()
    n_neg_r0 = (df_r0["LABEL"] == 0).sum()
    tot_r0 = len(df_r0)
    prev_r0 = n_pos_r0 / tot_r0 * 100
    pt_days_r0 = tot_r0 / 24.0

    # Risk Set R2^hybrid (delta = 2h): w_end <= T_TREATMENT_H_ICU - 2.0
    mask_r2 = df["T_TREATMENT_H_ICU"].isna() | (df["ABS_END_H"] <= df["T_TREATMENT_H_ICU"] - 2.0)
    df_r2 = df[mask_r2]
    stays_r2 = df_r2["ICUSTAY_ID"].nunique()
    n_pos_r2 = (df_r2["LABEL"] == 1).sum()
    n_neg_r2 = (df_r2["LABEL"] == 0).sum()
    tot_r2 = len(df_r2)
    prev_r2 = n_pos_r2 / tot_r2 * 100
    pt_days_r2 = tot_r2 / 24.0

    # Uniform Earliest-Prescription Task (R_delta^Rx): t + delta <= T_Rx,start across all stays
    case_rx_map = df_tri.set_index("ICUSTAY_ID")["T_TREATMENT_H_ICU"].to_dict()
    df_uniform = df.copy()
    df_uniform["T_RX_START_H_ICU"] = df_uniform["T_TREATMENT_H_ICU"]  # controls already have T_Rx,start
    is_case = df_uniform["SEPSIS_LABEL"] == 1
    df_uniform.loc[is_case, "T_RX_START_H_ICU"] = df_uniform.loc[is_case, "ICUSTAY_ID"].map(case_rx_map)

    # Uniform delta = 0h
    mask_rx0 = df_uniform["T_RX_START_H_ICU"].isna() | (df_uniform["ABS_END_H"] <= df_uniform["T_RX_START_H_ICU"])
    df_rx0 = df_uniform[mask_rx0]
    stays_rx0 = df_rx0["ICUSTAY_ID"].nunique()
    tot_rx0 = len(df_rx0)
    n_pos_rx0 = (df_rx0["LABEL"] == 1).sum()
    n_neg_rx0 = (df_rx0["LABEL"] == 0).sum()
    pt_days_rx0 = tot_rx0 / 24.0

    # Uniform delta = 2h
    mask_rx2 = df_uniform["T_RX_START_H_ICU"].isna() | (df_uniform["ABS_END_H"] <= df_uniform["T_RX_START_H_ICU"] - 2.0)
    df_rx2 = df_uniform[mask_rx2]
    stays_rx2 = df_rx2["ICUSTAY_ID"].nunique()
    tot_rx2 = len(df_rx2)
    n_pos_rx2 = (df_rx2["LABEL"] == 1).sum()
    n_neg_rx2 = (df_rx2["LABEL"] == 0).sum()
    pt_days_rx2 = tot_rx2 / 24.0

    # Accounting for missing/censored decision windows
    scheduled_windows = total_stays * 12
    missing_windows = scheduled_windows - total_windows  # 1,348
    post_onset_censored = 623
    monitoring_cessation_censored = 725

    rows = [
        {
            "Stage / Cohort Partition": "1. Base Prospective-Replay Test Cohort",
            "ICU Stays (N)": f"{total_stays:,}",
            "Evaluated Windows": f"{total_windows:,} (out of {scheduled_windows:,} scheduled)",
            "Surveillance Volume": f"{total_pt_days:.1f} patient-days ({total_hours:,.0f} patient-hours)",
            "Proportion / Window Prevalence": "100.0% of cohort",
            "Clinical Definition & Measurement Convention": (
                "Adult first ICU admissions in MIMIC-III MetaVision (2008–2012) with ICU stay >= 30h. Evaluated at up to 12 scheduled "
                f"hourly decision bins [t-1, t] at t in {{19..30}}. 1,348 windows right-censored ({post_onset_censored} "
                f"post-sepsis onset, {monitoring_cessation_censored} observation/monitoring cessation)."
            ),
        },
        {
            "Stage / Cohort Partition": "2. Non-Sepsis Control ICU Stays",
            "ICU Stays (N)": f"{control_stays:,}",
            "Evaluated Windows": "93,427 outcome-negative",
            "Surveillance Volume": f"{93427/24:.1f} patient-days (93,427 patient-hours)",
            "Proportion / Window Prevalence": f"{control_stays/total_stays*100:.2f}% of stays",
            "Clinical Definition & Measurement Convention": (
                "Patients without consensus Sepsis-3 during hospitalization. All windows serve as outcome-negative "
                f"surveillance (180 stays censored prior to hour 30 due to observation data cessation / monitoring termination, accounting for {monitoring_cessation_censored} windows)."
            ),
        },
        {
            "Stage / Cohort Partition": "3. Consensus Sepsis-3 Stays (S=1)",
            "ICU Stays (N)": f"{septic_stays:,}",
            "Evaluated Windows": "4,273 windows (678 positive, 3,595 negative)",
            "Surveillance Volume": f"{4273/24:.1f} patient-days (4,273 patient-hours)",
            "Proportion / Window Prevalence": f"{septic_stays/total_stays*100:.2f}% of stays",
            "Clinical Definition & Measurement Convention": (
                "Patients meeting Sepsis-3 consensus criteria (blood culture + antibiotic + acute Delta-SOFA >= 2). "
                "Median onset: hour 49.0 (IQR: 30.0h–77.0h; range: 20h–369h). 110 stays censored at onset prior to hour 30 (623 windows)."
            ),
        },
        {
            "Stage / Cohort Partition": "  3a. Late-Onset Sepsis (Onset >36h)",
            "ICU Stays (N)": f"{late_sepsis_stays:,}",
            "Evaluated Windows": "3,084 outcome-negative windows",
            "Surveillance Volume": f"{3084/24:.1f} patient-days (3,084 patient-hours)",
            "Proportion / Window Prevalence": f"{late_sepsis_stays/septic_stays*100:.1f}% of septic stays",
            "Clinical Definition & Measurement Convention": (
                "Onset occurs after hour 36. 6-hour lookback [T_S3 - 6, T_S3] does not intersect surveillance window [19, 30]. "
                "All windows are outcome-negative baseline surveillance (V_i = 0)."
            ),
        },
        {
            "Stage / Cohort Partition": "  3b. Surveillance-Covered Sepsis (V=1)",
            "ICU Stays (N)": f"{covered_sepsis_stays:,}",
            "Evaluated Windows": "1,189 windows (678 positive, 511 negative)",
            "Surveillance Volume": f"{1189/24:.1f} patient-days (1,189 patient-hours)",
            "Proportion / Window Prevalence": f"{covered_sepsis_stays/septic_stays*100:.1f}% of septic stays",
            "Clinical Definition & Measurement Convention": (
                "Onset occurs between ICU hours 20 and 36, such that [T_S3 - 6h, T_S3] intersects surveillance interval [19, 30] (V_i = 1)."
            ),
        },
        {
            "Stage / Cohort Partition": "4. Canonical Sepsis-3 Risk Set (H=6h)",
            "ICU Stays (N)": f"{total_stays:,} (151 horizon-positive)",
            "Evaluated Windows": f"{total_windows:,} (678 positive / {n_neg_canon:,} negative)",
            "Surveillance Volume": f"{total_pt_days:.1f} patient-days",
            "Proportion / Window Prevalence": f"{prev_canon:.3f}% window prevalence",
            "Clinical Definition & Measurement Convention": (
                "Full unconstrained risk set. Target windows have end-time within 6 hours before Sepsis-3 onset (0 < T_S3 - t <= 6h)."
            ),
        },
        {
            "Stage / Cohort Partition": "5. Outcome-Stratified Hybrid Set R0_hybrid (delta=0h)",
            "ICU Stays (N)": f"{stays_r0:,} (150 horizon-positive)",
            "Evaluated Windows": f"{tot_r0:,} ({n_pos_r0:,} positive / {n_neg_r0:,} negative)",
            "Surveillance Volume": f"{pt_days_r0:.1f} patient-days",
            "Proportion / Window Prevalence": f"{prev_r0:.3f}% window prevalence",
            "Clinical Definition & Measurement Convention": (
                "Exploratory outcome-stratified hybrid restriction censoring cases at infection-suspicion anchor T_susp and controls at earliest prescription T_Rx,start. "
                "150 out of 151 covered patients (99.3%) maintain >=1 pre-suspicion target window."
            ),
        },
        {
            "Stage / Cohort Partition": "6. Outcome-Stratified Hybrid Set R2_hybrid (delta=2h)",
            "ICU Stays (N)": f"{stays_r2:,} (149 horizon-positive)",
            "Evaluated Windows": f"{tot_r2:,} ({n_pos_r2:,} positive / {n_neg_r2:,} negative)",
            "Surveillance Volume": f"{pt_days_r2:.1f} patient-days",
            "Proportion / Window Prevalence": f"{prev_r2:.3f}% window prevalence",
            "Clinical Definition & Measurement Convention": (
                "Exploratory outcome-stratified hybrid restriction requiring a 2-hour proxy-timing margin (w_end + 2h <= C_i). "
                "149 out of 151 covered patients (98.7%) maintain >=1 actionable target window."
            ),
        },
        {
            "Stage / Cohort Partition": "7. Uniform Earliest-Rx Task R0_Rx (delta=0h)",
            "ICU Stays (N)": f"{stays_rx0:,} (0 horizon-positive)",
            "Evaluated Windows": f"{tot_rx0:,} ({n_pos_rx0:,} positive / {n_neg_rx0:,} negative)",
            "Surveillance Volume": f"{pt_days_rx0:.1f} patient-days",
            "Proportion / Window Prevalence": "0.000% (Not Meaningfully Estimable)",
            "Clinical Definition & Measurement Convention": (
                "Uniform eligibility task requiring t <= T_Rx,start across all stays. Under declared midnight assignment of prescription-start dates, "
                "no horizon-positive observations remained (0 positive, 65,926 negative; N_total = 65,926), preventing discrimination estimation. "
                "Control attrition identity: 65,926 = 93,427 - W_fully_excluded (27,388 from 2,287 stays) - W_partially_censored (113 from 83 stays)."
            ),
        },
        {
            "Stage / Cohort Partition": "8. Uniform Earliest-Rx Task R2_Rx (delta=2h)",
            "ICU Stays (N)": f"{stays_rx2:,} (0 horizon-positive)",
            "Evaluated Windows": f"{tot_rx2:,} ({n_pos_rx2:,} positive / {n_neg_rx2:,} negative)",
            "Surveillance Volume": f"{pt_days_rx2:.1f} patient-days",
            "Proportion / Window Prevalence": "0.000% (Not Meaningfully Estimable)",
            "Clinical Definition & Measurement Convention": (
                "Uniform eligibility task requiring a 2-hour proxy-timing margin (t + 2h <= T_Rx,start) across all stays. Under declared midnight assignment, "
                "no horizon-positive observations remained (0 positive, 65,877 negative; N_total = 65,877), preventing discrimination estimation. "
                "Buffer attrition identity: 65,877 = 65,926 - W_additional_margin (49 from 83 partially retained stays)."
            ),
        },
    ]

    out_df = pd.DataFrame(rows)
    out_csv = TABLE_DIR / "Table1_Cohort_Progression_Ledger.csv"
    out_df.to_csv(out_csv, index=False)
    logger.info("Saved Table 1 to %s", out_csv)
    print(out_df.to_string())


if __name__ == "__main__":
    main()
