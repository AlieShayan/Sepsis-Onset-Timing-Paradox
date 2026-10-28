#!/usr/bin/env python3
"""generate_table1_cohort_progression_ledger.py

Generates Table 1: Cohort Progression & Denominator Accounting Ledger.
Traces every single ICU admission from the base test cohort through surveillance eligibility,
sepsis onset timing, prediction lookback windows, and treatment opportunity.
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
    prevalent_sepsis_stays = (t_s <= 0).sum()

    # Early vs Late Incident
    early_incident_stays = ((t_s > 0) & (t_s <= 36.0)).sum()
    late_incident_stays = (t_s > 36.0).sum()

    # Window-level H=6h pre-onset lookback
    w_end = df["ABS_END_H"].values
    t_s_all = df["T_SEPSIS3_H_ICU"].values
    is_s_all = df["T_SEPSIS3_H_ICU"].notna().values

    pos_s3_6h = is_s_all & (w_end <= t_s_all) & (w_end >= t_s_all - 6.0)
    v1_stays = df[pos_s3_6h]["ICUSTAY_ID"].nunique()
    v1_pos_windows = pos_s3_6h.sum()

    # Pretreatment opportunity within V=1 stays (t <= T_treatment)
    t_tr_all = df["T_TREATMENT_H_ICU"].values
    pos_pretreat_6h = pos_s3_6h & (w_end <= t_tr_all)
    o1_stays = df[pos_pretreat_6h]["ICUSTAY_ID"].nunique()
    o1_pos_windows = pos_pretreat_6h.sum()

    # Control surveillance windows
    control_windows = total_windows - v1_pos_windows

    ledger_rows = [
        {
            "Progression Stage": "1. Base Test Cohort Monitored",
            "Stay / Patient Count": f"{total_stays:,}",
            "Observation Windows": f"{total_windows:,}",
            "Monitored Person-Time": f"{total_pt_days:.1f} patient-days",
            "Prevalence / Attrition": "100.0%",
            "Methodological Definition & Accounting Rule": "All patient-disjoint ICU stays in the held-out test partition with streaming replay during ICU hours 19–30.",
        },
        {
            "Progression Stage": "2. Control (Non-Sepsis) ICU Stays",
            "Stay / Patient Count": f"{control_stays:,}",
            "Observation Windows": f"{len(df[~septic_mask]):,}",
            "Monitored Person-Time": f"{len(df[~septic_mask])/24:.1f} patient-days",
            "Prevalence / Attrition": f"{control_stays/total_stays*100:.2f}% of stays",
            "Methodological Definition & Accounting Rule": "Patients without consensus Sepsis-3 during entire hospitalization. All windows serve as true negative surveillance.",
        },
        {
            "Progression Stage": "3. Consensus Sepsis-3 Stays (S=1)",
            "Stay / Patient Count": f"{septic_stays:,}",
            "Observation Windows": f"{len(df[septic_mask]):,}",
            "Monitored Person-Time": f"{len(df[septic_mask])/24:.1f} patient-days",
            "Prevalence / Attrition": f"{septic_stays/total_stays*100:.2f}% of stays",
            "Methodological Definition & Accounting Rule": "Patients meeting Sepsis-3 consensus criteria (blood culture + antibiotic + acute delta SOFA >= 2). All 408 had onset > 0h.",
        },
        {
            "Progression Stage": "  3a. Late-Onset Sepsis (Onset > 36h)",
            "Stay / Patient Count": f"{late_incident_stays:,}",
            "Observation Windows": f"{len(df[septic_mask & (df['T_SEPSIS3_H_ICU'] > 36.0)]):,}",
            "Monitored Person-Time": f"{len(df[septic_mask & (df['T_SEPSIS3_H_ICU'] > 36.0)])/24:.1f} patient-days",
            "Prevalence / Attrition": f"{late_incident_stays/septic_stays*100:.1f}% of septic stays",
            "Methodological Definition & Accounting Rule": "Onset occurs after surveillance window [19, 30]. Windows in hours 19–30 are legitimate pre-onset control surveillance (V=0).",
        },
        {
            "Progression Stage": "  3b. Surveillance-Covered Sepsis (V=1)",
            "Stay / Patient Count": f"{v1_stays:,}",
            "Observation Windows": f"{len(df[df['ICUSTAY_ID'].isin(df[pos_s3_6h]['ICUSTAY_ID'].unique())]):,}",
            "Monitored Person-Time": f"{len(df[df['ICUSTAY_ID'].isin(df[pos_s3_6h]['ICUSTAY_ID'].unique())])/24:.1f} patient-days",
            "Prevalence / Attrition": f"{v1_stays/septic_stays*100:.1f}% of septic stays",
            "Methodological Definition & Accounting Rule": "Onset occurs between ICU hours 20 and 36, so [T_S3 - 6h, T_S3] intersects the active surveillance window [19, 30].",
        },
        {
            "Progression Stage": "4. Evaluated Sepsis-3 Target Windows (H=6h)",
            "Stay / Patient Count": f"{v1_stays:,}",
            "Observation Windows": f"{v1_pos_windows:,} positive / {control_windows:,} control",
            "Monitored Person-Time": f"{v1_pos_windows/24:.1f} positive pt-days",
            "Prevalence / Attrition": f"{v1_pos_windows/total_windows*100:.3f}% window prevalence",
            "Methodological Definition & Accounting Rule": "Windows with end time within 6h before Sepsis-3 onset. Evaluated for window AUROC and AUPRC.",
        },
        {
            "Progression Stage": "5. Pretreatment Actionable Windows (O=1 | V=1)",
            "Stay / Patient Count": f"{o1_stays:,}",
            "Observation Windows": f"{o1_pos_windows:,} positive / {control_windows:,} control",
            "Monitored Person-Time": f"{o1_pos_windows/24:.1f} positive pt-days",
            "Prevalence / Attrition": f"{o1_stays/v1_stays*100:.1f}% of covered stays",
            "Methodological Definition & Accounting Rule": "Positive windows where prediction time strictly precedes systemic IV antibiotic initiation (w_end <= T_treatment).",
        },
    ]

    df_ledger = pd.DataFrame(ledger_rows)
    out_path = TABLE_DIR / "Table1_Cohort_Progression_Ledger.csv"
    df_ledger.to_csv(out_path, index=False)
    logger.info("Saved Table 1 to %s", out_path)
    print(df_ledger.to_string())


if __name__ == "__main__":
    main()
