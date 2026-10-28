#!/usr/bin/env python3
"""compute_interval_censored_bounds.py

Constructs Table 4: Two-Axis Clinical Subgroup & Timestamp Uncertainty Matrix.
Replaces artificial midnight-imputed hourly categories in MIMIC-III with
honest interval-censored bounds: [STARTDATE 00:00:00, STARTDATE + 1 day 00:00:00).

Contrasts:
  - MIMIC-III (MetaVision/CareVue): Bounded date-level prescription proxy with interval uncertainty.
  - MIMIC-IV: Exact hourly-resolved order and administration timeline.
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


def analyze_mimiciii_interval_censoring(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["INTIME"] = pd.to_datetime(df["INTIME"])
    df["T_ONSET_DT"] = pd.to_datetime(df["T_ONSET"])
    df["T_TREAT_DATE"] = pd.to_datetime(df["T_TREATMENT"])

    # Half-open interval [a_i, b_i)
    a_i = df["T_TREAT_DATE"]
    b_i = a_i + pd.Timedelta(days=1)
    intime = df["INTIME"]

    # 1. Classification relative to ICU entry
    # Definitely Pre-ICU: entire interval ended before admission
    def_pre_icu = b_i <= intime
    # Definitely In-ICU: interval starts strictly at or after admission
    def_in_icu = a_i >= intime
    # Indeterminate: admission time falls within the day of prescription
    indet_icu = (a_i < intime) & (b_i > intime)

    # 2. Classification relative to 22h surveillance opportunity threshold (b=18h, delta=4h)
    thresh_22h = intime + pd.Timedelta(hours=22)
    def_unavail = b_i <= thresh_22h
    def_avail = a_i >= thresh_22h
    indet_opp = (a_i < thresh_22h) & (b_i > thresh_22h)

    # Axis 1: Onset timing
    t_s = df["T_SEPSIS3_H_ICU"]
    onset_cat = pd.Series("Unknown", index=df.index)
    onset_cat[t_s <= 0] = "Present on Arrival (<=0h)"
    onset_cat[(t_s > 0) & (t_s <= 24)] = "Early Incident (0-24h)"
    onset_cat[t_s > 24] = "Late Incident (>24h)"

    df["onset_cat"] = onset_cat
    df["icu_timing_bound"] = np.where(
        def_pre_icu, "Definitely Pre-ICU",
        np.where(def_in_icu, "Definitely In-ICU", "Indeterminate (Same Calendar Day)")
    )
    df["opp_22h_bound"] = np.where(
        def_unavail, "Definitely Unavailable (<=22h)",
        np.where(def_avail, "Definitely Available (>=22h)", "Indeterminate (Crosses 22h)")
    )

    return df


def analyze_mimiciv_exact_timing(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    t_s = df["T_SEPSIS3_H_ICU"]
    t_tr = df["T_TREATMENT_H_ICU"]

    # Axis 1: Onset timing
    onset_cat = pd.Series("Unknown", index=df.index)
    onset_cat[t_s <= 0] = "Present on Arrival (<=0h)"
    onset_cat[(t_s > 0) & (t_s <= 24)] = "Early Incident (0-24h)"
    onset_cat[t_s > 24] = "Late Incident (>24h)"

    # Axis 2: Exact hourly treatment timing
    treat_cat = pd.Series("Unknown", index=df.index)
    treat_cat[t_tr <= 0] = "Pre-ICU / ED Treated (<=0h)"
    treat_cat[(t_tr > 0) & (t_tr <= 2)] = "Ultra-Early ICU Treated (0-2h)"
    treat_cat[(t_tr > 2) & (t_tr <= 22)] = "Early ICU Treated (2-22h)"
    treat_cat[t_tr > 22] = "Actionable Opportunity (>22h)"

    df["onset_cat"] = onset_cat
    df["treat_cat_exact"] = treat_cat
    return df


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    c3_path = TABLE_DIR / "tri_onset_labels_mimiciii.csv"
    c4_path = TABLE_DIR / "tri_onset_labels_mimiciv.csv"

    logger.info("Analyzing MIMIC-III interval-censored uncertainty bounds...")
    df3 = analyze_mimiciii_interval_censoring(c3_path)

    logger.info("Analyzing MIMIC-IV exact hourly timing...")
    df4 = analyze_mimiciv_exact_timing(c4_path)

    # 1. Compile MIMIC-III cross-tabulation
    ct3_icu = pd.crosstab(df3["onset_cat"], df3["icu_timing_bound"], margins=True)
    ct3_opp = pd.crosstab(df3["onset_cat"], df3["opp_22h_bound"], margins=True)

    # 2. Compile MIMIC-IV cross-tabulation
    ct4_exact = pd.crosstab(df4["onset_cat"], df4["treat_cat_exact"], margins=True)

    logger.info("=== MIMIC-III ICU Entry Timing Bounds ===")
    print(ct3_icu)
    logger.info("=== MIMIC-III 22h Opportunity Bounds ===")
    print(ct3_opp)
    logger.info("=== MIMIC-IV Exact Hourly Timing ===")
    print(ct4_exact)

    # Export a structured Table 4
    t4_rows = []
    # MIMIC-III summary rows
    for onset in ["Early Incident (0-24h)", "Late Incident (>24h)", "All"]:
        t4_rows.append({
            "Database": "MIMIC-III (Interval-Censored Date Proxy)",
            "Sepsis Onset Category": onset,
            "Total Episodes": ct3_icu.loc[onset, "All"],
            "Definitely Pre-ICU / Unavailable": f"{ct3_opp.loc[onset, 'Definitely Unavailable (<=22h)']} ({ct3_opp.loc[onset, 'Definitely Unavailable (<=22h)']/ct3_opp.loc[onset, 'All']*100:.1f}%)",
            "Definitely Available Opportunity (>=22h)": f"{ct3_opp.loc[onset, 'Definitely Available (>=22h)']} ({ct3_opp.loc[onset, 'Definitely Available (>=22h)']/ct3_opp.loc[onset, 'All']*100:.1f}%)",
            "Indeterminate (Date Crosses Threshold)": f"{ct3_opp.loc[onset, 'Indeterminate (Crosses 22h)']} ({ct3_opp.loc[onset, 'Indeterminate (Crosses 22h)']/ct3_opp.loc[onset, 'All']*100:.1f}%)",
            "Methodological Note": "Evaluated with [00:00, 24:00) interval bounds to propagate date truncation uncertainty without midnight imputation.",
        })

    # MIMIC-IV summary rows
    for onset in ["Early Incident (0-24h)", "Late Incident (>24h)", "All"]:
        t4_rows.append({
            "Database": "MIMIC-IV (Hourly-Resolved Administration Benchmark)",
            "Sepsis Onset Category": onset,
            "Total Episodes": ct4_exact.loc[onset, "All"],
            "Definitely Pre-ICU / Unavailable": f"{ct4_exact.loc[onset, 'Pre-ICU / ED Treated (<=0h)'] + ct4_exact.loc[onset, 'Ultra-Early ICU Treated (0-2h)'] + ct4_exact.loc[onset, 'Early ICU Treated (2-22h)']} ({(ct4_exact.loc[onset, 'Pre-ICU / ED Treated (<=0h)'] + ct4_exact.loc[onset, 'Ultra-Early ICU Treated (0-2h)'] + ct4_exact.loc[onset, 'Early ICU Treated (2-22h)'])/ct4_exact.loc[onset, 'All']*100:.1f}%)",
            "Definitely Available Opportunity (>=22h)": f"{ct4_exact.loc[onset, 'Actionable Opportunity (>22h)']} ({ct4_exact.loc[onset, 'Actionable Opportunity (>22h)']/ct4_exact.loc[onset, 'All']*100:.1f}%)",
            "Indeterminate (Date Crosses Threshold)": "0 (0.0%)",
            "Methodological Note": "Confirmed hourly timestamp resolution from electronic medication administration records (eMAR).",
        })

    df_t4 = pd.DataFrame(t4_rows)
    out_path = TABLE_DIR / "Table4_Two_Axis_Subgroup_Uncertainty_Matrix.csv"
    df_t4.to_csv(out_path, index=False)
    logger.info("Saved Table 4 to %s", out_path)


if __name__ == "__main__":
    main()
