#!/usr/bin/env python3
"""compute_interval_censored_bounds.py

Constructs Table 4: Two-Axis Clinical Subgroup & Timestamp Uncertainty Matrix.
Replaces artificial midnight-imputed hourly categories in MIMIC-III with
honest interval-censored bounds: [STARTDATE 00:00:00, STARTDATE + 1 day 00:00:00).

Contrasts:
  - MIMIC-III (MetaVision/CareVue): Bounded date-level prescription proxy with interval uncertainty.
  - MIMIC-IV: Exact hourly-resolved order and administration timeline (eMAR).

Opportunity deadline is defined relative to the surveillance protocol parameters:
  T_deadline = T_0 + b + delta = 18h + 4h = 22h after ICU admission.
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

    # Classification relative to 22h surveillance opportunity deadline (b=18h, delta=4h)
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
    df["opp_22h_bound"] = np.where(
        def_unavail, "Treatment at or before deadline (<=22h)",
        np.where(def_avail, "Treatment after deadline (>22h)", "Indeterminate (Date crosses deadline)")
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

    # Axis 2: Treatment relative to 22h deadline
    treat_cat = pd.Series("Unknown", index=df.index)
    treat_cat[t_tr <= 22] = "Treatment at or before deadline (<=22h)"
    treat_cat[t_tr > 22] = "Treatment after deadline (>22h)"

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

    ct3_opp = pd.crosstab(df3["onset_cat"], df3["opp_22h_bound"], margins=True)
    ct4_exact = pd.crosstab(df4["onset_cat"], df4["treat_cat_exact"], margins=True)

    t4_rows = []
    # MIMIC-III summary rows
    for onset in ["Early Incident (0-24h)", "Late Incident (>24h)", "All"]:
        t4_rows.append({
            "Database Release & Cohort": "MIMIC-III v1.4 (Broad Incident Sepsis, N=3,581)",
            "Sepsis Onset Category": onset,
            "Total Incident Episodes": ct3_opp.loc[onset, "All"],
            "Treatment at or before Opportunity Deadline (<=22h)": f"{ct3_opp.loc[onset, 'Treatment at or before deadline (<=22h)']} ({ct3_opp.loc[onset, 'Treatment at or before deadline (<=22h)']/ct3_opp.loc[onset, 'All']*100:.1f}%)",
            "Treatment after Opportunity Deadline (>22h)": f"{ct3_opp.loc[onset, 'Treatment after deadline (>22h)']} ({ct3_opp.loc[onset, 'Treatment after deadline (>22h)']/ct3_opp.loc[onset, 'All']*100:.1f}%)",
            "Indeterminate (Date Crosses Deadline)": f"{ct3_opp.loc[onset, 'Indeterminate (Date crosses deadline)']} ({ct3_opp.loc[onset, 'Indeterminate (Date crosses deadline)']/ct3_opp.loc[onset, 'All']*100:.1f}%)",
            "Clinical & Measurement Specification": "Prescription start date proxy evaluated with [00:00, 24:00) interval bounds. Deadline reflects b=18h lookback + delta=4h lead buffer.",
        })

    # MIMIC-IV summary rows
    for onset in ["Early Incident (0-24h)", "Late Incident (>24h)", "All"]:
        t4_rows.append({
            "Database Release & Cohort": "MIMIC-IV v3.1 (Broad Incident Sepsis, N=1,596)",
            "Sepsis Onset Category": onset,
            "Total Incident Episodes": ct4_exact.loc[onset, "All"],
            "Treatment at or before Opportunity Deadline (<=22h)": f"{ct4_exact.loc[onset, 'Treatment at or before deadline (<=22h)']} ({ct4_exact.loc[onset, 'Treatment at or before deadline (<=22h)']/ct4_exact.loc[onset, 'All']*100:.1f}%)",
            "Treatment after Opportunity Deadline (>22h)": f"{ct4_exact.loc[onset, 'Treatment after deadline (>22h)']} ({ct4_exact.loc[onset, 'Treatment after deadline (>22h)']/ct4_exact.loc[onset, 'All']*100:.1f}%)",
            "Indeterminate (Date Crosses Deadline)": "0 (0.0%)",
            "Clinical & Measurement Specification": "Confirmed hourly administration timestamps from electronic medication administration records (eMAR).",
        })

    df_t4 = pd.DataFrame(t4_rows)
    out_path = TABLE_DIR / "Table4_Two_Axis_Subgroup_Uncertainty_Matrix.csv"
    df_t4.to_csv(out_path, index=False)
    logger.info("Saved Table 4 to %s", out_path)
    print(df_t4.to_string())


if __name__ == "__main__":
    main()
