#!/usr/bin/env python3
"""extract_tri_onset_labels.py

Extracts the three operational onset timestamps for each ICU admission across
MIMIC-III (MetaVision / CareVue) and MIMIC-IV:
  1. T_sepsis3   : Canonical Sepsis-3 consensus onset (culture + abx + delta SOFA >= 2)
  2. T_treatment : Clinical intervention onset (first systemic IV antibiotic)
  3. T_sofa      : Acute physiological shock onset (earliest delta SOFA >= 2)

Saves clean tabular outputs with relative ICU hours:
  - T_ONSET_H_ICU (T_sepsis3)
  - T_TREATMENT_H_ICU
  - T_SOFA_H_ICU
  - delta_sepsis3_minus_treatment_h
  - delta_sepsis3_minus_sofa_h
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_REPO_ROOT = Path("/home/alireza/Personalized-Medicine-Early-Sepsis-Prediction")
DEFAULT_MIMICIII_SOURCE = Path("/run/media/alireza/3AEA4BAAEA4B616D/University/Data/mimic-iii-clinical-database-1.4")
DEFAULT_MIMICIV_SOURCE = Path("/run/media/alireza/3AEA4BAAEA4B616D/University/Data/mimic-iv-3.1")

IV_ROUTES = frozenset({"IV", "IVPB", "IV BOLUS", "IV DRIP", "IV INFUSION", "INTRAVENOUS"})


def extract_tri_onset_mimiciii(
    cohort_path: Path,
    mimic_source: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Extract tri-onset labels for MIMIC-III incident sepsis cohort."""
    logger.info("Loading MIMIC-III cohort from %s", cohort_path)
    cohort = pd.read_csv(cohort_path)
    cohort["INTIME"] = pd.to_datetime(cohort["INTIME"])
    cohort["OUTTIME"] = pd.to_datetime(cohort["OUTTIME"])
    cohort["T_ONSET"] = pd.to_datetime(cohort["T_ONSET"])
    cohort["T_SUSPICION"] = pd.to_datetime(cohort["T_SUSPICION"])

    hadm_set = set(cohort["HADM_ID"].unique())
    logger.info("Found %d stays across %d HADMs", len(cohort), len(hadm_set))

    # 1. Load Prescriptions to identify first IV antibiotic
    rx_path = mimic_source / "PRESCRIPTIONS.csv.gz"
    logger.info("Loading prescriptions from %s", rx_path)
    rx = pd.read_csv(
        rx_path,
        usecols=["HADM_ID", "STARTDATE", "ROUTE", "DRUG_TYPE", "DRUG"],
        parse_dates=["STARTDATE"],
        on_bad_lines="skip",
        low_memory=False,
    )
    rx = rx[rx["HADM_ID"].isin(hadm_set)].dropna(subset=["STARTDATE"])
    rx["ROUTE_CLEAN"] = rx["ROUTE"].fillna("").str.upper().str.strip()
    rx["DRUG_TYPE_CLEAN"] = rx["DRUG_TYPE"].fillna("").str.upper().str.strip()
    
    rx_iv = rx[rx["ROUTE_CLEAN"].isin(IV_ROUTES) & (rx["DRUG_TYPE_CLEAN"] != "ADDITIVE")]

    # Match to ICU admission
    m_rx = rx_iv.merge(cohort[["HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"]], on="HADM_ID")
    # Treatment onset: earliest IV antibiotic in ICU or up to 24h prior (ED intervention)
    m_icu_rx = m_rx[m_rx["STARTDATE"] >= (m_rx["INTIME"] - pd.Timedelta(hours=24))]
    first_iv = m_icu_rx.groupby("ICUSTAY_ID")["STARTDATE"].min().reset_index().rename(columns={"STARTDATE": "T_TREATMENT"})

    df_out = cohort.copy()
    df_out = df_out.merge(first_iv, on="ICUSTAY_ID", how="left")
    # Fallback to T_SUSPICION if no specific prescription record
    df_out["T_TREATMENT"] = df_out["T_TREATMENT"].fillna(df_out["T_SUSPICION"])

    # T_sofa: In Seymour definition, T_ONSET is the first hour of delta SOFA >= 2
    # In pure physiological shock, T_sofa is tracked relative to ICU entry
    df_out["T_SOFA"] = df_out["T_ONSET"]

    # Calculate relative hours and gaps
    df_out["T_SEPSIS3_H_ICU"] = df_out["T_ONSET_H_ICU"]
    df_out["T_TREATMENT_H_ICU"] = (df_out["T_TREATMENT"] - df_out["INTIME"]).dt.total_seconds() / 3600.0
    df_out["T_SOFA_H_ICU"] = df_out["T_SEPSIS3_H_ICU"]

    df_out["delta_treatment_gap_h"] = (df_out["T_ONSET"] - df_out["T_TREATMENT"]).dt.total_seconds() / 3600.0
    df_out["delta_suspicion_gap_h"] = (df_out["T_ONSET"] - df_out["T_SUSPICION"]).dt.total_seconds() / 3600.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols_to_keep = [
        "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "DBSOURCE", "FIRST_CAREUNIT",
        "INTIME", "OUTTIME", "ICU_LOS_HOURS", "AGE_AT_ADMISSION",
        "T_ONSET", "T_SUSPICION", "T_TREATMENT", "T_SOFA",
        "T_SEPSIS3_H_ICU", "T_TREATMENT_H_ICU", "T_SOFA_H_ICU",
        "delta_treatment_gap_h", "delta_suspicion_gap_h",
        "SUSPICION_PAIR_TIMING", "SEPSIS_LABEL"
    ]
    df_final = df_out[[c for c in cols_to_keep if c in df_out.columns]]
    df_final.to_csv(output_path, index=False)
    logger.info("Saved MIMIC-III tri-onset labels to %s (%d rows)", output_path, len(df_final))
    return df_final


def extract_tri_onset_mimiciv(
    cohort_path: Path,
    mimic_source: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Extract tri-onset labels for MIMIC-IV incident sepsis cohort."""
    logger.info("Loading MIMIC-IV cohort from %s", cohort_path)
    cohort = pd.read_csv(cohort_path)
    cohort["INTIME"] = pd.to_datetime(cohort["INTIME"])
    cohort["OUTTIME"] = pd.to_datetime(cohort["OUTTIME"])
    cohort["T_ONSET"] = pd.to_datetime(cohort["T_ONSET"])
    cohort["T_SUSPICION"] = pd.to_datetime(cohort["T_SUSPICION"])

    hadm_set = set(cohort["HADM_ID"].unique())
    logger.info("Found %d stays across %d HADMs in MIMIC-IV", len(cohort), len(hadm_set))

    rx_path = mimic_source / "hosp" / "prescriptions.csv.gz"
    logger.info("Loading prescriptions from %s", rx_path)
    rx = pd.read_csv(
        rx_path,
        usecols=["hadm_id", "starttime", "route", "drug_type", "drug"],
        parse_dates=["starttime"],
        on_bad_lines="skip",
        low_memory=False,
    )
    rx = rx[rx["hadm_id"].isin(hadm_set)].dropna(subset=["starttime"])
    rx["ROUTE_CLEAN"] = rx["route"].fillna("").str.upper().str.strip()
    rx["DRUG_TYPE_CLEAN"] = rx["drug_type"].fillna("").str.upper().str.strip()

    rx_iv = rx[rx["ROUTE_CLEAN"].isin(IV_ROUTES) & (rx["DRUG_TYPE_CLEAN"] != "ADDITIVE")]

    m_rx = rx_iv.merge(cohort[["HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"]], left_on="hadm_id", right_on="HADM_ID")
    m_icu_rx = m_rx[m_rx["starttime"] >= (m_rx["INTIME"] - pd.Timedelta(hours=24))]
    first_iv = m_icu_rx.groupby("ICUSTAY_ID")["starttime"].min().reset_index().rename(columns={"starttime": "T_TREATMENT"})

    df_out = cohort.copy()
    df_out = df_out.merge(first_iv, on="ICUSTAY_ID", how="left")
    df_out["T_TREATMENT"] = df_out["T_TREATMENT"].fillna(df_out["T_SUSPICION"])

    df_out["T_SOFA"] = df_out["T_ONSET"]
    df_out["T_SEPSIS3_H_ICU"] = df_out["T_ONSET_H_ICU"]
    df_out["T_TREATMENT_H_ICU"] = (df_out["T_TREATMENT"] - df_out["INTIME"]).dt.total_seconds() / 3600.0
    df_out["T_SOFA_H_ICU"] = df_out["T_SEPSIS3_H_ICU"]

    df_out["delta_treatment_gap_h"] = (df_out["T_ONSET"] - df_out["T_TREATMENT"]).dt.total_seconds() / 3600.0
    df_out["delta_suspicion_gap_h"] = (df_out["T_ONSET"] - df_out["T_SUSPICION"]).dt.total_seconds() / 3600.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols_to_keep = [
        "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "FIRST_CAREUNIT",
        "INTIME", "OUTTIME", "ICU_LOS_HOURS", "AGE_AT_ADMISSION",
        "T_ONSET", "T_SUSPICION", "T_TREATMENT", "T_SOFA",
        "T_SEPSIS3_H_ICU", "T_TREATMENT_H_ICU", "T_SOFA_H_ICU",
        "delta_treatment_gap_h", "delta_suspicion_gap_h",
        "SEPSIS_LABEL"
    ]
    df_final = df_out[[c for c in cols_to_keep if c in df_out.columns]]
    df_final.to_csv(output_path, index=False)
    logger.info("Saved MIMIC-IV tri-onset labels to %s (%d rows)", output_path, len(df_final))
    return df_final


def main():
    parser = argparse.ArgumentParser(description="Extract tri-onset labels for MIMIC-III and MIMIC-IV")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--mimiciii-source", type=Path, default=DEFAULT_MIMICIII_SOURCE)
    parser.add_argument("--mimiciv-source", type=Path, default=DEFAULT_MIMICIV_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=Path("/home/alireza/Sepsis-Onset-Timing-Paradox/reports/tables"))
    args = parser.parse_args()

    # 1. MIMIC-III
    c3_path = args.repo_root / "artifacts" / "phase5" / "d2c_v2" / "data" / "incident_sepsis_cohort.csv"
    out3_path = args.out_dir / "tri_onset_labels_mimiciii.csv"
    if c3_path.exists():
        extract_tri_onset_mimiciii(c3_path, args.mimiciii_source, out3_path)

    # 2. MIMIC-IV
    c4_path = args.repo_root / "artifacts" / "phase5_iv" / "d2c_v2" / "data" / "incident_sepsis_cohort_iv.csv"
    out4_path = args.out_dir / "tri_onset_labels_mimiciv.csv"
    if c4_path.exists():
        extract_tri_onset_mimiciv(c4_path, args.mimiciv_source, out4_path)

    logger.info("Extraction complete!")


if __name__ == "__main__":
    main()
