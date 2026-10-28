#!/usr/bin/env python3
"""compute_interval_censored_bounds.py

Constructs Table 4: Two-Axis Clinical Subgroup & Timestamp Uncertainty Matrix.
Strictly implements the executable opportunity definition requested by reviewers:
  A_i = { t in T_i : 0 < T_S3,i - t <= H }
  O_i(delta) = 1{ exists t in A_i : t + delta <= T_treatment,i }

Orthogonally categorizes by:
  - Axis 1: Sepsis-3 onset timing relative to ICU admission (Early <=24h, Intermediate 24-48h, Late >48h)
  - Axis 2: Measurement timestamp precision (MIMIC-IV hourly eMAR vs. MIMIC-III daily prescription proxy)

Reports within-subgroup bounds, within-covered bounds, and contributions to overall cohort bounds.
Conforms strictly to JAMIA / JAMIA Open reporting requirements.
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


def compute_mimiciv_test_table() -> list[dict]:
    df_test = pd.read_parquet(PROJECT_ROOT / "reports" / "predictions" / "canonical_test_predictions_harmonized.parquet")
    septic_test = df_test[df_test["SEPSIS_LABEL"] == 1]

    results = []
    for sid, group in septic_test.groupby("ICUSTAY_ID"):
        t_s3 = group["T_SEPSIS3_H_ICU"].iloc[0]
        t_tr = group["T_TREATMENT_H_ICU"].iloc[0]
        t_dec = group["ABS_END_H"].values
        A_i = t_dec[(t_s3 - t_dec > 0) & (t_s3 - t_dec <= 6.0)]
        v_i = int(len(A_i) > 0)
        o_0 = int(np.any(A_i <= t_tr)) if v_i else 0
        o_2 = int(np.any(A_i + 2.0 <= t_tr)) if v_i else 0
        results.append({
            "ICUSTAY_ID": sid,
            "T_S3": t_s3,
            "T_TR": t_tr,
            "V_i": v_i,
            "O_0": o_0,
            "O_2": o_2,
        })
    df_res = pd.DataFrame(results)
    df_res["onset_group"] = pd.cut(
        df_res["T_S3"],
        bins=[0, 24, 48, np.inf],
        labels=["Early Incident (<=24h)", "Intermediate Incident (24-48h)", "Late Incident (>48h)"],
    )

    n_tot = len(df_res)
    rows = []
    for grp_name in ["Early Incident (<=24h)", "Intermediate Incident (24-48h)", "Late Incident (>48h)"]:
        grp = df_res[df_res["onset_group"] == grp_name]
        n_sub = len(grp)
        v1 = grp["V_i"].sum()
        o0 = grp["O_0"].sum()
        inelig = n_sub - o0
        rows.append({
            "Cohort & Timestamp Source": "MIMIC-IV Prospective-Replay Test Cohort (eMAR Confirmed)",
            "Sepsis Onset Category": grp_name,
            "Subgroup Denominator (N)": n_sub,
            "Surveillance Covered Stays (V=1)": f"{v1} ({v1/n_sub*100:.1f}%)",
            "Definitely Eligible Count": o0,
            "Indeterminate Count": 0,
            "Definitely Ineligible Count": inelig,
            "Within-Subgroup Opportunity Bounds": f"[{o0/n_sub*100:.1f}%, {o0/n_sub*100:.1f}%]",
            "Within-Covered Opportunity Bounds": f"[{o0/v1*100:.1f}%, {o0/v1*100:.1f}%]" if v1 > 0 else "N/A (No covered stays)",
            "Contribution to Overall Cohort Bound": f"{o0} / {n_tot} ({o0/n_tot*100:.1f}%)",
        })

    # Overall MIMIC-IV row
    v1_tot = df_res["V_i"].sum()
    o0_tot = df_res["O_0"].sum()
    rows.append({
        "Cohort & Timestamp Source": "MIMIC-IV Prospective-Replay Test Cohort (eMAR Confirmed)",
        "Sepsis Onset Category": "All Incident Sepsis-3 Cases",
        "Subgroup Denominator (N)": n_tot,
        "Surveillance Covered Stays (V=1)": f"{v1_tot} ({v1_tot/n_tot*100:.1f}%)",
        "Definitely Eligible Count": o0_tot,
        "Indeterminate Count": 0,
        "Definitely Ineligible Count": n_tot - o0_tot,
        "Within-Subgroup Opportunity Bounds": f"[{o0_tot/n_tot*100:.1f}%, {o0_tot/n_tot*100:.1f}%]",
        "Within-Covered Opportunity Bounds": f"[{o0_tot/v1_tot*100:.1f}%, {o0_tot/v1_tot*100:.1f}%]",
        "Contribution to Overall Cohort Bound": f"{o0_tot} / {n_tot} ({o0_tot/n_tot*100:.1f}%)",
    })
    return rows


def compute_mimiciii_table() -> list[dict]:
    df3 = pd.read_csv(TABLE_DIR / "tri_onset_labels_mimiciii.csv")
    intime = pd.to_datetime(df3["INTIME"])
    treat_date = pd.to_datetime(df3["T_TREATMENT"].str.slice(0, 10))

    # [a_i, b_i) interval relative to INTIME in hours
    a_i = (treat_date - intime).dt.total_seconds() / 3600.0
    b_i = a_i + 24.0
    t_s3 = df3["T_SEPSIS3_H_ICU"].values

    results = []
    t_dec_all = np.arange(19, 31, 1.0)
    for idx in range(len(df3)):
        s3 = t_s3[idx]
        ai = a_i.iloc[idx]
        bi = b_i.iloc[idx]

        A_i = t_dec_all[(s3 - t_dec_all > 0) & (s3 - t_dec_all <= 6.0)]
        v_i = int(len(A_i) > 0)

        if not v_i:
            results.append({
                "ICUSTAY_ID": df3["ICUSTAY_ID"].iloc[idx],
                "T_S3": s3,
                "V_i": 0,
                "def_eligible": 0,
                "def_ineligible": 1,
                "indeterminate": 0,
            })
        else:
            def_elig = int(np.any(A_i <= ai))
            def_inelig = int(np.all(A_i > bi))
            indet = int(not def_elig and not def_inelig)
            results.append({
                "ICUSTAY_ID": df3["ICUSTAY_ID"].iloc[idx],
                "T_S3": s3,
                "V_i": 1,
                "def_eligible": def_elig,
                "def_ineligible": def_inelig,
                "indeterminate": indet,
            })

    df_res = pd.DataFrame(results)
    df_res["onset_group"] = pd.cut(
        df_res["T_S3"],
        bins=[0, 24, 48, np.inf],
        labels=["Early Incident (<=24h)", "Intermediate Incident (24-48h)", "Late Incident (>48h)"],
    )

    n_tot = len(df_res)
    rows = []
    for grp_name in ["Early Incident (<=24h)", "Intermediate Incident (24-48h)", "Late Incident (>48h)"]:
        grp = df_res[df_res["onset_group"] == grp_name]
        n_sub = len(grp)
        v1 = grp["V_i"].sum()
        d_el = grp["def_eligible"].sum()
        ind = grp["indeterminate"].sum()
        d_in = grp["def_ineligible"].sum()

        w_low = d_el / n_sub * 100
        w_high = (d_el + ind) / n_sub * 100
        cov_low = (d_el / v1 * 100) if v1 > 0 else 0
        cov_high = ((d_el + ind) / v1 * 100) if v1 > 0 else 0

        rows.append({
            "Cohort & Timestamp Source": "MIMIC-III Broad Incident Sepsis (Prescription Proxy)",
            "Sepsis Onset Category": grp_name,
            "Subgroup Denominator (N)": n_sub,
            "Surveillance Covered Stays (V=1)": f"{v1} ({v1/n_sub*100:.1f}%)",
            "Definitely Eligible Count": d_el,
            "Indeterminate Count": ind,
            "Definitely Ineligible Count": d_in,
            "Within-Subgroup Opportunity Bounds": f"[{w_low:.1f}%, {w_high:.1f}%]",
            "Within-Covered Opportunity Bounds": f"[{cov_low:.1f}%, {cov_high:.1f}%]" if v1 > 0 else "N/A (No covered stays)",
            "Contribution to Overall Cohort Bound": f"[{d_el}, {d_el+ind}] / {n_tot} ([{d_el/n_tot*100:.2f}%, {(d_el+ind)/n_tot*100:.2f}%])",
        })

    # Overall MIMIC-III row
    v1_tot = df_res["V_i"].sum()
    d_el_tot = df_res["def_eligible"].sum()
    ind_tot = df_res["indeterminate"].sum()
    d_in_tot = df_res["def_ineligible"].sum()
    rows.append({
        "Cohort & Timestamp Source": "MIMIC-III Broad Incident Sepsis (Prescription Proxy)",
        "Sepsis Onset Category": "All Incident Sepsis-3 Cases",
        "Subgroup Denominator (N)": n_tot,
        "Surveillance Covered Stays (V=1)": f"{v1_tot} ({v1_tot/n_tot*100:.1f}%)",
        "Definitely Eligible Count": d_el_tot,
        "Indeterminate Count": ind_tot,
        "Definitely Ineligible Count": d_in_tot,
        "Within-Subgroup Opportunity Bounds": f"[{d_el_tot/n_tot*100:.2f}%, {(d_el_tot+ind_tot)/n_tot*100:.2f}%]",
        "Within-Covered Opportunity Bounds": f"[{d_el_tot/v1_tot*100:.2f}%, {(d_el_tot+ind_tot)/v1_tot*100:.2f}%]",
        "Contribution to Overall Cohort Bound": f"[{d_el_tot}, {d_el_tot+ind_tot}] / {n_tot} ([{d_el_tot/n_tot*100:.2f}%, {(d_el_tot+ind_tot)/n_tot*100:.2f}%])",
    })
    return rows


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows_iv = compute_mimiciv_test_table()
    rows_iii = compute_mimiciii_table()

    all_rows = rows_iv + rows_iii
    df_t4 = pd.DataFrame(all_rows)
    out_csv = TABLE_DIR / "Table4_Two_Axis_Subgroup_Uncertainty_Matrix.csv"
    df_t4.to_csv(out_csv, index=False)
    logger.info("Saved Table 4 to %s", out_csv)
    print(df_t4.to_string())


if __name__ == "__main__":
    main()
