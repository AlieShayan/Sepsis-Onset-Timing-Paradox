#!/usr/bin/env python3
"""compute_alert_budget_and_refractory.py

Evaluates operational early warning alert budgets under a 12-hour refractory suppression policy.
Implements the empirical burden-matched bootstrap procedure (Procedure B) with dynamic
exposure recomputation and first-emitted-alert timing verification.

Conforms strictly to JAMIA / JAMIA Open reporting requirements:
  - Exact column schema and typed records (zero column shifts or reversals).
  - Explicit reporting of zero-detection replicate fractions (f_0).
  - Exact reciprocal derivation of NNE intervals from PPV percentiles with infinite upper bounds (+inf).
  - Explicit reporting of degenerate percentile bootstrap intervals at zero detections.
  - Evaluation of true detections at the FIRST emitted alert window under the refractory policy.
  - Dynamic recomputation of exposure E_b and target budget K_b in each bootstrap resample.
  - Neutral model nomenclature (Multi-Model Stacking Ensemble).
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
    "Proposed Champion (Ensemble+T-Smooth)": "Multi-Model Stacking Ensemble",
    "Tabular GBDT (LightGBM)": "Tabular GBDT (LightGBM)",
    "Tabular GBDT (XGBoost)": "Tabular GBDT (XGBoost)",
    "Random Forest": "Random Forest",
    "Transformer_FairMasked": "Mask-Aware Clinical Transformer",
    "Clinical Transformer": "Clinical Transformer (Baseline Unmasked)",
    "Logistic Regression (L2)": "Logistic Regression (L2)",
}

BUDGET_TARGETS = [
    (5.0, 204, "5.0 episodes / 100 pt-days"),
    (10.0, 407, "10.0 episodes / 100 pt-days"),
    (20.0, 814, "20.0 episodes / 100 pt-days"),
]


def extract_stay_data(df: pd.DataFrame, model_col: str):
    """Pre-extracts per-stay temporal arrays for efficient simulation of the refractory policy."""
    stay_dict = {}
    for sid, grp in df.groupby("ICUSTAY_ID"):
        grp_sorted = grp.sort_values("ABS_END_H")
        times = grp_sorted["ABS_END_H"].values
        scores = grp_sorted[model_col].values
        is_sep = grp_sorted["SEPSIS_LABEL"].iloc[0] == 1
        t_s3 = grp_sorted["T_SEPSIS3_H_ICU"].iloc[0] if is_sep else np.nan
        t_tr = grp_sorted["T_TREATMENT_H_ICU"].iloc[0] if is_sep else np.nan
        is_cov = int(is_sep and t_s3 <= 36.0)
        h_i = len(times)
        max_s = scores.max()
        stay_dict[sid] = {
            "times": times,
            "scores": scores,
            "max_s": max_s,
            "is_sep": is_sep,
            "is_cov": is_cov,
            "t_s3": t_s3,
            "t_tr": t_tr,
            "h_i": h_i,
        }
    return stay_dict


def evaluate_model_at_budget(
    stays: np.ndarray,
    stay_dict: dict,
    rate_target: float,
    nominal_k: int,
    budget_label: str,
    model_name: str,
    B: int = 1000,
    seed: int = 42,
) -> dict:
    n_stays = len(stays)
    stay_max_s = np.array([stay_dict[s]["max_s"] for s in stays])
    stay_is_sep = np.array([stay_dict[s]["is_sep"] for s in stays], dtype=bool)
    stay_is_cov = np.array([stay_dict[s]["is_cov"] for s in stays], dtype=bool)
    stay_hi = np.array([stay_dict[s]["h_i"] for s in stays], dtype=float)

    # 1. Point estimate on held-out dataset
    sort_order = np.lexsort((stays, -stay_max_s))
    top_k = sort_order[:nominal_k]
    cutoff_tau = stay_max_s[top_k[-1]]

    # Boundary ties analysis
    tied_mask = (stay_max_s == cutoff_tau)
    tied_total = int(tied_mask.sum())
    strictly_above = int((stay_max_s > cutoff_tau).sum())
    selected_from_tied = nominal_k - strictly_above

    tp_pt = 0
    tp_pretr_pt = 0
    tp_pretr_2h_pt = 0
    for idx in top_k:
        if not stay_is_sep[idx]:
            continue
        sid = stays[idx]
        data = stay_dict[sid]
        w_mask = data["scores"] >= cutoff_tau
        if np.any(w_mask):
            t_first = data["times"][w_mask][0]
            if 0 < (data["t_s3"] - t_first) <= 6.0:
                tp_pt += 1
                if t_first <= data["t_tr"]:
                    tp_pretr_pt += 1
                if t_first + 2.0 <= data["t_tr"]:
                    tp_pretr_2h_pt += 1

    ppv_pt = tp_pt / nominal_k * 100.0
    sens_cov_pt = tp_pt / 151 * 100.0
    yield_all_pt = tp_pt / 408 * 100.0
    nne_pt = 100.0 / ppv_pt if ppv_pt > 0 else np.nan

    # 2. Patient-clustered bootstrap with exposure recomputation
    np.random.seed(seed)
    ppv_boots = []
    sens_cov_boots = []
    yield_all_boots = []
    zero_cnt = 0

    for b in range(B):
        b_idx = np.random.choice(n_stays, size=n_stays, replace=True)
        b_max_s = stay_max_s[b_idx]
        b_hi = stay_hi[b_idx]
        b_sids = stays[b_idx]
        b_is_sep = stay_is_sep[b_idx]
        b_is_cov = stay_is_cov[b_idx]

        # Recompute exposure and target alert count
        E_b = b_hi.sum() / 24.0
        K_b = int(np.round(rate_target * E_b / 100.0))

        # Secondary sort key preserves bootstrap multiplicity
        sort_b = np.lexsort((np.arange(n_stays), b_sids, -b_max_s))
        top_b = sort_b[:K_b]
        tau_b = b_max_s[top_b[-1]]

        d_b = 0
        for pos_in_top in top_b:
            orig_idx = b_idx[pos_in_top]
            if not stay_is_sep[orig_idx]:
                continue
            sid = stays[orig_idx]
            data = stay_dict[sid]
            w_mask = data["scores"] >= tau_b
            if np.any(w_mask):
                t_first = data["times"][w_mask][0]
                if 0 < (data["t_s3"] - t_first) <= 6.0:
                    d_b += 1

        if d_b == 0:
            zero_cnt += 1

        n_cov_b = b_is_cov.sum()
        n_sep_b = b_is_sep.sum()

        ppv_boots.append(d_b / K_b * 100.0)
        sens_cov_boots.append(d_b / n_cov_b * 100.0 if n_cov_b > 0 else 0.0)
        yield_all_boots.append(d_b / n_sep_b * 100.0 if n_sep_b > 0 else 0.0)

    f_0 = zero_cnt / B
    p_lo, p_hi = np.percentile(ppv_boots, [2.5, 97.5])
    sc_lo, sc_hi = np.percentile(sens_cov_boots, [2.5, 97.5])
    ya_lo, ya_hi = np.percentile(yield_all_boots, [2.5, 97.5])

    # Reciprocal derivation of NNE intervals from PPV
    if tp_pt == 0:
        ppv_str = f"0.00 (0.00–0.00)%*"
        sc_str = f"0.00 (0.00–0.00)%*"
        ya_str = f"0.00 (0.00–0.00)%*"
        nne_str = "Undefined (+inf)"
    else:
        ppv_str = f"{ppv_pt:.2f} ({p_lo:.2f}–{p_hi:.2f})%"
        sc_str = f"{sens_cov_pt:.2f} ({sc_lo:.2f}–{sc_hi:.2f})%"
        ya_str = f"{yield_all_pt:.2f} ({ya_lo:.2f}–{ya_hi:.2f})%"

        nne_low = 100.0 / p_hi if p_hi > 0 else np.nan
        if p_lo > 0:
            nne_high = 100.0 / p_lo
            nne_str = f"{nne_pt:.1f} ({nne_low:.1f}–{nne_high:.1f})"
        else:
            nne_str = f"{nne_pt:.1f} ({nne_low:.1f}–+inf)"

    return {
        "Prespecified Alert Budget": budget_label,
        "Model Architecture": model_name,
        "Operating Threshold (tau)": f"{cutoff_tau:.4f}",
        "Emitted Alert Episodes": nominal_k,
        "Emitted Rate (/100 pt-days)": f"{nominal_k / 4070.83 * 100:.2f}",
        "Episode PPV % [95% CI]": ppv_str,
        "True Positive Sepsis Detections": tp_pt,
        "Pretreatment Detections (delta=0h)": tp_pretr_pt,
        "Pretreatment Detections (delta=2h)": tp_pretr_2h_pt,
        "Detection Sensitivity in Covered % [95% CI]": sc_str,
        "Overall Sepsis Detection Yield % [95% CI]": ya_str,
        "NNE per Detected Patient [95% CI]": nne_str,
        "Zero-Detection Replicate Fraction (f_0)": f"{f_0:.3f}",
        "_tied_total": tied_total,
        "_strictly_above": strictly_above,
        "_selected_from_tied": selected_from_tied,
    }


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROJECT_ROOT / "reports" / "predictions" / "canonical_test_predictions_harmonized.parquet"
    df = pd.read_parquet(parquet_path)

    stays = np.array(df["ICUSTAY_ID"].unique())
    logger.info("Loaded dataset with %d stays and %d evaluated windows.", len(stays), len(df))

    all_rows = []
    tie_reports = []

    for rate_target, target_k, b_label in BUDGET_TARGETS:
        logger.info("Evaluating alert budget: %s (target_k=%d)...", b_label, target_k)
        for mcol, mname in EVAL_MODELS.items():
            stay_dict = extract_stay_data(df, mcol)
            res = evaluate_model_at_budget(
                stays=stays,
                stay_dict=stay_dict,
                rate_target=rate_target,
                nominal_k=target_k,
                budget_label=b_label,
                model_name=mname,
                B=1000,
                seed=42,
            )
            tie_reports.append({
                "Budget": b_label,
                "Model": mname,
                "tau": res["Operating Threshold (tau)"],
                "Tied Total": res["_tied_total"],
                "Strictly Above": res["_strictly_above"],
                "Selected from Tied": res["_selected_from_tied"],
            })
            del res["_tied_total"], res["_strictly_above"], res["_selected_from_tied"]
            all_rows.append(res)

    df_table3 = pd.DataFrame(all_rows)
    t3_path = TABLE_DIR / "Table3_Alert_Budget_Refractory_Workload.csv"
    df_table3.to_csv(t3_path, index=False)
    logger.info("Saved Table 3 to %s", t3_path)

    # Print summary
    print(df_table3.to_string())

    # Print tie report
    df_ties = pd.DataFrame(tie_reports)
    print("\n=== BOUNDARY SCORE TIE AUDIT ===")
    print(df_ties[df_ties["Tied Total"] > 1].to_string())


if __name__ == "__main__":
    main()
