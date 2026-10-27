"""src/data/prediction_loader.py

Unified loader for model predictions and tri-onset metadata.
Integrates all 13 evaluated architectures from the canonical benchmark and fair transformer experiments.
"""
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit, expit

logger = logging.getLogger(__name__)

BENCHMARK_PREDS_PATH = Path(
    "/home/alireza/Personalized-Medicine-Early-Sepsis-Prediction/reports/main_benchmark_v2/predictions/test_predictions_20260905_145214.parquet"
)
FAIR_NPZ_PATH = Path(
    "/home/alireza/Personalized-Medicine-Early-Sepsis-Prediction/reports/fair_transformer_experiment/tables/test_prob_fold_raw.npz"
)
FAIR_META_PATH = Path(
    "/home/alireza/Personalized-Medicine-Early-Sepsis-Prediction/reports/fair_transformer_experiment/tables/meta_te.parquet"
)
TRI_ONSET_LABELS_PATH = Path(
    "/home/alireza/Sepsis-Onset-Timing-Paradox/reports/tables/tri_onset_labels_mimiciii.csv"
)

# Standardized Academic Display Names for All 13 Architectures
MODEL_NAME_MAP = {
    "Proposed Champion (Ensemble+T-Smooth)": "★ Proposed Champion (Ensemble)",
    "BiMamba_SSM": "BiMamba SSM (Proposed)",
    "SE-DualGated Mamba-LSTM": "SE-DualGated SSM",
    "Hierarchical Mamba-LSTM": "Hierarchical SSM",
    "Transformer_FairMasked": "Clinical Transformer (Fair Mask-Aware)",
    "Clinical Transformer": "Clinical Transformer (Baseline Unmasked)",
    "Transformer_Baseline": "Clinical Transformer (Baseline Unmasked)",
    "Multimodal Hybrid (BiLSTM+MLP)": "Multimodal Hybrid (BiLSTM+MLP)",
    "Standard Deep Bi-LSTM": "Standard Deep Bi-LSTM",
    "Standard Deep Bi-GRU": "Standard Deep Bi-GRU",
    "Dilated TCN": "Dilated TCN",
    "Tabular GBDT (XGBoost)": "Tabular GBDT (XGBoost)",
    "Tabular GBDT (LightGBM)": "Tabular GBDT (LightGBM)",
    "Random Forest": "Random Forest",
    "Logistic Regression (L2)": "Logistic Regression (L2)",
}


def load_harmonized_benchmark_dataset(
    benchmark_preds_path: Path = BENCHMARK_PREDS_PATH,
    fair_npz_path: Path = FAIR_NPZ_PATH,
    fair_meta_path: Path = FAIR_META_PATH,
    labels_path: Path = TRI_ONSET_LABELS_PATH,
) -> pd.DataFrame:
    """Loads and harmonizes predictions across all 13 architectures alongside tri-onset labels."""
    logger.info("Loading canonical benchmark predictions from %s", benchmark_preds_path)
    df_bench = pd.read_parquet(benchmark_preds_path)

    logger.info("Loading Fair Transformer and BiMamba fold predictions from %s", fair_npz_path)
    data_npz = np.load(fair_npz_path)
    meta_fair = pd.read_parquet(fair_meta_path)

    # Compute fold-averaged calibrated probabilities
    for m_prefix in ["Transformer_Baseline", "Transformer_FairMasked", "BiMamba_SSM"]:
        cal_keys = [f"{m_prefix}_fold_{f}_cal" for f in range(5)]
        probs = np.stack([data_npz[k] for k in cal_keys])
        Z = logit(np.clip(probs, 1e-6, 1.0 - 1e-6))
        meta_fair[m_prefix] = expit(Z.mean(axis=0))

    # Inner merge to align test observation windows
    merged = df_bench.merge(
        meta_fair[["ICUSTAY_ID", "ABS_END_H", "Transformer_Baseline", "Transformer_FairMasked", "BiMamba_SSM"]],
        on=["ICUSTAY_ID", "ABS_END_H"],
        how="inner",
    )
    logger.info("Merged evaluation dataset: %d windows across %d stays", len(merged), merged["ICUSTAY_ID"].nunique())

    # Merge tri-onset labels
    logger.info("Linking tri-onset metadata from %s", labels_path)
    labels = pd.read_csv(labels_path)
    cols_to_merge = [
        "ICUSTAY_ID", "T_SEPSIS3_H_ICU", "T_TREATMENT_H_ICU", "T_SOFA_H_ICU",
        "delta_treatment_gap_h", "delta_suspicion_gap_h", "SUSPICION_PAIR_TIMING"
    ]
    df_final = merged.merge(labels[[c for c in cols_to_merge if c in labels.columns]], on="ICUSTAY_ID", how="left")

    return df_final
