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

CANONICAL_HARMONIZED_PATH = Path(
    "/home/alireza/Sepsis-Onset-Timing-Paradox/reports/predictions/canonical_test_predictions_harmonized.parquet"
)

# Standardized Academic Display Names for Evaluated Architectures
MODEL_NAME_MAP = {
    "Proposed Champion (Ensemble+T-Smooth)": "★ Proposed Champion (Ensemble)",
    "BiMamba_SSM": "BiMamba SSM (Proposed)",
    "Random Forest": "Random Forest",
    "Transformer_FairMasked": "Mask-Aware Clinical Transformer",
    "Clinical Transformer": "Clinical Transformer (Baseline Unmasked)",
    "Transformer_Baseline": "Clinical Transformer (Baseline Unmasked)",
    "Tabular GBDT (XGBoost)": "Tabular GBDT (XGBoost)",
    "Tabular GBDT (LightGBM)": "Tabular GBDT (LightGBM)",
    "Logistic Regression (L2)": "Logistic Regression (L2)",
}


def load_harmonized_benchmark_dataset(
    dataset_path: Path = CANONICAL_HARMONIZED_PATH,
) -> pd.DataFrame:
    """Loads harmonized predictions across all architectures alongside tri-onset labels."""
    logger.info("Loading canonical harmonized dataset from %s", dataset_path)
    df = pd.read_parquet(dataset_path)
    logger.info("Loaded dataset: %d windows across %d stays", len(df), df["ICUSTAY_ID"].nunique())
    return df

