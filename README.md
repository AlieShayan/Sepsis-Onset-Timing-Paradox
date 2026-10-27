# The Onset Timing Paradox: How Diagnostic Ambiguity Deconstructs Lead-Time Dynamics and Operational Utility in Sepsis Early Warning

[![Target Venue](https://img.shields.io/badge/Target_Journal-JAMIA_Open_%2F_CMPB-blue.svg)](https://academic.oup.com/jamiaopen)
[![Zero Retraining](https://img.shields.io/badge/Zero_Retraining-Audit_Grade_Log-green.svg)](#)
[![Architectures](https://img.shields.io/badge/Evaluated_Models-13_Architectures_%2B_Fair_Transformer-orange.svg)](#)

## 📌 Overview

This repository contains the official implementation, query pipelines, evaluation suites, and figures for the empirical study:
> **"The Onset Timing Paradox: How Diagnostic Ambiguity Deconstructs Lead-Time Dynamics and Operational Utility in Sepsis Early Warning"**

### Abstract Summary
Deep learning systems for early sepsis detection in intensive care units (ICUs) routinely claim warning lead times of 4 to 12 hours prior to sepsis onset. However, in clinical practice, retrospective consensus definitions ($T_{\text{sepsis3}}$: co-occurrence of blood cultures, systemic antibiotics, and organ failure) lag significantly behind the bedside clinician's initial diagnostic suspicion and empiric intravenous antibiotic intervention ($T_{\text{treatment}}$).

This project investigates the **Delayed Chart Mirroring** phenomenon across **13 machine learning and deep learning architectures** (including State-Space Models such as BiMamba, the Fair Mask-Aware Clinical Transformer, Deep BiLSTM/BiGRU, TCN, and Gradient Boosted Trees), evaluating how model discrimination, sensitivity profiles, and lead-time decay behave under three distinct definitions of onset timing.

---

## 🔬 Tri-Onset Operational Definitions

1. **$T_{\text{sepsis3}}$ (Canonical Retrospective Benchmark):** The international Sepsis-3 consensus criterion (blood culture + antibiotic within $[-24\text{h}, +72\text{h}]$ window coupled with acute organ dysfunction $\Delta\text{SOFA} \ge 2$).
2. **$T_{\text{treatment}}$ (Clinical Intervention Onset):** The exact hour of the first intravenous (IV) systemic broad-spectrum antibiotic administration.
3. **$T_{\text{sofa}}$ (Pure Physiological Shock Onset):** The earliest timestamp meeting the acute organ dysfunction criterion ($\Delta\text{SOFA} \ge 2$) irrespective of blood culture ordering.

---

## 🏗️ Evaluated Architectural Suite (13 Models)
Evaluated directly from the unified benchmark suite (`notebooks/02_model_training_and_internal_validation_v2.ipynb`):
- **State-Space Models (SSM):** BiMamba SSM, SE-DualGated SSM, Hierarchical SSM-LSTM
- **Fair Mask-Aware Deep Sequence Baselines:** Clinical Transformer (Fair Mask-Aware & Pooled), Clinical Transformer (Baseline Unmasked), Deep BiLSTM, Deep BiGRU, Dilated TCN, Multimodal Hybrid (BiLSTM+MLP)
- **Tabular & Classic Baselines:** Tabular XGBoost, Tabular LightGBM, Random Forest, Logistic Regression
- **Meta-Learners:** Pareto-Convex Ensemble Blend (Champion)

---

## 📁 Repository Structure

```
.
├── manuscript/                    # Manuscript draft (Markdown / LaTeX)
├── notebooks/                     # Exploratory and audit notebooks
├── reports/
│   ├── figures/                   # Publication-ready figures (300 DPI PNG & PDF)
│   └── tables/                    # Computed summary tables (CSV & TeX)
├── scripts/
│   ├── extract_tri_onset_labels.py
│   └── figures/
│       ├── generate_onset_ambiguity_distribution.py   # Figure 1
│       ├── generate_onset_leadtime_decay.py            # Figure 2
│       └── generate_rank_invariance_plot.py            # Figure 3
└── src/
    ├── data/                      # Data loaders and linkage utilities
    ├── metrics/                   # Lead-time decay & rank correlation metrics
    └── utils/                     # Path management & configuration
```
