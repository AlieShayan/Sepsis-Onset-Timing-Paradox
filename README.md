# When Does Sepsis Prediction Precede Treatment? Retrospective Onset Mechanics, EHR Timelines, and Bedside Alert Workload in ICU Benchmarks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Audit_Grade-brightgreen.svg)](#reproducibility--verification)
[![MIMIC-III v1.4](https://img.shields.io/badge/EHR_Database-MIMIC--III_v1.4-lightgrey.svg)](https://physionet.org/content/mimiciii/1.4/)

This repository contains the official analytical audit suite, evaluation pipelines, data ledgers, and publication figures for the investigation:

> **"When Does Sepsis Prediction Precede Treatment? Retrospective Onset Mechanics, EHR Timelines, and Bedside Alert Workload in ICU Benchmarks"**  
> *Alireza Shayan, BSc; Somayeh Afrasiabi, PhD*  
> Department of Computer Science and Engineering & IT, Shiraz University, Shiraz, Iran.

---

## Executive Summary

Machine learning early-warning models for sepsis in intensive care units (ICUs) routinely report high discriminative performance (AUROC ≥ 0.80) ahead of consensus onset. However, when deployed at the bedside, systems frequently encounter alert fatigue, low positive predictive value (PPV), and skepticism regarding whether warnings precede meaningful clinical action.

This investigation provides a rigorous, code-level forensic audit of sepsis early-warning evaluation across **four operational cascade stages ($S \to V \to O \to D$)** using the schema-documented MIMIC-III (v1.4) database ($N = 8,254$ adult ICU stays; 97,700 evaluated hourly surveillance windows):

```
  Total Incident Sepsis (S)          N = 408 stays (100.0%)
            │
            ▼ [Surveillance Eligibility Window: Hours 19–30]
  Protocol Coverage (V)              N = 151 stays (37.0% of incident; 63.0% onset > 36h unreached)
            │
            ▼ [Clinical Process Anchor: Tsusp vs. TRx,start]
  Opportunity Eligibility (O)        Under Tsusp: 150/151 (99.3%) governed by 24h Lookback Floor
                                     Under TRx,start: 0/151 (0.0%) under midnight date coarsening
            │
            ▼ [Operational Emitted Alert Budget: 10 episodes / 100 pt-days + 12h Lockout]
  Timely Model Alert Detection (D)   N = 7 stays (1.72% whole-population yield; NNE = 58.1)
```

### Key Methodological & Mathematical Findings:

1. **The Retrospective Lookback Flooring Identity:**  
   Under consensus Sepsis-3 criteria (Seymour et al., *JAMA* 2016), organ dysfunction is evaluated across $[T_{\text{susp}} - 24\text{h}, T_{\text{susp}} + 12\text{h}]$ relative to suspected infection. In **67.2% of incident stays** (274 of 408) and **66.2% of surveillance-covered stays** (100 of 151), the qualifying $\Delta\text{SOFA} \ge 2$ is first identified at the earliest evaluated search bin ($T_{\text{susp}} - 24\text{h}$). We mathematically prove the resulting operational lead gap:

   $$G_i = T_{\text{susp}, i} - T_{\text{S3}, i} = 24\text{h} + h \cdot \text{frac}(x_i) < 25\text{h}$$

   This structural flooring algebraically guarantees pre-suspicion opportunity in 99.3% of covered stays, reflecting an artifact of the 24-hour lookback convention rather than empirical biological early warning.

2. **The Midnight Date-Coarsening Collapse:**  
   Prescription start dates standardly lack time of day (`prescriptions.startdate`). When assigned to midnight ($a_i = \text{00:00:00}$), all 408 sepsis stays exhibit documented antimicrobial orders prior to hour 3.83 of admission ($T_{\text{Rx,start}} \le 3.83\text{h}$, median $-12.6\text{h}$). Consequently, all surveillance decisions ($t \ge 19\text{h}$) occur post-prescription, collapsing evaluable pre-treatment cases to strictly **zero** ($N_+^{\text{Rx}} = 0$). Under interval-censored uncertainty $[a_i, b_i]$, pre-treatment opportunity is bounded within **[0.0%, 31.1%]** in surveillance-covered stays (and **[0.0%, 11.5%]** cohort-wide), compared to 99.3% under suspicion.

3. **Bedside Workload and Detection Yield:**  
   Across state-space models (BiMamba SSM), tree ensembles (LightGBM, XGBoost, Random Forest), deep transformers, and stacking meta-learners under natural prevalence ($\pi = 0.694\%$) and 12-hour refractory alert suppression:
   - At an operational alert budget of 10 episodes / 100 patient-days ($K = 407$), top models detect **7 distinct patients** (PPV = 1.72%).
   - Bedside evaluation workload requires **58.1 evaluations per detected patient** (NNE = 58.1), demonstrating that high offline AUROC does not circumvent clinical alarm burden.

---

## Repository Structure

```
.
├── manuscript/
│   ├── latex/
│   │   ├── main.tex                  # Primary two-column camera-ready LaTeX source (strictly 8 pages)
│   │   ├── main.pdf                  # Compiled two-column manuscript
│   │   └── figures/                  # Publication vector PDF and PNG figure assets
│   ├── arxiv/                        # Standalone arXiv preprint bundle (with public repo links)
│   └── supplementary_material.md     # Extended clinical methods and variable dictionaries
├── reports/
│   ├── figures/                      # High-resolution (300 DPI) publication figures
│   │   ├── Figure1_Clocks_and_Opportunity_Waterfall.{pdf,png}
│   │   ├── Figure2_Operational_Alert_Budget_Frontier.{pdf,png}
│   │   ├── Figure3_Three_Stage_Decomposition.{pdf,png}
│   │   └── Figure1_Vertical_Prototype.{pdf,png}
│   └── tables/                       # Ground-truth summary tables & ledgers
│       ├── Table1_Cohort_Progression_Ledger.csv
│       ├── Table2_Cross_Anchor_Discrimination.csv
│       ├── Table3_Alert_Budget_Refractory_Workload.csv
│       └── Table4_Two_Axis_Opportunity_Matrix.csv
├── scripts/
│   ├── figures/
│   │   ├── generate_jamia_figures.py # Regenerates Figures 1, 2, and 3 from summary tables
│   │   └── prototype_vertical_figure1.py # Regenerates vertical timeline prototype
│   └── metrics/
│       ├── compute_alert_budget_and_refractory.py # Table 3 workload & NNE calculations
│       ├── compute_interval_censored_bounds.py    # Table 4 interval bounds [a_i, b_i]
│       ├── compute_patient_clustered_bootstrap.py # Patient-clustered 95% CIs
│       ├── compute_pretreatment_sepsis_benchmark.py # Pre-treatment comparative pipeline
│       └── generate_table1_cohort_progression_ledger.py # Tracing all 8,254 stays
└── pyproject.toml                    # Environment specifications
```

---

## Reproducibility & Verification

To facilitate open science and independent scientific audit, reproducibility is organized into two distinct tiers:

### Tier 1: Instant Verification (No MIMIC-III Access Required)
The core analytical findings, statistical metrics, alert-budget frontiers, and publication figures can be regenerated in seconds directly from the packaged ground-truth ledgers in `reports/tables/`. This tier runs out-of-the-box on standard Python environments without requiring credentialed access or downloading the multi-gigabyte raw EHR database:

```bash
# 1. Clone the repository
git clone https://github.com/AlieShayan/Sepsis-Onset-Timing-Paradox.git
cd Sepsis-Onset-Timing-Paradox

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib pypdf

# 3. Regenerate all publication figures (Figures 1, 2, and 3):
python scripts/figures/generate_jamia_figures.py
```
*Generated high-resolution vector PDFs and PNGs will be written directly to `reports/figures/`.*

### Tier 2: Raw Clinical Telemetry & Database Governance
In compliance with international patient privacy regulations (HIPAA Safe Harbor) and the PhysioNet Credentialed Data Use Agreement (DUA), raw electronic health records from the **MIMIC-III v1.4** database cannot be redistributed within public source repositories.

Researchers seeking to re-execute patient-level feature extraction or raw model inference from the primary database should:
1. Complete human subjects training (CITI Data or Specimens Only Research).
2. Request credentialed access to MIMIC-III via [PhysioNet](https://physionet.org/content/mimiciii/1.4/).
3. Execute the data preparation and cohort extraction pipelines pointing to their authenticated local database instance.

---

## Summary of Evaluated Architectures

The benchmark evaluates predictive scores from eight representative model architectures under identical observation streams:
* **Selective State Space Models:** BiMamba SSM (Gu & Dao, 2024; Liang et al., 2024)
* **Ensemble Meta-Learners:** Multi-Model Stacking Ensemble
* **Gradient Boosted Decision Trees (GBDT):** Tabular LightGBM (Ke et al., 2017), Tabular XGBoost (Chen & Guestrin, 2016)
* **Classic Machine Learning:** Random Forest (Breiman, 2001)
* **Deep Sequence Baselines:** Mask-Aware Clinical Transformer, Clinical Transformer (Baseline), Deep BiLSTM

---

## Citation

If you use this codebase, audit framework, or findings in your research, please cite:

```bibtex
@article{shayan2026when,
  title={When Does Sepsis Prediction Precede Treatment? Retrospective Onset Mechanics, EHR Timelines, and Bedside Alert Workload in ICU Benchmarks},
  author={Shayan, Alireza and Afrasiabi, Somayeh},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License & Data Access

* **Codebase License:** Distributed under the terms of the [MIT License](LICENSE).
* **Clinical Data:** Clinical records are sourced from the de-identified **MIMIC-III v1.4** database, hosted by PhysioNet under the PhysioNet Credentialed Data Use Agreement. Users must complete human subjects training (CITI Program) and obtain institutional approval through PhysioNet to access the underlying patient records.
