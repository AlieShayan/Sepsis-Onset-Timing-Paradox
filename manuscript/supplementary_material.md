# Supplementary Material

**Title:** Surveillance Coverage, Treatment Precedence, and Workload Constraints in Machine Learning Early Warning for Sepsis: An Operational Evaluation  
**Target Journal:** *Journal of the American Medical Informatics Association (JAMIA)* / *JAMIA Open*  
**Track:** Research and Applications  
**Analytical Release Version:** `1.0.0-freeze` (Git Commit `b0932ca`)  

---

## Table S1: Complete Cohort Inclusion, Exclusion, and Progression Ledger

| Progression Stage / Cohort Partition | ICU Stays ($N$) | Evaluated Decision Windows | Surveillance Volume | Proportion / Window Prevalence | Clinical Definition & Measurement Convention |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Base Prospective-Replay Test Cohort** | 8,254 | 97,700 (out of 99,048 scheduled) | 4,070.8 evaluation-bin patient-days (97,700 patient-hours) | 100.0% of cohort | Adult first ICU admissions (approx. 2017–2019) conditioned on ICU stay $\ge 30\text{h}$. Evaluated at up to 12 scheduled hourly decision bins $[t-1, t]$ at $t \in \{19\dots30\}$. 1,348 windows right-censored (623 post-sepsis onset, 725 monitoring cessation). |
| **2. Non-Sepsis Control ICU Stays** | 7,846 | 93,427 outcome-negative | 3,892.8 patient-days (93,427 patient-hours) | 95.06% of stays | Patients without consensus Sepsis-3 during hospitalization. All windows serve as outcome-negative surveillance (180 stays censored prior to hour 30 due to routine observation cessation / monitoring termination, accounting for 725 windows). |
| **3. Consensus Sepsis-3 Stays ($S=1$)** | 408 | 4,273 windows (678 positive, 3,595 negative) | 178.0 patient-days (4,273 patient-hours) | 4.94% of stays | Patients meeting Sepsis-3 consensus criteria (blood culture + antibiotic + acute $\Delta\text{SOFA} \ge 2$). Median onset: hour 49.0 (IQR: 30.0h–77.0h; range: 20h–369h). 110 stays censored at onset prior to hour 30 (623 windows). |
| &emsp;**3a. Late-Onset Sepsis (Onset $>36\text{h}$)** | 257 | 3,084 outcome-negative windows | 128.5 patient-days (3,084 patient-hours) | 63.0% of septic stays | Onset occurs after hour 36. 6-hour lookback $[T_{\text{S3}} - 6, T_{\text{S3}}]$ does not intersect surveillance window $[19, 30]$. All windows are outcome-negative baseline surveillance ($V_i = 0$). |
| &emsp;**3b. Surveillance-Covered Sepsis ($V=1$)** | 151 | 1,189 windows (678 positive, 511 negative) | 49.5 patient-days (1,189 patient-hours) | 37.0% of septic stays | Onset occurs between ICU hours 20 and 36, such that $[T_{\text{S3}} - 6\text{h}, T_{\text{S3}}]$ intersects surveillance interval $[19, 30]$ ($V_i = 1$). |
| **4. Canonical Sepsis-3 Risk Set ($H=6\text{h}$)** | 8,254 (151 horizon-pos) | 97,700 (678 pos / 97,022 neg) | 4,070.8 patient-days | 0.694% window prevalence | Full unconstrained risk set. Target windows have end-time within 6 hours before Sepsis-3 onset ($0 < T_{\text{S3}} - t \le 6\text{h}$). |
| **5. Pretreatment Sepsis Risk Set $\mathcal{R}_0$ ($\delta=0\text{h}$)** | 5,967 (150 horizon-pos) | 70,192 (674 pos / 69,518 neg) | 2,924.7 patient-days | 0.960% window prevalence | Pretreatment risk set censoring all post-treatment windows ($w_{\text{end}} \le T_{\text{treatment}}$) for both cases and controls. 150 out of 151 covered patients (99.3%) maintain $\ge 1$ pretreatment target window. |
| **6. Pretreatment Sepsis Risk Set $\mathcal{R}_2$ ($\delta=2\text{h}$)** | 5,967 (149 horizon-pos) | 70,129 (663 pos / 69,466 neg) | 2,922.0 patient-days | 0.945% window prevalence | Pretreatment risk set requiring a 2-hour clinical buffer ($w_{\text{end}} \le T_{\text{treatment}} - 2\text{h}$) before antimicrobial therapy. 149 out of 151 covered patients (98.7%) maintain $\ge 1$ actionable target window. |

---

## Table S2: Complete Model Discrimination and Pairwise Differences

*Held-out point estimates ($B=0$) and $B=1,000$ patient-clustered bootstrap percentile 95% confidence intervals.*

| Model Architecture | Canonical Sepsis-3 ($H=6\text{h}$)<br>Prev: 0.694%<br>AUROC [95% CI] | Canonical Sepsis-3 ($H=6\text{h}$)<br>Prev: 0.694%<br>AUPRC % [95% CI] | Pretreatment Sepsis ($\mathcal{R}_0$, $\delta=0\text{h}$)<br>Prev: 0.960%<br>AUROC [95% CI] | Pretreatment Sepsis ($\mathcal{R}_0$, $\delta=0\text{h}$)<br>Prev: 0.960%<br>AUPRC % [95% CI] | Pretreatment ($\mathcal{R}_2$, $\delta=2\text{h}$)<br>Prev: 0.945%<br>AUROC [95% CI] | Pretreatment ($\mathcal{R}_2$, $\delta=2\text{h}$)<br>Prev: 0.945%<br>AUPRC % [95% CI] | Paired $\Delta\text{AUROC}$ ($\mathcal{R}_0 - \text{Canonical}$)<br>[95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BiMamba SSM (Proposed)** | 0.6326 (0.5872–0.6733) | 1.209% (0.892%–1.769%) | 0.6433 (0.5998–0.6835) | 1.656% (1.235%–2.329%) | 0.6447 (0.6002–0.6860) | 1.647% (1.220%–2.331%) | +0.0107 (+0.0060 to +0.0175) |
| **Multi-Model Stacking Ensemble** | 0.6284 (0.5829–0.6699) | 1.206% (0.882%–1.748%) | 0.6351 (0.5897–0.6771) | 1.621% (1.207%–2.263%) | 0.6370 (0.5913–0.6799) | 1.613% (1.198%–2.255%) | +0.0067 (+0.0021 to +0.0137) |
| **Random Forest** | 0.6108 (0.5643–0.6535) | 1.080% (0.821%–1.434%) | 0.6143 (0.5684–0.6575) | 1.470% (1.138%–1.920%) | 0.6173 (0.5712–0.6612) | 1.461% (1.120%–1.916%) | +0.0034 (-0.0008 to +0.0101) |
| **Mask-Aware Clinical Transformer** | 0.5942 (0.5467–0.6410) | 1.013% (0.766%–1.303%) | 0.5961 (0.5490–0.6422) | 1.382% (1.048%–1.760%) | 0.5954 (0.5480–0.6426) | 1.365% (1.032%–1.752%) | +0.0020 (-0.0024 to +0.0083) |
| **Tabular GBDT (XGBoost)** | 0.5670 (0.5208–0.6102) | 0.977% (0.739%–1.357%) | 0.5643 (0.5186–0.6082) | 1.320% (1.008%–1.817%) | 0.5652 (0.5192–0.6099) | 1.313% (0.998%–1.811%) | -0.0027 (-0.0070 to +0.0038) |
| **Tabular GBDT (LightGBM)** | 0.5613 (0.5161–0.6026) | 0.893% (0.694%–1.195%) | 0.5587 (0.5139–0.6010) | 1.222% (0.957%–1.629%) | 0.5598 (0.5145–0.6033) | 1.214% (0.949%–1.630%) | -0.0026 (-0.0070 to +0.0038) |
| **Logistic Regression (L2)** | 0.5593 (0.5097–0.6048) | 0.939% (0.716%–1.218%) | 0.5654 (0.5157–0.6107) | 1.337% (1.012%–1.740%) | 0.5657 (0.5150–0.6127) | 1.325% (1.002%–1.731%) | +0.0061 (+0.0023 to +0.0116) |
| **Clinical Transformer (Baseline)** | 0.5239 (0.4808–0.5653) | 0.808% (0.609%–1.381%) | 0.5296 (0.4851–0.5712) | 1.147% (0.857%–1.904%) | 0.5286 (0.4834–0.5705) | 1.131% (0.845%–1.894%) | +0.0057 (+0.0019 to +0.0110) |

---

## Table S3A: Extended Operational Alert-Budget Workload Under 12h Refractory Policy

*Evaluated under Procedure B (Empirical Burden-Matched Bootstrap, $B=1,000$). At each resample, exposure $E_b = \frac{1}{24}\sum h_i$ is dynamically recomputed to adjust the target alert count $K_b = \operatorname{round}(q \cdot E_b / 100)$. Secondary key sorting `[score DESC, ICUSTAY_ID ASC, copy_idx ASC]` preserves bootstrap multiplicity. Detections evaluated at first emitted alert $t_{\mathrm{first}} = \min\{t : \text{score}_t \ge \tau\}$.*

| Prespecified Alert Budget | Model Architecture | Operating Threshold ($\tau$) | Emitted Alert Episodes | Emitted Rate (/100 pt-days) | Episode PPV % [95% CI] | True Positive Sepsis Detections | Pretreatment Detections ($\delta=0\text{h}$) | Pretreatment Detections ($\delta=2\text{h}$) | Detection Sensitivity in Covered % [95% CI] | Overall Sepsis Detection Yield % [95% CI] | NNE per Detected Patient [95% CI] | Zero-Detection Fraction ($f_0$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5.0 episodes / 100 pt-days** | **Multi-Model Stacking Ensemble** | 0.0145 | 204 | 5.01 | 2.94 (0.98–5.88)% | 6 | 6 | 6 | 3.97 (1.31–7.65)% | 1.47 (0.49–2.86)% | 34.0 (17.0–102.0) | 0.002 |
| | **Tabular GBDT (LightGBM)** | 0.0117 | 204 | 5.01 | 2.94 (0.98–5.39)% | 6 | 6 | 6 | 3.97 (1.26–7.10)% | 1.47 (0.47–2.61)% | 34.0 (18.5–102.0) | 0.001 |
| | **BiMamba SSM (Proposed)** | 0.0187 | 204 | 5.01 | 1.96 (0.49–4.45)% | 4 | 4 | 4 | 2.65 (0.65–6.20)% | 0.98 (0.24–2.31)% | 51.0 (22.5–204.0) | 0.015 |
| | **Tabular GBDT (XGBoost)** | 0.0122 | 204 | 5.01 | 1.96 (0.49–3.94)% | 4 | 4 | 4 | 2.65 (0.63–5.63)% | 0.98 (0.23–2.09)% | 51.0 (25.4–204.0) | 0.015 |
| | **Random Forest** | 0.0177 | 204 | 5.01 | 1.47 (0.00–3.45)% | 3 | 3 | 3 | 1.99 (0.00–4.49)% | 0.74 (0.00–1.70)% | 68.0 (29.0–+inf) | 0.041 |
| | **Clinical Transformer (Baseline)** | 0.0101 | 204 | 5.01 | 0.98 (0.00–2.46)% | 2 | 2 | 2 | 1.32 (0.00–3.50)% | 0.49 (0.00–1.27)% | 102.0 (40.6–+inf) | 0.130 |
| | **Logistic Regression (L2)** | 0.0103 | 204 | 5.01 | 0.98 (0.00–2.94)% | 2 | 2 | 2 | 1.32 (0.00–3.85)% | 0.49 (0.00–1.45)% | 102.0 (34.0–+inf) | 0.138 |
| | **Mask-Aware Transformer** | 0.0120 | 204 | 5.01 | 0.00 (0.00–0.00)%* | 0 | 0 | 0 | 0.00 (0.00–0.00)%* | 0.00 (0.00–0.00)%* | Undefined (+inf) | 0.989 |
| **10.0 episodes / 100 pt-days** | **BiMamba SSM (Proposed)** | 0.0165 | 407 | 10.00 | 1.72 (0.49–2.95)% | 7 | 7 | 7 | 4.64 (1.33–8.16)% | 1.72 (0.49–2.93)% | 58.1 (33.9–203.5) | 0.001 |
| | **Tabular GBDT (LightGBM)** | 0.0110 | 407 | 10.00 | 1.72 (0.49–2.95)% | 7 | 7 | 7 | 4.64 (1.38–7.74)% | 1.72 (0.49–2.89)% | 58.1 (33.9–203.5) | 0.000 |
| | **Multi-Model Stacking Ensemble** | 0.0131 | 407 | 10.00 | 1.47 (0.25–2.46)% | 6 | 6 | 6 | 3.97 (0.70–6.98)% | 1.47 (0.26–2.54)% | 67.8 (40.7–407.0) | 0.004 |
| | **Tabular GBDT (XGBoost)** | 0.0114 | 407 | 10.00 | 1.47 (0.49–2.94)% | 6 | 6 | 6 | 3.97 (1.32–7.65)% | 1.47 (0.49–2.85)% | 67.8 (34.0–203.5) | 0.001 |
| | **Random Forest** | 0.0158 | 407 | 10.00 | 1.47 (0.49–2.70)% | 6 | 6 | 6 | 3.97 (1.26–7.29)% | 1.47 (0.47–2.70)% | 67.8 (37.0–204.0) | 0.004 |
| | **Mask-Aware Transformer** | 0.0110 | 407 | 10.00 | 1.23 (0.25–2.46)% | 5 | 5 | 5 | 3.31 (0.74–6.67)% | 1.23 (0.26–2.48)% | 81.4 (40.7–407.0) | 0.004 |
| | **Clinical Transformer (Baseline)** | 0.0094 | 407 | 10.00 | 1.23 (0.25–2.46)% | 5 | 5 | 5 | 3.31 (0.68–6.38)% | 1.23 (0.25–2.37)% | 81.4 (40.7–407.0) | 0.007 |
| | **Logistic Regression (L2)** | 0.0095 | 407 | 10.00 | 0.98 (0.25–1.97)% | 4 | 4 | 4 | 2.65 (0.59–5.51)% | 0.98 (0.23–2.07)% | 101.8 (50.8–408.0) | 0.021 |
| **20.0 episodes / 100 pt-days** | **Tabular GBDT (XGBoost)** | 0.0105 | 814 | 20.00 | 1.84 (0.98–2.70)% | 15 | 15 | 15 | 9.93 (5.26–14.47)% | 3.68 (1.87–5.29)% | 54.3 (37.0–101.9) | 0.000 |
| | **Mask-Aware Transformer** | 0.0099 | 814 | 20.00 | 1.72 (0.74–2.58)% | 14 | 14 | 14 | 9.27 (4.54–13.77)% | 3.43 (1.59–5.19)% | 58.1 (38.8–135.7) | 0.000 |
| | **Logistic Regression (L2)** | 0.0088 | 814 | 20.00 | 1.35 (0.49–2.09)% | 11 | 11 | 11 | 7.28 (2.74–11.11)% | 2.70 (0.99–4.18)% | 74.0 (47.8–203.7) | 0.000 |
| | **BiMamba SSM (Proposed)** | 0.0142 | 814 | 20.00 | 1.23 (0.49–2.09)% | 10 | 10 | 10 | 6.62 (3.07–10.89)% | 2.45 (1.05–4.10)% | 81.4 (47.9–203.2) | 0.000 |
| | **Tabular GBDT (LightGBM)** | 0.0105 | 814 | 20.00 | 1.23 (0.61–2.09)% | 10 | 10 | 10 | 6.62 (3.11–11.04)% | 2.45 (1.15–4.10)% | 81.4 (47.9–163.0) | 0.000 |
| | **Random Forest** | 0.0131 | 814 | 20.00 | 1.11 (0.49–1.85)% | 9 | 9 | 9 | 5.96 (2.70–9.68)% | 2.21 (0.96–3.69)% | 90.4 (54.2–203.7) | 0.001 |
| | **Multi-Model Stacking Ensemble** | 0.0115 | 814 | 20.00 | 0.98 (0.37–1.72)% | 8 | 8 | 8 | 5.30 (2.04–9.15)% | 1.96 (0.74–3.43)% | 101.8 (58.1–271.3) | 0.000 |
| | **Clinical Transformer (Baseline)** | 0.0088 | 814 | 20.00 | 0.86 (0.25–1.48)% | 7 | 7 | 7 | 4.64 (1.51–8.18)% | 1.72 (0.52–3.04)% | 116.3 (67.8–407.0) | 0.001 |

---

## Table S3B: Full-Precision IEEE 754 Boundary Score Tie Audit

*Evaluated on full-precision 64-bit floating point scores in held-out test data ($N=8,254$ stays).*

| Prespecified Budget | Model Architecture | Exact Cutoff Score (Float64) | Stays Strictly Above | Stays Tied at Boundary | Stays Selected from Tied | Selection Policy |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **5.0 episodes / 100 pt-days** | Multi-Model Stacking Ensemble | `0.014490108370127998` | 203 | 2 | 1 | Ascending `ICUSTAY_ID` priority |
| | Random Forest | `0.017658340327200716` | 202 | 2 | 2 | Both selected into quota |
| **20.0 episodes / 100 pt-days** | Tabular GBDT (LightGBM) | `0.010536972856183402` | 623 | 2636 | 191 | Ascending `ICUSTAY_ID` priority |
| | Tabular GBDT (XGBoost) | `0.010525625922113648` | 812 | 5 | 2 | Ascending `ICUSTAY_ID` priority |

*Note: For all other model and budget combinations, scores are strictly continuous with zero boundary ties (`strictly_above = K - 1`, `tied_exact = 1`, `selected_from_tied = 1`).*

---

## Table S4: Two-Axis Subgroup & Timestamp Uncertainty Matrix

*Opportunity Bounds Conditional on Fixed Reference Sepsis-3 Timestamps and Surveillance Coverage.*

| Cohort & Timestamp Source | Sepsis Onset Category | Subgroup Denominator ($N$) | Surveillance Covered Stays ($V=1$) | Definitely Eligible Count | Indeterminate Count | Surveillance Uncovered ($V=0$) | Covered but Ineligible ($V=1, O=0$) | Within-Subgroup Opportunity Bounds | Within-Covered Opportunity Bounds | Contribution to Overall Cohort Bound |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MIMIC-IV Prospective-Replay** | Early Incident (<=24h) | 47 | 47 (100.0%) | 47 | 0 | 0 | 0 | [100.0%, 100.0%] | [100.0%, 100.0%] | 47 / 408 (11.5%) |
| *(Hourly Confirmed eMAR)* | Intermediate Incident (24–48h) | 156 | 104 (66.7%) | 103 | 0 | 52 | 1 | [66.0%, 66.0%] | [99.0%, 99.0%] | 103 / 408 (25.2%) |
| | Late Incident (>48h) | 205 | 0 (0.0%) | 0 | 0 | 205 | 0 | [0.0%, 0.0%] | N/A (No covered stays) | 0 / 408 (0.0%) |
| | **All Incident Sepsis-3 Cases** | **408** | **151 (37.0%)** | **150** | **0** | **257** | **1** | **[36.8%, 36.8%]** | **[99.3%, 99.3%]** | **150 / 408 (36.8%)** |
| **MIMIC-III Broad Incident Sepsis** | Early Incident (<=24h) | 2,641 | 102 (3.9%) | 0 | 74 | 2,539 | 28 | [0.0%, 2.8%] | [0.0%, 72.5%] | [0, 74] / 3,581 ([0.00%, 2.07%]) |
| *(Date-Level Prescription Proxy)* | Intermediate Incident (24–48h) | 361 | 235 (65.1%) | 1 | 36 | 126 | 198 | [0.3%, 10.2%] | [0.4%, 15.7%] | [1, 37] / 3,581 ([0.03%, 1.03%]) |
| | Late Incident (>48h) | 579 | 0 (0.0%) | 0 | 0 | 579 | 0 | [0.0%, 0.0%] | N/A (No covered stays) | [0, 0] / 3,581 ([0.00%, 0.00%]) |
| | **All Incident Sepsis-3 Cases** | **3,581** | **337 (9.4%)** | **1** | **110** | **3,244** | **226** | **[0.03%, 3.10%]** | **[0.30%, 32.94%]** | **[1, 111] / 3,581 ([0.03%, 3.10%])** |

---

## Appendix 1: TRIPOD+AI Reporting Checklist Crosswalk

| Section / Item | TRIPOD+AI Recommendation | Implementation in Manuscript |
| :--- | :--- | :--- |
| **Title** | Identify the study as evaluating ML performance in operational context | Title explicitly specifies *Surveillance Coverage, Treatment Precedence, and Workload Constraints* |
| **Abstract** | Structured summary of objectives, design, setting, participants, findings, and operational metrics | 250-word structured abstract reporting PPV, NNE, and population waterfalls |
| **Study Design** | Explicitly state prospective-replay design, time horizon, and surveillance windows | Documented in Section 2.1: 12 scheduled hourly bins at ICU hours 19–30 |
| **Eligibility Criteria** | Clear inclusion and exclusion criteria | Section 2.1 and Table S1: Adult first ICU admissions, stay $\ge 30\text{h}$ |
| **Outcome Definition** | Unambiguous definition of reference event and diagnostic timing | Section 2.2: Sepsis-3 consensus criteria (blood culture, antibiotics, acute $\Delta\text{SOFA}\ge 2$) |
| **Treatment Precedence** | Distinction between pre-treatment and post-treatment predictions | Section 2.3: Formal risk sets $\mathcal{R}_0$ and $\mathcal{R}_2$ with eMAR confirmation |
| **Predictor Timing** | Specification of observation windows and feature generation rules | Section 2.1: Hourly vital signs, laboratory markers, and interventions |
| **Handling of Workload** | Reporting of threshold selection, refractory periods, and alert budgets | Section 2.4 and Table S3A: 12h refractory suppression at 5, 10, 20 alerts/100 pt-days |
| **Statistical Methods** | Clustered resampling and transparent confidence interval estimation | Section 2.5: Patient-clustered bootstrap ($B=1,000$) with exact reciprocal NNE intervals |
| **Interpretation** | Non-causal operational reporting without claims of clinical efficacy | Section 4: Workload-dependent yield and surveillance coverage constraints |
| **Reproducibility** | Cryptographic manifest, code availability, and environment reporting | Section 5 and Appendix 2: Release manifest version `1.0.0-freeze` (Git `b0932ca`) |

---

## Appendix 2: Computing Environment & Reproducibility Statement

- **Platform:** Linux (x86_64 architecture)
- **Python Version:** 3.10.19
- **Core Dependencies:** NumPy 1.26.4, Pandas 1.5.3, Scikit-learn 1.2.2, Scipy 1.10.1, Matplotlib 3.7.1
- **Release Checksum File:** `reports/manifest/release_manifest.json` (Git Commit `b0932ca`)
- **Code & Model Availability:** Complete analysis scripts, preprocessing pipelines, and evaluation harnesses are archived at GitHub: `https://github.com/alireza/Sepsis-Onset-Timing-Paradox` under release tag `v1.0.0-freeze`.
