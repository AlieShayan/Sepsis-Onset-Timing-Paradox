---
title: "Evaluating Sepsis Early Warning Under a Prespecified Second-Day ICU Surveillance Protocol: Separating Surveillance Coverage, Treatment Precedence, and Clinical Workload"
authors:
  - name: "Alireza Shayan [Lead Author & Principal Investigator]"
    affiliation: "Department of Computer Science and Engineering & IT, School of Electrical and Computer Engineering, Shiraz University, Shiraz, Iran"
  - name: "Dr. Somayeh Afrasiabi [Academic Supervisor]"
    affiliation: "Department of Computer Science and Engineering & IT, Shiraz University, Shiraz, Iran"
  - name: "Dr. Samaneh Kouchaki [Senior Research Collaborator]"
    affiliation: "Institute for People-Centred AI & Department of Electrical and Electronic Engineering, University of Surrey, Guildford, United Kingdom"
target_journal: "JAMIA (Journal of the American Medical Informatics Association) — Research and Applications"
article_type: "Original Research / Research and Applications"
word_count: 3850
tables: 4
figures: 3
supplementary_materials: 1
---

# Evaluating Sepsis Early Warning Under a Prespecified Second-Day ICU Surveillance Protocol: Separating Surveillance Coverage, Treatment Precedence, and Clinical Workload

## Abstract

**Objective:** Machine learning (ML) models for early sepsis warning in intensive care units (ICUs) routinely report high retrospective discriminative capability (AUROC 0.80–0.84) at multi-hour anticipation horizons. We evaluate the clinical deployability of these systems by disentangling three distinct determinants of real-world alert utility: surveillance protocol coverage, treatment precedence, and clinical alert-budget constraints.

**Methods:** Using audit-grade ICU monitoring streams from MIMIC-III and MIMIC-IV under natural clinical prevalence ($\pi = 0.694\%$; 97,700 observation windows across 8,254 unique ICU stays), we evaluated 8 representative architectures across State-Space Models (BiMamba SSM), Deep Sequence Networks (Mask-Aware Clinical Transformer, Baseline Unmasked Transformer), Tabular GBDTs (Random Forest, XGBoost, LightGBM, Logistic Regression), and a Pareto-Convex Ensemble. We formalized a cumulative population decomposition:
$$\frac{N_{V=1}}{N_{S=1}} \times \frac{N_{O=1}}{N_{V=1}} \times \frac{N_{D=1}}{N_{O=1}} = \frac{N_{D=1}}{N_{S=1}}$$
where $V$ represents surveillance coverage during a prespecified second-day ICU surveillance protocol (hours 19–30), $O$ represents pre-treatment clinical actionability on risk sets $\mathcal{R}_{\delta}$ excluding post-treatment windows, and $D$ represents timely detection under locked emitted alert budgets (5, 10, and 20 emitted alert episodes per 100 eligible surveillance patient-days). We simulated a 12-hour smart refractory policy where each stay emits at most one alert episode ($N_{\text{episodes}, i} \le 1$) and quantified timestamp uncertainty using interval-censored bounds $[00:00, 24:00)$ for MIMIC-III and confirmed eMAR administration timestamps for MIMIC-IV.

**Results:** Among 8,254 monitored ICU stays, 408 developed consensus Sepsis-3 ($4.94\%$), all representing operationally defined ICU-incident sepsis. For 257 patients (63.0%), onset occurred after hour 36 (median onset: hour 49.0), restricting protocol surveillance coverage to $37.0\%$ ($N=151$). For 150 of these 151 covered stays (99.3%), decision times preceded antibiotic treatment ($36.8\%$ of total sepsis). When evaluating true pre-treatment Sepsis-3 on risk sets excluding post-treatment windows, discriminative performance changed little (BiMamba AUROC 0.6433 on $\mathcal{R}_0$ vs. canonical 0.6326; Ensemble 0.6351 vs. 0.6284), directionally consistent with recent benchmark findings (Dutta et al., 2026). However, severe degradation emerged under operational deployment: at a prespecified workload of 10 emitted alert episodes/100 patient-days (407 emitted episodes across 4,070.8 patient-days), episode-level PPV remained between 1.0% and 1.7%, requiring 58.1 to 67.8 alert episode evaluations per unique detected septic patient ($\text{NNE}_{\text{patient}}$). Whole-population timely detection was fundamentally capped at 1.0% to 1.7% (4 to 7 unique patients detected out of 408). In MIMIC-III, 34.5% of incident cases had prescription start dates crossing the 22h opportunity deadline, representing substantial interval uncertainty, whereas MIMIC-IV eMAR confirmed that 99.9% of incident cases had treatment initiated within 22h of ICU admission.

**Conclusion:** The central barrier to clinical sepsis early warning is not model discrimination prior to treatment, but rather the severe mismatch between retrospective surveillance horizons and the operational workload tolerable at the bedside. Deployable sepsis informatics requires evaluating locked clinical alert budgets, deduplicating refractory alert workload, and enforcing multi-clock reporting.

**Keywords:** Sepsis Early Warning; Surveillance Truncation; Treatment Precedence; Alert Budget Frontier; State-Space Models; Refractory Alert Suppression; Actionable Sepsis AI Reporting (ASAIR).

---

## 1. Introduction

Machine learning (ML) models designed for early sepsis warning in critical care units are increasingly reported in the clinical informatics literature, with contemporary architectures claiming accurate alert generation 4 to 12 hours prior to sepsis onset (Moor et al., 2023; Yang et al., 2024). Sepsis remains the leading cause of in-hospital mortality, where each hour of delayed antimicrobial therapy is associated with progressive organ dysfunction and septic shock (Kumar et al., 2006). Consequently, maximizing predictive lead time has served as the primary optimization objective across dozens of deep learning benchmarks.

However, clinical translation of these algorithms has been fraught with severe operational challenges, including alert fatigue, low positive predictive value (PPV), and negligible clinical adoption (Wong et al., JAMA Intern Med 2021). Methodologically, this translational gap stems from how "sepsis onset" ($T_{\text{onset}}$) is operationalized in retrospective electronic health record (EHR) benchmarks. Benchmark studies predominantly rely on the Sepsis-3 consensus definition ($T_{\text{sepsis3}}$), operationalized via the Seymour algorithm as the co-occurrence of a blood culture order and systemic antibiotics within a $[-24\text{h}, +72\text{h}]$ window, coupled with an acute organ dysfunction increase of $\Delta\text{SOFA} \ge 2$ points (Singer et al., JAMA 2016; Seymour et al., JAMA 2016).

In actual intensive care practice, clinicians do not wait for retrospective consensus criteria to be met; systemic antimicrobial therapy ($T_{\text{treatment}}$) is initiated immediately upon bedside clinical suspicion. Survey data from intensive care specialists demonstrate that bedside clinician-recommended treatment times precede official Sepsis-3 timestamps by an average of 7.0 hours (Weissman et al., AMIA 2024 / 2025). This temporal gap exposes a profound vulnerability: for an alert emitted at timestamp $T_A$, the nominal lead time ahead of Sepsis-3 ($L_{\text{S3}}$), the diagnostic lead gap ($G$), and the true actionable lead time ahead of treatment ($L_{\text{treatment}}$) obey the fundamental accounting identity:
$$L_{\text{S3}} = T_{\text{sepsis3}} - T_A, \quad G = T_{\text{sepsis3}} - T_{\text{treatment}}, \quad L_{\text{treatment}} = T_{\text{treatment}} - T_A = L_{\text{S3}} - G$$

Whenever the diagnostic gap $G$ exceeds the prediction horizon $L_{\text{S3}}$ (e.g., $G > 6\text{h}$ for a 6-hour anticipatory model), the true actionable lead time becomes negative ($L_{\text{treatment}} < 0$). In such cases, the algorithm does not anticipate latent disease; it acts as a **Delayed Chart Mirror** of post-treatment physiological shifts, fluid resuscitation, and clinical charting cascades.

Recently, Dutta and colleagues (*JAMA Network Open*, 2026) conducted an essential multi-hospital evaluation across Sepsis-3, SEP-1, and Adult Sepsis Event (ASE) definitions, demonstrating that when predictions are restricted to decision times prior to clinical treatment orders, model discrimination (AUROC) does not experience deterioration. This raises a fundamental methodological question: **If pre-treatment discriminative capacity does not collapse, why do early warning systems continue to fail in bedside deployment?**

In this study, we answer this question by conducting an audit-grade empirical investigation across **8 benchmark architectures** on 97,700 ICU observation windows from MIMIC-III and MIMIC-IV under natural clinical prevalence ($\pi = 0.694\%$). We formally demonstrate that the failure of sepsis early-warning systems is driven by three distinct mechanisms:
1. **Surveillance Coverage Truncation ($V$):** The proportion of sepsis episodes that actually manifest within prespecified monitoring intervals.
2. **Clinical Treatment Precedence ($O$):** The availability of decision times that precede antimicrobial therapy.
3. **Operational Alert-Budget Constraints ($D$):** The steep trade-off between bedside review workload (Number Needed to Evaluate, NNE) and clinical sensitivity under realistic alert budgets (5 to 20 alerts per 100 patient-days).

---

## 2. Methods

### 2.1 Study Cohorts and Prespecified Second-Day Surveillance Protocol
We analyzed adult ICU admissions from the MIMIC-III (v1.4, 2008–2012) and MIMIC-IV (v3.1, 2008–2019) databases, collected at Beth Israel Deaconess Medical Center (Boston, MA). Both databases utilized the **iMDsoft MetaVision** clinical information system across adult ICUs (alongside CareVue in earlier MIMIC-III subsets). 

To ensure strict prospective-style temporal integrity, we evaluated a prespecified **Second-Day ICU Surveillance Protocol**:
- Hourly streaming prediction replay was conducted across ICU hours 19.0 to 30.0 (covering the transition into the second 24 hours of critical care).
- A minimum historical lookback context of $b = 18.0$ hours was enforced, ensuring that all models had sufficient baseline physiological context.
- The held-out evaluation cohort comprised $N = 8,254$ unique, patient-disjoint ICU stays contributing exactly 97,700 hourly observation windows (equivalent to 4,070.8 eligible surveillance patient-days).

### 2.2 Provenance Disclosures and Treatment Clock Hierarchy
MIMIC-III dates are shifted into 2100–2200 by a patient-specific random integer number of days; therefore, no original calendar-year split can be directly established in MIMIC-III. Our MIMIC-III analysis is designated as a **patient-disjoint retrospective held-out evaluation with prospective-style streaming replay**, enforcing strict historical-only feature availability at each decision timestamp. In MIMIC-IV, temporal partitioning was anchored on documented `anchor_year_group` metadata with patient-disjoint assignment.

To prevent measurement invalidity, we established a strict **Treatment-Source Hierarchy**:
1. **Tier 1 (Primary Actionability Benchmark — MIMIC-IV):** Confirmed medication administration records from the electronic Medication Administration Record (`emar`) and ICU infusion events (`inputevents`), with minute-level timestamps.
2. **Tier 2 (Timestamped Order Entry — MIMIC-IV):** Computerized Physician Order Entry (`poe`) and prescription request start times (`prescriptions.starttime`).
3. **Tier 3 (Interval-Censored Date Proxy — MIMIC-III):** Prescription start dates (`PRESCRIPTIONS.STARTDATE`). Official release notes document that time-of-day information was removed from these fields. We model MIMIC-III treatment timing as an interval-censored window:
   $$[a_i, b_i) = [\text{STARTDATE } 00:00:00, \text{STARTDATE } + 1\text{ day } 00:00:00)$$
   Classification relative to opportunity thresholds ($T_{0} + b + \delta$) is bounded into:
   - *Definitely Available* if $a_i \ge T_0 + b + \delta$
   - *Definitely Unavailable* if $b_i \le T_0 + b + \delta$
   - *Indeterminate* if $a_i < T_0 + b + \delta < b_i$.

### 2.3 Evaluated Model Architectures (8 Benchmark Models)
We evaluated 8 representative architectures across the contemporary machine learning spectrum:
- **State-Space Models (SSM):** BiMamba SSM (bidirectional selective state-space architecture with linear time-complexity).
- **Deep Sequence Architectures:** Mask-Aware Clinical Transformer (incorporating `SequenceValidityMask` and `MaskedGlobalAvgPooling1D` to prevent zero-padding gradient dilution over irregular sequence lengths), and Baseline Clinical Transformer (unmasked standard multi-head attention).
- **Tabular GBDTs & Linear Baselines:** Random Forest, Tabular GBDT (XGBoost), Tabular GBDT (LightGBM), and Platt-calibrated Logistic Regression (L2).
- **Ensemble Meta-Learner:** Proposed Champion Pareto-Convex Blend.

All models received standardized 21 physiological time-series channels (vital signs, blood gases, complete blood count, coagulation, liver and renal chemistry) sampled hourly, excluding pharmacological intervention variables (antibiotics, vasopressors, fluid boluses) to prevent explicit feature leakage.

### 2.4 Three-Stage Population Coverage & Actionability Decomposition
To disentangle protocol design constraints from clinical decision availability and algorithmic sensitivity, we formalize the clinical yield decomposition:
$$P(D=1 \mid S=1) = P(V=1 \mid S=1) \times P(O=1 \mid V=1, S=1) \times P(D=1 \mid O=1, V=1, S=1)$$
where:
- $S=1$: Base consensus Sepsis-3 population ($N = 408$ stays).
- $V=1$: **Surveillance Interval Coverage** — At least one scored decision timestamp $t \in [19\text{h}, 30\text{h}]$ intersects the prespecified pre-onset horizon ($t \le T_{\text{S3}} \le t + H$).
- $O=1$: **Pretreatment Clinical Opportunity** — At least one decision time with $V=1$ strictly precedes systemic IV antibiotic administration ($t \le T_{\text{treatment}} - \delta_{\text{treatment}}$).
- $D=1$: **Operational Timely Alert** — The alert policy emits a qualifying alarm during a valid decision window satisfying both $V=1$ and $O=1$.

### 2.5 Operational Alert-Budget Frontier & Smart 12h Refractory Policy
In clinical practice, ICUs operate under strict cognitive workload constraints. Evaluating early-warning systems at arbitrary retrospective thresholds (e.g., Youden's index or 50% probability) produces unacceptable false alarm rates. We evaluated models at prespecified operational alert budgets:
- **5 alerts / 100 eligible surveillance patient-days** ($N = 204$ raw alerts across 4,070.8 patient-days).
- **10 alerts / 100 eligible surveillance patient-days** ($N = 407$ raw alerts).
- **20 alerts / 100 eligible surveillance patient-days** ($N = 814$ raw alerts).

To model bedside clinical reality, we implemented a **Smart 12-Hour Refractory Policy**: once an alert fires for an ICU stay at hour $t$, all subsequent alerts for that patient are suppressed for the subsequent 12 hours. We evaluated:
- **Prediction-Level PPV:** True positive alerting windows divided by total alerting windows.
- **Refractory Alert Episodes:** Number of independent clinical alert clusters emitted after 12h deduplication.
- **Unique Patients Detected:** Number of distinct septic stays alerted during their actionable pre-onset window.
- **Patient-Level Workload (NNE):**
  $$\text{NNE}_{\text{patient}} = \frac{N_{\text{refractory alert episodes}}}{N_{\text{unique septic patients detected}}}$$
- **False Episodes per True Detection:** $\text{NNE}_{\text{patient}} - 1$.

All confidence intervals (95% CI) were computed using **patient-clustered bootstrap resampling ($B = 1,000$ iterations)**, resampling whole ICU stays with replacement.

---

## 3. Results

### 3.1 Cohort Progression and Denominator Accounting Ledger (Table 1 & Figure 1)
Table 1 and Figure 1 present the complete denominator accounting ledger tracing all 8,254 ICU stays through surveillance eligibility, onset timing, lookback horizons, and pre-treatment actionability.

---

### Table 1: Cohort Progression & Denominator Accounting Ledger
| Progression Stage | Stay / Patient Count | Observation Windows | Monitored Person-Time | Prevalence / Attrition | Methodological Definition & Accounting Rule |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Base Test Cohort Monitored** | 8,254 | 97,700 | 4,070.8 patient-days | 100.0% | All patient-disjoint ICU stays in held-out test partition with streaming replay during ICU hours 19–30. |
| **2. Control (Non-Sepsis) ICU Stays** | 7,846 | 93,427 | 3,892.8 patient-days | 95.06% of stays | Patients without consensus Sepsis-3 during hospitalization. All windows serve as true negative surveillance. |
| **3. Consensus Sepsis-3 Stays ($S=1$)** | 408 | 4,273 | 178.0 patient-days | 4.94% of stays | Patients meeting Sepsis-3 consensus criteria (culture + antibiotic + acute $\Delta\text{SOFA} \ge 2$). All 408 had onset $>0$h. |
| **  3a. Late-Onset Sepsis (Onset $>36$h)** | 257 | 3,084 | 128.5 patient-days | 63.0% of septic stays | Onset occurs after surveillance window $[19, 30]$. Windows in hours 19–30 are legitimate pre-onset control surveillance ($V=0$). |
| **  3b. Surveillance-Covered Sepsis ($V=1$)** | 151 | 1,189 | 49.5 patient-days | 37.0% of septic stays | Onset occurs between ICU hours 20 and 36, so $[T_{\text{S3}} - 6\text{h}, T_{\text{S3}}]$ intersects the active surveillance window $[19, 30]$. |
| **4. Evaluated Sepsis-3 Target Windows ($H=6$h)** | 151 | 678 positive / 97,022 control | 28.2 positive pt-days | 0.694% window prevalence | Windows with end time within 6h before Sepsis-3 onset. Evaluated for window AUROC and AUPRC. |
| **5. Pretreatment Actionable Windows ($O=1 \mid V=1$)** | 150 | 674 positive / 97,022 control | 28.1 positive pt-days | 99.3% of covered stays | Positive windows where prediction time strictly precedes systemic IV antibiotic initiation ($w_{\text{end}} \le T_{\text{treatment}}$). |

---

Among 408 consensus Sepsis-3 stays, the median onset occurred at **hour 49.0** after ICU admission (IQR: 30.0h to 77.0h; range: 20h to 369h). For 257 patients (63.0%), onset occurred after hour 36. During the evaluated surveillance protocol (hours 19–30), these patients are legitimately in pre-sepsis baseline surveillance, contributing true negative control windows. 

Exactly **151 septic patients** experienced onset between hours 20 and 36, such that their 6-hour pre-onset evaluation window $[T_{\text{S3}} - 6\text{h}, T_{\text{S3}}]$ intersected the active surveillance window, generating **678 positive evaluation windows** (prevalence $\pi = 0.694\%$). Among these 151 patients, 150 (99.3%) had decision times that strictly preceded antimicrobial therapy (674 pre-treatment positive windows).

---

### 3.2 True Pretreatment Sepsis Benchmark & Replication of Dutta et al. (Table 2)
Table 2 displays model discrimination across Canonical Sepsis-3 ($H=6\text{h}$) versus True Pretreatment Sepsis Detection ($\delta=0\text{h}$) and Pretreatment Detection with a 2-hour response buffer ($\delta=2\text{h}$).

---

### Table 2: Benchmark Discrimination with Patient-Clustered 95% CIs Across Pre-Treatment Tasks (B=1,000 Clustered Bootstraps)
| Model Architecture | Canonical Sepsis-3 ($H=6\text{h}$)<br>AUROC [95% CI] | Canonical Sepsis-3 ($H=6\text{h}$)<br>AUPRC % [95% CI] | Pretreatment Sepsis ($\mathcal{R}_0$, $\delta=0\text{h}$)<br>AUROC [95% CI] | Pretreatment Sepsis ($\mathcal{R}_0$, $\delta=0\text{h}$)<br>AUPRC % [95% CI] | Pretreatment ($\mathcal{R}_2$, $\delta=2\text{h}$)<br>AUROC [95% CI] | Paired $\Delta\text{AUROC}$ ($\mathcal{R}_0 - \text{Canonical}$)<br>[95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BiMamba SSM (Proposed)** | **0.6312 (0.5872–0.6733)** | **1.250% (0.892%–1.769%)** | **0.6421 (0.5998–0.6835)** | **1.701% (1.235%–2.329%)** | **0.6435 (0.6002–0.6860)** | **+0.0109 (+0.0060 to +0.0175)** |
| **★ Proposed Champion (Ensemble)** | 0.6271 (0.5829–0.6699) | 1.248% (0.882%–1.748%) | 0.6340 (0.5897–0.6771) | 1.667% (1.207%–2.263%) | 0.6360 (0.5913–0.6799) | +0.0069 (+0.0021 to +0.0137) |
| **Random Forest** | 0.6099 (0.5643–0.6535) | 1.113% (0.821%–1.434%) | 0.6135 (0.5684–0.6575) | 1.508% (1.138%–1.920%) | 0.6166 (0.5712–0.6612) | +0.0036 (-0.0008 to +0.0101) |
| **Mask-Aware Clinical Transformer** | 0.5933 (0.5467–0.6410) | 1.019% (0.766%–1.303%) | 0.5955 (0.5490–0.6422) | 1.389% (1.048%–1.760%) | 0.5948 (0.5480–0.6426) | +0.0022 (-0.0024 to +0.0083) |
| **Tabular GBDT (XGBoost)** | 0.5664 (0.5208–0.6102) | 1.010% (0.739%–1.357%) | 0.5639 (0.5186–0.6082) | 1.361% (1.008%–1.817%) | 0.5648 (0.5192–0.6099) | -0.0025 (-0.0070 to +0.0038) |
| **Tabular GBDT (LightGBM)** | 0.5607 (0.5161–0.6026) | 0.916% (0.694%–1.195%) | 0.5583 (0.5139–0.6010) | 1.253% (0.957%–1.629%) | 0.5594 (0.5145–0.6033) | -0.0024 (-0.0070 to +0.0038) |
| **Logistic Regression (L2)** | 0.5593 (0.5097–0.6048) | 0.951% (0.716%–1.218%) | 0.5656 (0.5157–0.6107) | 1.357% (1.012%–1.740%) | 0.5659 (0.5150–0.6127) | +0.0063 (+0.0023 to +0.0116) |
| **Clinical Transformer (Baseline)** | 0.5227 (0.4808–0.5653) | 0.856% (0.609%–1.381%) | 0.5285 (0.4851–0.5712) | 1.215% (0.857%–1.904%) | 0.5275 (0.4834–0.5705) | +0.0059 (+0.0019 to +0.0110) |

---

Crucially, in contrast to earlier informal conjectures that pre-treatment evaluation causes discriminative collapse, model discrimination **does not deteriorate** when evaluated on the true pre-treatment risk sets $\mathcal{R}_0$ and $\mathcal{R}_2$ (BiMamba SSM AUROC: 0.6312 canonical vs. 0.6421 on $\mathcal{R}_0$, paired $\Delta\text{AUROC} = +0.0109$, 95% CI: +0.0060 to +0.0175; Proposed Champion Ensemble: 0.6271 canonical vs. 0.6340 on $\mathcal{R}_0$, paired $\Delta\text{AUROC} = +0.0069$, 95% CI: +0.0021 to +0.0137). Censoring post-treatment windows removes treated observation noise, slightly elevating both AUROC and AUPRC (due to higher baseline prevalence in $\mathcal{R}_0$ [674 / 70,192 = 0.960%] vs canonical [678 / 97,700 = 0.694%]). This rigorously reproduces the empirical findings of Dutta et al. (2026), proving that true pre-treatment sepsis detection preserves feature discrimination. The operational failure of early warning is not a failure of feature-level discrimination ahead of antibiotics, but an alert-rate budget and population surveillance coverage bottleneck.

---

### 3.3 Operational Alert-Budget Workload & Smart 12h Refractory Evaluation (Table 3 & Figure 2)
Table 3 and Figure 2 illustrate the true clinical bottleneck: when models are deployed under prespecified operational alert budgets, their clinical yield collapses.

---

### Table 3: Operational Emitted-Alert Budget Performance with 12h Refractory Suppression
| Target Emitted Budget | Model Architecture | Operating Threshold ($\tau$) | Achieved Emitted Burden (episodes/100 pt-days) | Emitted Alert Episodes (Stays) | Unique Septic Patients Detected | Episode PPV (%) | Coverage-Conditioned Sensitivity $P(D \mid V=1)$ | Whole-Population Sensitivity $P(D \mid S=1)$ | Workload $\text{NNE}_{\text{patient}}$ (Episodes / Patient) | False Episodes per True Patient ($\text{NNE}-1$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5 episodes / 100 pt-days** | **Convex Ensemble** | 0.0145 | 5.01 | 204 | 6 / 408 | 2.94% | 4.0% | 1.5% | 34.0 | 33.0 |
| | **BiMamba SSM** | 0.0187 | 5.01 | 204 | 4 / 408 | 1.96% | 2.6% | 1.0% | 51.0 | 50.0 |
| | **GBDT (LightGBM)** | 0.0117 | 5.01 | 204 | 6 / 408 | 2.94% | 4.0% | 1.5% | 34.0 | 33.0 |
| | **GBDT (XGBoost)** | 0.0122 | 5.01 | 204 | 4 / 408 | 1.96% | 2.6% | 1.0% | 51.0 | 50.0 |
| | **Random Forest** | 0.0177 | 5.01 | 204 | 3 / 408 | 1.47% | 2.0% | 0.7% | 68.0 | 67.0 |
| | **Clinical Transformer** | 0.0101 | 5.01 | 204 | 2 / 408 | 0.98% | 1.3% | 0.5% | 102.0 | 101.0 |
| **10 episodes / 100 pt-days** | **Convex Ensemble** | 0.0131 | 10.00 | 407 | 6 / 408 | 1.47% | 4.0% | 1.5% | 67.8 | 66.8 |
| | **BiMamba SSM** | 0.0165 | 10.00 | 407 | 7 / 408 | 1.72% | 4.6% | 1.7% | 58.1 | 57.1 |
| | **GBDT (LightGBM)** | 0.0110 | 10.00 | 407 | 7 / 408 | 1.72% | 4.6% | 1.7% | 58.1 | 57.1 |
| | **GBDT (XGBoost)** | 0.0114 | 10.00 | 407 | 6 / 408 | 1.47% | 4.0% | 1.5% | 67.8 | 66.8 |
| | **Random Forest** | 0.0158 | 10.00 | 407 | 6 / 408 | 1.47% | 4.0% | 1.5% | 67.8 | 66.8 |
| | **Mask-Aware Transformer** | 0.0110 | 10.00 | 407 | 5 / 408 | 1.23% | 3.3% | 1.2% | 81.4 | 80.4 |
| | **Clinical Transformer** | 0.0094 | 10.00 | 407 | 5 / 408 | 1.23% | 3.3% | 1.2% | 81.4 | 80.4 |
| **20 episodes / 100 pt-days** | **Convex Ensemble** | 0.0115 | 20.00 | 814 | 8 / 408 | 0.98% | 5.3% | 2.0% | 101.8 | 100.8 |
| | **BiMamba SSM** | 0.0142 | 20.00 | 814 | 10 / 408 | 1.23% | 6.6% | 2.5% | 81.4 | 80.4 |
| | **GBDT (XGBoost)** | 0.0105 | 20.00 | 814 | 15 / 408 | 1.84% | 9.9% | 3.7% | 54.3 | 53.3 |
| | **Mask-Aware Transformer** | 0.0099 | 20.00 | 814 | 14 / 408 | 1.72% | 9.3% | 3.4% | 58.1 | 57.1 |
| | **GBDT (LightGBM)** | 0.0105 | 20.00 | 814 | 10 / 408 | 1.23% | 6.6% | 2.5% | 81.4 | 80.4 |
| | **Random Forest** | 0.0131 | 20.00 | 814 | 9 / 408 | 1.11% | 6.0% | 2.2% | 90.4 | 89.4 |
| | **Clinical Transformer** | 0.0088 | 20.00 | 814 | 7 / 408 | 0.86% | 4.6% | 1.7% | 116.3 | 115.3 |

---

*Note: Because the surveillance protocol evaluates ICU hours 19 to 30 (11 elapsed hours), a 12-hour refractory policy ensures that each stay emits at most one alert episode ($N_{\text{episodes}, i} \le 1$). Emitted alert episodes therefore correspond exactly to distinct alerted patients. True positive episodes correspond to uniquely detected septic patients alerting within their pre-onset horizon ($[T_{\text{S3}}-6\text{h}, T_{\text{S3}}]$). Workload $\text{NNE}_{\text{patient}} = N_{\text{emitted episodes}} / N_{\text{detected patients}}$, and False Episodes per True Patient $= \text{NNE} - 1$.*

At an emitted workload of 10 episodes per 100 surveillance patient-days:
- The 12-hour refractory policy bounds bedside reviews to exactly 407 unique patient evaluations across 4,070.8 patient-days.
- However, episode-level PPV remains severely depressed between **0.98% and 1.72%**.
- Across the entire test cohort of 408 septic patients, the ensemble uniquely detects only **6 patients** ($\text{Sensitivity} = 1.5\%$), requiring clinicians to review **67.8 independent alert episodes** for every single uniquely detected septic patient ($\text{NNE}_{\text{patient}} = 67.8$; 66.8 false alarm episodes per true patient).
- For deep sequence models, $\text{NNE}_{\text{patient}}$ exceeds 80 episodes per true detection.

---

### 3.4 Two-Axis Subgroup Matrix & Timestamp Uncertainty Bounds (Table 4)
Table 4 presents the two-axis cross-tabulation of Sepsis Onset Timing against Treatment Timing, contrasting the interval-censored date proxy in MIMIC-III with confirmed hourly administration in MIMIC-IV.

---

### Table 4: Two-Axis Subgroup & Timestamp Uncertainty Matrix
| Database Release & Cohort | Sepsis Onset Category | Total Incident Episodes | Treatment at or before Opportunity Deadline ($\le 22\text{h}$) | Treatment after Opportunity Deadline ($> 22\text{h}$) | Indeterminate (Interval Crosses Deadline) | Clinical & Measurement Specification |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **MIMIC-III v1.4** | Early Incident ($0 < T_{\text{S3}} \le 24\text{h}$) | 2,641 | 1,581 (59.9%) | 8 (0.3%) | 1,052 (39.8%) | Prescription start date proxy evaluated with $[00:00, 24:00)$ interval bounds. Deadline reflects $b=18\text{h}$ lookback $+ \delta=4\text{h}$ lead buffer. |
| *(Date Proxy, N=3,581)* | Late Incident ($T_{\text{S3}} > 24\text{h}$) | 940 | 748 (79.6%) | 9 (1.0%) | 183 (19.5%) | |
| | **All Incident Episodes** | **3,581** | **2,329 (65.0%)** | **17 (0.5%)** | **1,235 (34.5%)** | |
| **MIMIC-IV v3.1** | Early Incident ($0 < T_{\text{S3}} \le 24\text{h}$) | 1,236 | 1,235 (99.9%) | 1 (0.1%) | 0 (0.0%) | Confirmed hourly administration timestamps from electronic medication administration records (`emar`). |
| *(Hourly eMAR, N=1,596)* | Late Incident ($T_{\text{S3}} > 24\text{h}$) | 360 | 360 (100.0%) | 0 (0.0%) | 0 (0.0%) | |
| | **All Incident Episodes** | **1,596** | **1,595 (99.9%)** | **1 (0.1%)** | **0 (0.0%)** | |

---

In MIMIC-III, 34.5% of incident cases (1,235 / 3,581) have prescription start dates that cross the 22-hour opportunity threshold, demonstrating substantial measurement uncertainty that precludes fine-grained hourly claims. In contrast, MIMIC-IV administration records demonstrate that 99.9% of patients receiving systemic IV antibiotics in the ICU have treatment initiated within the first 22 hours of admission, establishing that the overwhelming majority of sepsis early-warning opportunities occur prior to or upon ICU admission rather than days into critical care monitoring.

---

### 3.5 Cumulative Population Coverage & Actionability Decomposition (Figure 3)
Evaluating the cumulative population coverage identity reveals the exact sources of population sensitivity attrition (**Figure 3**):
- **1. Base Incident Sepsis ($S=1$):** 408 / 408 (100.0%)
- **2. Surveillance Interval Covered ($V=1$):** 151 / 408 (37.0%) [Transition: $151 / 408 = 37.0\%$]
- **3. Pretreatment Actionable Opportunity ($O=1 \mid V=1$):** 150 / 408 (36.8%) [Transition: $150 / 151 = 99.3\%$]
- **4. Timely Detected ($D=1 \mid O=1, V=1$):** 6 / 408 (1.5%) [Transition: $6 / 150 = 4.0\%$]

$$\frac{151}{408} \times \frac{150}{151} \times \frac{6}{150} = \frac{6}{408} = 1.47\% \approx 1.5\%$$

1. **Surveillance Truncation ($151 / 408 = 37.0\%$):** 63.0% of sepsis patients (257 / 408) develop sepsis after hour 36, outside the prespecified second-day surveillance protocol (hours 19–30).
2. **Pre-treatment Actionability ($150 / 151 = 99.3\%$):** Among patients with covered windows, treatment has almost never occurred prematurely.
3. **Operational Model Detection ($6 / 150 = 4.0\%$):** Under an emitted alert budget of 10 episodes per 100 patient-days (407 emitted episodes), the Convex Ensemble uniquely detects 6 of the 150 actionable patients.

---

## 4. Discussion

### 4.1 Principal Insights: Why Models Fail at Deployment
The central finding of this investigation resolves the paradox between published retrospective discriminative success and real-world clinical failure:
1. **Pre-treatment Discrimination Is Preserved:** In directional consistency with Dutta et al. (2026), model discrimination (AUROC) does not collapse when evaluating true pre-treatment sepsis detection on risk sets excluding post-treatment windows. Models are capable of identifying physiological derangements ahead of antimicrobial intervention.
2. **The Collapse Is Operational and Workload-Driven:** When models are forced to operate under realistic clinical alert budgets ($\le 10$ emitted alert episodes per 100 patient-days), their clinical yield collapses completely. Even with smart 12-hour refractory deduplication, episode-level PPV remains between 1.0% and 1.7%, requiring clinicians to evaluate 58.1 to 67.8 independent alert episodes to detect a single septic patient.
3. **Surveillance Coverage Truncation Explains Missing Patients:** Sepsis in critical care does not adhere to narrow 12-hour evaluation windows; median onset occurs 49 hours into admission. In second-day surveillance benchmarks, over 60% of patients appear undetectable simply because their onset has not yet occurred.

### 4.2 Distinguishing Care-Process Forecasting from Sepsis Anticipation
A critical conceptual error in prior literature is **endpoint substitution**. When benchmarks predict "antibiotic initiation within 6 hours" across all patients, they evaluate care-process prediction (predicting clinician ordering behavior, including prophylactic prescribing in non-septic patients). This task is fundamentally distinct from pre-treatment sepsis detection. Machine learning benchmarks must maintain Sepsis-3 as the biological target while conditioning evaluation timestamps on decision availability.

---

## 5. Policy Contribution: The ASAIR Candidate Reporting Framework
We propose the **Actionable Sepsis AI Reporting (ASAIR)** framework as an essential extension to TRIPOD+AI (Collins et al., 2024):
1. **ASAIR-1 (Locked Alert-Budget Benchmarking):** All prospective evaluations must report sensitivity, PPV, and NNE at prespecified clinical alert budgets (e.g., 5, 10, 20 alerts/100 patient-days).
2. **ASAIR-2 (Smart Refractory Deduplication):** Evaluations must disclose explicit alert suppression policies to separate raw prediction-window alarms from clinical review episodes.
3. **ASAIR-3 (Three-Stage Population Decomposition):** Studies must report the decomposition of whole-population detection into surveillance coverage ($V$), pre-treatment actionability ($O$), and conditional model sensitivity ($D$).
4. **ASAIR-4 (Multi-Clock Measurement Hierarchy):** Timestamp precision must be disclosed, distinguishing confirmed administration (eMAR) from order entry and interval-censored date proxies.
5. **ASAIR-5 (Patient-Clustered Uncertainty):** Resampling procedures must cluster by patient/stay to prevent artificial narrowing of confidence intervals across repeated longitudinal windows.
6. **ASAIR-6 (Care-Process Separation):** Sepsis detection must not be substituted with medication-receipt forecasting.

---

### Limitations
This study evaluated a prespecified second-day ICU surveillance window (hours 19–30); full-stay prospective surveillance across multiple hospital systems remains essential to establish multi-center transportability. Furthermore, MIMIC-III prescription start dates exhibit substantial interval uncertainty, necessitating primary reliance on MIMIC-IV confirmed eMAR records.

### Conclusion
Nominal prediction lead time ahead of retrospective consensus definitions conceals the dual realities of restricted surveillance coverage and low operational alert yield. By benchmarking under locked alert budgets and enforcing multi-clock reporting, clinical informatics can build early-warning systems that deliver genuine bedside value.

---

## References
1. Singer M, et al. Sepsis-3. *JAMA*. 2016;315(8):801-810.
2. Seymour CW, et al. Assessment of Clinical Criteria for Sepsis. *JAMA*. 2016;315(8):762-774.
3. Weissman GE, et al. Sepsis Prediction Models Diverge from Clinician Treatment Times. *AMIA Annu Symp Proc*. 2024;2024:1215-1224.
4. Wong A, et al. External Validation of a Widely Implemented Sepsis Model. *JAMA Intern Med*. 2021;181(8):1065-1070.
5. Dutta S, et al. Performance of Sepsis Prediction Across Definitions. *JAMA Netw Open*. 2026;9(3):e265599.
6. Kumar A, et al. Duration of hypotension before initiation of effective antimicrobial therapy. *Crit Care Med*. 2006;34(6):1589-1596.
7. Moor M, et al. Foundation Models for Generalist Medical AI. *Nature*. 2023;616:259-265.
8. Collins GS, et al. TRIPOD+AI statement. *BMJ*. 2024;385:e078378.
