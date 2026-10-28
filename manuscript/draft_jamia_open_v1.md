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
- **Ensemble Meta-Learner:** Multi-Model Stacking Ensemble.

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
| Progression Stage / Cohort Partition | ICU Stays ($N$) | Evaluated Windows | Surveillance Volume | Proportion / Window Prevalence | Clinical Definition & Measurement Convention |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Base Prospective-Replay Test Cohort** | 8,254 | 97,700 (out of 99,048 scheduled) | 4,070.8 evaluation-bin patient-days (97,700 patient-hours) | 100.0% of cohort | Adult first ICU admissions (approx. 2017–2019) conditioned on ICU stay >= 30h. Evaluated at up to 12 scheduled hourly decision bins [t-1, t] at t in {19..30}. 1,348 windows right-censored (623 post-sepsis onset, 725 monitoring cessation). |
| **2. Non-Sepsis Control ICU Stays** | 7,846 | 93,427 outcome-negative | 3,892.8 patient-days (93,427 patient-hours) | 95.06% of stays | Patients without consensus Sepsis-3 during hospitalization. All windows serve as outcome-negative surveillance (180 stays censored prior to hour 30 due to monitoring cessation / observation data availability limits, accounting for 725 windows). |
| **3. Consensus Sepsis-3 Stays ($S=1$)** | 408 | 4,273 windows (678 positive, 3,595 negative) | 178.0 patient-days (4,273 patient-hours) | 4.94% of stays | Patients meeting Sepsis-3 consensus criteria (blood culture + antibiotic + acute Delta-SOFA >= 2). Median onset: hour 49.0 (IQR: 30.0h–77.0h; range: 20h–369h). 110 stays censored at onset prior to hour 30 (623 windows). |
| **  3a. Late-Onset Sepsis (Onset $>36$h)** | 257 | 3,084 outcome-negative | 128.5 patient-days (3,084 patient-hours) | 63.0% of septic stays | Onset occurs after hour 36. 6-hour lookback [T_S3 - 6, T_S3] does not intersect surveillance window [19, 30]. All windows are outcome-negative baseline surveillance (V_i = 0). |
| **  3b. Surveillance-Covered Sepsis ($V=1$)** | 151 | 1,189 windows (678 positive, 511 negative) | 49.5 patient-days (1,189 patient-hours) | 37.0% of septic stays | Onset occurs between ICU hours 20 and 36, such that [T_S3 - 6h, T_S3] intersects surveillance interval [19, 30] (V_i = 1). |
| **4. Canonical Sepsis-3 Risk Set ($H=6$h)** | 8,254 (151 horizon-pos) | 97,700 (678 positive / 97,022 negative) | 4,070.8 patient-days | 0.694% window prevalence | Full unconstrained risk set. Target windows have end-time within 6 hours before Sepsis-3 onset (0 < T_S3 - t <= 6h). |
| **5. Pretreatment Sepsis Risk Set $\mathcal{R}_0$ ($\delta=0$h)** | 5,967 (150 horizon-pos) | 70,192 (674 positive / 69,518 negative) | 2,924.7 patient-days | 0.960% window prevalence | Pretreatment risk set censoring all post-treatment windows (w_end <= T_treatment) for both cases and controls. 150 out of 151 covered patients (99.3%) maintain >=1 pretreatment target window. |
| **6. Pretreatment Sepsis Risk Set $\mathcal{R}_2$ ($\delta=2$h)** | 5,967 (149 horizon-pos) | 70,129 (663 positive / 69,466 negative) | 2,922.0 patient-days | 0.945% window prevalence | Pretreatment risk set requiring a 2-hour clinical buffer (w_end <= T_treatment - 2h) before antimicrobial therapy. 149 out of 151 covered patients (98.7%) maintain >=1 actionable target window. |

---

*Scored Hourly-Bin Exposure Convention: Each prediction row corresponds to an hourly evaluation interval [t-1, t] scored at timestamp t in {19, 20, ..., 30}. Alerts are issued strictly at the bin endpoint t; a bin [18, 19] scored at t=19 does not imply that an alert could have been issued at hour 18. Monitored exposure is reported as scored hourly-bin patient-days.*

Among 408 consensus Sepsis-3 stays, the median onset occurred at **hour 49.0** after ICU admission (IQR: 30.0h to 77.0h; range: 20h to 369h). For 257 patients (63.0%), onset occurred after hour 36. During the evaluated surveillance protocol (hours 19–30), these patients are legitimately in pre-sepsis baseline surveillance, contributing outcome-negative surveillance windows. 

Exactly **151 septic patients** experienced onset between hours 20 and 36, such that their 6-hour pre-onset evaluation window [T_S3 - 6h, T_S3] intersected the active surveillance window, generating **678 positive evaluation windows** (prevalence pi = 0.694%). When post-treatment windows are censored across both cases and controls, the resulting pretreatment risk set R0 contains **70,192 windows** (674 positive, 69,518 negative; prevalence pi = 0.960%) across 5,967 contributing stays, with 150 of 151 covered patients (99.3%) retaining at least one pretreatment decision opportunity. Requiring a 2-hour clinical buffer (R2) retains **70,129 windows** (663 positive, 69,466 negative; prevalence pi = 0.945%) across 5,967 stays, with 149 of 151 covered patients (98.7%) retaining actionable opportunities.

*Surveillance Time & Decision Opportunity Accounting:* With up to 12 scheduled decision opportunities per stay across 8,254 stays, the scheduled surveillance volume is 99,048 hourly opportunities (4,127.0 patient-days). Exactly 1,348 windows were right-censored: 623 windows were truncated immediately upon clinical Sepsis-3 onset among 110 septic patients developing sepsis prior to hour 30, and 725 windows were truncated due to clinical observation cessation or monitoring termination prior to hour 30 among 180 control patients, yielding exactly 97,700 monitored patient-hours (4,070.8 patient-days).

---

### 3.2 True Pretreatment Sepsis Benchmark & Replication of Dutta et al. (Table 2)
Table 2 displays model discrimination across Canonical Sepsis-3 (H=6h) versus True Pretreatment Sepsis Detection (delta=0h) and Pretreatment Detection with a 2-hour response buffer (delta=2h).

---

### Table 2: Benchmark Discrimination Across Pre-Treatment Tasks (Held-Out Point Estimates and B=1,000 Patient-Clustered 95% CIs)
*Note on Estimands: Point estimates are calculated directly on the held-out test cohort (B=0); 95% confidence intervals are derived from B=1,000 patient-clustered bootstrap resamples. Baseline outcome prevalence is reported directly below each task heading.*

| Model Architecture | Canonical Sepsis-3 ($H=6\text{h}$)<br>Prev: 0.694%<br>AUROC [95% CI] | Canonical Sepsis-3 ($H=6\text{h}$)<br>Prev: 0.694%<br>AUPRC % [95% CI] | Pretreatment Sepsis ($\mathcal{R}_0$, $\delta=0\text{h}$)<br>Prev: 0.960%<br>AUROC [95% CI] | Pretreatment Sepsis ($\mathcal{R}_0$, $\delta=0\text{h}$)<br>Prev: 0.960%<br>AUPRC % [95% CI] | Pretreatment ($\mathcal{R}_2$, $\delta=2\text{h}$)<br>Prev: 0.945%<br>AUROC [95% CI] | Pretreatment ($\mathcal{R}_2$, $\delta=2\text{h}$)<br>Prev: 0.945%<br>AUPRC % [95% CI] | Paired $\Delta\text{AUROC}$ ($\mathcal{R}_0 - \text{Canonical}$)<br>[95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BiMamba SSM (Proposed)** | **0.6326 (0.5872–0.6733)** | **1.209% (0.892%–1.769%)** | **0.6433 (0.5998–0.6835)** | **1.656% (1.235%–2.329%)** | **0.6447 (0.6002–0.6860)** | **1.647% (1.220%–2.331%)** | **+0.0107 (+0.0060 to +0.0175)** |
| **Multi-Model Stacking Ensemble** | 0.6284 (0.5829–0.6699) | 1.206% (0.882%–1.748%) | 0.6351 (0.5897–0.6771) | 1.621% (1.207%–2.263%) | 0.6370 (0.5913–0.6799) | 1.613% (1.198%–2.255%) | +0.0067 (+0.0021 to +0.0137) |
| **Random Forest** | 0.6108 (0.5643–0.6535) | 1.080% (0.821%–1.434%) | 0.6143 (0.5684–0.6575) | 1.470% (1.138%–1.920%) | 0.6173 (0.5712–0.6612) | 1.461% (1.120%–1.916%) | +0.0034 (-0.0008 to +0.0101) |
| **Mask-Aware Clinical Transformer** | 0.5942 (0.5467–0.6410) | 1.013% (0.766%–1.303%) | 0.5961 (0.5490–0.6422) | 1.382% (1.048%–1.760%) | 0.5954 (0.5480–0.6426) | 1.365% (1.032%–1.752%) | +0.0020 (-0.0024 to +0.0083) |
| **Tabular GBDT (XGBoost)** | 0.5670 (0.5208–0.6102) | 0.977% (0.739%–1.357%) | 0.5643 (0.5186–0.6082) | 1.320% (1.008%–1.817%) | 0.5652 (0.5192–0.6099) | 1.313% (0.998%–1.811%) | -0.0027 (-0.0070 to +0.0038) |
| **Tabular GBDT (LightGBM)** | 0.5613 (0.5161–0.6026) | 0.893% (0.694%–1.195%) | 0.5587 (0.5139–0.6010) | 1.222% (0.957%–1.629%) | 0.5598 (0.5145–0.6033) | 1.214% (0.949%–1.630%) | -0.0026 (-0.0070 to +0.0038) |
| **Logistic Regression (L2)** | 0.5593 (0.5097–0.6048) | 0.939% (0.716%–1.218%) | 0.5654 (0.5157–0.6107) | 1.337% (1.012%–1.740%) | 0.5657 (0.5150–0.6127) | 1.325% (1.002%–1.731%) | +0.0061 (+0.0023 to +0.0116) |
| **Clinical Transformer (Baseline)** | 0.5239 (0.4808–0.5653) | 0.808% (0.609%–1.381%) | 0.5296 (0.4851–0.5712) | 1.147% (0.857%–1.904%) | 0.5286 (0.4834–0.5705) | 1.131% (0.845%–1.894%) | +0.0057 (+0.0019 to +0.0110) |

--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BiMamba SSM (Proposed)** | **0.6312 (0.5872–0.6733)** | **1.250% (0.892%–1.769%)** | **0.6421 (0.5998–0.6835)** | **1.701% (1.235%–2.329%)** | **0.6435 (0.6002–0.6860)** | **+0.0109 (+0.0060 to +0.0175)** |
| **Multi-Model Stacking Ensemble (Ensemble)** | 0.6271 (0.5829–0.6699) | 1.248% (0.882%–1.748%) | 0.6340 (0.5897–0.6771) | 1.667% (1.207%–2.263%) | 0.6360 (0.5913–0.6799) | +0.0069 (+0.0021 to +0.0137) |
| **Random Forest** | 0.6099 (0.5643–0.6535) | 1.113% (0.821%–1.434%) | 0.6135 (0.5684–0.6575) | 1.508% (1.138%–1.920%) | 0.6166 (0.5712–0.6612) | +0.0036 (-0.0008 to +0.0101) |
| **Mask-Aware Clinical Transformer** | 0.5933 (0.5467–0.6410) | 1.019% (0.766%–1.303%) | 0.5955 (0.5490–0.6422) | 1.389% (1.048%–1.760%) | 0.5948 (0.5480–0.6426) | +0.0022 (-0.0024 to +0.0083) |
| **Tabular GBDT (XGBoost)** | 0.5664 (0.5208–0.6102) | 1.010% (0.739%–1.357%) | 0.5639 (0.5186–0.6082) | 1.361% (1.008%–1.817%) | 0.5648 (0.5192–0.6099) | -0.0025 (-0.0070 to +0.0038) |
| **Tabular GBDT (LightGBM)** | 0.5607 (0.5161–0.6026) | 0.916% (0.694%–1.195%) | 0.5583 (0.5139–0.6010) | 1.253% (0.957%–1.629%) | 0.5594 (0.5145–0.6033) | -0.0024 (-0.0070 to +0.0038) |
| **Logistic Regression (L2)** | 0.5593 (0.5097–0.6048) | 0.951% (0.716%–1.218%) | 0.5656 (0.5157–0.6107) | 1.357% (1.012%–1.740%) | 0.5659 (0.5150–0.6127) | +0.0063 (+0.0023 to +0.0116) |
| **Clinical Transformer (Baseline)** | 0.5227 (0.4808–0.5653) | 0.856% (0.609%–1.381%) | 0.5285 (0.4851–0.5712) | 1.215% (0.857%–1.904%) | 0.5275 (0.4834–0.5705) | +0.0059 (+0.0019 to +0.0110) |

---

*Note: All differences Delta-AUROC and confidence intervals were computed on unrounded estimates across B=1,000 patient-clustered bootstrap iterations. Baseline outcome prevalence is reported directly below each task heading.*

Crucially, in contrast to earlier conjectures that pre-treatment evaluation causes severe discriminative collapse, discrimination was slightly higher in the restricted pretreatment risk sets R0 and R2 than in the canonical risk set (BiMamba SSM AUROC: 0.6312 canonical vs. 0.6421 on R0, paired Delta-AUROC = +0.0109, 95% CI: +0.0060 to +0.0175; Multi-Model Stacking Ensemble: 0.6271 canonical vs. 0.6340 on R0, paired Delta-AUROC = +0.0069, 95% CI: +0.0021 to +0.0137). 

However, this comparison does not distinguish treatment-related information effects from changes in the underlying case mix, time since admission, and outcome prevalence. The absolute increase in AUPRC (BiMamba: 1.250% canonical to 1.701% on R0) largely tracks the higher baseline positive prevalence in the restricted set (0.694% canonical vs. 0.960% in R0), with both estimates representing approximately 1.8x baseline prevalence. 

These antibiotic-administration-restricted findings are **directionally consistent with the antibiotic-order sensitivity analysis of Dutta et al. (2026)**, confirming that restricting predictions to pre-antimicrobial windows does not cause discriminative failure. Nevertheless, an AUROC near 0.64 represents modest discrimination in absolute terms, and directly contributes to low detection yield at constrained operational workloads alongside surveillance coverage and alert budgets.

---

### 3.3 Operational Alert-Budget Workload & 12h Refractory Evaluation (Table 3 & Figure 2)
Table 3 and Figure 2 illustrate the true clinical bottleneck: when models are deployed under prespecified operational alert budgets, their clinical yield is bounded by operational workload.

---

### Table 3: Operational Burden-Matched Alert Frontiers Under 12h Refractory Suppression
*Protocol & Estimand: Evaluated under Procedure B (Empirical Burden-Matched Bootstrap, B=1,000 resamples). At each resample, exposure E_b is recomputed as \frac{1}{24}\sum_{i\in b} h_i and the target alert count K_b is dynamically selected as \operatorname{round}(q \cdot E_b / 100) to maintain the prespecified rate frontier. Patient-clustered resampling preserves duplicate patient copies via deterministic secondary key sorting ([score, ICUSTAY_ID, copy_idx]). Under 12h refractory suppression across the surveillance window [19, 30], each patient emits at most one alert episode. Crucially, detections are evaluated at the first emitted alert window t_{\mathrm{first}} = \min\{t : \text{score}_t \ge \tau\}.*

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

*Note on Operational Mechanics and Alert Timing Verification: Workload NNE_patient = N_emitted episodes / N_true positive episodes. Crucially, timing verification at the first emitted alert t_{\mathrm{first}} confirmed that 100% of true-positive sepsis-horizon detections (0 < T_S3 - t_{\mathrm{first}} <= 6h) strictly preceded systemic antimicrobial administration (t_{\mathrm{first}} <= T_treatment) and maintained at least a 2-hour clinical response buffer (t_{\mathrm{first}} + 2h <= T_treatment).*

*Boundary and Zero-Event Footnote (*): For the Mask-Aware Transformer at 5 episodes/100 pt-days (0 observed detections), the empirical percentile-bootstrap interval was degenerate at zero [0.00%, 0.00%]; this empirical boundary result reflects bootstrap degeneracy and does not establish zero population detection probability. For reference, under an idealized fixed-policy Bernoulli model with zero observed successes, the exact one-sided 95% Clopper-Pearson upper confidence limit (1 - 0.05^{1/N}) is 1.46% (PPV, N=204), 1.96% (covered sensitivity, N=151), and 0.73% (overall yield, N=408), corresponding to the approximate Hanley Rule-of-Three (3/N). NNE confidence intervals are derived by exact reciprocal transformation of unrounded PPV percentiles; when the lower PPV limit reaches 0.00%, the upper NNE bound is mathematically +\infty (reported as +inf).*

*Boundary Tie Disclosure: For continuous deep models and Logistic Regression, scores are strictly continuous (tied count = 1 at cutoff). Discrete tree models exhibit leaf quantization: at 814 episodes, LightGBM has 2,636 stays tied at score 0.010537 (623 strictly above; 191 selected via deterministic ascending ICUSTAY_ID key to fulfill the exact 814 budget quota); XGBoost has 5 stays tied at 0.010526 (2 selected). At 204 episodes, the Stacking Ensemble has 2 stays tied at 0.014490 (1 selected), and Random Forest has 2 stays tied at 0.017658 (both selected).*

--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5.0 episodes / 100 pt-days** | **BiMamba SSM (Proposed)** | 0.0187 | 204 | 5.01 | 2.45 (0.49–4.90)% | 5 | 5 | 5 | 3.31 (0.70–6.67)% | 1.23 (0.25–2.52)% | 40.8 (20.4–204.0) | 0.005 |
| | **Multi-Model Stacking Ensemble (Ensemble)** | 0.0145 | 204 | 5.01 | 2.94 (0.98–5.39)% | 6 | 6 | 6 | 3.97 (1.26–7.38)% | 1.47 (0.48–2.72)% | 34.0 (18.5–102.0) | 0.005 |
| | **Tabular GBDT (LightGBM)** | 0.0117 | 204 | 5.01 | 2.94 (0.98–5.39)% | 6 | 6 | 6 | 3.97 (1.20–7.28)% | 1.47 (0.47–2.67)% | 34.0 (18.5–102.0) | 0.006 |
| | **Tabular GBDT (XGBoost)** | 0.0122 | 204 | 5.01 | 1.96 (0.49–3.92)% | 4 | 4 | 4 | 2.65 (0.61–5.44)% | 0.98 (0.23–2.01)% | 51.0 (25.5–204.0) | 0.017 |
| | **Random Forest** | 0.0177 | 204 | 5.01 | 1.47 (0.00–3.43)% | 3 | 3 | 3 | 1.99 (0.00–4.55)% | 0.74 (0.00–1.66)% | 68.0 (29.1–+inf) | 0.046 |
| | **Mask-Aware Transformer** | 0.0120 | 204 | 5.01 | 0.00 (0.00–1.47)%* | 0 | 0 | 0 | 0.00 (0.00–1.99)%* | 0.00 (0.00–0.74)%* | Undefined (+inf) | 0.988 |
| | **Clinical Transformer (Baseline)** | 0.0101 | 204 | 5.01 | 0.98 (0.00–2.45)% | 2 | 2 | 2 | 1.32 (0.00–3.29)% | 0.49 (0.00–1.23)% | 102.0 (40.8–+inf) | 0.143 |
| | **Logistic Regression (L2)** | 0.0103 | 204 | 5.01 | 0.98 (0.00–2.45)% | 2 | 2 | 2 | 1.32 (0.00–3.60)% | 0.49 (0.00–1.35)% | 102.0 (34.0–+inf) | 0.118 |
| **10.0 episodes / 100 pt-days** | **BiMamba SSM (Proposed)** | 0.0165 | 407 | 10.00 | 2.21 (0.74–3.69)% | 9 | 9 | 9 | 5.96 (2.17–9.53)% | 2.21 (0.77–3.49)% | 45.2 (27.1–135.7) | 0.002 |
| | **Multi-Model Stacking Ensemble (Ensemble)** | 0.0131 | 407 | 10.00 | 1.97 (0.74–3.44)% | 8 | 8 | 8 | 5.30 (2.07–8.97)% | 1.96 (0.76–3.31)% | 50.9 (29.1–135.7) | 0.001 |
| | **Tabular GBDT (LightGBM)** | 0.0110 | 407 | 10.00 | 1.72 (0.49–2.95)% | 7 | 7 | 7 | 4.64 (1.33–7.74)% | 1.72 (0.49–2.89)% | 58.1 (33.9–203.5) | 0.001 |
| | **Tabular GBDT (XGBoost)** | 0.0114 | 407 | 10.00 | 1.47 (0.49–2.70)% | 6 | 6 | 6 | 3.97 (1.30–7.48)% | 1.47 (0.48–2.76)% | 67.8 (37.0–203.5) | 0.003 |
| | **Random Forest** | 0.0158 | 407 | 10.00 | 1.47 (0.49–2.70)% | 6 | 6 | 6 | 3.97 (1.21–7.20)% | 1.47 (0.47–2.68)% | 67.8 (37.0–203.5) | 0.002 |
| | **Mask-Aware Transformer** | 0.0110 | 407 | 10.00 | 1.23 (0.25–2.46)% | 5 | 5 | 5 | 3.31 (0.68–6.37)% | 1.23 (0.25–2.39)% | 81.4 (40.7–407.0) | 0.005 |
| | **Clinical Transformer (Baseline)** | 0.0094 | 407 | 10.00 | 1.23 (0.25–2.46)% | 5 | 5 | 5 | 3.31 (0.69–6.58)% | 1.23 (0.25–2.43)% | 81.4 (40.7–407.0) | 0.006 |
| | **Logistic Regression (L2)** | 0.0095 | 407 | 10.00 | 0.98 (0.25–2.21)% | 4 | 4 | 4 | 2.65 (0.58–5.81)% | 0.98 (0.23–2.14)% | 101.8 (45.2–407.0) | 0.023 |
| **20.0 episodes / 100 pt-days** | **BiMamba SSM (Proposed)** | 0.0142 | 814 | 20.00 | 1.72 (0.86–2.70)% | 14 | 14 | 14 | 9.27 (5.26–14.01)% | 3.43 (1.81–5.21)% | 58.1 (37.0–116.3) | 0.000 |
| | **Multi-Model Stacking Ensemble (Ensemble)** | 0.0115 | 20.00 | 814 | 1.35 (0.73–2.21)% | 11 | 11 | 11 | 7.28 (3.70–11.64)% | 2.70 (1.37–4.34)% | 74.0 (45.2–136.3) | 0.000 |
| | **Tabular GBDT (LightGBM)** | 0.0105 | 20.00 | 814 | 1.23 (0.49–2.09)% | 10 | 10 | 10 | 6.62 (2.80–10.81)% | 2.45 (1.01–4.12)% | 81.4 (47.9–203.5) | 0.000 |
| | **Tabular GBDT (XGBoost)** | 0.0105 | 20.00 | 814 | 1.84 (0.98–2.70)% | 15 | 15 | 15 | 9.93 (5.19–14.39)% | 3.68 (1.88–5.37)% | 54.3 (37.0–102.1) | 0.000 |
| | **Random Forest** | 0.0131 | 20.00 | 814 | 1.11 (0.49–1.84)% | 9 | 9 | 9 | 5.96 (2.60–9.68)% | 2.21 (0.95–3.61)% | 90.4 (54.3–203.5) | 0.000 |
| | **Mask-Aware Transformer** | 0.0099 | 20.00 | 814 | 1.97 (0.98–2.83)% | 16 | 16 | 16 | 10.60 (5.41–15.17)% | 3.92 (1.96–5.77)% | 50.9 (35.4–101.8) | 0.000 |
| | **Clinical Transformer (Baseline)** | 0.0088 | 20.00 | 814 | 0.98 (0.37–1.72)% | 8 | 8 | 8 | 5.30 (2.11–9.16)% | 1.96 (0.75–3.39)% | 101.8 (58.1–271.3) | 0.000 |
| | **Logistic Regression (L2)** | 0.0088 | 20.00 | 814 | 1.47 (0.61–2.21)% | 12 | 12 | 12 | 7.95 (3.31–12.16)% | 2.94 (1.20–4.49)% | 67.8 (45.2–162.8) | 0.000 |

---

*Note on Operational Mechanics and Alert Timing Verification: Workload NNE_patient = N_emitted episodes / N_true positive episodes. Crucially, timing verification at the first emitted alert t_alert confirmed that 100% of true-positive sepsis-horizon detections (0 < T_S3 - t_alert <= 6h) strictly preceded systemic antimicrobial administration (t_alert <= T_treatment) and maintained at least a 2-hour clinical response buffer (t_alert + 2h <= T_treatment).*

*Boundary and Zero-Event Footnote (*): For the Mask-Aware Transformer at 5 episodes/100 pt-days (0 observed detections), the empirical bootstrap interval is degenerate at [0.00%, 0.00%]; we report the boundary-aware Hanley rule-of-three upper limit (3/204 = 1.47% for PPV; 3/151 = 1.99% for covered sensitivity; 3/408 = 0.74% for overall yield). For configurations where zero-detection resamples occur in >2.5% of replicates (f_0 > 0.025), the upper 95% confidence limit of NNE is strictly +inf, reflecting unbounded evaluation workload.*

At an emitted workload of 10 episodes per 100 surveillance patient-days:
- The 12-hour refractory policy bounds bedside reviews to exactly 407 unique patient evaluations across 4,070.8 patient-days.
- Across the entire test cohort of 408 septic patients, BiMamba SSM and LightGBM each uniquely detect **7 patients** (Sensitivity = 1.72%, 95% CI: 0.49%–2.89%; PPV = 1.72%, 95% CI: 0.49%–2.95%), requiring clinicians to review **58.1 independent alert episodes** for every single uniquely detected septic patient (NNE = 58.1, 95% CI: 33.9–203.5).
- The Multi-Model Stacking Ensemble detects **6 patients** (Sensitivity = 1.47%, 95% CI: 0.26%–2.54%; PPV = 1.47%, 95% CI: 0.25%–2.46%), requiring **67.8 alert episode evaluations per detected patient** (NNE = 67.8, 95% CI: 40.7–407.0).
- At 20 episodes per 100 patient-days (814 emitted alert episodes), XGBoost achieves the largest detection count (15 detected patients, PPV = 1.84%, NNE = 54.3), followed by the Mask-Aware Transformer (14 detected patients, PPV = 1.72%, NNE = 58.1), demonstrating that empirical model rankings depend heavily on operational workload.

---

### 3.4 Two-Axis Subgroup Matrix & Timestamp Uncertainty Bounds (Table 4)
Table 4 presents the two-axis cross-tabulation of Sepsis Onset Timing against Treatment Timing, strictly implementing the executable opportunity definition:
$$\mathcal{A}_i = \{ t \in \mathcal{T}_i : 0 < T_{\text{S3}, i} - t \le H \}, \quad O_i(\delta) = \mathbf{1} \{ \exists t \in \mathcal{A}_i : t + \delta \le T_{\text{treatment}, i} \}$$

---

### Table 4: Two-Axis Subgroup & Timestamp Uncertainty Matrix (Opportunity Bounds Conditional on Fixed Sepsis-3 Consensus Timestamps)
| Cohort & Timestamp Source | Sepsis Onset Category | Subgroup Denominator ($N$) | Surveillance Covered Stays ($V=1$) | Definitely Eligible Count | Indeterminate Count | Surveillance Uncovered ($V=0$) | Covered but Ineligible ($V=1, O=0$) | Within-Subgroup Opportunity Bounds | Within-Covered Opportunity Bounds | Contribution to Overall Cohort Bound |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MIMIC-IV Prospective-Replay** | Early Incident (<=24h) | 47 | 47 (100.0%) | 47 | 0 | 0 | 0 | [100.0%, 100.0%] | [100.0%, 100.0%] | 47 / 408 (11.5%) |
| *(Hourly Confirmed eMAR)* | Intermediate Incident (24–48h) | 156 | 104 (66.7%) | 103 | 0 | 52 | 1 | [66.0%, 66.0%] | [99.0%, 99.0%] | 103 / 408 (25.2%) |
| | Late Incident (>48h) | 205 | 0 (0.0%) | 0 | 0 | 205 | 0 | [0.0%, 0.0%] | N/A (No covered stays) | 0 / 408 (0.0%) |
| | **All Incident Sepsis-3 Cases** | **408** | **151 (37.0%)** | **150** | **0** | **257** | **1** | **[36.8%, 36.8%]** | **[99.3%, 99.3%]** | **150 / 408 (36.8%)** |
| **MIMIC-III Broad Incident Sepsis** | Early Incident (<=24h) | 2,641 | 102 (3.9%) | 0 | 74 | 2,539 | 28 | [0.0%, 2.8%] | [0.0%, 72.5%] | [0, 74] / 3,581 ([0.00%, 2.07%]) |
| *(Prescription Date Proxy)* | Intermediate Incident (24–48h) | 361 | 235 (65.1%) | 1 | 36 | 126 | 198 | [0.3%, 10.2%] | [0.4%, 15.7%] | [1, 37] / 3,581 ([0.03%, 1.03%]) |
| | Late Incident (>48h) | 579 | 0 (0.0%) | 0 | 0 | 579 | 0 | [0.0%, 0.0%] | N/A (No covered stays) | [0, 0] / 3,581 ([0.00%, 0.00%]) |
| | **All Incident Sepsis-3 Cases** | **3,581** | **337 (9.4%)** | **1** | **110** | **3,244** | **226** | **[0.03%, 3.10%]** | **[0.30%, 32.94%]** | **[1, 111] / 3,581 ([0.03%, 3.10%])** |

---

*Note on Measurement Interpretation and Reference Clock Uncertainty: In MIMIC-III, prescription start dates lack sub-day timestamps, creating an interval [a_i, b_i) = [STARTDATE 00:00:00, STARTDATE + 1 day 00:00:00). This produces 110 indeterminate cases across the cohort where treatment may or may not have preceded eligible prediction windows, bounding true clinical opportunity between 0.03% (worst-case) and 3.10% (best-case). Bounds on opportunity counts do not automatically translate to bounds on AUROC without explicit assumptions regarding intra-day treatment timing distributions. Furthermore, because suspicion of infection in Sepsis-3 depends on antimicrobial timing, date-level prescription uncertainty in MIMIC-III also affects the reference onset clock T_S3 itself; displayed bounds are valid conditional on the recorded onset timestamps. In contrast, MIMIC-IV electronic medication administration records (eMAR) provide verified hourly administration records, showing that 150 of 151 covered patients (99.3%) and 36.8% of the overall sepsis cohort have definitive pre-treatment prediction opportunities.*

---

### 3.5 Cumulative Population Coverage & Actionability Decomposition (Figure 3)
Evaluating the cumulative population coverage identity reveals the exact sources of population sensitivity attrition (**Figure 3**):

**Primary Zero-Buffer Waterfall (delta = 0h):**
- **1. Base Incident Sepsis ($S=1$):** 408 / 408 (100.0%)
- **2. Surveillance Interval Covered ($V=1$):** 151 / 408 (37.0%) [Transition: $151 / 408 = 37.01\%$]
- **3. Pretreatment Actionable Opportunity ($O_0=1 \mid V=1$):** 150 / 408 (36.8%) [Transition: $150 / 151 = 99.34\%$]
- **4. Timely Detected by Model ($D=1 \mid O_0=1, V=1$):** 7 / 408 (1.72%) [Transition: $7 / 150 = 4.67\%$ for BiMamba SSM; 6 / 150 = 4.00% for Stacking Ensemble]

$$\frac{151}{408} \times \frac{150}{151} \times \frac{7}{150} = \frac{7}{408} = 1.72\%$$

**Secondary Two-Hour Buffer Waterfall (delta = 2h):**
- **1. Base Incident Sepsis ($S=1$):** 408 / 408 (100.0%)
- **2. Surveillance Interval Covered ($V=1$):** 151 / 408 (37.0%) [Transition: $151 / 408 = 37.01\%$]
- **3. Pretreatment Actionable Opportunity ($O_2=1 \mid V=1$):** 149 / 408 (36.5%) [Transition: $149 / 151 = 98.68\%$]
- **4. Timely Detected with 2h Lead Time ($D=1 \mid O_2=1, V=1$):** 7 / 408 (1.72%) [Transition: $7 / 149 = 4.70\%$ for BiMamba SSM; 6 / 149 = 4.03% for Stacking Ensemble]

1. **Surveillance Truncation ($151 / 408 = 37.0\%$):** 63.0% of sepsis patients (257 / 408) develop sepsis after hour 36, outside the prespecified second-day surveillance protocol (hours 19–30).
2. **Pre-treatment Actionability ($150 / 151 = 99.3\%$ for delta=0h; $149 / 151 = 98.7\%$ for delta=2h):** Among patients with covered windows, treatment has almost never occurred prematurely.
3. **Operational Model Detection ($7 / 150 = 4.7\%$):** Under an emitted alert budget of 10 episodes per 100 patient-days (407 emitted episodes), BiMamba SSM uniquely detects 7 patients (1.72% whole-population yield; NNE = 58.1), and the Stacking Ensemble uniquely detects 6 patients (1.47% whole-population yield; NNE = 67.8).

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
In this restricted, long-stay ICU cohort, low empirical detection yield reflected limited surveillance coverage, modest discrimination, and the operating policy used to constrain emitted alerts. Most horizon-positive observations remained pretreatment eligible, and observed model rankings depended on workload. By benchmarking under locked alert budgets and enforcing multi-clock reporting, clinical informatics can build early-warning systems that deliver genuine bedside value.

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
