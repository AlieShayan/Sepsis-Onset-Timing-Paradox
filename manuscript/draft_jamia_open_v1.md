---
title: "Surveillance Coverage, Treatment Precedence, and Workload Constraints in Machine Learning Early Warning for Sepsis: An Operational Evaluation"
authors:
  - name: "Alireza Shayan"
    affiliation: "Department of Computer Science and Engineering & IT, School of Electrical and Computer Engineering, Shiraz University, Shiraz, Iran"
    email: "Ali404@hafez.shirazu.ac.ir"
  - name: "Somayeh Afrasiabi"
    affiliation: "Department of Computer Science and Engineering & IT, School of Electrical and Computer Engineering, Shiraz University, Shiraz, Iran"
    email: "afrasiabi@shirazu.ac.ir"
target_journal: "JAMIA (Journal of the American Medical Informatics Association) — Research and Applications"
article_type: "Original Research / Research and Applications"
word_count: 3950
tables: 4
figures: 3
supplementary_materials: 1
---

# Surveillance Coverage, Treatment Precedence, and Workload Constraints in Machine Learning Early Warning for Sepsis: An Operational Evaluation

**Authors:**  
Alireza Shayan$^1$ (Email: Ali404@hafez.shirazu.ac.ir)  
Somayeh Afrasiabi$^1$ (Email: afrasiabi@shirazu.ac.ir)  

$^1$ Department of Computer Science and Engineering & IT, School of Electrical and Computer Engineering, Shiraz University, Shiraz, Iran  
*(Corresponding Author: Somayeh Afrasiabi, afrasiabi@shirazu.ac.ir)*

## Abstract

**Objectives:** To evaluate how surveillance protocol coverage, treatment precedence, and emitted-alert constraints govern machine learning early warning for sepsis in an intensive care unit (ICU) cohort.

**Materials and Methods:** We retrospectively evaluated eight predictive architectures in 8,254 adult first ICU stays from the MIMIC-III MetaVision clinical database (2008–2012) lasting at least 30 hours under a prespecified second-day surveillance protocol (hours 19–30; 97,700 hourly windows; prevalence $\pi = 0.694\%$). The broader historical CareVue cohort provided development data, and broad MIMIC-III provided a separate prescription-date sensitivity analysis. We formalized an operational cascade (surveillance coverage, pretreatment opportunity, and alert-budget detection), evaluated pre-treatment risk sets excluding post-antimicrobial windows ($\mathcal{R}_0$ and $\mathcal{R}_2$), and estimated burden-matched operating frontiers under a 12-hour alert suppression rule.

**Results:** Among 408 consensus Sepsis-3 stays (4.94%), 257 (63.0%) had clinical onset after hour 36, precluding evaluation under the second-day protocol ($V=0$). For 150 of 151 covered stays (99.3%), decision windows preceded antimicrobial administration. Model discrimination remained stable on pretreatment risk sets (BiMamba SSM: AUROC 0.6433 on $\mathcal{R}_0$ vs. 0.6326 canonical). Under an alert budget of 10 episodes per 100 patient-days, episode PPV ranged from 0.98% to 1.72%, corresponding to 58.1 to 101.8 simulated evaluations per detected patient (NNE), with whole-cohort timely yield bounded between 0.98% and 1.72%.

**Discussion:** Retrospective discrimination alone does not characterize clinical early-warning performance. Low observed yield accompanied restricted surveillance coverage, modest discrimination, and the evaluated alert-budget policy; pretreatment restriction changed discrimination little.

**Conclusion:** Sepsis early-warning systems must report surveillance coverage, verify treatment precedence, and evaluate operating characteristics against explicit alert-budget scenarios.

**Keywords:** Sepsis Early Warning, Surveillance Truncation, Treatment Precedence, Alert Budget Frontier, State-Space Models.

---

## 1. Introduction

### 1.1 Background and Significance
Sepsis remains a leading cause of critical illness, inpatient mortality, and healthcare expenditure globally (Singer et al., 2016; Seymour et al., 2016). Clinical treatment urgency depends critically on presentation: the Surviving Sepsis Campaign international guidelines recommend immediate antimicrobial therapy, ideally within one hour, for possible septic shock or a high likelihood of sepsis, while recommending rapid diagnostic assessment for possible sepsis without shock (Evans et al., 2021; Kumar et al., 2006). To support prompt intervention, the biomedical informatics community has heavily invested in machine learning (ML) models for early warning. Scores of retrospective investigations have reported high discriminative capability, frequently reporting area under the receiver operating characteristic curve (AUROC) values between 0.80 and 0.88 ahead of consensus onset (Moor et al., 2021). However, published ICU prediction studies differ substantially in outcome implementation, temporal alignment, and evaluation design; consequently, reported AUROCs should not be interpreted as directly comparable measures of deployment readiness (Moor et al., 2021; Wang et al., 2025). Time-sensitive sepsis benchmarks have also incorporated utility functions that reward and penalize alerts according to their timing (Reyna et al., 2020). Such approaches reinforce the need to report temporal evaluation rules, although benchmark utility is distinct from measured clinical benefit.

Clinical evaluations have reported heterogeneous performance and outcome associations (Wong et al., 2021; Adams et al., 2022), while clinician-vignette studies have identified discrepancies between operational onset labels and recommended treatment timing (Weissman et al., 2024). External evaluation of the original Epic Sepsis Model demonstrated that reported performance depended heavily on the evaluation level and timing rules: hospitalization-level and prediction-level workload estimates were not interchangeable, emphasizing the need to report the exact unit of evaluation (Wong et al., 2021). Prospective evaluations of early-warning systems such as TREWS have found associations between timely alert confirmation and improved outcomes among selected patients identified prior to antibiotic initiation (Adams et al., 2022; Henry et al., 2022). However, because provider response was observational rather than randomized, these findings support potential clinical utility without establishing universal causal efficacy (Adams et al., 2022). 

Furthermore, ICU clinicians already encounter substantial alarm exposure from bedside physiological monitors (Drew et al., 2014). In other clinical decision-support settings, repeated alerts and work complexity have been associated with lower alert acceptance, supporting explicit reporting of notification burden (Ancker et al., 2017; Paine et al., 2016).

This disconnect between retrospective benchmarks and prospective clinical utility is commonly attributed to discriminative collapse: the conjecture that ML models fail because they cannot anticipate sepsis prior to clinician recognition or empiric treatment (Kamran et al., 2024). Pretreatment evaluation has previously been proposed to distinguish genuine advance warning from recognition of care processes already underway (Kamran et al., 2024). However, existing evaluations frequently conflate three distinct operational mechanisms:
1. **Surveillance Protocol Coverage ($V$):** Protocol surveillance windows (e.g., fixed observation periods) may miss episodes occurring outside the evaluated interval.
2. **Treatment Precedence ($O$):** Decision windows may occur after clinicians have already recognized infection and initiated antimicrobial therapy.
3. **Clinical Alert-Budget Constraints ($D$):** Even when models maintain pre-treatment discrimination, bedside alert-rate constraints bound real-world sensitivity and positive predictive value (PPV).

To distinguish these evaluation components, we conducted an operational evaluation across 8 representative predictive architectures using retrospectively extracted electronic health record (EHR) data from the Beth Israel Deaconess Medical Center (BIDMC) MIMIC-III database (v1.4; CareVue development and MetaVision evaluation) (Johnson et al., 2016). Our contributions are fourfold:
1. **Pretreatment Discrimination Benchmark:** We show that model discrimination does not collapse when evaluating true pre-treatment Sepsis-3 detection on risk sets excluding post-treatment windows ($\mathcal{R}_0$ and $\mathcal{R}_2$).
2. **Operational Alert-Budget Frontier:** We estimate retrospective burden-matched operating characteristics under a 12-hour alert suppression rule across prespecified workload scenarios (5, 10, and 20 emitted alert episodes per 100 patient-days).
3. **Three-Stage Population Decomposition:** We formalize the cascade $S \to V \to O \to D$, demonstrating that under the specified schedule, 63.0% of sepsis episodes develop onset outside the surveillance window, whereas 99.3% of covered patients remain pre-treatment eligible.
4. **Multi-Clock Timestamp Uncertainty Analysis:** We bound prescription-proxy opportunity using calendar-day intervals in historical MIMIC-III prescriptions, conditional on assigned Sepsis-3 timestamps and the observed surveillance schedule, and contrast them with confirmed bedside medication administration records (`inputevents_mv`) in the MetaVision evaluation cohort.

---

## 2. Methods

### 2.1 Study Design and Cohort Definition
We conducted a retrospective evaluation with sequential prediction replay using the MIMIC-III critical care repository (v1.4) from BIDMC (Johnson et al., 2016), evaluating transportability across clinical information systems from CareVue (2001–2008 development) to MetaVision (2008–2012 prospective-replay benchmark), with broad historical prescriptions providing a sensitivity analysis. Analyses used versioned extraction queries with dataset identifiers and checksums recorded in the release manifest. The study cohort was defined as adult patients ($\ge 18$ years) admitted to an intensive care unit with a total ICU length of stay $\text{LOS} \ge 30\text{h}$. To avoid within-patient clustering across multiple admissions, only the chronological first ICU stay per patient was retained. Complete cohort derivation schemas, exclusion criteria, train/validation/test partitions, feature dictionaries, missing-data rules, model architectures, and executable Sepsis-3 definitions are detailed in the Supplementary Methods Package.

Surveillance was evaluated under a prespecified second-day ICU surveillance protocol: predictions were generated at 12 scheduled hourly decision bins $[t-1, t]$ for $t \in \{19, 20, \dots, 30\}$ hours post-admission. The test partition comprised 8,254 unique patient ICU stays (97,700 evaluated windows, 4,070.8 patient-days of surveillance volume). Stays were censored at Sepsis-3 onset ($N=110$ stays, 623 windows) or upon observation-availability cessation prior to hour 30 under the prespecified feature-generation schedule ($N=180$ control stays, 725 windows; all cohort stays had total $\text{LOS} \ge 30\text{h}$). Reporting was organized in accordance with TRIPOD+AI and STROBE recommendations (Collins et al., 2024; von Elm et al., 2007).

### 2.2 Outcome Definition and Operational Clocks
Sepsis was defined according to a study-specific operational adaptation of consensus Sepsis-3 criteria (Singer et al., 2016; Seymour et al., 2016): an acute increase in Sequential Organ Failure Assessment (SOFA) score $\ge 2$ points over baseline ($\text{SOFA}_{\text{base}}$, evaluated over the 24-hour lookback $[T_{\text{susp}} - 24\text{h}, T_{\text{susp}}]$) within a clinical evaluation window $[T_{\text{susp}} - 24\text{h}, T_{\text{susp}} + 12\text{h}]$ relative to suspected infection (defined as blood culture specimen acquisition paired with systemic antimicrobial administration; detailed in Supplementary Section S1.5).

For each patient $i$, we tracked four distinct temporal clocks:
- $T_{\text{admit}, i}$: ICU admission timestamp ($t=0$).
- $T_{\text{treatment}, i}$: First systemic antimicrobial administration (intravenous, intramuscular, or enteral systemic route).
- $T_{\text{culture}, i}$: First blood culture specimen collection timestamp.
- $T_{\text{S3}, i}$: Retrospective consensus Sepsis-3 onset timestamp.

### 2.3 Risk Sets, Anticipation Horizons, and Pretreatment Eligibility
Let $\mathcal{C}$ denote the evaluated test cohort of unique ICU stays ($N = 8,254$), and let $\mathcal{T}_i \subseteq \{19, 20, \dots, 30\}$ denote the set of hourly decision bins evaluated for stay $i$. For patients who never develop consensus Sepsis-3 during hospitalization, we set $T_{\text{S3}, i} = \infty$.

The full evaluated early-warning risk set is defined as:
$$\mathcal{R} = \bigl\{ (i, t) : i \in \mathcal{C}, \; t \in \mathcal{T}_i, \; t < T_{\text{S3}, i} \bigr\}$$

The binary early-warning prediction target at anticipation horizon $H = 6\text{h}$ is:
$$Y_i(t; H) = \mathbf{1} \bigl\{ t < T_{\text{S3}, i} \le t + H \bigr\}$$

In the canonical unconstrained risk set ($\mathcal{R}_{\text{canonical}} = \mathcal{R}$), this definition yields exactly 97,700 evaluated hourly windows, comprising 678 horizon-positive observations ($Y=1$) and 97,022 negative observations ($Y=0$, prevalence $\pi = 0.694\%$).

To evaluate true pre-treatment detection without feature distortion from prior antimicrobial exposure, we defined the pre-treatment risk sets $\mathcal{R}_\delta$ for clinical buffer $\delta \ge 0$:
$$\mathcal{R}_\delta = \bigl\{ (i, t) \in \mathcal{R} : t + \delta \le T_{\text{treatment}, i} \bigr\}$$
where $T_{\text{treatment}, i} = \infty$ for stays with no documented qualifying antimicrobial administration in the electronic medication administration record (eMAR). Specifically, in the MetaVision evaluation cohort, stays without charted administrations across confirmed bedside medication administration records (`inputevents_mv`) are classified as having no documented administration during ICU stay; potential outpatient, pre-ICU emergency department, or unrecorded administrations are acknowledged as unobserved in retrospective EHR tables. In the broad historical MIMIC-III cohort, prescription orders (`prescriptions`) lacking sub-day administration timestamps are evaluated using calendar-day interval bounds $[00:00, 24:00)$. Windows occurring after treatment initiation ($t + \delta > T_{\text{treatment}, i}$) were excluded from both cases and controls. We evaluated $\mathcal{R}_0$ ($\delta=0\text{h}$, censoring windows ending after treatment) and $\mathcal{R}_2$ ($\delta=2\text{h}$, requiring a 2-hour pre-treatment actionability buffer).

### 2.4 Alert-Budget Operating Frontiers and Refractory Policy
To quantify simulated emitted-alert burden, we evaluated fixed alert budgets $q \in \{5.0, 10.0, 20.0\}$ emitted alert episodes per 100 surveillance patient-days. Under a 12-hour suppression rule, each stay emits at most one alert episode within the evaluated 12-hour interval ($N_{\text{episodes}, i} \le 1$).

We estimated retrospective burden-matched operating characteristics by selecting score cutoffs from the held-out score distribution. These operating points describe empirical performance at matched emitted-alert rates; they do not represent thresholds prospectively fixed for online deployment. Patient selection used the maximum score over the evaluated interval, whereas detection timing was assessed at the first eligible threshold crossing under the complete tie-selection rule:
$$t_{\mathrm{first}, i} = \min \bigl\{ t \in \mathcal{T}_i : \operatorname{Elig}_i(t; \tau, \mathcal{B}) = 1 \bigr\}$$
where $\mathcal{B}$ denotes the deterministic prioritization rule:
$$\operatorname{Elig}_i(t; \tau, \mathcal{B}) = \mathbf{1} \bigl\{ i \in \mathcal{S}_{\text{alerted}}(K) \;\land\; s_i(t) \ge \tau \bigr\}$$
with $\mathcal{S}_{\text{alerted}}(K)$ defined as the top $K$ patient replicas under the lexicographical sort $(-\max_{t} s_i(t), \;\text{ICUSTAY\_ID}, \;\text{copy\_idx})$.

### 2.5 Predictive Models and Statistical Inference
We evaluated 8 models spanning three major architectural paradigms:
1. **State-Space Models:** Bidirectional Mamba (BiMamba SSM) utilizing linear-time selective state-space layers (Gu & Dao, 2023; Gu et al., 2022).
2. **Deep Sequence Transformers:** Mask-Aware Clinical Transformer (incorporating explicit missingness masks) and Baseline Unmasked Transformer.
3. **Tabular and Ensemble Baselines:** Random Forest, XGBoost, LightGBM, Logistic Regression (L2), and a Multi-Model Stacking Ensemble.

Models consumed a causal 20-hour observation lookback $[t-20\text{h}, t)$ across 21 physiological time series. All models were trained on fixed pre-surveillance development partitions with hyperparameters and feature normalization frozen prior to this operational evaluation. Complete architecture details, hyperparameter grids, and the stacking protocol are detailed in the Supplementary Methods Package.

Uncertainty was quantified using patient-clustered paired bootstrapping ($B=1,000$ resamples). For Procedure B (Empirical Burden-Matched Bootstrap), exposure $E_b = \frac{1}{24}\sum_{i\in b} h_i$ and alert target $K_b = \operatorname{round}(q \cdot E_b / 100)$ were dynamically recomputed within each resample. NNE confidence intervals were derived by exact reciprocal transformation of unrounded PPV percentiles: $\text{NNE}_{\text{low}} = 100 / \text{PPV}_{\text{upper}}$ and $\text{NNE}_{\text{high}} = 100 / \text{PPV}_{\text{low}}$ (with $+\infty$ when $\text{PPV}_{\text{low}} = 0$).

---

## 3. Results

### 3.1 Cohort Progression and Denominator Accounting
Table 1 traces all 8,254 ICU admissions through surveillance eligibility, onset timing, lookback horizons, and pre-treatment actionability. Among 408 consensus Sepsis-3 stays, 257 (63.0\%) had clinical onset after hour 36 (median onset: 49.0h post-admission). Under the specified observation schedule and 6-hour horizon, these 257 stays had no evaluated horizon-positive decision time ($V=0$). The remaining 151 stays (37.0\%) had active protocol surveillance coverage ($V=1$). Of the 151 covered stays, 150 had at least one horizon-positive decision time at or before recorded treatment initiation (99.3% within-covered opportunity; 36.8% of total incident sepsis), providing definitive pre-treatment opportunity ($\mathcal{R}_0$).

### Table 1: Cohort Progression & Denominator Accounting Ledger
| Progression Stage / Cohort Partition | ICU Stays ($N$) | Evaluated Windows | Surveillance Volume | Proportion / Window Prevalence | Clinical Definition & Measurement Convention |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Base Prospective-Replay Test Cohort** | 8,254 | 97,700 (out of 99,048 sched.) | 4,070.8 pt-days (97,700 pt-hours) | 100.0% of cohort | Adult first ICU admissions conditioned on ICU stay $\ge 30\text{h}$. Evaluated at scheduled hourly decision bins $[t-1, t]$ at $t \in \{19\dots 30\}$. 1,348 windows right-censored (623 post-onset, 725 monitoring cessation). |
| **2. Non-Sepsis Control ICU Stays** | 7,846 | 93,427 neg. windows | 3,892.8 pt-days (93,427 pt-hours) | 95.06% of stays | Patients without consensus Sepsis-3 during hospitalization. All windows serve as outcome-negative surveillance (180 stays ceased hourly charting prior to hour 30 due to observation-availability cessation, accounting for 725 windows; all stays had total $\text{LOS} \ge 30\text{h}$). |
| **3. Consensus Sepsis-3 Stays ($S=1$)** | 408 | 4,273 (678 pos, 3,595 neg) | 178.0 pt-days (4,273 pt-hours) | 4.94% of stays | Patients meeting Sepsis-3 consensus criteria (blood culture + antibiotic + acute $\Delta\text{SOFA}\ge 2$). Median onset: hour 49.0. 110 stays censored at onset prior to hour 30 (623 windows). |
| **  3a. Late-Onset Sepsis (Onset $>36$h)** | 257 | 3,084 neg. windows | 128.5 pt-days (3,084 pt-hours) | 63.0% of septic stays | Onset occurs after hour 36. 6-hour lookback $[T_{\text{S3}}-6, T_{\text{S3}}]$ does not intersect surveillance window $[19, 30]$ ($V=0$). |
| **  3b. Surveillance-Covered Sepsis ($V=1$)** | 151 | 1,189 (678 pos, 511 neg) | 49.5 pt-days (1,189 pt-hours) | 37.0% of septic stays | Onset occurs between ICU hours 20 and 36, such that $[T_{\text{S3}}-6\text{h}, T_{\text{S3}}]$ intersects surveillance interval $[19, 30]$ ($V=1$). |
| **4. Canonical Sepsis-3 Risk Set ($H=6$h)** | 8,254 (151 horizon-pos) | 97,700 (678 pos / 97,022 neg) | 4,070.8 pt-days | 0.694% prevalence | Full unconstrained risk set ($t \in \mathcal{T}_i, \; t < T_{\text{S3}, i}$). |
| **5. Pretreatment Sepsis Risk Set $\mathcal{R}_0$ ($\delta=0$h)** | 5,967 (150 horizon-pos) | 70,192 (674 pos / 69,518 neg) | 2,924.7 pt-days | 0.960% prevalence | Pretreatment risk set censoring post-treatment windows ($t \le T_{\text{treatment}}$) for cases and controls. 150/151 covered patients maintain pre-treatment decision opportunities. |
| **6. Pretreatment Sepsis Risk Set $\mathcal{R}_2$ ($\delta=2$h)** | 5,967 (149 horizon-pos) | 70,129 (663 pos / 69,466 neg) | 2,922.0 pt-days | 0.945% prevalence | Pretreatment risk set requiring a 2-hour clinical buffer ($t + 2\text{h} \le T_{\text{treatment}}$) before antimicrobial therapy. 149/151 covered patients maintain actionable opportunities. |

### 3.2 Pretreatment Discrimination Benchmark
Table 2 presents discrimination across Canonical Sepsis-3, Pretreatment Sepsis ($\mathcal{R}_0$), and Pretreatment with 2h Buffer ($\mathcal{R}_2$). Model discrimination remained stable or slightly higher in pre-treatment risk sets (BiMamba SSM: AUROC 0.6326 canonical vs. 0.6433 on $\mathcal{R}_0$, paired $\Delta\text{AUROC} = +0.0107$, 95\% CI: +0.0060 to +0.0175; Stacking Ensemble: 0.6284 vs. 0.6351, paired $\Delta\text{AUROC} = +0.0067$). The increase in AUPRC (BiMamba: 1.209\% to 1.656\%) reflects higher baseline positive prevalence in the restricted set (0.694\% vs. 0.960\%). These results are directionally consistent with the antibiotic-order sensitivity findings of Dutta et al. (2026), though differences in population selection, predictive models, and clinical-action timestamps preclude treating this as a direct replication.

### Table 2: Benchmark Discrimination Across Pretreatment Tasks
| Model Architecture | Canonical Sepsis-3 ($H=6\text{h}$) | Pretreatment $\mathcal{R}_0$ ($\delta=0\text{h}$) | Pretreatment $\mathcal{R}_2$ ($\delta=2\text{h}$) | Paired $\Delta\text{AUROC}$ ($\mathcal{R}_0 - \text{Canon}$) |
| :--- | :---: | :---: | :---: | :---: |
| **BiMamba SSM (Proposed)** | 0.6326 (0.5872–0.6733) <br> *AUPRC: 1.209% (0.892–1.769)* | **0.6433** (0.5998–0.6835) <br> **1.656%** (1.235–2.329) | **0.6447** (0.6002–0.6860) <br> **1.647%** (1.220–2.331) | **+0.0107** (+0.0060 to +0.0175) |
| **Multi-Model Stacking Ensemble** | 0.6284 (0.5829–0.6699) <br> *AUPRC: 1.206% (0.882–1.748)* | 0.6351 (0.5897–0.6771) <br> 1.621% (1.207–2.263) | 0.6370 (0.5913–0.6799) <br> 1.613% (1.198–2.255) | +0.0067 (+0.0021 to +0.0137) |
| **Random Forest** | 0.6108 (0.5643–0.6535) <br> *AUPRC: 1.080% (0.821–1.434)* | 0.6143 (0.5684–0.6575) <br> 1.470% (1.138–1.920) | 0.6173 (0.5712–0.6612) <br> 1.461% (1.120–1.916) | +0.0034 (-0.0008 to +0.0101) |
| **Mask-Aware Clinical Transformer** | 0.5942 (0.5467–0.6410) <br> *AUPRC: 1.013% (0.766–1.303)* | 0.5961 (0.5490–0.6422) <br> 1.382% (1.048–1.760) | 0.5954 (0.5480–0.6426) <br> 1.365% (1.032–1.752) | +0.0020 (-0.0024 to +0.0083) |
| **Tabular GBDT (XGBoost)** | 0.5670 (0.5208–0.6102) <br> *AUPRC: 0.977% (0.739–1.357)* | 0.5643 (0.5186–0.6082) <br> 1.320% (1.008–1.817) | 0.5652 (0.5192–0.6099) <br> 1.313% (0.998–1.811) | -0.0027 (-0.0070 to +0.0038) |
| **Tabular GBDT (LightGBM)** | 0.5613 (0.5161–0.6026) <br> *AUPRC: 0.893% (0.694–1.195)* | 0.5587 (0.5139–0.6010) <br> 1.222% (0.957–1.629) | 0.5598 (0.5145–0.6033) <br> 1.214% (0.949–1.630) | -0.0026 (-0.0070 to +0.0038) |
| **Logistic Regression (L2)** | 0.5593 (0.5097–0.6048) <br> *AUPRC: 0.939% (0.716–1.218)* | 0.5654 (0.5157–0.6107) <br> 1.337% (1.012–1.740) | 0.5657 (0.5150–0.6127) <br> 1.325% (1.002–1.731) | +0.0061 (+0.0023 to +0.0116) |
| **Clinical Transformer (Baseline)** | 0.5239 (0.4808–0.5653) <br> *AUPRC: 0.808% (0.609–1.381)* | 0.5296 (0.4851–0.5712) <br> 1.147% (0.857–1.904) | 0.5286 (0.4834–0.5705) <br> 1.131% (0.845–1.894) | +0.0057 (+0.0019 to +0.0110) |

### 3.3 Observed Burden–Detection Trade-Off Under This Protocol
Table 3 and Figure 2 demonstrate the observed burden–detection trade-off: under prespecified alert budgets, clinical detection yield is bounded.

At an emitted workload of 10 alert episodes per 100 patient-days (407 emitted episodes across 4,070.8 patient-days):
- Across all 8 models, episode-level PPV ranged from 0.98% to 1.72%, corresponding to 58.1 to 101.8 simulated evaluations per detected patient ($\text{NNE}$).
- BiMamba SSM and LightGBM each uniquely detected **7 patients** (PPV = 1.72%, 95% CI: 0.49%–2.95%; NNE = 58.1, 95% CI: 33.9–203.5).
- Stacking Ensemble, XGBoost, and Random Forest each detected **6 patients** (PPV = 1.47%, NNE = 67.8).
- All qualifying emitted alerts preceded the recorded antimicrobial administration by at least two hours ($t_{\text{first}} + 2\text{h} \le T_{\text{treatment}}$).
- Model rankings varied across budgets: at 20 episodes/100 pt-days, XGBoost detected 15 patients (NNE = 54.3) and Mask-Aware Transformer detected 14 patients (NNE = 58.1), highlighting workload-dependent operating behavior.

### Table 3: Operational Burden-Matched Alert Frontiers Under 12h Refractory Suppression
| Budget Scenario | Model Architecture | Cutoff ($\tau$) | Alerts ($K$) | Episode PPV % [95% CI] | Det ($N$) | Sens. Cov % | Yield % | NNE [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5.0 ep / 100 pt-d** | Stacking Ensemble | 0.0145 | 204 | 2.94 (0.98–5.88)% | 6 | 3.97 (1.31–7.65)% | 1.47 (0.49–2.86)% | 34.0 (17.0–102.0) |
| | LightGBM | 0.0117 | 204 | 2.94 (0.98–5.39)% | 6 | 3.97 (1.26–7.10)% | 1.47 (0.47–2.61)% | 34.0 (18.5–102.0) |
| | BiMamba SSM | 0.0187 | 204 | 1.96 (0.49–4.45)% | 4 | 2.65 (0.65–6.20)% | 0.98 (0.24–2.31)% | 51.0 (22.5–204.0) |
| | XGBoost | 0.0122 | 204 | 1.96 (0.49–3.94)% | 4 | 2.65 (0.63–5.63)% | 0.98 (0.23–2.09)% | 51.0 (25.4–204.0) |
| | Random Forest | 0.0177 | 204 | 1.47 (0.00–3.45)% | 3 | 1.99 (0.00–4.49)% | 0.74 (0.00–1.70)% | 68.0 (29.0–$\infty$) |
| | Clinical Transformer | 0.0101 | 204 | 0.98 (0.00–2.46)% | 2 | 1.32 (0.00–3.50)% | 0.49 (0.00–1.27)% | 102.0 (40.6–$\infty$) |
| | Logistic Regression | 0.0103 | 204 | 0.98 (0.00–2.94)% | 2 | 1.32 (0.00–3.85)% | 0.49 (0.00–1.45)% | 102.0 (34.0–$\infty$) |
| | Mask-Aware Transformer | 0.0120 | 204 | 0.00 (0.00–0.00)%$^*$ | 0 | 0.00 (0.00–0.00)%$^*$ | 0.00 (0.00–0.00)%$^*$ | Undefined ($\infty$) |
| **10.0 ep / 100 pt-d** | BiMamba SSM | 0.0165 | 407 | 1.72 (0.49–2.95)% | 7 | 4.64 (1.33–8.16)% | 1.72 (0.49–2.93)% | 58.1 (33.9–203.5) |
| | LightGBM | 0.0110 | 407 | 1.72 (0.49–2.95)% | 7 | 4.64 (1.38–7.74)% | 1.72 (0.49–2.89)% | 58.1 (33.9–203.5) |
| | Stacking Ensemble | 0.0131 | 407 | 1.47 (0.25–2.46)% | 6 | 3.97 (0.70–6.98)% | 1.47 (0.26–2.54)% | 67.8 (40.7–407.0) |
| | XGBoost | 0.0114 | 407 | 1.47 (0.49–2.94)% | 6 | 3.97 (1.32–7.65)% | 1.47 (0.49–2.85)% | 67.8 (34.0–203.5) |
| | Random Forest | 0.0158 | 407 | 1.47 (0.49–2.70)% | 6 | 3.97 (1.26–7.29)% | 1.47 (0.47–2.70)% | 67.8 (37.0–204.0) |
| | Mask-Aware Transformer | 0.0110 | 407 | 1.23 (0.25–2.46)% | 5 | 3.31 (0.74–6.67)% | 1.23 (0.26–2.48)% | 81.4 (40.7–407.0) |
| | Clinical Transformer | 0.0094 | 407 | 1.23 (0.25–2.46)% | 5 | 3.31 (0.68–6.38)% | 1.23 (0.25–2.37)% | 81.4 (40.7–407.0) |
| | Logistic Regression | 0.0095 | 407 | 0.98 (0.25–1.97)% | 4 | 2.65 (0.59–5.51)% | 0.98 (0.23–2.07)% | 101.8 (50.8–408.0) |
| **20.0 ep / 100 pt-d** | XGBoost | 0.0105 | 814 | 1.84 (0.98–2.70)% | 15 | 9.93 (5.26–14.47)% | 3.68 (1.87–5.29)% | 54.3 (37.0–101.9) |
| | Mask-Aware Transformer | 0.0099 | 814 | 1.72 (0.74–2.58)% | 14 | 9.27 (4.54–13.77)% | 3.43 (1.59–5.19)% | 58.1 (38.8–135.7) |
| | Logistic Regression | 0.0088 | 814 | 1.35 (0.49–2.09)% | 11 | 7.28 (2.74–11.11)% | 2.70 (0.99–4.18)% | 74.0 (47.8–203.7) |
| | BiMamba SSM | 0.0142 | 814 | 1.23 (0.49–2.09)% | 10 | 6.62 (3.07–10.89)% | 2.45 (1.05–4.10)% | 81.4 (47.9–203.2) |
| | LightGBM | 0.0105 | 814 | 1.23 (0.61–2.09)% | 10 | 6.62 (3.11–11.04)% | 2.45 (1.15–4.10)% | 81.4 (47.9–163.0) |
| | Random Forest | 0.0131 | 814 | 1.11 (0.49–1.85)% | 9 | 5.96 (2.70–9.68)% | 2.21 (0.96–3.69)% | 90.4 (54.2–203.7) |
| | Stacking Ensemble | 0.0115 | 814 | 0.98 (0.37–1.72)% | 8 | 5.30 (2.04–9.15)% | 1.96 (0.74–3.43)% | 101.8 (58.1–271.3) |
| | Clinical Transformer | 0.0088 | 814 | 0.86 (0.25–1.48)% | 7 | 4.64 (1.51–8.18)% | 1.72 (0.52–3.04)% | 116.3 (67.8–407.0) |

*$^*$At zero observed detections, the empirical percentile-bootstrap interval was degenerate at zero. This does not establish zero population detection probability. NNE intervals derived by exact reciprocal transformation. 100% of detected patients satisfied pre-treatment timing buffers ($\delta=0\text{h}$ and $\delta=2\text{h}$).*

### 3.4 Two-Axis Subgroup and Timestamp Uncertainty Matrix
Table 4 presents the cross-tabulation of Sepsis Onset Timing against Treatment Timing. In the MetaVision prospective-replay cohort, confirmed bedside administration records (`inputevents_mv`) verify that 150 of 151 covered patients (99.3\%) and 36.8\% of total incident sepsis episodes had definitive pre-treatment prediction opportunity. In the broad historical MIMIC-III cohort, prescription start dates (`prescriptions`) lack sub-day precision, producing calendar-day intervals $[\text{date at midnight}, \text{next date at midnight})$ that generate 110 indeterminate cases across the cohort, bounding prescription-proxy opportunity between 0.03\% and 3.10\% conditional on assigned consensus Sepsis-3 onset times.

### Table 4: Two-Axis Subgroup and Timestamp Uncertainty Matrix
| Cohort & Timestamp Source | Sepsis Onset Category | Total ($N$) | Covered ($V=1$) | Eligible ($O=1$) | Indeterminate | Subgroup Bounds | Contribution to Cohort |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MetaVision Prospective-Replay (MIMIC-III)** | Early Incident ($\le 24\text{h}$) | 47 | 47 (100.0%) | 47 | 0 | [100.0%, 100.0%] | 47 / 408 (11.5%) |
| *(Hourly Confirmed Bedside Administration: inputevents_mv)* | Intermediate (24–48h) | 156 | 104 (66.7%) | 103 | 0 | [66.0%, 66.0%] | 103 / 408 (25.2%) |
| | Late Incident ($>48\text{h}$) | 205 | 0 (0.0%) | 0 | 0 | [0.0%, 0.0%] | 0 / 408 (0.0%) |
| | **All Incident Sepsis-3** | **408** | **151 (37.0%)** | **150** | **0** | **[36.8%, 36.8%]** | **150 / 408 (36.8%)** |
| **Broad Historical MIMIC-III Sepsis** | Early Incident ($\le 24\text{h}$) | 2,641 | 102 (3.9%) | 0 | 74 | [0.0%, 2.8%] | [0, 74] / 3,581 (0.00–2.07%) |
| *(Date-Level Prescription Proxy: prescriptions)* | Intermediate (24–48h) | 361 | 235 (65.1%) | 1 | 36 | [0.3%, 10.2%] | [1, 37] / 3,581 (0.03–1.03%) |
| | Late Incident ($>48\text{h}$) | 579 | 0 (0.0%) | 0 | 0 | [0.0%, 0.0%] | [0, 0] / 3,581 (0.00%) |
| | **All Incident Sepsis-3** | **3,581** | **337 (9.4%)** | **1** | **110** | **[0.03%, 3.10%]** | **[1, 111] / 3,581 (0.03–3.10%)** |

### 3.5 Cumulative Population Coverage Decomposition
Figure 3 illustrates the cumulative population coverage decomposition:
$$\frac{151}{408} \times \frac{150}{151} \times \frac{7}{150} = \frac{7}{408} = 1.72\%$$

1. **Surveillance Truncation ($151/408 = 37.0\%$):** 63.0\% of sepsis patients develop onset after hour 36, outside the prespecified second-day surveillance protocol.
2. **Pretreatment Actionability ($150/151 = 99.3\%$):** Of the 151 covered stays, 150 had at least one horizon-positive decision time at or before recorded treatment initiation.
3. **Workload Detection Yield ($7/150 = 4.67\%$):** Under an alert budget of 10 episodes/100 patient-days, BiMamba SSM detects 7 patients (1.72\% whole-population yield; NNE = 58.1).

---

## 4. Discussion

### 4.1 Interpretation of the Operational Trade-Offs
This investigation characterizes the operational trade-offs governing machine learning early warning for sepsis in ICU surveillance:
1. **Pre-treatment Discrimination Is Preserved:** In directional consistency with Dutta et al. (2026), model discrimination (AUROC) changed little when evaluating pre-treatment sepsis detection on risk sets excluding post-treatment windows ($\mathcal{R}_0$ and $\mathcal{R}_2$).
2. **Detection Yield Is Constrained by Workload and Prevalence:** AUROC values near 0.64 are modest. At 10 emitted episodes per 100 evaluation-bin patient-days, PPV ranged from 0.98% to 1.72%, corresponding to 58.1 to 101.8 simulated evaluations per detected patient ($\text{NNE}$); observed sensitivity was below 2%.
3. **Surveillance Protocol Coverage Explains Unreached Cases:** Under the evaluated second-day protocol, 63.0% of incident cases develop onset after the monitoring window terminates, preventing multi-hour warning.

### 4.2 Distinguishing Care-Process Forecasting from Sepsis Anticipation
A critical conceptual error in prior literature is **endpoint substitution**. When benchmarks predict "antibiotic initiation within 6 hours across all patients", they evaluate care-process prediction (predicting clinician ordering behavior, including prophylactic prescribing in non-septic patients). This task is fundamentally distinct from pre-treatment sepsis detection. Pretreatment sepsis evaluation should retain its specified sepsis outcome while restricting eligible decision times; replacing that outcome with antibiotic receipt defines a different prediction task.

Furthermore, repeated and low-information notifications have been associated with reduced clinician acceptance in other decision-support settings (Ancker et al., 2017). While we quantify simulated emitted-alert exposure and evaluation yield, our analysis did not measure cognitive overload, alert acceptance, or alarm fatigue directly (Ancker et al., 2017; Paine et al., 2016). Clinical impact also depends on whether alerts are reviewed and acted upon (Henry et al., 2022). Our evaluation estimates opportunities and simulated review burden; it does not estimate provider adoption or changes in treatment timing.

### 4.3 Limitations
The analysis estimates operating characteristics in a selected long-stay ICU cohort rather than clinical effectiveness. Burden-matched cutoffs were derived retrospectively from complete evaluated score trajectories, and identifier-based selection was required for tied boundary scores. Accordingly, the frontier does not represent a prospectively locked policy or guarantee a specified alert rate in online deployment.

Observed treatment precedence does not establish that infection was unrecognized or that earlier antimicrobial administration would have been appropriate. The study did not measure clinician response, review duration, treatment changes, alarm fatigue, or patient outcomes; NNE therefore represents a simulated evaluation count rather than measured clinical workload. Alarm-reduction interventions may suppress both nonactionable and clinically important signals; our short evaluation interval cannot establish the safety or optimal duration of a refractory policy during sustained surveillance (Paine et al., 2016). Finally, our cohorts originate from a single medical center (BIDMC) across EHR system transitions (CareVue to MetaVision); multi-center validation across diverse health systems remains essential.

---

## 5. Conclusion
In this restricted, long-stay ICU cohort, observed detection yield depended on surveillance coverage, baseline discrimination, and the policy used to constrain emitted alerts. Most horizon-positive observations remained pretreatment eligible, and model rankings varied across workload levels. Deployable sepsis early-warning informatics requires benchmarking under explicitly specified emitted-alert policies and retrospectively burden-matched operating points, deduplicating refractory alert workload, and enforcing multi-clock reporting.

---

## Declarations

### Ethics Approval and Consent to Participate
This study was conducted using de-identified data from the MIMIC-III (v1.4) database, collected at Beth Israel Deaconess Medical Center (BIDMC). Data collection was approved by the Institutional Review Boards of BIDMC and the Massachusetts Institute of Technology (MIT). The data were de-identified in accordance with Health Insurance Portability and Accountability Act (HIPAA) Safe Harbor standards; informed consent was waived by the reviewing IRBs. Access was granted under PhysioNet Credentialed Data Use Agreement.

### Competing Interests
The authors declare that they have no competing financial or non-financial interests.

### Funding
This research received no external grant or commercial funding.

### Authors' Contributions
AS conceived the study, developed the analytical framework, implemented the pipelines, performed the experiments, and drafted the manuscript. SA provided academic supervision, contributed to methodological design, critically revised the manuscript, and approved the final draft.

### AI Assistance Disclosure
During the preparation of this manuscript, the authors utilized large language model and AI assistance (including Perplexity AI and Anthropic/Google AI interfaces) for adversarial peer-review pre-submission auditing, statistical consistency cross-checking, text refinement, and LaTeX typesetting. The authors independently designed and executed all empirical modeling pipelines, statistical data extraction algorithms, and bootstrap evaluations. The authors reviewed, validated, and edited all text and artifacts, and take full responsibility for the integrity, methodology, and conclusions of this work.

### Code and Data Availability
All preprocessing pipelines, model configurations, and release manifests are publicly available under MIT License at [https://github.com/alireza/Sepsis-Onset-Timing-Paradox](https://github.com/alireza/Sepsis-Onset-Timing-Paradox) (analytical freeze commit `b0932ca`, release manifest `1.0.0-freeze`; submission packaging commits `1292103` and `38aac8b`). De-identified patient data are available from PhysioNet under the MIMIC-III Data Use Agreement.

---

## References

1. Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*. 2016;315(8):801–810. DOI: [10.1001/jama.2016.0287](https://doi.org/10.1001/jama.2016.0287)
2. Seymour CW, Liu VX, Iwashyna TJ, et al. Assessment of clinical criteria for sepsis: for the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*. 2016;315(8):762–774. DOI: [10.1001/jama.2016.0288](https://doi.org/10.1001/jama.2016.0288)
3. Evans L, Rhodes A, Alhazzani W, et al. Surviving sepsis campaign: international guidelines for management of sepsis and septic shock 2021. *Intensive Care Med*. 2021;47(11):1181–1247. DOI: [10.1007/s00134-021-06506-y](https://doi.org/10.1007/s00134-021-06506-y)
4. Kumar A, Roberts D, Wood KE, et al. Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock. *Crit Care Med*. 2006;34(6):1589–1596. DOI: [10.1097/01.CCM.0000217961.75225.E9](https://doi.org/10.1097/01.CCM.0000217961.75225.E9)
5. Moor M, Rieck B, Horn M, Jutzeler CR, Borgwardt K. Early prediction of sepsis in the ICU using machine learning: a systematic review. *Front Med*. 2021;8:607952. DOI: [10.3389/fmed.2021.607952](https://doi.org/10.3389/fmed.2021.607952)
6. Wang Z, Wang W, Sun C, et al. A methodological systematic review of validation and performance of sepsis real-time prediction models. *npj Digit Med*. 2025;8:190. DOI: [10.1038/s41746-025-01587-1](https://doi.org/10.1038/s41746-025-01587-1)
7. Reyna MA, Josef CS, Jeter R, et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. *Crit Care Med*. 2020;48(2):210–217. DOI: [10.1097/CCM.0000000000004145](https://doi.org/10.1097/CCM.0000000000004145)
8. Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. *JAMA Intern Med*. 2021;181(8):1065–1070. DOI: [10.1001/jamainternmed.2021.2626](https://doi.org/10.1001/jamainternmed.2021.2626)
9. Adams R, Henry KE, Sridharan A, et al. Prospective, multi-site study of patient outcomes after implementation of the TREWS machine learning-based early warning system for sepsis. *Nat Med*. 2022;28(7):1455–1460. DOI: [10.1038/s41591-022-01894-0](https://doi.org/10.1038/s41591-022-01894-0)
10. Henry KE, Adams R, Parent C, et al. Factors driving provider adoption of the TREWS machine learning-based early warning system and its effects on sepsis treatment timing. *Nat Med*. 2022;28(7):1447–1454. DOI: [10.1038/s41591-022-01895-z](https://doi.org/10.1038/s41591-022-01895-z)
11. Weissman GE, Hubbard RA, Himes BE, et al. Sepsis prediction models are trained on labels that diverge from clinician-recommended treatment times. *AMIA Annu Symp Proc*. 2024;2024:1215–1224.
12. Kamran F, Tjandra D, Heiler A, et al. Evaluation of sepsis prediction models before onset of treatment. *NEJM AI*. 2024;1(3):AIoa2300032. DOI: [10.1056/AIoa2300032](https://doi.org/10.1056/AIoa2300032)
13. Dutta S, McMurry R, Tasi MC, et al. Performance of a sepsis prediction model across different sepsis definitions. *JAMA Netw Open*. 2026;9(4):e265599. DOI: [10.1001/jamanetworkopen.2026.5599](https://doi.org/10.1001/jamanetworkopen.2026.5599)
14. Drew BJ, Harris P, Zègre-Hemsey JK, et al. Insights into the problem of alarm fatigue with physiologic monitor devices: a comprehensive observational study of consecutive intensive care unit patients. *PLoS One*. 2014;9(10):e110274. DOI: [10.1371/journal.pone.0110274](https://doi.org/10.1371/journal.pone.0110274)
15. Ancker JS, Edwards A, Nosal S, et al. Effects of workload, work complexity, and repeated alerts on alert fatigue in a clinical decision support system. *BMC Med Inform Decis Mak*. 2017;17(1):36. DOI: [10.1186/s12911-017-0430-8](https://doi.org/10.1186/s12911-017-0430-8)
16. Paine CW, Goel VV, Ely E, et al. Systematic review of physiologic monitor alarm characteristics and pragmatic interventions to reduce alarm frequency. *J Hosp Med*. 2016;11(2):136–144. DOI: [10.1002/jhm.2520](https://doi.org/10.1002/jhm.2520)
17. Gu A, Dao T. Mamba: linear-time sequence modeling with selective state spaces. *arXiv preprint*. 2023;arXiv:2312.00752. DOI: [10.48550/arXiv.2312.00752](https://doi.org/10.48550/arXiv.2312.00752)
18. Gu A, Goel K, Ré C. Efficiently modeling long sequences with structured state spaces. In: *International Conference on Learning Representations (ICLR)*; 2022.
19. Johnson A, Pollard T, Mark R. MIMIC-III Clinical Database, version 1.4. *PhysioNet*; 2016. DOI: [10.13026/C2XW26](https://doi.org/10.13026/C2XW26)
20. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. DOI: [10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378)
21. von Elm E, Altman DG, Egger M, et al. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies. *Lancet*. 2007;370(9596):1453–1457. DOI: [10.1016/S0140-6736(07)61602-X](https://doi.org/10.1016/S0140-6736(07)61602-X)
