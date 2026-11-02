---
title: "Consensus Onset Mechanics, Timestamp Resolution, and Workload Constraints in Sepsis Early-Warning Benchmarks: An Operational Evaluation Audit"
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

# Consensus Onset Mechanics, Timestamp Resolution, and Workload Constraints in Sepsis Early-Warning Benchmarks: An Operational Evaluation Audit

**Authors:**  
Alireza Shayan$^1$ (Email: Ali404@hafez.shirazu.ac.ir)  
Somayeh Afrasiabi$^1$ (Email: afrasiabi@shirazu.ac.ir)  

$^1$ Department of Computer Science and Engineering & IT, School of Electrical and Computer Engineering, Shiraz University, Shiraz, Iran  
*(Corresponding Author: Somayeh Afrasiabi, afrasiabi@shirazu.ac.ir)*

## Abstract

**Objectives:** To evaluate how surveillance coverage, anchor selection, and alert constraints govern machine learning early warning for sepsis in ICU surveillance.

**Materials and Methods:** Using MIMIC-III as a reproducible benchmark substrate spanning an inter-era EHR vendor transition from CareVue (2001–2008 development) to MetaVision (2008–2012 prospective-replay audit; 8,254 first stays; 97,700 windows; prevalence 0.694%), we evaluated 8 predictive architectures during hours 19–30. We formalized an operational cascade ($S \to V \to O \to D$), evaluated outcome-stratified hybrid sets ($\mathcal{R}_0^{\mathrm{hybrid}}, \mathcal{R}_2^{\mathrm{hybrid}}$), and estimated burden-matched frontiers under 12-hour alert suppression.

**Results:** Among 408 Sepsis-3 stays, 257 (63.0%) developed onset after hour 36 ($V=0$). Discrimination changed little in the outcome-stratified hybrid analysis (BiMamba SSM: AUROC 0.6433 on $\mathcal{R}_0^{\mathrm{hybrid}}$ vs. 0.6326 canonical). Because cases and controls were restricted using different anchors, this finding does not establish preserved discrimination under a uniform pre-suspicion or pre-prescription policy. Within this restricted surveillance protocol and under declared midnight assignment of prescription-start dates, no horizon-positive observations remained under uniform earliest-prescription eligibility; discrimination was therefore not estimable for that task. At an alert budget of 10 episodes/100 patient-days, episode PPV was 0.98%–1.72% (NNE: 58.1–101.8), with timely yield 0.98%–1.72%.

**Discussion:** Canonical surveillance quantified population coverage and alert yield. Hybrid stability reflects outcome-specific censoring rather than uniform prospective policy.

**Conclusion:** Sepsis predictive evaluations must report protocol coverage, avoid outcome-dependent evaluation rules, and benchmark against explicit alert budgets.

**Keywords:** Sepsis Early Warning, Surveillance Truncation, Anchor Selection, Alert Budget Frontier, State-Space Models.

---

## 1. Introduction

### 1.1 Background and Significance
Sepsis remains a leading cause of critical illness, inpatient mortality, and healthcare expenditure globally (Singer et al., 2016; Seymour et al., 2016). Clinical treatment urgency depends critically on presentation: the Surviving Sepsis Campaign international guidelines recommend immediate antimicrobial therapy, ideally within one hour, for possible septic shock or a high likelihood of sepsis, while recommending rapid diagnostic assessment for possible sepsis without shock (Evans et al., 2021; Kumar et al., 2006). To support prompt intervention, the biomedical informatics community has heavily invested in machine learning (ML) models for early warning. Scores of retrospective investigations have reported high discriminative capability, frequently reporting area under the receiver operating characteristic curve (AUROC) values between 0.80 and 0.88 ahead of consensus onset (Moor et al., 2021). However, published ICU prediction studies differ substantially in outcome implementation, temporal alignment, and evaluation design; consequently, reported AUROCs should not be interpreted as directly comparable measures of deployment readiness (Moor et al., 2021; Wang et al., 2025). Prior MIMIC-III analyses have shown that alternative operational onset definitions can change measured performance as much as or more than changing model class (Cohen et al., 2024). These findings motivate separating reference-event construction from surveillance coverage and alert-policy evaluation. Time-sensitive sepsis benchmarks have also incorporated utility functions that reward and penalize alerts according to their timing (Reyna et al., 2020). Such approaches reinforce the need to report temporal evaluation rules, although benchmark utility is distinct from measured clinical benefit.

Clinical evaluations have reported heterogeneous performance and outcome associations (Wong et al., 2021; Adams et al., 2022), while clinician-vignette studies have identified discrepancies between operational onset labels and recommended treatment timing (Weissman et al., 2024). External evaluation of the original Epic Sepsis Model demonstrated that reported performance depended heavily on the evaluation level and timing rules: hospitalization-level and prediction-level workload estimates were not interchangeable, emphasizing the need to report the exact unit of evaluation (Wong et al., 2021). Prospective evaluations of early-warning systems such as TREWS have found associations between timely alert confirmation and improved outcomes among selected patients identified prior to antibiotic initiation (Adams et al., 2022; Henry et al., 2022). However, because provider response was observational rather than randomized, these findings support potential clinical utility without establishing universal causal efficacy (Adams et al., 2022). 

Furthermore, ICU clinicians already encounter substantial alarm exposure from bedside physiological monitors (Drew et al., 2014). In other clinical decision-support settings, repeated alerts and work complexity have been associated with lower alert acceptance, supporting explicit reporting of notification burden (Ancker et al., 2017; Paine et al., 2016).

This disconnect between retrospective benchmarks and prospective clinical utility is commonly attributed to discriminative collapse: the conjecture that ML models fail because they cannot anticipate sepsis prior to clinician recognition or empiric treatment (Kamran et al., 2024). Pretreatment evaluation has previously been proposed to distinguish genuine advance warning from recognition of care processes already underway (Kamran et al., 2024). However, existing evaluations frequently conflate three distinct operational mechanisms:
1. **Surveillance Protocol Coverage ($V$):** Protocol surveillance windows (e.g., fixed observation periods) may miss episodes occurring outside the evaluated interval.
2. **Treatment Precedence ($O$):** Decision windows may occur after clinicians have already recognized infection and initiated antimicrobial therapy.
3. **Clinical Alert-Budget Constraints ($D$):** Even when discrimination changes little in restricted analyses, bedside alert-rate constraints bound real-world sensitivity and positive predictive value (PPV).

To distinguish these evaluation components, we conducted an operational evaluation across 8 representative predictive architectures using retrospectively extracted electronic health record (EHR) data from the Beth Israel Deaconess Medical Center (BIDMC) MIMIC-III database (v1.4; CareVue development and MetaVision evaluation) (Johnson et al., 2016). Auditing published benchmarks requires examining the open, schema-documented computational substrate on which consensus phenotyping rules and prediction pipelines are routinely reproduced. We therefore evaluate MIMIC-III not to propose a deployable predictive model, but to isolate how retrospective onset conventions, timestamp resolution, and surveillance eligibility shape apparent early warning before cross-institutional variance is introduced. Our contributions are fourfold:
1. **Outcome-Stratified Hybrid Discrimination Analysis:** We evaluate discrimination on outcome-stratified hybrid sets ($\mathcal{R}_0^{\mathrm{hybrid}}$ and $\mathcal{R}_2^{\mathrm{hybrid}}$), demonstrating that while discrimination changed little in this restricted sample, because cases and controls were restricted using different anchors, this finding does not establish preserved discrimination under a uniform pre-suspicion or pre-prescription policy.
2. **Operational Alert-Budget Frontier:** We estimate retrospective burden-matched operating characteristics under a 12-hour alert suppression rule across prespecified workload scenarios (5, 10, and 20 emitted alert episodes per 100 patient-days).
3. **Three-Stage Population Decomposition and Evaluability:** We formalize the cascade $S \to V \to O \to D$, showing that under the second-day protocol, 63.0% of sepsis stays develop onset after surveillance, while evaluability under pre-exposure constraints is anchor-dependent (99.3% under $T_{\mathrm{susp}}$ vs. zero positive observations under uniform $T_{\mathrm{Rx,start}}$).
4. **Multi-Clock Timestamp Uncertainty Analysis:** We bound prescription-proxy opportunity using calendar-day intervals in historical MIMIC-III prescriptions, conditional on assigned Sepsis-3 timestamps and the observed surveillance schedule, and contrast interval bounds with declared point-assigned prescription proxies.

---

## 2. Methods

### 2.1 Study Design and Cohort Definition
We structured development and evaluation as an inter-era EHR vendor-transition audit using MIMIC-III (v1.4) (Johnson et al., 2016): CareVue-era candidate stays (2001–2008; $N=12,964$ stays; $10,371$ train, $2,593$ validation; $1,687$ incident sepsis) supported upstream development, whereas MetaVision-era stays (2008–2012; $N=8,254$ analyzed first adult ICU stays; $9,732$ candidates; $1,886$ sepsis) supported prospective-replay auditing across documentation and timestamping transitions within one institution. Adult first ICU stays with $\text{LOS} \ge 30\text{h}$ were included. Conditioning on $\text{LOS} \ge 30\text{h}$ ensures that included patients remain hospitalized through mid-stay surveillance (hours 19–30), but at surveillance entry ($t=19\text{h}$) it retrospectively conditions on 11 hours of future ICU survival. Stays with community-onset sepsis ($T_{\text{S3}} \le 0\text{h}$) or developing sepsis during the pre-surveillance baseline ($0 < T_{\text{S3}} \le 19\text{h}$, $N=1,478$) were excluded, yielding the 8,254 test stays ($408$ incident sepsis, $7,846$ controls). Beyond the primary MetaVision evaluation, historical sensitivity analyses evaluated $N=3,581$ incident sepsis stays partitioned into CareVue-only ($1,687$), MetaVision-only ($1,886$), and transitional stays bridging both database systems ($8$), where $1,687 + 1,886 + 8 = 3,581$ stays form a mutually exclusive, exhaustive partition. Reporting follows TRIPOD+AI and STROBE recommendations (Collins et al., 2024; von Elm et al., 2007).

Surveillance was evaluated under a prespecified second-day ICU surveillance protocol: predictions were generated at 12 scheduled hourly decision bins $[t-1, t]$ for $t \in \{19, 20, \dots, 30\}$ hours post-admission. The test partition comprised 8,254 unique patient ICU stays (97,700 evaluated windows, 4,070.8 patient-days of surveillance volume). Stays were censored at Sepsis-3 onset ($N=110$ stays, 623 windows) or upon observation-availability cessation prior to hour 30 under the prespecified feature-generation schedule ($N=180$ control stays, 725 windows; all cohort stays had total $\text{LOS} \ge 30\text{h}$). Reporting was organized in accordance with TRIPOD+AI and STROBE recommendations (Collins et al., 2024; von Elm et al., 2007).

### 2.2 Outcome Definition and Operational Clocks
Sepsis was defined according to a study-specific operational adaptation of consensus Sepsis-3 criteria (Singer et al., 2016; Seymour et al., 2016): an acute increase in Sequential Organ Failure Assessment (SOFA) score $\ge 2$ points over baseline ($\text{SOFA}_{\text{base}}$, evaluated over the 24-hour lookback $[T_{\text{susp}} - 24\text{h}, T_{\text{susp}}]$) within a clinical evaluation window $[T_{\text{susp}} - 24\text{h}, T_{\text{susp}} + 12\text{h}]$ relative to suspected infection (defined as blood culture specimen acquisition paired with systemic antimicrobial administration; detailed in Supplementary Section S1.5).

To resolve temporal precision and event ambiguity without conflating orders, intended start dates, and physical administrations, we track six operational milestones (with ICU admission as the reference origin $t=0$, followed by five operational event clocks; Figure 1):
- $T_{\text{admit}, i}$: ICU admission timestamp ($t=0$, reference origin).
- $T_{\mathrm{Rx,start}, i}$: Earliest documented systemic antimicrobial prescription start date (`prescriptions.startdate`, calendar-date precision, coarsened to midnight 00:00:00).
- $T_{\mathrm{abx,pair}, i}$: Systemic antimicrobial order date participating in the qualifying infection suspicion pair.
- $T_{\mathrm{culture,pair}, i}$: Blood culture specimen collection timestamp (`microbiologyevents.charttime`, minute-level precision).
- $T_{\mathrm{susp}, i}$: Operational infection-suspicion anchor, defined per Seymour et al. (2016) as $\min(T_{\mathrm{abx,pair}}, T_{\mathrm{culture,pair}})$.
- $T_{\mathrm{S3}, i}$: Retrospective consensus Sepsis-3 onset timestamp.

When blood culture precedes antimicrobial prescription ($T_{\mathrm{culture,pair}} < T_{\mathrm{abx,pair}}$), $T_{\mathrm{susp}} = T_{\mathrm{culture,pair}} \ne T_{\mathrm{abx,pair}}$. Let $h = 1\text{ hour}$ and define the dimensionless lookback coordinate $x_i = (T_{\mathrm{susp}, i} - 24h - T_{\text{admit}, i})/h$. The search window evaluates discrete hourly bins starting at $\text{baseline\_start\_h}_i = \lfloor x_i \rfloor$ and ending at $\text{window\_end\_h}_i = \lfloor(T_{\mathrm{susp}, i} + 12h - T_{\text{admit}, i})/h\rfloor$. Because the upper boundary bin is floored, evaluated hours never exceed $T_{\mathrm{susp}} + 12h$, strictly guaranteeing $T_{\mathrm{susp}, i} - T_{\text{S3}, i} \ge -12.0h$ (empirical minimum: $-11.2h$). For episodes where acute $\Delta\text{SOFA} \ge 2$ is first observed at the earliest search bin ($N=274/408$, 67.2%), onset is assigned as $T_{\text{S3}, i} = T_{\text{admit}, i} + \text{baseline\_start\_h}_i \times 1\text{h}$, yielding an operational search-window lead:
$$T_{\mathrm{susp}, i} - T_{\text{S3}, i} = 24.0\text{h} + h\,\operatorname{frac}(x_i) < 25.0\text{h}$$
Because $0 \le \operatorname{frac}(\cdot) < 1$, this hourly binning yields an analytical bound of $-12.0\text{h} \le T_{\mathrm{susp}, i} - T_{\text{S3}, i} < 25.0\text{h}$ (analytical supremum $25.0\text{h}$), explaining why the empirical cohort distribution exhibits a median of $+24.2\text{h}$ (IQR: $22.8\text{h}$ to $24.6\text{h}$; range: $-11.2\text{h}$ to $+24.99\text{h}$). Across all 274 boundary stays, the difference between $T_{\mathrm{susp}} - T_{\text{S3}}$ and $24.0 + h\,\operatorname{frac}(x_i)$ is identically zero ($< 10^{-13}\text{h}$). In 274 of 408 episodes, the assigned onset coincided with the first evaluated search bin. Consequently, onset timing and derived warning intervals were sensitive to the search-window boundary and bin-assignment convention and should not be interpreted as directly observed biological onset.

### 2.3 Operational Cascade and Pre-Suspicion Risk Sets
Let $\mathcal{C}$ denote the evaluated test cohort of unique ICU stays ($N = 8,254$), and let $\mathcal{T}_i \subseteq \{19, 20, \dots, 30\}$ denote the set of hourly decision bins evaluated for stay $i$. For patients who never develop consensus Sepsis-3 during hospitalization, we set $T_{\text{S3}, i} = \infty$.

The full evaluated early-warning risk set is defined as:
$$\mathcal{R} = \bigl\{ (i, t) : i \in \mathcal{C}, \; t \in \mathcal{T}_i, \; t < T_{\text{S3}, i} \bigr\}$$

The binary early-warning prediction target at anticipation horizon $H = 6\text{h}$ is:
$$Y_i(t; H) = \mathbf{1} \bigl\{ t < T_{\text{S3}, i} \le t + H \bigr\}$$

In the canonical unconstrained risk set ($\mathcal{R}_{\text{canonical}} = \mathcal{R}$), this definition yields exactly 97,700 evaluated hourly windows, comprising 678 horizon-positive observations ($Y=1$) and 97,022 negative observations ($Y=0$, prevalence $\pi = 0.694\%$).

To evaluate restricted decision windows, we define the outcome-stratified hybrid risk sets relative to reference clock $C_i$:
$$\mathcal{R}_\delta^{\mathrm{hybrid}} = \bigl\{ (i, t) \in \mathcal{R} : t + \delta \le C_i \bigr\}$$
where $C_i = T_{\mathrm{susp}, i} = \min(T_{\mathrm{culture,pair}, i}, T_{\mathrm{abx,pair}, i})$ for incident sepsis stays ($S_i=1$; determined by $T_{\mathrm{abx,pair}}$ in 399 stays and blood culture charttime in 9 stays), and $C_i = T_{\mathrm{Rx,start}, i}$ for non-sepsis control stays ($S_i=0$) if systemic antimicrobials were ordered during hospitalization ($C_i = \infty$ otherwise). Exactly 2,287 control stays had $T_{\mathrm{Rx,start}} \le 19\text{h}$, censoring all their windows. We evaluated $\mathcal{R}_0^{\mathrm{hybrid}}$ ($\delta=0\text{h}$; 70,192 windows across 5,967 stays; prevalence 0.960\%) and $\mathcal{R}_2^{\mathrm{hybrid}}$ ($\delta=2\text{h}$; 70,129 windows across 5,967 stays; prevalence 0.945\%).

Importantly, sepsis cases and controls were restricted using different clinical-process anchors. Differences from canonical discrimination therefore reflect both changes in the evaluated observations and the outcome-specific selection rule; they are not estimates of performance under a uniform prospective eligibility policy. Within this restricted surveillance protocol and under declared midnight assignment of prescription-start dates, no horizon-positive observations remained under uniform earliest-prescription eligibility ($\mathcal{R}_0^{\mathrm{Rx}}$: 5,559 control stays, 65,926 negative windows, 0 positive windows, $N_{\mathrm{total}} = 65,926$; $\mathcal{R}_2^{\mathrm{Rx}}$: 65,877 negative windows; $N_+^{\mathrm{Rx}} = 0$). Discrimination was therefore not estimable for that task. While the base cohort contains 408 eventual sepsis stays, the restricted task contains zero eligible horizon-positive decision windows. In pre-exposure sensitivity analyses, we evaluate $C_i = T_{\mathrm{Rx,start}, i}$ across all stays under declared midnight point assignment and calendar-day intervals $[a_i, b_i)$. Holding consensus $T_{\text{S3}}$ fixed evaluates precedence relative to operational EHR anchors; it does not establish robustness to date-level uncertainty in the reference phenotype itself.

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

Models consumed a causal 20-hour observation lookback $[t-20\text{h}, t)$ across 21 physiological time series, following established temporal feature engineering and sequential formulations in ICU deep learning (Shayan and Afrasiabi, 2025). All models were trained on fixed pre-surveillance development partitions with hyperparameters and feature normalization frozen prior to this operational evaluation. Complete architecture details, hyperparameter grids, and the stacking protocol are detailed in the Supplementary Methods Package.

Uncertainty was quantified using patient-clustered paired bootstrapping ($B=1,000$ resamples). For Procedure B (Empirical Burden-Matched Bootstrap), exposure $E_b = \frac{1}{24}\sum_{i\in b} h_i$ and alert target $K_b = \operatorname{round}(q \cdot E_b / 100)$ were dynamically recomputed within each resample. NNE confidence intervals were derived by exact reciprocal transformation of unrounded PPV percentiles: $\text{NNE}_{\text{low}} = 100 / \text{PPV}_{\text{upper}}$ and $\text{NNE}_{\text{high}} = 100 / \text{PPV}_{\text{low}}$ (with $+\infty$ when $\text{PPV}_{\text{low}} = 0$).

---

## 3. Results

### 3.1 Cohort Progression and Denominator Accounting
Table 1 traces all 8,254 ICU admissions through surveillance eligibility, onset timing, lookback horizons, and pre-treatment actionability. Among 408 consensus Sepsis-3 stays, 257 (63.0\%) had clinical onset after hour 36 (median onset: 49.0h post-admission). Under the specified observation schedule and 6-hour horizon, these 257 stays had no evaluated horizon-positive decision time ($V=0$). The remaining 151 stays (37.0\%) had active protocol surveillance coverage ($V=1$). Of the 151 covered stays, 150 had at least one horizon-positive decision time at or before recorded treatment initiation (99.3% within-covered opportunity; 36.8% of total incident sepsis), providing pre-proxy opportunity under midnight assignment ($\mathcal{R}_0$).

### Table 1: Cohort Progression & Denominator Accounting Ledger
| Progression Stage / Cohort Partition | ICU Stays ($N$) | Evaluated Windows | Surveillance Volume | Proportion / Window Prevalence | Clinical Definition & Measurement Convention |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Base Prospective-Replay Test Cohort** | 8,254 | 97,700 (out of 99,048 sched.) | 4,070.8 pt-days (97,700 pt-hours) | 100.0% of cohort | Adult first ICU admissions conditioned on ICU stay $\ge 30\text{h}$. Evaluated at scheduled hourly decision bins $[t-1, t]$ at $t \in \{19\dots 30\}$. 1,348 windows right-censored (623 post-onset, 725 monitoring cessation). |
| **2. Non-Sepsis Control ICU Stays** | 7,846 | 93,427 neg. windows | 3,892.8 pt-days (93,427 pt-hours) | 95.06% of stays | Patients without consensus Sepsis-3 during hospitalization. All windows serve as outcome-negative surveillance (180 stays ceased hourly charting prior to hour 30 due to observation-availability cessation, accounting for 725 windows; all stays had total $\text{LOS} \ge 30\text{h}$). |
| **3. Consensus Sepsis-3 Stays ($S=1$)** | 408 | 4,273 (678 pos, 3,595 neg) | 178.0 pt-days (4,273 pt-hours) | 4.94% of stays | Patients meeting Sepsis-3 consensus criteria (blood culture + antibiotic + acute $\Delta\text{SOFA}\ge 2$). Median onset: hour 49.0. 110 stays censored at onset prior to hour 30 (623 windows). |
| **  3a. Late-Onset Sepsis (Onset $>36$h)** | 257 | 3,084 neg. windows | 128.5 pt-days (3,084 pt-hours) | 63.0% of septic stays | Onset occurs after hour 36. 6-hour lookback $[T_{\text{S3}}-6, T_{\text{S3}}]$ does not intersect surveillance window $[19, 30]$ ($V=0$). |
| **  3b. Surveillance-Covered Sepsis ($V=1$)** | 151 | 1,189 (678 pos, 511 neg) | 49.5 pt-days (1,189 pt-hours) | 37.0% of septic stays | Onset occurs between ICU hours 20 and 36, such that $[T_{\text{S3}}-6\text{h}, T_{\text{S3}}]$ intersects surveillance interval $[19, 30]$ ($V=1$). |
| **4. Canonical Sepsis-3 Risk Set ($H=6$h)** | 8,254 (151 horizon-pos) | 97,700 (678 pos / 97,022 neg) | 4,070.8 pt-days | 0.694% prevalence | Full unconstrained risk set ($t \in \mathcal{T}_i, \; t < T_{\text{S3}, i}$). |
| **5. Outcome-Stratified Hybrid Set $\mathcal{R}_0^{\mathrm{hybrid}}$ ($\delta=0$h)** | 5,967 (150 horizon-pos) | 70,192 (674 pos / 69,518 neg) | 2,924.7 pt-days | 0.960% prevalence | Censoring post-anchor ($t \le C_i$; $T_{\mathrm{susp}}$ cases, $T_{\mathrm{Rx,start}}$ controls); 150/151 covered eligible. |
| **6. Outcome-Stratified Hybrid Set $\mathcal{R}_2^{\mathrm{hybrid}}$ ($\delta=2$h)** | 5,967 (149 horizon-pos) | 70,129 (663 pos / 69,466 neg) | 2,922.0 pt-days | 0.945% prevalence | 2-hour proxy-timing margin ($t + 2\text{h} \le C_i$); 149/151 covered eligible. |
| **7. Uniform Earliest-Rx Task $\mathcal{R}_0^{\mathrm{Rx}}$ ($\delta=0$h)** | 5,559 (0 horizon-pos) | 65,926 (0 pos / 65,926 neg) | 2,746.9 pt-days | 0.000% | Uniform eligibility task requiring $t \le T_{\mathrm{Rx,start}}$ across all stays. Under declared midnight assignment, no horizon-positive observations remain ($N_+=0$). Prevalence: 0.000%; discrimination metrics: not meaningfully estimable (with no eligible positive observations, AUROC was undefined and AUPRC did not provide an informative discrimination estimate. PPV would be zero for a policy emitting alerts in this all-negative evaluated set, and undefined if no alerts were emitted). Control attrition identity: $65,926 = 93,427 - W_{\mathrm{fully\ excluded}} (27,388\text{ from 2,287 stays}) - W_{\mathrm{partially\ censored}, 0} (113\text{ from 83 later-prescription control stays})$. |
| **8. Uniform Earliest-Rx Task $\mathcal{R}_2^{\mathrm{Rx}}$ ($\delta=2$h)** | 5,559 (0 horizon-pos) | 65,877 (0 pos / 65,877 neg) | 2,744.9 pt-days | 0.000% | Uniform eligibility task requiring a 2-hour proxy-timing margin ($t + 2\text{h} \le T_{\mathrm{Rx,start}}$) across all stays; no horizon-positive observations remain ($N_+=0$). Prevalence: 0.000%; discrimination metrics: not meaningfully estimable (with no eligible positive observations, AUROC was undefined and AUPRC did not provide an informative discrimination estimate. PPV would be zero for a policy emitting alerts in this all-negative evaluated set, and undefined if no alerts were emitted). Margin attrition identity: $65,877 = 65,926 - W_{\mathrm{additional\ margin}} (49\text{ from 83 later-prescription control stays})$. |

### 3.2 Outcome-Stratified Hybrid Discrimination Analysis
Table 2 presents discrimination across Canonical Sepsis-3, Hybrid Set ($\mathcal{R}_0^{\mathrm{hybrid}}$), and Hybrid Set with 2h Buffer ($\mathcal{R}_2^{\mathrm{hybrid}}$). Discrimination changed little in the outcome-stratified hybrid analysis (BiMamba SSM: AUROC 0.6326 canonical vs. 0.6433 on $\mathcal{R}_0^{\mathrm{hybrid}}$, paired $\Delta\text{AUROC} = +0.0107$, 95% CI: +0.0060 to +0.0175; Stacking Ensemble: 0.6284 vs. 0.6351, paired $\Delta\text{AUROC} = +0.0067$). Because cases and controls were restricted using different anchors, this finding does not establish preserved discrimination under a uniform pre-suspicion or pre-prescription policy. The slight numerical change in AUPRC (BiMamba: 1.209% to 1.656\%) reflects higher baseline positive prevalence in the restricted set (0.694% vs. 0.960\%). These results are directionally consistent with the antibiotic-order sensitivity findings of Dutta et al. (2026).

### Table 2: Benchmark Discrimination Across Outcome-Stratified Hybrid Risk Sets
*Held-out point estimates ($B=0$) and $B=1,000$ patient-clustered bootstrap percentile 95% confidence intervals. Evaluated on hybrid risk sets $\mathcal{R}_0^{\mathrm{hybrid}}$ and $\mathcal{R}_2^{\mathrm{hybrid}}$ censoring cases at infection-suspicion anchor $T_{\mathrm{susp}}$ and controls at earliest prescription $T_{\mathrm{Rx,start}}$. Sepsis cases and controls were restricted using different clinical-process anchors; differences from canonical discrimination therefore reflect both evaluated observations and outcome-specific selection rather than uniform prospective eligibility. Under uniform point-assigned earliest prescription restriction ($C_i = T_{\mathrm{Rx,start}, i}$), discrimination is not meaningfully estimable due to zero retained positive cases ($O_{\mathrm{Rx}}^{\mathrm{point}} = 0$). With no eligible positive observations, AUROC was undefined and AUPRC did not provide an informative discrimination estimate. PPV would be zero for a policy emitting alerts in this all-negative evaluated set, and undefined if no alerts were emitted.*

| Model Architecture | Canonical Sepsis-3 ($H=6\text{h}$)<br>AUROC [95% CI] | Hybrid $\mathcal{R}_0^{\mathrm{hybrid}}$ ($\delta=0\text{h}$)<br>AUROC [95% CI] | Hybrid $\mathcal{R}_2^{\mathrm{hybrid}}$ ($\delta=2\text{h}$)<br>AUROC [95% CI] | Paired $\Delta\text{AUROC}$ ($\mathcal{R}_0^{\mathrm{hybrid}} - \text{Canon}$)<br>[95% CI] |
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
- In this observed evaluation, the count of horizon-correct emitted alerts and the count of alerts additionally satisfying the two-hour pre-suspicion buffer ($t_{\mathrm{first}, i} + 2\text{h} \le T_{\mathrm{susp}, i}$) were identical across all models and operating points (e.g., 7 of 7 detected patients for BiMamba SSM and LightGBM at 10 ep/100 pt-d; equality in sample: $N_{\mathrm{det}} = N_{\mathrm{det}, \mathrm{pre\text{-}susp}}$). These detections occurred before the selected retrospective suspicion anchor ($T_{\mathrm{susp}}$), not necessarily before earliest antimicrobial prescribing ($T_{\mathrm{Rx,start}}$).
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

*$^*$At zero observed detections, the empirical percentile-bootstrap interval was degenerate at zero. This does not establish zero population detection probability. NNE intervals derived by exact reciprocal transformation. In this observed sample, horizon-correct detections and detections satisfying the two-hour pre-suspicion timing buffer ($t_{\mathrm{first}, i} + 2\text{h} \le T_{\mathrm{susp}, i}$) were identical across all models and operating points. Simulated alert episodes were evaluated across the full canonical surveillance stream (4,070.8 patient-days, 97,700 windows); reference anchors $C_i$ were used retrospectively to adjudicate advance-warning timeliness, not to filter observation volume prior to alert generation.*

### 3.4 Two-Axis Subgroup and Timestamp Uncertainty Matrix
Table 4 presents the cross-tabulation of Sepsis Onset Timing against Treatment Proxy Timing. Because MIMIC-III prescription start dates (`prescriptions.startdate`) lack time of day, order initiation is evaluated both under declared point assignment (midnight $a_i$) and under calendar-day intervals $[a_i, b_i) = [\text{midnight}, \text{next midnight})$.

Opportunity depended strongly on the selected clinical-process anchor. Under the primary benchmark evaluating decision opportunity relative to the infection-suspicion anchor ($T_{\mathrm{susp}}$), 150 of 151 covered patients (99.3% within-covered; 36.8% of total incident sepsis) had decision bins preceding $T_{\mathrm{susp}}$ under declared midnight point assignment $a_i$. Because $a_i \le T_{\mathrm{susp}, i}$ across $[a_i, b_i)$, these 150 stays are definitely eligible throughout the calendar-day interval ($O_i^{\mathrm{def}} = 150$), with only 1 indeterminate stay ($O_i^{\mathrm{indet}} = 1$), bounding opportunity between 36.8% and 37.0% ($[150, 151]/408$; $[99.3\%, 100.0\%]$ within-covered). All indicators strictly satisfy $O_i^{\mathrm{def}} \le O_i^{\mathrm{point}} \le O_i^{\mathrm{poss}}$ and $O_i^{\mathrm{poss}} = O_i^{\mathrm{def}} + O_i^{\mathrm{indet}}$.

In contrast, opportunity was substantially lower and uncertain relative to the earliest hospital prescription-start proxy ($T_{\mathrm{Rx,start}}$): exactly 0 covered stays had definite opportunity under midnight point-assignment ($O_i^{\mathrm{def}} = 0, O_i^{\mathrm{point}} = 0$), with 47 indeterminate stays ($O_i^{\mathrm{indet}} = 47$), establishing bounds $[0.0\%, 11.5\%]$ ($[0, 47]/408$; $[0.0\%, 31.1\%]$ within-covered). Because no horizon-positive cases remained before point-assigned earliest prescriptions, this task was not evaluable for discrimination under midnight point assignment. In broad historical MIMIC-III ($N=3,581$), intervals generate 1 definitely eligible stay, 110 indeterminate stays, and bounds $[0.03\%, 3.10\%]$. The 150/151 result under $T_{\mathrm{susp}}$ does not establish that patients were unexposed to antimicrobials or that models predicted prior to first administration. Equally, zero definite opportunity under $T_{\mathrm{Rx,start}}$ reflects date coarsening, not verified bedside exposure.

### Table 4: Two-Axis Subgroup and Timestamp Uncertainty Matrix
| Cohort & Timestamp Source | Sepsis Onset Category | Total ($N$) | Covered ($V=1$) | Definitely Eligible ($O^{\mathrm{def}}=1$) | Point-Assigned ($O^{\mathrm{point}}=1$) | Indeterminate ($O^{\mathrm{indet}}$) | Possibly Eligible ($O^{\mathrm{poss}}$) | Subgroup Bounds | Contribution to Cohort |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MetaVision Prospective-Replay** | Early Incident ($\le 24\text{h}$) | 47 | 47 (100.0%) | 47 | 47 | 0 | 47 | [100.0%, 100.0%] | [47, 47] / 408 (11.5%) |
| *(Primary Benchmark: Suspicion Anchor $T_{\mathrm{susp}}$)* | Intermediate (24–48h) | 156 | 104 (66.7%) | 103 | 103 | 1 | 104 | [66.0%, 66.7%] | [103, 104] / 408 (25.2–25.5%) |
| | Late Incident ($>48\text{h}$) | 205 | 0 (0.0%) | 0 | 0 | 0 | 0 | [0.0%, 0.0%] | 0 / 408 (0.0%) |
| | **All Incident Sepsis-3** | **408** | **151 (37.0%)** | **150** | **150** | **1** | **151** | **[36.8%, 37.0%]** | **[150, 151] / 408 (36.8–37.0%)** |
| **MetaVision Sensitivity: Earliest Rx** | Early Incident ($\le 24\text{h}$) | 47 | 47 (100.0%) | 0 | 0 | 38 | 38 | [0.0%, 80.9%] | [0, 38] / 408 (0.0–9.3%) |
| *(Pre-Exposure Sensitivity: Earliest Rx $T_{\mathrm{Rx,start}}$)* | Intermediate (24–48h) | 156 | 104 (66.7%) | 0 | 0 | 9 | 9 | [0.0%, 5.8%] | [0, 9] / 408 (0.0–2.2%) |
| | Late Incident ($>48\text{h}$) | 205 | 0 (0.0%) | 0 | 0 | 0 | 0 | [0.0%, 0.0%] | 0 / 408 (0.0%) |
| | **All Incident Sepsis-3** | **408** | **151 (37.0%)** | **0** | **0** | **47** | **47** | **[0.0%, 11.5%]** | **[0, 47] / 408 (0.0–11.5%)** |
| **Broad Historical MIMIC-III** | Early Incident ($\le 24\text{h}$) | 2,641 | 102 (3.9%) | 0 | 0 | 74 | 74 | [0.00%, 2.80%] | [0, 74] / 3,581 (0.00–2.07%) |
| *(Date-Level Proxy: Earliest Rx $T_{\mathrm{Rx,start}}$)* | Intermediate (24–48h) | 361 | 235 (65.1%) | 1 | 1 | 36 | 37 | [0.28%, 10.25%] | [1, 37] / 3,581 (0.03–1.03%) |
| | Late Incident ($>48\text{h}$) | 579 | 0 (0.0%) | 0 | 0 | 0 | 0 | [0.00%, 0.00%] | [0, 0] / 3,581 (0.00%) |
| | **All Incident Sepsis-3** | **3,581** | **337 (9.4%)** | **1** | **1** | **110** | **111** | **[0.03%, 3.10%]** | **[1, 111] / 3,581 (0.03–3.10%)** |

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
1. **Outcome-Dependent Anchor Selection Governs Evaluability:** The canonical surveillance analysis quantified population coverage and emitted-alert yield. A secondary outcome-stratified hybrid restriction produced small changes in discrimination, but did not represent a uniform prospective eligibility policy. Under a common earliest-prescription-start proxy and declared midnight assignment, no eligible horizon-positive observations remained, preventing discrimination estimation for that restricted task.
2. **Detection Yield Is Constrained by Workload and Prevalence:** AUROC values near 0.64 are modest. At 10 emitted episodes per 100 evaluation-bin patient-days, PPV ranged from 0.98% to 1.72%, corresponding to 58.1 to 101.8 simulated evaluations per detected patient ($\text{NNE}$); observed sensitivity was below 2%.
3. **Surveillance Protocol Coverage Explains Unreached Cases:** Under the evaluated second-day protocol, 63.0% of incident cases develop onset after the monitoring window terminates, preventing multi-hour warning.

### 4.2 Distinguishing Care-Process Forecasting from Sepsis Anticipation
A critical conceptual error in prior literature is **endpoint substitution**. When benchmarks predict "antibiotic initiation within 6 hours across all patients", they evaluate care-process prediction (predicting clinician ordering behavior, including prophylactic prescribing in non-septic patients). This task is fundamentally distinct from pre-treatment sepsis detection. Pretreatment sepsis evaluation should retain its specified sepsis outcome while restricting eligible decision times; replacing that outcome with antibiotic receipt defines a different prediction task.

Clinician-vignette studies address when clinicians judge treatment appropriate (Weissman et al., 2024); our analysis addresses prediction performance relative to operational EHR proxies. These are complementary questions, and proxy precedence does not establish clinical appropriateness. Furthermore, while changes under prescription-proxy restriction were small in this cohort, differences in outcome construction, population selection, timestamp precision, and clinical-action definition preclude direct replication of Dutta et al.'s order-based sensitivity analysis (Dutta et al., 2026). Recently, van der Vegt et al. (2026) demonstrated that conventional discrimination metrics fail to account for alert timing relative to clinical events and proposed balancing predictive accuracy, alert burden, and clinical utility across decision thresholds.

Repeated and low-information notifications have been associated with reduced clinician acceptance in other decision-support settings (Ancker et al., 2017). While we quantify simulated emitted-alert exposure and evaluation yield, our analysis did not measure cognitive overload, alert acceptance, or alarm fatigue directly (Ancker et al., 2017; Paine et al., 2016). Clinical impact also depends on whether alerts are reviewed and acted upon (Henry et al., 2022). Our evaluation estimates opportunities and simulated review burden; it does not estimate provider adoption or changes in treatment timing.

### 4.3 Limitations
The analysis estimates operating characteristics in a selected long-stay ICU cohort rather than clinical effectiveness. Burden-matched cutoffs were derived retrospectively from complete evaluated score trajectories, and identifier-based selection was required for tied boundary scores. Accordingly, the frontier does not represent a prospectively locked policy or guarantee a specified alert rate in online deployment.

Temporal precision also constrains interpretation: MIMIC-III prescription dates cannot be treated as directly observed administration times (Cohen et al., 2024; MIMIC-III release notes). Our order-proxy analyses therefore require explicit uncertainty bounds and cannot establish hour-resolved clinical treatment delay. Observed treatment precedence does not establish that infection was unrecognized or that earlier antimicrobial administration would have been appropriate. The study did not measure clinician response, review duration, treatment changes, alarm fatigue, or patient outcomes; NNE therefore represents a simulated evaluation count rather than measured clinical workload. Alarm-reduction interventions may suppress both nonactionable and clinically important signals; our short evaluation interval cannot establish the safety or optimal duration of a refractory policy during sustained surveillance (Paine et al., 2016). Finally, our findings possess two distinct epistemic dimensions: the floor-discretization proof ($G_i = 24\mathrm{h} + h\,\operatorname{frac}(x_i) < 25\mathrm{h}$) and timestamp-coarsening collapse ($N_+^{\mathrm{Rx}} = 0$) are exact, assumption-explicit algebraic identities that hold identically in any healthcare system applying discrete lookback flooring and midnight date truncation; conversely, the empirical detection yields (1.72%) and alarm burdens reflect this specific cohort, EHR era transition (CareVue to MetaVision), and protocol schedule, where multi-center studies remain necessary to establish their operational prevalence.

---

## 5. Conclusion
In this restricted, long-stay ICU cohort, observed detection yield depended on surveillance coverage, baseline discrimination, and emitted-alert policies. Evaluation feasibility was highly anchor-dependent: horizon-positive observations were retained before the selected suspicion anchor, but none remained before the earliest prescription-start proxy under midnight assignment, precluding discrimination estimation for that task. Machine learning early-warning evaluations must transparently identify outcome-dependent evaluation rules, report anchor-dependent evaluability, and benchmark under explicit workload constraints.

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
Prediction model architectures and frozen weight tensors are archived under model/data freeze commit `b0932ca` (release manifest `1.0.0-freeze`); evaluation implementation is archived under commit `1292103`; denominator recomputation is archived under commit `311cea3`; submission packaging is archived under commit `9b92113`, tag `v1.0.1-freeze`. All preprocessing pipelines and configurations are available under MIT License at [https://github.com/alireza/Sepsis-Onset-Timing-Paradox](https://github.com/alireza/Sepsis-Onset-Timing-Paradox). De-identified patient data are available from PhysioNet under the MIMIC-III Data Use Agreement.

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
22. Cohen SN, Foster J, Foster P, et al. Subtle variation in sepsis-III definitions markedly influences predictive performance within and across methods. *Sci Rep*. 2024;14:1920. DOI: [10.1038/s41598-024-51989-6](https://doi.org/10.1038/s41598-024-51989-6)
23. Shayan A, Afrasiabi S. Enhanced LSTM model for early sepsis prediction using systematic data preprocessing and temporal feature engineering. In: *15th Smart Grid Conference (SGC)*; 2025:1–6. DOI: [10.1109/SGC68038.2025.11372398](https://doi.org/10.1109/SGC68038.2025.11372398)
24. van der Vegt AH, Campbell VK, Webb R, et al. A novel, standardised approach to balancing effectiveness, efficiency and utility of surveillance AI prediction models for hospitalised patients using sepsis prediction as an exemplar. *J Am Med Inform Assoc*. 2026;33(2):304–315. DOI: [10.1093/jamia/ocaf192](https://doi.org/10.1093/jamia/ocaf192)
