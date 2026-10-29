# One-Page Authoritative Clock-to-Result Mapping

**Manuscript:** *Surveillance Coverage, Treatment Precedence, and Workload Constraints in Machine Learning Early Warning for Sepsis: An Operational Evaluation*  
**Authors:** Alireza Shayan, Somayeh Afrasiabi (Shiraz University)  
**Target Journal:** *JAMIA Open*  
**Model & Data Freeze:** Commit `b0932ca` (Release Manifest `1.0.0-freeze`)  
**Evaluation Implementation:** Commit `1292103`  
**Denominator Recomputation:** Commit `311cea3`  
**Submission Packaging:** Commit `9b92113`, Tag `v1.0.1-freeze`  

---

## 1. Operational Clocks & Mathematical Definitions

To resolve event ambiguity, care-process conflation, and timestamp coarsening across electronic health records, every reported result is mapped to explicit operational clocks defined relative to ICU admission ($t = 0$):

| Clock Symbol | Variable Name | Source Table & Column | Temporal Precision | Mathematical / Operational Definition |
| :--- | :--- | :--- | :--- | :--- |
| **Origin** | $T_{\text{admit}, i}$ | `icustays.intime` | Minute-level | ICU admission timestamp ($t = 0$, reference origin). |
| **Clock 1 (Primary)** | $T_{\mathrm{susp}, i}$ | `microbiologyevents.charttime`, `prescriptions.startdate` | Minute/Day hybrid | Infection-suspicion anchor: $\min(T_{\mathrm{culture,pair}, i}, T_{\mathrm{abx,pair}, i})$. Dictated by antibiotic order in 399 cases and blood culture in 9 cases. |
| **Clock 2 (Sensitivity)** | $T_{\mathrm{Rx,start}, i}$ | `prescriptions.startdate` | Calendar day (coarsened to 00:00:00) | Earliest documented systemic antimicrobial prescription start date in the hospitalization. ($T_{\mathrm{Rx,start}, i} = \infty$ for unexposed). |
| **Clock 3 (Phenotype)** | $T_{\text{S3}, i}$ | Derived SOFA trajectories | 1-hour discrete bin | Retrospective consensus Sepsis-3 onset timestamp: earliest discrete hourly bin where acute $\Delta\text{SOFA} \ge 2$ within $[T_{\mathrm{susp}}-24\text{h}, T_{\mathrm{susp}}+12\text{h}]$. |

### Outcome-Stratified Hybrid Risk Set Formulation (Option B)
For clinical warning buffer $\delta \ge 0$ (evaluating $\delta = 0\text{h}$ and $\delta = 2\text{h}$), the outcome-stratified hybrid risk set restricted by reference clock $C_i$ is formulated as:
$$\mathcal{R}_\delta^{\mathrm{hybrid}} = \bigl\{ (i, t) \in \mathcal{R} : t + \delta \le C_i \bigr\}$$

1. **Outcome-Dependent Clinical Anchor Mapping ($C_i$):**
   - For incident sepsis stays ($S_i = 1$): $C_i = T_{\mathrm{susp}, i} = \min(T_{\mathrm{culture,pair}, i}, T_{\mathrm{abx,pair}, i})$ (determined by antibiotic order in 399 stays and blood culture collection in 9 stays).
   - For non-sepsis control stays ($S_i = 0$): $C_i = T_{\mathrm{Rx,start}, i}$ if systemic antimicrobials were ordered during hospitalization; $C_i = \infty$ otherwise.
   - *Authoritative Methodological Disclosure:* Discrimination changed little in the outcome-stratified hybrid analysis. Because cases and controls were restricted using different anchors, this finding does not establish preserved discrimination under a uniform pre-suspicion or pre-prescription policy.
2. **Earliest-Prescription Non-Estimability Finding (Scoped):**
   - Within this restricted surveillance protocol and under declared midnight assignment of prescription-start dates, no horizon-positive observations remained under uniform earliest-prescription eligibility. Discrimination was therefore not estimable for that task.
   - *Denominator Distinction:* While the base cohort contains 408 eventual sepsis stays, the restricted task contains zero eligible horizon-positive decision windows ($N_+^{\mathrm{Rx}} = 0$). With no eligible positive observations, AUROC was undefined and AUPRC did not provide an informative discrimination estimate. PPV would be zero for a policy emitting alerts in this all-negative evaluated set, and undefined if no alerts were emitted.

---

## 2. Risk Set & Population Denominator Accounting

All evaluated hourly decision bins fall strictly within the prespecified second-day ICU surveillance schedule: $t \in \{19, 20, \dots, 30\}\text{h}$ post-admission (up to 12 windows per stay). Denominators strictly satisfy $N_{\mathrm{total}} = N_+ + N_-$.

| Risk Set / Partition | Conditioning Clock ($C_i$) | Proxy Margin ($\delta$) | Unique ICU Stays ($N$) | Evaluated Hourly Windows ($N_{\mathrm{total}}$) | Surveillance Volume | Positive Windows ($N_+$) | Negative Windows ($N_-$) | Positive Prevalence ($\pi$) | Discrimination Estimability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Canonical Risk Set $\mathcal{R}_{\text{canonical}}$** | None (unconstrained) | N/A | 8,254 | 97,700 | 4,070.8 pt-days | 678 | 97,022 | 0.694% | Standard benchmark (Table 2) |
| **Outcome-Stratified Hybrid Set $\mathcal{R}_0^{\mathrm{hybrid}}$** | $T_{\mathrm{susp}}$ cases / $T_{\mathrm{Rx,start}}$ controls | 0h | 5,967 | 70,192 | 2,924.7 pt-days | 674 | 69,518 | 0.960% | Preserved discrimination (Table 2) |
| **Outcome-Stratified Hybrid Set $\mathcal{R}_2^{\mathrm{hybrid}}$** | $T_{\mathrm{susp}}$ cases / $T_{\mathrm{Rx,start}}$ controls | 2h | 5,967 | 70,129 | 2,922.0 pt-days | 663 | 69,466 | 0.945% | Preserved discrimination (Table 2) |
| **Uniform Earliest-Rx Task $\mathcal{R}_0^{\mathrm{Rx}}$** | $T_{\mathrm{Rx,start}}$ across all stays | 0h | 5,559 | 65,926 | 2,746.9 pt-days | **0** | 65,926 | **0.000%** | Not meaningfully estimable$^*$ |
| **Uniform Earliest-Rx Task $\mathcal{R}_2^{\mathrm{Rx}}$** | $T_{\mathrm{Rx,start}}$ across all stays | 2h | 5,559 | 65,877 | 2,744.9 pt-days | **0** | 65,877 | **0.000%** | Not meaningfully estimable$^*$ |

$^*$*Note on Discrimination Estimability under Uniform Earliest-Rx:* Prevalence is strictly 0.000% ($N_+ = 0$). With no eligible positive observations, AUROC was undefined and AUPRC did not provide an informative discrimination estimate. PPV would be zero for a policy emitting alerts in this all-negative evaluated set, and undefined if no alerts were emitted.

### Strict Denominator Reconciliation & Attrition Mechanisms
1. **Hybrid Restrictions ($\mathcal{R}_0^{\mathrm{hybrid}}, \mathcal{R}_2^{\mathrm{hybrid}}$):**
   - Cases ($N=408$ stays): Evaluated against suspicion anchor $T_{\mathrm{susp}}$. Retain 4,266 windows (674 positive, 3,592 negative) at $\delta=0\text{h}$; 4,252 windows (663 positive, 3,589 negative) at $\delta=2\text{h}$. (7 case windows removed at $\delta=0\text{h}$ from stays with early suspicion).
   - Controls ($N=7,846$ stays, 93,427 canonical windows): Evaluated against earliest prescription $T_{\mathrm{Rx,start}}$.
     - *Fully Excluded Stays ($T_{\mathrm{Rx,start}} \le 19.0\text{h}$):* Exactly 2,287 control stays had documented prescription start dates prior to or at surveillance entry. These 2,287 stays account for exactly $W_{\mathrm{fully\ excluded}} = 27,388$ canonical windows (2,270 stays $\times 12 = 27,240$ windows, plus 17 stays with $<12$ windows totaling 148 windows due to observation-availability cessation before hour 30).
     - *Later-Prescription Controls ($T_{\mathrm{Rx,start}} > 19.0\text{h}$):* Exactly 83 control stays were later-prescription controls with prescription-start proxy after hour 19 (996 canonical windows; note that $T_{\mathrm{Rx,start}} > 19\text{h}$ includes orders after surveillance terminates). At $\delta=0\text{h}$, 883 windows satisfy $t \le T_{\mathrm{Rx,start}}$ and are retained, while $W_{\mathrm{partially\ censored}, 0} = 113$ windows occur post-prescription and are censored.
     - *Unexposed Stays ($T_{\mathrm{Rx,start}} = \infty$):* Exactly 5,476 control stays had no qualifying prescription records, contributing all 65,043 of their canonical windows.
     - *Control Accounting Identity ($\delta=0\text{h}$):*
       $$65,926 = 93,427 - W_{\mathrm{fully\ excluded}} - W_{\mathrm{partially\ censored}, 0} = 93,427 - 27,388 - 113$$
       Total control windows removed at $\delta=0\text{h}$ is $27,388 + 113 = 27,501$. (Note: the previously stated quantity 27,508 represents total cohort-wide window reduction from canonical to hybrid, $97,700 - 70,192 = 27,508$, which combines 27,501 removed control windows with 7 removed sepsis case windows).
     - *Control Accounting Identity ($\delta=2\text{h}$):*
       $$65,877 = 65,926 - W_{\mathrm{additional\ margin}} = 65,926 - 49$$
       where $W_{\mathrm{additional\ margin}} = 49$ windows removed by the 2-hour margin within the 83 later-prescription control stays ($883 - 834 = 49$).
   - *Sums:*
     - $\mathcal{R}_0^{\mathrm{hybrid}}$: $N_{\mathrm{total}} = 4,266 + 65,926 = 70,192$; $N_+ + N_- = 674 + 69,518 = 70,192$. Stays: $408 + 5,559 = 5,967$.
     - $\mathcal{R}_2^{\mathrm{hybrid}}$: $N_{\mathrm{total}} = 4,252 + 65,877 = 70,129$; $N_+ + N_- = 663 + 69,466 = 70,129$. Stays: $408 + 5,559 = 5,967$.
2. **Uniform Earliest-Prescription Task ($\mathcal{R}_0^{\mathrm{Rx}}, \mathcal{R}_2^{\mathrm{Rx}}$):**
   - Applied uniformly: $t + \delta \le T_{\mathrm{Rx,start}, i}$.
   - All 408 sepsis stays had documented prescription start dates coarsened to midnight with $T_{\mathrm{Rx,start}, i} \le 3.83\text{h} \le 19.0\text{h}$ post-ICU admission. Consequently, zero decision windows in surveillance hours 19–30 satisfy $t + \delta \le T_{\mathrm{Rx,start}, i}$ (0 case stays, 0 positive windows, 0 negative windows).
   - Contributing stays are exclusively the 5,559 surviving control stays ($5,476\text{ unexposed} + 83\text{ later-prescription controls}$):
     - $\mathcal{R}_0^{\mathrm{Rx}}$: $N_{\mathrm{total}} = 65,926$; $N_+ = 0$; $N_- = 65,926$. Exactly $0 + 65,926 = 65,926$. Volume: 2,746.9 pt-days.
     - $\mathcal{R}_2^{\mathrm{Rx}}$: $N_{\mathrm{total}} = 65,877$; $N_+ = 0$; $N_- = 65,877$. Exactly $0 + 65,877 = 65,877$. Volume: 2,744.9 pt-days.
   - *Metric Estimability with Zero Positives:* With no eligible positive observations, AUROC was undefined and AUPRC did not provide an informative discrimination estimate. PPV would be zero for a policy emitting alerts in this all-negative evaluated set, and undefined if no alerts were emitted.

---

## 3. Comprehensive Mapping of Manuscript Figures & Tables

| Manuscript Item | Displayed Content | Underlying Reference Clock(s) | Primary Evaluated Denominators | Essential Scientific Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **Table 1** | Cohort Progression Ledger | $T_{\text{admit}}$, $T_{\mathrm{susp}}$, $T_{\mathrm{Rx,start}}$, $T_{\text{S3}}$ | Base: 8,254 stays (408 sepsis / 7,846 controls). $\mathcal{R}_0$: 5,967 stays. $\mathcal{R}_0^{\mathrm{Rx}}$: 5,559 stays. | Traces stepwise attrition: 257/408 late sepsis unreached ($V=0$); 150/151 covered septic stays eligible before suspicion ($O=1$). Under uniform earliest-prescription, 0 positive windows remain. |
| **Table 2** | Discrimination Benchmark (8 Architectures) | Hybrid: $T_{\mathrm{susp}}$ cases, $T_{\mathrm{Rx,start}}$ controls | Canonical: 97,700 windows. $\mathcal{R}_0^{\mathrm{hybrid}}$: 70,192 windows. $\mathcal{R}_2^{\mathrm{hybrid}}$: 70,129 windows. | Discrimination changed little in hybrid analysis (BiMamba: 0.6433 on $\mathcal{R}_0^{\mathrm{hybrid}}$ vs. 0.6326 canonical). Does not establish preserved discrimination under uniform policy. Non-estimable under uniform $T_{\mathrm{Rx,start}}$ ($N_+=0$). |
| **Table 3** | Operational Alert-Budget Frontier (8 Models) | Full surveillance stream; timing adjudicated via $T_{\mathrm{susp}}$ | 8,254 test stays; 4,070.8 patient-days (97,700 windows); 204 alerts (5 ep), 407 alerts (10 ep), 814 alerts (20 ep). | Alerts selected across full stream (4,070.8 pt-d). Horizon-correct alerts and alerts satisfying $t_{\mathrm{first}} + 2\text{h} \le T_{\mathrm{susp}}$ are identical in sample ($N_{\mathrm{det}} = 7/7$ at 10 ep/100 pt-d). Adjudicated before retrospective suspicion anchor, not prescribing. |
| **Table 4** | Two-Axis Subgroup & Timestamp Uncertainty Matrix | Panel 1: $T_{\mathrm{susp}}$ (Primary) Panel 2: $T_{\mathrm{Rx,start}}$ (MetaVision) Panel 3: $T_{\mathrm{Rx,start}}$ (Broad MIMIC) | Panel 1: 408 sepsis stays. Panel 2: 408 sepsis stays. Panel 3: 3,581 sepsis stays. | Opportunity is anchor-dependent: 150/151 (99.3%) under suspicion anchor $T_{\mathrm{susp}}$, but [0.0%, 31.1%] (0 definite) under earliest prescription $T_{\mathrm{Rx,start}}$. |
| **Figure 1** | Panel A: Clocks & Risk Sets Panel B: Opportunity Waterfall | Panel A: $T_{\text{admit}}$, $T_{\mathrm{susp}}$, $T_{\mathrm{Rx,start}}$, $T_{\text{S3}}$ Panel B: $T_{\mathrm{Rx,start}}$ vs. $T_{\mathrm{susp}}$ | Panel A: Conceptual timeline. Panel B: MetaVision ($N=408$) vs. Broad MIMIC-III ($N=3,581$). | Visualizes floor-discretization lead $G_i = 24h + h\,\mathrm{frac}(x_i) < 25h$ (67.2% first-bin assigned) and opportunity bounds across clocks. |
| **Figure 2** | Operational Alert-Budget Frontiers (PPV & NNE) | Primary: $C_i^{\text{primary}}$ under 12h refractory policy | 8,254 test stays across 5, 10, and 20 ep / 100 pt-d budgets. | Demonstrates real-world performance trade-off: PPV < 3.0% and NNE 34.0–116.3 across all viable operational alert budgets. |
| **Figure 3** | Three-Stage Population Coverage Decomposition | $T_{\text{admit}}$, $T_{\text{S3}}$, $T_{\mathrm{susp}}$, Emitted Alert Policy | 408 incident Sepsis-3 stays. | Stepwise population decomposition: $\frac{151}{408} (37.0\%) \times \frac{150}{151} (99.3\%) \times \frac{7}{150} (4.67\%) = \frac{7}{408} = 1.72\%$ timely detection yield. |

---

## 4. Discrete Floor-Discretization Lead Formulation & Interpretation

For any episode where acute organ dysfunction first crosses threshold at the start of the retrospective search window (67.2% of stays, $N=274/408$):
- **Dimensionless Lookback Coordinate:** $x_i = \frac{T_{\mathrm{susp}, i} - 24h - T_{\text{admit}, i}}{h}$ where $h = 1\text{ hour}$.
- **Floored Boundary Start:** $\text{baseline\_start\_h}_i = \lfloor x_i \rfloor$.
- **Floored Window End:** $\text{window\_end\_h}_i = \lfloor (T_{\mathrm{susp}, i} + 12h - T_{\text{admit}, i})/h \rfloor$, strictly guaranteeing $T_{\mathrm{susp}, i} - T_{\text{S3}, i} \ge -12.0h$ (empirical minimum: $-11.2h$).
- **Search-Window Lead Formulation:**
  $$G_i = T_{\mathrm{susp}, i} - T_{\text{S3}, i} = 24h + h\,\operatorname{frac}(x_i) < 25.0h$$
- **Analytical Bound vs. Empirical Distribution:**
  - Analytical Infimum: $-12.0\text{h}$ (strictly guaranteed by floored upper window).
  - Analytical Supremum: $25.0\text{h}$ (unattained open upper bound, since $h\,\operatorname{frac}(x_i) < 1.0\text{h}$).
  - Empirical Distribution: Median $+24.2\text{h}$, IQR $22.8\text{h}$ to $24.6\text{h}$, empirical maximum $+24.99\text{h}$.
- **Phenotypic Interpretation:** In 274 of 408 episodes, assigned onset coincided with the first evaluated search bin. Consequently, onset timing and derived warning intervals were sensitive to the search-window boundary and bin-assignment convention and should not be interpreted as directly observed biological onset.
