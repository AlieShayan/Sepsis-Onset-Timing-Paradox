# One-Page Authoritative Clock-to-Result Mapping

**Manuscript:** *Surveillance Coverage, Treatment Precedence, and Workload Constraints in Machine Learning Early Warning for Sepsis: An Operational Evaluation*  
**Authors:** Alireza Shayan, Somayeh Afrasiabi (Shiraz University)  
**Target Journal:** *JAMIA Open*  
**Analytical Data Freeze:** Commit `b0932ca` (Release Manifest `1.0.0-freeze`)  
**Evaluation Pipeline Release:** Commit `1292103`  
**Manuscript Reporting & Packaging:** Commit `4038fa2`  

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
   - *Authoritative Methodological Disclosure:* Sepsis cases and controls were restricted using different clinical-process anchors. Differences from canonical discrimination therefore reflect both changes in the evaluated observations and the outcome-specific selection rule; they are not estimates of performance under a uniform prospective eligibility policy.
2. **Earliest-Prescription Non-Estimability Finding:**
   - Under uniform point-assigned earliest prescription start dates ($C_i = T_{\mathrm{Rx,start}, i}$ across all stays), exactly 0 of 151 covered sepsis stays have eligible pre-prescription decision bins ($O_{\mathrm{Rx}}^{\mathrm{point}} = 0, O_{\mathrm{Rx}}^{\mathrm{def}} = 0$).
   - Consequently, the number of horizon-positive observations is zero ($N=0$), precluding discrimination estimation (AUROC and AUPRC are mathematically not estimable under midnight assignment).

---

## 2. Risk Set & Population Denominator Accounting

All evaluated hourly decision bins fall strictly within the prespecified second-day ICU surveillance schedule: $t \in \{19, 20, \dots, 30\}\text{h}$ post-admission (up to 12 windows per stay).

| Risk Set / Partition | Conditioning Clock ($C_i$) | Buffer ($\delta$) | Unique ICU Stays ($N$) | Evaluated Hourly Windows | Surveillance Volume | Positive Windows ($Y=1$) | Negative Windows ($Y=0$) | Positive Prevalence ($\pi$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canonical Risk Set $\mathcal{R}_{\text{canonical}}$** | None (unconstrained) | N/A | 8,254 | 97,700 | 4,070.8 pt-days | 678 | 97,022 | 0.694% |
| **Outcome-Stratified Hybrid Set $\mathcal{R}_0^{\mathrm{hybrid}}$** | $T_{\mathrm{susp}}$ cases / $T_{\mathrm{Rx,start}}$ controls | 0h | 5,967 | 70,192 | 2,924.7 pt-days | 674 | 69,518 | 0.960% |
| **Outcome-Stratified Hybrid Set $\mathcal{R}_2^{\mathrm{hybrid}}$** | $T_{\mathrm{susp}}$ cases / $T_{\mathrm{Rx,start}}$ controls | 2h | 5,967 | 70,129 | 2,922.0 pt-days | 663 | 69,466 | 0.945% |
| **Uniform Earliest-Rx Task ($T_{\mathrm{Rx,start}}$)** | $T_{\mathrm{Rx,start}}$ across all stays | 0h | 5,967 | 70,129 | 2,922.0 pt-days | **0** | 69,518 | **0.0% (Incomputable)** |

*Accounting Note on Control Stays:* Exactly 2,287 non-sepsis control stays had $T_{\mathrm{Rx,start}} \le 19\text{h}$ (prior to surveillance start), censoring all 12 of their decision windows from $\mathcal{R}_0$ ($8,254 - 2,287 = 5,967$ stays; $97,700 - 27,508 = 70,192$ windows).

---

## 3. Comprehensive Mapping of Manuscript Figures & Tables

| Manuscript Item | Displayed Content | Underlying Reference Clock(s) | Primary Evaluated Denominators | Essential Scientific Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **Table 1** | Cohort Progression Ledger | $T_{\text{admit}}$, $T_{\mathrm{susp}}$, $T_{\mathrm{Rx,start}}$, $T_{\text{S3}}$ | Base: 8,254 stays (408 sepsis / 7,846 controls). $\mathcal{R}_0$: 5,967 stays. | Traces stepwise attrition: 257/408 late sepsis unreached ($V=0$); 150/151 covered septic stays eligible before suspicion ($O=1$). |
| **Table 2** | Discrimination Benchmark (8 Architectures) | Hybrid: $T_{\mathrm{susp}}$ cases, $T_{\mathrm{Rx,start}}$ controls | Canonical: 97,700 windows. $\mathcal{R}_0^{\mathrm{hybrid}}$: 70,192 windows. $\mathcal{R}_2^{\mathrm{hybrid}}$: 70,129 windows. | AUROC stable on hybrid sets (BiMamba: 0.6433 on $\mathcal{R}_0^{\mathrm{hybrid}}$ vs. 0.6326 canonical; $\Delta = +0.0107$). Non-estimable under point-assigned earliest prescriptions ($N=0$). |
| **Table 3** | Operational Alert-Budget Frontier (8 Models) | Full surveillance stream; timing adjudicated via $T_{\mathrm{susp}}$ | 8,254 test stays; 4,070.8 patient-days (97,700 windows); 204 alerts (5 ep), 407 alerts (10 ep), 814 alerts (20 ep). | Alerts selected across full stream (4,070.8 pt-d); 100% of detected sepsis stays satisfied two-hour pre-suspicion buffers ($t_{\mathrm{first}, i} + 2\text{h} \le T_{\mathrm{susp}, i}$). PPV: 0.98%–1.72% (NNE: 58.1–101.8). |
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
