# Supplementary Material

**Title:** Surveillance Coverage, Treatment Precedence, and Workload Constraints in Machine Learning Early Warning for Sepsis: An Operational Evaluation  
**Authors:** Alireza Shayan$^1$, Somayeh Afrasiabi$^1$  
$^1$ Department of Computer Science and Engineering & IT, School of Electrical and Computer Engineering, Shiraz University, Shiraz, Iran  
*(Corresponding Author: Somayeh Afrasiabi, afrasiabi@shirazu.ac.ir)*  
**Target Journal:** *Journal of the American Medical Informatics Association (JAMIA)* / *JAMIA Open*  
**Track:** Research and Applications  
**Analytical Release Version:** `1.0.0-freeze` (Analytical Commit `b0932ca`, Submission Packaging Commit `1292103`)  

---

# Section S1: Supplementary Methods Package — Model Development, Cohort Derivation, and Algorithmic Specifications

### S1.1 Cohort Construction, Eligibility Criteria, and Censoring Rules
The study cohort was derived from the Beth Israel Deaconess Medical Center (BIDMC) critical care repository: MIMIC-III (v1.4) under PhysioNet Credentialed Data Use Agreement. To evaluate transportability across electronic health record software systems and prevent temporal leakage, the cohort was partitioned chronologically across clinical software eras:
- **Development Cohort (Training & Tuning):** CareVue clinical information system era (2001–2008).
- **Evaluation Cohort (Primary Prospective-Replay Benchmark):** Successor MetaVision clinical information system era (2008–2012).
- **Prescription-Proxy Sensitivity Cohort:** Historical broad MIMIC-III cohort (2001–2012) evaluated under calendar-day prescription intervals to quantify interval-censored opportunity bounds.

1. **Eligibility Criteria:** Adult patients ($\ge 18$ years of age at ICU admission) admitted to intensive care units with a total ICU length of stay $\text{LOS} \ge 30\text{ hours}$ (calculated as $T_{\text{out}} - T_{\text{in}} \ge 30\text{h}$).
2. **First Admission Restriction:** To prevent within-patient correlation across multiple hospitalizations, only the chronological first ICU stay per unique patient was included.
3. **Multi-Stage Sepsis Exclusions:**
   - *Stage 1 (Pre-prediction Exclusion, D2-C):* Stays where consensus Sepsis-3 criteria were met at or prior to ICU admission ($T_{\text{S3}} \le 0\text{h}$, prevalent on arrival: $N = 3,384$) or where suspected infection/onset occurred after ICU discharge ($N = 1,324$) were excluded from candidate surveillance sets. Stays with dual EHR documentation (`DBSOURCE == 'both'`, $N = 44$) were excluded for era ambiguity.
   - *Stage 2 (Surveillance-Window Eligibility, D1):* Surveillance was prespecified as a second-day ICU protocol evaluating hourly decision bins $[t-1, t]$ for $t \in \{19, 20, \dots, 30\}$ hours post-admission. Patients who developed consensus Sepsis-3 during the pre-surveillance ICU period ($0 < T_{\text{S3}} \le 19\text{h}$, $N = 1,478$ in the MetaVision test partition) were excluded from active surveillance evaluation because acute organ dysfunction was already established prior to protocol initiation.
4. **Surveillance Protocol Interval:** Predictions were evaluated sequentially across 12 scheduled hourly decision bins $[t-1, t]$ for $t \in \{19, 20, \dots, 30\}$ hours post-admission.
5. **Observation-Availability Censoring vs. Administrative ICU Exit Rules:**
   - Because cohort eligibility strictly conditioned on total ICU length of stay $\text{LOS} \ge 30\text{h}$, all 8,254 cohort patients remained administratively in the ICU through at least hour 30. No patient was administratively discharged from the ICU prior to hour 30.
   - Right-censoring during the surveillance interval ($t \in [19, 30]$) occurred through exactly two distinct mechanisms:
     - *Consensus Sepsis-3 Onset ($N=110$ stays, 623 windows):* In accordance with early-warning principles, surveillance ceased immediately once acute Sepsis-3 organ dysfunction was confirmed ($t \ge T_{\text{S3}}$). Subsequent windows were not evaluated.
     - *Observation-Availability Exclusions ($N=180$ control stays, 725 windows):* Under the prespecified feature-generation rule, 725 scheduled decisions in 180 control stays were unavailable because the required charted observations were absent. These were observation-availability exclusions, not administrative ICU discharge. Each hourly decision bin was evaluated independently; once charting ceased in these 180 stays, no subsequent measurements were recorded during the surveillance interval, and scoring did not resume prior to hour 30.

### S1.2 Dataset Partitioning, Subject-Level Allocation Ledger, and Cohort Provenance Bridge
To prevent information leakage across operational EHR transitions, partitioning was executed strictly at the unique patient level (`SUBJECT_ID` / `ICUSTAY_ID`):

#### 1. MIMIC-III Developmental Split (D3 CareVue-to-MetaVision Era Partition):
The base developmental cohort comprises 27,448 unique adult first ICU admissions in MIMIC-III (`artifacts/phase5/d3/d3_cohort_summary.csv`). Each subject contributes exactly one stay (`unique_subjects == icu_stays == 27,448`).

| Partition / Progression Stage | Unique Subjects ($N$) | ICU Stays ($N$) | Sepsis Stays ($N$) | Control Stays ($N$) | Sepsis Prevalence | Allocation Policy & Accounting Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **CareVue Training Set** | 10,371 | 10,371 | 1,350 | 9,021 | 13.02% | 80.0% of CareVue candidates; patient-disjoint (`seed=42`, stratified by incident sepsis) |
| **CareVue Validation / Tuning Set** | 2,593 | 2,593 | 337 | 2,256 | 13.00% | 20.0% of CareVue candidates; patient-disjoint (`seed=42`, stratified by incident sepsis) |
| *CareVue Candidate Subtotal* | *12,964* | *12,964* | *1,687* | *11,277* | *13.01%* | *100.0% of eligible CareVue prediction candidates (CareVue bedside era, 2001–2008)* |
| **MetaVision Held-Out Era Test Candidates** | 9,732 | 9,732 | 1,886 | 7,846 | 19.38% | 100.0% of MetaVision prediction candidates (successor EHR era, 2008–2012) |
| **Excluded Transition Era** | 44 | 44 | 8 | 36 | 18.18% | Stays with `DBSOURCE == 'both'` during 2008 system transition; excluded for era ambiguity |
| **Prevalent Sepsis on Arrival Excluded** | 3,384 | 3,384 | 3,384 | 0 | 100.0% | Onset at or before admission ($T_{\text{S3}} \le 0\text{h}$); excluded from incident early warning |
| **Post-ICU Onset Excluded** | 1,324 | 1,324 | 0 | 1,324 | 0.0% | Infection/dysfunction onset after ICU discharge; excluded from in-ICU evaluation |
| **Total MIMIC-III Base Cohort** | **27,448** | **27,448** | **6,965** | **20,483** | **25.38%**$^\dagger$ | **All unique first adult ICU admissions meeting base criteria ($\text{LOS} \ge 30\text{h}$)** |

$^\dagger$*Arithmetic note:* Total base cohort sepsis prevalence is $100 \times \frac{6,965}{27,448} = \mathbf{25.38\%}$. The candidate sepsis prevalence prior to transition-source exclusion is $100 \times \frac{3,581}{22,740} = \mathbf{15.75\%}$ ($N=22,740$, comprising 12,964 CareVue candidates, 9,732 MetaVision candidates, and 44 transition-era stays with `DBSOURCE == 'both'`; following exclusion of the 44 transition stays for era ambiguity, the final prediction candidate pool contains 22,696 stays with 3,573 incident sepsis cases: $100 \times \frac{3,573}{22,696} = \mathbf{15.74\%}$).

*Cross-Partition Independence Audit:* Across all partitions, pairwise subject overlap is strictly zero (`train_val_overlap = 0`, `train_test_overlap = 0`, `val_test_overlap = 0`, verified in `artifacts/phase5/d3/d3_overlap_audit.csv`). There was zero cross-partition reassignment, subject addition, or data leakage between CareVue development and MetaVision test partitions.

#### 2. Source-Version-Tagged Primary Evaluation Cohort Bridge (MetaVision Test Set, $N=8,254$):
To clarify the precise relationship between the 9,732 MetaVision candidate stays and the 8,254 evaluated prospective-replay stays:

| Stage / Cohort Layer | Database & Version | Source Table / Selection Query | Unique Patients ($N$) | Unique ICU Stays ($N$) | Sepsis Stays ($N$) | Control Stays ($N$) | Sepsis Prevalence | Stage Transition / Exclusion Criterion |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. MetaVision Test Candidate Cohort** | MIMIC-III v1.4 | `icustays`, `admissions`, `patients` (`DBSOURCE == 'metavision'`) | 9,732 | 9,732 | 1,886 | 7,846 | 19.38% | Adult first ICU stay $\text{LOS} \ge 30\text{h}$; prevalent-on-arrival ($T_{\text{S3}} \le 0$) and post-ICU onset excluded |
| **2. Pre-Surveillance Early Onset Exclusion** | MIMIC-III v1.4 | `artifacts/phase5/d2c_v2/data/prediction_candidate_cohort.csv` | 1,478 | 1,478 | 1,478 | 0 | 100.0% | Sepsis onset during pre-surveillance interval ($0 < T_{\text{S3}} \le 19\text{h}$); patient already septic before hour 19 |
| **3. Primary Prospective-Replay Test Cohort** | MIMIC-III v1.4 | `reports/predictions/canonical_test_predictions_harmonized.parquet` | **8,254** | **8,254** | **408** | **7,846** | **4.94%** | **Evaluated at 12 hourly decision bins $t \in [19, 30]$; 97,700 evaluated windows (4,070.8 patient-days)** |
| &emsp;*3a. Surveillance-Covered ($V=1$)* | MIMIC-III v1.4 | Onset in $[20\text{h}, 36\text{h}]$; lookback intersects $[19, 30]$ | 151 | 151 | 151 | 0 | 100.0% | 678 horizon-positive ($Y=1$) and 511 negative ($Y=0$) evaluated windows; 49.5 pt-days |
| &emsp;*3b. Late-Onset Sepsis ($V=0$)* | MIMIC-III v1.4 | Onset $>36\text{h}$ post-admission (median: 49.0h) | 257 | 257 | 257 | 0 | 100.0% | 3,084 negative baseline surveillance windows ($Y=0$); 128.5 pt-days |
| &emsp;*3c. Non-Sepsis Control Stays* | MIMIC-III v1.4 | No consensus Sepsis-3 during hospitalization | 7,846 | 7,846 | 0 | 7,846 | 0.0% | 93,427 negative baseline surveillance windows ($Y=0$); 3,892.8 pt-days |

*Cohort Join Audit Result:*
- **Matched Primary Stays:** Exactly 8,254 out of 8,254 primary prediction stays (100.0%) map directly to `ICUSTAY_ID` records in the MetaVision candidate partition (`cand_test`).
- **Unmatched Primary Stays:** 0 stays.
- **Duplicate Mappings:** 0 stays (1:1 subject-to-stay bijection).
- **Database Namespace:** MIMIC-III Clinical Database (v1.4), BIDMC.
- **Source Identifier Key:** 6-digit `ICUSTAY_ID` namespace (range: 200021 to 299962).
- **Early-Onset Removal Accounting:** The entire numerical difference between the 9,732 MetaVision candidate population and the 8,254 evaluated surveillance population ($9,732 - 8,254 = 1,478$) consists exclusively of sepsis stays developing onset during the first 19 ICU hours ($1,886 - 408 = 1,478$; control count is identical at 7,846).

#### 3. Relation to Broad MIMIC-III Sensitivity Cohort ($N=3,581$):
- The broad MIMIC-III sensitivity cohort comprises all 3,581 incident sepsis cases identified across the full 27,448-stay MIMIC-III cohort (CareVue training: 1,350; CareVue tuning: 337; MetaVision test: 1,886; transition era: 8).
- This broad cohort was evaluated in Section 3.4 and Table S4 under calendar-day prescription intervals (`prescriptions.startdate`) to quantify the impact of date-level coarseness across historical EHR repositories.

### S1.3 Feature Dictionary, Availability Rules, and Preprocessing Protocol
Surveillance features comprise 21 continuous clinical time series grouped into three clinically grounded categories:
1. **Hemodynamics & Vital Signs (6 variables):** Heart Rate (bpm), Systolic Blood Pressure (mmHg), Diastolic Blood Pressure (mmHg), Mean Arterial Pressure (MAP, mmHg), Respiratory Rate (breaths/min), Temperature (°C).
2. **Oxygenation & Respiratory Support (2 variables):** Peripheral Oxygen Saturation ($\text{SpO}_2$, %), Fraction of Inspired Oxygen ($\text{FiO}_2$, ratio/percent).
3. **Laboratory Biomarkers (13 variables):** White Blood Cell Count (WBC, $\times 10^3/\mu\text{L}$), Platelet Count ($\times 10^3/\mu\text{L}$), Hemoglobin (g/dL), Hematocrit (%), Serum Creatinine (mg/dL), Blood Urea Nitrogen (BUN, mg/dL), Serum Glucose (mg/dL), Serum Potassium (mEq/L), Serum Lactate (mmol/L), Base Excess (mEq/L), Arterial pH, Partial Thromboplastin Time (PTT, sec), Aspartate Aminotransferase (AST, U/L).

#### Feature ItemID Cross-Reference and Physiological Validity Bounds

| Clinical Variable | Category | CareVue ItemIDs | MetaVision ItemIDs | Plausible Range | Hourly Aggregation |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Heart Rate** | Hemodynamics | `211`, `220045` | `220045` | 20–250 bpm | Mean |
| **Systolic BP** | Hemodynamics | `51`, `442`, `455`, `6701`, `220179`, `220050` | `220050`, `220179` | 40–300 mmHg | Mean |
| **Diastolic BP** | Hemodynamics | `8368`, `8440`, `8441`, `8555`, `220180`, `220051` | `220051`, `220180` | 20–200 mmHg | Mean |
| **MAP** | Hemodynamics | `52`, `456`, `443`, `220052`, `220181` | `220052`, `220181` | 20–220 mmHg | Mean |
| **Respiratory Rate** | Hemodynamics | `618`, `615`, `220210`, `224690` | `220210` | 4–60 /min | Mean |
| **Temperature** | Hemodynamics | `676`, `677`, `678`, `679`, `223762`, `223761` | `223761`, `223762` | 25–45 °C | Mean (conv. from °F) |
| **$\text{SpO}_2$** | Oxygenation | `646`, `220277` | `220277` | 50–100 % | Mean |
| **$\text{FiO}_2$** | Oxygenation / Support | `190`, `2981`, `3420`, `223835` | `223835` | 0.21–1.00 | Max |
| **WBC** | Laboratory | `51300`, `51301` | `51301` | 0.1–150 $\times 10^3/\mu\text{L}$ | Last charted |
| **Platelets** | Laboratory | `51265` | `51265` | 1–2000 $\times 10^3/\mu\text{L}$ | Last charted |
| **Hemoglobin** | Laboratory | `51222` | `51222` | 2–25 g/dL | Last charted |
| **Hematocrit** | Laboratory | `50399`, `51221` | `51221` | 10–75 % | Last charted |
| **Creatinine** | Laboratory | `50912` | `50912` | 0.1–30 mg/dL | Last charted |
| **BUN** | Laboratory | `51006` | `51006` | 1–250 mg/dL | Last charted |
| **Glucose** | Laboratory | `50931`, `807`, `811`, `1529`, `225664` | `50931`, `225664` | 10–1000 mg/dL | Mean |
| **Potassium** | Laboratory | `50971`, `50822` | `50971`, `50822` | 1.0–10.0 mEq/L | Last charted |
| **Lactate** | Laboratory | `50813` | `50813` | 0.1–30 mmol/L | Last charted |
| **Base Excess** | Laboratory | `50802` | `50802` | -30 to +30 mEq/L | Last charted |
| **Arterial pH** | Laboratory | `50820` | `50820` | 6.5–8.0 | Last charted |
| **PTT** | Laboratory | `51275` | `51275` | 10–150 sec | Last charted |
| **AST** | Laboratory | `50878` | `50878` | 1–10,000 U/L | Last charted |

**Temporal Latency, Availability, Lookback Boundary, and Pre-ICU Handling:**
- Features were extracted strictly within a causal right-open interval $[t-20\text{h}, t)$ preceding decision time $t$, comprising 20 hourly bins: $t-20, t-19, \dots, t-1$. No data timestamped at or after $t$ were accessible.
- **Handling of First Prediction Bin ($t=19\text{h}$):** For decision bin $t=19$, the lookback interval is $[19-20, 19) = [-1, 19)$. The earliest hourly bin (bin 0, corresponding to $[-1, 0)$ hours relative to ICU admission) represents the immediate pre-admission hour. If emergency department or pre-ICU observations were recorded in this interval, they were incorporated directly into bin 0; if unobserved, bin 0 was imputed using training medians and flagged with binary observation indicator `obs = 0` (and `in_icu_mask_h0 = 0`), ensuring strict causal temporal indexing and consistent tensor dimensions.
- Vital signs and oxygenation variables were carried forward for a maximum duration of 4 hours.
- Laboratory measurements were carried forward for a maximum duration of 24 hours within the 20-hour window. Data prior to the lookback boundary $[t-20\text{h}]$ were not retrieved.
- Unobserved segments prior to the initial measurement were imputed using training-set medians.
- *Measurement vs. Result Availability:* Because laboratory result-availability timestamps were not directly available in the source table used (`LABEVENTS` in MIMIC-III v1.4 contains collection `CHARTTIME` but lacks a distinct verification timestamp), the sequential replay applied a prespecified one-hour lag to recorded laboratory times ($T_{\text{available}} = T_{\text{chart}} + 1.0\text{h}$). This 1.0-hour lag represents an analytical scenario to evaluate decision latency under simulated reporting delay, rather than verified clinical availability for all laboratory analytes.

### S1.4 Model Architectures, Feature Representation Audit, and Training Protocols

#### Prediction-Time Leakage Audit & Input Representation Specification
To ensure strict temporal causality and prevent prediction-time leakage:
1. **D6 Observation Mask Sequence Representation:**
   - Under the canonical D6 explicit contract (`assemble_d6_tensor` in `src/models/d6_observation_mask.py`), sequence models consume a 3D tensor of shape $(N, 20, 2F)$ where $F=21$ base clinical variables and $2F=42$ channels per hourly timestep over the causal 20-hour lookback $[t-20\text{h}, t)$.
   - At each timestep, channels follow deterministic block order: 21 continuous normalized physiological values ($v_t$) concatenated with 21 explicit binary observation indicators ($m_t \in \{0, 1\}$).
   - Under D6 mode="off", sequence models consume only the 21 continuous values: $(N, 20, 21)$.
2. **Audit of Channel 4 Baseline Offset & Clarification of Drafting Text:**
   - A previous textual draft stated that Channel 4 used a patient-specific baseline computed across the "first 24 hours of ICU stay". That statement was a descriptive error that contradicted both the production pipeline and the codebase utilities.
   - *Production Sequence Models:* The production sequence architectures (BiMamba SSM, Mask-Aware Transformer, Baseline Transformer) consume strictly the D6 tensor ($2F=42$ or $F=21$), which contains **no baseline offset channel**.
   - *Data Loader Historical Baseline Utility:* In development data loading utilities where baseline offsets were computed (`compute_stay_baselines` in `src/models/benchmark_data_loader.py`), the function executed `d.drop_duplicates(subset=["ICUSTAY_ID"], keep="first")`, which extracted baseline values exclusively from the patient's *earliest evaluated historical window* ($t=19$, lookback $[0, 19\text{h})$ prior to surveillance onset). It never accessed observations from hours 20 to 24 or any future decision point.
   - *Time-Truncation Invariance:* All features are strict causal functions of data timestamped prior to decision time $t$ ($t$ right-open excluded). Feature vectors at hour $t$ are invariant to future censoring, treatment, or subsequent length of stay.
   - *Executable Verification Audit:* Formal automated testing across 50 stays, 12 decision hours, 547 unique patient-decision pairs, and 1,641 equality comparisons in `reports/manifest/time_truncation_invariance_audit.json` confirmed that removing future data yields an exact maximum feature discrepancy of `0.000000` for both sequence tensors and tabular summaries (PASS: no feature discrepancies detected in the tested prediction-time truncation comparisons).
3. **Tabular Feature Representation:**
   - Tree-based and linear models (LightGBM, XGBoost, Random Forest, L2-Logistic Regression) were evaluated using a mask-aware 6-summary vector (`compute_tabular_summary` in `src/models/benchmark_data_loader.py`). For each of the 21 variables, 6 summary statistics are computed across observed values within $[t-20\text{h}, t)$: mean, minimum, maximum, last charted value, overall trend (last minus first), and linear trajectory slope. This produces exactly $6 \times 21 = 126$ engineered input features.
   - *Handling of Sparse Observations in Summaries:* When a variable has zero observations within the lookback window ($cnt = 0$), mean, min, max, last, trend, and slope are imputed as 0.0. When exactly one observation is recorded ($cnt = 1$), trend and slope are set to 0.0, while mean, min, max, and last reflect the single observed measurement. Slope calculation uses least-squares regression with denominator check $\Delta t > 0$.

#### Authoritative Per-Model Serialized Input Specification

| Model Architecture | Input Representation Type | Serialized Input Shape / Dimension | Feature Channel Composition | Lookback Window |
| :--- | :--- | :---: | :--- | :---: |
| **BiMamba SSM (Proposed)** | 3D Sequence Tensor | $(N, 20, 42)$ | 21 normalized values + 21 binary observation masks | $[t-20\text{h}, t)$ (20 bins) |
| **Mask-Aware Clinical Transformer** | 3D Sequence Tensor | $(N, 20, 42)$ | 21 normalized values + 21 binary observation masks | $[t-20\text{h}, t)$ (20 bins) |
| **Baseline Unmasked Transformer** | 3D Sequence Tensor | $(N, 20, 21)$ | 21 normalized values only | $[t-20\text{h}, t)$ (20 bins) |
| **Tabular GBDT (LightGBM)** | 2D Summary Vector | $(N, 126)$ | 6 mask-aware summaries $\times$ 21 variables (mean, min, max, last, trend, slope) | $[t-20\text{h}, t)$ (20 bins) |
| **Tabular GBDT (XGBoost)** | 2D Summary Vector | $(N, 126)$ | 6 mask-aware summaries $\times$ 21 variables (mean, min, max, last, trend, slope) | $[t-20\text{h}, t)$ (20 bins) |
| **Random Forest** | 2D Summary Vector | $(N, 126)$ | 6 mask-aware summaries $\times$ 21 variables (mean, min, max, last, trend, slope) | $[t-20\text{h}, t)$ (20 bins) |
| **Logistic Regression (L2)** | 2D Summary Vector | $(N, 126)$ | 6 mask-aware summaries $\times$ 21 variables (mean, min, max, last, trend, slope) | $[t-20\text{h}, t)$ (20 bins) |
| **Multi-Model Stacking Ensemble** | 2D Meta-Feature Vector | $(N, 7)$ | Out-of-fold validation log-odds predictions from all 7 base architectures | Evaluated window $t$ |

#### Calibration Disambiguation: Temperature Scaling vs. Platt Scaling
Post-hoc probability calibration was fitted to improve agreement between predicted probabilities and observed outcomes:
1. **Temperature Scaling (1 Parameter):** A single scalar parameter $T > 0$ divides model logits $z$:
   $$p_{\text{cal}} = \sigma\left(\frac{z}{T}\right) = \frac{1}{1 + e^{-z/T}}$$
   Fitted via bounded scalar optimization on validation expected calibration error (ECE). Because $T > 0$ preserves strict monotonic order, temperature scaling preserves AUROC and AUPRC exactly.
2. **Platt Scaling (2 Parameters):** A univariate logistic regression model fitted on uncalibrated logits $z$ (or log-odds):
   $$p_{\text{cal}} = \sigma(a \cdot z + b) = \frac{1}{1 + e^{-(a \cdot z + b)}}$$
   with parameters $(a, b)$ estimated via maximum likelihood on the held-out validation split.
3. **Score Scale in Operational Alert Tables:** Operating thresholds ($\tau$) and IEEE 754 boundary ties in Main Table 3, Table S3A, and Table S3B are reported directly on the calibrated probability scale produced by validation recalibration (Temperature scaling for deep sequence models: $T = 1.15$; Platt scaling / calibrated probabilities for tabular baselines). Because temperature scaling ($T > 0$) and Platt scaling ($a > 0$, fitted $a \approx 0.92$) are strictly monotonic transformations, the set of alerted patient stays ($\mathcal{S}_{\text{alerted}}(K)$) and resulting detection counts are identical under calibrated and uncalibrated rankings.

#### Evaluated Architectures and Hyperparameters
1. **Bidirectional Mamba State-Space Model (BiMamba SSM):**
   - 4 bidirectional selective SSM layers ($d_{\text{state}} = 16$, $d_{\text{model}} = 64$, convolution width $d_{\text{conv}} = 4$, expansion factor $E = 2$).
   - Forward and backward SSM states combined via concatenation followed by linear projection.
   - Optimizer: Adam ($lr = 5 \times 10^{-4}$, weight decay $1 \times 10^{-5}$), batch size 256, max epochs 40.
   - Loss function: Focal Loss ($\gamma = 2.0, \alpha = 0.25$) combined with Maximum Mean Calibration Error (MMCE) penalty ($\lambda = 0.3, \text{bandwidth} = 0.4$).
   - Early stopping: patience = 8 epochs monitoring validation AUPRC.
2. **Mask-Aware Clinical Transformer:**
   - 4 Transformer encoder layers, 4 attention heads, $d_{\text{model}} = 64$, feedforward dimension 128, dropout 0.1, sinusoidal positional encoding, mean pooling over sequence length. Input: $(N, 20, 42)$.
   - Trained with Adam ($lr = 1 \times 10^{-4}$, weight decay $1 \times 10^{-4}$), batch size 256, binary cross-entropy with focal loss.
3. **Baseline Unmasked Clinical Transformer:**
   - Identical architecture to the Mask-Aware Transformer, but consuming only the 21 continuous normalized value channels: $(N, 20, 21)$.
4. **Gradient-Boosted Decision Trees (LightGBM):**
   - 200 trees, max depth 5, learning rate 0.05, feature fraction 0.8, min child samples 20, random seed 42. Input: $(N, 126)$.
5. **Gradient-Boosted Decision Trees (XGBoost):**
   - 200 trees, max depth 5, learning rate 0.05, subsample 0.8, colsample_bytree 0.8, eval_metric "logloss", random seed 42. Input: $(N, 126)$.
6. **Random Forest:**
   - 300 decision trees, max depth 12, min samples per leaf 10, class_weight "balanced_subsample", random seed 42. Input: $(N, 126)$.
7. **Logistic Regression (L2):**
   - Ridge-regularized logistic regression ($C = 1.0$), solver "lbfgs", max_iter 1,000, class_weight "balanced". Input: $(N, 126)$.
8. **Multi-Model Stacking Ensemble:**
   - L2-regularized logistic regression meta-learner ($C = 1.0$) trained on 5-fold out-of-fold cross-validated validation log-odds predictions from all 7 base architectures. Input: $(N, 7)$.

### S1.5 Executable Sepsis-3 Phenotype Implementation and Multi-Clock Temporal Formalization
The Sepsis-3 clinical consensus phenotype was implemented programmatically as a study-specific operational adaptation of Seymour et al. (JAMA 2016), tailored for retrospective ICU electronic health record surveillance under declared temporal resolution constraints:

1. **Formal Six-Milestone Temporal Tracking (Origin and Five Event Clocks):**
   To resolve temporal precision and event ambiguity without conflating orders, intended start dates, and physical administrations, we explicitly track six operational milestones (with ICU admission as the reference origin $t=0$, followed by five operational event clocks):
   - $T_{\text{admit}, i}$: ICU admission timestamp ($t=0$, reference origin).
   - $T_{\mathrm{Rx,start}, i}$: Earliest documented systemic antimicrobial prescription start date (`prescriptions.startdate`, calendar-date precision, coarsened to midnight 00:00:00).
   - $T_{\mathrm{abx,pair}, i}$: The specific systemic antimicrobial prescription start date participating in the qualifying infection suspicion pair.
   - $T_{\mathrm{culture,pair}, i}$: Blood culture specimen collection timestamp (`microbiologyevents.charttime`, minute-level precision).
   - $T_{\mathrm{susp}, i}$: Operational infection-suspicion anchor, defined per Seymour et al. (2016) as $\min(T_{\mathrm{abx,pair}, i}, T_{\mathrm{culture,pair}, i})$.
   - $T_{\mathrm{S3}, i}$: Assigned consensus Sepsis-3 organ-dysfunction onset timestamp, evaluated at hourly resolution as the earliest integer hour where $\Delta\text{SOFA}(t) = \text{SOFA}(t) - \text{SOFA}_{\text{base}} \ge 2$ within the 36-hour evaluation window $[T_{\mathrm{susp}, i} - 24\text{h}, T_{\mathrm{susp}, i} + 12\text{h}]$.

2. **Decoupling Treatment and Suspicion Clocks & Mathematical Formulation of Search-Window Lead:**
   - When blood culture collection precedes the antimicrobial order ($T_{\mathrm{culture,pair}} < T_{\mathrm{abx,pair}}$), the suspicion anchor is established by culture ($T_{\mathrm{susp}} = T_{\mathrm{culture,pair}} \ne T_{\mathrm{abx,pair}}$). Consequently, the suspicion anchor and the antimicrobial order proxy are not identical ($T_{\mathrm{treatment}} \ne T_{\mathrm{susp}}$).
   - **Mathematical Formulation of the Search-Window Lead ($G = T_{\mathrm{susp}} - T_{\mathrm{S3}} = +24.2\text{h}$ median):**
     In the executable Sepsis-3 algorithm, SOFA scores are discretized to integer hourly bins relative to ICU admission ($t \in \mathbb{Z}$). The retrospective search window for baseline SOFA and acute onset begins at the floored hourly integer:
     $$\text{baseline\_start\_h}_i = \left\lfloor \frac{T_{\mathrm{susp}, i} - 24\text{h} - T_{\text{admit}, i}}{1\text{ hour}} \right\rfloor$$
     When an acute increase $\Delta\text{SOFA} \ge 2$ is first observed at this lookback boundary bin (which occurred in 67.2% of incident sepsis stays; $N=274/408$), $T_{\mathrm{S3}}$ is assigned as the start of that hourly bin relative to admission:
     $$T_{\mathrm{S3}, i} = T_{\text{admit}, i} + \text{baseline\_start\_h}_i \times 1\text{ hour}$$
     Consequently, the operational search-window lead is:
     $$T_{\mathrm{susp}, i} - T_{\mathrm{S3}, i} = 24.0\text{h} + h\,\operatorname{frac}(x_i) < 25.0\text{h}$$
     where $h = 1\text{ hour}$ and $x_i = (T_{\mathrm{susp}, i} - 24h - T_{\text{admit}, i})/h$. The search window ends at the floored hourly integer:
     $$\text{window\_end\_h}_i = \left\lfloor \frac{T_{\mathrm{susp}, i} + 12h - T_{\text{admit}, i}}{h} \right\rfloor$$
     Because the upper boundary bin is floored, evaluated hours never exceed $T_{\mathrm{susp}} + 12h$, strictly guaranteeing $T_{\mathrm{susp}, i} - T_{\text{S3}, i} \ge -12.0h$ (empirical minimum: $-11.2h$). Because $0 \le \operatorname{frac}(\cdot) < 1$, this hourly binning yields an exact analytical bound of $-12.0\text{h} \le T_{\mathrm{susp}, i} - T_{\text{S3}, i} < 25.0\text{h}$ (analytical supremum $25.0\text{h}$), explaining why the empirical cohort distribution exhibits a median of $+24.2\text{h}$ (IQR: $22.8\text{h}$ to $24.6\text{h}$; range: $-11.2\text{h}$ to $+24.99\text{h}$). Across all 274 boundary stays, the difference between $T_{\mathrm{susp}} - T_{\text{S3}}$ and $24.0 + h\,\operatorname{frac}(x_i)$ is identically zero ($< 10^{-13}\text{h}$). In 274 of 408 episodes, the assigned onset coincided with the first evaluated search bin. Consequently, onset timing and derived warning intervals were sensitive to the search-window boundary and bin-assignment convention and should not be interpreted as directly observed biological onset.

3. **Complete 24-Class Antimicrobial Dictionary and Normalization Rules:**
   Qualifying systemic antimicrobials were identified across 24 pharmacological classes from BIDMC hospital formulary records in `prescriptions`:
   1. *Natural & Aminopenicillins:* Ampicillin, amoxicillin, penicillin G, penicillin V.
   2. *Antipseudomonal & Extended-Spectrum Penicillins:* Piperacillin, ticarcillin, mezlocillin.
   3. *Penicillinase-Resistant Penicillins:* Nafcillin, oxacillin, dicloxacillin.
   4. *Beta-Lactamase Inhibitor Combinations:* Piperacillin-tazobactam, ampicillin-sulbactam, amoxicillin-clavulanate.
   5. *First-Generation Cephalosporins:* Cefazolin, cephalexin, cefadroxil.
   6. *Second-Generation Cephalosporins:* Cefuroxime, cefoxitin, cefotetan, cefaclor, cefprozil.
   7. *Third-Generation Cephalosporins:* Ceftriaxone, ceftazidime, cefotaxime, cefpodoxime, cefdinir.
   8. *Fourth-Generation Cephalosporins:* Cefepime.
   9. *Fifth-Generation / Advanced Cephalosporins:* Ceftaroline fosamil.
   10. *Carbapenems:* Meropenem, imipenem-cilastatin, ertapenem, doripenem.
   11. *Monobactams:* Aztreonam.
   12. *Fluoroquinolones:* Ciprofloxacin, levofloxacin, moxifloxacin.
   13. *Aminoglycosides:* Gentamicin, tobramycin, amikacin, streptomycin.
   14. *Glycopeptides:* Vancomycin.
   15. *Lipopeptides & Lipoglycopeptides:* Daptomycin.
   16. *Oxazolidinones:* Linezolid.
   17. *Macrolides & Ketolides:* Azithromycin, clarithromycin, erythromycin.
   18. *Tetracyclines & Glycylcyclines:* Doxycycline, minocycline, tigecycline.
   19. *Sulfonamides & Trimethoprim:* Sulfamethoxazole-trimethoprim (co-trimoxazole).
   20. *Nitroimidazoles:* Metronidazole.
   21. *Lincosamides:* Clindamycin.
   22. *Polymyxins:* Colistin (colistimethate sodium), polymyxin B.
   23. *Rifamycins:* Rifampin.
   24. *Other Systemic Antibacterials, Antifungals, & Antivirals:* Nitrofurantoin, fosfomycin, fluconazole, voriconazole, posaconazole, amphotericin B, micafungin, caspofungin, anidulafungin, acyclovir, ganciclovir.

   *Route Normalization & Exclusions:* Enforced systemic routes: `IV`, `IVPB`, `IV BOLUS`, `IV DRIP`, `IV INFUSION`, `INTRAVENOUS`, `PO`, `PO/NG`, `NG`, `GT`, `OGT`, `ORAL`, `NASOGASTRIC`, `GASTROSTOMY TUBE`. Strictly excluded: topical formulations (creams, ointments), ophthalmic/otic drops, catheter flushes, bladder irrigations, inhalations, and intravenous additives (`DRUG_TYPE == "ADDITIVE"`).

#### Table S1C: Medication Clock Specification, Qualifying Formulations, and Ascertainment Ledger

| Specification Dimension | Operational Rule / Parameter | Source Table & Column | Population Count / Coverage |
| :--- | :--- | :--- | :---: |
| **Primary Clock Definition** | Prescription-start date proxy ($T_{\mathrm{Rx,start}}$) paired with blood culture collection ($T_{\mathrm{culture,pair}}$) defining infection suspicion $T_{\mathrm{susp}}$ | `prescriptions.startdate`, `microbiologyevents.charttime` | Evaluated across all 8,254 test stays |
| **Documented In-Hospital Orders** | Stays with documented qualifying systemic antimicrobial orders during hospital stay | `prescriptions` (systemic routes) | **2,778 / 8,254 (33.7%)** |
| &emsp;*Sepsis Stays ($S=1$)* | Consensus Sepsis-3 cases with confirmed antibiotic order paired with blood culture | `prescriptions` + `microbiologyevents` | **408 / 408 (100.0%)** |
| &emsp;*Control Stays ($S=0$)* | Non-sepsis controls with documented systemic antimicrobial orders for non-septic indications | `prescriptions` (systemic routes) | **2,370 / 7,846 (30.2%)** |
| **Undocumented In-Hospital Orders** | No qualifying prescription record was identified in the extracted hospitalization data. This does not verify absence of treatment ($T_{\mathrm{Rx,start}} = \infty$). | `prescriptions` | **5,476 / 7,846 (69.8%)** |
| **Qualifying Antimicrobial Classes** | 24 major antimicrobial agent classes enumerated in Section S1.5 (penicillins through systemic antivirals) | `prescriptions.drug` | 24 enumerated agent classes |
| **Qualifying Administration Routes** | Systemic routes only: IV, IVPB, IV Bolus, IV Drip, IV Infusion, Intravenous, PO, PO/NG, NG, GT, OGT, Oral, Nasogastric, Gastrostomy Tube | `prescriptions.route` | All systemic enteral and parenteral routes |
| **Excluded Routes & Formulations** | Topical creams, ophthalmic/otic solutions, bladder irrigations, inhalations, and additives (`DRUG_TYPE == "ADDITIVE"`) | `prescriptions.route`, `prescriptions.drug_type` | Strictly excluded from proxy clock |
| **Pre-Surveillance Orders ($T_{\mathrm{Rx,start}} \le 19\text{h}$)** | Stays receiving antimicrobial orders prior to hour 19; censored from pre-treatment risk set $\mathcal{R}_0$. Exactly 2,287 stays had $T_{\mathrm{Rx,start}} < 19.0\text{h}$; 0 stays had $T_{\mathrm{Rx,start}} == 19.0\text{h}$. | `T_TREATMENT_H_ICU <= 19.0` | 2,287 stays (2,278 controls, 9 sepsis) |
| **Retained Pretreatment Stays ($\mathcal{R}_0$)** | Stays retaining active surveillance decision windows at or before documented treatment ($w_{\mathrm{end}} \le T_{\mathrm{Rx,start}}$) | $t \le T_{\mathrm{Rx,start}}$ | **5,967 stays** ($8,254 - 2,287 = 5,967$; 150/151 covered sepsis) |
| **Ascertainment Boundaries** | Pre-hospital EMS administrations and outpatient prescriptions are unobserved in retrospective EHR tables. No qualifying prescription record was identified in the extracted hospitalization data. This does not verify absence of treatment. | Inpatient EHR tables (`HADM_ID`) | Hospitalization observation boundary |
4. **Directional Suspicion Pairing Rules ($T_{\text{susp}}$):**
   - *Antimicrobial precedes Blood Culture:* Blood culture must follow within $\le 24\text{ hours}$ ($0 \le t_{\text{culture}} - t_{\text{abx}} \le 24\text{h}$); $T_{\text{susp}} = t_{\text{abx}}$.
   - *Blood Culture precedes Antimicrobial:* Antimicrobial order must follow within $\le 72\text{ hours}$ ($0 < t_{\text{abx}} - t_{\text{culture}} \le 72\text{h}$); $T_{\text{susp}} = t_{\text{culture}}$.
   - *Multiple Candidate Pairs & Complete Episode Selection:* All candidate culture–antibiotic pairs within the hospital admission (`HADM_ID`) were evaluated. If an earlier suspected-infection pair did not produce acute organ dysfunction ($\Delta\text{SOFA} \ge 2$), subsequent candidate pairs within the admission were evaluated. The earliest qualifying pair that satisfies $\Delta\text{SOFA} \ge 2$ within $[T_{\text{susp}} - 24\text{h}, T_{\text{susp}} + 12\text{h}]$ confirms consensus Sepsis-3, and $T_{\text{S3}}$ is established at the earliest hour of confirmed organ dysfunction.
5. **Baseline SOFA ($\text{SOFA}_{\text{base}}$):**
   - Evaluated strictly over the 24-hour interval preceding suspicion: $[T_{\text{susp}} - 24\text{h}, T_{\text{susp}}]$.
   - $\text{SOFA}_{\text{base}} = \min\{\text{SOFA}(t) : t \in [T_{\text{susp}} - 24\text{h}, T_{\text{susp}}]\}$.
   - If no physiological or laboratory data were recorded in this interval, baseline SOFA was imputed as 0 (normal organ function prior to acute deterioration, adhering to Seymour et al. Table S2).
6. **Organ-Dysfunction Evaluation Window & Sepsis-3 Onset ($T_{\text{S3}}$):**
   - Evaluated across the 36-hour clinical evaluation window $[T_{\text{susp}} - 24\text{h}, T_{\text{susp}} + 12\text{h}]$.
   - Acute organ dysfunction is confirmed at the earliest integer hour $t$ satisfying:
     $$\Delta\text{SOFA}(t) = \text{SOFA}(t) - \text{SOFA}_{\text{base}} \ge 2$$
   - Sepsis-3 onset timestamp:
     $$T_{\text{S3}} = \min\bigl\{ t \in [T_{\text{susp}} - 24\text{h}, T_{\text{susp}} + 12\text{h}] : \text{SOFA}(t) - \text{SOFA}_{\text{base}} \ge 2 \bigr\}$$

### S1.6 Patient-Clustered Paired Bootstrap and Boundary Tie-Breaking
1. **Clustered Bootstrap Resampling:** Resampling was performed at the unique ICU stay level (`ICUSTAY_ID`, $B = 1,000$ iterations) to preserve within-patient temporal dependencies.
2. **Preservation of Multiplicity:** Duplicate selections of the same patient within a resample were indexed via an integer `copy_idx` ($0, 1, \dots$), maintaining identical exposure and alert multiplicity.
3. **Dynamic Exposure Recomputation (Procedure B):** In each resample $b$, surveillance exposure was computed as $E_b = \frac{1}{24}\sum_{i \in b} h_i$, and the alert quota was dynamically recalculated as $K_b = \operatorname{round}(q \cdot E_b / 100)$.
4. **Deterministic Float64 Tie-Breaking Rule:** Patients were prioritized using the lexicographical key:
   $$\mathcal{B} = \bigl( -\max_{t} s_i(t), \;\text{ICUSTAY\_ID}, \;\text{copy\_idx} \bigr)$$
5. **First-Alert Selection Predicate:** Detection timing was evaluated strictly at the first eligible threshold crossing:
   $$t_{\mathrm{first}, i} = \min \bigl\{ t \in \mathcal{T}_i : \operatorname{Elig}_i(t; \tau, \mathcal{B}) = 1 \bigr\}$$
   where $\operatorname{Elig}_i(t; \tau, \mathcal{B}) = \mathbf{1}\{i \in \mathcal{S}_{\text{alerted}}(K) \;\land\; s_i(t) \ge \tau\}$.
6. **Exact Reciprocal NNE Intervals:** For each resample, NNE was calculated as $\text{Alerts} / \text{Detections}$. Percentile confidence intervals were computed via exact reciprocal transformation of unrounded PPV percentiles ($\text{NNE}_{\text{low}} = 100 / \text{PPV}_{\text{upper}}$, $\text{NNE}_{\text{high}} = 100 / \text{PPV}_{\text{low}}$), setting $\text{NNE}_{\text{high}} = +\infty$ when $\text{PPV}_{\text{low}} = 0$.

---

## Table S1: Complete Cohort Inclusion, Exclusion, and Progression Ledger

| Progression Stage / Cohort Partition | ICU Stays ($N$) | Evaluated Decision Windows | Surveillance Volume | Proportion / Window Prevalence | Clinical Definition & Measurement Convention |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Base Prospective-Replay Test Cohort** | 8,254 | 97,700 (out of 99,048 scheduled) | 4,070.8 evaluation-bin patient-days (97,700 patient-hours) | 100.0% of cohort | Adult first ICU admissions (approx. 2017–2019) conditioned on total ICU stay $\text{LOS} \ge 30\text{h}$. Evaluated at up to 12 scheduled hourly decision bins $[t-1, t]$ at $t \in \{19\dots30\}$. 1,348 windows right-censored (623 post-sepsis onset, 725 observation-availability exclusions). |
| **2. Non-Sepsis Control ICU Stays** | 7,846 | 93,427 outcome-negative | 3,892.8 patient-days (93,427 patient-hours) | 95.06% of stays | Patients without consensus Sepsis-3 during hospitalization. All windows serve as outcome-negative surveillance (180 stays ceased charting prior to hour 30 due to routine observation-availability exclusions, accounting for 725 windows; all stays had total $\text{LOS} \ge 30\text{h}$). |
| **3. Consensus Sepsis-3 Stays ($S=1$)** | 408 | 4,273 windows (678 positive, 3,595 negative) | 178.0 patient-days (4,273 patient-hours) | 4.94% of stays | Patients meeting Sepsis-3 consensus criteria (blood culture + systemic antibiotic + acute $\Delta\text{SOFA} \ge 2$). Median onset: hour 49.0 (IQR: 30.0h–77.0h; range: 20h–369h). 110 stays right-censored upon onset prior to hour 30 (623 windows). |
| &emsp;**3a. Late-Onset Sepsis (Onset $>36\text{h}$)** | 257 | 3,084 outcome-negative windows | 128.5 patient-days (3,084 patient-hours) | 63.0% of septic stays | Onset occurs after hour 36. 6-hour lookback $[T_{\text{S3}} - 6, T_{\text{S3}}]$ does not intersect surveillance window $[19, 30]$. All windows are outcome-negative baseline surveillance ($V_i = 0$). |
| &emsp;**3b. Surveillance-Covered Sepsis ($V=1$)** | 151 | 1,189 windows (678 positive, 511 negative) | 49.5 patient-days (1,189 patient-hours) | 37.0% of septic stays | Onset occurs between ICU hours 20 and 36, such that $[T_{\text{S3}} - 6\text{h}, T_{\text{S3}}]$ intersects surveillance interval $[19, 30]$ ($V_i = 1$). |
| **4. Canonical Sepsis-3 Risk Set ($H=6\text{h}$)** | 8,254 (151 horizon-pos) | 97,700 (678 pos / 97,022 neg) | 4,070.8 patient-days | 0.694% window prevalence | Full unconstrained risk set. Target windows have end-time within 6 hours before Sepsis-3 onset ($0 < T_{\text{S3}} - t \le 6\text{h}$). |
| **5. Outcome-Stratified Hybrid Set $\mathcal{R}_0^{\mathrm{hybrid}}$ ($\delta=0\text{h}$)** | 5,967 (150 horizon-pos) | 70,192 (674 pos / 69,518 neg) | 2,924.7 patient-days | 0.960% window prevalence | Exploratory outcome-stratified hybrid restriction censoring cases at infection-suspicion anchor $T_{\mathrm{susp}}$ and controls at earliest prescription $T_{\mathrm{Rx,start}}$. 150 out of 151 covered patients (99.3%) maintain $\ge 1$ pre-suspicion target window. |
| **6. Outcome-Stratified Hybrid Set $\mathcal{R}_2^{\mathrm{hybrid}}$ ($\delta=2\text{h}$)** | 5,967 (149 horizon-pos) | 70,129 (663 pos / 69,466 neg) | 2,922.0 patient-days | 0.945% window prevalence | Exploratory outcome-stratified hybrid restriction requiring a 2-hour clinical buffer ($w_{\text{end}} + 2\text{h} \le C_i$). 149 out of 151 covered patients (98.7%) maintain $\ge 1$ actionable target window. |
| **7. Uniform Earliest-Rx Task $\mathcal{R}_0^{\mathrm{Rx}}$ ($\delta=0\text{h}$)** | 5,559 (0 horizon-pos) | 65,926 (0 pos / 65,926 neg) | 2,746.9 patient-days | 0.000% (Not Estimable) | Uniform eligibility task requiring $t \le T_{\mathrm{Rx,start}}$ across all stays. Under declared midnight assignment of prescription-start dates, no horizon-positive observations remained (0 positive, 65,926 negative; $N_{\mathrm{total}} = 65,926$), preventing discrimination estimation. |
| **8. Uniform Earliest-Rx Task $\mathcal{R}_2^{\mathrm{Rx}}$ ($\delta=2\text{h}$)** | 5,559 (0 horizon-pos) | 65,877 (0 pos / 65,877 neg) | 2,744.9 patient-days | 0.000% (Not Estimable) | Uniform eligibility task requiring $t + 2\text{h} \le T_{\mathrm{Rx,start}}$ across all stays. Under declared midnight assignment of prescription-start dates, no horizon-positive observations remained (0 positive, 65,877 negative; $N_{\mathrm{total}} = 65,877$), preventing discrimination estimation. |

---

## Table S2: Complete Model Discrimination and Pairwise Differences

*Held-out point estimates ($B=0$) and $B=1,000$ patient-clustered bootstrap percentile 95% confidence intervals.*

| Model Architecture | Canonical Sepsis-3 ($H=6\text{h}$)<br>Prev: 0.694%<br>AUROC [95% CI] | Canonical Sepsis-3 ($H=6\text{h}$)<br>Prev: 0.694%<br>AUPRC % [95% CI] | Outcome-Stratified Hybrid Set ($\mathcal{R}_0^{\mathrm{hybrid}}$, $\delta=0\text{h}$)<br>Prev: 0.960%<br>AUROC [95% CI] | Outcome-Stratified Hybrid Set ($\mathcal{R}_0^{\mathrm{hybrid}}$, $\delta=0\text{h}$)<br>Prev: 0.960%<br>AUPRC % [95% CI] | Outcome-Stratified Hybrid Set ($\mathcal{R}_2^{\mathrm{hybrid}}$, $\delta=2\text{h}$)<br>Prev: 0.945%<br>AUROC [95% CI] | Outcome-Stratified Hybrid Set ($\mathcal{R}_2^{\mathrm{hybrid}}$, $\delta=2\text{h}$)<br>Prev: 0.945%<br>AUPRC % [95% CI] | Paired $\Delta\text{AUROC}$ ($\mathcal{R}_0^{\mathrm{hybrid}} - \text{Canonical}$)<br>[95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BiMamba SSM (Proposed)** | 0.6326 (0.5872–0.6733) | 1.209% (0.892%–1.769%) | 0.6433 (0.5998–0.6835) | 1.656% (1.235%–2.329%) | 0.6447 (0.6002–0.6860) | 1.647% (1.220%–2.331%) | +0.0107 (+0.0060 to +0.0175) |
| **Multi-Model Stacking Ensemble** | 0.6284 (0.5829–0.6699) | 1.206% (0.882%–1.748%) | 0.6351 (0.5897–0.6771) | 1.621% (1.207%–2.263%) | 0.6370 (0.5913–0.6799) | 1.613% (1.198%–2.255%) | +0.0067 (+0.0021 to +0.0137) |
| **Random Forest** | 0.6108 (0.5643–0.6535) | 1.080% (0.821%–1.434%) | 0.6143 (0.5684–0.6575) | 1.470% (1.138%–1.920%) | 0.6173 (0.5712–0.6612) | 1.461% (1.120%–1.916%) | +0.0034 (-0.0008 to +0.0101) |
| **Mask-Aware Clinical Transformer** | 0.5942 (0.5467–0.6410) | 1.013% (0.766%–1.303%) | 0.5961 (0.5490–0.6422) | 1.382% (1.048%–1.760%) | 0.5954 (0.5480–0.6426) | 1.365% (1.032%–1.752%) | +0.0020 (-0.0024 to +0.0083) |
| **Tabular GBDT (XGBoost)** | 0.5670 (0.5208–0.6102) | 0.977% (0.739%–1.357%) | 0.5643 (0.5186–0.6082) | 1.320% (1.008%–1.817%) | 0.5652 (0.5192–0.6099) | 1.313% (0.998%–1.811%) | -0.0027 (-0.0070 to +0.0038) |
| **Tabular GBDT (LightGBM)** | 0.5613 (0.5161–0.6026) | 0.893% (0.694%–1.195%) | 0.5587 (0.5139–0.6010) | 1.222% (0.957%–1.629%) | 0.5598 (0.5145–0.6033) | 1.214% (0.949%–1.630%) | -0.0026 (-0.0070 to +0.0038) |
| **Logistic Regression (L2)** | 0.5593 (0.5097–0.6048) | 0.939% (0.716%–1.218%) | 0.5654 (0.5157–0.6107) | 1.337% (1.012%–1.740%) | 0.5657 (0.5150–0.6127) | 1.325% (1.002%–1.731%) | +0.0061 (+0.0023 to +0.0116) |
| **Clinical Transformer (Baseline)** | 0.5239 (0.4808–0.5653) | 0.808% (0.609%–1.381%) | 0.5296 (0.4851–0.5712) | 1.147% (0.857%–1.904%) | 0.5286 (0.4834–0.5705) | 1.131% (0.845%–1.894%) | +0.0057 (+0.0019 to +0.0110) |

*Note: Evaluated on hybrid risk sets $\mathcal{R}_0^{\mathrm{hybrid}}$ and $\mathcal{R}_2^{\mathrm{hybrid}}$ censoring cases at infection-suspicion anchor $T_{\mathrm{susp}}$ and controls at earliest prescription $T_{\mathrm{Rx,start}}$. Sepsis cases and controls were restricted using different clinical-process anchors; differences from canonical discrimination therefore reflect both evaluated observations and outcome-specific selection rather than uniform prospective eligibility. Under uniform point-assigned earliest prescription restriction ($C_i = T_{\mathrm{Rx,start}, i}$), discrimination is not estimable due to zero retained positive cases ($O_{\mathrm{Rx}}^{\mathrm{point}} = 0$).*

---

## Table S3A: Authoritative Operational Alert-Budget Workload Under 12h Refractory Policy

*Directly serialized from canonical `reports/tables/Table3_Alert_Budget_Refractory_Workload.csv` under Procedure B (Empirical Burden-Matched Bootstrap, $B=1,000$). At each resample, exposure $E_b = \frac{1}{24}\sum h_i$ is dynamically recomputed to adjust the target alert count $K_b = \operatorname{round}(q \cdot E_b / 100)$. Secondary key sorting `[-max_score, ICUSTAY_ID, copy_idx]` preserves bootstrap multiplicity. Detections evaluated at first eligible alert $t_{\mathrm{first}, i} = \min \{ t \in \mathcal{T}_i : \operatorname{Elig}_i(t; \tau, \mathcal{B}) = 1 \}$ where $\operatorname{Elig}_i(t; \tau, \mathcal{B}) = \mathbf{1}\{i \in \mathcal{S}_{\text{alerted}}(K) \;\land\; s_i(t) \ge \tau\}$.*

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

*Note: In this observed sample, horizon-correct detections and detections satisfying the two-hour pre-suspicion timing buffer ($t_{\mathrm{first}, i} + 2\text{h} \le T_{\mathrm{susp}, i}$) were identical across all models and operating points. Detections occurred before the selected retrospective suspicion anchor ($T_{\mathrm{susp}}$), not necessarily before earliest antimicrobial prescribing ($T_{\mathrm{Rx,start}}$). Alert episodes were evaluated across the full canonical surveillance stream (4,070.8 patient-days, 97,700 windows); reference anchors $C_i$ were used retrospectively to adjudicate advance-warning timeliness, not to filter observation volume prior to alert generation.*

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

| Cohort & Timestamp Source | Sepsis Onset Category | Subgroup Denominator ($N$) | Surveillance Covered Stays ($V=1$) | Definitely Eligible ($O^{\mathrm{def}}=1$) | Point-Assigned ($O^{\mathrm{point}}=1$) | Indeterminate ($O^{\mathrm{indet}}$) | Possibly Eligible ($O^{\mathrm{poss}}$) | Within-Subgroup Opportunity Bounds | Within-Covered Opportunity Bounds | Contribution to Overall Cohort Bound |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MetaVision Prospective-Replay** | Early Incident (<=24h) | 47 | 47 (100.0%) | 47 | 47 | 0 | 47 | [100.0%, 100.0%] | [100.0%, 100.0%] | [47, 47] / 408 (11.5%) |
| *(Primary Benchmark: Suspicion Anchor $T_{\mathrm{susp}}$)* | Intermediate Incident (24–48h) | 156 | 104 (66.7%) | 103 | 103 | 1 | 104 | [66.0%, 66.7%] | [99.0%, 100.0%] | [103, 104] / 408 (25.2–25.5%) |
| | Late Incident (>48h) | 205 | 0 (0.0%) | 0 | 0 | 0 | 0 | [0.0%, 0.0%] | N/A (No covered stays) | 0 / 408 (0.0%) |
| | **All Incident Sepsis-3 Cases** | **408** | **151 (37.0%)** | **150** | **150** | **1** | **151** | **[36.8%, 37.0%]** | **[99.3%, 100.0%]** | **[150, 151] / 408 (36.8–37.0%)** |
| **MetaVision Sensitivity: Earliest Rx** | Early Incident (<=24h) | 47 | 47 (100.0%) | 0 | 0 | 38 | 38 | [0.0%, 80.9%] | [0.0%, 80.9%] | [0, 38] / 408 (0.0–9.3%) |
| *(Pre-Exposure Sensitivity: Earliest Rx $T_{\mathrm{Rx,start}}$)* | Intermediate Incident (24–48h) | 156 | 104 (66.7%) | 0 | 0 | 9 | 9 | [0.0%, 5.8%] | [0.0%, 8.7%] | [0, 9] / 408 (0.0–2.2%) |
| | Late Incident (>48h) | 205 | 0 (0.0%) | 0 | 0 | 0 | 0 | [0.0%, 0.0%] | N/A (No covered stays) | 0 / 408 (0.0%) |
| | **All Incident Sepsis-3 Cases** | **408** | **151 (37.0%)** | **0** | **0** | **47** | **47** | **[0.0%, 11.5%]** | **[0.0%, 31.1%]** | **[0, 47] / 408 (0.0–11.5%)** |
| **MIMIC-III Broad Incident Sepsis** | Early Incident (<=24h) | 2,641 | 102 (3.9%) | 0 | 0 | 74 | 74 | [0.00%, 2.80%] | [0.0%, 72.5%] | [0, 74] / 3,581 ([0.00%, 2.07%]) |
| *(Date-Level Prescription Proxy: $T_{\mathrm{Rx,start}}$)* | Intermediate Incident (24–48h) | 361 | 235 (65.1%) | 1 | 1 | 36 | 37 | [0.28%, 10.25%] | [0.43%, 15.74%] | [1, 37] / 3,581 ([0.03%, 1.03%]) |
| | Late Incident (>48h) | 579 | 0 (0.0%) | 0 | 0 | 0 | 0 | [0.00%, 0.00%] | N/A (No covered stays) | [0, 0] / 3,581 ([0.00%, 0.00%]) |
| | **All Incident Sepsis-3 Cases** | **3,581** | **337 (9.4%)** | **1** | **1** | **110** | **111** | **[0.03%, 3.10%]** | **[0.30%, 32.94%]** | **[1, 111] / 3,581 ([0.03%, 3.10%])** |

---

## Appendix 1: Complete TRIPOD+AI Reporting Checklist Crosswalk

*Structured according to the official 27-item TRIPOD+AI Statement (Collins et al., BMJ 2024;385:e078378), with specific operational sepsis surveillance extensions detailed in the final column.*

| Item # | Official TRIPOD+AI Topic | Standard Recommendation | Manuscript & Supplement Section | Operational Sepsis Early-Warning Extension |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Title** | Identify the study as developing, validating, or evaluating a prediction model, target population, and outcome. | Title Page (Main text & Supplement) | Explicitly specifies operational constraints: *Surveillance Coverage, Treatment Precedence, and Workload Constraints*. |
| **2** | **Abstract** | Provide a structured summary of objectives, study design, setting, participants, sample size, predictors, outcome, statistical analysis, results, and conclusions. | Abstract (Main text, 235 words, 5 headings) | Reports episode PPV, reciprocal NNE workload, and population opportunity cascade ($S \to V \to O \to D$). |
| **3a** | **Background** | Explain clinical context, rationale for early-warning evaluation, and existing model limitations. | Section 1 (Introduction) | Identifies the gap between retrospective AUROC and deployable bedside alert burden / treatment precedence. |
| **3b** | **Objectives** | Specify primary research questions, hypotheses, and intended clinical application. | Section 1 (Introduction) | Fourfold operational evaluation objectives formally itemized. |
| **4a** | **Source of data** | Describe study design, healthcare setting, electronic health record databases, and data collection dates. | Section 2.1 & Section S1.1 | Retrospective replay across versioned BIDMC repository: MIMIC-III v1.4 (CareVue development and MetaVision evaluation). |
| **4b** | **Key dates** | Specify dates of cohort entry, surveillance intervals, and follow-up duration. | Section 2.1, Section S1.1, S1.2 | Evaluates second-day ICU surveillance (hours 19–30); development in CareVue era (2001–2008), testing in MetaVision (2008–2012). |
| **5a** | **Eligibility criteria** | Detail inclusion and exclusion criteria for study participants. | Section 2.1, Section S1.1, Table S1 | Adult first ICU admissions ($\ge 18$ years), total ICU $\text{LOS} \ge 30\text{h}$; prevalent sepsis on arrival ($T_{\text{S3}} \le 19\text{h}$) excluded. |
| **5b** | **Treatment received** | Describe clinical treatments administered and their relation to prediction windows. | Section 2.3, Section S1.5, Table S4 | Systemic antimicrobials tracked via hospital-wide prescription orders (`prescriptions`); pre-treatment risk sets $\mathcal{R}_0$ and $\mathcal{R}_2$ censor post-treatment observations. |
| **6a** | **Outcome definition** | Unambiguously define target outcome, diagnostic criteria, and reference standards. | Section 2.2, Section S1.5 | Consensus Sepsis-3 operational adaptation: acute $\Delta\text{SOFA} \ge 2$ over $[T_{\text{susp}}-24\text{h}, T_{\text{susp}}+12\text{h}]$ relative to suspected infection. |
| **6b** | **Outcome timing** | Specify exact timing and operational clock assigned to outcome onset. | Section 2.2, Section S1.5, Figure 1 | Earliest hour of SOFA increase $T_{\text{S3}}$; contrasted with infection suspicion anchor $T_{\mathrm{susp}}$ (operational search-window lead $G = T_{\mathrm{susp}} - T_{\text{S3}} = +24.2\text{h}$ median; $T_{\text{S3}} - T_{\mathrm{susp}} = -24.2\text{h}$). |
| **7a** | **Predictors** | Define all candidate predictors, clinical categories, units, and physiological validity filters. | Section S1.3 (Feature Dictionary Table) | 21 variables (6 vital signs, 2 oxygenation, 13 laboratory biomarkers) mapped across CareVue and MetaVision itemids. |
| **7b** | **Predictor timing** | Define observation lookback windows, measurement latency, and blinding to outcome. | Section 2.1, Section S1.3, S1.4 | Causal right-open $[t-20\text{h}, t)$ window; 1.0-hour laboratory turnaround buffer; maximum forward-fills: 4h vitals, 24h labs. |
| **8** | **Sample size** | Report sample size, number of events, and effective degrees of freedom. | Section 2.1, Table 1, Table S1 | 8,254 test stays, 97,700 windows, 408 consensus Sepsis-3 stays, 678 horizon-positive observations ($\pi = 0.694\%$). |
| **9** | **Missing data** | Detail missing data proportions, forward-filling rules, and explicit missingness indicators. | Section S1.3, Section S1.4 | Dynamic forward-fill limits; 21 explicit observation mask channels incorporated into 42-channel tensor and 126-summary vector. |
| **10a** | **Model development** | Describe model architectures, mathematical loss formulations, and optimization algorithms. | Section 2.5, Section S1.4 | BiMamba SSM (selective state space), Mask-Aware Transformers, GBDTs (XGBoost, LightGBM), Random Forest, L2-Logistic Regression, Stacking Ensemble. |
| **10b** | **Model specification** | Specify detailed model architectures, layer dimensions, and serialized input formats. | Section S1.4 (Per-Model Table) | BiMamba SSM: $(N, 20, 42)$; Transformers: $(N, 20, 42)$ and $(N, 20, 21)$; Tabular: $(N, 126)$; Stacking: $(N, 7)$. |
| **10c** | **Validation method** | Specify validation design, resampling protocols, and performance metrics. | Section 2.5, Section S1.6, Table S2 | Patient-clustered paired bootstrap ($B=1,000$); AUROC, AUPRC, episode PPV, NNE, and timely detection yield. |
| **10d** | **Calibration** | Describe post-processing probability calibration procedures and distinguish methods. | Section 2.5, Section S1.4 | Disambiguates 1-parameter temperature scaling $\sigma(z/T)$ vs. 2-parameter Platt scaling $\sigma(az+b)$ fitted on validation split. |
| **11** | **Decision thresholds** | Specify operational threshold selection policies and alert budgets. | Section 2.4, Section S1.6, Table 3, S3A | Prespecified alert budgets (5, 10, 20 episodes / 100 pt-days) under 12-hour refractory suppression rule. |
| **12** | **Cohort differences** | Compare development and evaluation data distributions and EHR source system eras. | Section S1.2, Table 4, Table S4 | CareVue (2001–2008) development vs. MetaVision (2008–2012) test within BIDMC. |
| **13a** | **Participant flow** | Provide detailed cohort derivation flow diagram and stage-wise accounting. | Figure 1, Figure 3, Table 1, Table S1 | Stage-wise population cascade: $S=408 \to V=151 \to O=150 \to D=6–15$ detected patients. |
| **13b** | **Demographics** | Summarize patient baseline characteristics and surveillance volume. | Section 2.1, Table 1 | 4,070.8 surveillance patient-days across 8,254 unique adult ICU stays. |
| **14** | **Model parameters** | Report final hyperparameters, state dimensions, and ensemble weights. | Section S1.4 | BiMamba: $d_{\text{state}}=16, d_{\text{model}}=64, E=2, lr=5\times 10^{-4}$; Stacking: L2 logistic meta-learner. |
| **15** | **Discrimination** | Report discrimination metrics with 95% confidence intervals across risk sets. | Section 3.2, Table 2, Table S2 | BiMamba SSM AUROC: 0.6326 (canonical) vs. 0.6433 ($\mathcal{R}_0$) vs. 0.6447 ($\mathcal{R}_2$); paired $\Delta\text{AUROC} = +0.0107$. |
| **16** | **Clinical utility** | Evaluate operational trade-offs, alert burden, and Number Needed to Evaluate (NNE). | Section 3.3, Table 3, Table S3A | NNE ranges from 34.0 to 116.3 across architectures; episode PPV ranges from 0.86% to 2.94%. |
| **17** | **Limitations** | Discuss study limitations, EHR proxy constraints, and retrospective boundaries. | Section 4 (Discussion) | Single-center origin, lack of clinician diagnostic rationale, and retrospective alert simulation under fixed surveillance hours. |
| **18** | **Interpretation** | Provide non-causal operational interpretation grounded in empirical findings. | Section 4 (Discussion) | Emphasizes that pre-treatment opportunity remains high ($99.3\%$) in covered stays, but alert budgets bound bedside sensitivity. |
| **19** | **Generalizability** | Discuss model transportability across EHR eras and clinical settings. | Section 4 (Discussion) | Cross-system stability from CareVue to MetaVision under standardized feature harmonization. |
| **20** | **Information accessibility** | Provide full access to detailed protocols, data schemas, and supplementary tables. | Supplementary Material (S1–S4) | Complete 4-section supplementary package with executable code references and boundary tie audits. |
| **21** | **Funding** | Disclose all sources of funding and institutional support. | Declarations | No external grant or commercial funding received. |
| **22** | **Conflicts of interest** | Disclose financial and non-financial competing interests. | Declarations | Authors declare no competing interests. |
| **23** | **Ethics approval** | Document Institutional Review Board approval and data use agreements. | Declarations | Approved by BIDMC and MIT IRBs under PhysioNet Credentialed Data Use Agreement; informed consent waived. |
| **24** | **Data sharing** | Detail public repository access, credentialing, and database versioning. | Section 2.1, Declarations | PhysioNet credentialed access: MIMIC-III (v1.4). |
| **25** | **Code availability** | Provide repository links, frozen release manifests, and container environments. | Declarations, Appendix 2 | GitHub: `alireza/Sepsis-Onset-Timing-Paradox`, Git commit `b0932ca`, release manifest `1.0.0-freeze`. |
| **26** | **AI disclosure** | Disclose the use of artificial intelligence and large language models. | Declarations | Pre-submission peer-review auditing and editorial polishing via Perplexity AI and Anthropic/Google AI interfaces. |
| **27** | **Author contributions** | Detail individual author roles, responsibilities, and guarantor status. | Title Page & Declarations | AS: study design, pipeline execution, statistical evaluation; SA: conceptualization, methodology supervision, critical review. |

---

## Appendix 2: Computing Environment & Reproducibility Statement

- **Operating System:** Linux (x86_64 architecture, Ubuntu/Debian compatible kernel)
- **Python Version:** `3.10.19` (Conda environment)
- **Deep Learning Frameworks:** PyTorch `2.1.2+cu121`, Mamba-ssm `1.2.0`, TensorFlow `2.14.0`
- **Machine Learning & Statistical Libraries:** LightGBM `3.3.5`, XGBoost `1.7.6`, Scikit-learn `1.2.2`, SciPy `1.10.1`, Pandas `1.5.3`, NumPy `1.26.4`
- **Visualization:** Matplotlib `3.7.1` / `3.10.1`, Seaborn `0.12.2`
- **Analytical Freeze Commit:** `b0932ca` (Release Manifest `1.0.0-freeze`, archived at `reports/manifest/release_manifest.json`)
- **Evaluation Pipeline & Bootstrap Release Commit:** `1292103`
- **Manuscript Reporting & Packaging Commit:** `4038fa2`
- **Automated Invariance Verification Artifacts:** `reports/manifest/time_truncation_invariance_audit.json` and `time_truncation_invariance_audit.md` (PASS: no feature discrepancies detected across 50 stays, 12 decision hours, 547 unique patient-decision pairs, and 1,641 equality comparisons).
- **Code & Model Availability:** Complete analysis scripts, preprocessing pipelines, and evaluation harnesses are archived at GitHub: `https://github.com/alireza/Sepsis-Onset-Timing-Paradox`.
