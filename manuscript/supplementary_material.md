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
The study cohorts were derived from the Beth Israel Deaconess Medical Center (BIDMC) critical care repositories: MIMIC-IV (v2.2, primary evaluation cohort) and MIMIC-III (v1.4, developmental and date-proxy sensitivity cohorts) under PhysioNet Credentialed Data Use Agreement.

1. **Eligibility Criteria:** Adult patients ($\ge 18$ years of age at ICU admission) admitted to intensive care units with a total ICU length of stay $\text{LOS} \ge 30\text{ hours}$ (calculated as $T_{\text{out}} - T_{\text{in}} \ge 30\text{h}$).
2. **First Admission Restriction:** To prevent within-patient correlation across multiple hospitalizations, only the chronological first ICU stay per unique patient was included.
3. **Prevalent Sepsis Exclusion:** Patients satisfying consensus Sepsis-3 criteria at or prior to ICU admission ($T_{\text{S3}} \le 0\text{h}$) or developing onset prior to or at the start of the surveillance protocol ($T_{\text{S3}} \le 19\text{h}$) were excluded from the incident-surveillance evaluation. These episodes occurred before the evaluated surveillance interval and were excluded from the incident-surveillance evaluation; acquisition setting and prior clinical recognition were not inferred from onset timing alone.
4. **Surveillance Protocol Interval:** Predictions were evaluated sequentially across 12 scheduled hourly decision bins $[t-1, t]$ for $t \in \{19, 20, \dots, 30\}$ hours post-admission.
5. **Observation-Availability Censoring vs. Administrative ICU Exit Rules:**
   - Because cohort eligibility strictly conditioned on total ICU length of stay $\text{LOS} \ge 30\text{h}$, all 8,254 cohort patients remained administratively in the ICU through at least hour 30. No patient was administratively discharged from the ICU prior to hour 30.
   - Right-censoring during the surveillance interval ($t \in [19, 30]$) occurred through exactly two distinct mechanisms:
     - *Consensus Sepsis-3 Onset ($N=110$ stays, 623 windows):* In accordance with early-warning principles, surveillance ceased immediately once acute Sepsis-3 organ dysfunction was confirmed ($t \ge T_{\text{S3}}$). Subsequent windows were not evaluated.
     - *Observation-Availability Exclusions ($N=180$ control stays, 725 windows):* Under the prespecified feature-generation rule, 725 scheduled decisions in 180 control stays were unavailable because the required charted observations were absent. These were observation-availability exclusions, not administrative ICU discharge. Each hourly decision bin was evaluated independently; once charting ceased in these 180 stays, no subsequent measurements were recorded during the surveillance interval, and scoring did not resume prior to hour 30.

### S1.2 Dataset Partitioning, Subject-Level Allocation Ledger, and Chronological Provenance
To reduce information leakage across operational EHR transitions, partitioning was executed strictly at the unique patient level (`SUBJECT_ID` / `ICUSTAY_ID`):

#### 1. MIMIC-III Developmental Split (D3 CareVue-to-MetaVision Era Partition):
The base developmental cohort comprises 27,448 unique adult first ICU admissions in MIMIC-III (`artifacts/phase5/d3/d3_cohort_summary.csv`). Each subject contributes exactly one stay (`unique_subjects == icu_stays == 27,448`).

| Partition / Progression Stage | Unique Subjects ($N$) | ICU Stays ($N$) | Incident Sepsis ($N$) | Controls ($N$) | Prevalence | Allocation Policy & Accounting Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **CareVue Training Set** | 10,371 | 10,371 | 1,350 | 9,021 | 13.02% | 80.0% of CareVue candidates; patient-disjoint (`seed=42`, stratified by incident sepsis) |
| **CareVue Validation / Tuning Set** | 2,593 | 2,593 | 337 | 2,256 | 13.00% | 20.0% of CareVue candidates; patient-disjoint (`seed=42`, stratified by incident sepsis) |
| *CareVue Candidate Subtotal* | *12,964* | *12,964* | *1,687* | *11,277* | *13.01%* | *100.0% of eligible CareVue prediction candidates (CareVue bedside era, 2001–2008)* |
| **MetaVision Held-Out Era Test** | 9,732 | 9,732 | 1,886 | 7,846 | 19.38% | 100.0% of MetaVision prediction candidates (successor EHR era, 2008–2012) |
| **Excluded Transition Era** | 44 | 44 | 8 | 36 | 18.18% | Stays with `DBSOURCE == 'both'` during 2008 system transition; excluded for era ambiguity |
| **Prevalent Sepsis on Arrival Excluded** | 3,384 | 3,384 | 3,384 | 0 | 100.0% | Onset at or before admission ($T_{\text{S3}} \le 0\text{h}$); excluded from incident early warning |
| **Post-ICU Onset Excluded** | 1,324 | 1,324 | 0 | 1,324 | 0.0% | Infection/dysfunction onset after ICU discharge; excluded from in-ICU evaluation |
| **Total MIMIC-III Base Cohort** | **27,448** | **27,448** | **6,965** | **20,483** | **15.75%** | **All unique first adult ICU admissions meeting base criteria** |

*Cross-Partition Independence Audit:* Across all partitions, pairwise subject overlap is strictly zero (`train_val_overlap = 0`, `train_test_overlap = 0`, `val_test_overlap = 0`, verified in `artifacts/phase5/d3/d3_overlap_audit.csv`). There was zero cross-partition reassignment, subject addition, or data leakage between CareVue development and MetaVision test partitions.

#### 2. MIMIC-IV Same-Institution Cross-Release Evaluation with Unresolved Cross-Release Patient Overlap:
- 8,254 adult first ICU stays (408 incident Sepsis-3 cases, 7,846 controls) evaluated across 97,700 surveillance windows.
- All model weights, calibration parameters, and preprocessing scalers were strictly frozen prior to evaluating this partition.
- *Cross-Release Independence Note:* MIMIC-III and MIMIC-IV share a source institution (BIDMC) and overlapping collection periods (MIMIC-III: 2001–2012; MIMIC-IV: 2008–2019). Because patient identifiers were re-randomized between MIMIC-III and MIMIC-IV releases, cross-release patient overlap is not fully ascertainable without restricted-access linkage; this evaluation is therefore characterized as a same-institution cross-release temporal evaluation rather than a fully independent multi-center external trial.

#### 3. Relation to Broad MIMIC-III Sensitivity Cohort ($N=3,581$):
- The broad MIMIC-III sensitivity cohort (3,581 incident cases) includes all adult first ICU admissions in MIMIC-III meeting the $\text{LOS} \ge 30\text{h}$ criterion that developed consensus Sepsis-3 at any point during hospitalization. It was utilized in Section 3.4 and Table S4 to compute population opportunity bounds under prescription-date uncertainty across the entire historical MIMIC-III repository.

### S1.3 Feature Dictionary, Availability Rules, and Preprocessing Protocol
Surveillance features comprise 21 continuous clinical time series grouped into three clinically grounded categories:
1. **Hemodynamics & Vital Signs (6 variables):** Heart Rate (bpm), Systolic Blood Pressure (mmHg), Diastolic Blood Pressure (mmHg), Mean Arterial Pressure (MAP, mmHg), Respiratory Rate (breaths/min), Temperature (°C).
2. **Oxygenation & Respiratory Support (2 variables):** Peripheral Oxygen Saturation ($\text{SpO}_2$, %), Fraction of Inspired Oxygen ($\text{FiO}_2$, ratio/percent).
3. **Laboratory Biomarkers (13 variables):** White Blood Cell Count (WBC, $\times 10^3/\mu\text{L}$), Platelet Count ($\times 10^3/\mu\text{L}$), Hemoglobin (g/dL), Hematocrit (%), Serum Creatinine (mg/dL), Blood Urea Nitrogen (BUN, mg/dL), Serum Glucose (mg/dL), Serum Potassium (mEq/L), Serum Lactate (mmol/L), Base Excess (mEq/L), Arterial pH, Partial Thromboplastin Time (PTT, sec), Aspartate Aminotransferase (AST, U/L).

#### Feature ItemID Cross-Reference and Physiological Validity Bounds

| Clinical Variable | Category | MIMIC-III ItemIDs (CareVue / MetaVision) | MIMIC-IV ItemIDs (`chartevents` / `labevents`) | Plausible Range | Hourly Aggregation |
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
- **Handling of First Prediction Bin ($t=19\text{h}$):** For decision bin $t=19$, the lookback interval is $[19-20, 19) = [-1, 19)$. The earliest hourly bin (bin 0, corresponding to $[-1, 0)$ hours relative to ICU admission) represents the immediate pre-admission hour. If emergency department or pre-ICU observations were recorded in this interval, they were incorporated directly into bin 0; if unobserved, bin 0 was imputed using training medians and flagged with binary observation indicator `obs = 0` (and `in_icu_mask_h0 = 0`), ensuring zero leakage and consistent tensor dimensions.
- Vital signs and oxygenation variables were carried forward for a maximum duration of 4 hours.
- Laboratory measurements were carried forward for a maximum duration of 24 hours within the 20-hour window. Data prior to the lookback boundary $[t-20\text{h}]$ were not retrieved.
- Unobserved segments prior to the initial measurement were imputed using training-set medians.
- *Measurement vs. Result Availability:* In MIMIC-IV, charted timestamps (`charttime`) were validated against specimen storage/result verification timestamps (`storetime`). In streaming simulation, a 1.0-hour laboratory turnaround buffer was enforced to ensure results were clinically posted and available before score calculation.

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
   - *Executable Verification Audit:* Formal automated testing across 1,548 patient-decision pairs in `reports/manifest/time_truncation_invariance_audit.json` confirmed that removing future data yields an exact maximum feature discrepancy of `0.000000` for both sequence tensors and tabular summaries.
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

### S1.5 Executable Sepsis-3 Phenotype Implementation (Operational Adaptation of Seymour et al., JAMA 2016)
The Sepsis-3 clinical consensus phenotype was implemented programmatically as a study-specific operational adaptation of Seymour et al. (JAMA 2016), tailored for ICU surveillance and electronic health record constraints:

1. **Four Design Adaptations from Seymour et al. (2016):**
   - *Culture Site Specificity:* While Seymour et al. included body fluid cultures from multiple anatomical sites (urine, sputum, cerebrospinal fluid, wounds), this study restricted qualifying cultures strictly to blood culture specimen collections (`microbiologyevents` where `spec_type_desc` indicates blood culture), reducing ambiguity from asymptomatic bacteriuria or superficial colonization.
   - *Antimicrobial Route Restriction:* Only new systemic antimicrobials (intravenous, intramuscular, or enteral routes with documented systemic bioavailability) were included; topical ointments, ophthalmic solutions, and catheter flushes were excluded. Both $T_{\text{treatment}}$ (treatment-precedence clock) and $T_{\text{abx}}$ (suspected infection clock) enforce this systemic route definition.
   - *Prescription vs. Confirmed Administration Records:* Seymour et al. evaluated hospital-wide cohorts using pharmacy dispensing/order timestamps. Here, MIMIC-IV leverages bedside electronic Medication Administration Records (`emar`), tracking exact administration times rather than order initiation.
   - *Evaluation Windows:* Organ dysfunction was evaluated across $[T_{\text{susp}} - 24\text{h}, T_{\text{susp}} + 12\text{h}]$ relative to suspected infection, with baseline SOFA defined over $[T_{\text{susp}} - 24\text{h}, T_{\text{susp}}]$.

2. **Dataset-Specific Source Tables, Fields, and Medication-Event Status Rules:**
   - **MIMIC-IV (v2.2):**
     - *Antimicrobials:* Extracted from `emar` and `emar_detail`, linked to `pharmacy`. Event verification required confirmed administration status recorded in `emar.event_txt` (e.g., `'Administered'`, `'Started'`, `'Restarted'`, `'Applied'`); orders flagged as `'Not Given'`, `'Held'`, or `'Cancelled'` were strictly excluded. Intravenous boluses and continuous infusions were additionally cross-verified against `inputevents`.
     - *Cultures:* Extracted from `microbiologyevents` using `chartdate` and `charttime`, filtering on blood culture specimen descriptions.
   - **MIMIC-III (v1.4):**
     - *Antimicrobials:* Bedside infusions and administrations extracted from `inputevents_cv` (CareVue) and `inputevents_mv` (MetaVision), capturing exact charted infusion start times. Provider orders extracted from `prescriptions.startdate` and `enddate`.
     - *Cultures:* Extracted from `microbiologyevents.charttime` (or `chartdate` when time was unrecorded).
3. **Directional Suspicion Pairing Rules ($T_{\text{susp}}$):**
   - *Antimicrobial precedes Blood Culture:* Blood culture must follow within $\le 24\text{ hours}$ ($0 \le t_{\text{culture}} - t_{\text{abx}} \le 24\text{h}$); $T_{\text{susp}} = t_{\text{abx}}$.
   - *Blood Culture precedes Antimicrobial:* Antimicrobial administration must follow within $\le 72\text{ hours}$ ($0 < t_{\text{abx}} - t_{\text{culture}} \le 72\text{h}$); $T_{\text{susp}} = t_{\text{culture}}$.
   - *Multiple Candidate Pairs & Complete Episode Selection:* All candidate culture–antibiotic pairs within the hospital admission (`HADM_ID`) were evaluated. If an earlier suspected-infection pair did not produce acute organ dysfunction ($\Delta\text{SOFA} \ge 2$), subsequent candidate pairs within the admission were evaluated. The earliest qualifying pair that satisfies $\Delta\text{SOFA} \ge 2$ within $[T_{\text{susp}} - 24\text{h}, T_{\text{susp}} + 12\text{h}]$ confirms consensus Sepsis-3, and $T_{\text{S3}}$ is established at the earliest hour of confirmed organ dysfunction.
4. **Baseline SOFA ($\text{SOFA}_{\text{base}}$):**
   - Evaluated strictly over the 24-hour interval preceding suspicion: $[T_{\text{susp}} - 24\text{h}, T_{\text{susp}}]$.
   - $\text{SOFA}_{\text{base}} = \min\{\text{SOFA}(t) : t \in [T_{\text{susp}} - 24\text{h}, T_{\text{susp}}]\}$.
   - If no physiological or laboratory data were recorded in this interval, baseline SOFA was imputed as 0 (normal organ function prior to acute deterioration, adhering to Seymour et al. Table S2).
5. **Organ-Dysfunction Evaluation Window & Sepsis-3 Onset ($T_{\text{S3}}$):**
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
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MIMIC-IV Prospective-Replay** | Early Incident (<=24h) | 47 | 47 (100.0%) | 47 | 0 | 0 | 0 | [100.0%, 100.0%] | [100.0%, 100.0%] | 47 / 408 (11.5%) |
| *(Hourly Confirmed eMAR)* | Intermediate Incident (24–48h) | 156 | 104 (66.7%) | 103 | 0 | 52 | 1 | [66.0%, 66.0%] | [99.0%, 99.0%] | 103 / 408 (25.2%) |
| | Late Incident (>48h) | 205 | 0 (0.0%) | 0 | 0 | 205 | 0 | [0.0%, 0.0%] | N/A (No covered stays) | 0 / 408 (0.0%) |
| | **All Incident Sepsis-3 Cases** | **408** | **151 (37.0%)** | **150** | **0** | **257** | **1** | **[36.8%, 36.8%]** | **[99.3%, 99.3%]** | **150 / 408 (36.8%)** |
| **MIMIC-III Broad Incident Sepsis** | Early Incident (<=24h) | 2,641 | 102 (3.9%) | 0 | 74 | 2,539 | 28 | [0.0%, 2.8%] | [0.0%, 72.5%] | [0, 74] / 3,581 ([0.00%, 2.07%]) |
| *(Date-Level Prescription Proxy)* | Intermediate Incident (24–48h) | 361 | 235 (65.1%) | 1 | 36 | 126 | 198 | [0.3%, 10.2%] | [0.4%, 15.7%] | [1, 37] / 3,581 ([0.03%, 1.03%]) |
| | Late Incident (>48h) | 579 | 0 (0.0%) | 0 | 0 | 579 | 0 | [0.0%, 0.0%] | N/A (No covered stays) | [0, 0] / 3,581 ([0.00%, 0.00%]) |
| | **All Incident Sepsis-3 Cases** | **3,581** | **337 (9.4%)** | **1** | **110** | **3,244** | **226** | **[0.03%, 3.10%]** | **[0.30%, 32.94%]** | **[1, 111] / 3,581 ([0.03%, 3.10%])** |

---

## Appendix 1: Complete TRIPOD+AI Reporting Checklist Crosswalk

*Structured according to the official 27-item TRIPOD+AI Statement (Collins et al., BMJ 2024;385:e078378), with specific operational sepsis surveillance extensions detailed in the final column.*

| Item # | Official TRIPOD+AI Topic | Standard Recommendation | Manuscript & Supplement Section | Operational Sepsis Early-Warning Extension |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Title** | Identify the study as developing, validating, or evaluating a prediction model, target population, and outcome. | Title Page (Main text & Supplement) | Explicitly specifies operational constraints: *Surveillance Coverage, Treatment Precedence, and Workload Constraints*. |
| **2** | **Abstract** | Provide a structured summary of objectives, study design, setting, participants, sample size, predictors, outcome, statistical analysis, results, and conclusions. | Abstract (Main text, 235 words, 5 headings) | Reports episode PPV, reciprocal NNE workload, and population opportunity cascade ($S \to V \to O \to D$). |
| **3a** | **Background** | Explain clinical context, rationale for early-warning evaluation, and existing model limitations. | Section 1 (Introduction) | Identifies the gap between retrospective AUROC and deployable bedside alert burden / treatment precedence. |
| **3b** | **Objectives** | Specify primary research questions, hypotheses, and intended clinical application. | Section 1 (Introduction) | Fourfold operational evaluation objectives formally itemized. |
| **4a** | **Source of data** | Describe study design, healthcare setting, electronic health record databases, and data collection dates. | Section 2.1 & Section S1.1 | Retrospective replay across versioned BIDMC repositories: MIMIC-IV v2.2 and MIMIC-III v1.4. |
| **4b** | **Key dates** | Specify dates of cohort entry, surveillance intervals, and follow-up duration. | Section 2.1, Section S1.1, S1.2 | Evaluates second-day ICU surveillance (hours 19–30); development in CareVue era (2001–2008), testing in MetaVision (2008–2012) and MIMIC-IV (2008–2019). |
| **5a** | **Eligibility criteria** | Detail inclusion and exclusion criteria for study participants. | Section 2.1, Section S1.1, Table S1 | Adult first ICU admissions ($\ge 18$ years), total ICU $\text{LOS} \ge 30\text{h}$; prevalent sepsis on arrival ($T_{\text{S3}} \le 19\text{h}$) excluded. |
| **5b** | **Treatment received** | Describe clinical treatments administered and their relation to prediction windows. | Section 2.3, Section S1.5, Table S4 | Systemic antimicrobials tracked via confirmed eMAR; pre-treatment risk sets $\mathcal{R}_0$ and $\mathcal{R}_2$ censor post-treatment observations. |
| **6a** | **Outcome definition** | Unambiguously define target outcome, diagnostic criteria, and reference standards. | Section 2.2, Section S1.5 | Consensus Sepsis-3 operational adaptation: acute $\Delta\text{SOFA} \ge 2$ over $[T_{\text{susp}}-24\text{h}, T_{\text{susp}}+12\text{h}]$ relative to suspected infection. |
| **6b** | **Outcome timing** | Specify exact timing and operational clock assigned to outcome onset. | Section 2.2, Section S1.5, Figure 1 | Earliest hour of SOFA increase $T_{\text{S3}}$; contrasted with treatment time $T_{\text{treatment}}$ (diagnostic lead gap $G = +24.2\text{h}$). |
| **7a** | **Predictors** | Define all candidate predictors, clinical categories, units, and physiological validity filters. | Section S1.3 (Feature Dictionary Table) | 21 variables (6 vital signs, 2 oxygenation, 13 laboratory biomarkers) mapped across CareVue, MetaVision, and MIMIC-IV itemids. |
| **7b** | **Predictor timing** | Define observation lookback windows, measurement latency, and blinding to outcome. | Section 2.1, Section S1.3, S1.4 | Causal right-open $[t-20\text{h}, t)$ window; 1.0-hour laboratory turnaround buffer; maximum forward-fills: 4h vitals, 24h labs. |
| **8** | **Sample size** | Report sample size, number of events, and effective degrees of freedom. | Section 2.1, Table 1, Table S1 | 8,254 test stays, 97,700 windows, 408 consensus Sepsis-3 stays, 678 horizon-positive observations ($\pi = 0.694\%$). |
| **9** | **Missing data** | Detail missing data proportions, forward-filling rules, and explicit missingness indicators. | Section S1.3, Section S1.4 | Dynamic forward-fill limits; 21 explicit observation mask channels incorporated into 42-channel tensor and 126-summary vector. |
| **10a** | **Model development** | Describe model architectures, mathematical loss formulations, and optimization algorithms. | Section 2.5, Section S1.4 | BiMamba SSM (selective state space), Mask-Aware Transformers, GBDTs (XGBoost, LightGBM), Random Forest, L2-Logistic Regression, Stacking Ensemble. |
| **10b** | **Model specification** | Specify detailed model architectures, layer dimensions, and serialized input formats. | Section S1.4 (Per-Model Table) | BiMamba SSM: $(N, 20, 42)$; Transformers: $(N, 20, 42)$ and $(N, 20, 21)$; Tabular: $(N, 126)$; Stacking: $(N, 7)$. |
| **10c** | **Validation method** | Specify validation design, resampling protocols, and performance metrics. | Section 2.5, Section S1.6, Table S2 | Patient-clustered paired bootstrap ($B=1,000$); AUROC, AUPRC, episode PPV, NNE, and timely detection yield. |
| **10d** | **Calibration** | Describe post-processing probability calibration procedures and distinguish methods. | Section 2.5, Section S1.4 | Disambiguates 1-parameter temperature scaling $\sigma(z/T)$ vs. 2-parameter Platt scaling $\sigma(az+b)$ fitted on validation split. |
| **11** | **Decision thresholds** | Specify operational threshold selection policies and alert budgets. | Section 2.4, Section S1.6, Table 3, S3A | Prespecified alert budgets (5, 10, 20 episodes / 100 pt-days) under 12-hour refractory suppression rule. |
| **12** | **Cohort differences** | Compare development and evaluation data distributions and EHR source system eras. | Section S1.2, Table 4, Table S4 | CareVue (2001–2008) development vs. MetaVision (2008–2012) test vs. MIMIC-IV (2008–2019) same-institution cross-release evaluation. |
| **13a** | **Participant flow** | Provide detailed cohort derivation flow diagram and stage-wise accounting. | Figure 1, Figure 3, Table 1, Table S1 | Stage-wise population cascade: $S=408 \to V=151 \to O=150 \to D=6–15$ detected patients. |
| **13b** | **Demographics** | Summarize patient baseline characteristics and surveillance volume. | Section 2.1, Table 1 | 4,070.8 surveillance patient-days across 8,254 unique adult ICU stays. |
| **14** | **Model parameters** | Report final hyperparameters, state dimensions, and ensemble weights. | Section S1.4 | BiMamba: $d_{\text{state}}=16, d_{\text{model}}=64, E=2, lr=5\times 10^{-4}$; Stacking: L2 logistic meta-learner. |
| **15** | **Discrimination** | Report discrimination metrics with 95% confidence intervals across risk sets. | Section 3.2, Table 2, Table S2 | BiMamba SSM AUROC: 0.6326 (canonical) vs. 0.6433 ($\mathcal{R}_0$) vs. 0.6447 ($\mathcal{R}_2$); paired $\Delta\text{AUROC} = +0.0107$. |
| **16** | **Clinical utility** | Evaluate operational trade-offs, alert burden, and Number Needed to Evaluate (NNE). | Section 3.3, Table 3, Table S3A | NNE ranges from 34.0 to 116.3 across architectures; episode PPV ranges from 0.86% to 2.94%. |
| **17** | **Limitations** | Discuss study limitations, EHR proxy constraints, and retrospective boundaries. | Section 4 (Discussion) | Single-center origin, eMAR rollout period (2014–2016), lack of clinician diagnostic rationale, and retrospective alert simulation. |
| **18** | **Interpretation** | Provide non-causal operational interpretation grounded in empirical findings. | Section 4 (Discussion) | Emphasizes that pre-treatment opportunity remains high ($99.3\%$) in covered stays, but alert budgets bound bedside sensitivity. |
| **19** | **Generalizability** | Discuss model transportability across EHR eras and clinical settings. | Section 4 (Discussion) | Cross-system stability from CareVue to MetaVision and MIMIC-IV under standardized feature harmonization. |
| **20** | **Information accessibility** | Provide full access to detailed protocols, data schemas, and supplementary tables. | Supplementary Material (S1–S4) | Complete 4-section supplementary package with executable code references and boundary tie audits. |
| **21** | **Funding** | Disclose all sources of funding and institutional support. | Declarations | No external grant or commercial funding received. |
| **22** | **Conflicts of interest** | Disclose financial and non-financial competing interests. | Declarations | Authors declare no competing interests. |
| **23** | **Ethics approval** | Document Institutional Review Board approval and data use agreements. | Declarations | Approved by BIDMC and MIT IRBs under PhysioNet Credentialed Data Use Agreement; informed consent waived. |
| **24** | **Data sharing** | Detail public repository access, credentialing, and database versioning. | Section 2.1, Declarations | PhysioNet credentialed access: MIMIC-IV (v2.2) and MIMIC-III (v1.4). |
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
- **Submission Packaging Commits:** `1292103` and `38aac8b` (repository HEAD)
- **Automated Invariance Verification Artifacts:** `reports/manifest/time_truncation_invariance_audit.json` and `time_truncation_invariance_audit.md` (Zero discrepancy across 1,548 patient-decision pairs).
- **Code & Model Availability:** Complete analysis scripts, preprocessing pipelines, and evaluation harnesses are archived at GitHub: `https://github.com/alireza/Sepsis-Onset-Timing-Paradox`.
