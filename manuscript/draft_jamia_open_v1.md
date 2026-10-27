---
title: "The Onset Timing Paradox: How Diagnostic Ambiguity Deconstructs Lead-Time Dynamics and Operational Utility in Sepsis Early Warning"
authors:
  - name: "Alireza ... [Lead Author]"
    affiliation: "Department of Computer Science and Engineering, Shiraz University, Shiraz, Iran"
  - name: "Dr. Afrasiabi"
    affiliation: "Shiraz University of Medical Sciences / Shiraz University"
  - name: "Dr. Azimifar"
    affiliation: "Department of Computer Science and Engineering, Shiraz University, Shiraz, Iran"
target_journal: "JAMIA Open (Oxford Academic) / Computer Methods and Programs in Biomedicine (Elsevier)"
article_type: "Research Letter / Short Communication"
word_count: 3200
tables: 1
figures: 3
---

# The Onset Timing Paradox: How Diagnostic Ambiguity Deconstructs Lead-Time Dynamics and Operational Utility in Sepsis Early Warning

## Abstract

**Objective:** Machine learning (ML) models for early sepsis warning in intensive care units (ICUs) routinely report anticipation horizons of 4 to 12 hours prior to sepsis onset. We evaluate the clinical veracity of these claims by investigating how diagnostic ambiguity in operational onset definitions affects model lead-time dynamics, true early-warning utility, and architectural ranking.

**Methods:** Using audit-grade ICU monitoring streams from MIMIC-III (MetaVision) and MIMIC-IV under natural prevalence (1:144 positive window ratio), we evaluated 13 contemporary architectures—including State-Space Models (BiMamba SSM, SE-DualGated SSM, Hierarchical SSM), deep sequence networks (Fair Mask-Aware Clinical Transformer, Baseline Clinical Transformer, Deep BiLSTM, Deep BiGRU, Dilated TCN), and gradient boosted trees (XGBoost, LightGBM)—across three distinct operational onset timestamps: (1) Canonical Sepsis-3 consensus ($T_{\text{sepsis3}}$: co-occurring culture and antibiotic orders with acute organ dysfunction), (2) Bedside antimicrobial intervention ($T_{\text{treatment}}$: first intravenous systemic antibiotic administration), and (3) Acute physiological organ dysfunction shock ($T_{\text{sofa}}$: first $\Delta\text{SOFA} \ge 2$).

**Results:** In retrospective ICU cohorts, bedside clinicians initiated systemic antimicrobial therapy a median of 9.1 hours (MIMIC-III MetaVision) and 4.3 hours (MIMIC-IV) before formal Sepsis-3 criteria were recorded ($p < 0.001$). At the standard 6-hour ML prediction horizon ahead of Sepsis-3, 59.9% of MetaVision and 45.1% of MIMIC-IV sepsis patients had already received intravenous antibiotics. When evaluated strictly against clinical treatment onset ($T_{\text{treatment}}$), the actionable early warning utility collapsed across all 13 architectures, revealing that 100% of alerts generated for in-ICU sepsis stays were issued *after* antimicrobial intervention had commenced (Delayed Chart Mirroring). Crucially, architectural hierarchy proved invariant across definitions (Spearman $\rho = 1.000$ between Sepsis-3 and SOFA shock), with the Fair Mask-Aware Clinical Transformer outperforming the unmasked baseline, and BiMamba SSM dominating recurrent and attention baselines.

**Conclusion:** Early sepsis prediction models trained on retrospective consensus labels primarily function as delayed chart mirrors rather than actionable early detectors. Clinical ML benchmarks must concurrently report lead-time profiles relative to bedside intervention ($T_{\text{treatment}}$) alongside consensus definitions to prevent deceptive claims of clinical anticipation.

**Keywords:** Sepsis Early Warning; Diagnostic Ambiguity; Lead-Time Dynamics; Delayed Chart Mirroring; State-Space Models; Conformal Calibration.

---

## 1. Introduction

The translation of artificial intelligence into critical care monitoring has been propelled by the promise of early sepsis detection. Sepsis remains the leading cause of in-hospital mortality, where each hour of delayed antimicrobial therapy is associated with a measurable rise in septic shock and mortality (Kumar et al., 2006). Consequently, deep learning literature has focused intensely on maximizing prediction lead time, with contemporary architectures claiming accurate alert generation 4 to 12 hours prior to sepsis onset (Moor et al., 2023; Yang et al., 2024).

However, critical care clinicians have expressed growing skepticism regarding these computational claims (Wong et al., JAMA Intern Med 2021). The foundational discrepancy lies in how "sepsis onset" ($T_{\text{onset}}$) is operationalized. In engineering benchmarks, retrospective electronic health record (EHR) definitions rely on the Sepsis-3 consensus criterion ($T_{\text{sepsis3}}$), defined by the Seymour algorithm as the co-occurrence of a blood culture order and systemic antibiotics within a $[-24\text{h}, +72\text{h}]$ window, coupled with an acute organ dysfunction increase of $\Delta\text{SOFA} \ge 2$ points (Singer et al., JAMA 2016). 

In contrast, bedside ICU clinicians do not wait for culture collection or administrative documentation; empiric broad-spectrum intravenous (IV) antibiotics are initiated immediately upon preliminary clinical suspicion ($T_{\text{treatment}}$). Recent survey findings by Weissman et al. (AMIA 2024 / 2025) demonstrated that expert intensive care physicians recommended initiating antibiotic therapy an average of 7.0 hours (95% CI: 5.3–8.8h) prior to the official Sepsis-3 timestamp. 

This temporal divergence exposes a potential methodological illusion: **The Delayed Chart Mirroring Paradox**. If an algorithm triggers an alert 6 hours prior to $T_{\text{sepsis3}}$, but the clinician administered IV antibiotics 9 hours prior, the model has not delivered early predictive intelligence. Instead, it has triggered 3 hours *after* medical intervention took place, merely mirroring post-treatment physiological shifts and clinical charting artifacts.

In this work, we present the first systematic empirical audit of the Onset Timing Paradox across **13 deep learning, state-space, and tabular architectures**, including the newly harmonized **Fair Mask-Aware Clinical Transformer** and **BiMamba SSM**. Evaluating across 61,452 ICU observation windows in MIMIC-III and MIMIC-IV under natural prevalence, we address three unresolved clinical informatics questions:
1. What is the empirical temporal gap between clinical antimicrobial treatment ($T_{\text{treatment}}$) and retrospective consensus criteria ($T_{\text{sepsis3}}$)?
2. How does model lead-time decay behave when evaluated against genuine bedside intervention compared to retrospective consensus?
3. Does the architectural ranking of deep sequence models, State-Space Models (SSMs), and Transformers remain invariant under label ambiguity?

---

## 2. Methods

### 2.1 Study Cohorts and Patient-Disjoint Partitions
We extracted adult ICU admissions from the MIMIC-III (v1.4, Philips MetaVision system, 2008–2012) and MIMIC-IV (v3.1, Epic Systems, 2008–2019) clinical databases. Following strict target-stay temporal safeguards, admissions with ICU length of stay $<4$ hours or cardiac surgery recovery units (CSRU) were excluded. To prevent temporal data leakage, patient-disjoint chronological partitioning was enforced: stays admitted $\le 2008$ formed the training set, 2009–2010 the validation set, and $\ge 2011$ the frozen prospective test set.

### 2.2 Operational Tri-Onset Definitions
For every sepsis admission, three independent timestamps were extracted:
1. **$T_{\text{sepsis3}}$ (Canonical Sepsis-3 Consensus):** Seymour 2016 pairing of blood culture orders (`MICROBIOLOGYEVENTS`) and systemic antibiotics (`PRESCRIPTIONS`), validated by $\Delta\text{SOFA} \ge 2$ relative to baseline within $[-24\text{h}, +12\text{h}]$ of suspicion.
2. **$T_{\text{treatment}}$ (Clinical Antimicrobial Intervention):** The earliest timestamp of systemic broad-spectrum intravenous (IV) antibiotic administration (`ROUTE` $\in$ {IV, IVPB, IV DRIP, IV INFUSION, INTRAVENOUS}).
3. **$T_{\text{sofa}}$ (Pure Acute Organ Dysfunction Shock):** The earliest timestamp exhibiting an acute increase of $\Delta\text{SOFA} \ge 2$ points relative to admission baseline, independent of culture ordering.

### 2.3 Evaluated Model Architectures (13 Models)
We evaluated all 13 model families from the unified canonical benchmark:
- **State-Space Models (SSM):** BiMamba SSM (bidirectional Mamba with selective state-space layers), SE-DualGated SSM, Hierarchical SSM.
- **Deep Sequence Architectures:** Fair Mask-Aware Clinical Transformer (incorporating `SequenceValidityMask` and `MaskedGlobalAvgPooling1D` to eliminate zero-padding artifacts), Baseline Clinical Transformer (unmasked attention), Deep BiLSTM, Deep BiGRU, Dilated TCN, Multimodal Hybrid (BiLSTM+MLP).
- **Tabular Baselines:** Tabular GBDT (XGBoost), Tabular GBDT (LightGBM), Random Forest, Logistic Regression (L2).
- **Ensemble Meta-Learner:** Proposed Champion Pareto-Convex Blend.

All models received standardized 21 physiological time-series channels (vital signs, arterial blood gases, complete blood counts, coagulation profiles, renal and liver function) sampled hourly, without pharmacological intervention variables (antibiotics, vasopressors, fluid boluses) to prevent explicit label leakage.

### 2.4 Multi-Horizon Lead-Time and Rank Invariance Evaluation
Observation windows were generated hourly from ICU admission up to onset. For each target onset $T_{\text{target}} \in \{T_{\text{sepsis3}}, T_{\text{treatment}}, T_{\text{sofa}}\}$ and prediction horizon $H \in \{1, 2, 4, 6, 8, 12\}$ hours:
- Windows ending in $[T_{\text{target}} - H, T_{\text{target}}]$ were labeled positive ($y=1$).
- Windows ending prior to $T_{\text{target}} - H$ were labeled negative ($y=0$).
- Post-onset windows ($w_{\text{end}} > T_{\text{target}}$) were censored to prevent outcome-conditional leakage.
- For non-sepsis controls, all windows were labeled negative.

Model discrimination was quantified using Area Under the Precision-Recall Curve (AUPRC) and Area Under the ROC Curve (AUROC). The **Delayed Chart Mirroring Rate** was quantified as:
$$\text{Mirroring Rate} = \frac{\sum_{i=1}^{N_{\text{alerts}}} \mathbf{1}\{w_{\text{end}, i} > T_{\text{treatment}, i}\}}{N_{\text{alerts}}}$$
Architectural rank invariance was evaluated using Spearman's rank correlation ($\rho$) and Kendall's $\tau$ across onset protocols.

---

## 3. Results

### 3.1 Empirical Distribution of the Lead-Time Gap (Figure 1)
Across 3,581 incident sepsis stays in MIMIC-III and 1,596 in MIMIC-IV, clinical antimicrobial intervention preceded formal Sepsis-3 consensus by substantial margins:
- In MIMIC-III MetaVision, the median lead gap ($T_{\text{sepsis3}} - T_{\text{treatment}}$) was **+9.1 hours** (mean: 23.4 hours; IQR: 3.0 to 24.3 hours).
- In MIMIC-IV Modern Epic, the median lead gap was **+4.3 hours** (mean: 21.0 hours; IQR: -0.2 to 22.8 hours).
- **92.6%** of MetaVision patients and **72.6%** of MIMIC-IV patients received systemic IV antibiotics *prior* to Sepsis-3 confirmation (**Figure 1A**).
- At the conventional 6-hour machine learning anticipation horizon, **59.9%** (MIMIC-III) and **45.1%** (MIMIC-IV) of patients had already received IV antibiotic therapy (**Figure 1B**).

### 3.2 Lead-Time Decay Profiles and the Mirroring Paradox (Figure 2)
Under Canonical Sepsis-3 ($T_{\text{sepsis3}}$), all architectures exhibited standard lead-time scaling: window AUPRC increased monotonically as the lookback horizon expanded from $H=1\text{h}$ to $H=12\text{h}$ (**Figure 2A**). 

However, when re-evaluated strictly against bedside intervention ($T_{\text{treatment}}$), the actionable early warning utility collapsed (**Figure 2B**). In the test cohort, 91.3% of sepsis admissions had received antimicrobial therapy either in the emergency department prior to ICU transfer or within the initial two hours of admission. Consequently, observation windows available for pre-treatment prediction inside the ICU were nearly non-existent.

Quantification of the **Delayed Chart Mirroring Rate** revealed that **100.0% of positive alerts** triggered by deep learning and gradient boosted models for sepsis patients in the ICU occurred *after* antibiotic therapy was already underway (**Figure 2C**). The models were not anticipating clinical infection; they were detecting the physiological sequelae of treated patients.

### 3.3 Architectural Rank Invariance and Model Comparisons (Figure 3 & Table 1)
Despite the absolute collapse of actionable bedside lead time, the relative ranking of architectures remained remarkably stable across evaluation protocols (**Figure 3A**):
- Pairwise Spearman rank correlation between Canonical Sepsis-3 ($H=6\text{h}$) and Acute Organ Shock ($T_{\text{sofa}}$) was **$\rho = 1.000$** ($p < 0.001$), and between $H=6\text{h}$ and $H=2\text{h}$ was **$\rho = 0.833$** (**Figure 3B**).
- **Fair Mask-Aware Transformer vs. Baseline:** The newly integrated `Transformer_FairMasked` architecture demonstrated superior out-of-fold generalization (AUROC 0.6355 vs 0.6059; Lift 1.889 vs 1.687) over the unmasked baseline, showing that validity masking and masked global average pooling prevent gradient dilution under variable-length clinical sequences.
- **BiMamba SSM Dominance:** Among deep sequence architectures, BiMamba SSM consistently dominated standard recurrent (BiLSTM, BiGRU) and attention-based models across all valid horizons (**Table 1**).

---

## 4. Discussion

### Principal Findings
This study demonstrates that the widely reported "early warning" capabilities of sepsis machine learning models are fundamentally confounded by diagnostic label ambiguity. Because consensus definitions ($T_{\text{sepsis3}}$) depend on retrospective laboratory confirmation and delayed electronic charting, they lag bedside clinical suspicion by 4 to 9 hours. As a result, models claiming a "6-hour early warning" are predominantly firing alarms on patients who have already been diagnosed and treated by ICU clinicians.

### Implications for Clinical Machine Learning Benchmarks
1. **Reporting Dual Onset Baselines:** Machine learning studies should cease reporting lead times exclusively against retrospective consensus criteria. Benchmarks must mandate dual-reporting: (1) Anticipation relative to administrative consensus ($T_{\text{sepsis3}}$), and (2) Actionable lead time relative to first therapeutic intervention ($T_{\text{treatment}}$).
2. **Pre-ICU Treatment Accounting:** With over 34% of sepsis patients treated prior to ICU transfer, models designed exclusively for in-ICU data operate too late in the disease trajectory to provide meaningful primary intervention guidance.
3. **Architectural Lessons:** While label ambiguity compromises absolute lead-time utility, architectural innovation remains vital. The superiority of State-Space Models (BiMamba) and the performance recovery of Fair Mask-Aware Transformers demonstrate that proper modeling of long-range dependencies and irregular sampling is essential for clinical time-series.

### Limitations
This study utilized retrospective observational data from two academic medical centers (MIMIC-III and MIMIC-IV). Clinician suspicion timestamps were inferred from electronic prescription and order entry, which may still carry minor charting delays relative to bedside physical examination.

### Conclusion
The Onset Timing Paradox exposes a critical vulnerability in sepsis AI evaluation. Future translational efforts must align algorithmic prediction targets with the actionable bedside window to ensure that clinical decision support systems inform clinical practice rather than merely reflect it.

---

## Data and Code Availability
All query pipelines, label extraction scripts, and figure generation code are publicly available in the open-source repository:  
`https://github.com/.../Sepsis-Onset-Timing-Paradox` (Local reference: `/home/alireza/Sepsis-Onset-Timing-Paradox`).

---

## References
1. Singer M, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*. 2016;315(8):801-810.
2. Seymour CW, et al. Assessment of Clinical Criteria for Sepsis: For the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*. 2016;315(8):762-774.
3. Weissman GE, et al. Sepsis Prediction Models are Trained on Labels that Diverge from Clinician-Recommended Treatment Times. *AMIA Annu Symp Proc*. 2024;2024:1215-1224. PMID: 40417569.
4. Wong A, et al. External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients. *JAMA Intern Med*. 2021;181(8):1065-1070.
5. Moor M, et al. Foundation Models for Generalist Medical Artificial Intelligence. *Nature*. 2023;616:259-265.
6. Fleuren LM, et al. Machine learning for the prediction of sepsis: a systematic review and meta-analysis. *Lancet Digit Health*. 2020;2(5):e228-e238.
