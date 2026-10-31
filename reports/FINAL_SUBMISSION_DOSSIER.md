# Final Submission Dossier & Audit Clearance

**Manuscript Title:** Surveillance Coverage, Clinical-Process Anchors, and Workload Constraints in Machine Learning Early Warning for Sepsis: An Operational Evaluation Audit  
**Target Journal:** *JAMIA Open* (Oxford University Press / AMIA)  
**Track:** Research and Applications  
**Authors:** Alireza Shayan, BSc$^{1}$; Somayeh Afrasiabi, PhD$^{1,*}$  
*$^{1}$Department of Computer Science and Engineering & IT, School of Electrical and Computer Engineering, Shiraz University, Shiraz, Iran*  
**Corresponding Author:** Somayeh Afrasiabi, PhD (`afrasiabi@shirazu.ac.ir`, Tel: +98-71-3613-3500)  
**Release Target:** Git Tag `v1.0.1-freeze` (Commit `5679b3e`)  
**Audit Decision:** **`[READY FOR SUBMISSION / GREEN LIGHT]`** (Round 8 Independent Audit Clearance)  

---

## 1. Executive Summary & Audit Clearance

Following an eight-round iterative pre-submission editorial and methodological audit, this manuscript and its accompanying release deliverables have achieved **full independent verification and unconditional clearance for submission (`GREEN LIGHT`)**. 

All previously identified methodological, statistical, bibliographic, typographical, and packaging defects have been definitively resolved:
1. **Mathematical Attainable Set Bounds (§3.4):** Piecewise formulation of $\mathcal{C}_i$, closed upper bound $[a_i, c_i]$ when $a_i \le c_i < b_i$, and logical conjunctions for definite/possible opportunity indicators are mathematically locked.
2. **Dynamic Bootstrap Denominators (§2.5 & Table 3):** Ground-truth source code inspection of `scripts/metrics/compute_alert_budget_and_refractory.py` confirmed that patient-clustered paired bootstrapping dynamically recalculates $N_{V,b} = \sum_{i \in b} V_i$ and $N_{S,b} = \sum_{i \in b} S_i$ per resample. Errant fixed-denominator draft claims were eliminated.
3. **Bibliographic Identity & Order:** All 26 references appear in strictly sequential numerical order (1 through 26) upon first in-text appearance. Ref [21] correctly cites Goodwin AJ, et al. (*Front Digit Health* 2022;4:932599, DOI: `10.3389/fdgth.2022.932599`, PMID: `36060541`).
4. **Table Typography & Readability:** Double-spaced review manuscript features dedicated landscape orientations for Tables 2–4 with legible font sizes ($\ge 8.17\text{ pt}$) and no global shrinking (`\resizebox` eliminated).
5. **Word Count Compliance:** Main text narrative body contains **3,853 words** in both camera-ready (`main.pdf`) and review (`main_review_doublespaced.pdf`) layouts, strictly complying with the *JAMIA Open* limit of $\le 4,000$ words.
6. **Substantive Prose Synchronization:** Lay summary and scientific qualifications (such as protocol coverage, retrospective baseline minimum, and simulated alert evaluations) are identical across both PDF formats.

---

## 2. Independent Audit Verification Ledger

| Verification Dimension | Evaluated Requirement | Observed Value / Status | Audit Status |
| :--- | :--- | :--- | :--- |
| **Camera-Ready Geometry** | Strictly 8 pages (two-column camera-ready) | Exactly 8 pages (`main.pdf`) | **PASS** |
| **Review Manuscript Geometry** | Double-spaced review layout | Exactly 23 pages (`main_review_doublespaced.pdf`) | **PASS** |
| **Camera-Ready SHA-256** | Cryptographic hash ledger match | `fe35d4da0a692953f139a3d6eff7eaf4db09dc05ee9755d77493eb5d2ffba41c` | **PASS** |
| **Review Manuscript SHA-256** | Cryptographic hash ledger match | `0978da15efc572920e645f7123fffa4c95b35f59012335394a0d508f1e21ef43` | **PASS** |
| **Word Count (Main Body)** | $\le 4,000$ words (*JAMIA Open* limit) | **3,853 text words** (71 header words) via `texcount` | **PASS** |
| **Structured Abstract** | $\le 250$ words (Objectives, Methods, Results, Disc, Conc) | **228 words** | **PASS** |
| **Lay Summary** | $\le 200$ words (Patient-oriented, non-technical) | **166 words** (synchronized verbatim across PDFs) | **PASS** |
| **Citation Sequence** | Strictly sequential first-appearance (1..26) | Verified monotonically increasing 1 to 26; 0 skips | **PASS** |
| **Reference 21 Metadata** | Accurate author, title, journal, DOI, PMID | Goodwin AJ et al., *Front Digit Health* (PMID: 36060541) | **PASS** |
| **Table Legibility (Review)** | Rendered font size $\ge 7.8\text{--}8.5\text{ pt}$ | Table 1: 7.97 pt; Table 2: 8.47 pt; Tables 3–4: 8.17 pt | **PASS** |
| **Table Orientation** | Wide multi-column tables on landscape pages | Table 1 portrait; Tables 2, 3, 4 dedicated landscape | **PASS** |
| **Bootstrap Denominators** | Dynamic cluster sampling documented | Documented in §2.5 and Table 3 notes; code-verified | **PASS** |

---

## 3. Cryptographic Release Ledger & File Manifest

The official frozen release directory is located at:  
`/home/alireza/Downloads/Sepsis_Freeze_v1.0.1/`

### SHA-256 Checksums (`SHA256SUMS`)
```
bd5e8d7d1f51f91a5b65fdbccb108cb90042affed0a5441f19c43886197df0f4  ./draft_jamia_open_v1.docx
fe35d4da0a692953f139a3d6eff7eaf4db09dc05ee9755d77493eb5d2ffba41c  ./main.pdf
0978da15efc572920e645f7123fffa4c95b35f59012335394a0d508f1e21ef43  ./main_review_doublespaced.pdf
93a573b2f80fa049caa7f3107b950ca360ace77846520752813d690c434fcf80  ./ONE_PAGE_CLOCK_TO_RESULT_MAPPING.md
764b3639bb6e06acb406f93123e5c8d697e3db7910805ff66a60f7f62f9557bc  ./supplementary_material.docx
7dcde126b36dc0ac963d149b222936100a6263a6a54824396d3ea635bf2ec936  ./Table1_Cohort_Progression_Ledger.csv
```

### Git Repository Provenance
* **Repository:** `https://github.com/alireza/Sepsis-Onset-Timing-Paradox` (MIT License)
* **Release Tag:** `v1.0.1-freeze`
* **Target Commit:** `5679b3e`
* **Analytical Provenance Breakdown:**
  * Model architectures & frozen weight tensors: Commit `b0932ca`
  * Execution & evaluation pipelines: Commit `1292103`
  * Denominator accounting & Table 1 ledger: Commit `311cea3`
  * Initial submission packaging: Commit `8f10435`
  * Final audit clearance & synchronized manuscript release: Commit `5679b3e`

---

## 4. Step-by-Step Submission Instructions for Authors

### A. Primary Submission Portal
* **Journal:** *JAMIA Open* (Oxford University Press)
* **Submission URL:** `https://mc.manuscriptcentral.com/jamiaopen`
* **Manuscript Type:** Research and Applications

### B. File Upload Designations in ScholarOne
1. **Main Document (Principal Review File):**  
   Upload `main_review_doublespaced.pdf`.  
   *Note:* *JAMIA Open* author instructions specifically mandate a double-spaced review manuscript with embedded/dedicated tables.
2. **Companion Camera-Ready Preview:**  
   Upload `main.pdf` with file designation: *"Supplementary File for Review"* or *"File Not for Review (Camera-Ready Layout Preview)"*.
3. **Supplementary Material:**  
   Upload `supplementary_material.docx` with file designation: *"Supplementary Data / Online Appendix"*.
4. **Editable Manuscript Draft (Optional / Backup):**  
   Upload `draft_jamia_open_v1.docx` with file designation: *"Supplementary File"*.
5. **Figures (Individual High-Resolution Files):**  
   Upload the vector PDF files from the `figures/` directory:
   * Figure 1: `Figure1_Clocks_and_Opportunity_Waterfall.pdf`
   * Figure 2: `Figure2_Operational_Alert_Budget_Frontier.pdf`
   * Figure 3: `Figure3_Three_Stage_Decomposition.pdf`
6. **Data Ledger:**  
   Upload `Table1_Cohort_Progression_Ledger.csv` with file designation: *"Supplementary Data"*.

---

## 5. Ready-to-Submit Cover Letter

Copy and paste the text below into the Cover Letter text box in ScholarOne:

```text
To: 
The Editorial Office
JAMIA Open (Oxford University Press / American Medical Informatics Association)

RE: Submission of Original Research Article — "Surveillance Coverage, Clinical-Process Anchors, and Workload Constraints in Machine Learning Early Warning for Sepsis: An Operational Evaluation Audit"

Dear Editor-in-Chief and Editorial Board,

We are pleased to submit our original research manuscript entitled "Surveillance Coverage, Clinical-Process Anchors, and Workload Constraints in Machine Learning Early Warning for Sepsis: An Operational Evaluation Audit" for consideration for publication as a Research and Applications article in JAMIA Open.

Study Motivation & Key Findings:
Over the past decade, dozens of retrospective investigations have reported impressive discriminative performance (AUROC 0.80–0.88) for machine learning early warning systems predicting sepsis onset hours before consensus criteria are met. However, bedside clinical trials and prospective implementations have yielded conflicting results regarding true pre-intervention lead time, alarm burden, and actionable warning windows.

In this study, we conducted a rigorous operational evaluation audit across 8,254 first adult ICU admissions (97,700 hourly evaluation windows) in the MIMIC-III MetaVision cohort, benchmarking eight diverse predictive architectures (including selective state-space models [BiMamba], masked clinical transformers, and gradient-boosted trees) under strict bedside alert workload budgets. 

To reconcile divergent findings in the informatics literature, we formalized a four-stage operational cascade (S -> V -> O -> D) that disentangles incident target sepsis (S), surveillance protocol coverage (V), clinical-process anchor eligibility (O), and emitted-alert detection under workload constraints (D). Our principal findings include:
1. Retrospective Lookback Artifacts: We mathematically prove that high pre-suspicion opportunity (99.3% of surveillance-covered stays) is largely an algebraic consequence of the discrete 24-hour SOFA lookback rule in first-bin assigned cases (66.2% of covered stays), which structurally guarantees pre-suspicion eligibility rather than providing independent biological early warning.
2. Timestamp Coarsening: When evaluating pre-intervention opportunity relative to earliest systemic antimicrobial prescription dates, calendar-day coarsening under declared midnight assignment eliminates all definite horizon-positive observations (bounds: [0.0%, 31.1%] in covered stays; [0.0%, 11.5%] cohort-wide), rendering prospective discrimination mathematically unestimable.
3. Bedside Workload Constraints: Under a realistic bedside workload ceiling of 10 alert episodes per 100 patient-days with a 12-hour refractory suppression policy, top-performing continuous models (BiMamba SSM and LightGBM) achieved an episode PPV of 1.72% (NNE = 58.1 simulated alerts per true case) and detected only 7 of 408 incident sepsis stays (1.72% population detection yield).

Journal Compliance & Declarations:
- Track: Research and Applications.
- Word Count (Main Narrative Text Body): 3,853 words (strictly complies with the <= 4,000 word ceiling).
- Structured Abstract: 228 words (strictly complies with the <= 250 word limit).
- Lay Summary: 166 words (strictly complies with the <= 200 word limit).
- Tables and Figures: 4 tables, 3 figures.
- Exclusivity: This manuscript represents original work, has not been published previously, and is not currently under consideration for publication elsewhere.
- Ethics & Consent: The study was performed using de-identified data from MIMIC-III under PhysioNet Credentialed Data Use Agreement. Review and waiver of informed consent were granted by the IRBs of BIDMC and MIT.
- Reproducibility & Open Science: All preprocessing pipelines, predictive model architectures, frozen weights, and bootstrap evaluation scripts are open-sourced under the MIT License at https://github.com/alireza/Sepsis-Onset-Timing-Paradox (Release Tag v1.0.1-freeze).

AI Assistance Disclosure:
In accordance with JAMIA Open author guidelines and COPE recommendations, large language model interfaces (Perplexity AI and Anthropic/Google AI) were utilized for pre-submission peer-review auditing, LaTeX typography formatting, and editorial consistency cross-checking. The authors independently designed and executed all analytical pipelines, empirical experiments, and statistical models, and take full responsibility for the integrity, validity, and conclusions of the work.

Thank you for considering our work for publication in JAMIA Open. We look forward to receiving the reviewers' comments.

Sincerely,

Somayeh Afrasiabi, PhD (Corresponding Author)
Assistant Professor, Department of Computer Science and Engineering & IT
School of Electrical and Computer Engineering, Shiraz University
Molla Sadra St., Shiraz 71348-51154, Fars, Iran
Telephone: +98-71-3613-3500 | Email: afrasiabi@shirazu.ac.ir

Alireza Shayan, BSc (First Author)
Department of Computer Science and Engineering & IT
School of Electrical and Computer Engineering, Shiraz University, Shiraz, Iran
Email: Ali404@hafez.shirazu.ac.ir
```
