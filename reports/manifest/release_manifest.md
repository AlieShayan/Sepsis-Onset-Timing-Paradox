# Analytical Release Manifest and Reproducibility Checksums

- **Manifest Version:** `1.0.0-freeze`
- **Timestamp:** `2026-10-28T19:14:07.891328+00:00`
- **Git Commit:** `73e7241`
- **Environment:** Python 3.10.19 | NumPy 1.26.4 | Pandas 1.5.3

## 1. Tie-Aware First-Alert Predicate Verification
- **Status:** PASSED: Full equivalence established between boundary selection and first-alert timestamps.
- **Target K=814 Selected:** 814
- **True Sepsis Detections:** 10
- **Pretreatment Verified:** 100.0%
- **2-Hour Lead Time Verified:** 100.0%

## 2. Boundary Ties Audit (Full IEEE 754 Float64 Precision)
| Prespecified Budget | Model | Full-Precision Cutoff Score | Strictly Above | Tied at Boundary | Selected from Tied |
| :--- | :--- | :--- | :---: | :---: | :---: |
| 5.0 episodes / 100 pt-days | Multi-Model Stacking Ensemble | `0.014490108370127998` | 203 | 2 | 1 |
| 20.0 episodes / 100 pt-days | Tabular GBDT (LightGBM) | `0.010536972856183402` | 623 | 2636 | 191 |
| 20.0 episodes / 100 pt-days | Tabular GBDT (XGBoost) | `0.010525625922113648` | 812 | 5 | 2 |
| 5.0 episodes / 100 pt-days | Random Forest | `0.017658340327200716` | 202 | 2 | 2 |

## 3. Cryptographic File Checksums (SHA256)
| Relative File Path | SHA256 Checksum | Size (Bytes) |
| :--- | :--- | :---: |
| `reports/predictions/canonical_test_predictions_harmonized.parquet` | `c436a1bcea5e387dd2dc142f093120704a292abe5b0b5d05b56b21e4df0580d4` | 7,785,009 |
| `reports/tables/Table1_Cohort_Progression_Ledger.csv` | `2619eeb455550153cd3f18274f4a859a0881ceeea5a8385e5a64c65b93f4310e` | 2,857 |
| `reports/tables/Table2_Pretreatment_Sepsis_Benchmark_CI.csv` | `4c2b765a09a33eff7ba955ae1852df1ae54ec38adb11465d5dce00b04dc44701` | 2,058 |
| `reports/tables/Table3_Alert_Budget_Refractory_Workload.csv` | `20fa591c271622e67a87982159e9f17a7f6c49f30d77c782abf84bab21270a7f` | 4,337 |
| `reports/tables/Table4_Two_Axis_Subgroup_Uncertainty_Matrix.csv` | `4b23f7b72752b58e1311bb5f91d22584fba1dea457db73a52e1f26014c8d3eaa` | 1,609 |
| `reports/figures/Figure1_Clocks_and_Opportunity_Waterfall.pdf` | `941e11e3ceb5919804cf2c922eef69944ee04177dff779a50ab1d76e76d46739` | 60,550 |
| `reports/figures/Figure1_Clocks_and_Opportunity_Waterfall.png` | `d95e5d22950afb4c2023cc8ef4b8d6f2beeb90ed4680d6e5be1e7d9c856ccd88` | 415,123 |
| `reports/figures/Figure2_Operational_Alert_Budget_Frontier.pdf` | `e980412af01c9aacea4615494fd7e7c83323076c5b1fb2b1f1f12665126bb1a0` | 32,902 |
| `reports/figures/Figure2_Operational_Alert_Budget_Frontier.png` | `3f653fd164789be8b6d756a514a7931b4a6d70aec3393d0c2896f97907c35cb7` | 437,004 |
| `reports/figures/Figure3_Three_Stage_Decomposition.pdf` | `3beff49a387d8c34819a5dae56eb873eaf3100559a99ca97643882db015b7c5a` | 31,723 |
| `reports/figures/Figure3_Three_Stage_Decomposition.png` | `8c1b97694d802b19cb6add1ed0aa4ff675e6dd56a0b1f1988088ea2ef5dd4665` | 298,458 |
| `scripts/metrics/generate_table1_cohort_progression_ledger.py` | `c81df43adc118c0747c72b68729b11f2df3dc21a977c49e7b5b60243a292c776` | 9,514 |
| `scripts/metrics/compute_pretreatment_sepsis_benchmark.py` | `24ec8045cc095b6dc12c4201a2606b529d1f03720f107db24ee849534f8e83ff` | 7,590 |
| `scripts/metrics/compute_alert_budget_and_refractory.py` | `f30e310a9938e542c8569e89c21188de71e2540f21a07937fa85b293884fa97f` | 9,812 |
| `scripts/metrics/compute_interval_censored_bounds.py` | `b1729e530f9b308d9b7098ef1189f7de64d2400094e40642d6058baf762925f8` | 8,619 |
| `scripts/figures/generate_jamia_figures.py` | `be6af2844f0754704c422c6de6cc83ce7077d2525fbd473d52e310f7b0d9300f` | 11,745 |
| `manuscript/draft_jamia_open_v1.md` | `24000f8931c2dbf78c24255fed1c1f279d77cb2939dfc8a32ec5b7292ecf175c` | 52,138 |
