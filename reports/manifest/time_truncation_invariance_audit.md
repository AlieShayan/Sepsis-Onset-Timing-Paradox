# Time-Truncation Invariance & Model Input Shape Audit

- **Audit Status:** `PASS: no feature discrepancies detected in the tested prediction-time truncation comparisons.`
- **Analytical Freeze Commit:** `b0932ca`
- **Manifest Version:** `1.0.0-freeze`
- **Timestamp (UTC):** `2026-10-29T10:42:50.224173+00:00`
- **Unique Patient ICU Stays Tested:** 50
- **Unique Patient-Decision Pairs Evaluated:** 547
- **Total Representation Comparisons:** 1641 (across sequence tensors and tabular summaries)
- **Numerical Tolerance:** Exact float equality (discrepancy threshold: 0.00000000)
- **Maximum Sequence Tensor Discrepancy:** `0.00000000`
- **Maximum Tabular Summary Discrepancy:** `0.00000000`

## Verified Per-Model Serialized Input Shapes

| Model Architecture | Input Shape | Channel Specification |
| :--- | :---: | :--- |
| BiMamba_SSM | `[None, 20, 42]` | Causal right-open $[t-20\mathrm{h}, t)$ |
| Mask_Aware_Transformer | `[None, 20, 42]` | Causal right-open $[t-20\mathrm{h}, t)$ |
| Baseline_Unmasked_Transformer | `[None, 20, 21]` | Causal right-open $[t-20\mathrm{h}, t)$ |
| Tabular_GBDT_LightGBM | `[None, 126]` | Causal right-open $[t-20\mathrm{h}, t)$ |
| Tabular_GBDT_XGBoost | `[None, 126]` | Causal right-open $[t-20\mathrm{h}, t)$ |
| Random_Forest | `[None, 126]` | Causal right-open $[t-20\mathrm{h}, t)$ |
| Logistic_Regression_L2 | `[None, 126]` | Causal right-open $[t-20\mathrm{h}, t)$ |
| Multi_Model_Stacking_Ensemble | `[None, 7]` | Causal right-open $[t-20\mathrm{h}, t)$ |

## Time-Truncation Invariance Conclusion

Features extracted at decision time $t$ remained unchanged in the reported tests when future data timestamped $\ge t$ were removed. No future observations timestamped $\ge t$ enter any model input tensor. Baseline-offset leakage channels are strictly absent.
