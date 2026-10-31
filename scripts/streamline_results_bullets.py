#!/usr/bin/env python3
"""
streamline_results_bullets.py
Streamlines the bullet list in Section 3.3 to save ~80 words.
"""

import os

for path in ['/home/alireza/Sepsis-Onset-Timing-Paradox/manuscript/latex/main_review_doublespaced.tex',
             '/home/alireza/Sepsis-Onset-Timing-Paradox/manuscript/latex/main.tex']:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Find the bullet list in Results
    start_str = r"At an emitted workload of 10 alert episodes per 100 patient-days ($K=407$ emitted episodes across 4,070.8 patient-days):"
    end_str = r"\end{itemize}"

    p_start = text.find(start_str)
    if p_start != -1:
        p_end = text.find(end_str, p_start) + len(end_str)
        old_block = text[p_start:p_end]

        new_block = r"""At an emitted workload of 10 alert episodes per 100 patient-days ($K=407$ emitted episodes across 4,070.8 patient-days), episode-level PPV across all 8 models ranged from 0.98\% to 1.72\%, corresponding to 58.1 to 101.8 simulated alert evaluations per detected patient ($\text{NNE}$). BiMamba SSM and LightGBM achieved the highest PPV at 1.72\% (95\% CI: 0.49--2.95\%; 7 detected cases out of 150 eligible; NNE = 58.1, 95\% CI: 33.9--203.5; covered sensitivity 4.64\%; population yield 1.72\%). Stacking Ensemble, XGBoost, and Random Forest detected 6 cases each (PPV 1.47\%, NNE = 67.8); Mask-Aware Transformer and Clinical Transformer detected 5 cases each (PPV 1.23\%, NNE = 81.4); and L2-Logistic Regression detected 4 cases (PPV 0.98\%, NNE = 101.8). At 20 episodes/100 pt-days, XGBoost detected 15 patients (NNE = 54.3) and Mask-Aware Transformer detected 14 patients (NNE = 58.1). In this sample, horizon-correct alerts and alerts satisfying the two-hour pre-suspicion margin ($t_{\mathrm{first}, i} + 2\text{h} \le T_{\mathrm{susp}, i}$) were identical across all models (7 of 7 for BiMamba SSM), occurring before suspicion ($T_{\mathrm{susp}}$) rather than earliest prescribing ($T_{\mathrm{Rx,start}}$).

Of the 407 emitted alerts for BiMamba SSM, 54 fired on sepsis stays (7 timely within $(0, 6]\text{h}$; 47 premature $>6\text{h}$ prior to onset) and 353 on control stays. Crucially, 71 control alerts (17.4\% of all alerts; 20.1\% of control alerts) fired on pre-treated controls ($T_{\mathrm{Rx,start}} \le 19\text{h}$), while 277 fired on untreated controls, and 5 on controls with prescriptions after hour 30. Across models, pre-treated controls consumed 17.4\% (BiMamba SSM, Stacking Ensemble; 71/407) to 31.4\% (LightGBM; 128/407) of all alerts. With 7 timely detections ($N_{\mathrm{false}} = 407 - 7 = 400$), retrospectively removing alerts on pre-treated controls removes 17.8\% (71/400) and 32.0\% (128/400) of false alerts for BiMamba SSM and LightGBM, respectively. These counts describe retrospective alert occupancy; a prospective suppression rule would apply to all patients and require verified active administration rather than date-coarsened orders, which would otherwise silence all 408 sepsis stays. Real-time effects on volume, detection, and safety were not prospectively evaluated."""

        text = text[:p_start] + new_block + text[p_end:]
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Successfully streamlined Results bullets in {os.path.basename(path)}")
    else:
        print(f"Start string not found in {os.path.basename(path)}")
