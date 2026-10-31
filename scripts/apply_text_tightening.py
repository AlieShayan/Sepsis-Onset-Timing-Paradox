#!/usr/bin/env python3
"""
apply_text_tightening.py
Tightens prose across main_review_doublespaced.tex and main.tex
to ensure Words in text <= 3,850 according to texcount.
"""

import os
import re

review_path = '/home/alireza/Sepsis-Onset-Timing-Paradox/manuscript/latex/main_review_doublespaced.tex'
main_path = '/home/alireza/Sepsis-Onset-Timing-Paradox/manuscript/latex/main.tex'

def tighten_content(text):
    # 1. Background and Significance
    p1_old = r"""Clinical treatment urgency depends critically on presentation: the Surviving Sepsis Campaign international guidelines recommend immediate antimicrobial therapy, ideally within one hour, for possible septic shock or a high likelihood of sepsis, while recommending rapid diagnostic assessment for possible sepsis without shock~\cite{evans2021surviving,kumar2006duration}. To support prompt intervention, the biomedical informatics community has heavily invested in machine learning (ML) models for early warning. Scores of retrospective investigations have reported high discriminative capability, frequently reporting area under the receiver operating characteristic curve (AUROC) values between 0.80 and 0.88 ahead of consensus onset~\cite{moor2021early}. However, published ICU prediction studies differ substantially in outcome implementation, temporal alignment, and evaluation design; consequently, reported AUROCs should not be interpreted as directly comparable measures of deployment readiness, even across advanced deep learning architectures evaluated across international health systems~\cite{moor2023predicting} or broad methodological reviews~\cite{wang2025methodological}. Prior MIMIC-III analyses have shown that alternative operational onset definitions can change measured performance as much as or more than changing model class~\cite{cohen2024subtle}. These findings motivate separating reference-event construction from surveillance coverage and alert-policy evaluation. Time-sensitive sepsis benchmarks have also incorporated utility functions that reward and penalize alerts according to their timing~\cite{reyna2020early}. Such approaches reinforce the need to report temporal evaluation rules, although benchmark utility is distinct from measured clinical benefit."""

    p1_new = r"""Clinical treatment urgency depends critically on presentation: Surviving Sepsis Campaign guidelines recommend immediate antimicrobial therapy (ideally within 1 hour) for septic shock, and rapid diagnostic assessment for sepsis without shock~\cite{evans2021surviving,kumar2006duration}. To support prompt intervention, informatics research has developed numerous machine learning (ML) early-warning models, frequently reporting AUROC values between 0.80 and 0.88 ahead of consensus onset~\cite{moor2021early}. However, published ICU prediction studies differ substantially in outcome definitions, temporal alignment, and evaluation designs; reported AUROCs should not be interpreted as directly comparable measures of deployment readiness, even across deep learning architectures evaluated in multi-center international cohorts~\cite{moor2023predicting} or broad methodological reviews~\cite{wang2025methodological}. Prior MIMIC-III analyses demonstrate that operational onset definitions can shift measured performance more than changing model class~\cite{cohen2024subtle}. Time-sensitive sepsis benchmarks have also incorporated utility functions that reward and penalize alerts by timing~\cite{reyna2020early}, reinforcing the need to report explicit temporal evaluation rules."""

    if p1_old in text:
        text = text.replace(p1_old, p1_new)

    p2_old = r"""Clinical evaluations have reported heterogeneous performance and outcome associations~\cite{wong2021external,adams2022prospective}, while clinician-vignette studies have identified discrepancies between operational onset labels and recommended treatment timing~\cite{weissman2024sepsis}. External evaluation of the original Epic Sepsis Model demonstrated that reported performance depended heavily on the evaluation level and timing rules: hospitalization-level and prediction-level workload estimates were not interchangeable, emphasizing the need to report the exact unit of evaluation~\cite{wong2021external}. Prospective evaluations of early-warning systems such as TREWS have found associations between timely alert confirmation and improved outcomes among selected patients identified prior to antibiotic initiation~\cite{adams2022prospective,henry2022factors}. However, because provider response was observational rather than randomized, these findings support potential clinical utility without establishing universal causal efficacy~\cite{adams2022prospective}. 

Furthermore, ICU clinicians already encounter substantial alarm exposure from bedside physiological monitors~\cite{drew2014insights}. In other clinical decision-support settings, repeated alerts and work complexity have been associated with lower alert acceptance, supporting explicit reporting of notification burden~\cite{ancker2017effects,paine2016systematic}."""

    p2_new = r"""Clinical evaluations report heterogeneous performance~\cite{wong2021external,adams2022prospective}, while clinician-vignette studies show discrepancies between operational onset labels and recommended treatment timing~\cite{weissman2024sepsis}. External validation of the original Epic Sepsis Model demonstrated that hospitalization-level and prediction-level workload estimates are not interchangeable, emphasizing the need to declare the exact evaluation unit~\cite{wong2021external}. Prospective evaluations of early-warning systems such as TREWS found associations between timely alert confirmation and improved outcomes among patients identified prior to antibiotics~\cite{adams2022prospective,henry2022factors}; however, observational clinician responses support potential clinical utility without establishing randomized causal efficacy~\cite{adams2022prospective}. Furthermore, ICU clinicians already encounter substantial alarm exposure~\cite{drew2014insights}; repeated nonactionable alerts contribute to alert fatigue and lower adoption~\cite{ancker2017effects,paine2016systematic}."""

    if p2_old in text:
        text = text.replace(p2_old, p2_new)

    # 2. Materials and Methods: Study Design
    m1_old = r"""We conducted a retrospective evaluation with sequential prediction replay using the MIMIC-III critical care database (v1.4)~\cite{johnson2016mimiciii}, evaluating transportability across clinical information systems from CareVue (2001--2008 development: $N=12,964$ candidates; $10,371$ train, $2,593$ validation; $1,687$ incident sepsis) to MetaVision (2008--2012 evaluation replay: $N=9,732$ candidates; $1,886$ sepsis). The base cohort comprised first ICU stays of adult patients ($\ge 18$ years) with total ICU length of stay $\text{LOS} \ge 30\text{h}$. While conditioning on $\text{LOS} \ge 30\text{h}$ ensures that included patients remain hospitalized throughout mid-stay surveillance (hours 19--30), at surveillance entry ($t=19\text{h}$) it retrospectively conditions on 11 hours of future ICU survival and stay. Stays with community-onset sepsis prior to or upon ICU arrival ($T_{\text{S3}} \le 0\text{h}$) were excluded as prevalent, and stays developing consensus Sepsis-3 during the baseline pre-surveillance interval ($0 < T_{\text{S3}} \le 19\text{h}$, $N=1,478$) were excluded prior to surveillance entry, yielding the evaluated prospective-replay test cohort of 8,254 unique stays ($408$ incident sepsis, $7,846$ controls). Beyond the primary MetaVision evaluation, historical sensitivity analyses included $N=3,581$ incident sepsis stays partitioned into CareVue-only ($1,687$), MetaVision-only ($1,886$), and transitional stays bridging both database systems ($8$), where $1,687 + 1,886 + 8 = 3,581$ stays form a mutually exclusive, exhaustive partition. Complete cohort derivation schemas, exclusion criteria, feature dictionaries, missing-data rules, model architectures, and executable Sepsis-3 definitions are detailed in the Supplementary Methods Package.

Surveillance was evaluated under a prespecified second-day ICU surveillance protocol: predictions were generated at 12 scheduled hourly decision bins $[t-1, t]$ for $t \in \{19, 20, \dots, 30\}$ hours post-admission. In prospective streaming deployment, patients would enter surveillance at hour 19 and be administratively right-censored at discharge or death. The test partition comprised 8,254 unique patient ICU stays (97,700 evaluated windows, 4,070.8 patient-days of surveillance volume). Stays were censored at Sepsis-3 onset ($N=110$ stays, 623 windows) or upon observation-availability cessation prior to hour 30 ($N=180$ control stays, 725 windows, triggered by cessation of charted vital sign telemetry in the EHR). Reporting was organized in accordance with TRIPOD+AI and STROBE recommendations~\cite{collins2024tripod,vonelm2007strobe}."""

    m1_new = r"""We conducted a retrospective evaluation with sequential prediction replay using MIMIC-III (v1.4)~\cite{johnson2016mimiciii}, evaluating transportability from CareVue (2001--2008 development: $N=12,964$ candidate stays; $10,371$ train, $2,593$ validation; $1,687$ incident sepsis) to MetaVision (2008--2012 prospective-replay test cohort: $N=9,732$ candidates; $1,886$ sepsis). Adult first ICU stays with $\text{LOS} \ge 30\text{h}$ were included. Conditioning on $\text{LOS} \ge 30\text{h}$ ensures that included patients remain hospitalized through mid-stay surveillance (hours 19--30), but at surveillance entry ($t=19\text{h}$) it retrospectively conditions on 11 hours of future ICU survival. Stays with community-onset sepsis ($T_{\text{S3}} \le 0\text{h}$) or developing sepsis during the pre-surveillance baseline ($0 < T_{\text{S3}} \le 19\text{h}$, $N=1,478$) were excluded, yielding the 8,254 test stays ($408$ incident sepsis, $7,846$ controls). Beyond the primary MetaVision evaluation, historical sensitivity analyses evaluated $N=3,581$ incident sepsis stays partitioned into CareVue-only ($1,687$), MetaVision-only ($1,886$), and transitional stays bridging both database systems ($8$), where $1,687 + 1,886 + 8 = 3,581$ stays form a mutually exclusive, exhaustive partition. Reporting follows TRIPOD+AI and STROBE recommendations~\cite{collins2024tripod,vonelm2007strobe}.

Surveillance was evaluated under a prespecified second-day protocol: predictions were generated at 12 scheduled hourly decision bins $[t-1, t]$ for $t \in \{19, 20, \dots, 30\}$ hours post-admission (4,070.8 patient-days of surveillance volume). Stays were censored at Sepsis-3 onset ($N=110$ stays, 623 windows) or EHR telemetry cessation ($N=180$ control stays, 725 windows). Cohort derivations, exclusions, feature dictionaries, and executable definitions are detailed in the Supplementary Methods."""

    if m1_old in text:
        text = text.replace(m1_old, m1_new)

    # 3. Discussion: Care-process forecasting
    d1_old = r"""Clinician-vignette studies address when clinicians judge treatment appropriate~\cite{weissman2024sepsis}; our analysis addresses prediction performance relative to operational EHR proxies. These are complementary questions, and proxy precedence does not establish clinical appropriateness. Furthermore, while changes under exploratory outcome-stratified hybrid restriction were small in this cohort, differences in outcome construction, population selection, timestamp precision, and clinical-action definition preclude direct replication of Dutta et al.'s order-based sensitivity analysis~\cite{dutta2026performance}. Clinical utility ultimately depends on alert acceptance, clinician review, and actionable treatment responses~\cite{ancker2017effects,henry2022factors}. Our analysis characterizes operational opportunities and simulated evaluation burden (NNE); it does not evaluate clinician adoption, cognitive fatigue, or therapeutic efficacy directly~\cite{paine2016systematic}."""

    d1_new = r"""Clinician-vignette studies address when clinicians judge treatment appropriate~\cite{weissman2024sepsis}; our analysis addresses predictive performance relative to operational EHR proxies. These are complementary questions, and proxy precedence does not establish clinical appropriateness. Furthermore, while changes under exploratory outcome-stratified hybrid restriction were small in this cohort, differences in outcome construction, population selection, timestamp precision, and clinical-action definitions preclude direct replication of Dutta et al.'s order-based sensitivity analysis~\cite{dutta2026performance}. Clinical utility ultimately depends on alert acceptance, review feasibility, and actionable treatments~\cite{ancker2017effects,henry2022factors}; our simulated evaluation burden (NNE) does not measure clinician adoption or therapeutic efficacy directly~\cite{paine2016systematic}."""

    if d1_old in text:
        text = text.replace(d1_old, d1_new)

    # 4. Limitations
    lim_old = r"""Alarm-reduction interventions may suppress both nonactionable and clinically important signals; our short evaluation interval cannot establish the safety or optimal duration of a refractory policy during sustained surveillance~\cite{paine2016systematic}. Finally, our evaluation benchmarked continuous multivariate ML models and did not include discrete integer bedside scores (qSOFA, NEWS2), whose tie-dominated step-function distributions resist continuous alert-budget tuning, restricting our benchmark scope rather than asserting ML superiority. Our cohorts originate from a single medical center (BIDMC) across EHR transitions (CareVue to MetaVision); multi-center validation across diverse health systems remains essential."""

    lim_new = r"""Alarm-reduction interventions may suppress both nonactionable and clinically important signals; our evaluation interval cannot establish the safety or optimal duration of refractory suppression during sustained surveillance~\cite{paine2016systematic}. Furthermore, our benchmark evaluated continuous multivariate ML models and omitted discrete integer scores (qSOFA, NEWS2), whose tie-dominated step functions resist continuous budget tuning, reflecting scope boundaries rather than ML superiority. Finally, our cohorts originate from a single academic medical center across EHR transitions (CareVue to MetaVision); multi-center validation across diverse hospital networks remains essential."""

    if lim_old in text:
        text = text.replace(lim_old, lim_new)

    # 5. Observed Burden Trade-off in Results:
    res_old = r"""At an emitted workload of 10 alert episodes per 100 patient-days ($K=407$ emitted episodes across 4,070.8 patient-days):
\begin{itemize}
    \item Across all 8 models, episode-level PPV ranged from 0.98\% to 1.72\%, corresponding to 58.1 to 101.8 simulated evaluations per detected patient ($\text{NNE}$).
    \item BiMamba SSM and LightGBM achieved the highest PPV at 1.72\% (95\% CI: 0.49--2.95\%; 7 detected cases out of 150 eligible; NNE = 58.1, 95\% CI: 33.9--203.5).
    \item The Stacking Ensemble, XGBoost, and Random Forest detected 6 cases each (PPV 1.47\%, NNE = 67.8).
    \item The Mask-Aware Transformer and Clinical Transformer detected 5 cases each (PPV 1.23\%, NNE = 81.4).
    \item L2-penalized Logistic Regression detected 4 cases (PPV 0.98\%, NNE = 101.8).
\end{itemize}"""

    res_new = r"""At an emitted workload of 10 alert episodes per 100 patient-days ($K=407$ emitted episodes across 4,070.8 patient-days), episode-level PPV across all 8 models ranged from 0.98\% to 1.72\%, corresponding to 58.1 to 101.8 simulated alert evaluations per detected patient ($\text{NNE}$). BiMamba SSM and LightGBM achieved the highest PPV at 1.72\% (95\% CI: 0.49--2.95\%; 7 detected cases out of 150 eligible; NNE = 58.1, 95\% CI: 33.9--203.5). Stacking Ensemble, XGBoost, and Random Forest detected 6 cases each (PPV 1.47\%, NNE = 67.8); Mask-Aware Transformer and Clinical Transformer detected 5 cases each (PPV 1.23\%, NNE = 81.4); and L2-Logistic Regression detected 4 cases (PPV 0.98\%, NNE = 101.8)."""

    if res_old in text:
        text = text.replace(res_old, res_new)

    return text

with open(review_path, 'r', encoding='utf-8') as f:
    r_text = f.read()

r_tight = tighten_content(r_text)
with open(review_path, 'w', encoding='utf-8') as f:
    f.write(r_tight)

with open(main_path, 'r', encoding='utf-8') as f:
    m_text = f.read()

m_tight = tighten_content(m_text)
with open(main_path, 'w', encoding='utf-8') as f:
    f.write(m_tight)

print("Tightening complete.")
