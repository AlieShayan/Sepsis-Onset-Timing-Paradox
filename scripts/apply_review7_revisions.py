#!/usr/bin/env python3
"""
apply_review7_revisions.py
Implements all Review #7 editorial audit corrections across:
  - manuscript/latex/main_review_doublespaced.tex
  - manuscript/latex/main.tex
"""

import os
import re
import subprocess

REPO_DIR = '/home/alireza/Sepsis-Onset-Timing-Paradox'
LATEX_DIR = os.path.join(REPO_DIR, 'manuscript/latex')

def update_review_manuscript():
    path = os.path.join(LATEX_DIR, 'main_review_doublespaced.tex')
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().replace('\r\n', '\n')

    # 1. Preamble: Add pdflscape
    if r'\usepackage{pdflscape}' not in text:
        text = text.replace(r'\usepackage{setspace}', r'\usepackage{setspace}' + '\n' + r'\usepackage{pdflscape}')

    # 2. Lay summary: Replace "requiring clinicians to review over 58 alerts"
    lay_target = "requiring clinicians to review over 58 alerts for every true case detected."
    lay_repl = "equivalent to approximately 58 simulated alert episodes per timely detected case."
    assert lay_target in text, "Lay summary target not found"
    text = text.replace(lay_target, lay_repl)

    # 3. Introduction Cascade item 4:
    item4_target = r"\item \textbf{Emitted-Alert Workload Constraints ($D$):} Operational alert budgets (5, 10, 20 episodes/100 patient-days) with 12-hour suppression, demonstrating bedside detection yield $<2\%$."
    item4_repl = r"\item \textbf{Emitted-Alert Workload Constraints ($D$):} Operational alert budgets (5, 10, 20 episodes/100 patient-days) with 12-hour suppression, including a maximum observed population yield of 1.72\% at 10 episodes/100 surveillance patient-days."
    assert item4_target in text, "Item 4 target not found"
    text = text.replace(item4_target, item4_repl)

    # 4. In-text citations and literature survey in §1.1:
    # First, survey sentence: Moor 2021 as systematic review, Moor 2023 for international deep learning, Wang 2025 for validation review
    lit_target = r"frequently reporting area under the receiver operating characteristic curve (AUROC) values between 0.80 and 0.88 ahead of consensus onset~\cite{moor2021early}. However, published ICU prediction studies differ substantially in outcome implementation, temporal alignment, and evaluation design; consequently, reported AUROCs should not be interpreted as directly comparable measures of deployment readiness~\cite{moor2021early,wang2025methodological}."
    lit_repl = r"frequently reporting area under the receiver operating characteristic curve (AUROC) values between 0.80 and 0.88 ahead of consensus onset~\cite{moor2021early}. However, published ICU prediction studies differ substantially in outcome implementation, temporal alignment, and evaluation design; consequently, reported AUROCs should not be interpreted as directly comparable measures of deployment readiness, even across advanced deep learning architectures evaluated across international health systems~\cite{moor2023predicting} or broad methodological reviews~\cite{wang2025methodological}."
    assert lit_target in text, "Lit target not found"
    text = text.replace(lit_target, lit_repl)

    # Cite Schamoni 2019 and van der Vegt 2022 sequentially after Do 2026:
    do_target = r"Do et al.~\cite{do2026continuous} further showed that continuous surveillance and alert silencing rules fundamentally alter clinical workload across health systems. This literature indicates that pre-intervention performance does not follow a uniform collapse, but depends on how onset labels, clinical-process anchors, and evaluation units are constructed."
    do_repl = r"Do et al.~\cite{do2026continuous} further showed that continuous surveillance and alert silencing rules fundamentally alter clinical workload across health systems. Addressing label contamination and circularity in sepsis definitions, Schamoni et al.~\cite{schamoni2019leveraging} emphasized that clinical recognition anchors require explicit separation from physiological deterioration. Furthermore, clinical database timestamps carry inherent recording latency, clock drift, and temporal coarsening~\cite{vandervegt2022timing}. This literature indicates that pre-intervention performance does not follow a uniform collapse, but depends on how onset labels, clinical-process anchors, and evaluation units are constructed."
    assert do_target in text, "Do target not found"
    text = text.replace(do_target, do_repl)

    # 5. Section 2.1: Historical cohort partition documentation
    cohort_target = r"Beyond the primary MetaVision evaluation, historical sensitivity analyses included $N=3,581$ incident sepsis stays across CareVue ($1,687$), MetaVision ($1,886$), and 8 transitional stays bridging both database systems ($1,687 + 1,886 + 8 = 3,581$)."
    cohort_repl = r"Beyond the primary MetaVision evaluation, historical sensitivity analyses included $N=3,581$ incident sepsis stays partitioned into CareVue-only ($1,687$), MetaVision-only ($1,886$), and transitional stays bridging both database systems ($8$), where $1,687 + 1,886 + 8 = 3,581$ stays form a mutually exclusive, exhaustive partition."
    assert cohort_target in text, "Cohort target not found"
    text = text.replace(cohort_target, cohort_repl)

    # 6. Section 2.2: Baseline SOFA wording
    sofa_target = r"this identifies the first qualifying bin relative to a retrospective search-window minimum rather than an observed increase from a pre-ICU baseline."
    sofa_repl = r"this identifies the first qualifying bin relative to a retrospective search-window minimum rather than an observed increase from a pre-ICU baseline; an alternative antecedent-baseline sensitivity analysis was not performed."
    assert sofa_target in text, "SOFA target not found"
    text = text.replace(sofa_target, sofa_repl)

    # 7. Section 3.2: AUPRC wording
    auprc_target = r"The increase in raw AUPRC (BiMamba: 1.209\% to 1.656\%) was mechanically driven by control-window censoring, which raised baseline positive prevalence by $1.384\times$ (from $\pi=0.694\%$ to $0.960\%$)."
    auprc_repl = r"The increase in raw AUPRC (BiMamba: 1.209\% to 1.656\%) accompanied control-window censoring and higher positive prevalence (raised by $1.384\times$, from $\pi=0.694\%$ to $0.960\%$); these comparisons do not isolate improved ranking performance."
    assert auprc_target in text, "AUPRC target not found"
    text = text.replace(auprc_target, auprc_repl)

    # 8. Section 3.4: Suspicion-Anchor Mathematics & Endpoint Error
    math_target = r"""Table~\ref{tab:table4} presents the cross-tabulation of Sepsis Onset Timing against Treatment Proxy Timing. Because MIMIC-III prescription start dates (\texttt{prescriptions.\allowbreak startdate}) lack time of day, order initiation is evaluated both under declared point assignment (midnight $a_i$) and under calendar-day intervals $[a_i, b_i) = [\text{midnight}, \text{next midnight})$. For stay $i$, let minute-resolved culture charttime be $c_i$ and the coarsened prescription interval be $[a_i, b_i)$. The transformed suspicion anchor interval is $[L_i, U_i)$, where $L_i = \min(a_i, c_i)$ and $U_i = \min(b_i, c_i)$. A stay has definite opportunity ($O_i^{\mathrm{def}} = 1$) if there exists an evaluated horizon-positive decision bin $t \in \mathcal{T}_i$ satisfying $t + \delta \le L_i$ ($\forall C \in [a_i, b_i), t + \delta \le \min(C, c_i)$); it has possible opportunity ($O_i^{\mathrm{poss}} = 1$) if there exists $t \in \mathcal{T}_i$ such that $t + \delta < U_i$ ($\exists C \in [a_i, b_i) : t + \delta \le \min(C, c_i)$). Because $b_i$ is excluded from $[a_i, b_i)$, strict inequality $t + \delta < U_i$ is required for non-empty feasible realizations. Stays with $O_i^{\mathrm{def}} = 0$ but $O_i^{\mathrm{poss}} = 1$ are indeterminate ($O_i^{\mathrm{indet}} = 1$). All indicators strictly satisfy $O_i^{\mathrm{def}} \le O_i^{\mathrm{point}} \le O_i^{\mathrm{poss}}$ and $O_i^{\mathrm{poss}} = O_i^{\mathrm{def}} + O_i^{\mathrm{indet}}$, with $O_i^{\mathrm{def}} \equiv O_i^{\mathrm{point}}$ by construction because midnight point assignment $a_i = \inf [a_i, b_i)$. Importantly, these opportunity bounds are strictly conditional on fixed operational Sepsis-3 timestamps and surveillance coverage; they do not propagate counterfactual antibiotic timings through the upstream phenotype (which could alter qualifying pairs, baseline SOFA, onset, or coverage).

Opportunity depended strongly on the selected clinical-process anchor. Under the infection-suspicion anchor ($T_{\mathrm{susp}}$), 150 of 151 covered patients (99.3\% within-covered; 36.8\% of total incident sepsis) had decision bins at or before $T_{\mathrm{susp}}$ under declared midnight point assignment $a_i$. Because $a_i \le T_{\mathrm{susp}, i}$ across $[a_i, b_i)$, these 150 stays are definitely eligible throughout the calendar-day interval ($O_i^{\mathrm{def}} = 150$), with only 1 indeterminate stay ($O_i^{\mathrm{indet}} = 1$), bounding opportunity between 36.8\% and 37.0\% ($[150, 151]/408$; $[99.3\%, 100.0\%]$ within-covered)."""

    math_repl = r"""Table~\ref{tab:table4} presents the cross-tabulation of Sepsis Onset Timing against Treatment Proxy Timing. Because MIMIC-III prescription start dates (\texttt{prescriptions.\allowbreak startdate}) lack time of day, order initiation is evaluated both under declared point assignment (midnight $a_i$) and under calendar-day intervals $[a_i, b_i) = [\text{midnight}, \text{next midnight})$. For stay $i$, let minute-resolved culture charttime be $c_i$, coarsened prescription interval be $[a_i, b_i)$, and pre-onset evaluation windows be $P_i = \{ t \in \mathcal{T}_i : 0 < T_{\mathrm{S3}, i} - t \le H \}$. Under the infection-suspicion anchor $T_{\mathrm{susp}, i} = \min(A_i, c_i)$ with unobserved $A_i \in [a_i, b_i)$, the attainable set $\mathcal{C}_i$ of $T_{\mathrm{susp}, i}$ is $\{c_i\}$ if $c_i < a_i$, $[a_i, c_i]$ if $a_i \le c_i < b_i$, and $[a_i, b_i)$ if $c_i \ge b_i$. Because $C_i = c_i$ is attained whenever $A_i \ge c_i$, the upper endpoint is closed when $a_i \le c_i < b_i$. Definite opportunity requires $t + \delta \le \min(A_i, c_i)$ for all $A_i \in [a_i, b_i)$, holding if and only if $t + \delta \le a_i$ and $t + \delta \le c_i$:
\begin{equation}
O_i^{\mathrm{def}}(\delta) = \mathbf{1} \bigl\{ \exists t \in P_i : t + \delta \le a_i \;\land\; t + \delta \le c_i \bigr\}
\end{equation}
Possible opportunity requires $t + \delta \le \min(A_i, c_i)$ for at least one $A_i \in [a_i, b_i)$, which holds if and only if $t + \delta < b_i$ and $t + \delta \le c_i$:
\begin{equation}
O_i^{\mathrm{poss}}(\delta) = \mathbf{1} \bigl\{ \exists t \in P_i : t + \delta < b_i \;\land\; t + \delta \le c_i \bigr\}
\end{equation}
For the earliest prescription proxy alone ($T_{\mathrm{Rx,start}}$):
\begin{equation}
O_i^{\mathrm{def}}(\delta) = \mathbf{1} \bigl\{ \exists t \in P_i : t + \delta \le a_i \bigr\}, \qquad O_i^{\mathrm{poss}}(\delta) = \mathbf{1} \bigl\{ \exists t \in P_i : t + \delta < b_i \bigr\}
\end{equation}
Stays with $O_i^{\mathrm{def}} = 0$ but $O_i^{\mathrm{poss}} = 1$ are indeterminate ($O_i^{\mathrm{indet}} = 1$). All indicators strictly satisfy $O_i^{\mathrm{def}} \le O_i^{\mathrm{point}} \le O_i^{\mathrm{poss}}$ and $O_i^{\mathrm{poss}} = O_i^{\mathrm{def}} + O_i^{\mathrm{indet}}$, with $O_i^{\mathrm{def}} \equiv O_i^{\mathrm{point}}$ by construction because midnight point assignment $a_i = \inf [a_i, b_i)$. Importantly, these opportunity bounds are strictly conditional on fixed operational Sepsis-3 timestamps and surveillance coverage; they do not propagate counterfactual antibiotic timings through the upstream phenotype (which could alter qualifying pairs, baseline SOFA, onset, or coverage).

Opportunity depended strongly on the selected clinical-process anchor. Under the infection-suspicion anchor ($T_{\mathrm{susp}}$), 150 of 151 covered stays (99.3\% within-covered; 36.8\% of total incident sepsis) had decision bins satisfying the definite opportunity condition under declared midnight point assignment $a_i$. Across the calendar-day interval, exactly 150 stays satisfy $O_i^{\mathrm{def}} = 1$, with only 1 indeterminate stay ($O_i^{\mathrm{indet}} = 1$), bounding opportunity between 36.8\% and 37.0\% ($[150, 151]/408$; $[99.3\%, 100.0\%]$ within-covered)."""

    assert math_target in text, "Math target not found"
    text = text.replace(math_target, math_repl)

    # 9. Section 3.5: Oracle paragraph wording
    oracle_target = r"Low empirical detection yield is thus driven by model discrimination and alert suppression, not a mathematical ceiling of the budget."
    oracle_repl = r"The low observed yield reflects performance of the tested score-and-first-alert policies within the restricted surveillance protocol, rather than a budget-imposed ceiling near 2\%."
    assert oracle_target in text, "Oracle target not found"
    text = text.replace(oracle_target, oracle_repl)

    # 10. Bibliography: Replace with 26 verified references
    bib_target = r"""\begin{thebibliography}{25}
\fontsize{8.5pt}{10.5pt}\selectfont
\setlength{\itemsep}{2pt}
\setlength{\parskip}{0pt}
\bibitem{singer2016sepsis} Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). \textit{JAMA}. 2016;315(8):801--810. DOI: 10.1001/jama.2016.0287.
\bibitem{seymour2016assessment} Seymour CW, Liu VX, Iwashyna TJ, et al. Assessment of clinical criteria for sepsis: for the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). \textit{JAMA}. 2016;315(8):762--774. DOI: 10.1001/jama.2016.0288.
\bibitem{evans2021surviving} Evans L, Rhodes A, Alhazzani W, et al. Surviving sepsis campaign: international guidelines for management of sepsis and septic shock 2021. \textit{Intensive Care Med}. 2021;47(11):1181--1247. DOI: 10.1007/s00134-021-06506-y.
\bibitem{kumar2006duration} Kumar A, Roberts D, Wood KE, et al. Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock. \textit{Crit Care Med}. 2006;34(6):1589--1596. DOI: 10.1097/01.CCM.0000217961.75225.E9.
\bibitem{moor2021early} Moor M, Bennett N, Plecko D, et al. Predicting sepsis using deep learning across international sites: a retrospective development and validation study. \textit{EClinicalMedicine}. 2023;62:102100. DOI: 10.1016/j.eclinm.2023.102100.
\bibitem{wang2025methodological} Wang Z, Wang W, Sun C, et al. A methodological systematic review of validation and performance of sepsis real-time prediction models. \textit{npj Digit Med}. 2025;8:190. DOI: 10.1038/s41746-025-01587-1.
\bibitem{cohen2024subtle} Cohen SN, Foster J, Foster P, et al. Subtle variation in sepsis-III definitions markedly influences predictive performance within and across methods. \textit{Sci Rep}. 2024;14:1920. DOI: 10.1038/s41598-024-51989-6.
\bibitem{reyna2020early} Reyna MA, Josef CS, Jeter R, et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. \textit{Crit Care Med}. 2020;48(2):210--217. DOI: 10.1097/CCM.0000000000004145.
\bibitem{wong2021external} Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. \textit{JAMA Intern Med}. 2021;181(8):1065--1070. DOI: 10.1001/jamainternmed.2021.2626.
\bibitem{adams2022prospective} Adams R, Henry KE, Sridharan A, et al. Prospective, multi-site study of patient outcomes after implementation of the TREWS machine learning-based early warning system for sepsis. \textit{Nat Med}. 2022;28(7):1455--1460. DOI: 10.1038/s41591-022-01894-0.
\bibitem{weissman2024sepsis} Weissman GE, Hubbard RA, Himes BE, et al. Sepsis prediction models are trained on labels that diverge from clinician-recommended treatment times. \textit{AMIA Annu Symp Proc}. 2024;2024:1215--1224.
\bibitem{henry2022factors} Henry KE, Adams R, Parent C, et al. Factors driving provider adoption of the TREWS machine learning-based early warning system and its effects on sepsis treatment timing. \textit{Nat Med}. 2022;28(7):1447--1454. DOI: 10.1038/s41591-022-01895-z.
\bibitem{drew2014insights} Drew BJ, Harris P, Zègre-Hemsey JK, et al. Insights into the problem of alarm fatigue with physiologic monitor devices: a comprehensive observational study of consecutive intensive care unit patients. \textit{PLoS One}. 2014;9(10):e110274. DOI: 10.1371/journal.pone.0110274.
\bibitem{ancker2017effects} Ancker JS, Edwards A, Nosal S, et al. Effects of workload, work complexity, and repeated alerts on alert fatigue in a clinical decision support system. \textit{BMC Med Inform Decis Mak}. 2017;17(1):36. DOI: 10.1186/s12911-017-0430-8.
\bibitem{paine2016systematic} Paine CW, Goel VV, Ely E, et al. Systematic review of physiologic monitor alarm characteristics and pragmatic interventions to reduce alarm frequency. \textit{J Hosp Med}. 2016;11(2):136--144. DOI: 10.1002/jhm.2520.
\bibitem{kamran2024evaluation} Kamran F, Tjandra D, Heiler A, et al. Evaluation of sepsis prediction models before onset of treatment. \textit{NEJM AI}. 2024;1(3):AIoa2300032. DOI: 10.1056/AIoa2300032.
\bibitem{dutta2026performance} Dutta S, McMurry R, Tasi MC, et al. Performance of a sepsis prediction model across different sepsis definitions. \textit{JAMA Netw Open}. 2026;9(4):e265599. DOI: 10.1001/jamanetworkopen.2026.5599.
\bibitem{do2026continuous} Do DK, Rockenschaub P, Boie SD, et al. The Impact of Evaluation Strategy on Sepsis Prediction Model Performance Metrics in Intensive Care Data: Retrospective Cohort Study. \textit{J Med Internet Res}. 2026;28:e72083. DOI: 10.2196/72083.
\bibitem{schamoni2019leveraging} Schamoni S, Lindner HA, Schneider-Lindner V, et al. Leveraging continuous physiological parameters for non-circular sepsis prediction. \textit{PLoS One}. 2019;14(10):e0223548. DOI: 10.1371/journal.pone.0223548.
\bibitem{goyal2022characterizing} Goyal R, et al. Characterizing timestamp discrepancies and data latency in electronic health records for clinical machine learning. \textit{J Am Med Inform Assoc}. 2022;29(5):890--899. DOI: 10.1093/jamia/ocab289.
\bibitem{johnson2016mimiciii} Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. \textit{Sci Data}. 2016;3:160035. DOI: 10.1038/sdata.2016.35.
\bibitem{collins2024tripod} Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. \textit{BMJ}. 2024;385:e078378. DOI: 10.1136/bmj-2023-078378.
\bibitem{vonelm2007strobe} von Elm E, Altman DG, Egger M, et al. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies. \textit{Lancet}. 2007;370(9596):1453--1457. DOI: 10.1016/S0140-6736(07)61602-X.
\bibitem{gu2023mamba} Gu A, Dao T. Mamba: Linear-time sequence modeling with selective state spaces. In: \textit{First Conference on Language Modeling (COLM)}; 2024.
\bibitem{gu2022efficiently} Gu A, Goel K, Ré C. Efficiently modeling long sequences with structured state spaces. In: \textit{International Conference on Learning Representations (ICLR)}; 2022.
\end{thebibliography}"""

    bib_repl = r"""\begin{thebibliography}{26}
\fontsize{8.5pt}{10.5pt}\selectfont
\setlength{\itemsep}{2pt}
\setlength{\parskip}{0pt}
\bibitem{singer2016sepsis} Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). \textit{JAMA}. 2016;315(8):801--810. DOI: 10.1001/jama.2016.0287.
\bibitem{seymour2016assessment} Seymour CW, Liu VX, Iwashyna TJ, et al. Assessment of clinical criteria for sepsis: for the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). \textit{JAMA}. 2016;315(8):762--774. DOI: 10.1001/jama.2016.0288.
\bibitem{evans2021surviving} Evans L, Rhodes A, Alhazzani W, et al. Surviving sepsis campaign: international guidelines for management of sepsis and septic shock 2021. \textit{Intensive Care Med}. 2021;47(11):1181--1247. DOI: 10.1007/s00134-021-06506-y.
\bibitem{kumar2006duration} Kumar A, Roberts D, Wood KE, et al. Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock. \textit{Crit Care Med}. 2006;34(6):1589--1596. DOI: 10.1097/01.CCM.0000217961.75225.E9.
\bibitem{moor2021early} Moor M, Rieck B, Horn M, et al. Early prediction of sepsis in the ICU using machine learning: a systematic review. \textit{Front Med (Lausanne)}. 2021;8:607952. DOI: 10.3389/fmed.2021.607952.
\bibitem{moor2023predicting} Moor M, Bennett N, Plecko D, et al. Predicting sepsis using deep learning across international sites: a retrospective development and validation study. \textit{EClinicalMedicine}. 2023;62:102124. DOI: 10.1016/j.eclinm.2023.102124.
\bibitem{wang2025methodological} Wang Z, Wang W, Sun C, et al. A methodological systematic review of validation and performance of sepsis real-time prediction models. \textit{npj Digit Med}. 2025;8:190. DOI: 10.1038/s41746-025-01587-1.
\bibitem{cohen2024subtle} Cohen SN, Foster J, Foster P, et al. Subtle variation in sepsis-III definitions markedly influences predictive performance within and across methods. \textit{Sci Rep}. 2024;14:1920. DOI: 10.1038/s41598-024-51989-6.
\bibitem{reyna2020early} Reyna MA, Josef CS, Jeter R, et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. \textit{Crit Care Med}. 2020;48(2):210--217. DOI: 10.1097/CCM.0000000000004145.
\bibitem{wong2021external} Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. \textit{JAMA Intern Med}. 2021;181(8):1065--1070. DOI: 10.1001/jamainternmed.2021.2626.
\bibitem{adams2022prospective} Adams R, Henry KE, Sridharan A, et al. Prospective, multi-site study of patient outcomes after implementation of the TREWS machine learning-based early warning system for sepsis. \textit{Nat Med}. 2022;28(7):1455--1460. DOI: 10.1038/s41591-022-01894-0.
\bibitem{weissman2024sepsis} Weissman GE, Hubbard RA, Himes BE, et al. Sepsis prediction models are trained on labels that diverge from clinician-recommended treatment times. \textit{AMIA Annu Symp Proc}. 2024;2024:1215--1224.
\bibitem{henry2022factors} Henry KE, Adams R, Parent C, et al. Factors driving provider adoption of the TREWS machine learning-based early warning system and its effects on sepsis treatment timing. \textit{Nat Med}. 2022;28(7):1447--1454. DOI: 10.1038/s41591-022-01895-z.
\bibitem{drew2014insights} Drew BJ, Harris P, Zègre-Hemsey JK, et al. Insights into the problem of alarm fatigue with physiologic monitor devices: a comprehensive observational study of consecutive intensive care unit patients. \textit{PLoS One}. 2014;9(10):e110274. DOI: 10.1371/journal.pone.0110274.
\bibitem{ancker2017effects} Ancker JS, Edwards A, Nosal S, et al. Effects of workload, work complexity, and repeated alerts on alert fatigue in a clinical decision support system. \textit{BMC Med Inform Decis Mak}. 2017;17(1):36. DOI: 10.1186/s12911-017-0430-8.
\bibitem{paine2016systematic} Paine CW, Goel VV, Ely E, et al. Systematic review of physiologic monitor alarm characteristics and pragmatic interventions to reduce alarm frequency. \textit{J Hosp Med}. 2016;11(2):136--144. DOI: 10.1002/jhm.2520.
\bibitem{kamran2024evaluation} Kamran F, Tjandra D, Heiler A, et al. Evaluation of sepsis prediction models before onset of treatment. \textit{NEJM AI}. 2024;1(3):AIoa2300032. DOI: 10.1056/AIoa2300032.
\bibitem{dutta2026performance} Dutta S, McMurry R, Tasi MC, et al. Performance of a sepsis prediction model across different sepsis definitions. \textit{JAMA Netw Open}. 2026;9(4):e265599. DOI: 10.1001/jamanetworkopen.2026.5599.
\bibitem{do2026continuous} Do DK, Rockenschaub P, Boie SD, et al. The Impact of Evaluation Strategy on Sepsis Prediction Model Performance Metrics in Intensive Care Data: Retrospective Cohort Study. \textit{J Med Internet Res}. 2026;28:e72083. DOI: 10.2196/72083.
\bibitem{schamoni2019leveraging} Schamoni S, Lindner HA, Schneider-Lindner V, et al. Leveraging implicit expert knowledge for non-circular machine learning in sepsis prediction. \textit{Artif Intell Med}. 2019;100:101725. DOI: 10.1016/j.artmed.2019.101725. PMID: 31607345.
\bibitem{vandervegt2022timing} van der Vegt AH, Campbell CA, Neale J, et al. Timing errors and temporal uncertainty in clinical databases---A narrative review. \textit{Front Digit Health}. 2022;4:932599. DOI: 10.3389/fdgth.2022.932599. PMID: 36248325.
\bibitem{johnson2016mimiciii} Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. \textit{Sci Data}. 2016;3:160035. DOI: 10.1038/sdata.2016.35.
\bibitem{collins2024tripod} Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. \textit{BMJ}. 2024;385:e078378. DOI: 10.1136/bmj-2023-078378.
\bibitem{vonelm2007strobe} von Elm E, Altman DG, Egger M, et al. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies. \textit{Lancet}. 2007;370(9596):1453--1457. DOI: 10.1016/S0140-6736(07)61602-X.
\bibitem{gu2023mamba} Gu A, Dao T. Mamba: Linear-time sequence modeling with selective state spaces. In: \textit{First Conference on Language Modeling (COLM)}; 2024.
\bibitem{gu2022efficiently} Gu A, Goel K, Ré C. Efficiently modeling long sequences with structured state spaces. In: \textit{International Conference on Learning Representations (ICLR)}; 2022.
\end{thebibliography}"""

    bib_repl = r"""\begin{thebibliography}{26}
\fontsize{7.5pt}{10.0pt}\selectfont
\setlength{\itemsep}{1pt}
\setlength{\parsep}{0pt}
\setlength{\parskip}{0pt}
\bibitem{singer2016sepsis} Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). \textit{JAMA}. 2016;315(8):801--810. DOI: 10.1001/jama.2016.0287.
\bibitem{seymour2016assessment} Seymour CW, Liu VX, Iwashyna TJ, et al. Assessment of clinical criteria for sepsis: for the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). \textit{JAMA}. 2016;315(8):762--774. DOI: 10.1001/jama.2016.0288.
\bibitem{evans2021surviving} Evans L, Rhodes A, Alhazzani W, et al. Surviving sepsis campaign: international guidelines for management of sepsis and septic shock 2021. \textit{Intensive Care Med}. 2021;47(11):1181--1247. DOI: 10.1007/s00134-021-06506-y.
\bibitem{kumar2006duration} Kumar A, Roberts D, Wood KE, et al. Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock. \textit{Crit Care Med}. 2006;34(6):1589--1596. DOI: 10.1097/01.CCM.0000217961.75225.E9.
\bibitem{moor2021early} Moor M, Rieck B, Horn M, et al. Early prediction of sepsis in the ICU using machine learning: a systematic review. \textit{Front Med (Lausanne)}. 2021;8:607952. DOI: 10.3389/fmed.2021.607952.
\bibitem{moor2023predicting} Moor M, Bennett N, Plecko D, et al. Predicting sepsis using deep learning across international sites: a retrospective development and validation study. \textit{EClinicalMedicine}. 2023;62:102124. DOI: 10.1016/j.eclinm.2023.102124.
\bibitem{wang2025methodological} Wang Z, Wang W, Sun C, et al. A methodological systematic review of validation and performance of sepsis real-time prediction models. \textit{npj Digit Med}. 2025;8:190. DOI: 10.1038/s41746-025-01587-1.
\bibitem{cohen2024subtle} Cohen SN, Foster J, Foster P, et al. Subtle variation in sepsis-III definitions markedly influences predictive performance within and across methods. \textit{Sci Rep}. 2024;14:1920. DOI: 10.1038/s41598-024-51989-6.
\bibitem{reyna2020early} Reyna MA, Josef CS, Jeter R, et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. \textit{Crit Care Med}. 2020;48(2):210--217. DOI: 10.1097/CCM.0000000000004145.
\bibitem{wong2021external} Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. \textit{JAMA Intern Med}. 2021;181(8):1065--1070. DOI: 10.1001/jamainternmed.2021.2626.
\bibitem{adams2022prospective} Adams R, Henry KE, Sridharan A, et al. Prospective, multi-site study of patient outcomes after implementation of the TREWS machine learning-based early warning system for sepsis. \textit{Nat Med}. 2022;28(7):1455--1460. DOI: 10.1038/s41591-022-01894-0.
\bibitem{weissman2024sepsis} Weissman GE, Hubbard RA, Himes BE, et al. Sepsis prediction models are trained on labels that diverge from clinician-recommended treatment times. \textit{AMIA Annu Symp Proc}. 2024;2024:1215--1224.
\bibitem{henry2022factors} Henry KE, Adams R, Parent C, et al. Factors driving provider adoption of the TREWS machine learning-based early warning system and its effects on sepsis treatment timing. \textit{Nat Med}. 2022;28(7):1447--1454. DOI: 10.1038/s41591-022-01895-z.
\bibitem{drew2014insights} Drew BJ, Harris P, Zègre-Hemsey JK, et al. Insights into the problem of alarm fatigue with physiologic monitor devices: a comprehensive observational study of consecutive intensive care unit patients. \textit{PLoS One}. 2014;9(10):e110274. DOI: 10.1371/journal.pone.0110274.
\bibitem{ancker2017effects} Ancker JS, Edwards A, Nosal S, et al. Effects of workload, work complexity, and repeated alerts on alert fatigue in a clinical decision support system. \textit{BMC Med Inform Decis Mak}. 2017;17(1):36. DOI: 10.1186/s12911-017-0430-8.
\bibitem{paine2016systematic} Paine CW, Goel VV, Ely E, et al. Systematic review of physiologic monitor alarm characteristics and pragmatic interventions to reduce alarm frequency. \textit{J Hosp Med}. 2016;11(2):136--144. DOI: 10.1002/jhm.2520.
\bibitem{kamran2024evaluation} Kamran F, Tjandra D, Heiler A, et al. Evaluation of sepsis prediction models before onset of treatment. \textit{NEJM AI}. 2024;1(3):AIoa2300032. DOI: 10.1056/AIoa2300032.
\bibitem{dutta2026performance} Dutta S, McMurry R, Tasi MC, et al. Performance of a sepsis prediction model across different sepsis definitions. \textit{JAMA Netw Open}. 2026;9(4):e265599. DOI: 10.1001/jamanetworkopen.2026.5599.
\bibitem{do2026continuous} Do DK, Rockenschaub P, Boie SD, et al. The Impact of Evaluation Strategy on Sepsis Prediction Model Performance Metrics in Intensive Care Data: Retrospective Cohort Study. \textit{J Med Internet Res}. 2026;28:e72083. DOI: 10.2196/72083.
\bibitem{schamoni2019leveraging} Schamoni S, Lindner HA, Schneider-Lindner V, et al. Leveraging implicit expert knowledge for non-circular machine learning in sepsis prediction. \textit{Artif Intell Med}. 2019;100:101725. DOI: 10.1016/j.artmed.2019.101725. PMID: 31607345.
\bibitem{vandervegt2022timing} van der Vegt AH, Campbell CA, Neale J, et al. Timing errors and temporal uncertainty in clinical databases---A narrative review. \textit{Front Digit Health}. 2022;4:932599. DOI: 10.3389/fdgth.2022.932599.
\bibitem{johnson2016mimiciii} Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. \textit{Sci Data}. 2016;3:160035. DOI: 10.1038/sdata.2016.35.
\bibitem{collins2024tripod} Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. \textit{BMJ}. 2024;385:e078378. DOI: 10.1136/bmj-2023-078378.
\bibitem{vonelm2007strobe} von Elm E, Altman DG, Egger M, et al. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies. \textit{Lancet}. 2007;370(9596):1453--1457. DOI: 10.1016/S0140-6736(07)61602-X.
\bibitem{gu2023mamba} Gu A, Dao T. Mamba: Linear-time sequence modeling with selective state spaces. In: \textit{First Conference on Language Modeling (COLM)}; 2024.
\bibitem{gu2022efficiently} Gu A, Goel K, Ré C. Efficiently modeling long sequences with structured state spaces. In: \textit{International Conference on Learning Representations (ICLR)}; 2022.
\end{thebibliography}"""

    assert bib_target in text, "Bib target not found"
    text = text.replace(bib_target, bib_repl)

    # 11. Table 1: Add historical cohort footnote and increase font size to 8.5pt
    t1_old = r"""\caption{\textbf{Cohort Progression and Denominator Accounting Ledger.} Denominators traced from the base test cohort through surveillance eligibility, sepsis onset timing, prediction horizons, and pre-treatment actionability. \textit{Note:} Rows 1--3b report cohort shares; rows 4--6 report window-level positive prevalence ($\pi$).}
\label{tab:table1}
\vspace{0.6em}
\fontsize{7.5pt}{9.8pt}\selectfont
\setlength{\tabcolsep}{2.5pt}
\begin{tabularx}{\textwidth}{@{}p{4.2cm} c c c c X@{}}"""

    t1_new = r"""\caption{\textbf{Cohort Progression and Denominator Accounting Ledger.} Denominators traced from the base test cohort through surveillance eligibility, sepsis onset timing, prediction horizons, and pre-treatment actionability. \textit{Note:} Rows 1--3b report cohort shares; rows 4--6 report window-level positive prevalence ($\pi$). \textit{Historical Cohort Reconciliation:} Beyond the 8,254 prospective-replay test stays (MetaVision), the broader historical cohort of 3,581 incident sepsis stays comprises 1,687 CareVue-only stays, 1,886 MetaVision-only stays, and 8 transitional stays bridging both database systems ($1,687 + 1,886 + 8 = 3,581$ stays). These categories form a mutually exclusive, exhaustive partition of all historical incident sepsis stays.}
\label{tab:table1}
\vspace{0.6em}
\fontsize{8.5pt}{11.5pt}\selectfont
\setlength{\tabcolsep}{3.5pt}
\begin{tabularx}{\textwidth}{@{}p{4.5cm} c c c c X@{}}"""

    assert t1_old in text, "Table 1 old not found"
    text = text.replace(t1_old, t1_new)

    # 12. Table 2: Remove resizebox, use compact column headers and 8.5pt font
    t2_old = r"""\fontsize{7.5pt}{10.0pt}\selectfont
\setlength{\tabcolsep}{2.5pt}
\resizebox{\textwidth}{!}{\begin{tabular}{l c c c c}
\toprule
\textbf{Model Architecture / Metric} & \textbf{Canonical Sepsis-3 ($H=6\text{h}$)} & \textbf{Hybrid $\mathcal{R}_0^{\mathrm{hybrid}}$ ($\delta=0\text{h}$)} & \textbf{Hybrid $\mathcal{R}_2^{\mathrm{hybrid}}$ ($\delta=2\text{h}$)} & \textbf{Paired $\Delta\text{AUROC}$ ($\mathcal{R}_0^{\mathrm{hybrid}} - \text{Canon}$)} \\"""

    t2_new = r"""\fontsize{8.5pt}{11.5pt}\selectfont
\setlength{\tabcolsep}{3.5pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l c c c c}
\toprule
\textbf{Model Architecture / Metric} & \textbf{\makecell{Canonical Sepsis-3\\($H=6\text{h}$)}} & \textbf{\makecell{Hybrid $\mathcal{R}_0^{\mathrm{hybrid}}$\\($\delta=0\text{h}$)}} & \textbf{\makecell{Hybrid $\mathcal{R}_2^{\mathrm{hybrid}}$\\($\delta=2\text{h}$)}} & \textbf{\makecell{Paired $\Delta\text{AUROC}$\\($\mathcal{R}_0^{\mathrm{hybrid}} - \text{Canon}$)}} \\"""

    assert t2_old in text, "Table 2 old not found"
    text = text.replace(t2_old, t2_new)

    t2_end_old = r"""\bottomrule
\end{tabular}}
\vspace{0.4em}
\raggedright
\fontsize{7.0pt}{8.5pt}\selectfont
$^*$Paired differences ($\Delta\mathrm{AUROC}$) were calculated from unrounded model predictions prior to display rounding."""

    t2_end_new = r"""\bottomrule
\end{tabular*}
\vspace{0.4em}
\raggedright
\fontsize{8.0pt}{10.0pt}\selectfont
$^*$Paired differences ($\Delta\mathrm{AUROC}$) were calculated from unrounded model predictions prior to display rounding."""

    assert t2_end_old in text, "Table 2 end old not found"
    text = text.replace(t2_end_old, t2_end_new)

    # 13. Table 3: Place in landscape, remove resizebox, use 8.5pt font
    t3_block_old = r"""\clearpage
% PAGE 21: TABLE 3
\begin{table}[p]
\begin{singlespace}
\centering
\caption{\textbf{Operational Burden-Matched Alert Frontiers Under 12h Refractory Suppression.} Evaluated under Procedure B with exposure recomputation and first-emitted-alert timing verification.}
\label{tab:table3}
\vspace{0.6em}
\fontsize{7.5pt}{9.8pt}\selectfont
\setlength{\tabcolsep}{2.5pt}
\resizebox{\textwidth}{!}{\begin{tabular}{llccccccc}"""

    t3_block_new = r"""\clearpage
\begin{landscape}
% TABLE 3
\begin{table}[p]
\begin{singlespace}
\centering
\caption{\textbf{Operational Burden-Matched Alert Frontiers Under 12h Refractory Suppression.} Evaluated under Procedure B with exposure recomputation and first-emitted-alert timing verification.}
\label{tab:table3}
\vspace{0.6em}
\fontsize{8.5pt}{11.5pt}\selectfont
\setlength{\tabcolsep}{5.0pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccccccc}"""

    assert t3_block_old in text, "Table 3 block old not found"
    text = text.replace(t3_block_old, t3_block_new)

    t3_close_old = r"""\bottomrule
\end{tabular}}
\vspace{0.4em}
\raggedright
\fontsize{7.0pt}{8.5pt}\selectfont
$^*$At zero observed detections, the empirical percentile-bootstrap interval was degenerate at zero. This does not establish zero population detection probability. Under Procedure B, exposure $E_b$ and integer alert quota $K_b = \operatorname{round}(q \cdot E_b / 100)$ are recomputed dynamically within each resample; NNE confidence intervals derive from exact unrounded reciprocal transformation ($100/\text{PPV}$). In this observed sample, horizon-correct detections and detections satisfying the two-hour pre-suspicion proxy-timing margin ($t_{\mathrm{first}, i} + 2\text{h} \le T_{\mathrm{susp}, i}$) were identical across all models and operating points.
\end{singlespace}
\end{table}"""

    t3_close_new = r"""\bottomrule
\end{tabular*}
\vspace{0.4em}
\raggedright
\fontsize{8.0pt}{10.0pt}\selectfont
$^*$At zero observed detections, the empirical percentile-bootstrap interval was degenerate at zero. This does not establish zero population detection probability. Under Procedure B, exposure $E_b$ and integer alert quota $K_b = \operatorname{round}(q \cdot E_b / 100)$ are recomputed dynamically within each resample; NNE confidence intervals derive from exact unrounded reciprocal transformation ($100/\text{PPV}$). In this observed sample, horizon-correct detections and detections satisfying the two-hour pre-suspicion proxy-timing margin ($t_{\mathrm{first}, i} + 2\text{h} \le T_{\mathrm{susp}, i}$) were identical across all models and operating points.
\end{singlespace}
\end{table}
\end{landscape}"""

    assert t3_close_old in text, "Table 3 close old not found"
    text = text.replace(t3_close_old, t3_close_new)

    # 14. Table 4: Place in landscape, remove resizebox, use 8.0pt font
    t4_block_old = r"""\clearpage
% PAGE 22: TABLE 4
\begin{table}[p]
\begin{singlespace}
\centering
\caption{\textbf{Two-Axis Subgroup and Timestamp Uncertainty Matrix.} Opportunity bounds conditional on fixed operational Sepsis-3 timestamps and surveillance coverage. \textit{Note:} Subgroup Bounds denote $[O^{\mathrm{def}}, O^{\mathrm{poss}}] / \text{Total}(N)$ within each onset stratum. For All Incident Sepsis-3, within-covered bounds ($[O^{\mathrm{def}}, O^{\mathrm{poss}}] / \text{Covered}[V=1]$) are $[150, 151]/151 = [99.3\%, 100.0\%]$ for MetaVision $T_{\mathrm{susp}}$, $[0, 47]/151 = [0.0\%, 31.1\%]$ for MetaVision Earliest Rx, and $[1, 111]/337 = [0.30\%, 32.94\%]$ for Broad Historical MIMIC-III.}
\label{tab:table4}
\vspace{0.6em}
\fontsize{7.5pt}{9.5pt}\selectfont
\setlength{\tabcolsep}{1.5pt}
\resizebox{\textwidth}{!}{\begin{tabular}{llcccccccc}"""

    t4_block_new = r"""\clearpage
\begin{landscape}
% TABLE 4
\begin{table}[p]
\begin{singlespace}
\centering
\caption{\textbf{Two-Axis Subgroup and Timestamp Uncertainty Matrix.} Opportunity bounds conditional on fixed operational Sepsis-3 timestamps and surveillance coverage. \textit{Note:} Subgroup Bounds denote $[O^{\mathrm{def}}, O^{\mathrm{poss}}] / \text{Total}(N)$ within each onset stratum. For All Incident Sepsis-3, within-covered bounds ($[O^{\mathrm{def}}, O^{\mathrm{poss}}] / \text{Covered}[V=1]$) are $[150, 151]/151 = [99.3\%, 100.0\%]$ for MetaVision $T_{\mathrm{susp}}$, $[0, 47]/151 = [0.0\%, 31.1\%]$ for MetaVision Earliest Rx, and $[1, 111]/337 = [0.30\%, 32.94\%]$ for Broad Historical MIMIC-III.}
\label{tab:table4}
\vspace{0.6em}
\fontsize{8.0pt}{10.8pt}\selectfont
\setlength{\tabcolsep}{3.5pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llcccccccc}"""

    assert t4_block_old in text, "Table 4 block old not found"
    text = text.replace(t4_block_old, t4_block_new)

    t4_close_old = r"""\bottomrule
\end{tabular}}
\end{singlespace}
\end{table}"""

    t4_close_new = r"""\bottomrule
\end{tabular*}
\end{singlespace}
\end{table}
\end{landscape}"""

    assert t4_close_old in text, "Table 4 close old not found"
    text = text.replace(t4_close_old, t4_close_new)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Successfully updated main_review_doublespaced.tex")

def update_main_camera_ready():
    path = os.path.join(LATEX_DIR, 'main.tex')
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().replace('\r\n', '\n')

    # 1. Lay summary
    lay_target = "requiring clinicians to review over 58 alerts for every true case detected."
    lay_repl = "equivalent to approximately 58 simulated alert episodes per timely detected case."
    if lay_target in text:
        text = text.replace(lay_target, lay_repl)

    # 2. Lit citations: Moor 2021 + Moor 2023 + Wang 2025
    lit_target = r"frequently reporting area under the receiver operating characteristic curve (AUROC) values between 0.80 and 0.88 ahead of consensus onset~\cite{moor2021early}. However, published ICU prediction studies differ substantially in outcome implementation, temporal alignment, and evaluation design; consequently, reported AUROCs should not be interpreted as directly comparable measures of deployment readiness~\cite{moor2021early,wang2025methodological}."
    lit_repl = r"frequently reporting area under the receiver operating characteristic curve (AUROC) values between 0.80 and 0.88 ahead of consensus onset~\cite{moor2021early}. However, published ICU prediction studies differ substantially in outcome implementation, temporal alignment, and evaluation design; consequently, reported AUROCs should not be interpreted as directly comparable measures of deployment readiness, even across advanced deep learning architectures evaluated across international health systems~\cite{moor2023predicting} or broad methodological reviews~\cite{wang2025methodological}."
    if lit_target in text:
        text = text.replace(lit_target, lit_repl)

    # Cite Schamoni 2019 and van der Vegt 2022 sequentially
    do_target = r"Do et al.~\cite{do2026continuous} further showed that continuous surveillance and alert silencing rules fundamentally alter clinical workload across health systems. This literature indicates that pre-intervention performance does not follow a uniform collapse, but depends on how onset labels, clinical-process anchors, and evaluation units are constructed."
    do_repl = r"Do et al.~\cite{do2026continuous} further showed that continuous surveillance and alert silencing rules fundamentally alter clinical workload across health systems. Addressing label contamination and circularity in sepsis definitions, Schamoni et al.~\cite{schamoni2019leveraging} emphasized that clinical recognition anchors require explicit separation from physiological deterioration. Furthermore, clinical database timestamps carry inherent recording latency, clock drift, and temporal coarsening~\cite{vandervegt2022timing}. This literature indicates that pre-intervention performance does not follow a uniform collapse, but depends on how onset labels, clinical-process anchors, and evaluation units are constructed."
    if do_target in text:
        text = text.replace(do_target, do_repl)

    # 3. Section 2.1: Historical cohort partition
    cohort_target = r"Beyond the primary MetaVision evaluation, historical sensitivity analyses included $N=3,581$ incident sepsis stays across CareVue ($1,687$), MetaVision ($1,886$), and 8 transitional stays bridging both database systems ($1,687 + 1,886 + 8 = 3,581$)."
    cohort_repl = r"Beyond the primary MetaVision evaluation, historical sensitivity analyses included $N=3,581$ incident sepsis stays partitioned into CareVue-only ($1,687$), MetaVision-only ($1,886$), and transitional stays bridging both database systems ($8$), where $1,687 + 1,886 + 8 = 3,581$ stays form a mutually exclusive, exhaustive partition."
    if cohort_target in text:
        text = text.replace(cohort_target, cohort_repl)

    # 4. Section 2.2: Baseline SOFA wording
    sofa_target = r"this identifies the first qualifying bin relative to a retrospective search-window minimum rather than an observed increase from a pre-ICU baseline."
    sofa_repl = r"this identifies the first qualifying bin relative to a retrospective search-window minimum rather than an observed increase from a pre-ICU baseline; an alternative antecedent-baseline sensitivity analysis was not performed."
    if sofa_target in text:
        text = text.replace(sofa_target, sofa_repl)

    # 5. Section 3.2: AUPRC wording
    auprc_target = r"The increase in raw AUPRC (BiMamba: 1.209\% to 1.656\%) was mechanically driven by control-window censoring, which raised baseline positive prevalence by $1.384\times$ (from $\pi=0.694\%$ to $0.960\%$)."
    auprc_repl = r"The increase in raw AUPRC (BiMamba: 1.209\% to 1.656\%) accompanied control-window censoring and higher positive prevalence (raised by $1.384\times$, from $\pi=0.694\%$ to $0.960\%$); these comparisons do not isolate improved ranking performance."
    if auprc_target in text:
        text = text.replace(auprc_target, auprc_repl)

    # 6. Section 3.4: Suspicion-Anchor Mathematics & Endpoint Error
    math_target = r"""Table~\ref{tab:table4} presents the cross-tabulation of Sepsis Onset Timing against Treatment Proxy Timing. Because MIMIC-III prescription start dates (\texttt{prescriptions.\allowbreak startdate}) lack time of day, order initiation is evaluated both under declared point assignment (midnight $a_i$) and under calendar-day intervals $[a_i, b_i) = [\text{midnight}, \text{next midnight})$. For stay $i$, let minute-resolved culture charttime be $c_i$ and the coarsened prescription interval be $[a_i, b_i)$. The transformed suspicion anchor interval is $[L_i, U_i)$, where $L_i = \min(a_i, c_i)$ and $U_i = \min(b_i, c_i)$. A stay has definite opportunity ($O_i^{\mathrm{def}} = 1$) if there exists an evaluated horizon-positive decision bin $t \in \mathcal{T}_i$ satisfying $t + \delta \le L_i$ ($\forall C \in [a_i, b_i), t + \delta \le \min(C, c_i)$); it has possible opportunity ($O_i^{\mathrm{poss}} = 1$) if there exists $t \in \mathcal{T}_i$ such that $t + \delta < U_i$ ($\exists C \in [a_i, b_i) : t + \delta \le \min(C, c_i)$). Because $b_i$ is excluded from $[a_i, b_i)$, strict inequality $t + \delta < U_i$ is required for non-empty feasible realizations. Stays with $O_i^{\mathrm{def}} = 0$ but $O_i^{\mathrm{poss}} = 1$ are indeterminate ($O_i^{\mathrm{indet}} = 1$). All indicators strictly satisfy $O_i^{\mathrm{def}} \le O_i^{\mathrm{point}} \le O_i^{\mathrm{poss}}$ and $O_i^{\mathrm{poss}} = O_i^{\mathrm{def}} + O_i^{\mathrm{indet}}$, with $O_i^{\mathrm{def}} \equiv O_i^{\mathrm{point}}$ by construction because midnight point assignment $a_i = \inf [a_i, b_i)$. Importantly, these opportunity bounds are strictly conditional on fixed operational Sepsis-3 timestamps and surveillance coverage; they do not propagate counterfactual antibiotic timings through the upstream phenotype (which could alter qualifying pairs, baseline SOFA, onset, or coverage).

Opportunity depended strongly on the selected clinical-process anchor. Under the infection-suspicion anchor ($T_{\mathrm{susp}}$), 150 of 151 covered patients (99.3\% within-covered; 36.8\% of total incident sepsis) had decision bins at or before $T_{\mathrm{susp}}$ under declared midnight point assignment $a_i$. Because $a_i \le T_{\mathrm{susp}, i}$ across $[a_i, b_i)$, these 150 stays are definitely eligible throughout the calendar-day interval ($O_i^{\mathrm{def}} = 150$), with only 1 indeterminate stay ($O_i^{\mathrm{indet}} = 1$), bounding opportunity between 36.8\% and 37.0\% ($[150, 151]/408$; $[99.3\%, 100.0\%]$ within-covered)."""

    math_repl = r"""Table~\ref{tab:table4} presents the cross-tabulation of Sepsis Onset Timing against Treatment Proxy Timing. Because MIMIC-III prescription start dates (\texttt{prescriptions.\allowbreak startdate}) lack time of day, order initiation is evaluated both under declared point assignment (midnight $a_i$) and under calendar-day intervals $[a_i, b_i) = [\text{midnight}, \text{next midnight})$. For stay $i$, let minute-resolved culture charttime be $c_i$, coarsened prescription interval be $[a_i, b_i)$, and pre-onset evaluation windows be $P_i = \{ t \in \mathcal{T}_i : 0 < T_{\mathrm{S3}, i} - t \le H \}$. Under the infection-suspicion anchor $T_{\mathrm{susp}, i} = \min(A_i, c_i)$ with unobserved $A_i \in [a_i, b_i)$, the attainable set $\mathcal{C}_i$ of $T_{\mathrm{susp}, i}$ is $\{c_i\}$ if $c_i < a_i$, $[a_i, c_i]$ if $a_i \le c_i < b_i$, and $[a_i, b_i)$ if $c_i \ge b_i$. Because $C_i = c_i$ is attained whenever $A_i \ge c_i$, the upper endpoint is closed when $a_i \le c_i < b_i$. Definite opportunity requires $t + \delta \le \min(A_i, c_i)$ for all $A_i \in [a_i, b_i)$, holding if and only if $t + \delta \le a_i$ and $t + \delta \le c_i$:
\begin{equation}
O_i^{\mathrm{def}}(\delta) = \mathbf{1} \bigl\{ \exists t \in P_i : t + \delta \le a_i \;\land\; t + \delta \le c_i \bigr\}
\end{equation}
Possible opportunity requires $t + \delta \le \min(A_i, c_i)$ for at least one $A_i \in [a_i, b_i)$, which holds if and only if $t + \delta < b_i$ and $t + \delta \le c_i$:
\begin{equation}
O_i^{\mathrm{poss}}(\delta) = \mathbf{1} \bigl\{ \exists t \in P_i : t + \delta < b_i \;\land\; t + \delta \le c_i \bigr\}
\end{equation}
For earliest prescription alone ($T_{\mathrm{Rx,start}}$):
\begin{equation}
O_i^{\mathrm{def}}(\delta) = \mathbf{1} \bigl\{ \exists t \in P_i : t + \delta \le a_i \bigr\}, \quad O_i^{\mathrm{poss}}(\delta) = \mathbf{1} \bigl\{ \exists t \in P_i : t + \delta < b_i \bigr\}
\end{equation}
Stays with $O_i^{\mathrm{def}} = 0$ but $O_i^{\mathrm{poss}} = 1$ are indeterminate ($O_i^{\mathrm{indet}} = 1$). All indicators strictly satisfy $O_i^{\mathrm{def}} \le O_i^{\mathrm{point}} \le O_i^{\mathrm{poss}}$ and $O_i^{\mathrm{poss}} = O_i^{\mathrm{def}} + O_i^{\mathrm{indet}}$, with $O_i^{\mathrm{def}} \equiv O_i^{\mathrm{point}}$ by construction because midnight point assignment $a_i = \inf [a_i, b_i)$. Importantly, these opportunity bounds are strictly conditional on fixed operational Sepsis-3 timestamps and surveillance coverage; they do not propagate counterfactual antibiotic timings through the upstream phenotype (which could alter qualifying pairs, baseline SOFA, onset, or coverage).

Opportunity depended strongly on the selected clinical-process anchor. Under the infection-suspicion anchor ($T_{\mathrm{susp}}$), 150 of 151 covered stays (99.3\% within-covered; 36.8\% of total incident sepsis) had decision bins satisfying the definite opportunity condition under declared midnight point assignment $a_i$. Across the calendar-day interval, exactly 150 stays satisfy $O_i^{\mathrm{def}} = 1$, with only 1 indeterminate stay ($O_i^{\mathrm{indet}} = 1$), bounding opportunity between 36.8\% and 37.0\% ($[150, 151]/408$; $[99.3\%, 100.0\%]$ within-covered)."""

    if math_target in text:
        text = text.replace(math_target, math_repl)

    # 7. Section 3.5: Oracle paragraph wording
    oracle_target = r"Low empirical detection yield is thus driven by model discrimination and alert suppression, not a mathematical ceiling of the budget."
    oracle_repl = r"The low observed yield reflects performance of the tested score-and-first-alert policies within the restricted surveillance protocol, rather than a budget-imposed ceiling near 2\%."
    if oracle_target in text:
        text = text.replace(oracle_target, oracle_repl)

    # 8. Table 1: Note reconciliation
    t1_old = r"""\caption{\textbf{Cohort Progression and Denominator Accounting Ledger.} Denominators traced from the base test cohort through surveillance eligibility, sepsis onset timing, prediction horizons, and pre-treatment actionability. \textit{Note:} Rows 1--3b report cohort shares; rows 4--6 report window-level positive prevalence ($\pi$).}"""
    t1_new = r"""\caption{\textbf{Cohort Progression and Denominator Accounting Ledger.} Denominators traced from the base test cohort through surveillance eligibility, sepsis onset timing, prediction horizons, and pre-treatment actionability. \textit{Note:} Rows 1--3b report cohort shares; rows 4--6 report window-level positive prevalence ($\pi$). Historical cohort ($N=3,581$) comprises CareVue-only (1,687), MetaVision-only (1,886), and 8 transitional stays ($1,687 + 1,886 + 8 = 3,581$).}"""
    if t1_old in text:
        text = text.replace(t1_old, t1_new)

    # 9. Bibliography: 26 references
    bib_target = r"""\begin{thebibliography}{25}
\fontsize{4.1pt}{4.6pt}\selectfont
\setlength{\itemsep}{0pt}
\setlength{\parsep}{0pt}
\setlength{\parskip}{0pt}
\bibitem{singer2016sepsis} Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). \textit{JAMA}. 2016;315(8):801--810. DOI: 10.1001/jama.2016.0287.
\bibitem{seymour2016assessment} Seymour CW, Liu VX, Iwashyna TJ, et al. Assessment of clinical criteria for sepsis: for the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). \textit{JAMA}. 2016;315(8):762--774. DOI: 10.1001/jama.2016.0288.
\bibitem{evans2021surviving} Evans L, Rhodes A, Alhazzani W, et al. Surviving sepsis campaign: international guidelines for management of sepsis and septic shock 2021. \textit{Intensive Care Med}. 2021;47(11):1181--1247. DOI: 10.1007/s00134-021-06506-y.
\bibitem{kumar2006duration} Kumar A, Roberts D, Wood KE, et al. Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock. \textit{Crit Care Med}. 2006;34(6):1589--1596. DOI: 10.1097/01.CCM.0000217961.75225.E9.
\bibitem{moor2021early} Moor M, Bennett N, Plecko D, et al. Predicting sepsis using deep learning across international sites: a retrospective development and validation study. \textit{EClinicalMedicine}. 2023;62:102100. DOI: 10.1016/j.eclinm.2023.102100.
\bibitem{wang2025methodological} Wang Z, Wang W, Sun C, et al. A methodological systematic review of validation and performance of sepsis real-time prediction models. \textit{npj Digit Med}. 2025;8:190. DOI: 10.1038/s41746-025-01587-1.
\bibitem{cohen2024subtle} Cohen SN, Foster J, Foster P, et al. Subtle variation in sepsis-III definitions markedly influences predictive performance within and across methods. \textit{Sci Rep}. 2024;14:1920. DOI: 10.1038/s41598-024-51989-6.
\bibitem{reyna2020early} Reyna MA, Josef CS, Jeter R, et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. \textit{Crit Care Med}. 2020;48(2):210--217. DOI: 10.1097/CCM.0000000000004145.
\bibitem{wong2021external} Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. \textit{JAMA Intern Med}. 2021;181(8):1065--1070. DOI: 10.1001/jamainternmed.2021.2626.
\bibitem{adams2022prospective} Adams R, Henry KE, Sridharan A, et al. Prospective, multi-site study of patient outcomes after implementation of the TREWS machine learning-based early warning system for sepsis. \textit{Nat Med}. 2022;28(7):1455--1460. DOI: 10.1038/s41591-022-01894-0.
\bibitem{weissman2024sepsis} Weissman GE, Hubbard RA, Himes BE, et al. Sepsis prediction models are trained on labels that diverge from clinician-recommended treatment times. \textit{AMIA Annu Symp Proc}. 2024;2024:1215--1224.
\bibitem{henry2022factors} Henry KE, Adams R, Parent C, et al. Factors driving provider adoption of the TREWS machine learning-based early warning system and its effects on sepsis treatment timing. \textit{Nat Med}. 2022;28(7):1447--1454. DOI: 10.1038/s41591-022-01895-z.
\bibitem{drew2014insights} Drew BJ, Harris P, Zègre-Hemsey JK, et al. Insights into the problem of alarm fatigue with physiologic monitor devices: a comprehensive observational study of consecutive intensive care unit patients. \textit{PLoS One}. 2014;9(10):e110274. DOI: 10.1371/journal.pone.0110274.
\bibitem{ancker2017effects} Ancker JS, Edwards A, Nosal S, et al. Effects of workload, work complexity, and repeated alerts on alert fatigue in a clinical decision support system. \textit{BMC Med Inform Decis Mak}. 2017;17(1):36. DOI: 10.1186/s12911-017-0430-8.
\bibitem{paine2016systematic} Paine CW, Goel VV, Ely E, et al. Systematic review of physiologic monitor alarm characteristics and pragmatic interventions to reduce alarm frequency. \textit{J Hosp Med}. 2016;11(2):136--144. DOI: 10.1002/jhm.2520.
\bibitem{kamran2024evaluation} Kamran F, Tjandra D, Heiler A, et al. Evaluation of sepsis prediction models before onset of treatment. \textit{NEJM AI}. 2024;1(3):AIoa2300032. DOI: 10.1056/AIoa2300032.
\bibitem{dutta2026performance} Dutta S, McMurry R, Tasi MC, et al. Performance of a sepsis prediction model across different sepsis definitions. \textit{JAMA Netw Open}. 2026;9(4):e265599. DOI: 10.1001/jamanetworkopen.2026.5599.
\bibitem{do2026continuous} Do DK, Rockenschaub P, Boie SD, et al. The Impact of Evaluation Strategy on Sepsis Prediction Model Performance Metrics in Intensive Care Data: Retrospective Cohort Study. \textit{J Med Internet Res}. 2026;28:e72083. DOI: 10.2196/72083.
\bibitem{schamoni2019leveraging} Schamoni S, Lindner HA, Schneider-Lindner V, et al. Leveraging continuous physiological parameters for non-circular sepsis prediction. \textit{PLoS One}. 2019;14(10):e0223548. DOI: 10.1371/journal.pone.0223548.
\bibitem{goyal2022characterizing} Goyal R, et al. Characterizing timestamp discrepancies and data latency in electronic health records for clinical machine learning. \textit{J Am Med Inform Assoc}. 2022;29(5):890--899. DOI: 10.1093/jamia/ocab289.
\bibitem{johnson2016mimiciii} Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. \textit{Sci Data}. 2016;3:160035. DOI: 10.1038/sdata.2016.35.
\bibitem{collins2024tripod} Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. \textit{BMJ}. 2024;385:e078378. DOI: 10.1136/bmj-2023-078378.
\bibitem{vonelm2007strobe} von Elm E, Altman DG, Egger M, et al. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies. \textit{Lancet}. 2007;370(9596):1453--1457. DOI: 10.1016/S0140-6736(07)61602-X.
\bibitem{gu2023mamba} Gu A, Dao T. Mamba: Linear-time sequence modeling with selective state spaces. In: \textit{First Conference on Language Modeling (COLM)}; 2024.
\bibitem{gu2022efficiently} Gu A, Goel K, Ré C. Efficiently modeling long sequences with structured state spaces. In: \textit{International Conference on Learning Representations (ICLR)}; 2022.
\end{thebibliography}"""

    bib_repl = r"""\begin{thebibliography}{26}
\fontsize{4.0pt}{4.5pt}\selectfont
\setlength{\itemsep}{0pt}
\setlength{\parsep}{0pt}
\setlength{\parskip}{0pt}
\bibitem{singer2016sepsis} Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). \textit{JAMA}. 2016;315(8):801--810. DOI: 10.1001/jama.2016.0287.
\bibitem{seymour2016assessment} Seymour CW, Liu VX, Iwashyna TJ, et al. Assessment of clinical criteria for sepsis: for the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). \textit{JAMA}. 2016;315(8):762--774. DOI: 10.1001/jama.2016.0288.
\bibitem{evans2021surviving} Evans L, Rhodes A, Alhazzani W, et al. Surviving sepsis campaign: international guidelines for management of sepsis and septic shock 2021. \textit{Intensive Care Med}. 2021;47(11):1181--1247. DOI: 10.1007/s00134-021-06506-y.
\bibitem{kumar2006duration} Kumar A, Roberts D, Wood KE, et al. Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock. \textit{Crit Care Med}. 2006;34(6):1589--1596. DOI: 10.1097/01.CCM.0000217961.75225.E9.
\bibitem{moor2021early} Moor M, Rieck B, Horn M, et al. Early prediction of sepsis in the ICU using machine learning: a systematic review. \textit{Front Med (Lausanne)}. 2021;8:607952. DOI: 10.3389/fmed.2021.607952.
\bibitem{moor2023predicting} Moor M, Bennett N, Plecko D, et al. Predicting sepsis using deep learning across international sites: a retrospective development and validation study. \textit{EClinicalMedicine}. 2023;62:102124. DOI: 10.1016/j.eclinm.2023.102124.
\bibitem{wang2025methodological} Wang Z, Wang W, Sun C, et al. A methodological systematic review of validation and performance of sepsis real-time prediction models. \textit{npj Digit Med}. 2025;8:190. DOI: 10.1038/s41746-025-01587-1.
\bibitem{cohen2024subtle} Cohen SN, Foster J, Foster P, et al. Subtle variation in sepsis-III definitions markedly influences predictive performance within and across methods. \textit{Sci Rep}. 2024;14:1920. DOI: 10.1038/s41598-024-51989-6.
\bibitem{reyna2020early} Reyna MA, Josef CS, Jeter R, et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. \textit{Crit Care Med}. 2020;48(2):210--217. DOI: 10.1097/CCM.0000000000004145.
\bibitem{wong2021external} Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. \textit{JAMA Intern Med}. 2021;181(8):1065--1070. DOI: 10.1001/jamainternmed.2021.2626.
\bibitem{adams2022prospective} Adams R, Henry KE, Sridharan A, et al. Prospective, multi-site study of patient outcomes after implementation of the TREWS machine learning-based early warning system for sepsis. \textit{Nat Med}. 2022;28(7):1455--1460. DOI: 10.1038/s41591-022-01894-0.
\bibitem{weissman2024sepsis} Weissman GE, Hubbard RA, Himes BE, et al. Sepsis prediction models are trained on labels that diverge from clinician-recommended treatment times. \textit{AMIA Annu Symp Proc}. 2024;2024:1215--1224.
\bibitem{henry2022factors} Henry KE, Adams R, Parent C, et al. Factors driving provider adoption of the TREWS machine learning-based early warning system and its effects on sepsis treatment timing. \textit{Nat Med}. 2022;28(7):1447--1454. DOI: 10.1038/s41591-022-01895-z.
\bibitem{drew2014insights} Drew BJ, Harris P, Zègre-Hemsey JK, et al. Insights into the problem of alarm fatigue with physiologic monitor devices: a comprehensive observational study of consecutive intensive care unit patients. \textit{PLoS One}. 2014;9(10):e110274. DOI: 10.1371/journal.pone.0110274.
\bibitem{ancker2017effects} Ancker JS, Edwards A, Nosal S, et al. Effects of workload, work complexity, and repeated alerts on alert fatigue in a clinical decision support system. \textit{BMC Med Inform Decis Mak}. 2017;17(1):36. DOI: 10.1186/s12911-017-0430-8.
\bibitem{paine2016systematic} Paine CW, Goel VV, Ely E, et al. Systematic review of physiologic monitor alarm characteristics and pragmatic interventions to reduce alarm frequency. \textit{J Hosp Med}. 2016;11(2):136--144. DOI: 10.1002/jhm.2520.
\bibitem{kamran2024evaluation} Kamran F, Tjandra D, Heiler A, et al. Evaluation of sepsis prediction models before onset of treatment. \textit{NEJM AI}. 2024;1(3):AIoa2300032. DOI: 10.1056/AIoa2300032.
\bibitem{dutta2026performance} Dutta S, McMurry R, Tasi MC, et al. Performance of a sepsis prediction model across different sepsis definitions. \textit{JAMA Netw Open}. 2026;9(4):e265599. DOI: 10.1001/jamanetworkopen.2026.5599.
\bibitem{do2026continuous} Do DK, Rockenschaub P, Boie SD, et al. The Impact of Evaluation Strategy on Sepsis Prediction Model Performance Metrics in Intensive Care Data: Retrospective Cohort Study. \textit{J Med Internet Res}. 2026;28:e72083. DOI: 10.2196/72083.
\bibitem{schamoni2019leveraging} Schamoni S, Lindner HA, Schneider-Lindner V, et al. Leveraging implicit expert knowledge for non-circular machine learning in sepsis prediction. \textit{Artif Intell Med}. 2019;100:101725. DOI: 10.1016/j.artmed.2019.101725. PMID: 31607345.
\bibitem{vandervegt2022timing} van der Vegt AH, Campbell CA, Neale J, et al. Timing errors and temporal uncertainty in clinical databases---A narrative review. \textit{Front Digit Health}. 2022;4:932599. DOI: 10.3389/fdgth.2022.932599.
\bibitem{johnson2016mimiciii} Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. \textit{Sci Data}. 2016;3:160035. DOI: 10.1038/sdata.2016.35.
\bibitem{collins2024tripod} Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. \textit{BMJ}. 2024;385:e078378. DOI: 10.1136/bmj-2023-078378.
\bibitem{vonelm2007strobe} von Elm E, Altman DG, Egger M, et al. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies. \textit{Lancet}. 2007;370(9596):1453--1457. DOI: 10.1016/S0140-6736(07)61602-X.
\bibitem{gu2023mamba} Gu A, Dao T. Mamba: Linear-time sequence modeling with selective state spaces. In: \textit{First Conference on Language Modeling (COLM)}; 2024.
\bibitem{gu2022efficiently} Gu A, Goel K, Ré C. Efficiently modeling long sequences with structured state spaces. In: \textit{International Conference on Learning Representations (ICLR)}; 2022.
\end{thebibliography}"""

    if bib_target in text:
        text = text.replace(bib_target, bib_repl)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Successfully updated main.tex")

if __name__ == '__main__':
    update_review_manuscript()
    update_main_camera_ready()
