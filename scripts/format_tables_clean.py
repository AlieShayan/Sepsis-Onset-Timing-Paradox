#!/usr/bin/env python3
"""
format_tables_clean.py
Re-formats Tables 1, 2, 3, 4 in main_review_doublespaced.tex:
- Table 1: Portrait, concise caption, footnote below, 8.0pt/10.5pt font, no float warning
- Table 2: Landscape, 8.5pt/12pt font, no overfull hbox
- Table 3: Landscape, 8.2pt/11pt font, no overfull hbox
- Table 4: Landscape, 7.8pt/10.2pt font, compact headers, no overfull hbox
"""

import os

path = '/home/alireza/Sepsis-Onset-Timing-Paradox/manuscript/latex/main_review_doublespaced.tex'

with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace from % PAGE 19: TABLE 1 to \end{document}
table_start_marker = r'\clearpage' + '\n' + r'% PAGE 19: TABLE 1'
assert table_start_marker in text, "Table start marker not found"

prefix = text[:text.find(table_start_marker)]

new_tables = r"""\clearpage
% TABLE 1 (PORTRAIT)
\begin{table}[p]
\begin{singlespace}
\centering
\caption{\textbf{Cohort Progression and Denominator Accounting Ledger.} Denominators traced from the base test cohort through surveillance eligibility, sepsis onset timing, prediction horizons, and pre-treatment actionability.}
\label{tab:table1}
\vspace{0.4em}
\fontsize{8.0pt}{10.2pt}\selectfont
\setlength{\tabcolsep}{2.5pt}
\begin{tabularx}{\textwidth}{@{}p{3.8cm} c c c c X@{}}
\toprule
\textbf{Progression Stage / Cohort Partition} & \textbf{Stays ($N$)} & \textbf{Evaluated Windows} & \textbf{Surveillance Volume} & \textbf{Prev. / Share} & \textbf{Clinical Definition \& Measurement Convention} \\
\midrule
1. Base Prospective-Replay Test Cohort & 8,254 & 97,700 / 99,048 sched. & 4,070.8 pt-days & 100.0\% & Adult first ICU stays $\ge 30\text{h}$; 1,348 censored windows (623 onset, 725 cessation) \\
2. Non-Sepsis Control ICU Stays & 7,846 & 93,427 neg. windows & 3,892.8 pt-days & 95.06\% & Stays without operational Sepsis-3 during hospitalization \\
3. Operational Sepsis-3 Stays ($S=1$) & 408 & 4,273 (678 pos, 3,595 neg) & 178.0 pt-days & 4.94\% & Blood culture + antibiotic + acute $\Delta\text{SOFA}\ge 2$; full cohort median onset 49.0h \\
\quad 3a. Late-Onset Sepsis (Onset $>36\text{h}$) & 257 & 3,084 neg. windows & 128.5 pt-days & 63.0\% of septic & Onset $>36\text{h}$ (median: 69.0h); anticipation horizon $[T_{\text{S3}}-6, T_{\text{S3}}]$ outside [19, 30] ($V=0$) \\
\quad 3b. Surveillance-Covered ($V=1$) & 151 & 1,189 (678 pos, 511 neg) & 49.5 pt-days & 37.0\% of septic & Onset in [20h, 36h] (median: 28.0h); anticipation horizon intersects window ($V=1$, earliest uncensored integer onset bin is 20h) \\
4. Canonical Sepsis-3 Risk Set ($H=6\text{h}$) & 8,254 & 97,700 (678 pos / 97,022 neg) & 4,070.8 pt-days & 0.694\% & Full unconstrained risk set ($t \in \mathcal{T}_i, \; t < T_{\text{S3}, i}$) \\
5. Outcome-Stratified Hybrid Set $\mathcal{R}_0^{\mathrm{hybrid}}$ ($\delta=0\text{h}$) & 5,967 & 70,192 (674 pos / 69,518 neg) & 2,924.7 pt-days & 0.960\% & Censoring post-anchor ($t \le C_i$; $T_{\mathrm{susp}}$ cases, $T_{\mathrm{Rx,start}}$ controls); 150/151 covered eligible \\
6. Outcome-Stratified Hybrid Set $\mathcal{R}_2^{\mathrm{hybrid}}$ ($\delta=2\text{h}$) & 5,967 & 70,129 (663 pos / 69,466 neg) & 2,922.0 pt-days & 0.945\% & 2-hour proxy-timing margin ($t + 2\text{h} \le C_i$); 149/151 covered eligible \\
\bottomrule
\end{tabularx}
\vspace{0.4em}
\raggedright
\fontsize{7.5pt}{9.2pt}\selectfont
\textit{Note:} Rows 1--3b report cohort shares; rows 4--6 report window-level positive prevalence ($\pi$). \textit{Historical Cohort Reconciliation:} Beyond the 8,254 prospective-replay test stays (MetaVision), the broader historical cohort of 3,581 incident sepsis stays comprises 1,687 CareVue-only stays, 1,886 MetaVision-only stays, and 8 transitional stays bridging both database systems ($1,687 + 1,886 + 8 = 3,581$ stays), forming a mutually exclusive, exhaustive partition of all historical incident sepsis stays.
\end{singlespace}
\end{table}

\clearpage
% TABLE 2 (LANDSCAPE)
\begin{landscape}
\begin{table}[p]
\begin{singlespace}
\centering
\caption{\textbf{Benchmark Discrimination Across Outcome-Stratified Hybrid Risk Sets (Held-Out Point Estimates and Clustered 95\% CIs).} Point estimates calculated directly on held-out test data ($B=0$); 95\% CIs derived from $B=1,000$ patient-clustered bootstrap resamples. Evaluated on hybrid risk sets $\mathcal{R}_0^{\mathrm{hybrid}}$ and $\mathcal{R}_2^{\mathrm{hybrid}}$ censoring cases at infection-suspicion anchor $T_{\mathrm{susp}}$ and controls at earliest prescription $T_{\mathrm{Rx,start}}$. Sepsis cases and controls were restricted using different clinical-process anchors; differences from canonical discrimination reflect both evaluated observations and outcome-specific selection, and do not represent a uniform prospective eligibility policy.}
\label{tab:table2}
\vspace{0.6em}
\fontsize{8.5pt}{11.8pt}\selectfont
\setlength{\tabcolsep}{5.0pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l c c c c}
\toprule
\textbf{Model Architecture / Metric} & \textbf{\makecell{Canonical Sepsis-3\\($H=6\text{h}$)}} & \textbf{\makecell{Hybrid $\mathcal{R}_0^{\mathrm{hybrid}}$\\($\delta=0\text{h}$)}} & \textbf{\makecell{Hybrid $\mathcal{R}_2^{\mathrm{hybrid}}$\\($\delta=2\text{h}$)}} & \textbf{\makecell{Paired $\Delta\text{AUROC}$\\($\mathcal{R}_0^{\mathrm{hybrid}} - \text{Canon}$)}} \\
\midrule
\textbf{BiMamba SSM} & 0.6326 (0.5872--0.6733) & \textbf{0.6433} (0.5998--0.6835) & \textbf{0.6447} (0.6002--0.6860) & \textbf{+0.0107} (+0.0060 to +0.0175) \\
\quad \textit{AUPRC} & \textit{1.209\% (0.892--1.769)} & \textbf{\textit{1.656\%}} \textit{(1.235--2.329)} & \textbf{\textit{1.647\%}} \textit{(1.220--2.331)} & --- \\[0.25em]
\textbf{Multi-Model Stacking Ensemble} & 0.6284 (0.5829--0.6699) & 0.6351 (0.5897--0.6771) & 0.6370 (0.5913--0.6799) & +0.0067 (+0.0021 to +0.0137) \\
\quad \textit{AUPRC} & \textit{1.206\% (0.882--1.748)} & \textit{1.621\% (1.207--2.263)} & \textit{1.613\% (1.198--2.255)} & --- \\[0.25em]
\textbf{Random Forest} & 0.6108 (0.5643--0.6535) & 0.6143 (0.5684--0.6575) & 0.6173 (0.5712--0.6612) & +0.0034 (-0.0008 to +0.0101) \\
\quad \textit{AUPRC} & \textit{1.080\% (0.821--1.434)} & \textit{1.470\% (1.138--1.920)} & \textit{1.461\% (1.120--1.916)} & --- \\[0.25em]
\textbf{Mask-Aware Clinical Transformer} & 0.5942 (0.5467--0.6410) & 0.5961 (0.5490--0.6422) & 0.5954 (0.5480--0.6426) & +0.0020 (-0.0024 to +0.0083) \\
\quad \textit{AUPRC} & \textit{1.013\% (0.766--1.303)} & \textit{1.382\% (1.048--1.760)} & \textit{1.365\% (1.032--1.752)} & --- \\[0.25em]
\textbf{Tabular GBDT (XGBoost)} & 0.5670 (0.5208--0.6102) & 0.5643 (0.5186--0.6082) & 0.5652 (0.5192--0.6099) & -0.0027 (-0.0070 to +0.0038) \\
\quad \textit{AUPRC} & \textit{0.977\% (0.739--1.357)} & \textit{1.320\% (1.008--1.817)} & \textit{1.313\% (0.998--1.811)} & --- \\[0.25em]
\textbf{Tabular GBDT (LightGBM)} & 0.5613 (0.5161--0.6026) & 0.5587 (0.5139--0.6010) & 0.5598 (0.5145--0.6033) & -0.0026 (-0.0070 to +0.0038) \\
\quad \textit{AUPRC} & \textit{0.893\% (0.694--1.195)} & \textit{1.222\% (0.957--1.629)} & \textit{1.214\% (0.949--1.630)} & --- \\[0.25em]
\textbf{Logistic Regression (L2)} & 0.5593 (0.5097--0.6048) & 0.5654 (0.5157--0.6107) & 0.5657 (0.5150--0.6127) & +0.0061 (+0.0023 to +0.0116) \\
\quad \textit{AUPRC} & \textit{0.939\% (0.716--1.218)} & \textit{1.337\% (1.012--1.740)} & \textit{1.325\% (1.002--1.731)} & --- \\[0.25em]
\textbf{Clinical Transformer (Baseline)} & 0.5239 (0.4808--0.5653) & 0.5296 (0.4851--0.5712) & 0.5286 (0.4834--0.5705) & +0.0057 (+0.0019 to +0.0110) \\
\quad \textit{AUPRC} & \textit{0.808\% (0.609--1.381)} & \textit{1.147\% (0.857--1.904)} & \textit{1.131\% (0.845--1.894)} & --- \\
\bottomrule
\end{tabular*}
\vspace{0.4em}
\raggedright
\fontsize{7.8pt}{9.8pt}\selectfont
$^*$Paired differences ($\Delta\mathrm{AUROC}$) were calculated from unrounded model predictions prior to display rounding.
\end{singlespace}
\end{table}
\end{landscape}

\clearpage
% TABLE 3 (LANDSCAPE)
\begin{landscape}
\begin{table}[p]
\begin{singlespace}
\centering
\caption{\textbf{Operational Burden-Matched Alert Frontiers Under 12h Refractory Suppression.} Evaluated under Procedure B with exposure recomputation and first-emitted-alert timing verification.}
\label{tab:table3}
\vspace{0.6em}
\fontsize{8.2pt}{11.2pt}\selectfont
\setlength{\tabcolsep}{4.0pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccccccc}
\toprule
\textbf{Budget} & \textbf{Model Architecture} & \textbf{\makecell{Cutoff\\($\tau$)}} & \textbf{\makecell{Alerts\\($K$)}} & \textbf{\makecell{Episode PPV \%\\{[95\% CI]}}} & \textbf{\makecell{Det\\($N$)}} & \textbf{\makecell{Sens. Cov \%\\{[95\% CI]}}} & \textbf{\makecell{Yield \%\\{[95\% CI]}}} & \textbf{\makecell{NNE\\{[95\% CI]}}} \\
\midrule
\textbf{5.0 ep / 100 pt-d} & Stacking Ensemble & 0.0145 & 204 & 2.94 (0.98--5.88)\% & 6 & 3.97 (1.31--7.65)\% & 1.47 (0.49--2.86)\% & 34.0 (17.0--102.0) \\
 & LightGBM & 0.0117 & 204 & 2.94 (0.98--5.39)\% & 6 & 3.97 (1.26--7.10)\% & 1.47 (0.47--2.61)\% & 34.0 (18.5--102.0) \\
 & BiMamba SSM & 0.0187 & 204 & 1.96 (0.49--4.45)\% & 4 & 2.65 (0.65--6.20)\% & 0.98 (0.24--2.31)\% & 51.0 (22.5--204.0) \\
 & XGBoost & 0.0122 & 204 & 1.96 (0.49--3.94)\% & 4 & 2.65 (0.63--5.63)\% & 0.98 (0.23--2.09)\% & 51.0 (25.4--204.0) \\
 & Random Forest & 0.0177 & 204 & 1.47 (0.00--3.45)\% & 3 & 1.99 (0.00--4.49)\% & 0.74 (0.00--1.70)\% & 68.0 (29.0--$\infty$) \\
 & Clinical Transformer & 0.0101 & 204 & 0.98 (0.00--2.46)\% & 2 & 1.32 (0.00--3.50)\% & 0.49 (0.00--1.27)\% & 102.0 (40.6--$\infty$) \\
 & Logistic Regression & 0.0103 & 204 & 0.98 (0.00--2.94)\% & 2 & 1.32 (0.00--3.85)\% & 0.49 (0.00--1.45)\% & 102.0 (34.0--$\infty$) \\
 & Mask-Aware Transformer & 0.0120 & 204 & 0.00 (0.00--0.00)\%$^*$ & 0 & 0.00 (0.00--0.00)\%$^*$ & 0.00 (0.00--0.00)\%$^*$ & Undefined ($\infty$) \\
\midrule
\textbf{10.0 ep / 100 pt-d} & BiMamba SSM & 0.0165 & 407 & 1.72 (0.49--2.95)\% & 7 & 4.64 (1.33--8.16)\% & 1.72 (0.49--2.93)\% & 58.1 (33.9--203.5) \\
 & LightGBM & 0.0110 & 407 & 1.72 (0.49--2.95)\% & 7 & 4.64 (1.38--7.74)\% & 1.72 (0.49--2.89)\% & 58.1 (33.9--203.5) \\
 & Stacking Ensemble & 0.0131 & 407 & 1.47 (0.25--2.46)\% & 6 & 3.97 (0.70--6.98)\% & 1.47 (0.26--2.54)\% & 67.8 (40.7--407.0) \\
 & XGBoost & 0.0114 & 407 & 1.47 (0.49--2.94)\% & 6 & 3.97 (1.32--7.65)\% & 1.47 (0.49--2.85)\% & 67.8 (34.0--203.5) \\
 & Random Forest & 0.0158 & 407 & 1.47 (0.49--2.70)\% & 6 & 3.97 (1.26--7.29)\% & 1.47 (0.47--2.70)\% & 67.8 (37.0--204.0) \\
 & Mask-Aware Transformer & 0.0110 & 407 & 1.23 (0.25--2.46)\% & 5 & 3.31 (0.74--6.67)\% & 1.23 (0.26--2.48)\% & 81.4 (40.7--407.0) \\
 & Clinical Transformer & 0.0094 & 407 & 1.23 (0.25--2.46)\% & 5 & 3.31 (0.68--6.38)\% & 1.23 (0.25--2.37)\% & 81.4 (40.7--407.0) \\
 & Logistic Regression & 0.0095 & 407 & 0.98 (0.25--1.97)\% & 4 & 2.65 (0.59--5.51)\% & 0.98 (0.23--2.07)\% & 101.8 (50.8--408.0) \\
\midrule
\textbf{20.0 ep / 100 pt-d} & XGBoost & 0.0105 & 814 & 1.84 (0.98--2.70)\% & 15 & 9.93 (5.26--14.47)\% & 3.68 (1.87--5.29)\% & 54.3 (37.0--101.9) \\
 & Mask-Aware Transformer & 0.0099 & 814 & 1.72 (0.74--2.58)\% & 14 & 9.27 (4.54--13.77)\% & 3.43 (1.59--5.19)\% & 58.1 (38.8--135.7) \\
 & Logistic Regression & 0.0088 & 814 & 1.35 (0.49--2.09)\% & 11 & 7.28 (2.74--11.11)\% & 2.70 (0.99--4.18)\% & 74.0 (47.8--203.7) \\
 & BiMamba SSM & 0.0142 & 814 & 1.23 (0.49--2.09)\% & 10 & 6.62 (3.07--10.89)\% & 2.45 (1.05--4.10)\% & 81.4 (47.9--203.2) \\
 & LightGBM & 0.0105 & 814 & 1.23 (0.61--2.09)\% & 10 & 6.62 (3.11--11.04)\% & 2.45 (1.15--4.10)\% & 81.4 (47.9--163.0) \\
 & Random Forest & 0.0131 & 814 & 1.11 (0.49--1.85)\% & 9 & 5.96 (2.70--9.68)\% & 2.21 (0.96--3.69)\% & 90.4 (54.2--203.7) \\
 & Stacking Ensemble & 0.0115 & 814 & 0.98 (0.37--1.72)\% & 8 & 5.30 (2.04--9.15)\% & 1.96 (0.74--3.43)\% & 101.8 (58.1--271.3) \\
 & Clinical Transformer & 0.0088 & 814 & 0.86 (0.25--1.48)\% & 7 & 4.64 (1.51--8.18)\% & 1.72 (0.52--3.04)\% & 116.3 (67.8--407.0) \\
\bottomrule
\end{tabular*}
\vspace{0.4em}
\raggedright
\fontsize{7.5pt}{9.5pt}\selectfont
$^*$At zero observed detections, the empirical percentile-bootstrap interval was degenerate at zero. This does not establish zero population detection probability. Under Procedure B, exposure $E_b$ and integer alert quota $K_b = \operatorname{round}(q \cdot E_b / 100)$ are recomputed dynamically within each resample; NNE confidence intervals derive from exact unrounded reciprocal transformation ($100/\text{PPV}$). In this observed sample, horizon-correct detections and detections satisfying the two-hour pre-suspicion proxy-timing margin ($t_{\mathrm{first}, i} + 2\text{h} \le T_{\mathrm{susp}, i}$) were identical across all models and operating points.
\end{singlespace}
\end{table}
\end{landscape}

\clearpage
% TABLE 4 (LANDSCAPE)
\begin{landscape}
\begin{table}[p]
\begin{singlespace}
\centering
\caption{\textbf{Two-Axis Subgroup and Timestamp Uncertainty Matrix.} Opportunity bounds conditional on fixed operational Sepsis-3 timestamps and surveillance coverage. \textit{Note:} Subgroup Bounds denote $[O^{\mathrm{def}}, O^{\mathrm{poss}}] / \text{Total}(N)$ within each onset stratum. For All Incident Sepsis-3, within-covered bounds ($[O^{\mathrm{def}}, O^{\mathrm{poss}}] / \text{Covered}[V=1]$) are $[150, 151]/151 = [99.3\%, 100.0\%]$ for MetaVision $T_{\mathrm{susp}}$, $[0, 47]/151 = [0.0\%, 31.1\%]$ for MetaVision Earliest Rx, and $[1, 111]/337 = [0.30\%, 32.94\%]$ for Broad Historical MIMIC-III.}
\label{tab:table4}
\vspace{0.6em}
\fontsize{7.8pt}{10.2pt}\selectfont
\setlength{\tabcolsep}{3.0pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llcccccccc}
\toprule
\textbf{\makecell{Cohort \& Timestamp\\Source}} & \textbf{\makecell{Sepsis Onset\\Category}} & \textbf{\makecell{Total\\($N$)}} & \textbf{\makecell{Covered\\($V=1$)}} & \textbf{\makecell{Def. Elig\\($O^{\mathrm{def}}$)}} & \textbf{\makecell{Point\\($O^{\mathrm{point}}$)}} & \textbf{\makecell{Indet\\($O^{\mathrm{indet}}$)}} & \textbf{\makecell{Poss\\($O^{\mathrm{poss}}$)}} & \textbf{\makecell{Subgroup\\Bounds}} & \textbf{\makecell{Contribution\\to Cohort}} \\
\midrule
\textbf{MetaVision Prospective-Replay} & Early Incident ($\le 24\text{h}$) & 47 & 47 (100.0\%) & 47 & 47 & 0 & 47 & [100.0\%, 100.0\%] & [47, 47] / 408 (11.5\%) \\
\textit{(Primary Benchmark: $T_{\mathrm{susp}}$)} & Intermediate (24--48h) & 156 & 104 (66.7\%) & 103 & 103 & 1 & 104 & [66.0\%, 66.7\%] & [103, 104] / 408 (25.2--25.5\%) \\
 & Late Incident ($>48\text{h}$) & 205 & 0 (0.0\%) & 0 & 0 & 0 & 0 & [0.0\%, 0.0\%] & 0 / 408 (0.0\%) \\
 & \textbf{All Incident Sepsis-3} & \textbf{408} & \textbf{151 (37.0\%)} & \textbf{150} & \textbf{150} & \textbf{1} & \textbf{151} & \textbf{[36.8\%, 37.0\%]} & \textbf{[150, 151] / 408 (36.8--37.0\%)} \\
\midrule
\textbf{MetaVision Sensitivity: Earliest Rx} & Early Incident ($\le 24\text{h}$) & 47 & 47 (100.0\%) & 0 & 0 & 38 & 38 & [0.0\%, 80.9\%] & [0, 38] / 408 (0.0--9.3\%) \\
\textit{(Pre-Exposure: $T_{\mathrm{Rx,start}}$)} & Intermediate (24--48h) & 156 & 104 (66.7\%) & 0 & 0 & 9 & 9 & [0.0\%, 5.8\%] & [0, 9] / 408 (0.0--2.2\%) \\
 & Late Incident ($>48\text{h}$) & 205 & 0 (0.0\%) & 0 & 0 & 0 & 0 & [0.0\%, 0.0\%] & 0 / 408 (0.0\%) \\
 & \textbf{All Incident Sepsis-3} & \textbf{408} & \textbf{151 (37.0\%)} & \textbf{0} & \textbf{0} & \textbf{47} & \textbf{47} & \textbf{[0.0\%, 11.5\%]} & \textbf{[0, 47] / 408 (0.0--11.5\%)} \\
\midrule
\textbf{Broad Historical MIMIC-III} & Early Incident ($\le 24\text{h}$) & 2,641 & 102 (3.9\%) & 0 & 0 & 74 & 74 & [0.00\%, 2.80\%] & [0, 74] / 3,581 (0.00--2.07\%) \\
\textit{(Date-Level Proxy: $T_{\mathrm{Rx,start}}$)} & Intermediate (24--48h) & 361 & 235 (65.1\%) & 1 & 1 & 36 & 37 & [0.28\%, 10.25\%] & [1, 37] / 3,581 (0.03--1.03\%) \\
 & Late Incident ($>48\text{h}$) & 579 & 0 (0.0\%) & 0 & 0 & 0 & 0 & [0.00\%, 0.00\%] & [0, 0] / 3,581 (0.00\%) \\
 & \textbf{All Incident Sepsis-3} & \textbf{3,581} & \textbf{337 (9.4\%)} & \textbf{1} & \textbf{1} & \textbf{110} & \textbf{111} & \textbf{[0.03\%, 3.10\%]} & \textbf{[1, 111] / 3,581 (0.03--3.10\%)} \\
\bottomrule
\end{tabular*}
\end{singlespace}
\end{table}
\end{landscape}

\end{document}
"""

with open(path, 'w', encoding='utf-8') as f:
    f.write(prefix + new_tables)

print("Successfully wrote formatted tables to main_review_doublespaced.tex")
