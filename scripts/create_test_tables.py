import os
import re

tex_path = '/home/alireza/Sepsis-Onset-Timing-Paradox/manuscript/latex/main_review_doublespaced.tex'
out_path = '/home/alireza/Sepsis-Onset-Timing-Paradox/manuscript/latex/test_tables.tex'

with open(tex_path, 'r', encoding='utf-8') as f:
    text = f.read().replace('\r\n', '\n')

# 1. Preamble
text = text.replace(r'\usepackage{setspace}', r'\usepackage{setspace}' + '\n' + r'\usepackage{pdflscape}')

# 2. Table 1
t1_old = r'''\fontsize{7.5pt}{9.8pt}\selectfont
\setlength{\tabcolsep}{2.5pt}
\begin{tabularx}{\textwidth}{@{}p{4.2cm} c c c c X@{}}'''
t1_new = r'''\fontsize{8.5pt}{11.5pt}\selectfont
\setlength{\tabcolsep}{3.5pt}
\begin{tabularx}{\textwidth}{@{}p{4.5cm} c c c c X@{}}'''
assert t1_old in text, 't1 not found'
text = text.replace(t1_old, t1_new)

# 3. Table 2
t2_old = r'''\fontsize{7.5pt}{10.0pt}\selectfont
\setlength{\tabcolsep}{2.5pt}
\resizebox{\textwidth}{!}{\begin{tabular}{l c c c c}
\toprule
\textbf{Model Architecture / Metric} & \textbf{Canonical Sepsis-3 ($H=6\text{h}$)} & \textbf{Hybrid $\mathcal{R}_0^{\mathrm{hybrid}}$ ($\delta=0\text{h}$)} & \textbf{Hybrid $\mathcal{R}_2^{\mathrm{hybrid}}$ ($\delta=2\text{h}$)} & \textbf{Paired $\Delta\text{AUROC}$ ($\mathcal{R}_0^{\mathrm{hybrid}} - \text{Canon}$)} \\'''

t2_new = r'''\fontsize{8.5pt}{11.5pt}\selectfont
\setlength{\tabcolsep}{4.0pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l c c c c}
\toprule
\textbf{Model Architecture / Metric} & \textbf{\makecell{Canonical Sepsis-3\\($H=6\text{h}$)}} & \textbf{\makecell{Hybrid $\mathcal{R}_0^{\mathrm{hybrid}}$\\($\delta=0\text{h}$)}} & \textbf{\makecell{Hybrid $\mathcal{R}_2^{\mathrm{hybrid}}$\\($\delta=2\text{h}$)}} & \textbf{\makecell{Paired $\Delta\text{AUROC}$\\($\mathcal{R}_0^{\mathrm{hybrid}} - \text{Canon}$)}} \\'''
assert t2_old in text, 't2 not found'
text = text.replace(t2_old, t2_new)
text = text.replace(r'\bottomrule' + '\n' + r'\end{tabular}}' + '\n' + r'\vspace{0.4em}' + '\n' + r'\raggedright' + '\n' + r'\fontsize{7.0pt}{8.5pt}\selectfont',
                    r'\bottomrule' + '\n' + r'\end{tabular*}' + '\n' + r'\vspace{0.4em}' + '\n' + r'\raggedright' + '\n' + r'\fontsize{8.0pt}{10.0pt}\selectfont')

# 4. Table 3 in landscape
t3_old = r'''\clearpage
% PAGE 21: TABLE 3
\begin{table}[p]
\begin{singlespace}
\centering
\caption{\textbf{Operational Burden-Matched Alert Frontiers Under 12h Refractory Suppression.} Evaluated under Procedure B with exposure recomputation and first-emitted-alert timing verification.}
\label{tab:table3}
\vspace{0.6em}
\fontsize{7.5pt}{9.8pt}\selectfont
\setlength{\tabcolsep}{2.5pt}
\resizebox{\textwidth}{!}{\begin{tabular}{llccccccc}'''

t3_new = r'''\clearpage
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
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccccccc}'''
assert t3_old in text, 't3 not found'
text = text.replace(t3_old, t3_new)

# Table 3 end
t3_end_old = r'''\bottomrule
\end{tabular}}
\vspace{0.4em}
\raggedright
\fontsize{7.0pt}{8.5pt}\selectfont
$^*$At zero observed detections'''

t3_end_new = r'''\bottomrule
\end{tabular*}
\vspace{0.4em}
\raggedright
\fontsize{8.0pt}{10.0pt}\selectfont
$^*$At zero observed detections'''

if t3_end_old not in text:
    # Try normalized newlines
    text = text.replace('\r\n', '\n')
assert t3_end_old in text, 't3 end not found'
text = text.replace(t3_end_old, t3_end_new)
text = text.replace(r'\end{singlespace}' + '\n' + r'\end{table}' + '\n\n' + r'\clearpage' + '\n' + r'% PAGE 22: TABLE 4',
                    r'\end{singlespace}' + '\n' + r'\end{table}' + '\n' + r'\end{landscape}' + '\n\n' + r'\clearpage' + '\n' + r'% TABLE 4')

# 5. Table 4 in landscape
t4_old = r'''% TABLE 4
\begin{table}[p]
\begin{singlespace}
\centering
\caption{\textbf{Two-Axis Subgroup and Timestamp Uncertainty Matrix.} Opportunity bounds conditional on fixed operational Sepsis-3 timestamps and surveillance coverage. \textit{Note:} Subgroup Bounds denote $[O^{\mathrm{def}}, O^{\mathrm{poss}}] / \text{Total}(N)$ within each onset stratum. For All Incident Sepsis-3, within-covered bounds ($[O^{\mathrm{def}}, O^{\mathrm{poss}}] / \text{Covered}[V=1]$) are $[150, 151]/151 = [99.3\%, 100.0\%]$ for MetaVision $T_{\mathrm{susp}}$, $[0, 47]/151 = [0.0\%, 31.1\%]$ for MetaVision Earliest Rx, and $[1, 111]/337 = [0.30\%, 32.94\%]$ for Broad Historical MIMIC-III.}
\label{tab:table4}
\vspace{0.6em}
\fontsize{7.5pt}{9.5pt}\selectfont
\setlength{\tabcolsep}{1.5pt}
\resizebox{\textwidth}{!}{\begin{tabular}{llcccccccc}'''

t4_new = r'''\begin{landscape}
% TABLE 4
\begin{table}[p]
\begin{singlespace}
\centering
\caption{\textbf{Two-Axis Subgroup and Timestamp Uncertainty Matrix.} Opportunity bounds conditional on fixed operational Sepsis-3 timestamps and surveillance coverage. \textit{Note:} Subgroup Bounds denote $[O^{\mathrm{def}}, O^{\mathrm{poss}}] / \text{Total}(N)$ within each onset stratum. For All Incident Sepsis-3, within-covered bounds ($[O^{\mathrm{def}}, O^{\mathrm{poss}}] / \text{Covered}[V=1]$) are $[150, 151]/151 = [99.3\%, 100.0\%]$ for MetaVision $T_{\mathrm{susp}}$, $[0, 47]/151 = [0.0\%, 31.1\%]$ for MetaVision Earliest Rx, and $[1, 111]/337 = [0.30\%, 32.94\%]$ for Broad Historical MIMIC-III.}
\label{tab:table4}
\vspace{0.6em}
\fontsize{8.0pt}{10.8pt}\selectfont
\setlength{\tabcolsep}{3.5pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llcccccccc}'''
assert t4_old in text, 't4 not found'
text = text.replace(t4_old, t4_new)

# Table 4 end
t4_end_old = r'''\bottomrule
\end{tabular}}
\end{singlespace}
\end{table}'''

t4_end_new = r'''\bottomrule
\end{tabular*}
\end{singlespace}
\end{table}
\end{landscape}'''
assert t4_end_old in text, 't4 end not found'
text = text.replace(t4_end_old, t4_end_new)

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully created test_tables.tex')
