import subprocess
import os

tex_file = '/home/alireza/Sepsis-Onset-Timing-Paradox/manuscript/latex/main_review_doublespaced.tex'
with open(tex_file, 'r', encoding='utf-8') as f:
    text = f.read()

# In JAMIA Open:
# Word count excludes: title page, author block, abstract, declarations, references, tables, figures, captions.
# Includes: Introduction, Methods, Results, Discussion, Conclusion.

intro_idx = text.find(r'\section{Introduction}')
decl_idx = text.find(r'\noindent\textbf{Declarations:}')
if decl_idx == -1:
    decl_idx = text.find(r'\begin{thebibliography}')

# Let us insert %TC:ignore before \section{Introduction} and after Conclusion
annotated_tex = r"%TC:ignore" + "\n" + text[:intro_idx] + r"%TC:endignore" + "\n" + text[intro_idx:decl_idx] + r"%TC:ignore" + "\n" + text[decl_idx:] + r"%TC:endignore"

temp_file = '/home/alireza/Sepsis-Onset-Timing-Paradox/manuscript/latex/temp_count.tex'
with open(temp_file, 'w', encoding='utf-8') as f:
    f.write(annotated_tex)

cmd = ['texcount', '-sum', '-sub=section', temp_file]
res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print(res.stdout)

if os.path.exists(temp_file):
    os.remove(temp_file)
