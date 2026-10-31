import re
import subprocess

tex_file = '/home/alireza/Sepsis-Onset-Timing-Paradox/manuscript/latex/main_review_doublespaced.tex'
with open(tex_file, 'r', encoding='utf-8') as f:
    text = f.read()

intro_idx = text.find(r'\section{Introduction}')
decl_idx = text.find(r'\subsection*{Declarations and Disclosures}')
if decl_idx == -1:
    decl_idx = text.find(r'Declarations and Disclosures')

print('Intro pos:', intro_idx)
print('Decl pos:', decl_idx)

body = text[intro_idx:decl_idx]

# Insert TC:ignore before and after body
annotated = r"%TC:ignore" + "\n" + text[:intro_idx] + r"%TC:endignore" + "\n" + body + r"%TC:ignore" + "\n" + text[decl_idx:] + r"%TC:endignore"

with open('/tmp/test_jamia_body.tex', 'w', encoding='utf-8') as f:
    f.write(annotated)

cmd = ['texcount', '-sum', '-sub=section', '/tmp/test_jamia_body.tex']
res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
print(res.stdout)
