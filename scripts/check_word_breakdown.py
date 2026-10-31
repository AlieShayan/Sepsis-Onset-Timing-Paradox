import re
import subprocess

tex_file = '/home/alireza/Sepsis-Onset-Timing-Paradox/manuscript/latex/main_review_doublespaced.tex'

with open(tex_file, 'r', encoding='utf-8') as f:
    text = f.read()

sec_names = ['Introduction', 'Materials and Methods', 'Results', 'Discussion', 'Conclusion']
sec_pos = []
for name in sec_names:
    p = text.find(r'\section{' + name + '}')
    sec_pos.append((name, p))

end_p = text.find(r'\noindent\textbf{Declarations:}')
if end_p == -1:
    end_p = text.find(r'\begin{thebibliography}')
sec_pos.append(('End', end_p))

total_words = 0
for i in range(len(sec_names)):
    s_name, s_start = sec_pos[i]
    e_name, s_end = sec_pos[i+1]
    sec_text = text[s_start:s_end]
    
    # Strip figure environments
    sec_clean = re.sub(r'\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}', '', sec_text, flags=re.S)
    sec_clean = re.sub(r'%.*', '', sec_clean)
    words = re.findall(r'[a-zA-Z0-9]+', sec_clean)
    total_words += len(words)
    print(f'{s_name}: ~{len(words)} words')

print(f'Total main-text words (alphanumeric regex): {total_words}')
