import re

tex_file = '/home/alireza/Sepsis-Onset-Timing-Paradox/manuscript/latex/main_review_doublespaced.tex'
with open(tex_file, 'r', encoding='utf-8') as f:
    text = f.read()

subsecs = list(re.finditer(r'\\(?:sub)*section\{([^}]+)\}', text))
for i in range(len(subsecs)):
    name = subsecs[i].group(1)
    start = subsecs[i].start()
    end = subsecs[i+1].start() if i+1 < len(subsecs) else text.find(r'\begin{thebibliography}')
    chunk = text[start:end]
    # Remove figure/table
    chunk = re.sub(r'\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}', '', chunk, flags=re.S)
    chunk = re.sub(r'%.*', '', chunk)
    # Remove math
    chunk_no_math = re.sub(r'\$.*?\$', '', chunk)
    chunk_no_math = re.sub(r'\\\[.*?\\\]', '', chunk_no_math, flags=re.S)
    chunk_no_math = re.sub(r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}', '', chunk_no_math, flags=re.S)
    words = re.findall(r'[a-zA-Z0-9]+', chunk_no_math)
    print(f'{name}: {len(words)} words')
