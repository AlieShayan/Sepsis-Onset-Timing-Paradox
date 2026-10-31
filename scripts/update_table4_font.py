import re
import subprocess

path = '/home/alireza/Sepsis-Onset-Timing-Paradox/manuscript/latex/main_review_doublespaced.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Let's update Word Count on title page to 3,843
text = text.replace(r'\textbf{Word Count (Main Text):} 3,875 words', r'\textbf{Word Count (Main Text):} 3,843 words')

# In Table 4, change font to 8.2pt / 10.8pt
text = text.replace(r'\fontsize{7.8pt}{10.2pt}\selectfont' + '\n' + r'\setlength{\tabcolsep}{3.0pt}',
                    r'\fontsize{8.2pt}{10.8pt}\selectfont' + '\n' + r'\setlength{\tabcolsep}{2.8pt}')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated main_review_doublespaced.tex")
