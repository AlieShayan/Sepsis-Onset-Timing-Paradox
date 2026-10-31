#!/usr/bin/env python3
"""
verify_citation_sequence.py
Verifies that all references are cited in strictly sequential order of appearance.
"""

import re
import os

for fname in ['main_review_doublespaced.tex', 'main.tex']:
    path = os.path.join('/home/alireza/Sepsis-Onset-Timing-Paradox/manuscript/latex', fname)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Find bibliography items in order
    bib_items = re.findall(r'\\bibitem\{([^}]+)\}', text)
    print(f"\n=== {fname} ===")
    print(f"Total bibliography items: {len(bib_items)}")

    # Find body text (up to \begin{thebibliography})
    body = text[:text.find(r'\begin{thebibliography}')]
    
    # Find all cite commands
    cites = re.findall(r'\\cite\{([^}]+)\}', body)
    cited_keys = []
    for c in cites:
        for k in c.split(','):
            cited_keys.append(k.strip())

    # Find first appearance of each key
    first_appearance = {}
    for i, k in enumerate(cited_keys):
        if k not in first_appearance:
            first_appearance[k] = len(first_appearance) + 1

    print(f"Total unique keys cited in text: {len(first_appearance)}")

    # Check if first appearance matches bib order
    all_ordered = True
    for expected_idx, bib_k in enumerate(bib_items, 1):
        actual_order = first_appearance.get(bib_k)
        if actual_order != expected_idx:
            print(f"MISMATCH: Bib item [{expected_idx}] '{bib_k}' first appeared at citation order {actual_order}")
            all_ordered = False

    if all_ordered:
        print("PERFECT: All bibliography items are cited in STRICTLY SEQUENTIAL order (1 to 26)!")
