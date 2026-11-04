#!/usr/bin/env python3
"""
format_docx.py

Converts Markdown manuscripts to styled Word (.docx) files adhering to:
- Times New Roman throughout
- Standard journal typography (10.5pt body, 1.15 line spacing)
- 14pt / 12pt / 11pt bold headings
- Monochrome academic styling (pure black and white, zero colored elements)
- Formal booktabs-style table borders (top, header-bottom, table-bottom)
"""

import subprocess
import os
import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_border(cell, **kwargs):
    """
    Set cell borders: top, bottom, left, right.
    Usage: set_cell_border(cell, top={"sz": 4, "val": "single", "color": "000000"})
    """
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = f'w:{edge}'
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key, val in edge_data.items():
                element.set(qn(f'w:{key}'), str(val))

def compile_and_style_docx(md_path, docx_path):
    print(f"Converting {md_path} -> {docx_path}...")
    
    # Run pandoc
    cmd = [
        'pandoc',
        os.path.basename(md_path),
        '-o', docx_path,
        '--from=markdown+pipe_tables+grid_tables+table_captions',
        '--toc=false'
    ]
    subprocess.run(cmd, check=True, cwd=os.path.dirname(md_path))
    
    # Load and style with docx
    doc = docx.Document(docx_path)
    
    # Margins: 0.8 in
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        section.header_distance = Inches(0.5)
        section.footer_distance = Inches(0.5)
        
    # Base Normal style
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(10.5)
    style_normal.font.color.rgb = RGBColor(0, 0, 0)
    style_normal.paragraph_format.line_spacing = 1.15
    style_normal.paragraph_format.space_after = Pt(4)
    style_normal.paragraph_format.space_before = Pt(0)
    
    # Heading styles
    for level in range(1, 4):
        style_name = f'Heading {level}'
        if style_name in doc.styles:
            h_style = doc.styles[style_name]
            h_style.font.name = 'Times New Roman'
            h_style.font.bold = True
            h_style.font.color.rgb = RGBColor(0, 0, 0)
            if level == 1:
                h_style.font.size = Pt(14)
                h_style.paragraph_format.space_before = Pt(12)
                h_style.paragraph_format.space_after = Pt(4)
            elif level == 2:
                h_style.font.size = Pt(12)
                h_style.paragraph_format.space_before = Pt(8)
                h_style.paragraph_format.space_after = Pt(3)
            elif level == 3:
                h_style.font.size = Pt(11)
                h_style.paragraph_format.space_before = Pt(6)
                h_style.paragraph_format.space_after = Pt(2)
                
    # Style all paragraphs and runs
    for p in doc.paragraphs:
        if p.style.name.startswith('Heading'):
            p.style.font.name = 'Times New Roman'
            p.style.font.color.rgb = RGBColor(0, 0, 0)
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.color.rgb = RGBColor(0, 0, 0)
            
    # Format Tables
    for table in doc.tables:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        n_rows = len(table.rows)
        for r_idx, row in enumerate(table.rows):
            is_header = (r_idx == 0)
            is_last = (r_idx == n_rows - 1)
            
            for cell in row.cells:
                # Border specs
                border_kwargs = {
                    'left': {'val': 'none'},
                    'right': {'val': 'none'},
                    'insideH': {'val': 'none'},
                    'insideV': {'val': 'none'}
                }
                if is_header:
                    border_kwargs['top'] = {'val': 'single', 'sz': 8, 'color': '000000'}
                    border_kwargs['bottom'] = {'val': 'single', 'sz': 6, 'color': '000000'}
                elif is_last:
                    border_kwargs['bottom'] = {'val': 'single', 'sz': 8, 'color': '000000'}
                else:
                    border_kwargs['top'] = {'val': 'none'}
                    border_kwargs['bottom'] = {'val': 'none'}
                    
                set_cell_border(cell, **border_kwargs)
                
                # Cell typography
                for cp in cell.paragraphs:
                    cp.paragraph_format.space_before = Pt(1)
                    cp.paragraph_format.space_after = Pt(1)
                    cp.paragraph_format.line_spacing = 1.05
                    for cr in cp.runs:
                        cr.font.name = 'Times New Roman'
                        cr.font.size = Pt(9.0)
                        cr.font.color.rgb = RGBColor(0, 0, 0)
                        if is_header:
                            cr.font.bold = True
                            
    doc.save(docx_path)
    print(f"Successfully generated and styled {docx_path}!")

if __name__ == '__main__':
    base_dir = '/home/alireza/Sepsis-Onset-Timing-Paradox/manuscript'
    compile_and_style_docx(
        os.path.join(base_dir, 'draft_jamia_open_v1.md'),
        os.path.join(base_dir, 'draft_jamia_open_v1.docx')
    )
    compile_and_style_docx(
        os.path.join(base_dir, 'supplementary_material.md'),
        os.path.join(base_dir, 'supplementary_material.docx')
    )
