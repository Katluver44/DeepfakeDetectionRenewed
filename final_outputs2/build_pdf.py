#!/usr/bin/env python3
"""Convert paper_style_summary.md to PDF using fpdf2."""
from fpdf import FPDF
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
md_path = HERE / "report" / "paper_style_summary.md"
pdf_path = HERE / "paper_style_summary.pdf"

text = md_path.read_text()
# Replace non-latin1 characters with ASCII equivalents
char_map = {
    '↔': '<->',   '→': '->',    '←': '<-',
    '−': '-',     '≈': '~',     '≤': '<=',    '≥': '>=',    '≠': '!=',
    'ρ': 'rho',   'β': 'beta',  'λ': 'lambda', '×': 'x',
    'Δ': 'Delta', '∈': 'in',    'α': 'alpha',  'γ': 'gamma',
    ''': "'", ''': "'", '"': '"', '"': '"',
    '–': '--',    '—': '--',     '²': '^2',    '³': '^3',
    '•': '*',
}
for uni, asc in char_map.items():
    text = text.replace(uni, asc)

class PDF(FPDF):
    def header(self):
        pass
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

pdf = PDF()
pdf.set_margins(22, 22, 22)
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

def write_line(line: str):
    """Render one line of markdown."""
    line = line.rstrip()

    # --- Heading levels ---
    if line.startswith("# ") and not line.startswith("## "):
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_text_color(20, 40, 80)
        pdf.multi_cell(0, 9, line[2:])
        pdf.ln(2)
        pdf.set_text_color(0)
        return
    if line.startswith("## "):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(20, 60, 120)
        pdf.multi_cell(0, 8, line[3:])
        pdf.ln(1)
        pdf.set_text_color(0)
        return
    if line.startswith("### "):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(50, 80, 140)
        pdf.multi_cell(0, 7, line[4:])
        pdf.ln(1)
        pdf.set_text_color(0)
        return

    # --- Horizontal rule ---
    if re.match(r"^-{3,}$", line):
        pdf.set_draw_color(180)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(3)
        return

    # --- Table row (render as single wrapped line to avoid layout issues) ---
    if line.startswith("|"):
        # skip separator rows
        if re.match(r"^\|[-| :]+\|$", line):
            return
        cells = [c.strip() for c in line.strip("|").split("|")]
        row_text = "  |  ".join(cells)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(40)
        # reset x position explicitly to left margin
        pdf.set_x(pdf.l_margin)
        safe = row_text.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 5, safe)
        return

    # --- Bullet point ---
    if line.startswith("- "):
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(0)
        # strip markdown bold
        content = re.sub(r"\*\*(.+?)\*\*", r"\1", line[2:])
        content = re.sub(r"`(.+?)`", r"\1", content)
        content = content.encode("latin-1", errors="replace").decode("latin-1")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5.5, f"*  {content}")
        return

    # --- Blank line ---
    if line == "":
        pdf.ln(2.5)
        return

    # --- Normal paragraph ---
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(0)
    # strip markdown formatting for plain text
    content = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
    content = re.sub(r"`(.+?)`", r"\1", content)
    content = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", content)
    content = content.encode("latin-1", errors="replace").decode("latin-1")
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5.5, content)

lines = text.split("\n")
for ln in lines:
    write_line(ln)

pdf.output(str(pdf_path))
print(f"PDF written to {pdf_path}  ({pdf_path.stat().st_size // 1024} KB)")
