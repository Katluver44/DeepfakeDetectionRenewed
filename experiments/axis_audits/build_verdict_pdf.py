#!/usr/bin/env python3
"""Convert COMPREHENSIVE_VERDICT.md to PDF (fpdf2), same approach as
final_outputs2/build_pdf.py."""
from fpdf import FPDF
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
md_path = HERE / "audits_outputs" / "COMPREHENSIVE_VERDICT.md"
pdf_path = HERE / "audits_outputs" / "comprehensive_verdict.pdf"

text = md_path.read_text()
char_map = {
    '↔': '<->', '→': '->', '←': '<-', '−': '-', '≈': '~', '≤': '<=', '≥': '>=',
    '≠': '!=', 'ρ': 'rho', 'β': 'beta', 'λ': 'lambda', '×': 'x', 'Δ': 'Delta',
    '∈': 'in', 'α': 'alpha', '–': '--', '—': '--', '²': '^2', '³': '^3',
    '•': '*', '✓': '[ok]', '⁻⁵': 'e-5', '⁻': '-', '↑': 'up', '↓': 'down',
    ''': "'", ''': "'", '"': '"', '"': '"', '§': 'sec.',
}
for u, a in char_map.items():
    text = text.replace(u, a)
text = text.encode("latin-1", "replace").decode("latin-1")

class PDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, f"axis_audits comprehensive verdict - page {self.page_no()}",
                  align="C")

pdf = PDF(format="A4")
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_margins(16, 16, 16)
pdf.add_page()
W = pdf.w - 32

for raw in text.split("\n"):
    line = raw.rstrip()
    if not line.strip():
        pdf.ln(2)
        continue
    if line.startswith("# "):
        pdf.set_font("Helvetica", "B", 15)
        pdf.multi_cell(W, 7, line[2:]); pdf.ln(2)
    elif line.startswith("## "):
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(W, 6, line[3:]); pdf.ln(1)
    elif line.startswith("### "):
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.multi_cell(W, 5.5, line[4:])
    elif line.strip() == "---":
        pdf.ln(1)
    else:
        body = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        body = re.sub(r"\*(.+?)\*", r"\1", body)
        body = re.sub(r"`(.+?)`", r"\1", body)
        bold_lead = bool(re.match(r"^\s*(\*\*|\d+\.\s+\*\*|[-*]\s+\*\*)", line))
        pdf.set_font("Helvetica", "B" if False else "", 9.5)
        if bold_lead:
            # bold the leading sentence up to the first period for emphasis lines
            m = re.match(r"^(.*?\.)\s*(.*)$", body)
            if m:
                pdf.set_font("Helvetica", "B", 9.5)
                pdf.write(5, m.group(1) + " ")
                pdf.set_font("Helvetica", "", 9.5)
                pdf.write(5, m.group(2))
                pdf.ln(6)
                continue
        pdf.multi_cell(W, 5, body)

pdf.output(str(pdf_path))
print(f"wrote {pdf_path}")
