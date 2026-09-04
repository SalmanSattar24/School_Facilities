from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "memo" / "draft_memo.md"
OUTPUT = ROOT / "output" / "pdf" / "memo.pdf"


def inline_markup(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`(.+?)`", r'<font name="Courier">\1</font>', escaped)
    return escaped


def footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(HexColor("#CBD5E1"))
    canvas.setLineWidth(0.35)
    canvas.line(0.48 * inch, 0.35 * inch, 8.02 * inch, 0.35 * inch)
    canvas.setFillColor(HexColor("#64748B"))
    canvas.setFont("Helvetica", 6.5)
    canvas.drawString(0.48 * inch, 0.23 * inch, "School Facilities Take-Home | Reproducible measurement and uncertainty pipeline")
    canvas.drawRightString(8.02 * inch, 0.23 * inch, f"Page {document.page}")
    canvas.restoreState()


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    title_style = ParagraphStyle(
        "Title", fontName="Helvetica-Bold", fontSize=14, leading=15.5,
        textColor=HexColor("#0F172A"), spaceAfter=4.5,
    )
    heading_style = ParagraphStyle(
        "Heading", fontName="Helvetica-Bold", fontSize=10, leading=11.5,
        textColor=HexColor("#174E78"), spaceBefore=3.6, spaceAfter=1.2,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body", fontName="Helvetica", fontSize=9, leading=10.75,
        textColor=HexColor("#1F2937"), alignment=TA_LEFT, spaceAfter=2.6,
    )
    story = []
    paragraph_lines: list[str] = []

    def flush() -> None:
        if paragraph_lines:
            story.append(Paragraph(inline_markup(" ".join(paragraph_lines)), body_style))
            paragraph_lines.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
        elif stripped.startswith("# "):
            flush()
            story.append(Paragraph(inline_markup(stripped[2:]), title_style))
            story.append(Spacer(1, 0.5))
        elif stripped.startswith("## "):
            flush()
            story.append(Paragraph(inline_markup(stripped[3:]), heading_style))
        else:
            paragraph_lines.append(stripped)
    flush()

    document = BaseDocTemplate(
        str(OUTPUT), pagesize=LETTER,
        leftMargin=0.48 * inch, rightMargin=0.48 * inch,
        topMargin=0.43 * inch, bottomMargin=0.43 * inch,
        title="School Facilities Measurement Pipeline - Memo",
        author="Salman",
    )
    frame = Frame(
        document.leftMargin, document.bottomMargin,
        document.width, document.height,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    document.addPageTemplates([PageTemplate(id="memo", frames=[frame], onPage=footer)])
    document.build(story)


if __name__ == "__main__":
    build()
