"""Build an editable Word version of the eight-page assignment report."""

from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION_START
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "build_report.py"
OUTPUT = ROOT / "report_revised.docx"


class RichText(HTMLParser):
    def __init__(self, paragraph):
        super().__init__()
        self.paragraph = paragraph
        self.bold = 0
        self.italic = 0

    def handle_starttag(self, tag, attrs):
        if tag == "b":
            self.bold += 1
        elif tag == "i":
            self.italic += 1

    def handle_endtag(self, tag):
        if tag == "b":
            self.bold = max(0, self.bold - 1)
        elif tag == "i":
            self.italic = max(0, self.italic - 1)

    def handle_data(self, data):
        if data:
            run = self.paragraph.add_run(unescape(data))
            run.bold = self.bold > 0
            run.italic = self.italic > 0


def plain(html: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", str(html)))


def collect_events():
    source = SOURCE.read_text(encoding="utf-8")
    prefix, rest = source.split("# Page 1: Part I Q1/Q2.", 1)
    body = rest.split("assert PAGE == 8", 1)[0]
    env = {"__file__": str(SOURCE), "__name__": "report_content"}
    exec(prefix, env)
    events = []

    def next_page(title, subtitle="", *, wide=False):
        env["W"], env["H"] = env["landscape"](env["A4"]) if wide else env["A4"]
        events.append(("page", title, subtitle, wide))
        return 500.0

    def heading(title, y):
        events.append(("heading", title))
        return y - 20

    def para(text, y, *, style="body", x=None, width=None, gap=9):
        text = text.replace("appear on pages 4-5", "appear in the results sections")
        text = text.replace("appears on page 7", "appears in the final-selection section")
        events.append(("paragraph", text, style))
        return y - 22

    def table(headers, rows, widths, y, **kwargs):
        events.append(("table", headers, rows, widths))
        return y - 100

    def figure(path, y, *, max_width, max_height, x=None):
        events.append(("figure", Path(path), max_width, max_height))
        return y - max_height

    def bullet(text, y, *, style="body"):
        events.append(("bullet", text, style))
        return y - 20

    env.update(next_page=next_page, heading=heading, para=para,
               table=table, figure=figure, bullet=bullet)
    exec(body, env)
    assert sum(event[0] == "page" for event in events) == 8
    return events


def set_cell_border(cell, color="D9D9D9"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for side in ("top", "left", "bottom", "right"):
        edge = OxmlElement(f"w:{side}")
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), "4")
        edge.set(qn("w:color"), color)
        borders.append(edge)


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(cell, top=75, start=75, bottom=75, end=75):
    tc_pr = cell._tc.get_or_add_tcPr()
    mar = tc_pr.first_child_found_in("w:tcMar")
    if mar is None:
        mar = OxmlElement("w:tcMar")
        tc_pr.append(mar)
    for side, val in (("top", top), ("start", start),
                      ("bottom", bottom), ("end", end)):
        item = OxmlElement(f"w:{side}")
        item.set(qn("w:w"), str(val))
        item.set(qn("w:type"), "dxa")
        mar.append(item)


def configure_section(section, wide):
    section.page_width = Inches(11.69 if wide else 8.27)
    section.page_height = Inches(8.27 if wide else 11.69)
    section.orientation = WD_ORIENT.LANDSCAPE if wide else WD_ORIENT.PORTRAIT
    section.left_margin = Inches(0.57)
    section.right_margin = Inches(0.57)
    section.top_margin = Inches(0.62)
    section.bottom_margin = Inches(0.62)
    section.header_distance = Inches(0.25)
    section.footer_distance = Inches(0.3)


def setup_styles(doc):
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(9.4)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.08
    for name, size, before, after in (
        ("Title", 15, 0, 5),
        ("Heading 1", 13, 0, 5),
        ("Heading 2", 10.5, 8, 4),
    ):
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        if name == "Title":
            ppr = style.element.get_or_add_pPr()
            for border in ppr.findall(qn("w:pBdr")):
                ppr.remove(border)


def add_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.add_run("Air University  |  Deep Learning  |  Assignment 02  |  ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    p._p.append(field)
    for run in p.runs:
        run.font.name = "Calibri"
        run.font.size = Pt(7)
        run.font.color.rgb = RGBColor(80, 80, 80)


def build():
    events = collect_events()
    doc = Document()
    setup_styles(doc)
    doc.core_properties.title = "CNN architectures training and ablation studies"
    doc.core_properties.subject = "CIFAR-10 assignment report"
    section_count = 0
    current_wide = False

    for event in events:
        kind = event[0]
        if kind == "page":
            _, title, subtitle, wide = event
            section = (doc.sections[0] if section_count == 0 else
                       doc.add_section(WD_SECTION_START.NEW_PAGE))
            configure_section(section, wide)
            if section_count == 0:
                add_footer(section)
            current_wide = wide
            section_count += 1
            p = doc.add_paragraph(style="Title" if section_count == 1 else "Heading 1")
            p.add_run(title.replace(" - ", " ").replace("(Q1-Q2)", "Q1 and Q2"))
            if subtitle:
                sub = doc.add_paragraph()
                sub.add_run(subtitle)
                sub.style = doc.styles["Normal"]
                sub.paragraph_format.space_after = Pt(9)
                for run in sub.runs:
                    run.font.size = Pt(8.2)
                    run.font.color.rgb = RGBColor(75, 75, 75)
        elif kind == "heading":
            doc.add_paragraph(event[1], style="Heading 2")
        elif kind in ("paragraph", "bullet"):
            text, style = event[1], event[2]
            p = doc.add_paragraph(style="List Bullet" if kind == "bullet" else "Normal")
            RichText(p).feed(text)
            if style == "small":
                for run in p.runs:
                    run.font.size = Pt(8.7)
            elif style == "caption":
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(8)
                for run in p.runs:
                    run.italic = True
                    run.font.size = Pt(8)
            if current_wide:
                for run in p.runs:
                    run.font.size = Pt(8)
        elif kind == "table":
            _, headers, rows, widths = event
            t = doc.add_table(rows=1, cols=len(headers))
            t.autofit = False
            usable = 10.55 if current_wide else 7.1
            scale = usable * 72 / sum(widths)
            col_widths = [Inches(w / 72 * scale) for w in widths]
            for i, (label, width) in enumerate(zip(headers, col_widths)):
                t.columns[i].width = width
                cell = t.rows[0].cells[i]
                cell.width = width
                cell.text = plain(label)
                shade(cell, "243B5A")
                set_cell_border(cell)
                set_cell_margins(cell, 65, 55, 65, 55)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(0)
                    for run in p.runs:
                        run.bold = True
                        run.font.size = Pt(6.4 if current_wide else 8)
                        run.font.color.rgb = RGBColor(255, 255, 255)
            tr_pr = t.rows[0]._tr.get_or_add_trPr()
            tr_pr.append(OxmlElement("w:tblHeader"))
            for j, row in enumerate(rows):
                cells = t.add_row().cells
                for i, (value, width) in enumerate(zip(row, col_widths)):
                    cell = cells[i]
                    cell.width = width
                    cell.text = plain(value)
                    if j % 2 == 0:
                        shade(cell, "F1F4F7")
                    set_cell_border(cell)
                    set_cell_margins(cell, 55, 55, 55, 55)
                    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    for p in cell.paragraphs:
                        p.paragraph_format.space_after = Pt(0)
                        p.paragraph_format.line_spacing = 1.0
                        for run in p.runs:
                            run.font.size = Pt(6.1 if current_wide else 7.6)
            doc.add_paragraph().paragraph_format.space_after = Pt(1)
        elif kind == "figure":
            _, path, max_width, max_height = event
            with Image.open(path) as img:
                ratio = img.width / img.height
            width = min(max_width / 72, 10.55 if current_wide else 7.1,
                        (max_height / 72) * ratio)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(2)
            p.add_run().add_picture(str(path), width=Inches(width))

    assert section_count == 8
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
