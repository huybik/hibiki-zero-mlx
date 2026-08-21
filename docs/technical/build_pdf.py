#!/usr/bin/env python
"""Build the technical report PDF from report.md using Pandoc + ReportLab.

Pandoc supplies a stable Markdown AST; ReportLab owns pagination, typography,
links, tables, headers, and the final PDF. SVG figures are rasterized at high
resolution only for the PDF build. The Markdown and SVG files remain the source
artifacts.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = HERE / "report.md"
OUTPUT = ROOT / "output" / "pdf" / "hibiki-zero-mlx-technical-report.pdf"

INK = colors.HexColor("#1B2540")
MUTED = colors.HexColor("#65708C")
INDIGO = colors.HexColor("#5F67C9")
TEAL = colors.HexColor("#178B79")
PALE = colors.HexColor("#F1F4FA")
BORDER = colors.HexColor("#DCE2EE")
WARM = colors.HexColor("#FFF4E7")


def pandoc_ast() -> dict:
    result = subprocess.run(
        ["pandoc", SOURCE.name, "-f", "gfm", "-t", "json"],
        cwd=HERE,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def meta_text(meta: dict, key: str, default: str = "") -> str:
    value = meta.get(key)
    if not value:
        return default
    if value["t"] == "MetaString":
        return value["c"]
    if value["t"] == "MetaInlines":
        return plain_inlines(value["c"])
    return default


def plain_inlines(inlines: list[dict]) -> str:
    out: list[str] = []
    for item in inlines:
        tag, content = item["t"], item.get("c")
        if tag == "Str":
            out.append(content)
        elif tag in {"Space", "SoftBreak", "LineBreak"}:
            out.append(" ")
        elif tag in {"Emph", "Strong", "Strikeout", "Underline"}:
            out.append(plain_inlines(content))
        elif tag == "Code":
            out.append(content[1])
        elif tag in {"Link", "Image", "Span"}:
            out.append(plain_inlines(content[1]))
        elif tag == "Quoted":
            out.append(plain_inlines(content[1]))
    return "".join(out).strip()


def rich_inlines(inlines: list[dict]) -> str:
    out: list[str] = []
    for item in inlines:
        tag, content = item["t"], item.get("c")
        if tag == "Str":
            out.append(escape(content))
        elif tag == "Space":
            out.append(" ")
        elif tag == "SoftBreak":
            out.append(" ")
        elif tag == "LineBreak":
            out.append("<br/>")
        elif tag == "Emph":
            out.append(f"<i>{rich_inlines(content)}</i>")
        elif tag == "Strong":
            out.append(f"<b>{rich_inlines(content)}</b>")
        elif tag == "Strikeout":
            out.append(f"<strike>{rich_inlines(content)}</strike>")
        elif tag == "Code":
            out.append(f'<font face="Courier" color="#384260">{escape(content[1])}</font>')
        elif tag == "Link":
            label = rich_inlines(content[1])
            url = escape(content[2][0], {'"': "&quot;"})
            out.append(f'<link href="{url}" color="#4F59B5">{label}</link>')
        elif tag == "Span":
            out.append(rich_inlines(content[1]))
        elif tag == "Quoted":
            out.append(f'&quot;{rich_inlines(content[1])}&quot;')
        elif tag == "Math":
            out.append(escape(content[1]))
        elif tag in {"RawInline", "Note", "Image"}:
            continue
    return "".join(out)


class TechnicalDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        page_w, page_h = A4
        cover = Frame(24 * mm, 18 * mm, page_w - 48 * mm, page_h - 36 * mm, id="cover")
        body = Frame(20 * mm, 19 * mm, page_w - 40 * mm, page_h - 36 * mm, id="body")
        self.addPageTemplates(
            [
                PageTemplate(id="cover", frames=[cover], onPage=self._cover_page),
                PageTemplate(id="body", frames=[body], onPage=self._body_page),
            ]
        )
        self._bookmark_id = 0

    def beforeDocument(self):
        # multiBuild lays out more than once while resolving the TOC. Stable
        # bookmark ids are required for the index entries to converge.
        self._bookmark_id = 0
        super().beforeDocument()

    @staticmethod
    def _cover_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#10162F"))
        canvas.rect(0, 0, A4[0], 12 * mm, stroke=0, fill=1)
        canvas.restoreState()

    @staticmethod
    def _body_page(canvas, doc):
        canvas.saveState()
        page_w, page_h = A4
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(20 * mm, page_h - 14 * mm, page_w - 20 * mm, page_h - 14 * mm)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(INDIGO)
        canvas.drawString(20 * mm, page_h - 10.5 * mm, "HIBIKI-ZERO MLX")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(page_w - 20 * mm, page_h - 10.5 * mm, "TECHNICAL REPORT")
        canvas.setStrokeColor(BORDER)
        canvas.line(20 * mm, 13 * mm, page_w - 20 * mm, 13 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(20 * mm, 8.5 * mm, "Evidence current to 21 August 2026")
        canvas.drawRightString(page_w - 20 * mm, 8.5 * mm, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable: Flowable):
        if not isinstance(flowable, Paragraph):
            return
        name = flowable.style.name
        if name not in {"Heading1", "Heading2"}:
            return
        level = 0 if name == "Heading1" else 1
        text = flowable.getPlainText()
        self._bookmark_id += 1
        key = f"heading-{self._bookmark_id}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=level == 0)
        self.notify("TOCEntry", (level, text, self.page, key))


def styles():
    base = getSampleStyleSheet()
    base.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=32,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=7,
        )
    )
    base.add(
        ParagraphStyle(
            name="CoverSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    base["BodyText"].fontName = "Helvetica"
    base["BodyText"].fontSize = 9.3
    base["BodyText"].leading = 13.8
    base["BodyText"].textColor = INK
    base["BodyText"].spaceAfter = 6.5
    base["BodyText"].alignment = TA_LEFT
    base["Heading1"].fontName = "Helvetica-Bold"
    base["Heading1"].fontSize = 22
    base["Heading1"].leading = 26
    base["Heading1"].textColor = INK
    base["Heading1"].spaceAfter = 12
    base["Heading2"].fontName = "Helvetica-Bold"
    base["Heading2"].fontSize = 14.5
    base["Heading2"].leading = 18
    base["Heading2"].textColor = INDIGO
    base["Heading2"].spaceBefore = 10
    base["Heading2"].spaceAfter = 6
    base["Heading3"].fontName = "Helvetica-Bold"
    base["Heading3"].fontSize = 11.2
    base["Heading3"].leading = 14
    base["Heading3"].textColor = TEAL
    base["Heading3"].spaceBefore = 8
    base["Heading3"].spaceAfter = 4
    base.add(
        ParagraphStyle(
            name="Quote",
            parent=base["BodyText"],
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#33415F"),
            leftIndent=4,
            rightIndent=4,
        )
    )
    base.add(
        ParagraphStyle(
            name="Caption",
            parent=base["BodyText"],
            fontSize=8,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )
    base.add(
        ParagraphStyle(
            name="Cell",
            parent=base["BodyText"],
            fontSize=7.5,
            leading=9.5,
            spaceAfter=0,
        )
    )
    base.add(
        ParagraphStyle(
            name="CellHead",
            parent=base["Cell"],
            fontName="Helvetica-Bold",
            textColor=colors.white,
        )
    )
    base.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.6,
            leading=10.2,
            textColor=colors.HexColor("#27324E"),
            backColor=PALE,
            borderColor=BORDER,
            borderWidth=0.5,
            borderPadding=7,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    return base


class ReportRenderer:
    def __init__(self, stylebook, tmp: Path):
        self.s = stylebook
        self.tmp = tmp
        self.available_width = A4[0] - 40 * mm
        self.image_cache: dict[Path, Path] = {}
        self.h1_seen = False

    def raster_image(self, rel: str) -> Path:
        source = (HERE / rel).resolve()
        if source.suffix.lower() != ".svg":
            return source
        if source in self.image_cache:
            return self.image_cache[source]
        target = self.tmp / f"{source.stem}.png"
        root = ET.parse(source).getroot()
        svg_w = float(root.attrib["width"])
        svg_h = float(root.attrib["height"])
        viewport_w = round(svg_w)
        viewport_h = round(svg_h)
        chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        process = subprocess.Popen(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=2",
                f"--window-size={viewport_w},{viewport_h}",
                f"--user-data-dir={self.tmp / 'chrome-profile'}",
                f"--screenshot={target}",
                source.as_uri(),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 30
        last_size = 0
        stable_reads = 0
        try:
            while stable_reads < 3:
                if target.exists():
                    size = target.stat().st_size
                    stable_reads = stable_reads + 1 if size == last_size and size > 0 else 0
                    last_size = size
                if process.poll() is not None and stable_reads < 3:
                    raise subprocess.CalledProcessError(process.returncode, process.args)
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(process.args, 30)
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self.image_cache[source] = target
        return target

    def image_flowables(self, inline: dict) -> list[Flowable]:
        content = inline["c"]
        alt = plain_inlines(content[1])
        rel = content[2][0]
        path = self.raster_image(rel)
        image = Image(str(path))
        max_w = self.available_width
        max_h = 112 * mm
        scale = min(max_w / image.imageWidth, max_h / image.imageHeight)
        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        image.hAlign = "CENTER"
        result: list[Flowable] = [Spacer(1, 4), image]
        if alt:
            result.append(Paragraph(escape(alt), self.s["Caption"]))
        return result

    def paragraph(self, inlines: list[dict], style="BodyText") -> list[Flowable]:
        if len(inlines) == 1 and inlines[0]["t"] == "Image":
            return self.image_flowables(inlines[0])
        content = rich_inlines(inlines)
        return [Paragraph(content, self.s[style])] if content.strip() else []

    def list_item(self, blocks: list[dict]) -> ListItem:
        flows: list[Flowable] = []
        for block in blocks:
            flows.extend(self.block(block, in_list=True))
        if not flows:
            flows = [Paragraph("", self.s["BodyText"])]
        return ListItem(flows, leftIndent=8, value=None)

    @staticmethod
    def _cell_content(cell) -> list:
        return cell["c"] if isinstance(cell, dict) else cell

    @classmethod
    def _cell_blocks(cls, cell) -> list[dict]:
        return cls._cell_content(cell)[4]

    @staticmethod
    def _row_cells(row) -> list:
        return row["c"][1] if isinstance(row, dict) else row[1]

    def cell_paragraph(self, cell: dict, head: bool = False) -> Paragraph:
        parts: list[str] = []
        for block in self._cell_blocks(cell):
            if block["t"] in {"Para", "Plain"}:
                parts.append(rich_inlines(block["c"]))
            elif block["t"] == "CodeBlock":
                parts.append(f'<font face="Courier">{escape(block["c"][1])}</font>')
            elif block["t"] in {"BulletList", "OrderedList"}:
                for item in block["c"] if block["t"] == "BulletList" else block["c"][1]:
                    text = " ".join(
                        rich_inlines(b["c"]) for b in item if b["t"] in {"Para", "Plain"}
                    )
                    parts.append(f"- {text}")
        return Paragraph("<br/>".join(parts), self.s["CellHead" if head else "Cell"])

    @staticmethod
    def table_rows(table_block: dict) -> tuple[list, list]:
        content = table_block["c"]
        head = content[3]["c"] if isinstance(content[3], dict) else content[3]
        head_rows = head[1]
        body_rows: list = []
        for body in content[4]:
            body_content = body["c"] if isinstance(body, dict) else body
            body_rows.extend(body_content[2])
            body_rows.extend(body_content[3])
        foot = content[5]["c"] if isinstance(content[5], dict) else content[5]
        foot_rows = foot[1]
        return head_rows, body_rows + foot_rows

    def table(self, block: dict) -> list[Flowable]:
        head_rows, body_rows = self.table_rows(block)
        rows = head_rows + body_rows
        if not rows:
            return []
        parsed: list[list[Paragraph]] = []
        raw_lengths: list[list[int]] = []
        span_cmds: list[tuple] = []
        for r_idx, row in enumerate(rows):
            parsed_row: list[Paragraph] = []
            lengths: list[int] = []
            c_idx = 0
            for cell in self._row_cells(row):
                cell_content = self._cell_content(cell)
                row_span, col_span = int(cell_content[2]), int(cell_content[3])
                para = self.cell_paragraph(cell, head=r_idx < len(head_rows))
                parsed_row.append(para)
                lengths.append(max(4, len(para.getPlainText())))
                for _ in range(col_span - 1):
                    parsed_row.append(Paragraph("", self.s["Cell"]))
                    lengths.append(1)
                if row_span > 1 or col_span > 1:
                    span_cmds.append(("SPAN", (c_idx, r_idx), (c_idx + col_span - 1, r_idx + row_span - 1)))
                c_idx += col_span
            parsed.append(parsed_row)
            raw_lengths.append(lengths)
        ncols = max(len(row) for row in parsed)
        for row, lengths in zip(parsed, raw_lengths):
            while len(row) < ncols:
                row.append(Paragraph("", self.s["Cell"]))
                lengths.append(1)
        weights = []
        for col in range(ncols):
            max_len = max(lengths[col] for lengths in raw_lengths)
            weights.append(min(8.5, max(2.7, math.sqrt(max_len))))
        total = sum(weights)
        col_widths = [self.available_width * weight / total for weight in weights]
        table = Table(parsed, colWidths=col_widths, repeatRows=len(head_rows), hAlign="LEFT")
        commands = [
            ("BACKGROUND", (0, 0), (-1, len(head_rows) - 1), INDIGO),
            ("TEXTCOLOR", (0, 0), (-1, len(head_rows) - 1), colors.white),
            ("BACKGROUND", (0, len(head_rows)), (-1, -1), colors.white),
            ("ROWBACKGROUNDS", (0, len(head_rows)), (-1, -1), [colors.white, colors.HexColor("#F8FAFD")]),
            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        commands.extend(span_cmds)
        table.setStyle(TableStyle(commands))
        return [Spacer(1, 3), table, Spacer(1, 9)]

    def block(self, block: dict, in_list: bool = False) -> list[Flowable]:
        tag, content = block["t"], block.get("c")
        if tag in {"Para", "Plain"}:
            return self.paragraph(content)
        if tag == "Header":
            level, _, inlines = content
            if level == 1:
                flows: list[Flowable] = []
                if self.h1_seen and not in_list:
                    flows.append(PageBreak())
                self.h1_seen = True
                flows.append(Paragraph(rich_inlines(inlines), self.s["Heading1"]))
                flows.append(HRFlowable(width="100%", thickness=1.2, color=INDIGO, spaceAfter=10))
                return flows
            return [Paragraph(rich_inlines(inlines), self.s[f"Heading{min(level, 3)}"])]
        if tag == "BlockQuote":
            inner: list[Flowable] = []
            for child in content:
                if child["t"] in {"Para", "Plain"}:
                    inner.extend(self.paragraph(child["c"], "Quote"))
                else:
                    inner.extend(self.block(child))
            box = Table([[inner]], colWidths=[self.available_width - 2])
            box.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), PALE),
                        ("LINEBEFORE", (0, 0), (0, -1), 4, INDIGO),
                        ("BOX", (0, 0), (-1, -1), 0.4, BORDER),
                        ("LEFTPADDING", (0, 0), (-1, -1), 12),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            return [Spacer(1, 4), box, Spacer(1, 8)]
        if tag == "BulletList":
            items = [self.list_item(item) for item in content]
            return [ListFlowable(items, bulletType="bullet", leftIndent=18, bulletFontName="Helvetica", bulletFontSize=7, spaceAfter=6)]
        if tag == "OrderedList":
            attrs, raw_items = content
            items = [self.list_item(item) for item in raw_items]
            return [ListFlowable(items, bulletType="1", start=attrs[0], leftIndent=20, bulletFontName="Helvetica", bulletFontSize=8, spaceAfter=6)]
        if tag == "CodeBlock":
            return [Preformatted(content[1], self.s["CodeBlock"])]
        if tag == "HorizontalRule":
            return [HRFlowable(width="100%", thickness=0.6, color=BORDER, spaceBefore=8, spaceAfter=10)]
        if tag == "Table":
            return self.table(block)
        if tag in {"Div", "Figure"}:
            children = content[1] if tag == "Div" else content[2]
            flows: list[Flowable] = []
            for child in children:
                flows.extend(self.block(child))
            return flows
        if tag == "RawBlock":
            return []
        return []


def cover_story(ast: dict, renderer: ReportRenderer) -> list[Flowable]:
    meta = ast.get("meta", {})
    title = meta_text(meta, "title", "Hibiki-Zero MLX")
    subtitle = meta_text(meta, "subtitle")
    date = meta_text(meta, "date")
    hero_inline = {"t": "Image", "c": [["", [], []], [{"t": "Str", "c": ""}], ["assets/hero.svg", ""]]}
    hero = renderer.image_flowables(hero_inline)[1]
    hero.drawWidth = A4[0] - 48 * mm
    hero.drawHeight = hero.drawWidth * 520 / 1200
    stats = Table(
        [
            [Paragraph("<b>3.0x RT</b><br/><font size=8 color='#65708C'>3B q4 on M4 Pro</font>", renderer.s["BodyText"]), Paragraph("<b>19.61 chrF</b><br/><font size=8 color='#65708C'>VI phase-1 val128</font>", renderer.s["BodyText"])],
            [Paragraph("<b>1.13 GB</b><br/><font size=8 color='#65708C'>1B phone LM</font>", renderer.s["BodyText"]), Paragraph("<b>80 ms</b><br/><font size=8 color='#65708C'>streaming frame budget</font>", renderer.s["BodyText"])],
        ],
        colWidths=[(A4[0] - 56 * mm) / 2] * 2,
        rowHeights=[19 * mm, 19 * mm],
    )
    stats.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    return [
        Spacer(1, 8 * mm),
        hero,
        Spacer(1, 8 * mm),
        Paragraph(escape(title), renderer.s["CoverTitle"]),
        Paragraph(escape(subtitle), renderer.s["CoverSubtitle"]),
        stats,
        Spacer(1, 10 * mm),
        Paragraph(
            f"<b>{escape(date)}</b><br/>Repository evidence at commit <font face='Courier'>92f31b8</font>. Measured, derived, and projected results are labeled separately.",
            renderer.s["CoverSubtitle"],
        ),
        NextPageTemplate("body"),
        PageBreak(),
    ]


def build() -> Path:
    ast = pandoc_ast()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = ROOT / "tmp" / "pdfs"
    tmp_root.mkdir(parents=True, exist_ok=True)
    stylebook = styles()
    with tempfile.TemporaryDirectory(prefix="hibiki-report-", dir=tmp_root) as tmp_name:
        renderer = ReportRenderer(stylebook, Path(tmp_name))
        story = cover_story(ast, renderer)
        toc = TableOfContents()
        toc.levelStyles = [
            ParagraphStyle(name="TOC1", fontName="Helvetica-Bold", fontSize=10.5, leading=15, leftIndent=0, textColor=INK, spaceBefore=4),
            ParagraphStyle(name="TOC2", fontName="Helvetica", fontSize=8.5, leading=12, leftIndent=16, textColor=MUTED),
        ]
        story.extend(
            [
                Paragraph("Contents", stylebook["Heading1"]),
                HRFlowable(width="100%", thickness=1.2, color=INDIGO, spaceAfter=12),
                toc,
                PageBreak(),
            ]
        )
        for block in ast["blocks"]:
            story.extend(renderer.block(block))
        doc = TechnicalDocTemplate(
            str(OUTPUT),
            pagesize=A4,
            title=meta_text(ast.get("meta", {}), "title", "Hibiki-Zero MLX"),
            author=meta_text(ast.get("meta", {}), "author", "Hibiki-Zero MLX project"),
            subject="Technical report for the Hibiki-Zero MLX simultaneous speech translation project",
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=19 * mm,
            bottomMargin=17 * mm,
        )
        doc.multiBuild(story)
    return OUTPUT


if __name__ == "__main__":
    print(f"Wrote {build()}")
