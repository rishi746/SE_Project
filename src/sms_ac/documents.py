"""Bounded, non-executing DOCX ingestion and readable Word report export."""
from __future__ import annotations

import copy
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
MAX_DOCUMENT_CHARS = 100_000
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class DocumentError(ValueError):
    """The document is unsupported, unsafe to parse, or exceeds a stated limit."""


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    block_count: int
    warnings: tuple[str, ...] = ()


def _xml(data: bytes, name: str) -> ET.Element:
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise DocumentError(f"DTD/entity declarations are unsupported: {name}")
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise DocumentError(f"Malformed XML in {name}") from exc


def _paragraph_text(element: ET.Element) -> str:
    result = []
    for node in element.iter():
        if node.tag == W + "t":
            result.append(node.text or "")
        elif node.tag == W + "tab":
            result.append("\t")
        elif node.tag in {W + "br", W + "cr"}:
            result.append("\n")
    return "".join(result).strip()


def _body_blocks(element: ET.Element) -> list[str]:
    blocks: list[str] = []
    for child in element:
        if child.tag == W + "p":
            text = _paragraph_text(child)
            if text:
                blocks.append(text)
        elif child.tag == W + "tbl":
            rows = []
            for row in child.findall(W + "tr"):
                cells = [" / ".join(_body_blocks(cell)) for cell in row.findall(W + "tc")]
                rows.append(" | ".join(cells))
            if rows:
                blocks.append("[Table]\n" + "\n".join(rows) + "\n[/Table]")
        elif child.tag in {W + "sdt", W + "sdtContent", W + "customXml"}:
            blocks.extend(_body_blocks(child))
    return blocks


def read_design_docx(source: str | Path | bytes, *, filename: str | None = None,
                     max_chars: int = MAX_DOCUMENT_CHARS) -> ExtractedDocument:
    """Read body paragraphs and tables in their original order without fetching relationships.

    Rejects macros, embedded objects, XML entities, tracked changes, text boxes and altChunk
    content. Images, headers and footers are declared as omitted rather than treated as read.
    Oversize content fails explicitly: the SDLC model must never see a silently clipped design.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if isinstance(source, bytes):
        name, data = filename or "upload.docx", source
    else:
        path = Path(source)
        name = path.name
        if path.stat().st_size > MAX_FILE_BYTES:
            raise DocumentError("DOCX exceeds the 8 MiB compressed file limit")
        data = path.read_bytes()
    if Path(name).suffix.lower() != ".docx":
        raise DocumentError("Only .docx is supported. Convert legacy .doc or macro-enabled files first.")
    if len(data) > MAX_FILE_BYTES:
        raise DocumentError("DOCX exceeds the 8 MiB compressed file limit")
    warnings: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as package:
            entries = package.infolist()
            if len(entries) > 2048 or sum(x.file_size for x in entries) > MAX_UNCOMPRESSED_BYTES:
                raise DocumentError("DOCX ZIP exceeds entry count or 32 MiB expansion limit")
            names = [x.filename for x in entries]
            if len(set(names)) != len(names):
                raise DocumentError("DOCX ZIP contains duplicate member names")
            for item in entries:
                normalized = item.filename.replace("\\", "/")
                if normalized.startswith("/") or ".." in PurePosixPath(normalized).parts:
                    raise DocumentError("DOCX ZIP contains an unsafe member path")
                if item.flag_bits & 1:
                    raise DocumentError("Encrypted DOCX ZIP entries are unsupported")
                if item.file_size > 16 * 1024 * 1024 or item.file_size / max(item.compress_size, 1) > 1000:
                    raise DocumentError("DOCX ZIP member exceeds expansion limits")
                low = normalized.lower()
                if "vbaproject" in low or "/embeddings/" in low or low.endswith((".exe", ".dll")):
                    raise DocumentError("Macro-enabled files and embedded executable/OLE objects are unsupported")
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise DocumentError("Not a supported Word DOCX package")
            content_types = package.read("[Content_Types].xml")
            _xml(content_types, "[Content_Types].xml")
            if b"macroEnabled" in content_types or b"vbaProject" in content_types:
                raise DocumentError("Macro-enabled document content is unsupported")
            root = _xml(package.read("word/document.xml"), "word/document.xml")
            unsupported = {"ins": "tracked changes", "del": "tracked changes",
                           "txbxContent": "text boxes", "altChunk": "imported HTML or external content"}
            for tag, description in unsupported.items():
                if root.find(".//" + W + tag) is not None:
                    raise DocumentError(f"Document contains {description}; flatten or accept changes before upload")
            body = root.find(W + "body")
            if body is None:
                raise DocumentError("DOCX has no document body")
            blocks = _body_blocks(body)
            if root.find(".//" + W + "drawing") is not None or root.find(".//" + W + "pict") is not None:
                warnings.append("Images and diagrams are not interpreted; include their design meaning in body text.")
            for part in names:
                if re.match(r"word/(header\d*|footer\d*|footnotes|endnotes|comments)\.xml$", part):
                    part_root = _xml(package.read(part), part)
                    if any((n.text or "").strip() for n in part_root.iter(W + "t")):
                        warnings.append(f"Text in {part} was omitted; move decision evidence to body paragraphs or tables.")
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise DocumentError("Invalid or unsupported DOCX ZIP package") from exc
    text = "\n\n".join(blocks)
    if not text:
        raise DocumentError("Document body has no readable text")
    if len(text) > max_chars:
        raise DocumentError(f"Document has {len(text):,} characters; limit is {max_chars:,}. Split or shorten it explicitly.")
    return ExtractedDocument(text, len(blocks), tuple(warnings))


def _add_inline(paragraph, text: str) -> None:
    """Readable basic Markdown; links retain visible URLs in exported Word text."""
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r"\1 (\2)", text)
    for index, fragment in enumerate(re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text)):
        run = paragraph.add_run(fragment[2:-2] if fragment.startswith("**") else
                                fragment[1:-1] if fragment.startswith("`") else fragment)
        if fragment.startswith("**"):
            run.bold = True
        if fragment.startswith("`"):
            run.font.name = "Consolas"


def markdown_to_docx(markdown: str, destination: str | Path, *, title: str,
                     demo: bool = False, template: str | Path | None = None,
                     preserve_heading_numbers: bool = False, include_toc: bool = False) -> Path:
    """Export generated Markdown. Optional reference retains its style and page definitions.

    The exporter is text-oriented: fenced diagrams remain readable monospaced source rather
    than pretending they were rendered. Generated documents still need a visual review.
    """
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_TAB_ALIGNMENT, WD_TAB_LEADER

    doc = Document(str(template)) if template else Document()
    if template:
        for child in list(doc.element.body):
            if child.tag != qn("w:sectPr"):
                doc.element.body.remove(child)
    else:
        for section in doc.sections:
            section.top_margin = section.bottom_margin = Inches(0.75)
            section.left_margin = section.right_margin = Inches(0.8)
        normal = doc.styles["Normal"]
        normal.font.name, normal.font.size = "Calibri", Pt(11)
        normal.paragraph_format.space_after = Pt(6)
        normal.paragraph_format.line_spacing = 1.08
        for name in ("Title", "Heading 1", "Heading 2", "Heading 3"):
            doc.styles[name].font.color.rgb = RGBColor(0, 0, 0)
    doc.core_properties.author = "SMS AC"
    doc.core_properties.title = title
    title_paragraph = doc.add_paragraph(re.sub(r"[^\w\s]", " ", title).strip(), style="Title")
    if include_toc:
        for properties in (doc.styles['Title'].element.find(qn('w:pPr')), title_paragraph._p.find(qn('w:pPr'))):
            if properties is not None:
                border = properties.find(qn('w:pBdr'))
                if border is not None:
                    properties.remove(border)
        doc.styles['Title'].font.color.rgb = RGBColor(0, 0, 0)
        contents = doc.add_paragraph('Table of Contents')
        contents.runs[0].bold = True
        contents.runs[0].font.size = Pt(16)
        run = contents.add_run()
        begin = OxmlElement('w:fldChar'); begin.set(qn('w:fldCharType'), 'begin')
        instruction = OxmlElement('w:instrText'); instruction.set(qn('xml:space'), 'preserve')
        instruction.text = ' TOC \\o "1-3" \\h \\z \\u '
        separate = OxmlElement('w:fldChar'); separate.set(qn('w:fldCharType'), 'separate')
        run._r.extend((begin, instruction, separate))
        first_title = False
        toc_headings = []
        toc_in_code = False
        for candidate in markdown.splitlines():
            if candidate.startswith('```'):
                toc_in_code = not toc_in_code
                continue
            if toc_in_code:
                continue
            match = re.match(r'^(#{1,6})\s+(.+)$', candidate)
            if not match:
                continue
            if len(match[1]) == 1 and not first_title:
                first_title = True
                continue
            toc_headings.append((min(max(len(match[1]) - 1, 1), 3), match[2]))
        width = doc.sections[0].page_width - doc.sections[0].left_margin - doc.sections[0].right_margin
        last = contents
        for number, (level, label) in enumerate(toc_headings, 1):
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.left_indent = Inches(.18 * (level - 1))
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.tab_stops.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
            link = OxmlElement('w:hyperlink'); link.set(qn('w:anchor'), f'srs_{number}')
            link_run = OxmlElement('w:r'); text_node = OxmlElement('w:t'); text_node.text = label
            link_run.append(text_node); link.append(link_run); paragraph._p.append(link)
            paragraph.add_run('\t')
            field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), f' PAGEREF srs_{number} \\h ')
            paragraph._p.append(field)
            last = paragraph
        end = OxmlElement('w:fldChar'); end.set(qn('w:fldCharType'), 'end'); last.add_run()._r.append(end)
        update = OxmlElement('w:updateFields'); update.set(qn('w:val'), 'true'); doc.settings.element.append(update)
    heading_number = 0
    if demo:
        p = doc.add_paragraph()
        p.add_run("DEMONSTRATION FIXTURE ").bold = True
        p.add_run("This deterministic example is not output from a trained model.")
    lines, index, in_code = markdown.splitlines(), 0, False
    first_title_skipped = False
    while index < len(lines):
        line = lines[index].rstrip()
        index += 1
        if line.startswith("```"):
            in_code = not in_code
            continue
        if not line.strip():
            continue
        if in_code:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(0)
            p.add_run(line).font.name = "Consolas"
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level, text = len(heading[1]), heading[2]
            if level == 1 and not first_title_skipped:
                first_title_skipped = True
                if not include_toc:
                    doc.add_paragraph(text, style='Subtitle' if 'Subtitle' in doc.styles else 'Normal')
                continue
            heading_pattern = r"[^\w\s.]" if preserve_heading_numbers else r"[^\w\s]"
            paragraph = doc.add_paragraph(re.sub(heading_pattern, " ", text).strip(), style=f"Heading {min(max(level - 1, 1), 3)}")
            if include_toc:
                if heading_number == 0:
                    paragraph.paragraph_format.page_break_before = True
                heading_number += 1
                bookmark = OxmlElement('w:bookmarkStart'); bookmark.set(qn('w:id'), str(heading_number)); bookmark.set(qn('w:name'), f'srs_{heading_number}')
                finish = OxmlElement('w:bookmarkEnd'); finish.set(qn('w:id'), str(heading_number))
                paragraph._p.insert(1 if paragraph._p.pPr is not None else 0, bookmark); paragraph._p.append(finish)
            continue
        if line.startswith("|") and index < len(lines) and re.match(r"^\|[\s:|\-]+\|\s*$", lines[index]):
            rows = [[c.strip() for c in line.strip().strip("|").split("|")]]
            index += 1
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                rows.append([c.strip() for c in lines[index].strip().strip("|").split("|")])
                index += 1
            columns = len(rows[0])
            table = doc.add_table(rows=0, cols=columns)
            if "Table Grid" in doc.styles:
                table.style = "Table Grid"
            for row_idx, values in enumerate(rows):
                cells = table.add_row().cells
                for col_idx in range(columns):
                    _add_inline(cells[col_idx].paragraphs[0], values[col_idx] if col_idx < len(values) else "")
                    if row_idx == 0:
                        for run in cells[col_idx].paragraphs[0].runs:
                            run.bold = True
                if row_idx == 0:
                    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
            continue
        bullet = re.match(r"^\s*[-*+]\s+(.+)$", line)
        numbered = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if bullet or numbered:
            style = "List Bullet" if bullet else "List Number"
            p = doc.add_paragraph(style=style if style in doc.styles else None)
            _add_inline(p, (bullet or numbered)[1])
        else:
            _add_inline(doc.add_paragraph(), line)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    return output
