"""Structured document extraction for the claim API.

Parsing stops at document blocks. Search and claim decisions consume the text
but do not control how source structure is represented here.
"""

import csv
from dataclasses import asdict, dataclass, field
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path

import fitz
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from openpyxl import load_workbook


@dataclass
class DocumentBlock:
    text: str
    type: str  # title, paragraph, list, or table
    order: int
    section: list[str] = field(default_factory=list)
    page_number: int | None = None
    positions: list[list[float]] = field(default_factory=list)
    level: int | None = None
    rows: list[list[str]] | None = None
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None


@dataclass
class ParsedDocument:
    name: str
    format: str
    blocks: list[DocumentBlock]
    page_count: int | None = None
    outline: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    return " ".join(text.split())


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data)

    def text(self) -> str:
        return _clean(" ".join(self.parts))


def parse_pdf(path: str | Path) -> ParsedDocument:
    path = Path(path)
    blocks: list[DocumentBlock] = []
    outline: list[dict] = []
    headings: list[str] = []
    with fitz.open(path) as pdf:
        for level, title, page in pdf.get_toc():
            outline.append({"title": title, "level": level, "page_number": page})

        for page_number, page in enumerate(pdf, start=1):
            page_blocks = []
            for raw in page.get_text("dict", sort=True)["blocks"]:
                if raw.get("type") != 0:
                    continue
                lines = raw.get("lines", [])
                text = _clean("\n".join("".join(span["text"] for span in line["spans"]) for line in lines))
                if not text:
                    continue
                font_sizes = [span["size"] for line in lines for span in line["spans"] if span["text"].strip()]
                page_blocks.append((raw, text, max(font_sizes, default=0)))

            # Font size is only a hint; PDF bookmarks remain the explicit outline.
            body_sizes = sorted(size for _, _, size in page_blocks)
            # Use the lower median so a page containing only one heading and
            # one body block does not treat the heading size as body text.
            body_size = body_sizes[(len(body_sizes) - 1) // 2] if body_sizes else 0
            for raw, text, size in page_blocks:
                is_title = len(text) <= 120 and size > body_size * 1.2
                if is_title:
                    headings = [text]
                blocks.append(
                    DocumentBlock(
                        text=text,
                        type="title" if is_title else "paragraph",
                        order=len(blocks),
                        section=headings.copy(),
                        page_number=page_number,
                        positions=[[round(float(value), 2) for value in raw["bbox"]]],
                    )
                )
        page_count = len(pdf)
    return ParsedDocument(path.name, "pdf", blocks, page_count, outline)


def _heading_level(paragraph: Paragraph) -> int | None:
    style = paragraph.style
    name = style.name if style else ""
    if name.lower().startswith("heading "):
        suffix = name.split()[-1]
        if suffix.isdigit():
            return int(suffix)
    return None


def _is_list(paragraph: Paragraph) -> bool:
    style = paragraph.style
    if style and style.name.lower().startswith("list"):
        return True
    return paragraph._p.pPr is not None and paragraph._p.pPr.numPr is not None


def parse_docx(path: str | Path) -> ParsedDocument:
    path = Path(path)
    document = Document(path)
    blocks: list[DocumentBlock] = []
    headings: list[str] = []
    outline: list[dict] = []

    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = _clean(item.text)
            if not text:
                continue
            level = _heading_level(item)
            if level is not None:
                headings = headings[: level - 1] + [text]
                outline.append({"title": text, "level": level})
            block_type = "title" if level is not None else "list" if _is_list(item) else "paragraph"
            blocks.append(DocumentBlock(text, block_type, len(blocks), headings.copy(), level=level))
        elif isinstance(item, Table):
            rows = [[_clean(cell.text) for cell in row.cells] for row in item.rows]
            if not any(any(cell for cell in row) for row in rows):
                continue
            text = "\n".join(" | ".join(row) for row in rows)
            blocks.append(DocumentBlock(text, "table", len(blocks), headings.copy(), rows=rows))

    return ParsedDocument(path.name, "docx", blocks, outline=outline)


def _email_body(message) -> str:
    plain_parts: list[str] = []
    html_parts: list[str] = []
    parts = message.walk() if message.is_multipart() else [message]
    for part in parts:
        if part.get_content_disposition() == "attachment":
            continue
        content_type = part.get_content_type()
        if content_type not in {"text/plain", "text/html"}:
            continue
        try:
            content = part.get_content()
        except (LookupError, UnicodeDecodeError):
            payload = part.get_payload(decode=True) or b""
            content = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        if content_type == "text/plain":
            plain_parts.append(content)
        else:
            html_parts.append(content)
    if plain_parts:
        return _clean("\n".join(plain_parts))
    extractor = _HTMLTextExtractor()
    extractor.feed("\n".join(html_parts))
    return extractor.text()


def parse_eml(path: str | Path) -> ParsedDocument:
    path = Path(path)
    with path.open("rb") as source:
        message = BytesParser(policy=policy.default).parse(source)
    metadata = {
        "subject": str(message.get("subject", "")),
        "sender": str(message.get("from", "")),
        "recipient": str(message.get("to", "")),
        "cc": str(message.get("cc", "")),
        "date": str(message.get("date", "")),
        "message_id": str(message.get("message-id", "")),
    }
    body = _email_body(message)
    metadata["body"] = body
    blocks = [DocumentBlock(body, "paragraph", 0, [metadata["subject"]])] if body else []
    return ParsedDocument(path.name, "eml", blocks, metadata=metadata)


def _cell_text(value) -> str:
    if value is None:
        return ""
    return _clean(str(value))


def _trim_empty_rows(rows: list[list[str]]) -> tuple[list[list[str]], int]:
    populated = [index for index, row in enumerate(rows) if any(row)]
    if not populated:
        return [], 1
    start, end = populated[0], populated[-1] + 1
    return rows[start:end], start + 1


def _table_block(rows: list[list[str]], order: int, sheet: str, row_start: int = 1) -> DocumentBlock:
    text = "\n".join(" | ".join(row) for row in rows)
    return DocumentBlock(
        text,
        "table",
        order,
        [sheet],
        rows=rows,
        sheet=sheet,
        row_start=row_start,
        row_end=row_start + len(rows) - 1,
    )


def parse_xlsx(path: str | Path) -> ParsedDocument:
    path = Path(path)
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet_names = workbook.sheetnames.copy()
    blocks: list[DocumentBlock] = []
    try:
        for worksheet in workbook.worksheets:
            rows = [[_cell_text(value) for value in row] for row in worksheet.iter_rows(values_only=True)]
            rows, row_start = _trim_empty_rows(rows)
            if rows:
                blocks.append(_table_block(rows, len(blocks), worksheet.title, row_start))
    finally:
        workbook.close()
    return ParsedDocument(path.name, "xlsx", blocks, metadata={"sheets": sheet_names})


def parse_csv(path: str | Path) -> ParsedDocument:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        sample = source.read(4096)
        source.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        rows = [[_clean(cell) for cell in row] for row in csv.reader(source, dialect)]
    rows, row_start = _trim_empty_rows(rows)
    sheet = path.stem
    blocks = [_table_block(rows, 0, sheet, row_start)] if rows else []
    return ParsedDocument(path.name, "csv", blocks, metadata={"sheets": [sheet]})


def parse_document(path: str | Path) -> ParsedDocument:
    path = Path(path)
    parsers = {
        ".pdf": parse_pdf,
        ".docx": parse_docx,
        ".eml": parse_eml,
        ".xlsx": parse_xlsx,
        ".csv": parse_csv,
    }
    try:
        parser = parsers[path.suffix.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported document format: {path.suffix or '<none>'}") from error
    return parser(path)
