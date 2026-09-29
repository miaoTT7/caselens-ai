"""Structure-aware chunking for parsed CaseLens documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field

from api.document_parser import DocumentBlock, ParsedDocument


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]|[^\w\s]", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？；;])\s+|\n+")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    type: str
    order: int
    token_count: int
    section_path: list[str]
    source_file: str
    source_format: str
    source_block_orders: list[int]
    page_numbers: list[int] = field(default_factory=list)
    positions: list[list[float]] = field(default_factory=list)
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    email_metadata: dict = field(default_factory=dict)
    context_above: str = ""
    context_below: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def count_tokens(text: str) -> int:
    """Return a deterministic local token estimate without loading an LLM."""
    return len(_TOKEN_RE.findall(text or ""))


def _take_tokens(text: str, limit: int) -> tuple[str, str]:
    matches = list(_TOKEN_RE.finditer(text))
    if len(matches) <= limit:
        return text, ""
    cut = matches[limit].start()
    return text[:cut].rstrip(), text[cut:].lstrip()


def _split_text(text: str, limit: int) -> list[str]:
    if count_tokens(text) <= limit:
        return [text] if text.strip() else []
    sentences = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
    if len(sentences) <= 1:
        pieces = []
        remaining = text.strip()
        while remaining:
            head, remaining = _take_tokens(remaining, limit)
            if not head:
                break
            pieces.append(head)
        return pieces

    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if count_tokens(sentence) > limit:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_split_text(sentence, limit))
            continue
        candidate = f"{current} {sentence}".strip()
        if current and count_tokens(candidate) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _section_prefix(section: list[str]) -> str:
    return " > ".join(value for value in section if value)


def _source_data(blocks: list[DocumentBlock]) -> tuple[list[int], list[int], list[list[float]]]:
    orders = [block.order for block in blocks]
    pages = sorted({block.page_number for block in blocks if block.page_number is not None})
    positions = [position for block in blocks for position in block.positions]
    return orders, pages, positions


def _email_metadata(document: ParsedDocument) -> dict:
    if document.format != "eml":
        return {}
    return {key: document.metadata.get(key, "") for key in ("subject", "sender", "recipient", "cc", "date", "message_id")}


def _chunk_id(document: ParsedDocument, chunk_type: str, order: int, source_orders: list[int], text: str) -> str:
    identity = f"{document.name}\0{document.format}\0{chunk_type}\0{order}\0{source_orders}\0{text}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _make_chunk(
    document: ParsedDocument,
    text: str,
    chunk_type: str,
    order: int,
    blocks: list[DocumentBlock],
    section: list[str],
    **kwargs,
) -> Chunk:
    source_orders, pages, positions = _source_data(blocks)
    return Chunk(
        chunk_id=_chunk_id(document, chunk_type, order, source_orders, text),
        text=text,
        type=chunk_type,
        order=order,
        token_count=count_tokens(text),
        section_path=section.copy(),
        source_file=document.name,
        source_format=document.format,
        source_block_orders=source_orders,
        page_numbers=pages,
        positions=positions,
        email_metadata=_email_metadata(document),
        **kwargs,
    )


def _render_text(section: list[str], body: str, document: ParsedDocument) -> str:
    context = []
    heading = _section_prefix(section)
    if heading:
        context.append(heading)
    if document.format == "eml":
        metadata = _email_metadata(document)
        context.extend(
            f"{label}: {metadata[key]}"
            for label, key in (("Subject", "subject"), ("From", "sender"), ("To", "recipient"), ("Date", "date"))
            if metadata.get(key)
        )
    context.append(body)
    return "\n".join(context)


def _chunk_table(
    document: ParsedDocument,
    block: DocumentBlock,
    max_tokens: int,
    start_order: int,
    title_blocks: list[DocumentBlock],
) -> list[Chunk]:
    source_blocks = [*title_blocks, block]
    rows = block.rows or []
    if not rows:
        text = _render_text(block.section, block.text, document)
        return [_make_chunk(document, text, "table", start_order, source_blocks, block.section)]

    header = rows[0]
    data_rows = rows[1:]
    groups: list[tuple[list[list[str]], int, int]] = []
    current: list[list[str]] = []
    current_start = 1
    for offset, row in enumerate(data_rows, start=1):
        candidate = [header, *current, row]
        rendered = "\n".join(" | ".join(item) for item in candidate)
        full_text = _render_text(block.section, rendered, document)
        if current and count_tokens(full_text) > max_tokens:
            groups.append(([header, *current], current_start, offset - 1))
            current = [row]
            current_start = offset
        else:
            current.append(row)
    if current or not data_rows:
        groups.append(([header, *current], current_start, max(current_start, len(data_rows))))

    chunks = []
    source_start = block.row_start or 1
    for index, (group, data_start, data_end) in enumerate(groups):
        rendered = "\n".join(" | ".join(row) for row in group)
        text = _render_text(block.section, rendered, document)
        row_start = source_start if not data_rows else source_start + data_start
        row_end = source_start if not data_rows else source_start + data_end
        chunks.append(
            _make_chunk(
                document,
                text,
                "spreadsheet" if block.sheet else "table",
                start_order + index,
                source_blocks,
                block.section,
                sheet=block.sheet,
                row_start=row_start,
                row_end=row_end,
            )
        )
    return chunks


def chunk_document(
    document: ParsedDocument,
    max_tokens: int = 512,
    overlap_tokens: int = 0,
    table_context_tokens: int = 64,
) -> list[Chunk]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be between 0 and max_tokens")

    chunks: list[Chunk] = []
    pending_blocks: list[DocumentBlock] = []
    pending_section: list[str] = []
    active_title_blocks: list[DocumentBlock] = []

    def append_text(text: str, blocks: list[DocumentBlock], section: list[str]) -> None:
        source_blocks = [*active_title_blocks, *blocks]
        prefix_only = _render_text(section, "", document).strip()
        body_limit = max(1, max_tokens - count_tokens(prefix_only))
        for piece in _split_text(text, body_limit):
            rendered = _render_text(section, piece, document)
            chunks.append(
                _make_chunk(
                    document,
                    rendered,
                    "email" if document.format == "eml" else "text",
                    len(chunks),
                    source_blocks,
                    section,
                )
            )

    def flush_pending() -> None:
        nonlocal pending_blocks, pending_section
        if pending_blocks:
            append_text("\n".join(block.text for block in pending_blocks), pending_blocks, pending_section)
        pending_blocks = []
        pending_section = []

    for block in document.blocks:
        if block.type == "title":
            flush_pending()
            level = block.level or 1
            active_title_blocks = active_title_blocks[: level - 1] + [block]
            continue
        if block.type == "table":
            flush_pending()
            chunks.extend(_chunk_table(document, block, max_tokens, len(chunks), active_title_blocks))
            continue

        rendered = _render_text(block.section, block.text, document)
        pending_text = "\n".join(item.text for item in pending_blocks)
        combined = _render_text(block.section, f"{pending_text}\n{block.text}".strip(), document)
        crosses_column = (
            pending_blocks
            and pending_blocks[-1].column_id is not None
            and block.column_id is not None
            and pending_blocks[-1].column_id != block.column_id
        )
        if pending_blocks and (block.section != pending_section or crosses_column or count_tokens(combined) > max_tokens):
            flush_pending()
        if count_tokens(rendered) > max_tokens:
            flush_pending()
            append_text(block.text, [block], block.section)
        else:
            if not pending_blocks:
                pending_section = block.section.copy()
            pending_blocks.append(block)
    flush_pending()

    if overlap_tokens:
        previous_text = ""
        for chunk in chunks:
            if chunk.type not in {"text", "email"}:
                previous_text = ""
                continue
            if previous_text:
                matches = list(_TOKEN_RE.finditer(previous_text))
                if matches:
                    start = matches[max(0, len(matches) - overlap_tokens)].start()
                    prefix = previous_text[start:].strip()
                    if prefix:
                        chunk.text = f"{prefix}\n{chunk.text}"
                        chunk.token_count = count_tokens(chunk.text)
            previous_text = chunk.text

    if table_context_tokens:
        for index, chunk in enumerate(chunks):
            if chunk.type not in {"table", "spreadsheet"}:
                continue
            if index > 0 and chunks[index - 1].type in {"text", "email"}:
                words = list(_TOKEN_RE.finditer(chunks[index - 1].text))
                start = words[max(0, len(words) - table_context_tokens)].start() if words else 0
                chunk.context_above = chunks[index - 1].text[start:]
            if index + 1 < len(chunks) and chunks[index + 1].type in {"text", "email"}:
                chunk.context_below, _ = _take_tokens(chunks[index + 1].text, table_context_tokens)

    for index, chunk in enumerate(chunks):
        chunk.order = index
        chunk.chunk_id = _chunk_id(document, chunk.type, index, chunk.source_block_orders, chunk.text)
    return chunks
