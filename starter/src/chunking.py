"""Fixed-size and structure-aware chunking."""
import re

from .models import Chunk, DocumentPage


def _make(page: DocumentPage, strategy: str, index: int, text: str, start: int | None = None) -> Chunk:
    meta = dict(page.metadata)
    meta.update({"strategy": strategy, "chunk_index": index})
    if start is not None:
        meta["char_start"] = start
    return Chunk(chunk_id=f"{page.document_id}:p{page.page}:{strategy}:{index}",
                 document_id=page.document_id, source=page.source, page=page.page,
                 text=text, metadata=meta)


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split text into overlapping character windows."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    text, step = page.text, size - overlap
    chunks = []
    for i, start in enumerate(range(0, max(len(text), 1), step)):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(_make(page, "fixed", i, piece, start))
        if start + size >= len(text):
            break
    return chunks


def _split_oversized(paragraph: str, max_size: int) -> list[str]:
    """Split on sentence boundaries, hard-cutting only as a last resort."""
    parts, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sentence) > max_size:
            if current:
                parts.append(current)
                current = ""
            parts.append(sentence[:max_size])
            sentence = sentence[max_size:]
        if current and len(current) + 1 + len(sentence) > max_size:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group whole paragraphs up to max_size; split oversized paragraphs by sentence."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page.text) if p.strip()]
    units: list[str] = []
    for p in paragraphs:
        units.extend(_split_oversized(p, max_size) if len(p) > max_size else [p])
    groups, current = [], ""
    for unit in units:
        if current and len(current) + 2 + len(unit) > max_size:
            groups.append(current)
            current = unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        groups.append(current)
    return [_make(page, "paragraph", i, g) for i, g in enumerate(groups)]
