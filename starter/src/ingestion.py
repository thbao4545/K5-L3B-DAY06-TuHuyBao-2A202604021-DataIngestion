"""Read and normalize source data."""
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)\s+([A-Z][^\n]{2,80})$", re.MULTILINE)


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while keeping paragraph breaks."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("­", "").replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single newline -> space
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page (1-based) and flag pages with no text."""
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    section = "front_matter"
    for number, pdf_page in enumerate(reader.pages, start=1):
        raw = pdf_page.extract_text() or ""
        for match in _HEADING.finditer(raw):  # headings detected before newlines are flattened
            section = f"{match.group(1)} {match.group(2).strip()}"
        text = normalize_text(raw)
        pages.append(DocumentPage(
            document_id=document_id, source=source, page=number, text=text,
            metadata={"empty": not text, "needs_ocr": not text,
                      "section": section, "char_count": len(text)},
        ))
    return pages
