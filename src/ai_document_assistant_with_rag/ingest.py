from pathlib import Path
from typing import Any

from pypdf import PdfReader
import re


def load_pdf(path: str) -> list[dict[str, Any]]:
    """Extract text from each PDF page as {source, page, text} dicts."""
    source = Path(path).name
    pages: list[dict[str, Any]] = []

    for page_number, page in enumerate(PdfReader(path).pages, start=1):
        raw_text = (page.extract_text() or "").strip()
        text = clean_text(raw_text)
        if not text:
            continue
        pages.append({"source": source, "page": page_number, "text": text})

    return pages


def chunk_pages(
    pages: list[dict[str, Any]], size: int, overlap: int
) -> list[dict[str, Any]]:
    """Split each page's text into overlapping chunks of `size` characters."""
    step = max(size - overlap, 1)
    chunks: list[dict[str, Any]] = []
    index = 0

    for page in pages:
        text = page.get("text") or ""
        source = page.get("source", "")
        page_num = page["page"]

        for start in range(0, len(text), step):
            piece = text[start : start + size]
            if not piece:
                break
            chunks.append(
                {
                    "text": piece,
                    "source": source,
                    "page": page_num,
                    "chunk_id": f"{page['source']}-p{page['page']}-c{index}"
                }
            )
            index += 1
            if start + size >= len(text):
                break

    return chunks

def clean_text(text: str) -> str:
    """Repair PDF line-wrapping artifacts."""
    text = re.sub(r"-\n(?=[a-z])", "", text)        # govern-\nments -> governments
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)    # single newline -> space
    text = re.sub(r"[ \t]+", " ", text)             # collapse repeated spaces
    return text.strip()