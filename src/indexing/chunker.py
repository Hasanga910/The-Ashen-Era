"""
chunker.py
----------
Splits extracted page/paragraph text into retrieval-sized chunks, keeping
source file + page number attached to every chunk (required for citations).

Owner: Search / ML lead

Chunking strategy: chunk by paragraph/section first, only falling back to a
fixed-size sliding window if a paragraph is unusually long. Avoid pure
fixed-character chunking -- it slices figure captions away from their
numbers/tables, which breaks exactly the kind of question in
sample_questions.json (e.g. "what attunement cost is listed...").
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    page_number: int
    text: str
    # Set when this chunk is associated with a specific extracted image/plate
    # (see src/ingest/pdf_ingest.py ExtractedImage.image_id).
    linked_image_id: str | None = None


def chunk_page_text(
    source_file: str,
    page_number: int,
    text: str,
    max_chars: int = 800,
    linked_image_id: str | None = None,
) -> list[Chunk]:
    """Split one page's text into paragraph-based chunks.

    Paragraphs longer than max_chars get split further on sentence boundaries.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []

    for i, para in enumerate(paragraphs):
        if len(para) <= max_chars:
            pieces = [para]
        else:
            pieces = _split_long_paragraph(para, max_chars)

        for j, piece in enumerate(pieces):
            chunk_id = f"{Path(source_file).stem}_p{page_number}_c{i}_{j}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_file=source_file,
                    page_number=page_number,
                    text=piece,
                    linked_image_id=linked_image_id,
                )
            )

    return chunks


def _split_long_paragraph(text: str, max_chars: int) -> list[str]:
    sentences = text.replace("\n", " ").split(". ")
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}." if current else f"{sentence}."
        if len(candidate) > max_chars and current:
            pieces.append(current.strip())
            current = f"{sentence}."
        else:
            current = candidate
    if current.strip():
        pieces.append(current.strip())
    return pieces


def save_chunks_json(chunks: list[Chunk], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)
