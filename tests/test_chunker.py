"""Basic sanity tests for chunker.py. Expand as pipeline stabilizes."""

from src.indexing.chunker import chunk_page_text


def test_short_page_produces_one_chunk_per_paragraph():
    text = "First paragraph about the seal.\n\nSecond paragraph about the council."
    chunks = chunk_page_text("codex.pdf", page_number=14, text=text)
    assert len(chunks) == 2
    assert chunks[0].page_number == 14
    assert chunks[0].source_file == "codex.pdf"


def test_chunk_ids_are_unique():
    text = "A.\n\nB.\n\nC."
    chunks = chunk_page_text("test.pdf", page_number=1, text=text)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
