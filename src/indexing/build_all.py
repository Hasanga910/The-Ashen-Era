"""
build_all.py
------------
Runs the full pipeline over every file in data/raw/:
  extract -> chunk -> save chunk JSON -> build vector index

Owner: Search / ML lead (integrates work from Data/Extraction lead)

Run this once after adding archive files to data/raw/, and again any time
you add more files or change chunking logic.

Usage:
    python -m src.indexing.build_all
"""

from __future__ import annotations

from pathlib import Path

from src.ingest.docx_ingest import extract_docx
from src.ingest.pdf_ingest import extract_pdf
from src.indexing.chunker import chunk_page_text, save_chunks_json, Chunk
from src.indexing.vector_index import build_index

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
VISUALS_DIR = Path("data/visuals")


def process_pdf(pdf_path: Path) -> list[Chunk]:
    pages = extract_pdf(pdf_path, visuals_out_dir=VISUALS_DIR)
    all_chunks: list[Chunk] = []

    for page in pages:
        # Link the first image on the page to this page's text chunks, if any.
        # TODO(search lead): refine this to link images to their SPECIFIC nearby
        # chunk once nearby_caption matching is validated against real pages,
        # rather than linking every chunk on the page to the first image found.
        linked_image_id = page.images[0].image_id if page.images else None

        if page.is_likely_scanned:
            print(f"  [!] {pdf_path.name} p{page.page_number} looks scanned — "
                  f"run src/ingest/ocr_ingest.py on it separately, skipping text chunk for now")
            continue

        page_chunks = chunk_page_text(
            source_file=pdf_path.name,
            page_number=page.page_number,
            text=page.text,
            linked_image_id=linked_image_id,
        )
        all_chunks.extend(page_chunks)

    return all_chunks


def process_docx(docx_path: Path) -> list[Chunk]:
    record = extract_docx(docx_path, visuals_out_dir=VISUALS_DIR)
    all_chunks: list[Chunk] = []

    full_text = "\n\n".join(record.paragraphs)
    linked_image_id = None
    if record.image_paths:
        linked_image_id = Path(record.image_paths[0]).stem

    chunks = chunk_page_text(
        source_file=docx_path.name,
        page_number=1,  # DOCX has no native page numbers; refine if needed
        text=full_text,
        linked_image_id=linked_image_id,
    )
    all_chunks.extend(chunks)
    return all_chunks


def main() -> None:
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print(f"No files found in {RAW_DIR}/. Add archive files there first.")
        return

    all_chunks: list[Chunk] = []
    files = sorted(RAW_DIR.glob("*"))

    for file_path in files:
        if file_path.name == ".gitkeep":
            continue

        print(f"Processing {file_path.name}...")
        try:
            if file_path.suffix.lower() == ".pdf":
                chunks = process_pdf(file_path)
            elif file_path.suffix.lower() == ".docx":
                chunks = process_docx(file_path)
            else:
                print(f"  [!] Skipping unsupported file type: {file_path.suffix}")
                continue
        except Exception as e:
            print(f"  [!] Failed to process {file_path.name}: {e}")
            continue

        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

        # Save per-file chunk JSON (hybrid_search.py's BM25Index reads *_chunks.json)
        out_path = PROCESSED_DIR / f"{file_path.stem}_chunks.json"
        save_chunks_json(chunks, out_path)

    if not all_chunks:
        print("No chunks produced. Check that data/raw/ has PDF or DOCX files.")
        return

    print(f"\nTotal chunks across all files: {len(all_chunks)}")
    print("Building vector index (this calls the embedding backend for every chunk)...")
    build_index(all_chunks)
    print("Done. Vector index + chunk JSON files are ready. You can now run:")
    print("    streamlit run src/app.py")


if __name__ == "__main__":
    main()
