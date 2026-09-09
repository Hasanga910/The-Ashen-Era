# Architecture

## Pipeline overview

```
Archive files (PDF/DOCX/MD/TXT/scans)
        |
        v
[ Ingestion ]  src/ingest/
  - pdf_ingest.py   : text + image bbox extraction (PyMuPDF)
  - docx_ingest.py  : paragraphs, tables, embedded images
  - ocr_ingest.py   : fallback OCR for scanned pages (Tesseract)
        |
        v
[ Indexing ]  src/indexing/
  - chunker.py      : paragraph-based chunking, keeps source_file + page_number
  - embeddings.py   : Voyage AI (or local sentence-transformers fallback)
  - vector_index.py : ChromaDB persistent collection
        |
        v
[ Retrieval ]  src/retrieval/
  - hybrid_search.py : BM25 (keyword) + vector (semantic) combined via
                        Reciprocal Rank Fusion
        |
        v
[ Generation ]  src/generation/
  - answer.py : LLM (OpenRouter) generates an answer using ONLY retrieved
                evidence; instructed to say "not enough evidence" rather
                than guess
        |
        v
[ App ]  src/app.py (Streamlit)
  - question box -> answer -> citations -> cropped visual evidence
  - "Inspect Source Context" toggle -> full page + highlighted bbox
```

## Why hybrid retrieval (BM25 + vector)

The sample question set includes near-identical proper nouns referring to
different entities, e.g. "Thrice-Bound Edge" (shards of will) vs.
"Thrice-Bound Lantern" (attunement cost). Semantic embeddings place these
close together, risking retrieval of the wrong figure plate. BM25 catches
the exact distinguishing keyword. We combine both rankings with Reciprocal
Rank Fusion rather than picking one.

## Why bounding-box cropping instead of whole-page screenshots

Standard multi-modal retrieval often just shows a full page image, forcing
the user to hunt for the relevant figure (weak visual grounding). Our
ingestion pipeline stores exact bounding-box coordinates for each
figure/table/seal, so the app can:
1. Show the cropped visual directly next to the answer.
2. Offer an "Inspect Source Context" toggle showing the full page with a
   translucent highlight over the exact region, for users who want to see
   it in original layout.

## Open questions / TODO

- [ ] Validate caption-detection heuristic in `pdf_ingest._guess_nearby_caption`
      against real "figure plate" / "portrait of" pages once we've looked at
      the archive.
- [ ] Decide Voyage AI vs local embeddings based on whether a team card works
      for international transactions.
- [ ] Add `src/indexing/build_all.py` to run ingestion + chunking + indexing
      over the whole `data/raw/` directory in one command.
- [ ] Add bbox-highlight rendering (Pillow/OpenCV) for "Inspect Source Context".
