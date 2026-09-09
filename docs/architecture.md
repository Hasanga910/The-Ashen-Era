# Architecture

**Ashen Era: Spatially-Grounded Multimodal RAG**
SLIIT Codefest 2026 — AI Competition — Sub-track 1A: Rich Answers, Not Just Text

## Pipeline overview

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

## Overview

AshenLens is a Retrieval-Augmented Generation system organized into three
boundaries: an **offline ingestion/indexing pipeline** that processes the
Ashen Era Corpus once, an **online retrieval + generation app** (Streamlit)
that serves user queries, and **external APIs** the online app calls out to.
The design goal throughout is *spatial grounding*: every answer traces back
to an exact bounding-box region on an exact page, not just a document name.

---

## Boundary 1 — Offline Data Ingestion & Indexing Pipeline

Run once (and re-run whenever the corpus changes) via `src/indexing/build_all.py`.

| Node | Component | Role |
|---|---|---|
| A | **Ashen Era Corpus** | Raw PDFs and DOCX files, read-only (`data/raw/`) |
| B | **Multimodal Parser** (PyMuPDF / python-docx) | Extracts text, layout, and embedded images per page |
| C | **Spatial Extraction Module** | Computes bounding boxes `(x0, y0, x1, y1)` for figures/tables/seals and crops them to PNG |
| D | **Metadata Store** | Local storage for bounding-box JSON + cropped visuals (`data/visuals/`) |
| E | **Text Chunker** | Splits page text into page-aware segments, keeping `source_file` + `page_number` attached |
| F | **BM25 Tokenizer** | Tokenizes chunks for exact-match/lexical search |
| G | **Lexical Index** | Persisted BM25 index for keyword search |
| H | **Embedding Client** | Sends chunks to the configured embedding backend |
| I | **Vector Store (ChromaDB)** | Stores dense embeddings for semantic search |

**Flow:** A → B → {C → D, E}. E → F → G. E → H → (external embedding API) → I.

**Code mapping:**
- B, C, D → `src/ingest/pdf_ingest.py`, `src/ingest/docx_ingest.py`
- E → `src/indexing/chunker.py`
- F, G → `src/retrieval/hybrid_search.py` (`BM25Index`)
- H, I → `src/indexing/embeddings.py`, `src/indexing/vector_index.py`
- Orchestration of the whole boundary → `src/indexing/build_all.py`

---

## Boundary 2 — Online Hybrid Retrieval & Generation (Streamlit App)

Serves live user queries against the indexes built in Boundary 1.

| Node | Component | Role |
|---|---|---|
| J | **User Interface (Streamlit)** | Accepts the natural-language question |
| K | **Hybrid Retrieval Engine** | Dispatches the query to both lexical and vector search |
| L | **Reciprocal Rank Fusion (RRF)** | Merges BM25 rankings and vector-similarity rankings into one ranked list |
| M | **Visual Provenance Matcher** | Cross-references top-ranked text chunks against the Metadata Store (D) to find their linked bounding-box visuals |
| N | **Prompt Compiler** | Builds a strict, context-bound generation prompt from fused text + visual references |
| O | **LLM Response Parser** | Parses the LLM's answer and citation references |
| P | **Dynamic UI Renderer** | Displays the answer alongside the cropped bounding-box visual and an "Inspect Source Context" full-page view |

**Flow:** J → K → {G, I} → L → M → D (lookup). M → N → (external LLM API) → O → P → J.

**Code mapping:**
- J, P → `src/app.py`
- K, L → `src/retrieval/hybrid_search.py` (`hybrid_search`, `reciprocal_rank_fusion`)
- M → linked via `linked_image_id` on each `Chunk` (see `src/indexing/chunker.py`)
- N, O → `src/generation/answer.py` (`build_prompt`, `generate_answer`)

---

## Boundary 3 — External Cloud APIs

| Node | Component | Role |
|---|---|---|
| Q | **Embedding API** | Turns text into dense vectors |
| R | **LLM API** | Generates the grounded natural-language answer |

**Design decision — backend-agnostic, not locked to one provider.**
The diagram shows Voyage AI (Q) and OpenRouter/Llama 3.1 (R) as the external
APIs, and those remain fully supported. In practice, `src/indexing/embeddings.py`
and `src/generation/answer.py` are both written as *pluggable backends*,
selected via `.env`, so the team is not dependent on any single paid or
rate-limited service during development or the live demo:

| Role | Options (set via `.env`) | Notes |
|---|---|---|
| Embeddings | `voyage` \| `ollama` \| `local` | Voyage = Q in the diagram. `ollama`/`local` avoid the free-tier token ceiling entirely. |
| LLM generation | `openrouter` \| `ollama` | OpenRouter = R in the diagram. `ollama` (including Ollama Cloud models) avoids OpenRouter's free-tier rate limits (see challenge appendix 9.3). |

This is a resilience decision, not a scope reduction — the same
hybrid-retrieval-then-grounded-generation architecture holds regardless of
which concrete API answers Q and R at runtime, and it should be called out
explicitly in `docs/decisions.md` and the AI Usage Disclosure as a deliberate
engineering trade-off (avoiding single-provider dependency for a time-boxed
competition).

---

## Full data flow (numbered, matches diagram arrows)

1. Ingest docs — A → B
2. Extract embedded images & tables — B → C
3. Save spatial coordinates & assets — C → D
4. Pass raw text and page numbers — B → E
5. Send text chunks (lexical) — E → F
6. Build exact-match index — F → G
7. Send text chunks (embedding) — E → H
8. API request for vectors — H → Q
9. Return numerical vectors — Q → H
10. Store embeddings in ChromaDB — H → I
11. Submit query — J → K
12. Execute lexical search — K → G
13. Execute semantic vector search — K → I
14. Return BM25 scores — G → L
15. Return vector similarities — I → L
16. Output top-K fused text chunks — L → M
17. Look up corresponding visual coordinates — M → D
18. Send fused context + visual IDs — M → N
19. Send strict generation prompt — N → R
20. Return generated answer with citations — R → O
21. Trigger rendering of answer + layout-aware visuals — O → P
22. Return final multimodal response to user — P → J

---

## Why hybrid retrieval (BM25 + vector, fused via RRF)

The sample question set includes near-identical proper nouns referring to
different entities — e.g. *"Thrice-Bound Edge"* (shards of will) vs.
*"Thrice-Bound Lantern"* (attunement cost). Semantic embeddings place these
close together in vector space, risking retrieval of the wrong figure plate.
BM25 anchors on the exact distinguishing keyword. RRF combines both
rankings rather than picking one, so neither weakness dominates.

## Why bounding-box cropping instead of whole-page screenshots

Standard multimodal retrieval often shows a full page image, forcing the
user to hunt for the relevant figure (weak visual grounding). This pipeline
stores exact bounding-box coordinates for each figure/table/seal so the app
can (a) show the cropped visual directly next to the answer, and (b) offer
an "Inspect Source Context" toggle showing the full page with a translucent
highlight over the exact region, for users who want original layout context.

## Open questions / TODO

- [ ] Validate the caption-detection heuristic in `pdf_ingest._guess_nearby_caption`
      against real "figure plate" / "portrait of" pages in the archive.
- [ ] Implement the bbox-highlight rendering (Pillow/OpenCV) for "Inspect
      Source Context" — currently a placeholder in `src/app.py`.
- [ ] Decide final default embedding/LLM backend based on measured
      accuracy on `sample_questions.json`, and document the choice in
      `docs/decisions.md`.
- [ ] Confirm no duplicate corpus files are being indexed twice (e.g. the
      same document present as both `.docx` and `.pdf`).
