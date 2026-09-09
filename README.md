# AshenLens

**SLIIT Codefest 2026 — AI Competition — Sub-track 1A: Rich Answers, Not Just Text**

AshenLens is a retrieval-augmented assistant for the Ashen Era Archive. Users ask questions
about the archive; the system retrieves grounded evidence (text, tables, figure plates,
portraits) and answers with citations — showing the *actual cropped image or plate*,
not just a text description, when a visual is relevant.

## Team

| Role | Member | Responsibility |
|---|---|---|
| Document Ingestion Lead | Dunith Desitha Ranawansha | Owns `src/ingest/` (`pdf_ingest.py`, `docx_ingest.py`, deprecated `ocr_ingest.py`). Engineered the raw document processing pipeline for the enterprise corpus; implemented layout-aware extraction to capture text alongside precise spatial bounding-box coordinates for images and tables. |
| Processing & Indexing Lead | Tharidi Pabasari Gamage | Owns `src/indexing/` (`chunker.py`, `embeddings.py`, `vector_index.py`, `build_all.py`). Architected the transformation of extracted content into searchable representations; designed the chunking strategy and mapped the embeddings pipeline into the ChromaDB vector store. |
| Retrieval Lead | Sajana Hasanga | Owns `src/retrieval/` (`hybrid_search.py`). Engineered the core search engine for finding relevant evidence; implemented BM25 + dense vector retrieval integration, Reciprocal Rank Fusion (RRF), and visual-intent prioritization logic. |
| Generation & Application Lead | Yasindu Sasmitha | Owns `src/generation/` (`answer.py`) and `src/app.py`. Transformed retrieved evidence into the final cited answer and built the interactive Streamlit UI; integrated Qwen vision handling, dynamic visual evidence selection, and deterministic citation fallback mechanisms for zero-hallucination outputs. |

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd ashen-era

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# edit .env: add OPENROUTER_API_KEY (required), VOYAGE_API_KEY (optional)
ollama signin                        # if not already
ollama run gemma4:31b-cloud          # test it responds, then /bye
ollama pull nomic-embed-text         # small local embedding model (~274MB)
# 5. Place the official archive files under data/raw/ (read-only, do not modify)
```

## Running the pipeline

```bash
# Step 1: Extract text/images/bboxes from each archive file
python src/ingest/pdf_ingest.py data/raw/<some_file>.pdf
python src/ingest/docx_ingest.py data/raw/<some_file>.docx

# Step 2: Chunk + build indexes (script to be added once extraction is stable)
# python src/indexing/build_all.py

# Step 3: Launch the app
streamlit run src/app.py
```

## Project structure

See `docs/architecture.md` for the full pipeline diagram and design rationale.

```
ashen-era/
├── data/
│   ├── raw/          # Official archive (read-only, not committed)
│   ├── processed/    # Extracted text, chunks, chroma DB
│   └── visuals/       # Cropped images / figure plates
├── src/
│   ├── ingest/        # PDF, DOCX, OCR extraction
│   ├── indexing/      # Chunking, embeddings, vector index
│   ├── retrieval/     # Hybrid (BM25 + vector) search
│   ├── generation/    # Grounded answer generation
│   └── app.py          # Streamlit application
├── tests/
├── docs/               # architecture.md, decisions.md, limitations.md
└── ai_usage/           # AI usage disclosure + exported chat logs (required by rules)
```

## Key design decisions

See `docs/decisions.md`. Highlights:
- **Hybrid search (BM25 + vector via RRF)** — the archive has near-identical proper
  nouns (e.g. "Thrice-Bound Edge" vs "Thrice-Bound Lantern"); pure embeddings confuse
  these, BM25 disambiguates.
- **Bounding-box provenance** — figures/tables are cropped to their exact region
  (via PyMuPDF), not shown as whole-page screenshots, with an "Inspect Source Context"
  toggle to view the full page with the region highlighted.
- **Strict grounding** — the LLM only ever sees retrieved evidence, and is instructed
  to say "not enough evidence" rather than guess.

## AI Usage

Per competition rules, our AI usage disclosure and exported chat logs are in `ai_usage/`.
