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

## 🚀 Quick Start

These instructions are written for a **first-time user or evaluator who has never used RAG before**. You do not need prior knowledge of RAG to run AshenLens.

### 1. Prerequisites

Install:

- **Python 3.11 or newer**
- **Ollama**
- Internet connection for the initial Python package and model downloads

AshenLens currently uses local Ollama models, so an external LLM API key is not required.

### 2. Open the project

Extract the project folder(AshenLens) and open PowerShell in it.

Example:

```powershell
cd C:\AshenLens
```

Verify the project files:

```powershell
Get-ChildItem
```

You should see files/folders similar to:

```text
data
docs
src
tests
.env.example
.gitignore
README.md
requirements.txt
```

### 3. Create a Python virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see:

```text
(.venv) PS C:\AshenLens>
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Python dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Install and prepare Ollama

Verify Ollama:

```powershell
ollama --version
```

Download the text-generation model:

```powershell
ollama pull llama3.2:3b
```

Download the vision-language model:

```powershell
ollama pull qwen2.5vl:3b
```

Verify both:

```powershell
ollama list
```

If Ollama is not already running:

```powershell
ollama serve
```

If `ollama serve` occupies the terminal, open another PowerShell window and continue there.

### 6. Configure the application

Create the local environment file:

```powershell
Copy-Item .env.example .env
```

The local-model configuration should contain:

```text
EMBEDDING_BACKEND=local
OLLAMA_MODEL=llama3.2:3b
OLLAMA_VISION_MODEL=qwen2.5vl:3b
```

Do not commit private API keys or other secrets.

### 7. Build the searchable archive

Before asking questions, AshenLens needs to prepare the archive for retrieval.

Run:

```powershell
python src\build_all.py
```

In simple terms, this step:

1. Reads the archive.
2. Extracts usable document content.
3. Splits content into smaller searchable pieces called **chunks**.
4. Creates numerical representations called **embeddings**.
5. Builds the local search index.
6. Associates relevant visual assets with retrieved content.

The initial indexing step may take some time depending on the machine.

### 8. Start AshenLens

```powershell
streamlit run src\app.py
```

Open the local address shown by Streamlit, normally:

```text
http://localhost:8501
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

## 🧩 Main Components

### `src/ingest/`

Responsible for processing source documents:

```text
docx_ingest.py
pdf_ingest.py
ocr_ingest.py
```

### `src/indexing/`

Responsible for preparing content for retrieval:

```text
chunker.py
embeddings.py
vector_index.py
build_all.py
```

General flow:

```text
Extracted content
       ↓
Chunking
       ↓
Embeddings
       ↓
Vector index
```

### `src/retrieval/`

Responsible for finding useful evidence:

```text
hybrid_search.py
```

It combines multiple retrieval signals and gives visual evidence additional priority for visually oriented queries.

### `src/generation/`

Responsible for generating the final grounded response:

```text
answer.py
```

It handles text generation, vision-language generation, visual evidence selection, source citations, and multimodal prompts.

### `src/app.py`

The Streamlit application connecting:

```text
User
 ↓
Question
 ↓
Retrieval
 ↓
Evidence
 ↓
Generation
 ↓
Answer + Sources + Visual Evidence
```

## 🧪 Testing

Run the automated tests:

```powershell
pytest
```

For evaluator testing, also manually test a normal text question and several visual/figure questions.

## 🛠️ Troubleshooting

### `python` is not recognized

Check:

```powershell
python --version
```

Install Python 3.11+ and ensure it is available from the terminal.

### Virtual environment does not activate

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### `ollama` is not recognized

Install Ollama and restart the terminal.

```powershell
ollama --version
```

### Model is missing

```powershell
ollama list
```

Then:

```powershell
ollama pull llama3.2:3b
ollama pull qwen2.5vl:3b
```

### Search index is missing

Run:

```powershell
python src\build_all.py
```

Then restart:

```powershell
streamlit run src\app.py
```

### Streamlit is not available

Make sure `.venv` is active and run:

```powershell
pip install -r requirements.txt
```

## 🔐 Configuration and Secrets

Create `.env` from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Do not place private credentials or API keys in source files.

