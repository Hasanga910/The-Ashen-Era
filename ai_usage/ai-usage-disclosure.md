# AI Usage Disclosure — Ashen Era: Spatially-Grounded Multimodal RAG

## 1. Purpose

AI tools were used throughout development as engineering assistants rather than as substitutes for
the team's implementation and decision-making. The team used AI to discuss architecture options,
generate and refine code ideas, debug implementation issues, improve prompts, and think through
edge cases.

The final implementation was reviewed, adapted, tested, and integrated by the team.

The four members divided the implementation into four main areas:

- Member 1 Dunith Desitha Ranawansha      — Document Ingestion
- Member 2 Tharidi Pabasari Gamage        — Processing & Indexing
- Member 3 Hewa Nalagamage Sajana Hasanga — Retrieval
- Member 4 Yasindu Sasmitha               — Generation & Application

Shared files such as documentation, configuration, tests, and repository integration were handled
collaboratively.

## 2. Member 1 Dunith Desitha Ranawansha — Document Ingestion

### Owned files
- `src/ingest/pdf_ingest.py`
- `src/ingest/docx_ingest.py`
- `src/ingest/ocr_ingest.py`

### How AI was used

AI assistance was mainly used to reason about how to process the mixed enterprise archive without
losing provenance. The team discussed PDF text extraction, DOCX paragraph/table extraction,
embedded image extraction, OCR fallback, and the metadata required for later visual retrieval.

A key design decision was to keep ingestion modular. PDF, DOCX, and OCR functionality were kept in
separate modules rather than placing all document handling into one large script.

For PDFs, the implementation uses PyMuPDF to extract page text and embedded images. Image locations
are represented using `(x0, y0, x1, y1)` bounding boxes and cropped visual assets are saved for
later use. Pages with very little extracted text can be treated as likely scans and passed through
the OCR fallback.

For DOCX files, the implementation extracts paragraphs, tables, and embedded media. The resulting
records retain source information so downstream indexing can connect retrieved chunks to their
original documents.

### Human decisions

The team decided:
1. provenance must be preserved at page level;
2. visual assets need stable IDs;
3. spatial metadata must survive beyond ingestion;
4. OCR should be a fallback rather than an expensive default for every page;
5. ingestion code should expose structured records to the indexing layer.

The implementation was tested by inspecting extracted text, generated images, and page metadata.

See: `ai_usage/member1_document_ingestion_ai_log.txt`

## 3. Member 2 Tharidi Pabasari Gamage — Processing & Indexing

### Owned files
- `src/indexing/chunker.py`
- `src/indexing/embeddings.py`
- `src/indexing/vector_index.py`
- `src/indexing/build_all.py`

### How AI was used

AI assistance was used to compare chunking approaches, discuss provenance-aware metadata, evaluate
embedding backends, and reason about persistent vector storage.

The team chose to retain source filename, page number, chunk ID, and linked visual information with
each chunk. This was important because retrieval alone is not enough for a multimodal answer: the
system must be able to trace a chunk back to the relevant visual asset.

The embedding implementation supports multiple backends through environment configuration,
including Voyage AI, Ollama, and a local sentence-transformers fallback. ChromaDB is used as the
persistent vector store.

The indexing pipeline processes documents into chunks and builds the vector index in batches.
Batching was selected to make large index construction more practical and to avoid unnecessarily
large individual ChromaDB operations.

### Human decisions

The team decided:
1. chunk metadata must preserve provenance;
2. visual links belong in chunk metadata;
3. the embedding backend should be configurable;
4. ChromaDB should persist locally;
5. indexing should operate in batches rather than making one operation per chunk.

The resulting pipeline is:

`documents -> chunks -> embeddings -> ChromaDB`

See: `ai_usage/member2_processing_indexing_ai_log.txt`

## 4. Member 3 Hewa Nalagamage Sajana Hasanga — Retrieval

### Owned file
- `src/retrieval/hybrid_search.py`

### How AI was used

AI assistance was used to reason about the limitations of relying on either lexical or semantic
retrieval alone.

The team identified two complementary retrieval needs:
- BM25 is useful when exact terminology, names, or distinguishing words matter.
- Vector retrieval is useful when the question and the evidence use different wording but have
  similar meaning.

The implementation therefore performs both searches and combines their rankings using Reciprocal
Rank Fusion (RRF). This avoids directly comparing incompatible BM25 and vector score scales.

A second important design decision was visual-intent prioritization. When the query clearly asks
about an image, visual appearance, a picture, portrait, emblem, banner, or similar visual detail,
the retrieval layer prioritizes retrieved chunks that contain linked visual evidence.

### Human decisions

The team decided:
1. not to replace BM25 with vector search;
2. to use both retrieval signals;
3. to use RRF for rank-level fusion;
4. to preserve source/page/image metadata during fusion;
5. to apply visual prioritization only when the query indicates visual intent.

This keeps ordinary factual questions from being unnecessarily dominated by visual evidence.

The retrieval implementation was tested with lexical, semantic, and visually oriented questions.

See: `ai_usage/member3_retrieval_ai_log.txt`

## 5. Member 4 Yasindu Sasmitha — Generation & Application

### Owned files
- `src/generation/answer.py`
- `src/app.py`

### How AI was used

AI assistance was used heavily for prompt design, citation enforcement, multimodal evidence
selection, model/API integration, and Streamlit UI structure.

The team designed the generation stage around evidence grounding. Retrieved chunks are numbered and
provided to the model as explicit evidence. The prompt instructs the model to make factual claims
only from the supplied evidence and to attach valid citation numbers.

For visual questions, the application resolves the `linked_image_id` values to the actual cropped
image files. Visual evidence can then be supplied to the vision-capable generation path. The
implementation also keeps relevant text evidence from the same source/page where appropriate.

Citation validation was treated as an engineering requirement rather than assuming the model would
always follow the prompt. The code checks for valid citation numbers, can request a citation repair,
and includes a deterministic fallback when necessary.

The Streamlit application connects the complete flow:

`user query -> hybrid retrieval -> grounded generation -> citations -> visual evidence -> UI`

### Human decisions

The team decided:
1. evidence must be explicitly numbered;
2. every factual claim should have a valid citation;
3. visual questions should use the actual retrieved image when available;
4. citation failures need a fallback;
5. the UI should expose both the answer and supporting evidence;
6. source-context inspection should remain optional so the main answer stays readable.

The current source also contains a TODO for full-page rendering with a translucent bounding-box
overlay. This is intentionally documented as unfinished rather than presented as a completed feature.

See: `ai_usage/member4_generation_application_ai_log.txt`

## 6. Shared / Integration Work

The following were treated as collaborative integration work rather than artificially assigning them
to one person:

- `README.md`
- `requirements.txt`
- `.env.example`
- `.gitignore`
- `src/build_all.py`
- `tests/`
- `docs/`

Team members reviewed integration points between ingestion, indexing, retrieval, and generation to
ensure that metadata and evidence could flow through the complete system.

## 7. Human–AI Collaboration Pattern

The team did not treat AI output as automatically correct. The general workflow was:

1. Identify an implementation problem.
2. Ask AI for possible approaches or explanations.
3. Compare the suggestions against the competition requirements and the existing code.
4. Select or modify an approach.
5. Implement it in the repository.
6. Run/test the implementation.
7. Investigate failures and edge cases.
8. Refine the code or prompt.
9. Keep the final implementation only after human review.

Examples of decisions that came from this process include hybrid rather than single-mode retrieval,
page-level provenance, linked visual IDs, visual-intent prioritization, configurable embedding
backends, citation validation, and deterministic citation fallback.

## 8. Important Note About AI Logs

The files in this directory are structured AI-usage disclosure/development-log drafts based on the
implemented repository and the team's documented ownership. They are not claimed to be verbatim
exports of historical AI conversations.

Where the competition requires raw exported chat histories, the team's actual ChatGPT/Claude/other
AI conversation exports should be placed alongside these disclosure documents. No reconstructed
conversation should be represented as an original historical transcript.
