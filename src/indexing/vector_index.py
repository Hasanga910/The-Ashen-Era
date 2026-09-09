"""
vector_index.py
----------------
Builds and queries a ChromaDB collection from chunks (see chunker.py).

Owner: Search / ML lead
"""

from __future__ import annotations

import os

import chromadb
from dotenv import load_dotenv

from src.indexing.chunker import Chunk
from src.indexing.embeddings import embed_texts

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/processed/chroma_db")
COLLECTION_NAME = "ashen_era_chunks"


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def build_index(chunks: list[Chunk], batch_size: int = 500) -> None:
    """Embed and index chunks in batches.

    Batching avoids Chroma's per-call max batch size limit and gives visible
    progress on large corpora (the embedding calls to Ollama are the slow
    part -- one HTTP request per chunk).
    """
    client = get_client()
    collection = client.get_or_create_collection(COLLECTION_NAME)

    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        print(f"  Indexing batch {start}-{start + len(batch)} of {total}...")

        texts = [c.text for c in batch]
        vectors = embed_texts(texts, input_type="document")

        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                {
                    "source_file": c.source_file,
                    "page_number": c.page_number if c.page_number is not None else -1,
                    "linked_image_id": c.linked_image_id or "",
                }
                for c in batch
            ],
        )


def query_index(question: str, top_k: int = 5) -> dict:
    client = get_client()
    collection = client.get_or_create_collection(COLLECTION_NAME)

    query_vector = embed_texts([question], input_type="query")[0]
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.indexing.vector_index '<question>'")
        sys.exit(1)

    results = query_index(sys.argv[1])
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"[{meta['source_file']} p{meta['page_number']}] {doc[:120]}...")
