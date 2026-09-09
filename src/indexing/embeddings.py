"""
embeddings.py
-------------
Turns chunk text into vectors. Supports three backends, chosen by
EMBEDDING_BACKEND in .env:

  - "voyage"      : Voyage AI (voyage-4 / voyage-4-lite) -- best quality,
                      needs API key + card, 200M free tokens
  - "ollama"      : local embedding model via Ollama (e.g. nomic-embed-text)
                      -- free, no card, no rate limits, works offline
  - "local"       : sentence-transformers, runs in-process (no server needed)

Owner: Search / ML lead

IMPORTANT: whichever backend you pick, use the SAME one for both indexing
and querying, or the vector space won't match -- don't switch mid-project
without re-indexing everything. If using Voyage, voyage-4 (docs) and
voyage-4-lite (queries) share a vector space, so mixing those two is fine.
"""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "ollama")  # "voyage" | "ollama" | "local"

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

_LOCAL_MODEL = None  # lazy-loaded sentence-transformers model


def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Embed a batch of texts. input_type is 'document' or 'query'.

    Voyage AI distinguishes doc vs query embeddings for better retrieval;
    the Ollama and local fallbacks ignore input_type.
    """
    if EMBEDDING_BACKEND == "voyage" and VOYAGE_API_KEY:
        return _embed_with_voyage(texts, input_type)
    if EMBEDDING_BACKEND == "ollama":
        return _embed_with_ollama(texts)
    return _embed_with_local_model(texts)


def _embed_with_ollama(texts: list[str]) -> list[list[float]]:
    """Call a locally running Ollama server's embedding endpoint.

    Requires the embedding model to be pulled first, e.g.:
        ollama pull nomic-embed-text

    Prints progress every 100 texts since this loops one HTTP call per text
    and can take a while on large batches (thousands of chunks).
    """
    vectors = []
    total = len(texts)
    for i, text in enumerate(texts):
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama embedding model '{OLLAMA_EMBED_MODEL}' not found. "
                f"Pull it first: `ollama pull {OLLAMA_EMBED_MODEL}`"
            )
        response.raise_for_status()
        vectors.append(response.json()["embedding"])

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"    embedded {i + 1}/{total} chunks")

    return vectors


def _embed_with_voyage(texts: list[str], input_type: str) -> list[list[float]]:
    model = "voyage-4" if input_type == "document" else "voyage-4-lite"
    response = requests.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
        json={"input": texts, "model": model, "input_type": input_type},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]


def _embed_with_local_model(texts: list[str]) -> list[list[float]]:
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _LOCAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _LOCAL_MODEL.encode(texts, convert_to_numpy=True).tolist()


if __name__ == "__main__":
    sample = ["The Thrice-Bound Edge requires three shards of will to attune."]
    vectors = embed_texts(sample, input_type="document")
    print(f"Backend: {'Voyage AI' if VOYAGE_API_KEY else 'local sentence-transformers'}")
    print(f"Embedded {len(sample)} text(s), vector dim = {len(vectors[0])}")
