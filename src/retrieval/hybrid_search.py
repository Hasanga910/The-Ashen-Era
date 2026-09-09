"""
hybrid_search.py
-----------------
Combines keyword search (BM25) with semantic search (vector/ChromaDB) using
Reciprocal Rank Fusion (RRF).

Owner: Search / ML lead

WHY THIS MATTERS FOR THIS ARCHIVE:
Sample questions reference near-identical proper nouns, e.g. "Thrice-Bound
Edge" vs "Thrice-Bound Lantern" -- two different artifacts. Pure embedding
search tends to conflate these because they're semantically very close.
BM25 catches the exact distinguishing word ("Edge" vs "Lantern") and pulls
rank back in the right direction. Don't skip this step.
"""

from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.indexing.vector_index import query_index


class BM25Index:
    """Simple in-memory BM25 index over all chunks.

    Rebuild is cheap enough for this corpus size (~1,277 pages) to just load
    from the processed chunk JSON files at startup rather than persisting a
    separate index.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    @classmethod
    def from_processed_dir(
        cls,
        processed_dir: str | Path = "data/processed",
    ) -> "BM25Index":
        chunks: list[dict] = []

        for path in Path(processed_dir).glob("*_chunks.json"):
            chunks.extend(
                json.loads(path.read_text(encoding="utf-8"))
            )

        return cls(chunks)

    def search(
        self,
        question: str,
        top_k: int = 10,
    ) -> list[tuple[dict, float]]:
        tokenized_query = question.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:top_k]


def reciprocal_rank_fusion(
    bm25_results: list[tuple[dict, float]],
    vector_chunk_ids: list[str],
    vector_documents: list[str],
    vector_metadatas: list[dict],
    k: int = 60,
) -> list[dict]:
    """Merge BM25 and vector rankings.

    Higher combined score = better.

    RRF formula:
        score(doc) = sum(1 / (k + rank))

    k=60 is the standard default from the RRF paper.
    """

    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    # BM25 ranking
    for rank, (chunk, _bm25_score) in enumerate(bm25_results):
        cid = chunk["chunk_id"]

        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)

        chunk_lookup[cid] = chunk

    # Vector ranking
    for rank, (
        cid,
        doc,
        meta,
    ) in enumerate(
        zip(
            vector_chunk_ids,
            vector_documents,
            vector_metadatas,
        )
    ):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)

        if cid not in chunk_lookup:
            chunk_lookup[cid] = {
                "chunk_id": cid,
                "text": doc,
                **meta,
            }

    ranked_ids = sorted(
        scores.keys(),
        key=lambda cid: scores[cid],
        reverse=True,
    )

    return [
        chunk_lookup[cid]
        for cid in ranked_ids
    ]


def hybrid_search(
    question: str,
    bm25_index: BM25Index,
    top_k: int = 5,
) -> list[dict]:
    """Run hybrid BM25 + vector retrieval.

    For visually-oriented questions, retrieved chunks containing an actual
    linked visual are prioritized so that the multimodal LLM receives the
    relevant image before semantically similar text-only evidence.
    """

    # ---------------------------------------------------------
    # 1. BM25 retrieval
    # ---------------------------------------------------------
    bm25_results = bm25_index.search(
        question,
        top_k=10,
    )

    # ---------------------------------------------------------
    # 2. Vector retrieval
    # ---------------------------------------------------------
    vector_results = query_index(
        question,
        top_k=10,
    )

    vector_chunk_ids = vector_results["ids"][0]
    vector_documents = vector_results["documents"][0]
    vector_metadatas = vector_results["metadatas"][0]

    # ---------------------------------------------------------
    # 3. Existing RRF fusion
    # ---------------------------------------------------------
    fused = reciprocal_rank_fusion(
        bm25_results,
        vector_chunk_ids,
        vector_documents,
        vector_metadatas,
    )

    # ---------------------------------------------------------
    # 4. Detect visual intent
    # ---------------------------------------------------------
    visual_terms = [
        "image",
        "picture",
        "portrait",
        "banner",
        "emblem",
        "symbol",
        "shown",
        "depicted",
        "depicts",
        "figure",
        "plate",
        "diagram",
        "illustration",
        "drawing",
        "appearance",
        "holding",
        "wearing",
        "looks like",
        "what color",
        "what colour",
        "visual",
        "on the banner",
        "in the portrait",
        "in the image",
        "in the picture",
        "according to the figure",
        "according to the plate",
    ]

    question_lower = question.lower()

    is_visual_query = any(
        term in question_lower
        for term in visual_terms
    )

    # ---------------------------------------------------------
    # 5. Prioritize visual evidence for visual questions
    # ---------------------------------------------------------
    if is_visual_query:
        visual_chunks = [
            chunk
            for chunk in fused
            if chunk.get("linked_image_id")
        ]

        text_chunks = [
            chunk
            for chunk in fused
            if not chunk.get("linked_image_id")
        ]

        # Preserve RRF ordering within each group.
        #
        # Visual evidence comes first because the downstream
        # LLM is vision-capable and needs the actual pixels for
        # questions such as:
        #
        #   - What is on the banner?
        #   - What is the character holding?
        #   - What does the figure plate say?
        #
        fused = visual_chunks + text_chunks

    return fused[:top_k]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.retrieval.hybrid_search '<question>'"
        )
        sys.exit(1)

    index = BM25Index.from_processed_dir()

    results = hybrid_search(
        sys.argv[1],
        index,
    )

    for result in results:
        print(
            f"[{result.get('source_file')} "
            f"p{result.get('page_number')}] "
            f"{result.get('text', '')[:120]}..."
        )