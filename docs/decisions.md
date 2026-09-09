# Key Decisions Log

Keep this updated as the team makes real technical choices — judges score
"technical judgment & decisions" (10%) partly from this file.

Format: one entry per decision, with the alternative considered and why it lost.

---

## 2026-XX-XX — Hybrid search over pure vector search

**Decision:** Combine BM25 keyword search with vector/semantic search via
Reciprocal Rank Fusion, rather than relying on embeddings alone.

**Why:** Sample questions reference near-identical proper nouns (e.g.
"Thrice-Bound Edge" vs "Thrice-Bound Lantern"). Testing showed pure
embedding search risks confusing these. BM25 anchors on the exact
distinguishing keyword.

**Alternative considered:** Pure vector search with a reranker model —
rejected for now as added complexity/latency without solving the
exact-keyword problem as directly.

---

## 2026-XX-XX — [next decision]

**Decision:**

**Why:**

**Alternative considered:**
