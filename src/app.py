"""
app.py
------
AshenLens Streamlit UI. Ties together: hybrid retrieval -> grounded answer
generation -> display of answer + citations + cropped visual evidence.

Owner: App / Integration lead

Run with:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

# Streamlit runs this script with `src/` (not the project root) as the
# working context, which breaks `from src.xxx import ...` imports below.
# Explicitly add the project root (parent of this file's directory) to
# sys.path so the `src` package resolves correctly no matter how/where
# this is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.generation.answer import generate_answer
from src.retrieval.hybrid_search import BM25Index, hybrid_search

st.set_page_config(page_title="AshenLens", layout="wide")

st.title("AshenLens")
st.caption("Spatially-Grounded Multimodal RAG on Ashen Era")


@st.cache_resource
def load_bm25_index():
    return BM25Index.from_processed_dir()


question = st.text_input(
    "Ask a question about the Ashen Era Archive",
    placeholder="e.g. What is the central emblem on the banner of House Morvain?",
)

if st.button("Search", type="primary") and question:
    with st.spinner("Searching the archive..."):
        try:
            bm25_index = load_bm25_index()
        except Exception as e:
            st.error(f"No processed chunks found yet. Run the ingestion + indexing pipeline first. ({e})")
            st.stop()

        evidence = hybrid_search(question, bm25_index, top_k=5)

        if not evidence:
            st.warning("No relevant evidence found in the archive for this question.")
            st.stop()

        answer_text = generate_answer(question, evidence)

    st.subheader("Answer")
    st.write(answer_text)

    st.subheader("Sources")
    for chunk in evidence:
        with st.expander(f"{chunk.get('source_file')} — page {chunk.get('page_number')}"):
            st.write(chunk.get("text", ""))

            linked_image_id = chunk.get("linked_image_id")
            if linked_image_id:
                matches = list(Path('data/visuals').glob(f'*{linked_image_id}.png'))
                if not matches:
                    continue
                image_path = str(matches[0])
                st.image(image_path, caption="Visual evidence (cropped)")

                if st.checkbox("Inspect Source Context (show full page + highlight)", key=f"ctx_{chunk.get('chunk_id')}"):
                    st.info(
                        "Full-page context view with highlighted source region is coming soon."
                    )
else:
    st.info("Type a question above and press Search to query the archive.")
