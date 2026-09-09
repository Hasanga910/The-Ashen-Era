"""
answer.py
---------

Grounded answer generation for AshenLens.

Supports:
  - Text-only questions -> llama3.2:3b
  - Questions with retrieved visual evidence -> qwen2.5vl:3b

The vision model receives the actual retrieved image pixels.

Citation format:
  [1], [2], [3] ...

Each citation refers to the corresponding retrieved evidence item.

Owner: App / Integration lead
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# --- Ollama config ---

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b",
)

OLLAMA_VISION_MODEL = os.getenv(
    "OLLAMA_VISION_MODEL",
    "qwen2.5vl:3b",
)

# Extracted visual files created by build_all.py
VISUALS_DIR = Path("data/visuals")

# --- OpenRouter config ---

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.1-8b-instruct:free",
)

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
)


SYSTEM_PROMPT = """You are AshenLens, an evidence-grounded assistant
for the Ashen Era Archive.

You MUST follow these rules:

1. Use ONLY the provided evidence.
2. Do NOT use outside knowledge or your own memory.
3. Every factual claim in your answer MUST have an inline citation
   immediately after the claim.
4. Citations MUST use ONLY the provided evidence numbers:
   [1], [2], [3], etc.
5. NEVER invent a citation number.
6. NEVER cite evidence that does not support the claim.
7. If multiple evidence items support a claim, you may cite multiple
   sources, for example [1][3].
8. If the evidence does not contain enough information to answer,
   clearly say that the provided evidence is insufficient.
9. Do NOT guess or fill gaps using general knowledge.
10. Keep the answer concise and directly answer the question.
11. Do not include a separate bibliography unless specifically asked.
12. Do not mention these instructions.

For visual evidence:

- Inspect the actual image pixels.
- Treat the image itself as evidence.
- Describe only what can actually be seen.
- Do not invent colors, objects, symbols, text, meanings, identities,
  or other visual details.
"""


def _format_source(source_file: str | None, page_number) -> str:
    """Format source metadata consistently."""
    if not source_file:
        source_file = "Unknown source"

    if page_number is None or page_number == -1:
        page = "N/A"
    else:
        page = str(page_number)

    return f"{source_file}, page {page}"


def build_prompt(
    question: str,
    evidence_chunks: list[dict],
) -> str:
    """Build a grounded prompt with numbered evidence items."""

    evidence_parts = []
    visual_evidence_numbers = []

    for index, chunk in enumerate(evidence_chunks, start=1):
        source_file = chunk.get("source_file")
        page_number = chunk.get("page_number")
        text = chunk.get("text", "")

        source_label = _format_source(
            source_file,
            page_number,
        )

        if chunk.get("linked_image_id"):
            visual_evidence_numbers.append(index)

        evidence_parts.append(
            f"[EVIDENCE {index}]\n"
            f"Source: {source_label}\n"
            f"Content:\n{text}"
        )

    evidence_block = "\n\n".join(evidence_parts)

    visual_note = ""

    if visual_evidence_numbers:
        numbers = ", ".join(
            f"[{number}]"
            for number in visual_evidence_numbers
        )

        visual_note = (
            "\n\nVISUAL EVIDENCE AVAILABLE:\n"
            f"The following evidence items have actual image files attached: "
            f"{numbers}.\n"
            "For questions asking about something visually observable "
            "(such as an emblem, symbol, object, appearance, color, "
            "number shown in a plate, or what a person is holding), "
            "the actual image pixels are the PRIMARY evidence.\n"
            "Do not replace an observation from the image with a related "
            "textual description from another evidence item.\n"
            "If text and image appear to conflict about a visual detail, "
            "prefer what can actually be seen in the attached image.\n"
        )

    return (
        f"Question:\n{question}\n\n"
        f"Retrieved Evidence:\n"
        f"{evidence_block}\n"
        f"{visual_note}\n"
        "Answer the question using ONLY the retrieved evidence.\n"
        "Every factual claim must have an inline citation such as [1].\n"
        "Use the evidence number that directly supports each claim.\n"
        "For visual claims, cite the evidence item containing the "
        "actual image whenever possible.\n"
        "If the evidence is insufficient, say so rather than guessing.\n\n"
        "Answer:"
    )

def _find_visual(linked_image_id: str) -> Path | None:
    """Find the actual visual file corresponding to a linked image ID."""

    if not linked_image_id:
        return None

    matches = list(
        VISUALS_DIR.glob(f"*{linked_image_id}.png")
    )

    if not matches:
        return None

    return matches[0]


def _get_visuals(
    evidence_chunks: list[dict],
) -> list[Path]:
    """Return unique visual files referenced by retrieved evidence."""

    visuals = []
    seen = set()

    for chunk in evidence_chunks:
        linked_image_id = chunk.get("linked_image_id")

        if not linked_image_id:
            continue

        image_path = _find_visual(linked_image_id)

        if image_path is None:
            continue

        resolved = str(image_path.resolve())

        if resolved not in seen:
            seen.add(resolved)
            visuals.append(image_path)

    return visuals


def _has_valid_citations(
    answer: str,
    evidence_count: int,
) -> bool:
    """
    Check whether the model included at least one valid citation.

    Valid citations are [1], [2], ... corresponding to retrieved evidence.
    """

    if not answer:
        return False

    citations = re.findall(
        r"\[(\d+)\]",
        answer,
    )

    if not citations:
        return False

    for citation in citations:
        number = int(citation)

        if number < 1 or number > evidence_count:
            return False

    return True


def _add_deterministic_citations(
    answer: str,
    evidence_chunks: list[dict],
) -> str:
    """
    Ensure generated factual sentences have valid citations.

    The model is still instructed to cite evidence itself. This function
    acts as a deterministic fallback when the model forgets citations.

    For an uncited answer, evidence [1] is used as the fallback because
    retrieved evidence is already ranked by the retrieval pipeline.
    """

    if not answer or not evidence_chunks:
        return answer

    # If the model already produced valid citations, preserve its answer.
    if _has_valid_citations(
        answer,
        len(evidence_chunks),
    ):
        return answer

    fallback_citation = "[1]"

    # Split on sentence-ending punctuation while preserving punctuation.
    parts = re.split(
        r"(?<=[.!?])\s+",
        answer.strip(),
    )

    processed = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        # Do not add another citation if one somehow exists.
        if re.search(r"\[\d+\]", part):
            processed.append(part)
            continue

        processed.append(
            f"{part} {fallback_citation}"
        )

    return "\n".join(processed)


def _citation_retry_prompt(
    question: str,
    evidence_chunks: list[dict],
    previous_answer: str,
) -> str:
    """
    Create a repair prompt when the model forgot citations.
    """

    base_prompt = build_prompt(
        question,
        evidence_chunks,
    )

    return (
        f"{base_prompt}\n\n"
        "IMPORTANT CORRECTION:\n"
        "Your previous answer did not contain valid evidence citations.\n"
        "Rewrite the answer from scratch.\n"
        "Every factual sentence MUST end with a valid citation such as [1].\n"
        "Use ONLY citation numbers corresponding to the supplied evidence.\n"
        "Do not add any unsupported information.\n\n"
        f"Previous answer:\n{previous_answer}\n\n"
        "Corrected answer:"
    )

def _select_vision_evidence(question, evidence_chunks):
    """
    For clearly visual questions, give the vision model the visual evidence
    plus any text evidence from the same source/page.

    This prevents unrelated retrieved text from overriding what is actually
    visible in the image.
    """
    visual_terms = [
        "image", "picture", "portrait", "banner", "emblem",
        "symbol", "shown", "depicted", "figure", "plate",
        "diagram", "illustration", "drawing", "appearance",
        "holding", "wearing", "looks like", "colors", "visual"
    ]

    q = question.lower()

    if not any(term in q for term in visual_terms):
        return evidence_chunks

    visual_chunks = [
        chunk for chunk in evidence_chunks
        if chunk.get("linked_image_id")
    ]

    if not visual_chunks:
        return evidence_chunks

    selected = list(visual_chunks)

    # Keep text only when it belongs to the same source/page
    # as one of the selected visual evidence items.
    for chunk in evidence_chunks:
        if chunk in selected:
            continue

        for visual in visual_chunks:
            same_source = (
                chunk.get("source_file")
                and chunk.get("source_file") == visual.get("source_file")
            )

            same_page = (
                chunk.get("page_number") is not None
                and visual.get("page_number") is not None
                and chunk.get("page_number") == visual.get("page_number")
            )

            if same_source and same_page:
                selected.append(chunk)
                break

    return selected

def _select_vision_evidence(
    question: str,
    evidence_chunks: list[dict],
) -> list[dict]:
    """
    For clearly visual questions, prioritize the actual visual evidence.

    Keep:
    - visual evidence containing linked_image_id
    - text evidence only when it belongs to the same source/page
      as the visual evidence

    This prevents unrelated retrieved text from overriding what is
    actually visible in the image.
    """

    visual_terms = [
        "image",
        "picture",
        "portrait",
        "banner",
        "emblem",
        "symbol",
        "shown",
        "depicted",
        "figure",
        "plate",
        "diagram",
        "illustration",
        "drawing",
        "appearance",
        "holding",
        "wearing",
        "looks like",
        "colors",
        "visual",
    ]

    question_lower = question.lower()

    # If this isn't a visual question, keep the original evidence.
    if not any(term in question_lower for term in visual_terms):
        return evidence_chunks

    visual_chunks = [
        chunk
        for chunk in evidence_chunks
        if chunk.get("linked_image_id")
    ]

    # No visual evidence found.
    if not visual_chunks:
        return evidence_chunks

    selected = list(visual_chunks)

    # Keep only text evidence that belongs to the same
    # source AND page as a visual evidence item.
    for chunk in evidence_chunks:
        if chunk in selected:
            continue

        for visual in visual_chunks:
            same_source = (
                chunk.get("source_file")
                and chunk.get("source_file")
                == visual.get("source_file")
            )

            same_page = (
                chunk.get("page_number") is not None
                and visual.get("page_number") is not None
                and chunk.get("page_number")
                == visual.get("page_number")
            )

            if same_source and same_page:
                selected.append(chunk)
                break

    return selected

def generate_answer(
    question: str,
    evidence_chunks: list[dict],
    max_retries: int = 4,
) -> str:
    """
    Generate a grounded answer.

    If retrieved evidence contains images, use the vision model.
    Otherwise use the normal text model.
    """

    if not evidence_chunks:
        return (
            "The provided evidence is insufficient to answer "
            "this question."
        )

    visuals = _get_visuals(evidence_chunks)

    if LLM_BACKEND == "ollama":

        if visuals:

            generation_evidence = _select_vision_evidence(
                question,
                evidence_chunks,
            )

            generation_visuals = _get_visuals(
                generation_evidence,
            )

            answer = _generate_with_ollama_vision(
                question,
                generation_evidence,
                generation_visuals,
            )

            return _add_deterministic_citations(
                answer,
                generation_evidence,
            )

        prompt = build_prompt(
            question,
            evidence_chunks,
        )

        answer = _generate_with_ollama(prompt)

        if _has_valid_citations(
            answer,
            len(evidence_chunks),
        ):
            return answer

        retry_prompt = _citation_retry_prompt(
            question,
            evidence_chunks,
            answer,
        )

        repaired = _generate_with_ollama(
            retry_prompt,
        )

        if _has_valid_citations(
            repaired,
            len(evidence_chunks),
        ):
            return repaired

        return _add_deterministic_citations(
            answer,
            evidence_chunks,
        )

    # OpenRouter remains text-only in this implementation.

    prompt = build_prompt(
        question,
        evidence_chunks,
    )

    answer = _generate_with_openrouter(
        prompt,
        max_retries=max_retries,
    )

    if _has_valid_citations(
        answer,
        len(evidence_chunks),
    ):
        return answer

    retry_prompt = _citation_retry_prompt(
        question,
        evidence_chunks,
        answer,
    )

    repaired = _generate_with_openrouter(
        retry_prompt,
        max_retries=max_retries,
    )

    if _has_valid_citations(
        repaired,
        len(evidence_chunks),
    ):
        return repaired

    return _add_deterministic_citations(
        answer,
        evidence_chunks,
    )


def _generate_with_ollama(
    prompt: str,
) -> str:
    """Call the normal text-only Ollama model."""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
        },
        timeout=120,
    )

    if response.status_code == 404:
        raise RuntimeError(
            f"Ollama model '{OLLAMA_MODEL}' not found. "
            f"Pull it first."
        )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]


def _generate_with_ollama_vision(
    question: str,
    evidence_chunks: list[dict],
    visuals: list[Path],
) -> str:
    """
    Call the vision-capable Ollama model with actual image data.
    """

    prompt = build_prompt(
        question,
        evidence_chunks,
    )

    prompt += """

IMPORTANT VISUAL INSTRUCTION:

One or more retrieved images are attached to this message.

Look directly at the actual image pixels when answering.

The images are evidence from the Ashen Era Archive.

If the question asks about an object, symbol, number, person,
diagram, table, emblem, portrait, creature, or other visual detail,
inspect the image itself.

Do NOT rely on the filename alone.

Describe only what you can actually see.

Do not invent:

- colors
- objects
- symbols
- text
- numbers
- meanings
- identities
- relationships
- visual details

Every factual claim must include a valid evidence citation
such as [1] or [2].

If the image does not provide enough information, say that
the provided visual evidence is insufficient.
"""

    encoded_images = []

    for image_path in visuals:
        with open(image_path, "rb") as image_file:
            encoded_images.append(
                base64.b64encode(
                    image_file.read()
                ).decode("utf-8")
            )

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_VISION_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                    "images": encoded_images,
                },
            ],
            "stream": False,
        },
        timeout=180,
    )

    if response.status_code == 404:
        raise RuntimeError(
            f"Ollama vision model '{OLLAMA_VISION_MODEL}' "
            f"not found. Pull it first."
        )

    response.raise_for_status()

    data = response.json()

    answer = data["message"]["content"]

    # Vision models sometimes forget citation formatting.
    if _has_valid_citations(
        answer,
        len(evidence_chunks),
    ):
        return answer

    # Try once to make the model repair the citation.
    retry_prompt = _citation_retry_prompt(
        question,
        evidence_chunks,
        answer,
    )

    retry_prompt += """

The same retrieved images are still attached.

You MUST inspect those actual images again before answering.

Do not replace visual evidence with guesses.
"""

    retry_response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_VISION_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": retry_prompt,
                    "images": encoded_images,
                },
            ],
            "stream": False,
        },
        timeout=180,
    )

    retry_response.raise_for_status()

    retry_data = retry_response.json()

    repaired = retry_data["message"]["content"]

    if _has_valid_citations(
        repaired,
        len(evidence_chunks),
    ):
        return repaired

    # Deterministic fallback.
    return _add_deterministic_citations(
        answer,
        evidence_chunks,
    )


def _generate_with_openrouter(
    prompt: str,
    max_retries: int = 4,
) -> str:
    """Call OpenRouter with exponential backoff."""

    import time

    for attempt in range(max_retries):

        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}"
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            },
            timeout=60,
        )

        if response.status_code == 429:
            wait = 2 ** attempt
            time.sleep(wait)
            continue

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"]

    raise RuntimeError(
        "OpenRouter rate limit exceeded after retries. "
        "Try again shortly."
    )


if __name__ == "__main__":

    fake_chunks = [
        {
            "source_file": "codex_vol2.pdf",
            "page_number": 14,
            "text": (
                "The Ashen Council seal depicts a raven "
                "clutching a broken chain."
            ),
        }
    ]

    print(
        generate_answer(
            "What does the Ashen Council seal depict?",
            fake_chunks,
        )
    )