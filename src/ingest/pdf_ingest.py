"""
pdf_ingest.py
-------------
Extracts text, images, and bounding-box metadata from PDF pages in the
Ashen Era Archive using PyMuPDF (fitz).

Owner: Data / Extraction lead

Output shape (one dict per page), designed to feed both:
  - src/indexing/  (text chunks -> embeddings + BM25)
  - src/retrieval/ (bbox crops for "show me the actual figure" answers)

Run directly for a quick sanity check on one file:
    python src/ingest/pdf_ingest.py data/raw/some_codex.pdf
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import fitz  # PyMuPDF


# Words that tend to introduce a labeled visual asset in this archive
# (banners, portraits, figure plates, threat-classification plates...).
# Extend this list as the extraction lead finds more patterns in the corpus.
FIGURE_CAPTION_HINTS = [
    "figure plate",
    "portrait of",
    "banner of",
    "threat-classification plate",
    "official illustration",
    "plate depicting",
]


@dataclass
class ExtractedImage:
    image_id: str
    page_number: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    image_path: str  # where the cropped PNG is saved
    nearby_caption: str | None = None  # best-guess caption text near the bbox


@dataclass
class ExtractedPage:
    source_file: str
    page_number: int
    text: str
    images: list[ExtractedImage] = field(default_factory=list)
    is_likely_scanned: bool = False  # True if page has ~no extractable text


def extract_pdf(pdf_path: str | Path, visuals_out_dir: str | Path) -> list[ExtractedPage]:
    """Extract all pages from one PDF file.

    Returns a list of ExtractedPage, one per page, with text + image metadata.
    Cropped images are saved to visuals_out_dir as PNGs.
    """
    pdf_path = Path(pdf_path)
    visuals_out_dir = Path(visuals_out_dir)
    visuals_out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages: list[ExtractedPage] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1  # human-readable, 1-indexed

        text = page.get_text("text")
        is_likely_scanned = len(text.strip()) < 20  # near-empty -> probably a scan, needs OCR

        page_record = ExtractedPage(
            source_file=pdf_path.name,
            page_number=page_number,
            text=text,
            is_likely_scanned=is_likely_scanned,
        )

        # --- Extract images with bounding boxes ---
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            bbox_rects = page.get_image_rects(xref)
            if not bbox_rects:
                continue
            bbox = bbox_rects[0]  # (x0, y0, x1, y1)

            # Crop the region directly from a high-res render of the page
            # (more reliable than extracting the raw embedded image stream).
            pix = page.get_pixmap(clip=bbox, dpi=200)
            image_id = f"{pdf_path.stem}_p{page_number}_img{img_index}"
            image_path = visuals_out_dir / f"{image_id}.png"
            pix.save(image_path)

            caption = _guess_nearby_caption(page, bbox)

            page_record.images.append(
                ExtractedImage(
                    image_id=image_id,
                    page_number=page_number,
                    bbox=tuple(bbox),
                    image_path=str(image_path),
                    nearby_caption=caption,
                )
            )

        pages.append(page_record)

    doc.close()
    return pages


def _guess_nearby_caption(page: "fitz.Page", bbox, search_margin: float = 40.0) -> str | None:
    """Look for text just below/above the image bbox that looks like a caption.

    TODO(search/extraction lead): this is a first-pass heuristic. Validate against
    real 'figure plate' / 'portrait of' pages and tighten the margin + matching
    logic once we see real layouts.
    """
    x0, y0, x1, y1 = bbox
    # Look in a strip just below the image first, then just above.
    below = fitz.Rect(x0 - search_margin, y1, x1 + search_margin, y1 + search_margin)
    above = fitz.Rect(x0 - search_margin, y0 - search_margin, x1 + search_margin, y0)

    for strip in (below, above):
        strip_text = page.get_textbox(strip).strip()
        if strip_text:
            lowered = strip_text.lower()
            if any(hint in lowered for hint in FIGURE_CAPTION_HINTS) or len(strip_text) < 150:
                return strip_text
    return None


def save_pages_json(pages: list[ExtractedPage], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in pages], f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/ingest/pdf_ingest.py <path_to_pdf>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    pages = extract_pdf(pdf_file, visuals_out_dir="data/visuals")

    scanned_count = sum(1 for p in pages if p.is_likely_scanned)
    image_count = sum(len(p.images) for p in pages)

    print(f"Extracted {len(pages)} pages from {pdf_file}")
    print(f"  - {scanned_count} page(s) look scanned (near-zero text -> needs OCR)")
    print(f"  - {image_count} image(s) extracted with bounding boxes")

    out_json = Path("data/processed") / f"{Path(pdf_file).stem}.json"
    save_pages_json(pages, out_json)
    print(f"Saved page records to {out_json}")
