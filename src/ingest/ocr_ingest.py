"""
ocr_ingest.py
-------------
OCR fallback for pages that pdf_ingest.py flags as `is_likely_scanned=True`
(i.e. near-zero extractable text -> the page is an image of text, not real text).

Owner: Data / Extraction lead

Uses Tesseract via pytesseract. Requires the tesseract-ocr system binary
to be installed (not just the Python package):
    Ubuntu/Debian:  sudo apt-get install tesseract-ocr
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytesseract
from PIL import Image


def ocr_page(pdf_path: str | Path, page_number: int, dpi: int = 300) -> str:
    """Render one page to an image and OCR it. page_number is 1-indexed."""
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    pix = page.get_pixmap(dpi=dpi)
    doc.close()

    img_path = Path("/tmp") / f"ocr_tmp_{Path(pdf_path).stem}_p{page_number}.png"
    pix.save(img_path)

    text = pytesseract.image_to_string(Image.open(img_path))
    img_path.unlink(missing_ok=True)
    return text.strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python src/ingest/ocr_ingest.py <path_to_pdf> <page_number>")
        sys.exit(1)

    result = ocr_page(sys.argv[1], int(sys.argv[2]))
    print(result)
