"""PDF medical-report text extraction module (PyMuPDF).

Extracts text from text-based PDFs. Scanned (image-only) PDFs cannot be read
without OCR; the module detects and reports this instead of guessing.
Extracted values are returned as *candidates for user review only* — nothing
is auto-interpreted and missing information is never fabricated.
"""

from __future__ import annotations

import re

import pymupdf as fitz

# (test_key, label, regex) — deliberately conservative patterns.
# Integer parts allow 1-3 digits so critically low single-digit results
# (e.g. hemoglobin 9.5, platelets 5) are still surfaced for review.
_LAB_PATTERNS = [
    ("hemoglobin", "Hemoglobin",
     r"(?:hemoglobin|haemoglobin|\bhb\b|\bhgb\b)[^\d]{0,25}"
     r"(\d{1,3}(?:\.\d+)?)\s*(?:g/?dl)?"),
    ("wbc", "WBC",
     r"(?:wbc|white\s+blood\s+cell(?:\s+count)?|leukocytes?)"
     r"[^\d]{0,25}(\d{1,3}(?:\.\d+)?)\s*(?:x?\s*10\^?3/?ul|k/?ul|thou/?ul)?"),
    ("platelets", "Platelets",
     r"(?:platelets?|plt|thrombocytes?)[^\d]{0,25}"
     r"(\d{1,4}(?:\.\d+)?)\s*(?:x?\s*10\^?3/?ul|k/?ul|thou/?ul)?"),
    ("glucose", "Glucose",
     r"(?:glucose|blood\s+glucose|fbg|rbg)[^\d]{0,25}"
     r"(\d{1,3}(?:\.\d+)?)\s*(?:mg/?dl|mmol/?l)?"),
    ("creatinine", "Creatinine",
     r"(?:creatinine|cr)[^\d]{0,25}"
     r"(\d{1,3}(?:\.\d{1,2})?)\s*(?:mg/?dl|umol/?l)?"),
]

_MIN_TEXT_LEN_FOR_TEXT_PAGE = 20


def extract_from_pdf(pdf_bytes: bytes) -> dict:
    """Extract per-page text from a PDF.

    Returns {"ok", "num_pages", "pages": [{page, text, scanned}], "needs_ocr"}.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        return {"ok": False, "error": f"Could not open PDF: {exc}",
                "num_pages": 0, "pages": [], "needs_ocr": False}
    if doc.page_count == 0:
        doc.close()
        return {"ok": False, "error": "The PDF contains no pages.",
                "num_pages": 0, "pages": [], "needs_ocr": False}

    pages = []
    for index, page in enumerate(doc):
        text = page.get_text("text").strip()
        scanned = (len(text) < _MIN_TEXT_LEN_FOR_TEXT_PAGE
                   and len(page.get_images(full=True)) > 0)
        pages.append({"page": index + 1, "text": text, "scanned": scanned})
    doc.close()

    needs_ocr = any(p["scanned"] for p in pages)
    return {"ok": True, "num_pages": len(pages), "pages": pages,
            "needs_ocr": needs_ocr,
            "ocr_message": (
                "Some pages appear to be scanned images with no extractable "
                "text. OCR is not enabled — please review the extraction "
                "below and correct it manually."
            ) if needs_ocr else None}


def find_lab_candidates(text: str) -> list[dict]:
    """Find possible lab values in reviewed text.

    Returned strictly as candidates for user review — not verified results.
    """
    candidates = []
    seen = set()
    for test_key, label, pattern in _LAB_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            raw = match.group(1)
            try:
                value = float(raw)
            except ValueError:
                continue
            # Guard against obvious unit confusions for glucose (mmol/L).
            snippet = text[match.start():match.end() + 12].lower()
            if test_key == "glucose" and "mmol" in snippet:
                continue
            key = (test_key, value, match.start())
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "test": test_key,
                "label": label,
                "value": value,
                "context": match.group(0).replace("\n", " ").strip()[:80],
            })
    return candidates
