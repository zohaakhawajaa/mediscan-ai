import pymupdf as fitz

from modules.report_extractor import extract_from_pdf, find_lab_candidates


def _pdf_bytes(text: str, blank_pages: int = 0) -> bytes:
    doc = fitz.open()
    if text:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    for _ in range(blank_pages):
        page = doc.new_page()
        page.insert_image(fitz.Rect(50, 50, 300, 300),
                          stream=_tiny_png())
    data = doc.tobytes()
    doc.close()
    return data


def _tiny_png() -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (10, 10), color=128).save(buf, format="PNG")
    return buf.getvalue()


def test_extract_text_based_pdf():
    result = extract_from_pdf(_pdf_bytes(
        "Patient lab report\nHemoglobin: 13.2 g/dL\nGlucose 95 mg/dL"))
    assert result["ok"]
    assert result["num_pages"] == 1
    assert not result["needs_ocr"]
    assert "Hemoglobin" in result["pages"][0]["text"]


def test_scanned_page_flagged_for_ocr():
    result = extract_from_pdf(_pdf_bytes("", blank_pages=2))
    assert result["ok"]
    assert result["needs_ocr"]
    assert "OCR" in result["ocr_message"]


def test_invalid_pdf_rejected():
    result = extract_from_pdf(b"not a pdf at all")
    assert not result["ok"]
    assert "error" in result


def test_find_lab_candidates():
    text = ("CBC: Hemoglobin 11.4 g/dL, WBC 12.3, Platelets 210, "
            "Glucose: 88 mg/dL, Creatinine 0.9")
    candidates = find_lab_candidates(text)
    found = {c["test"]: c["value"] for c in candidates}
    assert found["hemoglobin"] == 11.4
    assert found["wbc"] == 12.3
    assert found["platelets"] == 210
    assert found["glucose"] == 88
    assert found["creatinine"] == 0.9


def test_find_lab_candidates_ignores_mmol_glucose():
    candidates = find_lab_candidates("Glucose 5.2 mmol/L")
    assert all(c["test"] != "glucose" for c in candidates)
