import io

import fitz


def test_pages_render(client):
    for path in ("/", "/xray", "/labs", "/report", "/summary", "/about"):
        resp = client.get(path)
        assert resp.status_code == 200
        assert b"MediScan" in resp.data


def test_lab_interpretation_flow(client):
    resp = client.post("/labs/interpret", data={
        "test": ["hemoglobin", "glucose"],
        "value": ["10.5", "90"],
        "range": ["adult_female", "fasting_adult"],
    })
    assert resp.status_code == 200
    assert b"LOW" in resp.data
    assert b"NORMAL" in resp.data


def test_lab_empty_post_shows_error(client):
    resp = client.post("/labs/interpret", data={
        "test": [""], "value": [""], "range": ["adult"]})
    assert b"at least one" in resp.data


def test_xray_rejects_non_image(client):
    resp = client.post("/xray", data={
        "xray_image": (io.BytesIO(b"hello"), "notes.txt")},
        content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Unsupported image format" in resp.data


def test_pdf_flow_and_summary(client):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hemoglobin: 14.0 g/dL")
    pdf = doc.tobytes()
    doc.close()
    resp = client.post("/report", data={
        "pdf_file": (io.BytesIO(pdf), "report.pdf"), "action": "extract"},
        content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Hemoglobin" in resp.data

    resp = client.post("/report", data={
        "reviewed_text": "Hemoglobin: 14.0 g/dL", "action": "review"})
    assert resp.status_code == 200
    assert b"hemoglobin" in resp.data

    resp = client.get("/summary")
    assert resp.status_code == 200
    assert b"PDF report extraction" in resp.data

    resp = client.get("/summary/download")
    assert resp.status_code == 200
    assert b"MediScan AI" in resp.data


def test_upload_oversize_rejected(client):
    big = io.BytesIO(b"0" * (11 * 1024 * 1024))
    resp = client.post("/xray", data={"xray_image": (big, "big.png")},
                       content_type="multipart/form-data")
    assert resp.status_code == 413
    assert b"too large" in resp.data.lower()


def test_xray_result_page_does_not_present_scores_as_reliable(app, client,
                                                              monkeypatch):
    from modules.xray_analyzer import DISCLAIMER

    class FakeAnalyzer:
        def status(self):
            return {"available": True, "detail": "mocked"}

        def analyze_bytes(self, data):
            return {"ok": True, "label": "PNEUMONIA",
                    "pneumonia_probability": 0.97, "confidence": 0.97,
                    "model": "mock", "disclaimer": DISCLAIMER}

    app.extensions["xray_analyzer"] = FakeAnalyzer()
    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (256, 256), color=180).save(buf, format="PNG")
    resp = client.post("/xray", data={
        "xray_image": (io.BytesIO(buf.getvalue()), "xr.png")},
        content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Model confidence" in resp.data
    assert b"For research use only" in resp.data
    assert b"Not for diagnostic use" in resp.data
    # The old misleading label must not reappear as a bare claim.
    assert b"<th>Confidence</th>" not in resp.data


def test_low_resolution_upload_rejected_end_to_end(client):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (32, 32), color=128).save(buf, format="PNG")
    resp = client.post("/xray", data={
        "xray_image": (io.BytesIO(buf.getvalue()), "tiny.png")},
        content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"resolution too low" in resp.data
