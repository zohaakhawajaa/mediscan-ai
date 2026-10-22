import base64
import io
import json
import logging
import os
import re
import time
import urllib.parse
import zlib

import pymupdf as fitz
import pytest

from modules.lab_analyzer import analyze_entries
from modules.report_extractor import find_lab_candidates
from modules.routes import sweep_stale_uploads

UPLOAD_ERROR = b"mock analysis failure"


class _FailingAnalyzer:
    def status(self):
        return {"available": True, "detail": "mocked"}

    def analyze_bytes(self, data):
        return {"ok": False, "error": UPLOAD_ERROR.decode()}


class _OkAnalyzer:
    def status(self):
        return {"available": True, "detail": "mocked"}

    def analyze_bytes(self, data):
        from modules.xray_analyzer import DISCLAIMER
        return {"ok": True, "label": "NORMAL",
                "pneumonia_probability": 0.12, "confidence": 0.88,
                "model": "mock", "disclaimer": DISCLAIMER}


def _pdf_with_text(text):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _decode_session_cookie(set_cookie_headers):
    match = re.search(r"session=([^;\s]+)",
                      "\n".join(set_cookie_headers))
    assert match, "no session cookie was set"
    raw = urllib.parse.unquote(match.group(1)).split(".")[0]
    blob = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    try:
        blob = zlib.decompress(blob)
    except zlib.error:
        pass
    return blob.decode("utf-8", "replace")


def test_nonfinite_lab_values_rejected():
    for raw in ("nan", "inf", "-inf", "NaN", "INF"):
        result = analyze_entries([
            {"test": "hemoglobin", "value": raw, "range": "adult_male"}])
        assert result["results"] == [], raw
        assert result["errors"], raw


def test_nonfinite_lab_value_flashed_instead_of_normal(client):
    resp = client.post("/labs/interpret", data={
        "test": ["glucose"], "value": ["nan"], "range": ["fasting_adult"]})
    assert resp.status_code == 200
    assert b"finite number" in resp.data
    assert b"NORMAL" not in resp.data


def test_weird_filenames_never_return_500(client, app, monkeypatch):
    monkeypatch.setitem(app.extensions, "xray_analyzer", _FailingAnalyzer())
    for name in (".png", "..png", "图片.png", "scan"):
        resp = client.post("/xray", data={
            "xray_image": (io.BytesIO(b"junk-bytes"), name)},
            content_type="multipart/form-data")
        assert resp.status_code == 200, name
    assert os.listdir(app.config["UPLOAD_FOLDER"]) == []


def test_failed_analysis_leaves_no_file_behind(client, app, monkeypatch):
    monkeypatch.setitem(app.extensions, "xray_analyzer", _FailingAnalyzer())
    resp = client.post("/xray", data={
        "xray_image": (io.BytesIO(b"x" * 32), "chest.png")},
        content_type="multipart/form-data")
    assert resp.status_code == 200
    assert UPLOAD_ERROR in resp.data
    assert os.listdir(app.config["UPLOAD_FOLDER"]) == []


def test_new_upload_replaces_previous_file(client, app, monkeypatch):
    monkeypatch.setitem(app.extensions, "xray_analyzer", _OkAnalyzer())
    for index in range(2):
        resp = client.post("/xray", data={
            "xray_image": (io.BytesIO(b"x" * 32), f"chest{index}.png")},
            content_type="multipart/form-data")
        assert resp.status_code == 200
    assert len(os.listdir(app.config["UPLOAD_FOLDER"])) == 1


def test_pdf_without_extension_does_not_crash(client):
    resp = client.post("/report", data={
        "pdf_file": (io.BytesIO(b"not a pdf"), "labreport"),
        "action": "extract"}, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Unsupported file type" in resp.data


def test_candidate_form_unchecked_all_interprets_nothing(client):
    resp = client.post("/labs", data={
        "form_source": ["candidates"],
        "test": ["hemoglobin", "glucose"],
        "value": ["10.5", "90"],
        "range": ["adult_female", "fasting_adult"]})
    assert resp.status_code == 200
    assert b"Select at least one candidate" in resp.data
    assert b"LOW" not in resp.data
    assert b"NORMAL" not in resp.data


def test_candidate_form_interprets_only_checked_rows(client):
    resp = client.post("/labs", data={
        "form_source": ["candidates"],
        "test": ["hemoglobin", "glucose"],
        "value": ["10.5", "90"],
        "range": ["adult_female", "fasting_adult"],
        "use_candidate": ["0"]})
    assert resp.status_code == 200
    assert b"LOW" in resp.data
    assert b"NORMAL" not in resp.data


def test_manual_lab_form_still_interprets_every_row(client):
    resp = client.post("/labs/interpret", data={
        "test": ["hemoglobin", "glucose"],
        "value": ["10.5", "90"],
        "range": ["adult_female", "fasting_adult"]})
    assert resp.status_code == 200
    assert b"LOW" in resp.data
    assert b"NORMAL" in resp.data


def test_candidates_include_critical_low_and_wide_values():
    text = ("Hemoglobin 9.5 g/dL; WBC 4; Platelets 5; "
            "Glucose 4.2 mg/dL; Creatinine 12.0 mg/dL")
    found = {c["test"]: c["value"] for c in find_lab_candidates(text)}
    assert found["hemoglobin"] == 9.5
    assert found["wbc"] == 4
    assert found["platelets"] == 5
    assert found["glucose"] == 4.2
    assert found["creatinine"] == 12.0


def test_sweep_stale_uploads(tmp_path):
    old = tmp_path / "old.png"
    new = tmp_path / "new.png"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    past = time.time() - 48 * 60 * 60
    os.utime(old, (past, past))

    assert sweep_stale_uploads(str(tmp_path), 24 * 60 * 60) == 1
    assert not old.exists()
    assert new.exists()
    assert sweep_stale_uploads(str(tmp_path), 24 * 60 * 60) == 0
    assert sweep_stale_uploads(str(tmp_path), 0) == 0
    assert sweep_stale_uploads(str(tmp_path / "missing"), 100) == 0


def test_report_text_stored_server_side_not_in_cookie(client, app):
    resp = client.post("/report", data={
        "pdf_file": (io.BytesIO(_pdf_with_text("Hemoglobin: 14.0 g/dL")),
                     "report.pdf"),
        "action": "extract"}, content_type="multipart/form-data")
    assert resp.status_code == 200

    payload = _decode_session_cookie(resp.headers.getlist("Set-Cookie"))
    assert "Hemoglobin" not in payload
    assert "report_extraction" not in payload

    assert os.listdir(app.config["REPORT_STORE_FOLDER"])
    resp = client.get("/summary")
    assert resp.status_code == 200
    assert b"Hemoglobin" in resp.data


def test_expired_report_store_forgets_extraction(client, app):
    resp = client.post("/report", data={
        "pdf_file": (io.BytesIO(_pdf_with_text("Hemoglobin: 14.0 g/dL")),
                     "report.pdf"),
        "action": "extract"}, content_type="multipart/form-data")
    assert resp.status_code == 200

    store = app.config["REPORT_STORE_FOLDER"]
    name = os.listdir(store)[0]
    path = os.path.join(store, name)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["saved_at"] = time.time() - app.config["REPORT_TTL_SECONDS"] - 1
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    resp = client.get("/summary")
    assert b"(no PDF report processed)" in resp.data
    assert not os.path.exists(path)


def _png_bytes(size=256):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (size, size), color=180).save(buf, format="PNG")
    return buf.getvalue()


def test_decode_error_hides_internal_object_repr(client):
    resp = client.post("/xray", data={
        "xray_image": (io.BytesIO(b"definitely not an image"), "photo.png")},
        content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"object at 0x" not in resp.data
    assert b"BytesIO" not in resp.data
    assert b"cannot identify image" not in resp.data
    assert b"valid image file" in resp.data


def test_inference_error_is_generic_but_logged(monkeypatch, caplog):
    from modules.xray_analyzer import XrayAnalyzer

    internal_path = "C:\\internal\\models\\weights.bin"

    def _boom(self):
        raise RuntimeError("failed reading " + internal_path)

    monkeypatch.setattr(XrayAnalyzer, "_ensure_model", _boom)
    analyzer = XrayAnalyzer(model_dir=os.path.join("unused"))
    with caplog.at_level(logging.ERROR, logger="modules.xray_analyzer"):
        result = analyzer.analyze_bytes(_png_bytes())
    assert not result["ok"]
    assert internal_path not in result["error"]
    assert "internal error" in result["error"]
    assert internal_path in caplog.text


def test_production_refuses_missing_secret(monkeypatch):
    from app import create_app
    from conftest import TestConfig

    monkeypatch.setenv("MEDISCAN_ENV", "production")
    monkeypatch.delenv("MEDISCAN_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MEDISCAN_SECRET_KEY"):
        create_app(TestConfig)


def test_production_refuses_known_default_secret(monkeypatch):
    from app import create_app
    from conftest import TestConfig

    monkeypatch.setenv("MEDISCAN_ENV", "production")
    monkeypatch.setenv("MEDISCAN_SECRET_KEY", "dev-only-change-me")
    with pytest.raises(RuntimeError, match="MEDISCAN_SECRET_KEY"):
        create_app(TestConfig)


def test_production_starts_with_real_secret(monkeypatch):
    from app import create_app
    from conftest import TestConfig

    monkeypatch.setenv("MEDISCAN_ENV", "production")
    monkeypatch.setenv("MEDISCAN_SECRET_KEY", "unit-test-only-not-a-real-key")
    app = create_app(TestConfig)
    assert app.secret_key == "unit-test-only-not-a-real-key"


def test_development_still_starts_with_default_secret(monkeypatch):
    from app import create_app
    from conftest import TestConfig

    monkeypatch.delenv("MEDISCAN_ENV", raising=False)
    monkeypatch.delenv("MEDISCAN_SECRET_KEY", raising=False)
    app = create_app(TestConfig)
    assert app.secret_key == "dev-only-change-me"


def test_about_page_renders(client):
    resp = client.get("/about")
    assert resp.status_code == 200
    assert b"DenseNet121" in resp.data
    assert b"object at 0x" not in resp.data


def test_labs_page_has_no_reference_note_disclaimer(client):
    resp = client.post("/labs/interpret", data={
        "test": ["hemoglobin"], "value": ["10.8"],
        "range": ["adult_female"]})
    assert resp.status_code == 200
    assert b"LOW" in resp.data
    assert b"Flags indicate" not in resp.data
    assert b"not a diagnosis" not in resp.data


def test_summary_text_has_no_lab_note(client):
    client.post("/labs/interpret", data={
        "test": ["glucose"], "value": ["90"], "range": ["fasting_adult"]})
    resp = client.get("/summary/download")
    assert resp.status_code == 200
    assert b"Flags indicate" not in resp.data


def test_sample_cases_labelled_professionally(client):
    resp = client.get("/labs")
    assert resp.status_code == 200
    assert b"fictional" not in resp.data.lower()
    assert b"Sample cases" in resp.data
    assert b"Case A" in resp.data


def test_reset_clears_session_and_upload(app, client):
    resp = client.post("/labs/interpret", data={
        "test": ["hemoglobin"], "value": ["14.0"], "range": ["adult_male"]})
    assert resp.status_code == 200

    marker = "xray_reset_probe.png"
    upload_dir = app.config["UPLOAD_FOLDER"]
    with open(os.path.join(upload_dir, marker), "wb") as fh:
        fh.write(b"probe")
    with client.session_transaction() as sess:
        sess["xray_result"] = {"label": "NORMAL", "pneumonia_probability": 0.1,
                               "confidence": 0.9, "model": "mock",
                               "disclaimer": "mock disclaimer"}
        sess["xray_image_file"] = marker

    resp = client.post("/reset", follow_redirects=True)
    assert resp.status_code == 200
    assert b"Session cleared" in resp.data
    assert not os.path.exists(os.path.join(upload_dir, marker))

    resp = client.get("/summary")
    assert resp.data.count(b"Not run") == 3
    assert b"Completed" not in resp.data
