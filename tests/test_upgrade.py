import io
import json
import os
import re

import numpy as np
import pytest

from config import Config
from modules import evaluation, history, model_registry
from modules.dicom_support import available as dicom_available
from modules.dicom_support import convert_to_png
from modules.model_adapters import TorchvisionCheckpointAdapter, build_adapter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_ROOT = os.path.join(ROOT, "models")


# ------------------------------------------------------------------ registry
def test_builtin_registry_marks_densenet_available():
    specs = model_registry.list_models(MODELS_ROOT)
    keys = {spec["key"] for spec in specs}
    assert model_registry.DENSENET_KEY in keys
    dense = model_registry.get_model(MODELS_ROOT, model_registry.DENSENET_KEY)
    assert dense["adapter"] == "densenet_hf"
    assert dense["task"] == model_registry.TASK_XRAY
    if os.path.isfile(os.path.join(
            MODELS_ROOT, "xray", "checkpoints", "binary", "densenet121.pt")):
        assert dense["available"] is True


def test_unavailable_local_models_are_listed_not_hidden():
    specs = model_registry.list_models(MODELS_ROOT,
                                       task=model_registry.TASK_XRAY)
    local = [spec for spec in specs if spec["adapter"] == "torchvision_ckpt"]
    assert local, "expected optional local model entries"
    for spec in local:
        assert isinstance(spec["available"], bool)
        if not os.path.isfile(spec["checkpoint_path"]):
            assert spec["available"] is False


def test_custom_spec_discovery(tmp_path):
    custom = tmp_path / "custom" / "demo-model"
    custom.mkdir(parents=True)
    (custom / "best.pt").write_bytes(b"fake")
    spec = {
        "key": "demo-model", "name": "Demo", "architecture": "ResNet18",
        "task": "chest-xray-pneumonia", "labels": ["NORMAL", "PNEUMONIA"],
        "source": "local", "checkpoint": "custom/demo-model/best.pt",
        "version": "v0", "dataset": "demo", "license": "demo",
        "training_status": "trained", "limitations": ["demo only"],
        "preprocessing": "224", "adapter": "torchvision_ckpt",
        "gradcam_layer": "layer4",
    }
    (custom / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    bad = tmp_path / "custom" / "broken"
    bad.mkdir(parents=True)
    (bad / "spec.json").write_text("{}", encoding="utf-8")

    specs = model_registry.list_models(str(tmp_path))
    demo = model_registry.get_model(str(tmp_path), "demo-model")
    assert demo is not None
    assert demo["available"] is True
    assert demo["checkpoint_path"].endswith(os.path.join("demo-model",
                                                         "best.pt"))
    assert all(item["key"] != "broken" for item in specs)


def test_torchvision_adapter_reports_missing_checkpoint(tmp_path):
    spec = model_registry.get_model(MODELS_ROOT,
                                    "resnet18-pneumoniamnist-local")
    adapter = TorchvisionCheckpointAdapter(spec, str(tmp_path / "nope.pt"))
    assert adapter.status()["available"] is False
    result = adapter.analyze_bytes(b"irrelevant")
    assert not result["ok"] and result["label"] is None


def test_build_adapter_types():
    dense = model_registry.get_model(MODELS_ROOT, model_registry.DENSENET_KEY)
    adapter = build_adapter(dense, MODELS_ROOT)
    assert type(adapter).__name__ == "XrayAnalyzer"


# ---------------------------------------------------------------- evaluation
def test_compute_metrics_known_values():
    labels = [1, 1, 0, 0]
    probs = [0.9, 0.4, 0.6, 0.1]
    metrics = evaluation.compute_metrics(labels, probs, threshold=0.5)
    assert metrics["confusion"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert metrics["accuracy"] == 0.5
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["auc"] == 0.75
    assert metrics["accuracy_ci95"] is not None
    assert metrics["calibration"]
    assert sum(row["count"] for row in metrics["calibration"]) == 4


def test_compute_metrics_rejects_bad_input():
    with pytest.raises(ValueError):
        evaluation.compute_metrics([], [])
    with pytest.raises(ValueError):
        evaluation.compute_metrics([1, 0], [0.5])


def test_evaluate_npz_wiring_with_stub_adapter():
    npz = Config.DATASET_NPZ
    if not os.path.isfile(npz):
        pytest.skip("dataset not present")

    class StubAdapter:
        def analyze_image(self, image):
            return {"ok": True,
                    "pneumonia_probability": 0.1 if image.size[0] < 64 else 0.9}

    spec = {"key": "stub", "name": "Stub", "version": "test"}
    report = evaluation.evaluate_npz(StubAdapter(), spec, npz, split="test",
                                     limit=4)
    assert report["evaluated_samples"] == 4
    metrics = report["metrics"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["roc_points"]) >= 2
    assert report["dataset"].startswith("PneumoniaMNIST")


def test_report_store_roundtrip_and_expiry(tmp_path):
    config = {"EVAL_STORE_FOLDER": str(tmp_path)}
    report = {"model_key": "x", "model_name": "X", "metrics": {"accuracy": 1}}
    report_id = evaluation.save_report(config, report)
    loaded = evaluation.load_report(config, report_id)
    assert loaded["model_name"] == "X"
    assert evaluation.load_report(config, "../etc/passwd") is None
    assert len(evaluation.list_reports(config)) == 1
    old = os.path.join(str(tmp_path), "old.json")
    with open(old, "w", encoding="utf-8") as handle:
        json.dump({"model_name": "old"}, handle)
    os.utime(old, (1, 1))
    assert len(evaluation.list_reports(config)) == 1
    evaluation.sweep_reports(config)
    assert not os.path.isfile(old)


# ------------------------------------------------------------------- history
def test_history_metadata_only_and_clear(tmp_path):
    config = {"HISTORY_FOLDER": str(tmp_path), "HISTORY_MAX_ENTRIES": 3}
    sid = "a" * 32
    history.record(config, sid, "xray", "DenseNet: NORMAL",
                   {"model_key": "dense", "label": "NORMAL",
                    "nested": {"secret": "value"}, "pixels": [1, 2, 3]})
    history.record(config, sid, "labs", "2 measurements", {"count": 2})
    history.record(config, sid, "report", "PDF, 1 pages", {"pages": 1})
    history.record(config, sid, "summary", "exported", {"chars": 10})
    entries = history.entries(config, sid)
    assert len(entries) == 3
    assert entries[0]["kind"] == "summary"
    raw = (tmp_path / f"history_{sid}.json").read_text(encoding="utf-8")
    assert "secret" not in raw and "pixels" not in raw
    history.clear(config, sid)
    assert history.entries(config, sid) == []


def test_history_rejects_unsafe_ids(tmp_path):
    config = {"HISTORY_FOLDER": str(tmp_path)}
    history.record(config, "../../evil", "xray", "nope")
    assert os.listdir(tmp_path) == []


# --------------------------------------------------------------------- DICOM
def test_dicom_conversion_roundtrip(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    if not dicom_available():
        pytest.skip("pydicom unavailable")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    dataset = Dataset()
    dataset.file_meta = FileMetaDataset()
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.file_meta.MediaStorageSOPClassUID = (
        "1.2.840.10008.5.1.4.1.1.1")
    dataset.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "DX"
    dataset.Rows, dataset.Columns = 32, 32
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.arange(32 * 32, dtype=np.uint16).tobytes()
    buffer = io.BytesIO()
    pydicom.dcmwrite(buffer, dataset, write_like_original=False)

    png, meta = convert_to_png(buffer.getvalue())
    assert png.startswith(b"\x89PNG")
    assert meta["Modality"] == "DX"
    assert "PatientName" not in meta


def test_dicom_rejects_garbage():
    if not dicom_available():
        pytest.skip("pydicom unavailable")
    with pytest.raises(ValueError):
        convert_to_png(b"not a dicom file")


# ---------------------------------------------------------------- web pages
def test_new_pages_render(client):
    for path in ("/models", "/models/compare", "/status", "/eval", "/history",
                 "/models/densenet121-pneumonia-hf"):
        response = client.get(path)
        assert response.status_code == 200, path
    assert b"Model center" in client.get("/models").data
    assert b"DenseNet121" in client.get("/models").data


def test_unknown_model_and_report_are_404(client):
    assert client.get("/models/nope").status_code == 404
    assert client.get("/eval/report/" + "0" * 32).status_code == 404


def test_security_headers_and_csrf_input_present(client):
    response = client.get("/xray")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert b'name="csrf_token"' in response.data


def test_csrf_enforced_outside_testing(tmp_path):
    from app import create_app

    class StrictConfig(Config):
        TESTING = False
        UPLOAD_FOLDER = str(tmp_path / "uploads")
        REPORT_STORE_FOLDER = str(tmp_path / "reports")
        EVAL_STORE_FOLDER = str(tmp_path / "evals")
        HISTORY_FOLDER = str(tmp_path / "history")

    app = create_app(StrictConfig)
    client = app.test_client()
    assert client.post("/reset").status_code == 400
    page = client.get("/summary")
    match = re.search(rb'name="csrf_token" value="([^"]+)"', page.data)
    assert match
    token = match.group(1).decode()
    assert client.post("/reset",
                       data={"csrf_token": token}).status_code == 302

def test_roc_svg_renders_points():
    from modules import charts

    svg = charts.roc_svg([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert svg.startswith("<svg")
    assert "<polyline" in svg
    assert 'aria-label="ROC curve' in svg
    assert "0.00" in svg and "1.00" in svg
    assert charts.roc_svg([]) == ""

def test_eval_report_json_download(client):
    config = client.application.config
    report_id = evaluation.save_report(config, {
        "model_key": "densenet121-pneumonia-hf",
        "metrics": {"roc_points": [[0.0, 0.0], [1.0, 1.0]]},
        "evaluated_samples": 1,
    })
    response = client.get(f"/eval/report/{report_id}.json")
    assert response.status_code == 200
    assert response.mimetype == "application/json"
    assert "attachment" in response.headers["Content-Disposition"]
    payload = json.loads(response.get_data(as_text=True))
    assert payload["id"] == report_id
    assert payload["evaluated_samples"] == 1
    assert client.get("/eval/report/" + "0" * 32 + ".json").status_code == 404
