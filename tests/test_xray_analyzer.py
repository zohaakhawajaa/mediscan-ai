import io
import os

from PIL import Image

from modules.xray_analyzer import XrayAnalyzer, validate_image_upload


def _tiny_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("L", (256, 256), color=200).save(buf, format="PNG")
    return buf.getvalue()


def test_validate_image_upload():
    assert validate_image_upload("scan.png", {"png", "jpg"}) is None
    assert validate_image_upload("scan.gif", {"png", "jpg"}) is not None
    assert validate_image_upload("", {"png"}) is not None
    assert validate_image_upload("noext", {"png"}) is not None


def test_analyzer_without_model_fails_gracefully(tmp_path):
    analyzer = XrayAnalyzer(model_dir=str(tmp_path / "models"))
    result = analyzer.analyze_bytes(_tiny_png_bytes())
    # Either the download fails (offline) or inference runs; if it fails,
    # it must NOT fabricate a label or probability.
    if not result["ok"]:
        assert result["label"] is None
        assert result["pneumonia_probability"] is None
        assert result["error"]
    else:
        assert result["label"] in ("NORMAL", "PNEUMONIA")
        assert 0.0 <= result["pneumonia_probability"] <= 1.0


def test_corrupt_image_rejected(tmp_path):
    analyzer = XrayAnalyzer(model_dir=str(tmp_path / "models"))
    result = analyzer.analyze_bytes(b"definitely not an image")
    assert not result["ok"]
    assert result["label"] is None


def test_checkpoint_present_in_repo_models():
    """Integration: real checkpoint + real inference (skipped if absent)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(root, "models", "xray")
    ckpt = os.path.join(model_dir, "checkpoints", "binary", "densenet121.pt")
    if not os.path.isfile(ckpt):
        import pytest
        pytest.skip("checkpoint not downloaded")
    analyzer = XrayAnalyzer(model_dir=model_dir)
    result = analyzer.analyze_bytes(_tiny_png_bytes())
    assert result["ok"], result.get("error")
    assert result["label"] in ("NORMAL", "PNEUMONIA")
    assert 0.0 <= result["pneumonia_probability"] <= 1.0


def test_gradcam_heatmap_optional_and_valid():
    """Integration: heatmap only when requested, valid PNG otherwise."""
    import base64

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(root, "models", "xray")
    ckpt = os.path.join(model_dir, "checkpoints", "binary", "densenet121.pt")
    if not os.path.isfile(ckpt):
        import pytest
        pytest.skip("checkpoint not downloaded")
    analyzer = XrayAnalyzer(model_dir=model_dir)

    result = analyzer.analyze_bytes(_tiny_png_bytes(), gradcam=True)
    assert result["ok"], result.get("error")
    heatmap = result.get("heatmap_b64")
    assert heatmap, "expected a Grad-CAM heatmap"
    assert base64.b64decode(heatmap).startswith(b"\x89PNG")

    default = analyzer.analyze_bytes(_tiny_png_bytes())
    assert default["ok"], default.get("error")
    assert not default.get("heatmap_b64")
