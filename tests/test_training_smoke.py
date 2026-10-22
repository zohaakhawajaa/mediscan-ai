import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_training_smoke_run_writes_real_artifacts(tmp_path):
    """End-to-end training wiring on a tiny slice (no downloads)."""
    npz = os.path.join(ROOT, "validation_data", "pneumoniamnist.npz")
    if not os.path.isfile(npz):
        import pytest
        pytest.skip("dataset not present")
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "train_xray.py"),
         "--model", "resnet18", "--smoke", "--out", str(tmp_path / "custom")],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr

    run_dir = tmp_path / "custom" / "resnet18-pneumoniamnist-local"
    assert (run_dir / "best.pt").is_file()
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    assert spec["adapter"] == "torchvision_ckpt"
    assert spec["torchvision_arch"] == "resnet18"
    assert spec["task"] == "chest-xray-pneumonia"
    metrics = spec["reported_metrics"]
    assert "reduced" in metrics["note"]
    assert "SMOKE" in spec["notes"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 1
    assert (run_dir / "test_probabilities.npy").is_file()
