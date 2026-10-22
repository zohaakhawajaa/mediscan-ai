"""Regression tests from the validation audit (2026-10-18).

Covers the three audit findings:
1. Preprocessing must exactly match the model card's torchvision transform
   (bilinear resize) — a bicubic/pillow-default mismatch was found and fixed.
2. Inputs below MIN_INPUT_SIDE must be rejected: the audit showed the model
   saturates toward PNEUMONIA on upscaled 28x28 images (specificity 0.0 on
   the full PneumoniaMNIST test split), so small inputs are refused rather
   than given a misleading score.
3. On genuine, correctly-preprocessed chest X-rays the class-to-probability
   mapping must be directionally correct (p = P(pneumonia)).

Genuine-sample tests use PneumoniaMNIST (CC BY 4.0, MedMNIST/Kermany) stored
locally under validation_data/; they skip when the file is absent.
"""

import io
import os

import numpy as np
import pytest
from PIL import Image

from modules.xray_analyzer import (DISCLAIMER, MIN_INPUT_SIDE, XrayAnalyzer)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_PATH = os.path.join(ROOT, "validation_data", "pneumoniamnist.npz")
MODEL_DIR = os.path.join(ROOT, "models", "xray")


def _img_bytes(size=(256, 256), mode="L", color=128) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def _rank_auc(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for val in np.unique(scores):
        ranks[scores == val] = ranks[scores == val].mean()
    u = ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


# ------------------------------------------------------------- synthetic only
def test_preprocessing_parity_with_model_card(tmp_path):
    from torchvision import transforms
    analyzer = XrayAnalyzer(model_dir=str(tmp_path))
    model_card_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    rng = np.random.default_rng(7)
    img = Image.fromarray(
        rng.integers(0, 255, (300, 250), dtype=np.uint8), mode="L")
    ours = analyzer._preprocess(img.convert("RGB"))
    theirs = model_card_tf(img).unsqueeze(0)
    assert float((ours - theirs).abs().max()) < 1e-4


def test_low_resolution_input_rejected(tmp_path):
    analyzer = XrayAnalyzer(model_dir=str(tmp_path))
    result = analyzer.analyze_bytes(_img_bytes(size=(MIN_INPUT_SIDE - 1,
                                                     MIN_INPUT_SIDE - 1)))
    assert not result["ok"]
    assert result["label"] is None
    assert result["pneumonia_probability"] is None
    assert "resolution too low" in result["error"]


def test_disclaimer_is_compact_research_use_only():
    assert "For research use only" in DISCLAIMER
    assert "Not for diagnostic use" in DISCLAIMER
    assert "pediatric" in DISCLAIMER


# --------------------------------------------------- genuine X-rays (CC BY 4.0)
@pytest.mark.parametrize("subset", ["all"])
def test_genuine_pneumoniamnist_directional_mapping(tmp_path, subset):
    if not os.path.isfile(NPZ_PATH):
        pytest.skip("PneumoniaMNIST npz not present in validation_data/")
    if not os.path.isfile(os.path.join(MODEL_DIR, "checkpoints", "binary",
                                       "densenet121.pt")):
        pytest.skip("model checkpoint not downloaded")
    data = np.load(NPZ_PATH)
    images_arr = data["test_images"]
    labels = data["test_labels"].reshape(-1)
    # Balanced convenience subset: first 15 of each class (genuine X-rays).
    chosen = (list(np.where(labels == 0)[0][:15])
              + list(np.where(labels == 1)[0][:15]))
    analyzer = XrayAnalyzer(model_dir=MODEL_DIR)
    analyzer._ensure_model()
    import torch
    probs = []
    with torch.no_grad():
        for i in chosen:
            tensor = analyzer._preprocess(Image.fromarray(images_arr[i],
                                                          mode="L"))
            probs.append(float(torch.sigmoid(analyzer._model(tensor)).item()))
    chosen_labels = labels[chosen]
    auc = _rank_auc(probs, chosen_labels)
    mean_pos = float(np.mean(np.array(probs)[chosen_labels == 1]))
    mean_neg = float(np.mean(np.array(probs)[chosen_labels == 0]))
    # Directional sanity only (not clinical validation). An inverted or
    # broken class mapping would drive AUC far below 0.5.
    assert mean_pos > mean_neg
    assert auc > 0.6
