"""Validation audit for the MediScan pneumonia X-ray model.

Uses two GENUINE, openly-licensed sources (no synthetic inputs):
1. PneumoniaMNIST test split (624 images, CC BY 4.0, MedMNIST/Kermany).
   28x28 resolution — upscaled by the pipeline; expect degraded but
   directionally valid metrics. Quantitative sanity check only.
2. Full-resolution Kermany test images mirrored on Hugging Face
   (hf-vision/chest-xray-pneumonia, CC BY 4.0) — the same source dataset
   the model was trained on. Small sample; directional check only.

Outputs metrics and per-example predictions. NOT clinical validation.
"""

import io
import json
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.xray_analyzer import XrayAnalyzer  # noqa: E402

NPZ_PATH = os.path.join(ROOT, "validation_data", "pneumoniamnist.npz")
MODEL_DIR = os.path.join(ROOT, "models", "xray")
HF_DATASET = "hf-vision/chest-xray-pneumonia"
BATCH = 16


def auc_rank(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U / rank AUC
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    for val in np.unique(scores):
        mask = scores == val
        ranks[mask] = ranks[mask].mean()
    u = ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def batch_probs(analyzer, images):
    import torch
    model = analyzer._model
    probs = []
    with torch.no_grad():
        for i in range(0, len(images), BATCH):
            batch = torch.cat([analyzer._preprocess(im) for im in
                               images[i:i + BATCH]])
            probs.extend(torch.sigmoid(model(batch)).squeeze(1).tolist())
    return np.array(probs)


def metrics(probs, labels):
    preds = (probs >= 0.5).astype(int)
    labels = np.asarray(labels).astype(int)
    acc = float((preds == labels).mean())
    sens = float(preds[labels == 1].mean()) if (labels == 1).any() else float("nan")
    spec = float((preds[labels == 0] == 0).mean()) if (labels == 0).any() else float("nan")
    return {"n": len(labels), "accuracy@0.5": round(acc, 4),
            "sensitivity": round(sens, 4), "specificity": round(spec, 4),
            "auc": round(auc_rank(probs, labels), 4)}


def check_preprocessing_parity(analyzer):
    from torchvision import transforms
    model_card_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    rng = np.random.default_rng(42)
    img = Image.fromarray(rng.integers(0, 255, (300, 250), dtype=np.uint8),
                          mode="L")
    ours = analyzer._preprocess(img.convert("RGB"))
    theirs = model_card_tf(img).unsqueeze(0)
    diff = float((ours - theirs).abs().max())
    print(f"[parity] max |ours - model-card transform| = {diff:.6f}")
    assert diff < 1e-4, f"preprocessing mismatch: {diff}"
    return diff


def eval_pneumoniamnist(analyzer):
    data = np.load(NPZ_PATH)
    images_arr = data["test_images"]
    labels = data["test_labels"].reshape(-1)
    images = [Image.fromarray(arr, mode="L") for arr in images_arr]
    analyzer._ensure_model()
    probs = batch_probs(analyzer, images)
    m = metrics(probs, labels)
    print("[PneumoniaMNIST test, 624 genuine CC BY 4.0 images, 28x28 -> 224]",
          json.dumps(m))
    for name, want in (("NORMAL", 0), ("PNEUMONIA", 1)):
        idxs = np.where(labels == want)[0][:5]
        for i in idxs:
            print(f"    e.g. true={name:9s} p(pneumonia)={probs[i]:.3f}")
    return m


def _fetch_rows(offset, length=100):
    url = ("https://datasets-server.huggingface.co/rows?dataset="
           f"{urllib.request.quote(HF_DATASET)}&config=default&split=test"
           f"&offset={offset}&length={length}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)["rows"]


def fetch_fullres_samples(per_class=12):
    # Parquet is class-sorted: NORMAL rows come first, then PNEUMONIA.
    pages = [_fetch_rows(0), _fetch_rows(250)]
    picked = {0: [], 1: []}
    for page in pages:
        for row in page:
            r = row["row"]
            label = r["label"]
            if len(picked[label]) < per_class:
                picked[label].append(r["image"]["src"])
    images, labels = [], []
    for label in (0, 1):
        for src in picked[label]:
            with urllib.request.urlopen(src, timeout=60) as resp:
                img = Image.open(io.BytesIO(resp.read()))
                img.load()
            images.append(img.convert("RGB"))
            labels.append(label)
    return images, labels


def eval_fullres(analyzer, per_class=12):
    print(f"[full-res Kermany test via {HF_DATASET}, CC BY 4.0, "
          f"{per_class}/class]")
    images, labels = fetch_fullres_samples(per_class)
    analyzer._ensure_model()
    probs = batch_probs(analyzer, images)
    m = metrics(probs, labels)
    print("   ", json.dumps(m))
    for img_label, p in zip(labels, probs):
        print(f"    true={'NORMAL' if img_label == 0 else 'PNEUMONIA':9s} "
              f"p(pneumonia)={p:.3f}")
    return m


def main():
    analyzer = XrayAnalyzer(model_dir=MODEL_DIR)
    print("=== MediScan model validation (research sanity check only, "
          "NOT clinical validation) ===")
    check_preprocessing_parity(analyzer)
    mn = eval_pneumoniamnist(analyzer)
    fr = eval_fullres(analyzer)
    print("\nSummary")
    print(f"  PneumoniaMNIST (n={mn['n']}): AUC={mn['auc']} "
          f"acc={mn['accuracy@0.5']} sens={mn['sensitivity']} "
          f"spec={mn['specificity']}")
    print(f"  Full-res subset (n={fr['n']}): AUC={fr['auc']} "
          f"acc={fr['accuracy@0.5']} sens={fr['sensitivity']} "
          f"spec={fr['specificity']}")
    print("  NOTE: PneumoniaMNIST is 28x28 upscaled (domain mismatch vs "
          "training resolution); full-res set is a tiny convenience sample.")
    print("  These numbers are internal sanity checks on public data, NOT "
          "clinical validation.")


if __name__ == "__main__":
    main()
