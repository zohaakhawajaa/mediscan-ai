"""Central registry of AI models with honest, verified metadata.

Only facts that can be verified from the checkpoint, its model card, or a
local training run are recorded here. An entry is ``available`` only when
its checkpoint exists on disk; optional models stay listed as unavailable
instead of being silently faked or auto-downloaded.

Custom models trained with ``scripts/train_xray.py`` are discovered from
``models/custom/<key>/spec.json`` and appear only after a real run.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

TASK_XRAY = "chest-xray-pneumonia"
XRAY_LABELS = ("NORMAL", "PNEUMONIA")

DENSENET_KEY = "densenet121-pneumonia-hf"

BUILTIN_SPECS: list[dict] = [
    {
        "key": DENSENET_KEY,
        "name": "DenseNet121 Pneumonia (published weights)",
        "architecture": "DenseNet121 (torchvision), single-logit BCE head",
        "task": TASK_XRAY,
        "labels": list(XRAY_LABELS),
        "source": "https://huggingface.co/luanacarolina/pneumonia-chest-xray-classifier",
        "checkpoint": "xray/checkpoints/binary/densenet121.pt",
        "version": "binary-v1",
        "dataset": (
            "Kermany et al. (2018) pediatric chest X-ray dataset, Guangzhou "
            "Women and Children's Medical Center, ages 1-5"
        ),
        "license": "CC BY 4.0 (weights and model card)",
        "training_status": "pretrained (published weights, loaded as-is)",
        "reported_metrics": {
            "note": "Reported by the model card on the Kermany held-out test "
                    "split (mean over 3 seeds); NOT re-validated clinically by "
                    "this app.",
            "accuracy": 0.841,
            "sensitivity": 0.997,
            "f1": 0.887,
            "auc": 0.966,
        },
        "limitations": [
            "Validated only on pediatric AP chest X-rays from one institution.",
            "Modest specificity (~0.62 for the ensemble, per the model card): "
            "high recall comes with false alarms on normal lungs.",
            "Over-predicts PNEUMONIA on low-resolution/upscaled inputs "
            "(measured specificity 0.0 on 28x28 upscaled data).",
            "Not validated for adults or clinical use.",
        ],
        "preprocessing": "224x224 RGB, ImageNet mean/std, sigmoid on the logit",
        "adapter": "densenet_hf",
        "gradcam_layer": None,
        "notes": "Primary model used by the X-ray module.",
        "download_note": (
            "Auto-downloads (~84 MB) from Hugging Face on first use if the "
            "checkpoint is missing."
        ),
    },
    {
        "key": "resnet18-pneumoniamnist-local",
        "name": "ResNet18 (PneumoniaMNIST fine-tune)",
        "architecture": "ResNet18 (torchvision), 1-logit BCE head",
        "task": TASK_XRAY,
        "labels": list(XRAY_LABELS),
        "source": "trained locally by scripts/train_xray.py",
        "checkpoint": "custom/resnet18-pneumoniamnist-local/best.pt",
        "version": "local-unnamed",
        "dataset": "PneumoniaMNIST (MedMNIST v2, CC BY 4.0), 28x28 upscaled to 224",
        "license": "Torchvision weights: BSD-3-Clause; dataset: CC BY 4.0",
        "training_status": "not trained yet - run scripts/train_xray.py",
        "reported_metrics": None,
        "limitations": [
            "Checkpoint does not exist until you train it locally.",
            "PneumoniaMNIST is 28x28 upscaled to 224: strong distribution shift.",
            "Metrics from a local run are research-only, never clinical.",
        ],
        "preprocessing": "224x224 RGB, ImageNet mean/std, sigmoid on the logit",
        "adapter": "torchvision_ckpt",
        "torchvision_arch": "resnet18",
        "gradcam_layer": "layer4",
        "notes": "Optional second model; enables side-by-side comparison once trained.",
        "download_note": "No automatic download. Requires a local training run.",
    },
    {
        "key": "efficientnet_b0-pneumoniamnist-local",
        "name": "EfficientNet-B0 (PneumoniaMNIST fine-tune)",
        "architecture": "EfficientNet-B0 (torchvision), 1-logit BCE head",
        "task": TASK_XRAY,
        "labels": list(XRAY_LABELS),
        "source": "trained locally by scripts/train_xray.py",
        "checkpoint": "custom/efficientnet_b0-pneumoniamnist-local/best.pt",
        "version": "local-unnamed",
        "dataset": "PneumoniaMNIST (MedMNIST v2, CC BY 4.0), 28x28 upscaled to 224",
        "license": "Torchvision weights: BSD-3-Clause; dataset: CC BY 4.0",
        "training_status": "not trained yet - run scripts/train_xray.py",
        "reported_metrics": None,
        "limitations": [
            "Checkpoint does not exist until you train it locally.",
            "PneumoniaMNIST is 28x28 upscaled to 224: strong distribution shift.",
            "Metrics from a local run are research-only, never clinical.",
        ],
        "preprocessing": "224x224 RGB, ImageNet mean/std, sigmoid on the logit",
        "adapter": "torchvision_ckpt",
        "torchvision_arch": "efficientnet_b0",
        "gradcam_layer": "features.8",
        "notes": "Optional second model; enables side-by-side comparison once trained.",
        "download_note": "No automatic download. Requires a local training run.",
    },
]

REQUIRED_KEYS = {
    "key", "name", "architecture", "task", "labels", "source", "checkpoint",
    "version", "dataset", "license", "training_status", "limitations",
    "preprocessing", "adapter", "gradcam_layer",
}


def checkpoint_path(models_dir: str, spec: dict) -> str:
    return os.path.join(models_dir, *spec["checkpoint"].split("/"))


def is_available(models_dir: str, spec: dict) -> bool:
    return os.path.isfile(checkpoint_path(models_dir, spec))


def _load_custom_specs(models_dir: str) -> list[dict]:
    custom_root = os.path.join(models_dir, "custom")
    specs: list[dict] = []
    if not os.path.isdir(custom_root):
        return specs
    for name in sorted(os.listdir(custom_root)):
        spec_path = os.path.join(custom_root, name, "spec.json")
        if not os.path.isfile(spec_path):
            continue
        try:
            with open(spec_path, encoding="utf-8") as handle:
                spec = json.load(handle)
            missing = REQUIRED_KEYS - set(spec)
            if missing:
                logger.warning("Custom model '%s' missing keys: %s",
                               name, ", ".join(sorted(missing)))
                continue
            if not spec.get("reported_metrics"):
                spec["reported_metrics"] = None
            spec.setdefault("notes", "")
            spec.setdefault("download_note", "Trained locally in this repository.")
            spec["checkpoint"] = f"custom/{name}/best.pt"
            specs.append(spec)
        except (OSError, ValueError):
            logger.warning("Could not read custom model spec at %s", spec_path,
                           exc_info=True)
    return specs


def list_models(models_dir: str, task: str | None = None) -> list[dict]:
    """All registered models with an ``available`` flag (default task: all)."""
    specs = [dict(spec) for spec in BUILTIN_SPECS]
    specs.extend(_load_custom_specs(models_dir))
    if task:
        specs = [spec for spec in specs if spec["task"] == task]
    for spec in specs:
        spec["available"] = is_available(models_dir, spec)
        spec["checkpoint_path"] = checkpoint_path(models_dir, spec)
    return specs


def get_model(models_dir: str, key: str) -> dict | None:
    for spec in list_models(models_dir):
        if spec["key"] == key:
            return spec
    return None


def available_models(models_dir: str, task: str = TASK_XRAY) -> list[dict]:
    return [spec for spec in list_models(models_dir, task=task)
            if spec["available"]]
