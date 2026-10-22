"""Model adapters: one ``analyze_bytes``/``analyze_image`` interface per model.

The DenseNet Hugging Face checkpoint reuses :class:`XrayAnalyzer`. Locally
trained checkpoints (``scripts/train_xray.py``) are loaded through the
generic torchvision adapter. Nothing here fabricates predictions: if a
checkpoint is missing, inference fails with a clear message.
"""

from __future__ import annotations

import io
import logging
import os
import threading

logger = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """Raised when a registered model cannot run (missing deps/checkpoint)."""


def _adapt_head(model, arch: str) -> None:
    """Swap the classifier for a 1-logit pneumonia head (training parity)."""
    if arch.startswith("densenet"):
        model.classifier = _nn().Linear(model.classifier.in_features, 1)
    elif arch.startswith("resnet"):
        model.fc = _nn().Linear(model.fc.in_features, 1)
    elif arch.startswith("efficientnet"):
        model.classifier[1] = _nn().Linear(model.classifier[1].in_features, 1)
    else:  # pragma: no cover - registry only lists known arches
        raise ValueError(f"Unsupported architecture for local models: {arch}")


def _nn():
    import torch
    return torch.nn


class TorchvisionCheckpointAdapter:
    """Generic adapter for locally trained torchvision classification models."""

    def __init__(self, spec: dict, checkpoint_path: str):
        self.spec = spec
        self._checkpoint_path = checkpoint_path
        self._model = None
        self._lock = threading.Lock()
        self._last_error: str | None = None

    @property
    def checkpoint_path(self) -> str:
        return self._checkpoint_path

    def status(self) -> dict:
        from .xray_analyzer import ml_available
        if not ml_available():
            return {"available": False,
                    "detail": "ML dependencies are unavailable "
                              "(details in the server log)."}
        if not os.path.isfile(self.checkpoint_path):
            return {"available": False,
                    "detail": "Checkpoint not found. Train this model locally "
                              "with scripts/train_xray.py first."}
        if self._model is not None:
            return {"available": True, "detail": "Model loaded."}
        return {"available": True,
                "detail": "Checkpoint found; will load on first use."}

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not os.path.isfile(self.checkpoint_path):
                raise ModelUnavailable(
                    f"No checkpoint for '{self.spec['key']}' yet. Run "
                    "scripts/train_xray.py to train it locally.")
            import torch
            import torchvision
            arch = self.spec.get("torchvision_arch")
            if not arch:
                raise ModelUnavailable(
                    f"Model '{self.spec['key']}' does not declare a "
                    "torchvision_arch.")
            model = torchvision.models.get_model(arch, weights=None)
            _adapt_head(model, arch)
            state = torch.load(self.checkpoint_path, map_location="cpu",
                               weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            model.eval()
            self._model = model
            self._last_error = None
            logger.info("Local model loaded: %s", self.checkpoint_path)

    def analyze_bytes(self, data: bytes, gradcam: bool = False) -> dict:
        from .xray_analyzer import MIN_INPUT_SIDE, Image
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception as exc:
            logger.warning("Image decode failed (%s): %s",
                           type(exc).__name__, exc)
            return {"ok": False,
                    "error": ("Could not open this file as an image. Please "
                              "upload a valid image file (PNG, JPEG, BMP, "
                              "WebP or TIFF)."),
                    "label": None, "pneumonia_probability": None}
        if min(image.size) < MIN_INPUT_SIDE:
            return {"ok": False,
                    "error": (f"Image resolution too low "
                              f"({image.width}x{image.height}). At least "
                              f"{MIN_INPUT_SIDE}px on each side is required: "
                              "small or heavily upscaled images produce "
                              "unreliable scores in this model."),
                    "label": None, "pneumonia_probability": None}
        return self.analyze_image(image, gradcam=gradcam)

    def analyze_image(self, image, gradcam: bool = False) -> dict:
        from .xray_analyzer import (DISCLAIMER, compute_gradcam,
                                    heatmap_overlay_b64, preprocess_image,
                                    quality_warnings)
        import torch
        heatmap_b64 = None
        try:
            self._ensure_model()
            tensor = preprocess_image(image)
            if gradcam:
                cam = None
                try:
                    target = self._model.get_submodule(self.spec["gradcam_layer"])
                    cam = compute_gradcam(self._model, target, tensor)
                except Exception:
                    logger.exception("Grad-CAM pass failed; continuing "
                                     "without heatmap")
                with torch.no_grad():
                    p_pneumonia = float(
                        torch.sigmoid(self._model(tensor)).item())
                if cam is not None:
                    try:
                        heatmap_b64 = heatmap_overlay_b64(image, cam)
                    except Exception:
                        logger.exception("Heatmap overlay failed")
            else:
                with torch.no_grad():
                    logit = self._model(tensor)
                    p_pneumonia = float(torch.sigmoid(logit).item())
        except ModelUnavailable as exc:
            return {"ok": False, "error": str(exc),
                    "label": None, "pneumonia_probability": None}
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("Local model inference failed")
            return {"ok": False,
                    "error": ("Analysis could not be completed due to an "
                              "internal error. Please try again, or choose "
                              "another model."),
                    "label": None, "pneumonia_probability": None}
        label = "PNEUMONIA" if p_pneumonia >= 0.5 else "NORMAL"
        result = {
            "ok": True,
            "label": label,
            "pneumonia_probability": p_pneumonia,
            "confidence": max(p_pneumonia, 1.0 - p_pneumonia),
            "model": self.spec["name"],
            "model_key": self.spec["key"],
            "disclaimer": DISCLAIMER,
            "warnings": quality_warnings(image),
        }
        if heatmap_b64 is not None:
            result["heatmap_b64"] = heatmap_b64
        return result


def build_adapter(spec: dict, models_dir: str):
    """Instantiate the adapter for a registry spec (no model loading yet)."""
    from .model_registry import checkpoint_path
    if spec["adapter"] == "densenet_hf":
        from .xray_analyzer import (DEFAULT_REPO_ID, XrayAnalyzer)
        root, _, relpath = spec["checkpoint"].partition("/")
        return XrayAnalyzer(
            model_dir=os.path.join(models_dir, root),
            repo_id=spec.get("repo_id", DEFAULT_REPO_ID),
            checkpoint_relpath=relpath or "checkpoints/binary/densenet121.pt",
        )
    if spec["adapter"] == "torchvision_ckpt":
        return TorchvisionCheckpointAdapter(
            spec, checkpoint_path(models_dir, spec))
    raise ValueError(f"Unknown adapter type: {spec['adapter']}")
