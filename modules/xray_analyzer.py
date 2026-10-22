"""Chest X-ray pneumonia screening module (EXPERIMENTAL — not a medical device).

Model provenance
----------------
Repository: https://huggingface.co/luanacarolina/pneumonia-chest-xray-classifier
Checkpoint: checkpoints/binary/densenet121.pt (DenseNet121, binary BCE head)
License:    CC BY 4.0 (weights and model card)
Training data: Kermany et al. (2018) pediatric chest X-ray dataset
               (Guangzhou Women and Children's Medical Center, ages 1-5).

Preprocessing per model card: resize to 224x224, convert to 3 channels,
normalize with ImageNet mean/std, sigmoid on the single logit.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import torch
    import torchvision
    from huggingface_hub import hf_hub_download
    from PIL import Image, UnidentifiedImageError

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    logger.exception("ML dependency import failed")
    _IMPORT_ERROR = exc
    torch = None
    torchvision = None
    np = None
    Image = None

DEFAULT_REPO_ID = "luanacarolina/pneumonia-chest-xray-classifier"
DEFAULT_CHECKPOINT = "checkpoints/binary/densenet121.pt"
IMG_SIZE = 224
MIN_INPUT_SIDE = 128  # below this, scores saturate and become unreliable
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DISCLAIMER = (
    "For research use only. Not for diagnostic use. Trained on pediatric "
    "chest X-rays from a single institution; consult a qualified clinician."
)


class XrayAnalyzer:
    """Lazy-loading, thread-safe wrapper around the DenseNet121 checkpoint."""

    def __init__(self, model_dir: str, repo_id: str = DEFAULT_REPO_ID,
                 checkpoint_relpath: str = DEFAULT_CHECKPOINT):
        self.model_dir = model_dir
        self.repo_id = repo_id
        self.checkpoint_relpath = checkpoint_relpath
        self._model = None
        self._lock = threading.Lock()
        self._last_error: str | None = None

    # ------------------------------------------------------------------ status
    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.model_dir, *self.checkpoint_relpath.split("/"))

    def status(self) -> dict:
        if _IMPORT_ERROR is not None:
            # Raw exception text can embed local paths/versions; keep it in
            # the server log only.
            return {"available": False,
                    "detail": "ML dependencies are unavailable "
                              "(details in the server log)."}
        if self._model is not None:
            return {"available": True, "detail": "Model loaded."}
        if os.path.isfile(self.checkpoint_path):
            return {"available": True,
                    "detail": "Checkpoint downloaded; will load on first use."}
        return {"available": True,
                "detail": "Checkpoint will be downloaded on first use "
                          f"({self.repo_id})."}

    # ------------------------------------------------------------------- load
    def _ensure_model(self):
        if self._model is not None:
            return
        if _IMPORT_ERROR is not None:
            raise RuntimeError(f"ML dependencies unavailable: {_IMPORT_ERROR}")
        with self._lock:
            if self._model is not None:
                return
            os.makedirs(self.model_dir, exist_ok=True)
            if not os.path.isfile(self.checkpoint_path):
                logger.info("Downloading X-ray checkpoint from %s ...", self.repo_id)
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.checkpoint_relpath,
                    local_dir=self.model_dir,
                )
            state = torch.load(self.checkpoint_path, map_location="cpu",
                               weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model = torchvision.models.densenet121(weights=None)
            model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
            model.load_state_dict(state)
            model.eval()
            self._model = model
            self._last_error = None
            logger.info("X-ray model loaded from %s", self.checkpoint_path)

    # -------------------------------------------------------------- inference
    def _preprocess(self, image: "Image.Image"):
        # Bilinear matches the model card's torchvision transforms.Resize.
        img = image.convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return ((tensor - mean) / std).unsqueeze(0)

    def analyze_bytes(self, data: bytes, gradcam: bool = False) -> dict:
        """Run pneumonia screening on raw image bytes.

        Returns a dict with ok=True, label, pneumonia_probability and
        confidence on success; ok=False plus an error message otherwise.
        With gradcam=True, a successful result also carries heatmap_b64
        (a base64 PNG Grad-CAM attention overlay). Never fabricates
        predictions.
        """
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception as exc:
            # Details stay in the server log; the browser only gets a
            # generic message (PIL's text embeds internal object reprs).
            logger.warning("Image decode failed (%s): %s",
                           type(exc).__name__, exc)
            return {"ok": False,
                    "error": ("Could not open this file as an image. Please "
                              "upload a valid image file (PNG, JPEG, BMP, "
                              "WebP or TIFF)."),
                    "label": None, "pneumonia_probability": None}
        if min(image.size) < MIN_INPUT_SIDE:
            return {"ok": False,
                    "error": (f"Image resolution too low ({image.width}x{image.height}). "
                              f"At least {MIN_INPUT_SIDE}px on each side is required: "
                              "small or heavily upscaled images produce saturated, "
                              "unreliable pneumonia scores in this model."),
                    "label": None, "pneumonia_probability": None}
        return self.analyze_image(image, gradcam=gradcam)

    def analyze_image(self, image: "Image.Image", gradcam: bool = False) -> dict:
        heatmap_b64 = None
        try:
            self._ensure_model()
            tensor = self._preprocess(image)
            if gradcam:
                p_pneumonia, cam = self._gradcam_forward(tensor)
                if cam is not None:
                    try:
                        heatmap_b64 = self._heatmap_overlay(image, cam)
                    except Exception:
                        logger.exception("Grad-CAM overlay failed; "
                                         "continuing without heatmap")
            else:
                with torch.no_grad():
                    logit = self._model(tensor)
                    p_pneumonia = float(torch.sigmoid(logit).item())
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("X-ray inference failed")
            return {"ok": False,
                    "error": ("Analysis could not be completed due to an "
                              "internal error. Please try again, or use a "
                              "different image."),
                    "label": None, "pneumonia_probability": None}
        label = "PNEUMONIA" if p_pneumonia >= 0.5 else "NORMAL"
        confidence = max(p_pneumonia, 1.0 - p_pneumonia)
        result = {
            "ok": True,
            "label": label,
            "pneumonia_probability": p_pneumonia,
            "confidence": confidence,
            "model": "DenseNet121 binary pneumonia classifier",
            "disclaimer": DISCLAIMER,
        }
        if heatmap_b64 is not None:
            result["heatmap_b64"] = heatmap_b64
        return result

    # ----------------------------------------------------------------- gradcam
    def _gradcam_forward(self, tensor):
        """Forward pass computing Grad-CAM for the last conv block.

        Returns (probability, normalized CAM tensor or None). Any problem
        degrades to a plain forward pass instead of failing the analysis.
        """
        model = self._model
        try:
            feat_holder: dict = {}
            block = model.features.denseblock4
            h_fwd = block.register_forward_hook(
                lambda m, i, out: feat_holder.__setitem__("v", out))
            try:
                logits = model(tensor)
                p_pneumonia = float(torch.sigmoid(logits).item())
                feat = feat_holder.get("v")
                if feat is None or not feat.requires_grad:
                    return p_pneumonia, None
                # Grad w.r.t. the block output; avoids backward hooks, which
                # clash with DenseNet's inplace ReLU.
                (grad,) = torch.autograd.grad(logits.sum(), feat)
            finally:
                h_fwd.remove()
            feat = feat.detach()
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * feat).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(
                cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                align_corners=False)[0, 0]
            span = float(cam.max() - cam.min())
            if span < 1e-8:
                return p_pneumonia, None
            cam = (cam - cam.min()) / span
            return p_pneumonia, cam
        except Exception:
            logger.exception("Grad-CAM pass failed; "
                             "falling back to plain inference")
            with torch.no_grad():
                p_pneumonia = float(torch.sigmoid(model(tensor)).item())
            return p_pneumonia, None

    def _heatmap_overlay(self, image: "Image.Image", cam) -> str:
        """Blend a normalized CAM onto the image; returns PNG base64."""
        img = image.convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        base = np.asarray(img, dtype=np.float32) / 255.0
        if hasattr(cam, "detach"):
            cam = cam.detach().cpu().numpy()
        cam_np = np.asarray(cam, dtype=np.float32)
        heat = np.stack([cam_np,
                         0.25 * (1.0 - cam_np),
                         np.zeros_like(cam_np)], axis=-1)
        overlay = np.clip(0.55 * base + 0.45 * heat, 0.0, 1.0)
        buf = io.BytesIO()
        Image.fromarray((overlay * 255).astype(np.uint8)).save(
            buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")


def validate_image_upload(filename: str, allowed: set[str]) -> str | None:
    """Return an error message if the filename is not an allowed image type."""
    if not filename or "." not in filename:
        return "No image file provided."
    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in allowed:
        return (f"Unsupported image format '.{ext}'. "
                f"Allowed: {', '.join(sorted(allowed))}.")
    return None
