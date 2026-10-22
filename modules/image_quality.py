"""Lightweight input-quality hints for uploaded images (warnings only).

Deliberately cheap (numpy only, runs on a 224px grayscale copy) and never
blocking: they flag suspicious inputs so users do not over-trust a score.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MIN_SIDE_HINT = 256
LOW_CONTRAST_STD = 18.0
BLUR_VARIANCE = 25.0


def analyze_quality(image) -> list[str]:
    """Return human-readable warnings for an uploaded PIL image."""
    warnings: list[str] = []
    try:
        import numpy as np

        width, height = image.size
        if min(width, height) < MIN_SIDE_HINT:
            warnings.append(
                f"Low resolution ({width}×{height}px); fine detail may be lost.")
        ratio = max(width, height) / max(1, min(width, height))
        if ratio > 2.2:
            warnings.append(
                f"Unusual aspect ratio ({width}×{height}); chest X-rays are "
                "usually closer to square.")
        gray = np.asarray(image.convert("L").resize((224, 224)),
                          dtype=np.float32)
        if float(gray.std()) < LOW_CONTRAST_STD:
            warnings.append(
                "Low contrast: the image looks flat or over-exposed.")
        laplacian = (gray[1:-1, 1:-1] * 4.0
                     - gray[:-2, 1:-1] - gray[2:, 1:-1]
                     - gray[1:-1, :-2] - gray[1:-1, 2:])
        if float(laplacian.var()) < BLUR_VARIANCE:
            warnings.append(
                "Possibly blurry or heavily downscaled: fine lung detail may "
                "be missing.")
    except Exception:  # pragma: no cover - defensive
        logger.debug("Quality heuristics failed", exc_info=True)
    return warnings
