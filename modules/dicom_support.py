"""Optional single-frame DICOM import (requires the ``pydicom`` package).

Only converted pixels and a few non-identifying tags are kept; identifying
DICOM metadata (names, IDs, dates) is never stored or shown.
"""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

_SAFE_TAGS = ("Modality", "Rows", "Columns", "PhotometricInterpretation",
              "BitsStored")


def available() -> bool:
    try:
        import pydicom  # noqa: F401
        return True
    except Exception:
        return False


def convert_to_png(data: bytes) -> tuple[bytes, dict]:
    """Convert DICOM bytes to an 8-bit PNG; returns (png_bytes, safe_metadata)."""
    import numpy as np
    import pydicom
    from PIL import Image

    try:
        dataset = pydicom.dcmread(io.BytesIO(data), force=True)
        array = dataset.pixel_array
    except Exception as exc:
        logger.warning("DICOM decode failed (%s): %s", type(exc).__name__, exc)
        raise ValueError(
            "This DICOM file could not be decoded. Compressed transfer "
            "syntaxes may need extra codec packages.") from exc

    if array.ndim == 3:
        # Multi-frame or multi-sample: keep the first frame / colour planes.
        samples = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
        array = array[..., :3] if (samples > 1 and array.shape[-1] >= 3) \
            else array[0]
    if array.ndim not in (2, 3) or array.size == 0:
        raise ValueError("Unsupported DICOM pixel layout.")

    values = array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1) or 1)
    intercept = float(getattr(dataset, "RescaleIntercept", 0) or 0)
    values = values * slope + intercept
    if str(getattr(dataset, "PhotometricInterpretation", "")) == \
            "MONOCHROME1":
        values = values.max() - values
    low, high = float(values.min()), float(values.max())
    if high <= low:
        values = np.zeros_like(values, dtype=np.uint8)
    else:
        values = (values - low) / (high - low) * 255.0
    values = np.clip(values, 0, 255).astype(np.uint8)

    image = Image.fromarray(values) if values.ndim == 2 else \
        Image.fromarray(values[..., :3])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    meta = {}
    for tag in _SAFE_TAGS:
        value = getattr(dataset, tag, None)
        if value is not None:
            meta[tag] = str(value)
    logger.info("DICOM converted: %s", meta)
    return buffer.getvalue(), meta
