"""Offline evaluation on real labelled data (PneumoniaMNIST).

Everything here is measured from an actual inference run over a real split;
nothing is estimated or copied. Reports are stored as small JSON files with
a TTL so quick evaluations do not accumulate.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import uuid

logger = logging.getLogger(__name__)

EVAL_TTL_SECONDS = 24 * 60 * 60
MAX_QUICK_EVAL = 200
PNEUMONIA_LABEL = 1
_RID_RE = re.compile(r"^[0-9a-f]{32}$")


def load_pneumoniamnist(npz_path: str, split: str = "test"):
    """Load a PneumoniaMNIST split -> (images uint8 array, labels ints)."""
    import numpy as np
    if not os.path.isfile(npz_path):
        raise ValueError(f"Dataset file not found: {npz_path}")
    data = np.load(npz_path)
    images_key = f"{split}_images"
    labels_key = f"{split}_labels"
    if images_key not in data or labels_key not in data:
        raise ValueError(f"Split '{split}' not present in {npz_path}")
    images = data[images_key]
    labels = data[labels_key].reshape(-1).astype(int)
    if len(images) != len(labels):
        raise ValueError("Image/label count mismatch in dataset.")
    return images, labels


def wilson_interval(successes: int, total: int, z: float = 1.96):
    """95% Wilson score interval for a proportion (None if total == 0)."""
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [round(max(0.0, centre - half), 4),
            round(min(1.0, centre + half), 4)]


def _roc_points(labels, probs, max_points: int = 240):
    thresholds = sorted(set(probs), reverse=True)
    points = [[0.0, 0.0]]
    pos = sum(1 for label in labels if label == PNEUMONIA_LABEL)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return points + [[1.0, 1.0]]
    for threshold in thresholds:
        tp = sum(1 for label, prob in zip(labels, probs)
                 if label == PNEUMONIA_LABEL and prob >= threshold)
        fp = sum(1 for label, prob in zip(labels, probs)
                 if label != PNEUMONIA_LABEL and prob >= threshold)
        points.append([round(fp / neg, 5), round(tp / pos, 5)])
    points.append([1.0, 1.0])
    if len(points) > max_points:
        step = max(1, len(points) // max_points)
        decimated = points[::step]
        if decimated[-1] != [1.0, 1.0]:
            decimated.append([1.0, 1.0])
        points = decimated
    return points


def _auc(points) -> float:
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return round(area, 4)


def _calibration(labels, probs, bins: int = 10):
    rows = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [(label, prob) for label, prob in zip(labels, probs)
                   if low <= prob < high or (index == bins - 1 and prob == 1.0)]
        if not members:
            continue
        mean_prob = sum(prob for _, prob in members) / len(members)
        positives = sum(1 for label, _ in members
                        if label == PNEUMONIA_LABEL) / len(members)
        rows.append({
            "low": round(low, 2),
            "high": round(high, 2),
            "count": len(members),
            "mean_probability": round(mean_prob, 4),
            "fraction_positive": round(positives, 4),
        })
    return rows


def compute_metrics(labels, probs, threshold: float = 0.5) -> dict:
    """Confusion matrix + classification metrics + ROC/calibration."""
    if len(labels) != len(probs) or not labels:
        raise ValueError("Need equally sized, non-empty labels and scores.")
    tp = sum(1 for label, prob in zip(labels, probs)
             if label == PNEUMONIA_LABEL and prob >= threshold)
    tn = sum(1 for label, prob in zip(labels, probs)
             if label != PNEUMONIA_LABEL and prob < threshold)
    fp = sum(1 for label, prob in zip(labels, probs)
             if label != PNEUMONIA_LABEL and prob >= threshold)
    fn = sum(1 for label, prob in zip(labels, probs)
             if label == PNEUMONIA_LABEL and prob < threshold)
    total = tp + tn + fp + fn
    metrics = {
        "n": total,
        "threshold": threshold,
        "support": {"positive": tp + fn, "negative": tn + fp},
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": round((tp + tn) / total, 4),
        "accuracy_ci95": wilson_interval(tp + tn, total),
        "sensitivity": round(tp / (tp + fn), 4) if tp + fn else None,
        "specificity": round(tn / (tn + fp), 4) if tn + fp else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "f1": (round(2 * tp / (2 * tp + fp + fn), 4)
               if (2 * tp + fp + fn) else None),
    }
    roc = _roc_points(labels, probs)
    metrics["auc"] = _auc(roc) if len(set(labels)) > 1 else None
    metrics["roc_points"] = roc
    metrics["calibration"] = _calibration(labels, probs)
    return metrics


def evaluate_npz(adapter, spec: dict, npz_path: str, split: str = "test",
                 limit: int = MAX_QUICK_EVAL, threshold: float = 0.5) -> dict:
    """Run real inference over up to ``limit`` evenly spaced split samples."""
    import numpy as np
    from PIL import Image

    images, labels = load_pneumoniamnist(npz_path, split)
    count = min(limit, len(images))
    indices = np.linspace(0, len(images) - 1, count).round().astype(int)
    probs: list[float] = []
    used_labels: list[int] = []
    for index in indices:
        array = images[index]
        if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
            image = Image.fromarray(array.squeeze(), mode="L" if
                                    array.shape[-1] == 1 else "RGB")
        else:
            image = Image.fromarray(array)
        result = adapter.analyze_image(image)
        if not result.get("ok"):
            raise RuntimeError(
                f"Inference failed at sample {int(index)}: "
                f"{result.get('error', 'unknown error')}")
        probs.append(float(result["pneumonia_probability"]))
        used_labels.append(int(labels[index]))
    metrics = compute_metrics(used_labels, probs, threshold=threshold)
    return {
        "model_key": spec["key"],
        "model_name": spec["name"],
        "model_version": spec.get("version"),
        "dataset": "PneumoniaMNIST (MedMNIST v2)",
        "split": split,
        "available_samples": len(images),
        "evaluated_samples": count,
        "threshold": threshold,
        "metrics": metrics,
    }


# ------------------------------------------------------------------- storage
def _report_dir(config) -> str:
    return config["EVAL_STORE_FOLDER"]


def save_report(config, report: dict) -> str:
    os.makedirs(_report_dir(config), exist_ok=True)
    report = dict(report)
    report_id = uuid.uuid4().hex
    report["id"] = report_id
    report["created_at"] = time.time()
    path = os.path.join(_report_dir(config), f"{report_id}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False)
    return report_id


def load_report(config, report_id: str, ttl: int = EVAL_TTL_SECONDS):
    if not report_id or not _RID_RE.match(report_id):
        return None
    path = os.path.join(_report_dir(config), f"{report_id}.json")
    try:
        if not os.path.isfile(path):
            return None
        if ttl and time.time() - os.path.getmtime(path) > ttl:
            return None
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        logger.warning("Could not read evaluation report %s", report_id,
                       exc_info=True)
        return None


def list_reports(config, limit: int = 20, ttl: int = EVAL_TTL_SECONDS):
    directory = _report_dir(config)
    if not os.path.isdir(directory):
        return []
    reports = []
    now = time.time()
    for name in os.listdir(directory):
        if not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        try:
            if ttl and now - os.path.getmtime(path) > ttl:
                continue
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
            report["created_at"] = report.get("created_at") or \
                os.path.getmtime(path)
            reports.append(report)
        except (OSError, ValueError):
            continue
    reports.sort(key=lambda item: item.get("created_at", 0), reverse=True)
    return reports[:limit]


def sweep_reports(config, ttl: int = EVAL_TTL_SECONDS) -> int:
    directory = _report_dir(config)
    if not os.path.isdir(directory):
        return 0
    removed = 0
    now = time.time()
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        try:
            if (not name.endswith(".json")
                    or now - os.path.getmtime(path) > ttl):
                os.remove(path)
                removed += 1
        except OSError:
            continue
    return removed
