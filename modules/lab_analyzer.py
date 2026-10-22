"""Blood-test interpretation module.

Compares user-entered laboratory values against configurable reference
ranges and flags out-of-range values. This module NEVER diagnoses disease;
it only reports whether a value falls outside the selected range.
"""

from __future__ import annotations

import math

REFERENCE_RANGES: dict = {
    "hemoglobin": {
        "label": "Hemoglobin",
        "unit": "g/dL",
        "ranges": {
            "adult_male": {"low": 13.5, "high": 17.5},
            "adult_female": {"low": 12.0, "high": 15.5},
            "child_6_12y": {"low": 11.5, "high": 14.5},
        },
        "default_range": "adult_male",
    },
    "wbc": {
        "label": "White blood cell count (WBC)",
        "unit": "10*3/uL",
        "ranges": {
            "adult": {"low": 4.5, "high": 11.0},
            "child_6_12y": {"low": 5.0, "high": 13.0},
        },
        "default_range": "adult",
    },
    "platelets": {
        "label": "Platelets",
        "unit": "10*3/uL",
        "ranges": {
            "adult": {"low": 150.0, "high": 450.0},
        },
        "default_range": "adult",
    },
    "glucose": {
        "label": "Glucose",
        "unit": "mg/dL",
        "ranges": {
            "fasting_adult": {"low": 70.0, "high": 99.0},
            "random_adult": {"low": 70.0, "high": 140.0},
        },
        "default_range": "fasting_adult",
    },
    "creatinine": {
        "label": "Creatinine",
        "unit": "mg/dL",
        "ranges": {
            "adult_male": {"low": 0.74, "high": 1.35},
            "adult_female": {"low": 0.59, "high": 1.04},
        },
        "default_range": "adult_male",
    },
}

# Built-in example cases for demonstration purposes. No real patient data.
SAMPLE_CASES: dict = {
    "synthetic_normal": {
        "title": "Case A — all values within range",
        "entries": [
            {"test": "hemoglobin", "value": 14.2, "range": "adult_male"},
            {"test": "wbc", "value": 7.1, "range": "adult"},
            {"test": "platelets", "value": 240.0, "range": "adult"},
            {"test": "glucose", "value": 88.0, "range": "fasting_adult"},
        ],
    },
    "synthetic_anemia": {
        "title": "Case B — low hemoglobin pattern",
        "entries": [
            {"test": "hemoglobin", "value": 10.8, "range": "adult_female"},
            {"test": "wbc", "value": 6.4, "range": "adult"},
            {"test": "platelets", "value": 180.0, "range": "adult"},
            {"test": "glucose", "value": 92.0, "range": "fasting_adult"},
        ],
    },
    "synthetic_inflammation": {
        "title": "Case C — elevated WBC / platelets pattern",
        "entries": [
            {"test": "hemoglobin", "value": 13.8, "range": "adult_male"},
            {"test": "wbc", "value": 14.6, "range": "adult"},
            {"test": "platelets", "value": 495.0, "range": "adult"},
            {"test": "glucose", "value": 105.0, "range": "fasting_adult"},
        ],
    },
}


def range_options_for(test_key: str) -> list[str]:
    spec = REFERENCE_RANGES.get(test_key)
    if not spec:
        return []
    return list(spec["ranges"].keys())


def analyze_entries(entries: list[dict]) -> dict:
    """Interpret a list of {test, value, range} entries.

    Returns {"ok", "results", "errors"}.
    """
    results = []
    errors = []
    for entry in entries:
        test_key = str(entry.get("test", "")).strip()
        raw_value = entry.get("value")
        range_key = str(entry.get("range", "")).strip()

        spec = REFERENCE_RANGES.get(test_key)
        if spec is None:
            errors.append(f"Unknown test '{test_key}'.")
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"{spec['label']}: invalid value '{raw_value}'.")
            continue
        if not math.isfinite(value):
            errors.append(f"{spec['label']}: invalid value '{raw_value}' "
                          f"(must be a finite number).")
            continue
        bounds = spec["ranges"].get(range_key)
        if bounds is None:
            errors.append(
                f"{spec['label']}: unknown reference range '{range_key}'. "
                f"Available: {', '.join(spec['ranges'])}.")
            continue

        if value < bounds["low"]:
            flag = "low"
        elif value > bounds["high"]:
            flag = "high"
        else:
            flag = "normal"
        results.append({
            "test": test_key,
            "label": spec["label"],
            "value": value,
            "unit": spec["unit"],
            "range": range_key,
            "range_low": bounds["low"],
            "range_high": bounds["high"],
            "flag": flag,
        })
    return {"ok": not errors or bool(results),
            "results": results, "errors": errors}
