"""Evaluate a registered model on the real PneumoniaMNIST test split.

Uses actual inference only. Writes an optional JSON report and always prints
a summary; nothing is fabricated or cached across runs.

Examples:
    .venv\\Scripts\\python.exe scripts\\evaluate_model.py --model densenet121-pneumonia-hf --limit 100
    .venv\\Scripts\\python.exe scripts\\evaluate_model.py --split test --output eval_results\\densenet.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config  # noqa: E402
from modules import model_registry  # noqa: E402
from modules.evaluation import evaluate_npz  # noqa: E402
from modules.model_adapters import build_adapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=model_registry.DENSENET_KEY)
    parser.add_argument("--split", default="test",
                        choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=0,
                        help="0 evaluates the full split (slow on CPU).")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="",
                        help="Optional path for the JSON report.")
    args = parser.parse_args()

    spec = model_registry.get_model(Config.MODELS_ROOT, args.model)
    if spec is None:
        print(f"Unknown model key: {args.model}")
        print("Available: " + ", ".join(
            item["key"] for item in model_registry.list_models(
                Config.MODELS_ROOT)))
        return 2
    adapter = build_adapter(spec, Config.MODELS_ROOT)
    status = adapter.status()
    if not status["available"]:
        print(f"Model unavailable: {status['detail']}")
        return 2
    limit = args.limit if args.limit > 0 else 10 ** 9

    print(f"Evaluating {spec['name']} on {args.split} split "
          f"(limit {args.limit or 'full'}) ...")
    report = evaluate_npz(adapter, spec, Config.DATASET_NPZ, split=args.split,
                          limit=limit, threshold=args.threshold)
    metrics = report["metrics"]
    print(f"Samples: {report['evaluated_samples']} "
          f"(of {report['available_samples']})")
    for key in ("accuracy", "accuracy_ci95", "sensitivity", "specificity",
                "precision", "f1", "auc"):
        print(f"  {key}: {metrics[key]}")
    print(f"  confusion: {metrics['confusion']}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)),
                    exist_ok=True)
        report = dict(report)
        report["generated_at"] = datetime.now().isoformat(timespec="seconds")
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
