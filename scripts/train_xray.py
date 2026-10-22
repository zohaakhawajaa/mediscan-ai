"""Train a chest X-ray model locally on PneumoniaMNIST.

This is the only way a new model becomes available in the Model Center.
No pretrained ImageNet weights are downloaded: the backbone is created with
``weights=None`` so training works fully offline. Metrics written to
``spec.json`` come from the real held-out test split of that run.

Example (small, CPU-friendly):
    .venv\\Scripts\\python.exe scripts\\train_xray.py --model resnet18 --epochs 3
Smoke test (wiring only, ~1 minute on CPU):
    .venv\\Scripts\\python.exe scripts\\train_xray.py --model resnet18 --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision  # noqa: E402

from config import Config  # noqa: E402
from modules.evaluation import compute_metrics  # noqa: E402
from modules.model_adapters import _adapt_head  # noqa: E402

ARCH_CHOICES = ("resnet18", "efficientnet_b0")
GRADCAM_LAYERS = {"resnet18": "layer4", "efficientnet_b0": "features.8"}
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(npz_path: str, split: str):
    with np.load(npz_path) as data:
        images = data[f"{split}_images"]
        labels = data[f"{split}_labels"].reshape(-1).astype(np.float32)
    return images, labels


def to_batch(images: np.ndarray, indices) -> torch.Tensor:
    array = images[indices].astype(np.float32)
    if array.ndim == 3:
        array = array[..., None]
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2)
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear",
                           align_corners=False)
    return (tensor / 255.0 - MEAN) / STD


def predict(model, images: np.ndarray) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), 64):
            batch = to_batch(images, slice(start, start + 64))
            outputs.append(torch.sigmoid(model(batch)).reshape(-1))
    return torch.cat(outputs).numpy()


def build_model(arch: str, freeze: bool):
    model = torchvision.models.get_model(arch, weights=None)
    _adapt_head(model, arch)
    if freeze:
        for name, param in model.named_parameters():
            if not name.startswith(("fc.", "classifier.")):
                param.requires_grad = False
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=ARCH_CHOICES, default="resnet18")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny end-to-end run that verifies training "
                             "wiring without meaningful accuracy.")
    parser.add_argument("--npz", default=Config.DATASET_NPZ)
    parser.add_argument("--out", default=os.path.join(Config.MODELS_ROOT,
                                                      "custom"))
    args = parser.parse_args()

    set_seed(args.seed)
    if not os.path.isfile(args.npz):
        print(f"Dataset not found: {args.npz}")
        return 2

    train_images, train_labels = load_split(args.npz, "train")
    val_images, val_labels = load_split(args.npz, "val")
    test_images, test_labels = load_split(args.npz, "test")
    if args.smoke:
        train_images, train_labels = train_images[:48], train_labels[:48]
        val_images, val_labels = val_images[:24], val_labels[:24]
        test_images, test_labels = test_images[:24], test_labels[:24]
        args.epochs = 1
    print(f"Train {len(train_images)} / Val {len(val_images)} / "
          f"Test {len(test_images)} images")

    model = build_model(args.model, args.freeze_backbone)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    key = f"{args.model}-pneumoniamnist-local"
    run_dir = os.path.join(args.out, key)
    os.makedirs(run_dir, exist_ok=True)

    history = []
    best_auc = -1.0
    best_epoch = 0
    patience, stale = 3, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_images))
        total_loss, batches = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            batch_idx = order[start:start + args.batch_size]
            inputs = to_batch(train_images, batch_idx)
            targets = torch.from_numpy(train_labels[batch_idx]).view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        train_loss = total_loss / max(1, batches)
        val_probs = predict(model, val_images)
        val_metrics = compute_metrics(val_labels.astype(int).tolist(),
                                      val_probs.tolist())
        val_auc = val_metrics["auc"] or 0.0
        record = {"epoch": epoch, "train_loss": round(train_loss, 4),
                  "val_accuracy": val_metrics["accuracy"],
                  "val_auc": val_metrics["auc"]}
        history.append(record)
        print(f"epoch {epoch}: loss {train_loss:.4f} "
              f"val_acc {val_metrics['accuracy']:.3f} "
              f"val_auc {val_metrics['auc']}")
        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            torch.save({"model_state_dict": model.state_dict()},
                       os.path.join(run_dir, "best.pt"))
        else:
            stale += 1
            if stale >= patience and not args.smoke:
                print("Early stopping: validation AUC stopped improving.")
                break

    state = torch.load(os.path.join(run_dir, "best.pt"),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    test_probs = predict(model, test_images)
    test_metrics = compute_metrics(test_labels.astype(int).tolist(),
                                   test_probs.tolist())
    metrics_public = {key_: test_metrics[key_] for key_ in
                      ("accuracy", "accuracy_ci95", "sensitivity",
                       "specificity", "precision", "f1", "auc",
                       "confusion", "support", "n")}

    np.save(os.path.join(run_dir, "test_probabilities.npy"), test_probs)
    with open(os.path.join(run_dir, "history.json"), "w",
              encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    spec = {
        "key": key,
        "name": f"{args.model} (PneumoniaMNIST, local run "
                f"{datetime.now():%Y-%m-%d})",
        "architecture": f"{args.model} (torchvision), 1-logit BCE head",
        "task": "chest-xray-pneumonia",
        "labels": ["NORMAL", "PNEUMONIA"],
        "source": "trained locally by scripts/train_xray.py",
        "checkpoint": f"custom/{key}/best.pt",
        "version": datetime.now().strftime("local-%Y%m%d-%H%M"),
        "dataset": "PneumoniaMNIST (MedMNIST v2), 28x28 upscaled to 224",
        "license": "Torchvision backbone: BSD-3-Clause; dataset: CC BY 4.0",
        "training_status": "trained locally (measured metrics only)",
        "reported_metrics": {
            "note": "Measured by this training run on the PneumoniaMNIST "
                    f"{'reduced ' if args.smoke else ''}test split.",
            **metrics_public,
        },
        "limitations": [
            "PneumoniaMNIST is 28x28 upscaled to 224: strong distribution "
            "shift versus real radiographs.",
            "Single small dataset; not validated clinically.",
            "Research use only - never a diagnosis.",
        ],
        "preprocessing": "224x224 RGB, ImageNet mean/std, sigmoid on the logit",
        "adapter": "torchvision_ckpt",
        "torchvision_arch": args.model,
        "gradcam_layer": GRADCAM_LAYERS[args.model],
        "notes": f"Epochs run: {len(history)} (best epoch {best_epoch}), "
                 f"batch {args.batch_size}, lr {args.lr}, seed {args.seed}."
                 + (" SMOKE RUN: metrics are not meaningful." if args.smoke
                    else ""),
        "download_note": "Trained locally; no automatic download.",
    }
    with open(os.path.join(run_dir, "spec.json"), "w",
              encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)

    print("\nTest metrics:")
    for name, value in metrics_public.items():
        print(f"  {name}: {value}")
    print(f"\nSaved model + spec to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
