# MediScan AI

Flask-based medical **research prototype**: chest X-ray screening with multiple
registered models, blood-test reference-range flagging, and PDF report
extraction with human review — wrapped in a multi-page research workspace.

> **Not a medical device. Nothing here is a medical diagnosis.** All outputs are
> experimental research signals. Always consult a qualified clinician.
> For research use only.

## Modules

| Area | What it does |
|---|---|
| Dashboard | Entry point, module cards, recent activity (metadata only) |
| X-ray | Model selector, genuine DenseNet121 inference, optional Grad-CAM overlay with opacity control, image-quality warnings, DICOM (`.dcm`/`.dicom`) import with PHI-free metadata |
| Model Center | Registry of available/unavailable models with per-model cards, provenance, limitations |
| Compare | Run the same image through two models side by side |
| Evaluation | Quick evaluation on PneumoniaMNIST (capped at 200 images), ROC curve, confusion matrix, calibration bins, downloadable JSON report |
| Labs | Configurable reference ranges (age/sex variants), flag badges — never disease names |
| Report | PyMuPDF text extraction, OCR-needed detection, user-reviewed lab candidates |
| Summary | Session aggregation, print/PDF view, one-click reset |
| Status | Package versions, adapter readiness, dataset inventory |
| History | Acknowledged, metadata-only activity log (no patient text, no PHI) |

## Honesty principles

- **No fabricated results.** If a checkpoint is missing, the model is listed as
  unavailable and inference returns an error. No labels or scores are invented.
- **Reported metrics are quoted from source.** The app shows what the model card
  reports; anything measured locally is labeled as a local sanity check.
- **No silent downloads.** The DenseNet121 checkpoint is fetched once (or
  pre-downloaded); optional local models use `weights=None` backbones and must
  be trained from data already present on disk.
- **No PHI in logs or history.** DICOM metadata is reduced to a safe whitelist;
  history stores counters and labels only, never file text.

## Setup

Python 3.12 (torch-compatible). From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

The chest X-ray checkpoint (~84 MB) is downloaded once into `models/xray/`
(git-ignored) on first use, or pre-download it:

```powershell
.\.venv\Scripts\python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='luanacarolina/pneumonia-chest-xray-classifier', filename='checkpoints/binary/densenet121.pt', local_dir='models/xray')"
```

## Run

```powershell
.\.venv\Scripts\python serve.py          # http://127.0.0.1:5001
# or: $env:MEDISCAN_PORT=5002; .\.venv\Scripts\python serve.py
```

Port 5000 on this machine is used by another Flask app, so MediScan defaults to
**5001**. Both `serve.py` and `python app.py` honour `MEDISCAN_PORT`.

Production-style start (waitress, no debugger):

```powershell
$env:MEDISCAN_ENV = "production"
$env:MEDISCAN_SECRET_KEY = "<32+ random bytes, base64>"   # generate, never commit
.\.venv\Scripts\waitress-serve.exe --listen=127.0.0.1:5001 --threads=4 serve:app
```

Generate a key for a one-off run:

```powershell
$b = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b)
$env:MEDISCAN_SECRET_KEY = [Convert]::ToBase64String($b)
```

Production starts are refused if the key is missing or still the development
placeholder (`config.validate_secret_key`).

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `MEDISCAN_PORT` | `5001` | Port used by `serve.py` and by `python app.py` |
| `MEDISCAN_DEBUG` | off | `1` enables the Werkzeug debugger (loopback only — never on a shared host) |
| `MEDISCAN_ENV` | `development` | `production` enforces a real session secret |
| `MEDISCAN_SECRET_KEY` | dev placeholder | Session signing key; required in production |

CSRF protection is always on in normal runs and only bypassed by the test
configuration; it cannot be disabled by an environment variable.

Fixed server-side defaults: uploads ≤ 10 MB, upload sweep 24 h, report text
store 6 h, evaluation reports 24 h, history 6 h / 20 entries — all under
git-ignored directories.

## Models

| Key | Status | Backbone | Head | Checkpoint |
|---|---|---|---|---|
| `densenet121-pneumonia-hf` | **available** | torchvision DenseNet121 | 1-logit BCE → sigmoid | `models/xray/checkpoints/binary/densenet121.pt` (84 MB) |
| `resnet18-pneumoniamnist-local` | created by local training | torchvision ResNet18 | 1-logit BCE → sigmoid | `models/custom/resnet18-pneumoniamnist-local/best.pt` |
| `efficientnet_b0-pneumoniamnist-local` | created by local training | torchvision EfficientNet-B0 | 1-logit BCE → sigmoid | `models/custom/efficientnet_b0-pneumoniamnist-local/best.pt` |

Custom models are discovered from `models/custom/*/spec.json` (requires
`key`, `name`, `task`, `torchvision_arch`, and a checkpoint). Nothing is
downloaded for them; an untrained entry stays *unavailable* until you run the
training script.

### Provenance — `densenet121-pneumonia-hf`

| Item | Value |
|---|---|
| Repository | https://huggingface.co/luanacarolina/pneumonia-chest-xray-classifier |
| Architecture | torchvision DenseNet121, single-logit BCE head, sigmoid output |
| Input | 224×224, grayscale→3 channels, ImageNet mean/std normalization |
| Output | NORMAL vs PNEUMONIA (binary), plus P(pneumonia) |
| Training data | Kermany et al. (2018) pediatric AP chest X-rays (Guangzhou, ages 1–5) |
| Reported test metrics (model card, mean of 3 seeds) | Accuracy 84.1%, recall 99.7%, F1 88.7%, AUROC 96.6%; modest specificity (~0.62 for the ensemble) |
| License | CC BY 4.0 |
| Intended use | Research/education only — explicitly **not for clinical use** |

Verified locally: the checkpoint loads into `torchvision.models.densenet121`
with `weights=None`, and a forward pass on a genuine image produces a real
sigmoid probability (the integration test is skipped if the checkpoint is
absent).

## Local validation audit

`scripts/validate_model.py` reproduces a small sanity audit from genuine,
openly-licensed images — not synthetic noise.

**Sources:** PneumoniaMNIST test split (MedMNIST, Zenodo DOI
10.5281/zenodo.10519652; 624 images, 28×28, stored in
`validation_data/pneumoniamnist.npz`) and a 24-image full-resolution Kermany
sample from `hf-vision/chest-xray-pneumonia` (CC BY 4.0).

| Evaluation | n | AUC | Accuracy@0.5 | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Full-res Kermany sample | 24 | 1.000 | 1.000 | 1.000 | 1.000 |
| PneumoniaMNIST (28×28→224) | 624 | 0.711 | 0.625 | 1.000 | 0.000 |

- **Class mapping confirmed:** on genuine full-res images, the sigmoid output
  is the pneumonia probability (NORMAL ≈ 0.000, PNEUMONIA ≈ 1.000).
- **Preprocessing matches the model card** (bilinear resize, grayscale→3
  channels, ImageNet normalization).
- **Two defects found and fixed:** PIL's default bicubic resize was replaced by
  bilinear, and inputs below 128 px/side are now rejected instead of being
  scored — upscaled 28×28 data saturates toward PNEUMONIA (specificity 0.0
  above).
- **UI wording:** "Confidence" is presented as "Model certainty
  (uncalibrated)", with explicit out-of-distribution warnings.

This is an **internal sanity check on a few dozen genuine images — NOT
clinical validation**. The full-res sample is tiny and shares the training
distribution.

## Evaluation

- **In-app quick evaluation** (`/eval`): runs a registered model on up to 200
  PneumoniaMNIST test images (split selectable), computes accuracy, sensitivity,
  specificity, F1, AUC, Wilson intervals, calibration bins and an inline ROC
  curve, and stores a JSON report for 24 h.
- **Full CLI evaluation** for the complete split:

```powershell
.\.venv\Scripts\python scripts\evaluate_model.py --model densenet121-pneumonia-hf --split test --limit 0 --output eval_densenet_test.json
```

Reports are written to `.eval_store/` (git-ignored) when run without
`--output`; only measured values are included, never card numbers.

## Training optional local models

The script fine-tunes a torchvision backbone **from `weights=None`** (no
network, no hidden downloads) on the local PneumoniaMNIST npz. A smoke run is
cheap; a full run is a real training job (CPU is slow — expect hours).

```powershell
# smoke check (~seconds, tiny subset, writes a real but weak checkpoint)
.\.venv\Scripts\python scripts\train_xray.py --model resnet18 --smoke

# real run
.\.venv\Scripts\python scripts\train_xray.py --model efficientnet_b0 --epochs 8 --batch-size 64 --lr 1e-3
```

Outputs (per `--out`, default under `models/custom/<key>/`): `best.pt`
(state dict), `spec.json` (measured test metrics only, `notes` record the
setup), `history.json`, `test_probabilities.npy`. The model then appears as
available in the Model Center automatically. Training data is currently
28×28 upscaled to 224 — a small, low-resolution proxy, not a replacement for
the full-resolution pretrained model.

## Security & privacy

- **CSRF tokens** on every POST form (session-bound; enforced outside tests).
- **Security headers**: `X-Content-Type-Options: nosniff`, `X-Frame-Options:
  DENY`, `Referrer-Policy: same-origin`, and a CSP that allows only self-hosted
  scripts plus data-URI images (`default-src 'self'; img-src 'self' data:;
  style-src 'self' 'unsafe-inline'; script-src 'self'; base-uri 'self';
  form-action 'self'; frame-ancestors 'none'`).
- **Uploads** get random names, are served only to the owning session, and are
  swept after 24 h; requests over 10 MB are refused.
- **PDF text** lives server-side in `.report_store/` (6 h TTL); the cookie
  carries only a random store id and compact result fields.
- **DICOM** files are converted to PNG with a metadata whitelist (modality,
  rows/columns, photometric interpretation, bits stored) — no patient fields.
- **History** stores primitive metadata only (kind, model key, label, counters,
  timestamp); unsafe ids/kinds are rejected.
- **Errors** shown in the browser are generic; file/parse details go to the
  server log only. No external service receives user files.

## Tests

```powershell
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python -m pytest tests -q
```

**75 tests** cover: lab interpretation, PDF extraction, all page routes, upload
validation, UI disclaimer wording, preprocessing parity, low-resolution
rejection, genuine-X-ray directional mapping, real-checkpoint integration
(optional), Grad-CAM heatmap generation, upload cleanup, non-finite lab values,
non-standard filenames, candidate selection, server-side report storage,
session reset, the About page, the production secret-key gate, the model
registry (builtin + custom discovery), adapter unavailability, metric math
(AUC/confusion), evaluation report store round-trips and expiry, metadata-only
history, DICOM conversion, new pages/404s, security headers and CSRF
enforcement, and a subprocess training smoke run that verifies the produced
artifacts.

## Architecture

```
MediScanAI/
├── app.py                    # app factory, security headers, CSRF, sweeps
├── serve.py                  # dev/WSGI launcher (port 5001)
├── config.py                 # paths, limits, TTLs, model dir
├── modules/
│   ├── model_registry.py     # builtin + custom specs, availability
│   ├── model_adapters.py     # torchvision checkpoint adapter (lazy load)
│   ├── xray_analyzer.py      # preprocess, inference, Grad-CAM, quality checks
│   ├── evaluation.py         # PneumoniaMNIST eval, metrics, report store
│   ├── charts.py             # dependency-free ROC SVG
│   ├── history.py            # metadata-only activity log
│   ├── dicom_support.py      # optional pydicom conversion (PHI-free)
│   ├── image_quality.py      # blur/contrast/resolution hints
│   ├── lab_analyzer.py       # reference ranges + sample cases
│   ├── report_extractor.py   # PyMuPDF extraction, OCR detection, candidates
│   └── routes.py             # all blueprints
├── scripts/
│   ├── train_xray.py         # offline fine-tuning (PneumoniaMNIST)
│   ├── evaluate_model.py     # full-split evaluation CLI
│   └── validate_model.py     # reproducible audit script
├── templates/                # base + 13 pages
├── static/style.css, app.js  # premium dark/light UI, no inline scripts
├── tests/                    # pytest suite (75 tests)
├── models/xray/              # pretrained checkpoint (git-ignored)
├── models/custom/            # locally trained models (git-ignored)
├── validation_data/          # PneumoniaMNIST npz + samples (git-ignored)
├── uploads/                  # session uploads (git-ignored, swept)
├── .report_store/            # extracted PDF text, 6 h TTL (git-ignored)
├── .eval_store/              # evaluation reports, 24 h TTL (git-ignored)
└── .history_store/           # activity log, 6 h TTL (git-ignored)
```

## Design decisions

- **Lazy, thread-safe loading** per model; the DenseNet checkpoint downloads at
  most once, and uploads never trigger downloads.
- **Graceful degradation**: without torch or a checkpoint the app still runs;
  the model shows as unavailable and other modules keep working.
- **Grad-CAM** is opt-in per request, uses the model's own target layer
  (`denseblock4` for DenseNet, `layer4` / `features.8` for the local backbones),
  and is never stored. Failures degrade to a plain prediction.
- **Quality warnings** (small side, low contrast, blur) are advisory, not
  blocking; only <128 px/side is rejected.
- **Blood-test flags** compare against the selected reference range only — no
  disease names are ever emitted.
- **Session tools**: print/PDF output and a reset that clears results, deletes
  the session upload, and removes its stored report.

## Limitations

- Single-institution, **pediatric** AP chest X-rays; adults and other scanners
  are unvalidated.
- Modest specificity (~0.62 for the ensemble per the card): false alarms on
  normal X-rays are expected.
- **Out-of-distribution inputs over-predict PNEUMONIA** (measured specificity
  0.0 on 28×28→224 upscaled data). Inputs below 128 px/side are rejected, but
  other OOD inputs (non-chest photos, heavy corruption) can still look
  confidently wrong.
- Model certainty is uncalibrated (card: ECE ≈ 0.13–0.15 for single binary
  models) and is never presented as clinically reliable confidence.
- Local models trained through `scripts/train_xray.py` learn from 28×28
  upscaled data — useful for pipeline testing, not for meaningful performance.
- The local full-res audit uses 24 convenience images from the training
  distribution — a sanity check, not clinical validation.
- OCR is not implemented; scanned PDFs are detected and reported only.
- Reference ranges are textbook values, not lab-specific.
- The Flask dev server is for local prototyping; use waitress/gunicorn behind
  HTTPS for anything shared.

## Licenses

- Application code: provided as-is for research/education.
- DenseNet weights: CC BY 4.0
  (luanacarolina/pneumonia-chest-xray-classifier).
- Kermany dataset / MedMNIST: CC BY 4.0 data (MedMNIST code Apache-2.0).
  Cite Kermany et al., Cell (2018), DOI 10.1016/j.cell.2018.02.010 and
  Yang et al., MedMNIST v2, Sci. Data (2023), DOI 10.1038/s41597-022-01721-8.
- PyMuPDF: AGPL-3.0 (applies to distribution of the software).
