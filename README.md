# MediScan AI

Flask-based medical **research prototype** for:

1. **Chest X-ray pneumonia screening** (pretrained DenseNet121, genuine inference, optional Grad-CAM attention heatmap)
2. **Blood-test interpretation** (reference-range flagging, no diagnosis)
3. **PDF medical-report extraction** (PyMuPDF, with OCR detection and user review)

> **Not a medical device. Nothing here is a medical diagnosis.** All outputs are
> experimental research signals. Always consult a qualified clinician.

## Setup

Python 3.12 is used (torch-compatible). From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

The chest X-ray checkpoint (~84 MB) downloads automatically on first use into
`models/xray/` (git-ignored) from Hugging Face, or pre-download it:

```powershell
.\.venv\Scripts\python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='luanacarolina/pneumonia-chest-xray-classifier', filename='checkpoints/binary/densenet121.pt', local_dir='models/xray')"
```

## Run

```powershell
.\.venv\Scripts\python serve.py          # http://127.0.0.1:5001
# or: $env:MEDISCAN_PORT=5002; .\.venv\Scripts\python serve.py
```

Note: port 5000 on this machine is occupied by another Flask app (House Price
Predictor), so MediScan defaults to **5001**. Both `serve.py` and
`python app.py` honour the same variable.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `MEDISCAN_PORT` | `5001` | Port used by `serve.py` and by `python app.py` |
| `MEDISCAN_DEBUG` | off | `1` enables the Werkzeug debugger (loopback only — never on a shared host) |
| `MEDISCAN_ENV` | `development` | Set to `production` to enforce a real session secret |
| `MEDISCAN_SECRET_KEY` | dev placeholder | Session signing key; **must be set and non-default when `MEDISCAN_ENV=production`** |

Production starts are refused with a clear error if the key is missing or is
still the development placeholder (`config.validate_secret_key`). Development
needs no setup. Generate a key for a one-off production start — it is read
from the environment only and must never be committed:

```powershell
$b = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b)
$env:MEDISCAN_SECRET_KEY = [Convert]::ToBase64String($b)
$env:MEDISCAN_ENV = "production"
.\.venv\Scripts\python serve.py
```

## Validation audit (2026-10-18)

An audit was performed against **genuine, openly-licensed chest X-rays** —
not synthetic noise. `scripts/validate_model.py` reproduces it.

**Sources (both CC BY 4.0, same Kermany pediatric origin as training data):**
- PneumoniaMNIST test split (MedMNIST, Zenodo DOI 10.5281/zenodo.10519652) —
  624 images, 28×28, stored in `validation_data/pneumoniamnist.npz`.
- Full-resolution Kermany test images via `hf-vision/chest-xray-pneumonia`
  on Hugging Face (12 NORMAL + 12 PNEUMONIA convenience sample).

**Results:**

| Evaluation | n | AUC | Accuracy@0.5 | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Full-res Kermany sample | 24 | 1.000 | 1.000 | 1.000 | 1.000 |
| PneumoniaMNIST (28×28→224) | 624 | 0.711 | 0.625 | 1.000 | 0.000 |

- **Class mapping confirmed**: on genuine full-res images, p(sigmoid) is the
  pneumonia probability — NORMAL samples score ≈0.000, PNEUMONIA ≈1.000.
- **Preprocessing confirmed**: the pipeline now matches the model card's
  torchvision transform bit-exactly (max diff 0.000000).
- **Two defects found and fixed by the audit:**
  1. Resize used PIL's default bicubic interpolation instead of the model
     card's bilinear — corrected in `xray_analyzer._preprocess`.
  2. Low-resolution inputs saturate the model toward PNEUMONIA (specificity
     0.0 on upscaled 28×28 data) — inputs below 128 px/side are now rejected
     with a clear error rather than scored.
- **UI hardening**: "Confidence" renamed to "Model certainty (uncalibrated)";
  disclaimers now state explicitly that scores are not clinically reliable
  and that out-of-distribution inputs over-predict PNEUMONIA.

This is an **internal sanity check on a few dozen genuine images — NOT
clinical validation**. The full-res sample is tiny and drawn from the same
institution/distribution as the training data.

Regression tests: `tests/test_validation_regression.py` (preprocessing parity,
low-resolution rejection, genuine-sample directional AUC on PneumoniaMNIST)
plus UI wording and end-to-end rejection tests in `tests/test_app.py`.

## Tests

```powershell
.\.venv\Scripts\python -m pytest tests -q
```

55 tests cover lab interpretation, PDF extraction, app routes, upload
validation, UI disclaimer wording, preprocessing parity, low-resolution
rejection, genuine-X-ray directional mapping (skipped when validation data is
absent), a real-checkpoint integration test (skipped if the checkpoint is
absent), Grad-CAM heatmap generation, plus regression tests for upload
cleanup, non-finite lab values, non-standard filenames, candidate selection,
server-side report storage (no report text in cookies), redacted error
messages, session reset, the About page, and the production secret-key
gate.

## Model provenance (chest X-ray)

| Item | Value |
|---|---|
| Repository | https://huggingface.co/luanacarolina/pneumonia-chest-xray-classifier |
| Checkpoint | `checkpoints/binary/densenet121.pt` (84 MB, verified present) |
| Architecture | torchvision DenseNet121, single-logit BCE head, sigmoid output |
| Input | 224×224, grayscale→3 channels, ImageNet mean/std normalization |
| Output | NORMAL vs PNEUMONIA (binary), plus P(pneumonia) |
| Training data | Kermany et al. (2018) pediatric chest X-rays (Guangzhou, ages 1–5) |
| Reported test metrics | Accuracy 0.841, AUROC 0.966, recall 0.997, specificity ~0.62 |
| License | CC BY 4.0 |
| Intended use | Research/education only — explicitly **not for clinical use** |

Verified in this environment: the checkpoint downloads, loads with
`torchvision.models.densenet121`, and produces a real sigmoid probability on an
actual inference request (synthetic noise image → saturated P(pneumonia)≈1.0,
confirming a live forward pass, not a hardcoded result).

## Dataset attribution (MedMNIST / Kermany)

- [MedMNIST](https://github.com/MedMNIST/MedMNIST) (Apache-2.0 code, CC BY 4.0
  data) provides **PneumoniaMNIST**, the appropriate lightweight binary
  chest X-ray dataset variant (28×28, pediatric, from the same Kermany source).
  MedMNIST is a *dataset* collection — it ships no pretrained pneumonia
  classifier, so no MedMNIST training was performed; the pretrained HF model
  above was used instead.
- If PneumoniaMNIST data is ever used for training or evaluation, cite:
  Yang et al., *MedMNIST v2*, Nature Scientific Data (2023),
  DOI 10.1038/s41597-022-01721-8, and Kermany et al., Cell (2018),
  DOI 10.1016/j.cell.2018.02.010.

## Architecture

```
MediScanAI/
├── app.py                  # Flask app factory
├── serve.py                # dev server launcher (port 5001)
├── config.py               # upload limits, model dir, HF repo id
├── modules/
│   ├── xray_analyzer.py    # lazy-load DenseNet121, download-once, inference
│   ├── lab_analyzer.py     # configurable reference ranges + sample cases
│   ├── report_extractor.py # PyMuPDF text extraction, OCR detection, candidates
│   └── routes.py           # blueprints: main, xray, labs, report, summary
├── templates/              # Jinja2 pages (dashboard, xray, labs, report, summary, about)
├── static/style.css
├── tests/                  # pytest suite (55 tests)
├── scripts/validate_model.py  # reproducible validation-audit script
├── models/xray/            # downloaded checkpoint (git-ignored)
├── validation_data/        # PneumoniaMNIST npz + audit samples (git-ignored)
├── uploads/                # session-scoped uploads (git-ignored, auto-cleaned)
└── .report_store/          # server-side extracted PDF text, 6h TTL (git-ignored)
```

## Design decisions

- **Model loads once** (lazy, thread-safe) and the checkpoint is downloaded at
  most once into `models/xray/`; uploads never trigger re-downloads.
- **Graceful degradation**: if torch or the checkpoint is unavailable, the app
  still runs; the X-ray page shows a clear "unavailable" message and other
  modules keep working. Failed inference returns an error — labels and
  confidence scores are **never fabricated**.
- **Confidence** is `max(p, 1−p)` from the sigmoid output and may be
  miscalibrated; the UI presents it under a standard research-use-only line.
- **Blood-test flags** only compare against the selected reference range
  (age/sex variants where applicable). No disease names are ever emitted.
- **PDF extraction** flags image-only (scanned) pages as requiring OCR;
  extracted text must be reviewed by the user before any lab candidates are
  interpreted. Missing values are never invented.
- **Privacy**: all processing is local. Uploads are stored under `uploads/`
  with random names, served only back to the owning session, and deleted on
  replacement; stale uploads are swept after 24 h, and requests over 10 MB
  are refused (HTTP 413). Extracted PDF text is kept server-side in
  `.report_store/` (git-ignored) for 6 h — the session cookie carries only a
  random store id and compact result fields, never the report text. Error
  messages shown in the browser are generic; file/parse details go to the
  server log only. No external services receive user files.
- **Grad-CAM heatmap** is computed only when the user leaves the
  "Attention heatmap" option checked; it is generated per request, never
  stored, and heatmap failures degrade to a normal prediction instead of
  failing the analysis.
- **Session tools**: the summary page offers Print/PDF output and a
  one-click "Start new analysis" reset that clears results, deletes the
  session's upload, and removes its stored report.

## Limitations and known issues

- The X-ray model was trained on **pediatric** AP chest X-rays from a single
  institution; results on adult or non-standard images are unvalidated.
- Specificity is modest (~0.62 on the reported test set); false alarms on
  normal X-rays are common even in-distribution.
- **Out-of-distribution inputs over-predict PNEUMONIA with saturated scores**
  (measured: specificity 0.0 on 28×28→224 upscaled data). Inputs below
  128 px/side are rejected, but other OOD inputs (non-chest-X-ray photos,
  heavily corrupted images) can still trigger confident-looking wrong answers.
- Model certainty values are not calibrated (ECE ≈ 0.13 on the reported test
  set) and are never presented as clinically reliable confidence.
- The validation audit's full-res evaluation used only 24 convenience-sample
  images from the same source distribution as training — this is a sanity
  check, not clinical validation.
- OCR is not implemented; scanned PDFs are detected and reported only.
- Reference ranges are common textbook values, not lab-specific; users must
  confirm the correct range.
- The dev server (Flask built-in) is for local prototyping only — use a WSGI
  server (e.g. gunicorn/waitress) for anything beyond that.
- The PDF candidate regexes are conservative English patterns and will miss
  unusual report formats.

## Licenses

- Application code: provided as-is for research/education.
- Model weights: CC BY 4.0 (luanacarolina/pneumonia-chest-xray-classifier).
- MedMNIST code Apache-2.0 / data CC BY 4.0 (attribution above).
- PyMuPDF: AGPL-3.0 (note: AGPL applies to distribution of the software).
