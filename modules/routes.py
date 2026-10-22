"""Flask route blueprints for MediScan AI."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime

from flask import (Blueprint, abort, current_app, flash, redirect,
                   render_template, request, send_file, session,
                   send_from_directory, url_for)

from . import charts, dicom_support, evaluation, history as history_store
from . import model_registry
from .lab_analyzer import (REFERENCE_RANGES, SAMPLE_CASES, analyze_entries,
                           range_options_for)
from .model_adapters import build_adapter
from .report_extractor import extract_from_pdf, find_lab_candidates
from .xray_analyzer import validate_image_upload

main_bp = Blueprint("main", __name__)
xray_bp = Blueprint("xray", __name__)
lab_bp = Blueprint("labs", __name__)
report_bp = Blueprint("report", __name__)
summary_bp = Blueprint("summary", __name__)
models_bp = Blueprint("models", __name__)
eval_bp = Blueprint("eval", __name__)


# --------------------------------------------------------------------- helpers
def _analyzer():
    return current_app.extensions["xray_analyzer"]


def _models_root() -> str:
    return current_app.config["MODELS_ROOT"]


def _adapter_for(spec: dict):
    """Adapter for a registry spec; the default model stays swappable."""
    if spec["key"] == model_registry.DENSENET_KEY:
        return _analyzer()
    cache = current_app.extensions.setdefault("model_adapters", {})
    adapter = cache.get(spec["key"])
    if adapter is None:
        adapter = build_adapter(spec, _models_root())
        cache[spec["key"]] = adapter
    return adapter


def _activity_id() -> str:
    activity_id = session.get("_activity_id")
    if not activity_id:
        activity_id = uuid.uuid4().hex
        session["_activity_id"] = activity_id
    return activity_id


def _record(kind: str, label: str, extra: dict | None = None) -> None:
    history_store.record(current_app.config, _activity_id(), kind, label, extra)


def _remove_upload(filename: str | None):
    if not filename:
        return
    path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    try:
        os.remove(path)
    except OSError:
        pass


def _cleanup_previous_upload(session_key: str):
    _remove_upload(session.pop(session_key, None))


def sweep_stale_uploads(folder: str, max_age_seconds: float) -> int:
    """Delete upload files older than max_age_seconds.

    Returns the number of files removed. A non-positive max_age_seconds
    disables sweeping (returns 0 without touching anything).
    """
    if max_age_seconds <= 0 or not os.path.isdir(folder):
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            pass
    return removed


def _read_image_upload(file) -> tuple[bytes, str, dict]:
    """Validate an upload, converting DICOM to PNG. Raises ValueError."""
    config = current_app.config
    allowed = (set(config["ALLOWED_IMAGE_EXTENSIONS"])
               | set(config.get("ALLOWED_DICOM_EXTENSIONS", set())))
    error = validate_image_upload(file.filename, allowed)
    if error:
        raise ValueError(error)
    ext = file.filename.rsplit(".", 1)[-1].lower()
    data = file.read()
    meta: dict = {}
    if ext in set(config.get("ALLOWED_DICOM_EXTENSIONS", set())):
        if not dicom_support.available():
            raise ValueError(
                "DICOM support is not installed on this server "
                "(pip install pydicom).")
        data, meta = dicom_support.convert_to_png(data)
        ext = "png"
    return data, ext, meta


def _save_upload(data: bytes, ext: str, prefix: str = "xray") -> str:
    filename = f"{prefix}_{uuid.uuid4().hex}.{ext}"
    os.makedirs(current_app.config["UPLOAD_FOLDER"], exist_ok=True)
    with open(os.path.join(current_app.config["UPLOAD_FOLDER"], filename),
              "wb") as handle:
        handle.write(data)
    return filename


def _system_status() -> dict:
    from importlib import metadata
    packages = ("flask", "torch", "torchvision", "numpy", "pillow",
                "pymupdf", "huggingface-hub", "pydicom", "waitress")
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except Exception:
            versions[package] = None
    models = []
    for spec in model_registry.list_models(_models_root()):
        adapter = _adapter_for(spec)
        models.append({"spec": spec, "status": adapter.status()})
    return {
        "versions": versions,
        "models": models,
        "dataset": _dataset_info(),
        "dicom": dicom_support.available(),
        "environment": current_app.config.get("ENV_NAME", "development"),
        "config": current_app.config,
    }


def _dataset_info() -> dict:
    path = current_app.config.get("DATASET_NPZ")
    if not path or not os.path.isfile(path):
        return {"available": False, "counts": {}}
    try:
        import numpy as np
        with np.load(path) as data:
            counts = {split: int(len(data[f"{split}_images"]))
                      for split in ("train", "val", "test")
                      if f"{split}_images" in data}
        return {"available": True, "counts": counts,
                "name": "pneumoniamnist.npz",
                "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2)}
    except Exception:
        current_app.logger.warning("Could not inspect dataset", exc_info=True)
        return {"available": False, "counts": {}}


# --------------------------------------------- server-side report storage
def _report_path(store_id: str) -> str:
    return os.path.join(current_app.config["REPORT_STORE_FOLDER"],
                        f"report_{store_id}.json")


def _delete_report(store_id: str | None):
    if not store_id:
        return
    try:
        os.remove(_report_path(store_id))
    except OSError:
        pass


def _sweep_report_store():
    folder = current_app.config["REPORT_STORE_FOLDER"]
    ttl = current_app.config.get("REPORT_TTL_SECONDS", 0)
    if ttl <= 0 or not os.path.isdir(folder):
        return
    cutoff = time.time() - ttl
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


def _store_report(payload: dict) -> str:
    folder = current_app.config["REPORT_STORE_FOLDER"]
    os.makedirs(folder, exist_ok=True)
    store_id = uuid.uuid4().hex
    with open(_report_path(store_id), "w", encoding="utf-8") as fh:
        json.dump({**payload, "saved_at": time.time()}, fh)
    _sweep_report_store()
    return store_id


def _load_report() -> dict | None:
    """Return the extracted report for this session, or None.

    Older sessions kept the report text in the (signed, unencrypted) cookie;
    such payloads are migrated to the server store and dropped from the cookie.
    """
    legacy = session.pop("report_extraction", None)
    store_id = session.get("report_store_id")
    payload = None
    if store_id:
        try:
            with open(_report_path(store_id), encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            payload = None
        ttl = current_app.config.get("REPORT_TTL_SECONDS", 0)
        expired = (payload is None or
                   time.time() - payload.get("saved_at", 0) > ttl)
        if expired:
            _delete_report(store_id)
            session.pop("report_store_id", None)
            payload = None
    if payload is None and legacy:
        payload = legacy
        session["report_store_id"] = _store_report(legacy)
    return payload


# ----------------------------------------------------------------------- main
@main_bp.route("/")
def index():
    return render_template(
        "index.html", xray_status=_analyzer().status(),
        available_models=len(model_registry.available_models(_models_root())),
        total_models=len(model_registry.list_models(_models_root())))


@main_bp.route("/about")
def about():
    return render_template("about.html")


@main_bp.route("/status")
def status():
    return render_template("status.html", status=_system_status())


@main_bp.route("/history")
def history_view():
    return render_template(
        "history.html",
        entries=history_store.entries(current_app.config, _activity_id()),
        ttl_hours=round(current_app.config["HISTORY_TTL_SECONDS"] / 3600))


@main_bp.route("/history/clear", methods=["POST"])
def history_clear():
    history_store.clear(current_app.config, _activity_id())
    flash("Activity history cleared.", "info")
    return redirect(url_for("main.history_view"))


# ---------------------------------------------------------------------- models
@models_bp.route("/models")
def model_center():
    rows = []
    for spec in model_registry.list_models(_models_root()):
        rows.append({"spec": spec,
                     "status": _adapter_for(spec).status()})
    return render_template("models.html", models=rows)


@models_bp.route("/models/<key>")
def model_detail(key):
    spec = model_registry.get_model(_models_root(), key)
    if spec is None:
        abort(404)
    return render_template(
        "model_detail.html", spec=spec,
        status=_adapter_for(spec).status(),
        dataset=_dataset_info())


@models_bp.route("/models/compare", methods=["GET", "POST"])
def compare():
    specs = [spec for spec in model_registry.list_models(
        _models_root(), task=model_registry.TASK_XRAY)]
    comparisons = None
    image_filename = session.get("compare_image_file")
    if request.method == "POST":
        _cleanup_previous_upload("compare_image_file")
        image_filename = None
        file = request.files.get("xray_image")
        chosen = [request.form.get("model_a"), request.form.get("model_b")]
        heatmap = request.form.get("heatmap") == "on"
        if file is None or not file.filename:
            flash("Choose a chest X-ray image to compare models.", "error")
        elif len({key for key in chosen if key}) < 2:
            flash("Choose two different models to compare.", "error")
        else:
            try:
                data, ext, meta = _read_image_upload(file)
            except ValueError as exc:
                flash(str(exc), "error")
            else:
                image_filename = _save_upload(data, ext, prefix="compare")
                session["compare_image_file"] = image_filename
                comparisons = []
                for key in chosen:
                    spec = model_registry.get_model(_models_root(), key)
                    if spec is None:
                        flash("One of the selected models is unknown.", "error")
                        continue
                    adapter = _adapter_for(spec)
                    status = adapter.status()
                    if not status["available"]:
                        comparisons.append({
                            "spec": spec, "status": status, "result": None})
                        continue
                    if heatmap:
                        result = adapter.analyze_bytes(data, gradcam=True)
                    else:
                        result = adapter.analyze_bytes(data)
                    comparisons.append({
                        "spec": spec, "status": status, "result": result})
                    if result["ok"]:
                        _record("compare", f"{spec['name']}: {result['label']}",
                                {"model_key": spec["key"],
                                 "label": result["label"]})
                    else:
                        flash(f"{spec['name']}: {result['error']}", "error")
                if meta:
                    flash("DICOM imported and converted to PNG.", "info")
    return render_template("compare.html", models=specs,
                           comparisons=comparisons,
                           image_filename=image_filename)


@models_bp.route("/uploads/compare/<filename>")
def compare_image(filename):
    if filename != session.get("compare_image_file"):
        abort(404)
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)


# ------------------------------------------------------------------------ xray
@xray_bp.route("/xray", methods=["GET", "POST"])
def xray():
    sweep_stale_uploads(current_app.config["UPLOAD_FOLDER"],
                        current_app.config.get("UPLOAD_MAX_AGE_SECONDS", 0))
    specs = model_registry.list_models(_models_root(),
                                       task=model_registry.TASK_XRAY)
    selected_key = model_registry.DENSENET_KEY
    result = None
    image_filename = None
    if request.method == "POST":
        file = request.files.get("xray_image")
        selected_key = (request.form.get("model")
                        or model_registry.DENSENET_KEY)
        spec = model_registry.get_model(_models_root(), selected_key)
        if spec is None or spec["task"] != model_registry.TASK_XRAY:
            flash("Unknown model selected; using the default model.", "error")
            spec = model_registry.get_model(_models_root(),
                                            model_registry.DENSENET_KEY)
            selected_key = spec["key"]
        if file is None or not file.filename:
            flash("Please choose a chest X-ray image to upload.", "error")
        else:
            try:
                data, ext, meta = _read_image_upload(file)
            except ValueError as exc:
                flash(str(exc), "error")
            else:
                adapter = _adapter_for(spec)
                status = adapter.status()
                if not status["available"]:
                    flash(f"{spec['name']} is not available: "
                          f"{status['detail']}", "error")
                else:
                    _cleanup_previous_upload("xray_image_file")
                    image_filename = _save_upload(data, ext)
                    if request.form.get("heatmap") == "on":
                        result = adapter.analyze_bytes(data, gradcam=True)
                    else:
                        result = adapter.analyze_bytes(data)
                    if result["ok"]:
                        session["xray_image_file"] = image_filename
                        stored = {k: result[k] for k in
                                  ("label", "pneumonia_probability",
                                   "confidence", "model", "disclaimer")}
                        stored["model_key"] = spec["key"]
                        session["xray_result"] = stored
                        _record("xray", f"{spec['name']}: {result['label']}",
                                {"model_key": spec["key"],
                                 "label": result["label"]})
                        if meta:
                            flash("DICOM imported and converted to PNG.",
                                  "info")
                    else:
                        _remove_upload(image_filename)
                        image_filename = None
                        flash(result["error"], "error")
    return render_template("xray.html", result=result,
                           image_filename=image_filename,
                           models=specs, selected_key=selected_key,
                           status=_adapter_for(model_registry.get_model(
                               _models_root(), selected_key)).status())


@xray_bp.route("/uploads/xray/<filename>")
def uploaded_xray(filename):
    if filename != session.get("xray_image_file"):
        abort(404)
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)


# ------------------------------------------------------------------------ labs
@lab_bp.route("/labs", methods=["GET", "POST"])
@lab_bp.route("/labs/interpret", methods=["POST"])
def labs():
    prefill = []
    case_key = request.args.get("case")
    if request.method == "GET" and case_key in SAMPLE_CASES:
        prefill = SAMPLE_CASES[case_key]["entries"]
    result = None
    if request.method == "POST":
        entries = []
        # The candidate form posts a sentinel; the manual form does not.
        # Only the candidate form has selectable rows, so only it must treat
        # "no boxes sent" as "nothing selected" instead of "select everything".
        candidates_only = request.form.get("form_source") == "candidates"
        checked = set(request.form.getlist("use_candidate"))
        tests = request.form.getlist("test")
        values = request.form.getlist("value")
        ranges = request.form.getlist("range")
        for i, (t, v, r) in enumerate(zip(tests, values, ranges)):
            if candidates_only and str(i) not in checked:
                continue
            if str(t).strip() and str(v).strip():
                entries.append({"test": t, "value": v, "range": r})
        if not entries:
            if candidates_only:
                flash("Select at least one candidate to interpret.", "error")
            else:
                flash("Enter at least one laboratory measurement.", "error")
        else:
            result = analyze_entries(entries)
            if result["results"]:
                session["lab_result"] = result
                flags = sorted({row["flag"] for row in result["results"]})
                _record("labs", f"{len(result['results'])} measurements",
                        {"count": len(result["results"]),
                         "flags": ",".join(flags)})
            for err in result["errors"]:
                flash(err, "error")
    return render_template(
        "labs.html", result=result, ref_ranges=REFERENCE_RANGES,
        sample_cases=SAMPLE_CASES, prefill=prefill,
        range_options_for=range_options_for)


# ---------------------------------------------------------------------- report
@report_bp.route("/report", methods=["GET", "POST"])
def report():
    extraction = None
    candidates = None
    reviewed_text = ""
    if request.method == "POST":
        file = request.files.get("pdf_file")
        action = request.form.get("action", "extract")
        if action == "review":
            reviewed_text = request.form.get("reviewed_text", "")
            candidates = find_lab_candidates(reviewed_text)
            extraction = _load_report()
            if extraction:
                extraction = {**extraction, "reviewed": True}
                _record("report_review",
                        f"{len(candidates)} candidates found",
                        {"candidates": len(candidates)})
            flash("Review the detected candidates below, then interpret "
                  "the ones you trust.", "info")
        elif file is None or not file.filename:
            flash("Please choose a PDF report to upload.", "error")
        else:
            ext = (file.filename.rsplit(".", 1)[-1].lower()
                   if "." in file.filename else "")
            if ext not in current_app.config["ALLOWED_PDF_EXTENSIONS"]:
                shown = f".{ext}" if ext else file.filename
                flash(f"Unsupported file type '{shown}'. Only PDF is allowed.",
                      "error")
            else:
                extraction = extract_from_pdf(file.read())
                if extraction["ok"]:
                    session["report_store_id"] = _store_report({
                        "num_pages": extraction["num_pages"],
                        "pages": extraction["pages"],
                        "needs_ocr": extraction["needs_ocr"],
                        "ocr_message": extraction.get("ocr_message"),
                    })
                    reviewed_text = "\n\n".join(
                        p["text"] for p in extraction["pages"])
                    _record("report", f"PDF, {extraction['num_pages']} pages",
                            {"pages": extraction["num_pages"]})
                else:
                    flash(extraction["error"], "error")
                    extraction = None
    if extraction is None:
        extraction = _load_report()
    return render_template("report.html", extraction=extraction,
                           candidates=candidates, reviewed_text=reviewed_text)


# --------------------------------------------------------------------- summary
def _build_summary_text() -> str:
    lines = ["MediScan AI — Research Summary",
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             "For research use only — not for diagnostic use.", "=" * 60]

    xray = session.get("xray_result")
    lines.append("\n[Chest X-ray screening]")
    if xray:
        lines.append(f"  Model: {xray['model']}")
        lines.append(f"  Label: {xray['label']}")
        lines.append(f"  P(pneumonia): {xray['pneumonia_probability']:.3f}")
        lines.append(f"  Model confidence: {xray['confidence']:.1%}")
        lines.append(f"  Note: {xray['disclaimer']}")
    else:
        lines.append("  (no X-ray analysis performed)")

    lab = session.get("lab_result")
    lines.append("\n[Blood-test interpretation]")
    if lab and lab.get("results"):
        for r in lab["results"]:
            lines.append(f"  {r['label']}: {r['value']} {r['unit']} "
                         f"[{r['range_low']}-{r['range_high']}] -> "
                         f"{r['flag'].upper()}")
    else:
        lines.append("  (no blood-test interpretation performed)")

    rep = _load_report()
    lines.append("\n[PDF report extraction]")
    if rep:
        lines.append(f"  Pages extracted: {rep['num_pages']}")
        lines.append(f"  Scanned pages detected: "
                     f"{'yes — OCR required' if rep['needs_ocr'] else 'no'}")
        for page in rep["pages"]:
            preview = (page["text"][:200].replace("\n", " ")
                       if page["text"] else "(no extractable text)")
            lines.append(f"  Page {page['page']}: {preview}")
    else:
        lines.append("  (no PDF report processed)")

    lines.append("\n" + "=" * 60)
    lines.append("Privacy: all processing is local; files are not sent to "
                 "external services.")
    return "\n".join(lines)


@summary_bp.route("/summary")
def summary():
    return render_template("summary.html",
                           has_xray=bool(session.get("xray_result")),
                           has_lab=bool(session.get("lab_result")),
                           has_report=bool(_load_report()),
                           summary_text=_build_summary_text())


@summary_bp.route("/summary/download")
def summary_download():
    text = _build_summary_text()
    from io import BytesIO
    buf = BytesIO(text.encode("utf-8"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    _record("summary", f"Summary exported ({len(text)} chars)",
            {"chars": len(text)})
    return send_file(buf, mimetype="text/plain", as_attachment=True,
                     download_name=f"mediscan_summary_{stamp}.txt")


@summary_bp.route("/reset", methods=["POST"])
def reset_session():
    _remove_upload(session.get("xray_image_file"))
    _remove_upload(session.get("compare_image_file"))
    _delete_report(session.get("report_store_id"))
    history_store.clear(current_app.config, _activity_id())
    session.clear()
    flash("Session cleared. Ready for a new analysis.", "info")
    return redirect(url_for("main.index"))


# ----------------------------------------------------------------- evaluation
@eval_bp.route("/eval", methods=["GET", "POST"])
def dashboard():
    config = current_app.config
    specs = model_registry.list_models(_models_root(),
                                       task=model_registry.TASK_XRAY)
    dataset = _dataset_info()
    if request.method == "POST":
        spec = model_registry.get_model(_models_root(),
                                        request.form.get("model", ""))
        split = request.form.get("split", "test")
        if split not in ("train", "val", "test"):
            split = "test"
        try:
            limit = int(request.form.get("limit", "100"))
        except ValueError:
            limit = 100
        limit = max(10, min(limit, config["MAX_QUICK_EVAL"]))
        if spec is None or spec["task"] != model_registry.TASK_XRAY:
            flash("Choose a registered chest X-ray model.", "error")
        elif not dataset["available"]:
            flash("PneumoniaMNIST was not found on this server; quick "
                  "evaluation needs validation_data/pneumoniamnist.npz.",
                  "error")
        else:
            adapter = _adapter_for(spec)
            status = adapter.status()
            if not status["available"]:
                flash(f"{spec['name']} is not available: {status['detail']}",
                      "error")
            else:
                try:
                    report = evaluation.evaluate_npz(
                        adapter, spec, config["DATASET_NPZ"], split=split,
                        limit=limit)
                except Exception:
                    current_app.logger.exception("Quick evaluation failed")
                    flash("Evaluation failed. See the server log, then try "
                          "again.", "error")
                else:
                    report_id = evaluation.save_report(config, report)
                    _record("evaluation",
                            f"{spec['name']} on {split} "
                            f"({report['evaluated_samples']} samples)",
                            {"model_key": spec["key"],
                             "n": report["evaluated_samples"]})
                    flash("Evaluation finished; report saved for 24 hours.",
                          "info")
                    return redirect(url_for("eval.report_view",
                                            report_id=report_id))
    return render_template(
        "eval.html", models=specs, dataset=dataset,
        reports=evaluation.list_reports(config), max_limit=config["MAX_QUICK_EVAL"])


@eval_bp.route("/eval/report/<report_id>.json")
def report_download(report_id):
    report = evaluation.load_report(current_app.config, report_id)
    if report is None:
        abort(404)
    body = json.dumps(report, indent=2, sort_keys=True)
    return current_app.response_class(
        body, mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="mediscan_'
                                        f'evaluation_{report_id}.json"'})


@eval_bp.route("/eval/report/<report_id>")
def report_view(report_id):
    report = evaluation.load_report(current_app.config, report_id)
    if report is None:
        abort(404)
    return render_template(
        "eval_report.html", report=report,
        roc_svg=charts.roc_svg(report["metrics"]["roc_points"]),
        spec=model_registry.get_model(_models_root(), report["model_key"]))


def register_blueprints(app):
    for blueprint in (main_bp, xray_bp, lab_bp, report_bp, summary_bp,
                      models_bp, eval_bp):
        app.register_blueprint(blueprint)
