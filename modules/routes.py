"""Flask route blueprints for MediScan AI."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime

from flask import (Blueprint, current_app, flash, redirect, render_template,
                   request, send_file, session, url_for)

from .lab_analyzer import (REFERENCE_RANGES, SAMPLE_CASES, analyze_entries,
                           range_options_for)
from .report_extractor import extract_from_pdf, find_lab_candidates
from .xray_analyzer import validate_image_upload

main_bp = Blueprint("main", __name__)
xray_bp = Blueprint("xray", __name__)
lab_bp = Blueprint("labs", __name__)
report_bp = Blueprint("report", __name__)
summary_bp = Blueprint("summary", __name__)


def _analyzer():
    return current_app.extensions["xray_analyzer"]


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
    return render_template("index.html", xray_status=_analyzer().status())


@main_bp.route("/about")
def about():
    return render_template("about.html")


# ------------------------------------------------------------------------ xray
@xray_bp.route("/xray", methods=["GET", "POST"])
def xray():
    analyzer = _analyzer()
    sweep_stale_uploads(current_app.config["UPLOAD_FOLDER"],
                        current_app.config.get("UPLOAD_MAX_AGE_SECONDS", 0))
    result = None
    image_filename = None
    if request.method == "POST":
        file = request.files.get("xray_image")
        if file is None or not file.filename:
            flash("Please choose a chest X-ray image to upload.", "error")
        else:
            error = validate_image_upload(
                file.filename, current_app.config["ALLOWED_IMAGE_EXTENSIONS"])
            if error:
                flash(error, "error")
            else:
                data = file.read()
                ext = file.filename.rsplit(".", 1)[-1].lower()
                _cleanup_previous_upload("xray_image_file")
                image_filename = f"xray_{uuid.uuid4().hex}.{ext}"
                os.makedirs(current_app.config["UPLOAD_FOLDER"], exist_ok=True)
                with open(os.path.join(current_app.config["UPLOAD_FOLDER"],
                                       image_filename), "wb") as fh:
                    fh.write(data)
                if request.form.get("heatmap") == "on":
                    result = analyzer.analyze_bytes(data, gradcam=True)
                else:
                    result = analyzer.analyze_bytes(data)
                if result["ok"]:
                    session["xray_image_file"] = image_filename
                    session["xray_result"] = {
                        k: result[k] for k in
                        ("label", "pneumonia_probability", "confidence",
                         "model", "disclaimer")}
                else:
                    _remove_upload(image_filename)
                    image_filename = None
                    flash(result["error"], "error")
    return render_template("xray.html", result=result,
                           image_filename=image_filename,
                           status=analyzer.status())


@xray_bp.route("/uploads/xray/<filename>")
def uploaded_xray(filename):
    from flask import abort, send_from_directory
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
    return send_file(buf, mimetype="text/plain", as_attachment=True,
                     download_name=f"mediscan_summary_{stamp}.txt")


@summary_bp.route("/reset", methods=["POST"])
def reset_session():
    _remove_upload(session.pop("xray_image_file", None))
    session.pop("xray_result", None)
    session.pop("lab_result", None)
    _delete_report(session.pop("report_store_id", None))
    session.pop("report_extraction", None)
    flash("Session cleared. Ready for a new analysis.", "info")
    return redirect(url_for("main.index"))


def register_blueprints(app):
    app.register_blueprint(main_bp)
    app.register_blueprint(xray_bp)
    app.register_blueprint(lab_bp)
    app.register_blueprint(report_bp)
    app.register_blueprint(summary_bp)
