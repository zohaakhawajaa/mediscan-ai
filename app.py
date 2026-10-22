import os
import secrets
from datetime import datetime

from flask import Flask, abort, request, session

from config import Config, validate_secret_key
from modules import history as history_store
from modules import model_registry
from modules.evaluation import sweep_reports
from modules.routes import register_blueprints, sweep_stale_uploads
from modules.xray_analyzer import XrayAnalyzer

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
CSRF_EXEMPT_ENDPOINTS = {"static"}


def _csrf_token() -> str:
    """Per-session CSRF token, created lazily and exposed to templates."""
    token = session.get("_csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf"] = token
    return token


def _register_security(app: Flask) -> None:
    @app.before_request
    def csrf_protect():
        if app.config.get("TESTING") or not app.config.get("CSRF_ENABLED"):
            return
        if request.method in SAFE_METHODS:
            return
        if request.endpoint in CSRF_EXEMPT_ENDPOINTS:
            return
        expected = session.get("_csrf")
        supplied = (request.form.get("csrf_token")
                    or request.headers.get("X-CSRF-Token"))
        if (not expected or not supplied
                or not secrets.compare_digest(expected, supplied)):
            abort(400, description=(
                "Invalid or missing security token. Reload the page and try "
                "again."))

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self'; "
            "base-uri 'self'; form-action 'self'; frame-ancestors 'none'")
        return response

    app.jinja_env.globals["csrf_token"] = _csrf_token
    app.jinja_env.filters["ts"] = lambda value: (
        datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M")
        if value else "")


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    # Environment wins over the config class so the secret can be supplied
    # at process start, then validated before anything else initialises.
    env_secret = os.environ.get("MEDISCAN_SECRET_KEY")
    if env_secret:
        app.config["SECRET_KEY"] = env_secret
    validate_secret_key(app.config.get("SECRET_KEY"),
                        os.environ.get("MEDISCAN_ENV", "development"))
    for key in ("UPLOAD_FOLDER", "MODELS_ROOT", "EVAL_STORE_FOLDER",
                "HISTORY_FOLDER"):
        if app.config.get(key):
            os.makedirs(app.config[key], exist_ok=True)
    sweep_stale_uploads(app.config["UPLOAD_FOLDER"],
                        app.config.get("UPLOAD_MAX_AGE_SECONDS", 0))
    sweep_reports(app.config)
    history_store.sweep(app.config)
    analyzer = XrayAnalyzer(
        model_dir=app.config["MODEL_DIR"],
        repo_id=app.config["HF_REPO_ID"],
        checkpoint_relpath=app.config["HF_CHECKPOINT"],
    )
    app.extensions["xray_analyzer"] = analyzer
    # Lazily-built adapter cache; the default model is the shared analyzer.
    app.extensions["model_adapters"] = {
        model_registry.DENSENET_KEY: analyzer}
    register_blueprints(app)
    _register_security(app)
    return app


app = create_app()

if __name__ == "__main__":
    debug = os.environ.get("MEDISCAN_DEBUG") == "1"
    port = int(os.environ.get("MEDISCAN_PORT", "5001"))
    app.run(host="127.0.0.1", port=port, debug=debug)
