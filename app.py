import os

from flask import Flask

from config import Config, validate_secret_key
from modules.routes import register_blueprints, sweep_stale_uploads
from modules.xray_analyzer import XrayAnalyzer


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
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["MODEL_DIR"], exist_ok=True)
    sweep_stale_uploads(app.config["UPLOAD_FOLDER"],
                        app.config.get("UPLOAD_MAX_AGE_SECONDS", 0))
    analyzer = XrayAnalyzer(
        model_dir=app.config["MODEL_DIR"],
        repo_id=app.config["HF_REPO_ID"],
        checkpoint_relpath=app.config["HF_CHECKPOINT"],
    )
    app.extensions["xray_analyzer"] = analyzer
    register_blueprints(app)
    return app


app = create_app()

if __name__ == "__main__":
    debug = os.environ.get("MEDISCAN_DEBUG") == "1"
    port = int(os.environ.get("MEDISCAN_PORT", "5001"))
    app.run(host="127.0.0.1", port=port, debug=debug)
