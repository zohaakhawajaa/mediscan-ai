import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app import create_app
from config import Config


class TestConfig(Config):
    TESTING = True
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "test_uploads")
    REPORT_STORE_FOLDER = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "test_report_store")
    EVAL_STORE_FOLDER = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "test_eval_store")
    HISTORY_FOLDER = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "test_history_store")
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "models", "xray")


@pytest.fixture()
def app():
    app = create_app(TestConfig)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    yield app
    for name in os.listdir(app.config["UPLOAD_FOLDER"]):
        try:
            os.remove(os.path.join(app.config["UPLOAD_FOLDER"], name))
        except OSError:
            pass
    for key in ("REPORT_STORE_FOLDER", "EVAL_STORE_FOLDER", "HISTORY_FOLDER"):
        store = app.config[key]
        if os.path.isdir(store):
            for name in os.listdir(store):
                try:
                    os.remove(os.path.join(store, name))
                except OSError:
                    pass
            try:
                os.rmdir(store)
            except OSError:
                pass


@pytest.fixture()
def client(app):
    return app.test_client()
