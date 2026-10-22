import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Development-only placeholder. Refused outright when MEDISCAN_ENV=production.
DEV_SECRET_KEY = "dev-only-change-me"


def validate_secret_key(secret_key, env_name="development"):
    """Refuse to start in production with a missing or known-default secret.

    The session cookie is signed (not encrypted), so anyone knowing the key
    can forge sessions in development this is acceptable; in production it
    is not. Raises RuntimeError with a safe, copy-pasteable remedy.
    """
    if str(env_name or "development").strip().lower() not in ("production",
                                                              "prod"):
        return
    if not secret_key or secret_key == DEV_SECRET_KEY:
        raise RuntimeError(
            "MEDISCAN_ENV=production requires a real MEDISCAN_SECRET_KEY; "
            "the development default (or an empty value) would let anyone "
            "forge session cookies. Generate one before starting, e.g. in "
            "PowerShell: "
            "$b = New-Object byte[] 32; "
            "[System.Security.Cryptography.RandomNumberGenerator]::Create()"
            ".GetBytes($b); "
            "$env:MEDISCAN_SECRET_KEY = [Convert]::ToBase64String($b). "
            "Keep the value out of source control and never reuse it.")


class Config:
    ENV_NAME = os.environ.get("MEDISCAN_ENV", "development")
    SECRET_KEY = os.environ.get("MEDISCAN_SECRET_KEY") or DEV_SECRET_KEY
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB upload cap
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    # Files older than this are removed from UPLOAD_FOLDER (startup + GET /xray).
    UPLOAD_MAX_AGE_SECONDS = 24 * 60 * 60
    # Extracted report text lives on the server, NOT in the session cookie:
    # the cookie is signed but unencrypted, and large reports overflow the
    # ~4 KB browser cookie limit.
    REPORT_STORE_FOLDER = os.path.join(BASE_DIR, ".report_store")
    REPORT_TTL_SECONDS = 6 * 60 * 60
    MODEL_DIR = os.path.join(BASE_DIR, "models", "xray")
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"}
    ALLOWED_PDF_EXTENSIONS = {"pdf"}
    HF_REPO_ID = "luanacarolina/pneumonia-chest-xray-classifier"
    HF_CHECKPOINT = "checkpoints/binary/densenet121.pt"
