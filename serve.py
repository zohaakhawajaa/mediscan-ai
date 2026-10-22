import os

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("MEDISCAN_PORT", "5001"))
    app.run(host="127.0.0.1", port=port,
            debug=os.environ.get("MEDISCAN_DEBUG") == "1")
