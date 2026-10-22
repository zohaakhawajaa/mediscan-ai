import os

from app import app

if __name__ == "__main__":
    # Local development launcher. Binds to loopback by default; set
    # MEDISCAN_HOST=0.0.0.0 to expose on the LAN. MEDISCAN_PORT wins over the
    # conventional PORT variable; the default stays 5001 for local runs.
    port = int(os.environ.get("MEDISCAN_PORT")
               or os.environ.get("PORT") or "5001")
    host = os.environ.get("MEDISCAN_HOST", "127.0.0.1")
    app.run(host=host, port=port,
            debug=os.environ.get("MEDISCAN_DEBUG") == "1")
