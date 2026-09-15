"""Ephemeral loopback-only UI fixture. No real database, models or images.

For UI regression testing only. The published synthetic password below is NOT
a deployment credential. Stop this server after testing; it must not be tunneled.
"""
import os
import sys
import tempfile
from pathlib import Path

from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    with tempfile.TemporaryDirectory(prefix="woundcare_ui_fixture_") as directory:
        base = Path(directory)
        os.environ.update({
            "WOUNDCARE_ENV": "development", "WOUNDCARE_PROFESSOR_PREVIEW": "true",
            "WOUNDCARE_DB_PATH": str(base / "preview.db"),
            "WOUNDCARE_FERNET_KEY": Fernet.generate_key().decode(),
            "WOUNDCARE_BOOTSTRAP_PASSWORD": "SyntheticPreviewOnly!2026",
            "WOUNDCARE_CLS_MODEL": str(base / "missing-cls.pt"),
            "WOUNDCARE_SEG_MODEL": str(base / "missing-seg.pt"),
            "WOUNDCARE_DET_MODEL": str(base / "missing-det.pt"),
            "WOUNDCARE_ALLOWED_HOSTS": "127.0.0.1,localhost",
            "WOUNDCARE_ALLOWED_ORIGINS": "http://127.0.0.1:8097",
        })
        import backend_main
        import uvicorn
        print("Synthetic UI fixture; loopback only; no models loaded; test images used=0", flush=True)
        uvicorn.run(backend_main.app, host="127.0.0.1", port=8097, log_level="warning")


if __name__ == "__main__":
    main()
