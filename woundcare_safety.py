"""Side-effect-free startup checks; never open a patient database or a key."""
from pathlib import Path
from typing import Mapping


def validate_startup(environ: Mapping[str, str], root: Path) -> None:
    production = environ.get("WOUNDCARE_ENV", "development").lower() == "production"
    preview = environ.get("WOUNDCARE_PROFESSOR_PREVIEW", "false").strip().lower() in {"1", "true", "yes"}
    if production and not environ.get("WOUNDCARE_FERNET_KEY", "").strip():
        raise RuntimeError("Production requires an explicit WOUNDCARE_FERNET_KEY; local key fallback is forbidden")
    if preview:
        configured = environ.get("WOUNDCARE_DB_PATH", "").strip()
        if not configured:
            raise RuntimeError("Professor preview requires an explicit isolated WOUNDCARE_DB_PATH")
        candidate = Path(configured).resolve()
        clinical = (root / "healthcare.db").resolve()
        if candidate == clinical or (candidate.exists() and clinical.exists() and candidate.samefile(clinical)):
            raise RuntimeError("Professor preview cannot use the default clinical database")
        if not environ.get("WOUNDCARE_FERNET_KEY", "").strip():
            raise RuntimeError("Professor preview requires its own explicit WOUNDCARE_FERNET_KEY")
    password = environ.get("WOUNDCARE_BOOTSTRAP_PASSWORD", "")
    if password and (len(password) < 12 or len(password.encode("utf-8")) > 72
                     or "replace-with" in password.lower().replace("_", "-") or "change-me" in password.lower()):
        raise RuntimeError("Bootstrap password must be unique, at least 12 characters and at most 72 UTF-8 bytes")
