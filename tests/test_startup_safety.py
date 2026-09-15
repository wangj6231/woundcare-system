import os
import tempfile
import unittest
from pathlib import Path

from woundcare_safety import validate_startup


class StartupSafetyTests(unittest.TestCase):
    def test_local_development_unchanged(self):
        validate_startup({}, Path.cwd())

    def test_production_never_uses_local_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "secret.key").write_text("synthetic-not-a-key")
            with self.assertRaisesRegex(RuntimeError, "explicit WOUNDCARE_FERNET_KEY"):
                validate_startup({"WOUNDCARE_ENV": "production"}, root)

    def test_preview_requires_explicit_db_and_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {"WOUNDCARE_PROFESSOR_PREVIEW": "true"}
            with self.assertRaisesRegex(RuntimeError, "isolated"):
                validate_startup(env, root)
            env["WOUNDCARE_DB_PATH"] = str(root / "preview.db")
            with self.assertRaisesRegex(RuntimeError, "own explicit"):
                validate_startup(env, root)
            env["WOUNDCARE_FERNET_KEY"] = "synthetic-presence-only"
            validate_startup(env, root)
            self.assertFalse((root / "preview.db").exists())

    def test_preview_rejects_default_db_and_hardlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clinical = root / "healthcare.db"
            clinical.write_bytes(b"synthetic-not-a-db")
            alias = root / "preview.db"
            os.link(clinical, alias)
            for target in (clinical, alias):
                with self.assertRaisesRegex(RuntimeError, "clinical database"):
                    validate_startup({"WOUNDCARE_PROFESSOR_PREVIEW": "true", "WOUNDCARE_DB_PATH": str(target), "WOUNDCARE_FERNET_KEY": "test-only"}, root)

    def test_placeholder_and_bcrypt_limits(self):
        for password in ("short", "REPLACE_WITH_A_UNIQUE_PASSWORD", "傷" * 25):
            with self.assertRaises(RuntimeError):
                validate_startup({"WOUNDCARE_BOOTSTRAP_PASSWORD": password}, Path.cwd())
        validate_startup({"WOUNDCARE_BOOTSTRAP_PASSWORD": "Synthetic-Test-Password!2026"}, Path.cwd())
