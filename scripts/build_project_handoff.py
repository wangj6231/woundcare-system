"""Package an explicit source-only allowlist, never datasets, keys or weights."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "README.md", "backend_main.py", "woundcare_inference.py", "woundcare_safety.py",
    "requirements.txt", "requirements-dev.txt", "Dockerfile", ".dockerignore", ".gitignore",
    ".env.example", ".env.professor-preview.example", ".env.quick-preview.example",
    "docker-compose.professor-preview.yml", "docker-compose.quick-preview.yml", "Caddyfile.professor-preview",
    "wound_nurse_app/package.json", "wound_nurse_app/package-lock.json", "wound_nurse_app/index.html",
    "wound_nurse_app/tsconfig.json", "wound_nurse_app/tsconfig.app.json", "wound_nurse_app/tsconfig.node.json",
    "wound_nurse_app/vite.config.ts", "wound_nurse_app/eslint.config.js", "wound_nurse_app/README.md",
    "docs/PROJECT_HANDOFF_20260914.md", "docs/FUSEG_MODEL_CARD_20260914.md",
    "docs/APP_SECURITY_AND_OPERATIONS.md", "docs/PROFESSOR_PREVIEW_DEPLOYMENT.md",
    "scripts/build_project_handoff.py", "scripts/check_candidate_integration.py", "tests/run_local_preview_smoke.py",
    "tests/test_professor_preview_api.py", "tests/test_woundcare_inference.py", "tests/test_startup_safety.py",
)


def build(destination: Path):
    paths = [ROOT / name for name in FILES]
    # Current UI is CSS-based. Never include arbitrary image assets merely
    # because someone placed them under src/public.
    paths += [p for p in (ROOT / "wound_nurse_app/src").rglob("*") if p.is_file() and p.suffix in {".ts", ".tsx", ".css"}]
    paths += [ROOT / "wound_nurse_app/public/favicon.svg"]
    allowed_suffixes = {".py", ".md", ".txt", ".json", ".ts", ".tsx", ".js", ".css", ".html", ".svg", ".png", ".ico", ".yml", ".example"}
    payloads = []
    for path in sorted(set(paths)):
        name = path.relative_to(ROOT).as_posix()
        if not path.is_file() or path.is_symlink() or ROOT not in path.resolve().parents:
            raise ValueError(f"missing or escaped package member: {name}")
        if path.suffix not in allowed_suffixes and name not in FILES:
            raise ValueError(f"unexpected package file type: {name}")
        if path.suffix.lower() in {".pt", ".db", ".key", ".pem", ".zip"} or (path.name.startswith(".env") and not path.name.endswith(".example")):
            raise ValueError(f"forbidden sensitive package member: {name}")
        content = path.read_bytes()
        payloads.append((name, content, hashlib.sha256(content).hexdigest()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content, _ in payloads:
            archive.writestr("woundcare-project/" + name, content)
        archive.writestr("woundcare-project/SOURCE_MANIFEST.json", json.dumps({
            "scope": "noncommercial research prototype source handoff; no deployment approval",
            "contains_patient_data": False, "contains_weights": False,
            "members": [{"path": name, "sha256": digest, "bytes": len(content)} for name, content, digest in payloads],
        }, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        for name, _, digest in payloads:
            assert hashlib.sha256(archive.read("woundcare-project/" + name)).hexdigest() == digest
    print(json.dumps({"package": str(destination.resolve()), "source_files": len(payloads), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "bytes": destination.stat().st_size, "verified": True}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
