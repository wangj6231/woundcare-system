"""Record engineering checks without starting real services or model inference."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if (ROOT / "outputs").resolve() not in output.parents:
        raise ValueError("verification output must be within workspace outputs")
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    npm = ["cmd.exe", "/d", "/c", "npm"] if os.name == "nt" else ["npm"]
    jobs = [
        ("unittest", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"], ROOT),
        ("inference_synthetic", [sys.executable, "tests/test_woundcare_inference.py"], ROOT),
        ("frontend_lint", npm + ["run", "lint"], ROOT / "wound_nurse_app"),
        ("frontend_build", npm + ["run", "build"], ROOT / "wound_nurse_app"),
    ]
    results = []
    for label, command, cwd in jobs:
        try:
            run = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
            result = {"check": label, "exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = {"check": label, "exit_code": -1, "error": str(exc)}
        results.append(result)
        print(f"{label}: {'PASS' if result['exit_code'] == 0 else 'FAIL'}", flush=True)
    from experiments.review_v2.audit_project import snapshot_sealed, read_json
    bundle = ROOT / "outputs/fuseg_warmup_revision_20260914"
    result = read_json(bundle / "result.json")
    checkpoint = Path(result["checkpoint"])
    model_hash_matches = hashlib.sha256(checkpoint.read_bytes()).hexdigest() == result["checkpoint_sha256"]
    sealed_unchanged = snapshot_sealed() == read_json(bundle / "sealed_before.json")
    passed = all(r["exit_code"] == 0 for r in results) and model_hash_matches and sealed_unchanged
    payload = {"status": "PASS_ENGINEERING_ONLY" if passed else "FAIL", "created_utc": datetime.now(timezone.utc).isoformat(),
               "checks": results, "checkpoint_sha256_matches": model_hash_matches, "sealed_metadata_unchanged": sealed_unchanged,
               "test_images_used": 0, "clinical_database_opened": False, "app_model_replaced": False,
               "scope": "synthetic unit/API tests and frontend lint/build; not model accuracy or clinical deployment validation"}
    (output / "verification.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "checks"}, ensure_ascii=False), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
