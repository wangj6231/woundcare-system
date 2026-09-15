"""Save actual regression output without altering earlier audit evidence."""
import json
import os
import subprocess
import sys

from .audit_project import ROOT, read_json, snapshot_sealed, write_json


def main():
    audit = ROOT / "outputs/model_review_20260914"
    destination = audit / "verification_revision2.json"
    if destination.exists():
        raise FileExistsError(destination)
    commands = [[sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_experiment_review.py", "-v"],
                [sys.executable, "tests/test_woundcare_inference.py"]]
    records = []
    for cmd in commands:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=60)
        records.append({"command": cmd, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    unchanged = read_json(audit / "sealed_before_sha256.json") == snapshot_sealed()
    preflight = read_json(audit / "development_preflight/preflight.json")
    passed = all(r["exit_code"] == 0 for r in records) and unchanged
    output = {"status": "PASS_REGRESSION_NOT_TRAINING" if passed else "FAIL", "tests": records,
              "sealed_unchanged": unchanged, "new_development_preflight": preflight,
              "new_models_trained": 0, "test_images_opened": 0,
              "training_branch_validation": "mock unit tests only; real GPU training correctly blocked by source admission"}
    write_json(destination, output)
    print(json.dumps({k: v for k, v in output.items() if k != "tests"}, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
