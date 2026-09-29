"""Verify Phase A engineering checks. Never execute legacy model test scripts."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from .phase_a_audit import ROOT, dump, historical_snapshot, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=1)
    args = parser.parse_args()
    audit = args.audit.resolve()
    if not audit.is_relative_to((ROOT / "experiments/results/audit").resolve()):
        raise ValueError("unexpected audit directory")
    if args.iteration < 1:
        raise ValueError("iteration must be positive")
    destination = audit / ("verification_tests.json" if args.iteration == 1
                           else f"verification_tests_r{args.iteration}.json")
    if destination.exists():
        raise FileExistsError("verification is immutable; use a new audit run")
    before = json.loads((audit / "historical_before.json").read_text(encoding="utf-8"))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    checks = []
    jobs = [("targeted", ["-m", "unittest", "discover", "-s", "tests", "-p", "test_phase_a*.py", "-v"]),
            ("related", ["-m", "unittest", "discover", "-s", "tests", "-p", "test_experiment_review.py", "-v"]),
            ("full_tests_directory", ["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]),
            ("synthetic_cascade_script", ["tests/test_woundcare_inference.py"])]
    for name, command in jobs:
        run = subprocess.run([sys.executable, *command], cwd=ROOT, env=env, capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=180)
        output = run.stdout + run.stderr
        count = re.search(r"Ran (\d+) tests?", output)
        skipped = re.search(r"skipped=(\d+)", output)
        failure = re.search(r"failures=(\d+)", output)
        errors = re.search(r"errors=(\d+)", output)
        warnings = [line for line in output.splitlines() if re.search(r"\b\w+Warning:", line)]
        n = int(count[1]) if count else None
        failed = int(failure[1]) if failure else 0
        errored = int(errors[1]) if errors else 0
        skip = int(skipped[1]) if skipped else 0
        checks.append({"name": name, "command": [sys.executable, *command], "exit_code": run.returncode,
                       "tests_run": n, "passed": n - failed - errored - skip if n is not None else None,
                       "failed": failed, "errors": errored, "skipped": skip,
                       "warning_lines": warnings, "stdout": run.stdout, "stderr": run.stderr})
        print(json.dumps({k: v for k, v in checks[-1].items() if k not in {"command", "stdout", "stderr"}}), flush=True)
    # Synthetic demonstration of the OLD guard's trust boundary, not a dataset admission.
    from .review_v2.guards import validate_new_development_run
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        evidence = p / "SYNTHETIC_NOT_A_LICENSE.txt"
        evidence.write_text("synthetic unit fixture", encoding="utf-8")
        validate_new_development_run(
            {"role": "development", "test_images_used": 0, "sources": ["CO2Wounds-V2"]},
            {"CO2Wounds-V2": {"development_allowed": True, "evidence_file": str(evidence)}},
            train_groups={"a"}, val_groups={"b"}, ancestor_train_groups=None, generic_pretrained=True,
            output_dir=p / "not-created")
    after = historical_snapshot()
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    new_files = ["experiments/audit_classification_oof.py", "experiments/validate_dataset_isolation.py",
                 "experiments/phase_a_audit.py", "experiments/phase_a_verify.py", "tests/test_phase_a_audit.py",
                 "experiments/audit_classification_metrics.py", "tests/test_phase_a_metrics_audit.py"]
    result = {"status": "PASS_ENGINEERING_ONLY" if all(x["exit_code"] == 0 for x in checks) and not changed else "FAIL",
              "checks": checks, "historical_changed": changed, "historical_files_checked": len(before),
              "audit_code_sha256": {p: sha(ROOT / p) for p in new_files},
              "legacy_guard_probe": "ACCEPTED_SYNTHETIC_CO2_GATE; no training or dataset access occurred",
              "legacy_guard_scope": "Generic library trusts supplied admission; dedicated runners have additional hardcoded-source checks.",
              "test_images_used": 0, "CO2Wounds_images_used": 0, "model_inference_executed": False,
              "excluded_executable_tests": ["test_cls_model.py (loads locked test)",
                                            "experiments/test_phase1.py (mutates shared experiment log)"],
              "scope": "Entire synthetic tests directory plus standalone synthetic cascade checks; not real-model or deployment validation"}
    dump(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS_ENGINEERING_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
