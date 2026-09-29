"""Phase A artifact builder: metadata + explicit classification development only.

No training/inference, no bootstrap, no legacy imports, no protected image reads.
Output must be a NEW directory below experiments/results/audit. Run as a module.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

from .audit_classification_oof import audit_legacy_predictions, CANONICAL_FIELDS, CLASSES, SEEDS
from .validate_dataset_isolation import validate_dataset_isolation

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "yolo_wound_cls_dataset_v3"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def allowed_development_file(path, roots):
    resolved = Path(path).resolve()
    if not any(resolved.is_relative_to(Path(r).resolve()) for r in roots):
        raise ValueError("development path escaped fixed train/val roots")
    if {p.lower() for p in Path(path).parts + resolved.parts} & {"test", "testing", "blind_test", "locked_test"}:
        raise ValueError("protected path")
    return resolved


def development_manifest(dataset):
    """Hash only immediate class files inside explicit train/val allowlist."""
    roots = [dataset / split for split in ("train", "val")]
    # Verify roots themselves are not redirected outside the dataset.
    for r in roots:
        if not r.resolve().is_relative_to(dataset.resolve()) or r.is_symlink():
            raise ValueError("redirected development root")
    rows = []
    for split, root in zip(("train", "val"), roots):
        for label in CLASSES:
            directory = allowed_development_file(root / label, roots)
            for p in sorted(directory.iterdir()):
                if p.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
                    continue
                resolved = allowed_development_file(p, roots)
                relative = p.relative_to(dataset).as_posix()
                with resolved.open("rb") as stream:
                    md5 = hashlib.file_digest(stream, "md5").hexdigest()
                rows.append({"sample_id": relative, "image_path_relative": relative, "path": str(resolved),
                             "sha256": sha(resolved), "md5": md5, "phash": "", "perceptual_group": "",
                             "source_dataset": "UNVERIFIED_CLASSIFICATION_SOURCE", "source_case_id": "",
                             "patient_id": "", "video_id": "", "frame_family_id": "", "derived_family_id": "",
                             "derived_from": "", "class_label": label, "usage_role": split})
    return rows


def source_inventory():
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    extensions = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".tsx", ".ts"}
    paths = {ROOT / p for p in tracked if p and Path(p).suffix in extensions
             and "package-lock" not in p and not p.startswith("docs/evidence/")}
    for directory in ("work", "experiments/scripts", "experiments/review_v2", "experiments/configs"):
        paths.update(p for p in (ROOT / directory).rglob("*") if p.suffix in {".py", ".yaml", ".yml"})
    entries, hits = [], []
    patterns = {"count_3622": r"3,?622", "test_reference": r"\btest\b|blind[_ -]?test|testing",
                "co2_reference": r"co2wound", "color_route": r"\.predict\(|predict_materialized|RGB2BGR|BGR2RGB|imdecode|convert\([\"']RGB"}
    for p in sorted(paths):
        if not p.is_file() or not p.resolve().is_relative_to(ROOT):
            continue
        rel = p.relative_to(ROOT).as_posix()
        # Scope restricted to source/docs; no credentials, databases or dataset traversal.
        text = p.read_text(encoding="utf-8-sig", errors="replace")
        entries.append({"path": rel, "sha256": sha(p), "lines": len(text.splitlines())})
        if p.suffix == ".py":
            for number, line in enumerate(text.splitlines(), 1):
                for topic, pattern in patterns.items():
                    if re.search(pattern, line, re.I):
                        hits.append({"path": rel, "line": number, "topic": topic, "text": line[:600]})
    return {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_files": entries, "source_count": len(entries), "hits": hits,
            "scope": "Whole-file lexical source scan + targeted manual review, not exhaustive dynamic call graph proof.",
            "excluded": ["secrets/databases", "protected image folders", "third-party dependencies", "image/binary artifacts", "node_modules/dist"]}


def historical_snapshot():
    paths = set()
    for directory in ("experiments/configs", "experiments/scripts", "experiments/results/statistics",
                      "experiments/results/tables", "experiments/results/predictions/val"):
        paths.update(p for p in (ROOT / directory).rglob("*") if p.suffix in {".py", ".json", ".csv", ".md", ".tex", ".yaml"})
    for d in (ROOT / "experiments/results/raw").glob("C-Arch-05-MS_s*_fold*"):
        paths.update(p for p in d.rglob("*") if p.suffix in {".yaml", ".csv", ".pt"})
    formal = ROOT / "outputs/isic_fuseg_formal_seed42_20260914"
    paths.update(p for p in formal.rglob("*") if p.suffix in {".json", ".yaml", ".csv", ".pt"})
    paths.update(ROOT / p for p in ("experiments/run_experiments.py", "woundcare_inference.py", "woundcare_safety.py"))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "experiments/results/audit").resolve()):
        raise ValueError("audit output outside approved root")
    output.mkdir(parents=True, exist_ok=False)
    before = historical_snapshot()
    dump(output / "historical_before.json", before)
    inventory = source_inventory()
    dump(output / "repository_source_inventory.json", inventory)
    rows = development_manifest(DEVELOPMENT)
    with (output / "sample_identity_manifest.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    hashes = Counter(r["md5"] for r in rows)
    duplicate = [n for n in hashes.values() if n > 1]
    dev = {"files": len(rows), "unique_md5_groups": len(hashes), "singletons": sum(n == 1 for n in hashes.values()),
           "duplicate_groups": len(duplicate), "duplicate_related_files": sum(duplicate),
           "redundant_instances": len(rows) - len(hashes),
           "split_counts": dict(Counter(r["usage_role"] for r in rows)),
           "class_counts": dict(Counter(r["class_label"] for r in rows)),
           "manifest_sha256": sha(output / "sample_identity_manifest.csv"),
           "patient_ids_verified": False, "historical_oof_membership_verified": False}
    dump(output / "development_inventory.json", dev)
    isolation = validate_dataset_isolation(rows, allowed_roots=[DEVELOPMENT / "train", DEVELOPMENT / "val"])
    dump(output / "dataset_isolation_audit.json", isolation)
    predictions = ROOT / "experiments/results/predictions/val"
    bundles = []
    for p in sorted(predictions.glob("C-Arch-05-MS_s*_fold*_predictions.json")):
        match = re.fullmatch(r"C-Arch-05-MS_s(\d+)_fold(\d+)_predictions.json", p.name)
        if match:
            bundles.append({"seed": int(match[1]), "fold": int(match[2]), "source_file": p.relative_to(ROOT).as_posix(),
                            "data": json.loads(p.read_text(encoding="utf-8"))})
    historical = json.loads((ROOT / "experiments/results/statistics/C-Arch-05-MS_bootstrap_ci_report.json").read_text(encoding="utf-8"))
    report = audit_legacy_predictions(bundles, expected_per_seed=len(rows), historical_count=historical["total_predictions"])
    report.update({"development_inventory": dev, "test_images_used": 0, "models_loaded": 0,
                   "training_runs_started": 0, "bootstrap_executed": False,
                   "protected_image_roots_enumerated": False,
                   "bootstrap_verdict": "SUPERSEDED_STATISTICS_HISTORICAL_ROW_WISE_BOOTSTRAP",
                   "corrected_ci_available": False})
    dump(output / "classification_oof_audit.json", report)
    dump(output / "classification_oof_schema.json", {"version": 1, "fields": CANONICAL_FIELDS,
         "row_unit": "one development FILE per seed; not one unique content group", "expected_rows": len(rows) * len(SEEDS),
         "required_before_materialization": ["historical ordered prediction-row/image mapping with hashes",
              "checkpoint lineage", "exactly once per image per seed", "no unknown IDs",
              "class/probability alignment", "train/val membership per fold and group disjointness"],
         "master_created": False, "reason": report["identity_verdict"]})
    with (output / "classification_oof_audit.md").open("x", encoding="utf-8") as stream:
        stream.write("# Classification OOF forensic audit\n\n"
            f"Actual rows: {report['actual_rows']}; expected: {report['expected_rows']}; legacy report: {historical['total_predictions']}.\n\n"
            "| Seed | Stored rows | Identity coverage |\n|---|---:|---|\n" +
            "".join(f"| {s} | {n} | UNKNOWN |\n" for s, n in report["per_seed_rows"].items()) +
            "\nDuplicate/missing/unknown prediction identities: UNKNOWN (null), NOT zero.\n"
            "No 22 rows were removed. No canonical master was fabricated. See JSON for every source/fold.\n"
            "Legacy CI is historical row-wise bootstrap, not valid grouped uncertainty. No corrected CI computed.\n"
            "Status: BLOCKED_BY_EXTERNAL_EVIDENCE. Test images used: 0.\n")
    after = historical_snapshot()
    dump(output / "historical_after.json", after)
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    safety = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_head": inventory["git_head"],
              "historical_files_hashed": len(before), "historical_changed": changed,
              "test_images_used": 0, "CO2Wounds_images_used": 0, "models_loaded": 0, "training_runs_started": 0,
              "read_scope": "Classification train/val bytes; code/docs and stored metadata; no protected image enumeration",
              "status": "PRESERVED" if not changed else "FAIL_HISTORICAL_CHANGED"}
    dump(output / "safety_verification.json", safety)
    print(json.dumps({"output": str(output), "actual_rows": report["actual_rows"], "inventory": dev,
                      "numeric_verdict": report["numeric_verdict"], "safety": safety}, ensure_ascii=False, indent=2))
    return 0 if not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
