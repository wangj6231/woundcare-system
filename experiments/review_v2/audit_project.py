"""Read-only audit of stored experiment evidence and explicit DEVELOPMENT data.

Run: python -m experiments.review_v2.audit_project --output outputs/model_review_20260914
No model is loaded, trained, or evaluated. No test image is opened. Outputs are
new audit artifacts only; an existing output directory is never overwritten.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps
from scipy.fft import dctn

from .guards import validate_polygon_line
from .metrics import classification_metrics

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 123, 3407, 2026, 999)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False, default=str)


def snapshot_sealed() -> dict:
    files = [ROOT / "experiments/run_experiments.py", ROOT / "experiments/experiment_log.csv"]
    for name in ("experiments/scripts", "experiments/configs", "experiments/results/statistics",
                 "experiments/results/tables", "experiments/results/predictions",
                 "outputs/segmentation/co2wounds_external_test_20260831"):
        folder = ROOT / name
        files.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".json", ".md", ".csv"})
    return {rel(p): sha(p) for p in sorted(set(files)) if p.exists()}


def audit_classification() -> dict:
    runs, errors, all_true, all_pred, vocab = [], [], [], [], None
    seeds = defaultdict(list)
    for seed in SEEDS:
        for fold in range(1, 6):
            path = ROOT / f"experiments/results/predictions/val/C-Arch-05-MS_s{seed}_fold{fold}_predictions.json"
            if not path.exists():
                errors.append(f"missing {rel(path)}")
                continue
            doc = read_json(path)
            if vocab is None:
                vocab = doc["class_names"]
            if doc["class_names"] != vocab:
                errors.append(f"class vocabulary/order differs: {rel(path)}")
                continue
            metrics = classification_metrics(doc["y_true"], doc["y_pred"], len(vocab))
            probabilities = np.asarray(doc["y_prob"], dtype=float)
            valid_prob = (probabilities.shape == (metrics["n"], len(vocab)) and
                          np.isfinite(probabilities).all() and (probabilities >= 0).all() and
                          (probabilities <= 1).all() and np.allclose(probabilities.sum(axis=1), 1, atol=1e-4))
            if not valid_prob:
                errors.append(f"invalid probabilities: {rel(path)}")
            elif not np.array_equal(probabilities.argmax(axis=1), doc["y_pred"]):
                errors.append(f"probability argmax differs from stored prediction: {rel(path)}")
            if valid_prob:
                top = np.argsort(probabilities, axis=1)[:, -min(5, len(vocab)):]
                metrics["top5_accuracy"] = float(np.any(top == np.asarray(doc["y_true"])[:, None], axis=1).mean())
            all_true.extend(doc["y_true"])
            all_pred.extend(doc["y_pred"])
            row = {"seed": seed, "fold": fold, "path": rel(path), "sha256": sha(path), **metrics}
            runs.append(row)
            seeds[seed].append(row)
    if not runs:
        return {"status": "MISSING", "errors": errors}
    pooled = classification_metrics(all_true, all_pred, len(vocab))
    names = ["accuracy_all_samples", "macro_f1_fixed_classes", "weighted_f1", "top5_accuracy"]
    means = {}
    for name in names:
        v = np.array([r[name] for r in runs if name in r])
        seed_means = [np.mean([r[name] for r in s if name in r]) for s in seeds.values()]
        means[name] = {"run_mean": float(v.mean()), "run_sd_ddof0": float(v.std()),
                       "run_sd_ddof1": float(v.std(ddof=1)) if len(v) > 1 else None,
                       "between_seed_sd_of_unweighted_fold_means_ddof1": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else None}
    for i, name in enumerate(vocab):
        vals = [r["recall"][i] for r in runs if r["recall"][i] is not None]
        means[f"recall/{name}"] = {"run_mean": float(np.mean(vals)), "run_sd_ddof0": float(np.std(vals)),
                                   "run_sd_ddof1": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
    old = ROOT / "experiments/results/statistics/C-Arch-05-MS_bootstrap_ci_report.json"
    return {"status": "RECOMPUTED_DESCRIPTIVES_ONLY" if not errors else "DATA_VALIDATION_FAILED",
            "class_names": vocab, "errors": errors, "runs": runs, "run_count": len(runs),
            "n_rows": len(all_true), "rows_by_seed": {s: sum(r["n"] for r in rows) for s, rows in seeds.items()},
            "run_descriptives": means, "pooled_repeated_prediction_metrics": pooled,
            "pooled_not_independent_image_count": True,
            "identity_fields_in_legacy_files": False,
            "grouped_bootstrap_ci": None,
            "ci_reason": "No per-prediction image/hash/group identity in saved legacy arrays; cannot verify grouped resampling or independent patient observations.",
            "legacy_ci_report": read_json(old) if old.exists() else None}


def inventory_runs() -> tuple[list, list]:
    rows, errors, seen = [], [], set()
    for folder in (ROOT / "runs", ROOT / "experiments/results", ROOT / "Wound_AI_Detection_v2"):
        for path in sorted(folder.rglob("results.csv")):
            if path in seen:
                continue
            seen.add(path)
            try:
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    values = [{k.strip(): v.strip() for k, v in r.items() if k} for r in csv.DictReader(handle)]
                args_file = path.parent / "args.yaml"
                args = yaml.safe_load(args_file.read_text(encoding="utf-8-sig")) if args_file.exists() else {}
                last = values[-1] if values else {}
                numeric = {}
                nonfinite = []
                for idx, row in enumerate(values):
                    for key, value in row.items():
                        try:
                            number = float(value)
                            if not math.isfinite(number):
                                nonfinite.append({"row": idx + 1, "key": key})
                            elif idx == len(values) - 1:
                                numeric[key] = number
                        except ValueError:
                            pass
                final_metrics = {k: v for k, v in numeric.items() if k.startswith("metrics/")}
                requested = args.get("epochs")
                rows.append({"run_path": rel(path.parent), "task": args.get("task", "unknown"),
                             "model": str(args.get("model", "unknown")), "data": str(args.get("data", "unknown")),
                             "seed": args.get("seed"), "imgsz": args.get("imgsz"), "batch": args.get("batch"),
                             "epochs_requested": requested, "recorded_epoch_rows": len(values),
                             "last_epoch_field": last.get("epoch"), "patience": args.get("patience"),
                             "optimizer": args.get("optimizer"), "lr0": args.get("lr0"),
                             "warmup_bias_lr": args.get("warmup_bias_lr"),
                             "best_checkpoint_exists": (path.parent / "weights/best.pt").is_file(),
                             "last_checkpoint_exists": (path.parent / "weights/last.pt").is_file(),
                             "nonfinite_cells": nonfinite, "final_recorded_metrics": final_metrics,
                             "status": "EMPTY" if not values else "NONFINITE" if nonfinite else "STORED_HISTORY_NOT_COMPLETION_PROOF",
                             "completion_note": "Fewer rows than requested is not alone proof of early stopping, failure, or completion.",
                             "results_sha256": sha(path), "args_sha256": sha(args_file) if args_file.exists() else None})
            except Exception as exc:
                errors.append({"path": rel(path), "error": str(exc)})
    return rows, errors


def inventory_scripts() -> list:
    scripts = []
    for folder in ("work", "experiments/scripts", "scripts"):
        for path in sorted((ROOT / folder).glob("*.py")):
            if not any(word in path.name for word in ("train", "eval", "audit", "bootstrap", "cascade", "dseg", "statistic", "fold", "experiment", "supervisor")):
                continue
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"))
                syntax = "PASS"
            except Exception as exc:
                syntax = f"FAIL: {exc}"
            scripts.append({"path": rel(path), "sha256": sha(path), "syntax": syntax,
                            "scope": "static syntax only; no imports or execution"})
    return scripts


def phash(path: Path) -> int:
    # imagehash-compatible pHash: grayscale 32x32 DCT, upper-left 8x8,
    # median threshold including DC. This is similarity screening, not patient identity.
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        pixels = np.asarray(im.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    block = dctn(pixels, type=2, norm=None)[:8, :8]
    bits = (block > np.median(block)).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def audit_development_dataset(name: str, splits: tuple[str, ...], cache: dict) -> dict:
    # Explicit directory allowlist. Never enumerate or open test image directories.
    allowed = {"yolo_dataset_dseg06_small_multi_v1", "yolo_dataset_dseg07_yasin_wound_seg_v2", "yolo_dataset_dseg08_replay_v1", "yolo_dataset_fuseg_seg_v2"}
    if name not in allowed or any(s not in {"train", "val", "val_retention"} for s in splits):
        raise ValueError("not an allowed development dataset/split")
    base = ROOT / name
    by_split, errors = {}, []
    for split in splits:
        image_dir, label_dir = base / "images" / split, base / "labels" / split
        images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        rows = []
        for image in images:
            label = (label_dir / image.relative_to(image_dir)).with_suffix(".txt")
            try:
                image_sha = sha(image)
                if image_sha not in cache:
                    with Image.open(image) as im:
                        dimensions = im.size
                        im.verify()
                    cache[image_sha] = (phash(image), dimensions)
                p_hash, dimensions = cache[image_sha]
                if not label.exists():
                    raise ValueError("missing label; not an implicit negative")
                lines = [v for v in label.read_text(encoding="utf-8-sig").splitlines() if v.strip()]
                for line in lines:
                    validate_polygon_line(line)
                rows.append({"image": image.relative_to(base).as_posix(), "image_sha256": image_sha,
                             "label_sha256": sha(label), "phash": f"{p_hash:016x}",
                             "dimensions": dimensions, "instances": len(lines), "hardlink_count": image.stat().st_nlink,
                             "label_hardlink_count": label.stat().st_nlink})
            except Exception as exc:
                errors.append({"image": image.relative_to(base).as_posix(), "error": str(exc)})
        expected_labels = {p.with_suffix(".txt").relative_to(image_dir).as_posix() for p in images}
        actual_labels = {p.relative_to(label_dir).as_posix() for p in label_dir.rglob("*.txt")}
        orphan = sorted(actual_labels - expected_labels)
        if orphan:
            errors.append({"split": split, "orphan_labels": orphan})
        by_split[split] = {"image_files": len(images), "audited_images": len(rows),
                           "unique_content_sha256": len({r["image_sha256"] for r in rows}),
                           "negative_images": sum(r["instances"] == 0 for r in rows),
                           "instances": sum(r["instances"] for r in rows),
                           "hardlinked_images": sum(r["hardlink_count"] > 1 for r in rows),
                           "hardlinked_labels": sum(r["label_hardlink_count"] > 1 for r in rows), "rows": rows}
    overlaps = []
    for val in [s for s in splits if s != "train"]:
        train_unique = {r["image_sha256"]: r for r in by_split["train"]["rows"]}
        val_unique = {r["image_sha256"]: r for r in by_split[val]["rows"]}
        near = []
        for t in train_unique.values():
            for v in val_unique.values():
                distance = (int(t["phash"], 16) ^ int(v["phash"], 16)).bit_count()
                if distance <= 4:
                    near.append({"train": t["image"], "val": v["image"], "distance": distance})
        overlaps.append({"comparison": f"train vs {val}",
                         "exact_content_overlap": len(train_unique.keys() & val_unique.keys()),
                         "phash_distance_le4_pairs": near})
    return {"dataset": name, "status": "PASS_TECHNICAL_ONLY" if not errors and all(
        r["exact_content_overlap"] == 0 and not r["phash_distance_le4_pairs"] for r in overlaps) else "REVIEW_REQUIRED",
        "errors": errors, "splits": by_split, "cross_split": overlaps,
        "test_images_opened_this_audit": 0,
        "limitations": "No patient-level identity proof or clinical/semantic annotation approval; pHash does not prove independence. Existing labels are never changed, including hardlinks."}


def check_replay_manifest(dataset: dict) -> dict:
    path = ROOT / "outputs/dseg08_replay_manifest_20260830.csv"
    actual = {(split, r["image"]): r for split, data in dataset["splits"].items() for r in data["rows"]}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    issues, seen = [], set()
    counts = Counter()
    for row in manifest:
        key = (row["split"], row["destination_image"].replace("\\", "/"))
        if key in seen:
            issues.append({"duplicate_manifest_key": key})
        seen.add(key)
        observed = actual.get(key)
        if not observed:
            issues.append({"missing_or_failed_audit": key})
        elif any(observed[k] != row[k].lower() for k in ("image_sha256", "label_sha256")):
            issues.append({"changed_hash": key})
        counts[f'{row["split"]}/{row["source_dataset"]}'] += 1
    for missing in actual.keys() - seen:
        issues.append({"not_in_manifest": missing})
    return {"status": "PASS" if not issues else "FAIL", "manifest": rel(path), "manifest_sha256": sha(path),
            "records": len(manifest), "source_split_counts": dict(counts), "issues": issues}


def inventory_summaries() -> list:
    items = []
    for path in sorted((ROOT / "outputs").rglob("*.json")):
        if not (path.name.endswith("summary.json") or path.name.endswith("_results.json")) or path.stat().st_size > 350_000:
            continue
        try:
            doc = read_json(path)
            if not isinstance(doc, dict):
                continue
            selected = {k: v for k, v in doc.items() if k in {
                "status", "evaluation_role", "experiment_id", "run_name", "metrics", "primary_validation_metrics",
                "retention_validation_metrics", "test_images_used", "epochs_requested", "images", "n_images"}}
            if selected:
                items.append({"path": rel(path), "sha256": sha(path), "stored_fields": selected,
                              "invalidated_marker_in_same_directory": any(path.parent.glob("INVALIDATED*.json")),
                              "note": "Stored claims, not newly verified completion or comparable cohorts"})
        except Exception as exc:
            items.append({"path": rel(path), "error": str(exc)})
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    allowed_output = (ROOT / "outputs").resolve()
    if allowed_output not in output.parents:
        raise ValueError("output must be a NEW directory inside workspace outputs")
    output.mkdir(parents=True, exist_ok=False)
    before = snapshot_sealed()
    write_json(output / "sealed_before_sha256.json", before)
    print("Auditing stored classification predictions (no model inference)", flush=True)
    cls = audit_classification()
    write_json(output / "classification_recomputed.json", cls)
    runs, run_errors = inventory_runs()
    write_json(output / "training_run_inventory.json", {"runs": runs, "errors": run_errors})
    with (output / "training_run_inventory.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        fields = ["run_path", "task", "model", "data", "seed", "imgsz", "batch", "epochs_requested", "recorded_epoch_rows", "patience", "optimizer", "lr0", "best_checkpoint_exists", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    scripts = inventory_scripts()
    write_json(output / "script_inventory.json", scripts)
    write_json(output / "saved_summary_inventory.json", inventory_summaries())
    cache, datasets = {}, []
    for name, splits in [
        ("yolo_dataset_dseg06_small_multi_v1", ("train", "val")),
        ("yolo_dataset_dseg07_yasin_wound_seg_v2", ("train", "val")),
        ("yolo_dataset_dseg08_replay_v1", ("train", "val", "val_retention"))]:
        print(f"Auditing DEVELOPMENT files only: {name}", flush=True)
        result = audit_development_dataset(name, splits, cache)
        datasets.append(result)
        write_json(output / f"{name}_audit.json", result)
    manifest = check_replay_manifest(datasets[-1])
    write_json(output / "dseg08_manifest_verification.json", manifest)
    after = snapshot_sealed()
    integrity = {"files_checked": len(before), "unchanged": before == after,
                 "changed": [k for k in before.keys() | after.keys() if before.get(k) != after.get(k)],
                 "test_images_opened_this_audit": 0, "models_loaded": 0, "training_runs_started": 0}
    write_json(output / "sealed_integrity_verification.json", integrity)
    versions = {}
    for pkg in ("numpy", "scipy", "scikit-learn", "Pillow", "ultralytics", "torch", "torchvision", "PyYAML"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    summary = {"status": "AUDIT_COMPLETE_WITH_FINDINGS_NOT_MODEL_APPROVAL", "created_utc": datetime.now(timezone.utc).isoformat(),
               "run_inventory_count": len(runs), "script_count": len(scripts), "run_inventory_errors": run_errors,
               "classification_validation_errors": cls["errors"], "dataset_statuses": {r["dataset"]: r["status"] for r in datasets},
               "replay_manifest_status": manifest["status"], "sealed_integrity": integrity, "versions": versions,
               "source_access": {"Redscar": "USER_REPORTS_NO_REPLY; NOT_APPROVED; NOT_DOWNLOADED_BY_THIS_AUDIT"},
               "scope": "All discovered stored run histories and experiment scripts; full byte/label audits of active D-Seg-06/07v2/08 development only. Historical datasets/test pixels not re-audited."}
    write_json(output / "audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if integrity["unchanged"] and not run_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
