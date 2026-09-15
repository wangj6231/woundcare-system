"""Dedicated D-Seg-08R development runner; no blind/test mode exists.

Default is preflight only. Source admission must be backed by actual evidence,
not inferred from old training or a public download. Outputs never overwrite.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from .audit_project import ROOT, audit_development_dataset, check_replay_manifest, read_json, sha, write_json
from .guards import validate_new_development_run, validate_optimizer_settings

RECIPE = ROOT / "experiments/review_v2/recipes/D-Seg-08R_warmup_control.yaml"


def preflight(admission_file: Path, output: Path) -> dict:
    recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    validate_optimizer_settings(recipe)
    gates = read_json(admission_file)
    # Check source permission FIRST: no original pixels or model loaded on failure.
    for source in recipe["sources"]:
        g = gates.get(source, {})
        if g.get("development_allowed") is not True:
            raise ValueError(f"SOURCE_ADMISSION_BLOCKED: {source}: {g.get('status', 'missing evidence')}")
        evidence = Path(g.get("evidence_file", ""))
        if not evidence.is_absolute():
            evidence = ROOT / evidence
        if not evidence.is_file() or not g.get("evidence_sha256") or sha(evidence) != g["evidence_sha256"]:
            raise ValueError(f"SOURCE_EVIDENCE_MISSING_OR_CHANGED: {source}")
        g["evidence_file"] = str(evidence)
    cache = {}
    audits = [audit_development_dataset(name, splits, cache) for name, splits in [
        ("yolo_dataset_dseg06_small_multi_v1", ("train", "val")),
        ("yolo_dataset_dseg07_yasin_wound_seg_v2", ("train", "val")),
        ("yolo_dataset_dseg08_replay_v1", ("train", "val", "val_retention"))]]
    if any(a["status"] != "PASS_TECHNICAL_ONLY" for a in audits):
        raise ValueError("DEVELOPMENT_DATA_AUDIT_FAILED")
    replay = audits[-1]
    manifest = check_replay_manifest(replay)
    if manifest["status"] != "PASS":
        raise ValueError("FROZEN_MANIFEST_CHANGED")
    current_train = replay["splits"]["train"]["rows"]
    current_val = [r for split, s in replay["splits"].items() if split != "train" for r in s["rows"]]
    ancestor_train = [r for a in audits[:2] for r in a["splits"]["train"]["rows"]]
    if any((int(t["phash"], 16) ^ int(v["phash"], 16)).bit_count() <= 4 for t in ancestor_train for v in current_val):
        raise ValueError("ANCESTOR_PHASH_OVERLAP")
    validate_new_development_run(recipe, gates,
        train_groups={r["image_sha256"] for r in current_train},
        val_groups={r["image_sha256"] for r in current_val},
        ancestor_train_groups={r["image_sha256"] for r in ancestor_train},
        generic_pretrained=False, output_dir=output)
    # Only the exact audited development dataset and splits may reach YOLO.
    for key, val_split in (("data", "images/val"), ("retention_data", "images/val_retention")):
        data = yaml.safe_load((ROOT / recipe[key]).read_text(encoding="utf-8"))
        if ("test" in data or Path(data["path"]).resolve() != (ROOT / replay["dataset"]).resolve()
                or data.get("train") != "images/train" or data.get("val") != val_split
                or data.get("names") != {0: "Wound"}):
            raise ValueError("DATA_YAML_NOT_EXACT_AUDITED_DEVELOPMENT_SPLIT")
    checkpoint = ROOT / recipe["initialization"]
    if not checkpoint.is_file():
        raise ValueError("INITIALIZATION_MISSING")
    reference_args = yaml.safe_load((ROOT / recipe["reference_run"] / "args.yaml").read_text(encoding="utf-8"))
    if Path(reference_args["model"]).resolve() != checkpoint.resolve():
        raise ValueError("INITIALIZATION_DIFFERS_FROM_REFERENCE_ABLATION")
    # Freeze recipe, data/labels, and initialization as preflight evidence.
    return {"status": "PASS_DEVELOPMENT_PREFLIGHT_NOT_TRAINED", "recipe": recipe,
            "recipe_sha256": sha(RECIPE), "source_admission_sha256": sha(admission_file),
            "checkpoint_sha256": sha(checkpoint), "manifest": manifest,
            "test_images_used": 0, "blind_test_used": False,
            "data_yaml_sha256": {k: sha(ROOT / recipe[k]) for k in ("data", "retention_data")},
            "scope": "fixed development splits, content/pHash isolation; not patient-level or clinical authorization"}


def run_and_record(train, evaluate, output: Path) -> dict:
    """A failed evaluator cannot become PASS. Never touches legacy experiment_log."""
    output.mkdir(parents=True, exist_ok=False)
    phase = "TRAINING"
    try:
        trained = train()
        phase = "EVALUATION"
        metrics = evaluate(trained)
        if not metrics or any(not isinstance(domain, dict) or not domain or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) for v in domain.values())
                for domain in metrics.values()):
            raise ValueError("missing or nonfinite evaluation metrics")
        result = {"status": "PASS_DEVELOPMENT_ONLY", "metrics": metrics, "test_images_used": 0}
    except Exception as exc:
        result = {"status": f"FAILED_{phase}", "error_type": type(exc).__name__, "error": str(exc), "test_images_used": 0}
    write_json(output / "result.json", result)
    return result


def run_yolo(preflight_result: dict, output: Path) -> dict:
    # Import/load only after explicit --train AND full preflight; no downloads
    # are requested for initialization, which must be an existing local file.
    from ultralytics import YOLO
    from .audit_project import snapshot_sealed
    config = preflight_result["recipe"]
    initialization = ROOT / config["initialization"]
    before = snapshot_sealed()

    def train():
        if sha(initialization) != preflight_result["checkpoint_sha256"]:
            raise ValueError("checkpoint changed after preflight")
        for key in ("data", "retention_data"):
            if sha(ROOT / config[key]) != preflight_result["data_yaml_sha256"][key]:
                raise ValueError("data YAML changed after preflight")
        write_json(output / "preflight.json", preflight_result)
        model = YOLO(str(initialization))
        if model.task != "segment" or list(model.names.values()) != ["Wound"]:
            raise ValueError("wrong checkpoint task or vocabulary")
        model_config = getattr(model.model, "yaml", {})
        if model_config.get("scale") != "m":
            raise ValueError("checkpoint is not verified m-scale architecture")
        settings = {k: config[k] for k in (
            "imgsz", "batch", "epochs", "patience", "optimizer", "lr0", "warmup_bias_lr", "warmup_epochs",
            "lrf", "cos_lr", "weight_decay", "seed", "workers", "deterministic", "close_mosaic", "amp", "mask_ratio")}
        settings.update(config["augmentation"])
        settings.update({"data": str(ROOT / config["data"]), "project": str(output), "name": "training",
                         "exist_ok": False, "device": "0", "save_period": 10, "resume": False,
                         "cache": False, "fraction": 1.0, "val": True, "plots": True})
        write_json(output / "requested_train_arguments.json", settings)
        write_json(output / "model_identity.json", {"task": model.task, "names": model.names,
            "architecture": str(model_config.get("yaml_file")), "scale": model_config.get("scale"),
            "parameters": sum(p.numel() for p in model.model.parameters()), "initialization_sha256": sha(initialization)})
        model.train(**settings)
        best = output / "training/weights/best.pt"
        if not best.is_file():
            raise ValueError("best checkpoint missing")
        return best

    def evaluate(best):
        model = YOLO(str(best))
        metrics = {}
        for domain, key in (("primary", "data"), ("retention", "retention_data")):
            if sha(ROOT / config[key]) != preflight_result["data_yaml_sha256"][key]:
                raise ValueError("data YAML changed after preflight")
            result = model.val(data=str(ROOT / config[key]), split="val", imgsz=config["imgsz"],
                batch=config["batch"], conf=.001, device="0", workers=0, plots=True,
                project=str(output), name=f"validation_{domain}", exist_ok=False)
            metrics[domain] = {str(k): float(v) for k, v in result.results_dict.items()}
        write_json(output / "checkpoint.json", {"path": str(best), "sha256": sha(best)})
        if before != snapshot_sealed():
            raise ValueError("sealed artifacts changed during development run")
        return metrics

    result = run_and_record(train, evaluate, output)
    write_json(output / "sealed_integrity.json", {"unchanged": before == snapshot_sealed()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", action="store_true", help="requires complete preflight; development only")
    args = parser.parse_args()
    output = args.output.resolve()
    if (ROOT / "outputs").resolve() not in output.parents:
        raise ValueError("a NEW output directory inside workspace outputs is required")
    if output.exists():
        raise FileExistsError("refusing to overwrite output")
    try:
        result = preflight(args.source_admission.resolve(), output)
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=False)
        result = {"status": "BLOCKED_PREFLIGHT", "reason": str(exc), "training_started": False, "test_images_used": 0}
        write_json(output / "preflight.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2
    if args.train:
        result = run_yolo(result, output)
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "preflight.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
