"""Prepare and execute a bounded FUSeg-only noncommercial development revision.

--prepare creates a new copied dataset/manifest; --execute runs a 2-epoch
infrastructure smoke, then one 300-epoch-max development run, then validation.
No test mode, no automatic restart, no deployment or shared logger mutations.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .audit_project import ROOT, audit_development_dataset, read_json, rel, sha, snapshot_sealed, write_json
from .guards import validate_new_development_run, validate_optimizer_settings

DATA = ROOT / "yolo_dataset_fuseg_seg_v2"
REFERENCE = ROOT / "experiments/results/segmentation/D-Seg-03_yolo11m_fuseg_only_s42"
INITIALIZATION = ROOT / "yolo11m-seg.pt"
INITIALIZATION_SHA256 = "eb9a06f63e2206c35d68d839b08c362429ebecf933ad54c1ad68b2fd001c17cf"
PDF = ROOT / "official_detection_sources_20260812/fuseg/repository/data/Foot Ulcer Segmentation Challenge/FootUlcerSegmentationChallenge2021.pdf"
PDF_SHA256 = "f9a44fc14bc7589d03ce5c17307ad97e9dde5fcc0e2c596c8b86010a6d9547aa"
EVIDENCE = ROOT / "experiments/review_v2/evidence/FUSeg_noncommercial_research_20260914.md"
SOURCE_MANIFEST = ROOT / "outputs/fuseg_only_manifest_20260821.csv"
TRAIN_KEYS = (
    "imgsz", "batch", "epochs", "patience", "optimizer", "lr0", "warmup_bias_lr", "warmup_epochs",
    "warmup_momentum", "lrf", "cos_lr", "weight_decay", "momentum", "seed", "workers", "deterministic",
    "close_mosaic", "amp", "mask_ratio", "overlap_mask", "nbs", "box", "cls", "dfl", "dropout",
    "mosaic", "mixup", "copy_paste", "copy_paste_mode", "degrees", "translate", "scale", "shear",
    "perspective", "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "bgr", "multi_scale", "rect",
    "single_cls", "freeze", "fraction", "pretrained", "iou", "conf", "max_det", "retina_masks",
    "save_period", "cache", "val", "plots", "verbose")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def update_status(bundle: Path, **fields):
    destination = bundle / "status.json"
    state = read_json(destination) if destination.exists() else {}
    state.update(fields)
    state.update({"updated_utc": timestamp(), "pid": os.getpid(), "test_images_used": 0,
                  "scope": "noncommercial offline development only; no deployment"})
    temp = bundle / "status.json.tmp"
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(destination)


def train_arguments(reference: dict, *, smoke=False) -> dict:
    settings = {k: reference[k] for k in TRAIN_KEYS if k in reference}
    settings["warmup_bias_lr"] = 0.0
    validate_optimizer_settings(settings)
    if smoke:
        settings.update(epochs=2, fraction=.05, save_period=-1, plots=False)
    return settings


def check_dataset_yaml(doc: dict, expected_root: Path):
    if ("test" in doc or Path(doc["path"]).resolve() != expected_root.resolve()
            or doc.get("train") != "images/train" or doc.get("val") != "images/val"
            or doc.get("names") != {0: "Wound"}):
        raise ValueError("dataset is not the exact FUSeg-only development split")


def verify_inventory(bundle: Path, manifest: list[dict]):
    expected_images, expected_labels = set(), set()
    for row in manifest:
        image, label = bundle / row["image"], bundle / row["label"]
        if row["split"] not in {"train", "val"}:
            raise ValueError("forbidden split in manifest")
        for path in (image, label):
            if (bundle / "dataset").resolve() not in path.resolve().parents:
                raise ValueError("manifest path escapes copied development dataset")
        if sha(image) != row["image_sha256"] or sha(label) != row["label_sha256"]:
            raise ValueError(f"snapshot changed: {row['image']}")
        expected_images.add(image.resolve())
        expected_labels.add(label.resolve())
    actual_images = {p.resolve() for p in (bundle / "dataset/images").rglob("*") if p.is_file()}
    actual_labels = {p.resolve() for p in (bundle / "dataset/labels").rglob("*.txt")}
    if actual_images != expected_images or actual_labels != expected_labels:
        raise ValueError("snapshot contains missing/unregistered images or labels")


def prepare(bundle: Path):
    if bundle.exists():
        raise FileExistsError("bundle exists; refusing to overwrite")
    if sha(PDF) != PDF_SHA256 or sha(INITIALIZATION) != INITIALIZATION_SHA256:
        raise ValueError("pinned source evidence or generic checkpoint changed")
    reference = yaml.safe_load((REFERENCE / "args.yaml").read_text(encoding="utf-8"))
    if (Path(reference["model"]).resolve() != INITIALIZATION.resolve()
            or Path(reference["data"]).resolve() != (DATA / "dataset.yaml").resolve()):
        raise ValueError("historical reference is not the same generic initialization/dataset")
    check_dataset_yaml(yaml.safe_load((DATA / "dataset.yaml").read_text(encoding="utf-8")), DATA)
    audit = audit_development_dataset(DATA.name, ("train", "val"), {})
    if audit["status"] != "PASS_TECHNICAL_ONLY" or (audit["splits"]["train"]["image_files"], audit["splits"]["val"]["image_files"]) != (771, 191):
        raise ValueError("current data audit/count differs from frozen FUSeg cohort")
    with SOURCE_MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        upstream = list(csv.DictReader(handle))
    lookup = {str(Path(r["fuseg_image"]).as_posix()): r for r in upstream if r.get("fuseg_image")}
    for split, info in audit["splits"].items():
        for row in info["rows"]:
            key = (Path(DATA.name) / row["image"]).as_posix()
            original = lookup[key]
            if original["source"] != "FUSeg" or original["assigned_split"] != split or original["official_split"] not in {"train", "val", "validation"}:
                raise ValueError("non-FUSeg or test source encountered")
            if original["image_sha256"].lower() != row["image_sha256"] or sha(ROOT / original["source_mask"]) != original["mask_file_sha256"].lower():
                raise ValueError("source manifest image/mask hash mismatch")
    settings = train_arguments(reference)
    gates = {"FUSeg": {"development_allowed": True, "evidence_file": str(EVIDENCE)}}
    validate_new_development_run({"role": "development", "test_images_used": 0, "sources": ["FUSeg"]}, gates,
        train_groups={r["image_sha256"] for r in audit["splits"]["train"]["rows"]},
        val_groups={r["image_sha256"] for r in audit["splits"]["val"]["rows"]},
        ancestor_train_groups=None, generic_pretrained=True, output_dir=bundle)
    bundle.mkdir(parents=True, exist_ok=False)
    write_json(bundle / "sealed_before.json", snapshot_sealed())
    write_json(bundle / "data_audit.json", audit)
    manifest = []
    for split, info in audit["splits"].items():
        for row in info["rows"]:
            source_image = DATA / row["image"]
            source_label = DATA / "labels" / split / (source_image.stem + ".txt")
            destination_image = bundle / "dataset" / row["image"]
            destination_label = bundle / "dataset/labels" / split / source_label.name
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, destination_image)
            shutil.copy2(source_label, destination_label)
            manifest.append({"split": split, "source": "FUSeg", "image_id": source_image.name,
                "image": destination_image.relative_to(bundle).as_posix(), "label": destination_label.relative_to(bundle).as_posix(),
                "group_id": row["image_sha256"], "image_sha256": row["image_sha256"],
                "label_sha256": row["label_sha256"], "phash": row["phash"], "instances": row["instances"]})
    write_json(bundle / "manifest.json", manifest)
    with (bundle / "dataset/dataset.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump({"path": str(bundle / "dataset"), "train": "images/train", "val": "images/val", "nc": 1, "names": {0: "Wound"}}, handle, allow_unicode=True, sort_keys=False)
    verify_inventory(bundle, manifest)
    package_versions = {p: importlib.metadata.version(p) for p in ("ultralytics", "torch", "torchvision", "numpy", "scipy", "Pillow")}
    protocol = {"experiment_id": "D-Seg-03R_FUSeg_warmup_zero", "role": "noncommercial_offline_development",
        "created_utc": timestamp(), "source_gate": {"FUSeg": {"development_allowed": True,
        "evidence_file": str(EVIDENCE), "evidence_sha256": sha(EVIDENCE), "publisher_pdf_sha256": PDF_SHA256,
        "license_as_printed": "CC BY NC", "license_version": None, "commercial_use_authorized": False,
        "clinical_deployment_authorized": False}},
        "train_images": 771, "validation_images": 191, "test_images_used": 0,
        "excluded_sources": ["Yasin", "clinical", "WSNet", "AZH_patches", "BUBT", "Redscar", "CO2Wounds", "classification_blind_test"],
        "model": "YOLO11m-seg", "initialization": str(INITIALIZATION), "initialization_sha256": INITIALIZATION_SHA256,
        "manifest_sha256": sha(bundle / "manifest.json"), "source_manifest_sha256": sha(SOURCE_MANIFEST),
        "dataset_yaml_sha256": sha(bundle / "dataset/dataset.yaml"), "reference_args_sha256": sha(REFERENCE / "args.yaml"),
        "train_arguments": settings, "versions": package_versions,
        "changed_hyperparameters": {"warmup_bias_lr": {"reference": reference["warmup_bias_lr"], "revision": 0.0}},
        "reference": str(REFERENCE), "comparison_limitations": "matched configuration and fixed cohort; historical initial checkpoint hash/software state not fully archived, so not a prospective paired causal trial",
        "selection": "Ultralytics best checkpoint on development fitness; final same-val metrics remain development, not test",
        "future_evaluation": "automatic development validation only; no external or test unlock",
        "runner_sha256": sha(Path(__file__)), "guard_sha256": sha(ROOT / "experiments/review_v2/guards.py")}
    write_json(bundle / "protocol.json", protocol)
    update_status(bundle, status="PREFLIGHT_PASSED_NOT_TRAINED", train_images=771, validation_images=191,
                  max_epochs=settings["epochs"], patience=settings["patience"], latest_completed_epoch=0)
    print(json.dumps({"status": "PREFLIGHT_PASS", "bundle": str(bundle), "counts": [771, 191, 0], "changes": protocol["changed_hyperparameters"]}, ensure_ascii=False, indent=2), flush=True)


def validate_finite_metrics(metrics):
    if not metrics or any(not math.isfinite(float(v)) for v in metrics.values()):
        raise ValueError("empty/nonfinite validation metrics")


def execute(bundle: Path):
    protocol = read_json(bundle / "protocol.json")
    if protocol["role"] != "noncommercial_offline_development" or protocol["test_images_used"] != 0:
        raise ValueError("invalid experiment role")
    for path, expected in [(INITIALIZATION, protocol["initialization_sha256"]), (PDF, PDF_SHA256),
        (EVIDENCE, protocol["source_gate"]["FUSeg"]["evidence_sha256"]),
        (bundle / "manifest.json", protocol["manifest_sha256"]),
        (bundle / "dataset/dataset.yaml", protocol["dataset_yaml_sha256"]),
        (Path(__file__), protocol["runner_sha256"]),
        (ROOT / "experiments/review_v2/guards.py", protocol["guard_sha256"])]:
        if sha(path) != expected:
            raise ValueError(f"preflight fingerprint changed: {path.name}")
    if read_json(bundle / "sealed_before.json") != snapshot_sealed():
        raise ValueError("sealed files changed after prepare")
    manifest = read_json(bundle / "manifest.json")
    verify_inventory(bundle, manifest)
    with (bundle / "execution.lock").open("x", encoding="utf-8") as lock:
        lock.write(json.dumps({"pid": os.getpid(), "started_utc": timestamp()}))
    phase = "INITIALIZATION"
    try:
        import torch
        from ultralytics import YOLO
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no silent CPU fallback")
        versions = {p: importlib.metadata.version(p) for p in protocol["versions"]}
        if versions != protocol["versions"]:
            raise RuntimeError("software environment changed since preflight")
        reference = yaml.safe_load((REFERENCE / "args.yaml").read_text(encoding="utf-8"))
        if sha(REFERENCE / "args.yaml") != protocol["reference_args_sha256"]:
            raise ValueError("historical reference changed")
        for smoke in (True, False):
            phase = "SMOKE" if smoke else "FORMAL_TRAINING"
            run_name = "smoke" if smoke else "formal"
            run_dir = bundle / run_name
            if run_dir.exists():
                raise FileExistsError("refusing to overwrite existing model run")
            update_status(bundle, status=phase, latest_completed_epoch=0, current_run=run_name)
            model = YOLO(str(INITIALIZATION))
            if model.task != "segment" or model.model.yaml.get("scale") != "m" or len(model.names) != 80:
                raise ValueError("generic initialization is not the expected COCO m-scale segmenter")
            if not smoke:
                write_json(bundle / "model_identity.json", {"task": model.task, "scale": model.model.yaml.get("scale"),
                    "yaml_file": model.model.yaml.get("yaml_file"), "pretrained_classes": len(model.names),
                    "pretrained_parameters": sum(p.numel() for p in model.model.parameters()), "checkpoint_sha256": sha(INITIALIZATION)})

            def on_fit_epoch_end(trainer):
                epoch = int(trainer.epoch) + 1
                learning_rates = [float(p["lr"]) for p in trainer.optimizer.param_groups]
                if any(not math.isfinite(v) or v > protocol["train_arguments"]["lr0"] * 1.001 for v in learning_rates):
                    raise ValueError("optimizer learning rate exceeded corrected bound")
                metrics = {str(k): float(v) for k, v in trainer.metrics.items()}
                validate_finite_metrics(metrics)
                update_status(bundle, status=phase, latest_completed_epoch=epoch, current_run=run_name,
                    learning_rates=learning_rates, latest_development_metrics=metrics,
                    gpu_memory_allocated_mb=round(torch.cuda.memory_allocated()/1024**2, 1))

            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            settings = train_arguments(reference, smoke=smoke)
            settings.update(data=str(bundle / "dataset/dataset.yaml"), project=str(bundle), name=run_name,
                            exist_ok=False, resume=False, device="0")
            write_json(bundle / f"{run_name}_requested_arguments.json", settings)
            model.train(**settings)
            best = run_dir / "weights/best.pt"
            if not best.is_file():
                raise RuntimeError("training returned without best checkpoint")
            if smoke:
                write_json(bundle / "smoke_result.json", {"status": "PASS_INFRASTRUCTURE_ONLY", "epochs": 2,
                    "fraction": .05, "not_scientific_performance": True, "test_images_used": 0})
                del model
                torch.cuda.empty_cache()
                verify_inventory(bundle, manifest)
                continue
            phase = "DEVELOPMENT_EVALUATION"
            update_status(bundle, status=phase, best_checkpoint=str(best))
            del model
            torch.cuda.empty_cache()
            final_model = YOLO(str(best))
            if list(final_model.names.values()) != ["Wound"]:
                raise ValueError("trained class mapping is not Wound")
            result = final_model.val(data=str(bundle / "dataset/dataset.yaml"), split="val",
                imgsz=settings["imgsz"], batch=settings["batch"], device="0", workers=0,
                iou=reference["iou"], max_det=reference["max_det"], plots=True,
                project=str(bundle), name="development_validation", exist_ok=False)
            metrics = {str(k): float(v) for k, v in result.results_dict.items()}
            validate_finite_metrics(metrics)
            verify_inventory(bundle, manifest)
            sealed_unchanged = read_json(bundle / "sealed_before.json") == snapshot_sealed()
            if not sealed_unchanged:
                raise ValueError("sealed artifacts changed during training")
            write_json(bundle / "result.json", {"status": "PASS_DEVELOPMENT_ONLY_NOT_DEPLOYMENT",
                "metrics": metrics, "checkpoint": str(best), "checkpoint_sha256": sha(best),
                "train_images": 771, "validation_images": 191, "test_images_used": 0,
                "speed": result.speed, "evaluation_role": "repeated internal development validation",
                "sealed_unchanged": sealed_unchanged, "scope": protocol["role"],
                "commercial_use_authorized": False, "clinical_deployment_authorized": False})
            baseline = read_json(ROOT / "outputs/segmentation/D-Seg-03_yolo11m_fuseg_only_s42_summary.json")["metrics"]
            rows = [("Mask mAP50", "metrics/mAP50(M)", "mask_mAP50"), ("Mask mAP50-95", "metrics/mAP50-95(M)", "mask_mAP50_95"),
                    ("Mask Precision", "metrics/precision(M)", "mask_precision"), ("Mask Recall", "metrics/recall(M)", "mask_recall")]
            text = ["# FUSeg 暖身學習率修正版結果", "", "非商業離線研究；771 train／191 val；test=0。非部署許可。", "",
                    "| 指標 | 歷史 D-Seg-03 | 本次修正版 | 差異（百分點） |", "|---|---:|---:|---:|"]
            for title, new_key, old_key in rows:
                text.append(f"| {title} | {100*baseline[old_key]:.2f}% | {100*metrics[new_key]:.2f}% | {100*(metrics[new_key]-baseline[old_key]):+.2f} |")
            text += ["", "無论是否提高，都保留結果。不根據此驗證結果重跑 test 或部署。", "比較限制：" + protocol["comparison_limitations"], ""]
            with (bundle / "訓練完成報告.md").open("x", encoding="utf-8") as f:
                f.write("\n".join(text))
            update_status(bundle, status="COMPLETED_DEVELOPMENT_ONLY", result_file=str(bundle / "result.json"))
        return 0
    except BaseException as exc:
        update_status(bundle, status=f"FAILED_{phase}", error_type=type(exc).__name__, error=str(exc))
        write_json(bundle / "failure.json", {"phase": phase, "error": str(exc), "traceback": traceback.format_exc(), "test_images_used": 0})
        traceback.print_exc()
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if (ROOT / "outputs").resolve() not in bundle.parents:
        raise ValueError("bundle must be inside workspace outputs")
    if args.prepare:
        prepare(bundle)
        return 0
    return execute(bundle)


if __name__ == "__main__":
    raise SystemExit(main())
