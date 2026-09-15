"""Formal seed-42 ISIC auxiliary pretraining -> FUSeg fine-tuning run.

The smoke run must already pass before this module is allowed to start.  The
formal run reuses the hash-pinned, materialized train/val bundle from the smoke
run, never opens a test directory, and writes all outputs to a new directory.
It is intentionally a seed-42 gate; five-seed stability is a separate step and
is not started automatically.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from .audit_project import ROOT, read_json, sha, snapshot_sealed, write_json
from .isic_pretrain_smoke import finite_history
from .fuseg_warmup_experiment import update_status

SMOKE = ROOT / "outputs/isic_fuseg_pretrain_smoke_20260914"
SMOKE_RESULT = SMOKE / "result.json"
ISIC_DATA = SMOKE / "isic_dataset"
FUSEG_DATA = SMOKE / "fuseg_dataset"
INITIALIZATION = ROOT / "yolo11m-seg.pt"
INITIALIZATION_SHA256 = "eb9a06f63e2206c35d68d839b08c362429ebecf933ad54c1ad68b2fd001c17cf"
OUT_DEFAULT = ROOT / "outputs/isic_fuseg_formal_seed42_20260914"


def now():
    return datetime.now(timezone.utc).isoformat()


def out_path(value):
    path = Path(value).resolve()
    outputs = (ROOT / "outputs").resolve()
    if outputs not in path.parents:
        raise ValueError("output must be a child of this project's outputs directory")
    return path


def image_count(path: Path):
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def validate_bundle():
    if not SMOKE_RESULT.exists():
        raise ValueError("smoke result is missing; formal run is blocked")
    smoke = read_json(SMOKE_RESULT)
    if smoke.get("status") != "PASS_SMOKE_ONLY_NOT_MODEL_CANDIDATE":
        raise ValueError("smoke gate is not PASS")
    if smoke.get("test_images_used") != 0 or smoke.get("blind_test_used"):
        raise ValueError("smoke was not development-only")
    for dataset, expected in ((ISIC_DATA, {"train": 1800, "val": 200}),
                              (FUSEG_DATA, {"train": 771, "val": 191})):
        if not (dataset / "dataset.yaml").exists():
            raise ValueError(f"dataset yaml missing: {dataset}")
        for split, count in expected.items():
            actual = image_count(dataset / "images" / split)
            if actual != count:
                raise ValueError(f"{dataset.name} {split} count {actual}, expected {count}")
            # Only train/val are permitted in the formal bundle.
            if "test" in str((dataset / "images" / split).resolve()).lower():
                raise ValueError("formal dataset path contains a test component")
    if sha(INITIALIZATION) != INITIALIZATION_SHA256:
        raise ValueError("generic initialization hash changed")
    return smoke


def verify_stage_inventory(dataset: Path, rows: list, expected: dict, class_name: str):
    """Verify actual files against the pinned manifest, before loading images."""
    doc = yaml.safe_load((dataset / "dataset.yaml").read_text(encoding="utf-8-sig"))
    if (set(doc) != {"path", "train", "val", "names"}
            or Path(doc["path"]).resolve() != dataset.resolve()
            or doc["train"] != "images/train" or doc["val"] != "images/val"
            or doc["names"] != {0: class_name}):
        raise ValueError("dataset YAML changed or permits a forbidden split")
    if Counter(r["split"] for r in rows) != expected:
        raise ValueError("manifest counts or split roles changed")
    images, labels, hashes = set(), set(), {"train": set(), "val": set()}
    for row in rows:
        split = row["split"]
        image = (dataset.parent / row["image"]).resolve()
        label = (dataset / "labels" / split / (image.stem + ".txt")).resolve()
        if image.parent != (dataset / "images" / split).resolve():
            raise ValueError("image escapes the declared split")
        if "label" in row and (dataset.parent / row["label"]).resolve() != label:
            raise ValueError("label escapes the declared split")
        if image in images or label in labels:
            raise ValueError("duplicate manifest identity")
        if sha(image) != row["image_sha256"] or sha(label) != row["label_sha256"]:
            raise ValueError(f"dataset content hash changed: {image.name}")
        images.add(image)
        labels.add(label)
        hashes[split].add(row["image_sha256"])
    actual_images = {p.resolve() for p in (dataset / "images").rglob("*") if p.is_file()}
    actual_labels = {p.resolve() for p in (dataset / "labels").rglob("*.txt")}
    if images != actual_images or labels != actual_labels:
        raise ValueError("missing or unregistered dataset file")
    if hashes["train"] & hashes["val"]:
        raise ValueError("exact-content train/validation overlap")
    return {"images": len(images), "labels": len(labels), "counts": expected,
            "exact_train_val_overlap": 0, "hashes": hashes}


def verify_pinned_bundle(protocol):
    validate_bundle()
    lineage = protocol["lineage"]
    for path, key in ((SMOKE_RESULT, "smoke_result_sha256"),
                      (SMOKE / "isic_manifest.json", "isic_manifest_sha256"),
                      (SMOKE / "fuseg_manifest.json", "fuseg_manifest_sha256")):
        if sha(path) != lineage[key]:
            raise ValueError(f"pinned lineage changed: {path.name}")
    results = {}
    for name, data, expected, label in (("isic", ISIC_DATA, {"train": 1800, "val": 200}, "Foreground"),
                                       ("fuseg", FUSEG_DATA, {"train": 771, "val": 191}, "Wound")):
        results[name] = verify_stage_inventory(data, read_json(SMOKE / f"{name}_manifest.json"), expected, label)
    # Include auxiliary validation in exposure: it selects the pretraining checkpoint.
    isic_exposure = set.union(*results["isic"]["hashes"].values())
    if isic_exposure & results["fuseg"]["hashes"]["val"]:
        raise ValueError("auxiliary exposure overlaps wound validation")
    for result in results.values():
        result.pop("hashes")
    return results


def prepare(out: Path):
    if out.exists():
        raise FileExistsError("refusing to overwrite a formal run directory")
    smoke = validate_bundle()
    before = snapshot_sealed()
    out.mkdir(parents=True)
    protocol = {
        "experiment_id": "D-Formal-ISIC-FUSeg-seed42-20260914",
        "created_utc": now(),
        "role": "noncommercial_offline_development_only",
        "formal_seed": 42,
        "test_images_used": 0,
        "blind_test_used": False,
        "stages": [
            {"name": "ISIC_auxiliary_pretraining", "source": "ISIC2017",
             "role": "auxiliary_pretraining", "train_images": 1800,
             "validation_images": 200, "epochs": 100, "patience": 20,
             "dataset_yaml": str(ISIC_DATA / "dataset.yaml")},
            {"name": "FUSeg_wound_finetuning", "source": "FUSeg",
             "role": "wound_finetuning", "train_images": 771,
             "validation_images": 191, "epochs": 300, "patience": 80,
             "initialization": "ISIC formal best.pt",
             "dataset_yaml": str(FUSEG_DATA / "dataset.yaml")},
        ],
        "training": {"imgsz": 768, "batch": 4, "optimizer": "AdamW", "lr0": .0005,
                     "warmup_bias_lr": 0.0, "warmup_epochs": 5, "lrf": .01,
                     "cos_lr": True, "close_mosaic": 30, "seed": 42,
                     "deterministic": True, "workers": 2, "amp": True,
                     "mosaic": .1, "mixup": 0.0, "copy_paste": 0.0,
                     "degrees": 5.0, "translate": .05, "scale": .2,
                     "shear": .5, "perspective": .0002, "fliplr": .5,
                     "flipud": 0.0, "hsv_h": .01, "hsv_s": .4,
                     "hsv_v": .25, "device": "0", "val": True, "plots": True},
        "gates": ["smoke PASS", "train/val counts and hashes pinned", "finite histories",
                  "ISIC best.pt and FUSeg best.pt", "validation only", "test_images_used=0",
                  "sealed snapshot unchanged", "no App model replacement",
                  "seed-42 localization/development gate required before five seeds"],
        "lineage": {"smoke_result": str(SMOKE_RESULT), "smoke_result_sha256": sha(SMOKE_RESULT),
                     "isic_manifest_sha256": sha(SMOKE / "isic_manifest.json"),
                     "fuseg_manifest_sha256": sha(SMOKE / "fuseg_manifest.json"),
                     "initialization": str(INITIALIZATION),
                     "initialization_sha256": INITIALIZATION_SHA256},
        "sealed_before": before,
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "sealed_before.json", before)
    print(json.dumps({"status": "FORMAL_PREPARED_NOT_EXECUTED", "output": str(out),
                      "isic": [1800, 200, 100, 20], "fuseg": [771, 191, 300, 80],
                      "test_images_used": 0}), flush=True)


def stage_args(project: Path, name: str, data: Path, epochs: int, patience: int):
    return dict(data=str(data), epochs=epochs, patience=patience, imgsz=768, batch=4,
                optimizer="AdamW", lr0=.0005, lrf=.01, cos_lr=True, warmup_epochs=5,
                warmup_bias_lr=0.0, close_mosaic=30, save_period=-1, seed=42,
                deterministic=True, workers=2, amp=True, mosaic=.1, mixup=0.0,
                copy_paste=0.0, degrees=5.0, translate=.05, scale=.2, shear=.5,
                perspective=.0002, fliplr=.5, flipud=0.0, hsv_h=.01, hsv_s=.4,
                hsv_v=.25, project=str(project), name=name, exist_ok=False,
                device="0", val=True, plots=True, verbose=False)


def finite_and_epochs(run: Path, expected_max: int):
    rows = finite_history(run)
    # Ultralytics results.csv is already one-based. Do not truncate fractions.
    epochs = [float(row["epoch"]) for row in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise RuntimeError(f"non-contiguous epochs in {run}")
    if len(rows) > expected_max:
        raise RuntimeError(f"unexpected epoch count in {run}: {len(rows)}")
    return rows


def recovery_evidence(out: Path, protocol: dict):
    """Accept only the known post-ISIC bookkeeping failure, without retraining it."""
    failure = read_json(out / "failure.json")
    run = out / "runs/isic_auxiliary_formal"
    expected_error = repr(RuntimeError(f"non-contiguous epochs in {run}"))
    if failure.get("status") != "FAIL_FORMAL_SEED42" or failure.get("error") != expected_error:
        raise ValueError("recovery supports only the identified epoch bookkeeping error")
    if (out / "runs/fuseg_finetune_formal").exists() or (out / "result.json").exists():
        raise FileExistsError("FUSeg has already started or formal result exists")
    if sha(out / "protocol.json") != read_json(out / "execution.lock")["protocol_sha256"]:
        raise ValueError("original protocol changed after execution")
    rows = finite_and_epochs(run, 100)
    saved_args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8-sig"))
    for key, value in stage_args(out / "runs", "isic_auxiliary_formal", ISIC_DATA / "dataset.yaml", 100, 20).items():
        if saved_args.get(key) != value:
            raise ValueError(f"completed ISIC settings differ: {key}")
    best_path, last_path = run / "weights/best.pt", run / "weights/last.pt"
    if (sha(best_path) != "1ec926026473beafafb1ef8da6aa6efcc13d38c391ec8db48006c5d793d5f2a3"
            or sha(last_path) != "dc6586620be034b285325b9a5792a04875d2ae8ca980a85ca5b3d65e67d178f9"):
        raise ValueError("audited completed ISIC checkpoints changed")
    # These are trusted local outputs of this run; never load arbitrary downloads.
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    if last.get("epoch") != -1 or last.get("optimizer") is not None:
        raise ValueError("ISIC checkpoint is not finalized")
    saved_history = last.get("train_results", {})
    if saved_history.get("epoch") != [float(row["epoch"]) for row in rows]:
        raise ValueError("checkpoint/history epoch evidence disagrees")
    fitness = [sum(.1 * float(r[f"metrics/mAP50({kind})"]) +
                   .9 * float(r[f"metrics/mAP50-95({kind})"]) for kind in ("B", "M")) for r in rows]
    best_epoch = max(range(len(fitness)), key=fitness.__getitem__) + 1
    if len(rows) < 100 and len(rows) - best_epoch < 20:
        raise ValueError("neither full schedule nor patience termination is evidenced")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    checkpoint_fitness = best.get("train_metrics", {}).get("fitness")
    if checkpoint_fitness is None or not math.isclose(float(checkpoint_fitness), max(fitness), abs_tol=2e-5):
        raise ValueError("best checkpoint does not match the best recorded fitness")
    return {"isic_epochs_completed": len(rows), "best_epoch_from_csv": best_epoch,
            "isic_best_sha256": sha(best_path), "isic_last_sha256": sha(last_path),
            "history_sha256": sha(run / "results.csv"), "failure_sha256": sha(out / "failure.json"),
            "isic_retrained": False, "protocol_sha256": sha(out / "protocol.json")}


def progress_callback(out: Path, stage: str):
    def update(trainer):
        losses = trainer.tloss.detach().cpu().numpy()
        if not np.isfinite(losses).all():
            raise RuntimeError(f"non-finite loss in {stage}")
        update_status(out, status="TRAINING", stage=stage, epoch_completed=trainer.epoch + 1,
                      epochs_max=trainer.epochs, metrics={k: float(v) for k, v in trainer.metrics.items()})
    return update


def execute(out: Path, *, continue_fuseg=False):
    protocol = read_json(out / "protocol.json")
    before = read_json(out / "sealed_before.json")
    if protocol.get("test_images_used") != 0 or protocol.get("blind_test_used"):
        raise ValueError("formal protocol is not development-only")
    if snapshot_sealed() != before:
        raise ValueError("pre-run sealed gate failed")
    inventory = verify_pinned_bundle(protocol)
    if not torch.cuda.is_available():
        raise RuntimeError("declared CUDA device unavailable")
    recovery = recovery_evidence(out, protocol) if continue_fuseg else None
    lock = out / ("continue_fuseg.lock" if continue_fuseg else "execution.lock")
    if lock.exists():
        raise FileExistsError("formal execution lock already exists; refusing a second run")
    from .isic_fuseg_gate import prepare_gate_inputs
    gate_inputs = prepare_gate_inputs()
    write_json(lock, {"started_utc": now(), "protocol_sha256": sha(out / "protocol.json"),
                      "runner_sha256": sha(Path(__file__)), "inventory": inventory, "recovery": recovery})
    write_json(out / "development_gate_inputs.json", gate_inputs)
    runs = out / "runs"
    isic_run = runs / "isic_auxiliary_formal"
    fuseg_run = runs / "fuseg_finetune_formal"
    try:
        if not continue_fuseg:
            isic = YOLO(str(INITIALIZATION))
            isic.add_callback("on_fit_epoch_end", progress_callback(out, "ISIC"))
            isic.train(**stage_args(runs, "isic_auxiliary_formal", ISIC_DATA / "dataset.yaml", 100, 20))
        isic_best = isic_run / "weights/best.pt"
        if not isic_best.exists():
            raise RuntimeError("ISIC formal best.pt missing")
        isic_history = finite_and_epochs(isic_run, 100)
        if snapshot_sealed() != before:
            raise RuntimeError("sealed file changed before FUSeg")
        update_status(out, status="STARTING_FUSEG", isic_epochs_completed=len(isic_history),
                      isic_checkpoint_sha256=sha(isic_best), epoch_completed=0, epochs_max=300)
        # FUSeg may only initialize from the formal ISIC checkpoint.
        wound = YOLO(str(isic_best))
        wound.add_callback("on_fit_epoch_end", progress_callback(out, "FUSeg"))
        wound.train(**stage_args(runs, "fuseg_finetune_formal", FUSEG_DATA / "dataset.yaml", 300, 80))
        fuseg_best = fuseg_run / "weights/best.pt"
        if not fuseg_best.exists():
            raise RuntimeError("FUSeg formal best.pt missing")
        fuseg_history = finite_and_epochs(fuseg_run, 300)
        # Explicit validation is restricted to the materialized val splits.
        isic_val = YOLO(str(isic_best)).val(data=str(ISIC_DATA / "dataset.yaml"), split="val",
                                            imgsz=768, batch=4, device="0", plots=False,
                                            verbose=False, project=str(runs),
                                            name="isic_auxiliary_formal_val", exist_ok=False)
        fuseg_val = YOLO(str(fuseg_best)).val(data=str(FUSEG_DATA / "dataset.yaml"), split="val",
                                              imgsz=768, batch=4, device="0", plots=False,
                                              verbose=False, project=str(runs),
                                              name="fuseg_finetune_formal_val", exist_ok=False)
        if snapshot_sealed() != before:
            raise RuntimeError("sealed file changed during formal run")
        result = {
            "status": "PASS_FORMAL_SEED42_NEEDS_DEVELOPMENT_GATE",
            "finished_utc": now(), "test_images_used": 0, "blind_test_used": False,
            "isic": {"train_images": 1800, "val_images": 200, "epochs_completed": len(isic_history),
                     "checkpoint": str(isic_best), "checkpoint_sha256": sha(isic_best),
                     "val_box_mAP50": float(isic_val.box.map50), "val_mask_mAP50": float(isic_val.seg.map50)},
            "fuseg": {"train_images": 771, "val_images": 191, "epochs_completed": len(fuseg_history),
                      "checkpoint": str(fuseg_best), "checkpoint_sha256": sha(fuseg_best),
                      "val_box_mAP50": float(fuseg_val.box.map50), "val_box_mAP50_95": float(fuseg_val.box.map),
                      "val_mask_mAP50": float(fuseg_val.seg.map50), "val_mask_mAP50_95": float(fuseg_val.seg.map),
                      "val_box_precision": float(fuseg_val.box.mp), "val_box_recall": float(fuseg_val.box.mr),
                      "val_mask_precision": float(fuseg_val.seg.mp), "val_mask_recall": float(fuseg_val.seg.mr)},
            "integrity": {"sealed_unchanged": True, "app_model_replaced": False,
                          "test_images_used": 0, "blind_test_used": False,
                          "five_seed_run_started": False,
                          "candidate_gate": "pending localization/development acceptance"},
        }
        write_json(out / "result.json", result)
        (out / "實驗結果報告.md").write_text(
            "# Formal seed-42 ISIC auxiliary pretraining → FUSeg fine-tuning\n\n"
            "ISIC 1,800/200 以 100 epochs（patience 20）訓練，接續 FUSeg 771/191 以 300 epochs（patience 80）訓練。\n\n"
            f"- FUSeg validation Box mAP50：{result['fuseg']['val_box_mAP50']:.2%}\n"
            f"- FUSeg validation Mask mAP50：{result['fuseg']['val_mask_mAP50']:.2%}\n"
            f"- FUSeg validation Mask mAP50-95：{result['fuseg']['val_mask_mAP50_95']:.2%}\n"
            "- 這是 seed-42 development 結果；尚未通過定位/開發門檻前，不啟動 5 seeds。\n"
            "- 僅使用 train/val；test_images_used=0；未替換 App 模型；封版檔案未變更。\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        update_status(out, status=result["status"], stage="DEVELOPMENT_GATE_PENDING")
        from .isic_fuseg_gate import execute_gate
        execute_gate(out)
    except Exception as exc:
        update_status(out, status="FAILED", error=repr(exc))
        write_json(out / ("continue_fuseg_failure.json" if continue_fuseg else "failure.json"), {"status": "FAIL_FORMAL_SEED42", "error": repr(exc),
                                           "time": now(), "test_images_used": 0,
                                           "sealed_unchanged": snapshot_sealed() == before})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "execute", "continue-fuseg"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(out_path(args.output))
    else:
        execute(out_path(args.output), continue_fuseg=args.mode == "continue-fuseg")
