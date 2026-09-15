"""Two-stage full-data smoke for ISIC auxiliary pretraining -> FUSeg fine-tuning.

This script creates a new, hash-pinned bundle and runs exactly two epochs for
each stage. It never opens a test directory and never changes sealed artifacts.
The smoke checkpoints are process evidence only; they are not App candidates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from .audit_project import ROOT, read_json, sha, snapshot_sealed, write_json
from .fuseg_warmup_experiment import verify_inventory
from .localization_benchmark import BUNDLE

ISIC = ROOT / "official_segmentation_sources_20260821/isic2017"
ISIC_IMAGES = ISIC / "training_images"
ISIC_MASKS = ISIC / "ground_truth"
ISIC_AUDIT = ROOT / "outputs/isic2017_source_audit_20260821.json"
FUSEG_MANIFEST = ROOT / "outputs/fuseg_warmup_revision_20260914/manifest.json"
INITIALIZATION = ROOT / "yolo11m-seg.pt"
INITIALIZATION_SHA256 = "eb9a06f63e2206c35d68d839b08c362429ebecf933ad54c1ad68b2fd001c17cf"
FUSEG_PROTOCOL = BUNDLE / "protocol.json"
OUT_DEFAULT = ROOT / "outputs/isic_fuseg_pretrain_smoke_20260914"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def now():
    return datetime.now(timezone.utc).isoformat()


def output_path(value):
    path = Path(value).resolve()
    if (ROOT / "outputs").resolve() not in path.parents:
        raise ValueError("output must be a child of this project's outputs directory")
    return path


def copy_or_link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def find_isic_masks():
    masks = {}
    for p in ISIC_MASKS.rglob("*_segmentation.png"):
        key = p.name.removesuffix("_segmentation.png")
        if key in masks:
            raise ValueError(f"duplicate ISIC mask identity: {key}")
        masks[key] = p
    return masks


def polygon_lines(mask: np.ndarray):
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary.shape
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, max(.5, .002 * perimeter), True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        coords = " ".join(f"{min(1,max(0,float(x)/w)):.8f} {min(1,max(0,float(y)/h)):.8f}" for x,y in approx)
        lines.append(f"0 {coords}")
    return lines


def materialize_isic(out: Path):
    images = sorted(ISIC_IMAGES.glob("*.jpg"), key=lambda p: p.stem)
    masks = find_isic_masks()
    if len(images) != 2000 or len(masks) != 2000:
        raise ValueError(f"ISIC pair count must be 2000, got images={len(images)} masks={len(masks)}")
    rows, link_counts = [], Counter()
    for index, image in enumerate(images):
        if image.stem not in masks:
            raise ValueError(f"unmatched ISIC image: {image.name}")
        mask_path = masks[image.stem]
        with Image.open(image) as im:
            size = im.size
        with Image.open(mask_path) as im:
            mask = np.asarray(im.convert("L"))
        if mask.shape[::-1] != size:
            raise ValueError(f"image/mask dimension mismatch: {image.name}")
        split = "train" if index < 1800 else "val"
        out_image = out / "isic_dataset/images" / split / image.name
        out_label = out / "isic_dataset/labels" / split / f"{image.stem}.txt"
        link_counts[copy_or_link(image, out_image)] += 1
        out_label.parent.mkdir(parents=True, exist_ok=True)
        out_label.write_text("\n".join(polygon_lines(mask)) + ("\n" if polygon_lines(mask) else ""), encoding="utf-8")
        rows.append({"source": "ISIC2017", "role": "auxiliary_pretraining", "split": split,
                     "image_id": image.stem, "image": str(out_image.relative_to(out)).replace("\\", "/"),
                     "mask": str(mask_path.relative_to(ROOT)).replace("\\", "/"),
                     "image_sha256": sha(image), "mask_sha256": sha(mask_path),
                     "label_sha256": sha(out_label), "dimensions": list(size),
                     "instances": len(polygon_lines(mask)), "foreground_pixels": int((mask > 0).sum())})
    yaml_data = {"path": str((out / "isic_dataset").resolve()), "train": "images/train", "val": "images/val", "names": {0: "Foreground"}}
    (out / "isic_dataset/dataset.yaml").write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")
    return rows, link_counts


def materialize_fuseg(out: Path):
    manifest = read_json(BUNDLE / "manifest.json")
    verify_inventory(BUNDLE, manifest)
    rows, link_counts = [], Counter()
    for row in manifest:
        split = row["split"]
        src_image, src_label = BUNDLE / row["image"], BUNDLE / row["label"]
        dst_image = out / "fuseg_dataset/images" / split / Path(row["image"]).name
        dst_label = out / "fuseg_dataset/labels" / split / Path(row["label"]).name
        link_counts[copy_or_link(src_image, dst_image)] += 1
        link_counts[copy_or_link(src_label, dst_label)] += 1
        if sha(dst_image) != row["image_sha256"] or sha(dst_label) != row["label_sha256"]:
            raise ValueError(f"FUSeg copied identity changed: {row['image_id']}")
        rows.append({"source": "FUSeg", "role": "wound_finetuning", "split": split,
                     "image_id": row["image_id"], "image": str(dst_image.relative_to(out)).replace("\\", "/"),
                     "label": str(dst_label.relative_to(out)).replace("\\", "/"),
                     "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"],
                     "instances": row["instances"]})
    yaml_data = {"path": str((out / "fuseg_dataset").resolve()), "train": "images/train", "val": "images/val", "names": {0: "Wound"}}
    (out / "fuseg_dataset/dataset.yaml").write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")
    return rows, link_counts


def prepare(out: Path):
    if out.exists():
        raise FileExistsError("refusing to overwrite a smoke bundle")
    if sha(INITIALIZATION) != INITIALIZATION_SHA256:
        raise ValueError("generic YOLO11m-seg initialization changed")
    if read_json(ISIC_AUDIT).get("status") != "PASS_ISIC2017_OFFICIAL_SOURCE_AUDIT":
        raise ValueError("ISIC official source audit is not PASS")
    if read_json(ISIC_AUDIT).get("paired_ids") != 2000:
        raise ValueError("ISIC pair audit is incomplete")
    previous = read_json(FUSEG_PROTOCOL)
    if previous.get("test_images_used") != 0 or previous.get("model") != "YOLO11m-seg":
        raise ValueError("FUSeg frozen protocol is not development-only")
    out.mkdir(parents=True)
    before = snapshot_sealed()
    isic_rows, isic_links = materialize_isic(out)
    fuseg_rows, fuseg_links = materialize_fuseg(out)
    if Counter(r["split"] for r in isic_rows) != {"train": 1800, "val": 200}:
        raise ValueError("ISIC split count mismatch")
    if Counter(r["split"] for r in fuseg_rows) != {"train": 771, "val": 191}:
        raise ValueError("FUSeg split count mismatch")
    write_json(out / "isic_manifest.json", isic_rows)
    write_json(out / "fuseg_manifest.json", fuseg_rows)
    protocol = {
        "experiment_id": "D-Smoke-ISIC-FUSeg-20260914",
        "created_utc": now(), "role": "noncommercial_offline_development_only",
        "smoke_only": True, "test_images_used": 0, "blind_test_used": False,
        "stages": [
            {"name": "ISIC_auxiliary_pretraining", "source": "ISIC2017", "role": "auxiliary_pretraining",
             "train_images": 1800, "validation_images": 200, "epochs": 2, "patience": 2,
             "class_mapping": "foreground -> class 0 (pretraining only)", "dataset_yaml": str(out / "isic_dataset/dataset.yaml")},
            {"name": "FUSeg_wound_finetuning", "source": "FUSeg", "role": "wound_finetuning",
             "train_images": 771, "validation_images": 191, "epochs": 2, "patience": 2,
             "class_mapping": "class 0 -> Wound", "initialization": "ISIC smoke best.pt",
             "dataset_yaml": str(out / "fuseg_dataset/dataset.yaml")},
        ],
        "training": {"imgsz": 768, "batch": 4, "optimizer": "AdamW", "lr0": .0005,
                     "warmup_bias_lr": 0.0, "warmup_epochs": 5, "lrf": .01, "cos_lr": True,
                     "close_mosaic": 30, "seed": 42, "deterministic": True, "workers": 2,
                     "amp": True, "mosaic": .1, "mixup": 0.0, "copy_paste": 0.0,
                     "degrees": 5.0, "translate": .05, "scale": .2, "shear": .5,
                     "perspective": .0002, "fliplr": .5, "flipud": 0.0, "hsv_h": .01,
                     "hsv_s": .4, "hsv_v": .25, "device": "0", "val": True, "plots": True},
        "smoke_gates": ["both data manifests exact/hash verified", "loss finite", "best.pt exists for both stages",
                        "ISIC validation executes", "FUSeg validation executes", "test_images_used=0",
                        "sealed snapshot unchanged", "smoke checkpoints are lineage only; no App replacement"],
        "source_evidence": {"ISIC_audit": str(ISIC_AUDIT), "ISIC_audit_sha256": sha(ISIC_AUDIT),
                            "ISIC_license": "CC0 1.0; auxiliary skin-lesion pretraining only",
                            "FUSeg_protocol": str(FUSEG_PROTOCOL), "FUSeg_protocol_sha256": sha(FUSEG_PROTOCOL)},
        "initialization": str(INITIALIZATION), "initialization_sha256": INITIALIZATION_SHA256,
        "manifests": {"isic": sha(out / "isic_manifest.json"), "fuseg": sha(out / "fuseg_manifest.json")},
        "link_counts": {"isic": dict(isic_links), "fuseg": dict(fuseg_links)},
        "sealed_before": before, "runner_sha256": sha(Path(__file__)),
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "sealed_before.json", before)
    print(json.dumps({"status": "PREPARED_NOT_EXECUTED", "output": str(out),
                      "isic": [1800, 200], "fuseg": [771, 191], "test_images_used": 0}), flush=True)


def train_args(project: Path, name: str, data: Path):
    return dict(data=str(data), epochs=2, patience=2, imgsz=768, batch=4, optimizer="AdamW", lr0=.0005,
                lrf=.01, cos_lr=True, warmup_epochs=5, warmup_bias_lr=0.0, close_mosaic=30,
                save_period=-1, seed=42, deterministic=True, workers=2, amp=True, mosaic=.1,
                mixup=0.0, copy_paste=0.0, degrees=5.0, translate=.05, scale=.2, shear=.5,
                perspective=.0002, fliplr=.5, flipud=0.0, hsv_h=.01, hsv_s=.4, hsv_v=.25,
                project=str(project), name=name, exist_ok=False, device="0", val=True, plots=True,
                verbose=False)


def finite_history(path: Path):
    csv_path = path / "results.csv"
    if not csv_path.exists():
        raise RuntimeError(f"training history missing: {csv_path}")
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("training history empty")
    for row in rows:
        for key, value in row.items():
            if key != "epoch" and value and not np.isfinite(float(value)):
                raise RuntimeError(f"non-finite training value: {key}={value}")
    return rows


def execute(out: Path):
    import torch
    from ultralytics import YOLO
    protocol = read_json(out / "protocol.json")
    if protocol["test_images_used"] != 0 or snapshot_sealed() != read_json(out / "sealed_before.json"):
        raise ValueError("pre-run test/sealed gate failed")
    if not torch.cuda.is_available():
        raise RuntimeError("declared CUDA device unavailable")
    write_json(out / "execution.lock", {"started_utc": now(), "protocol_sha256": sha(out / "protocol.json")})
    runs = out / "runs"
    try:
        isic = YOLO(str(INITIALIZATION))
        isic.train(**train_args(runs, "isic_auxiliary_smoke", out / "isic_dataset/dataset.yaml"))
        isic_best = runs / "isic_auxiliary_smoke/weights/best.pt"
        if not isic_best.exists():
            raise RuntimeError("ISIC smoke best.pt missing")
        isic_history = finite_history(runs / "isic_auxiliary_smoke")
        isic_val = YOLO(str(isic_best)).val(data=str(out / "isic_dataset/dataset.yaml"), split="val",
                                            imgsz=768, batch=4, device="0", verbose=False)
        wound = YOLO(str(isic_best))
        wound.train(**train_args(runs, "fuseg_finetune_smoke", out / "fuseg_dataset/dataset.yaml"))
        wound_best = runs / "fuseg_finetune_smoke/weights/best.pt"
        if not wound_best.exists():
            raise RuntimeError("FUSeg smoke best.pt missing")
        wound_history = finite_history(runs / "fuseg_finetune_smoke")
        wound_val = YOLO(str(wound_best)).val(data=str(out / "fuseg_dataset/dataset.yaml"), split="val",
                                              imgsz=768, batch=4, device="0", verbose=False)
        if snapshot_sealed() != read_json(out / "sealed_before.json"):
            raise RuntimeError("sealed file changed during smoke")
        final = {"status": "PASS_SMOKE_ONLY_NOT_MODEL_CANDIDATE", "finished_utc": now(),
                 "test_images_used": 0, "blind_test_used": False,
                 "isic": {"train_images": 1800, "val_images": 200, "epochs_completed": len(isic_history),
                          "checkpoint": str(isic_best), "checkpoint_sha256": sha(isic_best),
                          "val_box_mAP50": float(isic_val.box.map50), "val_mask_mAP50": float(isic_val.seg.map50),
                          "val_box_precision": float(isic_val.box.mp), "val_box_recall": float(isic_val.box.mr)},
                 "fuseg": {"train_images": 771, "val_images": 191, "epochs_completed": len(wound_history),
                           "checkpoint": str(wound_best), "checkpoint_sha256": sha(wound_best),
                           "val_box_mAP50": float(wound_val.box.map50), "val_box_mAP50_95": float(wound_val.box.map),
                           "val_mask_mAP50": float(wound_val.seg.map50), "val_mask_mAP50_95": float(wound_val.seg.map),
                           "val_box_precision": float(wound_val.box.mp), "val_box_recall": float(wound_val.box.mr),
                           "val_mask_precision": float(wound_val.seg.mp), "val_mask_recall": float(wound_val.seg.mr)},
                 "integrity": {"sealed_unchanged": True, "app_model_replaced": False,
                               "test_images_used": 0, "smoke_checkpoints_only": True}}
        write_json(out / "result.json", final)
        (out / "實驗結果報告.md").write_text(
            "# ISIC auxiliary pretraining → FUSeg fine-tuning smoke\n\n"
            "完整資料 smoke 已通過：ISIC 1,800/200 各 2 epochs，接續 FUSeg 771/191 各 2 epochs。\n\n"
            f"- ISIC validation Mask mAP50：{final['isic']['val_mask_mAP50']:.2%}\n"
            f"- FUSeg validation Mask mAP50：{final['fuseg']['val_mask_mAP50']:.2%}\n"
            f"- FUSeg validation Mask mAP50-95：{final['fuseg']['val_mask_mAP50_95']:.2%}\n"
            "- 這些是 smoke 連通性結果，不是正式候選或臨床泛化結果。\n"
            "- test_images_used=0；沒有替換 App 模型；封版檔案未變更。\n", encoding="utf-8")
        print(json.dumps(final, ensure_ascii=False), flush=True)
    except Exception as exc:
        write_json(out / "failure.json", {"status": "FAIL_SMOKE", "error": repr(exc),
                                           "time": now(), "test_images_used": 0,
                                           "sealed_unchanged": snapshot_sealed() == read_json(out / "sealed_before.json")})
        raise


def finalize(out: Path):
    """Finalize an interrupted smoke run using its existing checkpoints only.

    This performs validation on the materialized validation splits (never a test
    split) and writes the same evidence files as a clean execute.  It does not
    retrain either stage or touch the sealed project artifacts.
    """
    import torch
    from ultralytics import YOLO

    protocol = read_json(out / "protocol.json")
    before = read_json(out / "sealed_before.json")
    if protocol["test_images_used"] != 0 or protocol["blind_test_used"]:
        raise ValueError("smoke protocol is not development-only")
    if snapshot_sealed() != before:
        raise ValueError("pre-finalize sealed gate failed")
    if not torch.cuda.is_available():
        raise RuntimeError("declared CUDA device unavailable")

    runs = out / "runs"
    isic_run = runs / "isic_auxiliary_smoke"
    fuseg_run = runs / "fuseg_finetune_smoke"
    isic_best = isic_run / "weights/best.pt"
    fuseg_best = fuseg_run / "weights/best.pt"
    if not isic_best.exists() or not fuseg_best.exists():
        raise RuntimeError("existing smoke checkpoints are incomplete")
    isic_history = finite_history(isic_run)
    wound_history = finite_history(fuseg_run)
    isic_val_images = sorted((out / "isic_dataset/images/val").glob("*"))
    fuseg_val_images = sorted((out / "fuseg_dataset/images/val").glob("*"))
    if len(isic_val_images) != 200 or len(fuseg_val_images) != 191:
        raise RuntimeError(f"validation split count mismatch: ISIC={len(isic_val_images)} FUSeg={len(fuseg_val_images)}")

    # Explicitly point both evaluations at materialized validation YAMLs.
    # Ultralytics may display the number of batches (48 for 191 images at batch 4);
    # that is not a 48-image test evaluation.
    isic_val = YOLO(str(isic_best)).val(
        data=str(out / "isic_dataset/dataset.yaml"), split="val", imgsz=768,
        batch=4, device="0", plots=False, verbose=False,
        project=str(runs), name="isic_auxiliary_smoke_finalize", exist_ok=False)
    wound_val = YOLO(str(fuseg_best)).val(
        data=str(out / "fuseg_dataset/dataset.yaml"), split="val", imgsz=768,
        batch=4, device="0", plots=False, verbose=False,
        project=str(runs), name="fuseg_finetune_smoke_finalize", exist_ok=False)
    if snapshot_sealed() != before:
        raise RuntimeError("sealed file changed during smoke finalize")

    final = {
        "status": "PASS_SMOKE_ONLY_NOT_MODEL_CANDIDATE",
        "finished_utc": now(), "test_images_used": 0, "blind_test_used": False,
        "isic": {"train_images": 1800, "val_images": 200,
                 "epochs_completed": len(isic_history), "checkpoint": str(isic_best),
                 "checkpoint_sha256": sha(isic_best),
                 "val_box_mAP50": float(isic_val.box.map50),
                 "val_mask_mAP50": float(isic_val.seg.map50),
                 "val_box_precision": float(isic_val.box.mp),
                 "val_box_recall": float(isic_val.box.mr)},
        "fuseg": {"train_images": 771, "val_images": 191,
                  "epochs_completed": len(wound_history), "checkpoint": str(fuseg_best),
                  "checkpoint_sha256": sha(fuseg_best),
                  "val_box_mAP50": float(wound_val.box.map50),
                  "val_box_mAP50_95": float(wound_val.box.map),
                  "val_mask_mAP50": float(wound_val.seg.map50),
                  "val_mask_mAP50_95": float(wound_val.seg.map),
                  "val_box_precision": float(wound_val.box.mp),
                  "val_box_recall": float(wound_val.box.mr),
                  "val_mask_precision": float(wound_val.seg.mp),
                  "val_mask_recall": float(wound_val.seg.mr)},
        "integrity": {"sealed_unchanged": True, "app_model_replaced": False,
                      "test_images_used": 0, "smoke_checkpoints_only": True,
                      "validation_images": {"isic": len(isic_val_images), "fuseg": len(fuseg_val_images)}}
    }
    write_json(out / "result.json", final)
    (out / "實驗結果報告.md").write_text(
        "# ISIC auxiliary pretraining → FUSeg fine-tuning smoke\n\n"
        "完整資料 smoke 已完成：ISIC 1,800/200 各 2 epochs，接續 FUSeg 771/191 各 2 epochs。\n\n"
        f"- ISIC validation Mask mAP50：{final['isic']['val_mask_mAP50']:.2%}\n"
        f"- FUSeg validation Box mAP50：{final['fuseg']['val_box_mAP50']:.2%}\n"
        f"- FUSeg validation Mask mAP50：{final['fuseg']['val_mask_mAP50']:.2%}\n"
        f"- FUSeg validation Mask mAP50-95：{final['fuseg']['val_mask_mAP50_95']:.2%}\n"
        "- 這些是 smoke 連通性結果，不是正式候選或臨床泛化結果。\n"
        "- 驗證只使用 materialized val split（ISIC 200、FUSeg 191）；test_images_used=0。\n"
        "- 沒有替換 App 模型；封版檔案未變更。\n", encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "execute", "finalize"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    mode_fn = {"prepare": prepare, "execute": execute, "finalize": finalize}[args.mode]
    mode_fn(output_path(args.output))
