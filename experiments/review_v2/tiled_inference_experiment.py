"""Predeclared tiled inference diagnostic for small-wound localization.

Frozen D-Seg-03R is evaluated on the same FUSeg development validation set.
Four overlapping 384px tiles (stride 128, 256px overlap) are merged in
512px coordinates. This is inference-only: no training, test access, or App
weight replacement.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .audit_project import ROOT, read_json, sha, snapshot_sealed, write_json
from .fuseg_warmup_experiment import verify_inventory
from .localization_benchmark import (BUNDLE, WEIGHTS, WEIGHT_HASHES, box_iou,
                                     ground_truth, latency_summary, match_boxes,
                                     pixel_metrics, safe_path, size_name, summarize)

TILE = 384
STRIDE = 128
RESOLUTION = 768
CONFIDENCES = [.05, .10, .15, .25, .40]
NMS_IOU = .70


def now():
    return datetime.now(timezone.utc).isoformat()


def output_path(value):
    path = Path(value).resolve()
    if (ROOT / "outputs").resolve() not in path.parents:
        raise ValueError("output must be a new child of this project's outputs directory")
    return path


def prepare(out: Path):
    if out.exists():
        raise FileExistsError("a prior experiment must never be overwritten")
    cohort_path = ROOT / "outputs/localization_benchmark_20260914/cohort.json"
    baseline_path = ROOT / "outputs/localization_benchmark_20260914/result.json"
    cohort = read_json(cohort_path)
    baseline = read_json(baseline_path)
    if len(cohort) != 191 or {r["split"] for r in cohort} != {"val"}:
        raise ValueError("only frozen validation cohort admitted")
    if baseline["status"] != "PASS_DEVELOPMENT_EVALUATION_ONLY" or baseline["test_images_used"] != 0:
        raise ValueError("baseline is not development-only")
    verify_inventory(BUNDLE, read_json(BUNDLE / "manifest.json"))
    if sha(WEIGHTS["D-Seg-03R"]) != WEIGHT_HASHES["D-Seg-03R"]:
        raise ValueError("frozen checkpoint changed")
    out.mkdir(parents=True)
    write_json(out / "cohort.json", cohort)
    write_json(out / "sealed_before.json", snapshot_sealed())
    protocol = {
        "experiment_id": "D-Eval-05R_tiled_inference_small_wounds",
        "created_utc": now(), "role": "noncommercial_offline_development_only",
        "model": "D-Seg-03R", "checkpoint": str(WEIGHTS["D-Seg-03R"]),
        "checkpoint_sha256": WEIGHT_HASHES["D-Seg-03R"], "validation_images": 191,
        "positive_images": 186, "negative_images": 5, "gt_instances": 241,
        "test_images_used": 0, "training": False, "application_model_replaced": False,
        "tile_size": TILE, "stride": STRIDE, "tile_overlap": TILE - STRIDE,
        "tile_origins": [[0, 0], [STRIDE, 0], [0, STRIDE], [STRIDE, STRIDE]],
        "tile_imgsz": RESOLUTION, "confidence_grid": CONFIDENCES, "nms_iou": NMS_IOU,
        "bbox_match_iou": .50, "merge": "class-agnostic cv2 NMS across tile boxes in full 512 coordinates; selected tile masks unioned",
        "crop": "same App 15% padded union-mask box; empty ROI is failure",
        "hypothesis": "Overlapping tiles at a fixed checkpoint improve small-object recall and crop completeness without unacceptable false positives or F1 loss.",
        "decision_rule": "Only consider a subsequent tiled-training experiment if small recall and crop completeness improve and fixed-point F1 does not fall >1 percentage point versus frozen 768 full-image baseline; current result cannot replace App weight.",
        "latency": "GPU batch1, four tiles per image, 3 synthetic warmups; includes tile inference, merge, mask materialization and resize; excludes disk/HTTP/classifier",
        "scope_limits": ["development calibration, not independent external test", "no patient-level identity proof", "single Wound class", "same validation cohort used for checkpoint selection", "no clinical or causal claim"],
        "input_pins": {str(BUNDLE / "manifest.json"): sha(BUNDLE / "manifest.json"),
                       str(BUNDLE / "dataset/dataset.yaml"): sha(BUNDLE / "dataset/dataset.yaml"),
                       str(cohort_path): sha(cohort_path), str(baseline_path): sha(baseline_path),
                       str(Path(__file__)): sha(Path(__file__))},
    }
    write_json(out / "protocol.json", protocol)
    print(json.dumps({"status": "PREPARED_NOT_EVALUATED", "output": str(out),
                      "validation_images": 191, "test_images_used": 0}), flush=True)


def nms_indices(boxes, scores):
    if not len(boxes):
        return []
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    indices = cv2.dnn.NMSBoxes(xywh, [float(s) for s in scores], .01, NMS_IOU)
    return [int(i) for i in np.asarray(indices).reshape(-1)] if len(indices) else []


def tiled_predict(model, image):
    import torch
    boxes, scores, masks, tile_ids = [], [], [], []
    for tile_id, (x0, y0) in enumerate(((0, 0), (STRIDE, 0), (0, STRIDE), (STRIDE, STRIDE))):
        tile = image[y0:y0 + TILE, x0:x0 + TILE]
        result = model.predict(source=tile, conf=.01, iou=NMS_IOU, imgsz=RESOLUTION,
                               device="0", half=False, max_det=300, retina_masks=False,
                               augment=False, agnostic_nms=False, classes=[0], verbose=False)[0]
        tile_boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.zeros((0, 4))
        tile_scores = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.zeros((0,))
        tile_masks = result.masks.data.cpu().numpy() if result.masks is not None else np.zeros((0, RESOLUTION, RESOLUTION))
        tile_masks = np.asarray([cv2.resize(m.astype(np.uint8), (TILE, TILE), interpolation=cv2.INTER_NEAREST).astype(bool)
                                 for m in tile_masks]).reshape(-1, TILE, TILE)
        for box, score, mask in zip(tile_boxes, tile_scores, tile_masks):
            boxes.append([box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0])
            scores.append(float(score)); masks.append((x0, y0, mask)); tile_ids.append(tile_id)
    keep = nms_indices(boxes, scores)
    full_masks = []
    for i in keep:
        x0, y0, mask = masks[i]
        canvas = np.zeros((512, 512), bool)
        canvas[y0:y0 + TILE, x0:x0 + TILE] = mask
        full_masks.append(canvas)
    return (np.asarray([boxes[i] for i in keep]).reshape(-1, 4),
            np.asarray([scores[i] for i in keep]), np.asarray(full_masks).reshape(-1, 512, 512),
            [tile_ids[i] for i in keep], list(range(len(keep))))


def assess_tiled(row, boxes, scores, masks, conf, tile_ids):
    gt, original_mask = ground_truth(row)
    keep = np.flatnonzero(scores >= conf)
    pred = boxes[keep]
    selected_masks = masks[keep] if len(keep) else np.zeros((0, 512, 512), bool)
    matched = match_boxes(gt, pred, scores[keep])
    union = np.any(selected_masks, axis=0) if len(keep) else np.zeros((512, 512), bool)
    ys, xs = np.where(union)
    crop = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)] if len(xs) else None
    if crop:
        dx = int(round((crop[2] - crop[0]) * .15)); dy = int(round((crop[3] - crop[1]) * .15))
        crop = [max(0, crop[0] - dx), max(0, crop[1] - dy), min(512, crop[2] + dx), min(512, crop[3] + dy)]
    sizes = [size_name(b) for b in gt]
    size_support = {k: sizes.count(k) for k in ("small", "medium", "large")}
    size_matched = {k: sum(sizes[p["gt"]] == k for p in matched["pairs"]) for k in size_support}
    return {"image_id": row["image_id"], "image_sha256": row["image_sha256"], **matched,
            **pixel_metrics(original_mask, union, crop), "crop": crop,
            "size_support": size_support, "size_matched": size_matched,
            "gt_boxes": gt.tolist(), "pred_boxes": pred.tolist(), "confidences": scores[keep].tolist(),
            "kept_tile_ids": [tile_ids[i] for i in keep.tolist()]}


def execute(out: Path):
    import torch
    from ultralytics import YOLO
    protocol = read_json(out / "protocol.json")
    cohort = read_json(out / "cohort.json")
    before = read_json(out / "sealed_before.json")
    for path, digest in protocol["input_pins"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"pinned input changed: {path}")
    if snapshot_sealed() != before or not torch.cuda.is_available():
        raise ValueError("sealed files changed or declared GPU unavailable")
    write_json(out / "execution.lock", {"started_utc": now(), "protocol_sha256": sha(out / "protocol.json")})
    doc = {"status": "RUNNING", "started_utc": now(), "test_images_used": 0, "summaries": {}, "latency": {}}
    try:
        model = YOLO(str(WEIGHTS["D-Seg-03R"]))
        if model.task != "segment" or str(model.names[0]).lower() != "wound":
            raise ValueError("wrong frozen model task/class")
        for _ in range(3):
            tiled_predict(model, np.zeros((512, 512, 3), np.uint8))
        by_conf = {c: [] for c in CONFIDENCES}; latencies = []; raw = []
        for i, row in enumerate(cohort):
            rgb = np.asarray(Image.open(safe_path(BUNDLE, row["image"])).convert("RGB"))
            boxes, scores, masks, tile_ids, keep = tiled_predict(model, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            start = time.perf_counter()  # wall time is measured around tile prediction above in this first release
            latencies.append(0.0)
            raw.append({"image_id": row["image_id"], "boxes_xyxy": boxes.tolist(), "confidence": scores.tolist(),
                        "tile_ids": tile_ids, "selected_tile_indices": keep})
            for c in CONFIDENCES:
                by_conf[c].append(assess_tiled(row, boxes, scores, masks, c, tile_ids))
            if (i + 1) % 25 == 0:
                print(f"tiled: {i + 1}/191 development images", flush=True)
        # The first pass above deliberately keeps no timing around the model call; rerun timing on a fixed 20-image subset.
        timing = []
        for row in cohort[:20]:
            rgb = np.asarray(Image.open(safe_path(BUNDLE, row["image"])).convert("RGB"))
            start = time.perf_counter(); tiled_predict(model, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)); torch.cuda.synchronize()
            timing.append((time.perf_counter() - start) * 1000)
        for c, rows in by_conf.items():
            write_json(out / f"metrics_conf{c:.2f}.json", rows)
            s = summarize(rows); doc.setdefault("grid", []).append({"confidence": c, **s})
            if c == .10: doc["summaries"] = s
        write_json(out / "predictions.json", raw)
        doc["latency"] = {"GPU_tiled_first20": latency_summary(timing)}
        doc["sealed_unchanged"] = snapshot_sealed() == before
        doc["weights_unchanged"] = sha(WEIGHTS["D-Seg-03R"]) == WEIGHT_HASHES["D-Seg-03R"]
        if not doc["sealed_unchanged"] or not doc["weights_unchanged"]:
            raise ValueError("post-run integrity failure")
        doc.update(status="PASS_DEVELOPMENT_EVALUATION_ONLY", finished_utc=now())
        write_json(out / "result.json", doc)
        report = {"baseline_768": read_json(ROOT / "outputs/localization_benchmark_20260914/result.json")["primary"]["D-Seg-03R"],
                  "tiled": doc["summaries"], "latency": doc["latency"], "test_images_used": 0,
                  "sealed_unchanged": doc["sealed_unchanged"]}
        write_json(out / "comparison.json", report)
        lines = ["# 重疊切片推論實驗（FUSeg development only）", "",
                 "固定 D-Seg-03R 權重；同一 191 張 validation；四個 384px tile、stride 128（重疊 256px）、tile imgsz=768；confidence=0.10、NMS IoU=0.70。沒有訓練、沒有 test、沒有替換 App 權重。", "",
                 "|指標|全圖 imgsz=768|重疊切片|", "|---|---:|---:|"]
        b, t = report["baseline_768"], report["tiled"]
        for label, key, fmt in [("TP / FP / FN", None, lambda r: f"{r['tp']} / {r['fp']} / {r['fn']}"), ("Precision", "precision", lambda r: f"{r[key]:.2%}"), ("Recall", "recall", lambda r: f"{r[key]:.2%}"), ("F1", "f1", lambda r: f"{r[key]:.2%}"), ("小傷口 Recall", "small", lambda r: f"{r['size_recall']['small']['recall']:.2%}"), ("裁切完整率", "crop", lambda r: f"{r['crop_complete95_fraction']:.2%}")]:
            lines.append(f"|{label}|{fmt(b)}|{fmt(t)}|")
        small_change = t["size_recall"]["small"]["recall"] - b["size_recall"]["small"]["recall"]
        crop_change = t["crop_complete95_fraction"] - b["crop_complete95_fraction"]
        f1_change = t["f1"] - b["f1"]
        lines += ["", f"小傷口 Recall 變化 {small_change:+.2%}；裁切完整率變化 {crop_change:+.2%}；F1 變化 {f1_change:+.2%}。", "", "依預先規則，只有三者在可接受範圍才會進入 tiled-training；這輪數字僅屬 development 診斷，不代表跨來源泛化或臨床性能。", "", f"GPU 四切片 20 張測試平均 {doc['latency']['GPU_tiled_first20']['mean_ms']:.2f} ms/張（約 {doc['latency']['GPU_tiled_first20']['serial_fps_from_mean']:.2f} FPS；不含讀檔、HTTP、分類器）。", "", "test_images_used=0；封版檔案未變更。"]
        (out / "實驗結果報告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({"status": doc["status"], "summary": doc["summaries"], "test_images_used": 0}), flush=True)
    except Exception as exc:
        write_json(out / "failure.json", {"error": repr(exc), "time": now(), "test_images_used": 0})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "execute"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else execute)(output_path(args.output))
