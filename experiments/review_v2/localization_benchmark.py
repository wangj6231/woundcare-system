"""Frozen-checkpoint DEVELOPMENT operating-point experiment, never a blind test.

Prepare first, then execute the pinned protocol once. Uses the existing copied
FUSeg development split and publisher binary masks, not any test directory.
No training, model selection, application changes or dataset mutation occurs.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .audit_project import ROOT, read_json, sha, snapshot_sealed, write_json
from .fuseg_warmup_experiment import EVIDENCE, PDF, SOURCE_MANIFEST, verify_inventory
from .guards import validate_polygon_line
from woundcare_inference import _padded_box, _segmentation_box

BUNDLE = ROOT / "outputs/fuseg_warmup_revision_20260914"
WEIGHTS = {
    "D-Seg-03": ROOT / "experiments/results/segmentation/D-Seg-03_yolo11m_fuseg_only_s42/weights/best.pt",
    "D-Seg-03R": BUNDLE / "formal/weights/best.pt",
}
WEIGHT_HASHES = {
    "D-Seg-03": "09a5b0003afd52b10cdbb2d0793bcc381453d3bb950d4c41e72b3108a483d3c8",
    "D-Seg-03R": "2a2d66e7036f2d8eb94e9da8907554f0fcd55484e138c25b024b14fbc6de678d",
}
THRESHOLDS = [.05, .10, .15, .25, .40, .50, .70]
NMS_VALUES = [.50, .70]


def now():
    return datetime.now(timezone.utc).isoformat()


def output_path(value):
    path = Path(value).resolve()
    if (ROOT / "outputs").resolve() not in path.parents:
        raise ValueError("output must be a new child of this project's outputs directory")
    return path


def safe_path(base: Path, relative: str):
    path = (base / relative).resolve()
    if base.resolve() not in path.parents:
        raise ValueError("path escapes the declared development directory")
    return path


def pinned_files():
    return [Path(__file__), ROOT / "woundcare_inference.py", EVIDENCE, PDF,
            SOURCE_MANIFEST, BUNDLE / "manifest.json", BUNDLE / "protocol.json",
            BUNDLE / "dataset/dataset.yaml", *WEIGHTS.values()]


def binary_mask(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[2] == 3:
        if not (np.array_equal(arr[:, :, 0], arr[:, :, 1]) and
                np.array_equal(arr[:, :, 0], arr[:, :, 2])):
            raise ValueError("RGB mask channels are not identical")
        arr = arr[:, :, 0]
    if arr.shape != (512, 512) or arr.dtype != np.uint8:
        raise ValueError("expected a 512x512 uint8 grayscale mask")
    # The frozen dataset builder uses >0, including the publisher's value 32.
    # prepare() independently verifies the original manifest's binary hash.
    return arr > 0


def prepare(out: Path):
    if out.exists():
        raise FileExistsError("a prior experiment must never be overwritten")
    previous = read_json(BUNDLE / "protocol.json")
    gate = previous["source_gate"]["FUSeg"]
    expected = {EVIDENCE: gate["evidence_sha256"], PDF: gate["publisher_pdf_sha256"],
                SOURCE_MANIFEST: previous["source_manifest_sha256"],
                BUNDLE / "manifest.json": previous["manifest_sha256"],
                BUNDLE / "dataset/dataset.yaml": previous["dataset_yaml_sha256"]}
    expected.update({p: WEIGHT_HASHES[k] for k, p in WEIGHTS.items()})
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError(f"frozen input changed: {path.name}")
    manifest = read_json(BUNDLE / "manifest.json")
    verify_inventory(BUNDLE, manifest)
    if Counter(r["split"] for r in manifest) != {"train": 771, "val": 191}:
        raise ValueError("unexpected development counts")
    train_ids = {r["image_sha256"] for r in manifest if r["split"] == "train"}
    val_ids = {r["image_sha256"] for r in manifest if r["split"] == "val"}
    if len(train_ids) != 771 or len(val_ids) != 191 or train_ids & val_ids:
        raise ValueError("duplicate/overlapping content")
    with SOURCE_MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        sources = [r for r in csv.DictReader(stream) if r["assigned_split"] == "val"]
    by_id = {Path(r["fuseg_image"]).name: r for r in sources}
    if len(by_id) != 191 or len(sources) != 191:
        raise ValueError("source validation mapping not one-to-one")
    selected, grayscale_notes = [], []
    mask_root = ROOT / "official_detection_sources_20260812/fuseg/repository/data/Foot Ulcer Segmentation Challenge/validation/labels"
    for r in sorted((r for r in manifest if r["split"] == "val"), key=lambda r: r["image_id"]):
        s = by_id[r["image_id"]]
        mask = (ROOT / s["source_mask"]).resolve()
        if (s["source"] != "FUSeg" or s["official_split"] != "val"
                or mask_root.resolve() not in mask.parents
                or s["image_sha256"] != r["image_sha256"]
                or sha(mask) != s["mask_file_sha256"]):
            raise ValueError("source mask is not the admitted validation identity")
        with Image.open(mask) as im:
            raw = np.asarray(im)
            arr = binary_mask(raw)
            if not set(np.unique(raw)) <= {0, 255}:
                grayscale_notes.append({"image_id": r["image_id"], "values": np.unique(raw).tolist()})
            payload = b"512x512:" + arr.astype(np.uint8).tobytes()
            if hashlib.sha256(payload).hexdigest() != s["mask_binary_sha256"]:
                raise ValueError("binary mask semantics differ from the frozen data builder")
            positive = bool(arr.any())
        with Image.open(safe_path(BUNDLE, r["image"])) as im:
            if im.size != (512, 512):
                raise ValueError("square 512-pixel images are required for exact mask resizing")
        if positive != (r["instances"] > 0):
            raise ValueError("original mask and converted polygons disagree on presence")
        selected.append({**r, "source_mask": s["source_mask"],
                         "mask_sha256": s["mask_file_sha256"], "positive": positive})
    out.mkdir(parents=True)
    write_json(out / "cohort.json", selected)
    protocol = {
        "experiment_id": "D-Eval-03R_localization_operating_points",
        "created_utc": now(), "role": "noncommercial_offline_development_only",
        "train_images": 771, "validation_images": 191,
        "positive_images": sum(r["positive"] for r in selected),
        "negative_images": sum(not r["positive"] for r in selected),
        "gt_polygon_instances": sum(r["instances"] for r in selected),
        "publisher_mask_conversion": "identical RGB channels to grayscale, >0 foreground; all binary hashes match original build manifest",
        "publisher_nonbinary_grayscale_notes": grayscale_notes,
        "test_images_used": 0, "training": False, "application_model_replaced": False,
        "models": {k: {"path": str(p), "sha256": WEIGHT_HASHES[k]} for k, p in WEIGHTS.items()},
        "primary": {"confidence": .10, "nms_iou": .70, "bbox_match_iou": .50},
        "confidence_grid": THRESHOLDS, "nms_grid": NMS_VALUES,
        "prediction": {"imgsz": 768, "batch": 1, "device": "0", "half": False,
                       "retina_masks": False, "max_det": 300, "conf": .01,
                       "augment": False, "agnostic_nms": False, "classes": [0]},
        "matching": "predictions descending confidence; unmatched GT with highest IoU >= 0.50; one-to-one",
        "mask_metrics": "union of predicted binary instance masks resized nearest 768 to 512 against original publisher binary mask; positive-image macro means",
        "crop": "actual App polygon-union bounding rectangle plus 15% per-side margin; success retains >=95% of original wound pixels; no ROI counts as failure",
        "size_strata": "GT bbox area/image area: small <1%, medium 1%-5%, large >=5%; not COCO size strata",
        "threshold_policy": "fixed primary point; full grid reported as development calibration only; no automatic threshold or model replacement",
        "latency": "3 synthetic warmups; GPU batch1 all191; CPU first20 sorted val IDs for candidate only; excludes disk read, includes predict and box/mask CPU materialization; no HTTP/classifier latency",
        "scope_limits": ["not an independent external test", "no patient-level grouping evidence",
                         "only Wound class, no seven-class/cascade accuracy", "only 5 negative images",
                         "no causal/significance claim for historical baseline comparison"],
        "pins": {str(p): sha(p) for p in pinned_files()},
        "cohort_sha256": sha(out / "cohort.json"),
        "software": {k: importlib.metadata.version(k) for k in ["ultralytics", "torch", "numpy", "pillow", "opencv-python"]},
    }
    write_json(out / "sealed_before.json", snapshot_sealed())
    write_json(out / "protocol.json", protocol)
    print(json.dumps({"status": "PREPARED_NOT_EVALUATED", "output": str(out),
                      "val": len(selected), "test_images_used": 0}), flush=True)


def box_iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    inter = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                       np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    areas_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    areas_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / np.maximum(areas_a[:, None] + areas_b[None, :] - inter, 1e-12)


def match_boxes(gt, pred, confidence, iou_threshold=.5):
    overlaps = box_iou(pred, gt)
    matched, pairs = set(), []
    for pi in np.argsort(-np.asarray(confidence), kind="stable"):
        choices = [j for j in range(len(gt)) if j not in matched and overlaps[pi, j] >= iou_threshold]
        if choices:
            gi = max(choices, key=lambda j: overlaps[pi, j])
            pairs.append({"prediction": int(pi), "gt": gi, "iou": float(overlaps[pi, gi])})
            matched.add(gi)
    return {"tp": len(pairs), "fp": len(pred) - len(pairs), "fn": len(gt) - len(pairs), "pairs": pairs}


def pixel_metrics(gt, predicted, crop):
    gt, predicted = np.asarray(gt, bool), np.asarray(predicted, bool)
    if gt.shape != predicted.shape:
        raise ValueError("unaligned masks")
    n = int(gt.sum())
    intersection = int((gt & predicted).sum())
    union = int((gt | predicted).sum())
    retained = int(gt[crop[1]:crop[3], crop[0]:crop[2]].sum()) if crop else 0
    return {"gt_pixels": n, "predicted_pixels": int(predicted.sum()),
            "mask_iou": intersection / union if n else None,
            "mask_dice": 2 * intersection / (n + int(predicted.sum())) if n else None,
            "crop_coverage": retained / n if n else None,
            "crop_complete95": bool(n and retained / n >= .95),
            "crop_area_fraction": ((crop[2] - crop[0]) * (crop[3] - crop[1]) / gt.size) if crop else 0.0}


def summarize(rows):
    positive = [r for r in rows if r["gt_pixels"] > 0]
    negative = [r for r in rows if r["gt_pixels"] == 0]
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    strata = {}
    for name in ("small", "medium", "large"):
        n = sum(r["size_support"][name] for r in rows)
        hits = sum(r["size_matched"][name] for r in rows)
        strata[name] = {"gt": n, "matched": hits, "recall": hits / n if n else None}
    mean = lambda key: float(np.mean([r[key] for r in positive])) if positive else None
    return {"images": len(rows), "positive_images": len(positive), "negative_images": len(negative),
            "tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "mask_iou_positive_mean": mean("mask_iou"), "mask_dice_positive_mean": mean("mask_dice"),
            "crop_coverage_positive_mean": mean("crop_coverage"),
            "crop_complete95_images": sum(r["crop_complete95"] for r in positive),
            "crop_complete95_fraction": mean("crop_complete95"),
            "positive_without_roi": sum(r["crop"] is None for r in positive),
            "negative_images_with_predictions": sum(r["tp"] + r["fp"] > 0 for r in negative),
            "crop_area_positive_mean": mean("crop_area_fraction"), "size_recall": strata}


def latency_summary(values):
    return {"n": len(values), "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, .95)),
            "serial_fps_from_mean": 1000 / float(np.mean(values))}


def ground_truth(row):
    polys = [np.asarray(validate_polygon_line(line)) * 512
             for line in safe_path(BUNDLE, row["label"]).read_text().splitlines() if line.strip()]
    boxes = [[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polys]
    with Image.open(ROOT / row["source_mask"]) as im:
        mask = binary_mask(np.asarray(im))
    return np.asarray(boxes).reshape(-1, 4), mask


def size_name(box):
    area = (box[2] - box[0]) * (box[3] - box[1]) / (512 * 512)
    return "small" if area < .01 else "medium" if area < .05 else "large"


def assess(row, result, mask_data, conf):
    scores = result.boxes.conf.cpu().numpy()
    keep = np.flatnonzero(scores >= conf)
    selected = result[keep.tolist()]
    pred = selected.boxes.xyxy.cpu().numpy()
    gt, original_mask = ground_truth(row)
    matched = match_boxes(gt, pred, scores[keep])
    raw, _, _ = _segmentation_box(selected, 512, 512)
    crop = _padded_box(raw, 512, 512, .15) if raw else None
    union = np.any(mask_data[keep], axis=0) if len(keep) else np.zeros((512, 512), bool)
    sizes = [size_name(b) for b in gt]
    size_support = {k: sizes.count(k) for k in ("small", "medium", "large")}
    size_matched = {k: sum(sizes[p["gt"]] == k for p in matched["pairs"]) for k in size_support}
    return {"image_id": row["image_id"], "image_sha256": row["image_sha256"],
            "mask_sha256": row["mask_sha256"], "crop": crop, **matched,
            **pixel_metrics(original_mask, union, crop), "size_support": size_support,
            "size_matched": size_matched, "gt_boxes": gt.tolist(), "pred_boxes": pred.tolist(),
            "confidences": scores[keep].tolist()}


def predict_materialized(model, image, settings):
    import torch
    gpu = str(settings["device"]) != "cpu"
    if gpu:
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = model.predict(source=image, verbose=False, save=False, **settings)[0]
    result.boxes.data.cpu().numpy()
    masks = result.masks.data.cpu().numpy() if result.masks is not None else np.zeros((0, 768, 768), bool)
    # Cohort is exactly square, hence no letterbox padding to remove.
    if len(masks) and masks.shape[1:] != (768, 768):
        raise ValueError("unexpected prediction mask dimensions")
    masks = np.asarray([cv2.resize(m.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
                        for m in masks]).reshape(-1, 512, 512)
    if gpu:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    return result, masks, elapsed


def error_panels(out, cohort, rows):
    lookup = {r["image_id"]: r for r in cohort}
    positives = [r for r in rows if r["gt_pixels"]]
    selected = sorted(positives, key=lambda r: (r["crop_coverage"], r["mask_iou"], r["image_id"]))[:12]
    write_json(out / "error_cases.json", selected)
    sheet = Image.new("RGB", (1024, 4 * 330), "white")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
    for i, r in enumerate(selected):
        im = Image.open(safe_path(BUNDLE, lookup[r["image_id"]]["image"])).convert("RGB")
        draw = ImageDraw.Draw(im)
        for box in r["gt_boxes"]:
            draw.rectangle(box, outline="#00ff00", width=3)
        for box in r["pred_boxes"]:
            draw.rectangle(box, outline="#ff2222", width=3)
        if r["crop"]:
            draw.rectangle(r["crop"], outline="#00ddff", width=3)
        x, y = (i % 3) * 341, (i // 3) * 330
        sheet.paste(im.resize((300, 270)), (x, y))
        ImageDraw.Draw(sheet).text((x, y + 275),
            f"{r['image_id']}  TP/FP/FN {r['tp']}/{r['fp']}/{r['fn']}\n"
            f"crop {r['crop_coverage']:.1%}  mask IoU {r['mask_iou']:.1%}", font=font, fill="black")
    sheet.save(out / "worst_crop_cases.png")


def report(out, result):
    lines = ["# 傷口定位與裁切：固定模型 development 驗證報告", "",
        "本輪未訓練、未更換 App 模型、未使用任何盲測影像。使用原 FUSeg 191 張驗證圖（186 有傷口、5 張無傷口），不是獨立外部測試。",
        "", "## 相同操作條件比較", "",
        "主設定：confidence=0.10、NMS IoU=0.70、bbox 配對 IoU≥0.50；單類 Wound。",
        "", "|模型|TP / FP / FN|Precision|Recall|F1|裁切完整率（≥95% 傷口像素）|",
        "|---|---:|---:|---:|---:|---:|"]
    for name, r in result["primary"].items():
        lines.append(f"|{name}|{r['tp']} / {r['fp']} / {r['fn']}|{r['precision']:.2%}|{r['recall']:.2%}|{r['f1']:.2%}|{r['crop_complete95_images']}/{r['positive_images']}（{r['crop_complete95_fraction']:.2%}）|")
    lines += ["", "這些是固定閾值結果，不是 mAP，也不是七類分類正確率。裁切完整率包含未產生 ROI 的失敗影像，不只計算成功偵測者。",
              "", "## 新模型錯誤分解", ""]
    r = result["primary"]["D-Seg-03R"]
    for size, s in r["size_recall"].items():
        value = f"{s['recall']:.2%}" if s["recall"] is not None else "不適用"
        lines.append(f"- {size}：{s['matched']}/{s['gt']}，Recall {value}。")
    lines += [f"- 有傷口但無 ROI：{r['positive_without_roi']}/{r['positive_images']}。",
              f"- 無傷口影像出現預測：{r['negative_images_with_predictions']}/{r['negative_images']}（樣本太少，不能外推誤報率）。",
              f"- 正樣本平均 Mask IoU：{r['mask_iou_positive_mean']:.2%}；Dice：{r['mask_dice_positive_mean']:.2%}。",
              f"- 正樣本平均裁切保留傷口像素：{r['crop_coverage_positive_mean']:.2%}；裁切平均佔原圖 {r['crop_area_positive_mean']:.2%}。",
              "", "small/medium/large 是原圖 bbox 面積 <1%、1%–5%、≥5%，不是傷口種類，也不是 COCO 尺度分類。",
              "", "## 速度", ""]
    for name, s in result["latency"].items():
        lines.append(f"- {name}：n={s['n']}，平均 {s['mean_ms']:.2f} ms，中位數 {s['median_ms']:.2f} ms，P95 {s['p95_ms']:.2f} ms，序列速度 {s['serial_fps_from_mean']:.2f} FPS。")
    lines += ["", "速度為 batch=1、3 次暖機後的定位推論與結果轉到 CPU；不含讀檔、HTTP、分類器與報表分析，不是整個 App 的端到端速度。CPU 只用事先固定的前 20 張驗證圖。",
              "", "## 可稽核產物", "",
              "- protocol.json / cohort.json：執行前固定的權重、191 張影像與原始 mask 雜湊、閾值與指標定義。",
              "- result.json：主設定、全部 28 組模型 × NMS × confidence 組合、配對影像變化及封版檢查。",
              "- 各模型子資料夾 predictions.json：逐圖偵測框、信心值、指標；masks/：原圖尺度預測 mask。",
              "- worst_crop_cases.png：新模型裁切保留率最低的 12 張；綠框 GT、紅框預測、青框含 15% 邊距裁切。",
              "", "## 解讀限制", "",
              "完整閾值網格只屬 development 校準，不自动採用最漂亮的數字作為 test 結果。兩模型皆曾使用這個驗證集挑選權重；比較不能證明泛化改善或 warmup 變更的因果。沒有病人 ID，不能聲稱病人級隔離。",
              "", "後續應依固定操作點的漏偵測與小傷口問題決定新訓練實驗；需取得未用且授權清楚的另一來源，才可驗證真正跨資料來源能力。不得回頭使用已評估的 CO2Wounds 或 48 張分類盲測調參。",
              "", f"封版雜湊未改變：{result['sealed_unchanged']}；test_images_used=0。"]
    (out / "實驗結果報告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(out: Path):
    import torch
    from ultralytics import YOLO
    protocol = read_json(out / "protocol.json")
    if sha(out / "cohort.json") != protocol["cohort_sha256"]:
        raise ValueError("cohort changed after preparation")
    for path, digest in protocol["pins"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"pinned input changed: {path}")
    cohort = read_json(out / "cohort.json")
    verify_inventory(BUNDLE, read_json(BUNDLE / "manifest.json"))
    for row in cohort:
        if row["split"] != "val" or sha(ROOT / row["source_mask"]) != row["mask_sha256"]:
            raise ValueError("mask/split identity changed")
    if not torch.cuda.is_available():
        raise RuntimeError("declared GPU is unavailable; no silent device change")
    before = read_json(out / "sealed_before.json")
    if snapshot_sealed() != before:
        raise ValueError("sealed files changed since preparation")
    write_json(out / "execution.lock", {"started_utc": now(), "protocol_sha256": sha(out / "protocol.json")})
    result_doc = {"status": "RUNNING", "started_utc": now(), "test_images_used": 0,
                  "gpu": torch.cuda.get_device_name(0), "grid": [], "primary": {}, "latency": {}}
    primary_rows = {}
    try:
        for name, weight in WEIGHTS.items():
            for nms in protocol["nms_grid"]:
                dest = out / f"{name}_nms{nms:.2f}"
                (dest / "masks").mkdir(parents=True)
                model = YOLO(str(weight))
                if model.task != "segment" or len(model.names) != 1 or str(model.names[0]).lower() != "wound":
                    raise ValueError("model is not the declared single-Wound segmenter")
                settings = {**protocol["prediction"], "iou": nms}
                for _ in range(3):
                    predict_materialized(model, np.zeros((512, 512, 3), np.uint8), settings)
                rows_by_conf = {c: [] for c in protocol["confidence_grid"]}
                raw_records, wall, infer = [], [], []
                for i, row in enumerate(cohort):
                    rgb = np.asarray(Image.open(safe_path(BUNDLE, row["image"])).convert("RGB"))
                    prediction, masks, elapsed = predict_materialized(model, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), settings)
                    wall.append(elapsed)
                    infer.append(prediction.speed["inference"])
                    mask_path = dest / "masks" / f"{Path(row['image_id']).stem}.npz"
                    np.savez_compressed(mask_path, masks=masks)
                    raw_records.append({"image_id": row["image_id"], "boxes_xyxy": prediction.boxes.xyxy.cpu().tolist(),
                                        "confidence": prediction.boxes.conf.cpu().tolist(), "mask_file": str(mask_path.relative_to(out)),
                                        "mask_file_sha256": sha(mask_path), "wall_ms": elapsed,
                                        "inference_ms": prediction.speed["inference"]})
                    for c in rows_by_conf:
                        rows_by_conf[c].append(assess(row, prediction, masks, c))
                    if (i + 1) % 50 == 0:
                        print(f"{name} NMS={nms}: {i + 1}/191 development images", flush=True)
                write_json(dest / "predictions.json", raw_records)
                for c, rows in rows_by_conf.items():
                    s = summarize(rows)
                    result_doc["grid"].append({"model": name, "nms_iou": nms, "confidence": c, **s})
                    write_json(dest / f"metrics_conf{c:.2f}.json", rows)
                    if nms == .7 and c == .1:
                        result_doc["primary"][name] = s
                        primary_rows[name] = rows
                result_doc["latency"][f"{name}_GPU_NMS{nms:.2f}_conf0.01_materialized"] = latency_summary(wall)
                result_doc["latency"][f"{name}_GPU_NMS{nms:.2f}_conf0.01_forward"] = latency_summary(infer)
                del model
                gc.collect()
                torch.cuda.empty_cache()
        model = YOLO(str(WEIGHTS["D-Seg-03R"]))
        settings = {**protocol["prediction"], "device": "cpu", "conf": .1, "iou": .7}
        for _ in range(3):
            predict_materialized(model, np.zeros((512, 512, 3), np.uint8), settings)
        cpu = []
        for row in cohort[:20]:
            rgb = np.asarray(Image.open(safe_path(BUNDLE, row["image"])).convert("RGB"))
            _, _, elapsed = predict_materialized(model, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), settings)
            cpu.append(elapsed)
        result_doc["latency"]["D-Seg-03R_CPU_NMS0.70_conf0.10_materialized_first20"] = latency_summary(cpu)
        result_doc["cpu_threads"] = torch.get_num_threads()
        old = {r["image_id"]: r for r in primary_rows["D-Seg-03"] if r["gt_pixels"]}
        new = {r["image_id"]: r for r in primary_rows["D-Seg-03R"] if r["gt_pixels"]}
        result_doc["paired_crop_complete95"] = {
            "improved_ids": [k for k in old if not old[k]["crop_complete95"] and new[k]["crop_complete95"]],
            "regressed_ids": [k for k in old if old[k]["crop_complete95"] and not new[k]["crop_complete95"]],
            "scope": "paired descriptive development counts; no independence/significance claim"}
        error_panels(out, cohort, primary_rows["D-Seg-03R"])
        result_doc["sealed_unchanged"] = snapshot_sealed() == before
        result_doc["weights_unchanged"] = all(sha(p) == WEIGHT_HASHES[k] for k, p in WEIGHTS.items())
        verify_inventory(BUNDLE, read_json(BUNDLE / "manifest.json"))
        result_doc["data_unchanged"] = all(sha(ROOT / r["source_mask"]) == r["mask_sha256"] for r in cohort)
        if not all(result_doc[k] for k in ("sealed_unchanged", "weights_unchanged", "data_unchanged")):
            raise ValueError("post-run integrity failure")
        result_doc.update(status="PASS_DEVELOPMENT_EVALUATION_ONLY", finished_utc=now())
        write_json(out / "result.json", result_doc)
        report(out, result_doc)
        print(json.dumps({"status": result_doc["status"], "primary": result_doc["primary"],
                          "test_images_used": 0}), flush=True)
    except Exception as exc:
        write_json(out / "failure.json", {"error": repr(exc), "time": now(), "test_images_used": 0,
                                         "sealed_unchanged": snapshot_sealed() == before})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "execute"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else execute)(output_path(args.output))
