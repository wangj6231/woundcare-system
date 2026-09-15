"""Predeclared inference-resolution ablation on the frozen FUSeg development set.

This is a development diagnostic only: one frozen checkpoint, no training,
no test access, and no application model replacement. It tests whether raising
inference resolution plausibly addresses the observed small-object misses.
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
from .localization_benchmark import (BUNDLE, WEIGHTS, WEIGHT_HASHES, assess,
                                     latency_summary, safe_path, summarize)

OUT_DEFAULT = ROOT / "outputs/highres_inference_20260914"
RESOLUTIONS = [768, 1024]
CONFIDENCES = [.05, .10, .15, .25, .40]


def now():
    return datetime.now(timezone.utc).isoformat()


def output_path(value):
    path = Path(value).resolve()
    if (ROOT / "outputs").resolve() not in path.parents:
        raise ValueError("output must be a new child of this project's outputs directory")
    return path


def make_protocol(out: Path):
    if out.exists():
        raise FileExistsError("a prior experiment must never be overwritten")
    base = read_json(ROOT / "outputs/localization_benchmark_20260914/protocol.json")
    cohort = read_json(ROOT / "outputs/localization_benchmark_20260914/cohort.json")
    if len(cohort) != 191 or {r["split"] for r in cohort} != {"val"}:
        raise ValueError("only the frozen validation cohort is admitted")
    verify_inventory(BUNDLE, read_json(BUNDLE / "manifest.json"))
    if sha(ROOT / "outputs/localization_benchmark_20260914/cohort.json") != base["cohort_sha256"]:
        raise ValueError("frozen cohort changed")
    if sha(WEIGHTS["D-Seg-03R"]) != WEIGHT_HASHES["D-Seg-03R"]:
        raise ValueError("frozen D-Seg-03R checkpoint changed")
    sealed = snapshot_sealed()
    out.mkdir(parents=True)
    write_json(out / "cohort.json", cohort)
    write_json(out / "sealed_before.json", sealed)
    protocol = {
        "experiment_id": "D-Eval-04R_inference_resolution_ablation",
        "created_utc": now(), "role": "noncommercial_offline_development_only",
        "model": "D-Seg-03R", "checkpoint": str(WEIGHTS["D-Seg-03R"]),
        "checkpoint_sha256": WEIGHT_HASHES["D-Seg-03R"],
        "validation_images": 191, "positive_images": 186, "negative_images": 5,
        "gt_instances": 241, "test_images_used": 0, "training": False,
        "application_model_replaced": False, "resolutions": RESOLUTIONS,
        "confidence_grid": CONFIDENCES, "nms_iou": .70, "bbox_match_iou": .50,
        "prediction": {"batch": 1, "device": "0", "half": False, "conf": .01,
                       "max_det": 300, "retina_masks": False, "augment": False,
                       "agnostic_nms": False, "classes": [0]},
        "hypothesis": "At fixed weights and NMS, imgsz=1024 will improve small-object box recall and >=95% crop retention relative to imgsz=768.",
        "decision_rule": "Do not train or replace App model unless 1024 improves small-object recall and total crop-complete fraction without reducing total fixed-point F1 by >1 percentage point; this rule is prospective and descriptive, not test selection.",
        "matching": "descending-confidence one-to-one bbox matching at IoU>=.50; all 241 GT instances remain denominator",
        "mask_and_crop": "same original publisher masks and current App 15% padded union crop as D-Eval-03R",
        "latency": "GPU batch1, 3 synthetic warmups per resolution; includes predict, NMS, mask CPU materialization and resize; excludes disk/HTTP/classifier; no CPU run",
        "scope_limits": ["development calibration, not independent external test", "no patient-level identity proof", "single Wound class", "same validation cohort used for checkpoint selection", "no causal or clinical claim"],
        "input_pins": {str(BUNDLE / "manifest.json"): sha(BUNDLE / "manifest.json"),
                       str(BUNDLE / "dataset/dataset.yaml"): sha(BUNDLE / "dataset/dataset.yaml"),
                       str(ROOT / "outputs/localization_benchmark_20260914/cohort.json"): sha(ROOT / "outputs/localization_benchmark_20260914/cohort.json"),
                       str(Path(__file__)): sha(Path(__file__))},
    }
    write_json(out / "protocol.json", protocol)
    print(json.dumps({"status": "PREPARED_NOT_EVALUATED", "output": str(out),
                      "validation_images": 191, "test_images_used": 0}), flush=True)


def predict(model, image, settings):
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = model.predict(source=image, verbose=False, save=False, **settings)[0]
    result.boxes.data.cpu().numpy()
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
    else:
        masks = np.zeros((0, settings["imgsz"], settings["imgsz"]), bool)
    masks = np.asarray([cv2.resize(m.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
                        for m in masks]).reshape(-1, 512, 512)
    torch.cuda.synchronize()
    return result, masks, (time.perf_counter() - start) * 1000


def write_report(out, result):
    rows = result["summaries"]
    lines = ["# 推論解析度單因素實驗（FUSeg development only）", "",
             "本輪使用固定 D-Seg-03R 權重與同一 191 張 validation cohort，只改 imgsz=768/1024。沒有訓練、沒有讀取 test、沒有替換 App 權重。",
             "", "## 固定操作點（confidence=0.10、NMS IoU=0.70、bbox IoU≥0.50）", "",
             "|推論解析度|TP/FP/FN|Precision|Recall|F1|小傷口 Recall|裁切完整率|平均延遲|",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for size in RESOLUTIONS:
        r = rows[str(size)]
        lines.append(f"|{size}|{r['tp']}/{r['fp']}/{r['fn']}|{r['precision']:.2%}|{r['recall']:.2%}|{r['f1']:.2%}|{r['size_recall']['small']['recall']:.2%}|{r['crop_complete95_fraction']:.2%}|{result['latency'][str(size)]['mean_ms']:.2f} ms|")
    lines += ["", "## 判斷", ""]
    base, high = rows["768"], rows["1024"]
    small_change = high["size_recall"]["small"]["recall"] - base["size_recall"]["small"]["recall"]
    crop_change = high["crop_complete95_fraction"] - base["crop_complete95_fraction"]
    f1_change = high["f1"] - base["f1"]
    lines.append(f"- 小傷口 Recall 變化：{small_change:+.2%}；裁切完整率變化：{crop_change:+.2%}；F1 變化：{f1_change:+.2%}。")
    if small_change > 0 and crop_change >= 0 and f1_change >= -.01:
        lines.append("- 預先規則：推論解析度方向值得進入下一輪單因素訓練/切片評估，但這不等同於已證明泛化提升。")
    else:
        lines.append("- 預先規則：目前證據不足以支持把 1024 當作下一輪訓練方向；不因解析度增加就更換 App 模型。")
    lines += ["", "## 限制", "", "所有數字是同來源 development validation；只能診斷小目標假設，不能稱為獨立外部或臨床測試。完整 confidence 網格保留在 result.json，不以最高數字選擇模型。", "", f"test_images_used=0；封版檔案未變更：{result['sealed_unchanged']}。"]
    (out / "實驗結果報告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        for resolution in RESOLUTIONS:
            model = YOLO(str(WEIGHTS["D-Seg-03R"]))
            if model.task != "segment" or str(model.names[0]).lower() != "wound":
                raise ValueError("wrong frozen model task/class")
            settings = {**protocol["prediction"], "imgsz": resolution, "iou": protocol["nms_iou"]}
            for _ in range(3):
                predict(model, np.zeros((512, 512, 3), np.uint8), settings)
            rows_by_conf, latency = {c: [] for c in CONFIDENCES}, []
            for i, row in enumerate(cohort):
                rgb = np.asarray(Image.open(safe_path(BUNDLE, row["image"])).convert("RGB"))
                prediction, masks, elapsed = predict(model, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), settings)
                latency.append(elapsed)
                for c in CONFIDENCES:
                    rows_by_conf[c].append(assess(row, prediction, masks, c))
                if (i + 1) % 50 == 0:
                    print(f"imgsz={resolution}: {i + 1}/191 development images", flush=True)
            dest = out / f"imgsz_{resolution}"
            dest.mkdir()
            for c, rows in rows_by_conf.items():
                write_json(dest / f"metrics_conf{c:.2f}.json", rows)
                summary = summarize(rows)
                doc.setdefault("grid", []).append({"imgsz": resolution, "confidence": c, **summary})
                if c == .10:
                    doc["summaries"][str(resolution)] = summary
            doc["latency"][str(resolution)] = latency_summary(latency)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        doc["sealed_unchanged"] = snapshot_sealed() == before
        doc["weights_unchanged"] = sha(WEIGHTS["D-Seg-03R"]) == WEIGHT_HASHES["D-Seg-03R"]
        doc["status"] = "PASS_DEVELOPMENT_EVALUATION_ONLY"
        doc["finished_utc"] = now()
        if not doc["sealed_unchanged"] or not doc["weights_unchanged"]:
            raise ValueError("post-run integrity failure")
        write_json(out / "result.json", doc)
        write_report(out, doc)
        print(json.dumps({"status": doc["status"], "summaries": doc["summaries"],
                          "test_images_used": 0}), flush=True)
    except Exception as exc:
        write_json(out / "failure.json", {"error": repr(exc), "time": now(), "test_images_used": 0})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "execute"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    (make_protocol if args.mode == "prepare" else execute)(output_path(args.output))
