"""Fixed development acceptance for the authorized seed-42 FUSeg candidate.

Reuses the audited 191-image validation cohort and the existing crop/matching
implementation. Does not calibrate thresholds or open any test dataset.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from .audit_project import ROOT, read_json, sha, snapshot_sealed, write_json
from .fuseg_warmup_experiment import update_status, verify_inventory
from . import localization_benchmark as loc

REFERENCE = ROOT / "outputs/localization_benchmark_20260914"
MINIMUM_PERCENT = {"precision": 87.18, "recall": 84.65, "f1": 85.89,
                   "crop_complete95_fraction": 90.32}


def decide(summary, latency):
    # The user's published minima have two percentage decimal places. Compare at
    # that precision, so the identical baseline cannot fail due to roundoff.
    checks = {name: summary.get(name) is not None and np.isfinite(summary[name])
              and round(100 * summary[name], 2) >= minimum
              for name, minimum in MINIMUM_PERCENT.items()}
    checks["gpu_mean_latency"] = bool(np.isfinite(latency.get("mean_ms", float("nan")))
                                      and latency["mean_ms"] <= 50)
    return {"passed": bool(all(checks.values())), "checks": checks,
            "minima_percent": MINIMUM_PERCENT, "latency_max_ms": 50,
            "comparison_precision": "published percentages rounded to 2 decimals"}


def prepare_gate_inputs():
    reference = read_json(REFERENCE / "protocol.json")
    if reference["primary"] != {"confidence": .1, "nms_iou": .7, "bbox_match_iou": .5}:
        raise ValueError("fixed development operating point changed")
    if sha(REFERENCE / "cohort.json") != reference["cohort_sha256"]:
        raise ValueError("validation cohort changed")
    cohort = read_json(REFERENCE / "cohort.json")
    if len(cohort) != 191 or any(r["split"] != "val" for r in cohort):
        raise ValueError("expected exactly 191 validation images")
    paths = [Path(__file__), Path(loc.__file__), ROOT / "woundcare_inference.py",
             REFERENCE / "cohort.json", REFERENCE / "protocol.json", REFERENCE / "result.json",
             loc.BUNDLE / "manifest.json"]
    for key in (str(ROOT / "woundcare_inference.py"), str(Path(loc.__file__))):
        if sha(Path(key)) != reference["pins"][key]:
            raise ValueError("original matching or crop implementation changed")
    return {"test_images_used": 0, "primary": reference["primary"],
            "prediction": reference["prediction"], "minima_percent": MINIMUM_PERCENT,
            "latency_max_ms": 50, "cohort_sha256": reference["cohort_sha256"],
            "pins": {str(p): sha(p) for p in paths}}


def execute_gate(formal: Path):
    import torch
    from ultralytics import YOLO
    completed = read_json(formal / "result.json")
    settings = read_json(formal / "development_gate_inputs.json")
    before = read_json(formal / "sealed_before.json")
    for path, digest in settings["pins"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"development input changed: {Path(path).name}")
    if snapshot_sealed() != before or completed["test_images_used"] != 0:
        raise ValueError("development safety gate failed")
    if not torch.cuda.is_available():
        raise RuntimeError("declared GPU unavailable")
    verify_inventory(loc.BUNDLE, read_json(loc.BUNDLE / "manifest.json"))
    cohort = read_json(REFERENCE / "cohort.json")
    training_manifest = read_json(ROOT / "outputs/isic_fuseg_pretrain_smoke_20260914/fuseg_manifest.json")
    val_hashes = {r["image_sha256"] for r in training_manifest if r["split"] == "val"}
    if {r["image_sha256"] for r in cohort} != val_hashes:
        raise ValueError("candidate and baseline validation cohorts differ")
    for row in cohort:
        mask = (ROOT / row["source_mask"]).resolve()
        if any(part.lower() in {"test", "testing", "blind_test"} for part in mask.parts):
            raise ValueError("forbidden mask source")
        if sha(mask) != row["mask_sha256"]:
            raise ValueError("validation mask changed")
    checkpoint = Path(completed["fuseg"]["checkpoint"])
    if sha(checkpoint) != completed["fuseg"]["checkpoint_sha256"]:
        raise ValueError("candidate checkpoint changed")
    output = formal / "development_gate"
    output.mkdir(exist_ok=False)
    write_json(output / "protocol.json", {**settings, "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha(checkpoint), "validation_images": 191,
               "class": "Wound", "test_images_used": 0})
    update_status(formal, status="EVALUATING_DEVELOPMENT", stage="FIXED_POINT_GATE")
    model = YOLO(str(checkpoint))
    if model.task != "segment" or model.names != {0: "Wound"}:
        raise ValueError("candidate is not a single-class Wound segmenter")
    predict = {**settings["prediction"], "iou": settings["primary"]["nms_iou"]}
    for _ in range(3):
        loc.predict_materialized(model, np.zeros((512, 512, 3), np.uint8), predict)
    rows, elapsed = [], []
    for row in cohort:
        with Image.open(loc.safe_path(loc.BUNDLE, row["image"])) as im:
            image = np.asarray(im.convert("RGB"))
        result, masks, milliseconds = loc.predict_materialized(model, image, predict)
        rows.append(loc.assess(row, result, masks, settings["primary"]["confidence"]))
        elapsed.append(milliseconds)
    summary, latency = loc.summarize(rows), loc.latency_summary(elapsed)
    decision = decide(summary, latency)
    if snapshot_sealed() != before:
        raise RuntimeError("sealed files changed during development gate")
    baseline = read_json(REFERENCE / "result.json")["primary"]["D-Seg-03"]
    report = {"status": "PASS_DEVELOPMENT_GATE" if decision["passed"] else "FAIL_DEVELOPMENT_GATE",
              "test_images_used": 0, "blind_test_used": False, "sealed_unchanged": True,
              "candidate": summary, "baseline": baseline, "gpu_latency": latency,
              "decision": decision, "five_seeds_started": False,
              "next_action": "five_seed_stability_eligible" if decision["passed"] else "stop_and_analyze_errors"}
    write_json(output / "predictions.json", rows)
    write_json(output / "result.json", report)
    loc.error_panels(output, cohort, rows)
    lines = ["# ISIC → FUSeg 正式 seed=42 開發門檻報告", "", f"狀態：{report['status']}", "",
             "固定 confidence=0.10、NMS IoU=0.70、bbox 匹配 IoU=0.50；191 張 FUSeg validation。",
             "", "|指標|候選|既有基準|最低門檻|", "|---|---:|---:|---:|"]
    for key, minimum in MINIMUM_PERCENT.items():
        value = f"{summary[key]:.2%}" if summary[key] is not None else "無法計算"
        lines.append(f"|{key}|{value}|{baseline[key]:.2%}|{minimum:.2f}%|")
    lines += ["", f"GPU batch=1 平均定位耗時：{latency['mean_ms']:.2f} ms，門檻 ≤50 ms。",
              "裁切成功：保留 ≥95% 傷口像素；所有正樣本均納入分母，無 ROI 視為失敗。",
              "", "使用過的 development validation 無法取代新來源測試；尚無跨來源通過證據。",
              "test_images_used=0；五種子尚未啟動；未替換 App 權重。",
              "", "下一步：" + ("可進入五種子穩定性評估。" if decision["passed"] else "停止候選擴跑並分析 error_cases.json 與 worst_crop_cases.png。")]
    (output / "實驗結果報告.md").write_text("\n".join(lines), encoding="utf-8")
    update_status(formal, status=report["status"], stage="DEVELOPMENT_GATE_COMPLETE", gate_result=str(output / "result.json"))
    return report
