"""Explicit opt-in local candidate smoke: synthetic arrays only, never test data.

This loads known local weights but neither imports backend_main nor mutates the
App model configuration/database. Success is compatibility, NOT model accuracy.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if (ROOT / "outputs").resolve() not in output.parents:
        raise ValueError("output must be in workspace outputs")
    output.mkdir(parents=True, exist_ok=False)
    bundle = ROOT / "outputs/fuseg_warmup_revision_20260914"
    report = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    segmentation = Path(report["checkpoint"])
    classification = ROOT / "experiments/results/raw/C-Arch-05_yolov8n_cls_s42_f0/weights/best.pt"
    expected_classifier = "45970709b3b1be9d2fb33b05d84f32f71c8dbe1977c34ff0c964c9743b834d9b"
    outcome = {"status": "NOT_RUN", "synthetic_only": True, "dataset_images_used": 0, "test_images_used": 0,
               "app_model_replaced": False, "clinical_database_opened": False,
               "scope": "candidate interface compatibility; no accuracy, specificity or clinical validation claim"}
    try:
        if digest(segmentation) != report["checkpoint_sha256"] or digest(classification) != expected_classifier:
            raise ValueError("candidate checkpoint identity changed")
        from experiments.review_v2.audit_project import snapshot_sealed, read_json
        sealed = snapshot_sealed()
        if sealed != read_json(bundle / "sealed_before.json"):
            raise ValueError("historical sealed metadata changed")
        import numpy as np
        import torch
        from ultralytics import YOLO
        from woundcare_inference import CascadeConfig, infer_segmentation_cascade

        torch.set_num_threads(4)
        segmenter, classifier = YOLO(str(segmentation)), YOLO(str(classification))
        expected_names = ["Abrasions", "Bruises", "Burns", "Cut", "Ingrown_nails", "Laceration", "Stab_wound"]
        if segmenter.task != "segment" or segmenter.model.yaml.get("scale") != "m" or list(segmenter.names.values()) != ["Wound"]:
            raise ValueError("segmenter task, scale or class mapping mismatch")
        if classifier.task != "classify" or list(classifier.names.values()) != expected_names:
            raise ValueError("classifier task or seven-class mapping mismatch")
        # Retain existing App thresholds. This is not threshold tuning.
        config = CascadeConfig(segmenter_confidence=.10, classifier_confidence=.60,
                               segmenter_imgsz=768, classifier_imgsz=224, device="cpu",
                               classification_source="full_image", segmenter_iou=.70)
        samples = [np.zeros((96, 128, 3), dtype=np.uint8),
                   np.random.default_rng(42).integers(0, 256, size=(192, 256, 3), dtype=np.uint8)]
        results = []
        for index, sample in enumerate(samples):
            start = time.perf_counter()
            value = infer_segmentation_cascade(sample, segmenter, classifier, config)
            elapsed = (time.perf_counter() - start) * 1000
            json.dumps(value, allow_nan=False)
            if value["mode"] != "full_image_primary":
                raise ValueError("default full-image classification contract changed")
            if value["label"] is not None and value["label"] not in expected_names:
                raise ValueError("unexpected classifier label")
            if not math.isfinite(elapsed):
                raise ValueError("nonfinite elapsed time")
            results.append({"fixture": "zero_array" if index == 0 else "seeded_noise", "shape": list(sample.shape),
                            "elapsed_ms_including_first_call_setup": round(elapsed, 2), "output": value})
        if digest(segmentation) != report["checkpoint_sha256"] or digest(classification) != expected_classifier or snapshot_sealed() != sealed:
            raise ValueError("weights or sealed metadata changed during smoke")
        outcome.update(status="PASS_SYNTHETIC_COMPATIBILITY_ONLY", candidate_config=asdict(config),
                       segmentation_sha256=report["checkpoint_sha256"], classification_sha256=expected_classifier,
                       segmenter_names=segmenter.names, classifier_names=classifier.names, results=results,
                       limitations=["Closed-set seven-class predictions on synthetic arrays are not diagnoses.",
                                    "No healthy/non-wound rejection claim; no clinical deployment approval.",
                                    "CPU fixture timings include first-call setup and are not a representative latency benchmark.",
                                    "No real image upload, independent cohort test or threshold selection performed."])
    except Exception as exc:
        outcome.update(status="FAILED_COMPATIBILITY", error_type=type(exc).__name__, error=str(exc))
    (output / "integration.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(outcome, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if outcome["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
