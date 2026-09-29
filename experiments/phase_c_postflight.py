"""Read-only saved-evidence verification; no model loading or inference.

The original execution verifier required boolean dtype even for empty arrays.
The pinned historical materializer returns float64 for np.asarray([]), with no
elements. This verifier accepts that exact zero-instance representation while
continuing to require boolean masks for every nonempty prediction. No execution
code, frozen protocol, saved prediction or metric implementation is modified.
"""
from pathlib import Path
import math
import json
import numpy as np

from experiments.phase_c_execution import (
    PhaseCError, PINNED_PROTOCOL_SHA256, PINNED_MANIFEST_SHA256,
    OUTPUT_REL, PROTOCOL_REL, read_json, check_digest, assert_relative_admitted,
    validate_result_integrity, choose_gate_status, write_json, now,
)
from experiments.phase_c0 import sha256_file


def verify_mask_representation(masks, count):
    if masks.shape != (count, 512, 512):
        raise PhaseCError("FAIL_RESULT_INTEGRITY: mask shape/count")
    if count == 0 and masks.size == 0:
        return "EMPTY_HISTORICAL_ARRAY"
    if masks.dtype != np.bool_:
        raise PhaseCError("FAIL_RESULT_INTEGRITY: nonempty mask must be boolean")
    return "BOOLEAN_MASKS"


def verify(root):
    out = root / OUTPUT_REL
    check_digest(root / PROTOCOL_REL, PINNED_PROTOCOL_SHA256, "FAIL_PROTOCOL_MUTATED")
    protocol = read_json(root / PROTOCOL_REL)
    manifest_path = root / protocol["validation_manifest"]["path"]
    check_digest(manifest_path, PINNED_MANIFEST_SHA256, "FAIL_VALIDATION_MANIFEST_MUTATED")
    manifest = read_json(manifest_path)
    result = read_json(out / "result.json")
    rows = read_json(out / "per_image_predictions.json")
    completion = read_json(out / "execution_completion.json")
    execution = read_json(out / "execution_manifest.json")
    check_digest(out / "result.json", completion["result_sha256"], "FAIL_RESULT_INTEGRITY")
    check_digest(out / "per_image_predictions.json", completion["per_image_predictions_sha256"], "FAIL_RESULT_INTEGRITY")
    for relative, expected in execution["protected_before"].items():
        check_digest(root / relative, expected, "FAIL_HISTORICAL_ARTIFACT_CHANGED")
    for runtime in execution["runtime_file_hashes"]:
        for kind in ("image", "mask", "label"):
            check_digest(root / runtime[kind + "_path"], runtime[kind + "_sha256"], "FAIL_RUNTIME_DATASET_HASH_MISMATCH")
    # Verify the frozen metric before importing the pure assessment functions.
    check_digest(root / protocol["metric_implementation"]["path"], protocol["metric_implementation"]["sha256"], "FAIL_METRIC_IMPLEMENTATION_CHANGED")
    from experiments.review_v2 import localization_benchmark as loc
    from experiments.review_v2.isic_fuseg_gate import decide

    summary = result["candidate"]
    integrity = validate_result_integrity(rows, summary, [r["sample_id"] for r in manifest["samples"]])
    samples = {r["sample_id"]: r for r in manifest["samples"]}
    originals = {r["image_id"]: r for r in read_json(root / protocol["provenance"]["historical_cohort_path"])}
    empty_ids = []
    for i, row in enumerate(rows):
        if read_json(out / "per_image" / f"{i:03d}.json") != row:
            raise PhaseCError("FAIL_RESULT_INTEGRITY: per-image evidence differs")
        sample = samples[row["sample_id"]]
        if row["image_sha256"] != sample["image_sha256"] or row["mask_sha256"] != sample["mask_sha256"]:
            raise PhaseCError("FAIL_RESULT_INTEGRITY: image/mask identity")
        mask_path = assert_relative_admitted(out, row["prediction_mask_reference"], out / "masks")
        check_digest(mask_path, row["prediction_mask_sha256"], "FAIL_RESULT_INTEGRITY")
        with np.load(mask_path, allow_pickle=False) as archive:
            masks, scores, boxes, indices = (archive[k] for k in ("masks", "confidences", "boxes", "retained_indices"))
            if verify_mask_representation(masks, row["floor_prediction_count"]) == "EMPTY_HISTORICAL_ARRAY":
                empty_ids.append(row["sample_id"])
            if (not np.array_equal(scores, np.asarray(row["floor_confidences"]))
                    or not np.array_equal(boxes, np.asarray(row["floor_bboxes"]).reshape(-1, 4))
                    or indices.tolist() != row["retained_mask_indices"]
                    or indices.tolist() != np.flatnonzero(scores >= protocol["operating_point"]["confidence"]).tolist()
                    or not np.array_equal(boxes[indices], np.asarray(row["pred_boxes"]).reshape(-1, 4))):
                raise PhaseCError("FAIL_RESULT_INTEGRITY: saved mask/box/confidence evidence")
            gt_boxes, gt_mask = loc.ground_truth(originals[row["sample_id"]])
            matching = loc.match_boxes(gt_boxes, boxes[indices], scores[indices])
            if any(matching[k] != row[k] for k in ("tp", "fp", "fn", "pairs")):
                raise PhaseCError("FAIL_RESULT_INTEGRITY: matching reconstruction")
            union = np.any(masks[indices], axis=0) if len(indices) else np.zeros((512, 512), bool)
            pixels = loc.pixel_metrics(gt_mask, union, row["crop"])
            for key, value in pixels.items():
                if row[key] != value:
                    raise PhaseCError("FAIL_RESULT_INTEGRITY: pixel reconstruction " + key)
            crop = row["crop"]
            retained = int(gt_mask[crop[1]:crop[3], crop[0]:crop[2]].sum()) if crop else 0
            if retained != row["retained_gt_wound_pixels"]:
                raise PhaseCError("FAIL_RESULT_INTEGRITY: retained pixel reconstruction")
            for instance in row["per_instance_gt"]:
                if instance["size_group"] != loc.size_name(gt_boxes[instance["index"]]):
                    raise PhaseCError("FAIL_RESULT_INTEGRITY: size group")
    reconstructed = loc.summarize(rows)
    if reconstructed != summary:
        raise PhaseCError("FAIL_RESULT_INTEGRITY: aggregate reconstruction")
    latency = loc.latency_summary([r["inference_latency_ms"] for r in rows])
    if latency != result["gpu_latency"]:
        raise PhaseCError("FAIL_RESULT_INTEGRITY: latency summary")
    decision = decide(summary, latency)
    performance = {k: decision["checks"][k] for k in protocol["acceptance_gate"]["minima_percent"]}
    if result["gate_result"] != choose_gate_status(performance, decision["checks"]["gpu_mean_latency"], result["latency_comparability"] == "VERIFIED"):
        raise PhaseCError("FAIL_RESULT_INTEGRITY: gate decision")
    for flat, nested in (("TP", "tp"), ("FP", "fp"), ("FN", "fn"), ("Precision", "precision"),
                         ("Recall", "recall"), ("F1", "f1"), ("crop_complete_fraction", "crop_complete95_fraction")):
        if result[flat] != summary[nested]:
            raise PhaseCError("FAIL_RESULT_INTEGRITY: reported metric alias")
    if result["development_images_inferred"] != 191 or any(result[k] for k in (
            "training_performed", "locked_test_used", "CO2Wounds_used", "multi_seed_started", "app_model_replaced")):
        raise PhaseCError("FAIL_RESULT_INTEGRITY: execution boundary")
    return {**integrity, "verified_utc": now(), "prediction_mask_files_verified": len(rows),
            "runtime_image_mask_label_hashes_reverified": len(rows),
            "matching_and_pixel_metrics_reconstructed": True, "protected_artifacts_unchanged": len(execution["protected_before"]),
            "verification_model_loads": 0, "verification_inference_calls": 0,
            "empty_historical_mask_arrays": empty_ids,
            "initial_checker_issue": "Three zero-instance arrays had the historical float64 empty dtype; the initial checker incorrectly required boolean even when no mask pixels existed. All other checks passed. Nonempty float masks remain forbidden.",
            "execution_code_or_results_changed": False,
            "postflight_verifier_sha256": sha256_file(Path(__file__))}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    verified = verify(root)
    write_json(root / OUTPUT_REL / "postflight_verification.json", verified)
    print(json.dumps(verified, ensure_ascii=False, indent=2))
