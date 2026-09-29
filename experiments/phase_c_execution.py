"""One authorized color-contract reevaluation; frozen metrics are imported intact.

No historical execute/prepare entrypoint is called: those write old directories
or scan broader inventories. Only the existing prediction, assessment, summary,
latency and decision functions are reused. There is no resume or retry mode.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.data_roles import Role, require_source_role, validate_model_input_contract
from experiments.phase_c0 import sha256_file, verify_historical_gate_settings

PINNED_PROTOCOL_SHA256 = "16718f5956548e002320ebd2ade83c5ecafdf77e7c79dc1f1d2b63818b7e6df3"
PINNED_CHECKPOINT_SHA256 = "cc2955d088928dc00c90d0ba8cab0aefeb49d4cc45da172b030c150ab0dce97a"
PINNED_MANIFEST_SHA256 = "ea58c4e7d36b699d2f1e232d23fea8df9a6010674d08769e5a2ca3aca66ab069"
OUTPUT_REL = "experiments/results/isic_fuseg_colorfix_v1"
PROTOCOL_REL = "experiments/protocols/ISIC_FUSEG_COLORFIX_V1_protocol.json"
BUNDLE_REL = "outputs/fuseg_warmup_revision_20260914"
MASK_ROOT_REL = "official_detection_sources_20260812/fuseg/repository/data/Foot Ulcer Segmentation Challenge/validation/labels"
SIZES = ("small", "medium", "large")


class PhaseCError(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def check_digest(path, expected, failure):
    if not path.is_file() or sha256_file(path) != expected:
        raise PhaseCError(f"{failure}: {path}")


def assert_relative_admitted(root, relative, admitted_root):
    raw = Path(relative)
    parts = {part.casefold() for part in raw.parts}
    if (raw.is_absolute() or ".." in parts or parts.intersection({"test", "testing", "blind_test", "locked_test"})
            or "co2" in str(raw).casefold()):
        raise PhaseCError("FAIL_SOURCE_ROLE: forbidden sample path")
    resolved = (root / raw).resolve()
    if not resolved.is_relative_to(admitted_root.resolve()):
        raise PhaseCError("FAIL_SOURCE_ROLE: sample outside admitted validation root")
    return resolved


def preflight(root):
    # The first read verifies the user-authorized digest, before imports/models/data.
    protocol_path = root / PROTOCOL_REL
    check_digest(protocol_path, PINNED_PROTOCOL_SHA256, "FAIL_PROTOCOL_MUTATED")
    if (root / OUTPUT_REL).exists():
        raise PhaseCError("FAIL_OUTPUT_ALREADY_EXISTS")
    sidecar = protocol_path.with_suffix(".sha256")
    if sidecar.read_text(encoding="ascii").split()[0] != PINNED_PROTOCOL_SHA256:
        raise PhaseCError("FAIL_PROTOCOL_MUTATED: sidecar")
    p = read_json(protocol_path)
    if (p["experiment_id"] != "ISIC_FUSEG_COLORFIX_V1"
            or p["allowed_semantic_change"] != "RGB_BGR_INPUT_CONTRACT_ONLY"
            or p["protocol_status"] != "FROZEN_BEFORE_EXECUTION"):
        raise PhaseCError("FAIL_PROTOCOL_MUTATED: identity")
    if p["checkpoint"]["sha256"] != PINNED_CHECKPOINT_SHA256:
        raise PhaseCError("FAIL_CHECKPOINT_IDENTITY_MISMATCH")
    check_digest(root / p["checkpoint"]["path"], PINNED_CHECKPOINT_SHA256,
                 "FAIL_CHECKPOINT_IDENTITY_MISMATCH")
    if (root / p["checkpoint"]["path"]).stat().st_size != p["checkpoint"]["size_bytes"]:
        raise PhaseCError("FAIL_CHECKPOINT_IDENTITY_MISMATCH: size")
    mpath = root / p["validation_manifest"]["path"]
    check_digest(mpath, PINNED_MANIFEST_SHA256, "FAIL_VALIDATION_MANIFEST_MUTATED")
    manifest = read_json(mpath)
    if (manifest["sample_count"] != 191 or len(manifest["samples"]) != 191
            or manifest["source_role"] != "DEVELOPMENT_VALIDATION"):
        raise PhaseCError("FAIL_VALIDATION_COHORT")
    if require_source_role(p["source"]["source_id"], "development_evaluate") != Role.DEVELOPMENT_VALIDATION:
        raise PhaseCError("FAIL_SOURCE_ROLE")
    pinned = {
        p["source"]["evidence_path"]: p["source"]["evidence_sha256"],
        p["metric_implementation"]["path"]: p["metric_implementation"]["sha256"],
        p["historical_reference"]["result_path"]: p["historical_reference"]["result_sha256"],
        p["color_contract"]["code_diff_audit_path"]: p["provenance"]["code_diff_audit_sha256"],
        "experiments/review_v2/isic_fuseg_gate.py": p["provenance"]["corrected_gate_script_sha256"],
    }
    for key in ("historical_cohort", "historical_protocol", "training_manifest", "crop_implementation"):
        pinned[p["provenance"][key + "_path"]] = p["provenance"][key + "_sha256"]
    for relative, digest in pinned.items():
        check_digest(root / relative, digest, "FAIL_FROZEN_IMPLEMENTATION_OR_EVIDENCE_CHANGED")
    historical_settings = read_json(root / p["provenance"]["historical_protocol_path"])
    verify_historical_gate_settings(historical_settings)
    if (p["prediction"] != historical_settings["prediction"]
            or p["operating_point"] != historical_settings["primary"]
            or p["acceptance_gate"]["minima_percent"] != historical_settings["minima_percent"]):
        raise PhaseCError("FAIL_FIXED_OPERATING_POINT")
    cohort = read_json(root / p["provenance"]["historical_cohort_path"])
    by_id = {r["image_id"]: r for r in cohort}
    if len(cohort) != 191 or len(by_id) != 191:
        raise PhaseCError("FAIL_VALIDATION_COHORT")
    runtime_records = []
    for sample in manifest["samples"]:
        if (sample["source_dataset"], sample["source_role"], sample["split"]) != (
                "FUSeg", "DEVELOPMENT_VALIDATION", "val"):
            raise PhaseCError("FAIL_SOURCE_ROLE")
        original = by_id[sample["sample_id"]]
        if (original["source"], original["split"]) != ("FUSeg", "val"):
            raise PhaseCError("FAIL_SOURCE_ROLE")
        image_rel = f"{BUNDLE_REL}/{original['image']}"
        label_rel = f"{BUNDLE_REL}/{original['label']}"
        if image_rel != sample["relative_image_path"] or original["source_mask"] != sample["relative_mask_path"]:
            raise PhaseCError("FAIL_VALIDATION_COHORT: path")
        record = {"sample_id": sample["sample_id"], "status": "PASS"}
        for kind, relative, admitted in (
            ("image", image_rel, root / BUNDLE_REL / "dataset/images/val"),
            ("label", label_rel, root / BUNDLE_REL / "dataset/labels/val"),
            ("mask", sample["relative_mask_path"], root / MASK_ROOT_REL),
        ):
            path = assert_relative_admitted(root, relative, admitted)
            require_source_role("FUSeg", "development_evaluate", path=path)
            expected = sample[kind + "_sha256"]
            if expected != original[kind + "_sha256"]:
                raise PhaseCError("FAIL_VALIDATION_COHORT: hash identity")
            check_digest(path, expected, "FAIL_RUNTIME_DATASET_HASH_MISMATCH")
            record[kind + "_path"] = relative
            record[kind + "_sha256"] = expected
        runtime_records.append(record)
    if {r["sample_id"] for r in runtime_records} != set(by_id):
        raise PhaseCError("FAIL_VALIDATION_COHORT: identities")
    return p, cohort, runtime_records


def protected_snapshot(root, protocol):
    # Explicit artifact directories only; never enumerate a test dataset.
    paths = [root / protocol["checkpoint"]["path"], root / PROTOCOL_REL,
             (root / PROTOCOL_REL).with_suffix(".sha256"),
             root / protocol["validation_manifest"]["path"],
             root / "woundcare_inference.py", root / "experiments/data_roles.py",
             root / "experiments/review_v2/guards.py", root / "experiments/phase_c0.py",
             root / "experiments/phase_c_execution.py",
             root / "experiments/review_v2/isic_fuseg_gate.py",
             root / protocol["metric_implementation"]["path"],
             root / "experiments/results/segmentation/D-Seg-03_yolo11m_fuseg_only_s42/weights/best.pt",
             root / "docs/CLASSIFICATION_RESULT_FREEZE_20260920.md",
             root / "docs/PHASE_C0_PREFLIGHT_REPORT_20260921.md",
             root / "experiments/results/statistics/C-Arch-05-MS_phase_b1_final_summary.json"]
    formal = root / "outputs/isic_fuseg_formal_seed42_20260914"
    paths += [formal / name for name in ("protocol.json", "result.json", "development_gate_inputs.json")]
    paths += [p for p in (formal / "development_gate").iterdir() if p.is_file()]
    paths += list((root / "experiments/results/tables").glob("Table*"))
    historical_co2 = root / "outputs/segmentation/co2wounds_external_test_20260831"
    paths += [p for p in historical_co2.iterdir() if p.is_file() and p.suffix in {".json", ".csv", ".md"}]
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(set(paths)) if p.is_file()}


def choose_gate_status(performance_checks, latency_threshold_passed, comparable):
    if not all(performance_checks.values()):
        return "FAIL_DEVELOPMENT_GATE"
    if not comparable:
        return "PARTIAL_LATENCY_UNVERIFIED"
    return "PASS_DEVELOPMENT_GATE" if latency_threshold_passed else "FAIL_DEVELOPMENT_GATE"


def validate_result_integrity(rows, summary, expected_ids):
    def require(condition, reason):
        if not condition:
            raise PhaseCError(f"FAIL_RESULT_INTEGRITY: {reason}")

    def equal(actual, expected, field):
        require(actual is None if expected is None else isinstance(actual, (int, float))
                and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), field)

    require(len(rows) == len(expected_ids) == len(set(expected_ids)), "cohort size")
    require(len({r["sample_id"] for r in rows}) == len(rows), "duplicate sample")
    require({r["sample_id"] for r in rows} == set(expected_ids), "cohort identities")
    for r in rows:
        require(all(isinstance(r[k], int) and r[k] >= 0 for k in ("tp", "fp", "fn")), "nonnegative counts")
        require(r["tp"] + r["fn"] == r["num_gt_instances"], "row TP+FN")
        require(r["tp"] + r["fp"] == r["num_predictions"], "row TP+FP")
        require(sum(r["size_support"].values()) == r["num_gt_instances"], "size support")
        require(sum(r["size_matched"].values()) == r["tp"], "size matched")
        require(all(0 <= r["size_matched"][s] <= r["size_support"][s] for s in SIZES), "size bounds")
        require(0 <= r["retained_gt_wound_pixels"] <= r["gt_pixels"], "retained pixels")
        if r["gt_pixels"]:
            equal(r["crop_coverage"], r["retained_gt_wound_pixels"] / r["gt_pixels"], "coverage")
            require(r["crop_complete95"] == (r["crop_coverage"] >= .95), "crop success")
    positive = [r for r in rows if r["gt_pixels"] > 0]
    negative = [r for r in rows if r["gt_pixels"] == 0]
    tp, fp, fn = [sum(r[k] for r in rows) for k in ("tp", "fp", "fn")]
    for key, expected in {
        "images": len(rows), "positive_images": len(positive), "negative_images": len(negative),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "crop_complete95_images": sum(r["crop_complete95"] for r in positive),
        "crop_complete95_fraction": sum(r["crop_complete95"] for r in positive)/len(positive) if positive else None,
        "positive_without_roi": sum(r["crop"] is None for r in positive),
        "negative_images_with_predictions": sum(r["num_predictions"] > 0 for r in negative),
    }.items():
        equal(summary[key], expected, key)
    for aggregate, key in (("mask_iou_positive_mean", "mask_iou"), ("mask_dice_positive_mean", "mask_dice"),
                           ("crop_coverage_positive_mean", "crop_coverage")):
        equal(summary[aggregate], sum(r[key] for r in positive)/len(positive) if positive else None, aggregate)
    for size in SIZES:
        support = sum(r["size_support"][size] for r in rows)
        hit = sum(r["size_matched"][size] for r in rows)
        stat = summary["size_recall"][size]
        equal(stat["gt"], support, size + " support")
        equal(stat["matched"], hit, size + " tp")
        equal(stat["recall"], hit/support if support else None, size + " recall")
    require(tp + fn == sum(r["num_gt_instances"] for r in rows), "aggregate GT denominator")
    return {"status": "PASS", "num_images": len(rows), "num_gt_instances": tp + fn,
            "metrics_recomputed_from_saved_counts": True, "size_support_equals_total_gt": True}


def runtime_environment(root):
    import torch
    if not torch.cuda.is_available():
        raise PhaseCError("FAIL_GPU_UNAVAILABLE: no CPU fallback")
    software = {k: importlib.metadata.version(k) for k in ("torch", "ultralytics", "numpy", "pillow", "opencv-python")}
    historical = read_json(root / "outputs/localization_benchmark_20260914/protocol.json")
    if software != historical["software"]:
        raise PhaseCError("FAIL_SOFTWARE_ENVIRONMENT_CHANGED")
    return {"device": "0", "gpu_name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__, "ultralytics_version": software["ultralytics"],
            "python_version": platform.python_version(), "software": software,
            "latency_comparability": "NOT_VERIFIED",
            "latency_comparability_reason": "Same software versions and timing implementation; historical gate lacks GPU/driver/load runtime metadata. Matching GPU family in the historical training log is supporting evidence only.",
            "latency_method": "3 synthetic warmups; GPU batch=1; includes prediction and CPU box/mask materialization; excludes disk I/O and diagnostics"}


def error_analysis(rows):
    return {
        "size_misses": {s: {"classification": "CONFIRMED_FROM_GT", "fn": sum(r["size_support"][s]-r["size_matched"][s] for r in rows)} for s in SIZES},
        "positive_no_roi": {"classification": "OBSERVED_FAILURE", "sample_ids": [r["sample_id"] for r in rows if r["gt_pixels"] and r["crop"] is None]},
        "incomplete_crop_with_roi": {"classification": "OBSERVED_FAILURE", "sample_ids": [r["sample_id"] for r in rows if r["gt_pixels"] and r["crop"] and not r["crop_complete95"]]},
        "negative_image_false_positives": {"classification": "CONFIRMED_FROM_GT", "sample_ids": [r["sample_id"] for r in rows if not r["gt_pixels"] and r["fp"]]},
        "multiple_wound_images_with_misses": {"classification": "CONFIRMED_FROM_GT", "sample_ids": [r["sample_id"] for r in rows if r["num_gt_instances"] > 1 and r["fn"]]},
        "possible_causes": {"classification": "SUSPECTED_CAUSE", "items": ["low contrast", "ambiguous boundaries", "edge truncation", "annotation ambiguity"], "note": "Not established by counts; requires visual review. No causal claim."},
    }


def build_comparison(historical, corrected):
    labels = {"precision": "Precision", "recall": "Recall", "f1": "F1",
              "crop_complete95_fraction": "Crop complete >=95%", "mask_iou_positive_mean": "Positive mean union IoU",
              "mask_dice_positive_mean": "Positive mean union Dice"}
    rows = [{"metric": k, "label": label, "historical": historical[k], "corrected": corrected[k],
             "absolute_delta": corrected[k]-historical[k], "delta_percentage_points": 100*(corrected[k]-historical[k])} for k, label in labels.items()]
    text = ["# Controlled RGB/BGR evaluation correction", "", "Historical value contains confirmed RGB/BGR input-contract defect.", "",
            "The comparison quantifies evaluation correction impact, not model/training improvement.", "",
            "| Metric | Historical defective | Corrected | Absolute delta (percentage points) |", "|---|---:|---:|---:|"]
    text += [f"|{r['label']}|{r['historical']:.2%}|{r['corrected']:.2%}|{r['delta_percentage_points']:+.2f}|" for r in rows]
    return rows, "\n".join(text) + "\n"


def verify_saved_execution(root):
    """Read-only verification of saved predictions; never loads a model/image."""
    import numpy as np
    out = root / OUTPUT_REL
    check_digest(root / PROTOCOL_REL, PINNED_PROTOCOL_SHA256, "FAIL_PROTOCOL_MUTATED")
    protocol = read_json(root / PROTOCOL_REL)
    manifest_path = root / protocol["validation_manifest"]["path"]
    check_digest(manifest_path, PINNED_MANIFEST_SHA256, "FAIL_VALIDATION_MANIFEST_MUTATED")
    manifest = read_json(manifest_path)
    result = read_json(out / "result.json")
    rows = read_json(out / "per_image_predictions.json")
    completion = read_json(out / "execution_completion.json")
    check_digest(out / "result.json", completion["result_sha256"], "FAIL_RESULT_INTEGRITY")
    check_digest(out / "per_image_predictions.json", completion["per_image_predictions_sha256"], "FAIL_RESULT_INTEGRITY")
    summary = result["candidate"]
    integrity = validate_result_integrity(rows, summary, [r["sample_id"] for r in manifest["samples"]])
    samples = {r["sample_id"]: r for r in manifest["samples"]}
    for i, row in enumerate(rows):
        if read_json(out / "per_image" / f"{i:03d}.json") != row:
            raise PhaseCError("FAIL_RESULT_INTEGRITY: per-image evidence differs")
        if row["image_sha256"] != samples[row["sample_id"]]["image_sha256"]:
            raise PhaseCError("FAIL_RESULT_INTEGRITY: image identity")
        mask_path = assert_relative_admitted(out, row["prediction_mask_reference"], out / "masks")
        check_digest(mask_path, row["prediction_mask_sha256"], "FAIL_RESULT_INTEGRITY")
        with np.load(mask_path, allow_pickle=False) as mask_file:
            masks, scores, boxes, indices = (mask_file[k] for k in ("masks", "confidences", "boxes", "retained_indices"))
            if (masks.shape != (row["floor_prediction_count"], 512, 512) or masks.dtype != np.bool_
                    or not np.array_equal(scores, np.asarray(row["floor_confidences"]))
                    or not np.array_equal(boxes, np.asarray(row["floor_bboxes"]).reshape(-1, 4))
                    or indices.tolist() != row["retained_mask_indices"]
                    or indices.tolist() != np.flatnonzero(scores >= protocol["operating_point"]["confidence"]).tolist()
                    or not np.array_equal(boxes[indices], np.asarray(row["pred_boxes"]).reshape(-1, 4))):
                raise PhaseCError("FAIL_RESULT_INTEGRITY: saved mask/box/confidence evidence")
    for flat, nested in (("TP", "tp"), ("FP", "fp"), ("FN", "fn"), ("Precision", "precision"),
                         ("Recall", "recall"), ("F1", "f1"), ("crop_complete_fraction", "crop_complete95_fraction")):
        if result[flat] != summary[nested]:
            raise PhaseCError("FAIL_RESULT_INTEGRITY: reported metric alias")
    if result["development_images_inferred"] != 191 or any(result[k] for k in (
            "training_performed", "locked_test_used", "CO2Wounds_used", "multi_seed_started", "app_model_replaced")):
        raise PhaseCError("FAIL_RESULT_INTEGRITY: execution boundary")
    return {**integrity, "prediction_mask_files_verified": len(rows), "read_only_verification": True}


def execute(root):
    p, cohort, runtime_records = preflight(root)
    environment = runtime_environment(root)
    protected = protected_snapshot(root, p)
    out = root / OUTPUT_REL
    out.mkdir(exist_ok=False)
    started = now()
    identity = {"experiment_id": p["experiment_id"], "execution_timestamp": started,
                "protocol_sha256": PINNED_PROTOCOL_SHA256, "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
                "validation_manifest_sha256": PINNED_MANIFEST_SHA256,
                "metric_script_sha256": p["metric_implementation"]["sha256"],
                "gate_script_sha256": p["provenance"]["corrected_gate_script_sha256"],
                "execution_adapter_sha256": sha256_file(Path(__file__)), **environment}
    write_json(out / "execution.lock", {**identity, "status": "RUNNING", "immutable_start_record": True})
    write_json(out / "execution_manifest.json", {**identity, "runtime_hashes_passed": len(runtime_records),
        "runtime_file_hashes": runtime_records, "protected_before": protected, "source_role": "DEVELOPMENT_VALIDATION",
        "prediction": {**p["prediction"], "iou": p["operating_point"]["nms_iou"]},
        "execution_adapter_changes": ["OUTPUT_VERSIONING", "SAFETY_GUARD", "LOGGING"],
        "inference_semantic_change": "RGB_BGR_INPUT_CONTRACT_ONLY", "test_images_used": 0, "CO2Wounds_used": False})
    rows = []
    attempted = 0
    try:
        import numpy as np
        from PIL import Image
        from ultralytics import YOLO
        from experiments.review_v2 import localization_benchmark as loc
        from experiments.review_v2.isic_fuseg_gate import decide

        # All file integrity and source checks above precede checkpoint loading.
        model = YOLO(str(root / p["checkpoint"]["path"]))
        if model.task != "segment" or model.names != {0: "Wound"}:
            raise PhaseCError("FAIL_CHECKPOINT_TASK_OR_CLASS")
        settings = {**p["prediction"], "iou": p["operating_point"]["nms_iou"]}
        for _ in range(3):
            loc.predict_materialized(model, np.zeros((512, 512, 3), np.uint8), settings)
        (out / "per_image").mkdir()
        (out / "masks").mkdir()
        for index, original in enumerate(cohort):
            image_path = loc.safe_path(loc.BUNDLE, original["image"])
            with Image.open(image_path) as image:
                if image.size != (512, 512):
                    raise PhaseCError("FAIL_RUNTIME_IMAGE_DIMENSIONS")
                bgr = validate_model_input_contract(image.convert("RGB"), source_type="PIL_RGB")
            attempted += 1
            prediction, masks, milliseconds = loc.predict_materialized(model, bgr, settings)
            row = loc.assess(original, prediction, masks, p["operating_point"]["confidence"])
            scores = prediction.boxes.conf.cpu().numpy()
            keep = np.flatnonzero(scores >= p["operating_point"]["confidence"])
            mask_rel = f"masks/{index:03d}.npz"
            with (out / mask_rel).open("xb") as handle:
                np.savez_compressed(handle, masks=masks, confidences=scores,
                                    boxes=prediction.boxes.xyxy.cpu().numpy(), retained_indices=keep)
            retained = int(round(row["crop_coverage"] * row["gt_pixels"])) if row["gt_pixels"] else 0
            gt_matched = {pair["gt"] for pair in row["pairs"]}
            row.update({"sample_id": original["image_id"], "num_gt_instances": len(row["gt_boxes"]),
                "num_predictions": len(row["pred_boxes"]), "matched_tp": row["tp"],
                "per_instance_gt": [{"index": i, "bbox": box, "size_group": loc.size_name(box),
                                     "matched": i in gt_matched} for i, box in enumerate(row["gt_boxes"])],
                "prediction_mask_reference": mask_rel, "prediction_mask_sha256": sha256_file(out / mask_rel),
                "retained_mask_indices": keep.tolist(), "floor_prediction_count": len(scores),
                "floor_confidences": scores.tolist(), "floor_bboxes": prediction.boxes.xyxy.cpu().numpy().tolist(),
                "retained_gt_wound_pixels": retained, "retained_fraction": row["crop_coverage"],
                "retained_pixels_derivation": "round(frozen crop_coverage * gt_pixels); diagnostic only",
                "no_roi": row["crop"] is None, "inference_latency_ms": milliseconds})
            write_json(out / "per_image" / f"{index:03d}.json", row)
            rows.append(row)
        write_json(out / "per_image_predictions.json", rows)
        # Reopen the saved evidence for integrity checks, without any more inference.
        saved = read_json(out / "per_image_predictions.json")
        summary = loc.summarize(saved)
        integrity = validate_result_integrity(saved, summary, [r["image_id"] for r in cohort])
        if sum(r["num_gt_instances"] for r in saved) != sum(r["instances"] for r in cohort):
            raise PhaseCError("FAIL_RESULT_INTEGRITY: original GT support")
        latency = loc.latency_summary([r["inference_latency_ms"] for r in saved])
        raw_decision = decide(summary, latency)
        performance = {k: raw_decision["checks"][k] for k in p["acceptance_gate"]["minima_percent"]}
        comparable = environment["latency_comparability"] == "VERIFIED"
        gate = choose_gate_status(performance, raw_decision["checks"]["gpu_mean_latency"], comparable)
        historical = read_json(root / p["historical_reference"]["result_path"])["candidate"]
        comparison, markdown = build_comparison(historical, summary)
        with (out / "historical_comparison.md").open("x", encoding="utf-8") as handle:
            handle.write(markdown)
        analysis = error_analysis(saved)
        write_json(out / "error_analysis.json", analysis)
        loc.error_panels(out, cohort, saved)
        after = protected_snapshot(root, p)
        if after != protected:
            raise PhaseCError("FAIL_HISTORICAL_ARTIFACT_CHANGED")
        write_json(out / "historical_integrity.json", {"status": "PASS", "count": len(protected),
            "before": protected, "after": after, "changed": [],
            "scope": "historical CO2 artifacts hashed only; no CO2 or locked-test dataset access"})
        result = {**identity, "phase_c_status": "COMPLETE", "gate_result": gate,
            "candidate": summary, "historical_comparison": comparison, "gpu_latency": latency,
            "latency_comparability": environment["latency_comparability"], "gate_thresholds": p["acceptance_gate"],
            "performance_gate_checks": performance,
            "failed_performance_criteria": [k for k, passed in performance.items() if not passed],
            "latency_threshold_observed_pass": raw_decision["checks"]["gpu_mean_latency"],
            "latency_gate_passed": raw_decision["checks"]["gpu_mean_latency"] if comparable else None,
            "result_integrity": integrity, "historical_artifacts_unchanged": True, "protected_artifact_count": len(protected),
            "development_images_inferred": attempted, "num_images": len(saved),
            "num_positive_images": summary["positive_images"], "num_negative_images": summary["negative_images"],
            "num_gt_instances": summary["tp"]+summary["fn"],
            "TP": summary["tp"], "FP": summary["fp"], "FN": summary["fn"],
            "Precision": summary["precision"], "Recall": summary["recall"], "F1": summary["f1"],
            "mean_mask_iou": summary["mask_iou_positive_mean"], "mean_dice": summary["mask_dice_positive_mean"],
            "crop_complete_count": summary["crop_complete95_images"], "crop_complete_fraction": summary["crop_complete95_fraction"],
            "no_roi_positive_count": summary["positive_without_roi"],
            "negative_with_prediction_count": summary["negative_images_with_predictions"],
            "mean_retained_gt_wound_pixels_positive": sum(r["retained_gt_wound_pixels"] for r in saved if r["gt_pixels"])/summary["positive_images"],
            "mean_retained_fraction_positive": summary["crop_coverage_positive_mean"],
            "latency_mean": latency["mean_ms"], "latency_median": latency["median_ms"], "latency_p95": latency["p95_ms"],
            "training_performed": False, "locked_test_used": False, "test_images_used": 0,
            "CO2Wounds_used": False, "multi_seed_started": False, "app_model_replaced": False,
            "bootstrap_CI_generated": False, "statistical_significance_test_performed": False,
            "READY_FOR_NEXT_RESEARCH_DECISION": True, "READY_FOR_ERROR_ANALYSIS": gate == "FAIL_DEVELOPMENT_GATE",
            "MULTI_SEED_AUTHORIZED": False, "APP_MODEL_REPLACEMENT_AUTHORIZED": False, "EXTERNAL_TEST_AUTHORIZED": False,
            "mask_mAP50": None, "mask_mAP50_95": None, "mAP_note": "Not supported by the historical fixed-point gate; not computed"}
        for size, stat in summary["size_recall"].items():
            result.update({f"{size}_support": stat["gt"], f"{size}_tp": stat["matched"],
                           f"{size}_fn": stat["gt"]-stat["matched"], f"{size}_recall": stat["recall"]})
        write_json(out / "result.json", result)
        write_json(out / "execution_completion.json", {"status": "COMPLETE", "completed_utc": now(),
            "development_images_inferred": attempted, "gate_result": gate,
            "result_sha256": sha256_file(out / "result.json"),
            "per_image_predictions_sha256": sha256_file(out / "per_image_predictions.json"),
            "inference_rerun_permitted": False})
        return result
    except BaseException as exc:
        write_json(out / "execution_failure.json", {"status": "FAILED_NO_RETRY", "timestamp": now(),
            "exception_type": type(exc).__name__, "error": str(exc), "images_attempted": attempted,
            "per_image_records_saved": len(rows), "test_images_used": 0, "CO2Wounds_used": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.verify_only:
        print(json.dumps(verify_saved_execution(root)))
    elif args.preflight_only:
        _, _, records = preflight(root)
        print(json.dumps({"status": "PASS_PREFLIGHT", "runtime_hashes_passed": len(records), "inference_started": False}))
    else:
        final = execute(root)
        print(json.dumps({key: final[key] for key in ("phase_c_status", "gate_result", "TP", "FP", "FN", "Precision", "Recall", "F1", "crop_complete_fraction", "latency_comparability")}, ensure_ascii=False))
