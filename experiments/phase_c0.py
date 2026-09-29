"""Phase C0: metadata-only preflight for the controlled RGB/BGR reevaluation.

This module deliberately has no torch or Ultralytics dependency.  It freezes
the historical checkpoint, FUSeg validation cohort, operating point, metric
implementation and the one permitted color-contract change before inference.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from experiments.data_roles import Role, require_source_role


EXPERIMENT_ID = "ISIC_FUSEG_COLORFIX_V1"
EXPECTED_VALIDATION_SAMPLES = 191
FUTURE_OUTPUT = "experiments/results/isic_fuseg_colorfix_v1"
EXPECTED_SOURCE_EVIDENCE_SHA256 = "c91ad4b43fabd0ef02242c2c76adca3807f5f98bf61f26df44fba540bc8a69dc"
FROZEN_PRIMARY = {"confidence": 0.1, "nms_iou": 0.7, "bbox_match_iou": 0.5}
FROZEN_PREDICTION = {
    "imgsz": 768,
    "batch": 1,
    "device": "0",
    "half": False,
    "retina_masks": False,
    "max_det": 300,
    "conf": 0.01,
    "augment": False,
    "agnostic_nms": False,
    "classes": [0],
}
FROZEN_MINIMA_PERCENT = {
    "precision": 87.18,
    "recall": 84.65,
    "f1": 85.89,
    "crop_complete95_fraction": 90.32,
}
HISTORICAL_RGB_LINE = 'image = np.asarray(im.convert("RGB"))'
CORRECTED_BGR_LINE = (
    'image = validate_model_input_contract('
    'im.convert("RGB"), source_type="PIL_RGB")'
)
GUARD_IMPORT = "from experiments.data_roles import validate_model_input_contract"


class C0Error(RuntimeError):
    """Fail-closed Phase C0 invariant violation."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value.casefold())


def verify_historical_gate_settings(historical_protocol: dict) -> None:
    """The actual historical artifact must agree with every C0 frozen setting."""
    if (
        historical_protocol.get("primary") != FROZEN_PRIMARY
        or historical_protocol.get("prediction") != FROZEN_PREDICTION
        or historical_protocol.get("minima_percent") != FROZEN_MINIMA_PERCENT
        or historical_protocol.get("latency_max_ms") != 50
        or historical_protocol.get("test_images_used") != 0
    ):
        raise C0Error("HISTORICAL_SETTINGS_MISMATCH")


def _reject_forbidden_path(value: str) -> None:
    parts = {part.casefold() for part in PurePosixPath(value.replace("\\", "/")).parts}
    joined = "/".join(parts)
    if parts.intersection({"test", "testing", "blind_test", "locked_test"}):
        raise C0Error("FORBIDDEN_LOCKED_TEST_PATH")
    if "co2wounds" in joined or "co2_wounds" in joined:
        raise C0Error("FORBIDDEN_HISTORICAL_EXTERNAL_PATH")


def build_validation_manifest(cohort: list[dict], training_manifest: list[dict]) -> dict:
    """Freeze FUSeg validation metadata without opening any image or mask."""
    role = require_source_role("FUSeg-Validation-191", "development_evaluate")
    if role is not Role.DEVELOPMENT_VALIDATION:
        raise C0Error("SOURCE_ROLE_NOT_DEVELOPMENT_VALIDATION")
    if len(cohort) != EXPECTED_VALIDATION_SAMPLES:
        raise C0Error("EXPECTED_191_VALIDATION_SAMPLES")

    training_val = {
        (row.get("image_sha256"), row.get("label_sha256"))
        for row in training_manifest
        if row.get("split") == "val" and row.get("source", "FUSeg") == "FUSeg"
    }
    samples: list[dict] = []
    identifiers: set[str] = set()
    cohort_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(cohort):
        sample_id = str(row.get("image_id") or f"sample-{index:03d}")
        if sample_id in identifiers:
            raise C0Error("DUPLICATE_SAMPLE_ID")
        identifiers.add(sample_id)
        if row.get("source", "FUSeg") != "FUSeg" or row.get("split") != "val":
            raise C0Error("SOURCE_ROLE_OR_SPLIT_MISMATCH")
        required_hashes = (row.get("image_sha256"), row.get("mask_sha256"), row.get("label_sha256"))
        if not all(_is_sha256(value) for value in required_hashes):
            raise C0Error("INVALID_SAMPLE_SHA256")
        image = str(row["image"])
        mask = str(row["source_mask"])
        _reject_forbidden_path(image)
        _reject_forbidden_path(mask)
        pair = (row["image_sha256"], row["label_sha256"])
        cohort_pairs.add(pair)
        samples.append(
            {
                "sample_id": sample_id,
                "relative_image_path": f"outputs/fuseg_warmup_revision_20260914/{image}",
                "image_sha256": row["image_sha256"],
                "relative_mask_path": mask.replace("\\", "/"),
                "mask_sha256": row["mask_sha256"],
                "label_sha256": row["label_sha256"],
                "source_dataset": "FUSeg",
                "source_role": Role.DEVELOPMENT_VALIDATION.value,
                "split": "val",
                "positive": bool(row.get("positive", True)),
            }
        )
    if cohort_pairs != training_val or len(training_val) != EXPECTED_VALIDATION_SAMPLES:
        raise C0Error("COHORT_MISMATCH_WITH_FROZEN_TRAINING_MANIFEST")
    samples.sort(key=lambda row: row["sample_id"])
    return {
        "manifest_schema": "phase-c0-validation-manifest-v1",
        "source_dataset": "FUSeg",
        "source_id": "FUSeg-Validation-191",
        "source_role": Role.DEVELOPMENT_VALIDATION.value,
        "sample_count": EXPECTED_VALIDATION_SAMPLES,
        "development_images_inferred": 0,
        "test_images_used": 0,
        "historical_external_images_used": 0,
        "samples": samples,
    }


def build_code_diff_audit(historical_text: str, corrected_text: str) -> dict:
    """Allow exactly the RGB-to-BGR contract guard and no other semantic edit."""
    old = historical_text.replace("\r\n", "\n")
    new = corrected_text.replace("\r\n", "\n")
    old_rgb = [line for line in old.splitlines() if line.strip() == HISTORICAL_RGB_LINE]
    new_bgr = [line for line in new.splitlines() if line.strip() == CORRECTED_BGR_LINE]
    imports = [line for line in new.splitlines() if line.strip() == GUARD_IMPORT]
    if len(imports) != 1 or len(old_rgb) != 1 or len(new_bgr) != 1:
        raise C0Error("NON_COLOR_SEMANTIC_DIFF")
    normalized = new.replace(imports[0] + "\n", "", 1).replace(
        new_bgr[0], old_rgb[0], 1
    )
    if normalized != old:
        raise C0Error("NON_COLOR_SEMANTIC_DIFF")
    return {
        "status": "PASS_COLOR_CONTRACT_ONLY",
        "allowed_change_count": 2,
        "changes": [
            {
                "classification": "SAFETY_GUARD",
                "before": None,
                "after": GUARD_IMPORT,
            },
            {
                "classification": "COLOR_CONTRACT",
                "before": HISTORICAL_RGB_LINE.strip(),
                "after": CORRECTED_BGR_LINE.strip(),
            },
        ],
        "non_color_semantic_changes": [],
        "threshold_change": False,
        "nms_change": False,
        "resize_change": False,
        "crop_change": False,
        "metric_change": False,
    }


def build_protocol(
    *,
    checkpoint_path: str,
    checkpoint_expected_sha256: str,
    checkpoint_actual_sha256: str,
    checkpoint_size_bytes: int,
    validation_manifest: dict,
    validation_manifest_sha256: str,
    source_evidence_path: str,
    source_evidence_sha256: str,
    source_evidence_verified: bool,
    metric_script_path: str,
    metric_expected_sha256: str,
    metric_actual_sha256: str,
    historical_gate_result_path: str,
    historical_gate_result_sha256: str,
    code_diff_audit: dict,
    future_output_dir: str = FUTURE_OUTPUT,
) -> dict:
    if checkpoint_actual_sha256 != checkpoint_expected_sha256:
        raise C0Error("CHECKPOINT_HASH_MISMATCH")
    if validation_manifest.get("sample_count") != EXPECTED_VALIDATION_SAMPLES:
        raise C0Error("EXPECTED_191_VALIDATION_SAMPLES")
    if sha256_bytes(canonical_json_bytes(validation_manifest)) != validation_manifest_sha256:
        raise C0Error("VALIDATION_MANIFEST_HASH_MISMATCH")
    if validation_manifest.get("source_role") != Role.DEVELOPMENT_VALIDATION.value:
        raise C0Error("SOURCE_ROLE_NOT_DEVELOPMENT_VALIDATION")
    if not source_evidence_verified or not _is_sha256(source_evidence_sha256):
        raise C0Error("SOURCE_EVIDENCE_NOT_VERIFIED")
    if metric_actual_sha256 != metric_expected_sha256:
        raise C0Error("METRIC_SCRIPT_HASH_MISMATCH")
    if code_diff_audit.get("status") != "PASS_COLOR_CONTRACT_ONLY":
        raise C0Error("NON_COLOR_SEMANTIC_DIFF")
    if code_diff_audit.get("non_color_semantic_changes"):
        raise C0Error("NON_COLOR_SEMANTIC_DIFF")
    if future_output_dir.replace("\\", "/") != FUTURE_OUTPUT:
        raise C0Error("UNEXPECTED_OUTPUT_DIRECTORY")

    return {
        "protocol_schema": "phase-c0-colorfix-preflight-v1",
        "protocol_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN_BEFORE_EXECUTION",
        "protocol_status": "FROZEN_BEFORE_EXECUTION",
        "research_question": "Effect on development-gate measurements of correcting only the RGB/BGR model-input contract while holding checkpoint, cohort, operating point and metric implementation fixed",
        "historical_experiment_id": "D-Formal-ISIC-FUSeg-seed42-20260914",
        "allowed_semantic_change": "RGB_BGR_INPUT_CONTRACT_ONLY",
        "scope": "CONTROLLED_RGB_BGR_REEVALUATION_PREFLIGHT_ONLY",
        "checkpoint": {
            "path": checkpoint_path.replace("\\", "/"),
            "sha256": checkpoint_actual_sha256,
            "size_bytes": checkpoint_size_bytes,
            "training_or_finetuning_allowed": False,
        },
        "source": {
            "dataset": "FUSeg",
            "source_id": "FUSeg-Validation-191",
            "role": Role.DEVELOPMENT_VALIDATION.value,
            "validation_samples": EXPECTED_VALIDATION_SAMPLES,
            "upstream_repository": "https://github.com/uwm-bigdata/wound-segmentation",
            "upstream_commit": "42a272dfe0679f20675e826385925cb7562934b6",
            "dataset_release_version": "not specified in existing source evidence",
            "data_usage_record": "CC BY NC, version unspecified; noncommercial offline academic/graduation research only",
            "evidence_path": source_evidence_path.replace("\\", "/"),
            "evidence_sha256": source_evidence_sha256,
        },
        "validation_manifest": {
            "path": "experiments/protocols/ISIC_FUSEG_COLORFIX_V1_validation_manifest.json",
            "sha256": validation_manifest_sha256,
            "sample_count": EXPECTED_VALIDATION_SAMPLES,
        },
        "color_contract": {
            "historical_input": "PIL RGB converted to NumPy RGB but passed to an interface expecting NumPy BGR",
            "corrected_input": "PIL RGB converted once to contiguous NumPy BGR",
            "only_permitted_semantic_change": True,
            "code_diff_audit_path": "experiments/protocols/ISIC_FUSEG_COLORFIX_V1_code_diff_audit.json",
        },
        "operating_point": FROZEN_PRIMARY,
        "prediction": FROZEN_PREDICTION,
        "device_policy": "GPU device 0; no silent CPU fallback",
        "resize_and_mask_contract": {
            "model_input_imgsz": 768,
            "ground_truth_canvas": [512, 512],
            "predicted_mask_materialization": "nearest-neighbor resize to 512x512 then boolean foreground",
            "mask_threshold": "Ultralytics masks.data materialized as foreground boolean; no tunable mask threshold",
        },
        "crop_rule": {
            "roi": "union bounding box of retained prediction polygons",
            "margin_fraction_per_side": 0.15,
            "complete_definition": "retains at least 95% of ground-truth wound pixels",
            "no_roi_policy": "positive image without ROI is failure",
        },
        "matching_rule": {
            "ordering": "predictions sorted by descending confidence",
            "assignment": "one-to-one unmatched ground truth with highest box IoU",
            "minimum_iou": 0.5,
            "denominator": "all ground-truth polygon instances, including unmatched instances",
            "size_bins_by_gt_image_fraction": {"small": "<1%", "medium": "1%-5%", "large": ">=5%"},
        },
        "metric_implementation": {
            "path": metric_script_path.replace("\\", "/"),
            "sha256": metric_actual_sha256,
            "frozen": True,
        },
        "historical_reference": {
            "result_path": historical_gate_result_path.replace("\\", "/"),
            "result_sha256": historical_gate_result_sha256,
            "status": "FAIL_DEVELOPMENT_GATE",
            "historical_evaluation": True,
            "confirmed_rgb_bgr_input_mismatch": True,
            "precision": 0.45588235294117646,
            "recall": 0.12863070539419086,
            "f1": 0.20064724919093851,
            "crop_complete95_fraction": 0.14516129032258066,
            "validity_note": "Historical result retained; comparison was invalidated by RGB/BGR input-contract defect.",
        },
        "acceptance_gate": {
            "comparison_precision": "published percentages rounded to 2 decimals",
            "minima_percent": FROZEN_MINIMA_PERCENT,
            "latency_max_ms": 50,
            "failure_action": "stop_and_analyze_errors",
            "success_action": "eligible_for_separately_authorized_next_phase_only",
        },
        "planned_outputs_after_separate_execution_authorization": {
            "future_output_dir": FUTURE_OUTPUT,
            "primary_metrics": [
                "images", "positive_images", "negative_images",
                "tp", "fp", "fn", "precision", "recall", "f1",
                "mask_iou_positive_mean", "mask_dice_positive_mean",
                "crop_coverage_positive_mean", "crop_complete95_fraction",
                "positive_without_roi", "negative_images_with_predictions", "size_recall",
            ],
            "secondary_diagnostics": [
                "gpu_latency", "per_image_predictions", "error_panels",
                "mean_retained_gt_wound_pixels (derived diagnostic only, not an acceptance metric)",
            ],
            "not_part_of_historical_gate": ["mask_mAP50", "mask_mAP50-95"],
            "historical_comparison_columns": [
                "metric", "historical_incorrect_color", "corrected_color", "absolute_delta"
            ],
            "permitted_interpretation": "Change in measured performance after correcting the evaluation input-color contract; not model or training improvement",
        },
        "execution_state": {
            "inference_started": False,
            "training_started": False,
            "multi_seed_started": False,
            "development_images_inferred": 0,
            "test_images_used": 0,
            "historical_external_images_used": 0,
        },
        "development_images_inferred": 0,
        "test_images_used": 0,
        "historical_external_images_used": 0,
        "model_inference": False,
        "training": False,
        "locked_test_used": False,
        "CO2Wounds_used": False,
        "bootstrap_CI_generated": False,
        "forbidden": [
            "training", "fine_tuning", "threshold_tuning", "multi_seed",
            "locked_test", "CO2Wounds", "App_weight_replacement",
        ],
    }


def verify_protocol_hash(protocol_path: Path, sha_path: Path) -> str:
    if not protocol_path.is_file() or not sha_path.is_file():
        raise C0Error("PROTOCOL_OR_HASH_MISSING")
    expected = sha_path.read_text(encoding="ascii").split()[0]
    actual = sha256_file(protocol_path)
    if not _is_sha256(expected) or actual != expected:
        raise C0Error("PROTOCOL_HASH_MISMATCH")
    return actual


def assert_execution_preflight(protocol_path: Path, sha_path: Path, output_dir: Path) -> str:
    digest = verify_protocol_hash(protocol_path, sha_path)
    if output_dir.exists():
        raise C0Error("OUTPUT_ALREADY_EXISTS")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    state = protocol.get("execution_state", {})
    if state != {
        "development_images_inferred": 0,
        "historical_external_images_used": 0,
        "inference_started": False,
        "multi_seed_started": False,
        "test_images_used": 0,
        "training_started": False,
    }:
        raise C0Error("PROTOCOL_NOT_PRISTINE")
    return digest


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head_blob(root: Path, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "cat-file", "blob", f"HEAD:{relative_path}"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8")


def generate_preflight(root: Path) -> dict[str, Path]:
    """Generate C0 evidence from metadata/code/checkpoint bytes only."""
    protocol_dir = root / "experiments/protocols"
    output_dir = root / FUTURE_OUTPUT
    targets = {
        "manifest": protocol_dir / f"{EXPERIMENT_ID}_validation_manifest.json",
        "diff": protocol_dir / f"{EXPERIMENT_ID}_code_diff_audit.json",
        "protocol": protocol_dir / f"{EXPERIMENT_ID}_protocol.json",
        "sha": protocol_dir / f"{EXPERIMENT_ID}_protocol.sha256",
    }
    if output_dir.exists():
        raise C0Error("OUTPUT_ALREADY_EXISTS")
    if any(path.exists() for path in targets.values()):
        raise C0Error("C0_ARTIFACT_ALREADY_EXISTS")

    historical_protocol_path = root / "outputs/isic_fuseg_formal_seed42_20260914/development_gate/protocol.json"
    historical_result_path = root / "outputs/isic_fuseg_formal_seed42_20260914/development_gate/result.json"
    historical_protocol = _read_json(historical_protocol_path)
    historical_result = _read_json(historical_result_path)
    verify_historical_gate_settings(historical_protocol)
    if historical_result.get("test_images_used") != 0 or historical_protocol.get("validation_images") != 191:
        raise C0Error("HISTORICAL_GATE_INVARIANT_FAILED")
    candidate = historical_result.get("candidate", {})
    if (
        historical_result.get("status") != "FAIL_DEVELOPMENT_GATE"
        or candidate.get("images") != 191
        or candidate.get("precision") != 0.45588235294117646
        or candidate.get("recall") != 0.12863070539419086
        or candidate.get("f1") != 0.20064724919093851
        or candidate.get("crop_complete95_fraction") != 0.14516129032258066
        or historical_result.get("decision", {}).get("minima_percent") != FROZEN_MINIMA_PERCENT
    ):
        raise C0Error("HISTORICAL_RESULT_MISMATCH")

    checkpoint = Path(historical_protocol["checkpoint"])
    checkpoint_actual = sha256_file(checkpoint)
    cohort_path = root / "outputs/localization_benchmark_20260914/cohort.json"
    training_manifest_path = root / "outputs/isic_fuseg_pretrain_smoke_20260914/fuseg_manifest.json"
    if sha256_file(cohort_path) != historical_protocol["cohort_sha256"]:
        raise C0Error("HISTORICAL_COHORT_HASH_MISMATCH")
    validation_manifest = build_validation_manifest(
        _read_json(cohort_path), _read_json(training_manifest_path)
    )
    validation_manifest_bytes = canonical_json_bytes(validation_manifest)

    evidence_path = root / "experiments/review_v2/evidence/FUSeg_noncommercial_research_20260914.md"
    evidence_text = evidence_path.read_text(encoding="utf-8")
    evidence_verified = sha256_file(evidence_path) == EXPECTED_SOURCE_EVIDENCE_SHA256 and all(
        marker in evidence_text
        for marker in (
            "https://github.com/uwm-bigdata/wound-segmentation",
            "42a272dfe0679f20675e826385925cb7562934b6",
            "CC BY NC",
            "771",
            "191",
        )
    )

    gate_relative = "experiments/review_v2/isic_fuseg_gate.py"
    historical_gate_text = _git_head_blob(root, gate_relative)
    historical_gate_sha = sha256_bytes(historical_gate_text.encode("utf-8"))
    expected_historical_gate_sha = historical_protocol["pins"][str(root / gate_relative)]
    if historical_gate_sha != expected_historical_gate_sha:
        raise C0Error("HISTORICAL_GATE_SOURCE_HASH_MISMATCH")
    corrected_gate_text = (root / gate_relative).read_text(encoding="utf-8")
    diff_audit = build_code_diff_audit(historical_gate_text, corrected_gate_text)
    diff_audit.update(
        {
            "historical_git_blob_sha256": historical_gate_sha,
            "corrected_worktree_sha256": sha256_file(root / gate_relative),
        }
    )
    diff_bytes = canonical_json_bytes(diff_audit)

    metric_path = root / "experiments/review_v2/localization_benchmark.py"
    metric_expected = historical_protocol["pins"][str(metric_path)]
    crop_script_path = root / "woundcare_inference.py"
    crop_script_expected = historical_protocol["pins"][str(crop_script_path)]
    if sha256_file(crop_script_path) != crop_script_expected:
        raise C0Error("CROP_IMPLEMENTATION_HASH_MISMATCH")
    protocol = build_protocol(
        checkpoint_path=str(checkpoint.relative_to(root)),
        checkpoint_expected_sha256=historical_protocol["checkpoint_sha256"],
        checkpoint_actual_sha256=checkpoint_actual,
        checkpoint_size_bytes=checkpoint.stat().st_size,
        validation_manifest=validation_manifest,
        validation_manifest_sha256=sha256_bytes(validation_manifest_bytes),
        source_evidence_path=str(evidence_path.relative_to(root)),
        source_evidence_sha256=sha256_file(evidence_path),
        source_evidence_verified=evidence_verified,
        metric_script_path=str(metric_path.relative_to(root)),
        metric_expected_sha256=metric_expected,
        metric_actual_sha256=sha256_file(metric_path),
        historical_gate_result_path=str(historical_result_path.relative_to(root)),
        historical_gate_result_sha256=sha256_file(historical_result_path),
        code_diff_audit=diff_audit,
    )
    protocol["provenance"] = {
        "historical_protocol_path": str(historical_protocol_path.relative_to(root)).replace("\\", "/"),
        "historical_protocol_sha256": sha256_file(historical_protocol_path),
        "historical_cohort_path": str(cohort_path.relative_to(root)).replace("\\", "/"),
        "historical_cohort_sha256": sha256_file(cohort_path),
        "training_manifest_path": str(training_manifest_path.relative_to(root)).replace("\\", "/"),
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "code_diff_audit_sha256": sha256_bytes(diff_bytes),
        "historical_gate_script_sha256": historical_gate_sha,
        "corrected_gate_script_sha256": sha256_file(root / gate_relative),
        "crop_implementation_path": str(crop_script_path.relative_to(root)).replace("\\", "/"),
        "crop_implementation_sha256": crop_script_expected,
    }
    protocol_bytes = canonical_json_bytes(protocol)
    protocol_digest = sha256_bytes(protocol_bytes)

    protocol_dir.mkdir(parents=True, exist_ok=True)
    targets["manifest"].write_bytes(validation_manifest_bytes)
    targets["diff"].write_bytes(diff_bytes)
    targets["protocol"].write_bytes(protocol_bytes)
    targets["sha"].write_text(
        f"{protocol_digest}  {targets['protocol'].name}\n", encoding="ascii"
    )
    verify_protocol_hash(targets["protocol"], targets["sha"])
    if output_dir.exists():
        raise C0Error("OUTPUT_CREATED_DURING_PREFLIGHT")
    return targets


if __name__ == "__main__":
    generated = generate_preflight(Path(__file__).resolve().parents[1])
    print(json.dumps({key: str(value) for key, value in generated.items()}, ensure_ascii=False, indent=2))
