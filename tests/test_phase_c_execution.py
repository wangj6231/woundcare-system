"""Synthetic tests for the one-shot execution boundary and saved-count integrity."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.phase_c_execution import (
    PhaseCError, PINNED_PROTOCOL_SHA256, assert_relative_admitted,
    check_digest, choose_gate_status, validate_result_integrity,
)


def test_execution_protocol_pin_is_user_authorized_digest():
    assert PINNED_PROTOCOL_SHA256 == "16718f5956548e002320ebd2ade83c5ecafdf77e7c79dc1f1d2b63818b7e6df3"


def test_execution_digest_rejects_changed_bytes(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"different")
    with pytest.raises(PhaseCError, match="FAIL_PROTOCOL_MUTATED"):
        check_digest(path, "0" * 64, "FAIL_PROTOCOL_MUTATED")


@pytest.mark.parametrize("relative", ["../outside.png", "dataset/test/a.png", "CO2Wounds/a.png", "dataset/train/a.png"])
def test_execution_rejects_unadmitted_sample_paths(tmp_path, relative):
    with pytest.raises(PhaseCError):
        assert_relative_admitted(tmp_path, relative, tmp_path / "dataset/val")


def test_execution_existing_output_rejected_before_model_import(tmp_path, monkeypatch):
    from experiments import phase_c_execution as execution
    monkeypatch.setattr(execution, "check_digest", lambda *args: None)
    output = tmp_path / execution.OUTPUT_REL
    output.mkdir(parents=True)
    with pytest.raises(PhaseCError, match="FAIL_OUTPUT_ALREADY_EXISTS"):
        execution.preflight(tmp_path)


def test_execution_performance_failure_not_hidden_by_latency_uncertainty():
    assert choose_gate_status({"recall": False}, True, False) == "FAIL_DEVELOPMENT_GATE"


def test_execution_unverified_latency_prevents_full_pass():
    assert choose_gate_status({"recall": True}, True, False) == "PARTIAL_LATENCY_UNVERIFIED"


def test_execution_latency_failure_prevents_verified_pass():
    assert choose_gate_status({"recall": True}, False, True) == "FAIL_DEVELOPMENT_GATE"


def fixture_result():
    row = {"sample_id": "synthetic", "tp": 1, "fp": 1, "fn": 1,
           "num_gt_instances": 2, "num_predictions": 2, "gt_pixels": 100,
           "crop": [0, 0, 10, 10], "crop_complete95": True,
           "crop_coverage": 0.96, "retained_gt_wound_pixels": 96,
           "mask_iou": 0.5, "mask_dice": 2/3,
           "size_support": {"small": 1, "medium": 1, "large": 0},
           "size_matched": {"small": 1, "medium": 0, "large": 0}}
    summary = {"images": 1, "positive_images": 1, "negative_images": 0,
               "tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5,
               "crop_complete95_images": 1, "crop_complete95_fraction": 1.0,
               "positive_without_roi": 0, "negative_images_with_predictions": 0,
               "mask_iou_positive_mean": 0.5, "mask_dice_positive_mean": 2/3,
               "crop_coverage_positive_mean": 0.96,
               "size_recall": {"small": {"gt": 1, "matched": 1, "recall": 1.0},
                               "medium": {"gt": 1, "matched": 0, "recall": 0.0},
                               "large": {"gt": 0, "matched": 0, "recall": None}}}
    return [row], summary


def test_saved_counts_reconstruct_primary_metrics():
    rows, summary = fixture_result()
    assert validate_result_integrity(rows, summary, ["synthetic"])["status"] == "PASS"


@pytest.mark.parametrize("field,value", [("tp", 2), ("precision", .7), ("recall", 1.1),
                                          ("f1", float("nan")), ("crop_complete95_fraction", .5)])
def test_saved_result_rejects_inconsistent_report(field, value):
    rows, summary = fixture_result()
    summary[field] = value
    with pytest.raises(PhaseCError, match="FAIL_RESULT_INTEGRITY"):
        validate_result_integrity(rows, summary, ["synthetic"])


def test_saved_result_requires_all_unique_cohort_members():
    rows, summary = fixture_result()
    with pytest.raises(PhaseCError, match="FAIL_RESULT_INTEGRITY"):
        validate_result_integrity(rows + deepcopy(rows), summary, ["synthetic", "missing"])


def test_saved_result_requires_size_support_equal_gt():
    rows, summary = fixture_result()
    rows[0]["size_support"]["small"] += 1
    with pytest.raises(PhaseCError, match="FAIL_RESULT_INTEGRITY"):
        validate_result_integrity(rows, summary, ["synthetic"])
