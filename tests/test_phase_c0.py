from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.data_roles import Role, require_source_role, validate_model_input_contract
from experiments.phase_c0 import (
    C0Error,
    assert_execution_preflight,
    build_code_diff_audit,
    build_protocol,
    build_validation_manifest,
    canonical_json_bytes,
    verify_historical_gate_settings,
    verify_protocol_hash,
)


HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


def _cohort(n: int = 191) -> list[dict]:
    return [
        {
            "image_id": f"sample-{i:03d}",
            "image": f"dataset/images/val/sample-{i:03d}.png",
            "source_mask": f"outputs/fuseg/masks/val/sample-{i:03d}.png",
            "image_sha256": f"{i:064x}",
            "mask_sha256": f"{i + 1000:064x}",
            "label_sha256": f"{i + 2000:064x}",
            "source": "FUSeg",
            "split": "val",
            "positive": True,
        }
        for i in range(n)
    ]


def _training_manifest(cohort: list[dict]) -> list[dict]:
    return [
        {
            "source": "FUSeg",
            "split": "val",
            "image_sha256": row["image_sha256"],
            "label_sha256": row["label_sha256"],
        }
        for row in cohort
    ]


def _diff() -> dict:
    old = 'from PIL import Image\n\nimage = np.asarray(im.convert("RGB"))\n'
    new = (
        "from PIL import Image\n"
        "from experiments.data_roles import validate_model_input_contract\n\n"
        'image = validate_model_input_contract(im.convert("RGB"), source_type="PIL_RGB")\n'
    )
    return build_code_diff_audit(old, new)


def _protocol(**overrides) -> dict:
    cohort = _cohort()
    manifest = build_validation_manifest(cohort, _training_manifest(cohort))
    kwargs = {
        "checkpoint_path": "outputs/formal/weights/best.pt",
        "checkpoint_expected_sha256": HEX_A,
        "checkpoint_actual_sha256": HEX_A,
        "checkpoint_size_bytes": 123,
        "validation_manifest": manifest,
        "validation_manifest_sha256": hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
        "source_evidence_path": "experiments/review_v2/evidence/fuseg.md",
        "source_evidence_sha256": HEX_B,
        "source_evidence_verified": True,
        "metric_script_path": "experiments/review_v2/localization_benchmark.py",
        "metric_expected_sha256": HEX_C,
        "metric_actual_sha256": HEX_C,
        "historical_gate_result_path": "outputs/formal/development_gate/result.json",
        "historical_gate_result_sha256": "d" * 64,
        "code_diff_audit": _diff(),
        "future_output_dir": "experiments/results/isic_fuseg_colorfix_v1",
    }
    kwargs.update(overrides)
    return build_protocol(**kwargs)


def test_c0_requires_same_checkpoint_hash():
    with pytest.raises(C0Error, match="CHECKPOINT_HASH_MISMATCH"):
        _protocol(checkpoint_actual_sha256=HEX_B)


def test_c0_requires_191_validation_samples():
    cohort = _cohort(190)
    with pytest.raises(C0Error, match="EXPECTED_191"):
        build_validation_manifest(cohort, _training_manifest(cohort))


def test_c0_requires_same_validation_manifest():
    cohort = _cohort()
    training = _training_manifest(cohort)
    training[0]["image_sha256"] = HEX_A
    with pytest.raises(C0Error, match="COHORT_MISMATCH"):
        build_validation_manifest(cohort, training)


def test_c0_rejects_unknown_source_role():
    with pytest.raises(ValueError, match="SOURCE_ROLE_NOT_VERIFIED"):
        require_source_role("unknown", "development_evaluate")


def test_c0_rejects_locked_test_role():
    with pytest.raises(ValueError, match="SOURCE_ROLE_FORBIDDEN"):
        require_source_role("C-Arch-05-Locked-Test", "development_evaluate")


def test_c0_rejects_historical_external_role():
    with pytest.raises(ValueError, match="SOURCE_ROLE_FORBIDDEN"):
        require_source_role("CO2Wounds-V2", "development_evaluate")


def test_c0_requires_source_evidence():
    with pytest.raises(C0Error, match="SOURCE_EVIDENCE_NOT_VERIFIED"):
        _protocol(source_evidence_verified=False)


def test_c0_requires_frozen_thresholds():
    protocol = _protocol()
    assert protocol["operating_point"] == {
        "confidence": 0.1,
        "nms_iou": 0.7,
        "bbox_match_iou": 0.5,
    }
    assert protocol["prediction"]["conf"] == 0.01
    assert protocol["crop_rule"]["margin_fraction_per_side"] == 0.15
    historical = {
        "primary": dict(protocol["operating_point"]),
        "prediction": protocol["prediction"],
        "minima_percent": protocol["acceptance_gate"]["minima_percent"],
        "latency_max_ms": protocol["acceptance_gate"]["latency_max_ms"],
        "test_images_used": 0,
    }
    verify_historical_gate_settings(historical)
    historical["primary"]["nms_iou"] = 0.65
    with pytest.raises(C0Error, match="HISTORICAL_SETTINGS_MISMATCH"):
        verify_historical_gate_settings(historical)


def test_c0_requires_same_metric_script():
    with pytest.raises(C0Error, match="METRIC_SCRIPT_HASH_MISMATCH"):
        _protocol(metric_actual_sha256=HEX_A)


def test_c0_rejects_non_color_semantic_diff():
    old = 'image = np.asarray(im.convert("RGB"))\n'
    new = 'image = np.asarray(im.resize((640, 640)).convert("RGB"))\n'
    with pytest.raises(C0Error, match="NON_COLOR_SEMANTIC_DIFF"):
        build_code_diff_audit(old, new)


def test_c0_rgb_bgr_synthetic_equivalence():
    rgb = np.array([[[1, 2, 3], [10, 20, 30]]], dtype=np.uint8)
    converted = validate_model_input_contract(Image.fromarray(rgb, "RGB"), source_type="PIL_RGB")
    expected_bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    assert np.array_equal(converted, expected_bgr)
    assert converted.flags.c_contiguous


def test_c0_rejects_existing_output(tmp_path: Path):
    output = tmp_path / "isic_fuseg_colorfix_v1"
    output.mkdir()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(canonical_json_bytes(_protocol()))
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    sha_path = tmp_path / "protocol.sha256"
    sha_path.write_text(f"{digest}  {protocol_path.name}\n", encoding="ascii")
    with pytest.raises(C0Error, match="OUTPUT_ALREADY_EXISTS"):
        assert_execution_preflight(protocol_path, sha_path, output)


def test_c0_protocol_hash_detects_mutation(tmp_path: Path):
    path = tmp_path / "protocol.json"
    path.write_bytes(canonical_json_bytes(_protocol()))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sha_path = tmp_path / "protocol.sha256"
    sha_path.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    assert verify_protocol_hash(path, sha_path) == digest
    path.write_text(json.dumps({"mutated": True}), encoding="utf-8")
    with pytest.raises(C0Error, match="PROTOCOL_HASH_MISMATCH"):
        verify_protocol_hash(path, sha_path)


def test_c0_does_not_load_model(monkeypatch):
    original_import = __import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "ultralytics"}:
            raise AssertionError("model framework import forbidden in C0")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    assert _protocol()["development_images_inferred"] == 0


def test_c0_does_not_read_validation_pixels(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("pixel read forbidden in C0")

    monkeypatch.setattr(Image, "open", forbidden)
    cohort = _cohort()
    manifest = build_validation_manifest(cohort, _training_manifest(cohort))
    assert len(manifest["samples"]) == 191


def test_c0_does_not_start_multiseed():
    protocol = _protocol()
    assert protocol["execution_state"] == {
        "inference_started": False,
        "training_started": False,
        "multi_seed_started": False,
        "development_images_inferred": 0,
        "test_images_used": 0,
        "historical_external_images_used": 0,
    }
    assert protocol["source"]["role"] == Role.DEVELOPMENT_VALIDATION.value
