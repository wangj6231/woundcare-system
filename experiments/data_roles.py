"""Fail-closed research source roles for new experiment entrypoints.

Role identity uses canonical IDs plus known path/manifest fingerprints where
available. Name-only admission is not proof of licensing, isolation or lineage.
"""
from __future__ import annotations

from enum import Enum
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


class Role(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    DEVELOPMENT_VALIDATION = "DEVELOPMENT_VALIDATION"
    INTERNAL_LOCKED_TEST = "INTERNAL_LOCKED_TEST"
    HISTORICAL_EXTERNAL = "HISTORICAL_EXTERNAL"
    TRAIN_ONLY = "TRAIN_ONLY"
    VALIDATION_ONLY = "VALIDATION_ONLY"


_IDS = {
    "C-Arch-05-Development": Role.DEVELOPMENT,
    "FUSeg": Role.DEVELOPMENT,
    "FUSeg-Validation-191": Role.DEVELOPMENT_VALIDATION,
    "D-Seg-07_manual": Role.DEVELOPMENT,
    "AZH": Role.TRAIN_ONLY,
    "BUBT_healthy": Role.TRAIN_ONLY,
    "ISIC2017": Role.TRAIN_ONLY,
    "C-Arch-05-Locked-Test": Role.INTERNAL_LOCKED_TEST,
    "CO2Wounds-V2": Role.HISTORICAL_EXTERNAL,
}
_DENIED = {Role.INTERNAL_LOCKED_TEST, Role.HISTORICAL_EXTERNAL}
_KNOWN_MANIFEST_SHA256 = {
    # Existing sealed CO2 external manifest; digest is of metadata CSV, not pixels.
    "be9b0bbe8374ca52c77764b5022617f736d6911f580433b69d6d8a7e6ed96fe4": Role.HISTORICAL_EXTERNAL,
}
_DEV_OPS = {"train", "fine_tune", "tune", "threshold_tune", "augment_select",
            "calibrate", "select", "development_evaluate", "development_gate"}


def _path_role(path: Path | str | None) -> Role | None:
    if path is None:
        return None
    parts = [p.casefold() for p in Path(path).parts]
    joined = "/".join(parts)
    if "co2wounds" in joined or "co2_wounds" in joined:
        return Role.HISTORICAL_EXTERNAL
    if any(p in {"test", "testing", "blind_test", "locked_test"} for p in parts):
        return Role.INTERNAL_LOCKED_TEST
    if "yolo_wound_cls_dataset_v3" in joined:
        return Role.DEVELOPMENT
    if "_gkfold_tmp" in joined:
        return Role.DEVELOPMENT
    if "fuseg" in joined and any(p in {"train", "val", "validation"} for p in parts):
        return Role.DEVELOPMENT
    return None


def require_source_role(source_id: str | None, operation: str, *,
                        path: Path | str | None = None,
                        manifest_source_id: str | None = None,
                        manifest_path: Path | str | None = None) -> Role:
    if operation not in _DEV_OPS:
        raise ValueError(f"SOURCE_OPERATION_NOT_VERIFIED: {operation}")
    manifest_role = None
    if manifest_path is not None:
        manifest = Path(manifest_path)
        if not manifest.is_file():
            raise ValueError("SOURCE_ROLE_NOT_VERIFIED: manifest missing")
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        manifest_role = _KNOWN_MANIFEST_SHA256.get(digest)
        if manifest_role is None:
            raise ValueError("SOURCE_ROLE_NOT_VERIFIED: manifest hash not registered")
    roles = [r for r in (_IDS.get(source_id), _path_role(path),
                         _IDS.get(manifest_source_id), manifest_role) if r]
    if any(r in _DENIED for r in roles):
        raise ValueError("SOURCE_ROLE_FORBIDDEN: locked or historical external source")
    if not roles or len(set(roles)) > 1:
        raise ValueError("SOURCE_ROLE_NOT_VERIFIED: unknown or conflicting source identity")
    role = roles[0]
    if role == Role.VALIDATION_ONLY and operation != "development_evaluate":
        raise ValueError("SOURCE_ROLE_FORBIDDEN: validation-only source")
    if role == Role.DEVELOPMENT_VALIDATION and operation not in {
        "development_evaluate", "development_gate"
    }:
        raise ValueError("SOURCE_ROLE_FORBIDDEN: development-validation source")
    if role == Role.TRAIN_ONLY and operation not in {"train", "fine_tune"}:
        raise ValueError("SOURCE_ROLE_FORBIDDEN: train-only source")
    return role


def validate_model_input_contract(value: Image.Image | np.ndarray, *, source_type: str) -> np.ndarray:
    """Return BGR ndarray for Ultralytics; caller declares source semantics."""
    if source_type == "PIL_RGB" and isinstance(value, Image.Image):
        if value.mode != "RGB":
            raise ValueError("PIL_RGB input requires RGB mode")
        array = np.asarray(value)
        return np.ascontiguousarray(array[:, :, ::-1])
    if source_type == "NUMPY_BGR" and isinstance(value, np.ndarray):
        if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
            raise ValueError("NUMPY_BGR input requires uint8 HWC 3-channel")
        return np.ascontiguousarray(value)
    raise ValueError(f"unsupported input contract: {source_type}; NUMPY_RGB direct input forbidden")
