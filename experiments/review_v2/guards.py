"""Fail-closed guards for NEW development experiments, independent of legacy code."""
from __future__ import annotations

import math
from pathlib import Path


def validate_optimizer_settings(settings: dict) -> None:
    """Explicit AdamW warmup avoids the inherited 0.1 bias LR in low-LR tuning."""
    if settings.get("optimizer") not in {"AdamW", "SGD"}:
        raise ValueError("explicit audited optimizer required; auto can override intended LR")
    for key in ("lr0", "warmup_bias_lr", "warmup_epochs"):
        value = settings.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"explicit finite nonnegative {key} required")
    if settings["lr0"] <= 0:
        raise ValueError("lr0 must be positive")
    if settings["optimizer"] == "AdamW" and settings["warmup_bias_lr"] > settings["lr0"]:
        raise ValueError("AdamW bias warmup exceeds base LR; requires a separate documented ablation")


def validate_polygon_line(line: str, class_count=1) -> list[tuple[float, float]]:
    tokens = line.split()
    if len(tokens) < 7 or (len(tokens) - 1) % 2:
        raise ValueError("polygon needs class and at least three xy pairs")
    cls = int(tokens[0])
    if str(cls) != tokens[0] or not 0 <= cls < class_count:
        raise ValueError("invalid class id")
    values = [float(x) for x in tokens[1:]]
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("coordinates must be finite and normalized")
    points = list(zip(values[::2], values[1::2]))
    area2 = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))
    if len(set(points)) < 3 or abs(area2) <= 1e-12:
        raise ValueError("degenerate/zero-area polygon")
    return points


def validate_new_development_run(config: dict, source_gates: dict, *,
                                 train_groups: set, val_groups: set,
                                 ancestor_train_groups: set | None,
                                 generic_pretrained: bool,
                                 output_dir: Path) -> None:
    """No test mode, no silent missing permissions, no ancestor-contaminated val.

    This is a guard library, not evidence of license approval or an executable
    training scheduler. Callers must verify checkpoint lineage by hash.
    """
    if config.get("role") != "development" or config.get("test_images_used") != 0:
        raise ValueError("explicit development role and zero test usage required")
    if any(k in config for k in ("test", "blind_test", "allow_blind_test")):
        raise ValueError("test paths/unlock flags are forbidden")
    if output_dir.exists():
        raise FileExistsError("run directory already exists; overwrite forbidden")
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("explicit data sources required")
    for name in sources:
        evidence = source_gates.get(name, {})
        if (evidence.get("development_allowed") is not True or not evidence.get("evidence_file")
                or not Path(evidence["evidence_file"]).is_file()):
            raise ValueError(f"source permission evidence missing: {name}")
    if not train_groups or not val_groups or train_groups & val_groups:
        raise ValueError("train/val must be nonempty and group-disjoint")
    if not generic_pretrained:
        if ancestor_train_groups is None:
            raise ValueError("unknown checkpoint training ancestry")
        if ancestor_train_groups & val_groups:
            raise ValueError("validation was used by an ancestor checkpoint")
    if config.get("repartitioned_cv") and not generic_pretrained:
        raise ValueError("new CV must start from verified generic pretraining, not prior wound checkpoints")
