"""Validated resume metric semantics; never translate macro recall to class recall."""
from __future__ import annotations

import math


def resume_stab_recall(log_row: dict, prediction: dict | None) -> float:
    if prediction is not None:
        names = prediction.get("class_names")
        truth = prediction.get("y_true")
        pred = prediction.get("y_pred")
        if (not isinstance(names, list) or names.count("Stab_wound") != 1
                or not isinstance(truth, list) or not isinstance(pred, list)
                or not truth or len(truth) != len(pred)):
            raise ValueError("RESUME_STAB_RECALL_UNVERIFIED: invalid saved prediction arrays")
        index = names.index("Stab_wound")
        if any(type(x) is not int or x < 0 or x >= len(names) for x in truth + pred):
            raise ValueError("RESUME_STAB_RECALL_UNVERIFIED: invalid class IDs")
        positives = sum(x == index for x in truth)
        if positives == 0:
            raise ValueError("RESUME_STAB_RECALL_UNVERIFIED: zero Stab_wound support")
        value = 100 * sum(t == index and p == index for t, p in zip(truth, pred)) / positives
        explicit = log_row.get("stab_wound_recall")
        if explicit not in (None, "") and abs(float(explicit) - value) > .011:
            raise ValueError("RESUME_STAB_RECALL_UNVERIFIED: explicit metric conflicts with arrays")
        return round(value, 2)
    explicit = log_row.get("stab_wound_recall")
    try:
        value = float(explicit)
    except (TypeError, ValueError):
        raise ValueError("RESUME_STAB_RECALL_UNVERIFIED: explicit class metric or saved arrays required") from None
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError("RESUME_STAB_RECALL_UNVERIFIED: invalid explicit class metric")
    return value


def combine_fold_records(resumed: list[dict], fresh: list[dict]) -> dict:
    rows = resumed + fresh
    keys = [(r["seed"], r["fold"]) for r in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError("duplicate or missing fold records")
    vals = [float(r["stab_wound_recall"]) for r in rows]
    if any(not math.isfinite(v) or not 0 <= v <= 100 for v in vals):
        raise ValueError("unverified class recall in fold record")
    return {"stab_wound_recall_mean": sum(vals) / len(vals), "fold_count": len(vals)}
