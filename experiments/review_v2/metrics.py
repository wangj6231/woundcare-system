"""Explicit estimands and identity-aware development prediction validation.

Legacy arrays cannot establish sample/group identity. Do not manufacture IDs
from row numbers, and do not call a repeated-row bootstrap a grouped CI.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score, recall_score


def classification_metrics(y_true, y_pred, class_count: int) -> dict:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if yt.ndim != 1 or yt.shape != yp.shape or not len(yt) or class_count < 2:
        raise ValueError("nonempty aligned one-dimensional predictions required")
    if yt.dtype.kind not in "iu" or yp.dtype.kind not in "iu":
        raise ValueError("integer labels required; -1 denotes abstention")
    if np.any((yt < 0) | (yt >= class_count)) or np.any((yp < -1) | (yp >= class_count)):
        raise ValueError("class outside declared vocabulary")
    accepted = yp != -1
    labels = np.arange(class_count)
    support = np.bincount(yt, minlength=class_count)
    recall = recall_score(yt, yp, labels=labels, average=None, zero_division=0)
    return {
        "n": len(yt), "accepted": int(accepted.sum()),
        "coverage": float(accepted.mean()),
        "accuracy_all_samples": float((yt == yp).mean()),
        "accuracy_accepted_only": float((yt[accepted] == yp[accepted]).mean()) if accepted.any() else None,
        "macro_f1_fixed_classes": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, labels=labels, average="weighted", zero_division=0)),
        "recall": [float(v) if support[i] else None for i, v in enumerate(recall)],
        "support": support.tolist(),
        "macro_policy": "all declared classes; zero F1 for unsupported classes; absent recall is null",
    }


def validate_prediction_records(records: list[dict], class_count: int) -> None:
    if not records:
        raise ValueError("empty prediction records")
    required = {"image_id", "content_sha256", "group_id", "seed", "fold", "y_true", "y_pred"}
    seen, identities, groups, digest_labels, seeds_by_image = set(), {}, {}, {}, {}
    for r in records:
        if not required <= r.keys() or any(r[k] in (None, "") for k in required):
            raise ValueError("image/hash/group/seed/fold identities required; legacy rows cannot be bootstrapped")
        digest = str(r["content_sha256"])
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("canonical SHA256 required")
        key = (r["seed"], r["image_id"])
        if key in seen:
            raise ValueError("an image must appear once per seed in out-of-fold evaluation")
        seen.add(key)
        identity = (digest, r["group_id"], r["y_true"])
        if identities.setdefault(r["image_id"], identity) != identity:
            raise ValueError("identity or ground truth changed between seeds")
        if groups.setdefault(digest, r["group_id"]) != r["group_id"]:
            raise ValueError("identical content assigned to different groups")
        if digest_labels.setdefault(digest, r["y_true"]) != r["y_true"]:
            raise ValueError("conflicting ground truth for identical content")
        seeds_by_image.setdefault(r["image_id"], set()).add(r["seed"])
    if len({frozenset(v) for v in seeds_by_image.values()}) != 1:
        raise ValueError("incomplete seed coverage; repeated OOF pool is unbalanced")
    group_folds = {}
    for r in records:
        key = (r["seed"], r["group_id"])
        if group_folds.setdefault(key, r["fold"]) != r["fold"]:
            raise ValueError("a group spans validation folds within the same seed")
    classification_metrics([r["y_true"] for r in records], [r["y_pred"] for r in records], class_count)


def conditional_group_accuracy_ci(records: list[dict], class_count: int, *, draws=2000, seed=42) -> dict:
    """Group-cluster percentile CI, conditional on stored fitted OOF models.

    Resample groups with all their images and all repeated seeds together.
    This does NOT estimate training-set or model-selection uncertainty.
    """
    validate_prediction_records(records, class_count)
    if draws < 100:
        raise ValueError("at least 100 draws required")
    clusters = {}
    for r in records:
        correct, n = clusters.get(r["group_id"], (0, 0))
        clusters[r["group_id"]] = correct + int(r["y_true"] == r["y_pred"]), n + 1
    if len(clusters) < 2:
        raise ValueError("at least two groups required")
    counts = np.asarray(list(clusters.values()), dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sampled = counts[rng.integers(0, len(counts), len(counts))].sum(axis=0)
        values.append(sampled[0] / sampled[1])
    return {"observed_accuracy": float(counts[:, 0].sum() / counts[:, 1].sum()),
            "ci95_percentile": np.quantile(values, [.025, .975]).tolist(),
            "unique_groups": len(clusters), "draws": draws,
            "scope": "conditional on fitted OOF models; not independent runs or training uncertainty"}


def paired_mcnemar(first: dict[str, bool], second: dict[str, bool]) -> dict:
    if not first or first.keys() != second.keys():
        raise ValueError("paired sample identifiers must match exactly")
    lost = sum(first[k] and not second[k] for k in first)
    gained = sum(not first[k] and second[k] for k in first)
    return {"lost": lost, "gained": gained,
            "p_exact_two_sided": float(binomtest(gained, gained + lost, .5).pvalue) if gained + lost else 1.0,
            "scope": "requires independent pairs; nonsignificance is not proof of equivalence"}
