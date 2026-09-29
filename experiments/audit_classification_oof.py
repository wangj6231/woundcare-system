"""Read-only legacy OOF forensic seam. Does not infer missing image identities.

Legacy arrays can establish counts and numeric consistency, NOT sample coverage.
Never import the legacy bootstrap/table scripts: some execute on import.
"""
from collections import Counter
import math

SEEDS = (42, 123, 3407, 2026, 999)
CLASSES = ("Abrasions", "Bruises", "Burns", "Cut", "Ingrown_nails", "Laceration", "Stab_wound")
CANONICAL_FIELDS = ["seed", "fold", "image_id", "image_path_relative", "md5_group",
                    "class_true", "class_pred", *["prob_" + x.lower() for x in CLASSES],
                    "correct", "checkpoint_id", "experiment_id", "prediction_source",
                    "prediction_row_index", "identity_evidence_sha256"]


def audit_legacy_predictions(bundles, *, expected_per_seed, seeds=SEEDS,
                             folds=(1, 2, 3, 4, 5), historical_count=None):
    """Inspect in-memory exported arrays. Identity counts stay null, never zero."""
    details, issues, seen, names = [], [], Counter(), None
    per_seed = {str(seed): 0 for seed in seeds}
    for bundle in bundles:
        seed, fold, data = bundle["seed"], bundle["fold"], bundle["data"]
        seen[(seed, fold)] += 1
        yt, yp, probs = (data.get(k, []) for k in ("y_true", "y_pred", "y_prob"))
        n = len(yt)
        classes = data.get("class_names", [])
        if names is None:
            names = classes
        if not classes or classes != names:
            issues.append({"source": bundle["source_file"], "issue": "CLASS_ORDER_MISMATCH"})
        if len(yp) != n or len(probs) != n:
            issues.append({"source": bundle["source_file"], "issue": "ARRAY_LENGTH_MISMATCH"})
        bad_labels = [i for i, pair in enumerate(zip(yt, yp))
                      if any(type(y) is not int or not 0 <= y < len(classes) for y in pair)]
        bad_probs, argmax_mismatch = [], []
        for i, p in enumerate(probs):
            if (not isinstance(p, list) or len(p) != len(classes) or not p
                    or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in p)
                    or abs(sum(p) - 1) > .002):
                bad_probs.append(i)
            elif (i < len(yp) and type(yp[i]) is int and 0 <= yp[i] < len(p)
                  and p[yp[i]] != max(p)):
                argmax_mismatch.append(i)
        if bad_labels or bad_probs or argmax_mismatch:
            issues.append({"source": bundle["source_file"], "issue": "INVALID_NUMERIC_ROWS",
                           "labels": bad_labels, "probabilities": bad_probs, "argmax": argmax_mismatch})
        if "n_samples" in data.get("metrics", {}) and data["metrics"]["n_samples"] != n:
            issues.append({"source": bundle["source_file"], "issue": "NSAMPLES_MISMATCH"})
        if seed not in seeds or fold not in folds:
            issues.append({"source": bundle["source_file"], "issue": "UNEXPECTED_SEED_OR_FOLD"})
        per_seed[str(seed)] = per_seed.get(str(seed), 0) + n
        details.append({"seed": seed, "fold": fold, "rows": n,
                        "class_counts": dict(Counter(map(str, yt))),
                        "source_file": bundle["source_file"], "exported_fields": sorted(data),
                        "prediction_image_id_present": False})
    missing_files = [{"seed": s, "fold": f} for s in seeds for f in folds if not seen[(s, f)]]
    duplicates = [{"seed": s, "fold": f, "files": n} for (s, f), n in seen.items() if n > 1]
    actual = sum(x["rows"] for x in details)
    return {"expected_rows": expected_per_seed * len(seeds), "actual_rows": actual,
            "historical_report_rows": historical_count,
            "historical_count_minus_actual": None if historical_count is None else historical_count - actual,
            "count_verdict": ("SUMMARY_COUNT_CONTRADICTS_STORED_PREDICTIONS"
                              if historical_count is not None and historical_count != actual else "COUNTS_RECONCILED"),
            "per_seed_rows": per_seed, "source_files": details,
            "missing_prediction_files": missing_files, "duplicate_seed_fold_files": duplicates,
            "numeric_issues": issues, "numeric_verdict": "FAIL" if issues else "PASS_NUMERIC_ONLY",
            "duplicated_rows": None, "missing_rows": None, "unknown_rows": None,
            "duplicate_ids": None, "identity_verdict": "BLOCKED_MISSING_PREDICTION_IDENTITIES",
            "affected_seeds": list(seeds), "affected_folds": list(folds),
            "canonical_master_permitted": False,
            "final_verdict": "BLOCKED_BY_EXTERNAL_EVIDENCE",
            "limitation": "Count equality is not image coverage. File and row ordinal identify evidence rows, not images."}


def audit_identity_coverage(records, expected, seeds):
    """Audit explicit image identities; caller must separately prove their lineage.

    Passing this is necessary, not sufficient, for canonical OOF acceptance.
    Expected maps file identity to exact content group, not patient identity.
    """
    counts, groups = Counter(), {}
    unknown, conflicts, unidentifiable = [], [], []
    for index, row in enumerate(records):
        if not all(row.get(k) is not None for k in ("seed", "fold", "image_id", "md5_group")):
            unidentifiable.append(index)
            continue
        seed, ident, group = row["seed"], row["image_id"], row["md5_group"]
        counts[(seed, ident)] += 1
        if seed not in seeds or ident not in expected:
            unknown.append({"seed": seed, "image_id": ident})
        elif expected[ident] != group:
            conflicts.append({"seed": seed, "image_id": ident})
        groups.setdefault((seed, group), set()).add(row["fold"])
    duplicates = [{"seed": s, "image_id": i, "count": n} for (s, i), n in counts.items() if n > 1]
    missing = [{"seed": s, "image_id": i} for s in seeds for i in expected if not counts[(s, i)]]
    cross = [{"seed": s, "md5_group": g, "folds": sorted(f)} for (s, g), f in groups.items() if len(f) > 1]
    return {"passed": not any((duplicates, missing, unknown, conflicts, cross, unidentifiable)),
            "duplicates": duplicates, "missing": missing, "unknown": unknown,
            "identity_conflicts": conflicts, "unidentifiable_row_indices": unidentifiable,
            "group_cross_fold": cross,
            "limitation": "Does not prove historical row mapping, training membership or checkpoint identity."}
