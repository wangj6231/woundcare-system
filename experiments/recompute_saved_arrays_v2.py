"""Versioned arithmetic on saved validation fold predictions only; no model/images/CI."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from .audit_classification_metrics import saved_array_arithmetic

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/results/predictions/val"
JSON_OUT = ROOT / "experiments/results/statistics/C-Arch-05-MS_saved_arrays_recomputed_v2.json"
MD_OUT = ROOT / "experiments/results/tables/Table2b_Classification_Performance_Recomputed_From_Saved_Arrays.md"
SEEDS = {42, 123, 3407, 2026, 999}


def compute_from_bundles(bundles: list[dict]) -> dict:
    expected = {(seed, fold) for seed in SEEDS for fold in range(1, 6)}
    actual = [(b["seed"], b["fold"]) for b in bundles]
    if len(actual) != 25 or set(actual) != expected or len(set(actual)) != 25:
        raise ValueError("expected exactly 25 unique saved seed/fold arrays")
    classes = None
    folds, by_seed = [], defaultdict(list)
    for b in sorted(bundles, key=lambda x: (x["seed"], x["fold"])):
        data = b["data"]
        if classes is None:
            classes = data["class_names"]
            if len(classes) != 7 or classes.count("Stab_wound") != 1:
                raise ValueError("expected seven-class classification order")
        if data["class_names"] != classes:
            raise ValueError("class order differs across saved folds")
        truth, pred = data["y_true"], data["y_pred"]
        if "y_prob" in data and len(data["y_prob"]) != len(truth):
            raise ValueError("saved probability array length differs")
        metrics, counts = saved_array_arithmetic(data)
        per_class = {}
        for name in classes:
            key = name.lower().replace(" ", "_").replace("/", "_")
            per_class[name] = {**counts[name],
                "precision_percent": metrics[f"{key}_precision"],
                "recall_percent": metrics[f"{key}_recall"],
                "f1_percent": metrics[f"{key}_f1"]}
        folds.append({"seed": b["seed"], "fold": b["fold"], "rows": len(truth),
                      "metrics_percent": {k: metrics[k] for k in
                          ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "stab_wound_recall")},
                      "per_class": per_class, "source_file": b["source_file"],
                      "sha256": b["sha256"]})
        by_seed[b["seed"]].append(data)
    seed_rows = []
    for seed in sorted(SEEDS):
        combined = {"class_names": classes,
                    "y_true": [x for d in by_seed[seed] for x in d["y_true"]],
                    "y_pred": [x for d in by_seed[seed] for x in d["y_pred"]]}
        if len(combined["y_true"]) != 720:
            raise ValueError(f"seed {seed} does not contain exactly 720 saved rows")
        metrics, counts = saved_array_arithmetic(combined)
        seed_rows.append({"seed": seed, "rows": 720, "metrics_percent": {
            k: metrics[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                                    "weighted_f1", "stab_wound_recall")},
            "per_class_recall_percent": {
                name: metrics[name.lower().replace(" ", "_").replace("/", "_") + "_recall"]
                for name in classes}, "per_class_counts": counts})
    if sum(f["rows"] for f in folds) != 3600:
        raise ValueError("expected 3600 saved fold rows")
    def describe(values):
        return {"mean_percent": round(mean(values), 4),
                "sd_percent_sample_ddof1": round(stdev(values), 4), "n": len(values)}
    seed_summary = {k: describe([s["metrics_percent"][k] for s in seed_rows]) for k in
                    ("accuracy", "macro_f1", "weighted_f1", "stab_wound_recall")}
    fold_summary = {k: describe([f["metrics_percent"][k] for f in folds]) for k in
                    ("accuracy", "macro_f1", "weighted_f1", "stab_wound_recall")}
    return {
        "SOURCE": "SAVED_FOLD_PREDICTION_ARRAYS",
        "ROW_IDENTITY": "NOT AVAILABLE",
        "PATIENT_INDEPENDENCE": "NOT VERIFIED",
        "GROUPED_BOOTSTRAP_CI": "NOT AVAILABLE",
        "GROUPED_CI_STATUS": "BLOCKED_BY_MISSING_HISTORICAL_ROW_IDENTITY",
        "NO_MODEL_INFERENCE_PERFORMED": True,
        "LOCKED_TEST_USED": False,
        "test_images_used": 0,
        "CO2Wounds_used_for_tuning": False,
        "interpretation": "seed-level aggregate metrics reconstructed from saved fold prediction arrays; not canonical image-identified OOF",
        "sd_definition": "sample SD (ddof=1) across the five seed aggregates; descriptive, not independent-dataset inference",
        "fold_distribution_definition": "descriptive fold variability only; 25 folds are not independent datasets",
        "classes": classes, "folds": folds, "seeds": seed_rows,
        "seed_aggregate_summary": seed_summary,
        "descriptive_fold_distribution": fold_summary,
        "historical_25_fold_mean_claim_check": {
            "accuracy_87_40_supported_at_2dp": round(fold_summary["accuracy"]["mean_percent"], 2) == 87.40,
            "macro_f1_87_36_supported_at_2dp": round(fold_summary["macro_f1"]["mean_percent"], 2) == 87.36,
            "note": "checks the historical mean of rounded fold scores, not the seed-pooled estimand"},
        "source_sha256": {f["source_file"]: f["sha256"] for f in folds},
    }


def render_table(report: dict) -> str:
    s = report["seed_aggregate_summary"]
    f = report["descriptive_fold_distribution"]
    lines = ["# Table 2b — Corrected classification metrics from saved arrays", "",
        "SOURCE = SAVED_FOLD_PREDICTION_ARRAYS; ROW_IDENTITY = NOT AVAILABLE; PATIENT_INDEPENDENCE = NOT VERIFIED.",
        "GROUPED_BOOTSTRAP_CI = NOT AVAILABLE; NO_MODEL_INFERENCE_PERFORMED = TRUE; LOCKED_TEST_USED = FALSE.",
        "This is seed-level aggregate metrics reconstructed from saved fold prediction arrays, not canonical image-identified OOF.",
        "", "| Metric | Mean of 5 seed-pooled scores (%) | Sample SD, ddof=1 (pp) | Historical 25-fold descriptive mean (%) |",
        "|---|---:|---:|---:|"]
    for key, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"),
                       ("weighted_f1", "Weighted-F1"), ("stab_wound_recall", "Stab_wound Recall")):
        lines.append(f"| {label} | {s[key]['mean_percent']:.2f} | {s[key]['sd_percent_sample_ddof1']:.2f} | {f[key]['mean_percent']:.2f} |")
    lines += ["", "The original 88.98 ± 9.30% Stab_wound claim is HISTORICAL_METRIC_SEMANTICS_ERROR; 17/25 fold summary values used Macro Recall. It is SUPERSEDED_FOR_THIS_METRIC.",
              "Historical Accuracy 87.40% and Macro-F1 87.36% are checked only as descriptive means across 25 saved folds; seed-pooled estimates are reported separately.",
              "No new CI: prediction-row to image/MD5-group linkage is unavailable. Do not treat 25 folds as independent datasets.",
              "", "| Seed | Rows | Accuracy (%) | Macro-F1 (%) | Stab_wound correct/support | Stab_wound Recall (%) |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in report["seeds"]:
        m, count = row["metrics_percent"], row["per_class_counts"]["Stab_wound"]
        lines.append(f"| {row['seed']} | {row['rows']} | {m['accuracy']:.2f} | {m['macro_f1']:.2f} | {count['correct']}/{count['support']} | {m['stab_wound_recall']:.2f} |")
    lines += ["", "Per-fold and all seven per-class precision/recall/F1, support and correct counts are in the companion versioned JSON.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if JSON_OUT.exists() or MD_OUT.exists():
        raise FileExistsError("versioned corrected report exists; overwrite forbidden")
    bundles = []
    for path in sorted(SOURCE.glob("C-Arch-05-MS_s*_fold*_predictions.json")):
        match = re.fullmatch(r"C-Arch-05-MS_s(\d+)_fold(\d+)_predictions.json", path.name)
        if match is None:
            raise ValueError("unexpected saved prediction name")
        raw = path.read_bytes()
        bundles.append({"seed": int(match[1]), "fold": int(match[2]),
                        "source_file": path.relative_to(ROOT).as_posix(),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "data": json.loads(raw.decode("utf-8"))})
    report = compute_from_bundles(bundles)
    if not JSON_OUT.parent.is_dir() or not MD_OUT.parent.is_dir():
        raise FileNotFoundError("versioned result directories must already exist")
    with JSON_OUT.open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
    with MD_OUT.open("x", encoding="utf-8") as out:
        out.write(render_table(report))
    print(json.dumps({"seed_aggregate_summary": report["seed_aggregate_summary"],
                      "descriptive_fold_distribution": report["descriptive_fold_distribution"],
                      "historical_check": report["historical_25_fold_mean_claim_check"]}, indent=2))


if __name__ == "__main__":
    main()
