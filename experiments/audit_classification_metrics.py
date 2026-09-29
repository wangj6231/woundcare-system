"""Forensic consistency of saved classification arrays, exports and summaries.

This is NOT canonical OOF reconstruction or a new performance/CI report.
Uses no image pixels, models, seeds aggregation or bootstrap. Originals immutable.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

from .phase_a_audit import ROOT, dump, sha


def saved_array_arithmetic(data):
    names, truth, pred = data["class_names"], data["y_true"], data["y_pred"]
    if not truth or len(truth) != len(pred) or not names or len(set(names)) != len(names):
        raise ValueError("missing/mismatched arrays or ambiguous class order")
    if any(type(y) is not int or not 0 <= y < len(names) for y in truth + pred):
        raise ValueError("invalid class id")
    support, predicted, correct = Counter(truth), Counter(pred), Counter(t for t, p in zip(truth, pred) if t == p)
    precision, recall, f1 = [], [], []
    metrics = {"accuracy": sum(correct.values()) / len(truth) * 100}
    for index, name in enumerate(names):
        p = correct[index] / predicted[index] if predicted[index] else 0.0
        r = correct[index] / support[index] if support[index] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        precision.append(p); recall.append(r); f1.append(f)
        key = name.lower().replace(" ", "_").replace("/", "_")
        metrics.update({key + "_precision": p * 100, key + "_recall": r * 100, key + "_f1": f * 100})
    metrics.update({"macro_precision": sum(precision) / len(names) * 100,
                    "macro_recall": sum(recall) / len(names) * 100,
                    "macro_f1": sum(f1) / len(names) * 100,
                    "weighted_f1": sum(f1[i] * support[i] for i in range(len(names))) / len(truth) * 100})
    return {k: round(v, 2) for k, v in metrics.items()}, {
        name: {"correct": correct[i], "support": support[i], "predicted": predicted[i]}
        for i, name in enumerate(names)}


def audit_saved_metric_consistency(bundles, fold_summary):
    def matches(value, expected):
        return type(value) in (int, float) and math.isfinite(value) and abs(value - expected) <= .005
    export_errors, summary_errors, records = [], [], []
    summary_index = {}
    for row in fold_summary:
        key = (row["seed"], row["fold"])
        if key in summary_index:
            raise ValueError("duplicate summary fold")
        summary_index[key] = row
    for bundle in bundles:
        values, counts = saved_array_arithmetic(bundle["data"])
        detail = {"seed": bundle["seed"], "fold": bundle["fold"], "source_file": bundle["source_file"]}
        for key, stored in bundle["data"].get("metrics", {}).items():
            if key in values and not matches(stored, values[key]):
                export_errors.append({**detail, "metric": key, "stored_export_percent": stored,
                                      "array_arithmetic_percent": values[key]})
        summary = summary_index.get((bundle["seed"], bundle["fold"]))
        if summary is None:
            raise ValueError("prediction has no matching summary fold")
        for key in ("accuracy", "macro_f1", "weighted_f1", "stab_wound_recall"):
            if key not in summary:
                continue
            if not matches(summary[key], values[key]):
                summary_errors.append({**detail, "metric": key, "stored_summary_percent": summary[key],
                    "array_arithmetic_percent": values[key],
                    "equals_macro_recall_instead": key == "stab_wound_recall" and matches(summary[key], values["macro_recall"])})
        records.append({**detail, "array_arithmetic_percent": values, "integer_class_counts": counts})
    return {"scope": "Per-fold saved-array arithmetic; NOT canonical OOF, official replacement metrics or CI",
            "folds_checked": len(records), "exported_metric_mismatches": export_errors,
            "summary_metric_mismatches": summary_errors, "records": records,
            "verdict": "INCONSISTENT_STORED_METRICS" if export_errors or summary_errors else "CONSISTENT_ARITHMETIC_ONLY",
            "test_images_used": 0, "model_inference_executed": False, "bootstrap_executed": False,
            "unverified": ["row/image identity", "patient independence", "historical generation provenance", "ROC-AUC arithmetic"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    output = args.audit.resolve()
    if not output.is_relative_to((ROOT / "experiments/results/audit").resolve()) or not output.is_dir():
        raise ValueError("existing Phase A audit directory required")
    destination = output / "saved_metric_consistency.json"
    if destination.exists():
        raise FileExistsError("forensic output already exists")
    paths = sorted((ROOT / "experiments/results/predictions/val").glob("C-Arch-05-MS_s*_fold*_predictions.json"))
    summary_path = ROOT / "experiments/results/statistics/C-Arch-05-MS_multiseed_summary.json"
    before = {str(p): sha(p) for p in [*paths, summary_path]}
    bundles = []
    for path in paths:
        match = re.fullmatch(r"C-Arch-05-MS_s(\d+)_fold(\d+)_predictions.json", path.name)
        if not match:
            raise ValueError("unexpected prediction filename")
        bundles.append({"seed": int(match[1]), "fold": int(match[2]), "source_file": path.relative_to(ROOT).as_posix(),
                        "data": json.loads(path.read_text(encoding="utf-8"))})
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = audit_saved_metric_consistency(bundles, summary["all_fold_results"])
    result["source_sha256"] = before
    result["sources_unchanged"] = before == {str(p): sha(p) for p in [*paths, summary_path]}
    result["mismatch_counts_by_metric"] = dict(Counter(r["metric"] for r in result["summary_metric_mismatches"]))
    dump(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"records", "source_sha256", "summary_metric_mismatches"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
