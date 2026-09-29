"""Phase B1 saved-array-only classification statistical reconstruction."""
from __future__ import annotations

import hashlib
import csv
import io
import json
import re
from pathlib import Path
import numpy as np
from statistics import mean, stdev

SEEDS = (42, 123, 3407, 2026, 999)
CLASS_NAMES = ("Abrasions", "Bruises", "Burns", "Cut", "Ingrown_nails", "Laceration", "Stab_wound")
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")


def load_saved_bundles(root: Path) -> list[dict]:
    """Read only the named saved validation JSON exports below a supplied root."""
    root = Path(root).resolve()
    source = (root / "experiments/results/predictions/val").resolve()
    if not source.is_relative_to(root) or any(
            "co2wounds" in p.casefold() or p.casefold() in {"test", "blind_test", "testing"}
            for p in source.parts):
        raise ValueError("FAIL_SAVED_PREDICTION_INTEGRITY: forbidden source path")
    bundles = []
    for path in sorted(source.glob("C-Arch-05-MS_s*_fold*_predictions.json")):
        match = re.fullmatch(r"C-Arch-05-MS_s(\d+)_fold(\d+)_predictions.json", path.name)
        resolved = path.resolve()
        if match is None or not resolved.is_relative_to(source) or not path.is_file():
            raise ValueError("FAIL_SAVED_PREDICTION_INTEGRITY: unexpected prediction export")
        raw = path.read_bytes()
        try:
            doc = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"FAIL_SAVED_PREDICTION_INTEGRITY: invalid JSON: {path.name}") from exc
        bundles.append({"seed": int(match[1]), "fold": int(match[2]),
                        "source_file": path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(raw).hexdigest(), "data": doc})
    return bundles


def _score(truth: list[int], pred: list[int]) -> tuple[dict, dict, list[list[int]]]:
    matrix = np.zeros((7, 7), dtype=np.int64)
    np.add.at(matrix, (truth, pred), 1)
    supports = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = matrix.diagonal()
    precision = np.divide(correct, predicted, out=np.zeros(7, dtype=float), where=predicted != 0)
    recall = np.divide(correct, supports, out=np.zeros(7, dtype=float), where=supports != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(7, dtype=float), where=(precision + recall) != 0)
    metrics = {"accuracy": float(correct.sum() / len(truth) * 100),
               "macro_precision": float(precision.mean() * 100),
               "macro_recall": float(recall.mean() * 100),
               "macro_f1": float(f1.mean() * 100),
               "weighted_f1": float(np.dot(f1, supports) / len(truth) * 100)}
    classes = {name: {"precision": float(precision[i] * 100),
                      "recall": float(recall[i] * 100), "f1": float(f1[i] * 100),
                      "support": int(supports[i]), "correct": int(correct[i])}
               for i, name in enumerate(CLASS_NAMES)}
    return metrics, classes, matrix.tolist()


def _describe(values: list[float]) -> dict:
    return {"mean": mean(values), "sample_sd_ddof1": stdev(values),
            "minimum": min(values), "maximum": max(values), "n": len(values)}


def build_phase_b1(bundles: list[dict]) -> dict:
    if len(bundles) != 25:
        raise ValueError("FAIL_SAVED_PREDICTION_INTEGRITY: expected 25 fold files")
    expected = {(seed, fold) for seed in SEEDS for fold in range(1, 6)}
    actual = [(row.get("seed"), row.get("fold")) for row in bundles]
    if set(actual) != expected or len(set(actual)) != 25:
        raise ValueError("FAIL_SAVED_PREDICTION_INTEGRITY: incomplete or duplicate seed/fold identity")
    for seed in SEEDS:
        try:
            counts = [len(row["data"]["y_true"]) for row in bundles if row["seed"] == seed]
        except (KeyError, TypeError):
            raise ValueError("FAIL_SAVED_PREDICTION_INTEGRITY: missing prediction arrays") from None
        if sum(counts) != 720 or any(n == 0 for n in counts):
            raise ValueError("FAIL_SAVED_PREDICTION_INTEGRITY: expected 720 rows per seed")
    if any(row.get("data", {}).get("class_names") != list(CLASS_NAMES) for row in bundles):
        raise ValueError("FAIL_SAVED_PREDICTION_INTEGRITY: class vocabulary/order mismatch")
    for row in bundles:
        data = row["data"]
        try:
            truth, pred, saved_prob = data["y_true"], data["y_pred"], data["y_prob"]
            if (not isinstance(truth, list) or not isinstance(pred, list)
                    or not isinstance(saved_prob, list) or len(truth) != len(pred) or len(pred) != len(saved_prob)
                    or any(type(y) is not int or not 0 <= y < len(CLASS_NAMES) for y in truth + pred)):
                raise ValueError("invalid label/array length")
            prob = np.asarray(saved_prob, dtype=float)
            if prob.shape != (len(truth), len(CLASS_NAMES)) or not np.isfinite(prob).all():
                raise ValueError("invalid probability shape/finite values")
            if ((prob < 0).any() or (prob > 1).any()
                    or not np.allclose(prob.sum(axis=1), 1.0, atol=1e-4)
                    or not np.array_equal(prob.argmax(axis=1), pred)):
                raise ValueError("invalid probability normalization or predicted class")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"FAIL_SAVED_PREDICTION_INTEGRITY: seed={row['seed']} fold={row['fold']}: {exc}") from exc
    folds = []
    seeds = []
    sorted_rows = sorted(bundles, key=lambda x: (x["seed"], x["fold"]))
    for row in sorted_rows:
        truth, pred = row["data"]["y_true"], row["data"]["y_pred"]
        metrics, classes, matrix = _score(truth, pred)
        folds.append({"seed": row["seed"], "fold": row["fold"], "rows": len(truth),
                      "metrics_percent": metrics, "per_class": classes,
                      "confusion_matrix": matrix, "source_file": row["source_file"],
                      "sha256": row["sha256"]})
    for seed in SEEDS:
        group = [row for row in sorted_rows if row["seed"] == seed]
        truth = [v for row in group for v in row["data"]["y_true"]]
        pred = [v for row in group for v in row["data"]["y_pred"]]
        metrics, classes, matrix = _score(truth, pred)
        seeds.append({"seed": seed, "folds": 5, "rows": len(truth),
                      "metrics_percent": metrics, "per_class": classes,
                      "confusion_matrix": matrix})
    seed_summary = {key: _describe([row["metrics_percent"][key] for row in seeds]) for key in METRICS}
    fold_summary = {key: _describe([row["metrics_percent"][key] for row in folds]) for key in METRICS}
    stab = _describe([row["per_class"]["Stab_wound"]["recall"] for row in seeds])
    per_class_summary = {}
    for name in CLASS_NAMES:
        per_class_summary[name] = {
            key: _describe([row["per_class"][name][key] for row in seeds])
            for key in ("precision", "recall", "f1")}
        per_class_summary[name]["support_per_seed"] = {
            str(row["seed"]): row["per_class"][name]["support"] for row in seeds}
    pooled_matrix = np.sum(np.asarray([row["confusion_matrix"] for row in seeds]), axis=0).tolist()
    return {
        "experiment_id": "C-Arch-05-MS", "source": "SAVED_FOLD_PREDICTION_ARRAYS",
        "source_prediction_files": [row["source_file"] for row in sorted_rows],
        "source_sha256": {row["source_file"]: row["sha256"] for row in sorted_rows},
        "num_seeds": 5, "num_folds": 5, "folds_per_seed": {str(seed): 5 for seed in SEEDS},
        "rows_per_seed": {str(seed): 720 for seed in SEEDS}, "total_saved_rows": 3600,
        "class_names": list(CLASS_NAMES), "probability_integrity": "PASS; shape, finite, range, sum atol=1e-4, argmax checked",
        "primary_estimand": "five seed-level aggregate evaluations; five held-out folds concatenated within each seed",
        "seed_metrics": seeds,
        "seed_mean": {key: seed_summary[key]["mean"] for key in METRICS},
        "seed_sample_sd": {key: seed_summary[key]["sample_sd_ddof1"] for key in METRICS},
        "seed_min": {key: seed_summary[key]["minimum"] for key in METRICS},
        "seed_max": {key: seed_summary[key]["maximum"] for key in METRICS},
        "stab_wound_seed_summary": stab,
        "fold_metrics": folds, "fold_descriptive_metrics": fold_summary,
        "fold_interpretation": "DESCRIPTIVE_FOLD_VARIABILITY_ONLY; not 25 independent experiments",
        "per_class_seed_metrics": {str(row["seed"]): row["per_class"] for row in seeds},
        "per_class_summary": per_class_summary,
        "descriptive_pooled_confusion_counts": pooled_matrix,
        "pooled_confusion_interpretation": "repeated predictions over the same development set under five random-seed evaluation procedures; not 3600 independent clinical observations",
        "historical_corrections": {
            "accuracy_25_fold_mean_percent": fold_summary["accuracy"]["mean"],
            "macro_f1_25_fold_mean_percent": fold_summary["macro_f1"]["mean"],
            "stab_historical_claim": "88.98 ± 9.30%; HISTORICAL_METRIC_SEMANTICS_ERROR; SUPERSEDED_FOR_STAB_WOUND_RECALL; 17/25 fold summary values affected",
            "fold_mean_vs_seed_pooled": "different estimands; Macro-F1 is nonlinear"},
        "ci_status": "BLOCKED_BY_MISSING_HISTORICAL_ROW_IDENTITY",
        "patient_independence_status": "NOT_VERIFIED",
        "external_7class_validation_status": "NOT_AVAILABLE",
        "clinical_validation_status": "NOT_CLINICALLY_VALIDATED",
        "locked_test_used": False, "external_dataset_used": False,
        "model_training_performed": False, "model_inference_performed": False,
        "bootstrap_CI_generated": False,
        "probability_based_metrics": "NOT_RECOMPUTED",
    }


def _output_paths(root: Path) -> dict[str, Path]:
    result = Path(root) / "experiments/results"
    return {
        "json": result / "statistics/C-Arch-05-MS_phase_b1_final_summary.json",
        "per_class_md": result / "tables/Table_B1_PerClass_SeedPooled_Performance.md",
        "per_class_csv": result / "tables/Table_B1_PerClass_SeedPooled_Performance.csv",
        "table2c": result / "tables/Table2c_Final_Classification_Development_Summary.md",
        "confusion_md": result / "tables/Table_B1_SeedPooled_Confusion_Counts.md",
    }


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _per_class_rows(report: dict) -> list[dict]:
    rows = []
    for name in CLASS_NAMES:
        item = report["per_class_summary"][name]
        rows.append({"Class": name,
                     "Mean Precision across seeds (%)": _fmt(item["precision"]["mean"]),
                     "SD Precision (pp)": _fmt(item["precision"]["sample_sd_ddof1"]),
                     "Mean Recall across seeds (%)": _fmt(item["recall"]["mean"]),
                     "SD Recall (pp)": _fmt(item["recall"]["sample_sd_ddof1"]),
                     "Mean F1 across seeds (%)": _fmt(item["f1"]["mean"]),
                     "SD F1 (pp)": _fmt(item["f1"]["sample_sd_ddof1"]),
                     "Support per seed": "; ".join(f"{seed}:{item['support_per_seed'][str(seed)]}" for seed in SEEDS)})
    return rows


def _markdown_table(rows: list[dict]) -> str:
    headers = list(rows[0])
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(str(row[h]) for h in headers) + " |" for row in rows)])


def _render_outputs(report: dict) -> dict[str, str]:
    per_class = _per_class_rows(report)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(per_class[0]))
    writer.writeheader(); writer.writerows(per_class)
    table2c = []
    for key, label in (("accuracy", "Accuracy"), ("macro_precision", "Macro Precision"),
                       ("macro_recall", "Macro Recall"), ("macro_f1", "Macro-F1"),
                       ("weighted_f1", "Weighted-F1")):
        fold = report["fold_descriptive_metrics"][key]
        table2c.append({"Metric": label,
                        "Primary five-seed pooled result": f"{_fmt(report['seed_mean'][key])}% ± {_fmt(report['seed_sample_sd'][key])} pp; range {_fmt(report['seed_min'][key])}–{_fmt(report['seed_max'][key])}%",
                        "Historical 25-fold descriptive result": f"{_fmt(fold['mean'])}% ± {_fmt(fold['sample_sd_ddof1'])} pp (descriptive sample SD)",
                        "Interpretation": "Different estimand; folds are not independent datasets"})
    stab = report["stab_wound_seed_summary"]
    table2c.append({"Metric": "Stab_wound Recall",
                    "Primary five-seed pooled result": f"{_fmt(stab['mean'])}% ± {_fmt(stab['sample_sd_ddof1'])} pp; range {_fmt(stab['minimum'])}–{_fmt(stab['maximum'])}%",
                    "Historical 25-fold descriptive result": "88.98% ± 9.30 pp — HISTORICAL_METRIC_SEMANTICS_ERROR",
                    "Interpretation": "Historical Stab claim superseded; 17/25 summary fold values used Macro Recall"})
    confusion_lines = ["# Phase B1 confusion counts reconstructed from saved arrays", "",
        "Each seed matrix aggregates five held-out folds. The pooled matrix repeats the same development dataset under five seed procedures; it is not 3,600 independent clinical observations.", ""]
    for item in report["seed_metrics"] + [{"seed": "DESCRIPTIVE_POOLED_FIVE_SEEDS",
                                             "confusion_matrix": report["descriptive_pooled_confusion_counts"]}]:
        confusion_lines += [f"## Seed {item['seed']}", "", "| True \\ predicted | " + " | ".join(CLASS_NAMES) + " |",
                            "|---|" + "---:|" * 7]
        for name, counts in zip(CLASS_NAMES, item["confusion_matrix"]):
            confusion_lines.append("| " + name + " | " + " | ".join(str(v) for v in counts) + " |")
        confusion_lines.append("")
    no_ci = "Confidence intervals were not recomputed because historical prediction rows cannot be reliably linked to image/content groups required for dependence-aware resampling."
    return {
        "json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "per_class_md": "# Phase B1 per-class performance — five seed-pooled evaluations\n\n" +
            "All percentages are means/sample SD across five seed-level aggregates (ddof=1), not CI.\n\n" +
            _markdown_table(per_class) + "\n\n" + no_ci + "\n",
        "per_class_csv": csv_buffer.getvalue(),
        "table2c": "# Table 2c — Final classification development summary\n\n" +
            "Primary: five seed-level aggregate evaluations. Historical: 25-fold descriptive variability only.\n\n" +
            _markdown_table(table2c) + "\n\n" + no_ci +
            "\n\nNo test or external cohort was combined with these development metrics.\n",
        "confusion_md": "\n".join(confusion_lines),
    }


def write_phase_b1_outputs(root: Path, report: dict) -> list[Path]:
    paths = _output_paths(root)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Phase B1 versioned output already exists; overwrite forbidden")
    if any(not path.parent.is_dir() for path in paths.values()):
        raise FileNotFoundError("Phase B1 result directories must already exist")
    rendered = _render_outputs(report)
    for key, path in paths.items():
        with path.open("x", encoding="utf-8", newline="") as out:
            out.write(rendered[key])
    return list(paths.values())


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if any(path.exists() for path in _output_paths(root).values()):
        raise FileExistsError("Phase B1 versioned output already exists; overwrite forbidden")
    bundles = load_saved_bundles(root)
    report = build_phase_b1(bundles)
    for path, digest in report["source_sha256"].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise ValueError("FAIL_SAVED_PREDICTION_INTEGRITY: source changed during reconstruction")
    paths = write_phase_b1_outputs(root, report)
    print(json.dumps({"outputs": [str(p) for p in paths],
                      "seed_mean": report["seed_mean"], "seed_sample_sd": report["seed_sample_sd"],
                      "stab_wound": report["stab_wound_seed_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
