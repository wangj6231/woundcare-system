"""Phase B1 contract tests use only synthetic saved-array JSON-shaped objects."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.phase_b1 import build_phase_b1, load_saved_bundles, write_phase_b1_outputs

SEEDS = (42, 123, 3407, 2026, 999)
NAMES = ("Abrasions", "Bruises", "Burns", "Cut", "Ingrown_nails", "Laceration", "Stab_wound")


def synthetic_bundles():
    bundles = []
    for seed in SEEDS:
        for fold in range(1, 6):
            truth = [i % 7 for i in range(144)]
            pred = truth.copy()
            probs = [[1.0 if c == y else 0.0 for c in range(7)] for y in pred]
            bundles.append({"seed": seed, "fold": fold, "source_file": f"synthetic_s{seed}_fold{fold}.json",
                            "sha256": "synthetic", "data": {"y_true": truth, "y_pred": pred,
                            "y_prob": probs, "class_names": list(NAMES)}})
    return bundles


class PhaseB1ContractTests(unittest.TestCase):
    def test_phase_b1_requires_25_fold_files(self):
        with self.assertRaisesRegex(ValueError, "FAIL_SAVED_PREDICTION_INTEGRITY"):
            build_phase_b1(synthetic_bundles()[:-1])

    def test_phase_b1_requires_five_complete_seeds(self):
        rows = synthetic_bundles()
        rows[-1]["seed"] = 42
        with self.assertRaisesRegex(ValueError, "FAIL_SAVED_PREDICTION_INTEGRITY"):
            build_phase_b1(rows)

    def test_phase_b1_rejects_duplicate_fold_identity(self):
        rows = synthetic_bundles()
        rows[-1]["seed"], rows[-1]["fold"] = rows[0]["seed"], rows[0]["fold"]
        with self.assertRaisesRegex(ValueError, "FAIL_SAVED_PREDICTION_INTEGRITY"):
            build_phase_b1(rows)

    def test_phase_b1_requires_720_rows_per_seed(self):
        rows = synthetic_bundles()
        for key in ("y_true", "y_pred", "y_prob"):
            rows[0]["data"][key].pop()
        with self.assertRaisesRegex(ValueError, "FAIL_SAVED_PREDICTION_INTEGRITY"):
            build_phase_b1(rows)

    def test_phase_b1_rejects_class_order_mismatch(self):
        rows = synthetic_bundles()
        rows[0]["data"]["class_names"] = list(reversed(NAMES))
        with self.assertRaisesRegex(ValueError, "FAIL_SAVED_PREDICTION_INTEGRITY"):
            build_phase_b1(rows)

    def test_phase_b1_rejects_invalid_probability_shape(self):
        rows = synthetic_bundles()
        rows[0]["data"]["y_prob"][0] = [1.0, 0.0]
        with self.assertRaisesRegex(ValueError, "FAIL_SAVED_PREDICTION_INTEGRITY"):
            build_phase_b1(rows)

    def test_phase_b1_uses_ddof1_for_seed_sd(self):
        rows = synthetic_bundles()
        for seed, errors in zip(SEEDS, (144, 108, 72, 36, 0)):
            remaining = errors
            for row in rows:
                if row["seed"] != seed:
                    continue
                data = row["data"]
                for index in range(min(remaining, len(data["y_true"]))):
                    data["y_pred"][index] = (data["y_true"][index] + 1) % 7
                    data["y_prob"][index] = [1.0 if c == data["y_pred"][index] else 0.0 for c in range(7)]
                remaining -= min(remaining, len(data["y_true"]))
        report = build_phase_b1(rows)
        self.assertEqual(report["seed_mean"]["accuracy"], 90.0)
        self.assertAlmostEqual(report["seed_sample_sd"]["accuracy"], 7.90569415, places=6)
        self.assertEqual(report["seed_min"]["accuracy"], 80.0)
        self.assertEqual(report["seed_max"]["accuracy"], 100.0)

    def test_phase_b1_rejects_prediction_probability_mismatch(self):
        rows = synthetic_bundles()
        rows[0]["data"]["y_pred"][0] = 2
        with self.assertRaisesRegex(ValueError, "FAIL_SAVED_PREDICTION_INTEGRITY"):
            build_phase_b1(rows)

    def test_phase_b1_rejects_corrupt_probability_and_missing_field(self):
        rows = synthetic_bundles()
        rows[0]["data"]["y_prob"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "FAIL_SAVED_PREDICTION_INTEGRITY"):
            build_phase_b1(rows)
        rows = synthetic_bundles()
        del rows[0]["data"]["y_prob"]
        with self.assertRaisesRegex(ValueError, "FAIL_SAVED_PREDICTION_INTEGRITY"):
            build_phase_b1(rows)

    def test_phase_b1_rejects_non_object_export(self):
        rows = synthetic_bundles()
        rows[0]["data"] = []
        with self.assertRaisesRegex(ValueError, "FAIL_SAVED_PREDICTION_INTEGRITY"):
            build_phase_b1(rows)

    def test_phase_b1_keeps_fold_and_seed_estimands_separate(self):
        rows = synthetic_bundles()
        # Fold 1 of seed 42 has only class-0; folds 2-5 remain seven-class.
        # Both estimands are valid but nonlinear Macro-F1 yields different values.
        row = next(x for x in rows if x["seed"] == 42 and x["fold"] == 1)
        row["data"]["y_true"] = [0] * 144
        row["data"]["y_pred"] = [0] * 144
        row["data"]["y_prob"] = [[1.0] + [0.0] * 6 for _ in range(144)]
        report = build_phase_b1(rows)
        self.assertNotAlmostEqual(report["seed_mean"]["macro_f1"],
                                  report["fold_descriptive_metrics"]["macro_f1"]["mean"])
        self.assertEqual(report["fold_interpretation"].split(";")[0],
                         "DESCRIPTIVE_FOLD_VARIABILITY_ONLY")

    def test_phase_b1_correct_stab_recall(self):
        rows = synthetic_bundles()
        remaining = 10
        for row in rows:
            if row["seed"] != 42:
                continue
            data = row["data"]
            for i, label in enumerate(data["y_true"]):
                if label == 6 and remaining:
                    data["y_pred"][i] = 0
                    data["y_prob"][i] = [1.0] + [0.0] * 6
                    remaining -= 1
        report = build_phase_b1(rows)
        self.assertEqual(report["per_class_seed_metrics"]["42"]["Stab_wound"]["correct"], 90)
        self.assertEqual(report["per_class_seed_metrics"]["42"]["Stab_wound"]["support"], 100)
        self.assertEqual(report["stab_wound_seed_summary"]["mean"], 98.0)
        self.assertAlmostEqual(report["stab_wound_seed_summary"]["sample_sd_ddof1"], 4.47213595)

    def test_phase_b1_generates_no_ci(self):
        report = build_phase_b1(synthetic_bundles())
        self.assertFalse(report["bootstrap_CI_generated"])
        self.assertEqual(report["ci_status"], "BLOCKED_BY_MISSING_HISTORICAL_ROW_IDENTITY")
        self.assertNotIn("confidence_intervals", report)

    def test_phase_b1_does_not_access_locked_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val = root / "experiments/results/predictions/val"
            val.mkdir(parents=True)
            decoy = root / "experiments/results/predictions/test/locked.json"
            decoy.parent.mkdir(parents=True)
            decoy.write_text("DO NOT READ", encoding="utf-8")
            real_read = Path.read_bytes
            def guarded(path):
                if path == decoy:
                    raise AssertionError("locked test accessed")
                return real_read(path)
            with patch.object(Path, "read_bytes", guarded):
                self.assertEqual(load_saved_bundles(root), [])

    def test_phase_b1_does_not_access_external_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiments/results/predictions/val").mkdir(parents=True)
            decoy = root / "outputs/segmentation/co2wounds_external_test_20260831/manifest.json"
            decoy.parent.mkdir(parents=True)
            decoy.write_text("DO NOT READ", encoding="utf-8")
            real_read = Path.read_bytes
            def guarded(path):
                if path == decoy:
                    raise AssertionError("external dataset accessed")
                return real_read(path)
            with patch.object(Path, "read_bytes", guarded):
                self.assertEqual(load_saved_bundles(root), [])

    def test_phase_b1_output_is_versioned_and_never_overwrites(self):
        report = build_phase_b1(synthetic_bundles())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = root / "experiments/results/statistics"
            tables = root / "experiments/results/tables"
            stats.mkdir(parents=True)
            tables.mkdir(parents=True)
            target = stats / "C-Arch-05-MS_phase_b1_final_summary.json"
            target.write_text("historical sentinel", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_phase_b1_outputs(root, report)
            self.assertEqual(target.read_text(encoding="utf-8"), "historical sentinel")
            self.assertEqual(list(tables.iterdir()), [])

    def test_phase_b1_writes_complete_per_class_and_confusion_outputs(self):
        report = build_phase_b1(synthetic_bundles())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiments/results/statistics").mkdir(parents=True)
            (root / "experiments/results/tables").mkdir(parents=True)
            files = write_phase_b1_outputs(root, report)
            self.assertEqual(len(files), 5)
            self.assertTrue(all(path.is_file() for path in files))
            csv_text = (root / "experiments/results/tables/Table_B1_PerClass_SeedPooled_Performance.csv").read_text(encoding="utf-8")
            self.assertEqual(len(csv_text.splitlines()), 8)
            loaded = json.loads((root / "experiments/results/statistics/C-Arch-05-MS_phase_b1_final_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(sum(sum(row) for row in loaded["descriptive_pooled_confusion_counts"]), 3600)
            self.assertEqual(len(loaded["seed_metrics"]), 5)


if __name__ == "__main__":
    unittest.main()
