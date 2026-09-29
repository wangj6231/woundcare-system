"""Synthetic arithmetic only: five seeds x five folds, no model or test images."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments import recompute_saved_arrays_v2 as recompute
from experiments.recompute_saved_arrays_v2 import compute_from_bundles, render_table, SEEDS


def synthetic_bundles():
    names = ["Abrasions", "Bruises", "Burns", "Cut", "Ingrown_nails", "Laceration", "Stab_wound"]
    rows = []
    for seed in SEEDS:
        for fold in range(1, 6):
            truth = [i % 7 for i in range(144)]
            pred = [0 if i % 7 == 6 and i % 2 == 0 else i % 7 for i in range(144)]
            rows.append({"seed": seed, "fold": fold, "source_file": f"synthetic-{seed}-{fold}",
                         "sha256": "synthetic", "data": {"class_names": names,
                         "y_true": truth, "y_pred": pred, "y_prob": [[] for _ in truth]}})
    return rows


class SavedArraysV2Tests(unittest.TestCase):
    def test_seed_aggregate_and_class_counts(self):
        report = compute_from_bundles(synthetic_bundles())
        self.assertEqual(len(report["folds"]), 25)
        self.assertEqual([s["rows"] for s in report["seeds"]], [720] * 5)
        self.assertEqual(sum(f["rows"] for f in report["folds"]), 3600)
        self.assertTrue(0 < report["seed_aggregate_summary"]["stab_wound_recall"]["mean_percent"] < 100)
        self.assertEqual(report["GROUPED_BOOTSTRAP_CI"], "NOT AVAILABLE")
        self.assertIn("Stab_wound", render_table(report))

    def test_duplicate_fold_rejected(self):
        rows = synthetic_bundles()
        rows[-1]["fold"] = rows[0]["fold"]
        rows[-1]["seed"] = rows[0]["seed"]
        with self.assertRaisesRegex(ValueError, "25 unique"):
            compute_from_bundles(rows)

    def test_missing_rows_rejected(self):
        rows = synthetic_bundles()
        rows[0]["data"]["y_true"].pop()
        with self.assertRaises(ValueError):
            compute_from_bundles(rows)

    def test_versioned_output_refuses_existing_without_reading_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "already.json"
            target.write_text("historical", encoding="utf-8")
            with patch.object(recompute, "JSON_OUT", target), \
                 patch.object(recompute, "MD_OUT", Path(tmp) / "new.md"), \
                 patch.object(recompute, "SOURCE", Path(tmp) / "absent"), \
                 patch("sys.argv", ["recompute_saved_arrays_v2"]):
                with self.assertRaises(FileExistsError):
                    recompute.main()
            self.assertEqual(target.read_text(encoding="utf-8"), "historical")


if __name__ == "__main__":
    unittest.main()
