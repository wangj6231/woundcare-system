import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "experiments/results/tables"
ASSETS = ROOT / "docs/report_assets/classification_5x5_20261003"


class Classification5x5ReportAssetsTest(unittest.TestCase):
    def test_committed_tables_cover_all_models_and_classes(self):
        with (TABLES / "Table_CV25_Overall_Performance.csv").open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            overall = list(csv.DictReader(handle))
        with (TABLES / "Table_CV25_PerClass_Performance.csv").open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            per_class = list(csv.DictReader(handle))

        self.assertEqual(len(overall), 25)
        self.assertEqual(len(per_class), 175)
        self.assertEqual({int(row["seed"]) for row in overall}, {42, 123, 999, 2026, 3407})
        self.assertEqual({int(row["fold"]) for row in overall}, {1, 2, 3, 4, 5})
        self.assertEqual(len({row["model_id"] for row in overall}), 25)
        self.assertEqual(len({row["class_name"] for row in per_class}), 7)

    def test_public_manifest_proves_no_locked_test_or_inference(self):
        manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["saved_prediction_files"], 25)
        self.assertEqual(manifest["overall_rows"], 25)
        self.assertEqual(manifest["per_class_rows"], 175)
        self.assertFalse(manifest["model_inference_performed"])
        self.assertFalse(manifest["locked_test_enumerated"])
        self.assertEqual(manifest["locked_test_images_used"], 0)
        self.assertEqual(manifest["image_row_prediction_mapping"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
