"""Saved-array arithmetic only; no real images, model or corrected CI."""
import unittest

from experiments.audit_classification_metrics import audit_saved_metric_consistency


class SavedMetricAuditTests(unittest.TestCase):
    def test_exported_accuracy_inconsistent_with_arrays_is_reported(self):
        bundles = [{"seed": 42, "fold": 1, "source_file": "synthetic.json", "data": {
            "class_names": ["A", "Stab_wound"], "y_true": [0, 1], "y_pred": [0, 0],
            "metrics": {"accuracy": 99.0}}}]
        report = audit_saved_metric_consistency(bundles, [{"seed": 42, "fold": 1, "accuracy": 50.0}])
        self.assertEqual(report["exported_metric_mismatches"][0]["array_arithmetic_percent"], 50.0)
        self.assertEqual(report["verdict"], "INCONSISTENT_STORED_METRICS")

    def test_duplicate_summary_fold_is_rejected(self):
        with self.assertRaises(ValueError):
            audit_saved_metric_consistency([], [{"seed": 42, "fold": 1}, {"seed": 42, "fold": 1}])

    def test_invalid_label_is_not_silently_excluded_from_denominator(self):
        bundles = [{"seed": 42, "fold": 1, "source_file": "synthetic.json", "data": {
            "class_names": ["A", "Stab_wound"], "y_true": [0, 1], "y_pred": [0, 9], "metrics": {}}}]
        with self.assertRaises(ValueError):
            audit_saved_metric_consistency(bundles, [{"seed": 42, "fold": 1}])

    def test_macro_recall_cannot_substitute_for_stab_recall(self):
        bundles = [{"seed": 42, "fold": 1, "source_file": "synthetic.json", "data": {
            "class_names": ["A", "Stab_wound"], "y_true": [0, 0, 1], "y_pred": [0, 1, 1],
            "metrics": {"accuracy": 66.67, "macro_recall": 75.0, "stab_wound_recall": 100.0}}}]
        summary = [{"seed": 42, "fold": 1, "accuracy": 66.67, "stab_wound_recall": 75.0}]
        report = audit_saved_metric_consistency(bundles, summary)
        self.assertEqual(report["exported_metric_mismatches"], [])
        self.assertEqual(len(report["summary_metric_mismatches"]), 1)
        mismatch = report["summary_metric_mismatches"][0]
        self.assertEqual(mismatch["metric"], "stab_wound_recall")
        self.assertEqual(mismatch["array_arithmetic_percent"], 100.0)
        self.assertEqual(mismatch["stored_summary_percent"], 75.0)
        self.assertTrue(mismatch["equals_macro_recall_instead"])


if __name__ == "__main__":
    unittest.main()
