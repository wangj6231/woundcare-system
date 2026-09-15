"""Offline regression tests. Synthetic arrays only; no model/test-image access."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.review_v2.guards import validate_new_development_run, validate_polygon_line, validate_optimizer_settings
from experiments.review_v2.metrics import (classification_metrics, conditional_group_accuracy_ci,
                                          paired_mcnemar, validate_prediction_records)
from woundcare_inference import CascadeConfig, classify_image, infer_segmentation_cascade
from experiments.review_v2.run_development import run_and_record, preflight


class MetricsTests(unittest.TestCase):
    def records(self):
        return [{"image_id": f"i{i}", "content_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                 "group_id": f"g{i}", "seed": seed, "fold": i, "y_true": i, "y_pred": i}
                for seed in (42, 123) for i in (0, 1)]

    def test_abstention_in_denominator(self):
        m = classification_metrics([0, 1, 1], [0, -1, 1], 2)
        self.assertAlmostEqual(m["accuracy_all_samples"], 2 / 3)
        self.assertEqual(m["accuracy_accepted_only"], 1)
        self.assertAlmostEqual(m["coverage"], 2 / 3)

    def test_absent_class_not_silently_dropped(self):
        m = classification_metrics([0, 0], [0, 0], 2)
        self.assertEqual(m["macro_f1_fixed_classes"], .5)
        self.assertIsNone(m["recall"][1])

    def test_no_accepted_predictions(self):
        self.assertIsNone(classification_metrics([0], [-1], 2)["accuracy_accepted_only"])

    def test_mismatched_arrays_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics([0, 1], [0], 2)

    def test_invalid_label_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics([0], [2], 2)

    def test_nan_prediction_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics([0], [float("nan")], 2)

    def test_legacy_identity_rejected(self):
        with self.assertRaises(ValueError):
            validate_prediction_records([{"y_true": 0, "y_pred": 0}], 2)

    def test_duplicate_oof_image_rejected(self):
        r = self.records()
        with self.assertRaises(ValueError):
            validate_prediction_records(r + [r[0]], 2)

    def test_incomplete_seed_coverage_rejected(self):
        with self.assertRaises(ValueError):
            validate_prediction_records(self.records()[:-1], 2)

    def test_content_conflict_rejected(self):
        r = self.records()
        r[1]["content_sha256"] = r[0]["content_sha256"]
        r[1]["group_id"] = r[0]["group_id"]
        with self.assertRaises(ValueError):
            validate_prediction_records(r, 2)

    def test_group_spans_folds_rejected(self):
        r = self.records()
        for x in r:
            x["group_id"] = "g"
        with self.assertRaises(ValueError):
            validate_prediction_records(r, 2)

    def test_cluster_resampling_deterministic_and_conditional(self):
        a = conditional_group_accuracy_ci(self.records(), 2, draws=100)
        self.assertEqual(a, conditional_group_accuracy_ci(self.records(), 2, draws=100))
        self.assertEqual(a["unique_groups"], 2)
        self.assertEqual(a["ci95_percentile"], [1, 1])
        self.assertIn("conditional", a["scope"])

    def test_exact_small_mcnemar(self):
        result = paired_mcnemar({"a": False, "b": False}, {"a": True, "b": True})
        self.assertEqual(result["p_exact_two_sided"], .5)

    def test_mcnemar_requires_paired_ids(self):
        with self.assertRaises(ValueError):
            paired_mcnemar({"a": True}, {"b": True})


class GuardTests(unittest.TestCase):
    def test_low_lr_bias_overshoot_rejected(self):
        with self.assertRaises(ValueError):
            validate_optimizer_settings({"optimizer": "AdamW", "lr0": .00002, "warmup_bias_lr": .1, "warmup_epochs": 2})

    def test_explicit_corrected_warmup(self):
        validate_optimizer_settings({"optimizer": "AdamW", "lr0": .00002, "warmup_bias_lr": 0, "warmup_epochs": 2})

    def test_missing_or_nonfinite_optimizer_setting(self):
        for settings in ({"optimizer": "auto"}, {"optimizer": "AdamW", "lr0": float("nan")},
                         {"optimizer": "AdamW", "lr0": .00002}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_optimizer_settings(settings)

    def test_valid_polygon(self):
        self.assertEqual(len(validate_polygon_line("0 0 0 1 0 1 1")), 3)

    def test_nonfinite_or_invalid_polygons(self):
        for line in ("0 nan 0 1 0 1 1", "0 inf 0 1 0 1 1", "0 0 0 0.5 0 1 0",
                     "0 0 0 1 0", "1 0 0 1 0 1 1", "0 -0.1 0 1 0 1 1"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                validate_polygon_line(line)

    def call_guard(self, **changes):
        config = {"role": "development", "test_images_used": 0, "sources": ["synthetic"]}
        config.update(changes.pop("config", {}))
        with tempfile.TemporaryDirectory() as folder:
            evidence = Path(folder) / "evidence.txt"
            evidence.write_text("Synthetic fixture only; not real dataset permission.", encoding="utf-8")
            kwargs = {"train_groups": {"a"}, "val_groups": {"b"}, "ancestor_train_groups": set(),
                      "generic_pretrained": True, "output_dir": Path(folder) / "new_run"}
            kwargs.update(changes)
            validate_new_development_run(config, {"synthetic": {"development_allowed": True, "evidence_file": str(evidence)}}, **kwargs)

    def test_clean_development_guard(self):
        self.call_guard()

    def test_test_unlock_rejected(self):
        for conf in ({"test": "images/test"}, {"allow_blind_test": False}, {"test_images_used": 1}, {"role": "test"}):
            with self.subTest(conf=conf), self.assertRaises(ValueError):
                self.call_guard(config=conf)

    def test_missing_source_evidence_rejected(self):
        with self.assertRaises(ValueError):
            self.call_guard(config={"sources": ["Redscar"]})

    def test_group_overlap_rejected(self):
        with self.assertRaises(ValueError):
            self.call_guard(val_groups={"a"})

    def test_ancestor_leakage_rejected(self):
        with self.assertRaises(ValueError):
            self.call_guard(generic_pretrained=False, ancestor_train_groups={"b"})

    def test_unknown_ancestry_rejected(self):
        with self.assertRaises(ValueError):
            self.call_guard(generic_pretrained=False, ancestor_train_groups=None)

    def test_repartitioned_finetuned_cv_rejected(self):
        with self.assertRaises(ValueError):
            self.call_guard(config={"repartitioned_cv": True}, generic_pretrained=False)

    def test_existing_run_rejected(self):
        with self.assertRaises(FileExistsError):
            self.call_guard(output_dir=Path(__file__).parent)


class FakeModel:
    names = {0: "Wound"}

    def __init__(self, result):
        self.result, self.calls = result, []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


class InferenceTests(unittest.TestCase):
    def test_nan_confidence_abstains(self):
        model = FakeModel(SimpleNamespace(probs=SimpleNamespace(top1=0, top1conf=float("nan"))))
        self.assertEqual(classify_image(model, np.zeros((8, 8, 3), np.uint8), CascadeConfig()), (None, None))

    def test_invalid_config_rejected(self):
        for config in ({"classifier_confidence": float("nan")}, {"segmenter_iou": 2},
                       {"classification_source": "unknown"}, {"segmenter_imgsz": 0}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                CascadeConfig(**config)

    def test_bgr_channels_not_silently_swapped(self):
        image = np.zeros((8, 8, 3), np.uint8)
        image[:, :, 0] = 231
        model = FakeModel(SimpleNamespace(probs=SimpleNamespace(top1=0, top1conf=.9)))
        classify_image(model, image, CascadeConfig())
        self.assertTrue(np.array_equal(model.calls[0]["source"], image))

    def test_invalid_image_rejected_before_predict(self):
        model = FakeModel(None)
        for image in (np.zeros((3, 3)), np.zeros((8, 8, 3), np.float32), np.zeros((0, 8, 3), np.uint8)):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                classify_image(model, image, CascadeConfig())
        self.assertEqual(model.calls, [])

    def test_nan_or_degenerate_mask_falls_back_safely(self):
        for polygon in (np.array([[np.nan, 0], [3, 0], [3, 3]]), np.array([[0, 0], [1, 1], [2, 2]])):
            segmenter = FakeModel(SimpleNamespace(masks=SimpleNamespace(xy=[polygon]), boxes=SimpleNamespace(conf=torch.tensor([.9]))))
            classifier = FakeModel(SimpleNamespace(probs=SimpleNamespace(top1=0, top1conf=.9)))
            out = infer_segmentation_cascade(np.zeros((8, 8, 3), np.uint8), segmenter, classifier,
                                             CascadeConfig(classification_source="segmentation_crop"))
            self.assertEqual(out["mode"], "full_image_fallback")
            self.assertIsNone(out["crop_box"])
            self.assertEqual(segmenter.calls[0]["iou"], .7)

    def test_mask_box_mismatch_no_reliable_roi(self):
        segmenter = FakeModel(SimpleNamespace(masks=SimpleNamespace(xy=[np.array([[0, 0], [5, 0], [5, 5]])]),
                                             boxes=SimpleNamespace(conf=torch.tensor([]))))
        classifier = FakeModel(SimpleNamespace(probs=SimpleNamespace(top1=0, top1conf=.9)))
        out = infer_segmentation_cascade(np.zeros((8, 8, 3), np.uint8), segmenter, classifier)
        self.assertIsNone(out["crop_box"])


class RunnerTests(unittest.TestCase):
    def test_evaluation_failure_cannot_pass(self):
        def fail(_):
            raise RuntimeError("synthetic evaluator failure")
        with tempfile.TemporaryDirectory() as folder:
            result = run_and_record(lambda: "fake-checkpoint", fail, Path(folder) / "run")
        self.assertEqual(result["status"], "FAILED_EVALUATION")

    def test_training_failure_cannot_evaluate(self):
        def fail():
            raise RuntimeError("synthetic train failure")
        def should_not_run(_):
            self.fail("evaluator ran despite training failure")
        with tempfile.TemporaryDirectory() as folder:
            result = run_and_record(fail, should_not_run, Path(folder) / "run")
        self.assertEqual(result["status"], "FAILED_TRAINING")

    def test_nonfinite_evaluation_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_and_record(lambda: None, lambda _: {"primary": {"map": float("nan")}}, Path(folder) / "run")
        self.assertEqual(result["status"], "FAILED_EVALUATION")

    def test_success_requires_complete_finite_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_and_record(lambda: None, lambda _: {"primary": {"map": .5}}, Path(folder) / "run")
        self.assertEqual(result["status"], "PASS_DEVELOPMENT_ONLY")

    def test_current_admission_stops_before_data_or_training(self):
        source_file = Path(__file__).resolve().parents[1] / "experiments/review_v2/source_admission.json"
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, "SOURCE_ADMISSION_BLOCKED"):
            preflight(source_file, Path(folder) / "run")


if __name__ == "__main__":
    unittest.main(verbosity=2)
