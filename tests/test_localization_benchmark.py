"""Synthetic metric/isolation regression tests; no research image is read."""
import unittest
from pathlib import Path

import numpy as np

from experiments.review_v2.localization_benchmark import (
    binary_mask, box_iou, match_boxes, pixel_metrics, safe_path, size_name,
)


class LocalizationBenchmarkTests(unittest.TestCase):
    def test_repeated_rgb_binary_mask_is_supported(self):
        a = np.full((512, 512, 3), 255, np.uint8)
        self.assertEqual(binary_mask(a).shape, (512, 512))
        self.assertTrue(binary_mask(a).all())

    def test_colored_mask_and_wrong_type_fail(self):
        for a in [np.full((512, 512), .5, float),
                  np.tile([0, 255, 0], (512, 512, 1))]:
            with self.assertRaises(ValueError):
                binary_mask(a)

    def test_publisher_gray32_uses_frozen_positive_semantics(self):
        self.assertTrue(binary_mask(np.full((512, 512), 32, np.uint8)).all())

    def test_iou_identity_and_disjoint(self):
        np.testing.assert_allclose(box_iou([[0, 0, 10, 10]], [[0, 0, 10, 10], [20, 20, 30, 30]]), [[1, 0]])

    def test_duplicate_prediction_not_two_true_positives(self):
        r = match_boxes([[0, 0, 10, 10]], [[0, 0, 10, 10]] * 2, [.1, .9])
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (1, 1, 0))
        self.assertEqual(r["pairs"][0]["prediction"], 1)

    def test_prediction_cannot_match_two_gt(self):
        r = match_boxes([[0, 0, 10, 10]] * 2, [[0, 0, 10, 10]], [.9])
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (1, 0, 1))

    def test_empty_cases_are_counted(self):
        a = match_boxes([[0, 0, 10, 10]], [], [])
        b = match_boxes([], [[0, 0, 10, 10]], [.9])
        self.assertEqual(a["fn"], 1)
        self.assertEqual(b["fp"], 1)

    def test_missing_crop_is_failure_not_excluded(self):
        r = pixel_metrics(np.ones((10, 10)), np.zeros((10, 10)), None)
        self.assertEqual(r["crop_coverage"], 0)
        self.assertFalse(r["crop_complete95"])
        self.assertEqual(r["mask_dice"], 0)

    def test_exact_masks_and_half_open_crop(self):
        r = pixel_metrics(np.ones((10, 10)), np.ones((10, 10)), [0, 0, 10, 10])
        self.assertEqual((r["mask_iou"], r["mask_dice"], r["crop_coverage"]), (1, 1, 1))
        self.assertTrue(r["crop_complete95"])

    def test_negative_masks_do_not_inflate_positive_means(self):
        r = pixel_metrics(np.zeros((10, 10)), np.zeros((10, 10)), None)
        self.assertIsNone(r["mask_iou"])
        self.assertIsNone(r["crop_coverage"])

    def test_misaligned_masks_fail(self):
        with self.assertRaises(ValueError):
            pixel_metrics(np.zeros((2, 2)), np.zeros((3, 3)), None)

    def test_escaping_path_rejected(self):
        with self.assertRaises(ValueError):
            safe_path(Path.cwd(), "../test/image.png")

    def test_relative_size_boundaries(self):
        self.assertEqual(size_name([0, 0, 10, 10]), "small")
        self.assertEqual(size_name([0, 0, 100, 100]), "medium")
        self.assertEqual(size_name([0, 0, 200, 200]), "large")


if __name__ == "__main__":
    unittest.main()
