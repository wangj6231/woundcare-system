"""Synthetic audit fixtures only: no model, clinical image or holdout reads."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.audit_classification_oof import audit_legacy_predictions, audit_identity_coverage


class OofAuditTests(unittest.TestCase):
    def test_malformed_probabilities_are_flagged_without_crashing(self):
        bundle = {"seed": 42, "fold": 1, "source_file": "bad.json",
                  "data": {"class_names": ["a", "b"], "y_true": [0, 1],
                           "y_pred": [0, 7], "y_prob": [[float("nan"), 0], [.5, .5]]}}
        result = audit_legacy_predictions([bundle], expected_per_seed=2, seeds=[42], folds=[1])
        self.assertEqual(result["numeric_verdict"], "FAIL")

    def test_complete_explicit_identity_can_pass_identity_checks_only(self):
        rows = [{"seed": s, "fold": 1, "image_id": i, "md5_group": "g"}
                for s in (42, 123) for i in ("a", "b")]
        self.assertTrue(audit_identity_coverage(rows, {"a": "g", "b": "g"}, [42, 123])["passed"])

    def test_unidentified_record_does_not_pass_identity_checks(self):
        result = audit_identity_coverage([{"seed": 42, "fold": 1}], {"a": "g"}, [42])
        self.assertFalse(result["passed"])
        self.assertEqual(result["unidentifiable_row_indices"], [0])

    def test_partial_fold_files_are_not_silently_complete(self):
        result = audit_legacy_predictions([], expected_per_seed=2, seeds=[42], folds=[1, 2])
        self.assertEqual(len(result["missing_prediction_files"]), 2)
        self.assertFalse(result["canonical_master_permitted"])

    def test_equal_counts_without_identity_do_not_prove_coverage(self):
        bundle = {"seed": 42, "fold": 1, "source_file": "synthetic.json",
                  "data": {"class_names": ["a", "b"], "y_true": [0, 1],
                           "y_pred": [0, 1], "y_prob": [[.9, .1], [.2, .8]]}}
        result = audit_legacy_predictions([bundle], expected_per_seed=2,
                                          seeds=[42], folds=[1], historical_count=3)
        self.assertEqual(result["actual_rows"], 2)
        self.assertEqual(result["historical_count_minus_actual"], 1)
        self.assertEqual(result["count_verdict"], "SUMMARY_COUNT_CONTRADICTS_STORED_PREDICTIONS")
        self.assertEqual(result["identity_verdict"], "BLOCKED_MISSING_PREDICTION_IDENTITIES")
        self.assertIsNone(result["duplicated_rows"])
        self.assertIsNone(result["missing_rows"])
        self.assertIsNone(result["unknown_rows"])
        self.assertFalse(result["canonical_master_permitted"])

    def test_duplicate_missing_and_unknown_are_explicit_when_ids_exist(self):
        expected = {"a": "g1", "b": "g2"}
        rows = [{"seed": 42, "fold": 1, "image_id": "a", "md5_group": "g1"},
                {"seed": 42, "fold": 2, "image_id": "a", "md5_group": "g1"},
                {"seed": 42, "fold": 2, "image_id": "unexpected", "md5_group": "g3"}]
        result = audit_identity_coverage(rows, expected, [42])
        self.assertEqual(result["duplicates"], [{"seed": 42, "image_id": "a", "count": 2}])
        self.assertEqual(result["missing"], [{"seed": 42, "image_id": "b"}])
        self.assertEqual(result["unknown"], [{"seed": 42, "image_id": "unexpected"}])
        self.assertEqual(result["group_cross_fold"], [{"seed": 42, "md5_group": "g1", "folds": [1, 2]}])
        self.assertFalse(result["passed"])


class IsolationAuditTests(unittest.TestCase):
    def test_missing_content_identity_cannot_pass_isolation_audit(self):
        from experiments.validate_dataset_isolation import validate_dataset_isolation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = validate_dataset_isolation(
                [{"sample_id": "a", "path": str(root / "train" / "a.png"),
                  "source_dataset": "FUSeg", "usage_role": "train"}],
                allowed_roots=[root], approved_sources=["FUSeg"])
            self.assertFalse(result["passed"])
            self.assertIn("MISSING_EXACT_CONTENT_IDENTITY", [v["code"] for v in result["violations"]])

    def test_development_inventory_never_opens_test_sentinel(self):
        from experiments.phase_a_audit import development_manifest
        from experiments.audit_classification_oof import CLASSES
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val", "test"):
                for label in CLASSES:
                    (root / split / label).mkdir(parents=True)
            (root / "train" / CLASSES[0] / "a.jpg").write_bytes(b"synthetic-train")
            (root / "val" / CLASSES[0] / "b.jpg").write_bytes(b"synthetic-val")
            sentinel = root / "test" / CLASSES[0] / "must-not-open.jpg"
            sentinel.write_bytes(b"SENTINEL")
            original = Path.open
            opened = []
            def checked(path, *args, **kwargs):
                self.assertNotIn("test", path.parts)
                opened.append(path)
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", checked):
                rows = development_manifest(root)
            self.assertEqual(len(rows), 2)
            self.assertNotIn(sentinel, opened)

    def test_test_path_and_path_escape_rejected_without_opening(self):
        from experiments.validate_dataset_isolation import validate_dataset_isolation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in (root / "test" / "sentinel.png", root / ".." / "escape.png"):
                with self.subTest(path=path):
                    result = validate_dataset_isolation(
                        [{"sample_id": "a", "path": str(path), "usage_role": "train", "source_dataset": "FUSeg"}],
                        allowed_roots=[root], approved_sources=["FUSeg"])
                    self.assertFalse(result["passed"])

    def test_known_group_patient_video_and_family_cross_split_rejected(self):
        from experiments.validate_dataset_isolation import validate_dataset_isolation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for field in ("md5", "sha256", "perceptual_group", "derived_family_id", "frame_family_id", "patient_id", "video_id"):
                with self.subTest(field=field):
                    rows = [{"sample_id": role, "path": str(root / role / "a.png"), "usage_role": role,
                             "source_dataset": "FUSeg", field: "same"} for role in ("train", "val")]
                    result = validate_dataset_isolation(rows, allowed_roots=[root], approved_sources=["FUSeg"])
                    self.assertFalse(result["passed"])
                    self.assertIn(field, [v.get("field") for v in result["violations"]])

    def test_renamed_protected_content_rejected_by_trusted_hash(self):
        from experiments.validate_dataset_isolation import validate_dataset_isolation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = validate_dataset_isolation(
                [{"sample_id": "a", "path": str(root / "train" / "renamed.png"),
                  "source_dataset": "FUSeg", "usage_role": "train", "sha256": "sealed"}],
                allowed_roots=[root], approved_sources=["FUSeg"], protected_hashes={"sealed"})
            self.assertIn("PROTECTED_CONTENT_HASH", [v["code"] for v in result["violations"]])

    def test_unknown_patient_and_sealed_hashes_are_not_verified(self):
        from experiments.validate_dataset_isolation import validate_dataset_isolation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = validate_dataset_isolation(
                [{"sample_id": "a", "path": str(root / "train" / "a.png"),
                  "source_dataset": "FUSeg", "usage_role": "train", "md5": "x"}],
                allowed_roots=[root], approved_sources=["FUSeg"])
            self.assertTrue(result["passed"])
            self.assertEqual(result["patient_verdict"], "PATIENT_LEVEL_INDEPENDENCE_NOT_VERIFIED")
            self.assertEqual(result["locked_content_verdict"], "NOT_VERIFIED_NO_TRUSTED_HASH_INVENTORY")

    def test_manifest_cannot_relabel_co2_as_development(self):
        from experiments.validate_dataset_isolation import validate_dataset_isolation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"sample_id": "a", "path": str(root / "train" / "a.png"),
                   "usage_role": "train", "source_dataset": "CO2Wounds-V2", "md5": "x"}
            result = validate_dataset_isolation([row], allowed_roots=[root],
                                                approved_sources=["CO2Wounds-V2"])
            self.assertFalse(result["passed"])
            self.assertIn("HISTORICAL_EXTERNAL_SOURCE", [e["code"] for e in result["violations"]])


class ColorContractAuditTests(unittest.TestCase):
    def test_installed_loader_reproduces_rgb_numpy_contract_mismatch(self):
        # No weights/inference. Observe the installed library's public input
        # loader using a three-distinct-channel image, not a grayscale fixture.
        import numpy as np
        from PIL import Image
        from ultralytics.data.loaders import LoadPilAndNumpy
        rgb = np.array([[[201, 71, 13], [202, 72, 14]]], dtype=np.uint8)
        pil_pixels = next(iter(LoadPilAndNumpy(Image.fromarray(rgb))))[1][0]
        bad_pixels = next(iter(LoadPilAndNumpy(rgb)))[1][0]
        bgr_pixels = next(iter(LoadPilAndNumpy(rgb[:, :, ::-1].copy())))[1][0]
        self.assertEqual(pil_pixels[0, 0].tolist(), [13, 71, 201])
        np.testing.assert_array_equal(pil_pixels, bgr_pixels)
        self.assertFalse(np.array_equal(pil_pixels, bad_pixels))


if __name__ == "__main__":
    unittest.main()
