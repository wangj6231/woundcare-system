"""Synthetic-only regression coverage for Phase A.5; no image datasets opened."""
import json
import builtins
import hashlib
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.scripts.resume_metrics import resume_stab_recall, combine_fold_records
from experiments.data_roles import require_source_role, validate_model_input_contract, Role
from experiments import data_roles


class ResumeSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.prediction = {"class_names": ["Other", "Stab_wound"],
                           "y_true": [0, 0, 1, 1], "y_pred": [0, 1, 1, 1]}

    def test_resume_never_uses_macro_recall_as_stab_recall(self):
        with self.assertRaisesRegex(ValueError, "RESUME_STAB_RECALL_UNVERIFIED"):
            resume_stab_recall({"recall": "75.0"}, None)

    def test_resume_recomputes_stab_recall_from_saved_predictions(self):
        self.assertEqual(resume_stab_recall({"recall": "75.0"}, self.prediction), 100.0)

    def test_resume_rejects_missing_per_class_metric_and_predictions(self):
        with self.assertRaisesRegex(ValueError, "RESUME_STAB_RECALL_UNVERIFIED"):
            resume_stab_recall({}, None)

    def test_mixed_resumed_and_new_folds_keep_same_metric_semantics(self):
        resumed = {"seed": 42, "fold": 1, "stab_wound_recall":
                   resume_stab_recall({"recall": "75.0"}, self.prediction)}
        fresh = {"seed": 42, "fold": 2, "stab_wound_recall": 50.0}
        result = combine_fold_records([resumed], [fresh])
        self.assertEqual(result["stab_wound_recall_mean"], 75.0)

    def test_conflicting_explicit_class_recall_and_arrays_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "conflicts with arrays"):
            resume_stab_recall({"stab_wound_recall": "75.0"}, self.prediction)


class RoleAndColorContractTests(unittest.TestCase):
    def test_unknown_role_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "SOURCE_ROLE_NOT_VERIFIED"):
            require_source_role("renamed_dataset", "train")

    def test_co2_cannot_be_renamed_into_training(self):
        with self.assertRaisesRegex(ValueError, "SOURCE_ROLE_FORBIDDEN"):
            require_source_role("new_alias", "train", path=Path("external_datasets/CO2Wounds-V2"))

    def test_locked_test_rejected_without_enumeration(self):
        with self.assertRaisesRegex(ValueError, "SOURCE_ROLE_FORBIDDEN"):
            require_source_role("C-Arch-05", "development_evaluate",
                                path=Path("yolo_wound_cls_dataset_v3/test"))

    def test_renamed_manifest_hash_still_identifies_external_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "innocent_name.csv"
            manifest.write_text("synthetic,external\n", encoding="utf-8")
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with patch.dict(data_roles._KNOWN_MANIFEST_SHA256, {digest: Role.HISTORICAL_EXTERNAL}):
                with self.assertRaisesRegex(ValueError, "SOURCE_ROLE_FORBIDDEN"):
                    require_source_role("renamed", "train", manifest_path=manifest)

    def test_pil_rgb_and_numpy_bgr_have_same_semantics(self):
        rgb = np.array([[[23, 101, 207]]], dtype=np.uint8)
        pil = Image.fromarray(rgb, "RGB")
        bgr_from_pil = validate_model_input_contract(pil, source_type="PIL_RGB")
        bgr_direct = validate_model_input_contract(rgb[:, :, ::-1].copy(), source_type="NUMPY_BGR")
        np.testing.assert_array_equal(bgr_from_pil, bgr_direct)

    def test_numpy_rgb_direct_rejected(self):
        with self.assertRaisesRegex(ValueError, "NUMPY_RGB"):
            validate_model_input_contract(np.zeros((1, 1, 3), np.uint8), source_type="NUMPY_RGB")

    def test_direct_classification_evaluator_denies_test_before_weights(self):
        from experiments.scripts.evaluate import evaluate, predict_yolo
        with self.assertRaisesRegex(ValueError, "SOURCE_ROLE_FORBIDDEN"):
            evaluate("missing.pt", "yolo_wound_cls_dataset_v3/test", split="test")
        with self.assertRaisesRegex(ValueError, "SOURCE_ROLE_FORBIDDEN"):
            predict_yolo("missing.pt", "yolo_wound_cls_dataset_v3/test")

    def test_direct_yolo_trainer_denies_locked_test_before_model_load(self):
        from experiments.scripts.train_cls_yolo import train_yolo_cls
        with self.assertRaisesRegex(ValueError, "SOURCE_ROLE_FORBIDDEN"):
            train_yolo_cls({"model": "unused.pt", "run_name": "synthetic",
                            "dataset": "yolo_wound_cls_dataset_v3/test"})


class ImportSafetyTests(unittest.TestCase):
    def test_importing_historical_scripts_has_no_write_side_effect(self):
        root = Path(__file__).resolve().parents[1]
        scripts = ("execute_bootstrap.py", "build_all_phase9_tables.py",
                   "generate_tables.py", "execute_tables.py", "statistical_analysis.py")
        real_open = builtins.open

        def guarded_open(file, mode="r", *args, **kwargs):
            if any(flag in mode for flag in "wax+"):
                raise AssertionError(f"import wrote file: {file}")
            return real_open(file, mode, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            old = Path.cwd()
            try:
                os.chdir(tmp)
                with patch("builtins.open", side_effect=guarded_open), \
                     patch.object(Path, "mkdir", side_effect=AssertionError("import created directory")):
                    for name in scripts:
                        with self.subTest(name=name):
                            runpy.run_path(str(root / "experiments" / "scripts" / name),
                                           run_name="phase_a5_import_probe")
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
