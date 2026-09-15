"""Synthetic-only regression checks for the isolated FUSeg training runner."""
import tempfile
import unittest
from pathlib import Path

from experiments.review_v2.fuseg_warmup_experiment import (
    check_dataset_yaml, sha, train_arguments, validate_finite_metrics,
    verify_inventory,
)


class FUSegRevisionTests(unittest.TestCase):
    def reference(self):
        return dict(optimizer="AdamW", lr0=.0005, warmup_bias_lr=.1,
                    warmup_epochs=5, imgsz=768, batch=4, epochs=300,
                    patience=80, seed=42, fraction=1.0, save_period=10,
                    plots=True, mosaic=.1, model="historical", data="historical")

    def test_only_bias_hyperparameter_changes(self):
        old = self.reference()
        new = train_arguments(old)
        self.assertEqual(new, {**{k: v for k, v in old.items()
                                 if k not in {"model", "data"}}, "warmup_bias_lr": 0.0})
        self.assertEqual(old["warmup_bias_lr"], .1)

    def test_smoke_is_bounded_without_mutating_formal(self):
        old = self.reference()
        smoke = train_arguments(old, smoke=True)
        self.assertEqual((smoke["epochs"], smoke["fraction"]), (2, .05))
        self.assertFalse(smoke["plots"])
        self.assertEqual(train_arguments(old)["epochs"], 300)

    def test_dataset_rejects_test_even_empty(self):
        root = Path.cwd()
        doc = dict(path=str(root), train="images/train", val="images/val", names={0: "Wound"})
        check_dataset_yaml(doc, root)
        with self.assertRaises(ValueError):
            check_dataset_yaml({**doc, "test": ""}, root)

    def test_dataset_rejects_wrong_class_and_split(self):
        root = Path.cwd()
        doc = dict(path=str(root), train="images/train", val="images/val", names={0: "Wound"})
        for change in ({"names": {0: "other"}}, {"val": "images/test"}):
            with self.assertRaises(ValueError):
                check_dataset_yaml({**doc, **change}, root)

    def test_inventory_integrity_and_unregistered_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            image = bundle / "dataset/images/train/synthetic.bin"
            label = bundle / "dataset/labels/train/synthetic.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"synthetic fixture only")
            label.write_text("0 .1 .1 .9 .1 .9 .9", encoding="utf-8")
            row = dict(split="train", image=image.relative_to(bundle).as_posix(),
                       label=label.relative_to(bundle).as_posix(),
                       image_sha256=sha(image), label_sha256=sha(label))
            verify_inventory(bundle, [row])
            extra = image.with_name("unregistered.bin")
            extra.write_bytes(b"extra")
            with self.assertRaises(ValueError):
                verify_inventory(bundle, [row])

    def test_inventory_rejects_forbidden_split_and_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            row = dict(split="test", image="../image.png", label="../label.txt")
            with self.assertRaisesRegex(ValueError, "forbidden split"):
                verify_inventory(bundle, [row])
            row["split"] = "train"
            with self.assertRaisesRegex(ValueError, "escapes"):
                verify_inventory(bundle, [row])

    def test_nonfinite_metrics_fail_closed(self):
        validate_finite_metrics({"map": 0.0, "loss": 2.0})
        for metrics in ({}, {"map": float("nan")}, {"loss": float("inf")}):
            with self.assertRaises(ValueError):
                validate_finite_metrics(metrics)


if __name__ == "__main__":
    unittest.main()
