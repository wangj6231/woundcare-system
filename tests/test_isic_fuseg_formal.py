import csv
import tempfile
import unittest
from pathlib import Path
import yaml

from experiments.review_v2.isic_fuseg_formal import finite_and_epochs, verify_stage_inventory, sha
from experiments.review_v2.isic_fuseg_gate import decide


class FormalHistoryTests(unittest.TestCase):
    def history(self, epochs, loss=1.0):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        run = Path(tmp.name)
        with (run / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['epoch', 'train/seg_loss'])
            writer.writeheader()
            writer.writerows({'epoch': e, 'train/seg_loss': loss} for e in epochs)
        return run

    def test_actual_ultralytics_one_based_history_can_advance_stage(self):
        run = self.history(range(1, 65))
        self.assertEqual(len(finite_and_epochs(run, 100)), 64)

    def test_missing_duplicate_and_fractional_epochs_are_rejected(self):
        for values in ([1, 3], [1, 1], [1, 2.5], [0, 1], [1, float('nan')]):
            with self.subTest(values=values), self.assertRaises((RuntimeError, ValueError)):
                finite_and_epochs(self.history(values), 100)

    def test_nonfinite_loss_empty_and_excess_history_are_rejected(self):
        for values, loss, limit in (([1], float('nan'), 100), ([], 1, 100), ([1, 2], 1, 1)):
            with self.subTest(values=values, loss=loss), self.assertRaises(RuntimeError):
                finite_and_epochs(self.history(values, loss), limit)


class RecoveryInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'fuseg_dataset'
        self.rows = []
        for split in ('train', 'val'):
            image = self.root / 'images' / split / 'sample.png'
            label = self.root / 'labels' / split / 'sample.txt'
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(split.encode())
            label.write_text('0 .1 .1 .5 .1 .5 .5', encoding='utf-8')
            self.rows.append(dict(split=split, image=image.relative_to(self.root.parent).as_posix(),
                                  label=label.relative_to(self.root.parent).as_posix(),
                                  image_sha256=sha(image), label_sha256=sha(label)))
        self.doc = dict(path=str(self.root), train='images/train', val='images/val', names={0: 'Wound'})
        self.yaml_path = self.root / 'dataset.yaml'
        self.yaml_path.write_text(yaml.safe_dump(self.doc), encoding='utf-8')

    def check(self):
        return verify_stage_inventory(self.root, self.rows, {'train': 1, 'val': 1}, 'Wound')

    def test_pinned_inventory_passes_and_content_change_fails(self):
        self.assertEqual(self.check()['images'], 2)
        (self.root / 'images/val/sample.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            self.check()

    def test_test_yaml_or_redirect_cannot_be_loaded(self):
        for changes in ({'test': ''}, {'val': 'images/test'}, {'path': str(self.root.parent)}):
            self.yaml_path.write_text(yaml.safe_dump({**self.doc, **changes}), encoding='utf-8')
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.check()

    def test_unregistered_image_and_manifest_escape_fail(self):
        (self.root / 'images/val/extra.png').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            self.check()
        self.rows[0]['image'] = '../outside.png'
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.check()

    def test_exact_overlap_fails_even_with_matching_hashes(self):
        (self.root / 'images/val/sample.png').write_bytes(b'train')
        self.rows[1]['image_sha256'] = self.rows[0]['image_sha256']
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.check()


class DevelopmentGateTests(unittest.TestCase):
    def test_baseline_rounding_is_not_a_false_failure(self):
        metrics = dict(precision=204/234, recall=204/241, f1=408/475,
                       crop_complete95_fraction=168/186)
        self.assertTrue(decide(metrics, {'mean_ms': 49})['passed'])

    def test_single_failed_or_missing_gate_blocks(self):
        metrics = dict(precision=.9, recall=.9, f1=.9, crop_complete95_fraction=.95)
        for key in metrics:
            with self.subTest(key=key):
                self.assertFalse(decide({**metrics, key: None}, {'mean_ms': 20})['passed'])
        self.assertFalse(decide(metrics, {'mean_ms': 50.01})['passed'])
        self.assertFalse(decide(metrics, {'mean_ms': float('nan')})['passed'])


if __name__ == '__main__':
    unittest.main()
