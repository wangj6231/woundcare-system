"""Shared stock original-label C4/H4 trainer binding; no execution at import."""
from pathlib import Path
import json
from experiments.f01_validator import SharedValidation768
from experiments.f1_v2_safety import RuntimeSafety, SafetyStop


class Sink:
    def __init__(self, out):
        self.files = {k: (out / name).open('x', encoding='utf-8') for k, name in {
            'optimizer': 'optimizer_telemetry.jsonl', 'numerical': 'numerical_safety_telemetry.jsonl',
            'anchors': 'anchor_telemetry.jsonl', 'epochs': 'epochs.jsonl', 'validation': 'validation_runtime.jsonl'}.items()}
    def __call__(self, kind, row):
        self.files[kind].write(json.dumps(row, allow_nan=False, sort_keys=True) + '\n')
        self.files[kind].flush()
    def close(self):
        for f in self.files.values():
            if not f.closed:
                f.close()


class FrozenSampler:
    def __init__(self, orders, lookup):
        self.orders, self.lookup, self.next_epoch = orders, lookup, 0
    def __len__(self):
        return 771
    def __iter__(self):
        e = self.next_epoch
        self.next_epoch += 1
        # InfiniteDataLoader may prefetch beyond epoch299. It must never be CONSUMED.
        return iter([self.lookup[x] for x in self.orders[min(e, 299)]])
    def rewind_for_reset(self, epoch):
        self.next_epoch = epoch


class Ledger:
    def __init__(self, root, cfg, rows, orders, sink):
        self.root, self.cfg, self.rows, self.orders, self.sink = root, cfg, rows, orders, sink
        self.safety = RuntimeSafety(sink)
        self.epoch, self.batch, self.position, self.completed_epochs = -1, -1, 0, 0
        self.consumed = 0
        self.best_epoch = None
        self.val_seen = False
        self.train_shape = None
    def start_epoch(self, epoch):
        if epoch != self.completed_epochs or not 0 <= epoch < 300:
            raise SafetyStop('FAIL_ANCHOR_ORDER_PARITY')
        self.epoch, self.batch, self.position = epoch, -1, 0
        self.epoch_start_ops = self.safety.state.scheduled
    def start_batch(self):
        self.batch += 1
        self.safety.set_batch(self.epoch, self.batch)
    def anchors(self, paths):
        expected = self.orders[self.epoch][self.position:self.position + 4]
        actual = [Path(p).name for p in paths]
        if actual != expected or len(actual) != min(4, 771-self.position):
            raise SafetyStop('FAIL_ANCHOR_ORDER_PARITY')
        admitted = (self.root / self.rows[0]['image_path']).resolve().parent
        if any(Path(p).resolve().parent != admitted for p in paths):
            raise SafetyStop('FAIL_TRAIN_SOURCE_ROLE')
        self.sink('anchors', dict(arm=self.cfg['arm'], epoch=self.epoch, batch_index=self.batch,
            start_position=self.position, sample_ids=actual))
        self.position += len(actual)
        self.consumed += len(actual)
    def end_epoch(self):
        scheduled = self.safety.state.scheduled - self.epoch_start_ops
        if self.position != 771 or self.batch != 192 or scheduled != self.cfg['budget']['scheduled_optimizer_calls_per_epoch'][self.epoch]:
            raise SafetyStop('FAIL_ANCHOR_OR_OPTIMIZER_BUDGET')
        self.completed_epochs = self.epoch + 1
        self.sink('epochs', dict(epoch=self.epoch, anchors=771, scheduled=scheduled,
                                cumulative=self.safety.summary()))


def make_trainer(base, ledger, loader_builder=None):
    """Both real arms instantiate this SAME factory and class path."""
    from experiments.paired_sampling_runner import default_train_loader
    builder = loader_builder or default_train_loader
    class SharedHigherScaleTrainer(SharedValidation768, base):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
            if mode != 'train':
                admitted = ledger.root / 'outputs/isic_fuseg_pretrain_smoke_20260914/fuseg_dataset/images/val'
                if Path(dataset_path).resolve() != admitted.resolve():
                    raise SafetyStop('FAIL_VALIDATION_ROLE')
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            admitted = (ledger.root / ledger.rows[0]['image_path']).resolve().parent
            if Path(dataset_path).resolve() != admitted or rank not in (-1, 0):
                raise SafetyStop('FAIL_TRAIN_SOURCE_ROLE')
            # After stock model/AMP setup but BEFORE constructing/reading the dataset.
            ledger.safety.bind_scaler(self)
            if not self.amp or not self.scaler.is_enabled() or self.scaler.get_scale() != 65536:
                raise SafetyStop('FAIL_AMP_POLICY')
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            files = [Path(p).resolve() for p in dataset.im_files]
            expected = {(ledger.root/r['image_path']).resolve() for r in ledger.rows}
            if len(files) != 771 or set(files) != expected:
                raise SafetyStop('FAIL_ORIGINAL_TRAIN_DATASET_IDENTITY')
            lookup = {p.name: i for i, p in enumerate(files)}
            self.paired_sampler = FrozenSampler(ledger.orders, lookup)
            return builder(dataset, self.paired_sampler, batch_size, self.args.workers)

        def build_optimizer(self, *args, **kwargs):
            optimizer = super().build_optimizer(*args, **kwargs)
            self.optimizer = optimizer
            ledger.safety.bind_optimizer(self)
            return optimizer

        def get_validator(self):
            validator = super().get_validator()
            if validator.args.imgsz != 768 or self.test_loader.dataset.imgsz != 768:
                raise SafetyStop('FAIL_CHECKPOINT_SELECTION_PARITY')
            original = validator.preprocess
            def observed(batch):
                result = original(batch)
                if not ledger.val_seen:
                    ledger.sink('validation', dict(train_nominal_imgsz=self.args.imgsz,
                        val_nominal_imgsz=validator.args.imgsz, dataset_nominal_imgsz=self.test_loader.dataset.imgsz,
                        first_batch_shape=list(result['img'].shape), validator_args=vars(validator.args),
                        shared_override='experiments.f01_validator.SharedValidation768',
                        rule='stock rect/pad/stride; nominal768 does not guarantee768x768 tensor'))
                    ledger.val_seen = True
                return result
            validator.preprocess = observed
            return validator

        def preprocess_batch(self, batch):
            ledger.anchors(batch['im_file'])
            result = super().preprocess_batch(batch)
            if ledger.train_shape is None:
                ledger.train_shape = list(result['img'].shape)
            if list(result['img'].shape[-2:]) != [self.args.imgsz, self.args.imgsz]:
                raise SafetyStop('FAIL_TRAIN_SCALE_RUNTIME')
            return result

        def optimizer_step(self):
            return ledger.safety.opportunity(self, super().optimizer_step)

        def run_callbacks(self, event):
            if event == 'on_train_epoch_start':
                ledger.start_epoch(self.epoch)
                if self.epoch == self.epochs-self.args.close_mosaic:
                    self.paired_sampler.rewind_for_reset(self.epoch)
            elif event == 'on_train_batch_start':
                ledger.start_batch()
            elif event == 'on_train_epoch_end':
                ledger.end_epoch()
            elif event == 'on_model_save' and self.best_fitness == self.fitness:
                ledger.best_epoch = self.epoch + 1
            return super().run_callbacks(event)
    return SharedHigherScaleTrainer
