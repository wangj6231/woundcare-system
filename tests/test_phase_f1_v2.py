"""F1 adapter behavioral tests use disposable CPU tensors, never research runs."""
from types import SimpleNamespace
import pytest
import torch
from experiments.f1_v2_safety import RuntimeSafety, SafetyStop


def fixture():
    model = torch.nn.Linear(1, 1, bias=False)
    opt = torch.optim.AdamW(model.parameters(), lr=.0005, betas=(.937, .999))
    tr = SimpleNamespace(model=model, optimizer=opt, scaler=torch.amp.GradScaler('cpu'),
                         accumulate=1, scheduler=SimpleNamespace(state_dict=lambda: {}), ema=None)
    log = []
    safety = RuntimeSafety(lambda kind, row: log.append((kind, row)))
    safety.bind_scaler(tr)
    safety.bind_optimizer(tr)
    safety.set_batch(0, 0)
    return tr, safety, log


def test_raw_loss_is_stopped_before_backward():
    tr, safety, log = fixture()
    tr.loss = tr.model.weight.sum() * float('nan')
    tr.loss_items = torch.ones(4)
    with pytest.raises(SafetyStop, match='FAIL_NONFINITE_LOSS'):
        tr.scaler.scale(tr.loss)
    assert tr.model.weight.grad is None
    assert log[-1][1]['stop_pair'] is True


def cpu_batch(tr, safety, i=0, overflow=False):
    safety.set_batch(i // 193, i % 193)
    tr.loss = tr.model.weight.square().sum()
    tr.loss_items = torch.ones(4)
    tr.scaler.scale(tr.loss).backward()
    if overflow:
        tr.model.weight.grad.fill_(float('inf'))


def stock_step(tr, safety):
    from ultralytics.engine.trainer import BaseTrainer
    return safety.opportunity(tr, lambda: BaseTrainer.optimizer_step(tr))


def test_real_cpu_stock_amp_applied_and_skipped_are_distinguished():
    tr, safety, log = fixture()
    cpu_batch(tr, safety)
    assert stock_step(tr, safety) is None
    cpu_batch(tr, safety, 1, overflow=True)
    assert stock_step(tr, safety) is None
    assert safety.summary()['applied'] == 1
    assert safety.summary()['skipped'] == 1
    assert safety.summary()['unknown'] == 0
    ops = [row for kind, row in log if kind == 'optimizer']
    assert [r['optimizer_post_hook_count'] for r in ops] == [1, 0]
    assert [r['gradient_nonfinite'] for r in ops] == [False, True]


@pytest.mark.parametrize('scale', [768, 1024])
def test_both_real_trainer_classes_observe_cpu_optimizer_and_shared_validation(scale, tmp_path):
    from experiments.f1_v2_runner import Ledger, make_trainer
    from experiments.phase_f1_v2 import configs
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    from ultralytics.cfg import get_cfg
    from torch.utils.data import DataLoader
    from PIL import Image
    cfg = configs()['C4' if scale == 768 else 'H4']
    events = []
    ledger = Ledger(tmp_path, cfg, [], [], lambda kind, row: events.append((kind,row)))
    cls = make_trainer(SegmentationTrainer, ledger)
    trainer = cls.__new__(cls)  # No research model construction / training.
    trainer.args = get_cfg(overrides=cfg['training_args'])
    trainer.model = torch.nn.Linear(1, 1, bias=True)
    trainer.optimizer = trainer.build_optimizer(trainer.model, name='AdamW', lr=.0005, momentum=.937, decay=.0005)
    trainer.scaler = torch.amp.GradScaler('cpu')
    trainer.accumulate = 1
    trainer.scheduler = SimpleNamespace(state_dict=lambda: {})
    trainer.ema = None
    ledger.safety.bind_scaler(trainer)
    cpu_batch(trainer, ledger.safety)
    trainer.optimizer_step()
    assert ledger.safety.summary()['applied'] == 1
    assert ledger.safety.summary()['unknown'] == 0
    # Stock dataset + validator preprocess on synthetic image, no forward/inference.
    images, labels = tmp_path/'images', tmp_path/'labels'
    images.mkdir(); labels.mkdir()
    Image.new('RGB', (512,512), (127,80,50)).save(images/'synthetic.png')
    (labels/'synthetic.txt').write_text('0 0.2 0.2 0.4 0.2 0.4 0.4 0.2 0.4\n', encoding='utf-8')
    trainer.data = {'names': {0:'Wound'}, 'nc':1, 'channels':3}
    trainer.model.stride = torch.tensor([32])
    trainer.device = torch.device('cpu')
    trainer.save_dir = tmp_path/'validation'
    trainer.callbacks = {}
    val = trainer.build_dataset(str(images), 'val', 4)
    trainer.test_loader = DataLoader(val, batch_size=4, num_workers=0, collate_fn=val.collate_fn)
    validator = trainer.get_validator()
    validator.device = torch.device('cpu')
    validator.preprocess(next(iter(trainer.test_loader)))
    observed = [r for k,r in events if k == 'validation']
    assert len(observed) == 1
    assert observed[0]['val_nominal_imgsz'] == 768
    assert observed[0]['first_batch_shape'] == [1,3,800,800]
    assert trainer.args.imgsz == scale
    assert not torch.cuda.is_initialized()


def test_uniform_sampler_consumed_ledger_matches_all_frozen_orders(tmp_path):
    from experiments.f1_v2_runner import FrozenSampler, Ledger
    from experiments.phase_f1_v2 import configs, read, ROOT
    cfg = configs()['C4']
    rows = read(ROOT/cfg['manifest']['path'])['samples']
    orders = read(ROOT/cfg['sampling']['order_path'])['orders']
    ids = [r['sample_id'] for r in rows]
    sampler = FrozenSampler(orders, {key:i for i,key in enumerate(ids)})
    ledger = Ledger(ROOT, cfg, rows, orders, lambda *a: None)
    parent = (ROOT/rows[0]['image_path']).parent
    for epoch in range(300):
        if epoch == 270:
            sampler.rewind_for_reset(epoch)
        order = [ids[i] for i in sampler]
        assert len(set(order)) == 771
        ledger.start_epoch(epoch)
        for start in range(0,771,4):
            ledger.start_batch()
            ledger.anchors([parent/i for i in order[start:start+4]])
        ledger.safety.state.scheduled += cfg['budget']['scheduled_optimizer_calls_per_epoch'][epoch]
        ledger.end_epoch()
    assert ledger.completed_epochs == 300
    assert ledger.consumed == 231300
    assert ledger.safety.state.scheduled == 3741


def gate_fixture():
    return {'candidate': {'tp':200,'fn':41,'fp':20,'positive_images':186,'crop_complete95_images':170,
        'size_recall': {s:{'gt':g,'matched':m} for s,g,m in [('small',137,100),('medium',90,86),('large',14,14)]}},
        'diagnostics': {'very_small': {'support':49,'TP':25}}}


def test_advancement_requires_two_very_small_and_every_safety_gate():
    from experiments.f1_v2_results import advancement
    c, h = gate_fixture(), gate_fixture()
    h['diagnostics']['very_small']['TP'] = 26
    assert advancement(c,h)['status'] == 'FAIL'
    h['diagnostics']['very_small']['TP'] = 27
    assert advancement(c,h)['status'] == 'PASS'
    h['candidate']['size_recall']['large']['matched'] = 13
    assert advancement(c,h)['checks']['large'] is False
    assert advancement(c,h)['status'] == 'FAIL'


def test_image_role_allowlists_exclude_all_test_and_external_paths():
    from experiments.f1_v2_data_guard import admitted_images
    train = admitted_images('training'); val = admitted_images('evaluation')
    assert len(train) == 962 and len(val) == 382
    for p in train | val:
        assert not any(x.lower() in {'test','co2wounds','blind_test'} for x in p.parts)


def test_raw_loss_components_nonfinite_blocks_backward():
    tr, safety, _ = fixture()
    tr.loss = tr.model.weight.square().sum()
    tr.loss_items = torch.tensor([1., float('inf')])
    with pytest.raises(SafetyStop, match='FAIL_NONFINITE_LOSS'):
        tr.scaler.scale(tr.loss)
    assert tr.model.weight.grad is None


def test_actual_optimizer_exception_preserves_unknown_and_stops_pair():
    tr, safety, log = fixture()
    cpu_batch(tr, safety)
    with pytest.raises(SafetyStop, match='TRAINING_NUMERICAL_RUNTIME_INVALID'):
        safety.opportunity(tr, lambda: (_ for _ in ()).throw(RuntimeError('injected optimizer exception')))
    assert safety.summary()['unknown'] == 1
    assert safety.summary()['skipped'] == 0


def test_real_amp_16_consecutive_skips_stop_and_cross_epochs():
    tr, safety, _ = fixture()
    for i in range(185, 200):
        cpu_batch(tr, safety, i, overflow=True)
        stock_step(tr, safety)
    cpu_batch(tr, safety, 200, overflow=True)
    with pytest.raises(SafetyStop, match='FAIL_PERSISTENT_AMP_UPDATE_SKIPS'):
        stock_step(tr, safety)
    assert safety.summary()['skipped'] == 16
    assert safety.summary()['minimum_scaler'] == 1


def test_parameter_corruption_after_applied_update_stops_before_next_batch():
    tr, safety, _ = fixture()
    tr.optimizer.register_step_post_hook(lambda opt, a, k: tr.model.weight.data.fill_(float('nan')))
    cpu_batch(tr, safety)
    with pytest.raises(SafetyStop, match='FAIL_NONFINITE_PARAMETERS'):
        stock_step(tr, safety)


def test_scale_below_one_rejected_before_stock_update():
    tr, safety, _ = fixture()
    cpu_batch(tr, safety)
    tr.scaler._scale.fill_(.5)
    with pytest.raises(SafetyStop, match='FAIL_AMP_SCALE_COLLAPSE'):
        stock_step(tr, safety)
    assert len(tr.optimizer.state) == 0


def test_smallest_bin_is_selected_by_label_not_sorted_json_position():
    import json
    from experiments.f1_v2_results import smallest_bin
    d = json.loads(json.dumps({'small_bins': {'<0.10%': {'support': 27, 'TP': 9},
        '0.10-0.25%': {'support': 22, 'TP': 11}}}, sort_keys=True))
    assert smallest_bin(d) == {'support': 27, 'TP': 9}


def test_telemetry_write_failure_does_not_double_count_optimizer_opportunity():
    tr, safety, _ = fixture()
    def broken_sink(kind, row):
        if kind == 'optimizer':
            raise OSError('injected full disk')
    safety.sink = broken_sink
    cpu_batch(tr, safety)
    with pytest.raises(SafetyStop, match='FAIL_TELEMETRY_INCOMPLETE'):
        stock_step(tr, safety)
    assert safety.summary()['scheduled'] == 1
    assert safety.summary()['applied'] == 1
    assert safety.summary()['unknown'] == 0


def test_failed_control_never_starts_higher_or_evaluation(monkeypatch, tmp_path):
    from experiments import phase_f1_v2 as f, f1_v2_results as report
    monkeypatch.setattr(f, 'PAIR', tmp_path/'pair')
    monkeypatch.setattr(f, 'REPORT', tmp_path/'report.md')
    monkeypatch.setattr(f, 'OUTPUTS', {'C4':tmp_path/'c', 'H4':tmp_path/'h'})
    monkeypatch.setattr(f, 'verify_execution', lambda: None)
    monkeypatch.setattr(f, 'verify_frozen', lambda: None)
    monkeypatch.setattr(f, 'runtime', lambda: {'CPU_fixture':True})
    monkeypatch.setattr(f, 'sha', lambda p: 'fixture')
    commands, failures = [], []
    def fail_process(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(pid=1, wait=lambda: 1)
    monkeypatch.setattr(f.subprocess, 'Popen', fail_process)
    monkeypatch.setattr(report, 'interrupted', lambda error: failures.append(str(error)))
    with pytest.raises(ValueError, match='C4_FAILED_EXIT_1'):
        f.execute()
    assert len(commands) == 1
    assert commands[0][-3] == 'C4'
    assert len(failures) == 1
    assert not f.OUTPUTS['H4'].exists()


def test_existing_output_is_preserved_without_starting_model(monkeypatch, tmp_path):
    from experiments import phase_f1_v2 as f
    monkeypatch.setattr(f, 'PAIR', tmp_path/'pair')
    monkeypatch.setattr(f, 'REPORT', tmp_path/'report.md')
    monkeypatch.setattr(f, 'OUTPUTS', {'C4':tmp_path/'c', 'H4':tmp_path/'h'})
    monkeypatch.setattr(f, 'verify_execution', lambda: None)
    monkeypatch.setattr(f, 'verify_frozen', lambda: None)
    f.OUTPUTS['C4'].mkdir()
    with pytest.raises(ValueError, match='FAIL_OUTPUT_ALREADY_EXISTS'):
        f.execute()
    assert f.OUTPUTS['C4'].is_dir()
    assert not f.PAIR.exists()


def test_windows_worker_guard_allows_local_cpu_loader_but_denies_other_images(tmp_path):
    import subprocess, sys, json
    script = '''
import sys, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from experiments.f1_v2_data_guard import install, install_worker_hook, guarded_seed_worker
if __name__ == '__main__':
    out = Path(sys.argv[1])
    install(out, 'training'); install_worker_hook()
    from PIL import Image
    try:
        Image.open(out.parent/'forbidden_fixture.png')
    except RuntimeError as error:
        assert 'FAIL_DATA_ROLE_ACCESS' in str(error)
    else:
        raise AssertionError('guard did not reject foreign image')
    loader = DataLoader(TensorDataset(torch.arange(4)), num_workers=2, worker_init_fn=guarded_seed_worker)
    assert [int(batch[0]) for batch in loader] == [0,1,2,3]
    print(json.dumps({'workers':2, 'CUDA_initialized':torch.cuda.is_initialized()}))
'''
    result = subprocess.run([sys.executable,'-B','-c',script,str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {'workers':2,'CUDA_initialized':False}
