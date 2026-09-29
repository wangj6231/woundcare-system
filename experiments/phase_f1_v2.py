"""One-shot F1 V2 orchestration. No resume/retry; separate C4 then H4 workers."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid
from experiments.phase_f02 import ROOT, PRO, sha, read, save, require, validate_pair, dataset_audit, historical_audit

PREFIX = 'F_HIGHER_SCALE_V2'
FREEZE_SHA = '4c50ca051ee5eb535d9d65259ebc2db9ed4de7d36129373d075e2f99ad5d7248'
REQUEST = Path('C:/Users/milo9/.codex/attachments/d834d458-3d3d-44cf-81dc-e74b1de6f800/貼上的文字.txt')
PAIR = ROOT / 'experiments/results/f_higher_scale_v2_seed42_pair_summary'
PRE = ROOT / 'experiments/results/f_higher_scale_v2_seed42_preflight'
REPORT = ROOT / 'docs/PHASE_F1_V2_PAIRED_HIGHER_SCALE_SEED42_20260927.md'
OUTPUTS = {'C4': ROOT / 'experiments/results/f_higher_scale_v2_seed42_control768',
           'H4': ROOT / 'experiments/results/f_higher_scale_v2_seed42_train1024'}
EXEC_FREEZE = PRO / 'F_HIGHER_SCALE_V2_F1_execution_freeze.json'
SOURCE_FILES = ['experiments/phase_f1_v2.py', 'experiments/f1_v2_runner.py', 'experiments/f1_v2_safety.py',
                'experiments/f1_v2_results.py', 'experiments/f1_v2_data_guard.py', 'tests/test_phase_f1_v2.py']
TEST_FILES = ['tests/test_phase_f1_v2.py', 'tests/test_phase_f02.py', 'tests/test_phase_d01.py',
              'tests/test_phase_d1.py', 'tests/test_phase_d0.py', 'tests/test_phase_c_execution.py',
              'tests/test_localization_benchmark.py', 'tests/test_phase_a5_guards.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def configs():
    c, h = [read(PRO/f'{PREFIX}_{kind}_config.json') for kind in ('control', 'experimental')]
    validate_pair(c, h)
    return {'C4': c, 'H4': h}


def verify_frozen():
    freeze = PRO/f'{PREFIX}_freeze.json'
    require(sha(freeze) == FREEZE_SHA, 'FAIL_PROTOCOL_MUTATED: freeze')
    for p, digest in read(freeze)['artifacts_sha256'].items():
        require(sha(ROOT/p) == digest, 'FAIL_PROTOCOL_MUTATED: ' + p)
    original = read(ROOT/'experiments/results/f_higher_scale_v2_protocol_revision/audit.json')
    for p, digest in original['protected_sha256'].items():
        require(sha(p) == digest, 'FAIL_PROTECTED_HASH: ' + p)
    require(historical_audit({}) == original['historical'], 'FAIL_HISTORICAL_EVIDENCE_MUTATED')
    return original


def verify_execution():
    lock = read(EXEC_FREEZE)
    require(lock['request_sha256'] == sha(REQUEST), 'FAIL_AUTHORIZATION_CHANGED')
    for p, digest in lock['source_sha256'].items():
        require(sha(ROOT/p) == digest, 'FAIL_EXECUTION_SOURCE_MUTATED: ' + p)
    require(sha(PRE/'tests.json') == lock['tests_sha256'], 'FAIL_BINDING_TEST_RECEIPT_MUTATED')
    return lock


def preflight():
    require(not PRE.exists() and not EXEC_FREEZE.exists() and not PAIR.exists() and not REPORT.exists()
            and all(not p.exists() for p in OUTPUTS.values()), 'FAIL_OUTPUT_ALREADY_EXISTS')
    old = verify_frozen()
    cfg = configs()
    data = dataset_audit(cfg['C4'], {})
    from experiments.phase_d1 import verify_data, verify_validation
    verify_data(ROOT, cfg['C4'])
    verify_validation(ROOT)
    PRE.mkdir()
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_ADDOPTS='', PYTEST_PLUGINS='')
    result = subprocess.run([sys.executable, '-B', '-m', 'pytest', *TEST_FILES, '-q'], cwd=ROOT,
                            env=env, capture_output=True, text=True, encoding='utf-8')
    save(PRE/'tests.json', dict(at=now(), exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
        test_files=TEST_FILES, scope='CPU-only binding tests and frozen protocol/role/evaluator tests; no research GPU run'))
    require(result.returncode == 0, 'FAIL_RUNTIME_SAFETY_ADAPTER_BINDING: targeted tests failed')
    save(PRE/'preflight.json', dict(data=data, protected_count=len(old['protected_sha256']),
        output_absent=True, test_images_used=0, CO2Wounds_used=False, initialization_deserialized=False,
        binding_evidence='real CPU GradScaler+AdamW+stock optimizer_step through same RuntimeSafety, both shared trainer arms and shared validator preprocessing'))
    save(EXEC_FREEZE, dict(at=now(), F1_authorized=True, request_sha256=sha(REQUEST), F02_freeze_sha256=FREEZE_SHA,
        source_sha256={p: sha(ROOT/p) for p in SOURCE_FILES}, tests_sha256=sha(PRE/'tests.json'),
        protocol_sha256=sha(PRO/f'{PREFIX}_protocol.json'), additional_experiments_authorized=False))
    print(result.stdout)


def runtime():
    import torch
    import numpy as np
    import ultralytics
    import importlib.metadata as metadata
    require(torch.cuda.is_available(), 'FAIL_GPU_UNAVAILABLE')
    return dict(Python=platform.python_version(), torch=torch.__version__, CUDA=torch.version.cuda,
        GPU=torch.cuda.get_device_name(0), Ultralytics=ultralytics.__version__, NumPy_actual=np.__version__,
        NumPy_import_path=np.__file__, NumPy_distribution_metadata=metadata.version('numpy'),
        albumentations=metadata.version('albumentations'))


def status(stage, **more):
    path = PAIR/'status.json'
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(dict(stage=stage, at=now(), test_images_used=0, LOCKED_TEST_USED=False,
            CO2Wounds_used=False, EXTERNAL_TEST_USED=False, **more), f, indent=2, allow_nan=False)
    os.replace(temp, path)


def rows_jsonl(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            require(bool(line.strip()), 'FAIL_EMPTY_TELEMETRY')
            yield json.loads(line)


def audit_arm(arm):
    out = OUTPUTS[arm]
    summary = read(out/'training_completion.json')
    require(summary['completed_epochs'] == 300 and summary['scheduled'] == 3741
        and summary['applied']+summary['skipped'] == 3741 and summary['unknown'] == 0
        and summary['failure'] is None and summary['runtime_exceptions'] == 0 and not summary['OOM'], 'FAIL_ARM_COMPLETION')
    orders = read(ROOT/configs()[arm]['sampling']['order_path'])['orders']
    consumed = []; current = 0; batches = 0
    for row in rows_jsonl(out/'anchor_telemetry.jsonl'):
        if row['epoch'] != current:
            require(row['epoch'] == current+1 and consumed == orders[current], 'FAIL_ANCHOR_ORDER_PARITY')
            current += 1; consumed = []
        require(row['start_position'] == len(consumed) and row['batch_index'] == len(consumed)//4, 'FAIL_ANCHOR_ORDER_PARITY')
        consumed.extend(row['sample_ids']); batches += 1
    require(current == 299 and consumed == orders[299] and batches == 57900, 'FAIL_ANCHOR_ORDER_PARITY')
    from experiments.f02_safety import NumericalSafety
    replay = NumericalSafety(); max_skip = 0; min_scale = float('inf'); per_epoch = [0]*300
    for row in rows_jsonl(out/'optimizer_telemetry.jsonl'):
        require(row['unknown'] == 0 and replay.observe(row) is None, 'FAIL_SAFETY_REPLAY')
        require(row['optimizer_post_hook_count'] == int(row['optimizer_update_applied']), 'FAIL_UPDATE_ACCOUNTING')
        per_epoch[row['epoch']] += 1
        max_skip = max(max_skip, replay.consecutive_skipped)
        min_scale = min(min_scale, row['scale_before'], row['scale_after'])
    for key in ('scheduled', 'applied', 'skipped', 'unknown'):
        require(replay.summary()[key] == summary[key], 'FAIL_TELEMETRY_TOTAL')
    require(per_epoch == configs()[arm]['budget']['scheduled_optimizer_calls_per_epoch'], 'FAIL_EPOCH_BUDGET')
    require(summary['max_consecutive_skips'] == max_skip and summary['minimum_scaler'] == min_scale, 'FAIL_SAFETY_SUMMARY')
    raw_count = op_count = 0
    for row in rows_jsonl(out/'numerical_safety_telemetry.jsonl'):
        if row['event'] == 'raw_loss':
            require(row['raw_total_loss_finite'] and row['raw_components_finite'], 'FAIL_RAW_LOSS')
            raw_count += 1
        elif row['event'] == 'optimizer_opportunity':
            require(not row['stop_pair'], 'FAIL_NUMERICAL_STOP')
            op_count += 1
        else:
            raise ValueError('FAIL_UNEXPECTED_NUMERICAL_EVENT')
    require(raw_count == 57900 and op_count == 3741, 'FAIL_MISSING_SAFETY_TELEMETRY')
    val = list(rows_jsonl(out/'validation_runtime.jsonl'))
    require(len(val) == 1 and val[0]['val_nominal_imgsz'] == val[0]['dataset_nominal_imgsz'] == 768, 'FAIL_CHECKPOINT_SELECTION_PARITY')
    require(val[0]['train_nominal_imgsz'] == configs()[arm]['training_args']['imgsz'], 'FAIL_SCALE')
    for name in ('best', 'last'):
        require(sha(out/f'{name}.pt') == summary[f'{name}_checkpoint_sha256'], 'FAIL_CHECKPOINT_MUTATED')
    return dict(status='PASS', anchor_batches=batches, raw_loss_batches=raw_count,
                scheduled=replay.scheduled, actual_anchor_order_matches_frozen=True, validation=val[0])


def train_arm(arm, token):
    lock = read(PAIR/'execution.lock')
    require(lock['token'] == token and lock['execution_freeze_sha256'] == sha(EXEC_FREEZE), 'FAIL_EXECUTION_AUTHORIZATION')
    verify_execution(); verify_frozen()
    cfg = configs()[arm]
    if arm == 'H4':
        audit_arm('C4')
    out = OUTPUTS[arm]
    require(not out.exists(), 'FAIL_OUTPUT_ALREADY_EXISTS')
    environment = runtime()
    require(environment == lock['environment'], 'FAIL_ENVIRONMENT_PARITY')
    out.mkdir()
    save(out/'execution.lock', dict(arm=arm, at=now(), pid=os.getpid(), protocol_sha256=sha(PRO/f'{PREFIX}_protocol.json'),
        config_sha256=sha(PRO/f'{PREFIX}_{"control" if arm == "C4" else "experimental"}_config.json'),
        runner_sha256=sha(ROOT/'experiments/f1_v2_runner.py'), safety_adapter_sha256=sha(ROOT/'experiments/f1_v2_safety.py'),
        manifest_sha256=cfg['manifest']['sha256'], anchor_order_sha256=cfg['sampling']['order_sha256'],
        initialization_sha256=cfg['initialization']['sha256'], environment=environment, resume=False))
    save(out/'runtime_environment.json', environment)
    from experiments.f1_v2_runner import Sink, Ledger, make_trainer
    from experiments.f1_v2_safety import SafetyStop
    sink = Sink(out)
    rows = read(ROOT/cfg['manifest']['path'])['samples']
    ledger = Ledger(ROOT, cfg, rows, read(ROOT/cfg['sampling']['order_path'])['orders'], sink)
    try:
        import torch
        from ultralytics import YOLO
        from ultralytics.models.yolo.segment.train import SegmentationTrainer
        from experiments.f1_v2_data_guard import install, install_worker_hook
        install(out, 'training'); install_worker_hook()
        torch.cuda.reset_peak_memory_stats()
        model = YOLO(str(ROOT/cfg['initialization']['path']))
        cls = make_trainer(SegmentationTrainer, ledger)
        def ready(trainer):
            for key, value in cfg['training_args'].items():
                require(getattr(trainer.args, key) == value, 'FAIL_RUNTIME_ARG: '+key)
            require(ledger.safety.handle is not None and ledger.safety.parameter_finite, 'FAIL_RUNTIME_SAFETY_ADAPTER_BINDING')
            require(type(trainer.optimizer).__name__ == 'AdamW' and not trainer.optimizer.state, 'FAIL_FRESH_OPTIMIZER')
            require(all(g['betas'] == (.937, .999) and g['lr'] == .0005 for g in trainer.optimizer.param_groups), 'FAIL_OPTIMIZER_RECIPE')
            require(sorted(g['weight_decay'] for g in trainer.optimizer.param_groups) == [0., 0., .0005], 'FAIL_WEIGHT_DECAY')
            require(trainer.amp and trainer.scaler.get_scale() == 65536 and trainer.start_epoch == 0, 'FAIL_AMP_OR_RESUME')
            require(trainer.accumulate == 16 and len(trainer.train_loader) == 193 and trainer.model.names == {0:'Wound'}, 'FAIL_RECIPE')
            require(sum(p.numel() for p in trainer.model.parameters()) == cfg['architecture']['runtime_parameters'], 'FAIL_MODEL_ARCHITECTURE')
            save(out/'runtime_binding.json', dict(status='PASS', raw_loss_guard_installed=True, optimizer_post_hook_installed=True,
                initialization_parameters_finite=True, val_nominal_imgsz=trainer.validator.args.imgsz,
                train_nominal_imgsz=trainer.args.imgsz, safety_adapter_sha256=sha(ROOT/'experiments/f1_v2_safety.py'),
                distinction='CPU capability proof before any research image; actual scaler bound before dataset construction; actual optimizer hook bound at stock optimizer construction before first batch'))
        def progress(trainer):
            status('TRAINING_'+arm, completed_epochs=ledger.completed_epochs, numerical=ledger.safety.summary())
        model.add_callback('on_train_start', ready)
        model.add_callback('on_fit_epoch_end', progress)
        args = deepcopy(cfg['training_args'])
        args.update(data=str(ROOT/cfg['data_yaml_path']), project=str(out), name='training')
        model.train(trainer=cls, **args)
        summary = dict(ledger.safety.summary(), completed_epochs=ledger.completed_epochs, best_epoch=ledger.best_epoch,
            last_epoch=ledger.completed_epochs, train_nominal_imgsz=cfg['training_args']['imgsz'], val_nominal_imgsz=768,
            train_first_batch_shape=ledger.train_shape, anchors_consumed=ledger.consumed,
            peak_GPU_allocated=torch.cuda.max_memory_allocated(), peak_GPU_reserved=torch.cuda.max_memory_reserved(),
            test_images_used=0, CO2Wounds_used=False, telemetry_complete=True,
            safety_adapter_sha256=sha(ROOT/'experiments/f1_v2_safety.py'))
        for name in ('best', 'last'):
            path = out/f'training/weights/{name}.pt'
            require(path.exists(), 'FAIL_CHECKPOINT_MISSING')
            os.link(path, out/f'{name}.pt')
            summary[f'{name}_checkpoint_sha256'] = sha(path)
        os.link(out/'training/results.csv', out/'results.csv')
        with (out/'results.csv').open(encoding='utf-8', newline='') as f:
            history = list(csv.DictReader(f))
        require([int(float(r['epoch'])) for r in history] == list(range(1, ledger.completed_epochs+1)), 'FAIL_EPOCH_HISTORY')
        sink.close()
        save(out/'training_completion.json', summary)
        require(ledger.completed_epochs == 300, 'EARLY_STOP_BEFORE_300_NO_RETRY')
        save(out/'telemetry_integrity.json', audit_arm(arm))
    except BaseException as exc:
        reason = str(exc) if isinstance(exc, (SafetyStop, ValueError)) else 'TRAINING_NUMERICAL_RUNTIME_INVALID'
        if 'out of memory' in str(exc).lower() or type(exc).__name__ == 'OutOfMemoryError':
            reason = 'FAIL_RESOURCE_RUNTIME'
            ledger.safety.oom = True
        if not isinstance(exc, (SafetyStop, ValueError)):
            ledger.safety.runtime_exceptions += 1
            ledger.safety.state.runtime_classification = 'TRAINING_NUMERICAL_RUNTIME_INVALID'
        ledger.safety.state.failure = ledger.safety.state.failure or reason
        save(out/'interruption.json', dict(at=now(), reason=reason, error_type=type(exc).__name__, error=str(exc),
            partial=ledger.safety.summary(), completed_epochs=ledger.completed_epochs, stop_pair=True, retry_allowed=False))
        raise
    finally:
        sink.close()


def execute():
    verify_execution(); verify_frozen()
    require(not PAIR.exists() and not REPORT.exists() and all(not p.exists() for p in OUTPUTS.values()), 'FAIL_OUTPUT_ALREADY_EXISTS')
    environment = runtime()
    PAIR.mkdir()
    token = str(uuid.uuid4())
    save(PAIR/'execution.lock', dict(at=now(), token=token, pid=os.getpid(), environment=environment,
        request_sha256=sha(REQUEST), execution_freeze_sha256=sha(EXEC_FREEZE), no_retry=True))
    try:
        for arm in ('C4', 'H4'):
            status('STARTING_'+arm)
            with (PAIR/f'train_{arm}.log').open('x', encoding='utf-8') as log:
                proc = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'experiments.phase_f1_v2', '--arm', arm, '--token', token],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                save(PAIR/f'{arm}_process.json', dict(pid=proc.pid, started_at=now()))
                code = proc.wait()
            require(code == 0, f'{arm}_FAILED_EXIT_{code}; no retry; see training log')
            audit_arm(arm); verify_frozen(); verify_execution()
        from experiments.f02_safety import compare_completed
        c, h = [read(OUTPUTS[a]/'training_completion.json') for a in ('C4', 'H4')]
        validity = compare_completed(c, h)
        require(validity['PAIRED_FIXED_BUDGET_VALID'] == 'YES', 'FAIL_PAIR_FIXED_BUDGET')
        save(PAIR/'pair_validity.json', validity)
        for arm in ('C4', 'H4'):
            status('EVALUATING_'+arm)
            with (PAIR/f'eval_{arm}.log').open('x', encoding='utf-8') as log:
                proc = subprocess.run([sys.executable, '-u', '-B', '-m', 'experiments.phase_f1_v2', '--eval', arm, '--token', token],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            require(proc.returncode == 0, 'FAIL_EVALUATION_'+arm)
        verify_frozen(); verify_execution()
        status('POSTFLIGHT')
        from experiments.f1_v2_results import finish
        finish()
    except BaseException as exc:
        from experiments.f1_v2_results import interrupted
        interrupted(exc)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--execute-authorized', action='store_true')
    parser.add_argument('--arm', choices=['C4', 'H4'])
    parser.add_argument('--eval', choices=['C4', 'H4'])
    parser.add_argument('--token')
    a = parser.parse_args()
    require(sum(bool(x) for x in (a.preflight, a.execute_authorized, a.arm, a.eval)) == 1, 'EXPLICIT_SINGLE_ACTION_REQUIRED')
    if a.preflight:
        preflight()
    elif a.execute_authorized:
        execute()
    elif a.arm:
        train_arm(a.arm, a.token)
    else:
        from experiments.f1_v2_results import evaluate
        evaluate(a.eval, a.token)
