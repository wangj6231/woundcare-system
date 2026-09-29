"""One fresh H4 replacement. Frozen F1 functions reused with explicit process-local routing.

No frozen source edits. No entry point can train C4 or original H4. Recovery evaluation
writes to NEW pair/evaluation directories, never the frozen comparator directory.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import platform
import subprocess
import sys
import uuid
import zipfile
from experiments.phase_f02 import ROOT, PRO, sha, read, save, require

R = ROOT/'experiments/results'
F11 = R/'f_higher_scale_v2_interruption_audit'
C4 = R/'f_higher_scale_v2_seed42_control768'
OLD_H4 = R/'f_higher_scale_v2_seed42_train1024'
OUT = R/'f_higher_scale_v2_recovery_seed42_h4_1024'
PRE = R/'f_higher_scale_v2_recovery_seed42_preflight'
PAIR = R/'f_higher_scale_v2_recovery_seed42_pair_summary'
REPORT = ROOT/'docs/PHASE_F12_H4_REPLACEMENT_RECOVERY_20260928.md'
REQUEST = Path('C:/Users/milo9/.codex/attachments/41294aa5-2c82-45bb-9058-1bd17b120cfb/貼上的文字.txt')
FREEZE = PRE/'execution_freeze.json'
PREFIX = 'F_HIGHER_SCALE_V2'
SCHEMA = 'F_HIGHER_INPUT_SCALE_V2_RECOVERY_H4_REPLACEMENT'
SOURCES = ['experiments/phase_f12.py', 'experiments/f12_results.py', 'tests/test_phase_f12.py']
_ORIGINAL_AUDIT = None


def now(): return datetime.now(timezone.utc).isoformat()


def env_vars():
    return dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_ADDOPTS='', PYTEST_PLUGINS='', NO_ALBUMENTATIONS_UPDATE='1')


def status(stage, **more):
    if stage == 'TRAINING_H4': stage = 'TRAINING_H4_R'
    path = PRE/'status.json'
    tmp = path.with_suffix('.tmp')
    payload = dict(stage=stage, at=now(), schema=SCHEMA, ORIGINAL_F1_STATUS='INVALID_OR_INTERRUPTED',
        ORIGINAL_PAIR_FIXED_BUDGET_VALID='NO', test_images_used=0, LOCKED_TEST_USED=False,
        CO2Wounds_used=False, EXTERNAL_TEST_USED=False, NEW_RETRY_AUTHORIZED='NO', **more)
    with tmp.open('w', encoding='utf-8') as stream: json.dump(payload, stream, indent=2, allow_nan=False)
    os.replace(tmp, path)


def configs():
    c = read(PRO/f'{PREFIX}_control_config.json')
    h = deepcopy(read(PRO/f'{PREFIX}_experimental_config.json'))
    h.update(arm='H4-R', schema=SCHEMA, experiment_id='F_HIGHER_SCALE_V2_RECOVERY_H4_R',
             output_path=str(OUT.relative_to(ROOT)))
    return {'C4': c, 'H4': h}


def validate_config(cfg):
    original = read(PRO/f'{PREFIX}_experimental_config.json')
    actual = deepcopy(cfg)
    for key in ('arm', 'schema', 'experiment_id', 'output_path'):
        actual[key] = original[key]
    require(actual == original, 'FAIL_CONFIG_PARITY')
    require(cfg['arm'] == 'H4-R' and cfg['training_args']['resume'] is False, 'FAIL_FRESH_IDENTITY')
    require((ROOT/cfg['output_path']).resolve() == OUT.resolve(), 'FAIL_OUTPUT_ROUTE')
    require(cfg['initialization']['sha256'] == '1ec926026473beafafb1ef8da6aa6efcc13d38c391ec8db48006c5d793d5f2a3', 'FAIL_FRESH_INITIALIZATION')
    return True


def historical_map():
    receipt = read(F11/'integrity.json')
    require(receipt['final_status']['PHASE_F11_STATUS'] == 'COMPLETE', 'FAIL_F11_NOT_COMPLETE')
    require(receipt['final_status']['ORIGINAL_F1_STATUS'] == 'INVALID_OR_INTERRUPTED', 'FAIL_HISTORY_STATUS')
    mapping = {row['path']: row['after_SHA256'] for row in receipt['comparisons']}
    mapping.update(receipt['audit_code_sha256'])
    mapping.update(receipt['delivered_artifact_sha256'])
    mapping[str(ROOT/'docs/PHASE_F11_INTERRUPTION_CHECKPOINT_AUDIT_20260928.md')] = receipt['report_sha256']
    mapping[str(F11/'integrity.json')] = sha(F11/'integrity.json')
    return mapping


def check_hashes(mapping):
    for path, digest in mapping.items():
        require(Path(path).is_file() and sha(path) == digest, 'FAIL_PROTECTED_HASH: '+path)


def verify_history():
    mapping = read(PRE/'historical_freeze.json')['sha256'] if (PRE/'historical_freeze.json').exists() else historical_map()
    check_hashes(mapping)
    old_snapshot = read(F11/'source_snapshot.json')
    for tree in (C4, OLD_H4, R/'f_higher_scale_v2_seed42_pair_summary', R/'f_higher_scale_v2_seed42_preflight'):
        expected = {r['path'] for r in old_snapshot['source_files'] if Path(r['path']).is_relative_to(tree)}
        require({str(p) for p in tree.rglob('*') if p.is_file()} == expected, 'FAIL_ORIGINAL_TREE_INVENTORY')
    return True


def no_original_evaluation():
    for tree in (C4, OLD_H4):
        for name in ('evaluation.lock', 'final_evaluation.json', 'per_image_predictions.json'):
            require(not (tree/name).exists(), 'FAIL_ORIGINAL_OPERATIONAL_EVALUATION_EXISTS')
    return True


def runtime_environment():
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    import torch, numpy, cv2, ultralytics, albumentations
    import importlib.metadata as metadata
    require(torch.cuda.is_available(), 'FAIL_GPU_UNAVAILABLE')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv,noheader'],
        capture_output=True, encoding='utf-8', check=True).stdout.strip()
    sources = {}
    cfg = configs()['H4']
    for relative, expected in cfg['architecture']['source_hashes'].items():
        path = Path(torch.__file__).parent/relative.removeprefix('torch/') if relative.startswith('torch/') else Path(ultralytics.__file__).parent/relative
        require(sha(path) == expected, 'FAIL_ACTIVE_PACKAGE_SEMANTIC_DRIFT: '+str(path))
        sources[str(path)] = sha(path)
    # Supplement version evidence with current import origins; no invented historical fields.
    packages = {}
    for name, module in [('torch',torch),('numpy',numpy),('cv2',cv2),('ultralytics',ultralytics),('albumentations',albumentations)]:
        packages[name] = dict(path=module.__file__, sha256=sha(module.__file__))
    return dict(Python=platform.python_version(), OS=platform.platform(), torch=torch.__version__,
        CUDA=torch.version.cuda, GPU=torch.cuda.get_device_name(0), GPU_VRAM=torch.cuda.get_device_properties(0).total_memory,
        gpu_driver_identity=gpu, Ultralytics=ultralytics.__version__, NumPy_actual=numpy.__version__,
        NumPy_import_path=numpy.__file__, NumPy_distribution_metadata=metadata.version('numpy'),
        OpenCV=cv2.__version__, albumentations=albumentations.__version__, active_source_sha256=sources, package_imports=packages)


def environment_parity(current):
    old = {arm:read(tree/'runtime_environment.json') for arm,tree in [('C4',C4),('original_H4',OLD_H4)]}
    mismatches = {arm:{k:dict(original=v,current=current.get(k)) for k,v in value.items() if current.get(k) != v} for arm,value in old.items()}
    require(not any(mismatches.values()), 'BLOCKED_BY_ENVIRONMENT_DRIFT: '+str(mismatches))
    return dict(ENVIRONMENT_PARITY='COMPATIBLE_WITH_DISCLOSED_DIFFERENCES', original=old, H4_R_pretraining=current,
        shared_recorded_fields_match=True, active_training_source_hashes_match=True,
        unrecorded_in_original=['OS', 'driver', 'GPU UUID', 'GPU VRAM in F1 runtime receipt', 'OpenCV'],
        disclosure='Original F1 receipts omit several platform fields. Current values recorded, historical equality UNKNOWN, not EXACT. Recorded critical versions and pinned training implementation hashes agree; no known semantic drift.')


def verify_execution():
    frozen = read(FREEZE)
    require(frozen['request_sha256'] == sha(REQUEST), 'FAIL_AUTHORIZATION_CHANGED')
    check_hashes(frozen['sources'])
    check_hashes(frozen['receipts'])
    validate_config(configs()['H4'])
    verify_history()
    no_original_evaluation()
    return frozen


def bind_runtime():
    """Reuse byte-identical functions, change only artifact/authority routing in this child."""
    import experiments.phase_f1_v2 as old
    global _ORIGINAL_AUDIT
    _ORIGINAL_AUDIT = old.audit_arm
    old.PAIR, old.PRE, old.EXEC_FREEZE = PRE, PRE, FREEZE
    old.OUTPUTS = {'C4':C4, 'H4':OUT}
    old.configs, old.verify_execution, old.verify_frozen, old.status = configs, verify_execution, verify_history, status
    def routed_save(path, value):
        if Path(path) == OUT/'execution.lock':
            value = dict(value, arm='H4-R', schema=SCHEMA, recovery_option='B',
                original_H4_role='INTERRUPTED_ENGINEERING_EVIDENCE',
                recovery_config_sha256=sha(PRE/'H4_R_config.json'), recovery_protocol_sha256=sha(PRE/'protocol.json'))
        return save(path,value)
    old.save = routed_save
    return old


def frozen_write_guard():
    roots = (C4, OLD_H4, F11, R/'f_higher_scale_v2_seed42_pair_summary', R/'f_higher_scale_v2_seed42_preflight')
    frozen_files = {Path(p).resolve() for p in read(PRE/'historical_freeze.json')['sha256']}
    frozen_files.update((PAIR/name/'best.pt').resolve() for name in ('eval_C4','eval_H4_R'))
    def blocked(value):
        if not isinstance(value, (str, bytes, Path)): return False
        p = Path(os.fsdecode(value)).resolve()
        return p in frozen_files or any(p.is_relative_to(r) for r in roots)
    def hook(event, args):
        if event == 'open':
            mode, flags = args[1] or '', args[2] or 0
            if (any(c in str(mode) for c in 'wax+') or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)) and blocked(args[0]):
                raise PermissionError('F12_FROZEN_WRITE_FORBIDDEN')
        elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.truncate'} and blocked(args[0]):
            raise PermissionError('F12_FROZEN_WRITE_FORBIDDEN')
        elif event in {'os.rename','os.replace'} and any(blocked(x) for x in args[:2]):
            raise PermissionError('F12_FROZEN_WRITE_FORBIDDEN')
    sys.addaudithook(hook)


def preflight():
    require(not OUT.exists() and not PAIR.exists() and not PRE.exists() and not REPORT.exists(), 'FAIL_OUTPUT_ALREADY_EXISTS')
    verify_history(); no_original_evaluation(); validate_config(configs()['H4'])
    from experiments import phase_f1_v2 as old
    old.verify_frozen(); old.verify_execution()
    c_audit = old.audit_arm('C4')
    from experiments.phase_d1 import verify_data, verify_validation
    samples = verify_data(ROOT, configs()['H4']); verify_validation(ROOT)
    require(len(samples) == 771 and sum(r['GT_count'] for r in samples) == 965, 'FAIL_DATASET_SUPPORT')
    require(sha(ROOT/configs()['H4']['sampling']['order_path']) == configs()['H4']['sampling']['order_sha256'], 'FAIL_ANCHOR_HASH')
    environment = runtime_environment()
    parity = environment_parity(environment)
    PRE.mkdir()
    save(PRE/'historical_freeze.json',dict(at=now(),sha256=historical_map(),original_status='INVALID_OR_INTERRUPTED'))
    save(PRE/'C4_RECOVERY_COMPARATOR_FREEZE.json',dict(at=now(),schema='C4_RECOVERY_COMPARATOR_FREEZE',
        C4_SOURCE='ORIGINAL_COMPLETED_F1_CONTROL', selected_checkpoint='best.pt', reselection_allowed=False,
        completion=read(C4/'training_completion.json'), runtime_integrity=c_audit,
        sha256={str(p):sha(p) for p in C4.rglob('*') if p.is_file()},
        training_config_sha256=sha(PRO/f'{PREFIX}_control_config.json'),
        dataset_manifest_sha256=configs()['C4']['manifest']['sha256'], initialization_sha256=configs()['C4']['initialization']['sha256']))
    save(PRE/'environment_parity.json',parity)
    save(PRE/'H4_R_config.json',configs()['H4'])
    protocol = dict(schema=SCHEMA, RECOVERY_OPTION='B', one_execution=True, no_resume=True, no_retry=True,
        authorization_sha256=sha(REQUEST), ORIGINAL_F1_STATUS='INVALID_OR_INTERRUPTED',
        C4_SOURCE='ORIGINAL_COMPLETED_F1_CONTROL', H4_SOURCE='FRESH_REPLACEMENT_EXECUTION',
        original_H4_role='INTERRUPTED_ENGINEERING_EVIDENCE', original_H4_performance_used=False,
        initialization=configs()['H4']['initialization'], training_args=configs()['H4']['training_args'],
        numerical_safety_contract_sha256=sha(PRO/f'{PREFIX}_numerical_safety_contract.json'),
        validation_override_sha256=sha(ROOT/'experiments/f01_validator.py'),
        evaluation_protocol_sha256=sha(PRO/f'{PREFIX}_evaluation_protocol.json'),
        advancement_gate=read(PRO/f'{PREFIX}_protocol.json')['advancement_gate'],
        routing='Frozen train_arm and trainer reused with process-local artifact/authority mapping; no source edits. C4 cannot be selected by CLI. Evaluation uses separate new shadow dirs.',
        evaluation_order=['valid_pair','C4@768','H4-R@768','paired_gate'],
        C4_evaluation_output=str(PAIR/'eval_C4'), H4_R_evaluation_output=str(PAIR/'eval_H4_R'),
        restrictions=dict(locked_test=False,CO2Wounds=False,external_test=False,multi_seed=False,app_replacement=False,H4_R_eval1024=False),
        interpretation='single-seed recovery paired development experiment; asymmetric replacement; no p-values, significance, population CI or clinical generalization', STOP_AFTER_F12=True)
    save(PRE/'protocol.json',protocol)
    test_files = ['tests/test_phase_f12.py',*old.TEST_FILES]
    run = subprocess.run([sys.executable,'-B','-m','pytest',*test_files,'-q','-p','no:cacheprovider','-k','not future_outputs_absent'],
        cwd=ROOT,env=env_vars(),capture_output=True,encoding='utf-8')
    save(PRE/'tests.json',dict(exit_code=run.returncode,stdout=run.stdout,stderr=run.stderr,
        scope='CPU synthetic adapter/validator binding and frozen guards; no research training. Historical future-output-absence tests excluded; recovery output absence separately asserted.'))
    require(run.returncode == 0, 'FAIL_PREFLIGHT_TESTS')
    verify_history(); no_original_evaluation()
    require(not OUT.exists() and not PAIR.exists(), 'FAIL_NEW_OUTPUT_NOT_ABSENT')
    save(PRE/'preflight.json',dict(READY_TO_EXECUTE_H4_REPLACEMENT='YES',at=now(),data_images=771,polygon_instances=965,
        validation_images=191, dataset_hashes='PASS', initialization_hash='PASS', config_parity='PASS',
        C4_frozen_comparator_integrity='PASS', F11_unchanged=True, original_H4_untouched=True,
        H4_R_output_absent=True, environment_parity=parity['ENVIRONMENT_PARITY'],
        safety_and_SharedValidation768_binding='same frozen factory/source plus passing CPU behavioral tests; real binding rechecked before first training batch',
        final_evaluation_performed=False,test_images_used=0))
    save(FREEZE,dict(at=now(),request_sha256=sha(REQUEST),
        sources={str(ROOT/p):sha(ROOT/p) for p in SOURCES},
        receipts={str(p):sha(p) for p in PRE.glob('*.json')},
        environment=environment, one_execution=True, new_retry_authorized=False))
    status('READY', READY_TO_EXECUTE_H4_REPLACEMENT='YES')
    from experiments.f12_results import write_report
    write_report('READY', {'preflight':read(PRE/'preflight.json')})
    print('READY_TO_EXECUTE_H4_REPLACEMENT=YES')


def train(token):
    old = bind_runtime()
    verify_execution()
    require(read(PRE/'execution.lock')['token'] == token, 'FAIL_TOKEN')
    require(not OUT.exists(), 'FAIL_ALREADY_STARTED_NO_RETRY')
    current = runtime_environment()
    require(current == read(FREEZE)['environment'], 'BLOCKED_BY_ENVIRONMENT_DRIFT')
    frozen_write_guard()
    # old train_arm checks token/freeze, fresh optimizer/scaler, original recipe, ledger and epoch budget.
    old.train_arm('H4', token)


def checkpoint_check(path):
    require(path.is_file() and path.stat().st_size > 0, 'FAIL_NONZERO_CHECKPOINT')
    with zipfile.ZipFile(path) as archive: require(archive.testzip() is None, 'FAIL_CHECKPOINT_CRC')
    run = subprocess.run([sys.executable,'-B',str(ROOT/'experiments/f11_checkpoint_reader.py'),str(path)],
        cwd=ROOT,env=env_vars(),capture_output=True,encoding='utf-8',timeout=120)
    require(run.returncode == 0, 'FAIL_CHECKPOINT_READABILITY: '+run.stderr)
    result = json.loads(run.stdout.strip().splitlines()[-1])
    require(result['integrity'] == 'VALID' and result['all_stored_tensors_finite'], 'FAIL_CHECKPOINT_STATE')
    result.update(path=str(path),SHA256=sha(path),size=path.stat().st_size)
    return result


def completion_gate():
    old = bind_runtime(); verify_execution()
    audit = old.audit_arm('H4')
    from experiments.phase_f11 import csv_epochs, tensorboard
    csv = csv_epochs(OUT/'results.csv')
    require(csv['epoch_numbers'] == list(range(1,301)) and not csv['invalid_lines'], 'FAIL_RESULTS_COMPLETION')
    events = [tensorboard(p) for p in OUT.rglob('events.out.tfevents.*')]
    require(events and all(not v['invalid_records'] for v in events) and max(max(v['distinct_steps']) for v in events) == 300, 'FAIL_TENSORBOARD_TAIL')
    checks = {name:checkpoint_check(OUT/f'{name}.pt') for name in ('best','last')}
    save(OUT/'checkpoint_integrity.json',dict(status='PASS',checkpoints=checks,results=csv,tensorboard=events))
    cv = list(old.rows_jsonl(C4/'validation_runtime.jsonl'))[0]
    hv = audit['validation']
    for key in ('val_nominal_imgsz','dataset_nominal_imgsz','first_batch_shape','shared_override'):
        require(cv[key] == hv[key], 'FAIL_SHARED_VALIDATOR_PARITY')
    for key in ('imgsz','conf','iou','rect','half','max_det','agnostic_nms','augment'):
        require(cv['validator_args'][key] == hv['validator_args'][key], 'FAIL_SHARED_VALIDATOR_ARGS')
    from experiments.f02_safety import compare_completed
    validity = compare_completed(read(C4/'training_completion.json'),read(OUT/'training_completion.json'))
    require(validity['PAIRED_FIXED_BUDGET_VALID'] == 'YES','FAIL_RECOVERY_BUDGET')
    return dict(RECOVERY_PAIR_VALID='YES',C4_frozen_integrity='PASS',H4_REPLACEMENT_COMPLETION_INTEGRITY='PASS',
                H4_REPLACEMENT_FIXED_BUDGET_VALID='YES', scope=SCHEMA, validity=validity, runtime_audit=audit,
                original_F1_status='INVALID_OR_INTERRUPTED',at=now())


def authorize_eval(arm, token):
    require(arm in ('C4','H4'), 'FAIL_EVAL_ARM')
    require(read(PRE/'execution.lock')['token'] == token, 'FAIL_TOKEN')
    require(read(PAIR/'recovery_pair_validity.json')['RECOVERY_PAIR_VALID'] == 'YES','FAIL_PAIR_BEFORE_EVAL')
    target = PAIR/('eval_C4' if arm == 'C4' else 'eval_H4_R')
    require(not target.exists(), 'FAIL_EVAL_ALREADY_STARTED')
    if arm == 'H4': require((PAIR/'eval_C4/final_evaluation.json').exists(),'FAIL_EVAL_ORDER')
    return target


def evaluate(arm, token):
    old = bind_runtime(); verify_execution()
    target = authorize_eval(arm,token)
    old.audit_arm('C4'); old.audit_arm('H4')
    source = C4 if arm == 'C4' else OUT
    target.mkdir()
    # New read-only-input shadow routes prevent any additions to frozen C4.
    os.link(source/'best.pt',target/'best.pt')
    save(target/'training_completion.json',read(source/'training_completion.json'))
    import experiments.f1_v2_results as results
    results.PAIR, results.EXEC_FREEZE = PAIR, FREEZE
    results.OUTPUTS = {'C4':PAIR/'eval_C4','H4':PAIR/'eval_H4_R'}
    results.verify_execution, results.verify_frozen, results.audit_arm = verify_execution, verify_history, old.audit_arm
    frozen_write_guard()
    results.evaluate(arm,token)


def execute():
    verify_execution()
    require(not OUT.exists() and not PAIR.exists() and not (PRE/'execution.lock').exists(),'FAIL_ALREADY_STARTED_NO_RETRY')
    require(runtime_environment() == read(FREEZE)['environment'],'BLOCKED_BY_ENVIRONMENT_DRIFT')
    token = str(uuid.uuid4())
    base = read(C4/'runtime_environment.json')
    save(PRE/'execution.lock',dict(at=now(),pid=os.getpid(),token=token,environment=base,
        execution_freeze_sha256=sha(FREEZE),request_sha256=sha(REQUEST),identity='H4-R',no_retry=True))
    try:
        status('STARTING_H4_R')
        with (PRE/'train_H4_R.log').open('x',encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable,'-u','-B','-m','experiments.phase_f12','--train','--token',token],
                cwd=ROOT,env=env_vars(),stdout=log,stderr=subprocess.STDOUT)
            save(PRE/'H4_R_process.json',dict(pid=proc.pid,at=now(),command='python -u -B -m experiments.phase_f12 --train [token redacted]'))
            code = proc.wait()
        save(PRE/'training_process_exit.json',dict(at=now(),exit_code=code))
        require(code == 0, 'H4_R_FAILED_EXIT_'+str(code)+'; NO RETRY')
        status('COMPLETION_INTEGRITY')
        validity = completion_gate()
        require(not PAIR.exists(), 'FAIL_PAIR_OUTPUT_EXISTS')
        PAIR.mkdir()
        save(PAIR/'recovery_pair_validity.json',validity)
        save(PAIR/'execution.lock',read(PRE/'execution.lock'))
        # Compatibility receipt consumed ONLY by unchanged evaluator, explicitly recovery-scoped.
        save(PAIR/'pair_validity.json',dict(PAIRED_FIXED_BUDGET_VALID='YES',scope=SCHEMA,original_pair_valid=False,
            recovery_validity_sha256=sha(PAIR/'recovery_pair_validity.json')))
        for arm in ('C4','H4'):
            status('EVALUATING_'+('C4' if arm == 'C4' else 'H4_R'))
            with (PRE/f'eval_{arm}.log').open('x',encoding='utf-8') as log:
                proc = subprocess.run([sys.executable,'-u','-B','-m','experiments.phase_f12','--eval',arm,'--token',token],
                    cwd=ROOT,env=env_vars(),stdout=log,stderr=subprocess.STDOUT)
            require(proc.returncode == 0,'FAIL_EVAL_'+arm+'_NO_RETRY')
        verify_execution()
        from experiments.f12_results import finish
        finish()
    except BaseException as exc:
        from experiments.f12_results import interrupted
        interrupted(exc)
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight',action='store_true'); group.add_argument('--execute-authorized',action='store_true')
    group.add_argument('--train',action='store_true'); group.add_argument('--eval',choices=['C4','H4'])
    p.add_argument('--token'); a=p.parse_args()
    if a.preflight: preflight()
    elif a.execute_authorized: execute()
    elif a.train: train(a.token)
    else: evaluate(a.eval,a.token)
