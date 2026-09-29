"""Frozen paired synthetic AMP preflight; no formal training authorization."""
from __future__ import annotations
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from experiments.phase_f0 import ROOT,PRO,read,save,sha,require,future_absent,INIT_SHA

PREFIX='F_HIGHER_SCALE_V1_NUMERICAL_PREFLIGHT'
OUT=ROOT/'experiments/results/f_higher_scale_v1_numerical_preflight'
F0OUT=ROOT/'experiments/results/f_higher_scale_v1_resource_audit'


def classify(rows):
    require(len(rows)<=8,'FAIL_MAX_ATTEMPTS')
    valid=lambda r:(r['loss_finite'] and r['gradient_tensor_count']>0 and r['nonfinite_gradient_tensor_count']==0
                    and r['scaler_after']>=r['scaler_before']>=1 and r['parameters_unchanged'])
    first=next((r for r in rows if valid(r)),None);window=None
    for i in range(len(rows)-2):
        if all(valid(r) for r in rows[i:i+3]):window=rows[i:i+3];break
    classification='PERSISTENT_NUMERICAL_INSTABILITY'
    if window:classification='STABLE_FROM_INITIAL_SCALE' if rows and valid(rows[0]) else 'SCALER_MANAGED_TRANSIENT_OVERFLOW'
    return {'classification':classification,'NUMERICALLY_STABLE':bool(window),
            'FIRST_FINITE_GRADIENT_ATTEMPT':first['attempt'] if first else 'NONE',
            'FIRST_FINITE_GRADIENT_SCALE':first['scaler_before'] if first else None,
            'stable_window':[r['attempt'] for r in window] if window else None,
            'stable_scale':min(r['scaler_before'] for r in window) if window else None}


def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+b'\0'+a.tobytes()).hexdigest()


def fixture():
    # Execute ONLY the pure generator AST from the immutable actual F0 run source.
    src=F0OUT/'executed_resource_smoke.py'
    require(sha(src)==read(F0OUT/'execution.lock')['code_sha256'],'FAIL_F0_EXECUTION_SOURCE_HASH')
    tree=ast.parse(src.read_text(encoding='utf8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='synthetic_source')
    ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),ns)
    return ns['synthetic_source']()


def fixture_hashes():
    images,polygons=fixture()
    return {'images_sha256':array_sha(images),'polygons_sha256':array_sha(polygons),
            'per_image_sha256':[array_sha(x) for x in images],
            'image_shape':list(images.shape),'polygon_shape':list(polygons.shape),
            'class_ids':[0]*64,'generator_source_sha256':sha(F0OUT/'executed_resource_smoke.py'),
            'original_F0_fixture_hash_was_recorded':False,
            'identity_basis':'Reconstructed from immutable executed F0 generator AST; runtime NumPy2.2.6, installed distribution metadata2.0.1 is stale. Original in-memory fixture was not retained; no historical array checksum comparison is claimed. No new generator design.',
            'numpy_version':np.__version__}


def prepare():
    import importlib.metadata
    require(not OUT.exists() and not list(PRO.glob(PREFIX+'*')),'FAIL_PREFLIGHT_EXISTS')
    future_absent()
    f0=read(PRO/'F_HIGHER_SCALE_V1_freeze.json')
    require(f0['PHASE_F0_STATUS']=='BLOCKED','FAIL_F0_STATUS')
    protected={str(ROOT/p):h for p,h in f0['artifacts_sha256'].items()}
    protected[str(PRO/'F_HIGHER_SCALE_V1_freeze.json')]=sha(PRO/'F_HIGHER_SCALE_V1_freeze.json')
    for p,h in protected.items():require(sha(p)==h,'FAIL_F0_CHANGED: '+p)
    require(importlib.metadata.version('numpy')=='2.0.1','FAIL_FIXTURE_NUMPY_VERSION')
    hashes=fixture_hashes();require(hashes==fixture_hashes(),'FAIL_SYNTHETIC_FIXTURE_IDENTITY')
    c=read(PRO/'F_HIGHER_SCALE_V1_control_config.json')
    require(sha(ROOT/c['initialization']['path'])==INIT_SHA,'FAIL_INIT')
    protocol={'schema':'F_HIGHER_INPUT_SCALE_V1_NUMERICAL_PREFLIGHT','created_at':datetime.now(timezone.utc).isoformat(),
        'arms':['N768','N1024'],'imgsz':[768,1024],'batch':4,'MAX_ATTEMPTS':8,'initial_scale':65536,
        'run_all_8_even_after_stability':True,'initial_scale_policy':'stock default verified65536, no manual override',
        'execution':'N768 fresh subprocess -> exit -> N1024 fresh subprocess -> exit; one execution per arm',
        'fixture_sha256':hashes,'initialization':c['initialization'],
        'recipe_source_sha256':sha(PRO/'F_HIGHER_SCALE_V1_control_config.json'),
        'same_recipe':['architecture','initialization','optimizer','LR','loss','batch','AMP','synthetic fixture'],
        'optimizer':'fresh AdamW(.0005,betas(.937,.999)); same decay groups; step forbidden',
        'attempt':'zero_grad(set_to_none=True), forward/loss, finite guard, scale/backward/unscale, every-gradient telemetry, scaler.update, parameter SHA equality',
        'stability':'3 consecutive finite-loss/all-finite-gradient attempts with no backoff and scale>=1 within8',
        'classifications':['STABLE_FROM_INITIAL_SCALE','SCALER_MANAGED_TRANSIENT_OVERFLOW','PERSISTENT_NUMERICAL_INSTABILITY'],
        'guard_policy':'behavioral CPU sentinels; per-GPU-arm counters all0; no save, optimizer.step, scaler.step or image decode',
        'F0_role':'IMMUTABLE_FIRST_ATTEMPT_NONFINITE_GRADIENT_OBSERVATION; not retrospectively PASS',
        'BN_buffers':'allowed to evolve only inside disposable model; trainable parameters must remain identical',
        'OOM_or_nonfinite_loss':'record incomplete arm and block; never lower batch, scale, or retry',
        'real_data_pixels_allowed':False,'research_validation_inference':False,'research_training':False,
        'F1_authorized':False,'STOP_AFTER_F01':True}
    save(PRO/f'{PREFIX}_protocol.json',protocol);save(PRO/f'{PREFIX}_fixture.json',hashes)
    package=Path('C:/Python312/Lib/site-packages/ultralytics')
    contract={'shared_override':'experiments.f01_validator.SharedValidation768','both_arms_same_class':True,
        'control':{'train_imgsz':768,'val_imgsz':768},'experimental':{'train_imgsz':1024,'val_imgsz':768},
        'dataset':'stock build_dataset called on shallow proxy with args copy; no label or augmentation rewrite',
        'validator':'stock SegmentationValidator constructed with768 args and same768 dataset; real validator.__call__ forbidden in F01',
        'thresholds':'preserve same stock training-checkpoint-validation defaults in both arms; confNone resolves.001, NMS.70, AP matching .50:.95',
        'final_evaluation':'separate frozen operational protocol unchanged: conf.10 floor.01 NMS.70 bbox match.50 imgsz768',
        'rect_padding':'stock val rect/pad preserved; report actual synthetic tensor shape, not just nominal img size',
        'sources':{str(p.relative_to(package)):sha(p) for p in [package/'models/yolo/detect/train.py',package/'models/yolo/segment/train.py',package/'engine/validator.py',package/'data/build.py']},
        'audit_scope':'synthetic CPU dataset and validator preprocessing only; no research image load, model construction or inference'}
    save(PRO/f'{PREFIX}_validator_contract.json',contract)
    save(OUT/'protected_f0_snapshot.json',protected)
    paths=[*PRO.glob(PREFIX+'*.json'),ROOT/'experiments/phase_f01.py',ROOT/'experiments/f01_numerical.py',ROOT/'experiments/f01_guards.py',ROOT/'experiments/f01_validator.py']
    save(OUT/'pre_execution_freeze.json',{'artifacts_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},
        'f0_freeze_sha256':sha(PRO/'F_HIGHER_SCALE_V1_freeze.json'),'timestamp':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'pre_execution_freeze':'PASS','fixture':hashes,'protected_F0_files':len(protected)}))


def execute():
    # Sequential blocking subprocess.run guarantees allocator/model exit before next arm.
    save(OUT/'paired_execution.lock',{'pid':__import__('os').getpid(),'order':['N768','N1024'],'retry':False})
    for arm in ['N768','N1024']:
        start=datetime.now(timezone.utc).isoformat()
        p=subprocess.run([sys.executable,'-m','experiments.f01_numerical','--arm',arm],capture_output=True,text=True,encoding='utf8')
        save(OUT/f'{arm}_process.json',{'command':[sys.executable,'-m','experiments.f01_numerical','--arm',arm],
            'started_at':start,'exited_at':datetime.now(timezone.utc).isoformat(),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
        require(p.returncode==0 and (OUT/arm/'summary.json').exists(),'FAIL_ARM_PROCESS: '+arm)
        s=read(OUT/arm/'summary.json')
        print(json.dumps({'arm':arm,'classification':s['classification'],'attempts':s['completed_attempts'],
                          'first_finite':s.get('FIRST_FINITE_GRADIENT_ATTEMPT'),'peak_allocated':s.get('peak_allocated_bytes')}),flush=True)
