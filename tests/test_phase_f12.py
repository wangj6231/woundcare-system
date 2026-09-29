"""Recovery preregistration/boundary tests. No research training or inference."""
from copy import deepcopy
import ast
import subprocess
import sys
import pytest
from experiments import phase_f12 as f


def test_config_changes_are_identity_only():
    assert f.validate_config(f.configs()['H4'])
    h=f.configs()['H4']
    assert h['arm']=='H4-R' and h['training_args']['imgsz']==1024
    assert h['training_args']['epochs']==300 and h['training_args']['patience']==80
    assert h['initialization']['resume'] is False


@pytest.mark.parametrize('key,value', [('imgsz',960),('epochs',299),('patience',81),('batch',2),('resume',True),
    ('seed',123),('lr0',.001),('workers',0),('amp',False),('mosaic',0),('box',8),('mask_ratio',2),('overlap_mask',False)])
def test_config_drift_rejected(key,value):
    cfg=deepcopy(f.configs()['H4']); cfg['training_args'][key]=value
    with pytest.raises(ValueError,match='FAIL_CONFIG_PARITY'): f.validate_config(cfg)


def test_interrupted_checkpoint_cannot_initialize():
    cfg=deepcopy(f.configs()['H4']); cfg['initialization']['path']=str(f.OLD_H4/'training/weights/best.pt')
    with pytest.raises(ValueError,match='FAIL_CONFIG_PARITY'): f.validate_config(cfg)


def test_no_C4_training_entry():
    source=(f.ROOT/'experiments/phase_f12.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='train_arm']
    assert len(calls)==1 and calls[0].args[0].value=='H4'
    assert "group.add_argument('--train',action='store_true')" in source
    assert "group.add_argument('--arm'" not in source


def test_no_original_outcome_reader_in_reporting():
    source=(f.ROOT/'experiments/f12_results.py').read_text(encoding='utf-8')
    assert 'OLD_H4' not in source
    assert 'INTERRUPTED_ENGINEERING_EVIDENCE' in source
    assert 'clinical generalization' in source and 'asymmetric replacement' in source


@pytest.fixture
def routing(tmp_path,monkeypatch):
    for name in ['PRE','PAIR','OUT','C4','OLD_H4']:
        monkeypatch.setattr(f,name,tmp_path/name)
    f.PRE.mkdir(); f.PAIR.mkdir()
    f.save(f.PRE/'execution.lock',{'token':'fixture'})
    return tmp_path


def test_eval_before_pair_valid_blocked(routing):
    f.save(f.PAIR/'recovery_pair_validity.json',{'RECOVERY_PAIR_VALID':'NO'})
    with pytest.raises(ValueError,match='FAIL_PAIR_BEFORE_EVAL'): f.authorize_eval('C4','fixture')
    assert not (f.PAIR/'eval_C4').exists()


def test_fixed_eval_order(routing):
    f.save(f.PAIR/'recovery_pair_validity.json',{'RECOVERY_PAIR_VALID':'YES'})
    with pytest.raises(ValueError,match='FAIL_EVAL_ORDER'): f.authorize_eval('H4','fixture')
    assert f.authorize_eval('C4','fixture')==f.PAIR/'eval_C4'
    (f.PAIR/'eval_C4').mkdir()
    f.save(f.PAIR/'eval_C4/final_evaluation.json',{'fixture':True})
    assert f.authorize_eval('H4','fixture')==f.PAIR/'eval_H4_R'


def test_one_shot_eval_not_reusable(routing):
    f.save(f.PAIR/'recovery_pair_validity.json',{'RECOVERY_PAIR_VALID':'YES'})
    (f.PAIR/'eval_C4').mkdir()
    with pytest.raises(ValueError,match='FAIL_EVAL_ALREADY_STARTED'): f.authorize_eval('C4','fixture')


def test_bad_token_and_arm_fail_closed(routing):
    with pytest.raises(ValueError,match='FAIL_TOKEN'): f.authorize_eval('C4','bad')
    with pytest.raises(ValueError,match='FAIL_EVAL_ARM'): f.authorize_eval('H4-R2','fixture')


def test_duplicate_execution_blocked_before_runtime(routing,monkeypatch):
    monkeypatch.setattr(f,'verify_execution',lambda:None)
    monkeypatch.setattr(f,'runtime_environment',lambda:pytest.fail('runtime must not be reached'))
    with pytest.raises(ValueError,match='FAIL_ALREADY_STARTED_NO_RETRY'): f.execute()


def test_environment_critical_drift_blocked(routing):
    for tree in [f.C4,f.OLD_H4]:
        tree.mkdir(); f.save(tree/'runtime_environment.json',{'torch':'frozen','CUDA':'12.4'})
    with pytest.raises(ValueError,match='BLOCKED_BY_ENVIRONMENT_DRIFT'): f.environment_parity({'torch':'changed','CUDA':'12.4'})
    good=f.environment_parity({'torch':'frozen','CUDA':'12.4'})
    assert good['ENVIRONMENT_PARITY']=='COMPATIBLE_WITH_DISCLOSED_DIFFERENCES'
    assert good['unrecorded_in_original']


def test_frozen_config_mapping_in_isolated_process():
    code="from experiments.phase_f12 import *; old=bind_runtime(); assert old.OUTPUTS['C4']==C4; assert old.OUTPUTS['H4']==OUT; assert old.PAIR==PRE; assert old.configs()['H4']['arm']=='H4-R'; assert OLD_H4 not in old.OUTPUTS.values(); print('ROUTING_PASS')"
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=f.ROOT,env=f.env_vars(),capture_output=True,encoding='utf-8')
    assert result.returncode==0,result.stderr
    assert 'ROUTING_PASS' in result.stdout


def test_empty_checkpoint_never_loaded(tmp_path,monkeypatch):
    file=tmp_path/'last.pt'; file.write_bytes(b'')
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('No child for zero-byte checkpoint'))
    with pytest.raises(ValueError,match='FAIL_NONZERO_CHECKPOINT'): f.checkpoint_check(file)


def test_final_gate_is_frozen_implementation():
    import experiments.f1_v2_results as old
    # Same function, same exact Fraction thresholds. Existing original behavioral tests cover boundaries.
    source=(f.ROOT/'experiments/f12_results.py').read_text(encoding='utf-8')
    assert "fr.advancement(e['C4'],e['H4-R'])" in source
    assert old.advancement.__module__=='experiments.f1_v2_results'


def test_no_new_training_threshold_or_checkpoint_policy():
    source=(f.ROOT/'experiments/phase_f12.py').read_text(encoding='utf-8')
    assert 'old.train_arm(\'H4\', token)' in source
    assert 'make_trainer(' not in source and '.backward(' not in source
    assert 'results.evaluate(arm,token)' in source
    assert 'NO_RETRY' in source


def test_shared_validator_hash_contract():
    contract=f.read(f.PRO/'F_HIGHER_SCALE_V2_validation_override_contract.json')
    assert f.sha(f.ROOT/contract['implementation']['path'])==contract['implementation']['sha256']
    assert contract['val_nominal_imgsz']==768


def test_saved_C4_receipt_still_300():
    c=f.read(f.C4/'training_completion.json')
    assert c['completed_epochs']==300 and c['scheduled']==3741 and c['unknown']==0
    assert c['best_checkpoint_sha256']==f.sha(f.C4/'best.pt')
