"""Forensic parser/boundary tests: no model execution or training imports."""
import ast
import json
import subprocess
import sys
from pathlib import Path
import pytest
from experiments import phase_f11 as f


@pytest.fixture(scope='module')
def evidence():
    return {key: f.read(f.OUT/(key+'.json')) for key in ['checkpoint_integrity', 'telemetry_integrity', 'anchor_integrity', 'source_snapshot', 'system_interruption_evidence']}


def assert_restriction(key):
    assert f.RESTRICTIONS[key] is False


def test_f11_no_training():
    assert_restriction('training')
    tree = ast.parse((f.ROOT/'experiments/phase_f11.py').read_text(encoding='utf-8'))
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not {'train', 'predict', 'forward', 'backward', 'val', 'fit', 'step', 'resume'} & set(calls)


def test_f11_no_resume(): assert_restriction('resume')
def test_f11_no_retry(): assert_restriction('retry')
def test_f11_no_inference(): assert_restriction('inference')
def test_f11_no_final_eval(evidence):
    assert_restriction('final_evaluation')
    assert evidence['telemetry_integrity']['final_evaluation_artifacts'] == []


def test_f11_c4_300_epochs(evidence):
    c = evidence['telemetry_integrity']['C4']
    assert c['completed_epoch_count'] == c['csv']['valid_row_count'] == 300
    assert c['full_epoch_scheduled_budget_match']


def test_f11_h4_completed_epoch_count(evidence):
    h = evidence['telemetry_integrity']['H4']
    assert h['completed_epoch_count'] == h['csv']['last_epoch_number'] == 297
    assert h['tensorboard'][0]['distinct_steps'] == list(range(1, 298))


def test_f11_h4_partial_epoch_preserved(evidence):
    h = evidence['telemetry_integrity']['H4']['partial']
    assert h['epoch_index'] == 297 and h['epoch_number'] == 298
    assert h['anchor_batches_observed'] == h['raw_loss_batches_observed'] == 193
    assert h['exact_completed_batch_count'] == 193
    assert h['actual_consumed_anchor_count'] == 771
    assert h['full_epoch_persistence_complete'] is False


def test_f11_zero_byte_last_detected(evidence):
    last = evidence['checkpoint_integrity']['H4_last']
    assert last['size'] == 0 and last['integrity'] == 'INVALID_ZERO_BYTE_FILE'


def test_f11_zero_byte_last_not_loaded(evidence):
    assert evidence['checkpoint_integrity']['H4_last']['deserialization_attempted'] is False
    for previous in evidence['checkpoint_integrity'].get('prior_audit_attempts', []):
        assert previous['H4_last']['deserialization_attempted'] is False


def test_f11_best_hash_recorded(evidence):
    digest = evidence['checkpoint_integrity']['H4_best']['SHA256']
    assert len(digest) == 64
    assert digest == f.sha(f.ARMS['H4']/'training/weights/best.pt')


def test_f11_best_integrity_checked(evidence):
    for key in ['C4_best', 'C4_last', 'H4_best']:
        c = evidence['checkpoint_integrity'][key]
        assert c['zip_crc_valid'] and c['all_stored_tensors_finite'] and c['all_tensors_cpu']
        assert c['integrity'] == 'VALID' and c['gpu_initialized'] is False
        assert not any(c[k] for k in ['forward', 'inference', 'optimizer_step', 'save'])


def test_f11_epoch_index_vs_count_explicit(evidence):
    h = evidence['telemetry_integrity']['H4']
    assert h['last_completed_epoch_index'] == 296
    assert h['last_fully_completed_epoch_number'] == 297
    assert evidence['checkpoint_integrity']['H4_best']['metadata']['epoch'] == 292


def test_f11_anchor_complete_epoch_match(evidence):
    for arm in ['C4', 'H4']:
        assert evidence['anchor_integrity'][arm]['EXPECTED_ORDER_MATCH'] == 'YES'
    assert f.compare_sequence(['a', 'b'], ['a', 'b'], True) == 'YES'
    assert f.compare_sequence(['b', 'a'], ['a', 'b'], True) == 'NO'
    assert f.compare_sequence([], ['a'], True) == 'UNKNOWN'


def test_f11_partial_epoch_prefix_match(evidence):
    assert evidence['anchor_integrity']['H4']['EXPECTED_PREFIX_MATCH'] == 'YES'
    assert f.compare_sequence(['a'], ['a', 'b'], False) == 'YES'
    assert f.compare_sequence(['b'], ['a', 'b'], False) == 'NO'
    assert f.compare_sequence([], ['a'], False) == 'UNKNOWN'


def test_f11_optimizer_telemetry_integrity(evidence):
    opt = evidence['telemetry_integrity']['H4']['optimizer']
    assert (opt['observed_scheduled'], opt['applied'], opt['skipped'], opt['unknown']) == (3717, 3707, 10, 0)
    assert opt['equation_valid'] and opt['monotonic_opportunities'] and opt['duplicate_opportunities'] == 0
    assert opt['frozen_contract_replay']['failure'] is None


def test_f11_numerical_hard_stop_audit(evidence):
    h = evidence['telemetry_integrity']['H4']
    assert h['recorded_numerical_stops'] == []
    assert h['numerical_hard_stop_verdict'] == 'NO_RECORDED_PREREGISTERED_NUMERICAL_HARD_STOP_BEFORE_INTERRUPTION'
    assert h['nonfinite_loss_records'] == h['nonfinite_parameter_records'] == []


def test_f11_no_false_interruption_cause(evidence):
    s = evidence['system_interruption_evidence']
    assert s['precise_hardware_cause'] == 'UNVERIFIED'
    assert s['training_process_termination_timestamp'] == 'UNKNOWN'
    assert '16:17:45' in s['chronology_conflict']


def test_f11_original_f1_remains_invalid():
    source = (f.ROOT/'experiments/phase_f11.py').read_text(encoding='utf-8')
    assert "ORIGINAL_F1_STATUS='INVALID_OR_INTERRUPTED'" in source


def test_f11_original_resume_forbidden():
    source = (f.ROOT/'experiments/phase_f11.py').read_text(encoding='utf-8')
    assert "ORIGINAL_F1_RESUME_AUTHORIZED='NO'" in source


def test_f11_no_locked_test():
    assert_restriction('locked_test_used')
    assert f.RESTRICTIONS['test_images_used'] == 0
def test_f11_no_co2(): assert_restriction('CO2Wounds_used')
def test_f11_no_external_test(): assert_restriction('external_test_used')


def test_truncated_event_is_not_a_skip(tmp_path):
    path = tmp_path/'partial.jsonl'
    path.write_bytes(b'{"event":"observed"}\n{"unfinished":\x00')
    rows, parsing = f.parse_jsonl(path)
    assert rows == [{'event':'observed'}]
    assert parsing['invalid_records'][0]['classification'] == 'TRUNCATED_EVENT'


def test_checkpoint_missing_resume_contract(evidence):
    c = evidence['checkpoint_integrity']['H4_best']
    assert c['states_present']['optimizer'] and c['states_present']['EMA']
    assert not any(c['states_present'][s] for s in ['scheduler', 'GradScaler', 'RNG', 'sampler', 'augmentation'])


@pytest.mark.parametrize('operation', [
    "open('forbidden.bin', 'wb')",
    "os.mkdir('forbidden_new_directory')",
    "__import__('subprocess').run([sys.executable, '-V'])",
    "__import__('socket').socket().connect(('127.0.0.1', 12345))",
])
def test_child_guard_rejects_side_effects(operation, tmp_path):
    code = f"import sys,os; sys.path.insert(0,{str(f.ROOT)!r}); from experiments.f11_checkpoint_reader import audit_hook; sys.addaudithook(audit_hook); {operation}"
    run = subprocess.run([sys.executable, '-B', '-c', code], cwd=tmp_path, capture_output=True, text=True, encoding='utf-8')
    assert run.returncode != 0 and 'F11_FORBIDDEN_OPERATION' in run.stderr
    assert not list(tmp_path.iterdir())


def test_audit_updates_cannot_target_original_source():
    with pytest.raises(RuntimeError, match='OUTPUT_UPDATE_DENIED'):
        f.update_audit('../f_higher_scale_v2_seed42_train1024/last.pt', {})


def test_timeline_timezone_conversion():
    result = f.times('2026-09-28T08:20:07+00:00')
    assert result['UTC'] == '2026-09-28T08:20:07+00:00'
    assert result['Asia/Taipei'] == '2026-09-28T16:20:07+08:00'


def test_saved_access_records_are_development_only(evidence):
    for arm in ['C4', 'H4']:
        access = evidence['telemetry_integrity'][arm]['saved_input_access_audit']
        assert access['outside_frozen_allowlist'] == []
        assert access['decoded_images_during_audit'] == 0
        assert all(not item['parsing']['invalid_records'] for item in access['files'])
