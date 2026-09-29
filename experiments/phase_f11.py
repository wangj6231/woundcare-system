"""F1.1 forensic audit; reads original artifacts and writes NEW audit artifacts only."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'experiments/results'
OUT = RESULTS / 'f_higher_scale_v2_interruption_audit'
ARMS = {'C4': RESULTS/'f_higher_scale_v2_seed42_control768', 'H4': RESULTS/'f_higher_scale_v2_seed42_train1024'}
PAIR = RESULTS/'f_higher_scale_v2_seed42_pair_summary'
PRE = RESULTS/'f_higher_scale_v2_seed42_preflight'
PRO = ROOT/'experiments/protocols'
TREES = [*ARMS.values(), PAIR, PRE]
TZ = timezone(timedelta(hours=8), 'Asia/Taipei')
RESTRICTIONS = dict(training=False, resume=False, retry=False, forward=False, inference=False,
                    final_evaluation=False, GPU_research_run=False, test_images_used=0,
                    locked_test_used=False, CO2Wounds_used=False, external_test_used=False)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def times(value):
    dt = datetime.fromtimestamp(value, timezone.utc) if isinstance(value, (int, float)) else datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return {'UTC': None, 'Asia/Taipei': None, 'raw_without_timezone': value}
    return {'UTC': dt.astimezone(timezone.utc).isoformat(), 'Asia/Taipei': dt.astimezone(TZ).isoformat()}


def record(path):
    p = Path(path)
    if not p.exists():
        return {'path': str(p), 'exists': False}
    stat = p.stat()
    return dict(path=str(p), exists=True, size=stat.st_size, SHA256=sha(p),
                creation_time=times(getattr(stat, 'st_birthtime', stat.st_ctime)), modified_time=times(stat.st_mtime))


def save(name, value):
    target = OUT/name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def update_audit(name, value, preserve_attempt=False):
    """Only derived, new F11 artifacts may be revised, never source trees."""
    allowed = {'checkpoint_integrity.json', 'telemetry_integrity.json', 'anchor_integrity.json', 'integrity.json', 'timeline.json'}
    if name not in allowed:
        raise RuntimeError('OUTPUT_UPDATE_DENIED')
    target = OUT/name
    if target.exists() and preserve_attempt:
        previous = read(target)
        value['prior_audit_attempts'] = previous.get('prior_audit_attempts', []) + [{k:v for k,v in previous.items() if k != 'prior_audit_attempts'}]
    with target.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def protected_inventory():
    base = read(RESULTS/'f_higher_scale_v2_protocol_revision/audit.json')['protected_sha256']
    freeze = read(PRO/'F_HIGHER_SCALE_V2_freeze.json')
    expected = dict(base)
    expected.update(freeze['artifacts_sha256'])
    execution = read(PRO/'F_HIGHER_SCALE_V2_F1_execution_freeze.json')
    expected.update(execution['source_sha256'])
    expected[str(PRO/'F_HIGHER_SCALE_V2_freeze.json')] = execution['F02_freeze_sha256']
    expected[str(PRE/'tests.json')] = execution['tests_sha256']
    rows = []
    for name, digest in expected.items():
        path = Path(name) if Path(name).is_absolute() else ROOT/name
        rec = record(path)
        rec.update(expected_SHA256=digest, matches_expected=rec.get('SHA256') == digest)
        rows.append(rec)
    return rows


def snapshot():
    if OUT.exists():
        raise RuntimeError('Audit output already exists; preserve it, do not overwrite')
    sources = [record(p) for tree in TREES for p in sorted(tree.rglob('*')) if p.is_file()]
    protected = protected_inventory()
    # Freeze itself is not inside its own content map.
    protected.append(record(PRO/'F_HIGHER_SCALE_V2_F1_execution_freeze.json'))
    save('source_snapshot.json', dict(at=times(datetime.now(timezone.utc).isoformat()),
         snapshot_type='Read-only content-hash inventory; NOT a byte-for-byte backup; original files not chmod-ed',
         source_files=sources, protected_files=protected, restrictions=RESTRICTIONS))
    print(json.dumps({'snapshot_source_files': len(sources), 'protected_files': len(protected),
                      'preexisting_hash_mismatches': sum(r.get('matches_expected') is False for r in protected)}))


def parse_jsonl(path):
    rows, errors = [], []
    offset = 0
    data = Path(path).read_bytes()
    for number, line in enumerate(data.splitlines(keepends=True), 1):
        if line.strip():
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('not object')
                rows.append(row)
            except (ValueError, UnicodeDecodeError):
                errors.append(dict(line=number, offset=offset, bytes=len(line), classification='TRUNCATED_EVENT'))
        offset += len(line)
    return rows, dict(valid_records=len(rows), invalid_records=errors, size=len(data), nul_bytes=data.count(b'\0'))


def optimizer_summary(rows, errors):
    applied = sum(r.get('optimizer_update_applied') is True for r in rows)
    skipped = sum(r.get('optimizer_update_skipped') is True for r in rows)
    unknown = sum(r.get('unknown', 0) for r in rows)
    invalid = [i for i, r in enumerate(rows) if (r.get('optimizer_update_applied') is True) == (r.get('optimizer_update_skipped') is True)]
    streak = maximum = 0
    for r in rows:
        streak = streak + 1 if r.get('optimizer_update_skipped') is True else 0
        maximum = max(maximum, streak)
    ids = [(r.get('epoch'), r.get('batch_index')) for r in rows]
    scales = [r[k] for r in rows for k in ('scale_before', 'scale_after') if isinstance(r.get(k), (int, float))]
    # Replay only pure Boolean/counter contract, never runtime training code.
    sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
    from experiments.f02_safety import NumericalSafety
    replay = NumericalSafety()
    for row in rows:
        if replay.observe(row):
            break
    return dict(observed_scheduled=len(rows), applied=applied, skipped=skipped, unknown=unknown,
                frozen_contract_replay=replay.summary(),
                equation_valid=applied + skipped == len(rows) and not invalid and unknown == 0,
                invalid_classification_record_indices=invalid, duplicate_opportunities=len(ids)-len(set(ids)),
                monotonic_opportunities=ids == sorted(ids), max_consecutive_skips=maximum,
                minimum_scaler=min(scales) if scales else None, last_observed=rows[-1] if rows else None,
                truncated_events=errors['invalid_records'], missing_unpersisted_events='UNKNOWN')


def compare_sequence(observed, expected, complete):
    if not observed:
        return 'UNKNOWN'
    return 'YES' if observed == (expected if complete else expected[:len(observed)]) else 'NO'


def anchors(path, orders, completed):
    rows, parsing = parse_jsonl(path)
    sequences = defaultdict(list)
    layout_errors = []
    for row in rows:
        seq = sequences[row['epoch']]
        if row['start_position'] != len(seq):
            layout_errors.append({'epoch': row['epoch'], 'batch_index': row['batch_index']})
        seq.extend(row['sample_ids'])
    full = [{'epoch_index': e, 'observed_anchors': len(sequences[e]),
             'EXPECTED_ORDER_MATCH': compare_sequence(sequences[e], orders[e], True)} for e in range(completed)]
    prefix = sequences[completed] if completed < len(orders) else []
    verdict = 'NO' if layout_errors or any(r['EXPECTED_ORDER_MATCH'] == 'NO' for r in full) else (
              'YES' if full and all(r['EXPECTED_ORDER_MATCH'] == 'YES' for r in full) else 'UNKNOWN')
    return dict(parsing=parsing, completed_epochs=full, EXPECTED_ORDER_MATCH=verdict,
                partial_epoch_index=completed if completed < 300 else None,
                partial_observed_anchors=len(prefix) if prefix else None,
                EXPECTED_PREFIX_MATCH=compare_sequence(prefix, orders[completed], False) if completed < 300 else 'NOT_APPLICABLE',
                layout_errors=layout_errors, missing_evidence='Zero-byte telemetry cannot establish actual consumed IDs' if not rows else None)


def csv_epochs(path):
    raw = Path(path).read_bytes()
    parsed = list(csv.reader(raw.decode('utf-8', errors='replace').splitlines()))
    header = parsed[0]
    epochs, bad = [], []
    for index, row in enumerate(parsed[1:], 2):
        try:
            if len(row) != len(header) or any('\0' in cell for cell in row):
                raise ValueError()
            epochs.append(int(row[0].strip()))
        except (ValueError, IndexError):
            bad.append(index)
    return dict(valid_row_count=len(epochs), epoch_numbers=epochs, last_epoch_number=max(epochs) if epochs else None,
                invalid_lines=bad, nul_bytes=raw.count(b'\0'), contiguous_from_one=epochs == list(range(1, len(epochs)+1)))


def tensorboard(path):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
    data = path.read_bytes()
    offset = 0
    rows, invalid = [], []
    while offset < len(data):
        start = offset
        if len(data)-offset < 12:
            invalid.append({'offset': start, 'classification': 'TRUNCATED_EVENT'}); break
        length = struct.unpack_from('<Q', data, offset)[0]
        if masked_crc32c(data[offset:offset+8]) != struct.unpack_from('<I', data, offset+8)[0]:
            invalid.append({'offset': start, 'classification': 'INVALID_LENGTH_CRC'}); break
        offset += 12
        if length > len(data)-offset-4:
            invalid.append({'offset': start, 'classification': 'TRUNCATED_EVENT'}); break
        body = data[offset:offset+length]
        if masked_crc32c(body) != struct.unpack_from('<I', data, offset+length)[0]:
            invalid.append({'offset': start, 'classification': 'INVALID_BODY_CRC'}); break
        event = Event.FromString(body)
        if event.HasField('summary'):
            rows.append(dict(step=event.step, wall_time=times(event.wall_time), tags=[v.tag for v in event.summary.value]))
        offset += length + 4
    return dict(path=str(path), scalar_values_not_read_or_compared=True, valid_summary_events=len(rows),
                distinct_steps=sorted({r['step'] for r in rows}), last_event=rows[-1] if rows else None,
                invalid_records=invalid, valid_bytes=offset, file_bytes=len(data))


def checkpoints():
    result = {}
    previous = read(OUT/'checkpoint_integrity.json') if (OUT/'checkpoint_integrity.json').exists() else {}
    for arm, name, path in [('C4', 'best', ARMS['C4']/'best.pt'), ('C4', 'last', ARMS['C4']/'last.pt'),
                            ('H4', 'best', ARMS['H4']/'training/weights/best.pt'), ('H4', 'last', ARMS['H4']/'training/weights/last.pt')]:
        rec = record(path)
        if not rec.get('size'):
            rec.update(integrity='INVALID_ZERO_BYTE_FILE' if rec.get('exists') else 'MISSING', deserialization_attempted=False)
        else:
            cached = previous.get(arm+'_'+name, {})
            if cached.get('child_exit_code') == 0 and cached.get('child_stdout') and cached.get('SHA256') == rec['SHA256']:
                stdout, stderr, code = cached['child_stdout'], cached['child_stderr'], 0
                rec['reparsed_existing_child_stdout_without_reloading'] = True
            else:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
                run = subprocess.run([sys.executable, '-B', str(ROOT/'experiments/f11_checkpoint_reader.py'), str(path)],
                                     capture_output=True, text=True, encoding='utf-8', env=env, timeout=120)
                stdout, stderr, code = run.stdout, run.stderr, run.returncode
            rec.update(child_exit_code=code, child_stderr=stderr)
            try:
                lines = stdout.strip().splitlines()
                rec.update(json.loads(lines[-1]))
                rec['blocked_import_side_effect_messages'] = lines[:-1]
            except ValueError:
                rec.update(integrity='UNKNOWN', child_stdout=stdout)
        result[arm+'_'+name] = rec
    update_audit('checkpoint_integrity.json', result, preserve_attempt=True)
    print(json.dumps({k: {'integrity': v['integrity'], 'metadata': v.get('metadata'), 'nonfinite': v.get('nonfinite_tensors')} for k, v in result.items() if k != 'prior_audit_attempts'}))


def analyze():
    cfg = read(PRO/'F_HIGHER_SCALE_V2_control_config.json')
    orders_data = read(ROOT/cfg['sampling']['order_path'])
    orders = orders_data['orders']
    allowed_images = {(ROOT/r['image_path']).resolve() for r in read(ROOT/cfg['manifest']['path'])['samples']}
    allowed_images.update((ROOT/'outputs/isic_fuseg_pretrain_smoke_20260914/fuseg_dataset/images/val'/r['sample_id']).resolve()
                          for r in read(PRO/'ISIC_FUSEG_COLORFIX_V1_validation_manifest.json')['samples'])
    tele, anch = {}, {}
    for arm, path in ARMS.items():
        epoch_rows, epoch_parse = parse_jsonl(path/'epochs.jsonl')
        opt_rows, opt_parse = parse_jsonl(path/'optimizer_telemetry.jsonl')
        completed = len(epoch_rows)
        partial = [r for r in opt_rows if r['epoch'] >= completed]
        numerical_rows, numerical_parse = parse_jsonl(path/'numerical_safety_telemetry.jsonl')
        anchor_rows, _ = parse_jsonl(path/'anchor_telemetry.jsonl')
        partial_anchors = [r for r in anchor_rows if r['epoch'] == completed]
        partial_raw_loss = [r for r in numerical_rows if r.get('epoch') == completed and r.get('event') == 'raw_loss']
        raw = (PAIR/f'train_{arm}.log').read_bytes()
        logtext = raw.decode('utf-8', errors='replace')
        # Console progress is distinct from committed epoch completion or batch-level ledger proof.
        progress = re.findall(r'(\d+)/300[^\r\n]*?(\d+)/193', logtext)
        stops = ['FAIL_NONFINITE_LOSS', 'FAIL_NONFINITE_PARAMETERS', 'FAIL_AMP_SCALE_COLLAPSE',
                 'FAIL_PERSISTENT_AMP_UPDATE_SKIPS', 'TRAINING_NUMERICAL_RUNTIME_INVALID',
                 'FAIL_RESOURCE_RUNTIME', 'FAIL_TELEMETRY_INCOMPLETE']
        oom = ['CUDA out of memory', 'OutOfMemoryError', 'CUBLAS', 'CUDA error', 'device-side assert', 'allocator failure']
        recorded_stops = sorted({s for s in stops if s in logtext or any(s in str(r.get('failure') or r.get('reason') or '') for r in opt_rows + numerical_rows)})
        serial = dict(completed_epoch_count=completed, last_completed_epoch_index=epoch_rows[-1]['epoch'] if epoch_rows else None,
                      last_fully_completed_epoch_number=completed, epoch_parse=epoch_parse,
                      epoch_indices_contiguous=[r['epoch'] for r in epoch_rows] == list(range(completed)),
                      last_epoch_ledger=epoch_rows[-1] if epoch_rows else None,
                      optimizer=optimizer_summary(opt_rows, opt_parse), optimizer_parse=opt_parse,
                      per_epoch_optimizer_counts=dict(Counter(r['epoch'] for r in opt_rows)),
                      numerical_parse=numerical_parse,
                      nonfinite_loss_records=[r for r in numerical_rows if r.get('raw_total_loss_finite') is False or r.get('raw_components_finite') is False],
                      nonfinite_parameter_records=[r for r in opt_rows if r.get('parameter_finite') is False],
                      csv=csv_epochs(path/'training/results.csv'),
                      tensorboard=[tensorboard(p) for p in path.rglob('events.out.tfevents.*')],
                      console=dict(nul_bytes=raw.count(b'\0'), last_training_progress=progress[-1] if progress else None,
                                   last_nonzero_text=logtext.rstrip('\0\r\n')[-1600:], event_timestamps='NOT_PRESENT'),
                      recorded_numerical_stops=recorded_stops,
                      numerical_hard_stop_verdict='RECORDED_STOP' if recorded_stops else 'NO_RECORDED_PREREGISTERED_NUMERICAL_HARD_STOP_BEFORE_INTERRUPTION',
                      recorded_OOM='YES' if any(s.lower() in logtext.lower() for s in oom) else 'NO',
                      absence_scope='Only surviving recorded data; missing intervals are UNKNOWN',
                      partial=dict(epoch_index=completed if partial else None, epoch_number=completed+1 if partial else None,
                                   optimizer=optimizer_summary(partial, {'invalid_records': []}),
                                   observed_optimizer_batch_indices=[r['batch_index'] for r in partial],
                                   last_optimizer_batch_index=partial[-1]['batch_index'] if partial else None,
                                   anchor_batches_observed=len(partial_anchors), raw_loss_batches_observed=len(partial_raw_loss),
                                   raw_loss_batch_indices=[r['batch'] for r in partial_raw_loss],
                                   last_raw_loss_batch_index=partial_raw_loss[-1]['batch'] if partial_raw_loss else None,
                                   training_batch_completion_console_evidence=progress[-1] if progress else None,
                                   full_epoch_persistence_complete=False if partial else None,
                                   exact_completed_batch_count=193 if partial and progress and tuple(progress[-1]) == (str(completed+1), '193') and len(partial_raw_loss) == 193 else 'UNKNOWN',
                                   actual_consumed_anchor_count=sum(len(r['sample_ids']) for r in partial_anchors) if partial_anchors else 'UNKNOWN',
                                   telemetry_event_timestamp='NOT_RECORDED', log_event_timestamp='NOT_RECORDED'))
        serial['full_epoch_scheduled_budget_match'] = all(
            serial['per_epoch_optimizer_counts'].get(e, 0) == cfg['budget']['scheduled_optimizer_calls_per_epoch'][e] for e in range(completed))
        access, invalid_access, access_files = [], [], []
        for log in path.glob('input_access_*.jsonl'):
            access_rows, access_parse = parse_jsonl(log)
            access_files.append(dict(path=str(log), parsing=access_parse))
            for row in access_rows:
                p = Path(row['path']).resolve()
                valid_path = p in allowed_images or p.is_relative_to(path.resolve()) or any(p.is_relative_to(Path(base).resolve()) for base in ['C:/Python312/Lib/site-packages/ultralytics/assets', 'C:/Python312/Lib/site-packages/matplotlib/mpl-data'])
                access.append(row)
                if not valid_path:
                    invalid_access.append(row)
        serial['saved_input_access_audit'] = dict(files=access_files, total_records=len(access),
            roles=dict(Counter(r['role'] for r in access)), outside_frozen_allowlist=invalid_access,
            decoded_images_during_audit=0, scope='Saved access records only; no claim about unpersisted events')
        tele[arm] = serial
        anch[arm] = anchors(path/'anchor_telemetry.jsonl', orders, completed)
    tele['persisted_pair_status'] = read(PAIR/'status.json')
    tele['C4_completion'] = read(ARMS['C4']/'training_completion.json')
    tele['final_evaluation_artifacts'] = [str(p) for tree in [*ARMS.values(), PAIR] for p in tree.rglob('*') if p.is_file() and ('evaluation.lock' in p.name or 'paired_comparison' in p.name or 'pair_validity' in p.name)]
    update_audit('telemetry_integrity.json', tele)
    update_audit('anchor_integrity.json', anch)
    print(json.dumps({arm: {'completed': tele[arm]['completed_epoch_count'], 'optimizer': {k:v for k,v in tele[arm]['optimizer'].items() if k != 'last_observed'}, 'console_progress':tele[arm]['console']['last_training_progress'], 'anchor_match': anch[arm]['EXPECTED_ORDER_MATCH']} for arm in ARMS}))


def system_evidence():
    command = """[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$events = @(Get-WinEvent -FilterHashtable @{LogName='System'; Id=41,6008,1074,6005; StartTime=[datetime]'2026-09-28T15:00:00'} -MaxEvents 12 | ForEach-Object { [PSCustomObject]@{id=$_.Id; provider=$_.ProviderName; time_UTC=$_.TimeCreated.ToUniversalTime().ToString('o'); time_Taipei=$_.TimeCreated.ToString('o'); message=$_.Message; xml=$_.ToXml()} })
$processes = @(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in 25652,12604 -or ($_.Name -match '^python' -and $_.CommandLine -match '(phase_f1_v2|f1_v2_runner)') } | Select-Object ProcessId,Name,CreationDate,CommandLine)
[PSCustomObject]@{events=$events; matching_processes=$processes; last_boot=(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')} | ConvertTo-Json -Depth 8
"""
    run = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command], capture_output=True,
                         encoding='utf-8', errors='replace', timeout=30)
    try:
        evidence = json.loads(run.stdout)
    except ValueError:
        evidence = {'events': [], 'read_error': run.stderr or run.stdout}
    ids = {e['id'] for e in evidence.get('events', [])}
    evidence.update(read_at=times(datetime.now(timezone.utc).isoformat()), exit_code=run.returncode,
                    INTERRUPTION_CAUSE='UNVERIFIED_HOST_OR_PROCESS_INTERRUPTION',
                    host_interruption_supported={41, 6008} <= ids,
                    precise_hardware_cause='UNVERIFIED', training_process_termination_timestamp='UNKNOWN',
                    chronology_conflict='6008 reports prior shutdown at 16:17:45+08, while surviving training artifacts record 16:20-16:21+08; exact interruption time cannot be reconciled from these records alone',
                    interpretation='Non-clean host shutdown and later restart recorded; cannot distinguish power loss, hang, crash or manual hard reset')
    save('system_interruption_evidence.json', evidence)


def finalize():
    test_run = subprocess.run([sys.executable, '-B', '-m', 'pytest', 'tests/test_phase_f11.py', '-q', '-p', 'no:cacheprovider'],
        cwd=ROOT, capture_output=True, encoding='utf-8', env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8'), timeout=60)
    if test_run.returncode:
        print(test_run.stdout + test_run.stderr)
        raise RuntimeError('F11_AUDIT_TESTS_FAILED')
    snapshot_data = read(OUT/'source_snapshot.json')
    checkpoint = read(OUT/'checkpoint_integrity.json')
    tele = read(OUT/'telemetry_integrity.json')
    anchor = read(OUT/'anchor_integrity.json')
    system = read(OUT/'system_interruption_evidence.json')
    comparisons = []
    for original in snapshot_data['source_files'] + snapshot_data['protected_files']:
        current = record(original['path'])
        comparisons.append(dict(path=original['path'], before_SHA256=original.get('SHA256'),
                                after_SHA256=current.get('SHA256'), before_size=original.get('size'), after_size=current.get('size'),
                                unchanged=original.get('SHA256') == current.get('SHA256') and original.get('size') == current.get('size')))
    existing = {r['path'] for r in snapshot_data['source_files']}
    now_files = {str(p) for tree in TREES for p in tree.rglob('*') if p.is_file()}
    changes = [r for r in comparisons if not r['unchanged']]
    best = checkpoint['H4_best']
    valid = best['integrity'] == 'VALID'
    states = best.get('states_present', {})
    missing = [key for key in ['optimizer', 'scheduler', 'GradScaler', 'RNG', 'sampler', 'augmentation', 'update_counter'] if not states.get(key)]
    recovery = 'VALID_BEST_CHECKPOINT_ONLY_NOT_EXACT_RESUME' if valid else ('NO_VALID_H4_CHECKPOINT' if best['integrity'] == 'INVALID' else 'INSUFFICIENT_EVIDENCE')
    audit_complete = valid and not changes and not now_files-existing and all(checkpoint[k]['integrity'] == 'VALID' for k in ['C4_best', 'C4_last']) and anchor['H4']['EXPECTED_ORDER_MATCH'] == 'YES' and anchor['H4']['EXPECTED_PREFIX_MATCH'] == 'YES'
    final_status = dict(PHASE_F11_STATUS='COMPLETE' if audit_complete else 'BLOCKED',
         ORIGINAL_F1_STATUS='INVALID_OR_INTERRUPTED', PAIR_FIXED_BUDGET_VALID='NO',
         C4_TRAINING_COMPLETE='YES', H4_TRAINING_COMPLETE='NO',
         FINAL_OPERATIONAL_EVALUATION_PERFORMED=False, HIGHER_SCALE_EFFECT_CONCLUSION='NOT_AVAILABLE',
         H4_LAST_CHECKPOINT_INTEGRITY=checkpoint['H4_last']['integrity'], H4_BEST_CHECKPOINT_INTEGRITY=best['integrity'],
         EXACT_RESUME_TECHNICALLY_POSSIBLE='NO' if valid and missing else 'UNKNOWN',
         ORIGINAL_F1_RESUME_AUTHORIZED='NO', RECOVERY_STATE=recovery, NEW_TRAINING_AUTHORIZED='NO')
    feasibility = dict(final_status=final_status, restrictions=RESTRICTIONS, missing_exact_resume_states=missing,
        checkpoint_artifact_corruption=True, model_state_corruption_during_training='NOT_ESTABLISHED',
        H4_best_research_artifact='VALID_RESEARCH_ARTIFACT' if valid else 'UNVERIFIED_OR_INVALID_RESEARCH_ARTIFACT',
        last_persisted_checkpoint_epoch_index=best.get('metadata', {}).get('epoch'),
        scheduler_telemetry='Scalar/dictionary observations survive, but not an atomic model/optimizer/scaler/RNG/sampler boundary snapshot',
        partial_epoch_rule='Restarting an earlier complete boundary repeats exposure; it is not exact mid-epoch resume',
        future_options={
            'A': 'Fresh C4 + fresh H4; cleanest fresh paired replication, highest cost; new preregistration required',
            'B': 'Reuse frozen completed C4 + fresh H4 replacement; asymmetric replacement must disclose interrupted H4 and freeze all choices before new execution',
            'C': 'Checkpoint restart/resume in separately versioned protocol; complex optimizer/scaler/RNG/partial-epoch continuity; available best is not last complete boundary and does not satisfy exact resume'},
        option_selected=None, STOP_AFTER_F11=True)
    save('recovery_feasibility.json', feasibility)
    events = []
    for r in snapshot_data['source_files']:
        if 'train1024' in r['path'] and (Path(r['path']).suffix in {'.pt', '.csv', '.jsonl'} or 'tfevents' in r['path']):
            events.append(dict(event='file modification', path=r['path'], time=r['modified_time'],
                               semantic='Filesystem mtime, NOT exact event completion or corruption-onset time'))
    for arm in ARMS:
        for tb in tele[arm]['tensorboard']:
            if tb['last_event']:
                events.append(dict(event=arm+' last CRC-valid TensorBoard summary', epoch_number=tb['last_event']['step'], time=tb['last_event']['wall_time']))
    events.append(dict(event='Last persisted pair status: H4 completed epoch297', time=times(tele['persisted_pair_status']['at'])))
    for e in system.get('events', []):
        events.append(dict(event='Windows event '+str(e['id']), time=times(e['time_UTC']), message=e['message']))
    for name in ['last partial raw-loss batch192', 'last optimizer opportunity batch181', 'last completed training batch192', 'actual process termination']:
        events.append(dict(event=name, time=None, time_status='UNKNOWN_NO_EVENT_TIMESTAMP'))
    events.sort(key=lambda e: e['time']['UTC'] if e.get('time') and e['time'].get('UTC') else 'Z')
    save('timeline.json', dict(events=events, chronology_conflict=system['chronology_conflict']))
    receipt = dict(restrictions=RESTRICTIONS, protected_and_source_hash_checks=len(comparisons),
        tests=dict(command='python -B -m pytest tests/test_phase_f11.py -q -p no:cacheprovider', exit_code=test_run.returncode, stdout=test_run.stdout, stderr=test_run.stderr,
                   earlier_harness_issue='Initial run: 27 passed, 4 subprocess-output decoding failures (cp950 vs UTF-8); explicit UTF-8 fixed in new F11 tests only'),
        changes=changes, added_original_files=sorted(now_files-existing), deleted_original_files=sorted(existing-now_files),
        preexisting_frozen_hash_mismatches=[r for r in snapshot_data['protected_files'] if r.get('matches_expected') is False],
        all_original_hashes_unchanged=not changes and now_files == existing, comparisons=comparisons,
        C4_completion_checkpoint_hash_match={name: checkpoint['C4_'+name]['SHA256'] == tele['C4_completion'][name+'_checkpoint_sha256'] for name in ['best', 'last']},
        audit_code_sha256={str(ROOT/p):sha(ROOT/p) for p in ['experiments/phase_f11.py','experiments/f11_checkpoint_reader.py','tests/test_phase_f11.py']},
        final_status=final_status)
    save('integrity.json', receipt)
    print(json.dumps(final_status))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['snapshot', 'checkpoints', 'analyze', 'system', 'finalize', 'verify'])
    args = parser.parse_args()
    {'snapshot': snapshot, 'checkpoints': checkpoints, 'analyze': analyze, 'system': system_evidence, 'finalize': finalize, 'verify': verify}[args.stage]()


def verify():
    """Final post-report verification; only update the NEW audit receipt."""
    receipt = read(OUT/'integrity.json')
    test_run = subprocess.run([sys.executable, '-B', '-m', 'pytest', 'tests/test_phase_f11.py', '-q', '-p', 'no:cacheprovider'],
        cwd=ROOT, capture_output=True, encoding='utf-8', env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8'), timeout=60)
    if test_run.returncode:
        raise RuntimeError(test_run.stdout + test_run.stderr)
    source = read(OUT/'source_snapshot.json')
    changes = [r['path'] for r in source['source_files'] + source['protected_files'] if not Path(r['path']).exists() or sha(r['path']) != r['SHA256']]
    old_paths = {r['path'] for r in source['source_files']}
    new_paths = {str(p) for t in TREES for p in t.rglob('*') if p.is_file()}
    if changes or old_paths != new_paths:
        raise RuntimeError('SOURCE_CHANGED:' + str(changes))
    receipt['tests'].update(exit_code=0, stdout=test_run.stdout, stderr=test_run.stderr)
    receipt['post_report_rehash_at'] = times(datetime.now(timezone.utc).isoformat())
    receipt['post_report_original_hashes_unchanged'] = True
    receipt['audit_code_sha256'] = {str(ROOT/p):sha(ROOT/p) for p in ['experiments/phase_f11.py','experiments/f11_checkpoint_reader.py','tests/test_phase_f11.py']}
    frozen = read(PRO/'F_HIGHER_SCALE_V2_F1_execution_freeze.json')
    receipt['additional_frozen_checks'] = {
        'F1_protocol_matches_execution_freeze': sha(PRO/'F_HIGHER_SCALE_V2_protocol.json') == frozen['protocol_sha256'],
        'F1_authorization_request_matches_execution_freeze': sha(Path('C:/Users/milo9/.codex/attachments/d834d458-3d3d-44cf-81dc-e74b1de6f800/貼上的文字.txt')) == frozen['request_sha256']}
    if not all(receipt['additional_frozen_checks'].values()):
        raise RuntimeError('F1_FREEZE_MISMATCH')
    timeline = read(OUT/'timeline.json')
    extra = []
    for r in source['source_files']:
        if Path(r['path']).name in {'train_H4.log', 'supervisor.stdout.log', 'supervisor.stderr.log'}:
            extra.append(dict(event='console/stdout/stderr file modification', path=r['path'], time=r['modified_time'],
                semantic='Filesystem mtime only; empty stderr is not proof of no unpersisted error'))
    timeline['events'].extend(extra)
    timeline['events'].sort(key=lambda e: e['time']['UTC'] if e.get('time') and e['time'].get('UTC') else 'Z')
    update_audit('timeline.json', timeline)
    receipt['delivered_artifact_sha256'] = {str(p):sha(p) for p in OUT.glob('*.json') if p.name != 'integrity.json'}
    report = ROOT/'docs/PHASE_F11_INTERRUPTION_CHECKPOINT_AUDIT_20260928.md'
    receipt['report_sha256'] = sha(report)
    update_audit('integrity.json', receipt)
    print(json.dumps({'tests':test_run.stdout.strip(), 'original_hashes_unchanged':True, 'report_sha256':receipt['report_sha256']}))


if __name__ == '__main__':
    main()
