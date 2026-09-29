"""Build frozen V2 paired preregistration. Never construct/train/infer a model."""
from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
from experiments import phase_d0 as d0
from experiments.paired_sampling_runner import OUTPUTS, config_diff, require_output_absent, plan_indices

PREFIX='D_SEG_SMALL_SAMPLING_V2'
RUNNER=Path('experiments/paired_sampling_runner.py')
REPORT=Path('docs/PHASE_D01_PAIRED_SAMPLING_PREREGISTRATION_20260922.md')
SUFFIXES=['protocol','control_config','experimental_config','telemetry_schema','evaluation_protocol','dry_audit']


def verified_inputs(root):
    p=root/'experiments/protocols';freeze=d0.read(p/f'{d0.PREFIX}_freeze.json')
    for name,h in freeze['artifacts_sha256'].items():
        if d0.sha(p/name)!=h:raise d0.D0Error('V1_FREEZE_CHANGED')
    if d0.sha(root/'experiments/phase_d0.py')!=freeze['analysis_code_sha256']:raise d0.D0Error('V1_CODE_CHANGED')
    if d0.read(p/f'{d0.PREFIX}_verification.json')['PHASE_D0_STATUS']!='BLOCKED':raise d0.D0Error('V1_BLOCKED_STATUS_CHANGED')
    audit=d0.read(p/f'{d0.PREFIX}_audit.json');protected=deepcopy(audit['protected_before'])
    for path,h in protected.items():
        if d0.sha(path)!=h:raise d0.D0Error('SEALED_INPUT_CHANGED: '+path)
    for path in list(p.glob(d0.PREFIX+'*'))+[root/'experiments/phase_d0.py',root/'tests/test_phase_d0.py',root/'docs/PHASE_D0_SMALL_OBJECT_SAMPLING_PREREGISTRATION_20260921.md']:
        protected[str(path)]=d0.sha(path)
    config=d0.read(p/f'{d0.PREFIX}_training_config.json')
    initial=config['control']['initialization'];expected='1ec926026473beafafb1ef8da6aa6efcc13d38c391ec8db48006c5d793d5f2a3'
    if initial['sha256']!=expected or d0.sha(root/initial['path'])!=expected:raise d0.D0Error('INITIALIZATION_MISMATCH')
    manifest=d0.read(p/f'{d0.PREFIX}_training_manifest.json')
    cp=d0.read(root/d0.C_PROTOCOL);val=d0.read(root/cp['validation_manifest']['path'])['samples']
    d0.validate_exclusion(manifest['samples'],val)
    for row in manifest['samples']:
        if d0.sha(root/row['image_path'])!=row['image_hash'] or d0.sha(root/row['label_path'])!=row['label_hash']:raise d0.D0Error('TRAIN_DATA_CHANGED')
    return p,config,manifest,protected


def build(root):
    root=Path(root).resolve();require_output_absent(root)
    dest,old,manifest,protected=verified_inputs(root)
    if list(dest.glob(PREFIX+'*')):raise d0.D0Error('V2_ALREADY_EXISTS_REFUSE_OVERWRITE')
    runner_sha=d0.sha(root/RUNNER)
    common=deepcopy(old['control']);common.pop('sampler')
    common['budget']['AMP_successful_update_count']='MEASURE_BOTH_FRESH_ARMS_VIA_OPTIMIZER_POST_HOOK'
    common['budget']['early_stop_rule']='Either arm <300 epochs: NOT_VALID, stop pair; never resume, top up, alter patience or rerun one arm.'
    common['manifest']={'path':f'experiments/protocols/{d0.PREFIX}_training_manifest.json','sha256':d0.sha(dest/f'{d0.PREFIX}_training_manifest.json')}
    common['runner']={'path':RUNNER.as_posix(),'sha256':runner_sha,'adapter':'make_trainer_adapter','base_class':'ultralytics.models.yolo.segment.train.SegmentationTrainer',
      'D1_entrypoint':'Separate explicit authorization/driver required; no execute entrypoint in D0.1',
      'runtime_guard':'Pinned software/source hashes and registered args must pass before model construction; AMP optimizer must not be fused or internal-AMP-aware.'}
    common['sampling']={'mode':'uniform_shuffled','weights':{'A':1.0,'B':1.0,'C':1.0},'replacement':False,
      'seed':42,'epoch_anchor_draws':771,'seed_policy':'Independent PCG64 SeedSequence([42,zero_based_epoch]); control permutation vs frozen V1 weighted choice',
      'sequence_identity_policy':'Different ordered sequences expected; actual consumed im_file paths must exactly match the declared epoch plan.'}
    common['data_yaml_path']=(d0.BUNDLE/'fuseg_dataset/dataset.yaml').as_posix()
    control=deepcopy(common);control.update(experiment_id=PREFIX+'_CONTROL',output_path=OUTPUTS['C'],arm_label='C')
    experimental=deepcopy(common);experimental.update(experiment_id=PREFIX+'_EXPERIMENTAL',output_path=OUTPUTS['S'],arm_label='S')
    experimental['sampling'].update(mode='small_aware_weighted',weights=deepcopy(d0.WEIGHTS),replacement=True)
    differences=config_diff(control,experimental)
    evaluation=deepcopy(d0.read(dest/f'{d0.PREFIX}_evaluation_protocol.json'))
    eval_path=dest/f'{PREFIX}_evaluation_protocol.json';d0.write(eval_path,evaluation)
    for cfg in [control,experimental]:cfg['evaluation_protocol_sha256']=d0.sha(eval_path)
    schema={'version':2,'time_index':'zero-based epoch, batch_index, global_batch; completed_epochs is a count',
      'optimizer_opportunity':{'required':['global_batch','epoch','batch_index','accumulation','optimizer_call_attempted','grad_scaler_scale_before','grad_scaler_scale_after','optimizer_step_applied','optimizer_step_skipped','skip_reason_if_known','scheduler_state','learning_rate','telemetry_status'],
        'detection':'Real optimizer.register_step_post_hook observed exactly once = applied; no post hook and normally returned original scaler step = AMP_STEP_SKIPPED.',
        'not_used':['optimizer return value','inferred overflow from scaler ratio','private found_inf tensors'],
        'unsupported':'fused/internal-AMP-aware optimizers: fail closed, because optimizer method invocation may not mean applied update',
        'exceptions':'Record FAILED with applied/skipped=null and observed hooks; stop, never classify exception as AMP skip.'},
      'anchor':{'required':['epoch','batch_index','anchor_position','sample_id','dataset_index','sampling_category','sampling_mode'],
        'experimental_additional':['draw_probability','weight'],'control_additional':['uniform_control_identity','marginal_probability'],
        'unit':'actual consumed anchor at preprocess_batch entry; prefetch-only draws not counted'},
      'completion_required':['initialization_sha256','best_checkpoint_sha256','last_checkpoint_sha256','completed_epochs','scheduled_optimizer_calls','applied_optimizer_updates','skipped_optimizer_updates','unknown_optimizer_opportunities','training_runtime_seconds','GPU','CUDA','torch','ultralytics','python','runner_sha256'],
      'persistence':'exclusive execution.lock; exclusive JSONL creation, flush each optimizer and anchor batch; no overwrite/resume',
      'manipulation_checks':['unique_anchors_seen','repeat_draws','A/B/C_exposure','small_GT','very_small_GT','medium_GT','large_GT','negative_image_exposure','multi_GT_anchor_exposure'],
      'augmentation_caveat':'Unaugmented GT attached to anchors, not a claim of pixel-identical augmentations or companion-image exposure.'}
    gate={'primary_comparator':'FRESH_CONTROL','all_required':{
      'small':'experimental_small_TP >= control_small_TP + 1; support 137',
      'precision':'experimental_P >= control_P - 0.01',
      'F1':'experimental_F1 >= control_F1 - 0.01',
      'crop':'experimental_crop_complete_count >= control_count - 1; positives 186',
      'medium':'experimental_medium_TP >= control_medium_TP - 1; support 90',
      'large':'experimental_large_TP >= control_large_TP; support 14'},
      'arithmetic':'Exact rational arithmetic for P/F1; no rounded-percentage boundary decisions',
      'success_status':'PASS_SMALL_SAMPLING_RESEARCH_ADVANCEMENT_GATE','historical_development_gate_separate':evaluation['acceptance_gate']}
    protocol={'experiment_id':PREFIX,'status':'DEFINED_BEFORE_TRAINING','completion_authority':f'{PREFIX}_freeze.json',
      'reason':'MISSING_HISTORICAL_TELEMETRY; not TRAINING_BUG_CONFIRMED','V1_role':'IMMUTABLE_BLOCKED_HISTORICAL_PREREGISTRATION',
      'primary_comparison':['FRESH_INSTRUMENTED_CONTROL','FRESH_INSTRUMENTED_SMALL_AWARE_EXPERIMENTAL'],
      'historical_reference':old['baseline']|{'role':'HISTORICAL_REFERENCE_COMPARATOR','reinference_allowed':False},
      'prohibited_comparator':'historical RGB/BGR-defective gate; methodological history only',
      'hypotheses':d0.read(dest/f'{d0.PREFIX}_protocol.json')['hypotheses']|{'H1':'Small-object-aware image sampling improves small recall relative to the fresh instrumented uniform control, not historical Phase C'},
      'single_training_intervention':'ANCHOR_IMAGE_SAMPLING_DISTRIBUTION',
      'training_order':['C','S'],'final_corrected_evaluation_order':['C','S'],
      'order_policy':'Train both arms before reading final corrected evaluation outcomes. Do not cancel S because of C performance. Safety/integrity failure or early stopping invalidates/stops the pair.',
      'native_epoch_validation':'Historical val=True/early-stopping validation preserved identically; not the final corrected Phase C operating-point evaluator. Never use its performance to revise weights or choose whether to launch S.',
      'fixed_budget_validity':{'required_completed_epochs_per_arm':300,'required_scheduled_optimizer_calls_per_arm':3741,
        'same_runner_AMP_scaler_recipe':True,'require_identical_realized_applied_updates':False,
        'early_stop_or_crash':'NOT_VALID; stop and report; no automatic resume, top-up or isolated retry'},
      'amp_sensitivity':{'report_any_skips':True,'imbalance_flag':'AMP_UPDATE_COUNT_IMBALANCE_OBSERVED',
        'imbalance_rule':'abs(control_skips-experimental_skips)>0, fixed conservatively before outcomes',
        'meaning':'Flag for separate interpretation, not automatic gate failure or retraining',
        'required_wording_if_different':'The arms received the same scheduled optimization budget, but realized successful optimizer-update counts differed because of AMP step skipping.',
        'forbidden_claim':'identical realized parameter-update count when counts differ'},
      'augmentation_policy':{'algorithms_probabilities_configuration_identical':True,'pixel_identical_claim':False,
        'anchor_RNG_and_mosaic_companion_changes':'DOWNSTREAM_EFFECT_OF_SAMPLING_INTERVENTION'},
      'advancement_gate':gate,'secondary_diagnostics':['five small-size bins unchanged from C1','very_small <0.25%','single/multi GT recall','single/multi GT crop failures','overall recall','no ROI','FP/FN counts','full C vs S vs historical C table'],
      'future_outputs':OUTPUTS,'locks':'Each future arm: atomic fresh directory, immutable exclusive execution.lock before model loading; absent directory required; no overwrite/delete-and-retry.',
      'runtime_integration_contract':'D1 driver validates freeze and data roles before model load; binds the same adapter to pinned SegmentationTrainer, records final checkpoint/runtime summary, checks pair validity, evaluates C then S with frozen Phase C evaluator.',
      'D1_authorized':False,'multi_seed_authorized':False,'app_replacement_authorized':False,
      'training_performed':False,'model_loaded':False,'model_inference':False,'locked_test_used':False,'CO2Wounds_used':False,'STOP_AFTER_D01':True}
    # Audit all 300 plans without decoding images, importing torch, or reading outcomes.
    totals={'C':Counter(),'S':Counter()}
    for epoch in range(300):
        for arm in ['C','S']:
            idx=plan_indices(manifest['samples'],arm,epoch)
            if len(idx)!=771 or any(i<0 or i>=771 for i in idx):raise d0.D0Error('INVALID_DRY_SAMPLER_PLAN')
            if arm=='C' and len(set(idx.tolist()))!=771:raise d0.D0Error('CONTROL_NOT_WITHOUT_REPLACEMENT')
            totals[arm].update(map(int,idx))
    require_output_absent(root)
    changed=[path for path,h in protected.items() if d0.sha(path)!=h]
    if changed:raise d0.D0Error('IMMUTABLE_INPUT_CHANGED')
    dry={'status':'PASS','TRAINING_SEMANTIC_DIFF':'SAMPLING_ONLY','allowed_config_differences':differences,
      'protected_before':protected,'protected_after':protected,'protected_files_unchanged':len(protected),
      'same_initialization_sha256':common['initialization']['sha256'],'same_training_manifest_sha256':common['manifest']['sha256'],
      'same_runner_sha256':runner_sha,'train_rows':771,'validation_rows_excluded':191,
      'sampler_dry_audit_epochs':300,'actual_training_anchors_consumed':0,
      'dry_index_draws_per_arm':{arm:sum(c.values()) for arm,c in totals.items()},
      'control_visits_per_sample':[min(totals['C'].values()),max(totals['C'].values())],
      'model_loaded':False,'model_inference':False,'training_performed':False,'image_pixels_decoded':0,
      'locked_test_used':False,'CO2Wounds_used':False,'output_dirs_absent':True,'tests_required_before_final_freeze':True}
    for suffix,value in [('protocol',protocol),('control_config',control),('experimental_config',experimental),('telemetry_schema',schema),('dry_audit',dry)]:
        d0.write(dest/f'{PREFIX}_{suffix}.json',value)
    return {'status':'PREPARED_NOT_TRAINING','protected_files_unchanged':len(protected),'config_diff':differences}


def freeze(root,verification):
    root=Path(root).resolve();p=root/'experiments/protocols';require_output_absent(root)
    if (p/f'{PREFIX}_freeze.json').exists():raise d0.D0Error('V2_FROZEN_REFUSE_OVERWRITE')
    dry=d0.read(p/f'{PREFIX}_dry_audit.json')
    if any(d0.sha(path)!=h for path,h in dry['protected_before'].items()):raise d0.D0Error('PROTECTED_INPUT_CHANGED')
    if not verification.get('all_required_tests_passed'):raise d0.D0Error('TEST_GATE_NOT_PASSED')
    artifacts={f'experiments/protocols/{PREFIX}_{s}.json':d0.sha(p/f'{PREFIX}_{s}.json') for s in SUFFIXES}
    for path in [RUNNER,Path('experiments/phase_d01.py'),Path('tests/test_phase_d01.py'),REPORT]:artifacts[path.as_posix()]=d0.sha(root/path)
    result={'status':'FROZEN_BEFORE_TRAINING','PHASE_D01_STATUS':'COMPLETE','READY_FOR_PHASE_D1_PAIRED_SINGLE_SEED_TRAINING':'YES',
      'artifacts_sha256':artifacts,'verification':verification,'D1_authorized':False,
      'MODEL_LOADED':False,'TRAINING_PERFORMED':False,'MODEL_INFERENCE':False,'LOCKED_TEST_USED':False,'CO2Wounds_used':False,
      'output_dirs_absent':True,'STOP_AFTER_D01':True}
    d0.write(p/f'{PREFIX}_freeze.json',result)
    return result


if __name__=='__main__':
    print(json.dumps(build(Path(__file__).resolve().parents[1]),indent=2))
