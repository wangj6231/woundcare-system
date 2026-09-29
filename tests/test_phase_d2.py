"""Synthetic D2 design, seed-equivalence, paired gate and namespace tests."""
from copy import deepcopy
from fractions import Fraction
import subprocess
import sys
import pytest
from experiments import phase_d2 as d
from experiments import paired_sampling_multiseed as seeded


def rows():
    return [{'sample_id':str(i),'source':'FUSeg','split':'train','role':'wound_finetuning','sampling_category':category,
      'sampling_weight':weight,'instance_geometry':[{'bbox_area_ratio':area,'size_group':'small' if area<.01 else 'medium'}]}
      for i,(category,weight,area) in enumerate([('A',2.,.001),('B',1.5,.005),('C',1.,.02)]*7)]


@pytest.mark.parametrize('arm',['C','S'])
def test_seed42_exact_algorithm_equivalence(arm):
    data=rows()
    for e in range(300):assert (seeded.indices(data,arm,e,42)==d.d1.paired.plan_indices(data,arm,e)).all()


@pytest.mark.parametrize('seed',[123,3407,2026,999])
def test_new_seed_is_deterministic_and_uniform_control_is_permutation(seed):
    a=seeded.indices(rows(),'C',0,seed);b=seeded.indices(rows(),'C',0,seed)
    assert (a==b).all() and len(set(a))==len(rows())
    assert not (a==seeded.indices(rows(),'C',0,42)).all()
    assert len(seeded.indices(rows(),'S',0,seed))==len(rows())


def test_private_namespace_does_not_mutate_frozen_module():
    original=d.d1.paired.plan_indices;paths=dict(d.d1.paired.OUTPUTS)
    module=seeded.make_runtime(d.ROOT,123)
    assert d.d1.paired.plan_indices is original and d.d1.paired.OUTPUTS==paths
    assert module.RunTelemetry.start_epoch.__code__.co_code==d.d1.paired.RunTelemetry.start_epoch.__code__.co_code
    assert module.make_trainer_adapter.__code__.co_code==d.d1.paired.make_trainer_adapter.__code__.co_code
    assert module.RunTelemetry.start_epoch.__globals__['plan_indices'] is module.plan_indices
    assert (module.plan_indices(rows(),'S',3)==seeded.indices(rows(),'S',3,123)).all()


def test_seed42_not_executable():
    with pytest.raises(d.d1.paired.PairError,match='SEED42_RETRAIN_FORBIDDEN'):d.configs(d.ROOT,42)
    with pytest.raises(d.d1.paired.PairError,match='SEED42_RETRAIN_FORBIDDEN'):d.delegated_driver(d.ROOT,42)


@pytest.mark.parametrize('seed',[0,124,-1,42.0])
def test_unregistered_seed_rejected(seed):
    with pytest.raises(ValueError):seeded.indices(rows(),'C',0,seed)


def metrics(small=104,large=14,fp=30):
    tp=small+85+large
    return {'tp':tp,'fp':fp,'fn':241-tp,'precision':tp/(tp+fp),'recall':tp/241,'f1':2*tp/(2*tp+fp+241-tp),
      'positive_images':186,'crop_complete95_images':165,'positive_without_roi':4,
      'size_recall':{s:{'gt':support,'matched':hit,'recall':hit/support} for s,support,hit in [('small',137,small),('medium',90,85),('large',14,large)]}}


def pair(seed,delta=4,large_delta=0):
    diagnostic={'very_small':{'TP':27},'single_GT':{'TP':139,'GT':151},'multi_GT':{'TP':64,'GT':90}}
    training={a:{'completed_epochs':300,'scheduled_optimizer_calls':3741,'applied_optimizer_updates':3731,
      'skipped_optimizer_updates':10,'unknown_optimizer_opportunities':0,
      'exposure':{k:n for k in ['A','small_GT','very_small_GT']}} for a,n in [('C',1),('S',2)]}
    ev={'C':{'candidate':metrics(),'diagnostics':deepcopy(diagnostic)},
        'S':{'candidate':metrics(104+delta,14+large_delta,27),'diagnostics':deepcopy(diagnostic)}}
    return {'seed':seed,'valid':True,'intervention_delivered':True,'training':training,'evaluations':ev}


def five():return [pair(s) for s in d.ALL_SEEDS]


def test_exact_minus_one_fails_seed_gate_not_catastrophic():
    pairs=five();pairs[1]=pair(123,4,-1)
    c,s=[d.d1.gate_counts(pairs[1]['evaluations'][a]['candidate']) for a in ['C','S']]
    single=d.d1.paired.paired_advancement(c,s,'VALID_SCHEDULED_BUDGET')
    assert single['checks']['large'] is False
    overall=d.confirmation(pairs)
    assert overall['MULTISEED_SMALL_SAMPLING_CONFIRMATION']=='PASS'
    assert overall['ordinary_large_minus_one_seeds']==[123] and overall['catastrophic_large_seeds']==[]


def test_minus_two_is_catastrophic_overall_fail():
    pairs=five();pairs[1]=pair(123,4,-2)
    result=d.confirmation(pairs)
    assert result['MULTISEED_SMALL_SAMPLING_CONFIRMATION']=='FAIL' and result['catastrophic_large_seeds']==[123]


def test_incomplete_preserves_five_seed_denominator():
    result=d.confirmation(five()[:4])
    assert result['MULTISEED_SMALL_SAMPLING_CONFIRMATION']=='INCOMPLETE'
    assert result['required_pairs']==5 and not result['checks']['direction_4_of_5']


def test_catastrophic_failure_and_incompleteness_both_reported():
    result=d.confirmation([pair(42),pair(123,4,-2)])
    assert result['MULTISEED_SMALL_SAMPLING_CONFIRMATION']=='FAIL'
    assert result['MULTISEED_CONFIRMATION_INCOMPLETE'] is True


def test_four_positive_sufficient_not_five_individual_passes():
    pairs=five();pairs[-1]=pair(999,0)
    result=d.confirmation(pairs)
    assert result['MULTISEED_SMALL_SAMPLING_CONFIRMATION']=='PASS' and result['small_paired_effect']['positive']==4


def test_three_positive_cannot_pass_despite_positive_mean():
    pairs=five();pairs[-1]=pair(999,0);pairs[-2]=pair(2026,0)
    result=d.confirmation(pairs)
    assert result['MULTISEED_SMALL_SAMPLING_CONFIRMATION']=='FAIL' and result['mean_paired_effects']['small_recall']>0


def test_sample_sd_is_paired_sample_sd():
    pairs=[pair(s,i) for s,i in zip(d.ALL_SEEDS,[1,2,3,4,5])]
    result=d.confirmation(pairs)
    assert result['exact_mean_paired_effects']['small_recall']=='3/137'
    assert result['small_paired_effect']['sample_SD']==pytest.approx((2.5**.5)/137)


def test_duplicate_seed_cannot_inflate_n():
    with pytest.raises(d.d1.paired.PairError):d.confirmation([pair(42)]*5)


def test_invalid_pair_not_in_confirmation():
    pairs=five();pairs[-1]['valid']=False
    assert d.confirmation(pairs)['MULTISEED_SMALL_SAMPLING_CONFIRMATION']=='INCOMPLETE'


def test_falsely_valid_budget_rejected():
    pairs=five();pairs[1]['training']['S']['completed_epochs']=299
    with pytest.raises(d.d1.paired.PairError,match='FAIL_SUMMARY_PAIR_VALIDITY'):d.confirmation(pairs)


def test_only_seed_changes_allowed():
    base={'training_args':{'seed':42,'lr':.0005},'sampling':{'seed':42,'seed_policy':'old'},'runner':{},'output_path':'old','experiment_id':'old'}
    changed=deepcopy(base);changed['training_args']['seed']=123;changed['sampling']['seed']=123
    assert d.config_seed_only(base,changed,123)
    changed['training_args']['lr']=.001
    with pytest.raises(d.d1.paired.PairError,match='BEYOND_SEED'):d.config_seed_only(base,changed,123)


def test_no_models_imported_by_d2():
    subprocess.run([sys.executable,'-c',"import sys; import experiments.phase_d2; assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"],check=True,cwd=d.ROOT)


def test_mean_precision_safety_not_bypassed_by_small_improvement():
    pairs=five()
    for p in pairs:p['evaluations']['S']['candidate']=metrics(108,14,150)
    result=d.confirmation(pairs)
    assert result['MULTISEED_SMALL_SAMPLING_CONFIRMATION']=='FAIL'
    assert not result['checks']['mean_precision_safety']


def test_positive_count_does_not_replace_positive_mean():
    pairs=[pair(seed,delta) for seed,delta in zip(d.ALL_SEEDS,[1,1,1,1,-10])]
    result=d.confirmation(pairs)
    assert result['checks']['direction_4_of_5'] and not result['checks']['mean_small_positive']
    assert result['MULTISEED_SMALL_SAMPLING_CONFIRMATION']=='FAIL'


def test_output_guard_is_non_destructive(tmp_path):
    prior=tmp_path/seeded.outputs(123)['C'];prior.mkdir(parents=True)
    (prior/'evidence').write_text('keep')
    with pytest.raises(d.d1.paired.PairError,match='FAIL_OUTPUT_ALREADY_EXISTS'):d.output_guard(tmp_path)
    assert (prior/'evidence').read_text()=='keep'


@pytest.mark.parametrize('seed',d.NEW_SEEDS)
@pytest.mark.parametrize('arm',['C','S'])
def test_seeded_sampler_delivers_exact_plan_to_unchanged_telemetry(tmp_path,seed,arm):
    data=rows()
    for r in data:r['image_path']='train/'+r['sample_id']+'.png'
    module=seeded.make_runtime(d.ROOT,seed);events=[]
    cfg={'arm_label':arm,'training_args':{'batch':4},'sampling':{'mode':arm}}
    telemetry=module.RunTelemetry(tmp_path,data,cfg,lambda kind,record:events.append((kind,record)))
    files=[tmp_path/r['image_path'] for r in reversed(data)]
    mapping=telemetry.bind_dataset(files);sampler=module.EpochPlanSampler(data,arm,mapping)
    telemetry.start_epoch(0);delivered=list(sampler)
    for offset in range(0,len(delivered),4):
        telemetry.start_batch();telemetry.anchors([files[i] for i in delivered[offset:offset+4]])
    observed=[x['sample_id'] for kind,event in events for x in event['anchors']]
    expected=[data[int(i)]['sample_id'] for i in seeded.indices(data,arm,0,seed)]
    assert observed==expected and telemetry.position==len(data)


def test_supervisor_control_flow_without_training(tmp_path,monkeypatch):
    # Dependency-injected orchestration proof only. No subprocess or model runs.
    from types import SimpleNamespace
    (tmp_path/d.MASTER).parent.mkdir(parents=True)
    monkeypatch.setattr(d,'sha',lambda p:'frozen')
    monkeypatch.setattr(d,'verify',lambda r:None)
    monkeypatch.setattr(d.d1,'validate_freeze',lambda r:({}, {}, {}))
    monkeypatch.setattr(d.d1,'verify_data',lambda *a:None)
    monkeypatch.setattr(d.d1,'verify_validation',lambda *a:None)
    monkeypatch.setattr(d.d1,'runtime',lambda c:{'synthetic':True})
    monkeypatch.setattr(d,'seed42_result',lambda r:{'seed':42})
    attempted=[]
    def run(r,seed,token,environment):
        attempted.append(seed);return {'seed':seed}
    monkeypatch.setattr(d,'run_seed',run)
    monkeypatch.setattr(d.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0))
    def report(r,pairs,aborted):
        assert aborted is None and [p['seed'] for p in pairs]==list(d.ALL_SEEDS)
        return {'PHASE_D2_STATUS':'COMPLETE','MULTISEED_SMALL_SAMPLING_CONFIRMATION':'PASS','VALID_PAIRED_SEEDS':5}
    monkeypatch.setattr(d,'save_report',report)
    d.execute(tmp_path,'frozen')
    assert attempted==list(d.NEW_SEEDS)
