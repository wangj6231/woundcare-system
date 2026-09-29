"""Synthetic D1 orchestration/integrity tests. No dataset/model construction."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from experiments import phase_d1 as d


def completed(skips=0):
    return {'completed_epochs':300,'scheduled_optimizer_calls':3741,'applied_optimizer_updates':3741-skips,
      'skipped_optimizer_updates':skips,'unknown_optimizer_opportunities':0,'exposure':{'A':1,'small_GT':1}}


def setup_fake(monkeypatch,tmp_path,fail_c=False,early_c=False):
    (tmp_path/'experiments/results').mkdir(parents=True);(tmp_path/'docs').mkdir()
    cfg={'same':'yes'};sequence=[]
    monkeypatch.setattr(d,'validate_freeze',lambda root:(cfg,cfg,{}))
    monkeypatch.setattr(d,'verify_data',lambda *args:[])
    monkeypatch.setattr(d,'verify_validation',lambda *args:None)
    monkeypatch.setattr(d,'runtime',lambda *args:{'GPU':'mock'})
    monkeypatch.setattr(d,'snapshot_extra',lambda *args:{})
    def run(args,**kwargs):
        if '--arm' not in args:sequence.append('postflight');return SimpleNamespace(returncode=0)
        arm=args[args.index('--arm')+1];sequence.append('train_'+arm)
        if arm=='C' and fail_c:return SimpleNamespace(returncode=1)
        out=tmp_path/d.paired.OUTPUTS[arm];out.mkdir()
        doc=completed()
        if arm=='S':doc['exposure']={'A':2,'small_GT':2}
        if arm=='C' and early_c:doc['completed_epochs']=299
        d.write(out/'training_completion.json',doc)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(d.subprocess,'run',run)
    def evaluate(root,arm):sequence.append('eval_'+arm);return {'simulated_performance':'poor' if arm=='C' else 'good'}
    monkeypatch.setattr(d,'evaluate_arm',evaluate)
    monkeypatch.setattr(d,'verify_saved_arm',lambda *args:{'status':'PASS'})
    monkeypatch.setattr(d,'complete_pair',lambda *args:({'SMALL_SAMPLING_RESEARCH_GATE':'FAIL'},'Synthetic report'))
    return sequence


def test_d1_order_is_both_training_then_both_evaluation(monkeypatch,tmp_path):
    sequence=setup_fake(monkeypatch,tmp_path);d.execute(tmp_path)
    assert sequence==['train_C','train_S','eval_C','eval_S','postflight']
    assert d.d0.read(tmp_path/d.PAIR/'status.json')['stage']=='COMPLETE'


@pytest.mark.parametrize('failure',['crash','early'])
def test_d1_control_failure_stops_without_s_or_evaluation(monkeypatch,tmp_path,failure):
    sequence=setup_fake(monkeypatch,tmp_path,fail_c=failure=='crash',early_c=failure=='early')
    with pytest.raises(d.paired.PairError):d.execute(tmp_path)
    assert sequence==['train_C']
    assert (tmp_path/d.PAIR/'interruption.json').exists()
    assert not (tmp_path/d.paired.OUTPUTS['S']).exists()


def test_d1_existing_output_is_not_deleted(monkeypatch,tmp_path):
    setup_fake(monkeypatch,tmp_path)
    prior=tmp_path/d.paired.OUTPUTS['C'];prior.mkdir();(prior/'evidence').write_text('preserve')
    with pytest.raises(d.paired.PairError):d.execute(tmp_path)
    assert (prior/'evidence').read_text()=='preserve'


def test_d1_frozen_index_mutation_rejected_before_framework_import(tmp_path):
    folder=tmp_path/'experiments/protocols';folder.mkdir(parents=True)
    (folder/f'{d.PREFIX}_freeze.json').write_text('{}')
    with pytest.raises(d.paired.PairError,match='FAIL_PROTOCOL_MUTATED'):d.validate_freeze(tmp_path)


def test_d1_runtime_version_mismatch_precedes_model_loading(monkeypatch):
    monkeypatch.setattr(d.metadata,'version',lambda name:'unexpected')
    with pytest.raises(d.paired.PairError,match='FAIL_SOFTWARE_VERSION'):d.runtime({'software':{'torch':'expected'}})


def test_d1_import_has_no_model_or_inference_side_effect():
    subprocess.run([sys.executable,'-c',"import sys; import experiments.phase_d1; assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"],check=True,cwd=d.ROOT)


def data771():
    return [{'sample_id':f'i{i}','source':'FUSeg','split':'train','role':'wound_finetuning','sampling_category':'A',
      'sampling_weight':2.0,'GT_count':1,'multi_GT':False,'instance_geometry':[{'bbox_area_ratio':.001,'size_group':'small'}]} for i in range(771)]


def fixture_telemetry(out,arm):
    rows=data771();plan=d.paired.plan_indices(rows,arm,0);p=d.paired.probabilities(rows)
    cfg={'arm_label':arm,'budget':{'scheduled_optimizer_calls_per_epoch':[1]}}
    with (out/'anchors.jsonl').open('w') as f:
        for b,start in enumerate(range(0,771,4)):
            anchors=[]
            for pos in range(start,min(start+4,771)):
                i=int(plan[pos]);a={'sample_id':rows[i]['sample_id'],'dataset_index':i,'anchor_position':pos,'sampling_category':'A'}
                if arm=='S':a.update(weight=2.,draw_probability=float(p[i]))
                anchors.append(a)
            f.write(json.dumps({'epoch':0,'batch_index':b,'anchors':anchors})+'\n')
    (out/'epochs.jsonl').write_text(json.dumps({'epoch':0})+'\n')
    op={'epoch':0,'batch_index':0,'global_batch':0,'telemetry_status':'COMPLETE','optimizer_call_attempted':True,'optimizer_step_applied':True,'optimizer_step_skipped':False}
    (out/'optimizer.jsonl').write_text(json.dumps(op)+'\n')
    summary={'completed_epochs':1,'scheduled_optimizer_calls':1,'applied_optimizer_updates':1,'skipped_optimizer_updates':0,
      'anchor_draws':771,'unique_anchors_seen':len(set(plan.tolist())),
      'exposure':{'A':771,'B':0,'C':0,'small_GT':771,'very_small_GT':771,'medium_GT':0,'large_GT':0,'negative_images':0,'multi_GT_anchors':0}}
    return rows,cfg,summary


@pytest.mark.parametrize('arm',['C','S'])
def test_d1_recounts_actual_consumed_telemetry(tmp_path,arm):
    rows,cfg,summary=fixture_telemetry(tmp_path,arm)
    audit=d.audit_telemetry(tmp_path,rows,cfg,summary)
    assert audit['status']=='PASS' and audit['total_batches']==193 and audit['anchor_draws']==771


def test_d1_corrupt_applied_total_rejected(tmp_path):
    rows,cfg,summary=fixture_telemetry(tmp_path,'C');summary['applied_optimizer_updates']=2
    with pytest.raises(d.paired.PairError,match='FAIL_OPTIMIZER_TOTALS'):d.audit_telemetry(tmp_path,rows,cfg,summary)


def test_d1_corrupt_anchor_rejected(tmp_path):
    rows,cfg,summary=fixture_telemetry(tmp_path,'C');path=tmp_path/'anchors.jsonl'
    text=path.read_text();path.write_text(text.replace('"sample_id": "i','"sample_id": "external',1))
    with pytest.raises(d.paired.PairError,match='FAIL_ACTUAL_ANCHOR_SEQUENCE'):d.audit_telemetry(tmp_path,rows,cfg,summary)


def test_d1_corrupt_exposure_rejected(tmp_path):
    rows,cfg,summary=fixture_telemetry(tmp_path,'C');summary['exposure']['small_GT']=1
    with pytest.raises(d.paired.PairError,match='FAIL_EXPOSURE_SUMMARY'):d.audit_telemetry(tmp_path,rows,cfg,summary)
