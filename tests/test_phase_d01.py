from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import json
import numpy as np
import pytest
from experiments import paired_sampling_runner as r
from experiments import phase_d0 as d0


def rows():
    return [{'sample_id':f'image{i}.png','image_path':f'train/image{i}.png','source':'FUSeg','split':'train','role':'wound_finetuning',
      'sampling_category':c,'sampling_weight':w,'GT_count':1,'multi_GT':False,
      'instance_geometry':[{'class_id':0,'bbox_area_ratio':area,'size_group':'small' if area<.01 else 'medium'}]}
      for i,(c,w,area) in enumerate([('A',2,.001),('B',1.5,.005),('C',1,.02)])]


def config(arm='C',epochs=2):
    return {'arm_label':arm,'experiment_id':'synthetic_'+arm,'output_path':r.OUTPUTS[arm],
      'initialization':{'path':'isic/best.pt','sha256':'same'},'dataset':{'sha256':'same'},'architecture':'YOLO11m-seg',
      'optimizer':{'name':'AdamW','lr':.0005},'loss':{'box':7.5},'augmentation':{'mosaic':.1},
      'training_args':{'batch':2,'epochs':epochs,'amp':True,'seed':42,'workers':2},
      'budget':{'batches_per_epoch':2,'scheduled_optimizer_calls_per_epoch':[2]*epochs},
      'evaluation':{'confidence':.1},'runner':{'sha256':'same'},
      'sampling':{'mode':'uniform_shuffled' if arm=='C' else 'small_aware_weighted',
         'weights':{'A':1,'B':1,'C':1} if arm=='C' else d0.WEIGHTS,'replacement':arm=='S','seed':42}}


@pytest.mark.parametrize('field',['initialization','dataset','architecture','optimizer','loss','augmentation','training_args','budget','evaluation','runner'],
 ids=['same_initialization','same_dataset','same_architecture','same_optimizer','same_loss','same_augmentation','same_training_schedule','same_budget','same_evaluation','same_runner'])
def test_v2_same_field(field):
    c,s=config('C'),config('S');r.config_diff(c,s)
    s[field]='modified'
    with pytest.raises(r.PairError,match='FAIL_PAIRED_CONFIG_DIFF'):r.config_diff(c,s)


def test_v2_only_sampling_semantics_differ():
    changes=r.config_diff(config('C'),config('S'))
    assert 'sampling.mode' in changes and 'sampling.replacement' in changes
    c,s=config('C'),config('S');c['nullable']=None
    with pytest.raises(r.PairError):r.config_diff(c,s)


class FakeOptimizer:
    def __init__(self):self.hooks=[];self.param_groups=[{'lr':.0005}];self.defaults={'fused':None};self.calls=0
    def register_step_post_hook(self,hook):self.hooks.append(hook);return SimpleNamespace(remove=lambda:None)
    def step(self):
        self.calls+=1
        for hook in self.hooks:hook(self,(),{})
        return None  # Real AdamW normally returns None even when it updates.


class FakeScaler:
    def __init__(self,skip=False):self.skip=skip;self.scale=1024
    def get_scale(self):return self.scale
    def step(self,opt):
        if not self.skip:return opt.step()
    def update(self):pass  # Deliberately unchanged: classification must not infer from scale.


def fixture(tmp_path,skip=False,arm='C'):
    log=[];t=r.RunTelemetry(tmp_path,rows(),config(arm),lambda kind,item:log.append((kind,item)))
    opt=FakeOptimizer();scaler=FakeScaler(skip)
    trainer=SimpleNamespace(optimizer=opt,scaler=scaler,accumulate=1,scheduler=SimpleNamespace(state_dict=lambda:{'last_epoch':0,'base_lrs':[.0005]}))
    t.bind_optimizer(opt);t.start_epoch(0);t.start_batch()
    return t,trainer,log


def do_step(t,trainer):
    def original():trainer.scaler.step(trainer.optimizer);trainer.scaler.update()
    return t.opportunity(trainer,original)


def test_v2_records_optimizer_attempts(tmp_path):
    t,tr,log=fixture(tmp_path);do_step(t,tr)
    assert t.scheduled==1 and log[-1][1]['optimizer_call_attempted'] is True
    assert log[-1][1]['global_batch']==0


def test_v2_records_amp_applied_step(tmp_path):
    t,tr,log=fixture(tmp_path);assert do_step(t,tr) is None
    assert t.applied==1 and t.skipped==0 and log[-1][1]['optimizer_step_applied'] is True


def test_v2_records_amp_skipped_step(tmp_path):
    t,tr,log=fixture(tmp_path,skip=True);do_step(t,tr)
    assert t.applied==0 and t.skipped==1
    assert log[-1][1]['skip_reason_if_known']=='AMP_STEP_SKIPPED'


def test_v2_records_scaler_state(tmp_path):
    t,tr,log=fixture(tmp_path,skip=True);do_step(t,tr);item=log[-1][1]
    assert item['grad_scaler_scale_before']==item['grad_scaler_scale_after']==1024
    assert item['learning_rate']==[.0005] and item['scheduler_state']['last_epoch']==0


def test_v2_does_not_call_execution_exception_an_amp_skip(tmp_path):
    t,tr,log=fixture(tmp_path)
    def failed():raise RuntimeError('simulated failure')
    with pytest.raises(RuntimeError):t.opportunity(tr,failed)
    assert t.unknown==1 and t.skipped==0
    assert log[-1][1]['optimizer_step_skipped'] is None


def test_v2_rejects_fused_internal_amp(tmp_path):
    t=r.RunTelemetry(tmp_path,rows(),config(),lambda *args:None);opt=FakeOptimizer();opt.defaults['fused']=True
    with pytest.raises(r.PairError,match='FUSED_INTERNAL_AMP'):t.bind_optimizer(opt)


def test_v2_rejects_uninstrumented_step(tmp_path):
    t,tr,log=fixture(tmp_path)
    with pytest.raises(r.PairError,match='UNINSTRUMENTED'):tr.optimizer.step()


@pytest.mark.parametrize('arm',['C','S'])
def test_v2_records_anchor_sequence(tmp_path,arm):
    log=[];data=rows();t=r.RunTelemetry(tmp_path,data,config(arm),lambda k,v:log.append((k,v)))
    paths=[tmp_path/x['image_path'] for x in data];mapping=t.bind_dataset(paths[::-1]);t.start_epoch(0);t.start_batch()
    plan=r.plan_indices(data,arm,0);t.anchors([paths[i] for i in plan[:2]])
    anchors=log[-1][1]['anchors']
    assert [a['sample_id'] for a in anchors]==[data[i]['sample_id'] for i in plan[:2]]
    assert [a['dataset_index'] for a in anchors]==[mapping[i] for i in plan[:2]]
    assert ('weight' in anchors[0])==(arm=='S')


@pytest.mark.parametrize('kind',['val','test','CO2Wounds-V2'])
def test_v2_validation_locked_co2_never_enters_sampler(kind):
    data=rows();data[0]['source' if kind.startswith('CO2') else 'split']=kind
    with pytest.raises(d0.D0Error):r.plan_indices(data,'S',0)


def test_v2_output_dirs_absent(tmp_path):
    r.require_output_absent(tmp_path)
    (tmp_path/r.OUTPUTS['C']).mkdir(parents=True)
    with pytest.raises(r.PairError):r.require_output_absent(tmp_path)


def test_v2_locks_refuse_missing_authorization_or_retry(tmp_path):
    (tmp_path/'experiments/results').mkdir(parents=True)
    with pytest.raises(r.PairError):r.claim_arm_output(tmp_path,config())
    out=r.claim_arm_output(tmp_path,config(),separately_authorized=True)
    assert (out/'execution.lock').exists()
    with pytest.raises(FileExistsError):r.claim_arm_output(tmp_path,config(),separately_authorized=True)


def test_v2_sampler_prefetch_reset_restarts_exact_epoch():
    data=rows();sampler=r.EpochPlanSampler(data,'S',[2,0,1])
    plans=[list(sampler) for _ in range(5)]  # simulated prefetch of later plans
    sampler.rewind_for_reset(2)
    assert list(sampler)==plans[2]
    for epoch in range(10):assert len(set(r.plan_indices(data,'C',epoch)))==3


class FakeBase:
    def __init__(self,root,data,epochs):
        self.files=[str(root/x['image_path']) for x in data];self.epochs=epochs
        self.args=SimpleNamespace(workers=2,close_mosaic=1)
        self.optimizer=FakeOptimizer();self.scaler=FakeScaler();self.scheduler=SimpleNamespace(state_dict=lambda:{'last_epoch':self.epoch})
        self.accumulate=1;self.original_step_calls=0;self.original_preprocess_calls=0
    def build_dataset(self,*args):return SimpleNamespace(im_files=self.files)
    def _setup_train(self,world_size):pass
    def run_callbacks(self,event):pass
    def preprocess_batch(self,batch):self.original_preprocess_calls+=1;return batch
    def optimizer_step(self):
        self.original_step_calls+=1;self.scaler.step(self.optimizer);self.scaler.update()


@pytest.mark.parametrize('arm',['C','S'])
def test_v2_mocked_runner_consumes_exact_plans_and_observes_steps(tmp_path,arm):
    data=rows();log=[];t=r.RunTelemetry(tmp_path,data,config(arm,3),lambda k,v:log.append((k,v)))
    def builder(dataset,sampler,batch,workers):return SimpleNamespace(dataset=dataset,sampler=sampler)
    cls=r.make_trainer_adapter(FakeBase,t,builder);trainer=cls(tmp_path,data,3)
    loader=trainer.get_dataloader(tmp_path/'train',batch_size=2,rank=-1)
    trainer._setup_train(1)
    for epoch in range(3):
        trainer.epoch=epoch;trainer.run_callbacks('on_train_epoch_start')
        ids=list(loader.sampler)
        for start in range(0,3,2):
            trainer.run_callbacks('on_train_batch_start')
            trainer.preprocess_batch({'im_file':[trainer.files[i] for i in ids[start:start+2]]})
            trainer.optimizer_step()
        trainer.run_callbacks('on_train_epoch_end')
    assert t.completed_epochs==3 and t.applied==6 and t.skipped==0
    assert trainer.original_step_calls==trainer.original_preprocess_calls==6
    assert t.summary()['anchor_draws']==9


def test_v2_adapter_rejects_forbidden_path_before_build(tmp_path):
    t=r.RunTelemetry(tmp_path,rows(),config(),lambda *args:None)
    cls=r.make_trainer_adapter(FakeBase,t);trainer=cls(tmp_path,rows(),2)
    with pytest.raises(r.PairError,match='FORBIDDEN_TRAINING'):trainer.get_dataloader(tmp_path/'test')


def test_v2_no_model_load_training_or_inference_in_d01():
    code="import sys; import experiments.phase_d01; import experiments.paired_sampling_runner; assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,cwd=Path(r.__file__).resolve().parents[1])


def complete(skips=0):
    return {'completed_epochs':300,'scheduled_optimizer_calls':3741,'applied_optimizer_updates':3741-skips,'skipped_optimizer_updates':skips,'unknown_optimizer_opportunities':0}


def test_v2_skips_can_differ_without_faking_realized_equality():
    c,s=complete(),complete(2)
    assert r.pair_validity(c,s)=='VALID_SCHEDULED_BUDGET'
    flags=r.amp_sensitivity(c,s)
    assert flags['AMP_UPDATE_COUNT_IMBALANCE_OBSERVED'] is True
    assert flags['identical_realized_updates_claim_allowed'] is False


def test_v2_early_stop_invalidates_pair():
    c,s=complete(),complete();s['completed_epochs']=299
    assert r.pair_validity(c,s)=='NOT_VALID'


def metrics(small=100,medium=86,large=14,fp=36,crop=167):
    tp=small+medium+large
    return {'tp':tp,'fp':fp,'fn':241-tp,'small_tp':small,'medium_tp':medium,'large_tp':large,
      'small_support':137,'medium_support':90,'large_support':14,'positive_images':186,'crop_complete_count':crop}


def test_v2_paired_gate_uses_fresh_control_not_historical_109():
    outcome=r.paired_advancement(metrics(100),metrics(101),'VALID_SCHEDULED_BUDGET')
    assert outcome['status']=='PASS_SMALL_SAMPLING_RESEARCH_ADVANCEMENT_GATE'


@pytest.mark.parametrize('candidate',[metrics(100),metrics(101,fp=100),metrics(101,crop=165),metrics(101,medium=84),metrics(101,large=13)])
def test_v2_each_gate_safety_failure_preserved(candidate):
    assert r.paired_advancement(metrics(),candidate,'VALID_SCHEDULED_BUDGET')['status'].startswith('FAIL')


def test_v2_corrupt_negative_telemetry_rejected():
    bad=complete(-1)
    assert r.pair_validity(complete(),bad)=='NOT_VALID'


@pytest.mark.parametrize('field,value',[('fp',-1),('crop_complete_count',187),('small_tp',138),('tp',True)])
def test_v2_corrupt_metric_counts_rejected(field,value):
    bad=metrics();bad[field]=value
    with pytest.raises(r.PairError):r.paired_advancement(metrics(),bad,'VALID_SCHEDULED_BUDGET')


def test_v2_jsonl_is_exclusive_and_flushed(tmp_path):
    with pytest.raises(r.PairError):r.JsonlSink(tmp_path)
    (tmp_path/'execution.lock').write_text('synthetic lock',encoding='utf8')
    sink=r.JsonlSink(tmp_path)
    try:
        sink('optimizer',{'optimizer_step_applied':True})
        assert json.loads((tmp_path/'optimizer.jsonl').read_text())['optimizer_step_applied'] is True
        with pytest.raises(FileExistsError):r.JsonlSink(tmp_path)
    finally:sink.close()


def test_v2_checkpoint_runtime_summary_uses_file_hashes_not_model_load(tmp_path):
    weights=tmp_path/'training/weights';weights.mkdir(parents=True)
    # Synthetic bytes, deliberately not a model serialization.
    (weights/'best.pt').write_bytes(b'synthetic-best')
    (weights/'last.pt').write_bytes(b'synthetic-last')
    t=SimpleNamespace(summary=lambda:complete(2))
    runtime={name:'synthetic' for name in ['GPU','CUDA','torch','ultralytics','python']}
    record=r.finalize_arm(tmp_path,t,config(),runtime)
    assert record['best_checkpoint_sha256']==r.sha(weights/'best.pt')
    assert record['last_checkpoint_sha256']==r.sha(weights/'last.pt')
    assert record['skipped_optimizer_updates']==2
    assert record['runtime']==runtime
    with pytest.raises(FileExistsError):r.finalize_arm(tmp_path,t,config(),runtime)
