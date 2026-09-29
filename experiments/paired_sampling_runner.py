"""Shared paired-arm sampler and observational telemetry. No model imports.

D0.1 uses dependency-injected mock trainers only. A separately authorized D1
driver may bind the same adapter to the pinned Ultralytics SegmentationTrainer.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import time
import numpy as np
from experiments.phase_d0 import epoch_indices, probabilities, sha

OUTPUTS={'C':'experiments/results/d_seg_small_sampling_v2_control',
         'S':'experiments/results/d_seg_small_sampling_v2_experimental'}
ALLOWED_DIFF={'sampling.mode','sampling.weights','sampling.replacement','experiment_id','output_path','arm_label'}


class PairError(RuntimeError):
    pass


def leaves(value,prefix=''):
    if isinstance(value,dict):
        if not value:return {prefix:{}}
        result={}
        for key,item in value.items():result.update(leaves(item,f'{prefix}.{key}' if prefix else key))
        return result
    return {prefix:value}


def config_diff(control,experimental):
    a,b=leaves(control),leaves(experimental)
    missing=object()
    changed=sorted(k for k in a.keys()|b.keys() if a.get(k,missing)!=b.get(k,missing))
    forbidden=[k for k in changed if not any(k==allowed or k.startswith(allowed+'.') for allowed in ALLOWED_DIFF)]
    if forbidden:raise PairError('FAIL_PAIRED_CONFIG_DIFF: '+','.join(forbidden))
    return changed


def require_output_absent(root):
    for path in OUTPUTS.values():
        if (root/path).exists():raise PairError('PAIRED_OUTPUT_ALREADY_EXISTS')


def claim_arm_output(root,config,*,separately_authorized=False):
    if not separately_authorized:raise PairError('D1_SEPARATE_AUTHORIZATION_REQUIRED')
    if config['arm_label'] not in OUTPUTS or config['output_path']!=OUTPUTS[config['arm_label']]:raise PairError('UNREGISTERED_OUTPUT')
    root=Path(root).resolve();out=root/config['output_path']
    if out.parent.resolve()!=(root/'experiments/results').resolve():raise PairError('OUTPUT_ESCAPE')
    # Atomic mkdir + exclusive lock. A crash between them leaves a blocked directory.
    out.mkdir(exist_ok=False)
    with (out/'execution.lock').open('x',encoding='utf8') as f:
        json.dump({'arm':config['arm_label'],'runner_sha256':sha(Path(__file__)),
                   'resume_allowed':False,'overwrite_allowed':False},f)
    return out


def plan_indices(rows,arm,epoch):
    probabilities(rows)  # Recheck frozen weights/source roles even for control.
    if arm not in OUTPUTS or epoch<0:raise PairError('INVALID_ARM_OR_EPOCH')
    if arm=='S':return epoch_indices(rows,epoch,42)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([42,epoch]))).permutation(len(rows))


class EpochPlanSampler:
    """Finite epoch plans consumed by InfiniteDataLoader's repeat sampler.

    Prefetched indices are not counted as actual exposure. On close_mosaic,
    rewind the planning cursor BEFORE the original loader reset; the consumer
    ledger checks the exact delivered sequence at each training batch.
    """
    def __init__(self,rows,arm,canonical_to_dataset):
        probabilities(rows)
        if sorted(canonical_to_dataset)!=list(range(len(rows))):raise PairError('INVALID_DATASET_INDEX_MAP')
        self.rows=rows;self.arm=arm;self.mapping=canonical_to_dataset;self.next_epoch=0

    def __len__(self):return len(self.rows)

    def __iter__(self):
        epoch=self.next_epoch;self.next_epoch+=1
        return iter([self.mapping[int(i)] for i in plan_indices(self.rows,self.arm,epoch)])

    def rewind_for_reset(self,epoch):
        if epoch<0:raise PairError('INVALID_RESET_EPOCH')
        self.next_epoch=epoch


class JsonlSink:
    def __init__(self,out):
        if not (out/'execution.lock').is_file():raise PairError('EXECUTION_LOCK_REQUIRED')
        self.files={name:(out/f'{name}.jsonl').open('x',encoding='utf8') for name in ['optimizer','anchors','epochs']}

    def __call__(self,kind,record):
        f=self.files[kind];f.write(json.dumps(record,allow_nan=False,sort_keys=True)+'\n');f.flush()

    def close(self):
        for f in self.files.values():f.close()


class RunTelemetry:
    def __init__(self,root,rows,config,sink):
        self.root=Path(root).resolve();self.rows=rows;self.config=config;self.sink=sink
        self.arm=config['arm_label'];self.p=probabilities(rows)
        self.epoch=-1;self.batch_index=-1;self.position=0;self.epoch_attempts=0
        self.scheduled=0;self.applied=0;self.skipped=0;self.unknown=0;self.completed_epochs=0
        self.context_active=False;self.observed_steps=0;self.handle=None;self.start_time=time.monotonic()
        self.draws=Counter();self.dataset_paths=None;self.index_map=None

    def bind_dataset(self,files):
        paths=[Path(p).resolve() for p in files]
        expected=[(self.root/r['image_path']).resolve() for r in self.rows]
        if len(paths)!=len(expected) or len(set(paths))!=len(paths) or set(paths)!=set(expected):raise PairError('TRAIN_DATASET_IDENTITY_MISMATCH')
        lookup={p:i for i,p in enumerate(paths)}
        self.index_map=[lookup[p] for p in expected];self.dataset_paths=paths
        return self.index_map

    def bind_optimizer(self,optimizer):
        if self.handle is not None:raise PairError('OPTIMIZER_HOOK_ALREADY_INSTALLED')
        if getattr(optimizer,'_step_supports_amp_scaling',False) or getattr(optimizer,'defaults',{}).get('fused'):
            raise PairError('FUSED_INTERNAL_AMP_SKIP_NOT_SUPPORTED_BY_POST_HOOK_TELEMETRY')
        def after_step(_optimizer,_args,_kwargs):
            if not self.context_active:raise PairError('UNINSTRUMENTED_OPTIMIZER_STEP')
            self.observed_steps+=1
        self.handle=optimizer.register_step_post_hook(after_step)

    def start_epoch(self,epoch):
        if epoch!=self.completed_epochs or self.position not in (0,len(self.rows)):raise PairError('EPOCH_CONTINUITY_FAILED')
        self.epoch=epoch;self.batch_index=-1;self.position=0;self.epoch_attempts=0
        self.plan=plan_indices(self.rows,self.arm,epoch)

    def start_batch(self):self.batch_index+=1

    def anchors(self,files):
        if self.index_map is None or self.epoch<0 or self.batch_index<0:raise PairError('ANCHOR_CONTEXT_NOT_READY')
        batch=int(self.config['training_args']['batch'])
        count=min(batch,len(self.rows)-self.position)
        if len(files)!=count or count<=0:raise PairError('ANCHOR_BATCH_COUNT_MISMATCH')
        records=[]
        for offset,file in enumerate(files):
            index=int(self.plan[self.position+offset]);row=self.rows[index]
            if Path(file).resolve()!=(self.root/row['image_path']).resolve():raise PairError('ACTUAL_ANCHOR_SEQUENCE_MISMATCH')
            self.draws[index]+=1
            record={'arm':self.arm,'epoch':self.epoch,'batch_index':self.batch_index,
              'anchor_position':self.position+offset,'sample_id':row['sample_id'],'dataset_index':self.index_map[index],
              'sampling_category':row['sampling_category'],'sampling_mode':self.config['sampling']['mode']}
            if self.arm=='S':record.update(weight=row['sampling_weight'],draw_probability=float(self.p[index]))
            else:record.update(uniform_control_identity='one shuffled visit per training image per epoch',marginal_probability=1/len(self.rows))
            records.append(record)
        self.sink('anchors',{'arm':self.arm,'epoch':self.epoch,'batch_index':self.batch_index,'anchors':records})
        self.position+=len(files)

    def opportunity(self,trainer,original_optimizer_step):
        if self.handle is None or self.context_active or self.batch_index<0:raise PairError('OPTIMIZER_CONTEXT_NOT_READY')
        nb=self.config['budget']['batches_per_epoch'];global_batch=self.epoch*nb+self.batch_index
        record={'arm':self.arm,'global_batch':global_batch,'epoch':self.epoch,'batch_index':self.batch_index,
          'accumulation':int(trainer.accumulate),'optimizer_call_attempted':True,
          'grad_scaler_scale_before':float(trainer.scaler.get_scale()),
          'scheduler_state':deepcopy(trainer.scheduler.state_dict()),
          'learning_rate':[float(g['lr']) for g in trainer.optimizer.param_groups]}
        self.context_active=True;self.observed_steps=0;self.scheduled+=1;self.epoch_attempts+=1
        try:
            value=original_optimizer_step()  # original unscale/clip/step/update/zero_grad/EMA unchanged
            if self.observed_steps not in (0,1):raise PairError('MULTIPLE_UPDATES_IN_ONE_OPPORTUNITY')
            applied=self.observed_steps==1
            record.update(grad_scaler_scale_after=float(trainer.scaler.get_scale()),optimizer_step_applied=applied,
              optimizer_step_skipped=not applied,skip_reason_if_known=None if applied else 'AMP_STEP_SKIPPED',
              telemetry_status='COMPLETE')
            self.applied+=int(applied);self.skipped+=int(not applied)
        except Exception as exc:
            self.unknown+=1
            record.update(grad_scaler_scale_after=float(trainer.scaler.get_scale()),optimizer_step_applied=None,
              optimizer_step_skipped=None,skip_reason_if_known='EXECUTION_ERROR_NOT_CLASSIFIED_AS_AMP_SKIP',
              telemetry_status='FAILED',error_type=type(exc).__name__,observed_optimizer_post_hooks=self.observed_steps)
            self.sink('optimizer',record)
            raise
        finally:self.context_active=False
        self.sink('optimizer',record)
        return value

    def end_epoch(self):
        expected=self.config['budget']['scheduled_optimizer_calls_per_epoch'][self.epoch]
        if self.position!=len(self.rows) or self.batch_index+1!=self.config['budget']['batches_per_epoch'] or self.epoch_attempts!=expected:
            raise PairError('EPOCH_ANCHOR_OR_OPTIMIZER_BUDGET_MISMATCH')
        self.completed_epochs=self.epoch+1
        self.sink('epochs',{'epoch':self.epoch,'anchors_consumed':self.position,'scheduled_optimizer_calls':self.epoch_attempts,
          'cumulative_applied':self.applied,'cumulative_skipped':self.skipped})

    def summary(self):
        counts={'A':0,'B':0,'C':0,'small_GT':0,'very_small_GT':0,'medium_GT':0,'large_GT':0,'negative_images':0,'multi_GT_anchors':0}
        for i,n in self.draws.items():
            r=self.rows[i];counts[r['sampling_category']]+=n
            counts['negative_images']+=n*int(r['GT_count']==0);counts['multi_GT_anchors']+=n*int(r['multi_GT'])
            for gt in r['instance_geometry']:
                counts[gt['size_group']+'_GT']+=n;counts['very_small_GT']+=n*int(gt['bbox_area_ratio']<.0025)
        total=sum(self.draws.values())
        return {'arm':self.arm,'completed_epochs':self.completed_epochs,'scheduled_optimizer_calls':self.scheduled,
          'applied_optimizer_updates':self.applied,'skipped_optimizer_updates':self.skipped,'unknown_optimizer_opportunities':self.unknown,
          'training_runtime_seconds':time.monotonic()-self.start_time,'unique_anchors_seen':len(self.draws),
          'anchor_draws':total,'repeat_draws_across_run':total-len(self.draws),'exposure':counts,
          'exposure_scope':'Consumed anchor identities and their unaugmented training GT, not mosaic companion/pixel exposure'}


def default_train_loader(dataset,sampler,batch,workers):
    """Deferred imports: called only by a separately authorized D1 trainer."""
    import torch
    from ultralytics.data.build import InfiniteDataLoader, seed_worker
    from ultralytics.data.utils import PIN_MEMORY
    generator=torch.Generator();generator.manual_seed(6148914691236517204)
    return InfiniteDataLoader(dataset,batch_size=batch,shuffle=False,sampler=sampler,num_workers=workers,
      pin_memory=PIN_MEMORY,collate_fn=dataset.collate_fn,worker_init_fn=seed_worker,generator=generator)


def make_trainer_adapter(base_class,telemetry,train_loader_builder=default_train_loader):
    """Same adapter factory for both arms; factory creation does not load a model."""
    class PairedInstrumentedTrainer(base_class):
        def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
            if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
            if rank not in (-1,0):raise PairError('SINGLE_GPU_ONLY')
            expected_dirs={(telemetry.root/r['image_path']).resolve().parent for r in telemetry.rows}
            if len(expected_dirs)!=1 or Path(dataset_path).resolve() not in expected_dirs:raise PairError('FORBIDDEN_TRAINING_DATASET_PATH')
            dataset=self.build_dataset(dataset_path,mode,batch_size)
            mapping=telemetry.bind_dataset(dataset.im_files)
            self.paired_sampler=EpochPlanSampler(telemetry.rows,telemetry.arm,mapping)
            return train_loader_builder(dataset,self.paired_sampler,batch_size,self.args.workers)

        def _setup_train(self,world_size):
            if world_size!=1:raise PairError('SINGLE_GPU_ONLY')
            super()._setup_train(world_size)
            telemetry.bind_optimizer(self.optimizer)

        def run_callbacks(self,event):
            if event=='on_train_epoch_start':
                telemetry.start_epoch(self.epoch)
                if self.epoch==self.epochs-self.args.close_mosaic:self.paired_sampler.rewind_for_reset(self.epoch)
            elif event=='on_train_batch_start':telemetry.start_batch()
            elif event=='on_train_epoch_end':telemetry.end_epoch()
            return super().run_callbacks(event)

        def preprocess_batch(self,batch):
            telemetry.anchors(batch['im_file'])
            return super().preprocess_batch(batch)

        def optimizer_step(self):
            return telemetry.opportunity(self,super().optimizer_step)
    return PairedInstrumentedTrainer


def finalize_arm(out,telemetry,config,runtime):
    """D1 driver calls after training/final checkpoint writes, never during D0.1."""
    summary=telemetry.summary()
    for field in ['GPU','CUDA','torch','ultralytics','python']:
        if not runtime.get(field):raise PairError('MISSING_RUNTIME_TELEMETRY: '+field)
    summary['runtime']=runtime;summary['runner_sha256']=sha(Path(__file__))
    summary['initialization_sha256']=config['initialization']['sha256']
    summary['best_checkpoint_sha256']=sha(out/'training/weights/best.pt')
    summary['last_checkpoint_sha256']=sha(out/'training/weights/last.pt')
    summary['PAIRED_FIXED_BUDGET_COMPARISON']=('ARM_BUDGET_VALID' if summary['completed_epochs']==300 and summary['scheduled_optimizer_calls']==3741 and not summary['unknown_optimizer_opportunities'] else 'NOT_VALID')
    with (out/'arm_completion.json').open('x',encoding='utf8') as f:json.dump(summary,f,sort_keys=True,indent=2)
    return summary


def pair_validity(control,experimental):
    for arm in [control,experimental]:
        if any(type(arm.get(key)) is not int or arm[key]<0 for key in ['completed_epochs','scheduled_optimizer_calls','applied_optimizer_updates','skipped_optimizer_updates']):return 'NOT_VALID'
        if arm['completed_epochs']!=300 or arm['scheduled_optimizer_calls']!=3741 or arm.get('unknown_optimizer_opportunities',0):return 'NOT_VALID'
        if arm['applied_optimizer_updates']+arm['skipped_optimizer_updates']!=arm['scheduled_optimizer_calls']:return 'NOT_VALID'
    return 'VALID_SCHEDULED_BUDGET'


def amp_sensitivity(control,experimental):
    c,s=control['skipped_optimizer_updates'],experimental['skipped_optimizer_updates']
    return {'separate_skip_report_required':c>0 or s>0,'absolute_skip_difference':abs(c-s),
      'AMP_UPDATE_COUNT_IMBALANCE_OBSERVED':c!=s,'threshold_frozen':'any nonzero difference flags imbalance; never auto-retrain',
      'identical_realized_updates_claim_allowed':control['applied_optimizer_updates']==experimental['applied_optimizer_updates']}


def paired_advancement(control,experimental,validity):
    if validity!='VALID_SCHEDULED_BUDGET':return {'status':'NOT_VALID','checks':{}}
    for arm in [control,experimental]:
        fields=['tp','fp','fn','small_tp','medium_tp','large_tp','small_support','medium_support','large_support','positive_images','crop_complete_count']
        if any(type(arm.get(key)) is not int or arm[key]<0 for key in fields):raise PairError('INVALID_EVALUATION_COUNT')
        if (arm['small_support'],arm['medium_support'],arm['large_support'],arm['positive_images'])!=(137,90,14,186):raise PairError('EVALUATION_DENOMINATOR_CHANGED')
        if any(arm[size+'_tp']>arm[size+'_support'] for size in ['small','medium','large']) or arm['crop_complete_count']>arm['positive_images']:raise PairError('INVALID_EVALUATION_COUNT')
        if arm['tp']+arm['fn']!=241 or arm['tp']!=sum(arm[s+'_tp'] for s in ['small','medium','large']):raise PairError('EVALUATION_COUNT_MISMATCH')
    def p(a):return Fraction(a['tp'],a['tp']+a['fp']) if a['tp']+a['fp'] else Fraction(0)
    def f1(a):return Fraction(2*a['tp'],2*a['tp']+a['fp']+a['fn'])
    checks={'small':experimental['small_tp']>=control['small_tp']+1,
      'precision':p(experimental)>=p(control)-Fraction(1,100),
      'f1':f1(experimental)>=f1(control)-Fraction(1,100),
      'crop':experimental['crop_complete_count']>=control['crop_complete_count']-1,
      'medium':experimental['medium_tp']>=control['medium_tp']-1,
      'large':experimental['large_tp']>=control['large_tp']}
    return {'status':'PASS_SMALL_SAMPLING_RESEARCH_ADVANCEMENT_GATE' if all(checks.values()) else 'FAIL_SMALL_SAMPLING_RESEARCH_ADVANCEMENT_GATE','checks':checks}
