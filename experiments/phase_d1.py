"""Authorized V2 orchestration only; frozen training/evaluation implementations unchanged.

No resume/retry entrypoint. Each worker gets a fresh process and exclusive lock.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid
import yaml

from experiments import phase_d0 as d0
from experiments import paired_sampling_runner as paired

ROOT=Path(__file__).resolve().parents[1]
PREFIX='D_SEG_SMALL_SAMPLING_V2'
FREEZE_SHA='16b0f66c3661662ea9840774c835cfc9ced1e6ce1da8f7fe1d0b1a1067f0245b'
PAIR=Path('experiments/results/d_seg_small_sampling_v2_pair_summary')
REPORT=Path('docs/PHASE_D1_PAIRED_SMALL_OBJECT_SAMPLING_20260922.md')
REQUEST=Path('C:/Users/milo9/.codex/attachments/9a7f82f8-0d46-4402-81b9-72b97e3f9413/貼上的文字.txt')
TESTS=['test_phase_d1','test_phase_d01','test_phase_d0','test_phase_c1','test_phase_c_execution',
 'test_phase_c_postflight','test_phase_c0','test_phase_b1','test_phase_a5_guards','test_phase_a5_recompute',
 'test_phase_a_audit','test_phase_a_metrics_audit','test_experiment_review','test_localization_benchmark',
 'test_isic_fuseg_formal','test_fuseg_warmup_experiment']


def now():return datetime.now(timezone.utc).isoformat()
def require(ok,reason):
    if not ok:raise paired.PairError(reason)
def write(path,value):d0.write(path,value)
def load(root,suffix):return d0.read(root/'experiments/protocols'/f'{PREFIX}_{suffix}.json')


def validate_freeze(root):
    p=root/'experiments/protocols';f=p/f'{PREFIX}_freeze.json'
    require(d0.sha(f)==FREEZE_SHA,'FAIL_PROTOCOL_MUTATED: freeze index')
    freeze=d0.read(f)
    for rel,digest in freeze['artifacts_sha256'].items():
        require(d0.sha(root/rel)==digest,'FAIL_PROTOCOL_MUTATED: '+rel)
    require(freeze['PHASE_D01_STATUS']=='COMPLETE','FAIL_D01_NOT_COMPLETE')
    protected=load(root,'dry_audit')['protected_before']
    for path,digest in protected.items():
        require(d0.sha(path)==digest,'FAIL_HISTORICAL_ARTIFACT_MUTATED: '+path)
    v1=d0.read(p/'D_SEG_SMALL_SAMPLING_V1_verification.json')
    require(v1['PHASE_D0_STATUS']=='BLOCKED' and v1['blockers']==['BLOCKED_BY_ACTUAL_OPTIMIZER_UPDATE_COUNT_NOT_VERIFIED'],'FAIL_V1_MUTATED')
    c,s=load(root,'control_config'),load(root,'experimental_config');paired.config_diff(c,s)
    return c,s,protected


def verify_data(root,cfg):
    require(d0.sha(root/cfg['manifest']['path'])==cfg['manifest']['sha256'],'FAIL_TRAIN_MANIFEST_MUTATED')
    rows=d0.read(root/cfg['manifest']['path'])['samples'];require(len(rows)==771,'FAIL_TRAIN_COUNT')
    val=load(root,'evaluation_protocol')['validation_manifest']
    require(d0.sha(root/val['path'])==val['sha256'],'FAIL_VAL_MANIFEST_MUTATED')
    vals=d0.read(root/val['path'])['samples'];require(len(vals)==191,'FAIL_VAL_COUNT')
    d0.validate_exclusion(rows,vals);d0.probabilities(rows)
    for row in rows:
        for field,hashfield,kind in [('image_path','image_hash','images'),('label_path','label_hash','labels')]:
            path=(root/row[field]).resolve()
            require(path.parent==(root/d0.BUNDLE/'fuseg_dataset'/kind/'train').resolve(),'FAIL_TRAIN_SOURCE_ROLE')
            require(d0.sha(path)==row[hashfield],'FAIL_TRAIN_HASH')
    source=d0.read(root/d0.BUNDLE/'fuseg_manifest.json')
    training_val=[r for r in source if r['split']=='val'];require(len(training_val)==191,'FAIL_NATIVE_VAL_COUNT')
    by_id={v['sample_id']:v for v in vals}
    for row in training_val:
        reference=by_id[row['image_id']]
        for kind in ['image','label']:
            path=(root/d0.BUNDLE/row[kind]).resolve()
            require(path.parent==(root/d0.BUNDLE/'fuseg_dataset'/(kind+'s')/'val').resolve(),'FAIL_NATIVE_VAL_ROLE')
            require(d0.sha(path)==row[kind+'_sha256']==reference[kind+'_sha256'],'FAIL_NATIVE_VAL_HASH')
    for kind,field,pattern in [('images','image_path','*.png'),('labels','label_path','*.txt')]:
        require({p.resolve() for p in (root/d0.BUNDLE/'fuseg_dataset'/kind/'train').glob(pattern)}=={(root/r[field]).resolve() for r in rows},'FAIL_TRAIN_INVENTORY')
    data=yaml.safe_load((root/cfg['data_yaml_path']).read_text(encoding='utf8'))
    require(set(data)=={'path','train','val','names'} and data['train']=='images/train' and data['val']=='images/val' and data['names']=={0:'Wound'},'FAIL_DATA_YAML_ROLE')
    require(Path(data['path']).resolve()==(root/d0.BUNDLE/'fuseg_dataset').resolve(),'FAIL_DATA_ROOT')
    require(d0.sha(root/cfg['initialization']['path'])==cfg['initialization']['sha256']=='1ec926026473beafafb1ef8da6aa6efcc13d38c391ec8db48006c5d793d5f2a3','FAIL_INITIALIZATION_MISMATCH')
    return rows


def runtime(cfg):
    for name,version in cfg['software'].items():require(metadata.version(name)==version,'FAIL_SOFTWARE_VERSION: '+name)
    import torch
    require(torch.cuda.is_available(),'FAIL_GPU_UNAVAILABLE')
    return {'GPU':torch.cuda.get_device_name(0),'CUDA':torch.version.cuda,'torch':torch.__version__,
      'ultralytics':metadata.version('ultralytics'),'python':platform.python_version(),
      'numpy':metadata.version('numpy'),'albumentations':metadata.version('albumentations')}


def status(root,stage,**extra):
    path=root/PAIR/'status.json';tmp=path.with_suffix('.tmp')
    value={'updated_at':now(),'stage':stage,'test_images_used':0,'LOCKED_TEST_USED':False,'CO2Wounds_used':False,**extra}
    with tmp.open('w',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
    os.replace(tmp,path)


def snapshot_extra(root):
    # Only research outputs/source files; never traverse clinical or external datasets.
    paths=[]
    for folder in ['experiments/results/tables','experiments/results/statistics']:
        paths.extend(p for p in (root/folder).rglob('*') if p.is_file())
    for pattern in ['**/*C1*','**/*B1*']:
        paths.extend(p for p in (root/'experiments/protocols').glob(pattern) if p.is_file())
    return {str(p):d0.sha(p) for p in sorted(set(paths))}


def verify_snapshot(values):
    for path,digest in values.items():require(d0.sha(path)==digest,'FAIL_HISTORICAL_ARTIFACT_MUTATED: '+path)


def read_jsonl(path):
    with path.open(encoding='utf8') as f:
        for line in f:
            require(bool(line.strip()),'FAIL_EMPTY_TELEMETRY_ROW')
            yield json.loads(line)


def audit_telemetry(out,rows,cfg,summary):
    epochs=list(read_jsonl(out/'epochs.jsonl')); require(len(epochs)==summary['completed_epochs'],'FAIL_EPOCH_TELEMETRY')
    operations=list(read_jsonl(out/'optimizer.jsonl'))
    require(len(operations)==summary['scheduled_optimizer_calls'],'FAIL_OPTIMIZER_TELEMETRY_INTEGRITY')
    require(all(x['telemetry_status']=='COMPLETE' and x['optimizer_call_attempted'] is True and
      type(x['optimizer_step_applied']) is bool and type(x['optimizer_step_skipped']) is bool and
      x['optimizer_step_applied']!=x['optimizer_step_skipped'] for x in operations),'FAIL_OPTIMIZER_TELEMETRY_INTEGRITY')
    require(sum(x['optimizer_step_applied'] for x in operations)==summary['applied_optimizer_updates'] and
      sum(x['optimizer_step_skipped'] for x in operations)==summary['skipped_optimizer_updates'],'FAIL_OPTIMIZER_TOTALS')
    op_counts=Counter(x['epoch'] for x in operations)
    last=-1
    for x in operations:
        require(x['global_batch']==x['epoch']*193+x['batch_index'] and x['global_batch']>last,'FAIL_OPTIMIZER_ORDER')
        last=x['global_batch']
    for e in range(len(epochs)):
        require(op_counts[e]==cfg['budget']['scheduled_optimizer_calls_per_epoch'][e],'FAIL_EPOCH_OPTIMIZER_BUDGET')
    counts=Counter(); batches=0; unique_epochs=[]; canonical={r['sample_id']:i for i,r in enumerate(rows)}
    probabilities=paired.probabilities(rows)
    current=-1; ids=[]; plan=None; mapping={}
    for batch in read_jsonl(out/'anchors.jsonl'):
        e,b=batch['epoch'],batch['batch_index']
        if e!=current:
            if current>=0:require(len(ids)==771,'FAIL_EPOCH_ANCHOR_COUNT');unique_epochs.append(len(set(ids)))
            require(e==current+1,'FAIL_ANCHOR_EPOCH_ORDER');current=e;ids=[];plan=paired.plan_indices(rows,cfg['arm_label'],e)
        require(b==len(ids)//4 and len(batch['anchors'])==min(4,771-len(ids)),'FAIL_ANCHOR_BATCH')
        for anchor in batch['anchors']:
            i=canonical.get(anchor['sample_id']);pos=len(ids)
            require(i is not None and i==int(plan[pos]) and anchor['anchor_position']==pos,'FAIL_ACTUAL_ANCHOR_SEQUENCE')
            require(anchor['sampling_category']==rows[i]['sampling_category'],'FAIL_ANCHOR_CATEGORY')
            if i in mapping:require(mapping[i]==anchor['dataset_index'],'FAIL_DATASET_INDEX_CHANGED')
            mapping[i]=anchor['dataset_index'];ids.append(i);counts[i]+=1
            if cfg['arm_label']=='S':
                require(anchor['weight']==rows[i]['sampling_weight'] and abs(anchor['draw_probability']-float(probabilities[i]))<1e-15,'FAIL_SAMPLING_PROBABILITY')
        batches+=1
    if current>=0:require(len(ids)==771,'FAIL_EPOCH_ANCHOR_COUNT');unique_epochs.append(len(set(ids)))
    require(len(unique_epochs)==len(epochs) and batches==193*len(epochs),'FAIL_ANCHOR_TOTALS')
    if cfg['arm_label']=='C':require(all(n==771 for n in unique_epochs),'FAIL_CONTROL_REPLACEMENT')
    require(sum(counts.values())==summary['anchor_draws'] and len(counts)==summary['unique_anchors_seen'],'FAIL_EXPOSURE_TOTALS')
    require(len(set(mapping.values()))==len(mapping),'FAIL_DATASET_INDEX_MAP')
    exposure=Counter({'A':0,'B':0,'C':0,'small_GT':0,'very_small_GT':0,'medium_GT':0,'large_GT':0,'negative_images':0,'multi_GT_anchors':0})
    for i,n in counts.items():
        row=rows[i];exposure[row['sampling_category']]+=n
        exposure['negative_images']+=n*int(row['GT_count']==0);exposure['multi_GT_anchors']+=n*int(row['multi_GT'])
        for gt in row['instance_geometry']:
            exposure[gt['size_group']+'_GT']+=n;exposure['very_small_GT']+=n*int(gt['bbox_area_ratio']<.0025)
    require(dict(exposure)==summary['exposure'],'FAIL_EXPOSURE_SUMMARY')
    return {'status':'PASS','total_batches':batches,'optimizer_rows':len(operations),'anchor_draws':sum(counts.values()),
      'unique_per_epoch':unique_epochs,'actual_plan_match':True,'consumed_ids_all_training':True}


def train_arm(root,arm,token):
    lock=d0.read(root/PAIR/'execution.lock');require(lock['token']==token,'FAIL_EXECUTION_AUTHORIZATION')
    require(lock['driver_sha256']==d0.sha(Path(__file__)),'FAIL_DRIVER_MUTATED')
    c,s,_=validate_freeze(root);cfg=c if arm=='C' else s
    rows=verify_data(root,cfg);environment=runtime(cfg)
    require(environment==lock['environment'],'FAIL_PAIRED_ENVIRONMENT_CHANGED')
    if arm=='S':
        control=d0.read(root/paired.OUTPUTS['C']/'training_completion.json')
        require(paired.pair_validity(control,control)=='VALID_SCHEDULED_BUDGET','FAIL_CONTROL_BUDGET')
    out=root/paired.OUTPUTS[arm];require(not out.exists(),'FAIL_OUTPUT_ALREADY_EXISTS');out.mkdir()
    write(out/'execution.lock',{'experiment_id':cfg['experiment_id'],'arm':arm,'start_time':now(),
      'protocol_sha256':d0.sha(root/'experiments/protocols'/f'{PREFIX}_protocol.json'),
      'config_sha256':d0.sha(root/'experiments/protocols'/f'{PREFIX}_{"control" if arm=="C" else "experimental"}_config.json'),
      'runner_sha256':cfg['runner']['sha256'],'driver_sha256':lock['driver_sha256'],
      'training_manifest_sha256':cfg['manifest']['sha256'],'initialization_sha256':cfg['initialization']['sha256'],
      'environment':environment,'immutable':True,'resume':False,'test_images_used':0})
    sink=paired.JsonlSink(out)
    os.link(out/'optimizer.jsonl',out/'optimizer_telemetry.jsonl');os.link(out/'anchors.jsonl',out/'anchor_telemetry.jsonl')
    telemetry=paired.RunTelemetry(root,rows,cfg,sink)
    try:
        from ultralytics import YOLO
        from ultralytics.models.yolo.segment.train import SegmentationTrainer
        model=YOLO(str(root/cfg['initialization']['path']))
        cls=paired.make_trainer_adapter(SegmentationTrainer,telemetry)
        def runtime_guard(trainer):
            for key,value in cfg['training_args'].items():
                require(getattr(trainer.args,key)==value,'FAIL_RUNTIME_ARG: '+key)
            require(trainer.amp and trainer.scaler.is_enabled(),'FAIL_AMP_POLICY')
            require(trainer.model.names=={0:'Wound'},'FAIL_MODEL_NAMES')
            require(sum(p.numel() for p in trainer.model.parameters())==cfg['architecture']['runtime_parameters'],'FAIL_ARCHITECTURE')
            require(type(trainer.optimizer).__name__=='AdamW' and len(trainer.optimizer.state)==0,'FAIL_FRESH_OPTIMIZER')
            require(all(g['betas']==(.937,.999) and g['lr']==.0005 for g in trainer.optimizer.param_groups),'FAIL_OPTIMIZER_RECIPE')
            require(sorted(g['weight_decay'] for g in trainer.optimizer.param_groups)==[0.,0.,.0005],'FAIL_WEIGHT_DECAY')
            require(trainer.start_epoch==0 and trainer.accumulate==16 and len(trainer.train_loader)==193,'FAIL_SCHEDULE')
            write(out/'runtime_training_guard.json',{'status':'PASS','environment':environment,'fresh_optimizer':True,
              'model_parameters':sum(p.numel() for p in trainer.model.parameters()),'amp_enabled':True,'scaler_class':type(trainer.scaler).__qualname__})
        def progress(trainer):status(root,'TRAINING_'+arm,completed_epochs=telemetry.completed_epochs,
          scheduled_calls=telemetry.scheduled,applied_updates=telemetry.applied,skipped_updates=telemetry.skipped)
        model.add_callback('on_train_start',runtime_guard)
        model.add_callback('on_fit_epoch_end',progress)
        args=deepcopy(cfg['training_args']);args.update(data=str(root/cfg['data_yaml_path']),project=str(out),name='training')
        model.train(trainer=cls,**args)
        sink.close()
        summary=paired.finalize_arm(out,telemetry,cfg,environment)
        summary.update(unknown_updates=summary['unknown_optimizer_opportunities'],unique_anchor_exposure=summary['unique_anchors_seen'],
          repeat_draws=summary['repeat_draws_across_run'],test_images_used=0,LOCKED_TEST_USED=False,CO2Wounds_used=False)
        audit=audit_telemetry(out,rows,cfg,summary);summary['total_batches']=audit['total_batches']
        with (out/'training/results.csv').open(encoding='utf8',newline='') as f:history=list(csv.DictReader(f))
        require([int(float(r['epoch'])) for r in history]==list(range(1,summary['completed_epochs']+1)),'FAIL_NATIVE_EPOCH_HISTORY')
        write(out/'telemetry_integrity.json',audit);write(out/'training_completion.json',summary)
        for name,original in [('best.pt','training/weights/best.pt'),('last.pt','training/weights/last.pt'),('results.csv','training/results.csv')]:os.link(out/original,out/name)
        require(paired.pair_validity(summary,summary)=='VALID_SCHEDULED_BUDGET','PAIRED_FIXED_BUDGET_COMPARISON_NOT_VALID: EARLY_STOP_OR_TELEMETRY')
        validate_freeze(root)
    except BaseException as exc:
        write(out/'interruption.json',{'PHASE_D1_STATUS':'INTERRUPTED','at':now(),'error_type':type(exc).__name__,
          'error':str(exc),'telemetry_partial':telemetry.summary(),'retry_allowed':False,'test_images_used':0})
        raise
    finally:sink.close()


def verify_validation(root):
    from experiments.phase_c_execution import assert_relative_admitted,BUNDLE_REL,MASK_ROOT_REL
    ep=load(root,'evaluation_protocol');m=ep['validation_manifest'];require(d0.sha(root/m['path'])==m['sha256'],'FAIL_VALIDATION_MANIFEST')
    samples=d0.read(root/m['path'])['samples'];require(len(samples)==191,'FAIL_VALIDATION_COUNT')
    cp=d0.read(root/d0.C_PROTOCOL);cohort=d0.read(root/cp['provenance']['historical_cohort_path'])
    require(d0.sha(root/cp['provenance']['historical_cohort_path'])==cp['provenance']['historical_cohort_sha256'],'FAIL_COHORT_MUTATED')
    originals={r['image_id']:r for r in cohort}
    for row in samples:
        original=originals[row['sample_id']]
        paths={'image':row['relative_image_path'],'label':BUNDLE_REL+'/'+original['label'],'mask':row['relative_mask_path']}
        for kind,rel in paths.items():
            admitted=root/(MASK_ROOT_REL if kind=='mask' else BUNDLE_REL+'/dataset/'+kind+'s/val')
            path=assert_relative_admitted(root,rel,admitted)
            require(d0.sha(path)==row[kind+'_sha256']==original[kind+'_sha256'],'FAIL_VALIDATION_RUNTIME_HASH')
    return ep,cohort


def diagnostics(rows):
    # Reuse C1 size-bin and single/multi grouping definitions, not its historical count assertions.
    from experiments.phase_c1 import SMALL_BINS,small_bin,bbox_features,group_metrics
    bins={b:{'support':0,'TP':0} for b in SMALL_BINS}
    for row in rows:
        matched={p['gt'] for p in row['pairs']}
        for i,box in enumerate(row['gt_boxes']):
            area=bbox_features(box)['bbox_area_ratio']
            if area<.01:
                b=bins[small_bin(area)];b['support']+=1;b['TP']+=int(i in matched)
    for b in bins.values():b['recall']=b['TP']/b['support'] if b['support'] else None
    support=sum(bins[b]['support'] for b in SMALL_BINS[:2]);tp=sum(bins[b]['TP'] for b in SMALL_BINS[:2])
    return {'small_bins':bins,'very_small':{'support':support,'TP':tp,'recall':tp/support if support else None},
      'single_GT':group_metrics([r for r in rows if r['num_gt_instances']==1]),
      'multi_GT':group_metrics([r for r in rows if r['num_gt_instances']>1])}


def evaluate_arm(root,arm):
    from experiments.phase_c_execution import validate_result_integrity
    from experiments.data_roles import validate_model_input_contract
    from experiments.review_v2 import localization_benchmark as loc
    from experiments.review_v2.isic_fuseg_gate import decide
    from PIL import Image
    import numpy as np
    from ultralytics import YOLO
    validate_freeze(root);ep,cohort=verify_validation(root)
    out=root/paired.OUTPUTS[arm];checkpoint=out/'best.pt';completion=d0.read(out/'training_completion.json')
    require(d0.sha(checkpoint)==completion['best_checkpoint_sha256'],'FAIL_CHECKPOINT_MUTATED')
    write(out/'evaluation.lock',{'arm':arm,'at':now(),'checkpoint_sha256':d0.sha(checkpoint),'validation_count':191,'test_images_used':0})
    model=YOLO(str(checkpoint));require(model.task=='segment' and model.names=={0:'Wound'},'FAIL_EVALUATION_MODEL')
    settings={**ep['prediction'],'iou':ep['operating_point']['nms_iou']}
    for _ in range(3):loc.predict_materialized(model,np.zeros((512,512,3),np.uint8),settings)
    (out/'masks').mkdir();(out/'per_image').mkdir();rows=[]
    for index,original in enumerate(cohort):
        with Image.open(loc.safe_path(loc.BUNDLE,original['image'])) as image:
            require(image.size==(512,512),'FAIL_VALIDATION_SIZE')
            bgr=validate_model_input_contract(image.convert('RGB'),source_type='PIL_RGB')
        prediction,masks,ms=loc.predict_materialized(model,bgr,settings)
        row=loc.assess(original,prediction,masks,ep['operating_point']['confidence'])
        scores=prediction.boxes.conf.cpu().numpy();keep=np.flatnonzero(scores>=ep['operating_point']['confidence'])
        maskrel=f'masks/{index:03d}.npz'
        with (out/maskrel).open('xb') as f:np.savez_compressed(f,masks=masks,confidences=scores,boxes=prediction.boxes.xyxy.cpu().numpy(),retained_indices=keep)
        matched={p['gt'] for p in row['pairs']}
        row.update(sample_id=original['image_id'],num_gt_instances=len(row['gt_boxes']),num_predictions=len(row['pred_boxes']),
          per_instance_gt=[{'index':i,'bbox':box,'size_group':loc.size_name(box),'matched':i in matched} for i,box in enumerate(row['gt_boxes'])],
          prediction_mask_reference=maskrel,prediction_mask_sha256=d0.sha(out/maskrel),retained_mask_indices=keep.tolist(),
          floor_prediction_count=len(scores),floor_confidences=scores.tolist(),floor_bboxes=prediction.boxes.xyxy.cpu().numpy().tolist(),
          retained_gt_wound_pixels=int(round(row['crop_coverage']*row['gt_pixels'])) if row['gt_pixels'] else 0,inference_latency_ms=ms)
        write(out/'per_image'/f'{index:03d}.json',row);rows.append(row)
    write(out/'per_image_predictions.json',rows);saved=d0.read(out/'per_image_predictions.json');summary=loc.summarize(saved)
    integrity=validate_result_integrity(saved,summary,[r['image_id'] for r in cohort]);latency=loc.latency_summary([r['inference_latency_ms'] for r in saved])
    decision=decide(summary,latency);performance={k:decision['checks'][k] for k in ep['acceptance_gate']['minima_percent']}
    result={'candidate':summary,'diagnostics':diagnostics(saved),'integrity':integrity,'latency':latency,
      'historical_development_gate':'PARTIAL' if all(performance.values()) else 'FAIL','performance_checks':performance,
      'latency_comparability':'NOT_VERIFIED; descriptive latency only','test_images_used':0,'CO2Wounds_used':False,
      'checkpoint_sha256':d0.sha(checkpoint),'predictions_sha256':d0.sha(out/'per_image_predictions.json')}
    write(out/'final_evaluation.json',result);return result


def gate_counts(summary):
    return {'tp':summary['tp'],'fp':summary['fp'],'fn':summary['fn'],'positive_images':summary['positive_images'],
      'crop_complete_count':summary['crop_complete95_images'],
      **{s+'_support':summary['size_recall'][s]['gt'] for s in ['small','medium','large']},
      **{s+'_tp':summary['size_recall'][s]['matched'] for s in ['small','medium','large']}}


def csv_output(path,rows):
    with path.open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def verify_saved_arm(root,arm,cfg):
    out=root/paired.OUTPUTS[arm];summary=d0.read(out/'training_completion.json')
    for name in ['best','last']:
        require(d0.sha(out/f'{name}.pt')==d0.sha(out/f'training/weights/{name}.pt')==summary[name+'_checkpoint_sha256'],'FAIL_CHECKPOINT_MUTATED')
    require(d0.sha(out/'optimizer.jsonl')==d0.sha(out/'optimizer_telemetry.jsonl') and
      d0.sha(out/'anchors.jsonl')==d0.sha(out/'anchor_telemetry.jsonl'),'FAIL_TELEMETRY_ALIAS')
    rows=d0.read(root/cfg['manifest']['path'])['samples']
    audit=audit_telemetry(out,rows,cfg,summary)
    if (out/'final_evaluation.json').exists():
        import numpy as np
        from experiments.phase_c_execution import validate_result_integrity
        result=d0.read(out/'final_evaluation.json');preds=d0.read(out/'per_image_predictions.json')
        require(result['predictions_sha256']==d0.sha(out/'per_image_predictions.json'),'FAIL_PREDICTIONS_MUTATED')
        manifest=d0.read(root/load(root,'evaluation_protocol')['validation_manifest']['path'])['samples']
        audit['evaluation']=validate_result_integrity(preds,result['candidate'],[r['sample_id'] for r in manifest])
        for index,row in enumerate(preds):
            require(d0.read(out/'per_image'/f'{index:03d}.json')==row,'FAIL_PER_IMAGE_EVIDENCE')
            require(d0.sha(out/row['prediction_mask_reference'])==row['prediction_mask_sha256'],'FAIL_PREDICTION_MASK_HASH')
            with np.load(out/row['prediction_mask_reference'],allow_pickle=False) as masks:
                require(masks['masks'].shape==(row['floor_prediction_count'],512,512),'FAIL_PREDICTION_MASK_SHAPE')
                require(masks['retained_indices'].tolist()==row['retained_mask_indices'],'FAIL_RETAINED_PREDICTIONS')
    return audit


def complete_pair(root,completions,evaluations):
    c,s=(evaluations[a] for a in ['C','S']);validity=paired.pair_validity(*[completions[a] for a in ['C','S']])
    gate=paired.paired_advancement(gate_counts(c['candidate']),gate_counts(s['candidate']),validity)
    amp=paired.amp_sensitivity(*[completions[a] for a in ['C','S']])
    require(completions['S']['exposure']['A']>completions['C']['exposure']['A'] and
      completions['S']['exposure']['small_GT']>completions['C']['exposure']['small_GT'],'FAIL_INTERVENTION_NOT_DELIVERED')
    h=d0.read(root/d0.C_OUT/'result.json')['candidate'];hd=diagnostics(d0.read(root/d0.C_OUT/'per_image_predictions.json'))
    allvals=[(h,hd),(c['candidate'],c['diagnostics']),(s['candidate'],s['diagnostics'])]
    table=['| Metric | Historical Phase C | Fresh C | Fresh S | S−C |','|---|---:|---:|---:|---:|']
    metrics=[('Precision',lambda x,d:x['precision']),('Recall',lambda x,d:x['recall']),('F1',lambda x,d:x['f1']),
      ('Small Recall',lambda x,d:x['size_recall']['small']['recall']),('Very-small Recall',lambda x,d:d['very_small']['recall']),
      ('Medium Recall',lambda x,d:x['size_recall']['medium']['recall']),('Large Recall',lambda x,d:x['size_recall']['large']['recall']),
      ('Crop complete',lambda x,d:x['crop_complete95_fraction']),('TP',lambda x,d:x['tp']),('FP',lambda x,d:x['fp']),('FN',lambda x,d:x['fn']),('No ROI',lambda x,d:x['positive_without_roi'])]
    for name,fn in metrics:
        v=[fn(x,d) for x,d in allvals];table.append('| '+name+' | '+' | '.join(f'{n:.6f}' for n in [*v,v[2]-v[1]])+' |')
    training_table=['| Training metric | Fresh C | Fresh S |','|---|---:|---:|']
    for key in ['completed_epochs','scheduled_optimizer_calls','applied_optimizer_updates','skipped_optimizer_updates','anchor_draws','unique_anchors_seen','repeat_draws_across_run']:
        training_table.append(f"| {key} | {completions['C'][key]} | {completions['S'][key]} |")
    exposures=[{'metric':k,'C':completions['C']['exposure'][k],'S':completions['S']['exposure'][k]} for k in completions['C']['exposure']]
    csv_output(root/PAIR/'sampling_exposure_comparison.csv',exposures)
    csv_output(root/PAIR/'amp_telemetry_comparison.csv',[{'arm':a,**{k:completions[a][k] for k in ['completed_epochs','scheduled_optimizer_calls','applied_optimizer_updates','skipped_optimizer_updates']}} for a in ['C','S']])
    result={'PHASE_D1_STATUS':'COMPLETE','PAIR_FIXED_BUDGET_VALID':'YES','SAMPLING_INTERVENTION_DELIVERED':'YES',
      'AMP_UPDATE_COUNT_IMBALANCE_OBSERVED':'YES' if amp['AMP_UPDATE_COUNT_IMBALANCE_OBSERVED'] else 'NO',
      'SMALL_SAMPLING_RESEARCH_GATE':'PASS' if gate['status'].startswith('PASS') else 'FAIL','paired_gate':gate,'amp':amp,
      'HISTORICAL_DEVELOPMENT_GATE_CONTROL':c['historical_development_gate'],'HISTORICAL_DEVELOPMENT_GATE_EXPERIMENTAL':s['historical_development_gate'],
      'MULTI_SEED_AUTHORIZED':'NO','APP_MODEL_REPLACEMENT_AUTHORIZED':'NO','EXTERNAL_TEST_AUTHORIZED':'NO',
      'LOCKED_TEST_USED':False,'CO2Wounds_used':False,'test_images_used':0,'training':completions,'evaluations':evaluations,
      'negative_result_preserved':not gate['status'].startswith('PASS'),'finished_at':now()}
    text='# Phase D1 配對抽樣實驗結果\n\n'+ '\n'.join(table)+'\n\n'+ '\n'.join(training_table)+'\n\n'
    text+='## 診斷與完整決策\n\n```json\n'+json.dumps({'gates':gate,'AMP':amp,'C_diagnostics':c['diagnostics'],'S_diagnostics':s['diagnostics'],'historical_diagnostics':hd,'exposure':exposures},ensure_ascii=False,indent=2)+'\n```\n\n'
    text+='兩組完整 300 epochs；scheduled/applied/skipped 見表。未 resume/retry，未修改 threshold/crop/metrics。未使用 locked test、CO2、external test；未啟動 multi-seed、未替換 App。\n\n'
    text+=('Under this single-seed paired development experiment, small-object-aware sampling improved observed small-wound recall relative to the fresh uniform-sampling control.\n\n' if s['candidate']['size_recall']['small']['recall']>c['candidate']['size_recall']['small']['recall'] else '沒有觀察到預登錄的 small Recall 改善。\n\n')
    text+='本結果不代表統計顯著性或臨床泛化；未新增 CI、p-value 或 McNemar。Latency 僅描述，歷史 gate 若性能全通過則標 PARTIAL，未假稱硬體可比。\n\n'
    if amp['AMP_UPDATE_COUNT_IMBALANCE_OBSERVED']:
        text+='The arms received the same scheduled optimization budget, but realized successful optimizer-update counts differed because of AMP step skipping. 不宣稱實際成功更新數相同，亦不因此重跑。\n\n'
    text+=('下一研究決策可考慮另次授權 multi-seed confirmation；現在停止。' if result['SMALL_SAMPLING_RESEARCH_GATE']=='PASS' else 'NEGATIVE_RESULT_PRESERVED；停止，保留負結果供另次研究決策，不改 weights 或重跑。')
    answers=[('Control 完整300 epochs',str(completions['C']['completed_epochs'])),('Experimental 完整300',str(completions['S']['completed_epochs'])),
      ('Scheduled opportunities',f"C={completions['C']['scheduled_optimizer_calls']}; S={completions['S']['scheduled_optimizer_calls']}"),
      ('Applied updates',f"C={completions['C']['applied_optimizer_updates']}; S={completions['S']['applied_optimizer_updates']}"),
      ('Skipped updates',f"C={completions['C']['skipped_optimizer_updates']}; S={completions['S']['skipped_optimizer_updates']}"),
      ('AMP imbalance',str(amp['AMP_UPDATE_COUNT_IMBALANCE_OBSERVED'])),('Sampling delivered','YES; 實際序列與各 frozen epoch plan 相符，exposure 差異見 JSON/CSV'),
      ('Control Small Recall',str(c['candidate']['size_recall']['small'])),('Experimental Small Recall',str(s['candidate']['size_recall']['small'])),
      ('Small TP 改變',str(s['candidate']['size_recall']['small']['matched']-c['candidate']['size_recall']['small']['matched'])),
      ('Precision 改變',str(s['candidate']['precision']-c['candidate']['precision'])),('F1 改變',str(s['candidate']['f1']-c['candidate']['f1'])),
      ('Crop complete 改變',str(s['candidate']['crop_complete95_images']-c['candidate']['crop_complete95_images'])+' 張'),
      ('Medium/Large safety',str({k:gate['checks'][k] for k in ['medium','large']})),('Very-small bins','見五 bins diagnostics；不事後改權重'),
      ('Multi-GT Recall','見 C/S single_GT、multi_GT diagnostics'),('FP/FN/no-ROI','見主表與 candidate summary'),
      ('Research gate',result['SMALL_SAMPLING_RESEARCH_GATE']),('Historical gate',f"C={c['historical_development_gate']}; S={s['historical_development_gate']}"),
      ('Fresh C vs historical C','見主表；不因兩者不同就認定 bug'),('Crash/resume/retry','NO'),('Threshold/crop/metrics 改變','NO'),
      ('Locked test','NO'),('CO2Wounds','NO'),('Multi-seed','NO'),('App replacement','NO'),('下一研究決策','另次授權 multi-seed confirmation' if result['SMALL_SAMPLING_RESEARCH_GATE']=='PASS' else 'Preserve negative result；另次 forensic/crop-policy 研究須重新授權')]
    text+='\n\n## 27 個指定問題\n\n'+'\n\n'.join(f'{i}. **{q}**：{a}' for i,(q,a) in enumerate(answers,1))+'\n'
    return result,text


def failure_report(root,error):
    result={'PHASE_D1_STATUS':'INTERRUPTED','PAIR_FIXED_BUDGET_VALID':'NO','error':str(error),'error_type':type(error).__name__,
      'at':now(),'test_images_used':0,'LOCKED_TEST_USED':False,'CO2Wounds_used':False,'retry_allowed':False,
      'MULTI_SEED_AUTHORIZED':'NO','APP_MODEL_REPLACEMENT_AUTHORIZED':'NO','EXTERNAL_TEST_AUTHORIZED':'NO'}
    write(root/PAIR/'interruption.json',result);status(root,'INTERRUPTED',error=str(error))
    if not (root/REPORT).exists():
        with (root/REPORT).open('x',encoding='utf8') as f:f.write('# Phase D1 中斷報告\n\n未完成正式配對比較。保留所有 partial artifacts，不重試／resume。\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n')


def execute(root):
    c,s,protected=validate_freeze(root);paired.require_output_absent(root)
    require(not (root/PAIR).exists() and not (root/REPORT).exists(),'FAIL_OUTPUT_ALREADY_EXISTS')
    verify_data(root,c);verify_validation(root);environment=runtime(c)
    extras=snapshot_extra(root);out=root/PAIR;out.mkdir();token=str(uuid.uuid4())
    write(out/'execution.lock',{'D1_authorized':True,'at':now(),'token':token,'parent_pid':os.getpid(),
      'request_sha256':d0.sha(REQUEST),'driver_sha256':d0.sha(Path(__file__)),'freeze_sha256':FREEZE_SHA,'environment':environment})
    write(out/'preflight.json',{'status':'PASS','train_images':771,'validation_images':191,'train_val_hash_overlap':0,
      'validation_ids_sampled':0,'test_images_used':0,'protected_before':protected,'additional_protected':extras})
    try:
        completions={};evaluations={}
        for arm in ['C','S']:
            status(root,'STARTING_'+arm)
            with (out/f'train_{arm}.log').open('x',encoding='utf8') as log:
                proc=subprocess.run([sys.executable,'-u','-m','experiments.phase_d1','--arm',arm,'--pair-token',token],cwd=root,stdout=log,stderr=subprocess.STDOUT)
            require(proc.returncode==0,f'ARM_{arm}_FAILED_EXIT_{proc.returncode}; see train_{arm}.log; no retry')
            completions[arm]=d0.read(root/paired.OUTPUTS[arm]/'training_completion.json')
            require(paired.pair_validity(completions[arm],completions[arm])=='VALID_SCHEDULED_BUDGET','FAIL_ARM_BUDGET')
            validate_freeze(root);verify_snapshot(extras)
        validity=paired.pair_validity(completions['C'],completions['S']);require(validity=='VALID_SCHEDULED_BUDGET','FAIL_PAIR_VALIDITY')
        audits={arm:verify_saved_arm(root,arm,c if arm=='C' else s) for arm in ['C','S']}
        require(completions['S']['exposure']['A']>completions['C']['exposure']['A'] and completions['S']['exposure']['small_GT']>completions['C']['exposure']['small_GT'],'FAIL_INTERVENTION_NOT_DELIVERED')
        write(out/'pair_validity.json',{'status':validity,'config_diff':paired.config_diff(c,s),'telemetry_audits':audits})
        for arm in ['C','S']:status(root,'EVALUATING_'+arm);evaluations[arm]=evaluate_arm(root,arm)
        result,text=complete_pair(root,completions,evaluations)
        status(root,'POSTFLIGHT');validate_freeze(root);verify_snapshot(extras)
        with (out/'postflight_tests.log').open('x',encoding='utf8') as log:
            tests=subprocess.run([sys.executable,'-m','pytest',*[f'tests/{name}.py' for name in TESTS],'-q'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
        require(tests.returncode==0,'FAIL_POSTFLIGHT_TESTS')
        validate_freeze(root);verify_snapshot(extras)
        result['postflight_tests_passed']=True;result['historical_integrity']='UNCHANGED'
        result['saved_artifact_postflight']={arm:verify_saved_arm(root,arm,c if arm=='C' else s) for arm in ['C','S']}
        write(out/'pair_summary.json',result)
        with (out/'paired_comparison.md').open('x',encoding='utf8') as f:f.write(text)
        with (root/REPORT).open('x',encoding='utf8') as f:f.write(text)
        status(root,'COMPLETE',research_gate=result['SMALL_SAMPLING_RESEARCH_GATE'])
    except BaseException as exc:failure_report(root,exc);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute-authorized',action='store_true');parser.add_argument('--arm',choices=['C','S']);parser.add_argument('--pair-token')
    args=parser.parse_args()
    if args.arm:train_arm(ROOT,args.arm,args.pair_token)
    elif args.execute_authorized:execute(ROOT)
    else:parser.error('Explicit --execute-authorized required; no resume option exists')
