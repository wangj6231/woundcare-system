"""D0 preregistration: metadata/byte hashes and index draws only; no model API."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import numpy as np
import yaml

PREFIX='D_SEG_SMALL_SAMPLING_V1'
FORMAL=Path('outputs/isic_fuseg_formal_seed42_20260914')
BUNDLE=Path('outputs/isic_fuseg_pretrain_smoke_20260914')
C_PROTOCOL=Path('experiments/protocols/ISIC_FUSEG_COLORFIX_V1_protocol.json')
C_OUT=Path('experiments/results/isic_fuseg_colorfix_v1')
OUTPUT=Path('experiments/results/d_seg_small_sampling_v1')
WEIGHTS={'A':2.0,'B':1.5,'C':1.0}
PINS={
 str(C_PROTOCOL):'16718f5956548e002320ebd2ade83c5ecafdf77e7c79dc1f1d2b63818b7e6df3',
 str(BUNDLE/'fuseg_manifest.json'):'69efe88935f1dd064b79ad9f2ceb67e2e358b0144eed985a79087f9459aa87f0',
 str(FORMAL/'protocol.json'):'a804c9dfdc94a63fd40b175a72088e03fd55ceaecf5c1b565318a3379a85a8a1',
 str(FORMAL/'runs/isic_auxiliary_formal/weights/best.pt'):'1ec926026473beafafb1ef8da6aa6efcc13d38c391ec8db48006c5d793d5f2a3',
 str(FORMAL/'runs/fuseg_finetune_formal/weights/best.pt'):'cc2955d088928dc00c90d0ba8cab0aefeb49d4cc45da172b030c150ab0dce97a',
 str(C_OUT/'result.json'):'d797561841227617ad3c8652164d12daa03d63924cada9739c9ef473007ca79e'}


class D0Error(ValueError):
    pass


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)


def require_train(row):
    if row.get('source')!='FUSeg' or row.get('split')!='train' or row.get('role')!='wound_finetuning':
        raise D0Error('FORBIDDEN_SAMPLER_SOURCE_ROLE')
    for key in ('image','label'):
        text=str(row.get(key,'')).replace('\\','/').lower()
        if 'co2' in text or 'locked' in text or 'blind' in text or '..' in text.split('/'):
            raise D0Error('FORBIDDEN_SAMPLER_PATH')
        if not text.startswith(f'fuseg_dataset/{"images" if key=="image" else "labels"}/train/') or len(text.split('/'))!=4:
            raise D0Error('FORBIDDEN_SAMPLER_PATH')


def polygon_geometry(text):
    result=[]
    for line in text.splitlines():
        if not line.strip():continue
        values=np.asarray([float(x) for x in line.split()],dtype=np.float64)
        if len(values)<7 or len(values)%2!=1 or not np.isfinite(values).all() or values[0]!=0:
            raise D0Error('INVALID_TRAINING_POLYGON')
        points=values[1:].reshape(-1,2)
        if np.any(points<0) or np.any(points>1):raise D0Error('INVALID_NORMALIZED_POLYGON')
        width,height=points.max(axis=0)-points.min(axis=0)
        if width<=0 or height<=0:raise D0Error('DEGENERATE_POLYGON_BBOX')
        area=float(width*height)
        result.append({'class_id':0,'bbox_area_ratio':area,
            'size_group':'small' if area<.01 else 'medium' if area<.05 else 'large'})
    return result


def category(areas):
    smallest=min(areas,default=math.inf)
    return 'A' if smallest<.0025 else 'B' if smallest<.01 else 'C'


def build_training_rows(root,source_rows):
    """The only sample-selection input is admitted training GT; no outcomes accepted."""
    result=[]
    for row in sorted(source_rows,key=lambda x:x['image_id']):
        require_train(row)
        image=(root/BUNDLE/row['image']).resolve();label=(root/BUNDLE/row['label']).resolve()
        for path,kind,key in [(image,'images','image_sha256'),(label,'labels','label_sha256')]:
            if path.parent!=(root/BUNDLE/'fuseg_dataset'/kind/'train').resolve():raise D0Error('PATH_ESCAPE')
            if sha(path)!=row[key]:raise D0Error('TRAINING_FILE_HASH_MISMATCH')
        # No image decoder; only normalized polygon label text is parsed.
        gt=polygon_geometry(label.read_text(encoding='utf8'))
        if len(gt)!=row['instances']:raise D0Error('GT_COUNT_MISMATCH')
        areas=[x['bbox_area_ratio'] for x in gt];group=category(areas)
        result.append({'sample_id':row['image_id'],'image_path':image.relative_to(root).as_posix(),
          'label_path':label.relative_to(root).as_posix(),'image_hash':row['image_sha256'],'label_hash':row['label_sha256'],
          'source':'FUSeg','split':'train','role':'wound_finetuning','GT_count':len(gt),
          'minimum_GT_bbox_area_ratio':min(areas) if areas else None,'contains_small':group in ('A','B'),
          'contains_very_small':group=='A','sampling_category':group,'sampling_weight':WEIGHTS[group],
          'instance_geometry':gt,'single_GT':len(gt)==1,'multi_GT':len(gt)>1,
          'exact_content_group':row['image_sha256'],'patient_case_family':None})
    return result


def validate_exclusion(train,val_metadata):
    ids=[r['sample_id'] for r in train];hashes=[r['image_hash'] for r in train]
    if len(set(ids))!=len(ids):raise D0Error('DUPLICATE_TRAINING_ID')
    if set(ids)&{r['sample_id'] for r in val_metadata}:raise D0Error('VALIDATION_ID_IN_SAMPLER')
    if set(hashes)&{r['image_sha256'] for r in val_metadata}:raise D0Error('VALIDATION_HASH_IN_SAMPLER')


def probabilities(rows):
    for row in rows:
        if row['source']!='FUSeg' or row['split']!='train' or row['role']!='wound_finetuning':raise D0Error('FORBIDDEN_SAMPLER_SOURCE_ROLE')
        expected=category([g['bbox_area_ratio'] for g in row['instance_geometry']])
        if row['sampling_category']!=expected or row['sampling_weight']!=WEIGHTS[expected]:raise D0Error('SAMPLING_WEIGHTS_NOT_FROZEN')
    weights=np.array([r['sampling_weight'] for r in rows],dtype=np.float64)
    if not len(weights):raise D0Error('EMPTY_TRAINING_SET')
    return weights/weights.sum()


def epoch_indices(rows,epoch,seed=42):
    if epoch<0 or seed!=42:raise D0Error('UNREGISTERED_SAMPLER_SEED_OR_EPOCH')
    # Stateless epoch-specific stream: independent of augmentation/global RNG.
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,epoch])))
    return rng.choice(len(rows),size=len(rows),replace=True,p=probabilities(rows))


def simulate(rows,epochs=300):
    p=probabilities(rows);n=len(rows);totals=np.zeros(n,dtype=np.int64);epoch_unique=[];epoch_max=[];draw_hash=hashlib.sha256()
    for epoch in range(epochs):
        drawn=epoch_indices(rows,epoch);draw_hash.update(drawn.astype('<i8').tobytes())
        count=np.bincount(drawn,minlength=n);totals+=count;epoch_unique.append(int(np.count_nonzero(count)));epoch_max.append(int(count.max()))
    by_category={};size_exposure={};class_exposure={};source_exposure={}
    for c in WEIGHTS:
        mask=np.array([r['sampling_category']==c for r in rows]);k=int(mask.sum())
        by_category[c]={'images':k,'original_proportion':k/n,'expected_proportion':float(p[mask].sum()),
            'simulated_proportion':float(totals[mask].sum()/(n*epochs)),'simulated_draws':int(totals[mask].sum())}
    def exposure(key,count_fn):
        counts=np.array([count_fn(r) for r in rows]);base=float(counts.sum());expected=float(np.dot(p,counts)*n)
        return {'original_instances_per_epoch':base,'expected_instances_per_epoch':expected,
                'simulated_instances_total':int(np.dot(totals,counts))}
    for s in ['small','medium','large','very_small']:
        size_exposure[s]=exposure(s,lambda r:sum(g['bbox_area_ratio']<.0025 if s=='very_small' else g['size_group']==s for g in r['instance_geometry']))
    class_exposure['0_Wound']=exposure('0',lambda r:r['GT_count'])
    source_exposure['FUSeg']={'original_fraction':1.0,'expected_fraction':1.0,'simulated_fraction':1.0}
    per_image=[{'sample_id':r['sample_id'],'category':r['sampling_category'],'probability':float(p[i]),
         'expected_draws_per_epoch':float(n*p[i]),'expected_draws_total':float(n*epochs*p[i]),
         'simulated_draws_total':int(totals[i])} for i,r in enumerate(rows)]
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['exact_content_group']].append(i)
    group_weights=[{'group_sha256':g,'images':len(indices),'summed_image_weight':sum(rows[i]['sampling_weight'] for i in indices),
      'expected_draws_per_epoch':float(n*p[indices].sum()),'simulated_draws_total':int(totals[indices].sum())} for g,indices in sorted(groups.items())]
    return {'simulation_epochs':epochs,'seed':42,'index_draws':n*epochs,'draw_sequence_sha256':draw_hash.hexdigest(),
      'category_exposure':by_category,'source_exposure':source_exposure,'class_exposure':class_exposure,'GT_size_exposure':size_exposure,
      'multi_GT_exposure':{'original_fraction':sum(r['multi_GT'] for r in rows)/n,'expected_fraction':float(sum(p[i] for i,r in enumerate(rows) if r['multi_GT'])),
                           'explicit_extra_weight':False},
      'negative_image_exposure':{'original_count':sum(r['GT_count']==0 for r in rows),'expected_draws_per_epoch':float(n*sum(p[i] for i,r in enumerate(rows) if r['GT_count']==0))},
      'repetition':{'control_unique_anchor_images_per_epoch':n,'expected_unique_per_epoch':float(sum(1-(1-p)**n)),
        'simulated_mean_unique_per_epoch':float(np.mean(epoch_unique)),
        'simulated_mean_repeated_draws_per_epoch':float(n-np.mean(epoch_unique)),
        'maximum_expected_image_reuse_per_epoch':float(n*p.max()),'maximum_expected_image_reuse_total':float(n*epochs*p.max()),
        'maximum_observed_image_reuse_single_epoch':max(epoch_max),'maximum_observed_image_reuse_total':int(totals.max()),
        'minimum_observed_image_reuse_total':int(totals.min()),'never_seen_images':int((totals==0).sum())},
      'per_image':per_image,'exact_content_group_weights':group_weights,
      'group_dependence':'GROUP_LEVEL_SAMPLING_DEPENDENCE_NOT_VERIFIED: patient/case/derived-image family IDs unavailable; exact-content groups only',
      'scope':'Anchor-image index draws only. Mosaic companion exposure is not counted or simulated; unchanged augmentation may alter realized pixel/instance exposures.',
      'audit_acceptance':bool(n*p.max()<=2.0+1e-12 and totals.max()<=3*epochs and not (totals==0).any() and max(n*p[v].sum() for v in groups.values())<=2.0+1e-12)}


def budget_schedule(args,n):
    batches=math.ceil(n/args['batch']);nw=max(round(args['warmup_epochs']*batches),100)
    last=-1;per_epoch=[];lrs=[]
    for epoch in range(args['epochs']):
        steps=0;factor=((1-math.cos(epoch*math.pi/args['epochs']))/2)*(args['lrf']-1)+1
        for i in range(batches):
            ni=i+batches*epoch
            accumulate=max(1,int(np.interp(ni,[0,nw],[1,args['nbs']/args['batch']]).round())) if ni<=nw else max(round(args['nbs']/args['batch']),1)
            if ni-last>=accumulate:steps+=1;last=ni
        per_epoch.append(steps)
        lrs.append(float(np.interp(ni,[0,nw],[0,args['lr0']*factor])) if ni<=nw else args['lr0']*factor)
    return {'epochs':args['epochs'],'patience':args['patience'],'images_sampled_per_epoch':n,'batch_size':args['batch'],
        'batches_per_epoch':batches,'last_batch_images':n%args['batch'] or args['batch'],
        'total_anchor_image_draws':n*args['epochs'],'total_batches':batches*args['epochs'],
        'gradient_accumulation_after_warmup':max(round(args['nbs']/args['batch']),1),'warmup_batch_iterations':nw,
        'scheduled_optimizer_calls_per_epoch':per_epoch,'total_scheduled_optimizer_calls':sum(per_epoch),
        'epoch_end_lr':lrs,'budget_definition':'Scheduled optimizer_step calls, not batches or confirmed successful AMP parameter updates',
        'AMP_successful_update_count':'NOT_VERIFIED_FROM_HISTORICAL_LOGS',
        'early_stop_rule':'Keep patience=80. If experimental stops before 300 epochs, strict realized-budget comparison is NOT_COMPARABLE; no override, restart or extra epochs.'}


FAIR_FIELDS=['architecture','initialization','dataset_manifest_sha256','training_args','runtime_optimizer','loss','augmentation','budget','evaluation_protocol_sha256']


def compare_configs(control,experimental):
    changed=[key for key in FAIR_FIELDS if control[key]!=experimental[key]]
    if changed:raise D0Error('FAIRNESS_MISMATCH: '+','.join(changed))
    return {key:'IDENTICAL_REGISTERED_SETTING' for key in FAIR_FIELDS}


def output_absent(root):
    if (root/OUTPUT).exists():raise D0Error('D1_OUTPUT_ALREADY_EXISTS')


def run(root):
    root=root.resolve();dest=root/'experiments/protocols';output_absent(root)
    if list(dest.glob(PREFIX+'*')):raise D0Error('D0_ALREADY_EXISTS_REFUSE_OVERWRITE')
    before={}
    def protect(path,expected=None):
        path=Path(path);actual=sha(path)
        if expected and actual!=expected:raise D0Error('PIN_MISMATCH: '+str(path))
        before[str(path)]=actual
        return actual
    for path,expected in PINS.items():protect(root/path,expected)
    cp=read(root/C_PROTOCOL);c_result=read(root/C_OUT/'result.json')['candidate']
    val_path=root/cp['validation_manifest']['path'];protect(val_path,cp['validation_manifest']['sha256']);val=read(val_path)['samples']
    for folder in [root/C_OUT]:
        # Hash-only immutability snapshot; never parse validation outcomes/FN lists.
        for file in folder.rglob('*'):
            if file.is_file():protect(file)
    for file in [root/'docs/PHASE_C1_ERROR_ANALYSIS_20260921.md',root/'woundcare_inference.py',root/cp['metric_implementation']['path'],root/'experiments/review_v2/isic_fuseg_gate.py']:
        protect(file)
    source=read(root/BUNDLE/'fuseg_manifest.json');train_source=[r for r in source if r['split']=='train'];source_val=[r for r in source if r['split']=='val']
    if len(train_source)!=771 or len(source_val)!=191 or len(source)!=962:raise D0Error('DATASET_COUNT_MISMATCH')
    by_id={r['sample_id']:r for r in val}
    if len(by_id)!=191 or any(r['image_id'] not in by_id or r['image_sha256']!=by_id[r['image_id']]['image_sha256'] or r['label_sha256']!=by_id[r['image_id']]['label_sha256'] for r in source_val):raise D0Error('VALIDATION_IDENTITY_MISMATCH')
    rows=build_training_rows(root,train_source);validate_exclusion(rows,val)
    for kind,field,extension in [('images','image_path','*.png'),('labels','label_path','*.txt')]:
        actual={p.resolve() for p in (root/BUNDLE/'fuseg_dataset'/kind/'train').glob(extension)}
        if actual!={(root/r[field]).resolve() for r in rows}:raise D0Error('TRAINING_INVENTORY_MISMATCH')
    for r in rows:
        protect(root/r['image_path'],r['image_hash']);protect(root/r['label_path'],r['label_hash'])
    args_path=root/FORMAL/'runs/fuseg_finetune_formal/args.yaml';args_sha=protect(args_path);args=yaml.safe_load(args_path.read_text(encoding='utf8'))
    lock_path=root/FORMAL/'continue_fuseg.lock';protect(lock_path);lock=read(lock_path)
    protect(root/'experiments/review_v2/isic_fuseg_formal.py',lock['runner_sha256'])
    historical=read(root/FORMAL/'protocol.json')
    for key,value in historical['training'].items():
        if args.get(key)!=value:raise D0Error('BASELINE_ARGS_PROTOCOL_MISMATCH: '+key)
    if any(args[k]!=v for k,v in {'epochs':300,'patience':80,'batch':4,'nbs':64,'seed':42,'resume':False}.items()):raise D0Error('BASELINE_TRAINING_BUDGET_CHANGED')
    initialization=root/FORMAL/'runs/isic_auxiliary_formal/weights/best.pt'
    if Path(args['model']).resolve()!=initialization or lock['recovery']['isic_best_sha256']!=sha(initialization):raise D0Error('BLOCKED_BY_BASELINE_INITIALIZATION_MISMATCH')
    data_path=root/BUNDLE/'fuseg_dataset/dataset.yaml';protect(data_path);data=yaml.safe_load(data_path.read_text(encoding='utf8'))
    if Path(args['data']).resolve()!=data_path or data['names']!={0:'Wound'} or data['train']!='images/train' or data['val']!='images/val' or 'test' in data:raise D0Error('DATASET_CONFIG_CHANGED')
    log_path=root/FORMAL/'continue_fuseg.stdout.log';protect(log_path);log=log_path.read_text(encoding='utf8')
    if 'AdamW(lr=0.0005, momentum=0.937)' not in log or 'Transferred 711/711' not in log or '300 epochs completed' not in log:raise D0Error('BASELINE_RUNTIME_IDENTITY_NOT_VERIFIED')
    history_path=root/FORMAL/'runs/fuseg_finetune_formal/results.csv';protect(history_path)
    with history_path.open(encoding='utf8',newline='') as f:history=list(csv.DictReader(f))
    if [int(float(r['epoch'])) for r in history]!=list(range(1,301)):raise D0Error('BASELINE_EPOCHS_NOT_VERIFIED')
    budget=budget_schedule(args,771)
    for row,lr in zip(history,budget['epoch_end_lr']):
        if any(not math.isclose(float(row[k]),lr,rel_tol=1e-4,abs_tol=1e-10) for k in ['lr/pg0','lr/pg1','lr/pg2']):raise D0Error('LR_SCHEDULE_MISMATCH')
    software={name:metadata.version(name) for name in ['ultralytics','torch','numpy','albumentations']}
    if software['ultralytics']!='8.3.53' or software['torch']!='2.5.1+cu124':raise D0Error('RUNTIME_VERSION_MISMATCH')
    package=Path(metadata.distribution('ultralytics').locate_file('ultralytics'))
    implementations={}
    for file in ['engine/trainer.py','data/build.py','data/augment.py','data/dataset.py','models/yolo/segment/train.py','utils/loss.py','utils/tal.py','cfg/models/11/yolo11-seg.yaml']:
        implementations[file]=protect(package/file)
    scaler_source=Path(metadata.distribution('torch').locate_file('torch/amp/grad_scaler.py'))
    implementations['torch/amp/grad_scaler.py']=protect(scaler_source)
    # TensorBoard tag/step metadata only; no metric values used for sample selection.
    import struct
    from tensorboard.compat.proto.event_pb2 import Event
    event_tags=defaultdict(list)
    for event_path in (root/FORMAL/'runs/fuseg_finetune_formal').glob('events.out.tfevents.*'):
        protect(event_path)
        with event_path.open('rb') as stream:
            while header:=stream.read(12):
                if len(header)!=12:raise D0Error('TRUNCATED_EVENT_FILE')
                length=struct.unpack('<Q',header[:8])[0];event=Event.FromString(stream.read(length));stream.read(4)
                for value in event.summary.value:event_tags[value.tag].append(event.step)
    evidence=root/cp['source']['evidence_path'];protect(evidence,cp['source']['evidence_sha256'])
    simulation=simulate(rows)
    # All definitions, including safety/advancement gates, fixed before any D1 result exists.
    evaluation={k:deepcopy(cp[k]) for k in ['validation_manifest','metric_implementation','color_contract','operating_point','prediction','matching_rule','crop_rule','resize_and_mask_contract','acceptance_gate']}
    evaluation.update({'inherited_phase_c_protocol_sha256':sha(root/C_PROTOCOL),'evaluation_training_difference_allowed':False,
      'gate_script_sha256':protect(root/'experiments/review_v2/isic_fuseg_gate.py'),
      'crop_implementation_sha256':protect(root/'woundcare_inference.py'),
      'test_images_used':0,'CO2Wounds_used':False})
    manifest={'schema':'d0-training-only-gt-v1','sample_count':771,'source_manifest_sha256':sha(root/BUNDLE/'fuseg_manifest.json'),
      'source_admission':{k:v for k,v in cp['source'].items() if k not in ['role','source_id','validation_samples']}|{'role':'TRAIN_ONLY_FUSEG_DEVELOPMENT_TRAIN','source_id':'FUSeg-Train-771','training_samples':771},
      'selection_input':'Training normalized polygon labels only; validation manifest IDs/hashes are exclusion checks only',
      'primary_small_definition':'bbox_area_ratio <0.01','very_small_definition':'bbox_area_ratio <0.0025','samples':rows}
    manifest_path=dest/f'{PREFIX}_training_manifest.json';write(manifest_path,manifest)
    evaluation_path=dest/f'{PREFIX}_evaluation_protocol.json';write(evaluation_path,evaluation)
    baseline={'baseline_experiment_id':'D-Formal-ISIC-FUSeg-seed42-20260914','baseline_checkpoint_path':str(cp['checkpoint']['path']),
      'baseline_checkpoint_sha256':cp['checkpoint']['sha256'],'baseline_training_config_sha256':args_sha,
      'baseline_dataset_manifest_sha256':sha(root/BUNDLE/'fuseg_manifest.json'),
      'baseline_initialization_sha256':sha(initialization),'baseline_initialization_path':initialization.relative_to(root).as_posix(),
      'initialization_semantics':'Same finalized ISIC auxiliary checkpoint; FUSeg starts fresh optimizer, resume=False. No FUSeg-trained continuation; no ISIC retraining.',
      'corrected_result_path':(C_OUT/'result.json').as_posix(),'corrected_result_sha256':sha(root/C_OUT/'result.json'),
      'epochs_completed':300,'runtime_evidence':{'log_sha256':sha(log_path),'history_sha256':sha(history_path),'continuation_lock_sha256':sha(lock_path)}}
    normalized_args={k:v for k,v in args.items() if k not in ['model','data','project','name','save_dir']}
    augmentation={k:args[k] for k in ['hsv_h','hsv_s','hsv_v','degrees','translate','scale','shear','perspective','flipud','fliplr','bgr','mosaic','mixup','copy_paste','copy_paste_mode','close_mosaic','auto_augment','erasing','crop_fraction']}
    augmentation['runtime_albumentations']='Blur p=.01 limit3-7; MedianBlur p=.01 limit3-7; ToGray p=.01 weighted_average 3channels; CLAHE p=.01 clip1-4 grid8x8 (historical log)'
    control={'architecture':{'name':'YOLO11m-seg','class_names':{0:'Wound'},'runtime_layers':445,'runtime_parameters':22359987,'source_hashes':implementations},
      'initialization':{'path':baseline['baseline_initialization_path'],'sha256':baseline['baseline_initialization_sha256'],'resume':False},
      'dataset_manifest_sha256':sha(manifest_path),'training_args':normalized_args,
      'runtime_optimizer':{'name':'AdamW','initial_lr_all_groups':.0005,'betas':[.937,.999],'decay_weight_group':.0005,'decay_bias_norm_groups':0.0,'nbs':64,'gradient_clip_norm':10.0,'amp':True,'lr_schedule':'cosine +5 epochs warmup; all 300 CSV LR entries verified'},
      'loss':{'implementation':'ultralytics.utils.loss.v8SegmentationLoss','implementation_sha256':implementations['utils/loss.py'],'box':7.5,'seg_gain':'same box gain 7.5','cls':.5,'dfl':1.5,'overlap_mask':True,'mask_ratio':4},
      'augmentation':augmentation,'budget':budget,'evaluation_protocol_sha256':sha(evaluation_path),'software':software,
      'sampler':'historical shuffled RandomSampler without replacement; each of 771 anchors once per epoch'}
    experimental=deepcopy(control);experimental['sampler']='Frozen weighted categorical anchor draws with replacement; 771 per epoch, independent PCG64 SeedSequence([42,epoch])'
    fairness=compare_configs(control,experimental)
    sampling={'algorithm':'numpy Generator(PCG64(SeedSequence([42,zero_based_epoch]))).choice(N,size=N,replace=True,p=normalized_weights)',
      'weights':WEIGHTS,'weight_formula':'minimum unaugmented training GT bbox area; no multi-GT bonus',
      'replacement':True,'epoch_sample_count':771,'seed':42,'numpy_version':software['numpy'],'training_manifest_sha256':sha(manifest_path),
      'categories':{'A':'at least one bbox area <0.25%','B':'at least one bbox area <1%, none <0.25%','C':'no bbox area <1%, including empty labels'},
      'expected_category_distribution':simulation['category_exposure'],
      'D1_integration_contract':'Map sampler index by frozen sample_id, not assumed dataset order. Epoch plans must survive InfiniteDataLoader prefetch/reset and close_mosaic without skipping/duplicating anchors. Do not change companion-image selection or augmentation.',
      'simulation_acceptance_rules':{'maximum_expected_reuse_per_image_per_epoch':2.0,'maximum_expected_reuse_per_exact_content_group_per_epoch':2.0,'maximum_observed_reuse_total_multiplier_vs_control':3.0,'never_seen_after_300_epochs':0},
      'no_new_training_authorization':True}
    baseline_p=209/245;baseline_f1=418/486
    gate={'primary_endpoint':'Small Recall','combine':'ALL_REQUIRED','small_TP_min':110,'small_support':137,
      'precision_min':baseline_p-.01,'overall_F1_min':baseline_f1-.01,'crop_complete_min_count':166,'crop_positive_denominator':186,
      'medium_TP_min':85,'medium_support':90,'large_TP_min':14,'large_support':14,
      'medium_large_safety_definition':'At most one additional medium FN, no additional large FN. Fixed conservative definition, not significance inference.',
      'comparison_arithmetic':'Use unrounded fractions; no baseline rounding at gate boundaries',
      'secondary_endpoints':['overall_precision','overall_recall','overall_F1','medium_recall','large_recall','crop_complete95','FP_count','FN_count','no_ROI_positive','very_small_recall','five_small_bins','single_vs_multi_recall_and_crop_failure'],
      'very_small_baseline':{'TP':29,'support':49,'recall':29/49},
      'original_development_gate':deepcopy(cp['acceptance_gate']),'deployment_authorization':False}
    # Historical AMP skips were not logged. Do not conflate scheduled calls with verified successful parameter updates.
    blockers=['BLOCKED_BY_ACTUAL_OPTIMIZER_UPDATE_COUNT_NOT_VERIFIED']
    if not simulation['audit_acceptance']:blockers.append('BLOCKED_BY_SAMPLING_DISTRIBUTION_AUDIT')
    protocol={'experiment_id':PREFIX,'phase':'D0','status':'FROZEN_PREREGISTRATION_BLOCKED','hypotheses':{
      'H1':'Small-object-aware image sampling improves small recall relative to frozen baseline',
      'R1':'Oversampling may increase FP or reduce Precision','R2':'Small recall improvement may not improve multi-instance crop completeness'},
      'baseline':baseline,'fairness_registered_settings':fairness,'only_proposed_training_difference':'anchor-image sampling distribution',
      'advancement_gate':gate,'budget_fairness_status':'Scheduled budgets identical; actual historical AMP-applied update count NOT_VERIFIED',
      'blockers':blockers,'training_output_dir':OUTPUT.as_posix(),'output_immutability':'D1 must require absent target, acquire exclusive lock before work, refuse overwrite/resume. No D1 output exists in D0.',
      'training_performed':False,'model_inference':False,'locked_test_used':False,'CO2Wounds_used':False,
      'READY_FOR_PHASE_D1_SINGLE_SEED_TRAINING':'NO','STOP_AFTER_D0':True,
      'D1_authorization':'Separate explicit user authorization required even after all blockers resolved; frozen V1 cannot be silently amended.'}
    for suffix,doc in [('sampling_protocol',sampling),('training_config',{'control':control,'experimental':experimental,'baseline':baseline}),('sampling_simulation',simulation),('protocol',protocol)]:
        write(dest/f'{PREFIX}_{suffix}.json',doc)
    output_absent(root)
    changed=[p for p,h in before.items() if sha(p)!=h]
    if changed:raise D0Error('PROTECTED_INPUT_CHANGED: '+str(changed))
    audit={'status':'PASS_AUDIT_WITH_FAIRNESS_BLOCKER','source_files_unchanged':len(before),'protected_before':before,'protected_after':before,
      'train_images':771,'validation_metadata_images':191,'exact_train_val_hash_overlap':0,'validation_ids_in_sampler':0,
      'training_pixels_decoded':0,'validation_pixels_read':0,'test_images_used':0,'CO2Wounds_used':False,
      'validation_outcomes_read_by_sampler':False,'model_loaded':False,'training_performed':False,'inference_performed':False,
      'group_status':simulation['group_dependence'],'D1_output_exists':False,
      'actual_optimizer_update_evidence':'AMP scaler.step may skip updates on non-finite gradients. Historical epoch CSV records loss/LR but not skipped/applied steps; stripped final checkpoints must not be loaded in D0.',
      'tensorboard_tag_step_inventory':{tag:{'events':len(steps),'min_step':min(steps),'max_step':max(steps)} for tag,steps in sorted(event_tags.items())},
      'scheduled_budget':budget,'simulation_acceptable':simulation['audit_acceptance']}
    write(dest/f'{PREFIX}_audit.json',audit)
    artifacts={p.name:sha(p) for p in sorted(dest.glob(PREFIX+'*.json'))}
    write(dest/f'{PREFIX}_freeze.json',{'status':'FROZEN_BLOCKED_NO_D1_AUTHORIZATION','artifacts_sha256':artifacts,
         'analysis_code_sha256':sha(Path(__file__)),'training_output_absent':True})
    return {'PHASE_D0_STATUS':'BLOCKED','READY_FOR_PHASE_D1_SINGLE_SEED_TRAINING':'NO','blockers':blockers,
      'category_exposure':simulation['category_exposure'],'repetition':simulation['repetition'],
      'total_scheduled_optimizer_calls':budget['total_scheduled_optimizer_calls'],'protected_files_unchanged':len(before)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args()
    print(json.dumps(run(Path(__file__).resolve().parents[1]),ensure_ascii=False,indent=2))
