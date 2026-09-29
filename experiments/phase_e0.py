"""Saved-evidence-only very-small audit. No model loading, inference or training."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import importlib.abc
import json
from pathlib import Path
import sys
import numpy as np
from experiments import phase_d2 as d2
from experiments.phase_c1 import bbox_features, small_bin, stats, iou
from experiments.phase_c_execution import assert_relative_admitted, validate_result_integrity

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('experiments/results/very_small_failure_audit')
REPORT=Path('docs/PHASE_E0_VERY_SMALL_FAILURE_AUDIT_20260925.md')
SEEDS=(42,123,3407,2026,999)
MODELS=[f'{a}{s}' for s in SEEDS for a in ('C','S')]
CATEGORIES=['PERSISTENTLY_MISSED','SAMPLING_RESPONSIVE','SAMPLING_HARMED','SEED_UNSTABLE','CONSISTENTLY_DETECTED']
RULES={
 'scope':'All 49 GT bbox instances with area/image_area <0.0025 in fixed FUSeg development validation; no case selection',
 'identity':'sample_id + zero-based GT index in original frozen polygon-label order; cross-model boxes must be identical',
 'matching':'Saved pairs are authoritative; verify using unchanged frozen matching at IoU 0.50, conf 0.10. No threshold sweep or new predictions.',
 'category_precedence':['all_10_count==10: CONSISTENTLY_DETECTED','all_10_count<=1: PERSISTENTLY_MISSED',
   'S_count>C_count: SAMPLING_RESPONSIVE','S_count<C_count: SAMPLING_HARMED','otherwise: SEED_UNSTABLE'],
 'near_always_missed':'at least 9/10 misses; also report strict 10/10 misses and full histogram; cutoff defined before E0 matrix derivation',
 'net_effect_note':'Responsive/harmed are descriptive net count labels, not causal claims; category precedence can mask a one-hit gain/loss, so all net counts and paired gains/losses are also reported.',
 'exclusive_S_only':'C_count=0 and S_count>0, regardless of primary category; not assumed reproducible',
 'image_features':{'brightness':'whole-image cv2 RGB2GRAY uint8 mean (0..255)',
   'contrast':'whole-image grayscale population SD', 'sharpness':'whole-image cv2 Laplacian CV_64F variance, ksize=1',
   'local_contrast':'absolute difference of grayscale mean inside GT bbox and a fixed 8-pixel outer rectangular ring; excludes ALL image GT bboxes from ring. Bbox proxy, not instance tissue/mask contrast.',
   'local_CNR':'same absolute mean difference / ring SD; undefined when no ring pixels or zero ring SD',
   'rounding':'floor left/top, ceil right/bottom, clipped to image; no image rescaling or model crop change'},
 'saved_miss_diagnostics':['matched => DETECTED','unmatched with retained pred IoU>=0.5 => MATCH_COMPETITION',
   'else saved floor pred conf in [0.01,0.10) with IoU>=0.5 => BELOW_FROZEN_CONFIDENCE',
   'else retained overlap IoU>0 => RETAINED_LOCALIZATION_BELOW_IOU',
   'else NO_RETAINED_OVERLAP'],
 'diagnostic_limits':'These are geometric associations at existing frozen thresholds, not rematching outcomes or measured recall under another threshold. Floor outputs are post-NMS, so suppressed/pre-floor proposals are unobservable.',
 'statistics':'descriptive only; same images, dependent GT/model observations; no CI, p-value, causal interpretation or independent n=490',
 'exposure':'Consumed unaugmented anchor identities/GT, not mosaic companion or effective lesion pixels; separately count anchors containing very-small/small GT and GT-instance exposure.',
 'NEW_TRAINING_AUTHORIZED':'NO','NEW_INFERENCE_AUTHORIZED':'NO','APP_MODEL_REPLACEMENT_AUTHORIZED':'NO',
 'EXTERNAL_TEST_AUTHORIZED':'NO','test_images_used':0,'CO2Wounds_used':False,'STOP_AFTER_E0':True}

class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','ultralytics','tensorflow','onnxruntime'}:
            raise RuntimeError('E0_FORBIDDEN_MODEL_IMPORT: '+fullname)

def require(ok,message):
    if not ok:raise RuntimeError('E0_BLOCKED: '+message)

def read(path):return d2.read(path)
def write(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def table(path,rows):
    require(bool(rows),'empty required table '+path.name)
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with path.open(encoding='utf-8-sig',newline='') as f:
        saved=list(csv.DictReader(f))
    require(len(saved)==len(rows) and list(saved[0])==list(rows[0]),'CSV roundtrip')

def classify(c,s):
    require(type(c) is int and type(s) is int and 0<=c<=5 and 0<=s<=5,'invalid detection count')
    if c+s==10:return 'CONSISTENTLY_DETECTED'
    if c+s<=1:return 'PERSISTENTLY_MISSED'
    if s>c:return 'SAMPLING_RESPONSIVE'
    if s<c:return 'SAMPLING_HARMED'
    return 'SEED_UNSTABLE'

def pattern(c,s):
    require(len(c)==len(s)==5 and all(x in (0,1) for x in c+s),'invalid detection vector')
    gain=sum(a==0 and b==1 for a,b in zip(c,s));loss=sum(a==1 and b==0 for a,b in zip(c,s))
    return {'control_detection_count':sum(c),'experimental_detection_count':sum(s),'all_10_detection_count':sum(c+s),
      'net_detection_count_delta':sum(s)-sum(c),'paired_gain_seeds':gain,'paired_loss_seeds':loss,
      'mixed_gain_and_loss':int(gain>0 and loss>0),'S_only_detected':int(sum(c)==0 and sum(s)>0),
      'C_only_detected':int(sum(s)==0 and sum(c)>0),'category':classify(sum(c),sum(s))}

def verify_row(row,gt,matcher):
    require(row['gt_boxes']==gt,'cross-model GT identity/order')
    recomputed=matcher(gt,row['pred_boxes'],row['confidences'],.5)
    require(recomputed['pairs']==row['pairs'],'saved matching differs from frozen matcher')
    require(all(recomputed[k]==row[k] for k in ('tp','fp','fn')),'matching counts')
    matched={p['gt'] for p in row['pairs']}
    require(len(row['per_instance_gt'])==len(gt),'instance flags length')
    for i,g in enumerate(row['per_instance_gt']):
        require(g['index']==i and g['bbox']==gt[i] and g['matched']==(i in matched),'instance flags')
    require(all(c>=.1 for c in row['confidences']),'retained confidence')
    idx=row['retained_mask_indices']
    require(idx==[i for i,c in enumerate(row['floor_confidences']) if c>=.1],'retention identity')
    require([row['floor_bboxes'][i] for i in idx]==row['pred_boxes'],'floor/retained boxes')

def miss_diagnostic(row,j):
    if j in {p['gt'] for p in row['pairs']}:return 'DETECTED'
    overlaps=[iou(row['gt_boxes'][j],b) for b in row['pred_boxes']]
    if max(overlaps,default=0)>=.5:return 'MATCH_COMPETITION'
    if any(.01<=c<.1 and iou(row['gt_boxes'][j],b)>=.5 for c,b in zip(row['floor_confidences'],row['floor_bboxes'])):
        return 'BELOW_FROZEN_CONFIDENCE'
    return 'RETAINED_LOCALIZATION_BELOW_IOU' if max(overlaps,default=0)>0 else 'NO_RETAINED_OVERLAP'

def local_features(rgb,box,all_boxes):
    import cv2
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);h,w=gray.shape
    def bounds(b):return max(0,int(np.floor(b[0]))),max(0,int(np.floor(b[1]))),min(w,int(np.ceil(b[2]))),min(h,int(np.ceil(b[3])))
    x1,y1,x2,y2=bounds(box);lesion=gray[y1:y2,x1:x2]
    require(lesion.size>0,'empty bbox pixels')
    ring=np.zeros_like(gray,dtype=bool);ring[max(0,y1-8):min(h,y2+8),max(0,x1-8):min(w,x2+8)]=True
    for b in all_boxes:
        a,c,e,f=bounds(b);ring[c:f,a:e]=False
    background=gray[ring];delta=abs(float(lesion.mean())-float(background.mean())) if len(background) else None
    return {'brightness_gray_mean_0_255':float(gray.mean()),'contrast_gray_SD_0_255':float(gray.std()),
      'sharpness_laplacian_variance':float(cv2.Laplacian(gray,cv2.CV_64F).var()),
      'bbox_brightness_0_255':float(lesion.mean()),'bbox_contrast_SD':float(lesion.std()),
      'ring_pixel_count':len(background),'local_bbox_background_abs_mean_difference':delta,
      'local_bbox_background_CNR':delta/float(background.std()) if len(background) and background.std()>0 else None}

def aggregate(items):
    n=len(items);miss=sum(10-r['all_10_detection_count'] for r in items)
    return {'GT_instances':n,'images':len({r['sample_id'] for r in items}),
      'categories':{c:sum(r['category']==c for r in items) for c in CATEGORIES},
      'detection_count_histogram':{str(i):sum(r['all_10_detection_count']==i for r in items) for i in range(11)},
      'strict_all_10_missed':sum(r['all_10_detection_count']==0 for r in items),
      'net_positive_GT':sum(r['net_detection_count_delta']>0 for r in items),
      'net_negative_GT':sum(r['net_detection_count_delta']<0 for r in items),
      'S_only_detected_GT':sum(r['S_only_detected'] for r in items),'C_only_detected_GT':sum(r['C_only_detected'] for r in items),
      'mixed_gain_loss_GT':sum(r['mixed_gain_and_loss'] for r in items),
      'model_GT_miss_observations':miss,'model_GT_opportunities':10*n,
      'miss_fraction':miss/(10*n) if n else None,
      'mean_delta_TP_per_seed':sum(r['net_detection_count_delta'] for r in items)/5,
      'mean_delta_recall_pp':100*sum(r['net_detection_count_delta'] for r in items)/(5*n) if n else None}

def derive(predictions):
    first=predictions['C42'];matrix=[];mechanisms=[]
    for sid in sorted(first):
        row=first[sid]
        for j,box in enumerate(row['gt_boxes']):
            g=bbox_features(box)
            if g['bbox_area_ratio']>=.0025:continue
            flags={key:int(j in {p['gt'] for p in predictions[key][sid]['pairs']}) for key in MODELS}
            entry={'sample_id':sid,'GT_instance_id':j,'instance_key':f'{sid}::GT{j}',
              'bbox_area_ratio':g['bbox_area_ratio'],'size_bin':small_bin(g['bbox_area_ratio']),
              **{key+'_detected':flags[key] for key in MODELS},
              **pattern([flags[f'C{s}'] for s in SEEDS],[flags[f'S{s}'] for s in SEEDS]),
              **g,'GT_count_per_image':len(row['gt_boxes']),'multi_GT':int(len(row['gt_boxes'])>1),
              'bbox_x1':box[0],'bbox_y1':box[1],'bbox_x2':box[2],'bbox_y2':box[3]}
            matrix.append(entry)
            for model in MODELS:
                mechanisms.append({'instance_key':entry['instance_key'],'sample_id':sid,'GT_instance_id':j,
                  'model':model,'size_bin':entry['size_bin'],'category':entry['category'],
                  'diagnostic':miss_diagnostic(predictions[model][sid],j)})
    return matrix,mechanisms

def exposure_rows(root,pairs,train_rows):
    by_id={r['sample_id']:r for r in train_rows};out=[]
    for pair in pairs:
        seed=pair['seed'];result={'seed':seed};armdata={}
        for arm in ('C','S'):
            folder=root/(d2.d1.paired.OUTPUTS[arm] if seed==42 else d2.seeded.outputs(seed)[arm])
            counts=Counter()
            for b in d2.d1.read_jsonl(folder/'anchors.jsonl'):
                for a in b['anchors']:
                    require(a['sample_id'] in by_id,'non-training anchor');counts[a['sample_id']]+=1
            t=pair['training'][arm]
            require(sum(counts.values())==t['anchor_draws'] and len(counts)==t['unique_anchors_seen'],'anchor totals')
            tiny={sid for sid,r in by_id.items() if any(g['bbox_area_ratio']<.0025 for g in r['instance_geometry'])}
            small={sid for sid,r in by_id.items() if any(g['bbox_area_ratio']<.01 for g in r['instance_geometry'])}
            armdata[arm]={'very_small_anchor_draws':sum(counts[s] for s in tiny),'small_anchor_draws':sum(counts[s] for s in small),
              'unique_images':len(counts),'unique_very_small_images':len(set(counts)&tiny),
              'repeat_draws':sum(counts.values())-len(counts),'within_epoch_repeats':0,
              'very_small_GT_exposure':t['exposure']['very_small_GT'],'small_GT_exposure':t['exposure']['small_GT'],
              'applied_updates':t['applied_optimizer_updates'],'skipped_updates':t['skipped_optimizer_updates']}
            for e in range(300):
                indices=d2.seeded.indices(train_rows,arm,e,seed)
                armdata[arm]['within_epoch_repeats']+=len(indices)-len(set(indices.tolist()))
            for k,v in armdata[arm].items():result[arm+'_'+k]=v
            result[arm+'_validation_very_small_TP']=pair['evaluations'][arm]['diagnostics']['very_small']['TP']
        for k in armdata['C']:result['delta_'+k]=armdata['S'][k]-armdata['C'][k]
        result['delta_validation_very_small_TP']=result['S_validation_very_small_TP']-result['C_validation_very_small_TP']
        result['delta_validation_very_small_recall_pp']=100*result['delta_validation_very_small_TP']/49
        out.append(result)
    return out

def figures(folder,matrix,images):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    folder.mkdir();inventory=[]
    ordered=sorted(matrix,key=lambda r:(CATEGORIES.index(r['category']),r['size_bin'],r['sample_id'],r['GT_instance_id']))
    fig,ax=plt.subplots(figsize=(10,16));arr=np.array([[r[k+'_detected'] for k in MODELS] for r in ordered])
    ax.imshow(arr,aspect='auto',cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
    ax.set_xticks(range(10),MODELS,rotation=45);ax.set_yticks(range(len(ordered)),[f"{r['sample_id']} G{r['GT_instance_id']} ({r['control_detection_count']}/{r['experimental_detection_count']})" for r in ordered],fontsize=7)
    ax.set_title('All very-small GT: frozen detections (blue=1, white=0)\nLabels end in C/S detection counts out of five');fig.tight_layout();fig.savefig(folder/'detection_matrix.png',dpi=140);plt.close(fig)
    for start in range(0,len(ordered),9):
        batch=ordered[start:start+9];fig,axes=plt.subplots(3,3,figsize=(12,13))
        for ax in axes.flat:ax.axis('off')
        for ax,r in zip(axes.flat,batch):
            ax.imshow(images[r['sample_id']]);b=[r[k] for k in ('bbox_x1','bbox_y1','bbox_x2','bbox_y2')]
            ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='#ffff00',linewidth=2))
            ax.set_title(f"{r['sample_id']} G{r['GT_instance_id']} {r['size_bin']}\n{r['category']}\nC={r['control_detection_count']}/5 S={r['experimental_detection_count']}/5",fontsize=8)
        fig.suptitle('All instances, rule-sorted; yellow=frozen target GT; display only',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.96));name=f'all_instances_{start//9+1:02d}.png';fig.savefig(folder/name,dpi=120);plt.close(fig)
        inventory.append({'file':name,'instance_keys':[r['instance_key'] for r in batch]})
    require([k for p in inventory for k in p['instance_keys']]==[r['instance_key'] for r in ordered],'figure coverage')
    return inventory

def run(root):
    require(not any(m.split('.')[0] in {'torch','ultralytics','tensorflow','onnxruntime'} for m in sys.modules),'model library already loaded')
    sys.meta_path.insert(0,NoModels())
    out=root/OUT;require(not out.exists() and not (root/REPORT).exists(),'output exists; no overwrite')
    d2.verify(root)
    summary=read(root/d2.MASTER/'multiseed_summary.json')
    require(summary['PHASE_D2_STATUS']=='COMPLETE' and summary['VALID_PAIRED_SEEDS']==5 and summary['MULTISEED_SMALL_SAMPLING_CONFIRMATION']=='FAIL','D2 prerequisite')
    pairs=summary['pairs'];require([p['seed'] for p in pairs]==list(SEEDS),'paired seed identity')
    require(d2.confirmation(pairs)['small_paired_effect']==summary['small_paired_effect'],'D2 recomputation')
    out.mkdir();write(out/'analysis_protocol.json',{'frozen_at':d2.d1.now(),'rules':RULES,'code_sha256':d2.sha(Path(__file__))})
    protected={}
    folders=[root/d2.MASTER,root/d2.d1.PAIR]+[root/p for p in d2.d1.paired.OUTPUTS.values()]
    folders += [root/p for s in SEEDS[1:] for p in d2.seeded.outputs(s).values()]
    for folder in folders:
        for p in folder.rglob('*'):
            if p.is_file():protected[str(p)]=d2.sha(p)
    protected[str(root/d2.REPORT)]=d2.sha(root/d2.REPORT)
    write(out/'source_snapshot_before.json',protected)
    from experiments.review_v2 import localization_benchmark as loc
    ep,cohort=d2.d1.verify_validation(root);manifest=read(root/ep['validation_manifest']['path'])['samples']
    truth={r['image_id']:loc.ground_truth(r)[0].tolist() for r in cohort}
    train=read(root/d2.d1.load(root,'control_config')['manifest']['path'])['samples']
    predictions={}
    for pair in pairs:
        seed=pair['seed'];driver=d2.d1 if seed==42 else d2.delegated_driver(root,seed)
        cfgs=(d2.d1.load(root,'control_config'),d2.d1.load(root,'experimental_config')) if seed==42 else d2.configs(root,seed)
        for arm,cfg in zip(('C','S'),cfgs):
            folder=root/driver.paired.OUTPUTS[arm];print(f'Verify saved {arm}{seed}',flush=True)
            driver.verify_saved_arm(root,arm,cfg)
            ev=read(folder/'final_evaluation.json');require(ev==pair['evaluations'][arm],'D2 evaluation lineage')
            require(read(folder/'training_completion.json')==pair['training'][arm],'training lineage')
            rows=read(folder/'per_image_predictions.json')
            validate_result_integrity(rows,ev['candidate'],[r['sample_id'] for r in manifest])
            for row in rows:verify_row(row,truth[row['sample_id']],loc.match_boxes)
            predictions[arm+str(seed)]={r['sample_id']:r for r in rows}
    matrix,mechanisms=derive(predictions)
    require(len(matrix)==49 and Counter(r['size_bin'] for r in matrix)=={'<0.10%':27,'0.10-0.25%':22},'very-small denominator')
    from PIL import Image
    images={};by_id={r['sample_id']:r for r in manifest};geometry=[]
    for r in matrix:
        sid=r['sample_id'];sample=by_id[sid]
        path=assert_relative_admitted(root,sample['relative_image_path'],root/'outputs/fuseg_warmup_revision_20260914/dataset/images/val')
        require(d2.sha(path)==sample['image_sha256'],'image hash')
        if sid not in images:
            with Image.open(path) as im:
                require(im.size==(512,512),'image dimensions');images[sid]=np.array(im.convert('RGB'))
        box=truth[sid][r['GT_instance_id']]
        geometry.append({**r,**local_features(images[sid],box,truth[sid])})
    exposures=exposure_rows(root,pairs,train)
    stability=[]
    for p in pairs:
        s=p['seed'];row={'seed':s}
        for b in ('<0.10%','0.10-0.25%','ALL_VERY_SMALL'):
            g=[r for r in matrix if b=='ALL_VERY_SMALL' or r['size_bin']==b]
            c=sum(r[f'C{s}_detected'] for r in g);e=sum(r[f'S{s}_detected'] for r in g)
            stability.append({'seed':s,'size_bin':b,'GT_support':len(g),'C_TP':c,'S_TP':e,'delta_TP':e-c,'delta_recall_pp':100*(e-c)/len(g),
              'paired_gains':sum(not r[f'C{s}_detected'] and r[f'S{s}_detected'] for r in g),
              'paired_losses':sum(r[f'C{s}_detected'] and not r[f'S{s}_detected'] for r in g)})
            if b=='ALL_VERY_SMALL':require(c==p['evaluations']['C']['diagnostics']['very_small']['TP'] and e==p['evaluations']['S']['diagnostics']['very_small']['TP'],'reconcile D2 tiny TP')
    groupings={'all':matrix,**{b:[r for r in matrix if r['size_bin']==b] for b in ('<0.10%','0.10-0.25%')},
      'single_GT':[r for r in matrix if not r['multi_GT']],'multi_GT':[r for r in matrix if r['multi_GT']]}
    strata={k:aggregate(v) for k,v in groupings.items()}
    for b in ('<0.10%','0.10-0.25%'):
        require(abs(strata[b]['mean_delta_recall_pp']/100-summary['small_bin_mean_paired_effects'][b]['mean_delta_recall'])<1e-12,'D2 bin reconciliation')
    features=['bbox_width_pixels','bbox_height_pixels','bbox_area_ratio','aspect_ratio','nearest_border_distance_normalized','GT_count_per_image',
      'brightness_gray_mean_0_255','contrast_gray_SD_0_255','sharpness_laplacian_variance','local_bbox_background_abs_mean_difference','local_bbox_background_CNR']
    geomsummary={}
    for group in ['ALL',*CATEGORIES,'<0.10%','0.10-0.25%']:
        rows=[r for r in geometry if group=='ALL' or r['category']==group or r['size_bin']==group]
        geomsummary[group]={k:stats([r[k] for r in rows if r[k] is not None]) for k in features}
    correlations={}
    y=[r['delta_validation_very_small_TP'] for r in exposures]
    for k in ('delta_very_small_anchor_draws','delta_very_small_GT_exposure','delta_small_anchor_draws','delta_unique_images','delta_repeat_draws'):
        x=[r[k] for r in exposures]
        correlations[k]={'n_paired_seeds':5,'pearson_r_descriptive':float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None,
          'undefined_reason':'constant exposure delta' if np.std(x)==0 else None,'causal':False}
    result={'PHASE_E0_STATUS':'ANALYSIS_READY','rules':RULES,'strata':strata,'geometry_summary':geomsummary,
      'saved_miss_diagnostics':{k:dict(Counter(r['diagnostic'] for r in mechanisms if k=='all' or r['size_bin']==k)) for k in ('all','<0.10%','0.10-0.25%')},
      'exposure_analysis':exposures,'descriptive_correlations':correlations,'seed_stability':stability,
      'source_integrity':'PASS','models':MODELS,'new_inference_performed':False,'training_performed':False,
      'NEW_TRAINING_AUTHORIZED':'NO','APP_MODEL_REPLACEMENT_AUTHORIZED':'NO','EXTERNAL_TEST_AUTHORIZED':'NO',
      'test_images_used':0,'CO2Wounds_used':False,'STOP_AFTER_E0':True}
    for name,rows in [('persistent_failure_matrix',matrix),('seed_stability',stability),('geometry_analysis',geometry),('training_exposure',exposures),('saved_miss_diagnostics',mechanisms)]:table(out/(name+'.csv'),rows)
    result['visualization_inventory']=figures(out/'failure_visualizations',matrix,images)
    for p,h in protected.items():require(d2.sha(p)==h,'historical mutation: '+p)
    d2.verify(root)
    require(not any(m.split('.')[0] in {'torch','ultralytics','tensorflow','onnxruntime'} for m in sys.modules),'model import')
    write(out/'integrity.json',{'status':'PASS','protected_files_unchanged':len(protected),'saved_models_verified':10,
      'GT_instances':49,'saved_model_GT_observations':490,'independent_n_not_490':True,
      'csv_row_counts':{'persistent_failure_matrix':len(matrix),'seed_stability':len(stability),'geometry_analysis':len(geometry),'training_exposure':len(exposures),'saved_miss_diagnostics':len(mechanisms)},
      'all_case_visualization_coverage':49,'model_import_guard':'PASS','model_inference_performed':False,'training_performed':False})
    write(out/'analysis_results.json',result)
    print(json.dumps({'status':'ANALYSIS_READY','strata':strata,'diagnostics':result['saved_miss_diagnostics'],'exposure':exposures},indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute-audit',action='store_true');args=parser.parse_args()
    if not args.execute_audit:parser.error('Explicit --execute-audit required')
    try:run(ROOT)
    except Exception as exc:
        if (ROOT/OUT).exists() and not (ROOT/OUT/'blocked.json').exists():write(ROOT/OUT/'blocked.json',{'PHASE_E0_STATUS':'BLOCKED','reason':str(exc),'training_performed':False})
        raise
