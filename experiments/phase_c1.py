"""Descriptive analysis of immutable Phase C evidence, without model imports.

Saved matching is authoritative. IoU below is a diagnostic for unmatched
predictions, never a replacement matching procedure or threshold sweep.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np

from experiments.phase_c0 import sha256_file
from experiments.phase_c_execution import (
    PINNED_PROTOCOL_SHA256, PINNED_CHECKPOINT_SHA256, PINNED_MANIFEST_SHA256,
    PROTOCOL_REL, OUTPUT_REL, read_json, check_digest, validate_result_integrity,
    assert_relative_admitted,
)

EXPECTED = {'TP':209,'FP':36,'FN':32,'GT':241,'small':137,'medium':90,'large':14,
            'positive':186,'negative':5,'crop_pass':167,'crop_fail':19,'no_roi_positive':4}
RESULT_SHA = 'd797561841227617ad3c8652164d12daa03d63924cada9739c9ef473007ca79e'
ROWS_SHA = '1197dea64a089f9c86d0e710ab19fe4bbb199f1995a73441eb6520a147f5e9e4'
SMALL_BINS = ['<0.10%','0.10-0.25%','0.25-0.50%','0.50-0.75%','0.75-1.00%']
SEVERITY_BINS = ['0%','>0-25%','>25-50%','>50-75%','>75-90%','>90-95%','>=95%']


class C1Error(RuntimeError):
    pass


def verify_expected_counts(counts):
    if counts != EXPECTED:
        raise C1Error('FAIL_PHASE_C1_INPUT_INTEGRITY: totals differ from immutable Phase C')


def stats(values):
    a=np.asarray(values,dtype=float)
    if not len(a):
        return dict.fromkeys(['mean','median','p25','p75','min','max']) | {'n':0}
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
            'p25':float(np.quantile(a,.25)),'p75':float(np.quantile(a,.75)),
            'min':float(a.min()),'max':float(a.max())}


def bbox_features(box):
    x1,y1,x2,y2=map(float,box); width=x2-x1; height=y2-y1
    return {'bbox_width_pixels':width,'bbox_height_pixels':height,'bbox_area_pixels':width*height,
            'bbox_width_ratio':width/512,'bbox_height_ratio':height/512,'bbox_area_ratio':width*height/(512*512),
            'image_width':512,'image_height':512,'aspect_ratio':width/height if height else None,
            'nearest_border_distance_normalized':min(x1/512,(512-x2)/512,y1/512,(512-y2)/512)}


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return float(intersection/max(union,1e-12))


def classify_fp(row, index):
    box=row['pred_boxes'][index]
    overlaps=[iou(box,gt) for gt in row['gt_boxes']]
    maximum=max(overlaps,default=0.0)
    matched={p['gt'] for p in row['pairs']}
    duplicate=any(value>=.5 and i in matched for i,value in enumerate(overlaps))
    if row['gt_pixels']==0:
        primary,tag='FP_NEGATIVE_IMAGE','FP_NEGATIVE_IMAGE'
    elif duplicate:
        primary,tag='FP_DUPLICATE_NEAR_GT','FP_POSSIBLE_DUPLICATE'
    elif 0<maximum<.5:
        primary,tag='FP_LOCALIZATION_MISMATCH','FP_NEAR_GT_LOW_IOU'
    elif maximum==0:
        primary,tag='FP_ON_POSITIVE_IMAGE_OTHER_REGION','FP_SPATIALLY_DISTANT'
    else:
        primary,tag='FP_UNCLEAR','FP_UNCLEAR'
    low_bin=('(0,0.10)' if maximum<.1 else '[0.10,0.25)' if maximum<.25 else '[0.25,0.50)') if 0<maximum<.5 else 'NOT_APPLICABLE'
    return {'sample_id':row['sample_id'],'prediction_id':index,'prediction_bbox':box,
            'confidence':row['confidences'][index],'max_iou_with_gt':maximum,
            'nearest_gt_id':int(np.argmax(overlaps)) if overlaps else None,
            'primary_failure_type':primary,'geometric_tag':tag,'low_iou_bin':low_bin,
            'positive_image':row['gt_pixels']>0,'multi_gt_image':row['num_gt_instances']>1,
            'evidence_class':'CONFIRMED_FROM_GT' if primary=='FP_NEGATIVE_IMAGE' else 'OBSERVED_FAILURE',
            'human_review_status':'UNKNOWN','human_failure_tags':'UNKNOWN'}


def severity_bin(fraction):
    if fraction==0:return '0%'
    if fraction<=.25:return '>0-25%'
    if fraction<=.5:return '>25-50%'
    if fraction<=.75:return '>50-75%'
    if fraction<=.9:return '>75-90%'
    if fraction<.95:return '>90-95%'
    return '>=95%'


def small_bin(area):
    if not 0<=area<.01:raise C1Error('Not a primary-small instance')
    return SMALL_BINS[next(i for i,limit in enumerate((.001,.0025,.005,.0075,.01)) if area<limit)]


def crop_failure_type(row):
    crop=row['crop']; matched={p['gt'] for p in row['pairs']}
    outside=[]
    if crop:
        outside=[i for i,b in enumerate(row['gt_boxes']) if i not in matched and
                 not (crop[0]<=b[0] and crop[1]<=b[1] and crop[2]>=b[2] and crop[3]>=b[3])]
    tags=[]
    if crop is None:primary='NO_ROI'
    else:
        tags.append('PARTIAL_GT_COVERAGE')
        if row['num_gt_instances']>1 and outside:tags.append('MISSED_SECOND_INSTANCE')
        if row['tp']==0 and row['num_predictions']>0:tags.append('LOCALIZED_WRONG_REGION')
        primary=('LOCALIZED_WRONG_REGION' if 'LOCALIZED_WRONG_REGION' in tags else
                 'MISSED_SECOND_INSTANCE' if 'MISSED_SECOND_INSTANCE' in tags else 'PARTIAL_GT_COVERAGE')
    return {'primary_failure_type':primary,'secondary_failure_types':[t for t in tags if t!=primary],
            'unmatched_gt_bbox_partly_outside_crop':outside,
            'evidence_class':'OBSERVED_FAILURE',
            'causal_status':'Descriptive geometry; does not establish the cause of a missed instance'}


def group_metrics(rows):
    n=len(rows); gt=sum(r['num_gt_instances'] for r in rows); tp=sum(r['tp'] for r in rows)
    positives=[r for r in rows if r['gt_pixels']]
    return {'image_count':n,'GT':gt,'TP':tp,'FN':gt-tp,'instance_recall':tp/gt if gt else None,
            'images_with_any_miss':sum(r['fn']>0 for r in rows),
            'image_any_miss_rate':sum(r['fn']>0 for r in rows)/n if n else None,
            'crop_pass':sum(r['crop_complete95'] for r in positives),
            'crop_fail':sum(not r['crop_complete95'] for r in positives),
            'crop_complete_rate':sum(r['crop_complete95'] for r in positives)/len(positives) if positives else None,
            'mean_retained_fraction':float(np.mean([r['crop_coverage'] for r in positives])) if positives else None,
            'FP':sum(r['fp'] for r in rows),'FP_per_image':sum(r['fp'] for r in rows)/n if n else None}


def pareto(categories):
    counts=Counter(categories); total=sum(counts.values()); cumulative=0; result=[]
    for category,count in sorted(counts.items(),key=lambda kv:(-kv[1],kv[0])):
        cumulative+=count
        result.append({'primary_failure_category':category,'count':count,'denominator':total,
                       'percentage':100*count/total,'cumulative_percentage':100*cumulative/total})
    return result


def derive(rows):
    gt=[]; fn=[]; fp=[]; crops=[]; confidence={'TP':[],'FP':[]}; review=[]
    for r in rows:
        matched_gt={p['gt'] for p in r['pairs']}; matched_pred={p['prediction'] for p in r['pairs']}
        for j,box in enumerate(r['gt_boxes']):
            size=r['per_instance_gt'][j]['size_group']
            instance={'sample_id':r['sample_id'],'gt_instance_id':j,'size_group':size,'gt_bbox':box,
                      'detected':j in matched_gt,'matched_prediction':j in matched_gt,
                      'image_has_any_prediction':r['num_predictions']>0,'image_has_roi':r['crop'] is not None,
                      'multi_gt_image':r['num_gt_instances']>1,'GT_instance_count_in_image':r['num_gt_instances'],
                      **bbox_features(box)}
            instance['gt_area_ratio']=instance['bbox_area_ratio']
            instance['small_diagnostic_bin']=small_bin(instance['bbox_area_ratio']) if size=='small' else 'NOT_SMALL'
            gt.append(instance)
            if j not in matched_gt:
                missed={**instance,'primary_failure_type':f"{size.upper()}_FN_{'MULTI_GT' if instance['multi_gt_image'] else 'SINGLE_GT'}",
                        'evidence_class':'CONFIRMED_FROM_GT'}
                fn.append(missed)
        for j,score in enumerate(r['confidences']):
            confidence['TP' if j in matched_pred else 'FP'].append(score)
            if j not in matched_pred:fp.append(classify_fp(r,j))
        crop_entry=None
        if r['gt_pixels'] and not r['crop_complete95']:
            crop_entry={'sample_id':r['sample_id'],'GT_count':r['num_gt_instances'],'TP':r['tp'],'FP':r['fp'],'FN':r['fn'],
                        'retained_fraction':r['crop_coverage'],'severity_bin':severity_bin(r['crop_coverage']),
                        'crop':r['crop'],'no_roi':r['crop'] is None,'multi_wound':r['num_gt_instances']>1,
                        'small_GT':r['size_support']['small'],'medium_GT':r['size_support']['medium'],'large_GT':r['size_support']['large'],
                        **crop_failure_type(r)}
            crops.append(crop_entry)
        tags=[]
        if r['fn']:tags.append('FN_PRESENT')
        tags += sorted({f['primary_failure_type'] for f in fp if f['sample_id']==r['sample_id']})
        if crop_entry:tags += [crop_entry['primary_failure_type']]+crop_entry['secondary_failure_types']
        review.append({'sample_id':r['sample_id'],'GT_count':r['num_gt_instances'],'TP':r['tp'],'FP':r['fp'],'FN':r['fn'],
                       'small_GT':r['size_support']['small'],'medium_GT':r['size_support']['medium'],'large_GT':r['size_support']['large'],
                       'crop_retained_fraction':r['crop_coverage'],'crop_complete':r['crop_complete95'] if r['gt_pixels'] else None,
                       'no_roi':r['crop'] is None,'multi_wound':r['num_gt_instances']>1,
                       'machine_failure_tags':sorted(set(tags)), 'human_review_status':'UNKNOWN',
                       'human_failure_tags':'UNKNOWN','notes':''})
    positive=[r for r in rows if r['gt_pixels']]
    counts={'TP':sum(r['tp'] for r in rows),'FP':len(fp),'FN':len(fn),'GT':len(gt),
            **{s:sum(g['size_group']==s for g in gt) for s in ('small','medium','large')},
            'positive':len(positive),'negative':len(rows)-len(positive),'crop_pass':sum(r['crop_complete95'] for r in positive),
            'crop_fail':len(crops),'no_roi_positive':sum(r['crop'] is None for r in positive)}
    features=['bbox_area_ratio','bbox_width_ratio','bbox_height_ratio','bbox_width_pixels','bbox_height_pixels','bbox_area_pixels',
              'image_width','image_height','aspect_ratio','GT_instance_count_in_image','nearest_border_distance_normalized']
    geometry={}
    for scope,group in [('all_GT',gt),('small_GT',[g for g in gt if g['size_group']=='small'])]:
        geometry[scope]={state:{key:stats([g[key] for g in group if g['detected']==detected]) for key in features}
                         for state,detected in [('detected',True),('missed',False)]}
    size_bins=[]
    for bin_name in SMALL_BINS:
        group=[g for g in gt if g['small_diagnostic_bin']==bin_name]; tp=sum(g['detected'] for g in group)
        size_bins.append({'bin':bin_name,'support':len(group),'TP':tp,'FN':len(group)-tp,
                          'recall':tp/len(group) if group else None,'interpretation':'descriptive only; no independence or significance claim'})
    severity=[]
    for name in SEVERITY_BINS:
        values=[c['retained_fraction'] for c in crops if c['severity_bin']==name]
        severity.append({'bin':name,'count':len(values),'percentage_of_19_failures':100*len(values)/len(crops) if crops else 0,**stats(values)})
    grouping={'single_GT':group_metrics([r for r in positive if r['num_gt_instances']==1]),
              'multi_GT':group_metrics([r for r in positive if r['num_gt_instances']>1])}
    instance_strata={key:group_metrics([r for r in positive if condition(r['num_gt_instances'])])
                     for key,condition in [('1',lambda n:n==1),('2',lambda n:n==2),('3+',lambda n:n>=3)]}
    small_crop={
        'any_small':group_metrics([r for r in positive if r['size_support']['small']>0]),
        'no_small':group_metrics([r for r in positive if r['size_support']['small']==0]),
        'single_small':group_metrics([r for r in positive if r['num_gt_instances']==1 and r['size_support']['small']==1]),
        'multi_small_only':group_metrics([r for r in positive if r['num_gt_instances']>1 and r['size_support']['small']==r['num_gt_instances']]),
        'small_plus_medium_large':group_metrics([r for r in positive if 0<r['size_support']['small']<r['num_gt_instances']])}
    sets={'FN':{g['sample_id'] for g in fn},'FP':{g['sample_id'] for g in fp},'CROP':{g['sample_id'] for g in crops}}
    overlap=Counter('+'.join(k for k,s in sets.items() if r['sample_id'] in s) or 'NONE' for r in rows)
    confidence_stats={k:stats(v) for k,v in confidence.items()}
    hist_edges=np.linspace(0,1,11)
    tp_hist=np.histogram(confidence['TP'],bins=hist_edges)[0]; fp_hist=np.histogram(confidence['FP'],bins=hist_edges)[0]
    overlap_coefficient=float(np.minimum(tp_hist/max(1,len(confidence['TP'])),fp_hist/max(1,len(confidence['FP']))).sum())
    t=confidence_stats['TP']
    confidence_details={'statistics':confidence_stats,'histogram_edges':hist_edges.tolist(),
        'histogram_counts':{'TP':tp_hist.tolist(),'FP':fp_hist.tolist()},'histogram_overlap_coefficient':overlap_coefficient,
        'FP_fraction_within_TP_IQR':sum(t['p25']<=v<=t['p75'] for v in confidence['FP'])/len(confidence['FP']) if confidence['FP'] else None,
        'FP_fraction_within_TP_range':sum(t['min']<=v<=t['max'] for v in confidence['FP'])/len(confidence['FP']) if confidence['FP'] else None,
        'frozen_confidence':.10,'threshold_sweep_performed':False,'values':confidence}
    taxonomy=[]
    for unit,items,idkey in [('FN_INSTANCE',fn,'gt_instance_id'),('FP_PREDICTION',fp,'prediction_id'),('CROP_IMAGE',crops,None)]:
        for item in items:
            taxonomy.append({'unit':unit,'sample_id':item['sample_id'],'instance_or_prediction_id':item[idkey] if idkey else '',
                'primary_failure_category':item['primary_failure_type'],'secondary_tags':item.get('secondary_failure_types',[item.get('geometric_tag','')]),
                'evidence_class':item['evidence_class'],'human_review_status':'UNKNOWN'})
    concentration={'small_FN_of_all_FN':{'numerator':sum(g['size_group']=='small' for g in fn),'denominator':len(fn)},
        'multi_wound_FN_of_all_FN':{'numerator':sum(g['multi_gt_image'] for g in fn),'denominator':len(fn)},
        'no_ROI_of_crop_failures':{'numerator':sum(c['no_roi'] for c in crops),'denominator':len(crops)},
        'multi_GT_of_crop_failures':{'numerator':sum(c['multi_wound'] for c in crops),'denominator':len(crops)}}
    for item in concentration.values():item['fraction']=item['numerator']/item['denominator'] if item['denominator'] else None
    return {'counts':counts,'new_inference_performed':False,'training_performed':False,
        'gt_instances':gt,'FN_instances':fn,'FP_predictions':fp,'crop_failures':crops,'review_sheet':review,'failure_taxonomy':taxonomy,
        'geometry_comparison':geometry,'small_size_bins':size_bins,'crop_failure_severity':severity,
        'crop_failure_retained_fraction_summary':stats([c['retained_fraction'] for c in crops]),
        'single_vs_multi':grouping,'instance_count_strata':instance_strata,'crop_by_small_presence':small_crop,
        'failure_set_image_counts':{k:len(s) for k,s in sets.items()},'failure_set_image_intersections':dict(overlap),
        'confidence_analysis':confidence_details,'concentration':concentration,
        'FP_primary_counts':dict(Counter(f['primary_failure_type'] for f in fp)),
        'FP_geometry_counts':dict(Counter(f['geometric_tag'] for f in fp)),
        'FP_low_iou_bins':dict(Counter(f['low_iou_bin'] for f in fp if f['geometric_tag']=='FP_NEAR_GT_LOW_IOU')),
        'crop_primary_counts':dict(Counter(c['primary_failure_type'] for c in crops)),
        'pareto':{'FN_instances':pareto([g['primary_failure_type'] for g in fn]),
                  'FP_predictions':pareto([g['primary_failure_type'] for g in fp]),
                  'crop_images':pareto([g['primary_failure_type'] for g in crops])},
        'distance_to_gate_positive_images':1,
        'mask_occupancy_note':'Not assigned per instance: historical publisher mask is an image-level union; no instance mask identity is invented.'}


def load_verified(root):
    out=root/OUTPUT_REL
    check_digest(root/PROTOCOL_REL,PINNED_PROTOCOL_SHA256,'FAIL_PHASE_C1_INPUT_INTEGRITY')
    p=read_json(root/PROTOCOL_REL)
    check_digest(root/p['checkpoint']['path'],PINNED_CHECKPOINT_SHA256,'FAIL_PHASE_C1_INPUT_INTEGRITY')
    check_digest(root/p['validation_manifest']['path'],PINNED_MANIFEST_SHA256,'FAIL_PHASE_C1_INPUT_INTEGRITY')
    check_digest(out/'result.json',RESULT_SHA,'FAIL_PHASE_C1_INPUT_INTEGRITY')
    check_digest(out/'per_image_predictions.json',ROWS_SHA,'FAIL_PHASE_C1_INPUT_INTEGRITY')
    rows=read_json(out/'per_image_predictions.json'); result=read_json(out/'result.json')
    manifest=read_json(root/p['validation_manifest']['path'])
    validate_result_integrity(rows,result['candidate'],[s['sample_id'] for s in manifest['samples']])
    for i,row in enumerate(rows):
        if read_json(out/'per_image'/f'{i:03d}.json')!=row:raise C1Error('FAIL_PHASE_C1_INPUT_INTEGRITY: per-image')
        path=assert_relative_admitted(out,row['prediction_mask_reference'],out/'masks')
        check_digest(path,row['prediction_mask_sha256'],'FAIL_PHASE_C1_INPUT_INTEGRITY')
        matched={p['gt'] for p in row['pairs']}
        if len(row['per_instance_gt'])!=len(row['gt_boxes']):raise C1Error('FAIL_PHASE_C1_INPUT_INTEGRITY: GT identity')
        for j,instance in enumerate(row['per_instance_gt']):
            if instance['index']!=j or instance['matched']!=(j in matched) or instance['bbox']!=row['gt_boxes'][j]:
                raise C1Error('FAIL_PHASE_C1_INPUT_INTEGRITY: saved matching flags')
    # Only Phase C files, its protocol/code and candidate checkpoint, no external dataset reads.
    protected=[f for f in out.rglob('*') if f.is_file() and not f.relative_to(out).parts[0].startswith('phase_c1')]
    protected += [root/PROTOCOL_REL,(root/PROTOCOL_REL).with_suffix('.sha256'),root/p['validation_manifest']['path'],
        root/p['checkpoint']['path'],root/p['metric_implementation']['path'],root/'woundcare_inference.py',
        root/'experiments/review_v2/isic_fuseg_gate.py',root/'experiments/phase_c_execution.py',
        root/'docs/PHASE_C_RGB_BGR_CONTROLLED_REEVALUATION_20260921.md']
    snapshot={str(f.relative_to(root)).replace('\\','/'):sha256_file(f) for f in sorted(set(protected))}
    return rows,manifest,snapshot


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def write_json(path,data):
    with path.open('x',encoding='utf8') as handle:json.dump(data,handle,ensure_ascii=False,indent=2,allow_nan=False)


def images_and_quality(root,manifest,rows):
    import cv2
    from PIL import Image
    images={}; quality=[]; by_id={r['sample_id']:r for r in rows}
    allowed=root/'outputs/fuseg_warmup_revision_20260914/dataset/images/val'
    for sample in manifest['samples']:
        path=assert_relative_admitted(root,sample['relative_image_path'],allowed)
        check_digest(path,sample['image_sha256'],'FAIL_PHASE_C1_INPUT_INTEGRITY')
        with Image.open(path) as im:
            if im.size!=(512,512):raise C1Error('FAIL_PHASE_C1_INPUT_INTEGRITY: image dimensions')
            rgb=np.asarray(im.convert('RGB')).copy()
        images[sample['sample_id']]=rgb
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        saturation=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[:,:,1]
        r=by_id[sample['sample_id']]
        quality.append({'sample_id':r['sample_id'],'group':'negative' if not r['gt_pixels'] else 'any_missed_GT' if r['fn'] else 'all_GT_detected',
            'brightness_gray_mean_0_255':float(gray.mean()),'contrast_gray_std_0_255':float(gray.std()),
            'sharpness_laplacian_variance':float(cv2.Laplacian(gray,cv2.CV_64F).var()),
            'saturation_mean_0_1':float(saturation.mean()/255)})
    return images,quality


def generate_figures(out,images,rows,analysis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    folder=out/'phase_c1_figures';folder.mkdir(exist_ok=False)
    by_id={r['sample_id']:r for r in rows}; inventory=[]
    def grid(name,entries,unit):
        for start in range(0,len(entries),12):
            batch=entries[start:start+12]; fig,axes=plt.subplots(4,3,figsize=(12,17))
            for ax in axes.flat:ax.axis('off')
            for ax,item in zip(axes.flat,batch):
                r=by_id[item['sample_id']];ax.imshow(images[item['sample_id']])
                for i,b in enumerate(r['gt_boxes']):
                    color='#ffb000' if unit=='GT_INSTANCE' and i==item.get('gt_instance_id') else '#2bdd5a'
                    ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor=color,linewidth=1.5))
                    ax.text(b[0],b[1],f'G{i}',color=color,fontsize=7,backgroundcolor='black')
                for i,b in enumerate(r['pred_boxes']):
                    color='#ffb000' if unit=='FP_PREDICTION' and i==item.get('prediction_id') else '#ff3848'
                    ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor=color,linewidth=1.3))
                if r['crop']:
                    b=r['crop'];ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='#00c9e8',linewidth=1.2))
                coverage=f"{r['crop_coverage']:.1%}" if r['crop_coverage'] is not None else 'N/A'
                label=item.get('primary_failure_type','')
                title=f"{r['sample_id']}  TP/FP/FN={r['tp']}/{r['fp']}/{r['fn']}\nGT={r['num_gt_instances']} crop={coverage}"
                if unit=='GT_INSTANCE':title+=f" | target G{item['gt_instance_id']}"
                if unit=='FP_PREDICTION':title+=f" | target P{item['prediction_id']} c={item['confidence']:.3f}"
                if label:title+='\n'+label.replace('FP_ON_POSITIVE_IMAGE_OTHER_REGION','FP_POSITIVE_OTHER_REGION')
                ax.set_title(title,fontsize=8)
            fig.suptitle(f'{name} | page {start//12+1} | all cases included\nGreen=GT, red=prediction, cyan=crop, orange=focused instance',fontsize=12)
            fig.tight_layout(rect=(0,0,1,.96));path=folder/f'{name}_{start//12+1:02d}.png'
            fig.savefig(path,dpi=140);plt.close(fig)
            inventory.append({'file':path.name,'unit':unit,'sample_ids':[e['sample_id'] for e in batch],
                              'instance_ids':[e.get('gt_instance_id',e.get('prediction_id')) for e in batch]})
    grid('small_wound_misses',[r for r in analysis['FN_instances'] if r['size_group']=='small'],'GT_INSTANCE')
    grid('all_FN_instances',analysis['FN_instances'],'GT_INSTANCE')
    grid('multi_wound_miss_images',[{'sample_id':r['sample_id']} for r in rows if r['num_gt_instances']>1 and r['fn']],'IMAGE')
    grid('FP_by_taxonomy',sorted(analysis['FP_predictions'],key=lambda r:(r['primary_failure_type'],r['sample_id'],r['prediction_id'])),'FP_PREDICTION')
    grid('all_crop_failures',analysis['crop_failures'],'IMAGE')
    confidence=analysis['confidence_analysis'];fig,axes=plt.subplots(1,2,figsize=(11,4))
    for key,color in [('TP','#167d8d'),('FP','#d3593b')]:
        vals=np.asarray(confidence['values'][key]);counts=np.histogram(vals,bins=confidence['histogram_edges'])[0]
        axes[0].stairs(counts/max(1,len(vals)),confidence['histogram_edges'],label=f'{key} n={len(vals)}',color=color,linewidth=2)
        axes[1].step(np.sort(vals),np.arange(1,len(vals)+1)/max(1,len(vals)),where='post',label=key,color=color)
    for ax in axes:
        ax.axvline(.1,color='gray',linestyle='--',label='Frozen conf=0.10');ax.set_xlim(0,1);ax.set_xlabel('Saved prediction confidence');ax.legend(fontsize=8)
    axes[0].set_ylabel('Fraction within group');axes[0].set_title('Fixed-bin distribution (width 0.10)')
    axes[1].set_ylabel('ECDF');axes[1].set_title('TP and FP confidence, no threshold search')
    fig.tight_layout();fig.savefig(folder/'confidence_histogram_ecdf.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key,label in [(axes[0],'bbox_area_ratio','Small GT bbox area (% of image)'),
                         (axes[1],'nearest_border_distance_normalized','Small GT normalized border distance')]:
        data=[[g[key]*(100 if key=='bbox_area_ratio' else 1) for g in analysis['gt_instances'] if g['size_group']=='small' and g['detected']==flag] for flag in (True,False)]
        ax.boxplot(data,labels=['Detected (109)','Missed (28)'],showmeans=True);ax.set_ylabel(label);ax.set_title('Descriptive geometry; no significance test')
    fig.tight_layout();fig.savefig(folder/'small_detected_missed_geometry.png',dpi=180);plt.close(fig)
    inventory += [{'file':name,'unit':'AGGREGATE','sample_ids':[]} for name in ['confidence_histogram_ecdf.png','small_detected_missed_geometry.png']]
    return inventory


def verify_grid_inventory(analysis):
    groups={
        'small_wound_misses':[(r['sample_id'],r['gt_instance_id']) for r in analysis['FN_instances'] if r['size_group']=='small'],
        'all_FN_instances':[(r['sample_id'],r['gt_instance_id']) for r in analysis['FN_instances']],
        'multi_wound_miss_images':[(r['sample_id'],None) for r in analysis['review_sheet'] if r['multi_wound'] and r['FN']],
        'FP_by_taxonomy':[(r['sample_id'],r['prediction_id']) for r in analysis['FP_predictions']],
        'all_crop_failures':[(r['sample_id'],None) for r in analysis['crop_failures']]}
    for prefix,expected in groups.items():
        actual=[]
        for page in analysis['visualization_inventory']:
            if page['file'].startswith(prefix+'_'):
                if len(page['sample_ids'])!=len(page['instance_ids']):raise C1Error('GRID_ID_LENGTH_MISMATCH')
                actual.extend(zip(page['sample_ids'],page['instance_ids']))
        if Counter(actual)!=Counter(expected):raise C1Error('GRID_COVERAGE_MISMATCH: '+prefix)
    return {name:len(rows) for name,rows in groups.items()}


def verify_artifacts(root):
    rows,manifest,snapshot=load_verified(root);out=root/OUTPUT_REL
    analysis=read_json(out/'phase_c1_error_analysis.json');verify_expected_counts(analysis['counts'])
    fresh=derive(rows)
    for key,value in fresh.items():
        if analysis[key]!=value:raise C1Error('ANALYSIS_REPRODUCTION_MISMATCH: '+key)
    integrity=read_json(out/'phase_c1_integrity.json')
    if snapshot!=integrity['before'] or snapshot!=integrity['after']:raise C1Error('FAIL_PHASE_C1_INPUT_INTEGRITY')
    csv_counts={}
    expected={'failure_taxonomy':87,'small_wound_analysis':137,'fp_analysis':36,'fn_analysis':32,
              'crop_failure_analysis':19,'review_sheet':191,'all_gt_geometry':241,'small_size_bins':5,
              'crop_severity':7,'image_quality':191}
    for name,count in expected.items():
        with (out/f'phase_c1_{name}.csv').open(encoding='utf-8-sig',newline='') as handle:
            records=list(csv.DictReader(handle))
        if len(records)!=count:raise C1Error('CSV_COUNT_MISMATCH: '+name)
        if name=='review_sheet' and any(r['human_review_status']!='UNKNOWN' or r['human_failure_tags']!='UNKNOWN' for r in records):
            raise C1Error('HUMAN_REVIEW_MUST_REMAIN_UNKNOWN')
        csv_counts[name]=count
    coverage=verify_grid_inventory(analysis)
    from PIL import Image
    for page in analysis['visualization_inventory']:
        with Image.open(out/'phase_c1_figures'/page['file']) as picture:picture.verify()
    known={f'fuseg__{sid}.png' for sid in ['0456','0816','0349','0548','0192','0798']}
    if not known.issubset({r['sample_id'] for r in analysis['crop_failures']}):raise C1Error('KNOWN_EXAMPLES_MISSING')
    return {'status':'PASS','protected_files_unchanged':len(snapshot),'mask_artifacts_verified':len(rows),
            'derived_fields_reproduced':list(fresh),'csv_row_counts':csv_counts,'grid_coverage':coverage,
            'figure_files_verified':len(analysis['visualization_inventory']),'human_review_rows_UNKNOWN':191,
            'known_examples_included':sorted(known),'model_inference_performed':False,'training_performed':False,
            'locked_test_used':False,'CO2Wounds_used':False}


def run(root,preview=False):
    rows,manifest,before=load_verified(root);analysis=derive(rows);verify_expected_counts(analysis['counts'])
    if preview:
        return {k:v for k,v in analysis.items() if k not in {'gt_instances','FN_instances','FP_predictions','crop_failures','review_sheet','failure_taxonomy'}}
    out=root/OUTPUT_REL
    if any(out.glob('phase_c1*')):raise C1Error('PHASE_C1_OUTPUT_ALREADY_EXISTS')
    images,quality=images_and_quality(root,manifest,rows)
    quality_features=list(quality[0])[2:]
    analysis['image_quality_definitions']={'brightness':'mean cv2 RGB2GRAY uint8 intensity (0-255)',
        'contrast':'population standard deviation of same grayscale pixels',
        'sharpness':'variance of cv2.Laplacian(gray,CV_64F), default ksize=1',
        'saturation':'mean cv2 RGB2HSV saturation /255',
        'unit':'one whole image, not wound ROI; positive images grouped all-GT-detected versus any-GT-missed',
        'interpretation':'exploratory association only; framing/background and GT count may confound these descriptors'}
    analysis['image_quality_comparison']={group:{key:stats([r[key] for r in quality if r['group']==group]) for key in quality_features}
                                         for group in ['all_GT_detected','any_missed_GT']}
    files={
        'failure_taxonomy':analysis['failure_taxonomy'], 'small_wound_analysis':[g for g in analysis['gt_instances'] if g['size_group']=='small'],
        'fp_analysis':analysis['FP_predictions'],'fn_analysis':analysis['FN_instances'],'crop_failure_analysis':analysis['crop_failures'],
        'review_sheet':analysis['review_sheet'],'all_gt_geometry':analysis['gt_instances'],'small_size_bins':analysis['small_size_bins'],
        'crop_severity':analysis['crop_failure_severity'],'image_quality':quality}
    for suffix,records in files.items():write_csv(out/f'phase_c1_{suffix}.csv',records)
    analysis['visualization_inventory']=generate_figures(out,images,rows,analysis)
    analysis['recommendation_status']='CANDIDATE_ONLY'
    analysis['recommended_next_single_training_experiment']={
        'name':'Training-GT-only small-object-aware image sampling',
        'evidence':'28/32 FN are small; 20/28 small FN have bbox area <0.25%; <0.10% recall is 13/27.',
        'single_factor':'Training image sampling distribution only',
        'controls':'Reuse the same pre-FUSeg initialization and all other baseline settings, seed, number of sampled images per epoch and training budget; preserve the fixed Phase C comparator.',
        'preregistration_required':'Define training-GT size rules, sampling weights, paired control and advancement gates before running. Never select training images using validation failure IDs.',
        'limitation':'May improve tiny-lesion recall; improvement in precision or crop completeness is not guaranteed.',
        'status':'CANDIDATE_ONLY','authorized':False}
    analysis['candidate_intervention_mapping']=[
        {'evidence':'Very-small lesion FN concentration','possible_intervention':'Small-object-aware image sampling: the single prioritized training proposal','new_experiment_required':True,'status':'CANDIDATE_ONLY'},
        {'evidence':'12/19 crop failures are multi-GT; 12/15 ROI-present failures have an unmatched GT bbox partly outside the crop','possible_intervention':'Separate crop-policy / multi-ROI research; cannot recover an undetected lesion by unioning existing predictions alone','new_experiment_required':True,'status':'CANDIDATE_ONLY'},
        {'evidence':'15/36 FP have 0<max IoU<0.50','possible_intervention':'Future localization-quality research question, not an additional prioritized training experiment','new_experiment_required':True,'status':'CANDIDATE_ONLY'},
        {'evidence':'FP median confidence is lower but its range overlaps TP','possible_intervention':'Future pre-registered threshold study; no threshold value proposed or swept','new_experiment_required':True,'status':'CANDIDATE_ONLY'}]
    analysis['rules']={
        'matching':'saved Phase C pairs used verbatim; no rematching',
        'FP_priority':['GT-negative image','IoU>=0.50 with already matched GT: possible duplicate','0<max IoU<0.50: localization mismatch','max IoU=0 on positive image: other region','unclear'],
        'crop_priority':['NO_ROI','LOCALIZED_WRONG_REGION: retained predictions but TP=0','MISSED_SECOND_INSTANCE: multi-GT and unmatched bbox not wholly contained by saved crop','PARTIAL_GT_COVERAGE','OTHER'],
        'crop_secondary_tags':'nonexclusive descriptive geometry, never a causal attribution',
        'severity_intervals':'0; (0,.25]; (.25,.50]; (.50,.75]; (.75,.90]; (.90,.95); [.95,1]',
        'small_intervals':'[0,.001); [.001,.0025); [.0025,.005); [.005,.0075); [.0075,.01)',
        'human_review':'all 191 rows UNKNOWN until a human enters findings; no annotation_error or clinical_error assigned',
        'denominators':'FN/FP are instances; crop failures are positive images. Pareto tables kept separate; 87 mixed-unit records are not 87 independent failures.',
        'unit_dependence':'multiple instances in one image may be dependent; all analyses descriptive; no p-value, bootstrap or CI'}
    analysis['integrity']={'status':'PASS','protocol_sha256':PINNED_PROTOCOL_SHA256,'checkpoint_sha256':PINNED_CHECKPOINT_SHA256,
        'result_sha256':RESULT_SHA,'per_image_predictions_sha256':ROWS_SHA,'mask_artifacts_verified':191,'protected_files':len(before)}
    changed=[rel for rel,digest in before.items() if sha256_file(root/rel)!=digest]
    if changed:raise C1Error('FAIL_PHASE_C1_INPUT_INTEGRITY: source artifact changed')
    write_json(out/'phase_c1_integrity.json',{'status':'PASS','before':before,'after':before,'changed':[],
        'model_inference_performed':False,'training_performed':False,'locked_test_used':False,'CO2Wounds_used':False})
    write_json(out/'phase_c1_error_analysis.json',analysis)
    return {'phase_c1_analysis_status':'COMPLETE','counts':analysis['counts'],'protected_files_unchanged':len(before),
            'visualization_pages':len(analysis['visualization_inventory']),'model_inference_performed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--preview',action='store_true');modes.add_argument('--verify-artifacts',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    if args.verify_artifacts:
        result=verify_artifacts(root)
        write_json(root/OUTPUT_REL/'phase_c1_artifact_verification.json',result)
    else:result=run(root,args.preview)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
