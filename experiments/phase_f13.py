"""F1.3: saved-artifact-only descriptive audit. No model imports or inference.

GT identity = sample_id + zero-based nonempty frozen polygon-label line index.
Saved matching is authoritative; recomputation only checks its fixed contract.
CSV files are plain scientific evidence tables, not editable Excel models.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from collections import Counter
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'experiments/results'
PAIR = RESULTS / 'f_higher_scale_v2_recovery_seed42_pair_summary'
PRE = RESULTS / 'f_higher_scale_v2_recovery_seed42_preflight'
OUT = RESULTS / 'f_higher_scale_v2_paired_error_audit'
REPORT = ROOT / 'docs/PHASE_F13_PAIRED_PREDICTION_ERROR_AUDIT_20260929.md'
REQUEST = Path('C:/Users/milo9/.codex/attachments/1977d8cd-440d-44af-9561-98875b47053b/貼上的文字.txt')
MANIFEST = ROOT / 'experiments/protocols/ISIC_FUSEG_COLORFIX_V1_validation_manifest.json'
PROTOCOL = ROOT / 'experiments/protocols/F_HIGHER_SCALE_V2_evaluation_protocol.json'
ARMS = {'C4': PAIR/'eval_C4', 'H4': PAIR/'eval_H4_R'}
PINS = {'C4':'2ee2cc68c9e9413ca8446fe27f1ad8e794b655a3486ba1fa7db0e607ac39f683',
        'H4':'af47361faf4e4c77b35dab342db0fc6a06138d49edc9edc895f012b77a9305e0'}
BINS = ['<0.10%', '0.10-<0.25%', '0.25-<0.50%', '0.50-<0.75%', '0.75-<1.00%', 'Medium 1-<5%', 'Large >=5%']
TRANSITIONS = ['BOTH_DETECTED','C4_ONLY','H4_ONLY','BOTH_MISSED']
FP_CATS = ['NEGATIVE_IMAGE_FP','DUPLICATE_OR_MATCH_COMPETITION_FP','LOCALIZATION_MISMATCH_FP','OTHER_REGION_FP']
MISS_CATS = ['MATCH_COMPETITION','BELOW_FROZEN_CONFIDENCE','RETAINED_LOCALIZATION_BELOW_IOU','NO_RETAINED_OVERLAP']
FLAGS = dict(NEW_INFERENCE=False, MODEL_LOADED=False, TRAINING=False, THRESHOLD_CHANGED=False,
             NMS_CHANGED=False, MATCH_IOU_CHANGED=False, LOCKED_TEST_USED=False, CO2Wounds_USED=False,
             EXTERNAL_TEST_USED=False, test_images_used=0, NEW_TRAINING_AUTHORIZED='NO',
             APP_MODEL_REPLACEMENT_AUTHORIZED='NO', MULTI_SEED_AUTHORIZED='NO', STOP_AFTER_F13=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csv_write(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    require(bool(fields), 'EMPTY_CSV_SCHEMA: '+name)
    def value(v):
        if v is None: return 'NA'
        if isinstance(v, (list, dict)): return json.dumps(v, ensure_ascii=False)
        return v
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:value(r.get(k)) for k in fields} for r in rows)
    with (OUT/name).open(newline='', encoding='utf-8-sig') as stream:
        reread = list(csv.DictReader(stream))
    require(len(reread) == len(rows) and all(set(r)==set(fields) for r in reread), 'CSV_ROUNDTRIP_FAILED')


def iou(a, b):
    inter = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-inter
    return inter/max(union, 1e-12)


def distance(a, b):
    """Euclidean minimum edge-to-edge bbox distance in original 512px canvas."""
    return math.hypot(max(a[0]-b[2], b[0]-a[2], 0), max(a[1]-b[3], b[1]-a[3], 0))


def stats(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    return dict(n=len(v), mean=float(v.mean()) if len(v) else None,
                median=float(np.median(v)) if len(v) else None,
                q25=float(np.quantile(v,.25)) if len(v) else None,
                q75=float(np.quantile(v,.75)) if len(v) else None,
                min=float(v.min()) if len(v) else None, max=float(v.max()) if len(v) else None)


def size_bin(ratio):
    return BINS[next((i for i, v in enumerate([.001,.0025,.005,.0075,.01,.05]) if ratio < v), 6)]


def group(n):
    return 'negative' if n == 0 else 'single' if n == 1 else 'multi'


def binary_mask(array):
    # Frozen publisher-mask normalization: identical RGB channels, then >0.
    a=np.asarray(array)
    if a.ndim==3 and a.shape[2]==3:
        require(np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,1],a[:,:,2]),'GT_MASK_RGB_CHANNEL_MISMATCH')
        a=a[:,:,0]
    require(a.shape==(512,512) and a.dtype==np.uint8,'GT_MASK_FORMAT')
    return a>0


def transition(c, h):
    return 'BOTH_DETECTED' if c and h else 'C4_ONLY' if c else 'H4_ONLY' if h else 'BOTH_MISSED'


def frozen_pairs(row):
    """Integrity-only reproduction, never replaces the saved pair identities."""
    used, pairs = set(), []
    for pi in sorted(range(len(row['confidences'])), key=lambda i:(-row['confidences'][i], i)):
        choices = [j for j, b in enumerate(row['gt_boxes']) if j not in used and iou(row['pred_boxes'][pi],b)>=.5]
        if choices:
            gi = max(choices, key=lambda j:iou(row['pred_boxes'][pi], row['gt_boxes'][j]))
            used.add(gi)
            pairs.append(dict(prediction=pi, gt=gi, iou=iou(row['pred_boxes'][pi], row['gt_boxes'][gi])))
    return pairs


def miss(row, gi):
    gt = row['gt_boxes'][gi]
    retained = [iou(b,gt) for b in row['pred_boxes']]
    floor = [iou(b,gt) for b in row['floor_bboxes']]
    assigned = {p['prediction']:p['gt'] for p in row['pairs']}
    competition = [i for i,v in enumerate(retained) if v>=.5 and i in assigned and assigned[i]!=gi]
    below = [i for i,v in enumerate(floor) if .01<=row['floor_confidences'][i]<.1 and v>=.5]
    diag = (MISS_CATS[0] if competition else MISS_CATS[1] if below else
            MISS_CATS[2] if 0<max(retained,default=0)<.5 else MISS_CATS[3])
    require(not any(v>=.5 and i not in assigned for i,v in enumerate(retained)), 'UNEXPLAINED_UNMATCHED_VALID_CANDIDATE')
    best = min(range(len(floor)), key=lambda i:(-floor[i],-row['floor_confidences'][i],i)) if floor else None
    return dict(diagnostic=diag, best_saved_candidate_index=best,
                best_saved_candidate_confidence=row['floor_confidences'][best] if best is not None else None,
                best_saved_candidate_iou=floor[best] if best is not None else None,
                max_retained_iou=max(retained,default=0), competition_prediction_indices=competition,
                below_frozen_confidence_candidate_indices=below)


def fp_record(arm, row, pi):
    box = row['pred_boxes'][pi]
    overlaps = [iou(box,b) for b in row['gt_boxes']]
    maximum = max(overlaps, default=0.)
    category = FP_CATS[0] if not overlaps else FP_CATS[1] if maximum>=.5 else FP_CATS[2] if maximum>0 else FP_CATS[3]
    distances = [distance(box,b) for b in row['gt_boxes']]
    return dict(arm=arm,sample_id=row['sample_id'],prediction_index=pi,
                floor_prediction_index=row['retained_mask_indices'][pi], bbox=box,
                confidence=row['confidences'][pi],category=category,max_bbox_iou_to_GT=maximum,
                nearest_GT_bbox_distance_px=min(distances) if distances else None,
                nearest_GT_bbox_index=min(range(len(distances)),key=lambda j:(distances[j],j)) if distances else None,
                image_GT_count=len(overlaps),single_multi=group(len(overlaps)))


def pair_fp(cs, hs):
    """IoU desc, min pair confidence desc, max confidence desc, C/H indices asc."""
    edges = [(iou(c['bbox'],h['bbox']), c, h) for c in cs for h in hs if iou(c['bbox'],h['bbox'])>=.5]
    edges.sort(key=lambda e:(-e[0],-min(e[1]['confidence'],e[2]['confidence']),
                            -max(e[1]['confidence'],e[2]['confidence']),e[1]['prediction_index'],e[2]['prediction_index']))
    usedc, usedh, result = set(), set(), []
    def record(kind,c,h,overlap):
        r = c or h
        return dict(sample_id=r['sample_id'],pairing=kind,single_multi=r['single_multi'],
                    C4_prediction_index=c['prediction_index'] if c else None,
                    H4_prediction_index=h['prediction_index'] if h else None,
                    C4_confidence=c['confidence'] if c else None,H4_confidence=h['confidence'] if h else None,
                    C4_category=c['category'] if c else None,H4_category=h['category'] if h else None,
                    cross_model_bbox_iou=overlap)
    for overlap,c,h in edges:
        ci,hi=c['prediction_index'],h['prediction_index']
        if ci not in usedc and hi not in usedh:
            usedc.add(ci); usedh.add(hi); result.append(record('SHARED_FP',c,h,overlap))
    result += [record('C4_ONLY_FP',c,None,None) for c in cs if c['prediction_index'] not in usedc]
    result += [record('H4_ONLY_FP',None,h,None) for h in hs if h['prediction_index'] not in usedh]
    return result


def guard():
    """Runtime deny model imports, external actions and writes beyond F1.3 outputs."""
    def admitted(path):
        p=Path(path).resolve()
        return p.is_relative_to(OUT.resolve()) or p==REPORT.resolve()
    def hook(event,args):
        if event=='import' and args[0].split('.')[0] in {'torch','ultralytics','tensorflow','onnxruntime'}:
            raise PermissionError('F13_MODEL_IMPORT_FORBIDDEN')
        if event.startswith(('subprocess.','socket.')) or event in {'os.system','os.startfile'}:
            raise PermissionError('F13_EXTERNAL_ACTION_FORBIDDEN')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            mode,flags=args[1],args[2]
            writing = (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if writing and not admitted(os.fsdecode(args[0])):
                raise PermissionError('F13_WRITE_OUTSIDE_OUTPUT: '+str(args[0]))
        if event in {'os.remove','os.rmdir','os.mkdir','os.rename'}:
            for p in (args[:2] if event=='os.rename' else args[:1]):
                if isinstance(p,(str,bytes,os.PathLike)) and not admitted(os.fsdecode(p)):
                    raise PermissionError('F13_MUTATION_OUTSIDE_OUTPUT')
    sys.addaudithook(hook)


def snapshot(paths):
    return {str(p):sha(p) for p in sorted(set(paths)) if p.is_file()}


def preflight():
    comparison = read(PAIR/'paired_comparison.json')
    require(comparison['PHASE_F12_STATUS']=='COMPLETE' and comparison['RECOVERY_PAIR_VALID']=='YES'
            and comparison['RECOVERY_RESEARCH_GATE']=='FAIL','F12_STATE_MISMATCH')
    require(read(PAIR/'recovery_pair_validity.json')['RECOVERY_PAIR_VALID']=='YES','RECOVERY_VALIDITY_MISMATCH')
    require((PAIR/'paired_comparison.md').read_bytes()==(ROOT/'docs/PHASE_F12_H4_REPLACEMENT_RECOVERY_20260928.md').read_bytes(),'F12_REPORT_COPY_MISMATCH')
    for arm,folder in [('C4','f_higher_scale_v2_seed42_control768'),('H4-R','f_higher_scale_v2_recovery_seed42_h4_1024')]:
        completion=comparison['training'][arm]
        require(read(RESULTS/folder/'training_completion.json')==completion,'TRAINING_RECEIPT_MISMATCH')
        for stem in ['best','last']:
            require(sha(RESULTS/folder/(stem+'.pt'))==completion[stem+'_checkpoint_sha256'],'COMPLETION_CHECKPOINT_HASH_MISMATCH')
    execution = read(PRE/'execution_freeze.json')
    checks = {}
    for mapping in [execution['sources'],execution['receipts'],read(PRE/'historical_freeze.json')['sha256']]:
        for name,expected in mapping.items():
            require(sha(name)==expected,'PROTECTED_SHA256_MISMATCH: '+name)
            checks[name]=expected
    ep=read(PROTOCOL)
    require(ep['operating_point']==dict(bbox_match_iou=.5,confidence=.1,nms_iou=.7),'OPERATING_POINT_CHANGED')
    require(ep['prediction']['conf']==.01 and ep['prediction']['imgsz']==768,'PREDICTION_CONTRACT_CHANGED')
    require(sha(MANIFEST)==ep['validation_manifest']['sha256']=='ea58c4e7d36b699d2f1e232d23fea8df9a6010674d08769e5a2ca3aca66ab069','GT_MANIFEST_HASH_MISMATCH')
    require(sha(ROOT/ep['metric_implementation']['path'])==ep['metric_implementation']['sha256'],'MATCH_IMPLEMENTATION_CHANGED')
    manifest = read(MANIFEST)['samples']
    require(len(manifest)==191 and len({r['sample_id'] for r in manifest})==191,'IMAGE_IDENTITY_COUNT')
    data={}; gt={}; inputs=set(map(Path,checks))|{MANIFEST,PROTOCOL,PAIR/'paired_comparison.json',REQUEST}
    for row in manifest:
        require(row['source_role']=='DEVELOPMENT_VALIDATION' and row['split']=='val','DATA_ROLE_NOT_VAL')
        im=ROOT/row['relative_image_path']; mask=ROOT/row['relative_mask_path']
        label=im.parent.parent.parent/'labels'/'val'/im.with_suffix('.txt').name
        for path,key in [(im,'image_sha256'),(mask,'mask_sha256'),(label,'label_sha256')]:
            require(sha(path)==row[key],'GT_SOURCE_HASH_MISMATCH: '+str(path)); inputs.add(path)
        boxes=[]
        for line in label.read_text().splitlines():
            if not line.strip():continue
            values=list(map(float,line.split()))
            require(values[0]==0 and len(values)>=7 and (len(values)-1)%2==0,'GT_POLYGON_CONTRACT')
            polygon=np.asarray(values[1:]).reshape(-1,2)*512
            boxes.append([float(polygon[:,0].min()),float(polygon[:,1].min()),float(polygon[:,0].max()),float(polygon[:,1].max())])
        gt[row['sample_id']]=boxes
    for arm,directory in ARMS.items():
        result=read(directory/'final_evaluation.json'); rows=read(directory/'per_image_predictions.json')
        require(sha(directory/'per_image_predictions.json')==result['predictions_sha256']==PINS[arm],'PREDICTIONS_HASH_MISMATCH')
        ck = comparison['training']['C4' if arm=='C4' else 'H4-R']
        require(result==comparison['evaluations']['C4' if arm=='C4' else 'H4-R'],'SAVED_EVALUATION_SUMMARY_MISMATCH')
        require(sha(directory/'best.pt')==ck['best_checkpoint_sha256']==result['checkpoint_sha256'],'CHECKPOINT_HASH_MISMATCH')
        lock=read(directory/'evaluation.lock')
        require(sha(PROTOCOL)==lock['evaluation_protocol_sha256'],'EVALUATION_PROTOCOL_HASH_MISMATCH')
        require(len(rows)==191 and len({r['sample_id'] for r in rows})==191 and {r['sample_id'] for r in rows}==set(gt),'IMAGE_IDENTITIES_MISMATCH')
        for index,row in enumerate(rows):
            require(read(directory/'per_image'/f'{index:03d}.json')==row,'PER_IMAGE_MISMATCH')
            require(row['gt_boxes']==gt[row['sample_id']],'ORDERED_GT_POLYGON_IDENTITY_MISMATCH')
            require(row['num_gt_instances']==len(row['gt_boxes']),'GT_COUNT_MISMATCH')
            computed=frozen_pairs(row)
            require(len(computed)==len(row['pairs']),'FROZEN_MATCHING_COUNT')
            for a,b in zip(computed,row['pairs']):
                require(a['gt']==b['gt'] and a['prediction']==b['prediction'] and abs(a['iou']-b['iou'])<1e-10,'FROZEN_MATCHING_MISMATCH')
            require((row['tp'],row['fp'],row['fn'])==(len(computed),len(row['pred_boxes'])-len(computed),len(row['gt_boxes'])-len(computed)),'PER_IMAGE_COUNTS')
            keep=[i for i,c in enumerate(row['floor_confidences']) if c>=.1]
            require(keep==row['retained_mask_indices'],'RETAINED_INDEX_MISMATCH')
            require([row['floor_bboxes'][i] for i in keep]==row['pred_boxes'] and [row['floor_confidences'][i] for i in keep]==row['confidences'],'FLOOR_RETAINED_ALIGNMENT')
            require(all(c>=.01 for c in row['floor_confidences']),'FLOOR_CONTRACT')
            matched={p['gt'] for p in row['pairs']}
            require(row['per_instance_gt']==[dict(index=i,bbox=b,size_group='small' if size_bin((b[2]-b[0])*(b[3]-b[1])/512**2) in BINS[:5] else 'medium' if (b[2]-b[0])*(b[3]-b[1])/512**2<.05 else 'large',matched=i in matched) for i,b in enumerate(row['gt_boxes'])],'INSTANCE_RECORD_MISMATCH')
            path=directory/row['prediction_mask_reference']
            require(path.resolve().is_relative_to(directory.resolve()) and sha(path)==row['prediction_mask_sha256'],'SAVED_MASK_HASH')
            with np.load(path,allow_pickle=False) as masks:
                require(masks['masks'].shape==(row['floor_prediction_count'],512,512),'MASK_SHAPE')
                require(masks['boxes'].tolist()==row['floor_bboxes'] and masks['confidences'].tolist()==row['floor_confidences'] and masks['retained_indices'].tolist()==keep,'MASK_ARCHIVE_ALIGNMENT')
            ref=next(r for r in manifest if r['sample_id']==row['sample_id'])
            require(row['image_sha256']==ref['image_sha256'] and row['mask_sha256']==ref['mask_sha256'],'ROW_GT_HASH')
            with Image.open(ROOT/ref['relative_mask_path']) as image:
                mask=binary_mask(image)
            require(mask.ndim==2 and mask.shape==(512,512) and int(mask.sum())==row['gt_pixels'],'GT_MASK_PIXELS')
            crop=row['crop']; retained=int(mask[crop[1]:crop[3],crop[0]:crop[2]].sum()) if crop else 0
            require(retained==row['retained_gt_wound_pixels'],'CROP_RETAINED_PIXELS')
            require(row['crop_complete95']==bool(row['gt_pixels'] and retained/row['gt_pixels']>=.95),'CROP_CONTRACT')
            if row['gt_pixels']:require(abs(row['crop_coverage']-retained/row['gt_pixels'])<1e-12,'CROP_COVERAGE')
        totals=[sum(r[k] for r in rows) for k in ('tp','fp','fn')]
        require(totals==([203,30,38] if arm=='C4' else [203,35,38]),'AGGREGATE_COUNT_MISMATCH')
        require(sum(r['crop_complete95'] for r in rows)==(169 if arm=='C4' else 168),'CROP_TOTAL_MISMATCH')
        data[arm]={r['sample_id']:r for r in rows}
    require(sum(len(b) for b in gt.values())==241,'GT_IDENTITY_COUNT')
    # Snapshot every frozen F1.2 artifact, including checkpoints; hashing is not deserialization.
    for directory in [PAIR,PRE,RESULTS/'f_higher_scale_v2_recovery_seed42_h4_1024']:
        inputs.update(p for p in directory.rglob('*') if p.is_file())
    inputs.add(ROOT/'docs/PHASE_F12_H4_REPLACEMENT_RECOVERY_20260928.md')
    before=snapshot(inputs)
    return data,manifest,comparison,before


def analyze(data):
    master=[]; misses=[]; fps=[]; pairing=[]; crops=[]; images=[]; quality=[]; negatives=[]
    for sid in sorted(data['C4']):
        c,h=data['C4'][sid],data['H4'][sid]; n=len(c['gt_boxes']); grouping=group(n)
        matched={a:{p['gt']:p for p in data[a][sid]['pairs']} for a in ARMS}
        for gi,b in enumerate(c['gt_boxes']):
            ratio=(b[2]-b[0])*(b[3]-b[1])/512**2
            row=dict(sample_id=sid,GT_instance_id=gi,size_ratio=ratio,size_bin=size_bin(ratio),image_GT_count=n,single_multi=grouping,
                     C4_detected=gi in matched['C4'],H4_detected=gi in matched['H4'],transition=transition(gi in matched['C4'],gi in matched['H4']))
            for a,r in [('C4',c),('H4',h)]:
                p=matched[a].get(gi)
                row[a+'_matched_prediction_index']=p['prediction'] if p else None
                row[a+'_matched_confidence']=r['confidences'][p['prediction']] if p else None
                row[a+'_bbox_iou']=p['iou'] if p else None
                row[a+'_mask_iou']=None; row[a+'_mask_dice']=None
                diag=miss(r,gi) if not p else None
                row[a+'_miss_diagnostic']=diag['diagnostic'] if diag else None
                if row['transition'] in ['C4_ONLY','H4_ONLY'] and not p:
                    other='H4' if a=='C4' else 'C4'; success=matched[other][gi]; sr=data[other][sid]
                    misses.append(dict(sample_id=sid,GT_instance_id=gi,transition=row['transition'],size_bin=row['size_bin'],size_ratio=ratio,
                                       image_GT_count=n,single_multi=grouping,missed_arm=a,**diag,
                                       success_confidence=sr['confidences'][success['prediction']],success_bbox_iou=success['iou'],
                                       success_mask_iou=None,success_mask_dice=None))
            master.append(row)
            if row['transition']=='BOTH_DETECTED':
                quality.append({**row,'delta_bbox_iou':row['H4_bbox_iou']-row['C4_bbox_iou'],
                                'delta_confidence':row['H4_matched_confidence']-row['C4_matched_confidence'],
                                'delta_mask_iou':None,'delta_mask_dice':None,'mask_metric_availability':'NA_NO_AUTHORITATIVE_INSTANCE_MASK_IDENTITY'})
        local={}
        for a,r in [('C4',c),('H4',h)]:
            used={p['prediction'] for p in r['pairs']}
            local[a]=[fp_record(a,r,i) for i in range(len(r['pred_boxes'])) if i not in used]
            fps.extend(local[a])
        pairing.extend(pair_fp(local['C4'],local['H4']))
        croptrans = ('BOTH_CROP_PASS' if c['crop_complete95'] and h['crop_complete95'] else
                     'C4_ONLY_CROP_PASS' if c['crop_complete95'] else 'H4_ONLY_CROP_PASS' if h['crop_complete95'] else 'BOTH_CROP_FAIL') if n else None
        image=dict(sample_id=sid,GT_count=n,single_multi=grouping,
                   **{f'{a}_{k.upper()}':r[k] for a,r in [('C4',c),('H4',h)] for k in ['tp','fp','fn']},
                   **{f'delta_{k.upper()}':h[k]-c[k] for k in ['tp','fp','fn']},
                   C4_crop_pass=c['crop_complete95'] if n else None,H4_crop_pass=h['crop_complete95'] if n else None,crop_transition=croptrans)
        images.append(image)
        if n:
            cr=dict(sample_id=sid,image_GT_count=n,single_multi=grouping,transition=croptrans)
            for a,r in [('C4',c),('H4',h)]:
                crop=r['crop']; failure=None if r['crop_complete95'] else 'NO_ROI' if crop is None else 'ROI_PRESENT_BUT_GT_RETENTION_BELOW_95'
                excluded=[i for i,b in enumerate(r['gt_boxes']) if crop and i not in matched[a] and
                          not(crop[0]<=b[0] and crop[1]<=b[1] and crop[2]>=b[2] and crop[3]>=b[3])]
                cr.update({a+'_crop_pass':r['crop_complete95'],a+'_failure':failure,a+'_retained_fraction':r['crop_coverage'],
                           a+'_crop_bbox':crop,a+'_unmatched_GT_bboxes_not_fully_inside_ROI':excluded,
                           a+'_PRIMARY_ROI_MISSES_OTHER_GT':None,
                           a+'_multi_unmatched_bbox_exclusion':bool(n>1 and failure and crop and excluded)})
            crops.append(cr)
        else:
            negatives.append(dict(sample_id=sid,C4_prediction_count=len(c['pred_boxes']),H4_prediction_count=len(h['pred_boxes']),
                                  C4_confidences=c['confidences'],H4_confidences=h['confidences']))
    counts=lambda rs,key:{k:sum(r[key]==k for r in rs) for k in (TRANSITIONS if key=='transition' else [])}
    overall=counts(master,'transition')
    require(overall['C4_ONLY']==overall['H4_ONLY'],'FAIL_TRANSITION_ACCOUNTING')
    sizes=[dict(size_bin=b,support=sum(r['size_bin']==b for r in master),**counts([r for r in master if r['size_bin']==b],'transition')) for b in BINS]
    require([r['support'] for r in sizes]==[27,22,38,30,20,90,14],'SIZE_SUPPORT_MISMATCH')
    for a,expected in [('C4',[10,16,33,28,17,85,14]),('H4',[9,16,35,24,18,87,14])]:
        require([r['BOTH_DETECTED']+r[a+'_ONLY'] for r in sizes]==expected,'SIZE_TP_MISMATCH')
    cats=[dict(category=k,C4=sum(r['arm']=='C4' and r['category']==k for r in fps),H4=sum(r['arm']=='H4' and r['category']==k for r in fps)) for k in FP_CATS]
    for r in cats:r['delta']=r['H4']-r['C4']
    pc=dict(Counter(r['pairing'] for r in pairing))
    require(pc.get('SHARED_FP',0)+pc.get('C4_ONLY_FP',0)==30 and pc.get('SHARED_FP',0)+pc.get('H4_ONLY_FP',0)==35,'FP_CHURN_ACCOUNTING')
    fpstrata={g:{a:sum(r['single_multi']==g and r['arm']==a for r in fps) for a in ARMS} for g in ['single','multi','negative']}
    require(fpstrata=={'single':{'C4':21,'H4':22},'multi':{'C4':7,'H4':11},'negative':{'C4':2,'H4':2}},'FP_STRATA_MISMATCH')
    cropcounts=lambda rs:{k:sum(r['transition']==k for r in rs) for k in ['BOTH_CROP_PASS','C4_ONLY_CROP_PASS','H4_ONLY_CROP_PASS','BOTH_CROP_FAIL']}
    cropgroups={g:cropcounts([r for r in crops if r['single_multi']==g]) for g in ['single','multi']}
    for g,cc,hh in [('single',147,144),('multi',22,24)]:
        rc=cropgroups[g]
        require(rc.get('BOTH_CROP_PASS',0)+rc.get('C4_ONLY_CROP_PASS',0)==cc and rc.get('BOTH_CROP_PASS',0)+rc.get('H4_ONLY_CROP_PASS',0)==hh,'CROP_STRATA_MISMATCH')
    fpconfidence={g:{a:stats(r[a+'_confidence'] for r in pairing if r['pairing']==g) for a in ARMS} for g in ['SHARED_FP','C4_ONLY_FP','H4_ONLY_FP']}
    summary=dict(overall=overall,very_small=counts([r for r in master if r['size_ratio']<.0025],'transition'),
                 size_bins=sizes,GT_group_transitions={g:counts([r for r in master if r['single_multi']==g],'transition') for g in ['single','multi']},
                 miss_diagnostics={t:{k:sum(r['transition']==t and r['diagnostic']==k for r in misses) for k in MISS_CATS} for t in ['C4_ONLY','H4_ONLY']},
                 fp_categories=cats,fp_pairing=pc,fp_strata=fpstrata,fp_confidence=fpconfidence,
                 fp_pairing_strata={g:{k:sum(r['single_multi']==g and r['pairing']==k for r in pairing) for k in ['SHARED_FP','C4_ONLY_FP','H4_ONLY_FP']} for g in ['single','multi','negative']},
                 crop_transitions=cropcounts(crops),crop_group_transitions=cropgroups,negative_images=negatives,
                 negative_image_sets_equal={r['sample_id'] for r in negatives if r['C4_prediction_count']}=={r['sample_id'] for r in negatives if r['H4_prediction_count']},
                 paired_quality={metric:stats(r[metric] for r in quality) for metric in ['delta_bbox_iou','delta_confidence','delta_mask_iou','delta_mask_dice']},
                 image_union_mask_quality={m:stats(data['H4'][sid][m]-data['C4'][sid][m] for sid in data['C4'] if data['C4'][sid]['gt_pixels']) for m in ['mask_iou','mask_dice']})
    tables={'gt_transition_matrix.csv':master,'very_small_transition_matrix.csv':[r for r in master if r['size_ratio']<.0025],
            'size_bin_transition_summary.csv':sizes,'miss_diagnostics.csv':misses,'fp_master.csv':fps,'fp_pairing.csv':pairing,
            'fp_category_summary.csv':cats,'crop_transition_matrix.csv':crops,'image_level_pair_summary.csv':images,
            'paired_prediction_quality.csv':quality}
    return tables,summary


def render(data,manifest,tables):
    dest=OUT/'paired_error_visualizations'; dest.mkdir(exist_ok=True)
    selected={}
    for r in tables['very_small_transition_matrix.csv']:
        if r['transition'] in ['C4_ONLY','H4_ONLY']:selected.setdefault(r['sample_id'],[]).append(r['transition']+':GT'+str(r['GT_instance_id']))
    for r in tables['fp_pairing.csv']:
        if r['pairing']!='SHARED_FP':selected.setdefault(r['sample_id'],[]).append(r['pairing'])
    for r in tables['crop_transition_matrix.csv']:
        if r['transition'] in ['C4_ONLY_CROP_PASS','H4_ONLY_CROP_PASS']:selected.setdefault(r['sample_id'],[]).append(r['transition'])
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    refs={r['sample_id']:r for r in manifest}; index=[]
    for sid in sorted(selected):
        with Image.open(ROOT/refs[sid]['relative_image_path']) as source:image=source.convert('RGB')
        panel=Image.new('RGB',(1048,648),'white'); draw=ImageDraw.Draw(panel)
        draw.text((12,6),sid+' | '+', '.join(sorted(set(selected[sid]))),font=font,fill='black')
        for a,x in [('C4',8),('H4',528)]:
            row=data[a][sid]; tile=image.copy(); d=ImageDraw.Draw(tile)
            matches={p['prediction'] for p in row['pairs']}
            for gi,b in enumerate(row['gt_boxes']):
                d.rectangle(b,outline='#00ff50',width=2); d.text((b[0],max(0,b[1]-18)),f'GT{gi}',font=font,fill='#00ff50',stroke_width=1,stroke_fill='black')
            for pi,b in enumerate(row['pred_boxes']):
                color='#ff4040' if pi not in matches else '#ffb000'
                d.rectangle(b,outline=color,width=2); d.text((b[0],min(490,b[3])),f'P{pi} {row["confidences"][pi]:.3f}',font=font,fill=color,stroke_width=1,stroke_fill='black')
            if row['crop']:d.rectangle(row['crop'],outline='#00dfff',width=2)
            panel.paste(tile,(x,62))
            draw.text((x,36),f'{a} | TP {row["tp"]} FP {row["fp"]} FN {row["fn"]}',font=font,fill='black')
            coverage='NA' if row['crop_coverage'] is None else f'{row["crop_coverage"]:.4f}'
            draw.text((x,580),'Crop retained fraction: '+coverage,font=font,fill='black')
        draw.text((8,610),'Green: frozen GT | Orange: matched prediction | Red: FP | Cyan: saved union ROI',font=font,fill='black')
        path=dest/(Path(sid).stem+'.png'); panel.save(path)
        index.append(dict(sample_id=sid,reasons=sorted(set(selected[sid])),path=str(path.relative_to(OUT)),sha256=sha(path)))
    save(dest/'index.json',index)
    return index


def md_table(rows,keys):
    return '| '+' | '.join(keys)+' |\n| '+' | '.join('---' for _ in keys)+' |\n'+'\n'.join('| '+' | '.join('NA' if r.get(k) is None else str(r[k]) for k in keys)+' |' for r in rows)+'\n'


def report(summary,tables,visuals):
    s=summary;o=s['overall'];v=s['very_small'];f=s['fp_pairing'];ct=s['crop_transitions']
    lines=['# Phase F1.3 — 配對偵測轉換、誤報與裁切稽核','',
           'PHASE_F13_STATUS = COMPLETE。僅使用 F1.2 已保存結果，未載入模型、未重新推論、未訓練。',
           '', '## 方法與可追溯性','',
           'F1.2 原 gate=FAIL 保持不變。本次是 single-seed、asymmetric replacement recovery、同一 development validation 的 paired descriptive error analysis；不是統計確認、泛化估計或臨床驗證。不計算 p-value 或 bootstrap CI。',
           '', '191 個 sample_id、241 個 GT 身分以相同影像與凍結 polygon 非空行的零起算 index 對齊。逐行 bbox 與 SHA256 均核對，保存 matching 為權威；固定規則重算只作一致性驗證。',
           '', 'Confidence=.10、prediction floor=.01、NMS=.70、match IoU=.50、推論尺寸768、15% ROI margin 不變。floor 候選已經過既有 NMS，不能拿來判斷 pre-NMS 是否曾有候選。',
           '', 'FP 跨模型 pairing：同 image、IoU≥.50，依 IoU 由高到低；同值依兩框較低 confidence 由高到低、較高 confidence 由高到低、C4 index、H4 index 由小到大。SHARED_FP 只是此幾何規則配對，不代表已確認相同組織。',
           '', '每個成功 GT 的 bbox IoU/confidence 可追溯到保存的 pair。逐 GT mask IoU/Dice=NA：GT 原始 mask 是語義聯集，未保存逐 polygon 對應的權威 instance mask，不能將影像 union mask 分數充當 GT instance 分數。',
           '', 'Crop 直接核對保存 ROI 在原 GT union mask 中保留≥95%像素，不重建或修改 ROI。原規則是所有保留 polygon 的 union ROI，不是 primary lesion ROI，所以 PRIMARY_ROI_MISSES_OTHER_GT 標 NA；另提供 multi-GT 中 unmatched bbox 未完全包含於 ROI 的幾何旗標，不當作逐實例像素證明。',
           '', '## 核心轉換','',md_table([o],TRANSITIONS),
           f"總 TP 均203，但有 {o['C4_ONLY']} 個 C4-only 與 {o['H4_ONLY']} 個 H4-only，另有 {o['BOTH_MISSED']} 個共同漏偵測。數量相等已逐身分核實。",
           '',md_table(s['size_bins'],['size_bin','support',*TRANSITIONS]),
           '', '## 21 個問題的回答','',
           f"1. C4_ONLY：{o['C4_ONLY']} 個。",
           f"2. H4_ONLY：{o['H4_ONLY']} 個。",
           f"3. 兩者相等，且 BOTH_DETECTED+任一 only=203；總身分=241，並非僅 aggregate 算術假設。",
           '4. 所有 size-bin 四格 counts 見上表與 size_bin_transition_summary.csv。',
           f"5. Very-small：失去 {v['C4_ONLY']}、得到 {v['H4_ONLY']}，共同偵測 {v['BOTH_DETECTED']}、共同漏掉 {v['BOTH_MISSED']}，淨 -1。",
           '6. <0.10% 的全部 gained/lost 身分如下（沒有人工挑例）：','']
    changed=[r for r in tables['gt_transition_matrix.csv'] if r['transition'] in ['C4_ONLY','H4_ONLY']]
    lines.append(md_table([r for r in changed if r['size_bin']=='<0.10%'],['sample_id','GT_instance_id','size_ratio','transition']))
    for q,b in [(7,'0.25-<0.50%'),(8,'0.50-<0.75%')]:
        r=next(r for r in s['size_bins'] if r['size_bin']==b)
        lines += [f"{q}. {b}：gain {r['H4_ONLY']}、loss {r['C4_ONLY']}，所以淨 {r['H4_ONLY']-r['C4_ONLY']:+d} TP。機制見 miss 診斷，不宣稱因果。",'',md_table([r for r in changed if r['size_bin']==b],['sample_id','GT_instance_id','size_ratio','transition'])]
    lines += ['9. Single/multi GT 交換：','',md_table([dict(group=g,**r) for g,r in s['GT_group_transitions'].items()],['group',*TRANSITIONS]),
              '10–11. 固定優先順序的雙向 miss 診斷：','',md_table([dict(direction=k,**r) for k,r in s['miss_diagnostics'].items()],['direction',*MISS_CATS]),
              'C4→H4-R 的8次 loss 中，6次有保留框但定位 IoU 不足；例如 fuseg__0604.png 的 GT1 IoU=0.4999786039，嚴格低於0.50，不可四捨五入成 TP。反方向8次 gain 中，4次在 C4 保存了低於 operating threshold 的候選。',
              'BELOW_FROZEN_CONFIDENCE 僅表示保存候選存在於 frozen operating threshold 以下，不表示降低門檻會改善整體表現。','',
              '12. FP 類型與淨差：','',md_table(s['fp_categories'],['category','C4','H4','delta']),
              'OTHER_REGION_FP +5 是主要淨增來源；localization +1 與 duplicate/competition −1 抵銷。OTHER_REGION 指與凍結 GT bbox 的 IoU=0，不等於經醫護確認為正常皮膚，也不能排除標註未涵蓋的病灶。',
              f"13. H4_ONLY_FP={f.get('H4_ONLY_FP',0)}。",
              f"14. C4_ONLY_FP={f.get('C4_ONLY_FP',0)}；SHARED_FP={f.get('SHARED_FP',0)}。+5 是新增與消失抵銷後的淨差，不是只有五個新 FP。",
              '15. Multi-GT FP 7→11（+4）；single 21→22（+1）；negative 2→2（0）。淨增加的4/5在 multi images，但僅為描述性關聯。實際 unshared FP 按分組見附表。',
              f"16. Negative-image FP 是否同一批：{s['negative_image_sets_equal']}。全部五張的數量與置信度：",'',md_table(s['negative_images'],['sample_id','C4_prediction_count','H4_prediction_count','C4_confidences','H4_confidences']),
              f"17. Crop gain={ct.get('H4_ONLY_CROP_PASS',0)}、loss={ct.get('C4_ONLY_CROP_PASS',0)}；其餘見下表。",'',md_table([dict(group='all',**ct)]+[dict(group=g,**r) for g,r in s['crop_group_transitions'].items()],['group','BOTH_CROP_PASS','C4_ONLY_CROP_PASS','H4_ONLY_CROP_PASS','BOTH_CROP_FAIL'])]
    for q,g in [(18,'single'),(19,'multi')]:
        r=s['crop_group_transitions'][g]
        lines.append(f"{q}. {g} crop gain={r.get('H4_ONLY_CROP_PASS',0)}、loss={r.get('C4_ONLY_CROP_PASS',0)}，淨 {r.get('H4_ONLY_CROP_PASS',0)-r.get('C4_ONLY_CROP_PASS',0):+d}；逐張 retained fraction、NO_ROI/不足95%與 bbox exclusion 見 crop_transition_matrix.csv。")
    lines += [f"20. {s['PRIMARY_FAILURE_STRUCTURE']}。總 TP 未增加，不支持 global sensitivity gain。",
              f"21. 唯一建議：{s['RECOMMENDED_NEXT_SINGLE_INTERVENTION']}。{s['recommendation_rationale']}",
              '', '## 配對品質（H4-R − C4）','',md_table([dict(metric=k,**r) for k,r in s['paired_quality'].items()],['metric','n','mean','median','q25','q75','min','max']),
              '', 'mask NA 的原因見方法。image-union mask 品質另存 JSON，只能作影像層次描述。',
              '', '## FP churn 置信度分布','',md_table([dict(pairing=g,arm=a,**r) for g,arms in s['fp_confidence'].items() for a,r in arms.items()],['pairing','arm','n','mean','median','q25','q75','min','max']),
              '', '空組 n=0，其統計值 NA，不補零。未做 threshold sweep 或提出 confidence threshold。',
              '', '## FP churn 按影像組別','',md_table([dict(group=g,**r) for g,r in s['fp_pairing_strata'].items()],['group','SHARED_FP','C4_ONLY_FP','H4_ONLY_FP']),
              '', '## 規則化視覺化','',f'全部符合條件的 {len(visuals)} 張唯一影像均保存，每張左右分別 C4/H4-R。index.json 列出納入理由及 SHA256。以下只展示 sample_id 排序的前3張，不以好壞選圖。','']
    for r in visuals[:3]:lines += [f"![{r['sample_id']}]({(OUT/r['path']).as_posix()})",'']
    lines += ['## 交付與邊界','', '10 份 CSV、audit_summary.json、integrity.json 與全部規則化圖表位於 experiments/results/f_higher_scale_v2_paired_error_audit/。CSV 一列一個分析單位，NA 為無資料／不適用，原始 F1.2 不覆寫。',
              '', '工程紀錄：首次新稽核介面漏做凍結規則中的相同RGB通道轉灰階，於輸出分析表之前停止。只修正F1.3介面並新增回歸測試；未更改GT或F1.2，原停止紀錄保留 initial_engineering_attempt.json。完整重新核對通過後才完成報告。',
              '', '```ini','PHASE_F13_STATUS = COMPLETE','PAIRED_ERROR_ANALYSIS = COMPLETE',
              *[f'{k} = {str(v).lower() if isinstance(v,bool) else v}' for k,v in FLAGS.items()],
              'PRIMARY_FAILURE_STRUCTURE = '+s['PRIMARY_FAILURE_STRUCTURE'],
              'RECOMMENDED_NEXT_SINGLE_INTERVENTION = '+s['RECOMMENDED_NEXT_SINGLE_INTERVENTION'],'```','',
              'STOP AFTER F1.3。建議不是執行授權；不訓練、不重跑、不 multi-seed、不換尺度、不調門檻、不改 crop、不 external test、不替換 App。','']
    return '\n'.join(lines).replace('| None |','| NA |')


def finalize():
    require(OUT.exists() and not REPORT.exists(),'F13_FINALIZE_OUTPUT_CONTRACT')
    require(read(OUT/'audit_summary.json')['PHASE_F13_STATUS']=='ANALYZED_PENDING_REPORT','F13_NOT_PENDING_REPORT')
    guard()
    try:
        baseline=read(OUT/'input_snapshot.json')['sha256']
        require(snapshot(map(Path,baseline))==baseline,'PROTECTED_SHA256_MISMATCH')
        data,manifest,comparison,verified=preflight()
        require(verified==baseline,'PROTECTED_INVENTORY_OR_HASH_CHANGED')
        tables,s=analyze(data)
        # Evidence reviewed only AFTER complete analysis. One recommendation, not execution.
        s.update(PRIMARY_FAILURE_STRUCTURE='MIXED_PATTERN: DETECTION_REDISTRIBUTION + FP_INFLATION',
                 RECOMMENDED_NEXT_SINGLE_INTERVENTION='FALSE_POSITIVE_CONTROL',
                 recommendation_rationale='建議下一階段僅預登錄「train-only、來源可追溯且已確認非傷口的背景負樣本監督」作單一因素，固定模型、尺度、loss、threshold及crop；不得把這191張validation的FP搬進training或拿未標註區域逕當負樣本。依據是 OTHER_REGION_FP 9→14，且multi-GT淨FP +4。這不保證能解決6次定位型loss或提升very-small TP，必須保留小傷口recall與crop安全門檻，未具備可信負樣本前不得啟動。',
                 PHASE_F13_STATUS='COMPLETE',PAIRED_ERROR_ANALYSIS='COMPLETE',**FLAGS)
        for name,rows in tables.items():
            with (OUT/name).open(newline='',encoding='utf-8-sig') as f:
                old=list(csv.DictReader(f))
            require(len(old)==len(rows),'DELIVERED_CSV_COUNT_CHANGED')
            for x,y in zip(old,rows):
                for k,v in y.items():
                    wanted='NA' if v is None else json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else str(v)
                    require(x[k]==wanted,'DELIVERED_CSV_VALUE_CHANGED: '+name+':'+k)
        visuals=read(OUT/'paired_error_visualizations/index.json')
        for v in visuals:require(sha(OUT/v['path'])==v['sha256'],'VISUALIZATION_CHANGED')
        expected=set()
        for r in tables['very_small_transition_matrix.csv']:
            if r['transition'] in ['C4_ONLY','H4_ONLY']:expected.add(r['sample_id'])
        for r in tables['fp_pairing.csv']:
            if r['pairing']!='SHARED_FP':expected.add(r['sample_id'])
        for r in tables['crop_transition_matrix.csv']:
            if r['transition'] in ['C4_ONLY_CROP_PASS','H4_ONLY_CROP_PASS']:expected.add(r['sample_id'])
        require([v['sample_id'] for v in visuals]==sorted(expected),'VISUALIZATION_COVERAGE_FAILED')
        s.update(identity_contract=dict(images=191,GT_instances=241,aligned=True,GT_key=['sample_id','zero_based_nonempty_polygon_line_index']),
                 visualization_image_count=len(visuals),F12_gate_unchanged='FAIL',
                 csv_rows={name:len(rows) for name,rows in tables.items()},
                 caveats=['Single seed; asymmetric replacement; same development validation.',
                          'GT-instance mask metrics NA: no authoritative instance mask association.',
                          'Saved floor candidates are post-NMS; no threshold performance inference.',
                          'OTHER_REGION is geometry relative to annotation, not a confirmed normal-tissue diagnosis.'],
                 recommendation_executed=False,finished_at=datetime.now(timezone.utc).isoformat())
        save(OUT/'audit_summary.json',s)
        REPORT.write_text(report(s,tables,visuals),encoding='utf-8')
        require(snapshot(map(Path,baseline))==baseline,'POSTFLIGHT_PROTECTED_SHA256_MISMATCH')
        require(not any(m in sys.modules for m in ['torch','ultralytics','tensorflow','onnxruntime']),'MODEL_MODULE_LOADED')
        outputs=snapshot([p for p in OUT.rglob('*') if p.is_file() and p.name!='integrity.json']+[REPORT])
        save(OUT/'integrity.json',dict(status='PASS',PHASE_F13_STATUS='COMPLETE',historical_expected_hashes='PASS',
             before_equals_after=True,source_file_count=len(baseline),source_sha256=baseline,output_sha256=outputs,
             audit_source_sha256=sha(Path(__file__)),tests_source_sha256=sha(ROOT/'tests/test_phase_f13.py'),
             checks=['191_IMAGE_IDENTITIES','241_ORDERED_GT_IDENTITIES','C4_203_30_38','H4_203_35_38','SIZE_COUNTS','CROP_169_168',
                     'BEST_LAST_CHECKPOINT_HASHES','MASK_ARCHIVE_HASHES','FIXED_MATCHING','FLOOR_RETAINED_ALIGNMENT',
                     'ALL_CSV_VALUES_RECHECKED','35_RULE_SELECTED_IMAGES_COMPLETE'],**FLAGS))
        print(json.dumps(dict(status='COMPLETE',overall=s['overall'],very_small=s['very_small'],fp_pairing=s['fp_pairing'],crop=s['crop_transitions'],recommendation=s['RECOMMENDED_NEXT_SINGLE_INTERVENTION']),ensure_ascii=False))
    except Exception as exc:
        save(OUT/'audit_summary.json',dict(PHASE_F13_STATUS='BLOCKED',error=str(exc),**FLAGS))
        save(OUT/'integrity.json',dict(status='FAIL',error=str(exc),**FLAGS))
        raise


def run():
    repair = OUT.exists() and {p.name for p in OUT.iterdir()}=={'audit_summary.json','integrity.json'} and read(OUT/'audit_summary.json').get('error')=='GT_MASK_PIXELS'
    require((not OUT.exists() or repair) and not REPORT.exists(),'F13_OUTPUT_EXISTS_REVIEW_BEFORE_OVERWRITE')
    guard(); OUT.mkdir(exist_ok=repair)
    if repair:
        save(OUT/'initial_engineering_attempt.json',dict(summary=read(OUT/'audit_summary.json'),integrity=read(OUT/'integrity.json'),
             cause='New audit adapter omitted frozen identical-RGB-to-grayscale mask normalization. Corrected audit adapter only; no input hash mismatch, no inference, no previous analysis tables produced.'))
    before={}
    try:
        data,manifest,comparison,before=preflight()
        save(OUT/'input_snapshot.json',dict(sha256=before,at=datetime.now(timezone.utc).isoformat()))
        print('F13 preflight PASS: saved hashes, 191 images, 241 ordered GT, frozen matching/crop.',flush=True)
        tables,summary=analyze(data)
        # Recommendation is deliberately deferred until every transition audit is computed.
        summary.update(PRIMARY_FAILURE_STRUCTURE='MIXED_PATTERN: DETECTION_REDISTRIBUTION + FP_INFLATION',
                       RECOMMENDED_NEXT_SINGLE_INTERVENTION='PENDING_EVIDENCE_REVIEW',recommendation_rationale='Pending completed descriptive evidence review.',
                       PHASE_F13_STATUS='ANALYZED_PENDING_REPORT',PAIRED_ERROR_ANALYSIS='COMPLETE',**FLAGS)
        for name,rows in tables.items():csv_write(name,rows)
        visuals=render(data,manifest,tables)
        summary['visualization_image_count']=len(visuals)
        summary['identity_contract']=dict(images=191,GT_instances=241,GT_key=['sample_id','zero_based_nonempty_polygon_line_index'],aligned=True)
        summary['F12_gate_unchanged']='FAIL'
        after=snapshot(map(Path,before))
        require(before==after,'PROTECTED_POSTFLIGHT_SHA256_MISMATCH')
        require(not any(m in sys.modules for m in ['torch','ultralytics','tensorflow','onnxruntime']),'MODEL_MODULE_LOADED')
        save(OUT/'audit_summary.json',summary)
        save(OUT/'integrity.json',dict(status='PASS',historical_expected_hashes='PASS',before_equals_after=True,
                                      source_file_count=len(before),source_sha256=after,checks=['191_IMAGE_IDENTITIES','241_ORDERED_GT_IDENTITIES','C4_203_30_38','H4_203_35_38','SIZE_COUNTS','CROP_169_168','MASK_ARCHIVE_HASHES','FIXED_MATCHING','FLOOR_RETAINED_ALIGNMENT'],
                                      audit_source_sha256=sha(Path(__file__)),**FLAGS))
        print(json.dumps({k:v for k,v in summary.items() if k not in ['fp_confidence','image_union_mask_quality']},ensure_ascii=False,indent=2))
    except Exception as exc:
        save(OUT/'audit_summary.json',dict(PHASE_F13_STATUS='BLOCKED',error=str(exc),**FLAGS))
        save(OUT/'integrity.json',dict(status='FAIL',error=str(exc),before_equals_after=(before==snapshot(map(Path,before))) if before else None,**FLAGS))
        raise


if __name__=='__main__':
    if sys.argv[1:]==['finalize']:finalize()
    else:run()
