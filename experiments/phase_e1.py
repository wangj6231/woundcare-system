"""E1 preregistration and training-label simulation only; no execution entrypoint for E2."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

from experiments import patch_geometry as g
from experiments.phase_d0 import sha, read, write

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'E_PATCH_V1'
PROTOCOLS = Path('experiments/protocols')
AUDIT = Path('experiments/results/e_patch_v1_preregistration_audit')
FUTURE = Path('experiments/results/e_patch_v1_seed42')
REPORT = Path('docs/PHASE_E1_PATCH_TRAINING_PREREGISTRATION_20260925.md')
REQUEST = Path('C:/Users/milo9/.codex/attachments/d0d91c56-d25c-4460-a674-3177cb02553c/貼上的文字.txt')
CONTROL = Path('experiments/results/d_seg_small_sampling_v2_control')


def output(suffix):
    return ROOT / PROTOCOLS / f'{PREFIX}_{suffix}.json'


def require_future_absent(root):
    g.require(not (root/FUTURE).exists(), 'FAIL_FUTURE_OUTPUT_ALREADY_EXISTS')


def synthetic_figures(out):
    """Synthetic geometry diagrams; no real training or validation pixels."""
    from PIL import Image, ImageDraw
    def square(x,y,w,h): return [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
    scenes = {'center_target':[square(250,250,12,12)], 'corner_target':[square(0,0,8,8)],
              'edge_target':[square(500,220,12,14)], 'multiple_GT':[square(250,250,8,8),square(300,300,30,30)],
              'cross_boundary_secondary_GT':[square(250,250,8,8),square(360,270,70,70)],
              'tiny_polygon':[square(100,100,.5,.5)]}
    files = []
    for name, polygons in scenes.items():
        r = g.transform(polygons,0)
        canvas = Image.new('RGB',(1050,570),'white');d=ImageDraw.Draw(canvas)
        d.rectangle((8,40,520,552),fill='#eeeeee');d.rectangle((530,40,1042,552),fill='#eeeeee')
        d.text((8,12),name+' | synthetic original 512',fill='black');d.text((530,12),'256 patch (displayed at 2x)',fill='black')
        for i,p in enumerate(polygons):
            d.polygon([(x+8,y+40) for x,y in p],outline='red' if i==r['target_GT_id'] else 'blue',width=2)
        x,y,x2,y2 = r['patch_bounds'];d.rectangle((x+8,y+40,x2+8,y2+40),outline='green',width=2)
        for item in r['records']:
            if item['polygon']:
                d.polygon([(a*512+530,b*512+40) for a,b in item['polygon']],outline='red' if item['GT_instance_id']==r['target_GT_id'] else 'blue',width=2)
        path=out/(name+'.png');canvas.save(path);files.append(path.relative_to(ROOT).as_posix())
    return files


def build():
    g.model_guard();require_future_absent(ROOT)
    g.require(not list((ROOT/PROTOCOLS).glob(PREFIX+'*')), 'FAIL_E1_EXISTS_REFUSE_OVERWRITE')
    g.require(not (ROOT/AUDIT).exists(), 'FAIL_E1_AUDIT_EXISTS_REFUSE_OVERWRITE')
    protected = {}
    def protect(path, expected=None):
        path=Path(path).resolve();h=sha(path)
        g.require(expected is None or h==expected, 'FAIL_SOURCE_HASH: '+str(path))
        protected[str(path)]=h
        return h
    # Verify frozen config/code identities, without reading held-out pixels or predictions.
    freeze=ROOT/PROTOCOLS/'D_SEG_SMALL_SAMPLING_V2_freeze.json'
    protect(freeze,'16b0f66c3661662ea9840774c835cfc9ced1e6ce1da8f7fe1d0b1a1067f0245b')
    for rel,h in read(freeze)['artifacts_sha256'].items(): protect(ROOT/rel,h)
    e0=ROOT/'experiments/results/very_small_failure_audit'
    e0index=read(e0/'artifact_manifest.json');protect(e0/'artifact_manifest.json')
    for rel,h in e0index['sha256'].items():
        if rel.endswith('very_small_failure_summary.json'):protect(ROOT/rel,h)
    g.require(read(e0/'very_small_failure_summary.json')['PHASE_E0_STATUS']=='COMPLETE','FAIL_E0_STATUS')
    cfgpath=ROOT/PROTOCOLS/'D_SEG_SMALL_SAMPLING_V2_control_config.json';cfg=read(cfgpath)
    mp=ROOT/cfg['manifest']['path'];protect(mp,cfg['manifest']['sha256']);rows=read(mp)['samples']
    g.require(len(rows)==771,'FAIL_TRAIN_COUNT');ids=[r['sample_id'] for r in rows]
    # Existing validation manifest is read ONLY for exclusion identities, never image/label data or outcomes.
    ep=read(ROOT/PROTOCOLS/'D_SEG_SMALL_SAMPLING_V2_evaluation_protocol.json')
    vp=ROOT/ep['validation_manifest']['path'];protect(vp,ep['validation_manifest']['sha256'])
    val=read(vp)['samples']
    g.require(not set(ids)&{r['sample_id'] for r in val},'FAIL_VALIDATION_ID_OVERLAP')
    g.require(not {r['image_hash'] for r in rows}&{r['image_sha256'] for r in val},'FAIL_VALIDATION_HASH_OVERLAP')
    protect(ROOT/cfg['initialization']['path'],cfg['initialization']['sha256'])
    snapshotpath=ROOT/PROTOCOLS/'D_SEG_SMALL_SAMPLING_V2_D2_historical_snapshot.json'
    d2freeze=ROOT/PROTOCOLS/'D_SEG_SMALL_SAMPLING_V2_D2_freeze.json';protect(d2freeze)
    index=read(d2freeze)['artifacts_sha256']
    protect(snapshotpath,index[snapshotpath.relative_to(ROOT).as_posix()])
    historic=read(snapshotpath)['seed42_files']
    for name in ['arm_completion.json','final_evaluation.json','anchors.jsonl','optimizer.jsonl','best.pt','last.pt']:
        p=(ROOT/CONTROL/name).resolve();protect(p,historic[str(p)])
    completed=read(ROOT/CONTROL/'arm_completion.json');baseline=read(ROOT/CONTROL/'final_evaluation.json')
    g.require(completed['completed_epochs']==300 and completed['scheduled_optimizer_calls']==3741
              and completed['applied_optimizer_updates']+completed['skipped_optimizer_updates']==3741
              and completed['unknown_optimizer_opportunities']==0,'FAIL_CONTROL_BUDGET')
    g.require(baseline['diagnostics']['very_small']=={'TP':27,'support':49,'recall':27/49},'FAIL_CONTROL_ENDPOINT')
    # Reconstruct ALL recorded consumed anchors, not just the declared sampler recipe.
    epoch_hashes=[];actual=[];current=0;total=0
    with (ROOT/CONTROL/'anchors.jsonl').open(encoding='utf8') as f:
        for line in f:
            batch=json.loads(line);epoch=batch['epoch']
            if epoch!=current:
                g.require(epoch==current+1 and actual==g.uniform_order(ids,current),'FAIL_ANCHOR_ORDER')
                epoch_hashes.append(g.digest(actual));actual=[];current=epoch
            for item in batch['anchors']:
                g.require(item['anchor_position']==len(actual),'FAIL_ANCHOR_POSITION')
                actual.append(item['sample_id']);total+=1
    g.require(current==299 and actual==g.uniform_order(ids,299) and total==231300,'FAIL_ANCHOR_ORDER')
    epoch_hashes.append(g.digest(actual))
    # Source inspection only. No importing Ultralytics or constructing its datasets.
    package=Path(metadata.distribution('ultralytics').locate_file('ultralytics'))
    source_info={}
    for rel in ['data/base.py','data/augment.py','data/dataset.py']:
        p=package/rel;protect(p,cfg['architecture']['source_hashes'].get(rel))
        source_info[rel]={'path':str(p),'sha256':sha(p)}
    from PIL import Image
    invalid=[];patches=[];manifest=[];metadata_count=0
    for row in rows:
        g.admit(row)
        for kind in ('image','label'):
            p=(ROOT/row[kind+'_path']).resolve()
            g.require(p.parent==(ROOT/'outputs/isic_fuseg_pretrain_smoke_20260914/fuseg_dataset'/('images' if kind=='image' else 'labels')/'train').resolve(),'FAIL_PATH_ESCAPE')
            protect(p,row[kind+'_hash'])
        with Image.open(ROOT/row['image_path']) as im:
            g.require(im.size==(512,512),'FAIL_IMAGE_CANVAS');metadata_count+=1
        polygons=g.parse_labels((ROOT/row['label_path']).read_text(encoding='utf8'))
        targets=g.eligible_ids(polygons)
        g.require(len(polygons)==row['GT_count'] and bool(targets)==row['contains_very_small'],'FAIL_FROZEN_GT_MISMATCH')
        g.require(len(targets)==sum(x['bbox_area_ratio']<.0025 for x in row['instance_geometry']),'FAIL_FROZEN_TARGET_COUNT')
        entry={k:row[k] for k in ['sample_id','image_path','label_path','image_hash','label_hash','source','split','role','GT_count']}
        entry.update(eligible_GT_ids=targets,representation='PATCH' if targets else 'FULL_IMAGE',sampling_weight=1.,anchor_visits_per_epoch=1)
        manifest.append(entry)
        if not targets:
            g.require(g.transform(polygons,0)['polygons']==polygons,'FAIL_NONELIGIBLE_CHANGED');continue
        for i,p in enumerate(polygons):
            try:g.validated_polygon(p)
            except g.PatchError as exc:
                invalid.append({'sample_id':row['sample_id'],'GT_instance_id':i,'selected_target_eligible':i in targets,
                                'reason':str(exc),'label_sha256':row['label_hash']})
        for cycle,target in enumerate(targets):
            item={'sample_id':row['sample_id'],'cycle_index':cycle,'cycle_length':len(targets),'target_GT_id':target,
                  'original_GT_count':len(polygons),'source_image_sha256':row['image_hash'],'source_label_sha256':row['label_hash']}
            # Bbox coverage is independently testable even when polygon topology is invalid.
            try:item['bbox_containment_verified']=True;item['desired_patch_bounds']=g.patch_bounds(polygons[target])
            except g.PatchError as exc:item['bbox_containment_verified']=False;item['bbox_error']=str(exc)
            try:
                result=g.transform(polygons,cycle)
                g.require(result==g.transform(polygons,cycle) and result==g.transform(polygons,cycle+len(targets)), 'FAIL_NONDETERMINISTIC')
                item.update(result,status='PASS')
            except g.PatchError as exc:item.update(status='BLOCKED',reason=str(exc))
            patches.append(item)
    g.require(sum(bool(r['eligible_GT_ids']) for r in manifest)==161 and len(patches)==182,'FAIL_ELIGIBILITY_COUNTS')
    valid=[p for p in patches if p['status']=='PASS'];blocked=[p for p in patches if p['status']=='BLOCKED']
    other=Counter();full=Counter();multi=[p for p in valid if p['original_GT_count']>1]
    for p in valid:
        for r in p['records']:
            full[r['state']]+=1
            if r['GT_instance_id']!=p['target_GT_id']:other[r['state']]+=1
    # Fractions here use patch-cycles, not unique-image denominators.
    patterns=Counter(('some_clipped' if p['partially_clipped_GT_count'] else 'none_clipped')+' / '+('some_dropped' if p['dropped_GT_count'] else 'none_dropped') for p in multi)
    audit={'PHASE_E1_STATUS':'BLOCKED' if blocked else 'PENDING_FINAL_VERIFICATION',
           'eligible_images':161,'very_small_targets':182,'training_images':771,
           'negative_images':sum(r['GT_count']==0 for r in manifest),
           'noneligible_positive_images':sum(r['GT_count']>0 and not r['eligible_GT_ids'] for r in manifest),
           'eligible_multi_GT_images':sum(bool(r['eligible_GT_ids']) and r['GT_count']>1 for r in manifest),
           'original_invalid_polygons_in_eligible_images':invalid,
           'invalid_polygon_count':len(invalid),'invalid_polygon_image_count':len({x['sample_id'] for x in invalid}),
           'bbox_contained_targets':sum(p['bbox_containment_verified'] for p in patches),
           'patch_cycles_attempted':len(patches),'patch_cycles_passed':len(valid),'patch_cycles_blocked':len(blocked),
           'selected_polygon_retention_verified':len(valid),'selected_polygon_retention_unverified':len(blocked),
           'invalid_transformed_polygons_on_successful_cycles':0,'empty_patches_on_successful_cycles':0,
           'all_182_validity_or_empty_checks_passed':not blocked,
           'other_GT_counts_successful_cycles_only':dict(other),'all_GT_counts_successful_cycles_only':dict(full),
           'successful_single_GT_source_cycles':sum(p['original_GT_count']==1 for p in valid),
           'successful_multi_GT_source_cycles':len(multi),'successful_patches_containing_ge2_GT':sum(p['retained_GT_count']>=2 for p in valid),
           'multi_GT_context_patterns_successful_cycles_only':dict(patterns),
           'multi_GT_image_outcomes':{r['sample_id']:{'total_cycles':len(r['eligible_GT_ids']),
               'blocked_cycles':sum(p['sample_id']==r['sample_id'] for p in blocked),
               'successful_cycles_with_clipping':sum(p['sample_id']==r['sample_id'] and p['partially_clipped_GT_count']>0 for p in valid),
               'successful_cycles_with_drops':sum(p['sample_id']==r['sample_id'] and p['dropped_GT_count']>0 for p in valid)}
               for r in manifest if r['eligible_GT_ids'] and r['GT_count']>1},
           'patches':patches,'TRAINING_PIXELS_USED_FOR_PATCH_AUDIT':False,'training_image_headers_read':metadata_count,
           'validation_images_used':0,'test_images_used':0,'CO2Wounds_used':False,'model_loaded':False,
           'training_performed':False,'model_inference':False,'synthetic_visualizations':[],
           'EXACT_ANCHOR_ORDER_PARITY':'VERIFIED','parity_scope':'All 300 historical consumed control epoch orders reconstructed; future experimental consumption NOT YET OBSERVED',
           'historical_epoch_order_sha256':epoch_hashes,'negative_and_noneligible_full_image_identity_verified':True,
           'mosaic_runtime_integration':'NOT_IMPLEMENTED_OR_VERIFIED; fail-closed geometry blocker prevents E2 readiness',
           'raster_fractional_window_contract':'UNRESOLVED: exact continuous bounds audited; integer raster/interpolation must not be silently chosen',
           'source_inspection':source_info,'software':{'numpy':metadata.version('numpy'),'shapely':metadata.version('shapely')},
           'blockers':sorted({p['reason'].split(':')[0] for p in blocked})}
    out=ROOT/AUDIT;out.mkdir()
    audit['synthetic_visualizations']=synthetic_figures(out)
    transform={'status':'GEOMETRY_ONLY_NOT_RUNTIME_ADAPTER','patch_size':[256,256],'source_canvas':[512,512],
       'eligible':'TRAINING_GT_BBOX_AREA_RATIO < 0.0025','epoch_index':'zero-based',
       'target':'original annotation order; eligible_GT_list[epoch_index % len(eligible_GT_list)]',
       'bounds':'left=clamp(cx-128,0,256); top=clamp(cy-128,0,256); right=left+256; bottom=top+256',
       'coordinates':'continuous original source pixels; subtract left/top, divide by256; no rounding in geometry simulation',
       'selected_target':'whole original polygon required; boundary covers exact, no area tolerance acceptance',
       'partial_GT':'keep every nonzero-area valid single polygon; zero-area line/point touch is not lesion area',
       'multiparts_and_holes':'fail closed, no splitting into new GT identities, no silent part discard',
       'source_invalid_policy':'FAIL_PATCH_LABEL_INVALID; no buffer(0), make_valid, simplification, relabel or dropping',
       'ordering':'pre-augmentation, pre-label-resampling, using original source coordinates; not crop an already resized 768 image',
       'mosaic_contract':'Both anchor and companion get_image_and_label routes require exactly one representation transform. Keep companion algorithm/probability/buffer identity semantics. No adapter attached in E1.',
       'epoch_contract':'Future workers=2 and prefetch must carry immutable zero-based epoch token; mutable global epoch is insufficient. Pending adapter tests.',
       'cache':'on-the-fly intended; do not mutate shared source image/labels, no validation patches',
       'required_telemetry':['sample_id','epoch','target_GT_id','patch_bounds','input_image_SHA256','source_label_SHA256','transformed_label_hash','anchor_or_companion'],
       'representation_linear_multiplier':2,'representation_bbox_area_multiplier':4,
       'performance_claim':False,'raster_implementation':'NOT_IMPLEMENTED; fractional raster alignment requires explicit contract in subsequent revision'}
    candidate=baseline['candidate'];counts={'very_small':27,'small':candidate['size_recall']['small']['matched'],
        'medium':candidate['size_recall']['medium']['matched'],'large':candidate['size_recall']['large']['matched'],
        'crop_complete':candidate['crop_complete95_images'],'tp':candidate['tp'],'fp':candidate['fp'],'fn':candidate['fn']}
    protocol={'experiment_id':PREFIX,'PHASE_E1_STATUS':audit['PHASE_E1_STATUS'],'READY_FOR_PHASE_E2_PATCH_SEED42_TRAINING':'NO',
      'question':'FULL_IMAGE vs GT_CENTERED_PATCH_REPRESENTATION under identical uniform anchor sampling and frozen training/evaluation recipe',
      'primary_comparator':'D2 Fresh Uniform Control seed42 (immutable D1 seed42 C reused in D2)',
      'secondary_comparator':'Historical corrected Phase C','context_only':'D2 Sampling S; never control',
      'control_config_path':cfgpath.relative_to(ROOT).as_posix(),'control_config_sha256':sha(cfgpath),
      'control_checkpoint_sha256':completed['best_checkpoint_sha256'],'control_budget':completed,
      'baseline_counts':counts,'baseline_very_small_27_of_49':True,
      'frozen_training_recipe':deepcopy(cfg),
      'future_routing_only_override':{'experiment_id':PREFIX+'_SEED42','output_path':FUTURE.as_posix(),'arm_label':'PATCH'},
      'frozen_recipe_note':'Old control runner paths identify historical semantics; NOT an executable E2 config. Future adapter must be separately integrated and verified.',
      'single_intervention':'PATCH_REPRESENTATION_FOR_VERY_SMALL_ELIGIBLE_TRAINING_IMAGES',
      'sampling':{'uniform':True,'replacement':False,'anchors_per_epoch':771,'seed':42,'weights':'all 1','order':'PCG64 SeedSequence([42,epoch]); same canonical sample order as frozen control'},
      'primary_endpoint':{'name':'Very-small Recall <0.25%','support':49,'required_TP':29,'minimum_delta_TP':2,'minimum_delta_pp':200/49,'statistical_significance_claim':False},
      'key_secondary_endpoint':{'name':'<0.10% recall','support':27,'control_TP':11},
      'advancement_gate':{'all_required':{'very_small_TP_min':29,'small_TP_min':103,'medium_TP_min':84,'large_TP_min':14,
            'crop_complete_min':164,'precision_min_exact':'203/233 - 1/100','f1_min_exact':'406/474 - 1/100'},
            'arithmetic':'Exact rational arithmetic on counts; no rounded gate decisions','status_if_all_pass':'PASS_PATCH_RESEARCH_ADVANCEMENT_GATE'},
      'secondary_endpoints':['Small Recall <1%','Precision','Recall','F1','Medium Recall','Large Recall','Crop completeness','TP','FP','FN','No ROI','single-GT recall','multi-GT recall'],
      'future_persistent_case_diagnostic':'E0 IDs only post-hoc descriptive evaluation, NEVER training selection or mining',
      'fixed_budget':'300 completed epochs;3741 scheduled;applied+skipped=3741;unknown=0;early stop/crash NOT_VALID;no resume/top-up/retry',
      'AMP_rule':'same scheduled budget, report realized skips (control 10 / applied3731); imbalance flag, not identical-realized-updates claim',
      'hypothesis_status':'TESTABLE_NEXT_HYPOTHESIS_NOT_PROVEN_SOLUTION',
      'risks':['global context loss / increased FP','multi-GT contextual lesions clipped or dropped','medium/large regression','train/eval scale mismatch'],
      'prohibited':['training','inference','model_loading','weighted_sampling','sampler_V3','resolution_change','loss_change','architecture_change','threshold_NMS_IoU_tuning','validation_patches','test_time_patches','crop_policy_change','multi_ROI','locked_test','CO2Wounds','FUSeg_official_test','external_test','App_replacement'],
      'NEW_TRAINING_AUTHORIZED':'NO','APP_MODEL_REPLACEMENT_AUTHORIZED':'NO','EXTERNAL_TEST_AUTHORIZED':'NO','STOP_AFTER_E1':True}
    for suffix,value in [('protocol',protocol),('training_transform',transform),('training_manifest',{'schema':'E1-training-only-v1','source_manifest_sha256':sha(mp),'samples':manifest}),('geometry_audit',audit),('evaluation_protocol',ep)]:write(output(suffix),value)
    protect(REQUEST)
    write(out/'source_snapshot_before.json',protected)
    g.require(all(sha(p)==h for p,h in protected.items()),'FAIL_SOURCE_CHANGED')
    write(out/'integrity.json',{'protected_files_unchanged':len(protected),'model_loaded':False,'training':False,'inference':False,
       'validation_pixels_used':0,'test_images_used':0,'CO2Wounds_used':False,'future_output_absent':True})
    require_future_absent(ROOT)
    print(json.dumps({k:audit[k] for k in ['PHASE_E1_STATUS','eligible_images','very_small_targets','invalid_polygon_count','invalid_polygon_image_count','patch_cycles_passed','patch_cycles_blocked','other_GT_counts_successful_cycles_only','eligible_multi_GT_images']},indent=2))


if __name__ == '__main__':
    build()
