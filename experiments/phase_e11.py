"""E1.1 model-free topology/raster audit and paired preregistration. Stop here."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import importlib.metadata as metadata
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from experiments.phase_d0 import sha, read, write
from experiments.patch_geometry import uniform_order
from experiments import patch_raster as r
from experiments.patch_dataset_adapter import runtime_probe

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'experiments/protocols'
PREFIX='E_PATCH_V2'
OUT=ROOT/'experiments/results/e_patch_v2_preregistration_audit'
REPORT=ROOT/'docs/PHASE_E11_TOPOLOGY_RASTER_PREREGISTRATION_20260925.md'
FUTURE=['experiments/results/e_patch_v2_seed42_control','experiments/results/e_patch_v2_seed42_experimental']
REQUEST=Path('C:/Users/milo9/.codex/attachments/28917535-6338-4c48-b697-72a292d76a54/貼上的文字.txt')


def artifact(suffix):return P/f'{PREFIX}_{suffix}.json'


def future_absent():
    r.require(not any((ROOT/p).exists() for p in FUTURE),'FAIL_FUTURE_OUTPUT_ALREADY_EXISTS')


def build():
    r.model_guard();future_absent()
    r.require(not list(P.glob(PREFIX+'*')) and not OUT.exists(),'FAIL_V2_ALREADY_EXISTS')
    protected={}
    def protect(path,h=None):
        path=Path(path).resolve();actual=sha(path)
        r.require(h is None or h==actual,'FAIL_FROZEN_INPUT_CHANGED: '+str(path));protected[str(path)]=actual
    v1freeze=P/'E_PATCH_V1_freeze.json';protect(v1freeze)
    for rel,h in read(v1freeze)['artifacts_sha256'].items():protect(ROOT/rel,h)
    r.require(read(v1freeze)['PHASE_E1_STATUS']=='BLOCKED','FAIL_V1_STATUS')
    for path,h in read(ROOT/'experiments/results/e_patch_v1_preregistration_audit/source_snapshot_before.json').items():protect(path,h)
    protect(REQUEST)
    cfg=read(P/'D_SEG_SMALL_SAMPLING_V2_control_config.json')
    manifest=read(P/'E_PATCH_V1_training_manifest.json')['samples']
    original_audit=read(P/'E_PATCH_V1_geometry_audit.json')
    bad={(x['sample_id'],x['GT_instance_id']) for x in original_audit['original_invalid_polygons_in_eligible_images']}
    r.require(len(bad)==17,'FAIL_E1_INVALID_SET')
    package=Path(metadata.distribution('ultralytics').locate_file('ultralytics'))
    raster=r.RasterContract(package,cfg['architecture']['source_hashes'])
    for v in raster.sources.values():protect(v['path'],v['sha256'])
    records=[];samples=[];cycles=[];identity_invalid_noneligible=0
    for row in manifest:
        r.require(row['split']=='train' and row['source']=='FUSeg','FAIL_SOURCE_ROLE')
        lines,original=r.parse_labels((ROOT/row['label_path']).read_text(encoding='utf8'))
        source_masks=raster.rasterize(original)
        canonical=[];instances=[];targets=row['eligible_GT_ids']
        for i,polygon in enumerate(original):
            key=(row['sample_id'],i);p=Polygon(polygon*512)
            if key in bad:
                result,candidate=r.canonicalize(original,i,raster)
                result.update(sample_id=row['sample_id'],GT_instance_id=i,source_label_sha256=row['label_hash'])
                records.append(result);canonical.append(candidate)
                status=result['canonicalization_status']
            elif targets and (not p.is_valid or p.area<=0):
                status='BLOCKED_UNREGISTERED_INVALID_ELIGIBLE_INSTANCE';candidate=None;canonical.append(None)
            else:
                candidate=polygon.copy();canonical.append(candidate)
                status='IDENTITY_VALID_SOURCE' if p.is_valid and p.area>0 else 'IDENTITY_NONELIGIBLE_UNMODIFIED_SOURCE'
                identity_invalid_noneligible+=int(status=='IDENTITY_NONELIGIBLE_UNMODIFIED_SOURCE')
            instances.append({'GT_instance_id':i,'status':status,'source_line_sha256':r.digest(lines[i]),
                'source_polygon_float32_hash':r.digest(polygon.tolist()),'source_raster_sha256':r.array_sha(source_masks[i]),
                'canonical_polygon':None if candidate is None else candidate.tolist(),
                'canonical_polygon_hash':None if candidate is None else r.digest(candidate.tolist())})
        if all(p is not None for p in canonical):
            allm=raster.rasterize(canonical)
            r.require(all(np.array_equal(a,b) for a,b in zip(source_masks,allm)),'FAIL_COMBINED_CANONICAL_SCENE')
        samples.append({'sample_id':row['sample_id'],'split':'train','source':'FUSeg','role':row['role'],
            'source_image_path':row['image_path'],'source_image_sha256':row['image_hash'],
            'source_label_path':row['label_path'],'source_label_sha256':row['label_hash'],
            'eligible_GT_ids':targets,'GT_count':len(original),'instances':instances,
            'canonical_status':'READY_LABEL_VIEW' if all(p is not None for p in canonical) else 'BLOCKED_LABEL_VIEW'})
        for epoch,target in enumerate(targets):
            case={'sample_id':row['sample_id'],'cycle_index':epoch,'target_GT_id':target,'cycle_length':len(targets),
                  'original_GT_count':len(original),'multi_GT':len(original)>1}
            try:case.update(r.patch_cycle(canonical,target,raster,source_masks))
            except r.PatchError as exc:
                case.update(status='BLOCKED',failures=[str(exc)],records=None,selected_target_mask_retained=None,
                            empty_patch=None)
            cycles.append(case)
    r.require(len(records)==17 and len(cycles)==182,'FAIL_COVERAGE')
    r.require({(x['sample_id'],x['GT_instance_id']) for x in records}==bad,'FAIL_INVALID_COVERAGE')
    resolved=sum(x['canonicalization_status'] in ('IDENTITY','EXACT_RASTER_EQUIVALENT') for x in records)
    evaluated=[x for x in cycles if x['records'] is not None];passed=[x for x in cycles if x['status']=='PASS']
    aggregate=Counter();other=Counter();multi_patterns=Counter()
    for c in evaluated:
        for item in c['records']:
            aggregate[item['state']]+=1
            if item['GT_instance_id']!=c['target_GT_id']:other[item['state']]+=1
        if c['multi_GT']:multi_patterns[f"partial={c['PARTIAL']>0},dropped={c['DROPPED']>0}"]+=1
    audit={'PHASE_E11_STATUS':'BLOCKED' if resolved!=17 or len(passed)!=182 else 'PENDING_TESTS',
       'cycles_attempted':182,'cycles_passed':len(passed),'cycles_blocked':182-len(passed),
       'cycles_with_raster_evaluation':len(evaluated),'cycles_without_raster_evaluation':182-len(evaluated),
       'selected_target_retention_verified':sum(x['selected_target_mask_retained'] is True for x in cycles),
       'selected_target_retention_unknown':sum(x['selected_target_mask_retained'] is None for x in cycles),
       'raster_mismatch_cycles':sum(any(t['xor_pixels'] for t in x['records']) for x in evaluated),
       'raster_mismatch_instance_observations':sum(t['xor_pixels']>0 for x in evaluated for t in x['records']),
       'invalid_transformed_label_cycles':sum('FAIL_PATCH_LABEL_INVALID' in x['failures'] for x in cycles),
       'empty_evaluated_patches':sum(x['empty_patch'] is True for x in evaluated),
       'unresolved_instance_cycles':sum('BLOCKED_UNRESOLVED_SOURCE_INSTANCE' in x['failures'] for x in cycles),
       'all_GT_counts_evaluated_cycles_only':dict(aggregate),'other_GT_counts_evaluated_cycles_only':dict(other),
       'eligible_multi_GT_images':54,'multi_GT_patterns_evaluated_cycles_only':dict(multi_patterns),
       'selected_retained_other_dropped_cycles':sum(x.get('selected_retained_other_dropped',False) for x in evaluated),
       'counting_unit':'unique target-cycle / original GT-instance-in-cycle; not independent images and not epoch exposure',
       'blocker_counts':dict(Counter(f for x in cycles for f in x['failures'])),'cycles':cycles,
       'training_image_pixels_read':0,'validation_images_used':0,'test_images_used':0,'CO2Wounds_used':False}
    ca={'required_source_invalid_count':17,'resolved_exact_count':resolved,'unresolved_count':17-resolved,
        'records':records,'candidate_scope':'Fixed ordered family: raster contour, make_valid candidate, buffer(0) candidate; not exhaustive impossibility proof',
        'exploratory_training_only_checks':'Before audit freeze, contour subpixel buffer .125 also failed exact raster equality on all17; not promoted as a registered canonical view',
        'identity_noneligible_invalid_source_count':identity_invalid_noneligible,
        'noneligible_policy':'Original parsed coordinates unchanged in both arms; no clipping or unilateral normalization. Only E1-listed17 invalid eligible instances are canonicalization targets.',
        'authoritative_connectivity':'8-connected foreground / RETR_TREE odd-depth holes; also report4-connectivity, never interpret8-connected as proof of one valid vector ring'}
    OUT.mkdir();labeldir=P/(PREFIX+'_canonical_labels');labeldir.mkdir()
    write(labeldir/'view.json',{'status':'BLOCKED_NOT_TRAINING_READY' if resolved!=17 else 'CANDIDATE_VIEW','samples':samples})
    write(artifact('canonical_label_manifest'),{'source_manifest':{'path':'experiments/protocols/E_PATCH_V1_training_manifest.json','sha256':sha(P/'E_PATCH_V1_training_manifest.json')},
        'view_path':(labeldir/'view.json').relative_to(ROOT).as_posix(),'view_sha256':sha(labeldir/'view.json'),
        'sample_count':771,'eligible_images':161,'targets':182,'same_view_required_for_C2_P2':True,
        'unresolved_instances':17-resolved,'source_files_rewritten':False})
    topology_contract={'SOURCE_POLYGON_RASTER_CONTRACT':'FLOAT32_LABEL_PARSE -> frozen SEGMENT_RESAMPLING -> SOURCE_CANVAS512 -> frozen polygon2mask int32/fillPoly/downsample1',
        'authoritative_stage':'pre-augmentation, per-instance source binary mask; not downstream randomized768/mask_ratio4 composite training masks',
        'source_pipeline_audit':{'label_parse':'verify_image_label float32; class0; original instance identity',
          'resample':'update_labels_info uses1000 unless max vertex count>1000 then max+1; resample_segments retains original points and inserts interpolation points',
          'original_loader':'load_image resizes before update_labels_info; V2 source transform must precede that resize',
          'raster':'Format denormalizes current canvas, polygon2mask castsint32, cv2.fillPoly then cv2.resize',
          'training_final':'mask_ratio4; overlap_mask true sorts by area and builds index mask; unchanged downstream, not substituted by source binary mask'},
        'sources':raster.sources,'software':{k:metadata.version(k) for k in ['ultralytics','numpy','shapely','opencv-python']},
        'valid_source':'IDENTITY; do not simplify valid polygons','invalid_source':'E1-listed17 only; candidate needs valid single polygon and exact per-instance AND sibling raster equality',
        'acceptance':{'xor_pixels':0,'raster_IoU':1.0,'same_area':True,'single_instance':True,'holes':0},
        'candidate_order':['RASTER_EXTERNAL_CHAIN_NONE','SOURCE_MAKE_VALID_CANDIDATE','SOURCE_BUFFER_ZERO_CANDIDATE'],
        'no_component_discard':True,'no_hole_fill':True,'no_source_rewrite':True,'pixel_connectivity':8,'also_report_connectivity4':True,
        'mask_hash':'SHA256(str((shape,dtype.str)) + NUL + contiguous mask bytes)'}
    rc={'source_canvas':[512,512],'patch_size':[256,256],
        'bounds':'left=clamp(floor(cx-128),0,256); top=clamp(floor(cy-128),0,256); right/bottom=left/top+256',
        'center':'bbox of selected canonical polygon in source coordinates; eligibility/order stays frozen original training GT IDs',
        'pixel_crop':'image[top:top+256,left:left+256].copy() BEFORE resize',
        'target_containment':'vector bbox containment AND full authoritative source-mask pixel count retained',
        'clipping':'canonical vector intersection with integer rectangle, patch-local float32 normalized by256',
        'acceptance':'every instance source-mask slice exactly equals transformed vector mask; no approximate IoU',
        'resampling':'same pinned segmentation resampling stage at local256 source canvas; image-context vertex count included',
        'image_resize_after_crop':'existing cv2.INTER_LINEAR to768; subsequent augmentation unchanged',
        'mismatch_policy':'BLOCKED, no alternate floor/round/ceil, contour reconstruction or tolerance added on failure'}
    runtime=runtime_probe(package)
    adapter_contract={'adapter_path':'experiments/patch_dataset_adapter.py','adapter_sha256':sha(ROOT/'experiments/patch_dataset_adapter.py'),
       'arms':['C2','P2'],'shared_label_view':sha(labeldir/'view.json'),
       'scope':runtime['runtime_scope'],'source_stage':'exactly once before resize/standard augmentation',
       'C2':'identity full image through same adapter','P2':'eligible anchor AND eligible companion patched once',
       'token':'frozen FetchToken(sample_id,epoch_token,generation); per-anchor ScopedCompanions binds immutable parent epoch',
       'prefetch_reset':'generation token discards stale prefetched results; generation does not affect patch content',
       'runtime_test':runtime,'production_training_driver_bound':False,
       'remaining_integration_requirement':'Separate authorized E2 must bind this source-stage adapter to pinned trainer without changing original companion selector/buffer/RNG or augmentation. No executable training route in E1.1.'}
    ids=[s['sample_id'] for s in samples];plans=[uniform_order(ids,e) for e in range(300)]
    r.require(all(len(p)==len(set(p))==771 for p in plans),'FAIL_UNIFORM_ORDER')
    write(OUT/'anchor_orders.json',{'seed':42,'epochs':plans,'epoch_sha256':[r.digest(p) for p in plans]})
    base=deepcopy(cfg)
    for key in ['experiment_id','arm_label','output_path','runner']:base.pop(key,None)
    base['manifest']={'path':artifact('canonical_label_manifest').relative_to(ROOT).as_posix(),'sha256':sha(artifact('canonical_label_manifest'))}
    base['sampling']={'mode':'uniform_shuffled','replacement':False,'seed':42,'epoch_anchor_draws':771,
        'weights':'all1','order_path':(OUT/'anchor_orders.json').relative_to(ROOT).as_posix(),'order_sha256':sha(OUT/'anchor_orders.json')}
    base['runtime_adapter_contract']={'path':artifact('runtime_adapter_contract').relative_to(ROOT).as_posix()}
    base['primary_comparison']='FRESH_CANONICAL_C2_vs_FRESH_CANONICAL_P2'
    base['historical_D2_C_role']='HISTORICAL_REFERENCE_ONLY'
    base['advancement_gate']={'primary':'P2 very_small_TP >= C2 very_small_TP+2; support49','small':'P2>=C2-1','medium':'P2>=C2-1','large':'P2>=C2',
        'precision':'P2>=C2-1/100','f1':'P2>=C2-1/100','crop_complete':'P2>=C2-1','arithmetic':'exact rational counts; operational criterion not statistical significance',
        'historical27_of49_not_gate_baseline':True,'all_required':True}
    base['fixed_budget_contract']={'epochs':300,'scheduled':3741,'applied_plus_skipped':3741,'unknown':0,
        'patience':80,'early_stop_or_crash':'NOT_VALID; no resume/top-up/retry','AMP_imbalance':'report; no identical applied-count claim unless measured equal'}
    base['execution_authorized']=False;base['MULTI_SEED_PATCH_AUTHORIZED']='NO'
    base['status']='BLOCKED_PREREGISTRATION_NOT_EXECUTABLE'
    for arm,out,representation in [('C2',FUTURE[0],'IDENTITY_FULL_IMAGE'),('P2',FUTURE[1],'ELIGIBLE_GT_CENTERED_PATCH')]:
        c=deepcopy(base);c.update(experiment_id=PREFIX+'_'+arm,arm_label=arm,output_path=out,representation=representation)
        write(artifact('control_config' if arm=='C2' else 'experimental_config'),c)
    evaluation=read(P/'E_PATCH_V1_evaluation_protocol.json');write(artifact('evaluation_protocol'),evaluation)
    for suffix,value in [('topology_contract',topology_contract),('canonicalization_audit',ca),('raster_contract',rc),
                         ('runtime_adapter_contract',adapter_contract),('patch_geometry_audit',audit)]:write(artifact(suffix),value)
    write(OUT/'source_snapshot.json',protected)
    r.require(all(sha(p)==h for p,h in protected.items()),'FAIL_SOURCE_CHANGED')
    future_absent()
    write(OUT/'integrity.json',{'PHASE_E11_STATUS':audit['PHASE_E11_STATUS'],'protected_files_unchanged':len(protected),
       'MODEL_LOADING':False,'TRAINING':False,'INFERENCE':False,'training_image_pixels_read':0,
       'training_label_rasters_generated':True,'synthetic_image_pixels_used':True,'validation_images_used':0,'test_images_used':0,
       'CO2Wounds_used':False,'future_outputs_absent':True})
    print(json.dumps({'resolved':resolved,'unresolved':17-resolved,**{k:v for k,v in audit.items() if k not in ['cycles']},'runtime_status':runtime['status']},indent=2))


if __name__=='__main__':build()
