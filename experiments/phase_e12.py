"""Training-only E1.2 audit. No network imports, inference, or training."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import explain_validity
from experiments import mask_first as m
from experiments.patch_raster import RasterContract, parse_labels, file_sha, array_sha, topology, compare
from experiments.patch_geometry import require, admit, uniform_order, digest, model_guard

ROOT=Path(__file__).resolve().parents[1]
PRO=ROOT/'experiments/protocols'
OUT=ROOT/'experiments/results/e_patch_v3_feasibility_audit'
PACKAGE=Path('C:/Python312/Lib/site-packages/ultralytics')


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8',newline='\n') as f:
        json.dump(data,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def audit_scene(polygons,raster):
    masks=raster.rasterize(polygons);repeat=raster.rasterize(polygons);rows=[]
    for j,(p,mask,again) in enumerate(zip(polygons,masks,repeat)):
        q=Polygon(p*512);bbox=m.tight_bbox(mask)
        require(bbox is not None,'FAIL_EMPTY_SOURCE_MASK')
        ratio=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])/512**2
        rows.append({'source_GT_instance_id':j,'mask_area':int(mask.sum()),**topology(mask),
            'bbox_from_mask':bbox,'mask_bbox_area_ratio':ratio,'mask_eligible':ratio<.0025,
            'polygon_bbox_float32':list(map(float,q.bounds)),
            'source_polygon_valid':bool(q.is_valid),'source_polygon_topology_status':explain_validity(q),
            'source_mask_sha256':array_sha(mask),'deterministic':bool(np.array_equal(mask,again))})
    require(len(rows)==len(polygons) and all(r['deterministic'] for r in rows),'FAIL_SOURCE_IDENTITY')
    return rows,np.asarray(masks,np.uint8).reshape(-1,512,512)


def protected_snapshot():
    snapshot=read(OUT.parent/'e_patch_v2_preregistration_audit/source_snapshot.json')
    v2=PRO/'E_PATCH_V2_freeze.json'
    for p,h in read(v2)['artifacts_sha256'].items():snapshot[str(ROOT/p)]=h
    snapshot[str(v2)]=file_sha(v2)
    for p,h in snapshot.items():require(file_sha(p)==h,'FAIL_PROTECTED_SOURCE_CHANGED: '+p)
    return snapshot


def characterize(raster):
    # Simple convex quadrilateral; finite fixed diagnostic transforms, not RNG tuning.
    p=np.array([[.111,.203],[.323,.181],[.351,.427],[.129,.401]],np.float32)
    require(Polygon(p).is_valid,'FAIL_SYNTHETIC_POLYGON')
    source=raster.rasterize([p])[0];points=raster.resample([p.copy()],n=1000)[0]*512
    image=np.repeat((source*255)[:,:,None],3,axis=2);ms=source[None]
    results=[]
    operations=[('identity',np.eye(3),512),('resize',np.diag([1.5,1.5,1]),768),
                ('horizontal_flip',np.array([[-1,0,512],[0,1,0],[0,0,1]],float),512),
                ('fixed_affine',np.array([[.97,-.07,17],[.07,.97,-6],[0,0,1]],float),512)]
    for name,M,size in operations:
        homogeneous=np.c_[points,np.ones(len(points))]@M.T
        vertices=(homogeneous[:,:2]/homogeneous[:,2:]).astype(np.float32)
        old=raster.polygon2mask((size,size),[vertices.reshape(-1)],1,1)
        if name=='identity':new=source.copy()
        elif name=='resize':new=m.resize_scene(image,ms,['x'],size)['masks'][0]
        elif name=='horizontal_flip':new=m.flip_scene(image,ms,['x'],'horizontal')['masks'][0]
        else:new=m.warp_scene(image,ms,['x'],M,size)['masks'][0]
        low_old=raster.polygon2mask((size,size),[vertices.reshape(-1)],1,4)
        low_new=cv2.resize(new,(size//4,size//4),interpolation=cv2.INTER_NEAREST)
        results.append({'operation':name,'matrix':M.tolist(),'full_resolution':compare(old,new),'ratio4':compare(low_old,low_new)})
    return {'scope':'synthetic legal-simple polygon; pinned raster primitive after fixed point transform versus raster-native; not full historical runtime replay',
            'historical_flip_coordinate_convention':'continuous x -> width-x; raster flip -> width-1-x',
            'purpose':'descriptive pixel differences only; never optimize authority to match historical path','rows':results}


def run():
    model_guard();cv2.setNumThreads(1)
    require(not OUT.exists(),'FAIL_AUDIT_OUTPUT_EXISTS')
    for arm in ('control','experimental'):
        require(not (OUT.parent/f'e_patch_v3_seed42_{arm}').exists(),'FAIL_FUTURE_OUTPUT_EXISTS')
    protected=protected_snapshot();save(OUT/'source_snapshot.json',protected)
    original=read(PRO/'E_PATCH_V1_training_manifest.json')['samples']
    require(len(original)==771 and len({r['sample_id'] for r in original})==771,'FAIL_TRAIN_SIZE')
    pins=read(PRO/'E_PATCH_V2_topology_contract.json')['sources']
    raster=RasterContract(PACKAGE,{k:v['sha256'] for k,v in pins.items()})
    samples=[];cycles=[];removed=[];added=[];all_instances=[]
    for row in original:
        admit(row)
        for kind in ('image','label'):require(file_sha(ROOT/row[kind+'_path'])==row[kind+'_hash'],'FAIL_SOURCE_HASH')
        polygons=parse_labels((ROOT/row['label_path']).read_text(encoding='utf8'))[1]
        instances,masks=audit_scene(polygons,raster);require(len(instances)==row['GT_count'],'FAIL_GT_COUNT')
        old=set(row['eligible_GT_ids']);new={r['source_GT_instance_id'] for r in instances if r['mask_eligible']}
        for ids,out in ((old-new,removed),(new-old,added)):
            out.extend({'sample_id':row['sample_id'],'GT_instance_id':j} for j in sorted(ids))
        for item in instances:
            item.update(source_sample_id=row['sample_id'],source_label_sha256=row['label_hash'],
                        original_polygon_eligible=item['source_GT_instance_id'] in old)
        for target in sorted(old|new):
            sliced=m.patch_masks(masks,[f"{row['sample_id']}:{j}" for j in range(len(masks))],target)
            other=[j for j in range(len(masks)) if j!=target]
            counts=Counter(sliced['states'][j] for j in other)
            before=sum(sliced['source_pixels'][j] for j in other);after=sum(sliced['retained_pixels'][j] for j in other)
            cycles.append({'sample_id':row['sample_id'],'GT_instance_id':target,'original_E1_target':target in old,
                'V3_target':target in new,'bounds':sliced['bounds'],'selected_fully_retained':sliced['selected_retained'],
                'selected_area':sliced['source_pixels'][target],'other_GT_states':dict(counts),
                'all_instance_states':sliced['states'],'source_instance_pixels':sliced['source_pixels'],
                'retained_instance_pixels':sliced['retained_pixels'],'other_pixels_source':before,
                'other_pixels_retained':after,'other_pixels_retention_ratio':after/before if before else None})
        samples.append(dict(row,representation='AUTHORITATIVE_PER_INSTANCE_SOURCE_RASTER',
                            original_eligible_GT_ids=sorted(old),eligible_GT_ids=sorted(new),
                            mask_eligible_GT_ids=sorted(new),mask_hashes=[r['source_mask_sha256'] for r in instances],instances=instances))
        all_instances.extend(instances)
    def summary(rows):
        counts=Counter()
        for r in rows:counts.update(r['other_GT_states'])
        a=sum(r['other_pixels_source'] for r in rows);b=sum(r['other_pixels_retained'] for r in rows)
        return {'target_cycles':len(rows),'selected_retained':sum(r['selected_fully_retained'] for r in rows),
            'empty_selected_patches':sum(r['selected_area']==0 for r in rows),'other_GT_states':dict(counts),
            'selected_retained_other_dropped_cycles':sum(r['other_GT_states'].get('DROPPED',0)>0 for r in rows),
            'other_pixels_source':a,'other_pixels_retained':b,'aggregate_other_pixel_retention_ratio':b/a if a else None}
    old_summary=summary([r for r in cycles if r['original_E1_target']]);new_summary=summary([r for r in cycles if r['V3_target']])
    require(old_summary['target_cycles']==182 and old_summary['selected_retained']==182,'FAIL_182_RETENTION')
    stats={'training_images':len(samples),'GT_instances':len(all_instances),'deterministic_masks':sum(r['deterministic'] for r in all_instances),
        'invalid_source_polygons':sum(not r['source_polygon_valid'] for r in all_instances),
        'multicomponent_instances':sum(r['connected_components']>1 for r in all_instances),
        'instances_with_holes':sum(r['hole_count']>0 for r in all_instances),
        'original_eligible_images':sum(bool(r['original_eligible_GT_ids']) for r in samples),
        'mask_eligible_images':sum(bool(r['eligible_GT_ids']) for r in samples),
        'original_eligible_GT':old_summary['target_cycles'],'mask_eligible_GT':new_summary['target_cycles']}
    contract={'schema':'E_PATCH_V3_MASK_FIRST','source_authority':'E1.1 float32 normalized polygons -> pinned per-image resample -> *512 -> polygon2mask ratio1 int32 fillPoly',
        'source_files':raster.sources,'instance_rule':'one original annotation line = one ID and one mask, including all components and holes',
        'after_source_rasterization':'no mask-to-polygon; no polygon repair, clipping, hull, simplification or contour reconstruction',
        'bbox_convention':'positive pixel tight bbox [xmin,ymin,xmax+1,ymax+1]; normalized xywh from current transformed mask',
        'ELIGIBILITY_IDENTITY':'CHANGED' if removed or added else 'UNCHANGED',
        'V3_eligibility':'mask bbox pixel width*height /512^2 <0.0025 (strict); new contract, NOT same E1 eligible set',
        'removed':removed,'added':added,'patch_size':[256,256],
        'crop':'floor(mask bbox center -128), clamp[0,256]; direct integer image and instance-mask slice',
        'target_selection':'sorted eligible original GT IDs; epoch_token % count; zero-based epoch; generation not a randomness input',
        'storage':'deterministic on-demand raster masks with per-instance hashes; no original label rewrite or lossy serialized contour',
        'label_scope':'TRAINING ONLY; no changes to original development validation labels','stats':stats}
    save(PRO/'E_PATCH_V3_mask_authority_contract.json',contract)
    save(PRO/'E_PATCH_V3_training_mask_manifest.json',{'schema':'E_PATCH_V3_MASK_FIRST','samples':samples,'stats':stats})
    save(PRO/'E_PATCH_V3_patch_audit.json',{'original182':old_summary,'V3_membership':new_summary,
        'unresolved_source_topology_blockers':0,'vector_raster_roundtrips':0,'raster_mismatch_blockers':0,
        'identity_warning':'direct slices avoid re-rasterization; this does not certify clinical annotation correctness',
        'cycles':cycles})
    save(OUT/'control_characterization.json',characterize(raster))
    # Deterministic training-only subset: first invalid, first eligible, first multi-GT,
    # then first remaining ID. No outcome or validation access in selection.
    subset=[]
    for predicate in (lambda r:any(not i['source_polygon_valid'] for i in r['instances']),
                      lambda r:bool(r['eligible_GT_ids']),lambda r:r['GT_count']>1,lambda r:True):
        subset.append(next(r for r in samples if r not in subset and predicate(r)))
    save(OUT/'smoke_training_subset.json',{'selection':'first sorted IDs matching invalid, eligible, multi-GT, remaining; training labels only','samples':subset})
    ids=[r['sample_id'] for r in samples];orders=[uniform_order(ids,e) for e in range(300)]
    require(all(len(o)==771 and set(o)==set(ids) for o in orders),'FAIL_ORDER')
    save(OUT/'anchor_orders.json',{'seed':42,'orders':orders,'epoch_sha256':[digest(o) for o in orders],
                                 'replacement':False,'weights':'all1','shared_arms':['C3','P3']})
    print(json.dumps({'source_audit':stats,'original182':old_summary,'next':'run CPU DataLoader subset, then finalization'},ensure_ascii=False))


if __name__=='__main__':run()
