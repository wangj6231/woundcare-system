"""E_PATCH_V2 model-free raster authority and candidate verification.

Read the pinned functions as AST; never import the Ultralytics package. Candidate
operators are NOT authority: a candidate is usable only after exact raster tests.
"""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from shapely import make_valid
from shapely.geometry import Polygon, box
from shapely.validation import explain_validity

from experiments.patch_geometry import PatchError, require, digest, NoModels, model_guard


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def array_sha(array):
    a=np.ascontiguousarray(array)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+b'\0'+a.tobytes()).hexdigest()


def extract_function(path, name, namespace):
    tree=ast.parse(Path(path).read_text(encoding='utf8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


class RasterContract:
    def __init__(self, package, pins=None):
        self.package=Path(package);self.sources={}
        for rel in ('data/utils.py','utils/ops.py','data/dataset.py','data/augment.py','data/base.py','utils/instance.py'):
            path=self.package/rel;h=file_sha(path)
            if pins and rel in pins:require(h==pins[rel],'FAIL_BASELINE_SOURCE_CHANGED: '+rel)
            self.sources[rel]={'path':str(path),'sha256':h}
        ns={'np':np,'cv2':cv2}
        self.polygon2mask=extract_function(self.package/'data/utils.py','polygon2mask',ns)
        self.resample=extract_function(self.package/'utils/ops.py','resample_segments',ns)

    @staticmethod
    def resample_count(polygons):
        longest=max(map(len,polygons),default=0)
        return longest+1 if longest>1000 else 1000

    def rasterize(self, polygons, canvas=512):
        """Coordinates are normalized float32, before augmentation/resizing.

        Exactly the frozen segmentation resampling rule and polygon2mask primitive,
        evaluated at source canvas with ratio1; not a claim of matching augmented
        historical 768/ratio4 training tensors or their overlap-composited masks.
        """
        ps=[np.asarray(p,dtype=np.float32).copy() for p in polygons]
        if not ps:return []
        resampled=self.resample(ps,n=self.resample_count(ps))
        return [self.polygon2mask((canvas,canvas),[(p*np.float32(canvas)).reshape(-1)],1,1) for p in resampled]


def parse_labels(text):
    lines=[l for l in text.splitlines() if l.strip()]
    polygons=[]
    for line in lines:
        values=line.split();require(values[0]=='0' or float(values[0])==0,'FAIL_LABEL_CLASS')
        p=np.asarray(values[1:],dtype=np.float32)
        require(len(p)>=6 and len(p)%2==0 and np.isfinite(p).all() and np.all((p>=0)&(p<=1)), 'FAIL_LABEL_COORDINATES')
        polygons.append(p.reshape(-1,2))
    return lines,polygons


def compare(a,b):
    require(a.shape==b.shape,'FAIL_MASK_SHAPE')
    a=a.astype(bool);b=b.astype(bool);intersection=int(np.count_nonzero(a&b));union=int(np.count_nonzero(a|b))
    return {'xor_pixels':int(np.count_nonzero(a!=b)), 'intersection':intersection,'union':union,
            'raster_IoU':intersection/union if union else 1.,'area_original_raster':int(a.sum()),'area_canonical_raster':int(b.sum())}


def topology(mask):
    mask=np.asarray(mask,dtype=np.uint8)
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    holes=0
    if hierarchy is not None:
        for item in hierarchy[0]:
            depth=0;parent=int(item[3])
            while parent>=0:depth+=1;parent=int(hierarchy[0,parent,3])
            holes+=depth%2
    return {'connected_components':int(cv2.connectedComponents(mask,connectivity=8)[0]-1),
            'connected_components_4':int(cv2.connectedComponents(mask,connectivity=4)[0]-1),'hole_count':holes}


def mask_representable(mask):
    t=topology(mask)
    require(t['connected_components']<=1,'BLOCKED_BY_MULTIPART_INSTANCE_REPRESENTATION')
    require(t['hole_count']==0,'BLOCKED_BY_HOLE_REPRESENTATION')
    return t


def canonicalize(polygons, index, raster):
    """Deterministic ordered candidate family, selected on TRAINING raster only.

    No iteration/tuning on validation. No discarded components or filled holes.
    All siblings are re-rasterized: segment-resample count is image-dependent.
    """
    source=raster.rasterize(polygons);mask=source[index];p=Polygon(polygons[index]*512)
    result={'validity_reason':explain_validity(p),'source_raster_sha256':array_sha(mask),**topology(mask),
            'canonicalization_method':None,'canonical_polygon':None,'canonical_raster_sha256':None,
            'xor_pixels':None,'raster_IoU':None,'intersection':None,'union':None,
            'area_original_raster':int(mask.sum()),'area_canonical_raster':None,'candidates':[]}
    if p.is_valid and p.area>0:
        return result|{'canonicalization_status':'IDENTITY','canonicalization_method':'IDENTITY',
                      'canonical_polygon':polygons[index].tolist(),'canonical_raster_sha256':array_sha(mask),**compare(mask,mask)},polygons[index].copy()
    if result['connected_components']>1:
        return result|{'canonicalization_status':'BLOCKED_BY_MULTIPART_INSTANCE_REPRESENTATION'},None
    if result['hole_count']:
        return result|{'canonicalization_status':'BLOCKED_BY_HOLE_REPRESENTATION'},None
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    candidates=[('RASTER_EXTERNAL_CHAIN_NONE',Polygon(contours[0].reshape(-1,2))) if contours and len(contours[0])>=3 else ('RASTER_EXTERNAL_CHAIN_NONE',Polygon()),
                ('SOURCE_MAKE_VALID_CANDIDATE',make_valid(p)),('SOURCE_BUFFER_ZERO_CANDIDATE',p.buffer(0))]
    for method,q in candidates:
        trial={'method':method,'geometry_type':q.geom_type,'accepted':False}
        if q.geom_type!='Polygon':
            trial['reason']='BLOCKED_BY_MULTIPART_INSTANCE_REPRESENTATION' if q.geom_type=='MultiPolygon' else 'UNSUPPORTED_GEOMETRY_COLLECTION_NO_COMPONENT_DISCARD'
        elif q.is_empty or not q.is_valid or q.area<=0:trial['reason']='INVALID_CANDIDATE_TOPOLOGY'
        elif len(q.interiors):trial['reason']='BLOCKED_BY_HOLE_REPRESENTATION'
        else:
            candidate=np.asarray(q.exterior.coords[:-1],dtype=np.float32)/np.float32(512)
            q32=Polygon(candidate*512)
            if not q32.is_valid or q32.area<=0:trial['reason']='FLOAT32_TOPOLOGY_INVALID'
            else:
                scene=[x.copy() for x in polygons];scene[index]=candidate
                masks=raster.rasterize(scene);m=compare(mask,masks[index])
                trial.update(m,canonical_raster_sha256=array_sha(masks[index]),candidate_polygon=candidate.tolist(),
                             sibling_xor_pixels=[compare(a,b)['xor_pixels'] for a,b in zip(source,masks)])
                trial['accepted']=all(v==0 for v in trial['sibling_xor_pixels']) and bool(mask.sum())
                trial['reason']='EXACT_RASTER_EQUIVALENT' if trial['accepted'] else 'RASTER_MISMATCH'
                if trial['accepted']:
                    result['candidates'].append(trial)
                    return result|{'canonicalization_status':'EXACT_RASTER_EQUIVALENT','canonicalization_method':method,
                                   'canonical_polygon':candidate.tolist(),'canonical_raster_sha256':array_sha(masks[index]),**m},candidate
        result['candidates'].append(trial)
    return result|{'canonicalization_status':'BLOCKED_NO_EXACT_SINGLE_POLYGON_CANDIDATE'},None


def integer_bounds(target):
    xy=np.asarray(target,dtype=np.float64)*512;a,b=xy.min(axis=0);c,d=xy.max(axis=0)
    left=max(0,min(256,math.floor((a+c)/2-128)));top=max(0,min(256,math.floor((b+d)/2-128)))
    require(left<=a and top<=b and c<=left+256 and d<=top+256,'FAIL_INTEGER_PATCH_CONTAINMENT')
    return left,top,left+256,top+256


def positive_parts(q):
    if q.is_empty or q.area==0:return []
    if q.geom_type=='Polygon':return [q]
    return [p for child in q.geoms for p in positive_parts(child)]


def patch_cycle(canonical, selected_id, raster, authoritative_masks=None):
    require(all(p is not None for p in canonical),'BLOCKED_UNRESOLVED_SOURCE_INSTANCE')
    for p in canonical:
        q=Polygon(p*512);require(q.is_valid and q.area>0,'FAIL_CANONICAL_TOPOLOGY')
    left,top,right,bottom=integer_bounds(canonical[selected_id]);window=box(left,top,right,bottom)
    masks=authoritative_masks if authoritative_masks is not None else raster.rasterize(canonical)
    output=[];records=[];geometry_errors=[]
    for i,p in enumerate(canonical):
        expected=masks[i][top:bottom,left:right];q=Polygon(p*512);clipped=q.intersection(window);parts=positive_parts(clipped)
        r={'GT_instance_id':i,'source_raster_sha256':array_sha(masks[i]),'source_patch_sha256':array_sha(expected),
           'visible_source_pixels':int(expected.sum()),'state':'DROPPED' if not expected.any() else 'FULL' if expected.sum()==masks[i].sum() else 'PARTIAL',
           'polygon':None}
        if len(parts)>1:r['error']='BLOCKED_BY_MULTIPART_INSTANCE_REPRESENTATION'
        elif parts and len(parts[0].interiors):r['error']='BLOCKED_BY_HOLE_REPRESENTATION'
        elif parts:
            local=(np.asarray(parts[0].exterior.coords[:-1],dtype=np.float32)-np.asarray([left,top],dtype=np.float32))/256
            v=Polygon(local)
            if not v.is_valid or v.area<=0 or not np.isfinite(local).all() or not np.all((local>=0)&(local<=1)):
                r['error']='FAIL_PATCH_LABEL_INVALID'
            else:r['polygon']=local.tolist();output.append((i,local))
        if not parts and expected.any():r['error']='FAIL_VISIBLE_FRAGMENT_WITHOUT_POLYGON'
        if 'error' in r:geometry_errors.append(r['error'])
        records.append(r)
    generated=raster.rasterize([p for i,p in output],256);by_id={i:m for (i,p),m in zip(output,generated)}
    for r in records:
        i=r['GT_instance_id'];m=by_id.get(i,np.zeros((256,256),np.uint8))
        r.update(compare(masks[i][top:bottom,left:right],m),vector_patch_sha256=array_sha(m))
    target_kept=int(masks[selected_id][top:bottom,left:right].sum())==int(masks[selected_id].sum()) and bool(masks[selected_id].sum())
    failures=geometry_errors.copy()
    if not target_kept:failures.append('FAIL_SELECTED_TARGET_MASK_RETENTION')
    if not output:failures.append('FAIL_ELIGIBLE_PATCH_EMPTY')
    if any(r['xor_pixels'] for r in records):failures.append('FAIL_RASTER_PATCH_MISMATCH')
    return {'status':'PASS' if not failures else 'BLOCKED','failures':sorted(set(failures)),
            'patch_bounds':[left,top,right,bottom],'target_GT_id':selected_id,'selected_target_mask_retained':target_kept,
            'original_GT_count':len(canonical),'retained_GT_count':len(output),'records':records,
            'FULL':sum(r['state']=='FULL' for r in records),'PARTIAL':sum(r['state']=='PARTIAL' for r in records),
            'DROPPED':sum(r['state']=='DROPPED' for r in records),'empty_patch':not output,
            'transformed_label_hash':digest([r['polygon'] for r in records]),
            'selected_retained_other_dropped':target_kept and any(r['state']=='DROPPED' for r in records if r['GT_instance_id']!=selected_id)}
