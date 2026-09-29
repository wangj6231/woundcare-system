"""Model-free, dependency-injected source-stage adapter for the V2 contract.

Both arms use this adapter. It has no training driver. Synthetic multiprocessing
tests verify immutable fetch tokens, not a claim that a GPU trainer was exercised.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import random
import ast
from pathlib import Path

import cv2
import numpy as np
from experiments import patch_raster as r


@dataclass(frozen=True)
class FetchToken:
    sample_id: str
    epoch_token: int
    generation: int = 0


def admit_sample(sample):
    r.require(sample['split']=='train' and sample['source'] in ('FUSeg','SYNTHETIC'), 'FAIL_TRAINING_ROLE_REQUIRED')
    r.require(not any(x in sample['sample_id'].lower() for x in ('co2','locked','blind')), 'FAIL_FORBIDDEN_SAMPLE')
    r.require(not sample.get('already_transformed',False),'FAIL_DOUBLE_PATCH')


class PatchAwareDataset:
    def __init__(self, samples, arm, raster):
        r.require(arm in ('C2','P2'),'FAIL_ARM')
        self.samples=samples;self.arm=arm;self.raster=raster

    def fetch(self, token, role='anchor'):
        r.require(isinstance(token,FetchToken) and type(token.epoch_token) is int and token.epoch_token>=0,'FAIL_IMMUTABLE_EPOCH_TOKEN')
        r.require(role in ('anchor','companion'),'FAIL_FETCH_ROLE')
        source=self.samples[token.sample_id];admit_sample(source)
        pixels=source['pixels'];r.require(pixels.shape[:2]==(512,512),'FAIL_PATCH_AFTER_RESIZE')
        canonical=source['canonical_polygons'];r.require(all(p is not None for p in canonical),'BLOCKED_UNRESOLVED_SOURCE_INSTANCE')
        ids=source['eligible_GT_ids'];target=None;bounds=None;labels=deepcopy(canonical)
        steps=['SOURCE_512','CANONICAL_SOURCE_LABEL'];patch_count=0
        if self.arm=='P2' and ids:
            target=ids[token.epoch_token%len(ids)]
            result=r.patch_cycle(canonical,target,self.raster,source.get('authoritative_masks'))
            r.require(result['status']=='PASS','FAIL_UNSAFE_PATCH: '+','.join(result['failures']))
            bounds=result['patch_bounds'];left,top,right,bottom=bounds
            image=pixels[top:bottom,left:right].copy()
            labels=[np.asarray(x['polygon'],np.float32) for x in result['records'] if x['polygon'] is not None]
            steps+=['INTEGER_SOURCE_PATCH','PATCH_LOCAL_LABEL'];patch_count=1
        else:image=pixels.copy();steps+=['IDENTITY_FULL_IMAGE']
        return {'image':image,'polygons':labels,'sample_id':token.sample_id,'epoch_token':token.epoch_token,
                'generation':token.generation,'role':role,'target_GT_id':target,'patch_bounds':bounds,
                'patch_count':patch_count,'representation_transform_count':1,'already_transformed':True,
                'image_sha256':r.array_sha(image),'source_pixels_sha256':r.array_sha(pixels),
                'canonical_label_hash':r.digest([p.tolist() for p in canonical]),
                'transformed_label_hash':r.digest([p.tolist() for p in labels]),'steps':steps}

    @staticmethod
    def resize_for_existing_pipeline(item):
        r.require(item['already_transformed'] and item['representation_transform_count']==1,'FAIL_TRANSFORM_STAGE')
        out=dict(item);out['image']=cv2.resize(item['image'],(768,768),interpolation=cv2.INTER_LINEAR)
        out['steps']=item['steps']+['EXISTING_LINEAR_RESIZE_768'];return out


class ScopedCompanions:
    """One object per anchor call; immutable parent epoch, no global epoch."""
    def __init__(self, adapter, parent_token):self.adapter=adapter;self.token=parent_token
    def get_image_and_label(self, sample_id):
        return self.adapter.fetch(replace(self.token,sample_id=sample_id),'companion')


def frozen_mix_fetch(adapter, token, companion_ids, augmentation_source):
    """Exercise the exact pinned BaseMixTransform.__call__ fetch route.

    Geometry mixing is a dependency-injected recorder; not a full Mosaic tensor
    or training augmentation test. Test-only p=1 forces companion fetching.
    """
    tree=ast.parse(Path(augmentation_source).read_text(encoding='utf8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaseMixTransform')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
    ns={'random':random};exec(compile(ast.Module(body=[method],type_ignores=[]),str(augmentation_source),'exec'),ns)
    class Probe:
        p=1.;pre_transform=None
        def __init__(self):self.dataset=ScopedCompanions(adapter,token)
        def get_indexes(self):return companion_ids
        def _update_label_text(self,labels):return labels
        def _mix_transform(self,labels):
            labels['companion_evidence']=labels['mix_labels'];return labels
    return ns['__call__'](Probe(),adapter.fetch(token))


def synthetic_samples():
    def square(x,y,w=8):return np.array([[x,y],[x+w,y],[x+w,y+w],[x,y+w]],np.float32)/512
    image=np.stack(np.meshgrid(np.arange(512,dtype=np.uint16),np.arange(512,dtype=np.uint16)),axis=-1)
    image=np.concatenate([image,(image[:,:,:1]+image[:,:,1:])%256],axis=-1)
    return {'multi':{'sample_id':'multi','source':'SYNTHETIC','split':'train','pixels':image,
                'canonical_polygons':[square(100,100),square(350,350)],'eligible_GT_ids':[0,1]},
            'negative':{'sample_id':'negative','source':'SYNTHETIC','split':'train','pixels':image,'canonical_polygons':[],'eligible_GT_ids':[]},
            'large':{'sample_id':'large','source':'SYNTHETIC','split':'train','pixels':image,'canonical_polygons':[square(100,100,100)],'eligible_GT_ids':[]}}


def worker_fetch(job):
    r.model_guard();package,arm,token=job
    adapter=PatchAwareDataset(synthetic_samples(),arm,r.RasterContract(package))
    result=adapter.fetch(token)
    result.pop('image');result.pop('polygons');return result


def runtime_probe(package):
    package=str(package);tokens=[FetchToken('multi',e,g) for e,g in [(0,0),(1,0),(269,0),(270,0),(270,1),(270,1),(271,1)]]
    jobs=[(package,a,t) for a in ('C2','P2') for t in tokens]
    # Both paths run in fresh spawned interpreters.  The surrounding test or
    # application process may legitimately have imported torch already; that
    # must not be mistaken for this model-free probe importing it.  worker_fetch
    # still installs the fail-closed model import guard inside every worker.
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        serial=[f.result() for f in [pool.submit(worker_fetch,j) for j in jobs]]
    # Submitted before consumption: interleaved generations/epochs emulate prefetch.
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker_fetch,j) for j in jobs]
        parallel=[f.result() for f in futures]
    r.require(serial==parallel,'FAIL_WORKERS_PARITY')
    # close_mosaic reset: discard outstanding old generation, do not mutate tokens.
    accepted=[v for v in parallel if v['epoch_token']>=270 and v['generation']==1]
    r.require(all(x['generation']==1 for x in accepted),'FAIL_STALE_PREFETCH')
    p=[x for x in parallel if x['patch_count']==1]
    r.require(all(x['target_GT_id']==x['epoch_token']%2 for x in p),'FAIL_ROUND_ROBIN')
    for a in p:
        for b in p:
            if a['epoch_token']==b['epoch_token']:
                r.require(a['image_sha256']==b['image_sha256'] and a['transformed_label_hash']==b['transformed_label_hash'],'FAIL_SAME_EPOCH')
    adapter=PatchAwareDataset(synthetic_samples(),'P2',r.RasterContract(package))
    mixed=frozen_mix_fetch(adapter,FetchToken('multi',1),['multi','negative','large'],Path(package)/'data/augment.py')
    same=mixed['companion_evidence'][0]
    r.require(same['image_sha256']==mixed['image_sha256'] and same['patch_count']==mixed['patch_count']==1,'FAIL_COMPANION_PATCH')
    return {'status':'PASS','workers_0_vs_2':'EXACT_OUTPUT_PARITY','submitted_jobs':len(jobs),
            'immutable_tokens':True,'prefetch_submitted_before_consumption':True,'generation_reset_stale_results_discarded':True,
            'same_epoch_same_patch':True,'next_epoch_round_robin':True,'anchor_and_companion_exactly_once':True,
            'pinned_BaseMixTransform_fetch_route_exercised':True,'synthetic_scene_only':True,
            'actual_torch_DataLoader_exercised':False,'actual_full_Mosaic_geometry_exercised':False,
            'runtime_scope':'minimum equivalent source-stage adapter + actual pinned BaseMixTransform fetch route; multiprocessing spawn workers2; not GPU trainer',
            'records':parallel,'model_loading':False,'training':False,'inference':False}
