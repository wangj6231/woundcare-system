"""CPU-only diagnostic DataLoader, NOT a training runner or full augmentation adapter."""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
import importlib.abc
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from experiments import mask_first as m
from experiments.patch_geometry import admit, require
from experiments.patch_raster import RasterContract, parse_labels, file_sha, array_sha


def forbidden(*args, **kwargs):
    raise RuntimeError('E12_FORBIDDEN_MODEL_FORWARD_OPTIMIZER_GPU')


class NoNetworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ultralytics','tensorflow','onnxruntime'}:
            forbidden()


def cpu_guard(_=None):
    import torch
    torch.set_num_threads(1)
    torch.load=forbidden;torch.jit.load=forbidden
    torch.nn.Module.__init__=forbidden;torch.nn.Module._call_impl=forbidden
    torch.optim.Optimizer.__init__=forbidden;torch.cuda._lazy_init=forbidden
    sys.meta_path.insert(0,NoNetworks())
    cv2.setNumThreads(1)


@dataclass(frozen=True)
class FetchToken:
    sample_id: str
    epoch_token: int
    generation: int


class DiagnosticDataset:
    def __init__(self, rows, arm, synthetic=False):
        self.rows={x['sample_id']:x for x in rows};self.ids=list(self.rows)
        self.arm=arm;self.synthetic=synthetic
    def __len__(self):return len(self.rows)
    def scene(self, sid, epoch):
        row=self.rows[sid]
        if self.synthetic:
            masks=np.zeros((2,512,512),np.uint8)
            masks[0,248:254,248:254]=1;masks[1,5:30,5:30]=1
            image=np.repeat((masks.any(axis=0)*255).astype(np.uint8)[:,:,None],3,axis=2)
            eligible=[0,1]
        else:
            admit(row)
            for k in ('image','label'):
                require(file_sha(row[k+'_path'])==row[k+'_hash'],'FAIL_SOURCE_HASH')
            image=cv2.imread(row['image_path']);require(image is not None and image.shape==(512,512,3),'FAIL_IMAGE')
            polygons=parse_labels(Path(row['label_path']).read_text(encoding='utf8'))[1]
            masks=np.asarray(RasterContract('C:/Python312/Lib/site-packages/ultralytics').rasterize(polygons),np.uint8).reshape(-1,512,512)
            require([array_sha(x) for x in masks]==row['mask_hashes'],'FAIL_MASK_HASH')
            eligible=row['mask_eligible_GT_ids']
        ids=[f'{sid}:{j}' for j in range(len(masks))];target=None
        if self.arm=='P3' and eligible:
            target=eligible[epoch%len(eligible)];patch=m.patch_masks(masks,ids,target)
            x1,y1,x2,y2=patch['bounds'];image=image[y1:y2,x1:x2].copy();masks=patch['masks']
        result=m.resize_scene(image,masks,ids,768)
        return result,target
    def __getitem__(self, token):
        require(isinstance(token,FetchToken) and token.sample_id in self.rows,'FAIL_FETCH_TOKEN')
        scene,target=self.scene(token.sample_id,token.epoch_token)
        # Force branch for diagnostic coverage only; NOT the production p=.1 RNG.
        mosaic=token.epoch_token<270
        if mosaic:
            start=self.ids.index(token.sample_id)
            scenes=[scene]+[self.scene(self.ids[(start+i)%len(self.ids)],token.epoch_token)[0] for i in range(1,4)]
            scene=m.mosaic_scene(scenes,(768,768))
            scene=m.warp_scene(scene['image'],scene['masks'],scene['ids'],np.array([[1,0,-384],[0,1,-384],[0,0,1]],float),768)
        scene=m.flip_scene(scene['image'],scene['masks'],scene['ids'],'horizontal')
        scene=m.hsv_scene(scene['image'],scene['masks'],scene['ids'],[1.01,1.1,.95])
        targets=m.format_targets(scene['masks'],scene['ids'])
        return dict(targets,image=scene['image'],token=asdict(token),selected_target=target,mosaic=mosaic)


def collate(rows):
    import torch
    return {'img':torch.from_numpy(np.stack([r['image'][:,:,::-1].transpose(2,0,1) for r in rows])),
            'masks':torch.from_numpy(np.stack([r['masks'] for r in rows])),
            'cls':torch.from_numpy(np.concatenate([r['cls'] for r in rows])),
            'bboxes':torch.from_numpy(np.concatenate([r['bboxes'] for r in rows])),
            'batch_idx':torch.from_numpy(np.concatenate([np.full(len(r['ids']),i,np.int64) for i,r in enumerate(rows)])),
            'metadata':[{k:r[k] for k in ('ids','token','selected_target','mosaic','lowres_empty_ids','fully_occluded_ids')} for r in rows]}


def consume(dataset, tokens, workers):
    from torch.utils.data import DataLoader
    opts={'multiprocessing_context':'spawn','prefetch_factor':2,'worker_init_fn':cpu_guard} if workers else {}
    loader=DataLoader(dataset,batch_size=4,sampler=tokens,num_workers=workers,collate_fn=collate,pin_memory=False,**opts)
    output=[]
    for batch in loader:
        n=len(batch['cls'])
        require(batch['img'].shape==(4,3,768,768) and batch['masks'].shape==(4,192,192),'FAIL_BATCH_SHAPE')
        require(batch['bboxes'].shape==(n,4) and batch['batch_idx'].shape==(n,),'FAIL_LOSS_TARGET_FIELDS')
        require(bool(((batch['bboxes']>=0)&(batch['bboxes']<=1)).all()),'FAIL_NORMALIZED_BOX')
        for i,meta in enumerate(batch['metadata']):
            require(int((batch['batch_idx']==i).sum())==len(meta['ids']),'FAIL_TARGET_INDEX')
            require(int(batch['masks'][i].max())<=len(meta['ids']),'FAIL_OVERLAP_INDEX')
        output.append({'tensor_sha256':{k:array_sha(batch[k].numpy()) for k in ('img','masks','cls','bboxes','batch_idx')},
                       'metadata':batch['metadata'],'instance_count':n})
    return output


def run(rows, synthetic=False):
    cpu_guard()
    import torch
    ids=[x['sample_id'] for x in rows];require(len(ids)==4,'FAIL_SMOKE_SUBSET_SIZE')
    tokens=[FetchToken(sid,epoch,0) for epoch in (269,270) for sid in ids]
    results={}
    for arm in ('C3','P3'):
        ds=DiagnosticDataset(rows,arm,synthetic)
        a=consume(ds,tokens,0);b=consume(ds,tokens,2)
        require(a==b,'FAIL_WORKER_PARITY')
        reset=consume(ds,[FetchToken(sid,270,1) for sid in ids],2)
        require(reset[0]['tensor_sha256']==b[1]['tensor_sha256'],'FAIL_RESET_PIXEL_PARITY')
        require(all(x['token']['generation']==1 and not x['mosaic'] for x in reset[0]['metadata']),'FAIL_RESET_TOKEN')
        results[arm]={'batches':b,'reset':reset,'worker_parity':True}
    require(not torch.cuda.is_initialized(),'FAIL_CUDA_INITIALIZED')
    require(torch.load is forbidden and torch.nn.Module.__init__ is forbidden
            and torch.nn.Module._call_impl is forbidden and torch.optim.Optimizer.__init__ is forbidden,'FAIL_CPU_GUARDS')
    require(not any(k.split('.')[0] in {'ultralytics','tensorflow','onnxruntime'} for k in sys.modules),'FAIL_NETWORK_PACKAGE_IMPORTED')
    return {'scope':'DIAGNOSTIC_PRIMITIVES_NOT_FULL_BASELINE_AUGMENTATION','worker_parity':True,
            'batch_image_shape':[4,3,768,768],'batch_mask_shape':[4,192,192],
            'workers':[0,2],'prefetch_factor':2,'close_mosaic_token_boundary':[269,270],
            'reset_method':'new iterator and workers; generation immutable; no stale iterator consumption',
            'synthetic':synthetic,'training_sample_ids':[] if synthetic else ids,
            'MODEL_LOADED':False,'FORWARD_PASS':False,'TRAINING':False,'GPU_USED':False,
            'guard_checks_passed':True,'torch_version':torch.__version__,'CUDA_initialized':torch.cuda.is_initialized(),
            'validation_images_used':0,'test_images_used':0,'CO2Wounds_used':False,'arms':results}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--synthetic',action='store_true');p.add_argument('--manifest');a=p.parse_args()
    require(a.synthetic != bool(a.manifest),'FAIL_CHOOSE_INPUT')
    rows=[{'sample_id':f'synthetic{i}'} for i in range(4)] if a.synthetic else json.loads(Path(a.manifest).read_text(encoding='utf8'))['samples']
    print(json.dumps(run(rows,a.synthetic),allow_nan=False))
