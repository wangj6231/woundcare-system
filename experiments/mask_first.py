"""E1.2 CPU feasibility interfaces. Not an authorized training pipeline.

After source rasterization masks are the only GT. Never reconstruct polygons.
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from experiments.patch_raster import require


def resize_scene(image,masks,ids,size):
    require(image.shape[:2]==masks.shape[1:] and len(ids)==len(masks),'FAIL_SCENE_ALIGNMENT')
    output=np.stack([cv2.resize(x,(size,size),interpolation=cv2.INTER_NEAREST) for x in masks]) if len(masks) else np.zeros((0,size,size),np.uint8)
    return {'image':cv2.resize(image,(size,size),interpolation=cv2.INTER_LINEAR),'masks':output,'ids':list(ids),
            'bboxes':[tight_bbox(x) for x in output]}


def warp_scene(image,masks,ids,matrix,size):
    require(image.shape[:2]==masks.shape[1:] and len(ids)==len(masks),'FAIL_SCENE_ALIGNMENT')
    require(np.asarray(matrix).shape==(3,3) and np.isfinite(matrix).all(),'FAIL_MATRIX')
    output=np.stack([cv2.warpPerspective(x,matrix,(size,size),flags=cv2.INTER_NEAREST,borderValue=0) for x in masks]) if len(masks) else np.zeros((0,size,size),np.uint8)
    return {'image':cv2.warpPerspective(image,matrix,(size,size),flags=cv2.INTER_LINEAR,borderValue=(114,114,114)),
            'masks':output,'ids':list(ids),'bboxes':[tight_bbox(x) for x in output]}


def flip_scene(image,masks,ids,direction):
    require(direction in ('horizontal','vertical'),'FAIL_FLIP_DIRECTION')
    axis=1 if direction=='horizontal' else 0
    output=np.flip(masks,axis+1).copy()
    return {'image':np.flip(image,axis).copy(),'masks':output,'ids':list(ids),'bboxes':[tight_bbox(x) for x in output]}


def mosaic_scene(scenes,center):
    require(len(scenes)==4,'FAIL_MOSAIC_TILES')
    s=scenes[0]['image'].shape[0];xc,yc=center
    require(s//2<=xc<=3*s//2 and s//2<=yc<=3*s//2,'FAIL_MOSAIC_CENTER')
    image=np.full((2*s,2*s,3),114,np.uint8);masks=[];ids=[];lineage=[];placements=[]
    for i,scene in enumerate(scenes):
        h,w=scene['image'].shape[:2];require((h,w)==(s,s),'FAIL_TILE_SIZE')
        if i==0:
            a=(max(xc-w,0),max(yc-h,0),xc,yc);b=(w-(a[2]-a[0]),h-(a[3]-a[1]),w,h)
        elif i==1:
            a=(xc,max(yc-h,0),min(xc+w,2*s),yc);b=(0,h-(a[3]-a[1]),min(w,a[2]-a[0]),h)
        elif i==2:
            a=(max(xc-w,0),yc,xc,min(2*s,yc+h));b=(w-(a[2]-a[0]),0,w,min(h,a[3]-a[1]))
        else:
            a=(xc,yc,min(xc+w,2*s),min(2*s,yc+h));b=(0,0,min(w,a[2]-a[0]),min(h,a[3]-a[1]))
        image[a[1]:a[3],a[0]:a[2]]=scene['image'][b[1]:b[3],b[0]:b[2]]
        for source_id,mask in zip(scene['ids'],scene['masks']):
            canvas=np.zeros((2*s,2*s),np.uint8);canvas[a[1]:a[3],a[0]:a[2]]=mask[b[1]:b[3],b[0]:b[2]]
            masks.append(canvas);ids.append(f'occurrence{i}/{source_id}');lineage.append({'occurrence':i,'source_instance_id':source_id})
        placements.append({'tile':i,'destination':a,'source_slice':b})
    output=np.stack(masks) if masks else np.zeros((0,2*s,2*s),np.uint8)
    require(len(set(ids))==len(ids),'FAIL_MOSAIC_ID_COLLISION')
    return {'image':image,'masks':output,'ids':ids,'source_lineage':lineage,'placements':placements,'bboxes':[tight_bbox(x) for x in output]}


def format_targets(masks,ids,mask_ratio=4):
    require(len(masks)==len(ids) and len(set(ids))==len(ids),'FAIL_INSTANCE_IDENTITY')
    h,w=masks.shape[1:];require(h%mask_ratio==0 and w%mask_ratio==0,'FAIL_MASK_RESOLUTION')
    visible=[i for i,x in enumerate(masks) if x.any()]
    small=[cv2.resize(masks[i],(w//mask_ratio,h//mask_ratio),interpolation=cv2.INTER_NEAREST) for i in visible]
    areas=np.array([x.sum() for x in small],dtype=np.int64)
    order=np.argsort(-areas);ordered=[visible[int(j)] for j in order]
    encoded=np.zeros((h//mask_ratio,w//mask_ratio),dtype=np.int32)
    bboxes=[]
    for rank,j in enumerate(order):
        encoded[small[int(j)]>0]=rank+1
        x1,y1,x2,y2=tight_bbox(masks[visible[int(j)]])
        bboxes.append([(x1+x2)/(2*w),(y1+y2)/(2*h),(x2-x1)/w,(y2-y1)/h])
    return {'masks':encoded,'bboxes':np.asarray(bboxes,np.float32).reshape(-1,4),
            'cls':np.zeros((len(ordered),1),np.float32),'ids':[ids[i] for i in ordered],
            'source_indices':ordered,'dropped_empty_ids':[ids[i] for i in range(len(ids)) if i not in visible],
            'lowres_empty_ids':[ids[visible[int(j)]] for j in order if not small[int(j)].any()],
            'fully_occluded_ids':[ids[ordered[j]] for j in range(len(ordered)) if not (encoded==j+1).any()]}


def hsv_scene(image,masks,ids,gains):
    require(len(gains)==3 and np.isfinite(gains).all(),'FAIL_HSV_GAINS')
    hue,sat,val=cv2.split(cv2.cvtColor(image,cv2.COLOR_BGR2HSV));x=np.arange(256,dtype=np.float64)
    lh=((x*gains[0])%180).astype(np.uint8);ls=np.clip(x*gains[1],0,255).astype(np.uint8);lv=np.clip(x*gains[2],0,255).astype(np.uint8)
    output=cv2.cvtColor(cv2.merge((cv2.LUT(hue,lh),cv2.LUT(sat,ls),cv2.LUT(val,lv))),cv2.COLOR_HSV2BGR)
    return {'image':output,'masks':masks.copy(),'ids':list(ids),'bboxes':[tight_bbox(x) for x in masks]}


def tight_bbox(mask):
    ys,xs=np.where(mask>0)
    return None if not len(xs) else [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]


def patch_masks(masks,ids,target):
    require(masks.ndim==3 and masks.shape[1:]==(512,512),'FAIL_SOURCE_MASK_SHAPE')
    require(len(ids)==len(masks) and len(set(ids))==len(ids),'FAIL_INSTANCE_IDENTITY')
    require(0<=target<len(masks),'FAIL_TARGET_ID')
    b=tight_bbox(masks[target]);require(b is not None,'FAIL_EMPTY_SELECTED_MASK')
    x=max(0,min(256,math.floor((b[0]+b[2])/2-128)));y=max(0,min(256,math.floor((b[1]+b[3])/2-128)))
    sliced=masks[:,y:y+256,x:x+256].copy();before=masks.sum(axis=(1,2));after=sliced.sum(axis=(1,2))
    require(before[target]==after[target],'FAIL_SELECTED_MASK_RETENTION')
    return {'masks':sliced,'ids':list(ids),'bounds':[x,y,x+256,y+256],
            'states':['DROPPED' if b==0 else 'FULL' if a==b else 'PARTIAL' for a,b in zip(before,after)],
            'source_pixels':[int(x) for x in before],'retained_pixels':[int(x) for x in after],
            'selected_retained':True,'bbox_from_mask':[tight_bbox(mask) for mask in sliced]}
