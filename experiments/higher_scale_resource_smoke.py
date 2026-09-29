"""One-shot synthetic1024/batch4 forward/backward resource diagnostic, no updates."""
from __future__ import annotations
from pathlib import Path
import numpy as np


class SmokeAccess:
    def __init__(self,root,checkpoint,out):
        self.root=Path(root).resolve();self.checkpoint=Path(checkpoint).resolve();self.out=Path(out).resolve()
        self.project_reads=set();self.denied=[]
    def allowed_read(self,path):
        p=Path(path).resolve()
        if p==self.checkpoint or p.is_relative_to(self.out):return True
        if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.npy','.npz','.pt','.pth','.ckpt','.safetensors'}:return False
        if not p.is_relative_to(self.root):return True  # installed libraries/runtime only, no external-data input API
        return p.is_relative_to(self.root/'experiments') and p.suffix in {'.py','.pyc'}
    def allowed_write(self,path):
        p=Path(path).resolve()
        if p.suffix.lower() in {'.pt','.pth','.ckpt','.safetensors'}:return False
        if p.is_relative_to(self.root):return p.is_relative_to(self.out) and p.suffix in {'.json','.log','.lock'}
        return True  # runtime caches outside project; checkpoint extensions remain forbidden
    def audit(self,event,args):
        if event=='socket.connect':raise RuntimeError('F0_NETWORK_FORBIDDEN')
        if event!='open' or not isinstance(args[0],(str,bytes,Path)):return
        import os
        p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR)))
        allowed=self.allowed_write(p) if writing else self.allowed_read(p)
        if not allowed:
            self.denied.append(str(p));raise RuntimeError('F0_FILE_ACCESS_FORBIDDEN: '+str(p))
        if not writing and p.is_relative_to(self.root):self.project_reads.add(str(p))


def forbidden_operation(*args,**kwargs):
    raise RuntimeError('F0_OPTIMIZER_STEP_OR_CHECKPOINT_SAVE_FORBIDDEN')


def synthetic_source():
    images=np.random.Generator(np.random.PCG64(42)).integers(0,256,(4,512,512,3),dtype=np.uint8)
    rectangles=[]
    for y in range(4):
        for x in range(4):
            left=.04+x*.24;top=.04+y*.24;size=.025+(x+y)*.007
            rectangles.append([[left,top],[left+size,top],[left+size,top+size],[left,top+size]])
    polygons=np.repeat(np.asarray(rectangles,np.float32)[None],4,axis=0)
    return images,polygons


def run():
    import gc
    import hashlib
    import json
    import os
    import sys
    import time
    from types import SimpleNamespace
    from experiments.phase_f0 import ROOT,PRO,OUT,PREFIX,INIT_SHA,read,save,sha,require,future_absent
    protocol=read(PRO/f'{PREFIX}_resource_smoke_protocol.json')
    config=read(PRO/f'{PREFIX}_control_config.json')
    require(protocol['imgsz']==1024 and protocol['batch']==4 and protocol['attempts_allowed']==1,'FAIL_SMOKE_PROTOCOL')
    seal=read(OUT/'pre_smoke_seal.json')
    for p,h in seal['artifacts_sha256'].items():require(sha(ROOT/p)==h,'FAIL_PREFREEZE_CHANGED')
    checkpoint=ROOT/protocol['initialization']['path'];require(sha(checkpoint)==INIT_SHA,'FAIL_INIT_HASH')
    future_absent()
    save(OUT/'execution.lock',{'attempt':1,'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(PRO/f'{PREFIX}_resource_smoke_protocol.json'),
                              'no_retry':True,'official_training':False})
    access=SmokeAccess(ROOT,checkpoint,OUT);sys.dont_write_bytecode=True
    sys.addaudithook(access.audit)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['YOLO_AUTOINSTALL']='false'
    started=time.monotonic();torch=None;model=None
    result={'kind':'SYNTHETIC_RESOURCE_SMOKE','imgsz':1024,'batch':4,'AMP':True,
        'forward_success':False,'backward_success':False,'OOM':False,'NaN_Inf':False,
        'optimizer_steps':0,'checkpoint_saves':0,'research_metrics_generated':False,
        'real_dataset_pixels_used':0,'validation_images_used':0,'test_images_used':0,'CO2Wounds_used':False,
        'official_training':False,'RESOURCE_FEASIBILITY':'FAIL','iterations_completed':[]}
    try:
        import torch
        import cv2
        torch.set_num_threads(2);cv2.setNumThreads(1)
        torch.manual_seed(42);np.random.seed(42)
        torch.use_deterministic_algorithms(True,warn_only=True)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        require(torch.cuda.is_available(),'FAIL_NO_CUDA')
        torch.cuda.set_device(0);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        prop=torch.cuda.get_device_properties(0);free,total=torch.cuda.mem_get_info()
        result.update(GPU_name=prop.name,total_VRAM_bytes=prop.total_memory,free_VRAM_before_bytes=free,
                      torch_version=torch.__version__,cuda_version=torch.version.cuda)
        raw_load=torch.load;load_count=0
        def approved_load(path,*args,**kwargs):
            nonlocal load_count
            require(isinstance(path,(str,Path)) and Path(path).resolve()==checkpoint.resolve() and load_count==0,'FAIL_CHECKPOINT_LOAD_SCOPE')
            require(kwargs.get('map_location')=='cpu','FAIL_CHECKPOINT_LOAD_DEVICE')
            load_count+=1
            return raw_load(path,*args,**kwargs)
        torch.load=approved_load;torch.save=forbidden_operation;torch.jit.save=forbidden_operation
        from ultralytics.nn.tasks import SegmentationModel
        from ultralytics.data.utils import polygons2masks_overlap
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
        pretrained=ckpt.get('ema') or ckpt['model']
        model=SegmentationModel(pretrained.yaml,ch=3,nc=1,verbose=False)
        model.load(pretrained,verbose=False);model.names={0:'Wound'}
        del pretrained,ckpt;gc.collect()
        require(sum(p.numel() for p in model.parameters())==config['architecture']['runtime_parameters'],'FAIL_ARCHITECTURE')
        for name,param in model.named_parameters():param.requires_grad_('.dfl.' not in name)
        model.args=SimpleNamespace(**config['training_args']);model=model.to('cuda:0').train()
        def parameter_hash():
            h=hashlib.sha256()
            for name,param in model.named_parameters():
                h.update(name.encode());h.update(param.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        before=parameter_hash()
        groups=[[],[],[]];norms=tuple(v for k,v in torch.nn.__dict__.items() if 'Norm' in k)
        for module_name,module in model.named_modules():
            for name,param in module.named_parameters(recurse=False):
                index=2 if 'bias' in name else 1 if isinstance(module,norms) else 0
                groups[index].append(param)
        optimizer=torch.optim.AdamW(groups[2],lr=.0005,betas=(.937,.999),weight_decay=0.)
        optimizer.add_param_group({'params':groups[0],'weight_decay':.0005});optimizer.add_param_group({'params':groups[1],'weight_decay':0.})
        optimizer.step=forbidden_operation;torch.optim.AdamW.step=forbidden_operation;torch.optim.Optimizer.step=forbidden_operation
        scaler=torch.amp.GradScaler('cuda',enabled=True);scaler.step=forbidden_operation
        images,polygons=synthetic_source();masks=[];boxes=[]
        for ps in polygons:
            mask,order=polygons2masks_overlap((1024,1024),(ps*1024).reshape(16,-1),downsample_ratio=4)
            masks.append(mask)
            low=ps.min(axis=1);high=ps.max(axis=1)
            boxes.extend(np.concatenate(((low+high)/2,high-low),axis=1)[order])
        image=np.stack([cv2.resize(x,(1024,1024),interpolation=cv2.INTER_LINEAR)[:,:,::-1].transpose(2,0,1) for x in images])
        batch={'img':torch.from_numpy(image).to('cuda').float()/255.,
            'masks':torch.from_numpy(np.stack(masks)).to('cuda').float(),
            'bboxes':torch.tensor(np.asarray(boxes),device='cuda',dtype=torch.float32),
            'cls':torch.zeros(64,1,device='cuda'),'batch_idx':torch.arange(4,device='cuda').repeat_interleave(16).float()}
        result['synthetic_GT_instances']=64;result['target_mask_shape']=list(batch['masks'].shape)
        # One warmup forward+loss, retaining a normal training autograd graph until released.
        with torch.autocast(device_type='cuda',enabled=True):loss,parts=model(batch)
        require(bool(torch.isfinite(loss).all()) and bool(torch.isfinite(parts).all()),'FAIL_NONFINITE_LOSS')
        torch.cuda.synchronize();del loss,parts;model.zero_grad(set_to_none=True)
        result['iterations_completed'].append('warmup_forward_loss')
        # Exactly one fixed stress backward; no step, no epoch, no scale selection.
        with torch.autocast(device_type='cuda',enabled=True):loss,parts=model(batch)
        result['forward_success']=True
        require(bool(torch.isfinite(loss).all()) and bool(torch.isfinite(parts).all()),'FAIL_NONFINITE_LOSS')
        result['GradScaler_scale_before']=scaler.get_scale()
        scaler.scale(loss).backward();torch.cuda.synchronize();result['backward_success']=True
        scaler.unscale_(optimizer)
        finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        result['NaN_Inf']=not finite;result['unscaled_gradients_finite']=finite
        scaler.update();result['GradScaler_scale_after']=scaler.get_scale()
        result['iterations_completed'].append('fixed_stress_forward_loss_backward')
        require(len(optimizer.state)==0,'FAIL_OPTIMIZER_STATE_UPDATED')
        result['parameters_unchanged']=parameter_hash()==before
        require(result['parameters_unchanged'],'FAIL_PARAMETERS_CHANGED')
        result['RESOURCE_FEASIBILITY']='PASS'
        result['NUMERICAL_FEASIBILITY']='PASS' if finite else 'FAIL_NONFINITE_GRADIENTS'
        result['checkpoint_loads']=load_count
        result['update_save_guards_installed']=optimizer.step is forbidden_operation and scaler.step is forbidden_operation and torch.save is forbidden_operation
    except Exception as exc:
        result['error_type']=type(exc).__name__;result['error']=str(exc)
        result['OOM']='out of memory' in str(exc).lower() or 'OutOfMemory' in type(exc).__name__
        result['NaN_Inf']=result['NaN_Inf'] or 'NONFINITE' in str(exc)
        result['RESOURCE_FEASIBILITY']='FAIL'
    finally:
        if torch is not None and torch.cuda.is_initialized():
            result['max_memory_allocated_bytes']=torch.cuda.max_memory_allocated()
            result['max_memory_reserved_bytes']=torch.cuda.max_memory_reserved()
        result['elapsed_seconds']=time.monotonic()-started
        result['project_files_opened_for_read']=sorted(access.project_reads)
        result['denied_file_accesses']=access.denied
        result['research_model_metrics']='NOT_COMPUTED'
        result['limitations']=['No Adam moment state allocation (step forbidden)','Synthetic workload only, no real Mosaic/DataLoader/fragmentation guarantee',
                              'BatchNorm buffers may update in disposable RAM model; no parameters updated and nothing saved']
        save(OUT/'resource_smoke_result.json',result)
        print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':run()
