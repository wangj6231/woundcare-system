"""Disposable-process, eight-attempt AMP backoff diagnostic. No parameter steps."""
import numpy as np


def gradient_stats(named_arrays):
    total=finite=nan_count=inf_count=0;bad=[];maximum=0.
    for name,a in named_arrays:
        total+=1;mask=np.isfinite(a);isnan=bool(np.isnan(a).any());isinf=bool(np.isinf(a).any())
        nan_count+=isnan;inf_count+=isinf
        if mask.all():finite+=1
        else:bad.append(name)
        if mask.any():maximum=max(maximum,float(np.abs(a[mask]).max()))
    return {'gradient_tensor_count':total,'finite_gradient_tensor_count':finite,
        'nonfinite_gradient_tensor_count':total-finite,'nan_gradient_tensor_count':nan_count,
        'inf_gradient_tensor_count':inf_count,'nonfinite_parameter_names':bad,'max_abs_finite_gradient':maximum}


def run(arm):
    import os,sys,json,hashlib,gc,time
    from datetime import datetime,timezone
    from types import SimpleNamespace
    from pathlib import Path
    from experiments.phase_f01 import OUT,PRO,PREFIX,ROOT,read,save,sha,require,fixture,fixture_hashes,classify,INIT_SHA
    from experiments.f01_guards import Guards
    require(arm in ('N768','N1024'),'FAIL_ARM');size=int(arm[1:])
    protocol=read(PRO/f'{PREFIX}_protocol.json');expected=read(PRO/f'{PREFIX}_fixture.json')
    seal=read(OUT/'pre_execution_freeze.json')
    for path,h in seal['artifacts_sha256'].items():require(sha(ROOT/path)==h,'FAIL_F01_PREFREEZE_CHANGED')
    require(protocol['MAX_ATTEMPTS']==8 and protocol['initial_scale']==65536 and protocol['batch']==4,'FAIL_FROZEN_NUMERICAL_CONTRACT')
    out=OUT/arm;out.mkdir(exist_ok=False)
    config=read(PRO/'F_HIGHER_SCALE_V1_control_config.json');checkpoint=ROOT/config['initialization']['path']
    require(sha(checkpoint)==INIT_SHA,'FAIL_INIT')
    save(out/'execution.lock',{'pid':os.getpid(),'arm':arm,'created_at':datetime.now(timezone.utc).isoformat(),
        'code_sha256':sha(Path(__file__)),'initialization_sha256':sha(checkpoint),'fresh_process':True,'no_retry':True})
    guards=Guards(ROOT,checkpoint,OUT);rows=[];torch=None;start=time.monotonic()
    result={'arm':arm,'pid':os.getpid(),'imgsz':size,'batch':4,'initial_scale':65536,
        'OOM':False,'fresh_model':True,'fresh_optimizer':True,'fresh_GradScaler':True,
        'initialization_sha256':sha(checkpoint),'parameter_update_count':0,'research_training':False,'research_data_inference':False}
    try:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['YOLO_AUTOINSTALL']='false'
        import torch,cv2
        from ultralytics.nn.tasks import SegmentationModel
        from ultralytics.data.utils import polygons2masks_overlap
        torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(42);np.random.seed(42)
        torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        guards.install(torch,cv2)
        hashes=fixture_hashes();require(hashes==expected,'FAIL_SYNTHETIC_FIXTURE_IDENTITY');result['fixture']=hashes
        require(torch.cuda.is_available(),'FAIL_NO_CUDA');torch.cuda.set_device(0);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        prop=torch.cuda.get_device_properties(0);result.update(GPU_name=prop.name,total_VRAM_bytes=prop.total_memory)
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False);pretrained=ckpt.get('ema') or ckpt['model']
        model=SegmentationModel(pretrained.yaml,ch=3,nc=1,verbose=False);model.load(pretrained,verbose=False);model.names={0:'Wound'}
        del pretrained,ckpt;gc.collect()
        require(sum(p.numel() for p in model.parameters())==22359987,'FAIL_ARCHITECTURE')
        for name,p in model.named_parameters():p.requires_grad_('.dfl.' not in name)
        model.args=SimpleNamespace(**dict(config['training_args'],imgsz=size));model=model.to('cuda').train()
        def parameter_hash():
            h=hashlib.sha256()
            for name,p in model.named_parameters():
                if p.requires_grad:h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        before=parameter_hash();result['initial_trainable_parameter_sha256']=before
        groups=[[],[],[]];norms=tuple(v for k,v in torch.nn.__dict__.items() if 'Norm' in k)
        for name,module in model.named_modules():
            for p_name,p in module.named_parameters(recurse=False):groups[2 if 'bias' in p_name else 1 if isinstance(module,norms) else 0].append(p)
        optimizer=torch.optim.AdamW(groups[2],lr=.0005,betas=(.937,.999),weight_decay=0.)
        optimizer.add_param_group({'params':groups[0],'weight_decay':.0005});optimizer.add_param_group({'params':groups[1],'weight_decay':0.})
        scaler=torch.amp.GradScaler('cuda',enabled=True);require(scaler.get_scale()==65536,'FAIL_STOCK_INITIAL_SCALE')
        guards.bind(optimizer,scaler)
        images,polygons=fixture();masks=[];boxes=[]
        for ps in polygons:
            mask,order=polygons2masks_overlap((size,size),(ps*size).reshape(16,-1),downsample_ratio=4);masks.append(mask)
            lo=ps.min(axis=1);hi=ps.max(axis=1);boxes.extend(np.concatenate(((lo+hi)/2,hi-lo),axis=1)[order])
        inputs=np.stack([cv2.resize(x,(size,size),interpolation=cv2.INTER_LINEAR)[:,:,::-1].transpose(2,0,1) for x in images])
        batch={'img':torch.from_numpy(inputs).to('cuda').float()/255.,'masks':torch.from_numpy(np.stack(masks)).to('cuda').float(),
               'bboxes':torch.tensor(np.asarray(boxes),device='cuda',dtype=torch.float32),
               'cls':torch.zeros(64,1,device='cuda'),'batch_idx':torch.arange(4,device='cuda').repeat_interleave(16).float()}
        result['target_mask_shape']=list(batch['masks'].shape)
        for attempt in range(1,9):
            optimizer.zero_grad(set_to_none=True)
            scale_before=scaler.get_scale()
            with torch.autocast(device_type='cuda',enabled=True):loss,parts=model(batch)
            loss_finite=bool(torch.isfinite(loss).all() and torch.isfinite(parts).all())
            require(loss_finite,'FAIL_NONFINITE_LOSS')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.cuda.synchronize()
            stats=gradient_stats((name,p.grad.detach().cpu().numpy()) for name,p in model.named_parameters() if p.grad is not None)
            scaler.update();after=parameter_hash()
            row=dict(stats,arm=arm,attempt=attempt,imgsz=size,batch=4,loss_finite=loss_finite,
                scaler_before=scale_before,scaler_after=scaler.get_scale(),parameters_unchanged=before==after,
                trainable_parameter_sha256=after,guard_counts=dict(guards.counts))
            rows.append(row);save(out/f'attempt_{attempt:02d}.json',row)
            require(before==after,'FAIL_PARAMETER_UPDATE');require(not any(guards.counts.values()),'FAIL_EXECUTION_GUARD')
            require(not optimizer.state,'FAIL_OPTIMIZER_STATE_UPDATE')
            del loss,parts
        result.update(classify(rows),completed_attempts=len(rows),parameters_unchanged=all(r['parameters_unchanged'] for r in rows))
    except Exception as e:
        result.update(error_type=type(e).__name__,error=str(e),completed_attempts=len(rows),
            classification='INCOMPLETE_NUMERICAL_PREFLIGHT',NUMERICALLY_STABLE=False)
        result['OOM']='out of memory' in str(e).lower() or 'OutOfMemory' in type(e).__name__
    finally:
        if torch is not None and torch.cuda.is_initialized():
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        result.update(guards=guards.evidence(),elapsed_seconds=time.monotonic()-start,
            attempts=rows,synthetic_only=True,test_images_used=0,CO2Wounds_used=False,checkpoint_saved=False,
            limitations='No optimizer step; Adam moments not allocated; BN buffers may change; not a300-epoch training guarantee')
        save(out/'summary.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('attempts','fixture','guards')},ensure_ascii=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['N768','N1024']);a=p.parse_args();run(a.arm)
