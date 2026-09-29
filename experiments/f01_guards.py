"""Behavioral guards for synthetic numerical preflight, with process-local counters."""
from pathlib import Path
import os
import sys

class ForbiddenWriteError(RuntimeError):pass
class ForbiddenOptimizerStep(RuntimeError):pass
class ForbiddenScalerStep(RuntimeError):pass
class ForbiddenImageAccess(RuntimeError):pass


class Guards:
    def __init__(self,root,checkpoint,out):
        self.root=Path(root).resolve();self.checkpoint=Path(checkpoint).resolve();self.out=Path(out).resolve()
        self.counts={'save_attempts':0,'checkpoint_write_attempts':0,'optimizer_step_attempts':0,'scaler_step_attempts':0,'image_access_attempts':0}
        self.image_accesses=[];self.project_reads=set()
    def save(self,*a,**kw):
        self.counts['save_attempts']+=1;raise ForbiddenWriteError('F01_SAVE_FORBIDDEN')
    def optimizer_step(self,*a,**kw):
        self.counts['optimizer_step_attempts']+=1;raise ForbiddenOptimizerStep('F01_OPTIMIZER_STEP_FORBIDDEN')
    def scaler_step(self,*a,**kw):
        self.counts['scaler_step_attempts']+=1;raise ForbiddenScalerStep('F01_SCALER_STEP_FORBIDDEN')
    def image(self,path=None,*a,**kw):
        self.counts['image_access_attempts']+=1
        self.image_accesses.append(str(path) if isinstance(path,(str,Path)) else '<memory-decode-buffer>')
        raise ForbiddenImageAccess('F01_NO_IMAGE_OPEN_OR_DECODE')
    def audit(self,event,args):
        if event=='socket.connect':raise RuntimeError('F01_NETWORK_FORBIDDEN')
        if event!='open' or not isinstance(args[0],(str,bytes,Path)):return
        p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR)))
        if writing and (p.suffix.lower() in {'.pt','.pth','.ckpt','.safetensors'} or 'weights' in p.parts):
            self.counts['checkpoint_write_attempts']+=1;raise ForbiddenWriteError('F01_CHECKPOINT_WRITE_FORBIDDEN')
        if not writing and p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}:self.image(p)
        if not writing and p.suffix.lower() in {'.pt','.pth','.ckpt','.safetensors'} and p!=self.checkpoint:
            raise RuntimeError('F01_OTHER_CHECKPOINT_READ_FORBIDDEN')
        if p.is_relative_to(self.root):
            if writing and not p.is_relative_to(self.out):raise ForbiddenWriteError('F01_PROJECT_WRITE_FORBIDDEN')
            if not writing:
                allowed=(p==self.checkpoint or p.is_relative_to(self.out) or
                    p.is_relative_to(self.root/'experiments/protocols') and p.suffix=='.json' or
                    p.is_relative_to(self.root/'experiments') and p.suffix in {'.py','.pyc'} or
                    p.is_relative_to(self.root/'experiments/results/f_higher_scale_v1_resource_audit') and p.suffix in {'.py','.json','.lock'})
                if not allowed:raise RuntimeError('F01_RESEARCH_FILE_READ_FORBIDDEN: '+str(p))
                self.project_reads.add(str(p))
    def install(self,torch,cv2):
        from PIL import Image
        # Install AFTER Ultralytics has applied its wrappers. Test behavior, not identity.
        torch.save=self.save;torch.jit.save=self.save
        cv2.imread=self.image;cv2.imdecode=self.image;Image.open=self.image
        sys.dont_write_bytecode=True;sys.addaudithook(self.audit)
    def bind(self,optimizer,scaler):
        optimizer.step=self.optimizer_step;scaler.step=self.scaler_step
    def evidence(self):return dict(self.counts,image_open_decode_paths=self.image_accesses,project_reads=sorted(self.project_reads))


def self_test():
    import tempfile,json
    import torch,cv2
    import ultralytics  # wrapper installation precedes sentinel
    from experiments.phase_f0 import ROOT
    with tempfile.TemporaryDirectory(prefix='f01-guard-') as td:
        p=Path(td)/'fake.pt';g=Guards(ROOT,Path(td)/'init.pt',Path(td))
        tensor=torch.tensor([1.],requires_grad=True);optimizer=torch.optim.AdamW([tensor],lr=.0005)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        g.install(torch,cv2);g.bind(optimizer,scaler);result={}
        for name,action,exception in [
            ('save',lambda:torch.save({'synthetic':1},p),ForbiddenWriteError),
            ('write',lambda:p.write_bytes(b'fake'),ForbiddenWriteError),
            ('optimizer',lambda:optimizer.step(),ForbiddenOptimizerStep),
            ('scaler',lambda:scaler.step(optimizer),ForbiddenScalerStep),
            ('decode',lambda:cv2.imread(str(Path(td)/'FUSeg_val.png')),ForbiddenImageAccess)]:
            try:action();result[name+'_raised']=False
            except exception:result[name+'_raised']=True
        result.update(file_exists=p.exists(),CUDA_initialized=torch.cuda.is_initialized(),counters=g.evidence())
        assert all(result[x+'_raised'] for x in ['save','write','optimizer','scaler','decode']) and not p.exists()
        print(json.dumps(result))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
