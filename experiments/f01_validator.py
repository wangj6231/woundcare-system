"""Shared stock segmentation-trainer validation-scale override; synthetic dry test."""
from copy import copy


class SharedValidation768:
    """Both arms use the same mixin. Only a shallow proxy's validation args change."""
    def build_dataset(self,img_path,mode='train',batch=None):
        if mode!='val':return super().build_dataset(img_path,mode,batch)
        proxy=copy(self);proxy.args=copy(self.args);proxy.args.imgsz=768
        dataset=super(SharedValidation768,proxy).build_dataset(img_path,mode,batch)
        if dataset.imgsz!=768:raise ValueError('FAIL_VALIDATION_DATASET_SCALE')
        return dataset
    def get_validator(self):
        if self.test_loader.dataset.imgsz!=768:raise ValueError('FAIL_VALIDATION_LOADER_SCALE')
        proxy=copy(self);proxy.args=copy(self.args);proxy.args.imgsz=768
        validator=super(SharedValidation768,proxy).get_validator()
        self.loss_names=proxy.loss_names
        if validator.args.imgsz!=768:raise ValueError('FAIL_VALIDATOR_SCALE')
        return validator


def dry_audit():
    import json,sys,tempfile,os
    from pathlib import Path
    import torch,cv2
    from torch.utils.data import DataLoader
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils import LOGGER
    from experiments.phase_f01 import fixture,fixture_hashes
    from experiments.phase_f0 import ROOT,PRO,read,sha
    LOGGER.setLevel('ERROR');torch.set_num_threads(1);cv2.setNumThreads(1)
    def no_inference(*a,**kw):raise RuntimeError('F01_NO_VALIDATION_INFERENCE_OR_MODEL')
    BaseValidator.__call__=no_inference;torch.nn.Module.__init__=no_inference
    class Trainer(SharedValidation768,SegmentationTrainer):pass
    opened=[];arms=[]
    images,polygons=fixture()
    with tempfile.TemporaryDirectory(prefix='f01-synthetic-validator-') as td:
        root=Path(td);imdir=root/'images';labdir=root/'labels';imdir.mkdir();labdir.mkdir()
        for i,(image,ps) in enumerate(zip(images,polygons)):
            cv2.imwrite(str(imdir/f'synthetic{i}.png'),image)
            # Generated fixture artifact, not a modification of source labels.
            with (labdir/f'synthetic{i}.txt').open('x',encoding='utf8') as f:
                for p in ps:f.write('0 '+' '.join(map(str,p.reshape(-1).tolist()))+'\n')
        allowed={str(p.resolve()) for p in imdir.glob('*.png')}
        def audit(event,args):
            if event=='open' and isinstance(args[0],(str,bytes,Path)):
                p=Path(os.fsdecode(args[0])).resolve()
                if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}:
                    if str(p) not in allowed:raise RuntimeError('F01_REAL_IMAGE_OPEN_FORBIDDEN: '+str(p))
                    opened.append(str(p))
            if event=='socket.connect':raise RuntimeError('F01_NETWORK_FORBIDDEN')
        sys.addaudithook(audit)
        for scale in (768,1024):
            t=Trainer.__new__(Trainer)
            args=dict(read(PRO/'F_HIGHER_SCALE_V1_control_config.json')['training_args'],imgsz=scale)
            t.args=get_cfg(overrides=args);t.model=None;t.data={'names':{0:'Wound'},'nc':1,'channels':3}
            t.save_dir=root/f'validator{scale}';t.callbacks={};t.device=torch.device('cpu')
            train=t.build_dataset(str(imdir),'train',4)
            val=t.build_dataset(str(imdir),'val',4)
            t.test_loader=DataLoader(val,batch_size=4,num_workers=0,collate_fn=val.collate_fn)
            v=t.get_validator();batch=next(iter(t.test_loader));v.device=torch.device('cpu')
            processed=v.preprocess(batch)
            arms.append({'train_imgsz':train.imgsz,'val_imgsz':val.imgsz,'validator_imgsz':v.args.imgsz,
                'trainer_imgsz_after_override':t.args.imgsz,'actual_val_tensor_shape':list(processed['img'].shape),
                'validator_class':type(v).__name__,'shared_override_class':'SharedValidation768',
                'conf':v.args.conf,'NMS_iou':v.args.iou,'matching_IoU_vector':v.iouv.tolist(),
                'batch_image_max_dim':max(processed['img'].shape[-2:]),'rect':val.rect})
        equal=all(arms[0][k]==arms[1][k] for k in ['val_imgsz','validator_imgsz','conf','NMS_iou','matching_IoU_vector','actual_val_tensor_shape','shared_override_class'])
        result={'arms':arms,'BEST_CHECKPOINT_SELECTION_PARITY':equal,
            'synthetic_fixture':fixture_hashes(),'synthetic_image_open_paths':sorted(set(opened)),
            'real_validation_images_loaded':0,'validation_inference':False,'model_constructed':False,
            'CUDA_initialized':torch.cuda.is_initialized(),
            'rect_padding_note':'768 is stock nominal imgsz; rect=True and pad=.5 may produce800x800 for square input. Same in both arms, not a hidden1024 validation.',
            'training_checkpoint_validation_note':'stock validation defaults including conf=.001 and AP matching remain identical; separate frozen final operational evaluation conf=.10/floor.01/match.50 unchanged',
            'frozen_final_evaluation_sha256':sha(PRO/'F_HIGHER_SCALE_V1_evaluation_protocol.json')}
        assert equal and [a['trainer_imgsz_after_override'] for a in arms]==[768,1024]
        print(json.dumps(result))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dry-audit',action='store_true');a=p.parse_args()
    if a.dry_audit:dry_audit()
