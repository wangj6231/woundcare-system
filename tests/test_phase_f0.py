"""F0 public protocol and synthetic resource boundaries; no GPU in pytest."""
from copy import deepcopy
from pathlib import Path
import unittest


class F0Tests(unittest.TestCase):
    def pair(self):
        from experiments.phase_f0 import make_pair
        return make_pair()

    def test_f0_no_patch_adapter(self):
        for cfg in self.pair():self.assertEqual(cfg['dataset_adapter'],'STOCK_YOLODataset')

    def test_f0_no_canonical_labels(self):
        for cfg in self.pair():self.assertEqual(cfg['label_path'],'ORIGINAL_STOCK_YOLO_POLYGON')

    def test_f0_same_dataset(self):
        c,h=self.pair();self.assertEqual(c['manifest'],h['manifest'])
        self.assertEqual(c['manifest']['path'],'experiments/protocols/D_SEG_SMALL_SAMPLING_V1_training_manifest.json')

    def test_f0_same_initialization(self):
        c,h=self.pair();self.assertEqual(c['initialization'],h['initialization']);self.assertFalse(c['initialization']['resume'])
        self.assertEqual(c['initialization']['sha256'],'1ec926026473beafafb1ef8da6aa6efcc13d38c391ec8db48006c5d793d5f2a3')

    def test_f0_same_architecture(self):
        c,h=self.pair();self.assertEqual(c['architecture'],h['architecture']);self.assertEqual(c['architecture']['name'],'YOLO11m-seg')

    def test_f0_same_sample_order(self):
        from experiments.phase_f0 import read,PRO,PREFIX
        c,h=self.pair();self.assertEqual(c['sampling'],h['sampling'])
        orders=read(PRO/f'{PREFIX}_anchor_orders.json')['orders'];self.assertEqual(len(orders),300)
        ids=set(orders[0]);self.assertEqual(len(ids),771)
        self.assertTrue(all(len(o)==771 and set(o)==ids for o in orders))

    def test_f0_same_batch(self):
        self.assertEqual([x['training_args']['batch'] for x in self.pair()],[4,4])

    def test_f0_same_optimizer(self):
        c,h=self.pair();self.assertEqual(c['runtime_optimizer'],h['runtime_optimizer'])
        self.assertEqual(c['runtime_optimizer']['betas'],[.937,.999]);self.assertEqual(c['runtime_optimizer']['name'],'AdamW')
        self.assertFalse(c['budget_contract']['gradient_accumulation_compensation'])

    def test_f0_same_loss(self):
        c,h=self.pair();self.assertEqual(c['loss'],h['loss']);self.assertEqual(c['loss']['mask_ratio'],4)
        self.assertEqual([c['loss'][k] for k in ['box','cls','dfl']],[7.5,.5,1.5]);self.assertTrue(c['loss']['overlap_mask'])

    def test_f0_same_augmentation_config(self):
        c,h=self.pair();self.assertEqual(c['augmentation'],h['augmentation'])
        expected={'mosaic':.1,'close_mosaic':30,'mixup':0.,'copy_paste':0.,'degrees':5.,'translate':.05,
                  'scale':.2,'shear':.5,'perspective':.0002,'fliplr':.5,'flipud':0.,'hsv_h':.01,'hsv_s':.4,'hsv_v':.25,'bgr':0.}
        self.assertEqual({k:c['augmentation'][k] for k in expected},expected)

    def test_f0_same_training_budget(self):
        c,h=self.pair();self.assertEqual(c['budget'],h['budget']);self.assertEqual(c['budget_contract'],h['budget_contract'])
        self.assertEqual(c['budget_contract']['scheduled'],3741)
        self.assertEqual([c['training_args'][k] for k in ['epochs','patience','nbs']],[300,80,64])

    def test_f0_control_imgsz_768(self):self.assertEqual(self.pair()[0]['training_args']['imgsz'],768)
    def test_f0_experimental_imgsz_1024(self):self.assertEqual(self.pair()[1]['training_args']['imgsz'],1024)
    def test_f0_eval_both_768(self):
        self.assertEqual([c['evaluation']['imgsz'] for c in self.pair()],[768,768])
        self.assertEqual([c['checkpoint_selection']['both_arms_validation_imgsz'] for c in self.pair()],[768,768])

    def test_f0_resource_smoke_no_optimizer_step(self):
        from experiments.higher_scale_resource_smoke import forbidden_operation
        with self.assertRaisesRegex(RuntimeError,'OPTIMIZER_STEP_OR_CHECKPOINT_SAVE_FORBIDDEN'):forbidden_operation()

    def denied_role(self,source,split):
        from experiments.phase_f0 import admit_training_row
        with self.assertRaisesRegex(ValueError,'FAIL_ORIGINAL_TRAINING_ONLY'):
            admit_training_row({'source':source,'split':split,'role':'wound_finetuning'})
    def test_f0_validation_not_used(self):self.denied_role('FUSeg','val')
    def test_f0_locked_test_rejected(self):self.denied_role('FUSeg','test')
    def test_f0_co2_rejected(self):self.denied_role('CO2Wounds','train')

    def test_f0_future_outputs_absent(self):
        from experiments.phase_f0 import future_absent,FUTURE
        import tempfile
        self.assertTrue(future_absent())
        with tempfile.TemporaryDirectory(prefix='f0-guard-') as tmp:
            (Path(tmp)/FUTURE[0]).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'FAIL_FUTURE_OUTPUT_EXISTS'):future_absent(Path(tmp))

    def test_f0_nonfinite_gradients_do_not_authorize_f1(self):
        from experiments.phase_f0 import completion_status
        result=completion_status({'RESOURCE_FEASIBILITY':'PASS','NaN_Inf':True,'parameters_unchanged':True},tests_pass=True)
        self.assertEqual(result['PHASE_F0_STATUS'],'BLOCKED')
        self.assertEqual(result['READY_FOR_PHASE_F1_PAIRED_HIGHER_SCALE_SEED42'],'NO')

    def test_f0_external_pixels_rejected_even_outside_workspace(self):
        from experiments.higher_scale_resource_smoke import SmokeAccess
        a=SmokeAccess(Path('C:/audit_root'),Path('C:/audit_root/init.pt'),Path('C:/audit_root/out'))
        self.assertFalse(a.allowed_read(Path('C:/external_dataset/wound.png')))

    def test_f0_resource_access_rejects_all_real_data(self):
        from experiments.higher_scale_resource_smoke import SmokeAccess
        root=Path('C:/audit_root');checkpoint=root/'outputs/init/best.pt';out=root/'experiments/results/resource'
        access=SmokeAccess(root,checkpoint,out)
        self.assertTrue(access.allowed_read(checkpoint))
        for path in ['outputs/fuseg/images/train/a.png','outputs/fuseg/images/val/a.png','data/test/a.png','data/CO2/a.png']:
            self.assertFalse(access.allowed_read(root/path))
        self.assertFalse(access.allowed_write(root/'outputs/new.pt'))
        self.assertTrue(access.allowed_write(out/'result.json'))

    def test_f0_resource_smoke_synthetic_only(self):
        from experiments.higher_scale_resource_smoke import synthetic_source
        import numpy as np
        images,polygons=synthetic_source()
        self.assertEqual(images.shape,(4,512,512,3))
        self.assertEqual(polygons.shape,(4,16,4,2))
        self.assertTrue(np.isfinite(polygons).all())
        self.assertTrue(((polygons>0)&(polygons<1)).all())
        np.testing.assert_array_equal(images,synthetic_source()[0])

    def test_f0_original_labels_only(self):
        from experiments.phase_f0 import admit_training_row
        good={'source':'FUSeg','split':'train','role':'wound_finetuning',
              'image_path':'outputs/isic_fuseg_pretrain_smoke_20260914/fuseg_dataset/images/train/fuseg__0011.png',
              'label_path':'outputs/isic_fuseg_pretrain_smoke_20260914/fuseg_dataset/labels/train/fuseg__0011.txt'}
        self.assertTrue(admit_training_row(good))
        for change in ({'split':'val'},{'split':'test'},{'source':'CO2Wounds'},
                       {'label_path':'experiments/protocols/E_PATCH_V2_canonical_labels/x.txt'}):
            with self.assertRaisesRegex(ValueError,'FAIL_ORIGINAL_TRAINING_ONLY'):admit_training_row(dict(good,**change))

    def test_f0_only_imgsz_differs(self):
        from experiments.phase_f0 import make_pair,validate_pair
        c,h=make_pair()
        self.assertEqual(validate_pair(c,h),['arm','experiment_id','output_path','training_args.imgsz'])
        changed=deepcopy(h);changed['training_args']['batch']=2
        with self.assertRaisesRegex(ValueError,'FAIL_NON_SCALE_CONFIG_DIFF'):validate_pair(c,changed)


if __name__=='__main__':unittest.main()
