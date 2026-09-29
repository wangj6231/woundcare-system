"""Public mask-first interfaces; synthetic expected pixels, no network."""
import unittest
import json
import subprocess
import sys
import numpy as np
from experiments import mask_first as m


class MaskFirstTests(unittest.TestCase):
    def test_diagnostic_loader_rejects_validation_test_and_co2_before_read(self):
        from experiments.mask_first_loader import DiagnosticDataset,FetchToken
        for source,split in [('FUSeg','val'),('FUSeg','test'),('CO2Wounds','train')]:
            row={'sample_id':'forbidden','source':source,'split':split,'role':'wound_finetuning'}
            with self.assertRaisesRegex(ValueError,'FAIL_NOT_TRAINING'):
                DiagnosticDataset([row],'P3')[FetchToken('forbidden',0,0)]

    def test_partial_and_dropped_other_instances_keep_identity(self):
        masks=np.zeros((3,512,512),np.uint8)
        masks[0,250:253,250:253]=1;masks[1,120:126,125:128]=1;masks[2,0:5,0:5]=1
        result=m.patch_masks(masks,['selected','partial','dropped'],0)
        self.assertEqual(result['states'],['FULL','PARTIAL','DROPPED'])
        self.assertEqual(result['retained_pixels'],[9,9,0])
        self.assertEqual(result['ids'],['selected','partial','dropped'])

    def test_feasibility_refuses_unverified_baseline_augmentation(self):
        from experiments.phase_e12_finalize import feasibility
        result=feasibility(source_ok=True,retention_ok=True,loader_ok=True,augmentation_parity='NOT_VERIFIED')
        self.assertEqual(result['PHASE_E12_STATUS'],'BLOCKED')
        self.assertEqual(result['PATCH_BASED_TRAINING_ROUTE'],'STOP')
        self.assertEqual(result['READY_FOR_E2_MASK_FIRST_PAIRED_PATCH_TRAINING'],'NO')

    def test_invalid_source_polygon_remains_one_authoritative_instance(self):
        from experiments.phase_e12 import audit_scene
        from experiments.patch_raster import RasterContract
        raster=RasterContract('C:/Python312/Lib/site-packages/ultralytics')
        polygon=np.array([[.4,.4],[.6,.6],[.4,.6],[.6,.4]],np.float32)
        result,masks=audit_scene([polygon],raster)
        self.assertEqual(len(result),1)
        self.assertFalse(result[0]['source_polygon_valid'])
        self.assertGreater(result[0]['mask_area'],0)
        self.assertEqual(result[0]['source_GT_instance_id'],0)
        self.assertTrue(result[0]['deterministic'])

    def test_real_torch_dataloader_worker_prefetch_without_network(self):
        proc=subprocess.run([sys.executable,'-m','experiments.mask_first_loader','--synthetic'],capture_output=True,text=True,encoding='utf8')
        self.assertEqual(proc.returncode,0,proc.stderr)
        result=json.loads(proc.stdout)
        self.assertTrue(result['worker_parity'])
        self.assertEqual(result['batch_image_shape'],[4,3,768,768])
        self.assertEqual(result['batch_mask_shape'],[4,192,192])
        self.assertFalse(result['MODEL_LOADED']);self.assertFalse(result['FORWARD_PASS'])
        self.assertTrue(result['guard_checks_passed']);self.assertFalse(result['CUDA_initialized'])
        for i,target in [(0,1),(1,0)]:
            rows=result['arms']['P3']['batches'][i]['metadata']
            self.assertTrue(all(r['selected_target']==target for r in rows))
            self.assertTrue(all(r['mosaic']==(i==0) for r in rows))
        self.assertTrue(all(r['token']['generation']==1 for r in result['arms']['P3']['reset'][0]['metadata']))

    def test_photometric_transform_changes_image_only(self):
        masks=np.ones((1,8,8),np.uint8);image=np.full((8,8,3),100,np.uint8)
        v=m.hsv_scene(image,masks,['x'],[1.,1.,1.2])
        np.testing.assert_array_equal(v['masks'],masks)
        self.assertTrue(np.all(v['image']==120))

    def test_overlap_encoding_keeps_bbox_class_mask_index_in_same_order(self):
        masks=np.zeros((2,16,16),np.uint8);masks[0,4:8,4:8]=1;masks[1,0:12,0:12]=1
        v=m.format_targets(masks,['small','large'],mask_ratio=4)
        self.assertEqual(v['ids'],['large','small'])
        self.assertEqual(v['masks'][1,1],2)
        np.testing.assert_allclose(v['bboxes'],[[.375,.375,.75,.75],[.375,.375,.25,.25]])
        self.assertEqual(v['cls'].tolist(),[[0.],[0.]])

    def test_four_tile_mosaic_keeps_repeated_source_instances_distinct(self):
        masks=np.zeros((1,8,8),np.uint8);masks[0,1:3,1:3]=1
        image=np.repeat((masks[0]*255)[:,:,None],3,axis=2)
        v=m.mosaic_scene([{'image':image,'masks':masks,'ids':['same:0']} for _ in range(4)],(8,8))
        self.assertEqual(v['bboxes'],[[1,1,3,3],[9,1,11,3],[1,9,3,11],[9,9,11,11]])
        self.assertEqual(len(set(v['ids'])),4)
        np.testing.assert_array_equal(v['image'][:,:,0]==255,v['masks'].any(axis=0))

    def test_horizontal_and_vertical_flip_keep_image_mask_alignment(self):
        masks=np.zeros((1,8,8),np.uint8);masks[0,1:3,1:3]=1
        image=np.repeat((masks[0]*255)[:,:,None],3,axis=2)
        h=m.flip_scene(image,masks,['x'],'horizontal');v=m.flip_scene(image,masks,['x'],'vertical')
        self.assertEqual((h['bboxes'],v['bboxes']),([[5,1,7,3]],[[1,5,3,7]]))
        np.testing.assert_array_equal(h['image'][:,:,0],h['masks'][0]*255)

    def test_fixed_affine_uses_identical_geometry_and_preserves_binary_masks(self):
        mask=np.zeros((1,8,8),np.uint8);mask[0,1:3,1:3]=1
        image=np.repeat((mask[0]*255)[:,:,None],3,axis=2)
        result=m.warp_scene(image,mask,['x'],np.array([[1,0,2],[0,1,1],[0,0,1]],float),8)
        self.assertEqual(result['bboxes'],[[3,2,5,4]])
        np.testing.assert_array_equal(result['image'][:,:,0]==255,result['masks'][0]==1)

    def test_resize_uses_nearest_mask_and_rederives_exclusive_bbox(self):
        mask=np.zeros((1,8,8),np.uint8);mask[0,1:3,1:3]=1
        image=np.repeat((mask[0]*255)[:,:,None],3,axis=2)
        result=m.resize_scene(image,mask,['x'],16)
        self.assertEqual((result['image'].shape,result['masks'].sum(),result['bboxes']),((16,16,3),16,[[2,2,6,6]]))

    def test_selected_mask_and_instance_identity_survive_integer_patch(self):
        masks=np.zeros((2,512,512),np.uint8)
        masks[0,250:253,250:253]=1;masks[1,10:13,10:13]=1
        result=m.patch_masks(masks,['a:0','a:1'],0)
        self.assertEqual((result['bounds'],result['states'],result['ids'],result['masks'].sum()),
                         ([123,123,379,379],['FULL','DROPPED'],['a:0','a:1'],9))


if __name__=='__main__':unittest.main()
