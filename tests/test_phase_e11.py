"""Synthetic source-mask and minimum-equivalent runtime contract tests."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import importlib.metadata as metadata
import inspect
import ast
import sys
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from experiments import patch_raster as r
from experiments import patch_dataset_adapter as a


def square(x,y,w=8,h=None):
    h=w if h is None else h
    return np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],np.float32)/512


class E11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package=Path(metadata.distribution('ultralytics').locate_file('ultralytics'))
        cls.raster=r.RasterContract(cls.package)

    def adapter(self,arm='P2'):return a.PatchAwareDataset(a.synthetic_samples(),arm,self.raster)

    def test_valid_source_identity(self):
        source=square(100,100);record,p=r.canonicalize([source],0,self.raster)
        self.assertEqual(record['canonicalization_status'],'IDENTITY');np.testing.assert_array_equal(source,p)

    def test_invalid_source_candidate_exact(self):
        p=np.array([[4,4],[8,4],[8,8],[4,8],[4,4],[4,8]],np.float32)/512
        result,canonical=r.canonicalize([p],0,self.raster)
        self.assertTrue(result['candidates']);self.assertEqual(result['canonicalization_status'],'EXACT_RASTER_EQUIVALENT')
        self.assertEqual(result['xor_pixels'],0);self.assertEqual(result['raster_IoU'],1.)

    def test_canonical_raster_exact_not_near(self):
        x=np.ones((10,10),np.uint8);y=x.copy();y[0,0]=0
        self.assertEqual(r.compare(x,y)['xor_pixels'],1);self.assertEqual(r.compare(x,y)['raster_IoU'],.99)

    def test_multipart_rejection(self):
        m=np.zeros((20,20),np.uint8);m[1:4,1:4]=1;m[10:13,10:13]=1
        with self.assertRaisesRegex(r.PatchError,'MULTIPART'):r.mask_representable(m)

    def test_hole_rejection(self):
        m=np.zeros((20,20),np.uint8);m[1:18,1:18]=1;m[5:8,5:8]=0
        with self.assertRaisesRegex(r.PatchError,'HOLE'):r.mask_representable(m)

    def test_integer_patch_floor(self):
        self.assertEqual(r.integer_bounds(square(250.25,251.75,8)),(126,127,382,383))

    def test_selected_target_containment(self):
        for p in [square(0,0),square(504,504)]:
            v=r.patch_cycle([p],0,self.raster);self.assertEqual(v['status'],'PASS');self.assertTrue(v['selected_target_mask_retained'])
        with self.assertRaisesRegex(r.PatchError,'INTEGER_PATCH_CONTAINMENT'):r.integer_bounds(square(0,100,300,1))

    def test_raster_crop_vector_equivalence(self):
        v=r.patch_cycle([square(100,100)],0,self.raster)
        self.assertEqual(v['status'],'PASS');self.assertEqual(v['records'][0]['xor_pixels'],0)

    def test_multi_gt_accounting(self):
        v=r.patch_cycle([square(100,100),square(240,100,30),square(400,400,20)],0,self.raster)
        self.assertEqual((v['FULL'],v['PARTIAL'],v['DROPPED']),(1,1,1))
        self.assertTrue(v['selected_retained_other_dropped'])

    def test_negative_unchanged(self):
        adapter=self.adapter();v=adapter.fetch(a.FetchToken('negative',0))
        self.assertEqual(v['patch_count'],0);self.assertEqual(v['image_sha256'],v['source_pixels_sha256'])

    def test_noneligible_unchanged(self):
        v=self.adapter().fetch(a.FetchToken('large',0));self.assertEqual(v['patch_count'],0)
        self.assertEqual(v['canonical_label_hash'],v['transformed_label_hash'])

    def test_control_identity(self):
        v=self.adapter('C2').fetch(a.FetchToken('multi',1));self.assertEqual(v['image'].shape[:2],(512,512))
        self.assertEqual(v['patch_count'],0);self.assertEqual(v['canonical_label_hash'],v['transformed_label_hash'])

    def test_experimental_eligible(self):
        v=self.adapter().fetch(a.FetchToken('multi',1));self.assertEqual(v['image'].shape[:2],(256,256))
        self.assertEqual(v['patch_count'],1);self.assertEqual(v['target_GT_id'],1)

    def test_anchor_exactly_once(self):
        adapter=self.adapter();v=adapter.fetch(a.FetchToken('multi',0))
        self.assertEqual(v['representation_transform_count'],1)
        with self.assertRaisesRegex(r.PatchError,'DOUBLE_PATCH'):a.admit_sample(adapter.samples['multi']|{'already_transformed':True})

    def test_mosaic_companion_exactly_once(self):
        v=a.frozen_mix_fetch(self.adapter(),a.FetchToken('multi',1),['multi'],self.package/'data/augment.py')
        companion=v['companion_evidence'][0]
        self.assertEqual(companion['patch_count'],1);self.assertEqual(v['image_sha256'],companion['image_sha256'])

    def test_same_epoch_deterministic(self):
        adapter=self.adapter();t=a.FetchToken('multi',3)
        one=adapter.fetch(t);two=adapter.fetch(t)
        self.assertEqual(one['image_sha256'],two['image_sha256']);self.assertEqual(one['transformed_label_hash'],two['transformed_label_hash'])

    def test_next_epoch_round_robin(self):
        adapter=self.adapter();self.assertEqual([adapter.fetch(a.FetchToken('multi',e))['target_GT_id'] for e in range(3)],[0,1,0])

    def test_workers2_prefetch_epoch_and_close_mosaic_reset(self):
        result=a.runtime_probe(self.package)
        self.assertEqual(result['status'],'PASS');self.assertTrue(result['generation_reset_stale_results_discarded'])
        self.assertFalse(result['actual_torch_DataLoader_exercised']);self.assertFalse(result['actual_full_Mosaic_geometry_exercised'])

    def test_runtime_probe_isolated_from_parent_model_import_state(self):
        with mock.patch.dict(sys.modules, {'torch': object()}):
            result=a.runtime_probe(self.package)
        self.assertEqual(result['status'],'PASS')

    def test_epoch_token_is_immutable(self):
        t=a.FetchToken('multi',3)
        with self.assertRaises(FrozenInstanceError):t.epoch_token=4

    def test_reset_generation_does_not_change_patch(self):
        adapter=self.adapter();t=a.FetchToken('multi',270)
        self.assertEqual(adapter.fetch(t)['image_sha256'],adapter.fetch(replace(t,generation=1))['image_sha256'])

    def test_validation_never_canonicalized_or_patched(self):
        adapter=self.adapter();adapter.samples['multi']['split']='val'
        with self.assertRaisesRegex(r.PatchError,'TRAINING_ROLE'):adapter.fetch(a.FetchToken('multi',0))

    def test_locked_test_rejected(self):
        with self.assertRaises(r.PatchError):a.admit_sample({'split':'test','source':'FUSeg','sample_id':'locked'})

    def test_co2_rejected(self):
        with self.assertRaises(r.PatchError):a.admit_sample({'split':'train','source':'CO2Wounds','sample_id':'co2'})

    def test_no_model_load(self):
        for module in ['torch','ultralytics','tensorflow','onnxruntime']:
            with self.assertRaises(r.PatchError):r.NoModels().find_spec(module)

    def test_no_training(self):
        for mod in [r,a]:
            tree=ast.parse(inspect.getsource(mod))
            self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'train','fit','load_state_dict'} for n in ast.walk(tree)))

    def test_no_inference(self):
        for mod in [r,a]:
            tree=ast.parse(inspect.getsource(mod))
            self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'predict','forward','infer'} for n in ast.walk(tree)))

    def test_crop_before_resize(self):
        adapter=self.adapter();v=adapter.fetch(a.FetchToken('multi',0));w=adapter.resize_for_existing_pipeline(v)
        self.assertEqual(w['image'].shape[:2],(768,768));self.assertEqual(w['steps'][-1],'EXISTING_LINEAR_RESIZE_768')
        adapter.samples['multi']['pixels']=np.zeros((768,768,3),np.uint8)
        with self.assertRaisesRegex(r.PatchError,'PATCH_AFTER_RESIZE'):adapter.fetch(a.FetchToken('multi',0))

    def test_unresolved_blocked_both_arms(self):
        for arm in ['C2','P2']:
            adapter=self.adapter(arm);adapter.samples['multi']['canonical_polygons'][0]=None
            with self.assertRaisesRegex(r.PatchError,'UNRESOLVED'):adapter.fetch(a.FetchToken('multi',0))

    def test_same_canonical_view_for_both_arms(self):
        c=self.adapter('C2').fetch(a.FetchToken('multi',0));p=self.adapter('P2').fetch(a.FetchToken('multi',0))
        self.assertEqual(c['canonical_label_hash'],p['canonical_label_hash'])

    def test_source_pixels_and_labels_immutable(self):
        adapter=self.adapter();before=deepcopy(adapter.samples['multi']);adapter.fetch(a.FetchToken('multi',0))
        np.testing.assert_array_equal(before['pixels'],adapter.samples['multi']['pixels'])
        for x,y in zip(before['canonical_polygons'],adapter.samples['multi']['canonical_polygons']):np.testing.assert_array_equal(x,y)

    def test_resample_count_boundary(self):
        self.assertEqual(self.raster.resample_count([np.zeros((1000,2))]),1000)
        self.assertEqual(self.raster.resample_count([np.zeros((1001,2))]),1002)


if __name__=='__main__':unittest.main()
