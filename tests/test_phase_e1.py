"""Synthetic E1 geometry and fail-closed preregistration tests; never model code."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import tempfile
import unittest

from experiments import patch_geometry as g
from experiments.phase_e1 import require_future_absent, FUTURE


def square(x, y, w=10, h=10):
    return [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]


class E1Tests(unittest.TestCase):
    def test_e1_selected_target_fully_retained(self):
        r=g.transform([square(250,250)],0)
        self.assertEqual(r['selected_target_retention'],1.)
        self.assertEqual(r['records'][0]['state'],'fully_retained')

    def test_e1_patch_exact_256_square(self):
        x,y,r,b=g.patch_bounds(square(220.25,100.5))
        self.assertEqual((r-x,b-y),(256,256))

    def test_e1_boundary_clamping(self):
        self.assertEqual(g.patch_bounds(square(0,0)),(0,0,256,256))
        self.assertEqual(g.patch_bounds(square(502,502)),(256,256,512,512))

    def test_e1_polygon_clipping(self):
        r=g.transform([square(250,250),square(370,200,50,50)],0)
        self.assertEqual(r['records'][1]['state'],'partially_clipped')
        self.assertAlmostEqual(r['records'][1]['retention'],13/50)

    def test_e1_polygon_coordinate_translation(self):
        r=g.transform([square(250,250)],0)
        pts=r['records'][0]['polygon']
        self.assertEqual(min(p[0] for p in pts),123/256)

    def test_e1_polygon_normalization(self):
        r=g.transform([square(250,250),square(370,200,50,50)],0)
        self.assertEqual(max(p[0] for p in r['records'][1]['polygon']),1.)
        self.assertTrue(all(0<=v<=1 for x in r['records'] for p in x['polygon'] for v in p))

    def test_e1_invalid_polygon_rejected(self):
        with self.assertRaisesRegex(g.PatchError,'FAIL_PATCH_LABEL_INVALID'):
            g.transform([[(250,250),(260,260),(250,260),(260,250)]],0)

    def test_e1_no_empty_eligible_patch(self):
        for x,y in [(0,0),(502,502),(250,250),(100,0)]:
            r=g.transform([square(x,y)],0)
            self.assertGreaterEqual(r['retained_GT_count'],1)
            self.assertIsNotNone(r['records'][r['target_GT_id']]['polygon'])

    def test_e1_multi_gt_clipping_recorded(self):
        r=g.transform([square(250,250),square(370,200,50,50),square(0,0,60,60)],0)
        self.assertEqual((r['original_GT_count'],r['retained_GT_count'],r['fully_retained_GT_count'],r['partially_clipped_GT_count'],r['dropped_GT_count']),(3,2,1,1,1))

    def test_e1_negative_images_unchanged(self):
        self.assertEqual(g.transform([],0),{'representation':'FULL_IMAGE','polygons':[],'target_GT_id':None})

    def test_e1_noneligible_images_unchanged(self):
        p=[square(100,100,100,100)];before=deepcopy(p)
        self.assertEqual(g.transform(p,0)['polygons'],before)
        self.assertEqual(p,before)

    def test_e1_uniform_sampler_unchanged(self):
        from experiments.paired_sampling_runner import plan_indices
        rows=[{'source':'FUSeg','split':'train','role':'wound_finetuning','instance_geometry':[],
               'sampling_category':'C','sampling_weight':1.} for _ in range(771)]
        ids=[str(i) for i in range(771)]
        for epoch in (0,1,269,270,299):
            self.assertEqual(g.uniform_order(ids,epoch),[ids[int(i)] for i in plan_indices(rows,'C',epoch)])
            self.assertEqual(len(set(g.uniform_order(ids,epoch))),771)

    def test_e1_validation_never_patched(self):
        with self.assertRaisesRegex(g.PatchError,'FAIL_NOT_TRAINING'):g.transform([square(5,5)],0,split='val')

    def test_e1_test_never_patched(self):
        with self.assertRaisesRegex(g.PatchError,'FAIL_NOT_TRAINING'):g.transform([square(5,5)],0,split='test')

    def test_e1_no_validation_outcome_used(self):
        self.assertEqual(set(inspect.signature(g.transform).parameters),{'polygons','epoch','split','already_transformed'})
        row={'source':'FUSeg','split':'train','role':'wound_finetuning','validation_failure':True}
        with self.assertRaisesRegex(g.PatchError,'FAIL_OUTCOME_DEPENDENCY'):g.admit(row)

    def test_e1_deterministic_target_round_robin(self):
        p=[square(100,100),square(200,200),square(50,50,70,70)]
        self.assertEqual([g.transform(p,e)['target_GT_id'] for e in range(5)],[0,1,0,1,0])
        self.assertEqual(g.transform(p,0),g.transform(p,2))

    def test_e1_no_model_load(self):
        for name in ['torch','ultralytics','onnxruntime','tensorflow.keras']:
            with self.assertRaises(g.PatchError):g.NoModels().find_spec(name)

    def test_e1_no_training(self):
        tree=ast.parse(inspect.getsource(g))
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'train','fit','load','load_state_dict'} for n in ast.walk(tree)))

    def test_e1_no_inference(self):
        tree=ast.parse(inspect.getsource(g))
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'predict','forward','infer'} for n in ast.walk(tree)))

    def test_e1_double_patch_rejected(self):
        with self.assertRaisesRegex(g.PatchError,'FAIL_DOUBLE_PATCH'):g.transform([square(20,20)],0,already_transformed=True)

    def test_e1_target_too_wide_fail(self):
        with self.assertRaisesRegex(g.PatchError,'FAIL_PATCH_TARGET_NOT_FULLY_CONTAINED'):g.transform([square(0,100,300,1)],0)

    def test_e1_nan_and_class_rejected(self):
        for s in ['0 nan 0 .1 0 .1 .1','1 .1 .1 .2 .1 .2 .2','0 0 0 0 0 0']:
            with self.assertRaises(g.PatchError):g.parse_labels(s)

    def test_e1_tiny_positive_polygon(self):
        r=g.transform([square(100,100,.001,.001)],0)
        self.assertEqual(r['selected_target_retention'],1)

    def test_e1_zero_area_contact_is_not_visible_area(self):
        r=g.transform([square(250,250),square(383,200,30,30)],0)
        self.assertEqual(r['records'][1]['state'],'dropped')

    def test_e1_frozen_gate_plus_one_is_not_pass(self):
        c={'very_small':27,'small':104,'medium':85,'large':14,'crop_complete':165,'tp':203,'fp':30,'fn':38}
        self.assertFalse(g.advancement(c,c|{'very_small':28})['very_small'])
        self.assertTrue(all(g.advancement(c,c|{'very_small':29}).values()))
        self.assertFalse(g.advancement(c,c|{'very_small':29,'large':13})['large'])

    def test_e1_future_output_must_be_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);require_future_absent(p);(p/FUTURE).mkdir(parents=True)
            with self.assertRaisesRegex(g.PatchError,'FAIL_FUTURE_OUTPUT_ALREADY_EXISTS'):require_future_absent(p)

    def test_e1_no_source_mutation(self):
        p=[square(250,250),square(370,200,50,50)];before=deepcopy(p)
        g.transform(p,0);self.assertEqual(p,before)

    def test_e1_fractional_bounds_preserved(self):
        x,y,_,_=g.patch_bounds(square(250.1,250.3))
        self.assertAlmostEqual(x,127.1);self.assertAlmostEqual(y,127.3)


if __name__=='__main__':unittest.main()
