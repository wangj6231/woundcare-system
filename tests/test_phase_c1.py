from copy import deepcopy
import builtins
import numpy as np
import pytest

from experiments.phase_c1 import (
    C1Error, bbox_features, classify_fp, crop_failure_type, severity_bin,
    small_bin, pareto, verify_expected_counts, derive, verify_grid_inventory,
)


@pytest.mark.parametrize('fraction,expected', [(0,'0%'),(.25,'>0-25%'),(.5,'>25-50%'),
    (.75,'>50-75%'),(.9,'>75-90%'),(.94999,'>90-95%'),(.95,'>=95%'),(1,'>=95%')])
def test_crop_severity_bins_do_not_reclassify_primary_gate(fraction, expected):
    assert severity_bin(fraction) == expected


@pytest.mark.parametrize('area,expected', [(0.00099,'<0.10%'),(.001,'0.10-0.25%'),
    (.0025,'0.25-0.50%'),(.005,'0.50-0.75%'),(.0075,'0.75-1.00%')])
def test_small_diagnostic_bin_boundaries(area, expected):
    assert small_bin(area) == expected


def test_small_primary_boundary_not_in_diagnostic_bins():
    with pytest.raises(C1Error):
        small_bin(.01)


def row():
    return {'sample_id':'synthetic', 'gt_boxes':[[10,10,20,20]], 'pred_boxes':[[10,10,20,20]],
            'confidences':[.8], 'pairs':[{'gt':0,'prediction':0,'iou':1.0}], 'tp':1,'fp':0,'fn':0,
            'num_gt_instances':1,'num_predictions':1,'gt_pixels':100,'crop':[8,8,22,22],
            'crop_complete95':True,'crop_coverage':1.0,'per_instance_gt':[{'index':0,'size_group':'small','matched':True,'bbox':[10,10,20,20]}],
            'size_support':{'small':1,'medium':0,'large':0},'size_matched':{'small':1,'medium':0,'large':0}}


def test_fp_on_negative_image_is_distinct_from_positive_region():
    r=row(); r['gt_boxes']=[]; r['pairs']=[]; r['gt_pixels']=0
    assert classify_fp(r,0)['primary_failure_type']=='FP_NEGATIVE_IMAGE'


def test_duplicate_requires_overlap_at_frozen_match_iou_with_matched_gt():
    r=row(); r['pred_boxes'].append([10,10,20,20]); r['confidences'].append(.2)
    assert classify_fp(r,1)['primary_failure_type']=='FP_DUPLICATE_NEAR_GT'


def test_low_iou_is_localization_mismatch_not_duplicate_or_background():
    r=row(); r['pred_boxes']=[[18,10,28,20]]; r['pairs']=[]
    answer=classify_fp(r,0)
    assert answer['primary_failure_type']=='FP_LOCALIZATION_MISMATCH'
    assert answer['geometric_tag']=='FP_NEAR_GT_LOW_IOU'


def test_zero_overlap_on_positive_image_is_not_declared_background():
    r=row(); r['pred_boxes']=[[30,30,40,40]]; r['pairs']=[]
    assert classify_fp(r,0)['primary_failure_type']=='FP_ON_POSITIVE_IMAGE_OTHER_REGION'


def test_crop_failure_multilabel_priority_and_outside_gt_evidence():
    r=row(); r['gt_boxes'].append([40,40,50,50]); r['num_gt_instances']=2
    r['fn']=1; r['crop_complete95']=False; r['crop_coverage']=.5
    result=crop_failure_type(r)
    assert result['primary_failure_type']=='MISSED_SECOND_INSTANCE'
    assert 'PARTIAL_GT_COVERAGE' in result['secondary_failure_types']


def test_no_roi_primary_is_exclusive():
    r=row(); r['crop']=None; r['crop_coverage']=0; r['crop_complete95']=False
    assert crop_failure_type(r)['primary_failure_type']=='NO_ROI'


def test_geometry_preserves_bbox_area_and_border_definition():
    f=bbox_features([0,128,16,144])
    assert f['bbox_area_pixels']==256
    assert f['bbox_area_ratio']==256/(512*512)
    assert f['nearest_border_distance_normalized']==0


def test_pareto_primary_categories_sum_to_correct_denominator():
    table=pareto(['a','a','b'])
    assert sum(r['count'] for r in table)==3
    assert table[-1]['cumulative_percentage']==100


def test_phase_c1_counts_fail_closed():
    counts={'TP':209,'FP':36,'FN':32,'GT':241,'small':137,'medium':90,'large':14,
            'positive':186,'negative':5,'crop_pass':167,'crop_fail':19,'no_roi_positive':4}
    verify_expected_counts(counts)
    counts['FN']=31
    with pytest.raises(C1Error,match='FAIL_PHASE_C1_INPUT_INTEGRITY'):
        verify_expected_counts(counts)


def test_analysis_preserves_saved_rows_and_never_imports_model(monkeypatch):
    r=row(); before=deepcopy(r)
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in {'torch','ultralytics'}:
            raise AssertionError('model framework import prohibited')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    analysis=derive([r])
    assert r==before
    assert analysis['counts']['TP']==1
    assert analysis['new_inference_performed'] is False


def test_grid_inventory_requires_every_instance_without_duplicate_substitution():
    a={'FN_instances':[{'sample_id':'x','gt_instance_id':0,'size_group':'small'}],
       'review_sheet':[],'FP_predictions':[],'crop_failures':[],
       'visualization_inventory':[
           {'file':'small_wound_misses_01.png','sample_ids':['x'],'instance_ids':[0]},
           {'file':'all_FN_instances_01.png','sample_ids':['x'],'instance_ids':[0]}]}
    assert verify_grid_inventory(a)['all_FN_instances']==1
    a['visualization_inventory'][1]['instance_ids']=[1]
    with pytest.raises(C1Error,match='GRID_COVERAGE_MISMATCH'):
        verify_grid_inventory(a)


def test_grid_inventory_rejects_missing_page():
    a={'FN_instances':[],'review_sheet':[],'FP_predictions':[],
       'crop_failures':[{'sample_id':'x'}],'visualization_inventory':[]}
    with pytest.raises(C1Error,match='GRID_COVERAGE_MISMATCH'):
        verify_grid_inventory(a)
