"""Synthetic E0 tests: no model, dataset, training or inference."""
from copy import deepcopy
import numpy as np
import pytest
from experiments import phase_e0 as e

@pytest.mark.parametrize('c,s,expected',[(0,0,'PERSISTENTLY_MISSED'),(0,1,'PERSISTENTLY_MISSED'),
 (1,0,'PERSISTENTLY_MISSED'),(5,5,'CONSISTENTLY_DETECTED'),(1,2,'SAMPLING_RESPONSIVE'),
 (3,1,'SAMPLING_HARMED'),(1,1,'SEED_UNSTABLE'),(4,4,'SEED_UNSTABLE')])
def test_categories(c,s,expected):assert e.classify(c,s)==expected

def test_all_patterns_exhaustive_and_net_effect_retained():
    for c in range(6):
        for s in range(6):assert e.classify(c,s) in e.CATEGORIES
    p=e.pattern([0]*5,[0,1,0,0,0])
    assert p['category']=='PERSISTENTLY_MISSED' and p['S_only_detected']==1 and p['net_detection_count_delta']==1

def test_cancelled_gains_losses_not_hidden():
    p=e.pattern([1,0,0,1,0],[0,1,1,0,0])
    assert p['category']=='SEED_UNSTABLE' and p['paired_gain_seeds']==p['paired_loss_seeds']==2 and p['mixed_gain_and_loss']==1

@pytest.mark.parametrize('c,s',[(-1,0),(6,1),(1.0,1),(True,0)])
def test_bad_count(c,s):
    with pytest.raises(RuntimeError):e.classify(c,s)

def row():
    return {'gt_boxes':[[0,0,10,10]],'pairs':[],'pred_boxes':[],'floor_bboxes':[], 'floor_confidences':[]}

def test_miss_diagnostic_fixed_thresholds():
    r=row();assert e.miss_diagnostic(r,0)=='NO_RETAINED_OVERLAP'
    r.update(floor_bboxes=[[0,0,10,10]],floor_confidences=[.05])
    assert e.miss_diagnostic(r,0)=='BELOW_FROZEN_CONFIDENCE'
    r['pred_boxes']=[[0,0,10,10]]
    assert e.miss_diagnostic(r,0)=='MATCH_COMPETITION'
    r['pairs']=[{'gt':0,'prediction':0,'iou':1.}]
    assert e.miss_diagnostic(r,0)=='DETECTED'

def test_local_features_constant_image_and_ring_exclusion():
    im=np.full((512,512,3),100,np.uint8)
    f=e.local_features(im,[20,20,24,24],[[20,20,24,24]])
    assert f['local_bbox_background_abs_mean_difference']==0 and f['local_bbox_background_CNR'] is None
    assert f['sharpness_laplacian_variance']==0 and f['ring_pixel_count']==20*20-4*4
    f=e.local_features(im,[20,20,24,24],[[0,0,512,512]])
    assert f['ring_pixel_count']==0 and f['local_bbox_background_abs_mean_difference'] is None

def test_forbidden_imports():
    guard=e.NoModels()
    for name in ('torch','ultralytics.engine','onnxruntime','tensorflow.keras'):
        with pytest.raises(RuntimeError,match='FORBIDDEN_MODEL_IMPORT'):guard.find_spec(name)
    assert guard.find_spec('numpy') is None

def test_cross_model_gt_mismatch_rejected():
    with pytest.raises(RuntimeError,match='GT identity'):e.verify_row(row(),[[1,1,10,10]],None)

def test_csv_roundtrip_and_no_overwrite(tmp_path):
    path=tmp_path/'matrix.csv';e.table(path,[{'sample_id':'a','n':0},{'sample_id':'b','n':1}])
    with pytest.raises(FileExistsError):e.table(path,[{'sample_id':'changed','n':2}])

def test_very_small_boundary():
    # Area ratios exactly .001 and .0025: former second bin; latter excluded.
    pred={k:{'x':{'gt_boxes':[[0,0,16,16.384],[0,0,32,20.48]],'pairs':[],
                   'pred_boxes':[],'floor_bboxes':[],'floor_confidences':[]}} for k in e.MODELS}
    m,diag=e.derive(pred)
    assert len(m)==1 and m[0]['size_bin']=='0.10-0.25%' and len(diag)==10
