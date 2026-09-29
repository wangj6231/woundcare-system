"""Synthetic-only tests for the saved-output F1.3 audit."""
import pytest
from experiments import phase_f13 as a


def row(boxes, scores, gt, pairs, floor=None, fs=None):
    return dict(sample_id='synthetic',pred_boxes=boxes,confidences=scores,gt_boxes=gt,pairs=pairs,
                floor_bboxes=boxes if floor is None else floor,floor_confidences=scores if fs is None else fs,
                retained_mask_indices=list(range(len(boxes))))


@pytest.mark.parametrize('x,expected',[(0,a.BINS[0]),(.001,a.BINS[1]),(.0025,a.BINS[2]),(.005,a.BINS[3]),(.0075,a.BINS[4]),(.01,a.BINS[5]),(.05,a.BINS[6])])
def test_bins(x,expected):
    assert a.size_bin(x)==expected


@pytest.mark.parametrize('c,h,t',[(True,True,'BOTH_DETECTED'),(True,False,'C4_ONLY'),(False,True,'H4_ONLY'),(False,False,'BOTH_MISSED')])
def test_transitions(c,h,t):
    assert a.transition(c,h)==t


def test_miss_order_competition_before_floor():
    box=[0,0,10,10]
    r=row([box],[.8],[box,box],[dict(prediction=0,gt=0,iou=1)],floor=[box,box],fs=[.8,.05])
    assert a.miss(r,1)['diagnostic']=='MATCH_COMPETITION'


def test_below_before_localization():
    r=row([[0,0,4,4]],[.8],[[0,0,10,10]],[],floor=[[0,0,4,4],[0,0,10,10]],fs=[.8,.05])
    assert a.miss(r,0)['diagnostic']=='BELOW_FROZEN_CONFIDENCE'
    r['floor_confidences'][1]=.009
    assert a.miss(r,0)['diagnostic']=='RETAINED_LOCALIZATION_BELOW_IOU'


def test_empty_miss_is_not_zero_confidence():
    d=a.miss(row([],[],[[0,0,10,10]],[]),0)
    assert d['diagnostic']=='NO_RETAINED_OVERLAP'
    assert d['best_saved_candidate_confidence'] is None


@pytest.mark.parametrize('gt,box,cat',[
    ([],[0,0,10,10],a.FP_CATS[0]),
    ([[0,0,10,10]],[0,0,10,10],a.FP_CATS[1]),
    ([[0,0,10,10]],[0,0,5,5],a.FP_CATS[2]),
    ([[0,0,10,10]],[20,20,30,30],a.FP_CATS[3])])
def test_fp_categories(gt,box,cat):
    assert a.fp_record('C4',row([box],[.2],gt,[]),0)['category']==cat


def test_pairing_deterministic_confidence_and_index():
    cs=[a.fp_record('C4',row([[0,0,10,10]]*2,[.8,.8],[],[]),i) for i in range(2)]
    hs=[a.fp_record('H4',row([[0,0,10,10]],[.9],[],[]),0)]
    result=a.pair_fp(cs,hs)
    assert result[0]['C4_prediction_index']==0
    assert result[1]['pairing']=='C4_ONLY_FP'
    assert a.pair_fp(list(reversed(cs)),hs)==result


def test_frozen_matching_is_confidence_first_not_iou_first():
    r=row([[0,0,8,10],[0,0,10,10]],[.9,.8],[[0,0,10,10]],[])
    assert a.frozen_pairs(r)==[dict(prediction=0,gt=0,iou=.8)]


def test_no_authoritative_instance_masks_and_no_model_imports():
    import ast
    tree=ast.parse(a.Path(a.__file__).read_text(encoding='utf-8'))
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [alias.name for n in ast.walk(tree) if isinstance(n,ast.Import) for alias in n.names]
    assert not any(n and n.split('.')[0] in {'torch','ultralytics','tensorflow','onnxruntime'} for n in imports)


def test_stats_missing_and_single():
    assert a.stats([None])['mean'] is None
    assert a.stats([.5])['q25']==.5


def test_distance_and_iou_boundary():
    assert a.distance([0,0,10,10],[13,14,20,20])==5
    assert a.iou([0,0,10,10],[10,0,20,10])==0


def test_frozen_rgb_mask_normalization():
    arr=a.np.zeros((512,512,3),dtype=a.np.uint8)
    arr[0,0,:]=32
    assert a.binary_mask(arr).sum()==1
    arr[0,0,1]=0
    with pytest.raises(ValueError,match='RGB_CHANNEL'):
        a.binary_mask(arr)
