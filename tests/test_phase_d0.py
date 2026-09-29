from copy import deepcopy
import ast
import inspect
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from experiments import phase_d0 as d0


def row(area=.001,name='synthetic'):
    c=d0.category([area])
    return {'sample_id':name,'source':'FUSeg','split':'train','role':'wound_finetuning',
      'image_hash':name,'label_hash':'label-'+name,'GT_count':1,'multi_GT':False,
      'instance_geometry':[{'class_id':0,'bbox_area_ratio':area,'size_group':'small' if area<.01 else 'medium'}],
      'sampling_category':c,'sampling_weight':d0.WEIGHTS[c],'exact_content_group':name}


def test_d0_sampler_uses_training_gt_only():
    a=row();a['human_failure_tag']='UNKNOWN';a['validation_FN']=999
    b=deepcopy(a);b['validation_FN']=0
    assert np.array_equal(d0.epoch_indices([a],0),d0.epoch_indices([b],0))
    assert d0.category([.0001,.04])=='A'


def test_d0_sampler_never_reads_validation_outcomes(monkeypatch):
    from builtins import open as real_open
    def forbidden(*args,**kwargs):raise AssertionError('sampler must do no file reads')
    monkeypatch.setattr('builtins.open',forbidden)
    monkeypatch.setattr(Path,'open',forbidden)
    d0.simulate([row(.001,'a'),row(.005,'b'),row(.02,'c')],epochs=5)


@pytest.mark.parametrize('field,value',[('split','val'),('split','test'),('source','CO2Wounds-V2'),('role','HISTORICAL_EXTERNAL')])
def test_d0_validation_not_sampled(field,value):
    a=row();a[field]=value
    with pytest.raises(d0.D0Error):d0.epoch_indices([a],0)


def test_d0_locked_test_rejected():
    a={'source':'FUSeg','role':'wound_finetuning','split':'train','image':'fuseg_dataset/images/train/../locked_test/x.png','label':'fuseg_dataset/labels/train/x.txt'}
    with pytest.raises(d0.D0Error):d0.require_train(a)


def test_d0_co2_rejected():
    a={'source':'FUSeg','role':'wound_finetuning','split':'train','image':'fuseg_dataset/images/train/co2wounds.png','label':'fuseg_dataset/labels/train/x.txt'}
    with pytest.raises(d0.D0Error):d0.require_train(a)


@pytest.mark.parametrize('field',['sample_id','image_hash'])
def test_d0_exclusion_checks_identity_and_hash(field):
    a=row();v={'sample_id':'other','image_sha256':'other'}
    v['sample_id' if field=='sample_id' else 'image_sha256']=a[field]
    with pytest.raises(d0.D0Error):d0.validate_exclusion([a],[v])


@pytest.mark.parametrize('field',d0.FAIR_FIELDS,ids=['same_architecture','same_initialization','same_dataset','same_training_args','same_optimizer','same_loss','same_augmentation','same_training_budget','same_evaluation_protocol'])
def test_d0_same_registered_setting(field):
    control={k:{'v':1} for k in d0.FAIR_FIELDS};experimental=deepcopy(control)
    d0.compare_configs(control,experimental)
    experimental[field]['v']=2
    with pytest.raises(d0.D0Error,match='FAIRNESS_MISMATCH'):d0.compare_configs(control,experimental)


def test_d0_sampling_weights_frozen():
    a=row();a['sampling_weight']=2.1
    with pytest.raises(d0.D0Error,match='SAMPLING_WEIGHTS_NOT_FROZEN'):d0.probabilities([a])


def test_d0_expected_sampling_distribution():
    rows=[row(.001,'a'),row(.005,'b'),row(.02,'c')]
    np.testing.assert_allclose(d0.probabilities(rows),[2/4.5,1.5/4.5,1/4.5])
    sim=d0.simulate(rows)
    assert sim['index_draws']==900
    assert sum(x['simulated_draws'] for x in sim['category_exposure'].values())==900
    assert sim['category_exposure']['A']['expected_proportion']==pytest.approx(2/4.5)


def test_d0_output_directory_absent(tmp_path):
    d0.output_absent(tmp_path)
    (tmp_path/d0.OUTPUT).mkdir(parents=True)
    with pytest.raises(d0.D0Error,match='D1_OUTPUT_ALREADY_EXISTS'):d0.output_absent(tmp_path)


def test_d0_no_model_load():
    code="import sys; import experiments.phase_d0; assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],check=True,cwd=Path(d0.__file__).resolve().parents[1],capture_output=True)


@pytest.mark.parametrize('operation',['train','predict','val','load_state_dict'])
def test_d0_no_training_or_inference(operation):
    tree=ast.parse(inspect.getsource(d0))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr==operation for n in ast.walk(tree))


@pytest.mark.parametrize('area,expected',[(.0024999,'A'),(.0025,'B'),(.0099999,'B'),(.01,'C')])
def test_d0_size_boundary(area,expected):
    assert d0.category([area])==expected


def test_d0_empty_labels_remain_in_category_c():
    assert d0.polygon_geometry('')==[]
    assert d0.category([])=='C'


@pytest.mark.parametrize('label',['1 0 0 0.1 0 0.1 0.1','0 0 0 nan 0 .1 .1','0 0 0 2 0 2 1','0 0 0 0 0 0 0'])
def test_d0_invalid_polygons_rejected(label):
    with pytest.raises(d0.D0Error):d0.polygon_geometry(label)


def test_d0_sampling_does_not_mutate_rows_and_is_epoch_reproducible():
    rows=[row(.001,'a'),row(.005,'b'),row(.02,'c')];before=deepcopy(rows)
    first=d0.epoch_indices(rows,19);d0.epoch_indices(rows,1)
    np.testing.assert_array_equal(first,d0.epoch_indices(rows,19))
    assert before==rows


def test_d0_budget_distinguishes_batches_scheduled_calls_and_amp_updates():
    a={'batch':4,'epochs':300,'warmup_epochs':5,'nbs':64,'patience':80,'lr0':.0005,'lrf':.01}
    b=d0.budget_schedule(a,771)
    assert b['total_batches']==57900
    assert b['images_sampled_per_epoch']==771
    assert b['last_batch_images']==3
    assert b['total_scheduled_optimizer_calls']<b['total_batches']
    assert b['AMP_successful_update_count']=='NOT_VERIFIED_FROM_HISTORICAL_LOGS'
