"""Synthetic G0 contract tests: no model loading, datasets or inference."""
import ast
from pathlib import Path
import pytest
from experiments.phase_g0 import label_state, evidence_level, eligible


@pytest.mark.parametrize('text,state,n', [
    (None,'MISSING',None), ('','EMPTY',0), (' \n\t','EMPTY',0),
    ('0 0 0 1 0 1 1','NONEMPTY',1),
    ('0 0 0 1 0 1 1\n0 0 0 0 1 1 1','NONEMPTY',2),
    ('1 0 0 1 0 1 1','INVALID',None), ('0 0 0 1 0','INVALID',None),
    ('0 0 0 nan 0 1 1','INVALID',None), ('0 0 0 2 0 1 1','INVALID',None),
    ('0 0 0 0 0 0 0','INVALID',None), ('bad label','INVALID',None)])
def test_label_state(text,state,n):
    assert label_state(text) == (state,n)


def test_no_automatic_negative():
    assert evidence_level('EMPTY',True,False,True) == 'EMPTY_LABEL_ONLY'
    assert evidence_level('EMPTY',True,True,False) == 'EMPTY_LABEL_ONLY'
    assert evidence_level('EMPTY',False,True,True) == 'EMPTY_LABEL_ONLY'
    assert evidence_level('MISSING',True,True,True) == 'MISSING_LABEL'
    assert evidence_level('INVALID',True,True,True) == 'INVALID_LABEL'
    assert evidence_level('NONEMPTY',True,True,True) == 'POSITIVE_IMAGE_UNANNOTATED_REGION'


def test_verified_native_or_expert():
    assert evidence_level('EMPTY',True,True,True) == 'VERIFIED_NEGATIVE'
    assert evidence_level('EMPTY',True,False,False,'CONFIRMED_NO_TARGET_WOUND') == 'VERIFIED_NEGATIVE'
    assert evidence_level('EMPTY',True,True,True,'UNCERTAIN') == 'UNCERTAIN'
    assert evidence_level('EMPTY',True,True,True,'TARGET_WOUND_PRESENT') == 'TARGET_WOUND_PRESENT'


@pytest.mark.parametrize('level', ['EMPTY_LABEL_ONLY','UNCERTAIN','MISSING_LABEL','INVALID_LABEL','POSITIVE_IMAGE_UNANNOTATED_REGION','DEVELOPMENT_VALIDATION_EVIDENCE_ONLY','TEST'])
def test_excluded_level(level):
    assert not eligible(level,[])


def test_leakage_excludes_verified():
    assert eligible('VERIFIED_NEGATIVE',[])
    assert not eligible('VERIFIED_NEGATIVE',['validation'])
    assert not eligible('VERIFIED_NEGATIVE',['locked_test'])


def test_no_model_import_or_training_call():
    tree = ast.parse(Path('experiments/phase_g0.py').read_text(encoding='utf-8'))
    forbidden = {'torch','ultralytics','tensorflow','onnxruntime','subprocess'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            assert all(n.name.split('.')[0] not in forbidden for n in node.names)
        if isinstance(node,ast.ImportFrom):
            assert node.module.split('.')[0] not in forbidden
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in {'train','predict','val','load_state_dict','imread'}
