"""E1 training-only, model-free polygon geometry. No raster/data-loader adapter.

Invalid source rings are rejected, never repaired. This module cannot train,
load models, infer, or select cases from validation outcomes.
"""
from __future__ import annotations
import hashlib
import importlib.abc
import json
import math
import sys
from fractions import Fraction

import numpy as np
from shapely.geometry import Polygon, box
from shapely.validation import explain_validity

PATCH_SIZE = 256
CANVAS_SIZE = 512


class PatchError(ValueError):
    pass


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'ultralytics', 'tensorflow', 'onnxruntime'}:
            raise PatchError('FAIL_MODEL_IMPORT_FORBIDDEN: ' + fullname)


def model_guard():
    if any(k.split('.')[0] in {'torch', 'ultralytics', 'tensorflow', 'onnxruntime'} for k in sys.modules):
        raise PatchError('FAIL_MODEL_ALREADY_IMPORTED')
    sys.meta_path.insert(0, NoModels())


def require(condition, message):
    if not condition:
        raise PatchError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def admit(row):
    require(row.get('source') == 'FUSeg' and row.get('split') == 'train'
            and row.get('role') == 'wound_finetuning', 'FAIL_NOT_TRAINING')
    require(not any('validation' in k or 'prediction' in k or 'failure' in k for k in row),
            'FAIL_OUTCOME_DEPENDENCY')
    for kind in ('image', 'label'):
        path = row[kind + '_path'].replace('\\', '/')
        prefix = 'outputs/isic_fuseg_pretrain_smoke_20260914/fuseg_dataset/' + ('images' if kind == 'image' else 'labels') + '/train/'
        require(path.startswith(prefix) and len(path[len(prefix):].split('/')) == 1
                and '..' not in path.split('/'), 'FAIL_PATH_ROLE')


def parse_labels(text):
    """Keep original nonempty annotation line order; validate topology separately."""
    result = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            values = list(map(float, line.split()))
        except ValueError as exc:
            raise PatchError('FAIL_PATCH_LABEL_INVALID') from exc
        require(len(values) >= 7 and len(values) % 2 == 1 and values[0] == 0
                and all(math.isfinite(v) for v in values), 'FAIL_PATCH_LABEL_INVALID')
        coords = values[1:]
        require(all(0 <= x <= 1 for x in coords), 'FAIL_PATCH_LABEL_INVALID')
        result.append([(coords[i] * 512, coords[i + 1] * 512) for i in range(0, len(coords), 2)])
    return result


def bounds(points):
    return min(x for x, y in points), min(y for x, y in points), max(x for x, y in points), max(y for x, y in points)


def area_ratio(points):
    a, b, c, d = bounds(points)
    return (c - a) * (d - b) / 512**2


def eligible_ids(polygons):
    return [i for i, p in enumerate(polygons) if area_ratio(p) < 0.0025]


def validated_polygon(points):
    require(len(set(map(tuple, points))) >= 3 and all(math.isfinite(v) for p in points for v in p),
            'FAIL_PATCH_LABEL_INVALID')
    p = Polygon(points)
    require(p.is_valid and not p.is_empty and p.area > 0,
            'FAIL_PATCH_LABEL_INVALID: ' + explain_validity(p))
    return p


def patch_bounds(target):
    a, b, c, d = bounds(target)
    x = max(0., min(256., (a + c) / 2 - 128))
    y = max(0., min(256., (b + d) / 2 - 128))
    require(x <= a and y <= b and c <= x + 256 and d <= y + 256,
            'FAIL_PATCH_TARGET_NOT_FULLY_CONTAINED')
    return x, y, x + 256, y + 256


def positive_parts(geometry):
    if geometry.is_empty or geometry.area == 0:
        return []
    if geometry.geom_type == 'Polygon':
        return [geometry]
    return [p for g in geometry.geoms for p in positive_parts(g)]


def transform(polygons, epoch, *, split='train', already_transformed=False):
    require(split == 'train', 'FAIL_NOT_TRAINING')
    require(not already_transformed, 'FAIL_DOUBLE_PATCH')
    require(type(epoch) is int and epoch >= 0, 'FAIL_EPOCH')
    ids = eligible_ids(polygons)
    if not ids:
        return {'representation': 'FULL_IMAGE', 'polygons': polygons, 'target_GT_id': None}
    target = ids[epoch % len(ids)]
    window = patch_bounds(polygons[target])
    rectangle = box(*window)
    records = []
    for i, points in enumerate(polygons):
        p = validated_polygon(points)
        clipped = p.intersection(rectangle)
        parts = positive_parts(clipped)
        # A single source GT must remain one instance. Never silently split it.
        require(len(parts) <= 1, 'FAIL_MULTIPART_INSTANCE_POLICY_UNREGISTERED')
        if not parts:
            records.append({'GT_instance_id': i, 'state': 'dropped', 'retention': 0., 'polygon': None})
            continue
        q = parts[0]
        require(not q.interiors, 'FAIL_POLYGON_HOLE_POLICY_UNREGISTERED')
        full = rectangle.covers(p)
        local = [[(x - window[0]) / 256, (y - window[1]) / 256] for x, y in list(q.exterior.coords)[:-1]]
        require(all(0 <= v <= 1 for point in local for v in point), 'FAIL_PATCH_LABEL_INVALID')
        validated_polygon(local)
        records.append({'GT_instance_id': i, 'state': 'fully_retained' if full else 'partially_clipped',
                        'retention': 1. if full else q.area / p.area, 'polygon': local})
    require(records[target]['state'] == 'fully_retained' and records[target]['retention'] == 1.,
            'FAIL_SELECTED_TARGET_CLIPPED')
    kept = sum(r['state'] != 'dropped' for r in records)
    require(kept > 0, 'FAIL_ELIGIBLE_PATCH_EMPTY')
    a, b, c, d = bounds(polygons[target])
    return {'representation': 'GT_CENTERED_PATCH_REPRESENTATION', 'target_GT_id': target,
            'patch_bounds': window, 'original_GT_count': len(polygons), 'retained_GT_count': kept,
            'fully_retained_GT_count': sum(r['state'] == 'fully_retained' for r in records),
            'partially_clipped_GT_count': sum(r['state'] == 'partially_clipped' for r in records),
            'dropped_GT_count': sum(r['state'] == 'dropped' for r in records), 'records': records,
            'selected_target_retention': 1., 'source_bbox_area_ratio': area_ratio(polygons[target]),
            'patch_bbox_area_ratio': area_ratio(polygons[target]) * 4,
            'context_distances_left_top_right_bottom': [a-window[0], b-window[1], window[2]-c, window[3]-d],
            'transformed_label_hash': digest(records)}


def uniform_order(sample_ids, epoch):
    require(len(set(sample_ids)) == len(sample_ids) and epoch >= 0, 'FAIL_UNIFORM_PLAN')
    order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([42, epoch]))).permutation(len(sample_ids))
    return [sample_ids[int(i)] for i in order]


def advancement(control, patch):
    """Future saved-count decision only. No evaluation or model execution."""
    required = ('very_small', 'small', 'medium', 'large', 'crop_complete', 'tp', 'fp', 'fn')
    for arm in (control, patch):
        require(all(type(arm[k]) is int and arm[k] >= 0 for k in required), 'FAIL_GATE_COUNTS')
        require(arm['tp'] + arm['fp'] > 0 and 2*arm['tp'] + arm['fp'] + arm['fn'] > 0, 'FAIL_GATE_DENOMINATOR')
    def precision(a): return Fraction(a['tp'], a['tp']+a['fp'])
    def f1(a): return Fraction(2*a['tp'], 2*a['tp']+a['fp']+a['fn'])
    return {'very_small': patch['very_small'] >= control['very_small']+2,
            'small': patch['small'] >= control['small']-1,
            'medium': patch['medium'] >= control['medium']-1,
            'large': patch['large'] >= control['large'],
            'crop_complete': patch['crop_complete'] >= control['crop_complete']-1,
            'precision': precision(patch) >= precision(control)-Fraction(1, 100),
            'f1': f1(patch) >= f1(control)-Fraction(1, 100)}
