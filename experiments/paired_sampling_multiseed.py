"""Seed-parameterized runtime namespace; never edit the sealed seed42 source.

The verified D0.1 source is executed in a private module namespace unchanged.
Only plan_indices and output routing are rebound. All trainer, AMP, loader,
accumulation and telemetry methods remain the original compiled source.
"""
from pathlib import Path
import sys
import types
import numpy as np
from experiments.phase_d0 import sha, probabilities

BASE_SHA='c996783410e4d0e61d3523d08bc3b65079c205487081937c454f61f6ecbaec40'
SEEDS=(42,123,3407,2026,999)


def outputs(seed):
    if seed not in SEEDS:raise ValueError('UNREGISTERED_SEED')
    return {a:f'experiments/results/d_seg_small_sampling_v2_seed{seed}_{label}' for a,label in [('C','control'),('S','experimental')]}


def indices(rows,arm,epoch,seed):
    if type(seed) is not int or seed not in SEEDS or epoch<0 or arm not in ('C','S'):raise ValueError('UNREGISTERED_SAMPLER_ARGUMENT')
    p=probabilities(rows)
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,epoch])))
    return rng.permutation(len(rows)) if arm=='C' else rng.choice(len(rows),size=len(rows),replace=True,p=p)


def make_runtime(root,seed):
    if seed not in SEEDS:raise ValueError('UNREGISTERED_SEED')
    path=Path(root)/'experiments/paired_sampling_runner.py'
    if sha(path)!=BASE_SHA:raise ValueError('FAIL_FROZEN_RUNNER_MUTATED')
    name=f'experiments._paired_seed_{seed}'
    module=types.ModuleType(name);module.__file__=__file__
    exec(compile(path.read_text(encoding='utf8'),str(path),'exec'),module.__dict__)
    module.OUTPUTS=outputs(seed)
    module.plan_indices=lambda rows,arm,epoch:indices(rows,arm,epoch,seed)
    # The unchanged methods resolve only these intentionally rebound globals.
    sys.modules[name]=module
    return module
