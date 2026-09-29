"""Allowlisted image access for main process and stock-loader worker processes."""
import os
from pathlib import Path
import sys
import json

_installed = False
_stock_seed_worker = None


def admitted_images(mode):
    from experiments.phase_f02 import ROOT, PRO, read
    allowed = set()
    if mode == 'training':
        cfg = read(PRO/'F_HIGHER_SCALE_V2_control_config.json')
        allowed.update((ROOT/r['image_path']).resolve() for r in read(ROOT/cfg['manifest']['path'])['samples'])
    rows = read(PRO/'ISIC_FUSEG_COLORFIX_V1_validation_manifest.json')['samples']
    for row in rows:
        if mode == 'training':
            allowed.add((ROOT/'outputs/isic_fuseg_pretrain_smoke_20260914/fuseg_dataset/images/val'/row['sample_id']).resolve())
        else:
            allowed.add((ROOT/row['relative_image_path']).resolve())
            allowed.add((ROOT/row['relative_mask_path']).resolve())
    return allowed


def install(out, mode):
    global _installed
    if _installed:
        return
    import cv2
    import numpy as np
    from PIL import Image
    from experiments.phase_f02 import ROOT
    allowed = admitted_images(mode)
    out = Path(out).resolve()
    package_assets = Path('C:/Python312/Lib/site-packages/ultralytics/assets').resolve()
    package_mpl = Path('C:/Python312/Lib/site-packages/matplotlib/mpl-data').resolve()
    log = (out/f'input_access_{os.getpid()}.jsonl').open('x', encoding='utf-8')
    seen = set()
    def check(value):
        if not isinstance(value, (str, bytes, Path)):
            return
        path = Path(os.fsdecode(value)).resolve()
        if path not in allowed and not path.is_relative_to(out) and not path.is_relative_to(package_assets) and not path.is_relative_to(package_mpl):
            raise RuntimeError('FAIL_DATA_ROLE_ACCESS: '+str(path))
        if path not in seen:
            seen.add(path)
            log.write(json.dumps(dict(path=str(path), role=('FROZEN_DEVELOPMENT' if path in allowed else 'GENERATED_OR_FRAMEWORK_ASSET'))) + '\n')
            log.flush()
    original_imread, original_open, original_fromfile = cv2.imread, Image.open, np.fromfile
    def imread(file, *a, **kw):
        check(file)
        return original_imread(file, *a, **kw)
    def image_open(file, *a, **kw):
        check(file)
        return original_open(file, *a, **kw)
    def fromfile(file, *a, **kw):
        if isinstance(file, (str, bytes, Path)) and Path(os.fsdecode(file)).suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}:
            check(file)
        return original_fromfile(file, *a, **kw)
    cv2.imread, Image.open, np.fromfile = imread, image_open, fromfile
    def audit(event, args):
        if event == 'socket.connect':
            raise ConnectionError('F1_NETWORK_FORBIDDEN')
        if event == 'open' and isinstance(args[0], (str, bytes)):
            p = Path(os.fsdecode(args[0]))
            if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}:
                mode_arg = args[1] or ''
                writing = any(x in mode_arg for x in 'wax+') or bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
                if writing:
                    if not p.resolve().is_relative_to(out):
                        raise RuntimeError('FAIL_SOURCE_IMAGE_WRITE')
                else:
                    check(p)
    sys.addaudithook(audit)
    _installed = True
    os.environ['WOUND_F1_GUARD_OUTPUT'] = str(out)
    os.environ['WOUND_F1_GUARD_MODE'] = mode


def guarded_seed_worker(worker_id):
    global _stock_seed_worker
    if _stock_seed_worker is None:
        from ultralytics.data.build import seed_worker
        _stock_seed_worker = seed_worker
    if _stock_seed_worker is guarded_seed_worker:
        raise RuntimeError('FAIL_WORKER_GUARD_BINDING')
    _stock_seed_worker(worker_id)
    install(Path(os.environ['WOUND_F1_GUARD_OUTPUT']), os.environ['WOUND_F1_GUARD_MODE'])


def install_worker_hook():
    global _stock_seed_worker
    from ultralytics.data import build
    _stock_seed_worker = build.seed_worker
    build.seed_worker = guarded_seed_worker
