# ============================================================
# train_cls_yolo.py — Phase 2 (定版)
# 職責：YOLO 分類模型訓練，回傳 train_record dict
# 不呼叫 logger（由 run_experiments.py orchestrator 負責）
# ============================================================
import os
import sys
import time
import yaml
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent

CLASS_NAMES = ['Abrasions', 'Bruises', 'Burns', 'Cut',
               'Ingrown_nails', 'Laceration', 'Stab_wound']


def train_yolo_cls(flat_cfg: dict) -> dict:
    """
    執行 YOLO 分類模型訓練。

    Args:
        flat_cfg: _cfg_to_flat() 產生的扁平 dict

    Returns:
        dict 含 weights_path, best_epoch, train_samples, val_samples
    """
    from ultralytics import YOLO

    model_name  = flat_cfg['model']
    run_name    = flat_cfg['run_name']
    epochs      = flat_cfg.get('epochs', 150)
    batch       = flat_cfg.get('batch_size', 16)
    imgsz       = flat_cfg.get('img_size', 224)
    seed        = flat_cfg.get('seed', 42)
    dataset     = flat_cfg.get('dataset', '')
    optimizer   = flat_cfg.get('optimizer', 'auto')
    lr0         = flat_cfg.get('lr0', 0.01)
    weight_decay = flat_cfg.get('weight_decay', 0.0005)
    aug         = flat_cfg.get('augmentation', {})

    # 增強參數
    scale   = aug.get('scale', 0.0)
    fliplr  = aug.get('fliplr', 0.0)
    flipud  = aug.get('flipud', 0.0)
    degrees = aug.get('degrees', 0.0)
    erasing = aug.get('erasing', 0.0)

    out_dir = str(BASE_DIR / 'experiments' / 'results' / 'raw')

    print(f"   [{run_name}] Loading model: {model_name}")

    # 優先使用本地 .pt 檔
    local_pt = BASE_DIR / model_name
    model_path = str(local_pt) if local_pt.exists() else model_name
    model = YOLO(model_path)

    # 計算資料量（訓練前掃描）
    train_samples = _count_images(Path(dataset) / 'train')
    val_samples   = _count_images(Path(dataset) / 'val')

    results = model.train(
        data=dataset,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        name=run_name,
        project=out_dir,
        optimizer=optimizer,
        lr0=lr0,
        weight_decay=weight_decay,
        scale=scale,
        fliplr=fliplr,
        flipud=flipud,
        degrees=degrees,
        erasing=erasing,
        verbose=False,
        exist_ok=True,
    )

    best_top1  = float(results.results_dict.get('metrics/accuracy_top1', 0))
    best_epoch = results.best_epoch if hasattr(results, 'best_epoch') else epochs
    weights_path = str(Path(out_dir) / run_name / 'weights' / 'best.pt')

    print(f"   [{run_name}] Done — Best Top-1: {best_top1*100:.2f}%  Best Epoch: {best_epoch}")

    return {
        'weights_path':  weights_path,
        'best_epoch':    best_epoch,
        'val_top1':      round(best_top1 * 100, 2),
        'train_samples': train_samples,
        'val_samples':   val_samples,
    }


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob('*')
               if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))


# ── 允許直接執行（供測試用）──────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to experiment YAML config')
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from experiment_logger import log_start, log_done, log_failed

    SCRIPTS = Path(__file__).parent
    sys.path.insert(0, str(SCRIPTS.parent.parent / 'experiments' / 'scripts'))

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 建立 flat_cfg（模擬 orchestrator 行為）
    exp_id = cfg['experiment']['id']
    flat_cfg = {
        'exp_id':       exp_id,
        'model':        cfg['model']['name'],
        'run_name':     f"{cfg['experiment']['id']}_{cfg['experiment']['name']}",
        'dataset':      str(BASE_DIR / cfg['dataset']['path']),
        'epochs':       cfg['training']['epochs'],
        'batch_size':   cfg['training']['batch'],
        'img_size':     cfg['training']['imgsz'],
        'seed':         cfg['training']['seed'],
        'optimizer':    cfg['optimizer']['name'],
        'lr0':          cfg['optimizer']['lr0'],
        'weight_decay': cfg['optimizer']['weight_decay'],
        'augmentation': cfg.get('augmentation', {}),
    }

    log_start(exp_id, cfg)
    try:
        record = train_yolo_cls(flat_cfg)
        log_done(exp_id, cfg, {
            'training_time_min': 0,
            'model_path': record['weights_path'],
            'best_epoch': record['best_epoch'],
            'train_samples': record['train_samples'],
            'val_samples': record['val_samples'],
        })
    except Exception as e:
        log_failed(exp_id, cfg, error=str(e))
        raise
