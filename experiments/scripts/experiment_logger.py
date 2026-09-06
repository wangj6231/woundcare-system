# ============================================================
# experiment_logger.py — Phase 1 (定版)
# 固定 30 欄位 Schema，所有模型產生相同格式 CSV
# 支援 status = running / done / FAILED
# ============================================================
import csv
import os
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 固定 Schema（30 欄位，順序不可改動）
# ──────────────────────────────────────────────────────────────
SCHEMA = [
    # 實驗識別
    'experiment_id',      # e.g. C-Arch-05
    'experiment_type',    # classification / detection / ablation
    # 模型
    'model',              # e.g. yolov8n-cls.pt / resnet50
    'framework',          # ultralytics / torch
    # 資料集
    'dataset',            # e.g. yolo_wound_cls_dataset_v3
    'dataset_version',    # v3
    # 實驗條件
    'fold',               # 0=no fold, 1-5=fold number
    'seed',               # 42 / 123 / 3407 / 2026 / 999
    'epochs',
    'batch_size',
    'image_size',
    'optimizer',
    'learning_rate',
    'weight_decay',
    'augmentation',       # summary string
    # 資料量
    'train_samples',
    'val_samples',
    # 分類指標（偵測模型這些欄位留空）
    'accuracy',           # Top-1 %
    'top5_accuracy',      # Top-5 %
    'precision',          # Macro Precision %
    'recall',             # Macro Recall %
    'macro_f1',           # Macro F1 %
    'weighted_f1',        # Weighted F1 %
    'roc_auc',            # Macro OvR AUC
    # 效率
    'inference_ms',       # ms per image
    # 訓練資訊
    'training_time',      # minutes
    'best_epoch',
    # 路徑
    'model_path',
    # 狀態與時間
    'status',             # running / done / FAILED
    'timestamp',
]

LOG_PATH = Path(__file__).parent.parent / 'experiment_log.csv'


# ──────────────────────────────────────────────────────────────
# 公開 API
# ──────────────────────────────────────────────────────────────

def log_start(experiment_id: str, cfg: dict) -> None:
    """
    實驗開始時寫一筆 status=running 的記錄。
    確保即使訓練中途崩潰，experiment_log.csv 也有留下記錄。
    """
    record = _build_base_record(experiment_id, cfg)
    record['status'] = 'running'
    _write_row(record)


def log_done(experiment_id: str, cfg: dict, metrics: dict) -> None:
    """
    實驗完成後寫 status=done 的完整記錄（含所有指標）。
    """
    record = _build_base_record(experiment_id, cfg)
    record['status'] = 'done'
    record.update({
        'accuracy':       _fmt(metrics.get('accuracy')),
        'top5_accuracy':  _fmt(metrics.get('top5_accuracy')),
        'precision':      _fmt(metrics.get('macro_precision')),
        'recall':         _fmt(metrics.get('macro_recall')),
        'macro_f1':       _fmt(metrics.get('macro_f1')),
        'weighted_f1':    _fmt(metrics.get('weighted_f1')),
        'roc_auc':        _fmt(metrics.get('roc_auc_macro')),
        'inference_ms':   _fmt(metrics.get('inference_ms')),
        'training_time':  _fmt(metrics.get('training_time_min')),
        'best_epoch':     _fmt(metrics.get('best_epoch')),
        'model_path':     metrics.get('model_path', ''),
        'train_samples':  _fmt(metrics.get('train_samples')),
        'val_samples':    _fmt(metrics.get('val_samples')),
    })
    _write_row(record)
    _print_summary(record)


def log_failed(experiment_id: str, cfg: dict, error: str = '') -> None:
    """
    訓練失敗時寫 status=FAILED 的記錄，確保永遠有痕跡。
    """
    record = _build_base_record(experiment_id, cfg)
    record['status'] = 'FAILED'
    record['model_path'] = f'ERROR: {error[:200]}'
    _write_row(record)
    print(f"\n❌ FAILED logged: {experiment_id}")


# ──────────────────────────────────────────────────────────────
# 內部工具
# ──────────────────────────────────────────────────────────────

def _build_base_record(experiment_id: str, cfg: dict) -> dict:
    """從 cfg 建立含實驗條件的基底 record（指標欄位留空）"""
    aug = cfg.get('augmentation', {})
    if isinstance(aug, dict):
        aug_str = ','.join(f"{k}={v}" for k, v in aug.items())
    else:
        aug_str = str(aug)

    exp_cfg    = cfg.get('experiment', {})
    model_cfg  = cfg.get('model', {})
    ds_cfg     = cfg.get('dataset', {})
    train_cfg  = cfg.get('training', {})
    opt_cfg    = cfg.get('optimizer', {})

    record = {f: '' for f in SCHEMA}  # 先全部填空
    record.update({
        'experiment_id':   experiment_id,
        'experiment_type': exp_cfg.get('type', cfg.get('task', '')),
        'model':           model_cfg.get('name', cfg.get('model', '')),
        'framework':       model_cfg.get('framework', cfg.get('framework', '')),
        'dataset':         ds_cfg.get('path', cfg.get('dataset', '')),
        'dataset_version': ds_cfg.get('version', cfg.get('dataset_version', '')),
        'fold':            cfg.get('fold', 0),
        'seed':            train_cfg.get('seed', cfg.get('seed', '')),
        'epochs':          train_cfg.get('epochs', cfg.get('epochs', '')),
        'batch_size':      train_cfg.get('batch', cfg.get('batch_size', '')),
        'image_size':      train_cfg.get('imgsz', cfg.get('img_size', '')),
        'optimizer':       opt_cfg.get('name', cfg.get('optimizer', '')),
        'learning_rate':   opt_cfg.get('lr0', cfg.get('lr0', '')),
        'weight_decay':    opt_cfg.get('weight_decay', cfg.get('weight_decay', '')),
        'augmentation':    aug_str,
        'timestamp':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })
    return record


def _write_row(record: dict) -> None:
    """安全寫入一行 CSV（追加模式）"""
    log_path = LOG_PATH.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 確保所有欄位存在，缺的填空字串
    for f in SCHEMA:
        record.setdefault(f, '')

    file_exists = log_path.exists()
    with open(log_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def _fmt(val) -> str:
    """安全格式化數值"""
    if val is None or val == '':
        return ''
    if isinstance(val, float):
        return f'{val:.2f}' if val > 1 else f'{val:.4f}'
    return str(val)


def _print_summary(record: dict) -> None:
    sep = '─' * 55
    print(f"\n{sep}")
    print(f"✅ {record['experiment_id']}  |  {record['model']}  |  seed={record['seed']}  fold={record['fold']}")
    if record.get('accuracy'):
        print(f"   Acc={record['accuracy']}%  Macro-F1={record['macro_f1']}%  WF1={record['weighted_f1']}%")
    if record.get('model_path') and 'ERROR' not in str(record.get('model_path', '')):
        print(f"   Model: {record['model_path']}")
    print(sep)
