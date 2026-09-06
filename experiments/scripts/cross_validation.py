# ============================================================
# cross_validation.py — Phase 6 (定版)
#
# 真正的 Stratified K-Fold CV：
#   ① 收集 Development Set（train/ + val/，不含 test/）
#   ② StratifiedKFold 重新切分
#   ③ 每 Fold 建立實體資料夾（Windows 相容：複製而非 symlink）
#   ④ 訓練 → 評估 → log_start/done/failed（與 run_experiments.py 相同 API）
#   ⑤ 彙總 Mean ± SD → JSON + CSV 各一行
#
# ⚠️  test/ 資料夾完全不接觸（Blind Test 保護）
#
# 使用方式：
#   python experiments/run_experiments.py --kfold experiments/configs/C-Arch-05_yolov8n_cls.yaml
# ============================================================
import json
import os
import shutil
import sys
import yaml
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from experiment_logger import log_start, log_done, log_failed, SCHEMA
from evaluate import evaluate

BASE_DIR    = Path(__file__).parent.parent.parent
TEMP_DIR    = Path(__file__).parent.parent / 'results' / 'raw' / '_kfold_tmp'
STATS_DIR   = Path(__file__).parent.parent / 'results' / 'statistics'
LOG_PATH    = Path(__file__).parent.parent / 'experiment_log.csv'

CLASS_NAMES = ['Abrasions', 'Bruises', 'Burns', 'Cut',
               'Ingrown_nails', 'Laceration', 'Stab_wound']
TORCH_MODELS = {'resnet50', 'efficientnet_b0', 'mobilenet_v3_large', 'vit_s_16'}

# Development Set = train + val（test/ 完全不碰）
DEV_SPLITS = ['train', 'val']


# ──────────────────────────────────────────────────────────────
# Step 1: 收集 Development Set 全部影像
# ──────────────────────────────────────────────────────────────
def collect_dev_set(dataset_path: Path) -> tuple:
    """
    掃描 train/ 和 val/ 目錄，回傳所有影像的 (paths, labels)。
    labels 是整數索引，對應 CLASS_NAMES。
    test/ 完全略過。
    """
    all_images, all_labels = [], []
    total_by_class = {c: 0 for c in CLASS_NAMES}

    for split in DEV_SPLITS:
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cls_dir = dataset_path / split / cls_name
            if not cls_dir.exists():
                continue
            imgs = list(cls_dir.glob('*.[jp][pn]g')) + list(cls_dir.glob('*.jpeg'))
            for img in imgs:
                all_images.append(str(img))
                all_labels.append(cls_idx)
                total_by_class[cls_name] += 1

    print(f"\n   Development Set summary (train+val, test excluded):")
    for cls_name, cnt in total_by_class.items():
        print(f"     {cls_name:<20}: {cnt}")
    print(f"     {'TOTAL':<20}: {len(all_images)}")

    return all_images, all_labels


# ──────────────────────────────────────────────────────────────
# Step 2: 建立 Fold 臨時資料夾（複製，Windows 相容）
# ──────────────────────────────────────────────────────────────
def build_fold_dir(fold_dir: Path,
                   train_imgs, train_lbls,
                   val_imgs,   val_lbls) -> None:
    """
    在 fold_dir 下建立 train/ 和 val/ 子資料夾，
    並把圖片複製進去（Windows 不支援一般使用者建 symlink）。
    若圖片名稱衝突（不同類別同名），加上 class 前綴避免覆蓋。
    """
    if fold_dir.exists():
        shutil.rmtree(fold_dir)

    for cls_name in CLASS_NAMES:
        (fold_dir / 'train' / cls_name).mkdir(parents=True, exist_ok=True)
        (fold_dir / 'val'   / cls_name).mkdir(parents=True, exist_ok=True)

    def _copy(src, split, cls_idx):
        cls_name = CLASS_NAMES[cls_idx]
        src_path = Path(src)
        dst_name = f"{cls_name}_{src_path.name}"   # 加 class 前綴，防止同名衝突
        dst = fold_dir / split / cls_name / dst_name
        shutil.copy2(src, dst)

    for img, lbl in zip(train_imgs, train_lbls):
        _copy(img, 'train', lbl)
    for img, lbl in zip(val_imgs, val_lbls):
        _copy(img, 'val', lbl)

    train_total = sum(1 for _ in (fold_dir / 'train').rglob('*.[jp][pn]g'))
    val_total   = sum(1 for _ in (fold_dir / 'val').rglob('*.[jp][pn]g'))
    print(f"   Fold dir built: train={train_total}, val={val_total}")


# ──────────────────────────────────────────────────────────────
# Step 3: 建立 flat_cfg（與 run_experiments._cfg_to_flat 一致）
# ──────────────────────────────────────────────────────────────
def _fold_flat_cfg(cfg: dict, fold_dir: Path, fold_run_name: str, fold_idx: int) -> dict:
    aug = cfg.get('augmentation', {})
    aug_str = ','.join(f"{k}={v}" for k, v in aug.items()) if isinstance(aug, dict) else str(aug)
    train_cfg = cfg['training']
    opt_cfg   = cfg['optimizer']
    return {
        'exp_id':          cfg['experiment']['id'],   # 保持原 exp_id（log 用）
        'experiment_name': cfg['experiment']['name'],
        'task':            cfg['experiment']['type'],
        'model':           cfg['model']['name'],
        'framework':       cfg['model']['framework'],
        'dataset':         str(fold_dir),             # ← fold 臨時資料夾
        'dataset_version': cfg['dataset']['version'],
        'run_name':        fold_run_name,
        'epochs':          train_cfg['epochs'],
        'batch_size':      train_cfg['batch'],
        'img_size':        train_cfg['imgsz'],
        'seed':            train_cfg['seed'],
        'fold':            fold_idx,
        'optimizer':       opt_cfg['name'],
        'lr0':             opt_cfg['lr0'],
        'weight_decay':    opt_cfg['weight_decay'],
        'augmentation':    aug,
        'augmentation_note': aug_str,
        'notes':           f"K-Fold {fold_idx}: {fold_run_name}",
    }


# ──────────────────────────────────────────────────────────────
# Main K-Fold Runner
# ──────────────────────────────────────────────────────────────
def run_kfold(cfg: dict, k: int = 5) -> dict:
    import time

    dataset_path = BASE_DIR / cfg['dataset']['path']
    exp_id       = cfg['experiment']['id']
    exp_name     = cfg['experiment']['name']
    model_name   = cfg['model']['name']
    framework    = cfg['model']['framework']
    seed         = cfg['training']['seed']

    print(f"\n{'='*62}")
    print(f"🔁  Stratified {k}-Fold CV")
    print(f"    Experiment: {exp_id} | {exp_name}")
    print(f"    Model: {model_name}  |  Seed: {seed}")
    print(f"    Blind Test (test/) is excluded from all folds.")
    print(f"{'='*62}")

    # ── 收集 Development Set ──────────────────────────────────
    all_images, all_labels = collect_dev_set(dataset_path)
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_images, all_labels), 1):
        fold_run_name = f'{exp_id}_kfold{fold_idx}'
        print(f"\n{'─'*62}")
        print(f"  Fold {fold_idx}/{k}  |  run: {fold_run_name}")

        train_imgs = all_images[train_idx].tolist()
        train_lbls = all_labels[train_idx].tolist()
        val_imgs   = all_images[val_idx].tolist()
        val_lbls   = all_labels[val_idx].tolist()

        # ── 建立 fold 資料夾 ──────────────────────────────────
        fold_dir = TEMP_DIR / fold_run_name
        build_fold_dir(fold_dir, train_imgs, train_lbls, val_imgs, val_lbls)

        flat_cfg = _fold_flat_cfg(cfg, fold_dir, fold_run_name, fold_idx)

        # ── log_start（fold_idx 寫入 CSV）────────────────────
        # 注意：用原始 cfg 但覆寫 fold 欄位
        log_start(exp_id, {**cfg, '_fold_override': fold_idx})

        # ── 訓練 ─────────────────────────────────────────────
        t0 = time.time()
        weights_path = ''
        train_record = {}

        try:
            is_torch = framework == 'torch' or model_name.replace('.pt', '') in TORCH_MODELS
            if is_torch:
                from train_cls_torch import train_torch_cls
                train_record = train_torch_cls(flat_cfg)
            else:
                from train_cls_yolo import train_yolo_cls
                train_record = train_yolo_cls(flat_cfg)
            weights_path = train_record.get('weights_path', '')

        except Exception as e:
            elapsed = (time.time() - t0) / 60
            log_failed(exp_id, cfg, error=f"Fold {fold_idx}: {str(e)}")
            print(f"  ❌ Fold {fold_idx} FAILED: {e}")
            import traceback
            traceback.print_exc()
            fold_results.append({'fold': fold_idx, 'accuracy': 0, 'macro_f1': 0,
                                  'weighted_f1': 0, 'stab_wound_recall': 0, 'status': 'FAILED'})
            continue

        elapsed = (time.time() - t0) / 60

        # ── 評估 ─────────────────────────────────────────────
        metrics = {}
        try:
            metrics = evaluate(
                weights_path=weights_path,
                data_dir=str(fold_dir / 'val'),
                framework='torch' if (framework == 'torch' or
                          model_name.replace('.pt','') in TORCH_MODELS) else 'ultralytics',
                exp_id=fold_run_name,
                split='val',
                save_figures=True
            )
        except Exception as e:
            print(f"  ⚠️  Evaluate failed for fold {fold_idx}: {e}")

        metrics['training_time_min'] = round(elapsed, 1)
        metrics['model_path']  = weights_path
        metrics['best_epoch']  = train_record.get('best_epoch', '')
        metrics['train_samples'] = len(train_imgs)
        metrics['val_samples']   = len(val_imgs)

        fold_result = {
            'fold':              fold_idx,
            'accuracy':          metrics.get('accuracy', 0),
            'macro_f1':          metrics.get('macro_f1', 0),
            'weighted_f1':       metrics.get('weighted_f1', 0),
            'stab_wound_recall': metrics.get('stab_wound_recall', 0),
            'status':            'done',
        }
        fold_results.append(fold_result)

        # ── log_done（fold=fold_idx 寫入 CSV）─────────────────
        # 需要把 fold 資訊注入 cfg，讓 _build_base_record 能讀到
        fold_cfg = {**cfg, 'fold': fold_idx}
        log_done(exp_id, fold_cfg, metrics)

        print(f"  ✅ Fold {fold_idx}: Acc={metrics.get('accuracy',0):.2f}%  "
              f"F1={metrics.get('macro_f1',0):.2f}%  "
              f"StabRec={metrics.get('stab_wound_recall',0):.2f}%")

    # ── 清理暫時資料夾 ────────────────────────────────────────
    if TEMP_DIR.exists():
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"\n🧹 Temp fold dirs cleaned.")
        except Exception as e:
            print(f"  ⚠️  Could not clean temp dir: {e}")

    # ── 彙總 Mean ± SD ────────────────────────────────────────
    done_folds = [r for r in fold_results if r.get('status') != 'FAILED']
    if not done_folds:
        print("❌ All folds failed. No summary generated.")
        return {}

    accs   = [r['accuracy']          for r in done_folds]
    f1s    = [r['macro_f1']          for r in done_folds]
    wf1s   = [r['weighted_f1']       for r in done_folds]
    stabs  = [r['stab_wound_recall'] for r in done_folds]

    summary = {
        'experiment_id':    exp_id,
        'model':            model_name,
        'k_folds':          k,
        'completed_folds':  len(done_folds),
        'seed':             seed,
        'accuracy_mean':    round(float(np.mean(accs)),  2),
        'accuracy_std':     round(float(np.std(accs)),   2),
        'macro_f1_mean':    round(float(np.mean(f1s)),   2),
        'macro_f1_std':     round(float(np.std(f1s)),    2),
        'weighted_f1_mean': round(float(np.mean(wf1s)),  2),
        'weighted_f1_std':  round(float(np.std(wf1s)),   2),
        'stab_recall_mean': round(float(np.mean(stabs)), 2),
        'stab_recall_std':  round(float(np.std(stabs)),  2),
        'fold_results':     fold_results,
    }

    # 儲存 JSON
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = STATS_DIR / f'{exp_id}_kfold_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 印出論文格式結果
    sep = '=' * 62
    print(f"\n{sep}")
    print(f"✅  {k}-Fold CV Complete: {exp_id}")
    print(f"    Model: {model_name}  |  Seed: {seed}  |  Folds done: {len(done_folds)}/{k}")
    print(f"{'─'*62}")
    print(f"    {'Fold':<8} {'Accuracy':>10} {'Macro-F1':>10} {'WF1':>10} {'StabRec':>10}")
    for r in fold_results:
        status = '❌' if r.get('status') == 'FAILED' else ''
        print(f"    {r['fold']:<8} {r['accuracy']:>10.2f} {r['macro_f1']:>10.2f} "
              f"{r['weighted_f1']:>10.2f} {r['stab_wound_recall']:>10.2f}  {status}")
    print(f"{'─'*62}")
    print(f"    {'Mean':<8} {summary['accuracy_mean']:>10.2f} {summary['macro_f1_mean']:>10.2f} "
          f"{summary['weighted_f1_mean']:>10.2f} {summary['stab_recall_mean']:>10.2f}")
    print(f"    {'±SD':<8} {summary['accuracy_std']:>10.2f} {summary['macro_f1_std']:>10.2f} "
          f"{summary['weighted_f1_std']:>10.2f} {summary['stab_recall_std']:>10.2f}")
    print(f"{'─'*62}")
    print(f"    📝 論文格式:")
    print(f"       Top-1 Accuracy = {summary['accuracy_mean']}% ± {summary['accuracy_std']}%  ({k}-Fold CV, seed={seed})")
    print(f"       Macro-F1       = {summary['macro_f1_mean']}% ± {summary['macro_f1_std']}%")
    print(f"       Stab Recall    = {summary['stab_recall_mean']}% ± {summary['stab_recall_std']}%")
    print(f"    💾 Summary: {json_path}")
    print(sep)

    return summary
