# ============================================================
# cross_validation_group.py — Phase 6.6 (定版 v2)
#
# StratifiedGroupKFold：以 MD5 duplicate group 為不可拆分單位
# 三道 Zero-Leakage Gate（訓練前強制通過）：
#   Gate 1: Train MD5 ∩ Val MD5 = ∅
#   Gate 2: Train group_ids ∩ Val group_ids = ∅
#   Gate 3: 7 classes 全部出現在 Train 和 Val
#
# 使用方式：
#   python experiments/run_experiments.py --kfold-group \
#          experiments/configs/C-Arch-05_yolov8n_cls.yaml
# ============================================================
import hashlib
import json
import shutil
import sys
import time
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from experiment_logger import log_start, log_done, log_failed
from evaluate import evaluate

BASE_DIR    = Path(__file__).parent.parent.parent
TEMP_DIR    = Path(__file__).parent.parent / 'results' / 'raw' / '_gkfold_tmp'
STATS_DIR   = Path(__file__).parent.parent / 'results' / 'statistics'

CLASS_NAMES  = ['Abrasions', 'Bruises', 'Burns', 'Cut',
                'Ingrown_nails', 'Laceration', 'Stab_wound']
NUM_CLASSES  = len(CLASS_NAMES)
DEV_SPLITS   = ['train', 'val']
TORCH_MODELS = {'resnet50', 'efficientnet_b0', 'mobilenet_v3_large', 'vit_s_16'}
K = 5


# ──────────────────────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────────────────────
def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_dev_set_with_groups(dataset_path: Path):
    """
    收集 Development Set（train+val），計算 MD5，
    回傳 (paths, labels, group_ids, metadata)
    相同 MD5 → 相同 group_id（整數）
    """
    items = []
    for split in DEV_SPLITS:
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cls_dir = dataset_path / split / cls_name
            if not cls_dir.exists():
                continue
            for img in list(cls_dir.glob('*.[jp][pn]g')) + list(cls_dir.glob('*.jpeg')):
                items.append((img, cls_idx))

    print(f"  Computing MD5 for {len(items)} images...")
    hash_to_gid = {}   # md5 → group_id (int)
    gid_counter = [0]
    paths, labels, groups = [], [], []
    all_md5s = []

    for img_path, cls_idx in items:
        md5 = file_md5(img_path)
        if md5 not in hash_to_gid:
            hash_to_gid[md5] = gid_counter[0]
            gid_counter[0] += 1
        paths.append(str(img_path))
        labels.append(cls_idx)
        groups.append(hash_to_gid[md5])
        all_md5s.append(md5)

    total_groups = len(hash_to_gid)
    gsize = Counter(groups)
    n_singleton = sum(1 for v in gsize.values() if v == 1)
    n_dup       = total_groups - n_singleton

    print(f"  Total images:        {len(items)}")
    print(f"  Unique MD5 groups:   {total_groups}")
    print(f"  Singleton groups:    {n_singleton}")
    print(f"  Duplicate groups:    {n_dup}")

    return (np.array(paths), np.array(labels), np.array(groups),
            all_md5s, hash_to_gid, total_groups, n_dup)


# ──────────────────────────────────────────────────────────────
# 三道 Gate 驗證
# ──────────────────────────────────────────────────────────────
def run_leakage_gates(fold_idx: int,
                      train_imgs, train_lbls, train_groups,
                      val_imgs,   val_lbls,   val_groups) -> dict:
    """
    三道 Gate，全部 PASS 才允許訓練。
    返回 dict，含每個 gate 的結果及 overall passed。
    """
    result = {'fold': fold_idx, 'passed': True, 'gates': {}}

    # ── Gate 1: MD5 Isolation ──────────────────────────────────
    train_md5s = {file_md5(Path(p)) for p in train_imgs}
    val_md5s   = {file_md5(Path(p)) for p in val_imgs}
    md5_overlap = train_md5s & val_md5s

    g1 = {
        'name':          'MD5 Isolation',
        'train_unique':  len(train_md5s),
        'val_unique':    len(val_md5s),
        'overlap':       len(md5_overlap),
        'passed':        len(md5_overlap) == 0,
    }
    result['gates']['gate1_md5'] = g1

    # ── Gate 2: Group ID Isolation ─────────────────────────────
    train_gids = set(int(g) for g in train_groups)
    val_gids   = set(int(g) for g in val_groups)
    gid_overlap = train_gids & val_gids

    g2 = {
        'name':          'Group Isolation',
        'train_groups':  len(train_gids),
        'val_groups':    len(val_gids),
        'overlap':       len(gid_overlap),
        'passed':        len(gid_overlap) == 0,
    }
    result['gates']['gate2_group'] = g2

    # ── Gate 3: Class Coverage ─────────────────────────────────
    train_classes = set(int(l) for l in train_lbls)
    val_classes   = set(int(l) for l in val_lbls)
    full_set      = set(range(NUM_CLASSES))
    train_missing = full_set - train_classes
    val_missing   = full_set - val_classes

    g3 = {
        'name':          'Class Coverage',
        'train_classes': len(train_classes),
        'val_classes':   len(val_classes),
        'train_missing': [CLASS_NAMES[i] for i in train_missing],
        'val_missing':   [CLASS_NAMES[i] for i in val_missing],
        'train_dist':    {CLASS_NAMES[i]: int((np.array(train_lbls) == i).sum())
                          for i in range(NUM_CLASSES)},
        'val_dist':      {CLASS_NAMES[i]: int((np.array(val_lbls) == i).sum())
                          for i in range(NUM_CLASSES)},
        'passed':        len(val_missing) == 0,   # val 必須有全部 7 class
    }
    result['gates']['gate3_class'] = g3

    # ── 彙總 ──────────────────────────────────────────────────
    all_passed = g1['passed'] and g2['passed'] and g3['passed']
    result['passed'] = all_passed
    return result


def print_gate_result(gate_result: dict):
    fold_idx = gate_result['fold']
    g1 = gate_result['gates']['gate1_md5']
    g2 = gate_result['gates']['gate2_group']
    g3 = gate_result['gates']['gate3_class']

    def icon(p): return '✅' if p else '❌'

    print(f"  Gate 1 MD5 Isolation:   {icon(g1['passed'])} "
          f"overlap={g1['overlap']}  "
          f"(train_unique={g1['train_unique']}, val_unique={g1['val_unique']})")
    print(f"  Gate 2 Group Isolation: {icon(g2['passed'])} "
          f"overlap={g2['overlap']}  "
          f"(train_grps={g2['train_groups']}, val_grps={g2['val_groups']})")
    print(f"  Gate 3 Class Coverage:  {icon(g3['passed'])} "
          f"val_missing={g3['val_missing'] or 'none'}")
    if g3['val_missing']:
        print(f"  ⚠️  Missing classes in val: {g3['val_missing']}")
    print(f"  → Fold {fold_idx} Gate Result: {'✅ ALL PASS' if gate_result['passed'] else '❌ BLOCKED'}")


# ──────────────────────────────────────────────────────────────
# 建立 Fold 臨時目錄
# ──────────────────────────────────────────────────────────────
def build_fold_dir(fold_dir: Path, train_imgs, train_lbls, val_imgs, val_lbls):
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    for cls_name in CLASS_NAMES:
        (fold_dir / 'train' / cls_name).mkdir(parents=True, exist_ok=True)
        (fold_dir / 'val'   / cls_name).mkdir(parents=True, exist_ok=True)

    def _copy(src, split, cls_idx):
        src_p = Path(src)
        dst_name = f"{CLASS_NAMES[cls_idx]}_{src_p.name}"
        shutil.copy2(src, fold_dir / split / CLASS_NAMES[cls_idx] / dst_name)

    for img, lbl in zip(train_imgs, train_lbls):
        _copy(img, 'train', lbl)
    for img, lbl in zip(val_imgs, val_lbls):
        _copy(img, 'val', lbl)


def _fold_flat_cfg(cfg, fold_dir, fold_run_name, fold_idx):
    aug = cfg.get('augmentation', {})
    aug_str = ','.join(f"{k}={v}" for k, v in aug.items()) if isinstance(aug, dict) else str(aug)
    return {
        'exp_id':          cfg['experiment']['id'],
        'experiment_name': cfg['experiment']['name'],
        'task':            cfg['experiment']['type'],
        'model':           cfg['model']['name'],
        'framework':       cfg['model']['framework'],
        'dataset':         str(fold_dir),
        'dataset_version': cfg['dataset']['version'],
        'run_name':        fold_run_name,
        'epochs':          cfg['training']['epochs'],
        'batch_size':      cfg['training']['batch'],
        'img_size':        cfg['training']['imgsz'],
        'seed':            cfg['training']['seed'],
        'fold':            fold_idx,
        'optimizer':       cfg['optimizer']['name'],
        'lr0':             cfg['optimizer']['lr0'],
        'weight_decay':    cfg['optimizer']['weight_decay'],
        'augmentation':    aug,
        'augmentation_note': aug_str,
        'notes':           f"GroupKFold Fold{fold_idx}: {fold_run_name}",
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def run_group_kfold(cfg: dict, k: int = K) -> dict:
    dataset_path = BASE_DIR / cfg['dataset']['path']
    orig_exp_id  = cfg['experiment']['id']
    exp_id       = f"{orig_exp_id}-GKF"
    exp_name     = cfg['experiment']['name']
    model_name   = cfg['model']['name']
    framework    = cfg['model']['framework']
    seed         = cfg['training']['seed']

    print(f"\n{'='*62}")
    print(f"🔁  Phase 6.6: StratifiedGroupKFold (MD5-group-aware)")
    print(f"    Experiment: {exp_id}  |  Model: {model_name}")
    print(f"    Seed: {seed}  |  k={k}")
    print(f"    Zero-Leakage Gates: MD5 + GroupID + ClassCoverage")
    print(f"    Blind Test (test/) excluded.")
    print(f"{'='*62}")

    # ── Step 1: Dev Set + MD5 Groups ─────────────────────────
    (all_paths, all_labels, all_groups,
     all_md5s, hash_to_gid, total_groups,
     n_dup) = collect_dev_set_with_groups(dataset_path)

    # ── Step 2: StratifiedGroupKFold ─────────────────────────
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_results = []
    gate_results = []
    total_md5_overlap = 0
    total_gid_overlap = 0

    for fold_idx, (train_idx, val_idx) in enumerate(
            sgkf.split(all_paths, all_labels, groups=all_groups), 1):

        fold_run_name = f'{exp_id}_kfold{fold_idx}'
        print(f"\n{'─'*62}")
        print(f"  Fold {fold_idx}/{k}  |  {fold_run_name}")

        train_imgs   = all_paths[train_idx].tolist()
        train_lbls   = all_labels[train_idx].tolist()
        train_groups = all_groups[train_idx].tolist()
        val_imgs     = all_paths[val_idx].tolist()
        val_lbls     = all_labels[val_idx].tolist()
        val_groups_f = all_groups[val_idx].tolist()

        print(f"  Train: {len(train_imgs)}  Val: {len(val_imgs)}  "
              f"({len(val_imgs)/len(all_paths)*100:.1f}% val)")

        # ── 三道 Gate ─────────────────────────────────────────
        gate = run_leakage_gates(
            fold_idx,
            train_imgs, train_lbls, train_groups,
            val_imgs,   val_lbls,   val_groups_f
        )
        gate_results.append(gate)
        print_gate_result(gate)

        total_md5_overlap += gate['gates']['gate1_md5']['overlap']
        total_gid_overlap += gate['gates']['gate2_group']['overlap']

        if not gate['passed']:
            reason = []
            if not gate['gates']['gate1_md5']['passed']:
                reason.append(f"MD5 overlap={gate['gates']['gate1_md5']['overlap']}")
            if not gate['gates']['gate2_group']['passed']:
                reason.append(f"group overlap={gate['gates']['gate2_group']['overlap']}")
            if not gate['gates']['gate3_class']['passed']:
                reason.append(f"missing classes={gate['gates']['gate3_class']['val_missing']}")
            err_msg = '; '.join(reason)
            log_failed(exp_id, {**cfg, 'fold': fold_idx}, error=f"Gate FAIL: {err_msg}")
            fold_results.append({
                'fold':              fold_idx,
                'train_size':        len(train_imgs),
                'val_size':          len(val_imgs),
                'gate_passed':       False,
                'gate_failure':      err_msg,
                'accuracy':          0,
                'macro_f1':          0,
                'weighted_f1':       0,
                'stab_wound_recall': 0,
                'status':            'BLOCKED_BY_GATE',
            })
            continue

        # ── 建立 Fold 目錄 ────────────────────────────────────
        fold_dir = TEMP_DIR / fold_run_name
        build_fold_dir(fold_dir, train_imgs, train_lbls, val_imgs, val_lbls)

        flat_cfg = _fold_flat_cfg(cfg, fold_dir, fold_run_name, fold_idx)

        # ── log_start ─────────────────────────────────────────
        log_start(exp_id, {**cfg, 'fold': fold_idx})

        # ── 訓練 ─────────────────────────────────────────────
        t0 = time.time()
        weights_path = ''
        train_record = {}
        try:
            is_torch = (framework == 'torch' or
                        model_name.replace('.pt', '') in TORCH_MODELS)
            if is_torch:
                from train_cls_torch import train_torch_cls
                train_record = train_torch_cls(flat_cfg)
            else:
                from train_cls_yolo import train_yolo_cls
                train_record = train_yolo_cls(flat_cfg)
            weights_path = train_record.get('weights_path', '')
        except Exception as e:
            log_failed(exp_id, {**cfg, 'fold': fold_idx}, error=str(e))
            print(f"  ❌ Fold {fold_idx} FAILED: {e}")
            import traceback; traceback.print_exc()
            fold_results.append({'fold': fold_idx, 'gate_passed': True,
                                  'accuracy': 0, 'macro_f1': 0,
                                  'weighted_f1': 0, 'stab_wound_recall': 0,
                                  'status': 'FAILED'})
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
            print(f"  ⚠️  Evaluate failed: {e}")

        metrics['training_time_min'] = round(elapsed, 1)
        metrics['model_path']    = weights_path
        metrics['best_epoch']    = train_record.get('best_epoch', '')
        metrics['train_samples'] = len(train_imgs)
        metrics['val_samples']   = len(val_imgs)

        fold_result = {
            'fold':               fold_idx,
            'train_size':         len(train_imgs),
            'val_size':           len(val_imgs),
            'gate_passed':        True,
            'md5_overlap':        gate['gates']['gate1_md5']['overlap'],
            'gid_overlap':        gate['gates']['gate2_group']['overlap'],
            'val_class_dist':     gate['gates']['gate3_class']['val_dist'],
            'accuracy':           metrics.get('accuracy', 0),
            'macro_f1':           metrics.get('macro_f1', 0),
            'weighted_f1':        metrics.get('weighted_f1', 0),
            'stab_wound_recall':  metrics.get('stab_wound_recall', 0),
            'training_time_min':  round(elapsed, 1),
            'status':             'done',
        }
        fold_results.append(fold_result)

        log_done(exp_id, {**cfg, 'fold': fold_idx}, metrics)
        print(f"  ✅ Fold {fold_idx}: Acc={metrics.get('accuracy',0):.2f}%  "
              f"F1={metrics.get('macro_f1',0):.2f}%  "
              f"StabRec={metrics.get('stab_wound_recall',0):.2f}%")

    # ── 清理 ─────────────────────────────────────────────────
    if TEMP_DIR.exists():
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception:
            pass

    # ── 彙總 ─────────────────────────────────────────────────
    done_folds = [r for r in fold_results if r.get('status') == 'done']
    all_gates_pass = all(g['passed'] for g in gate_results)
    leakage_status = 'PASS' if total_md5_overlap == 0 and total_gid_overlap == 0 else 'FAIL'

    summary_base = {
        'accuracy_mean':    0, 'accuracy_std':    0,
        'macro_f1_mean':    0, 'macro_f1_std':    0,
        'weighted_f1_mean': 0, 'weighted_f1_std': 0,
        'stab_recall_mean': 0, 'stab_recall_std': 0,
    }
    if done_folds:
        accs  = [r['accuracy']          for r in done_folds]
        f1s   = [r['macro_f1']          for r in done_folds]
        wf1s  = [r['weighted_f1']       for r in done_folds]
        stabs = [r['stab_wound_recall'] for r in done_folds]
        summary_base = {
            'accuracy_mean':    round(float(np.mean(accs)),  2),
            'accuracy_std':     round(float(np.std(accs)),   2),
            'macro_f1_mean':    round(float(np.mean(f1s)),   2),
            'macro_f1_std':     round(float(np.std(f1s)),    2),
            'weighted_f1_mean': round(float(np.mean(wf1s)),  2),
            'weighted_f1_std':  round(float(np.std(wf1s)),   2),
            'stab_recall_mean': round(float(np.mean(stabs)), 2),
            'stab_recall_std':  round(float(np.std(stabs)),  2),
        }

    summary = {
        # 實驗識別
        'experiment_id':    exp_id,
        'original_exp_id':  orig_exp_id,
        'model':            model_name,
        'framework':        framework,
        'k_folds':          k,
        'completed_folds':  len(done_folds),
        'seed':             seed,
        # 指標（Mean ± SD）
        **summary_base,
        # 資料完整性
        'data_integrity': {
            'method':                  'MD5 + StratifiedGroupKFold',
            'total_images':            len(all_paths),
            'unique_md5_count':        total_groups,
            'duplicate_group_count':   n_dup,
            'md5_train_val_overlap':   total_md5_overlap,
            'gid_train_val_overlap':   total_gid_overlap,
            'all_7_classes_in_val':    all(g['gates']['gate3_class']['passed']
                                           for g in gate_results if g['passed'] or True),
            'leakage_status':          leakage_status,
            'gates_all_pass':          all_gates_pass,
        },
        # Fold 明細
        'fold_results':    fold_results,
        'gate_results':    [{
            'fold':     g['fold'],
            'passed':   g['passed'],
            'md5_overlap': g['gates']['gate1_md5']['overlap'],
            'gid_overlap': g['gates']['gate2_group']['overlap'],
            'val_missing': g['gates']['gate3_class']['val_missing'],
        } for g in gate_results],
        # 歷史記錄（保留被 invalidate 的結果）
        'invalidated_result': {
            'experiment_id': orig_exp_id,
            'method':        'StratifiedKFold (image-level)',
            'accuracy_mean': 98.34,
            'accuracy_std':  1.21,
            'status':        'INVALIDATED_BY_DATA_INTEGRITY_AUDIT',
            'reason':        '178/206 duplicate groups crossed Train/Val boundary',
        },
    }

    # 儲存 JSON
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = STATS_DIR / f'{exp_id}_gkfold_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 最終輸出 ──────────────────────────────────────────────
    sep = '=' * 62
    print(f"\n{sep}")
    print(f"{'✅' if all_gates_pass else '⚠️ '}  Phase 6.6 Complete: {exp_id}")
    print(f"    Method: StratifiedGroupKFold + 3-Gate Zero-Leakage")
    print(f"    All gates passed: {all_gates_pass}")
    print(f"{'─'*62}")
    print(f"    {'Fold':<6} {'Train':>6} {'Val':>6} "
          f"{'MD5ovlp':>8} {'GIDovlp':>8} "
          f"{'Acc%':>7} {'F1%':>7} {'StbRec%':>8}")
    for r in fold_results:
        g = next((x for x in gate_results if x['fold'] == r['fold']), {})
        status_icon = {'done': '', 'FAILED': '❌', 'BLOCKED_BY_GATE': '🚫'}.get(r.get('status',''), '?')
        print(f"    {r['fold']:<6} {r.get('train_size','?'):>6} {r.get('val_size','?'):>6} "
              f"{r.get('md5_overlap','-'):>8} {r.get('gid_overlap','-'):>8} "
              f"{r.get('accuracy',0):>7.2f} {r.get('macro_f1',0):>7.2f} "
              f"{r.get('stab_wound_recall',0):>8.2f} {status_icon}")
    print(f"{'─'*62}")
    if done_folds:
        print(f"    {'Mean':<6} {'':>6} {'':>6} {'0':>8} {'0':>8} "
              f"{summary['accuracy_mean']:>7.2f} {summary['macro_f1_mean']:>7.2f} "
              f"{summary['stab_recall_mean']:>8.2f}")
        print(f"    {'±SD':<6} {'':>6} {'':>6} {'':>8} {'':>8} "
              f"{summary['accuracy_std']:>7.2f} {summary['macro_f1_std']:>7.2f} "
              f"{summary['stab_recall_std']:>8.2f}")
    print(f"{'─'*62}")
    print(f"    Data Integrity: {leakage_status}")
    print(f"    MD5 overlap (total across folds): {total_md5_overlap}")
    print(f"    GID overlap (total across folds): {total_gid_overlap}")
    if done_folds:
        print(f"\n    📝 論文格式 (Leakage-Free, MD5-GroupKFold):")
        print(f"       Top-1  = {summary['accuracy_mean']}% ± {summary['accuracy_std']}%  "
              f"({k}-Fold, seed={seed})")
        print(f"       Macro-F1 = {summary['macro_f1_mean']}% ± {summary['macro_f1_std']}%")
        print(f"       Stab Recall = {summary['stab_recall_mean']}% ± {summary['stab_recall_std']}%")
    print(f"\n    ⚠️  Invalidated (reference only):")
    print(f"       C-Arch-05 StratifiedKFold: 98.34% ± 1.21% [INVALIDATED]")
    print(f"    💾 {json_path}")
    print(sep)

    return summary
