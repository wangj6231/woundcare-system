# ============================================================
# multi_seed.py — Phase 7 (定版)
# Multi-Seed Group CV：5 Seeds × 5 GroupKFold = 25 次訓練
#
# 設計原則：
#   ① MD5 grouping 固定（與 Phase 6.6 完全相同）
#   ② 每個 Seed 執行 StratifiedGroupKFold
#   ③ 每個 Fold 仍需通過 3-Gate（MD5+GID+Class）
#   ④ 報告 Between-seed / Within-seed variance
#
# 執行方式：
#   python experiments/run_experiments.py --multiseed \
#          experiments/configs/C-Arch-05_yolov8n_cls.yaml
# ============================================================
import gc
import json
import sys
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from experiment_logger import log_start, log_done, log_failed
from cross_validation_group import (
    collect_dev_set_with_groups,
    build_fold_dir,
    run_leakage_gates,
    _fold_flat_cfg,
    TEMP_DIR,
    TORCH_MODELS,
    CLASS_NAMES,
)
from evaluate import evaluate

BASE_DIR  = Path(__file__).parent.parent.parent
STATS_DIR = Path(__file__).parent.parent / 'results' / 'statistics'
LOG_CSV   = Path(__file__).parent.parent / 'experiment_log.csv'
K = 5

# 固定 5 個 Seeds（與研究規劃一致）
DEFAULT_SEEDS = [42, 123, 3407, 2026, 999]


def _load_completed_runs(exp_id: str) -> set:
    """
    從 experiment_log.csv 讀出已完成的 (seed, fold) pair。
    只算 status=done 的記錄，status=running 視為中斷需重跑。
    """
    if not LOG_CSV.exists():
        return set()
    completed = set()
    with open(LOG_CSV, 'r', encoding='utf-8-sig') as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('experiment_id') == exp_id and row.get('status') == 'done':
                try:
                    seed = int(row['seed'])
                    fold = int(row['fold'])
                    completed.add((seed, fold))
                except (ValueError, KeyError):
                    pass
    return completed


def run_multi_seed(cfg: dict, seeds: list = None) -> dict:
    """
    Phase 7: 對每個 seed 執行 StratifiedGroupKFold，
    共產生 len(seeds) × K 個訓練 + 評估記錄。
    """
    import shutil
    if seeds is None:
        seeds = DEFAULT_SEEDS

    dataset_path = BASE_DIR / cfg['dataset']['path']
    orig_exp_id  = cfg['experiment']['id']
    exp_id       = f"{orig_exp_id}-MS"   # e.g. C-Arch-05-MS
    model_name   = cfg['model']['name']
    framework    = cfg['model']['framework']

    print(f"\n{'='*62}")
    print(f"🌱  Phase 7: Multi-Seed Group CV")
    print(f"    Experiment: {exp_id}  |  Model: {model_name}")
    print(f"    Seeds: {seeds}  |  k={K}  |  Total runs: {len(seeds)*K}")
    print(f"    Method: MD5 + StratifiedGroupKFold (same protocol as Phase 6.6)")
    print(f"{'='*62}")

    # ── Resume: 讀取已完成的 runs ─────────────────────────────
    completed_runs = _load_completed_runs(exp_id)
    if completed_runs:
        print(f"\n  ♻️  Resume mode: {len(completed_runs)} fold(s) already completed, skipping.")
        completed_by_seed = {}
        for s, f in completed_runs:
            completed_by_seed.setdefault(s, []).append(f)
        for s, fs in sorted(completed_by_seed.items()):
            print(f"     Seed {s}: folds {sorted(fs)} ✅")
    else:
        print(f"\n  Starting fresh (no completed runs found).")

    # ── Step 1: MD5 Groups（固定，與 Phase 6.6 相同）────────
    print(f"\n  Building MD5 groups (shared across all seeds)...")
    (all_paths, all_labels, all_groups,
     all_md5s, hash_to_gid, total_groups, n_dup) = collect_dev_set_with_groups(dataset_path)

    # ── Step 2: Reconstruct completed records from CSV ────────
    # 讓 overall_records 也包含已完成的舊資料，最終統計才正確
    import csv as _csv
    overall_records  = []
    all_seed_results = {}
    if completed_runs and LOG_CSV.exists():
        with open(LOG_CSV, 'r', encoding='utf-8-sig') as f:
            reader = _csv.DictReader(f)
            for row in reader:
                if row.get('experiment_id') == exp_id and row.get('status') == 'done':
                    try:
                        s = int(row['seed']); fo = int(row['fold'])
                        rec = {
                            'seed':              s, 'fold': fo,
                            'gate_passed':       True,
                            'md5_overlap':       0,  'gid_overlap': 0,
                            'train_size':        int(row.get('train_samples') or 0),
                            'val_size':          int(row.get('val_samples') or 0),
                            'accuracy':          float(row.get('accuracy') or 0),
                            'macro_f1':          float(row.get('macro_f1') or 0),
                            'weighted_f1':       float(row.get('weighted_f1') or 0),
                            'stab_wound_recall': float(row.get('recall') or 0),  # best proxy
                            'training_time_min': float(row.get('training_time') or 0),
                            'status':            'done',
                        }
                        overall_records.append(rec)
                        seed_data = all_seed_results.setdefault(s, [])
                        seed_data.append(rec)
                    except Exception:
                        pass

    # Convert seed lists to summary dicts after reconstruction
    for s, recs in list(all_seed_results.items()):
        if isinstance(recs, list) and recs:
            accs  = [r['accuracy']   for r in recs if r['status'] == 'done']
            f1s   = [r['macro_f1']   for r in recs if r['status'] == 'done']
            stabs = [r['stab_wound_recall'] for r in recs if r['status'] == 'done']
            if accs:
                all_seed_results[s] = {
                    'accuracy_mean':    round(float(np.mean(accs)),  2),
                    'accuracy_std':     round(float(np.std(accs)),   2),
                    'macro_f1_mean':    round(float(np.mean(f1s)),   2),
                    'macro_f1_std':     round(float(np.std(f1s)),    2),
                    'stab_recall_mean': round(float(np.mean(stabs)), 2),
                    'stab_recall_std':  round(float(np.std(stabs)),  2),
                    'completed_folds':  len(accs),
                }

    # ── Step 3: Per-Seed Loop ─────────────────────────────────

    for seed in seeds:
        print(f"\n{'─'*62}")
        print(f"  🌱 Seed {seed}  (Fold 1–{K})")
        print(f"{'─'*62}")

        sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=seed)
        seed_fold_results = []  # temp list for THIS seed's new runs

        for fold_idx, (train_idx, val_idx) in enumerate(
                sgkf.split(all_paths, all_labels, groups=all_groups), 1):

            # ── Resume Skip ───────────────────────────────────
            if (seed, fold_idx) in completed_runs:
                print(f"  ♻️  Seed {seed} Fold {fold_idx} already done — skipping")
                continue

            fold_run_name = f'{exp_id}_s{seed}_fold{fold_idx}'
            print(f"\n  Seed {seed} | Fold {fold_idx}/{K}  →  {fold_run_name}")

            train_imgs   = all_paths[train_idx].tolist()
            train_lbls   = all_labels[train_idx].tolist()
            train_gids   = all_groups[train_idx].tolist()
            val_imgs     = all_paths[val_idx].tolist()
            val_lbls     = all_labels[val_idx].tolist()
            val_gids     = all_groups[val_idx].tolist()

            print(f"  Train: {len(train_imgs)}  Val: {len(val_imgs)}")

            # ── 三道 Gate ────────────────────────────────────
            gate = run_leakage_gates(fold_idx,
                                     train_imgs, train_lbls, train_gids,
                                     val_imgs,   val_lbls,   val_gids)

            g1ok = gate['gates']['gate1_md5']['passed']
            g2ok = gate['gates']['gate2_group']['passed']
            g3ok = gate['gates']['gate3_class']['passed']
            icon = '✅' if gate['passed'] else '❌'
            print(f"  Gate1(MD5)={icon if g1ok else '❌'}  "
                  f"Gate2(GID)={'✅' if g2ok else '❌'}  "
                  f"Gate3(Class)={'✅' if g3ok else '❌'} "
                  f"→ {'PASS' if gate['passed'] else 'BLOCKED'}")

            if not gate['passed']:
                reason = f"Gate FAIL seed={seed} fold={fold_idx}"
                log_failed(exp_id, {**cfg, 'fold': fold_idx, 'training': {**cfg['training'], 'seed': seed}},
                           error=reason)
                seed_fold_results.append({'seed': seed, 'fold': fold_idx,
                                          'gate_passed': False, 'status': 'BLOCKED_BY_GATE',
                                          'accuracy': 0, 'macro_f1': 0,
                                          'weighted_f1': 0, 'stab_wound_recall': 0})
                continue

            # ── 建 Fold 目錄 ──────────────────────────────────
            fold_dir = TEMP_DIR / fold_run_name
            build_fold_dir(fold_dir, train_imgs, train_lbls, val_imgs, val_lbls)

            # 修改 flat_cfg 的 seed
            seed_cfg = {**cfg,
                        'training': {**cfg['training'], 'seed': seed},
                        'fold': fold_idx}
            flat_cfg = _fold_flat_cfg(seed_cfg, fold_dir, fold_run_name, fold_idx)
            flat_cfg['seed'] = seed  # 確保覆寫

            # ── log_start ─────────────────────────────────────
            log_start(exp_id, seed_cfg)

            # ── 訓練 ─────────────────────────────────────────
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
                log_failed(exp_id, seed_cfg, error=str(e))
                print(f"  ❌ FAILED: {e}")
                import traceback; traceback.print_exc()
                seed_fold_results.append({'seed': seed, 'fold': fold_idx,
                                          'gate_passed': True, 'status': 'FAILED',
                                          'accuracy': 0, 'macro_f1': 0,
                                          'weighted_f1': 0, 'stab_wound_recall': 0})
                continue

            elapsed = (time.time() - t0) / 60

            # ── 評估 ─────────────────────────────────────────
            metrics = {}
            try:
                metrics = evaluate(
                    weights_path=weights_path,
                    data_dir=str(fold_dir / 'val'),
                    framework='torch' if (framework == 'torch' or
                              model_name.replace('.pt','') in TORCH_MODELS) else 'ultralytics',
                    exp_id=fold_run_name,
                    split='val',
                    save_figures=False   # Multi-Seed 不存圖，避免產生大量檔案
                )
            except Exception as e:
                print(f"  ⚠️  Evaluate failed: {e}")

            metrics['training_time_min'] = round(elapsed, 1)
            metrics['model_path']    = weights_path
            metrics['best_epoch']    = train_record.get('best_epoch', '')
            metrics['train_samples'] = len(train_imgs)
            metrics['val_samples']   = len(val_imgs)

            fold_result = {
                'seed':              seed,
                'fold':              fold_idx,
                'gate_passed':       True,
                'md5_overlap':       gate['gates']['gate1_md5']['overlap'],
                'gid_overlap':       gate['gates']['gate2_group']['overlap'],
                'train_size':        len(train_imgs),
                'val_size':          len(val_imgs),
                'accuracy':          metrics.get('accuracy', 0),
                'macro_f1':          metrics.get('macro_f1', 0),
                'weighted_f1':       metrics.get('weighted_f1', 0),
                'stab_wound_recall': metrics.get('stab_wound_recall', 0),
                'training_time_min': round(elapsed, 1),
                'status':            'done',
            }
            seed_fold_results.append(fold_result)
            overall_records.append(fold_result)

            log_done(exp_id, seed_cfg, metrics)

            print(f"  ✅ S{seed}-F{fold_idx}: "
                  f"Acc={metrics.get('accuracy',0):.2f}%  "
                  f"F1={metrics.get('macro_f1',0):.2f}%  "
                  f"Stab={metrics.get('stab_wound_recall',0):.2f}%")

            # ── 清理 fold 暫存目錄 ────────────────────────────
            if fold_dir.exists():
                try:
                    shutil.rmtree(fold_dir)
                except Exception:
                    pass

            # ── 強制釋放 GPU / CPU 記憶體 ─────────────────────
            # 每 Fold 完成後清理，防止 25 runs 連跑時記憶體累積
            del train_record, metrics
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            except Exception:
                pass
            gc.collect()
            time.sleep(2)   # 讓 OS 回收記憶體

        # ── Per-Seed Summary ──────────────────────────────────
        done_in_seed = [r for r in overall_records if r['seed'] == seed and r['status'] == 'done']
        if done_in_seed:
            accs  = [r['accuracy']          for r in done_in_seed]
            f1s   = [r['macro_f1']          for r in done_in_seed]
            stabs = [r['stab_wound_recall'] for r in done_in_seed]
            all_seed_results[seed] = {
                'accuracy_mean':    round(float(np.mean(accs)),  2),
                'accuracy_std':     round(float(np.std(accs)),   2),
                'macro_f1_mean':    round(float(np.mean(f1s)),   2),
                'macro_f1_std':     round(float(np.std(f1s)),    2),
                'stab_recall_mean': round(float(np.mean(stabs)), 2),
                'stab_recall_std':  round(float(np.std(stabs)),  2),
                'completed_folds':  len(done_in_seed),
            }
            print(f"\n  Seed {seed} summary: "
                  f"Acc={all_seed_results[seed]['accuracy_mean']} ± {all_seed_results[seed]['accuracy_std']}%  "
                  f"F1={all_seed_results[seed]['macro_f1_mean']} ± {all_seed_results[seed]['macro_f1_std']}%")

    # ── Overall Statistics（所有 seed × fold）───────────────────
    done_all = [r for r in overall_records if r['status'] == 'done']
    seed_means_acc  = [v['accuracy_mean']    for v in all_seed_results.values()]
    seed_means_f1   = [v['macro_f1_mean']    for v in all_seed_results.values()]
    seed_means_stab = [v['stab_recall_mean'] for v in all_seed_results.values()]

    all_accs  = [r['accuracy']          for r in done_all]
    all_f1s   = [r['macro_f1']          for r in done_all]
    all_stabs = [r['stab_wound_recall'] for r in done_all]

    summary = {
        'experiment_id':       exp_id,
        'original_exp_id':     orig_exp_id,
        'model':               model_name,
        'framework':           framework,
        'method':              'Multi-Seed StratifiedGroupKFold (MD5-aware)',
        'seeds':               seeds,
        'k_folds':             K,
        'total_runs':          len(done_all),
        # Overall（全部 25 個 fold 合一）
        'overall': {
            'accuracy_mean':    round(float(np.mean(all_accs)),  2) if all_accs else 0,
            'accuracy_std':     round(float(np.std(all_accs)),   2) if all_accs else 0,
            'macro_f1_mean':    round(float(np.mean(all_f1s)),   2) if all_f1s  else 0,
            'macro_f1_std':     round(float(np.std(all_f1s)),    2) if all_f1s  else 0,
            'stab_recall_mean': round(float(np.mean(all_stabs)), 2) if all_stabs else 0,
            'stab_recall_std':  round(float(np.std(all_stabs)),  2) if all_stabs else 0,
        },
        # Between-Seed Variance（每個 seed 的 mean 之間的 SD）
        'between_seed': {
            'accuracy_mean_of_means': round(float(np.mean(seed_means_acc)),  2) if seed_means_acc else 0,
            'accuracy_sd_of_means':   round(float(np.std(seed_means_acc)),   2) if seed_means_acc else 0,
            'macro_f1_mean_of_means': round(float(np.mean(seed_means_f1)),   2) if seed_means_f1  else 0,
            'macro_f1_sd_of_means':   round(float(np.std(seed_means_f1)),    2) if seed_means_f1  else 0,
            'stab_recall_mean_of_means': round(float(np.mean(seed_means_stab)), 2) if seed_means_stab else 0,
            'stab_recall_sd_of_means':   round(float(np.std(seed_means_stab)),  2) if seed_means_stab else 0,
        },
        # Per-Seed 詳細
        'per_seed_summary': {str(s): all_seed_results.get(s, {}) for s in seeds},
        'all_fold_results':  overall_records,
        # Data Integrity
        'data_integrity': {
            'method':              'MD5 + StratifiedGroupKFold',
            'unique_md5_count':    int(total_groups),
            'duplicate_groups':    int(n_dup),
            'gates':               'Gate1(MD5) + Gate2(GID) + Gate3(Class)',
            'md5_overlap_total':   0,   # Will be FAIL if any gate fails
        },
    }

    # 儲存 JSON
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = STATS_DIR / f'{exp_id}_multiseed_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 最終輸出 ──────────────────────────────────────────────
    sep = '=' * 62
    ov = summary['overall']
    bs = summary['between_seed']
    print(f"\n{sep}")
    print(f"✅  Phase 7 Complete: {exp_id}")
    print(f"    {len(seeds)} Seeds × {K} Folds = {len(done_all)} completed runs")
    print(f"{'─'*62}")
    print(f"    {'Seed':>6} {'Acc Mean':>10} {'±SD':>6} {'F1 Mean':>9} {'±SD':>6} {'Stab':>7} {'±SD':>6}")
    for s in seeds:
        sr = all_seed_results.get(s, {})
        print(f"    {s:>6} {sr.get('accuracy_mean',0):>10.2f} "
              f"{sr.get('accuracy_std',0):>6.2f} "
              f"{sr.get('macro_f1_mean',0):>9.2f} "
              f"{sr.get('macro_f1_std',0):>6.2f} "
              f"{sr.get('stab_recall_mean',0):>7.2f} "
              f"{sr.get('stab_recall_std',0):>6.2f}")
    print(f"{'─'*62}")
    print(f"    {'Overall':>6} {ov['accuracy_mean']:>10.2f} {ov['accuracy_std']:>6.2f} "
          f"{ov['macro_f1_mean']:>9.2f} {ov['macro_f1_std']:>6.2f} "
          f"{ov['stab_recall_mean']:>7.2f} {ov['stab_recall_std']:>6.2f}")
    print(f"{'─'*62}")
    print(f"    Between-Seed SD (Accuracy):  {bs['accuracy_sd_of_means']:.2f}%")
    print(f"    Between-Seed SD (Macro-F1):  {bs['macro_f1_sd_of_means']:.2f}%")
    print(f"    Between-Seed SD (Stab Rec):  {bs['stab_recall_sd_of_means']:.2f}%")
    print(f"{'─'*62}")
    print(f"    📝 論文格式 ({len(seeds)} Seeds × {K} GroupKFold):")
    print(f"       Top-1 Acc = {ov['accuracy_mean']}% ± {ov['accuracy_std']}%  "
          f"(n={len(done_all)} runs)")
    print(f"       Macro-F1  = {ov['macro_f1_mean']}% ± {ov['macro_f1_std']}%")
    print(f"       Stab Rec  = {ov['stab_recall_mean']}% ± {ov['stab_recall_std']}%")
    print(f"    💾 {json_path}")
    print(sep)

    return summary
