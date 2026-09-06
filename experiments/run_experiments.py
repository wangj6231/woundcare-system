# ============================================================
# run_experiments.py — Orchestrator (定版 v3)
#
# 職責：控制整個實驗流程（不含訓練細節）
# 模式：development（預設）/ blind_test（需明確授權）
#
# 使用方式：
#   python experiments/run_experiments.py --list
#   python experiments/run_experiments.py --batch C-Arch
#   python experiments/run_experiments.py --single configs/C-Arch-01_resnet50.yaml
#   python experiments/run_experiments.py --kfold  configs/C-Arch-05_yolov8n_cls.yaml
#   python experiments/run_experiments.py --multiseed configs/C-Arch-05_yolov8n_cls.yaml
#
# Blind Test 保護機制（必須同時加兩個 flag）：
#   python experiments/run_experiments.py --mode blind_test --allow-blind-test \
#          --single configs/C-Arch-05_yolov8n_cls.yaml
# ============================================================
import argparse
import glob
import sys
import time
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SCRIPTS  = BASE_DIR / 'experiments' / 'scripts'
CONFIGS  = BASE_DIR / 'experiments' / 'configs'
sys.path.insert(0, str(SCRIPTS))

TORCH_MODEL_NAMES = {'resnet50', 'efficientnet_b0', 'mobilenet_v3_large', 'vit_s_16'}


# ──────────────────────────────────────────────────────────────
# Blind Test 保護
# ──────────────────────────────────────────────────────────────
BLIND_TEST_SPLIT = 'test'
DEVELOPMENT_SPLIT = 'val'

def _guard_blind_test(mode: str, allow_flag: bool) -> str:
    """
    回傳本次要評估的 split 名稱。
    blind_test 模式必須同時加 --allow-blind-test，否則強制退出。
    """
    if mode == 'blind_test':
        if not allow_flag:
            print("\n" + "="*62)
            print("🚫  BLIND TEST BLOCKED")
            print("    Blind Test is reserved for final model evaluation only.")
            print("    To unlock, add: --allow-blind-test")
            print("    Example:")
            print("      python run_experiments.py --mode blind_test \\")
            print("             --allow-blind-test --single configs/xxx.yaml")
            print("="*62)
            sys.exit(1)
        print("\n⚠️  BLIND TEST MODE ACTIVE — this is your final evaluation.")
        print("    Results will be logged. This cannot be undone.\n")
        return BLIND_TEST_SPLIT
    return DEVELOPMENT_SPLIT


# ──────────────────────────────────────────────────────────────
# YAML 工具
# ──────────────────────────────────────────────────────────────
def _load_cfg(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _cfg_to_flat(cfg: dict, override_seed: int = None, override_fold: int = None) -> dict:
    """巢狀 YAML → 扁平 dict（供 train_cls_*.py 使用）"""
    aug = cfg.get('augmentation', {})
    aug_str = ','.join(f"{k}={v}" for k, v in aug.items()) if isinstance(aug, dict) else str(aug)
    exp_cfg   = cfg.get('experiment', {})
    model_cfg = cfg.get('model', {})
    ds_cfg    = cfg.get('dataset', {})
    train_cfg = cfg.get('training', {})
    opt_cfg   = cfg.get('optimizer', {})

    seed = override_seed if override_seed is not None else train_cfg.get('seed', 42)
    fold = override_fold if override_fold is not None else 0

    return {
        'exp_id':          exp_cfg.get('id', 'unknown'),
        'experiment_name': exp_cfg.get('name', ''),
        'task':            exp_cfg.get('type', 'classification'),
        'model':           model_cfg.get('name', ''),
        'framework':       model_cfg.get('framework', 'ultralytics'),
        'dataset':         str(BASE_DIR / ds_cfg.get('path', '')),
        'dataset_version': ds_cfg.get('version', ''),
        'run_name':        f"{exp_cfg.get('id')}_{exp_cfg.get('name')}_s{seed}_f{fold}",
        'epochs':          train_cfg.get('epochs', 150),
        'batch_size':      train_cfg.get('batch', 16),
        'img_size':        train_cfg.get('imgsz', 224),
        'seed':            seed,
        'fold':            fold,
        'optimizer':       opt_cfg.get('name', 'auto'),
        'lr0':             opt_cfg.get('lr0', 0.01),
        'weight_decay':    opt_cfg.get('weight_decay', 0.0005),
        'augmentation':    aug,
        'augmentation_note': aug_str,
        'notes':           cfg.get('notes', ''),
    }


# ──────────────────────────────────────────────────────────────
# 單一實驗（核心）
# ──────────────────────────────────────────────────────────────
def run_single(config_path: str, eval_split: str = 'val',
               override_seed: int = None, override_fold: int = None) -> dict:
    """
    完整實驗流程：
      1. 讀 YAML → 建立 flat cfg
      2. log_start (status=running)
      3. 訓練
      4. evaluate
      5. log_done / log_failed
    """
    from experiment_logger import log_start, log_done, log_failed
    from evaluate import evaluate

    cfg      = _load_cfg(config_path)
    flat_cfg = _cfg_to_flat(cfg, override_seed=override_seed, override_fold=override_fold)
    exp_id   = flat_cfg['exp_id']
    model_name = flat_cfg['model']
    framework  = flat_cfg['framework']
    dataset_path = Path(flat_cfg['dataset'])

    print(f"\n{'='*62}")
    print(f"▶  {exp_id}  |  {model_name}  |  seed={flat_cfg['seed']}  fold={flat_cfg['fold']}")
    print(f"   split={eval_split}  |  epochs={flat_cfg['epochs']}")
    print(f"{'='*62}")

    # ── Step 1: log_start ─────────────────────────────────────
    log_start(exp_id, cfg)

    # ── Step 2: Train ─────────────────────────────────────────
    t0 = time.time()
    weights_path = ''
    train_record = {}

    try:
        is_torch = framework == 'torch' or model_name.replace('.pt', '') in TORCH_MODEL_NAMES
        if is_torch:
            from train_cls_torch import train_torch_cls
            train_record = train_torch_cls(flat_cfg)
        else:
            from train_cls_yolo import train_yolo_cls
            train_record = train_yolo_cls(flat_cfg)

        weights_path = train_record.get('weights_path', '')

    except Exception as e:
        elapsed = (time.time() - t0) / 60.0
        log_failed(exp_id, cfg, error=str(e))
        import traceback
        traceback.print_exc()
        return {'status': 'FAILED', 'experiment_id': exp_id}

    elapsed = (time.time() - t0) / 60.0

    # ── Step 3: Evaluate ──────────────────────────────────────
    metrics = {}
    try:
        val_dir = str(dataset_path / eval_split)
        metrics = evaluate(
            weights_path=weights_path,
            data_dir=val_dir,
            framework='torch' if (framework == 'torch' or model_name.replace('.pt','') in TORCH_MODEL_NAMES) else 'ultralytics',
            exp_id=exp_id,
            split=eval_split,
            save_figures=True
        )
    except Exception as e:
        print(f"⚠️  Evaluate failed: {e} — logging partial result.")

    metrics['training_time_min'] = round(elapsed, 1)
    metrics['model_path']  = weights_path
    metrics['best_epoch']  = train_record.get('best_epoch', '')
    metrics['train_samples'] = train_record.get('train_samples', '')
    metrics['val_samples']   = train_record.get('val_samples', '')

    # ── Step 4: log_done ──────────────────────────────────────
    log_done(exp_id, cfg, metrics)
    return {'status': 'done', 'experiment_id': exp_id, **metrics}


# ──────────────────────────────────────────────────────────────
# Batch / List / K-Fold / Multi-Seed
# ──────────────────────────────────────────────────────────────
def run_batch(prefix: str, eval_split: str):
    configs = sorted(glob.glob(str(CONFIGS / f'{prefix}*.yaml')))
    if not configs:
        print(f"No configs found: {CONFIGS}/{prefix}*.yaml")
        return
    print(f"\n🚀 Batch [{prefix}]  {len(configs)} experiments  split={eval_split}")
    for i, cfg_path in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {Path(cfg_path).stem}")
        run_single(cfg_path, eval_split=eval_split)


def list_configs():
    all_cfgs = sorted(CONFIGS.glob('*.yaml'))
    print(f"\n{'─'*65}")
    print(f"{'ID':<20} {'Model':<26} {'Framework':<14} {'Seed':<6} {'Epochs'}")
    print(f"{'─'*65}")
    for c in all_cfgs:
        cfg = _load_cfg(str(c))
        print(
            f"{cfg['experiment']['id']:<20} "
            f"{cfg['model']['name']:<26} "
            f"{cfg['model']['framework']:<14} "
            f"{cfg['training']['seed']:<6} "
            f"{cfg['training']['epochs']}"
        )
    print(f"{'─'*65}")
    print(f"Total: {len(all_cfgs)} configs\n")


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Wound AI Experiment Orchestrator v3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  development  (default) — uses val/ split, Blind Test locked
  blind_test             — uses test/ split, requires --allow-blind-test

Examples:
  python experiments/run_experiments.py --list
  python experiments/run_experiments.py --batch C-Arch
  python experiments/run_experiments.py --single experiments/configs/C-Arch-01_resnet50.yaml
  python experiments/run_experiments.py --kfold  experiments/configs/C-Arch-05_yolov8n_cls.yaml
  python experiments/run_experiments.py --multiseed experiments/configs/C-Arch-05_yolov8n_cls.yaml

  # Final Blind Test (unlock explicitly):
  python experiments/run_experiments.py --mode blind_test --allow-blind-test \\
         --single experiments/configs/C-Arch-05_yolov8n_cls.yaml
        """
    )
    parser.add_argument('--batch',            help='Run all configs with given prefix')
    parser.add_argument('--single',           help='Path to a single config YAML')
    parser.add_argument('--kfold',            help='Run 5-Fold CV (StratifiedKFold) for a config')
    parser.add_argument('--kfold-group',      help='Phase 6.6: Run 5-Fold GroupKFold (MD5-aware, leakage-free)')
    parser.add_argument('--multiseed',        help='Run Multi-Seed for a config')
    parser.add_argument('--stats',            help='Phase 8: Run Bootstrap 95% CI statistical analysis for exp_id (e.g. C-Arch-05-MS)')
    parser.add_argument('--tables',           action='store_true',
                                              help='Phase 9: Auto-generate publication paper tables (CSV, MD, LaTeX)')
    parser.add_argument('--audit',            action='store_true',
                                              help='Phase 6.5: Run data split integrity audit')
    parser.add_argument('--list',             action='store_true')
    parser.add_argument('--mode',             default='development',
                                              choices=['development', 'blind_test'])
    parser.add_argument('--allow-blind-test', action='store_true',
                                              help='Required to unlock Blind Test mode')
    args = parser.parse_args()

    # 確定評估 split（含 Blind Test 保護）
    eval_split = _guard_blind_test(args.mode, args.allow_blind_test)

    if args.audit:
        from audit_split import run_audit
        run_audit()
        print("\n" + "="*62)
        print("  Running Phase 6.5-B: Duplicate Group Audit...")
        print("="*62)
        from audit_duplicates import run_duplicate_group_audit
        run_duplicate_group_audit()

    elif args.list:
        list_configs()

    elif args.single:
        run_single(args.single, eval_split=eval_split)

    elif args.batch:
        run_batch(args.batch, eval_split=eval_split)

    elif args.kfold:
        sys.path.insert(0, str(SCRIPTS))
        from cross_validation import run_kfold
        run_kfold(_load_cfg(args.kfold), k=5)

    elif args.kfold_group:
        sys.path.insert(0, str(SCRIPTS))
        from cross_validation_group import run_group_kfold
        run_group_kfold(_load_cfg(args.kfold_group), k=5)

    elif args.multiseed:
        sys.path.insert(0, str(SCRIPTS))
        from multi_seed import run_multi_seed
        run_multi_seed(_load_cfg(args.multiseed))

    elif args.stats:
        sys.path.insert(0, str(SCRIPTS))
        from statistical_analysis import run_phase8_analysis
        run_phase8_analysis(exp_id=args.stats)

    elif args.tables:
        sys.path.insert(0, str(SCRIPTS))
        from generate_tables import run_phase9_table_generation
        run_phase9_table_generation()

    else:
        parser.print_help()
