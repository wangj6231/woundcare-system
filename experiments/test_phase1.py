# ============================================================
# test_phase1.py — Phase 1 驗證腳本
# 不需要 GPU、不需要真實模型，直接測試 logger 是否正常運作
# 使用方式：python experiments/test_phase1.py
# ============================================================
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
from experiment_logger import log_start, log_done, log_failed, LOG_PATH, SCHEMA

BASE_DIR = Path(__file__).parent.parent

# ── 模擬實驗設定（巢狀 YAML 格式）──────────────────────────
DUMMY_CFG = {
    'experiment': {'id': 'TEST-001', 'name': 'dummy_test', 'type': 'classification'},
    'model':      {'name': 'dummy_model', 'framework': 'torch'},
    'dataset':    {'path': 'yolo_wound_cls_dataset_v3', 'version': 'v3'},
    'training':   {'epochs': 5, 'batch': 16, 'imgsz': 224, 'seed': 42},
    'optimizer':  {'name': 'adam', 'lr0': 0.001, 'weight_decay': 0.0005},
    'augmentation': {'fliplr': 0.5, 'degrees': 45.0},
    'notes': 'Phase 1 test run',
}

DUMMY_METRICS_SUCCESS = {
    'accuracy':       92.68,
    'top5_accuracy':  100.0,
    'macro_precision': 91.2,
    'macro_recall':    90.8,
    'macro_f1':        91.0,
    'weighted_f1':     92.5,
    'roc_auc_macro':   0.987,
    'inference_ms':    0.52,
    'training_time_min': 3.7,
    'best_epoch':      48,
    'model_path':      '/fake/path/best.pt',
    'train_samples':   679,
    'val_samples':     41,
}

def run_tests():
    print("=" * 62)
    print("Phase 1 Test: experiment_logger.py")
    print("=" * 62)

    # ── Test 1: Schema 完整性 ────────────────────────────────
    print(f"\n[T1] Schema check: {len(SCHEMA)} fields")
    assert len(SCHEMA) == 30, f"Expected 30 fields, got {len(SCHEMA)}"
    print(f"     ✅ PASS — {len(SCHEMA)} fields confirmed")

    # ── Test 2: log_start ────────────────────────────────────
    print(f"\n[T2] log_start → status=running")
    log_start('TEST-001', DUMMY_CFG)
    assert LOG_PATH.exists(), "CSV not created"
    content = LOG_PATH.read_text(encoding='utf-8-sig')
    assert 'TEST-001' in content
    assert 'running' in content
    print(f"     ✅ PASS — running row written")

    # ── Test 3: log_done ─────────────────────────────────────
    print(f"\n[T3] log_done → status=done, all metrics")
    log_done('TEST-001', DUMMY_CFG, DUMMY_METRICS_SUCCESS)
    content = LOG_PATH.read_text(encoding='utf-8-sig')
    assert 'done' in content
    assert '92.68' in content
    assert '91.00' in content
    print(f"     ✅ PASS — done row with metrics written")

    # ── Test 4: log_failed ───────────────────────────────────
    print(f"\n[T4] log_failed → status=FAILED")
    FAIL_CFG = dict(DUMMY_CFG)
    FAIL_CFG['experiment'] = {'id': 'TEST-002', 'name': 'fail_test', 'type': 'classification'}
    log_failed('TEST-002', FAIL_CFG, error='RuntimeError: CUDA out of memory')
    content = LOG_PATH.read_text(encoding='utf-8-sig')
    assert 'TEST-002' in content
    assert 'FAILED' in content
    print(f"     ✅ PASS — FAILED row written")

    # ── Test 5: 多筆累積（CSV 不會覆蓋）────────────────────
    print(f"\n[T5] Multiple rows accumulate correctly")
    rows = [r for r in content.split('\n') if r.strip() and not r.startswith('experiment')]
    print(f"     Data rows in CSV: {len(rows)}")
    assert len(rows) >= 3, f"Expected >= 3 data rows, got {len(rows)}"
    print(f"     ✅ PASS — {len(rows)} rows accumulated")

    # ── Test 6: 欄位順序固定 ────────────────────────────────
    print(f"\n[T6] Column order is fixed")
    header = content.split('\n')[0].strip()
    first_col = header.split(',')[0]
    assert first_col == 'experiment_id', f"First col should be experiment_id, got {first_col}"
    last_col = header.split(',')[-1]
    assert last_col == 'timestamp', f"Last col should be timestamp, got {last_col}"
    print(f"     ✅ PASS — experiment_id...timestamp")

    # ── Test 7: Blind Test 保護（不需要模型，只測 CLI） ─────
    print(f"\n[T7] Blind Test guard (CLI level)")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / 'experiments' / 'run_experiments.py'),
         '--mode', 'blind_test',
         '--single', 'fake_config.yaml'],
        capture_output=True, text=True
    )
    assert 'BLIND TEST BLOCKED' in result.stdout or 'BLIND TEST BLOCKED' in result.stderr, \
        "Blind Test guard did not trigger"
    assert result.returncode != 0, "Should have exited with error"
    print(f"     ✅ PASS — Blind Test blocked without --allow-blind-test")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"✅ ALL PHASE 1 TESTS PASSED  (7/7)")
    print(f"   CSV path: {LOG_PATH}")
    print(f"{'='*62}")

    # 清理測試資料
    _cleanup_test_rows()


def _cleanup_test_rows():
    """移除 TEST-001, TEST-002 測試記錄"""
    if not LOG_PATH.exists():
        return
    lines = LOG_PATH.read_text(encoding='utf-8-sig').split('\n')
    cleaned = [l for l in lines if 'TEST-001' not in l and 'TEST-002' not in l]
    LOG_PATH.write_text('\n'.join(cleaned), encoding='utf-8-sig')
    print(f"\n🧹 Test rows cleaned from {LOG_PATH.name}")


if __name__ == '__main__':
    run_tests()
