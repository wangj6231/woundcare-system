# ============================================================
# verify_phase7.py — Phase 7 Integrity Audit & Summary Fix
#
# 目標：
#   ① 修正 C-Arch-05-MS_multiseed_summary.json 的 completed_folds 與彙總計算
#   ② 執行 Phase 7 Integrity Gate 檢查（7 項條款）：
#      [1] len(all_fold_results) == 25
#      [2] 每個 seed completed_folds == 5
#      [3] 全部 gate_passed == True
#      [4] 全部 md5_overlap == 0
#      [5] 全部 gid_overlap == 0
#      [6] 全部 status == done
#      [7] 無遺漏 runs (5 seeds x 5 folds = 25)
#
# 使用方式：
#   python experiments/scripts/verify_phase7.py
# ============================================================
import json
import sys
import numpy as np
from pathlib import Path

BASE_DIR  = Path(__file__).parent.parent.parent
STATS_DIR = Path(__file__).parent.parent / 'results' / 'statistics'
SUMMARY_PATH = STATS_DIR / 'C-Arch-05-MS_multiseed_summary.json'

DEFAULT_SEEDS = [42, 123, 3407, 2026, 999]
K = 5


def fix_and_verify_phase7():
    print(f"\n{'='*62}")
    print(f"🔍 Phase 7 Integrity Audit & Summary Fix")
    print(f"   Target: {SUMMARY_PATH}")
    print(f"{'='*62}")

    if not SUMMARY_PATH.exists():
        print(f"❌ Summary JSON not found: {SUMMARY_PATH}")
        sys.exit(1)

    with open(SUMMARY_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_runs = data.get('all_fold_results', [])
    print(f"  Loaded {len(all_runs)} fold records from JSON.")

    # ── Step 1: Recompute Per-Seed & Overall Summaries ─────────
    per_seed = {}
    done_runs = []

    for seed in DEFAULT_SEEDS:
        seed_runs = [r for r in all_runs if r.get('seed') == seed and r.get('status') == 'done']
        done_runs.extend(seed_runs)

        if seed_runs:
            accs  = [r['accuracy']          for r in seed_runs]
            f1s   = [r['macro_f1']          for r in seed_runs]
            stabs = [r['stab_wound_recall'] for r in seed_runs]

            per_seed[str(seed)] = {
                'accuracy_mean':    round(float(np.mean(accs)),  2),
                'accuracy_std':     round(float(np.std(accs)),   2),
                'macro_f1_mean':    round(float(np.mean(f1s)),   2),
                'macro_f1_std':     round(float(np.std(f1s)),    2),
                'stab_recall_mean': round(float(np.mean(stabs)), 2),
                'stab_recall_std':  round(float(np.std(stabs)),  2),
                'completed_folds':  len(seed_runs),
            }

    all_accs  = [r['accuracy']          for r in done_runs]
    all_f1s   = [r['macro_f1']          for r in done_runs]
    all_stabs = [r['stab_wound_recall'] for r in done_runs]

    seed_means_acc  = [v['accuracy_mean']    for v in per_seed.values()]
    seed_means_f1   = [v['macro_f1_mean']    for v in per_seed.values()]
    seed_means_stab = [v['stab_recall_mean'] for v in per_seed.values()]

    overall = {
        'accuracy_mean':    round(float(np.mean(all_accs)),  2),
        'accuracy_std':     round(float(np.std(all_accs)),   2),
        'macro_f1_mean':    round(float(np.mean(all_f1s)),   2),
        'macro_f1_std':     round(float(np.std(all_f1s)),    2),
        'stab_recall_mean': round(float(np.mean(all_stabs)), 2),
        'stab_recall_std':  round(float(np.std(all_stabs)),  2),
    }

    between_seed = {
        'accuracy_mean_of_means': round(float(np.mean(seed_means_acc)),  2),
        'accuracy_sd_of_means':   round(float(np.std(seed_means_acc)),   2),
        'macro_f1_mean_of_means': round(float(np.mean(seed_means_f1)),   2),
        'macro_f1_sd_of_means':   round(float(np.std(seed_means_f1)),    2),
        'stab_recall_mean_of_means': round(float(np.mean(seed_means_stab)), 2),
        'stab_recall_sd_of_means':   round(float(np.std(seed_means_stab)),  2),
    }

    # 更新 JSON 資料結構
    data['total_runs']       = len(done_runs)
    data['overall']          = overall
    data['between_seed']     = between_seed
    data['per_seed_summary'] = per_seed

    # 存檔
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Corrected summary saved to {SUMMARY_PATH}")

    # ── Step 2: Phase 7 Integrity Gate Checks ─────────────────
    print(f"\n{'─'*62}")
    print(f"  Phase 7 Integrity Gate Checks (7 Criteria)")
    print(f"{'─'*62}")

    c1_total_runs = len(all_runs) == 25
    c2_seeds_folds = all(per_seed[str(s)]['completed_folds'] == 5 for s in DEFAULT_SEEDS)
    c3_gate_passed = all(r.get('gate_passed') is True for r in all_runs)
    c4_md5_zero    = all(r.get('md5_overlap', -1) == 0 for r in all_runs)
    c5_gid_zero    = all(r.get('gid_overlap', -1) == 0 for r in all_runs)
    c6_status_done = all(r.get('status') == 'done' for r in all_runs)

    # 檢查是否包含所有 (seed, fold)
    expected_pairs = {(s, f) for s in DEFAULT_SEEDS for f in range(1, K+1)}
    actual_pairs   = {(r['seed'], r['fold']) for r in all_runs}
    c7_all_pairs   = (expected_pairs == actual_pairs)

    checks = [
        ("1. Total runs == 25",                      c1_total_runs, f"count={len(all_runs)}"),
        ("2. Each seed completed_folds == 5",        c2_seeds_folds, f"{[per_seed[str(s)]['completed_folds'] for s in DEFAULT_SEEDS]}"),
        ("3. All gate_passed == True",               c3_gate_passed, f"passed={sum(1 for r in all_runs if r.get('gate_passed'))}/25"),
        ("4. All MD5 overlap == 0",                  c4_md5_zero,    f"zero={sum(1 for r in all_runs if r.get('md5_overlap')==0)}/25"),
        ("5. All Group ID overlap == 0",             c5_gid_zero,    f"zero={sum(1 for r in all_runs if r.get('gid_overlap')==0)}/25"),
        ("6. All status == 'done'",                  c6_status_done, f"done={sum(1 for r in all_runs if r.get('status')=='done')}/25"),
        ("7. All 25 (seed, fold) pairs present",     c7_all_pairs,   f"missing={len(expected_pairs - actual_pairs)}"),
    ]

    all_passed = True
    for label, ok, detail in checks:
        icon = '✅ PASS' if ok else '❌ FAIL'
        if not ok:
            all_passed = False
        print(f"  {icon:<8} | {label:<38} ({detail})")

    print(f"{'─'*62}")

    print(f"\n  📊 Per-Seed Summary (Corrected):")
    print(f"  {'Seed':<6} {'Completed':<10} {'Acc Mean ± SD':<18} {'Macro-F1 ± SD':<18} {'Stab Rec ± SD':<18}")
    for s in DEFAULT_SEEDS:
        ps = per_seed[str(s)]
        print(f"  {s:<6} {ps['completed_folds']:<10} "
              f"{ps['accuracy_mean']:>5.2f} ± {ps['accuracy_std']:<5.2f}     "
              f"{ps['macro_f1_mean']:>5.2f} ± {ps['macro_f1_std']:<5.2f}     "
              f"{ps['stab_recall_mean']:>5.2f} ± {ps['stab_recall_std']:<5.2f}")

    print(f"\n  📈 Overall (25 Runs):")
    print(f"     Top-1 Accuracy:  {overall['accuracy_mean']}% ± {overall['accuracy_std']}%")
    print(f"     Macro-F1:        {overall['macro_f1_mean']}% ± {overall['macro_f1_std']}%")
    print(f"     Stab Recall:     {overall['stab_recall_mean']}% ± {overall['stab_recall_std']}%")
    print(f"     Between-Seed SD: Acc={between_seed['accuracy_sd_of_means']}%, F1={between_seed['macro_f1_sd_of_means']}%, Stab={between_seed['stab_recall_sd_of_means']}%")

    print(f"\n{'='*62}")
    if all_passed:
        print(f"🎉 PHASE 7 INTEGRITY VERDICT: ALL PASS ✅")
        print(f"   Phase 7 is officially SEALED and ready for Phase 8 Statistical Analysis.")
    else:
        print(f"❌ PHASE 7 INTEGRITY VERDICT: FAIL")
    print(f"{'='*62}\n")


if __name__ == '__main__':
    fix_and_verify_phase7()
