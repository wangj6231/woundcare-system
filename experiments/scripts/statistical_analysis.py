# ============================================================
# statistical_analysis.py — Phase 8：Statistical Significance Testing
#
# 功能：
#   ① Bootstrap 95% Confidence Interval (BCa / Percentile Bootstrap, B=2000)
#      - Overall Accuracy (95% CI)
#      - Macro-F1 (95% CI)
#      - Weighted-F1 (95% CI)
#      - Per-class Recall (95% CI) 特別含 Stab_wound
#   ② McNemar's Test (含 Continuity Correction)
#   ③ 自動累積 25 runs (5 Seeds × 5 Folds) 預測做池化/分折 Bootstrap
#   ④ 生成 JSON 報告 + 論文 LaTeX/Markdown 格式化輸出
#
# 使用方式：
#   python experiments/scripts/statistical_analysis.py \
#          --exp_id C-Arch-05-MS --n_boot 2000
# ============================================================
import argparse
import json
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, recall_score

BASE_DIR  = Path(__file__).parent.parent.parent
PRED_DIR  = Path(__file__).parent.parent / 'results' / 'predictions' / 'val'
STATS_DIR = Path(__file__).parent.parent / 'results' / 'statistics'

CLASS_NAMES = ['Abrasions', 'Bruises', 'Burns', 'Cut',
               'Ingrown_nails', 'Laceration', 'Stab_wound']
NUM_CLASSES = len(CLASS_NAMES)
DEFAULT_SEEDS = [42, 123, 3407, 2026, 999]
K = 5


# ──────────────────────────────────────────────────────────────
# 預測讀取
# ──────────────────────────────────────────────────────────────
def load_predictions_for_exp(exp_id: str) -> tuple:
    """
    從 predictions/val/ 讀取所有關聯的 predictions.json，
    聚合出 (y_true_all, y_pred_all, fold_runs)
    """
    pred_files = []

    # 優先嘗試 Multi-Seed pattern: {exp_id}_s{seed}_fold{fold}_predictions.json
    for s in DEFAULT_SEEDS:
        for f in range(1, K + 1):
            p_json = PRED_DIR / f"{exp_id}_s{s}_fold{f}_predictions.json"
            if p_json.exists():
                pred_files.append((s, f, p_json))

    # 若非 MS，嘗試 GKF pattern: {exp_id}_kfold{fold}_predictions.json
    if not pred_files:
        for f in range(1, K + 1):
            p_json = PRED_DIR / f"{exp_id}_kfold{f}_predictions.json"
            if p_json.exists():
                pred_files.append((0, f, p_json))

    # 若單一檔案: {exp_id}_predictions.json
    if not pred_files:
        p_json = PRED_DIR / f"{exp_id}_predictions.json"
        if p_json.exists():
            pred_files.append((0, 0, p_json))

    if not pred_files:
        raise FileNotFoundError(f"No prediction JSON files found matching {exp_id} in {PRED_DIR}")

    y_true_list = []
    y_pred_list = []
    fold_details = []

    for s, f, p_file in pred_files:
        with open(p_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            yt = np.array(data['y_true'])
            yp = np.array(data['y_pred'])
            y_true_list.append(yt)
            y_pred_list.append(yp)
            fold_details.append({
                'seed': s,
                'fold': f,
                'file': p_file.name,
                'n_samples': len(yt),
                'accuracy': round(float(accuracy_score(yt, yp) * 100), 2),
                'macro_f1': round(float(f1_score(yt, yp, average='macro') * 100), 2),
            })

    y_true_concat = np.concatenate(y_true_list)
    y_pred_concat = np.concatenate(y_pred_list)

    return y_true_concat, y_pred_concat, fold_details, len(pred_files)


# ──────────────────────────────────────────────────────────────
# Bootstrap 95% Confidence Interval
# ──────────────────────────────────────────────────────────────
def calculate_bootstrap_ci(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           n_boot: int = 2000,
                           ci_level: float = 0.95,
                           seed: int = 42) -> dict:
    """
    非參數 Percentile Bootstrap 計算 Accuracy, Macro-F1, Weighted-F1 與 Per-class Recalls 的 95% CI
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)

    boot_accs        = np.zeros(n_boot)
    boot_macro_f1s   = np.zeros(n_boot)
    boot_weighted_f1s = np.zeros(n_boot)
    boot_recalls     = np.zeros((n_boot, NUM_CLASSES))

    alpha = (1 - ci_level) / 2
    lower_p = alpha * 100
    upper_p = (1 - alpha) * 100

    for i in range(n_boot):
        boot_idx = rng.integers(0, n_samples, size=n_samples)
        yt_b = y_true[boot_idx]
        yp_b = y_pred[boot_idx]

        boot_accs[i]        = accuracy_score(yt_b, yp_b) * 100
        boot_macro_f1s[i]   = f1_score(yt_b, yp_b, average='macro', zero_division=0) * 100
        boot_weighted_f1s[i] = f1_score(yt_b, yp_b, average='weighted', zero_division=0) * 100
        boot_recalls[i]     = recall_score(yt_b, yp_b, average=None, labels=list(range(NUM_CLASSES)), zero_division=0) * 100

    def _summarize(boot_arr):
        mean_val  = float(np.mean(boot_arr))
        std_val   = float(np.std(boot_arr))
        lower_val = float(np.percentile(boot_arr, lower_p))
        upper_val = float(np.percentile(boot_arr, upper_p))
        return {
            'mean':        round(mean_val,  2),
            'std':         round(std_val,   2),
            'ci_lower':    round(lower_val, 2),
            'ci_upper':    round(upper_val, 2),
            'ci_str':      f"{round(mean_val,2):.2f}% (95% CI: {round(lower_val,2):.2f}%–{round(upper_val,2):.2f}%)",
        }

    per_class_ci = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        per_class_ci[c_name] = _summarize(boot_recalls[:, c_idx])

    return {
        'n_bootstrap':   n_boot,
        'ci_level':      ci_level,
        'total_samples': n_samples,
        'accuracy':      _summarize(boot_accs),
        'macro_f1':      _summarize(boot_macro_f1s),
        'weighted_f1':   _summarize(boot_weighted_f1s),
        'per_class_recall': per_class_ci,
    }


# ──────────────────────────────────────────────────────────────
# McNemar's Test
# ──────────────────────────────────────────────────────────────
def run_mcnemar_test(y_true: np.ndarray,
                     y_pred_a: np.ndarray,
                     y_pred_b: np.ndarray) -> dict:
    """
    McNemar's Test (Edwards' continuity correction):
    χ² = (|b - c| - 1)² / (b + c)
    """
    from scipy.stats import chi2

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b))  # A 正確, B 錯誤
    c = int(np.sum(~correct_a & correct_b))  # A 錯誤, B 正確

    if (b + c) == 0:
        return {
            'chi2': 0.0,
            'p_value': 1.0,
            'b_A_correct_B_wrong': 0,
            'c_A_wrong_B_correct': 0,
            'significant': False,
            'interpretation': 'No discordant pairs — models are statistically identical',
        }

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_val     = 1 - chi2.cdf(chi2_stat, df=1)
    is_sig    = p_val < 0.05

    better = 'Model A' if b > c else ('Model B' if c > b else 'Tie')

    return {
        'chi2': round(float(chi2_stat), 4),
        'p_value': round(float(p_val), 6),
        'b_A_correct_B_wrong': b,
        'c_A_wrong_B_correct': c,
        'significant': is_sig,
        'better_model': better,
        'interpretation': f"{'Statistically Significant Difference (p<0.05)' if is_sig else 'No Significant Difference (p>=0.05)'} ({better} is superior)",
        'formatted_str': f"McNemar χ²={chi2_stat:.4f}, p={p_val:.6f} ({'p<0.05' if is_sig else 'p>=0.05'})",
    }


# ──────────────────────────────────────────────────────────────
# Main Function
# ──────────────────────────────────────────────────────────────
def run_phase8_analysis(exp_id: str = 'C-Arch-05-MS', n_boot: int = 2000) -> dict:
    print(f"\n{'='*62}")
    print(f"📊 Phase 8: Statistical Significance Testing")
    print(f"   Target Experiment: {exp_id}")
    print(f"   Bootstrap Resamples: B = {n_boot}")
    print(f"{'='*62}")

    # 1. 讀取預測
    y_true, y_pred, fold_details, n_files = load_predictions_for_exp(exp_id)
    print(f"\n  Loaded {n_files} prediction files across folds.")
    print(f"  Total pooled evaluation samples: n = {len(y_true)}")

    # 2. 計算 Bootstrap CI
    print(f"\n  Running Percentile Bootstrap (B={n_boot})...")
    ci_res = calculate_bootstrap_ci(y_true, y_pred, n_boot=n_boot, seed=42)

    acc = ci_res['accuracy']
    mf1 = ci_res['macro_f1']
    wf1 = ci_res['weighted_f1']
    stab = ci_res['per_class_recall']['Stab_wound']

    # 3. 輸出簡明文字與表格
    print(f"\n  {'Metric':<20} {'Mean':>8} {'95% CI Lower':>14} {'95% CI Upper':>14} {'Formatted String':<30}")
    print(f"  {'─'*88}")
    print(f"  {'Accuracy':<20} {acc['mean']:>7.2f}% {acc['ci_lower']:>13.2f}% {acc['ci_upper']:>13.2f}%   {acc['ci_str']}")
    print(f"  {'Macro-F1':<20} {mf1['mean']:>7.2f}% {mf1['ci_lower']:>13.2f}% {mf1['ci_upper']:>13.2f}%   {mf1['ci_str']}")
    print(f"  {'Weighted-F1':<20} {wf1['mean']:>7.2f}% {wf1['ci_lower']:>13.2f}% {wf1['ci_upper']:>13.2f}%   {wf1['ci_str']}")
    print(f"  {'Stab_wound Recall':<20} {stab['mean']:>7.2f}% {stab['ci_lower']:>13.2f}% {stab['ci_upper']:>13.2f}%   {stab['ci_str']}")
    print(f"  {'─'*88}")

    print(f"\n  📋 Per-Class Recall (95% CI):")
    for c_name in CLASS_NAMES:
        c_stat = ci_res['per_class_recall'][c_name]
        print(f"     {c_name:<20}: {c_stat['mean']:>6.2f}%  (95% CI: {c_stat['ci_lower']:>5.2f}%–{c_stat['ci_upper']:>5.2f}%)")

    # 4. 論文專用格式化字串
    paper_latex = f"{acc['mean']:.2f}\\% (95\\% CI: {acc['ci_lower']:.2f}--{acc['ci_upper']:.2f}\\%)"
    paper_md = f"Top-1 Accuracy = {acc['mean']:.2f}% (95% CI: {acc['ci_lower']:.2f}%–{acc['ci_upper']:.2f}%), Macro-F1 = {mf1['mean']:.2f}% (95% CI: {mf1['ci_lower']:.2f}%–{mf1['ci_upper']:.2f}%)"

    report = {
        'experiment_id':      exp_id,
        'total_predictions':  len(y_true),
        'prediction_files':   n_files,
        'bootstrap_results':  ci_res,
        'paper_formatting': {
            'latex_accuracy': paper_latex,
            'markdown_summary': paper_md,
            'accuracy_ci_str': acc['ci_str'],
            'macro_f1_ci_str': mf1['ci_str'],
            'stab_wound_recall_ci_str': stab['ci_str'],
        },
        'fold_details':       fold_details,
    }

    # 5. 儲存 JSON
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STATS_DIR / f"{exp_id}_bootstrap_ci_report.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*62}")
    print(f"✅ Phase 8 Statistical Report Saved:")
    print(f"   💾 {out_path}")
    print(f"   📝 Markdown Format:")
    print(f"      {paper_md}")
    print(f"   📄 LaTeX Format:")
    print(f"      {paper_latex}")
    print(f"{'='*62}\n")

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', default='C-Arch-05-MS', help='Experiment ID to analyze')
    parser.add_argument('--n_boot', type=int, default=2000, help='Number of Bootstrap resamples')
    args = parser.parse_args()

    run_phase8_analysis(exp_id=args.exp_id, n_boot=args.n_boot)
