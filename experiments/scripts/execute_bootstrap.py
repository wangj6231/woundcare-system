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

def do_compute():
    pred_files = []
    for s in DEFAULT_SEEDS:
        for f in range(1, K + 1):
            p_json = PRED_DIR / f"C-Arch-05-MS_s{s}_fold{f}_predictions.json"
            if p_json.exists():
                pred_files.append((s, f, p_json))

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
                'seed': s, 'fold': f, 'file': p_file.name,
                'n_samples': len(yt),
                'accuracy': round(float(accuracy_score(yt, yp) * 100), 2),
                'macro_f1': round(float(f1_score(yt, yp, average='macro') * 100), 2),
            })

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    n_boot = 2000
    ci_level = 0.95
    rng = np.random.default_rng(42)
    n_samples = len(y_true)

    boot_accs         = np.zeros(n_boot)
    boot_macro_f1s    = np.zeros(n_boot)
    boot_weighted_f1s  = np.zeros(n_boot)
    boot_recalls      = np.zeros((n_boot, NUM_CLASSES))

    alpha = (1 - ci_level) / 2
    lower_p = alpha * 100
    upper_p = (1 - alpha) * 100

    for i in range(n_boot):
        boot_idx = rng.integers(0, n_samples, size=n_samples)
        yt_b = y_true[boot_idx]
        yp_b = y_pred[boot_idx]

        boot_accs[i]         = accuracy_score(yt_b, yp_b) * 100
        boot_macro_f1s[i]    = f1_score(yt_b, yp_b, average='macro', zero_division=0) * 100
        boot_weighted_f1s[i]  = f1_score(yt_b, yp_b, average='weighted', zero_division=0) * 100
        boot_recalls[i]      = recall_score(yt_b, yp_b, average=None, labels=list(range(NUM_CLASSES)), zero_division=0) * 100

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

    acc  = _summarize(boot_accs)
    mf1  = _summarize(boot_macro_f1s)
    wf1  = _summarize(boot_weighted_f1s)
    stab = per_class_ci['Stab_wound']

    paper_latex = f"{acc['mean']:.2f}\\% (95\\% CI: {acc['ci_lower']:.2f}--{acc['ci_upper']:.2f}\\%)"
    paper_md = f"Top-1 Accuracy = {acc['mean']:.2f}% (95% CI: {acc['ci_lower']:.2f}%–{acc['ci_upper']:.2f}%), Macro-F1 = {mf1['mean']:.2f}% (95% CI: {mf1['ci_lower']:.2f}%–{mf1['ci_upper']:.2f}%)"

    report = {
        'experiment_id':      'C-Arch-05-MS',
        'total_predictions':  len(y_true),
        'prediction_files':   len(pred_files),
        'bootstrap_results': {
            'n_bootstrap': n_boot,
            'ci_level': ci_level,
            'total_samples': n_samples,
            'accuracy': acc,
            'macro_f1': mf1,
            'weighted_f1': wf1,
            'per_class_recall': per_class_ci,
        },
        'paper_formatting': {
            'latex_accuracy': paper_latex,
            'markdown_summary': paper_md,
            'accuracy_ci_str': acc['ci_str'],
            'macro_f1_ci_str': mf1['ci_str'],
            'stab_wound_recall_ci_str': stab['ci_str'],
        },
        'fold_details': fold_details,
    }

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STATS_DIR / "C-Arch-05-MS_bootstrap_ci_report.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"DONE writing {out_path}")

do_compute()
