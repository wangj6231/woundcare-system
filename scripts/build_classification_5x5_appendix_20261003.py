"""Build the saved-result-only 5x5 classification appendix.

Public outputs contain aggregate metrics and plots only.  A separate ignored
professor appendix contains development-image contact sheets.  The script never
enumerates or opens the locked ``test`` directory and performs no inference.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from experiments.scripts.cross_validation_group import CLASS_NAMES


PREDICTIONS = ROOT / "experiments/results/predictions/val"
TABLES = ROOT / "experiments/results/tables"
PUBLIC_ASSETS = ROOT / "docs/report_assets/classification_5x5_20261003"
PRIVATE_REPORT = ROOT / "docs/private/PROFESSOR_CLASSIFICATION_PREDICTION_GALLERY_20261003.md"
RAW_RESULTS = ROOT / "experiments/results/raw"
SEEDS = (42, 123, 3407, 2026, 999)
FOLDS = (1, 2, 3, 4, 5)
PATTERN = re.compile(r"C-Arch-05-MS_s(?P<seed>\d+)_fold(?P<fold>\d+)_predictions\.json$")


def _read_saved_runs() -> list[dict]:
    files = sorted(PREDICTIONS.glob("C-Arch-05-MS_s*_fold*_predictions.json"))
    if len(files) != 25:
        raise RuntimeError(f"expected 25 saved validation files, found {len(files)}")
    runs = []
    seen = set()
    for path in files:
        match = PATTERN.match(path.name)
        if not match:
            raise RuntimeError(f"unexpected prediction filename: {path.name}")
        seed, fold = int(match["seed"]), int(match["fold"])
        if (seed, fold) in seen:
            raise RuntimeError(f"duplicate run: seed={seed}, fold={fold}")
        seen.add((seed, fold))
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("class_names") != CLASS_NAMES:
            raise RuntimeError(f"class order mismatch: {path.name}")
        y_true = np.asarray(data["y_true"], dtype=int)
        y_pred = np.asarray(data["y_pred"], dtype=int)
        y_prob = np.asarray(data["y_prob"], dtype=float)
        if y_true.shape != y_pred.shape or y_prob.shape != (len(y_true), len(CLASS_NAMES)):
            raise RuntimeError(f"prediction shape mismatch: {path.name}")
        if not np.isfinite(y_prob).all() or not np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-4):
            raise RuntimeError(f"invalid probabilities: {path.name}")
        report = classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        overall = {
            "seed": seed,
            "fold": fold,
            "model_id": f"C-Arch-05-MS_s{seed}_fold{fold}",
            "validation_images": int(len(y_true)),
            "accuracy_percent": 100 * accuracy_score(y_true, y_pred),
            "macro_precision_percent": 100 * report["macro avg"]["precision"],
            "macro_recall_percent": 100 * report["macro avg"]["recall"],
            "macro_f1_percent": 100 * report["macro avg"]["f1-score"],
            "weighted_f1_percent": 100 * report["weighted avg"]["f1-score"],
            "roc_auc_macro_ovr": roc_auc_score(
                y_true,
                y_prob,
                labels=np.arange(len(CLASS_NAMES)),
                multi_class="ovr",
                average="macro",
            ),
        }
        per_class = []
        for name in CLASS_NAMES:
            values = report[name]
            per_class.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "model_id": overall["model_id"],
                    "class_name": name,
                    "support": int(values["support"]),
                    "precision_percent": 100 * values["precision"],
                    "recall_percent": 100 * values["recall"],
                    "f1_percent": 100 * values["f1-score"],
                }
            )
        runs.append({"overall": overall, "per_class": per_class, "y_true": y_true})
    expected = {(seed, fold) for seed in SEEDS for fold in FOLDS}
    if seen != expected:
        raise RuntimeError(f"run grid mismatch: missing={sorted(expected-seen)} extra={sorted(seen-expected)}")
    return sorted(runs, key=lambda row: (SEEDS.index(row["overall"]["seed"]), row["overall"]["fold"]))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _write_markdown_tables(overall: list[dict], per_class: list[dict]) -> None:
    overall_md = [
        "# C-Arch-05-MS：5 Seeds × 5 Folds 各模型整體成績",
        "",
        "> 全部數字由25份既存validation prediction arrays重算；沒有載入模型、沒有重新推論、沒有使用48張locked test。每列是一個seed/fold模型實例，25列不是25個獨立資料集。",
        "",
        "| Seed | Fold | Model ID | Val N | Accuracy (%) | Macro-P (%) | Macro-R (%) | Macro-F1 (%) | Weighted-F1 (%) | Macro ROC-AUC |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        overall_md.append(
            "| {seed} | {fold} | `{model_id}` | {validation_images} | {accuracy_percent:.2f} | "
            "{macro_precision_percent:.2f} | {macro_recall_percent:.2f} | {macro_f1_percent:.2f} | "
            "{weighted_f1_percent:.2f} | {roc_auc_macro_ovr:.4f} |".format(**row)
        )
    overall_md += [
        "",
        "說明：不同fold的validation張數與類別support不完全相同，因MD5 group不可拆分；因此不能要求每折張數一樣。",
        "",
    ]
    (TABLES / "Table_CV25_Overall_Performance.md").write_text("\n".join(overall_md), encoding="utf-8")

    class_md = [
        "# C-Arch-05-MS：25模型 × 七類別詳細成績",
        "",
        "> Precision、Recall與F1均以該seed/fold保存的validation predictions計算；support是真實類別張數。共25×7＝175列。",
        "",
        "| Seed | Fold | Class | Support | Precision (%) | Recall (%) | F1 (%) |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in per_class:
        class_md.append(
            "| {seed} | {fold} | {class_name} | {support} | {precision_percent:.2f} | "
            "{recall_percent:.2f} | {f1_percent:.2f} |".format(**row)
        )
    class_md += [
        "",
        "這些分數描述模型在development cross-validation中的表現，不是48張locked test、外部驗證或病人層級推論。",
        "",
    ]
    (TABLES / "Table_CV25_PerClass_Performance.md").write_text("\n".join(class_md), encoding="utf-8")


def _matrix(rows: list[dict], key: str) -> np.ndarray:
    index = {(row["seed"], row["fold"]): row[key] for row in rows}
    return np.asarray([[index[(seed, fold)] for fold in FOLDS] for seed in SEEDS], dtype=float)


def _plot_heatmaps(overall: list[dict], per_class: list[dict]) -> None:
    PUBLIC_ASSETS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for ax, key, title in (
        (axes[0], "accuracy_percent", "Top-1 Accuracy (%)"),
        (axes[1], "macro_f1_percent", "Macro-F1 (%)"),
    ):
        values = _matrix(overall, key)
        image = ax.imshow(values, cmap="YlGnBu", vmin=80, vmax=94)
        ax.set_xticks(range(5), [f"Fold {fold}" for fold in FOLDS])
        ax.set_yticks(range(5), [f"Seed {seed}" for seed in SEEDS])
        ax.set_title(title)
        for y in range(5):
            for x in range(5):
                ax.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if values[y, x] < 84 or values[y, x] > 91 else "black")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("C-Arch-05-MS: 25 leakage-free development runs", fontsize=14)
    fig.savefig(PUBLIC_ASSETS / "classification_5x5_overall_heatmaps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    by_class = {name: [row for row in per_class if row["class_name"] == name] for name in CLASS_NAMES}
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    axes = axes.ravel()
    for ax, name in zip(axes, CLASS_NAMES):
        values = _matrix(by_class[name], "f1_percent")
        image = ax.imshow(values, cmap="RdYlGn", vmin=50, vmax=100)
        ax.set_xticks(range(5), FOLDS)
        ax.set_yticks(range(5), SEEDS)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Seed")
        ax.set_title(name)
        for y in range(5):
            for x in range(5):
                ax.text(x, y, f"{values[y, x]:.0f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, shrink=0.75)
    axes[-1].axis("off")
    fig.suptitle("Per-class F1 (%) for every seed/fold model", fontsize=15)
    fig.savefig(PUBLIC_ASSETS / "classification_5x5_per_class_f1_heatmaps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_private_prediction_gallery(runs: list[dict]) -> int:
    PRIVATE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 教授查閱版：分類模型5×5預測影像",
        "",
        "日期：2026-10-03。以下直接引用25次訓練保存的`val_batch0_pred.jpg`；未重新推論、未開啟或複製48張locked test。",
        "",
        "> 重要：C-Arch-05是整張圖分類模型，不會產生傷口bbox或ROI crop。圖中每張影像左上角的文字是Ultralytics保存的預測類別；這是分類預測標示，不是傷口定位框。",
        "",
        "> 七類資料的原始URL、版本與授權仍為UNVERIFIED。本附錄只供本機與教授查閱，不提交公開GitHub。",
        "",
    ]
    count = 0
    for run in runs:
        row = run["overall"]
        montage = RAW_RESULTS / row["model_id"] / "val_batch0_pred.jpg"
        if not montage.is_file():
            raise RuntimeError(f"missing saved classification prediction montage: {montage}")
        relative = Path("..") / ".." / "experiments" / "results" / "raw" / row["model_id"] / montage.name
        lines += [
            f"## Seed {row['seed']} / Fold {row['fold']}",
            "",
            f"Validation N={row['validation_images']}；Accuracy={row['accuracy_percent']:.2f}%；Macro-F1={row['macro_f1_percent']:.2f}%。",
            "",
            f"![{row['model_id']} saved validation predictions]({relative.as_posix()})",
            "",
        ]
        count += 1
    lines += [
        "## 解讀限制",
        "",
        "- 每張montage只是該run保存的第一批validation預測，不代表完整fold；完整分數看25-run與175-row表。",
        "- 圖中文字是預測類別；不能把方形拼圖邊界解讀成傷口bbox。",
        "- 逐模型分數以公開的25-run整體表與175-row逐類表為準。",
        "- 若要把圖片公開，必須先補齊原始資料來源、版本、授權及逐圖對應證據。",
        "",
    ]
    PRIVATE_REPORT.write_text("\n".join(lines), encoding="utf-8")
    return count


def main() -> None:
    runs = _read_saved_runs()
    overall = [run["overall"] for run in runs]
    per_class = [row for run in runs for row in run["per_class"]]
    _write_csv(TABLES / "Table_CV25_Overall_Performance.csv", overall)
    _write_csv(TABLES / "Table_CV25_PerClass_Performance.csv", per_class)
    _write_markdown_tables(overall, per_class)
    _plot_heatmaps(overall, per_class)
    prediction_montages = _write_private_prediction_gallery(runs)
    manifest = {
        "schema": "CLASSIFICATION_5X5_APPENDIX_V1",
        "date": "2026-10-03",
        "saved_prediction_files": 25,
        "overall_rows": len(overall),
        "per_class_rows": len(per_class),
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "classes": CLASS_NAMES,
        "model_inference_performed": False,
        "locked_test_enumerated": False,
        "locked_test_images_used": 0,
        "image_row_prediction_mapping": "UNAVAILABLE",
        "private_prediction_montages": prediction_montages,
    }
    (PUBLIC_ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
