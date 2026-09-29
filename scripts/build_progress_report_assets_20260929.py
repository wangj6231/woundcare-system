"""Build reproducible public charts and a reviewed FUSeg case gallery.

This script reads aggregate development-result files and copies only five
pre-rendered, reviewed FUSeg comparison figures. It never opens the locked test,
clinical database, checkpoints, or raw clinical/field images.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "report_assets" / "progress_20260929"
PAIR_DIR = ROOT / "experiments" / "results" / "f_higher_scale_v2_paired_error_audit"
SELECTED_CASES = (
    "fuseg__0003.png",  # higher-scale-only FP
    "fuseg__0412.png",  # prediction redistribution and FP
    "fuseg__0604.png",  # multi-GT localization loss / FP churn
    "fuseg__0867.png",  # very-small C4-only detection
    "fuseg__0971.png",  # H4-only recovery and crop gain
)


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def save(fig, name: str):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grouped_bars(ax, labels, left, right, left_label, right_label, ylabel, title):
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, left, width, label=left_label, color="#2463EB")
    ax.bar(x + width / 2, right, width, label=right_label, color="#E8792E")
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    for i, value in enumerate(left):
        ax.text(i - width / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    for i, value in enumerate(right):
        ax.text(i + width / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    b1 = read_json("experiments/results/statistics/C-Arch-05-MS_phase_b1_final_summary.json")
    seed_rows = b1["seed_metrics"]
    seeds = [str(row["seed"]) for row in seed_rows]
    acc = [row["metrics_percent"]["accuracy"] for row in seed_rows]
    macro = [row["metrics_percent"]["macro_f1"] for row in seed_rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    grouped_bars(ax, seeds, acc, macro, "Accuracy", "Macro-F1", "Percent", "Classification: five seed-pooled development results")
    ax.set_ylim(82, 91)
    ax.set_xlabel("Saved-file seed (720 development rows each)")
    save(fig, "classification_seed_metrics.png")

    color = read_json("experiments/results/isic_fuseg_colorfix_v1/result.json")
    historical = {row["metric"]: row for row in color["historical_comparison"]}
    metrics = ("precision", "recall", "f1", "crop_complete95_fraction")
    labels = ["Precision", "Recall", "F1", "Crop complete"]
    before = [historical[key]["historical"] * 100 for key in metrics]
    after = [historical[key]["corrected"] * 100 for key in metrics]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    grouped_bars(ax, labels, before, after, "Historical RGB-as-NumPy", "Corrected BGR NumPy", "Percent", "Controlled correction of the model input color contract")
    ax.set_ylim(0, 105)
    save(fig, "rgb_bgr_controlled_reevaluation.png")

    d2 = read_json("experiments/results/d_seg_small_sampling_v2_multiseed_summary/multiseed_summary.json")
    pairs = d2["pairs"]
    d2_seeds = [str(row["seed"]) for row in pairs]
    control = [row["evaluations"]["C"]["candidate"]["size_recall"]["small"]["matched"] for row in pairs]
    sampling = [row["evaluations"]["S"]["candidate"]["size_recall"]["small"]["matched"] for row in pairs]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    grouped_bars(ax, d2_seeds, control, sampling, "Uniform control", "Small-object sampling", "Matched small GT (support 137)", "D2 paired five-seed small-object results")
    ax.set_ylim(95, 114)
    ax.set_xlabel("Seed; only 42 and 2026 passed the full seed-level gate")
    save(fig, "d2_multiseed_small_tp.png")

    f12 = read_json("experiments/results/f_higher_scale_v2_recovery_seed42_pair_summary/paired_comparison.json")
    c4 = f12["evaluations"]["C4"]["candidate"]
    h4 = f12["evaluations"]["H4-R"]["candidate"]
    labels = ["Precision", "Recall", "F1", "Crop complete", "Very-small recall"]
    c4_values = [c4["precision"] * 100, c4["recall"] * 100, c4["f1"] * 100, c4["crop_complete95_fraction"] * 100,
                 f12["evaluations"]["C4"]["diagnostics"]["very_small"]["recall"] * 100]
    h4_values = [h4["precision"] * 100, h4["recall"] * 100, h4["f1"] * 100, h4["crop_complete95_fraction"] * 100,
                 f12["evaluations"]["H4-R"]["diagnostics"]["very_small"]["recall"] * 100]
    fig, ax = plt.subplots(figsize=(9.3, 4.7))
    grouped_bars(ax, labels, c4_values, h4_values, "Train 768 (C4)", "Train 1024 (H4-R)", "Percent", "F1.2 paired operational evaluation at the same 768 inference scale")
    ax.set_ylim(45, 105)
    save(fig, "f12_higher_scale_pair.png")

    f13 = read_json("experiments/results/f_higher_scale_v2_paired_error_audit/audit_summary.json")
    fp = f13["fp_categories"]
    labels = [row["category"].replace("_", "\n") for row in fp]
    c4_fp = [row["C4"] for row in fp]
    h4_fp = [row["H4"] for row in fp]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    grouped_bars(ax, labels, c4_fp, h4_fp, "C4", "H4-R", "False-positive count", "F1.3 false-positive structure")
    ax.set_ylim(0, max(h4_fp + c4_fp) + 5)
    save(fig, "f13_false_positive_categories.png")

    index = read_json("experiments/results/f_higher_scale_v2_paired_error_audit/paired_error_visualizations/index.json")
    by_id = {row["sample_id"]: row for row in index}
    copied = []
    for name in SELECTED_CASES:
        row = by_id[name]
        source = PAIR_DIR / "paired_error_visualizations" / name
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise RuntimeError(f"Visualization hash mismatch: {name}")
        target = OUT / name
        shutil.copy2(source, target)
        copied.append({"sample_id": name, "reasons": row["reasons"], "sha256": digest,
                       "source": source.relative_to(ROOT).as_posix(), "published_copy": target.relative_to(ROOT).as_posix()})

    manifest = {
        "scope": "Aggregate development charts and five reviewed FUSeg paired-error composites only",
        "source_dataset": "FUSeg / Foot Ulcer Segmentation Challenge",
        "source_repository": "https://github.com/uwm-bigdata/wound-segmentation",
        "pinned_commit": "42a272dfe0679f20675e826385925cb7562934b6",
        "license_evidence": "Challenge PDF records CC BY NC without a version; use is noncommercial academic reporting",
        "clinical_or_field_images_included": False,
        "locked_test_images_included": False,
        "selection_policy": "Five pre-rendered development examples chosen to illustrate FP, multi-GT, very-small loss and crop recovery; not a performance sample",
        "selected_cases": copied,
        "charts": [
            "classification_seed_metrics.png", "rgb_bgr_controlled_reevaluation.png",
            "d2_multiseed_small_tp.png", "f12_higher_scale_pair.png",
            "f13_false_positive_categories.png",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "charts": 5, "case_images": len(copied)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
