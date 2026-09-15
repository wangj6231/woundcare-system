"""Independently check persisted predictions and render development diagnostics.

No models are loaded. No model inference, training or test access is performed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .audit_project import read_json, sha, snapshot_sealed, write_json
from .localization_benchmark import output_path


def verify(out: Path):
    result = read_json(out / "result.json")
    protocol = read_json(out / "protocol.json")
    cohort = read_json(out / "cohort.json")
    ids = {r["image_id"] for r in cohort}
    if len(ids) != 191 or any(r["split"] != "val" for r in cohort):
        raise ValueError("unexpected validation cohort")
    if result["status"] != "PASS_DEVELOPMENT_EVALUATION_ONLY" or result["test_images_used"] != 0:
        raise ValueError("run is not successful development-only evidence")
    checked, primary = 0, {}
    for model in protocol["models"]:
        for nms in protocol["nms_grid"]:
            dest = out / f"{model}_nms{nms:.2f}"
            raw = read_json(dest / "predictions.json")
            if len(raw) != 191 or {r["image_id"] for r in raw} != ids:
                raise ValueError("raw predictions are incomplete")
            raw_by_id = {r["image_id"]: r for r in raw}
            for r in raw:
                mask_path = (out / r["mask_file"]).resolve()
                if out.resolve() not in mask_path.parents or sha(mask_path) != r["mask_file_sha256"]:
                    raise ValueError("saved masks changed")
                with np.load(mask_path, allow_pickle=False) as saved:
                    if saved["masks"].shape != (len(r["confidence"]), 512, 512):
                        raise ValueError("mask-prediction count mismatch")
            for conf in protocol["confidence_grid"]:
                rows = read_json(dest / f"metrics_conf{conf:.2f}.json")
                if len(rows) != 191 or {r["image_id"] for r in rows} != ids:
                    raise ValueError("metric rows incomplete")
                for r in rows:
                    raw_row = raw_by_id[r["image_id"]]
                    expected = [i for i, c in enumerate(raw_row["confidence"]) if c >= conf]
                    if r["pred_boxes"] != [raw_row["boxes_xyxy"][i] for i in expected]:
                        raise ValueError("threshold selection not reproducible from stored raw predictions")
                    if r["confidences"] != [raw_row["confidence"][i] for i in expected]:
                        raise ValueError("confidence values changed")
                    if r["tp"] + r["fp"] != len(expected) or r["tp"] + r["fn"] != len(r["gt_boxes"]):
                        raise ValueError("denominator excludes misses or false positives")
                    pairs = r["pairs"]
                    if (r["tp"] != len(pairs) or len({p["gt"] for p in pairs}) != len(pairs)
                            or len({p["prediction"] for p in pairs}) != len(pairs)):
                        raise ValueError("non-unique one-to-one match")
                    for p in pairs:
                        a, b = np.array(r["pred_boxes"][p["prediction"]]), np.array(r["gt_boxes"][p["gt"]])
                        intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
                        overlap = intersection / ((a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - intersection)
                        if overlap < .5 or not np.isclose(overlap, p["iou"]):
                            raise ValueError("reported TP fails the declared IoU requirement")
                    if r["gt_pixels"] and r["crop"] is None and (r["crop_coverage"] != 0 or r["crop_complete95"]):
                        raise ValueError("empty crop incorrectly counted as success")
                grid = next(r for r in result["grid"] if r["model"] == model and r["nms_iou"] == nms and r["confidence"] == conf)
                tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
                positives = [r for r in rows if r["gt_pixels"]]
                if tp + fn != 241 or len(positives) != 186:
                    raise ValueError("wrong fixed instance/positive denominator")
                expected_summary = {"tp": tp, "fp": fp, "fn": fn,
                    "precision": tp / (tp + fp), "recall": tp / (tp + fn),
                    "f1": 2 * tp / (2 * tp + fp + fn),
                    "crop_complete95_fraction": sum(r["crop_complete95"] for r in positives) / 186}
                if any(not np.isclose(grid[k], v) for k, v in expected_summary.items()):
                    raise ValueError("summary arithmetic does not match per-image records")
                if conf == .1 and nms == .7:
                    primary[model] = grid
                checked += len(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    colors = {"D-Seg-03": "#64748b", "D-Seg-03R": "#087f8c"}
    for name in protocol["models"]:
        for nms in protocol["nms_grid"]:
            rows = sorted([r for r in result["grid"] if r["model"] == name and r["nms_iou"] == nms], key=lambda r: r["confidence"])
            for ax, field in zip(axes, ["f1", "recall", "crop_complete95_fraction"]):
                ax.plot([r["confidence"] for r in rows], [100 * r[field] for r in rows],
                        "o-" if nms == .7 else "x--", color=colors[name], label=f"{name}, NMS={nms:.1f}")
                ax.set_xlabel("Confidence threshold")
                ax.set_ylabel("Percent")
                ax.grid(alpha=.2)
    for ax, title in zip(axes, ["Box F1 (IoU >= .50)", "Box recall / 241 GT instances", "Crop retains >=95% pixels / 186 positives"]):
        ax.set_title(title, fontsize=10)
        ax.axvline(.1, color="#d97706", alpha=.5, linewidth=1)
    axes[0].legend(fontsize=8)
    fig.suptitle("FUSeg development only | 191 images | full predeclared grid | primary confidence = 0.10", fontsize=12)
    fig.savefig(out / "operating_point_curves.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), constrained_layout=True)
    for ax, (name, r) in zip(axes, primary.items()):
        data = np.array([[r["tp"], r["fn"]], [r["fp"], np.nan]])
        ax.imshow(np.ma.masked_invalid(data), cmap="Blues", vmin=0, vmax=241)
        for y in range(2):
            for x in range(2):
                value = "not defined" if np.isnan(data[y, x]) else str(int(data[y, x]))
                ax.text(x, y, value, ha="center", va="center", color="black", fontsize=14)
        ax.set_xticks([0, 1], ["Predicted wound", "No matched prediction"])
        ax.set_yticks([0, 1], ["GT wound", "Unmatched prediction"])
        ax.tick_params(axis="both", labelsize=8)
        ax.set_title(name)
    fig.suptitle("Object matching counts: TP / FN / FP (true negatives undefined)\nDevelopment primary conf=.10, NMS=.70, matching IoU>=.50", fontsize=10)
    fig.savefig(out / "object_matching_matrix.png", dpi=160)
    plt.close(fig)
    unchanged = snapshot_sealed() == read_json(out / "sealed_before.json")
    if not unchanged:
        raise ValueError("sealed artifact changed after execution")
    write_json(out / "verification.json", {"status": "PASS_STORED_PREDICTION_VERIFICATION",
        "checked_metric_rows": checked, "model_nms_passes": 4, "operating_points": 28,
        "raw_image_predictions": 764, "sealed_unchanged": unchanged, "test_images_used": 0,
        "result_sha256": sha(out / "result.json"), "verifier_sha256": sha(Path(__file__)),
        "limitations": "checks persisted predictions, mask integrity, matching and arithmetic; not an independent annotation audit or clinical validation"})
    print("PASS: 764 raw image predictions, 28 operating points, 5348 metric rows; no model rerun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    verify(output_path(parser.parse_args().output))
