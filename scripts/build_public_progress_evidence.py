"""Export aggregate progress evidence; never open image pixels or clinical DBs.

Only explicitly selected report fields are published. Directory counts are file
counts, not unique patients. Run from a complete research checkout.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/evidence/progress_20260915.json"
EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def extract(relative, keys):
    path = ROOT / relative
    if not path.exists():
        return {"source": relative, "available": False}
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    return {"source": relative, "sha256": hashlib.sha256(raw).hexdigest(),
            "values": {key: data[key] for key in keys if key in data}}


def main():
    datasets = []
    roots = sorted(ROOT.glob("yolo_dataset_*")) + [ROOT / "yolo_wound_cls_dataset_v3",
             ROOT / "clinical_detection_isolated_20260812", ROOT / "clinical_seg_candidate_20260820"]
    roots += [ROOT / "outputs/isic_fuseg_pretrain_smoke_20260914" / name
              for name in ("isic_dataset", "fuseg_dataset")]
    for folder in roots:
        if not folder.is_dir():
            continue
        counts = {}
        for split in ("train", "val", "test", "retention_val"):
            candidates = [folder / "images" / split, folder / split / "images", folder / split]
            image_dir = next((p for p in candidates if p.is_dir()), None)
            if image_dir is None:
                continue
            counts[split] = sum(p.is_file() and p.suffix.lower() in EXT for p in image_dir.rglob("*"))
        datasets.append({"dataset": folder.relative_to(ROOT).as_posix(), "file_counts": counts,
                         "count_only_no_pixels_read": True})
    common = ["status", "experiment_id", "stage", "train_images", "train_samples", "val_images",
              "validation_images", "primary_validation_images", "retention_validation_images",
              "train_positive_images", "train_negative_images", "epochs_completed", "epochs_requested",
              "metrics", "primary_validation_metrics", "retention_validation_metrics", "test_images_used"]
    summaries = [extract(p.relative_to(ROOT).as_posix(), common)
                 for p in sorted((ROOT / "outputs/segmentation").glob("D-Seg-*_s42_summary.json"))]
    selected = [
        ("experiments/results/statistics/C-Arch-05-MS_multiseed_summary.json",
         ["method", "seeds", "k_folds", "total_runs", "overall", "between_seed"]),
        ("experiments/results/statistics/C-Arch-05-GKF_gkfold_summary.json", ["data_integrity", "fold_results"]),
        ("experiments/results/statistics/C-Arch-05-MS_bootstrap_ci_report.json", ["total_predictions", "prediction_files", "bootstrap_results"]),
        ("outputs/fuseg_warmup_revision_20260914/result.json", ["status", "metrics", "checkpoint_sha256", "test_images_used"]),
        ("outputs/isic_fuseg_formal_seed42_20260914/protocol.json", ["experiment_id", "training", "formal_seed", "test_images_used"]),
        ("outputs/isic_fuseg_formal_seed42_20260914/status.json", ["status", "stage", "updated_utc", "epoch_completed", "isic_epochs_completed", "metrics", "test_images_used"]),
        ("outputs/isic_fuseg_formal_seed42_20260914/development_gate/result.json",
         ["status", "candidate", "baseline", "gpu_latency", "decision", "five_seeds_started", "sealed_unchanged", "test_images_used"]),
        ("outputs/segmentation/dseg09b_carch05_oof_cascade_bgr_corrected_20260831/summary.json",
         ["status", "sample_count", "class_counts", "integrity", "detector", "full_image_raw", "cascade_raw", "cascade_gated", "paired_comparison", "limitations"]),
        ("outputs/github_progress_verification_20260915/verification.json",
         ["status", "created_utc", "checkpoint_sha256_matches", "sealed_metadata_unchanged", "scope", "test_images_used", "clinical_database_opened", "app_model_replaced"]),
        ("outputs/segmentation/co2wounds_external_test_20260831/D-Seg-09B_CO2Wounds-V2_external_results.json",
         ["status", "evaluation_role", "images_evaluated", "content_groups", "official_unlabeled_test_images_used", "training_or_tuning_images_used_from_co2wounds", "full_cohort_coco_instance_segmentation", "canonical_representative_coco_sensitivity", "fixed_operating_point", "annotation_quality_disclosure", "rerun_permitted"]),
    ]
    evidence = [extract(path, keys) for path, keys in selected]
    checks_path = ROOT / "outputs/github_progress_verification_20260915/verification.json"
    checks = []
    if checks_path.exists():
        checks = [{"check": r["check"], "exit_code": r["exit_code"]}
                  for r in json.loads(checks_path.read_text(encoding="utf-8"))["checks"]]
    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "aggregates only; stored metrics are evidence, not independent certification",
               "dataset_counts": datasets, "historical_segmentation": summaries,
               "evidence": evidence, "engineering_checks": checks,
               "review_notes": [
                   "ISIC-FUSeg fixed-point gate passes RGB ndarray where Ultralytics expects BGR. Its low scores are diagnostic outputs pending corrected validation, not a valid comparison with BGR baseline.",
                   "Classification bootstrap total_predictions=3622 needs reconciliation with 720*5=3600; sealed values are retained as historical reports.",
                   "File counts do not establish licenses or patient-level independence."]}
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(DEST), "datasets": len(datasets), "summaries": len(summaries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
