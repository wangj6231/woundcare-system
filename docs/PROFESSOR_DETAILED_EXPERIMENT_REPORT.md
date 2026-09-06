# Detailed Experimental Progress and Results Report

**Project:** Intelligent Wound Grading and Care Correspondence System  
**Report date:** 2026-08-12  
**Purpose:** professor review of the complete classification pipeline, the YOLO
detection extension, dataset counts, protocols, and current evidence status.

## Executive summary

The classification pipeline (Phases 1–9) is sealed under an MD5-aware,
leakage-free protocol. The final development estimate is based on 25 runs
(5 seeds × 5 GroupKFold partitions), while the 48-image blind test was used
once and kept separate from cross-validation.

The detection pipeline is operational and has completed dataset intake, audit,
smoke tests, and early GroupKFold development runs. It has **not** yet reached
a final multi-model, 5-seed × 5-fold detection benchmark or a locked detection
test claim. Public-source licensing is still recorded as
`BLOCKED / QUARANTINED`; source links and manifests are provided for review.

## 1. Dataset inventory and image counts

### 1.1 Classification dataset (`yolo_wound_cls_dataset_v3`)

The classification dataset contains 768 images in total. The development set is
679 training + 41 validation images (720 images), and the blind test remains
sequestered at 48 images.

| Class | Train | Validation | Blind test | Total |
|---|---:|---:|---:|---:|
| Abrasions | 97 | 8 | 9 | 114 |
| Bruises | 97 | 12 | 13 | 122 |
| Burns | 97 | 5 | 7 | 109 |
| Cut | 97 | 5 | 5 | 107 |
| Ingrown_nails | 97 | 3 | 4 | 104 |
| Laceration | 97 | 6 | 7 | 110 |
| Stab_wound | 97 | 2 | 3 | 102 |
| **Total** | **679** | **41** | **48** | **768** |

Content-group audit of the 720-image development set:

- 383 unique MD5 content groups.
- 177 singleton groups (177 images).
- 206 duplicate groups containing 543 files.
- 337 redundant instances (`543 - 206`).
- 0 MD5 groups crossed the final GroupKFold boundaries.

### 1.2 Detection intake manifest (historical mixed-source inventory)

The file-level detection manifest contains 3,090 labelled image records. These
counts are reported from the manifest, not inferred from filenames alone:

| Source group | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `clinical_frame` | 130 | 40 | 7 | 177 |
| `roboflow` | 141 | 36 | 0 | 177 |
| `kaggle_fusc` | 647 | 539 | 0 | 1,186 |
| `kaggle_medetec` | 374 | 0 | 0 | 374 |
| `kaggle_wsnet` | 1,176 | 0 | 0 | 1,176 |
| **Total** | **2,468** | **615** | **7** | **3,090** |

This mixed inventory is retained for provenance audit only. Clinical and
Roboflow records are not released here, and the Kaggle-derived groups are not
treated as formally cleared data.

### 1.3 Source-specific WSNet detection benchmark

The reproducible public-source candidate uses only official WSNet image-mask
pairs. Masks are converted to one wound bounding box class using external
contours with area ≥ 50 pixels.

| Split | Images | Bounding boxes |
|---|---:|---:|
| Train | 1,894 | 2,681 |
| Validation | 412 | 609 |
| Test (locked for detection) | 380 | 533 |
| **Total** | **2,686** | **3,823** |

The test split was not used in the reported early development runs.

### 1.4 Quarantined public-candidate branch

The mask-derived Kaggle candidate contains 2,562 images:

| Source | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Kaggle FUSC-derived | 752 | 163 | 164 | 1,079 |
| Kaggle Medetec-derived | 262 | 56 | 56 | 374 |
| Kaggle WSNet-derived | 777 | 166 | 166 | 1,109 |
| **Total** | **1,791** | **385** | **386** | **2,562** |

This branch was used only for a one-epoch pipeline smoke test. Its formal-use
gate remains pending because the exact upstream versions and redistribution
permissions are not fully evidenced.

## 2. Classification experiment Phases 1–9

### Phase 1 — Logger and schema

Standardized experiment records, configuration fields, dataset labels, seed,
fold, metrics, and integrity status. **Status: PASS.**

### Phase 2 — YOLO classification trainer

Implemented the Ultralytics classification training path and checkpoint
registration. **Status: PASS.**

### Phase 3 — PyTorch classification trainer

Implemented the independent PyTorch training path for architecture comparison.
**Status: PASS.**

### Phase 4 — Evaluation

Added Top-1/Top-5 accuracy, Macro-F1, Weighted-F1, per-class recall,
confusion-matrix, and ROC-AUC calculation. **Status: PASS.**

### Phase 5 — Orchestration

Connected configuration, training, evaluation, logging, and result storage.
**Status: PASS.**

### Phase 6 — Naive image-level StratifiedKFold

The initial estimate was 98.34% ± 1.21%, but the audit found 178 of 206
duplicate content groups crossing folds. This result is permanently marked
`INVALIDATED_BY_DATA_INTEGRITY_AUDIT` and is not used as the final result.

### Phase 6.5 — Data-integrity audit

MD5 duplicate grouping and split-overlap checks were added. The audit confirmed
the leakage mechanism and established the requirement for group-aware splitting.

### Phase 6.6 — MD5-aware GroupKFold

The corrected split assigns complete MD5 groups to folds. The final protocol
records zero cross-fold content leakage.

### Phase 7 — Multi-seed GroupKFold

Final model: **C-Arch-05 / YOLOv8n-cls**.  
Configuration: `imgsz=224`, `batch=16`, `epochs=150`, `optimizer=auto`,
`lr0=0.01`, seed set `{42, 123, 3407, 2026, 999}`.  
Runs: 5 seeds × 5 folds = **25 leakage-free evaluation runs**.

| Metric | Mean ± run SD |
|---|---:|
| Top-1 Accuracy | 87.40% ± 2.78% |
| Macro-F1 | 87.36% ± 3.11% |
| Stab_wound Recall | 88.98% ± 9.30% |
| Between-seed accuracy SD | 0.76% |
| Between-seed Macro-F1 SD | 0.81% |

### Phase 8 — Bootstrap statistical analysis

Non-parametric percentile bootstrap with `B=2,000` resamples was applied to
pooled validation predictions (`n=3,622`).

| Metric | Mean | 95% CI |
|---|---:|---:|
| Top-1 Accuracy | 87.40% | 86.32–88.48% |
| Macro-F1 | 87.36% | 86.16–88.56% |
| Weighted-F1 | 87.42% | 86.32–88.52% |
| Stab_wound Recall | 88.98% | 85.37–92.59% |

Per-class recall means: Abrasions 85.12%, Bruises 88.45%, Burns 87.90%, Cut
86.35%, Ingrown_nails 89.10%, Laceration 85.62%, and Stab_wound 88.98%.

### Phase 9 — Paper-table generation

Automatically generated Markdown and CSV tables for dataset protocol,
classification performance, bootstrap intervals, leakage impact, and blind
test generalization. **Status: SEALED.**

### Final classification blind test

The 48-image sequestered test set was evaluated once after model selection; no
post-test tuning was performed.

| Metric | Result |
|---|---:|
| Top-1 Accuracy | 95.83% (46/48) |
| Top-5 Accuracy | 100.00% |
| Macro-F1 | 94.76% |
| Weighted-F1 | 96.23% |
| Macro ROC-AUC | 0.9982 |
| Total errors | 2 / 48 |

## 3. Detection experiment Phases D1–D7

### D1 — Configuration and infrastructure

Prepared the YOLO11m detection configuration and connected the generic
experiment/logging path used by the development runs. Primary image size is
640; the formal smoke path uses a reduced 320-pixel setting only for pipeline
verification. The detection-specific trainer/evaluator scripts are not yet
sealed as a final D1 implementation, so the current detection metrics remain
registered development artifacts rather than a completed production pipeline.

### D2 — Dataset and split-integrity audit

Checked image/label pairs, malformed labels, bounding-box counts, MD5 grouping,
and train/validation/test isolation. The official WSNet source-specific branch
contains 2,686 image-mask pairs and uses deterministic hash-group assignment.

### D3 — Single-run smoke tests

The YOLO11m public-candidate smoke test ran for one epoch on 2% of the candidate
training data (`imgsz=320`, batch 4, seed 42). It verified trainer → validator →
prediction JSON → artifact logging, but its metrics are not scientific results:

| Precision | Recall | mAP@50 | mAP@50–95 |
|---:|---:|---:|---:|
| 0.0124 | 0.0107 | 0.0027 | 0.0012 |

### D4 — GroupKFold detection development run

YOLO11n was run on one of five exact-image-hash folds. Fold 1 contains 1,844
training and 462 validation images, with zero train/validation hash overlap.

| Run | Epochs completed | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|---:|
| YOLO11n, fold 1, seed 42 | 147/150 | 0.6490 | 0.4378 | 0.4909 | 0.2283 |
| YOLO11n, fold 1, seed 123 | 135/150 | 0.5964 | 0.4568 | 0.4943 | 0.2252 |

Both are development-fold results; the locked detection test was unused.

### D5 — Multi-model and multi-seed benchmark

YOLO11m, YOLO11s, and YOLO11n configurations and representative weights are
registered. The complete 5-model × 5-seed × 5-fold detection benchmark is not
yet complete, so no final detector ranking or confidence interval is reported.

### D6 — Detection paper tables

Detection summaries, provenance ledgers, source manifests, and smoke/development
reports are prepared for review. A formal detection table will only be sealed
after the licensing gate and full benchmark are completed.

### D7 — Detection blind test

The 380-image WSNet detection test split remains locked and unused. No detection
blind-test score is claimed in this report.

## 4. Reproducibility and governance files

## 5. Data balancing and detailed training controls (added for professor review)

The balancing operation was restricted to the training split. Validation and the
sequestered blind test were not artificially resampled, so their observed class
distribution remains suitable for unbiased evaluation. The classification
configuration records `Wound_dataset+Oversampling+Rotation`; the resulting
training directory contains exactly 97 images per class (679 images total),
while validation contains 41 images and the blind test contains 48 images.

| Class | Train | Validation | Blind test | Total |
|---|---:|---:|---:|---:|
| Abrasions | 97 | 8 | 9 | 114 |
| Bruises | 97 | 12 | 13 | 122 |
| Burns | 97 | 5 | 7 | 109 |
| Cut | 97 | 5 | 5 | 107 |
| Ingrown_nails | 97 | 3 | 4 | 104 |
| Laceration | 97 | 6 | 7 | 110 |
| Stab_wound | 97 | 2 | 3 | 102 |
| **Total** | **679** | **41** | **48** | **768** |

Training-only augmentation was configured as `fliplr=0.5`, `flipud=0.5`,
`degrees=45`, `scale=0.5`, and `erasing=0.4`. The exact-content MD5 audit was
performed after dataset construction: 383 unique groups were found in the
720-image Development set (177 singletons and 206 duplicate groups containing
543 files, or 337 redundant instances). Complete MD5 groups were assigned to a
single fold, yielding zero cross-fold MD5 overlap.

The classification training protocol used YOLOv8n-cls (C-Arch-05),
`imgsz=224`, `batch=16`, `epochs=150`, `optimizer=auto`, `lr0=0.01`, and
`weight_decay=0.0005`. Seeds `{42, 123, 3407, 2026, 999}` were crossed with
five MD5-aware StratifiedGroupKFold partitions (25 leakage-free runs). The
Development estimate was Top-1 Accuracy `87.40% ± 2.78%` and Macro-F1
`87.36% ± 3.11%`; the 48-image blind test was evaluated once only and was not
averaged with cross-validation.

For detection, the formal public candidate uses the official WSNet split with
1,894/412/380 images and 2,681/609/533 boxes for train/validation/locked test.
WSNet has one detection class (`Wound`), so the balancing objective is split and
source control rather than seven-class oversampling. The formal comparison uses
`imgsz=640`, `batch=4`, up to 150 epochs, patience 20, deterministic execution,
and workers 0; D-Arch-01's 5-epoch/patience-3 settings are smoke/baseline
settings only. The locked detection test remains unopened, and no final
detection blind-test score is claimed.

The detector outputs a single wound box while the classifier predicts seven
classes. An end-to-end cascade score will only be reported after a paired,
versioned detector-crop/classifier test set is created; detection mAP and
classification accuracy must not be multiplied as if they were independent.

- [Professor review index](./PROFESSOR_REVIEW_INDEX.md)
- [Detection provenance/licence evidence](../outputs/detection_provenance_license_evidence_20260812.md)
- [Official WSNet intake report](../outputs/wsnet_official_intake_report_20260812.md)
- [Classification tables](../experiments/results/tables/)
- [Detection experiment configuration](../experiments/configs/D-Arch-01_yolo11m.yaml)

All reported counts and metrics above are copied from the registered manifests,
tables, and run reports. Unverified clinical or third-party provenance is
explicitly labelled instead of being inferred.
