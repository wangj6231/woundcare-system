# YOLO11m formal public candidate smoke test report

Date: 2026-08-12  
Status: **PASS — pipeline and artifact validation only**

## Scope

This was a one-epoch smoke test of the public-source candidate dataset. It was performed to verify that the dataset YAML, YOLO11m training path, validation path, and prediction JSON export operate end-to-end. It is **not** a formal detection result and must not be used for model selection or publication claims.

- Dataset: `yolo_dataset_wound_formal_public_candidate_v1/dataset.yaml`
- Model: YOLO11m (pretrained initialization)
- Training: 1 epoch, 2% fraction, image size 320, batch 4, seed 42
- Candidate composition: 2,562 mask-backed Kaggle-derived images only
- Split: train 1,791; validation 385; test 386
- License/provenance status: **PENDING**; formal training remains blocked until upstream terms and versions are evidenced.

## Smoke metrics

| Metric | Value |
|---|---:|
| Precision | 0.0124 |
| Recall | 0.0107 |
| mAP@50 | 0.0027 |
| mAP@50–95 | 0.0012 |

The low values are expected from a one-epoch, 2%-fraction smoke run and are not estimates of final model performance.

## Verification outcome

- YOLO11m model instantiated successfully with one detection class.
- Training completed successfully on the configured GPU.
- Validation completed on the candidate validation split.
- Prediction JSON export completed.
- Best and last weights were written.
- No formal test-set evaluation was performed.

## Artifacts

- Dataset YAML: `yolo_dataset_wound_formal_public_candidate_v1/dataset.yaml`
- Smoke summary: `outputs/detection_formal_public_smoke_summary_20260812.json`
- Weights: `experiments/results/detection_smoke/YOLO11m_formal_public_candidate_smoke_20260812/weights/best.pt`
- Validation predictions: `experiments/results/detection_smoke/YOLO11m_formal_public_candidate_smoke_20260812_val/predictions.json`

## Reproducibility note

Ultralytics emitted permission warnings while attempting to write `%APPDATA%\\Ultralytics\\settings.json` and `persistent_cache.json`. The warnings did not prevent training or artifact creation. They should be resolved or documented before the final benchmark environment is frozen.

## Release gate

Do not start the formal multi-model / 5-fold / multi-seed benchmark or publish detection results until the provenance and licensing ledger is completed for FUSC, Medetec, WSNet, and any additional source included in the analysis.
