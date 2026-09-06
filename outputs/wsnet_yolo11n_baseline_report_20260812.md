# YOLO11n WSNet official development baseline

Date: **2026-08-12**  
Status: **PASS — development baseline, not a formal paper result**

## Protocol

- Dataset: `yolo_dataset_wsnet_official_v1/dataset.yaml`
- Source revision: `bdcc1e1cef607bfd7a4f98fc28f31a638a0e0ca9`
- Licence: CC BY-NC 4.0; internal non-commercial research only
- Model: YOLO11n pretrained initialization
- Train/val/test: 1,894 / 412 / 380 images
- Training: 5 epochs, full training split, image size 640, batch 16, seed 42
- Test split: **not used**

## Validation metrics

| Metric | Value |
|---|---:|
| Precision | 0.4338 |
| Recall | 0.3548 |
| mAP@50 | 0.3049 |
| mAP@50–95 | 0.1144 |

These are early development metrics after only five epochs. They are not suitable for confidence intervals, final model selection, or publication claims.

## Artifacts

- Training script: `work/run_wsnet_yolo11n_baseline.py`
- Best weights: `experiments/results/detection/YOLO11n_WSNet_official_baseline_20260812/weights/best.pt`
- Training log: `experiments/results/detection/YOLO11n_WSNet_official_baseline_20260812/results.csv`
- Summary: `outputs/wsnet_yolo11n_baseline_summary_20260812.json`

The next valid step is a controlled multi-model benchmark with a pre-declared batch policy, followed by 5-fold × 5-seed evaluation and locked test evaluation only after model selection.
