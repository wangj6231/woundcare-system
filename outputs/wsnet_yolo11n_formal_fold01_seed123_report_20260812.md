# YOLO11n formal GroupKFold run: fold 1 / seed 123

Date: **2026-08-12**  
Status: **COMPLETED — development benchmark, locked test unused**

## Protocol

- Dataset: official WSNet source-specific dataset
- Source revision: `bdcc1e1cef607bfd7a4f98fc28f31a638a0e0ca9`
- Fold: 1 of 5 exact-image-hash GroupKFold partitions
- Fold 1 train/validation: 1,844 / 462 images
- Train/validation exact-hash overlap: 0
- Model: YOLO11n pretrained initialization
- Seed: 123
- Maximum epochs: 150
- Completed epochs: 135; early stopping terminated training before the ceiling
- Batch: 4
- Image size: 640
- Optimizer: SGD (`lr0=0.01`, `lrf=0.01`, momentum 0.937, weight decay 0.0005)
- Locked test split: **not used**

## Registered best-weight validation metrics

| Metric | Value |
|---|---:|
| Precision | 0.5964 |
| Recall | 0.4568 |
| mAP@50 | 0.4943 |
| mAP@50–95 | 0.2252 |

These are one development fold/seed results. They are not a final detector result, confidence interval, or locked-test result.

## Artifacts

- Run directory: `experiments/results/detection_groupkfold/YOLO11n_WSNet_groupkfold_fold01_seed123_20260812/`
- Best weights: `experiments/results/detection_groupkfold/YOLO11n_WSNet_groupkfold_fold01_seed123_20260812/weights/best.pt`
- Last weights: `experiments/results/detection_groupkfold/YOLO11n_WSNet_groupkfold_fold01_seed123_20260812/weights/last.pt`
- Training history: `experiments/results/detection_groupkfold/YOLO11n_WSNet_groupkfold_fold01_seed123_20260812/results.csv`
- Registered JSON: `outputs/wsnet_groupkfold_runs_20260812/YOLO11n_WSNet_groupkfold_fold01_seed123_20260812.json`

## SHA-256

- `best.pt`: `AB6BBD75EDFCE1A26157F12E0A845C93E2185A60AF487E8C62F3AB35D0B98610`
- `last.pt`: `FF2C50BB60A4429C3404AE897075358475CFE8677153644B46928DB3F4A51252`
- `results.csv`: `E2DAE2D107BB1754262DA84A6ED8731628C84F8A99735E7F1E38985BCD001386`

## Interpretation

The registered mAP@50 is close to the completed seed-42 fold-1 run (0.4909 versus 0.4943), while mAP@50–95 is also similar (0.2283 versus 0.2252). This is an early indication of seed stability for this fold, not a substitute for all five folds and five seeds.
