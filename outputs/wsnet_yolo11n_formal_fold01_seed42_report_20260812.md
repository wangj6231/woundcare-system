# YOLO11n formal GroupKFold run: fold 1 / seed 42

Date: **2026-08-12**  
Status: **COMPLETED_PRIMARY — development benchmark, locked test unused**

## Protocol

- Dataset: official WSNet source-specific dataset
- Source revision: `bdcc1e1cef607bfd7a4f98fc28f31a638a0e0ca9`
- Fold: 1 of 5 exact-image-hash GroupKFold partitions
- Development fold: 1,844 train images / 462 validation images
- Train/validation exact-hash overlap: 0
- Model: YOLO11n pretrained initialization
- Seed: 42
- Maximum epochs: 150
- Completed epochs: 147 recorded epochs; early stopping terminated training before the 150-epoch ceiling
- Batch: 4
- Image size: 640
- Optimizer: SGD (`lr0=0.01`, `lrf=0.01`, momentum 0.937, weight decay 0.0005)
- Locked test split: **not used**

## Validation metrics from the registered best-weight evaluation

| Metric | Value |
|---|---:|
| Precision | 0.6490 |
| Recall | 0.4378 |
| mAP@50 | 0.4909 |
| mAP@50–95 | 0.2283 |

These are one fold/one seed development results. They are not yet a final model result, confidence interval, or locked-test result.

## Primary artifacts

- Run directory: `experiments/results/detection_groupkfold/YOLO11n_WSNet_groupkfold_fold01_seed42_202608123/`
- Best weights: `experiments/results/detection_groupkfold/YOLO11n_WSNet_groupkfold_fold01_seed42_202608123/weights/best.pt`
- Last weights: `experiments/results/detection_groupkfold/YOLO11n_WSNet_groupkfold_fold01_seed42_202608123/weights/last.pt`
- Training history: `experiments/results/detection_groupkfold/YOLO11n_WSNet_groupkfold_fold01_seed42_202608123/results.csv`
- Registered JSON: `outputs/wsnet_groupkfold_runs_20260812/YOLO11n_WSNet_groupkfold_fold01_seed42_20260812.json`

## SHA-256 records

- `best.pt`: `3DE18C770B6E397C8925DF1AE71910FF1F2AE083D4262C9E8A278CE6EBEAF326`
- `last.pt`: `76A26CB7306153E7AB07C96F07D5AD11138DD0DCE76742D276D89BF961C08ABA`
- `results.csv`: `E5051DDF5E48271D0A20B7C18C65734468F47D4629E6456743459BA6AA6A16AE`

## Duplicate invocation handling

An overlapping invocation created `YOLO11n_WSNet_groupkfold_fold01_seed42_202608124/`. It is retained as `DUPLICATE_NOT_FOR_STATISTICS` and is not merged with the primary run. The runner was corrected so future summaries record the actual auto-suffixed result directory, preventing summary collisions.
