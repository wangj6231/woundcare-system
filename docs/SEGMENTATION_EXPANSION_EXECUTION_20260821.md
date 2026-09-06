# YOLO11m wound-segmentation expansion execution record

Date: 2026-08-21

## Experimental objective

The deployable front end is treated as a one-class `Wound` localisation and segmentation problem. Diagnostic subtype classification remains a later cascade stage. The immediate development objective is to improve wound localisation/mask quality while preventing source leakage, test-driven tuning, and all-background collapse.

## Protected-data policy

- WSNet/WOUNDSEG images used: 0.
- FUSeg challenge test images used: 0.
- AZH archive test pixels read: 0.
- BUBT healthy images used in validation/test: 0.
- CO2Wounds-V2 external test images used: 0; status remains `LOCKED_NOT_ACCESSED`.
- Every metric below is development validation, not a blind/external-test result.

## Audited data pools

### D-Seg-03 baseline: FUSeg only

- Training: 771 images.
- Fixed validation: 191 images.
- Test: 0 images.
- Initialisation: COCO-pretrained `yolo11m-seg.pt`; no WSNet checkpoint.

### D-Seg-04 positive-source expansion: FUSeg + AZH

- Training: 1,587 images (FUSeg 771 + AZH 816).
- Fixed validation: the same 191 FUSeg images.
- Cross-split exact overlap: 0.
- Cross-split pHash overlap at Hamming distance <= 4: 0.
- One AZH 224×224 patch was excluded because its non-empty mask contained only three foreground pixels and could not form a valid segmentation polygon. The exclusion is recorded rather than silently enlarged or relabelled.

### D-Seg-05 background-control expansion: FUSeg + AZH + BUBT healthy

- Training: 1,899 images (FUSeg 771 + AZH 816 + BUBT 312).
- Fixed validation: the same 191 FUSeg images.
- Positive training images: 1,519.
- Existing empty-label images: 68.
- Added BUBT empty-label images: 312.
- Total empty-label images: 380, equal to 25.02% of positive images.
- BUBT sample is deterministic (seed 42), filename-sex-prefix balanced (156/156), and limited to one image per conservative pHash group.
- BUBT participant identifiers are unavailable; therefore its images are training-only and cannot contribute validation/test metrics.

## BUBT source audit

- Official source: https://data.mendeley.com/datasets/hsj38fwnvr/2
- DOI: `10.17632/hsj38fwnvr.2`.
- Licence: CC BY 4.0.
- Verified healthy files: 2,757 (776 female-prefixed + 1,981 male-prefixed).
- Internal exact duplicates: 0.
- Exact overlaps with FUSeg train/validation: 0/0.
- pHash overlaps with FUSeg train/validation at distance <= 4: 0/0.
- The archive's 2,686 wound images and 2,686 wound masks correspond to the rejected WSNet/WOUNDSEG content and were not opened by the BUBT audit.

## Training curriculum

1. `D-Seg-03`: COCO-pretrained YOLO11m-seg → FUSeg-only baseline.
2. `D-Seg-04`: reset optimiser; initialise from D-Seg-03 best; fine-tune on FUSeg + AZH.
3. `D-Seg-05`: reset optimiser; initialise from D-Seg-04 best; fine-tune with capped BUBT healthy negatives.
4. Compare all stages on the unchanged 191-image FUSeg development validation split.
5. Do not open an external test until the complete protocol, checkpoint, and inference thresholds have been sealed.

All three training stages use image size 768, batch size 4, AdamW, seed 42, deterministic execution, maximum 300 epochs, and patience 80. D-Seg-04/D-Seg-05 use lower learning rates (`2e-4`/`1e-4`) because they are curriculum fine-tuning stages with a reset optimiser.

## Current execution state

- D-Seg-03 is running normally. At epoch 80/300, the development metrics were:
  - Box mAP50: 88.86%.
  - Box mAP50-95: 64.22%.
  - Mask mAP50: 88.27%.
  - Mask mAP50-95: 62.53%.
- These are interim epoch metrics, not the final best-checkpoint report.
- The fail-closed curriculum supervisor is active. It will start D-Seg-04 only after D-Seg-03 produces its expected PASS summary, and D-Seg-05 only after D-Seg-04 produces its PASS summary.
- ISIC 2017 Task 1 images are downloading from the official S3 source for a separate auxiliary-pretraining branch. ISIC will never be counted as wound data or included in wound validation/test metrics.

## Machine-readable evidence

- `outputs/bubt_healthy_negative_audit_20260821.csv/json`
- `outputs/fuseg_azh_manifest_20260821.csv`
- `outputs/fuseg_azh_audit_20260821.json`
- `outputs/fuseg_azh_bubt_manifest_20260821.csv`
- `outputs/fuseg_azh_bubt_audit_20260821.json`
- `outputs/segmentation/curriculum_supervisor_20260821/state.json`
