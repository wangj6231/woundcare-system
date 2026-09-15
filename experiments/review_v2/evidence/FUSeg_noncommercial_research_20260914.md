# FUSeg official data-use evidence (2026-09-14)

## Primary evidence inspected

- Official repository: https://github.com/uwm-bigdata/wound-segmentation
- Pinned checkout: `42a272dfe0679f20675e826385925cb7562934b6`.
- Publisher PDF: https://github.com/uwm-bigdata/wound-segmentation/blob/42a272dfe0679f20675e826385925cb7562934b6/data/Foot%20Ulcer%20Segmentation%20Challenge/FootUlcerSegmentationChallenge2021.pdf
- Local PDF: `official_detection_sources_20260812/fuseg/repository/data/Foot Ulcer Segmentation Challenge/FootUlcerSegmentationChallenge2021.pdf`.
- Local and pinned upstream PDF SHA256 independently matched: `f9a44fc14bc7589d03ce5c17307ad97e9dde5fcc0e2c596c8b86010a6d9547aa`.
- The COMPLETE relevant page 5 was extracted and visually inspected. Under Data usage agreement, the filled-in answer below the template examples is **“CC BY NC.”** This is a data-use response, not the article's publication license or the repository's code license.
- The document does not specify a Creative Commons license version here. Do not invent 4.0 or unrestricted rights.

## Admission scope

The present local training is admitted only as an attributed, non-commercial, offline academic/graduation model-development experiment. No clinical decisions, deployment, commercial use, public model release, image redistribution, or patient-level independence claim is authorized by this record. This narrow experiment admission does not establish that a resulting model can later be used in a commercial nursing product.

Source attribution: Wang et al., FUSeg: The Foot Ulcer Segmentation Challenge, https://arxiv.org/abs/2201.00414 ; UWM Big Data Lab / AZH Wound and Vascular Center.

## Data boundary

- Only the already screened FUSeg development train/validation inventory: 771 train and 191 validation images.
- Official unlabeled test, classification Blind Test (48 images), CO2Wounds, and all clinical/Yasin/Roboflow/WSNet/AZH-patch/BUBT images are excluded from THIS experiment.
- Initialize from the existing generic COCO YOLO11m-seg checkpoint, not D-Seg-07/08/09B or any other wound-tuned checkpoint.
- This evidence does not extend the FUSeg challenge license to the separate AZH patches or to other datasets.

The earlier UNKNOWN entry reflected an incomplete evidence review. This record adds the previously missed official PDF evidence; it does not erase the historical audit or manufacture upstream permission.
