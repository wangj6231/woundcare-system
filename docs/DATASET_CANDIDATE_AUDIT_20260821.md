# Public Segmentation Dataset Candidate Audit - 2026-08-21

## Decision

No single public dataset currently satisfies all of the following at once: large scale, independent wound source, pixel masks, immediate download, verified provenance, and a licence compatible with later system development. The recommended expansion therefore separates wound-specific training, auxiliary segmentation pretraining, negative-image training, and external testing.

## Recommended sources

| Source | Images / records | Annotation | Licence | Intended role | Status |
|---|---:|---|---|---|---|
| WoundsDB | 188 wound records from 79 visits / 47 patients | Expert wound masks; colour plus multimodal data | CC BY 4.0 | New wound-positive source; patient-group split required | Candidate; registration required; website TLS certificate currently invalid |
| ISIC 2017 Task 1 | 2,000 training images plus masks | Binary skin-lesion masks | CC0 | Auxiliary segmentation pretraining only, followed by wound fine-tuning | Approved candidate; do not mix into wound validation or report as wound data |
| BUBT healthy-foot subset | 2,757 verified (776 female-prefixed + 1,981 male-prefixed) | Image-level healthy label; use empty segmentation labels | CC BY 4.0 | Training-only negatives to reduce false-positive wound detections | Audited: 0 exact/pHash overlap with FUSeg train/val; patient IDs unavailable, so validation/test use is prohibited; all wound folders excluded |
| REDSCAR | 394 post-surgical wound images | Binary wound and staple masks | Access controlled; site states all rights reserved | Future surgical-wound domain candidate only after written use permission | Conditional candidate; response may take up to five days |
| Pressure-ulcer Figshare set | 20 images | LabelMe boundary annotations | CC BY 4.0 | Qualitative edge cases only | Too small for primary training |

## Excluded or protected sources

| Source | Reason |
|---|---|
| DFUC2022 | Current licence prohibits clinical and commercial purposes and requires an authorised institutional signatory; excluded from the deployable model |
| WOUNDSEG / WSNet, 2,686 images | Previously evaluated and rejected because of poor transfer; must not be reintroduced through Mendeley or Kaggle repackaging |
| Lower Limb and Feet Wound Image Dataset - wound folders | The 2,686 wound images originate from the WOUNDSEG / WSNet collection; only the independently collected healthy-foot subset is eligible |
| CO2Wounds-V2 | Reserved external test source; never use for training, tuning, threshold selection, or early stopping |
| Roboflow/Kaggle compilations | Excluded unless upstream source, version, annotation provenance, and licence can all be verified |

## Recommended leakage-controlled experiment

1. Pretrain the segmentation model on ISIC 2017 Task 1 under a separate experiment ID.
2. Reset the optimiser and fine-tune on the approved wound-positive development pool only.
3. Add an audited, capped sample of BUBT healthy-foot images as empty-label negatives to training only. Do not place them in validation/test because participant identifiers are unavailable; preserve their pHash groups in the manifest.
4. Add WoundsDB only after safe registration and MD5/pHash/patient-group auditing.
5. Perform clinical-domain fine-tuning using the project's `Wound` masks without introducing diagnostic subtype labels.
6. Select thresholds using development validation only.
7. Evaluate once on the protected CO2Wounds-V2 external test after the complete protocol is sealed.

## Required safeguards

- Never mix ISIC images into wound validation metrics.
- Never randomly split multiple images from the same participant or patient across train and validation.
- Audit exact and perceptual duplicates across every source before materialising a combined dataset.
- Cap healthy negative sampling to avoid all-background collapse.
- Keep source IDs in every manifest and prediction record.
- Report source-specific metrics in addition to pooled metrics to expose domain failure.

## Official source links

- WoundsDB: https://chronicwounddatabase.eu/
- WoundsDB licence: https://chronicwounddatabase.eu/Terms
- ISIC challenge data: https://challenge.isic-archive.com/data/
- Lower Limb and Feet Wound Dataset: https://data.mendeley.com/datasets/hsj38fwnvr/2
- REDSCAR: https://redscar.uib.es/dataset.html
- Pressure-ulcer Figshare dataset: https://figshare.com/articles/dataset/images_of_pressure_ulcer_2_/17206940
- CO2Wounds-V2: https://data.mendeley.com/datasets/s2w7rjwz49/1
