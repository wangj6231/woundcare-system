# Detection provenance and licence evidence review

Date reviewed: **2026-08-12**  
Decision: **Evidence pack completed; formal-use gate remains BLOCKED**

This record separates technical provenance from authorization. A local hash match proves which files were used; it does not, by itself, grant permission to train, publish, redistribute, or release derived labels.

## Local evidence completed

The workspace contains a reproducible file-level mapping:

| Local group | Rows | Exact SHA-256 match to `data_wound_seg/` | Origin IDs mapped |
|---|---:|---:|---:|
| FUSC | 1,186 | 1,186 | 1,186 |
| Medetec | 374 | 374 | 374 |
| WSNet | 1,176 | 1,176 | 1,176 |

The mapping is recorded in [local_dataset_provenance_mapping_20260812.csv](local_dataset_provenance_mapping_20260812.csv) and is based on `correspondence_table.xlsx`. This establishes local identity and reproducibility, but not the original download version or rights.

## Public-source evidence

### Kaggle compilation

Source page: [Kaggle — Wound images segmentation (2760 samples)](https://www.kaggle.com/datasets/leoscode/wound-segmentation-images)

The page identifies the compilation as Medetec + FUSeg + WSNet, reports 2,760 samples, asks users to cite the underlying works, states that a substantial portion of annotations were not clinician-validated and that the dataset is for research only, and displays an MIT licence for the Kaggle compilation. The workspace does not contain a saved Kaggle archive, version identifier, download date, or metadata snapshot. Therefore the Kaggle page is recorded as a source reference, not as proof that the uploader could relicense every upstream image and annotation.

**Decision: RESTRICTED — research-only evidence recorded; upstream rights and exact version still require confirmation.**

### FUSeg / FUSC

Primary project evidence: [UWM wound-segmentation FUSeg README](https://github.com/uwm-bigdata/wound-segmentation/blob/master/data/Foot%20Ulcer%20Segmentation%20Challenge/README.MD)

The project README describes an open challenge, says the images were collected from clinical visits, and states that the images were de-identified by removing HIPAA-defined personal identifiers. It does not state a clear dataset licence or a blanket permission to redistribute derived labels. The Kaggle page supplies a compilation licence, but that is not sufficient to resolve the upstream rights question.

**Decision: RESTRICTED — source identity and de-identification statement verified; dataset licence/publication and derivative-label permission not verified.**

### Medetec

Primary source evidence: [Medetec image database terms](https://medetec.co.uk/files/medetec-images.html)

Medetec states that its images may be downloaded free of charge and used without restriction for reports, essays, dissertations, training and education, provided that the Medetec copyright notice is not removed. The page separately directs users to contact Medetec for high-resolution commercial or publishing use. The exact archive/version used in this workspace and the permission to redistribute resized images or derived bounding-box labels are not recorded.

**Decision: RESTRICTED — academic-use statement recorded; retain attribution and obtain written confirmation for publication/derived-label release before formal use.**

### WSNet

Primary project evidence: [authors' WSNET repository](https://github.com/subbareddy248/WSNET) and [Wseg dataset card](https://huggingface.co/datasets/subbareddyoota/wseg_dataset)

The authors' repository links the Wseg dataset and displays an MIT licence for the repository. The linked Hugging Face dataset card displays **CC BY-NC 4.0** for that dataset. The local files are hash-matched to `data_wound_seg/wsnet_*.png`, but the workspace does not contain an archive hash or a direct file-level match to the Hugging Face release. Consequently, the repository's MIT code licence must not be treated as an automatic licence for the images.

**Decision: RESTRICTED — candidate dataset identity is documented; exact image-release match and applicable image licence remain to be confirmed.**

## Evidence not present in the workspace

The following cannot be completed from filenames or hashes:

- Roboflow workspace/project URL, version, export date, upstream dataset and licence terms.
- Clinical video owner/institution, IRB or exemption number, consent/waiver basis, de-identification review, and publication restrictions.
- Clinical frame extraction manifest and augmentation script/version/hash linking each `frame_*_aug_*` file to its original frame.
- A saved Kaggle download metadata/version snapshot for the 2,760-sample compilation.

These are explicitly **not inferred** from naming conventions.

## Gate result

The current governance gate remains **BLOCKED / QUARANTINED**. No source is cleared for formal training, model comparison, cross-validation, cascade evaluation, blind testing, or publication reporting. The formal public candidate smoke test remains a pipeline diagnostic only.

## Required attachments to clear the gate

1. Kaggle metadata or download record showing the exact version/date used.
2. Written or official terms resolving FUSeg, Medetec and WSNet image/annotation reuse and publication of derived labels.
3. Completed Roboflow provenance record with project URL, version and licence.
4. Completed clinical governance record with IRB/consent/de-identification and augmentation lineage.

