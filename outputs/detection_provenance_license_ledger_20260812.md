# Detection provenance and license ledger (2026-08-12)

Status: **D0 provenance gate pending**

The local filename audit identified five provenance groups within three broad source families (internal clinical, Kaggle imports, and Roboflow export). A file-level SHA-256 join against `data_wound_seg/` and `correspondence_table.xlsx` now confirms all 1,186 FUSC, 374 Medetec and 1,176 WSNet rows at the local-file level, including origin-ID mappings. This resolves local provenance identity, but not the original download version or license.

## Required D0 evidence

1. Original dataset name, version, download date and upstream URL.
2. License or terms that explicitly permit research training and publication of derived results.
3. Source image identifiers and a mapping from upstream ID to local filename.
4. Patient/case/video grouping metadata where applicable.
5. Cross-source MD5 and perceptual-overlap report.
6. For clinical frames: consent/de-identification record and augmentation lineage.

## Current decision

The conservative `yolo_dataset_wound_clean_v1` is suitable for smoke testing only. It is not a final paper dataset until external terms, clinical governance evidence, Roboflow upstream identity and the 55 annotation-conflict groups are resolved.

The exact local provenance mapping is recorded in `outputs/local_dataset_provenance_mapping_20260812.csv`; it covers all three Kaggle-derived groups with 100% exact SHA-256 matches.

For the three locally matched Kaggle groups, `outputs/mask_canonical_label_audit_20260812.csv` applies the existing `prepare_yolo_dataset.py` mask-to-bbox rule. The resulting `yolo_dataset_wound_clean_v2` uses mask-derived labels for 2,562 rows and excludes 11 rows with conflicting mask sources. This resolves the label-generation path for the retained Kaggle candidates, but it does not grant upstream license rights.

Reference links are recorded in the CSV ledger; no public data was downloaded or redistributed during this audit.

## Public evidence review update (2026-08-12)

The public-source pages and terms were reviewed and captured in:

- `outputs/detection_provenance_license_evidence_20260812.md`
- `outputs/detection_provenance_license_evidence_20260812.csv`

This review confirms source descriptions and some usage statements, but it does **not** clear the dataset for formal use. The Kaggle compilation version/download record, exact upstream image licences for FUSeg and WSNet, publication/derived-label permission for Medetec, Roboflow project terms, and all clinical governance records are still missing. The governance decision therefore remains `BLOCKED / QUARANTINED`.
