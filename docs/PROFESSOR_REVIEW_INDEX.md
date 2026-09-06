# Professor Review Index

Updated: 2026-08-12

This index is the public, reproducible entry point for the wound detection and
classification experiments. Raw clinical images and third-party dataset copies
are intentionally not committed. Use the source links and manifests below to
retrieve data from the original providers.

For the complete phase-by-phase protocol, image counts, metrics, and governance
status, see the [detailed experimental report](./PROFESSOR_DETAILED_EXPERIMENT_REPORT.md).

## 1. Dataset sources and access

| Source | Official access | Version / revision recorded locally | Current release decision |
|---|---|---|---|
| WSNet / Wseg | [Hugging Face dataset](https://huggingface.co/datasets/subbareddyoota/wseg_dataset) | `bdcc1e1cef607bfd7a4f98fc28f31a638a0e0ca9` | Research-only candidate; CC BY-NC 4.0; source-image and derived-label release policy remains under review |
| WSNet code | [Authors' repository](https://github.com/subbareddy248/WSNET) | `cd407a9b877125352f95954bab3b015a8c58477c` | MIT code licence recorded; this does not automatically license the images |
| FUSeg / FUSC | [UWM wound-segmentation project](https://github.com/uwm-bigdata/wound-segmentation/blob/master/data/Foot%20Ulcer%20Segmentation%20Challenge/README.MD) | Source page recorded; archive/version not fixed | De-identification statement recorded; upstream image licence and redistribution terms remain unverified |
| Medetec | [Medetec image database terms](https://medetec.co.uk/files/medetec-images.html) | Terms page accessed 2026-08-12 | Academic-use statement recorded; attribution and publication/derived-label permission require confirmation |
| Kaggle compilation | [Wound images segmentation](https://www.kaggle.com/datasets/leoscode/wound-segmentation-images) | Exact download version/date not captured | Research-only evidence; upstream licences are not assumed from the compilation page |

Local evidence and checksums:

- [Detection provenance and licence ledger](../outputs/detection_provenance_license_ledger_20260812.md)
- [Licence evidence review](../outputs/detection_provenance_license_evidence_20260812.md)
- [Official source metadata](../outputs/detection_official_source_metadata_20260812.json)
- [WSNet intake report and archive SHA-256](../outputs/wsnet_official_intake_report_20260812.md)
- [WSNet file manifest](../outputs/wsnet_official_file_manifest_20260812.csv)
- [Local provenance mapping](../outputs/local_dataset_provenance_mapping_20260812.csv)

**Governance note:** the local evidence pack currently records the formal-use
gate as `BLOCKED / QUARANTINED`. The public links above are provided for
reproducibility and review; they are not a claim that every source is cleared
for redistribution or publication.

## 2. Detection model artifacts

The following representative weights are from the official WSNet-derived
development experiments. They are stored with Git LFS and are not final locked
test results.

| Model artifact | Protocol status | SHA-256 |
|---|---|---|
| `experiments/results/detection/YOLO11m_WSNet_official_baseline_20260812/weights/best.pt` | 5-epoch development baseline; test split unused | `2B06B62189FE0724FBBCA9D0C7A61688F98D69CBB8D76332A602094234BAA4BD` |
| `experiments/results/detection/YOLO11s_WSNet_official_common_batch4_20260812/weights/best.pt` | 5-epoch comparison baseline; test split unused | `389F63D197159FE6361DCC60D823D0A480C6DFC6994C51700006F29C78F8B8F3` |
| `experiments/results/detection_groupkfold/YOLO11n_WSNet_groupkfold_fold01_seed42_202608123/weights/best.pt` | Fold 1 / seed 42 development benchmark; locked test unused | `3DE18C770B6E397C8925DF1AE71910FF1F2AE083D4262C9E8A278CE6EBEAF326` |

The clinical-data classification checkpoint is intentionally not published in
this public repository until its clinical governance and model-release review
is complete. Its validated results remain available in the tables below.

## 3. Result summaries

### Classification (sealed leakage-free development protocol)

- [Table 1 — Dataset and protocol](../experiments/results/tables/Table1_Dataset_and_Protocol_Summary.md)
- [Table 2 — Classification performance](../experiments/results/tables/Table2_Classification_Performance.md)
- [Table 3 — Bootstrap confidence intervals](../experiments/results/tables/Table3_Statistical_Bootstrap_CI.md)
- [Table 4 — Leakage impact analysis](../experiments/results/tables/Table4_Data_Leakage_Impact_Analysis.md)
- [Table 5 — Blind-test generalization](../experiments/results/tables/Table5_Final_Blind_Test_Generalization.md)

### Detection (development diagnostics)

- [YOLO11n baseline report](../outputs/wsnet_yolo11n_baseline_report_20260812.md)
- [YOLO11n fold 1 / seed 42 report](../outputs/wsnet_yolo11n_formal_fold01_seed42_report_20260812.md)
- [YOLO11n fold 1 / seed 123 report](../outputs/wsnet_yolo11n_formal_fold01_seed123_report_20260812.md)
- [YOLO11m public-candidate smoke report](../outputs/detection_formal_public_smoke_report_20260812.md)

The detection reports clearly distinguish smoke/development metrics from a
formal multi-fold, multi-seed result. No locked detection test claim is made in
this index.

## 4. Reproducibility code

- Experiment runner: [`experiments/run_experiments.py`](../experiments/run_experiments.py)
- Detection configuration: [`experiments/configs/D-Arch-01_yolo11m.yaml`](../experiments/configs/D-Arch-01_yolo11m.yaml)
- Detection/classification scripts: [`experiments/scripts/`](../experiments/scripts/)
- Dataset preparation scripts: [`prepare_yolo_dataset.py`](../prepare_yolo_dataset.py), [`prepare_cls_dataset.py`](../prepare_cls_dataset.py)

All paths above are relative to the repository root. Results and checkpoints
must be interpreted together with their provenance and governance status.
