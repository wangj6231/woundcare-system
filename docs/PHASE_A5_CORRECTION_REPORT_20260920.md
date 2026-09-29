# Phase A.5 — historical metric correction and research guard hardening

Date: 2026-09-20. Scope: saved classification **validation** arrays and synthetic regression tests only. No training, classification/segmentation inference, development-image reevaluation, locked-test enumeration/access, CO2Wounds tuning, or new bootstrap CI occurred. This phase changes defensive source code under the user's explicit Phase A.5 authorization; historical results and model weights remain immutable.

## Decision summary

| Item | Result |
|---|---|
| Saved prediction inputs | 25 validation fold JSON files; exactly 720 rows per seed, 3,600 saved rows total |
| Historical 25-fold descriptive mean | Accuracy 87.398% → **87.40%**; Macro-F1 87.356% → **87.36%** |
| New mean of five seed-pooled scores | Accuracy **87.39%**; Macro-F1 **87.47%**; Weighted-F1 **87.43%** |
| Corrected Stab_wound Recall | Five seed-pooled scores: 89.90%, 89.90%, 89.90%, 89.90%, 100.00%; mean **91.92%**, sample SD **4.52 percentage points** (ddof=1) |
| Descriptive 25-fold Stab distribution | Mean 91.4596%, sample SD 13.6325 pp; **not** independent-run inference |
| Grouped bootstrap CI | **Unavailable**, no row→image/MD5-group linkage |
| Test/CO2/training in Phase A.5 | 0 locked-test images; no CO2Wounds tuning; no training or inference |

The corrected [saved-array JSON](../experiments/results/statistics/C-Arch-05-MS_saved_arrays_recomputed_v2.json) contains per-fold Accuracy, Macro Precision/Recall/F1, Weighted-F1, and all seven per-class precision/recall/F1/support/correct counts, plus each seed's five-fold pooled metrics. [Table 2b](../experiments/results/tables/Table2b_Classification_Performance_Recomputed_From_Saved_Arrays.md) is a concise version. Source SHA256 values are recorded in the JSON. These are arithmetic reconstructions of label/prediction pairs, **not canonical image-identified OOF** and not a new generalization benchmark.

## Required questions

1. **Correct Stab_wound Recall?** 91.92% mean of five seed-pooled recall values, sample SD 4.52 pp. Four seeds have 89/99 correct and one 99/99 in the saved arrays. The fold-level descriptive mean is 91.4596%; these estimands differ.
2. **Why did 88.98 ± 9.30 fail?** The CSV `recall` field means Macro Recall, but the old `multi_seed.py` resume path assigned it to `stab_wound_recall`. For 17/25 completed folds the historical summary contains that wrong value. The old files are preserved and registered as `HISTORICAL_METRIC_SEMANTICS_ERROR`, `NOT_VERIFIED_AS_STAB_WOUND_RECALL`, `SUPERSEDED_FOR_THIS_METRIC` in [CORR-003](RESEARCH_CORRECTION_REGISTRY.md).
3. **Are 87.40 Accuracy / 87.36 Macro-F1 supported?** Yes **only as the historical descriptive average of 25 saved-fold scores** to two decimals. Pooling five folds within each seed gives 87.39% Accuracy and 87.47% Macro-F1 on average; Macro-F1 is nonlinear. Neither is 25 independent datasets, a patient-level estimate, or a new CI.
4. **Any classification inference?** No. The v2 tool reads stored `y_true/y_pred/y_prob/class_names` only; no model or image loader is invoked.
5. **Any locked test access?** No. Phase A.5 did not enumerate, open or infer on any of the 48 locked-test images. The count is a historical split claim, not a new test audit.
6. **Any new CI?** No. Old row-wise bootstrap generators now fail closed; no replacement CI was created.
7. **Which old statistics are superseded?** Old Stab 88.98 ± 9.30 and its per-class CI/derived comparisons as verified Stab estimates; old N=3,622 row-wise CIs in Table 3 as valid group-aware inference. Accuracy/Macro-F1 descriptive fold means remain arithmetically supported but their old CI interpretation is not. Original Tables 1–4 and JSON remain intact.
8. **What cannot be strictly recovered from current artifacts?** Ordered prediction-row→image/MD5/patient/augmentation/checkpoint mapping, per-image five-seed coverage or missing/duplicate row audit, patient independence, group-aware resampling units, original 3,622 denominator provenance, and valid grouped/patient-level CIs. Current array arithmetic cannot manufacture those identities.
9. **Which RGB/BGR results need reevaluation?** The historical ISIC→FUSeg fixed gate and RGB-NumPy classifier/cascade paths listed in [the affected-results inventory](RGB_BGR_AFFECTED_HISTORICAL_RESULTS.md). The corrected gate is code-only; no 191-image rerun occurred.
10. **Which source-role bypasses were fixed?** `experiments/data_roles.py` rejects known locked-test and CO2Wounds IDs/paths, a registered CO2 manifest SHA256, and unknown/conflicting roles. The new formal development guard and its preflight use it; direct classification `evaluate`, `predict_yolo`, `predict_torch`, `train_yolo_cls`, `train_torch_cls` and multi-seed runner now check roles before model execution. Known ID alone is **not** license approval: the formal runner retains separate evidence-file SHA and exact prepared-dataset audits. Arbitrary copied pixels without an attached trusted manifest and other legacy scripts are **not** proven fully protected; do not call the whole repository bypass-proof.
11. **External evidence still missing?** The original classification publisher/URL/license/version/archive digest and patient/augmentation lineage; original historical row→image manifest/checkpoint mapping; provenance of 3,622; independent new-source testing/permission evidence where needed. Noncommercial intent or file naming is not evidence of permission. No replacement source was inferred.
12. **Ready for Phase C controlled development reevaluation?** **Not yet as an executable gate.** The corrected ISIC gate changes its pinned script hash, while old `development_gate_inputs.json` pins the prior hash and the old `development_gate` output directory already exists. Phase C must create a *new versioned*, non-overwriting protocol/output with frozen FUSeg validation cohort, fixed inference thresholds/crops, verified source permission and checkpoint hashes. It must not reuse CO2Wounds or the locked classification test for selection. No Phase C evaluation was run here.

## Engineering changes and verification

- Resume accepts an explicit `stab_wound_recall` or computes from same-fold saved arrays with seven-class order, otherwise raises `RESUME_STAB_RECALL_UNVERIFIED`; no Macro Recall/zero proxy. Mixed fresh/resumed fold semantics covered by synthetic tests. Failed or incomplete new evaluation no longer logs a successful fold.
- Legacy table/bootstrap imports no longer create directories or execute reports; direct row-wise Bootstrap entrypoints are disabled as superseded. Existing research output filenames fail before overwrite. New v2 JSON/Markdown are opened exclusively, never replacing historical tables.
- Explicit image-input contract: PIL RGB is converted to contiguous BGR NumPy; already-BGR NumPy is preserved; direct RGB NumPy is rejected as a declared source type. Historical benchmark routes that already convert RGB→BGR are not mislabeled affected.
- Full synthetic suite: **118 tests passed, 0 failed, 0 skipped**; Phase A.5 targeted **18 tests passed**. Import-side-effect and source-role tests used temporary/synthetic data. Historical 25 prediction files, old summary/bootstrap and original Table 1–4 files: **35/35 SHA256 unchanged**. Existing checkpoints: **54/54 SHA256 match Phase A snapshot**. Deliberately corrected scripts are not counted as unchanged sealed source.
- `.gitignore` now exposes only the three new versioned correction artifacts under `experiments/results/`; raw predictions, weights and legacy results remain ignored. No commit or push was performed in this phase.

## Formal status

```text
PHASE_A5_STATUS = COMPLETE
READY_FOR_PHASE_C_CONTROLLED_REEVALUATION = NO
READY_FOR_FULL_PHASE_B_GROUPED_BOOTSTRAP = NO
GROUPED_CI_STATUS = BLOCKED_BY_MISSING_HISTORICAL_ROW_IDENTITY
test_images_used = 0
CO2Wounds_used_for_tuning = false
new_model_training = false
```
