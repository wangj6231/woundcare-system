# Classification development-result freeze — 2026-09-20

**Scope of freeze:** C-Arch-05-MS / YOLOv8n-cls development evidence reconstructed from 25 saved validation prediction files, five seeds × five held-out folds. The source arrays pass seven-class-order, probability-shape/finite/normalization/argmax, five-complete-seed, 720-rows-per-seed and 3,600-total-row checks. Full precision and source SHA256 values are in [the B1 machine-readable summary](../experiments/results/statistics/C-Arch-05-MS_phase_b1_final_summary.json). This is an evidence freeze, not a model retraining or clinical clearance.

## Primary — five seed-level aggregate evaluations

| Metric | Mean across five seeds | Sample SD, ddof=1 | Seed minimum–maximum |
|---|---:|---:|---:|
| Accuracy | **87.39%** | 0.79 percentage points | 86.39–88.06% |
| Macro Precision | 88.10% | 0.71 pp | 87.10–88.73% |
| Macro Recall | 87.36% | 0.78 pp | 86.36–88.01% |
| Macro-F1 | **87.47%** | 0.78 pp | 86.54–88.07% |
| Weighted-F1 | **87.43%** | 0.79 pp | 86.52–88.04% |
| Stab_wound Recall | **91.92%** | **4.52 pp** | 89.90–100.00% |

The seed-labeled saved arrays give Stab_wound 89/99 correct for seeds 42, 123, 2026 and 999, and 99/99 for seed **3407**. The Phase B1 request's seed-specific list transposed 3407 and 999; the total 91.92% ± 4.52 pp is unchanged. We freeze the mapping supported by file names and arrays, not the transposed list. Because row→image/MD5 identity is absent, this does not prove patient independence or canonical image-identified OOF.

## Secondary — historical descriptive fold statistics

- Accuracy mean across 25 held-out fold scores: **87.40%** (unrounded 87.3984%).
- Macro-F1 mean across 25 held-out fold scores: **87.36%** (unrounded 87.3556%).
- These are `DESCRIPTIVE_FOLD_VARIABILITY_ONLY`, not 25 independent experiments. Seed-pooled Macro-F1 differs because F1 is nonlinear. The older 88.98 ± 9.30% Stab claim is `HISTORICAL_METRIC_SEMANTICS_ERROR` / `SUPERSEDED_FOR_STAB_WOUND_RECALL`; 17/25 summary fold values used Macro Recall.

## Separate historical internal holdout

Existing historical documentation records **46/48 = 95.83%** on a one-shot internal locked holdout. B1 did **not** reopen, enumerate or infer on those images; it did not combine 48 test cases with the 3,600 repeated saved development predictions. This is **not external clinical validation** and must not be used for post-hoc tuning. Existing Table 5 remains unchanged.

## Authoritative interpretation and limits

Use [Table 2c](../experiments/results/tables/Table2c_Final_Classification_Development_Summary.md) for subsequent development reporting; preserve historical Table 2 and Table 2b as lineage. [Per-class table](../experiments/results/tables/Table_B1_PerClass_SeedPooled_Performance.md) and [seed confusion counts](../experiments/results/tables/Table_B1_SeedPooled_Confusion_Counts.md) are B1 descriptive outputs. No CI was generated; [Phase B2 remains blocked](PHASE_B2_BLOCKER.md). The original classification source/version/license and patient/derived-family lineage remain unverified.

```text
DEVELOPMENT_EVALUATION_RECONSTRUCTED
INTERNAL_HOLDOUT_HISTORICALLY_EVALUATED
GROUPED_CI_UNAVAILABLE
PATIENT_INDEPENDENCE_NOT_VERIFIED
EXTERNAL_7CLASS_VALIDATION_NOT_AVAILABLE
NOT_CLINICALLY_VALIDATED
```

`CLASSIFICATION_RESULT_FREEZE = COMPLETE` means the **current saved-array development interpretation is frozen**. It does not authorize training, a new test, grouped CI, external validation claims, or clinical deployment.
