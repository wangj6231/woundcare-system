# Phase B1 — classification statistical reconstruction and result freeze

Artifact label: 2026-09-20, as requested. Final verification completed 2026-09-21 (Asia/Taipei). Scope: 25 saved **validation prediction JSON** exports from C-Arch-05-MS (five seeds × five held-out folds); 720 saved rows per seed, 3,600 rows total. No image pixels, model weights or locked/external cohorts were loaded for computation. The five-seed procedure repeatedly evaluates the same development collection; 3,600 rows are **not** independent clinical observations.

## Primary results — five seed-level aggregate evaluations

Each seed score is calculated after concatenating that seed's five held-out fold `y_true/y_pred` arrays. Values below are percentages; SD is the sample SD across five seeds (ddof=1), expressed in percentage points. It is not a confidence interval.

| Metric | Mean | Seed sample SD | Seed min–max |
|---|---:|---:|---:|
| Accuracy | **87.39%** | **0.79 pp** | 86.39–88.06% |
| Macro Precision | 88.10% | 0.71 pp | 87.10–88.73% |
| Macro Recall | 87.36% | 0.78 pp | 86.36–88.01% |
| Macro-F1 | **87.47%** | **0.78 pp** | 86.54–88.07% |
| Weighted-F1 | **87.43%** | **0.79 pp** | 86.52–88.04% |
| Stab_wound Recall | **91.92%** | **4.52 pp** | 89.90–100.00% |

| Saved-file seed | Pooled rows | Accuracy | Macro-F1 | Weighted-F1 | Stab correct/support | Stab Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 720 | 87.92% | 88.07% | 88.04% | 89/99 | 89.90% |
| 123 | 720 | 87.92% | 88.01% | 87.95% | 89/99 | 89.90% |
| 3407 | 720 | 86.67% | 86.70% | 86.61% | **99/99** | **100.00%** |
| 2026 | 720 | 88.06% | 88.03% | 88.02% | 89/99 | 89.90% |
| 999 | 720 | 86.39% | 86.54% | 86.52% | **89/99** | **89.90%** |

**Per-seed attribution correction:** the Phase B1 request listed seed 999 as 100% and seed 3407 as 89.90%. The file-named saved arrays show the reverse. No row identity or checkpoint provenance was fabricated; the table reports what those seed-labeled exports support. The five-seed Stab mean and SD are unchanged by that transposition. Source SHA256 values and unrounded metrics are in the [machine-readable B1 summary](../experiments/results/statistics/C-Arch-05-MS_phase_b1_final_summary.json).

## Secondary descriptive results

The mean of the 25 separate fold scores remains **87.40% Accuracy** (87.3984% unrounded) and **87.36% Macro-F1** (87.3556% unrounded). Their sample fold SDs are 2.84 and 3.18 pp; these are `DESCRIPTIVE_FOLD_VARIABILITY_ONLY`, not 25 independent experiments. The historical ±2.78 / ±3.11 used population-style fold SD; neither fold SD is a CI. Fold-mean Macro-F1 differs from pooled-seed Macro-F1 because F1 depends nonlinearly on pooled TP/FP/FN and class supports vary by fold. Accuracy differs slightly because fold sizes vary. Do not substitute either estimand for the other.

The original **88.98 ± 9.30% Stab_wound Recall** remains in historical files but is `HISTORICAL_METRIC_SEMANTICS_ERROR` / `SUPERSEDED_FOR_STAB_WOUND_RECALL`: the old resume path used logged Macro Recall for Stab in 17/25 fold summary values. [Correction registry CORR-003 and CORR-006](RESEARCH_CORRECTION_REGISTRY.md) preserves the lineage and specifies the new reporting priority.

## Required questions and decisions

1. **Five seed-pooled Accuracy scores:** 42=87.92%, 123=87.92%, 3407=86.67%, 2026=88.06%, 999=86.39% (display rounded; exact values in JSON).
2. **Five seed-pooled Macro-F1 scores:** 42=88.07%, 123=88.01%, 3407=86.70%, 2026=88.03%, 999=86.54%.
3. **Mean ± seed SD:** Accuracy 87.39 ± 0.79 pp; Macro-F1 87.47 ± 0.78 pp; Weighted-F1 87.43 ± 0.79 pp; Stab_wound Recall 91.92 ± 4.52 pp. All are descriptive summaries across five seed-level aggregate evaluations.
4. **Corrected Stab:** Yes, 91.92 ± 4.52 pp from saved arrays, with four 89/99 and one 99/99 as identified above. It is not a CI.
5. **25-fold descriptive figures:** Yes, Accuracy 87.40% and Macro-F1 87.36% remain supported as secondary fold means only.
6. **Why different Macro-F1?** Mean of fold-level F1 and F1 computed after within-seed concatenation are different nonlinear operations. Fold class supports and fold sizes also vary.
7. **New CI?** No—no percentile, BCa, grouped, cluster or patient CI was calculated.
8. **Why no grouped CI?** The old prediction rows lack verified ordered row→image/content-group identity. [Phase B2 blocker](PHASE_B2_BLOCKER.md) defines the missing evidence; guesses from current order or file names are prohibited.
9. **Locked test access?** None in B1. The existing one-shot internal holdout figure of **46/48 = 95.83%** is quoted only from prior documentation and remains separate from development. It is not external clinical validation.
10. **CO2Wounds access?** None in B1; it was not used for training, tuning, selection or this summary.
11. **Model inference?** None. Only saved `y_true/y_pred/y_prob/class_names` JSON values were read; probabilities were checked for shape, finite values, [0,1] range, row sum within the previously used tolerance and argmax agreement, but Top-5/ROC-AUC/calibration/ECE were not newly reported.
12. **Can development evidence freeze?** Yes, as the versioned **saved-array statistical interpretation** in [Classification Result Freeze](CLASSIFICATION_RESULT_FREEZE_20260920.md). It does not certify row/image identity, patient independence, external seven-class validation, original source licensing or clinical safety.

## Output, verification and safety

- Preferred downstream table: [Table 2c](../experiments/results/tables/Table2c_Final_Classification_Development_Summary.md); old Table 2 and Table 2b remain intact. [Seven-class Markdown/CSV](../experiments/results/tables/Table_B1_PerClass_SeedPooled_Performance.md) and [five seed matrices plus descriptive pooled counts](../experiments/results/tables/Table_B1_SeedPooled_Confusion_Counts.md) are versioned. The pooled matrix repeats predictions over the same development data, not 3,600 independent cases.
- Assertions passed: 25 unique file names; five complete seeds, five folds each; 720 rows/seed and 3,600 total; fixed seven-class order; matching true/pred/prob lengths; valid finite normalized 7-column probabilities and probability argmax equal to stored prediction. Any violation fails `FAIL_SAVED_PREDICTION_INTEGRITY` before output.
- Tests: B1 targeted **17 passed / 0 failed / 0 skipped**; related Phase A/A.5 **36 passed / 0 failed / 0 skipped**; full synthetic suite **135 passed / 0 failed / 0 skipped**. Targeted and full runs showed no warnings; related run showed two third-party `thop`/`distutils` deprecation warnings, not test failures.
- SHA256 before/after: **91/91 protected files unchanged**—25 saved predictions, old multiseed summary and bootstrap report, old Tables 1–4, both historical locked-test Table 5 formats, and 54 checkpoint `.pt` files. Hashing the old Table 5 files did not open locked-test images or rerun evaluation. No historical file was overwritten or deleted.
- New outputs use exclusive file creation; existing versioned outputs fail before overwrite. The root README was not rewritten; [proposed README patch](README_CLASSIFICATION_RESULT_PATCH.md) is separate.

```text
PHASE_B1_STATUS = COMPLETE
CLASSIFICATION_RESULT_FREEZE = COMPLETE
READY_FOR_FULL_PHASE_B2_GROUPED_BOOTSTRAP = NO
READY_FOR_PHASE_C0 = YES
model_training = false
model_inference = false
locked_test_used = false
CO2Wounds_used = false
bootstrap_CI_generated = false
historical_files_overwritten = false
```

`READY_FOR_PHASE_C0 = YES` means controlled **planning and preflight** may begin using this frozen B1 evidence. It does not authorize Phase C model inference, retuning on any locked/external cohort, or replacement of historical results without a new versioned protocol.
