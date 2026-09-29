# Table 2b — Corrected classification metrics from saved arrays

SOURCE = SAVED_FOLD_PREDICTION_ARRAYS; ROW_IDENTITY = NOT AVAILABLE; PATIENT_INDEPENDENCE = NOT VERIFIED.
GROUPED_BOOTSTRAP_CI = NOT AVAILABLE; NO_MODEL_INFERENCE_PERFORMED = TRUE; LOCKED_TEST_USED = FALSE.
This is seed-level aggregate metrics reconstructed from saved fold prediction arrays, not canonical image-identified OOF.

| Metric | Mean of 5 seed-pooled scores (%) | Sample SD, ddof=1 (pp) | Historical 25-fold descriptive mean (%) |
|---|---:|---:|---:|
| Accuracy | 87.39 | 0.80 | 87.40 |
| Macro-F1 | 87.47 | 0.78 | 87.36 |
| Weighted-F1 | 87.43 | 0.79 | 87.35 |
| Stab_wound Recall | 91.92 | 4.52 | 91.46 |

The original 88.98 ± 9.30% Stab_wound claim is HISTORICAL_METRIC_SEMANTICS_ERROR; 17/25 fold summary values used Macro Recall. It is SUPERSEDED_FOR_THIS_METRIC.
Historical Accuracy 87.40% and Macro-F1 87.36% are checked only as descriptive means across 25 saved folds; seed-pooled estimates are reported separately.
No new CI: prediction-row to image/MD5-group linkage is unavailable. Do not treat 25 folds as independent datasets.

| Seed | Rows | Accuracy (%) | Macro-F1 (%) | Stab_wound correct/support | Stab_wound Recall (%) |
|---:|---:|---:|---:|---:|---:|
| 42 | 720 | 87.92 | 88.07 | 89/99 | 89.90 |
| 123 | 720 | 87.92 | 88.01 | 89/99 | 89.90 |
| 999 | 720 | 86.39 | 86.54 | 89/99 | 89.90 |
| 2026 | 720 | 88.06 | 88.03 | 89/99 | 89.90 |
| 3407 | 720 | 86.67 | 86.70 | 99/99 | 100.00 |

Per-fold and all seven per-class precision/recall/F1, support and correct counts are in the companion versioned JSON.
