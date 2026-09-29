# Table 2c — Final classification development summary

Primary: five seed-level aggregate evaluations. Historical: 25-fold descriptive variability only.

| Metric | Primary five-seed pooled result | Historical 25-fold descriptive result | Interpretation |
| --- | --- | --- | --- |
| Accuracy | 87.39% ± 0.79 pp; range 86.39–88.06% | 87.40% ± 2.84 pp (descriptive sample SD) | Different estimand; folds are not independent datasets |
| Macro Precision | 88.10% ± 0.71 pp; range 87.10–88.73% | 88.80% ± 2.62 pp (descriptive sample SD) | Different estimand; folds are not independent datasets |
| Macro Recall | 87.36% ± 0.78 pp; range 86.36–88.01% | 87.56% ± 3.16 pp (descriptive sample SD) | Different estimand; folds are not independent datasets |
| Macro-F1 | 87.47% ± 0.78 pp; range 86.54–88.07% | 87.36% ± 3.18 pp (descriptive sample SD) | Different estimand; folds are not independent datasets |
| Weighted-F1 | 87.43% ± 0.79 pp; range 86.52–88.04% | 87.35% ± 2.91 pp (descriptive sample SD) | Different estimand; folds are not independent datasets |
| Stab_wound Recall | 91.92% ± 4.52 pp; range 89.90–100.00% | 88.98% ± 9.30 pp — HISTORICAL_METRIC_SEMANTICS_ERROR | Historical Stab claim superseded; 17/25 summary fold values used Macro Recall |

Confidence intervals were not recomputed because historical prediction rows cannot be reliably linked to image/content groups required for dependence-aware resampling.

No test or external cohort was combined with these development metrics.
