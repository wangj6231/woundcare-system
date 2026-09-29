# Classification statistical definitions — Phase B1

All metrics in the Phase B1 machine-readable summary are percentages unless marked as counts. Seven classes are fixed in the order Abrasions, Bruises, Burns, Cut, Ingrown_nails, Laceration, Stab_wound. Per-class precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R). Absent denominators yield 0 for arithmetic comparability; actual seed-pooled class supports are included. Macro scores average the seven class scores equally; Weighted-F1 weights class F1 by true-class support; Accuracy = correct/total.

| Term | Unit / calculation | Appropriate interpretation |
|---|---|---|
| Fold score | Metric on one held-out fold's saved `y_true/y_pred` | One partition evaluation; folds share training/source data and are not independent datasets. |
| Seed-pooled score | Concatenate that seed's five held-out fold arrays (720 rows), then compute the metric once | Primary B1 evaluation unit. Still no verified row→image or patient mapping. |
| Mean across seeds | Arithmetic mean of five seed-pooled scores | Primary development point summary; five seed procedures over the same development source, not five external cohorts. |
| Seed sample SD | `sqrt(sum((score-mean)^2)/(5-1))`, ddof=1 | Describes between-seed variation in the five observed procedures; **not a CI**. |
| Descriptive fold mean/SD | Mean and sample SD (ddof=1) of 25 separate fold scores | Secondary `DESCRIPTIVE_FOLD_VARIABILITY_ONLY`, not independent-run inference. The older published `±2.78`/`±3.11` were population-style fold SD (ddof=0); B1 uses explicitly labeled sample SD. |
| Bootstrap CI | Interval from predeclared resampling over independent or justified cluster units | **Unavailable** for these saved historical rows because ordered row→image/content-group linkage is absent. |
| Patient-level CI | Patient-cluster inference, requiring verified patient identities and an appropriate sampling design | **Unavailable**; MD5 content groups are not patients. |

`mean(Macro-F1 per fold) ≠ Macro-F1(concatenated folds within each seed)` in general: F1 is nonlinear in TP/FP/FN, and fold class supports differ. This is why the historical descriptive 25-fold Macro-F1 87.36% and the B1 primary five-seed pooled Macro-F1 87.47% can both be arithmetically correct without measuring the same estimand. Accuracy differs slightly as fold sizes vary.

The five-seed pooled confusion-count sum is repeated predictions over the *same* development set under five random-seed evaluation procedures; it is not 3,600 independent clinical observations. Top-5, ROC-AUC, calibration and ECE were not recomputed in B1 despite saved probabilities, avoiding an unrequested probability-based result with separate calibration and class-order assumptions.

Confidence intervals were not recomputed because historical prediction rows cannot be reliably linked to image/content groups required for dependence-aware resampling.
