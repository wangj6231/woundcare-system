# Suggested future README classification-result text

This is proposed wording only; the root README has **not** been rewritten.

> Leakage-aware development evaluation across five random seeds produced approximately 87.4% Accuracy and 87.5% Macro-F1 when the five held-out fold predictions were pooled within each seed. Corrected Stab_wound Recall was 91.92% ± 4.52 percentage points across the five seed-level evaluations (sample SD, not a confidence interval).
>
> Historical 25-fold descriptive averages were 87.40% Accuracy and 87.36% Macro-F1. They represent a different estimand and should not replace the seed-pooled values or be treated as 25 independent datasets.
>
> Dependence-aware confidence intervals could not be reconstructed because historical prediction rows lack preserved image/content-group identity. The historical one-shot internal locked holdout reported 46/48 correct (95.83%); it was not combined with cross-validation and is not external clinical validation.

Preferred development table: [Table 2c](../experiments/results/tables/Table2c_Final_Classification_Development_Summary.md). Dataset source provenance and patient independence remain unverified; this is not clinical validation or deployment authorization.
