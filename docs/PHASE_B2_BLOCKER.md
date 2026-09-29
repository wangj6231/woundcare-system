# Phase B2 grouped-bootstrap blocker — 2026-09-20

`READY_FOR_FULL_PHASE_B2_GROUPED_BOOTSTRAP = NO`.

The 25 historical fold prediction exports retain `y_true`, `y_pred`, `y_prob` and `class_names`, but no verified ordered prediction-row→image/content-group identity. Exactly 3,600 saved prediction rows exist (720 per seed); this count is **not** 3,600 independent clinical observations. Each seed reevaluated the same development collection under a different partition/training procedure.

Before historical dependence-aware grouped resampling can be considered, every exported row must have a verifiable `prediction_row_id`, `image_id`, and `content_group_id`/MD5 group linked to the original fold and checkpoint. `patient_id`, `derived_family_id`, and `source_id` are strongly preferred for patient/family/source dependence and overlap auditing. The mapping must be recovered from contemporaneous provenance, not inferred from current file order, class labels, fold sizes, filenames or post-hoc matches. Any recreated mapping must prove row uniqueness, complete 720-row coverage within each seed, consistent class labels and no cross-split content/family leakage.

If those original ordered identities no longer exist anywhere, **the historical grouped bootstrap is permanently unrecoverable from the surviving artifacts**. A future new experiment with an identity-preserving export could support a new grouped analysis, but it cannot repair the historical one by relabeling old rows. Phase B1 therefore reports seed-level descriptive mean and sample SD only—no percentile, BCa, cluster or patient CI.

For every *new* classification experiment, the prediction export contract must preserve:

```text
sample_id
relative_path
sha256
md5_group
source_id
patient_id (if available)
derived_family_id
seed
fold
checkpoint_sha256
y_true
y_pred
y_prob
```

The new runner must verify stable per-row identity, original-image/hash provenance, group-disjoint train/validation partitions, and checkpoint lineage **before** statistical inference. Unknown patient IDs remain unknown rather than guessed. No locked test or CO2Wounds cohort may be used to fill missing development identities or tune a model.
