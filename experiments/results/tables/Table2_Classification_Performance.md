# Table 2: Classification Architecture Performance (Leakage-Free Protocol)

| Experiment ID | Model | Validation Method | Runs (n) | Top-1 Accuracy (%) | Macro-F1 (%) | Stab_wound Recall (%) | Between-Seed SD (%) | Leakage Status |
|---|---|---|---|---|---|---|---|---|
| C-Arch-05-MS | YOLOv8n-cls | 5-Seed x 5-Fold StratifiedGroupKFold | 25 | 87.40 ± 2.78 | 87.36 ± 3.11 | 88.98 ± 9.30 | 0.76% | PASS (0 MD5 overlap) |

*Footnote: Evaluated across 25 leakage-free evaluation runs using 5 random seeds (42, 123, 3407, 2026, 999) and 5-fold MD5-aware StratifiedGroupKFold cross-validation. Naive StratifiedKFold results (98.34%) were invalidated due to cross-fold content leakage and are excluded from model comparison.*
