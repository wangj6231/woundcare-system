# Phase B1 per-class performance — five seed-pooled evaluations

All percentages are means/sample SD across five seed-level aggregates (ddof=1), not CI.

| Class | Mean Precision across seeds (%) | SD Precision (pp) | Mean Recall across seeds (%) | SD Recall (pp) | Mean F1 across seeds (%) | SD F1 (pp) | Support per seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Abrasions | 84.36 | 3.34 | 88.76 | 1.56 | 86.48 | 2.14 | 42:105; 123:105; 3407:105; 2026:105; 999:105 |
| Bruises | 84.16 | 4.34 | 93.58 | 1.72 | 88.59 | 2.95 | 42:109; 123:109; 3407:109; 2026:109; 999:109 |
| Burns | 89.99 | 4.37 | 78.43 | 3.25 | 83.78 | 3.24 | 42:102; 123:102; 3407:102; 2026:102; 999:102 |
| Cut | 94.76 | 2.25 | 80.78 | 1.32 | 87.20 | 1.08 | 42:102; 123:102; 3407:102; 2026:102; 999:102 |
| Ingrown_nails | 93.59 | 3.26 | 92.00 | 2.83 | 92.75 | 2.10 | 42:100; 123:100; 3407:100; 2026:100; 999:100 |
| Laceration | 76.94 | 4.89 | 86.02 | 4.04 | 81.10 | 2.92 | 42:103; 123:103; 3407:103; 2026:103; 999:103 |
| Stab_wound | 92.92 | 3.90 | 91.92 | 4.52 | 92.38 | 3.59 | 42:99; 123:99; 3407:99; 2026:99; 999:99 |

Confidence intervals were not recomputed because historical prediction rows cannot be reliably linked to image/content groups required for dependence-aware resampling.
