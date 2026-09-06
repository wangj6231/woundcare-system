# Table 5: Final Model Generalization on Sequestered Blind Test Set (n=48)

| Wound Class | Test Support (n) | Correct | Recall (%) | Precision (%) | F1-Score (%) | Misclassifications |
|---|---|---|---|---|---|---|
| Abrasions | 9 | 9 | 100.00% | 100.00% | 100.00% | None (9/9 correct) |
| Bruises | 13 | 12 | 92.31% | 100.00% | 96.00% | 1 image predicted as Stab_wound |
| Burns | 7 | 7 | 100.00% | 100.00% | 100.00% | None (7/7 correct) |
| Cut | 5 | 5 | 100.00% | 100.00% | 100.00% | None (5/5 correct) |
| Ingrown_nails | 4 | 4 | 100.00% | 100.00% | 100.00% | None (4/4 correct) |
| Laceration | 7 | 6 | 85.71% | 100.00% | 92.31% | 1 image predicted as Stab_wound |
| Stab_wound | 3 | 3 | 100.00% | 60.00% | 75.00% | 3/3 detected (+2 FP from Bruises/Laceration) |
| **Overall / Macro** | **48** | **46** | **96.86%** | **94.29%** | **94.76%** | **Total Errors: 2 / 48 (4.17%)** |

---

### Overall Summary Metrics
- **Top-1 Accuracy**: **95.83%** (46/48)
- **Top-5 Accuracy**: **100.00%** (48/48)
- **Macro-F1 Score**: **94.76%**
- **Weighted-F1 Score**: **96.23%**
- **Macro ROC-AUC**: **0.9982**

*Protocol Note: Evaluated strictly once on the sequestered 48-image blind test set (0 MD5 overlap with development set). No post-evaluation hyperparameter tuning was conducted.*

### Class-imbalance interpretation

The locked test set is a sequestered holdout with unequal per-class support (n=3–13), not a balanced test sample. Consequently, one error changes class recall by 7.69 percentage points for Bruises (1/13), 14.29 points for a seven-image class, 20 points for Cut (1/5), 25 points for Ingrown_nails (1/4), and 33.33 points for Stab_wound (1/3). The two observed top-1 errors were **Bruises → Stab_wound (1)** and **Laceration → Stab_wound (1)**. Stab_wound recall remained 3/3 = 100%, but precision fell to 60% because it received two false positives. These support-dependent metrics are therefore reported with the confusion direction and should be interpreted as an internal one-shot benchmark, not as external clinical generalization.
