# Table 4: Empirical Impact of Data Leakage on Model Performance Assessment

*Comparison of Naive Image-Level Cross-Validation versus MD5-Aware Group Cross-Validation on YOLOv8n-cls.*

| Validation Strategy | Sampling Unit | Top-1 Accuracy (%) | Macro-F1 (%) | Stab Recall (%) | Cross-Fold Leakage | Status & Verdict |
|---|---|---|---|---|---|---|
| Naive StratifiedKFold (Image-Level) | Individual Image | 98.34% ± 1.21% | 98.34% ± 1.21% | 100.0% ± 0.0% | 178 / 206 groups leaked | INVALIDATED (Optimistic Bias) |
| MD5 StratifiedGroupKFold (Group-Aware) | MD5 Group (383 groups) | 87.40% ± 2.78% | 87.36% ± 3.11% | 88.98% ± 9.30% | 0 groups leaked | VALIDATED (Leakage-Free Baseline) |
| Empirical Difference (Leakage Bias) | N/A | +10.94 percentage points | +10.98 percentage points | +11.02 percentage points | 178 leaked groups | Demonstrates Severe Data Leakage Inflation |

*Conclusion: Unaware image-level partitioning on datasets containing duplicate or augmented images inflates reported Accuracy by over 10.9 percentage points. Group-aware partitioning is strictly required for sound biomedical ML evaluation.*
