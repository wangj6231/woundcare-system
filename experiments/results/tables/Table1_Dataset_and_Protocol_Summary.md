# Table 1: Dataset Partition & Experimental Protocol Summary

| Subset / Category | Count / Value | Description |
|---|---|---|
| Total Raw Images | 768 images | Complete collected wound dataset |
| Development Set | 720 images | Used for 5-Fold StratifiedGroupKFold cross-validation |
| Unique Content Groups | 383 MD5 groups | Identified exact binary duplicate groups |
| Duplicate Images | 337 images (206 groups) | Images sharing exact content with at least one other image |
| Blind Test Set (Sequestered) | 48 images | Strictly locked for final untouched generalization benchmark |
| Cross-Validation Protocol | 5-Fold GroupKFold | Group-aware split by MD5 hash; 0 cross-fold content leakage |
| Multi-Seed Evaluation | 5 Seeds x 5 Folds (n=25) | Seeds: 42, 123, 3407, 2026, 999; 25 leakage-free evaluation runs |
