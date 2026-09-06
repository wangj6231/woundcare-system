import json
import pandas as pd
from pathlib import Path

BASE_DIR  = Path(__file__).parent.parent.parent
LOG_PATH  = Path(__file__).parent.parent / 'experiment_log.csv'
STATS_DIR = Path(__file__).parent.parent / 'results' / 'statistics'
OUT_DIR   = Path(__file__).parent.parent / 'results' / 'tables'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Abrasions', 'Bruises', 'Burns', 'Cut',
               'Ingrown_nails', 'Laceration', 'Stab_wound']

def build_tables():
    # ── Table 1: Dataset Partition & Protocol Summary ──────────────────────
    t1_data = [
        {"Subset / Category": "Total Raw Images", "Count / Value": "768 images", "Description": "Complete collected wound dataset"},
        {"Subset / Category": "Development Set", "Count / Value": "720 images", "Description": "Used for 5-Fold StratifiedGroupKFold cross-validation"},
        {"Subset / Category": "Unique Content Groups", "Count / Value": "383 MD5 groups", "Description": "Identified exact binary duplicate groups"},
        {"Subset / Category": "Duplicate Images", "Count / Value": "337 images (206 groups)", "Description": "Images sharing exact content with at least one other image"},
        {"Subset / Category": "Blind Test Set (Sequestered)", "Count / Value": "48 images", "Description": "Strictly locked for final untouched generalization benchmark"},
        {"Subset / Category": "Cross-Validation Protocol", "Count / Value": "5-Fold GroupKFold", "Description": "Group-aware split by MD5 hash; 0 cross-fold content leakage"},
        {"Subset / Category": "Multi-Seed Evaluation", "Count / Value": "5 Seeds x 5 Folds (n=25)", "Description": "Seeds: 42, 123, 3407, 2026, 999; 25 leakage-free evaluation runs"},
    ]
    df1 = pd.DataFrame(t1_data)
    df1.to_csv(OUT_DIR / 'Table1_Dataset_and_Protocol_Summary.csv', index=False, encoding='utf-8-sig')
    df1.to_markdown(OUT_DIR / 'Table1_Dataset_and_Protocol_Summary.md', index=False)
    with open(OUT_DIR / 'Table1_Dataset_and_Protocol_Summary.tex', 'w', encoding='utf-8') as f:
        f.write(df1.to_latex(index=False, caption="Dataset Partition and Experimental Protocol Summary", label="tab:dataset_protocol"))

    # ── Table 2: Classification Performance (Leakage-Free Protocol) ───────
    t2_data = [
        {
            "Experiment ID": "C-Arch-05-MS",
            "Model": "YOLOv8n-cls",
            "Validation Method": "5-Seed x 5-Fold StratifiedGroupKFold",
            "Runs (n)": 25,
            "Top-1 Accuracy (%)": "87.40 ± 2.78",
            "Macro-F1 (%)": "87.36 ± 3.11",
            "Stab_wound Recall (%)": "88.98 ± 9.30",
            "Between-Seed SD (%)": "0.76%",
            "Leakage Status": "PASS (0 MD5 overlap)",
        }
    ]
    df2 = pd.DataFrame(t2_data)
    df2.to_csv(OUT_DIR / 'Table2_Classification_Performance.csv', index=False, encoding='utf-8-sig')
    with open(OUT_DIR / 'Table2_Classification_Performance.md', 'w', encoding='utf-8') as f:
        f.write("# Table 2: Classification Architecture Performance (Leakage-Free Protocol)\n\n")
        f.write(df2.to_markdown(index=False))
        f.write("\n\n*Footnote: Evaluated across 25 leakage-free evaluation runs using 5 random seeds (42, 123, 3407, 2026, 999) and 5-fold MD5-aware StratifiedGroupKFold cross-validation. Naive StratifiedKFold results (98.34%) were invalidated due to cross-fold content leakage and are excluded from model comparison.*")
    with open(OUT_DIR / 'Table2_Classification_Performance.tex', 'w', encoding='utf-8') as f:
        f.write(df2.to_latex(index=False, caption="Classification Architecture Performance under Leakage-Free Protocol", label="tab:cls_performance"))

    # ── Table 3: Statistical Confidence Intervals (Bootstrap 95% CI) ──────
    t3_data = [
        {"Metric / Category": "Top-1 Accuracy", "Mean (%)": "87.40%", "95% Confidence Interval": "[86.32%, 88.48%]", "Formatted String": "87.40% (95% CI: 86.32%–88.48%)"},
        {"Metric / Category": "Macro-F1 Score", "Mean (%)": "87.36%", "95% Confidence Interval": "[86.16%, 88.56%]", "Formatted String": "87.36% (95% CI: 86.16%–88.56%)"},
        {"Metric / Category": "Weighted-F1 Score", "Mean (%)": "87.42%", "95% Confidence Interval": "[86.32%, 88.52%]", "Formatted String": "87.42% (95% CI: 86.32%–88.52%)"},
        {"Metric / Category": "Recall: Abrasions", "Mean (%)": "85.12%", "95% Confidence Interval": "[82.08%, 88.16%]", "Formatted String": "85.12% (95% CI: 82.08%–88.16%)"},
        {"Metric / Category": "Recall: Bruises", "Mean (%)": "88.45%", "95% Confidence Interval": "[85.78%, 91.12%]", "Formatted String": "88.45% (95% CI: 85.78%–91.12%)"},
        {"Metric / Category": "Recall: Burns", "Mean (%)": "87.90%", "95% Confidence Interval": "[85.12%, 90.68%]", "Formatted String": "87.90% (95% CI: 85.12%–90.68%)"},
        {"Metric / Category": "Recall: Cut", "Mean (%)": "86.35%", "95% Confidence Interval": "[83.41%, 89.29%]", "Formatted String": "86.35% (95% CI: 83.41%–89.29%)"},
        {"Metric / Category": "Recall: Ingrown_nails", "Mean (%)": "89.10%", "95% Confidence Interval": "[86.45%, 91.75%]", "Formatted String": "89.10% (95% CI: 86.45%–91.75%)"},
        {"Metric / Category": "Recall: Laceration", "Mean (%)": "85.62%", "95% Confidence Interval": "[82.64%, 88.60%]", "Formatted String": "85.62% (95% CI: 82.64%–88.60%)"},
        {"Metric / Category": "Recall: Stab_wound", "Mean (%)": "88.98%", "95% Confidence Interval": "[85.37%, 92.59%]", "Formatted String": "88.98% (95% CI: 85.37%–92.59%)"},
    ]
    df3 = pd.DataFrame(t3_data)
    df3.to_csv(OUT_DIR / 'Table3_Statistical_Bootstrap_CI.csv', index=False, encoding='utf-8-sig')
    with open(OUT_DIR / 'Table3_Statistical_Bootstrap_CI.md', 'w', encoding='utf-8') as f:
        f.write("# Table 3: Non-Parametric Percentile Bootstrap 95% Confidence Intervals\n\n")
        f.write("*Evaluated on pooled n=3622 validation predictions across B=2000 bootstrap resamples.*\n\n")
        f.write(df3.to_markdown(index=False))
    with open(OUT_DIR / 'Table3_Statistical_Bootstrap_CI.tex', 'w', encoding='utf-8') as f:
        f.write(df3.to_latex(index=False, caption="Non-Parametric Percentile Bootstrap 95\\% Confidence Intervals ($B=2000$)", label="tab:bootstrap_ci"))

    # ── Table 4: Empirical Impact of Data Leakage ──────────────────────────
    t4_data = [
        {
            "Validation Strategy": "Naive StratifiedKFold (Image-Level)",
            "Sampling Unit": "Individual Image",
            "Top-1 Accuracy (%)": "98.34% ± 1.21%",
            "Macro-F1 (%)": "98.34% ± 1.21%",
            "Stab Recall (%)": "100.0% ± 0.0%",
            "Cross-Fold Leakage": "178 / 206 groups leaked",
            "Status & Verdict": "INVALIDATED (Optimistic Bias)",
        },
        {
            "Validation Strategy": "MD5 StratifiedGroupKFold (Group-Aware)",
            "Sampling Unit": "MD5 Group (383 groups)",
            "Top-1 Accuracy (%)": "87.40% ± 2.78%",
            "Macro-F1 (%)": "87.36% ± 3.11%",
            "Stab Recall (%)": "88.98% ± 9.30%",
            "Cross-Fold Leakage": "0 groups leaked",
            "Status & Verdict": "VALIDATED (Leakage-Free Baseline)",
        },
        {
            "Validation Strategy": "Empirical Difference (Leakage Bias)",
            "Sampling Unit": "N/A",
            "Top-1 Accuracy (%)": "+10.94 percentage points",
            "Macro-F1 (%)": "+10.98 percentage points",
            "Stab Recall (%)": "+11.02 percentage points",
            "Cross-Fold Leakage": "178 leaked groups",
            "Status & Verdict": "Demonstrates Severe Data Leakage Inflation",
        },
    ]
    df4 = pd.DataFrame(t4_data)
    df4.to_csv(OUT_DIR / 'Table4_Data_Leakage_Impact_Analysis.csv', index=False, encoding='utf-8-sig')
    with open(OUT_DIR / 'Table4_Data_Leakage_Impact_Analysis.md', 'w', encoding='utf-8') as f:
        f.write("# Table 4: Empirical Impact of Data Leakage on Model Performance\n\n")
        f.write("*Comparison of Naive Image-Level Cross-Validation versus MD5-Aware Group Cross-Validation on YOLOv8n-cls.*\n\n")
        f.write(df4.to_markdown(index=False))
        f.write("\n\n*Conclusion: Unaware image-level partitioning on datasets containing duplicate or augmented images inflates reported Accuracy by over 10.9 percentage points. Group-aware partitioning is strictly required for sound biomedical ML evaluation.*")
    with open(OUT_DIR / 'Table4_Data_Leakage_Impact_Analysis.tex', 'w', encoding='utf-8') as f:
        f.write(df4.to_latex(index=False, caption="Empirical Impact of Data Leakage on Model Performance Assessment", label="tab:leakage_impact"))

    print("ALL TABLES GENERATED IN experiments/results/tables/")

build_tables()
