# 智慧型傷口 AI 實驗框架

## 目錄結構

```
experiments/
├── run_experiments.py          # Orchestrator (唯一入口)
├── experiment_log.csv          # 自動累積所有實驗結果
│
├── configs/                    # 實驗條件 YAML (只改這裡，code 不動)
│   ├── C-Arch-01_resnet50.yaml
│   ├── C-Arch-02_efficientnet_b0.yaml
│   ├── C-Arch-03_mobilenet_v3.yaml
│   ├── C-Arch-04_vit_s16.yaml
│   ├── C-Arch-05_yolov8n_cls.yaml
│   ├── C-Arch-06_yolov8s_cls.yaml
│   └── C-Arch-07_yolov8m_cls.yaml
│
├── scripts/
│   ├── experiment_logger.py    # Phase 1: 統一日誌
│   ├── train_cls_yolo.py       # Phase 2: YOLO 分類訓練
│   ├── train_cls_torch.py      # Phase 3: PyTorch 訓練
│   ├── evaluate.py             # Phase 4: 完整評估
│   ├── cross_validation.py     # Phase 6: 5-Fold CV
│   ├── multi_seed.py           # Phase 7: Multi-Seed
│   ├── statistical_analysis.py # Phase 8: Bootstrap + McNemar
│   └── generate_tables.py      # Phase 9: 論文表格
│
└── results/
    ├── raw/                    # 各實驗訓練輸出
    ├── predictions/
    │   ├── validation/         # Val 預測 JSON (y_true, y_pred, y_prob)
    │   └── blind_test/         # ⚠️ 最後才放這裡
    ├── statistics/
    │   ├── *_kfold_summary.json
    │   └── *_multiseed_summary.json
    ├── figures/
    │   ├── confusion_matrix/
    │   └── (roc/, calibration/, gradcam/ 後續新增)
    └── tables/
        ├── Table1_Detection_Architecture.csv
        ├── Table2_Classification_Architecture.csv
        ├── Table3_Imbalance_Ablation.csv
        └── Table4_Augmentation_Ablation.csv
```

---

## 使用指令

```bash
# 列出所有可執行的實驗
python experiments/run_experiments.py --list

# 執行第一批（Tier1 分類架構，C-Arch-01~07）
python experiments/run_experiments.py --batch C-Arch

# 執行單一實驗
python experiments/run_experiments.py --single experiments/configs/C-Arch-01_resnet50.yaml

# 5-Fold Cross-Validation（選出最佳架構後執行）
python experiments/run_experiments.py --kfold experiments/configs/C-Arch-05_yolov8n_cls.yaml

# Multi-Seed 穩定性（最佳模型確定後執行）
python experiments/run_experiments.py --multiseed experiments/configs/C-Arch-05_yolov8n_cls.yaml

# 生成論文表格（跑完若干實驗後執行）
python experiments/scripts/generate_tables.py

# 統計分析：Bootstrap CI + McNemar（最後階段）
python experiments/scripts/statistical_analysis.py --mode both \
    --weights_a path_to_best_A.pt \
    --weights_b path_to_best_B.pt \
    --split val
```

---

## Blind Test 使用規範

> ⚠️ **test/ 資料夾 = Blind Test，在所有模型選擇完成前禁止使用**

正確流程：
1. Tier 1 Architecture 實驗 → 只看 `val` 結果
2. Tier 2 Ablation 實驗 → 只看 `val` 結果
3. K-Fold CV → 選出最佳架構
4. Multi-Seed → 確認穩定性
5. 最終模型鎖定
6. **一次性執行 Blind Test** (`--split test`)
7. 計算 Bootstrap 95% CI + McNemar

---

## experiment_log.csv 欄位說明

| 欄位 | 說明 |
|---|---|
| experiment_id | e.g. C-Arch-05 |
| fold | 0=無折, 1-5=K-Fold |
| seed | 42, 123, 3407, 2026, 999 |
| accuracy | Top-1 Val% |
| macro_f1 | Macro F1% (論文主要指標) |
| stab_wound_recall | 少數類別召回率 (RQ3 關鍵) |
| model_path | best.pt 絕對路徑 |

---

## 研究問題對應

| RQ | 對應實驗批次 |
|---|---|
| RQ1 Architecture | C-Arch-01~07 |
| RQ2 Data/Augmentation | C-Aug-01~05, D-Src-01~04 |
| RQ3 Imbalance | C-Imb-01~05 |
| RQ4 Generalization | 最終 Blind Test + McNemar + Bootstrap |
