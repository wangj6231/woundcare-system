# 智慧型傷口分級與照護對應系統：專案檔案結構

更新日期：2026-08-12  
整理原則：訓練執行期間只建立索引，不搬移、不覆寫、不刪除正在使用的檔案。

## 1. 目前核心資料夾

| 類別 | 位置 | 用途 | 狀態 |
|---|---|---|---|
| 分類資料 | `Wound_dataset/` | 原始七類分類影像 | 保留，不修改 |
| 分類標準資料 | `yolo_wound_cls_dataset_v3/` | 分類 train/val/test | Phase 1–9 已封版 |
| 分類結果 | `runs/classify/` | 分類訓練權重與結果 | 保留，不與偵測混用 |
| 臨床偵測資料 | `clinical_detection_isolated_20260812/` | 臨床 frame 與 YOLO labels | 隔離，不與公開來源混合 |
| WSNet 官方資料 | `yolo_dataset_wsnet_official_v1/` | 目前偵測 benchmark 唯一訓練來源 | test split 鎖定 |
| 官方來源保存 | `official_detection_sources_20260812/` | WSNet/FUSeg/Medetec 原始證據與雜湊 | 僅 WSNet 已進入 benchmark |
| 實驗設定 | `experiments/configs/` | 分類與偵測設定檔 | 保留 |
| 實驗清單 | `experiments/manifests/` | split、hash、75-run 計畫 | 保留 |
| 實驗結果 | `experiments/results/` | 分類、偵測、統計與圖表結果 | 訓練完成前不搬移 |
| 報告與證據 | `outputs/` | 教授報告、來源、稽核、摘要 | 目前正式文件位置 |
| 開發腳本 | `work/` | 建置、稽核、訓練與監控腳本 | 保留，待訓練後再分層 |

## 2. 正在執行的正式 run

`experiments/results/detection_groupkfold/YOLO11n_WSNet_groupkfold_fold01_seed123_20260812/`

- 模型：YOLO11n
- Fold：1
- Seed：123
- 最大 epochs：150
- Test split：未使用
- 目前狀態：RUNNING（整理時不得搬移）

其監控紀錄：

- `outputs/wsnet_formal_benchmark_status_20260812.md`
- `work/monitor_formal_yolo11n.ps1`

## 3. 正式 benchmark 來源與規範

- GroupKFold manifests：`experiments/manifests/wsnet_official_groupkfold_20260812/`
- 75-run 清單：`experiments/manifests/wsnet_formal_benchmark_runs_20260812.csv`
- 正式計畫：`outputs/wsnet_formal_benchmark_plan_20260812.md`
- WSNet 版本：`bdcc1e1cef607bfd7a4f98fc28f31a638a0e0ca9`
- Development：2,306 images / 2,149 exact image-hash groups
- Locked test：380 images，尚未開啟

## 4. 分類與偵測的邊界

分類 Phase 1–9 已封版；偵測 benchmark 是獨立實驗，不得改寫分類結果。現階段尚未建立可驗證的同影像 `bbox + 七類 wound class` paired set，因此不得把 WSNet 偵測結果直接宣稱為偵測→分類端到端正確率。

## 5. 已刪除／不再使用的資料

Kaggle 與 Roboflow 偵測副本已依使用者要求移除；臨床資料保留並隔離；分類資料與分類結果保留。詳見：

`outputs/detection_public_data_deletion_20260812.md`

## 6. 訓練完成後才可執行的整理

1. 將已完成的正式 run 與 pilot run 依 `formal/`, `pilot/`, `aborted/` 建立索引或資料夾。
2. 將重複產生的 suffix run（例如 `...202608122`、`...202608123`）標記為 aborted/diagnostic；先不刪除。
3. 將空白或失敗的啟動日誌集中至 `outputs/runtime_logs/`，保留原始內容。
4. 對所有正式權重、JSON、CSV 產生 SHA-256 manifest。
5. 更新教授進度報告與 benchmark summary。

在正式 run 完成前，不執行上述實體搬移，以避免 Windows 檔案鎖定、權重寫入中斷或結果路徑失效。
