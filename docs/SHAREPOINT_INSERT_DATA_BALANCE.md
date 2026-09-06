# 可加入教授進度報告的新增章節：資料平衡與詳細訓練設定

## 資料平衡與訓練資料處理策略

本研究的資料平衡只作用於訓練資料，不對驗證集或封存測試集進行人工補樣。這樣可以同時降低少數類別在訓練時的偏置，並保留驗證與盲測資料的實際分布，避免以人為平衡後的測試集高估模型效能。

### 1. 七類傷口分類資料的平衡方式

分類資料集共有 768 張影像，其中 Development set 為 720 張，另外 48 張為完全隔離的 Blind Test。依目前 `C-Arch-05_yolov8n_cls.yaml` 設定，資料來源欄位記錄為 `Wound_dataset+Oversampling+Rotation`；實際訓練目錄的結果是七個類別各有 97 張訓練影像，因此訓練集總數為 679 張。換句話說，過採樣與旋轉增強只用來補足訓練階段的類別差異，沒有把 Validation 或 Blind Test 強制改成每類相同數量。

| 類別 | Train | Validation | Blind Test | 總數 |
|---|---:|---:|---:|---:|
| Abrasions | 97 | 8 | 9 | 114 |
| Bruises | 97 | 12 | 13 | 122 |
| Burns | 97 | 5 | 7 | 109 |
| Cut | 97 | 5 | 5 | 107 |
| Ingrown_nails | 97 | 3 | 4 | 104 |
| Laceration | 97 | 6 | 7 | 110 |
| Stab_wound | 97 | 2 | 3 | 102 |
| **合計** | **679** | **41** | **48** | **768** |

訓練階段的影像增強設定為：水平翻轉機率 `fliplr=0.5`、垂直翻轉機率 `flipud=0.5`、旋轉角度 `degrees=45`、縮放 `scale=0.5`、隨機擦除 `erasing=0.4`。這些操作只在訓練流程中啟用；驗證與 Blind Test 僅使用原始影像和固定前處理，以確保評估結果可解釋。

### 2. 重複內容稽核與資料洩漏控制

平衡後的訓練影像仍以 MD5 進行 exact-content grouping。720 張 Development image 形成 383 個 unique content groups，其中 177 個為 singleton groups，206 個為 duplicate groups（涵蓋 543 個檔案）；每個 duplicate group 以一個內容群組處理，故冗餘複本為 `543-206=337`。GroupKFold 將完整群組分配到同一 fold，最終稽核為 0 個 MD5 group 跨越 train/validation fold。這一步避免同一張影像或其完全相同複本同時出現在訓練與驗證資料中。

### 3. 分類模型訓練與統計程序

最終候選模型為 YOLOv8n-cls（C-Arch-05），`imgsz=224`、`batch=16`、`epochs=150`、`optimizer=auto`、`lr0=0.01`、`weight_decay=0.0005`。正式穩定性分析使用 seeds `{42, 123, 3407, 2026, 999}` 與 5-fold MD5-aware StratifiedGroupKFold，共 25 次 leakage-free evaluation runs。正式 Development 結果為 Top-1 Accuracy `87.40% ± 2.78%`、Macro-F1 `87.36% ± 3.11%`；以 `B=2,000` 非參數 bootstrap 得到 Accuracy 95% CI `[86.32%, 88.48%]`。

48 張 Blind Test 僅在模型選定後執行一次，不回頭調參，也不與 25 次交叉驗證結果重新平均。一次性結果為 Accuracy `95.83% (46/48)`、Top-5 `100%`、Macro-F1 `94.76%`、Weighted-F1 `96.23%`、Macro ROC-AUC `0.9982`。

### 4. YOLO11 偵測資料的平衡與隔離方式

目前可重現的公開來源候選資料採用官方 WSNet image-mask split，而非把尚未完成來源授權確認的混合資料直接當作正式結果。WSNet 為單一 `Wound` detection class，因此不進行七類分類式的類別過採樣；平衡重點改為維持官方 train/validation/test 比例、確認影像與標註框一一對應，以及以 exact image hash 進行群組隔離。

| Split | Images | Bounding boxes | 用途 |
|---|---:|---:|---|
| Train | 1,894 | 2,681 | 模型訓練 |
| Validation | 412 | 609 | 開發期選模與早停 |
| Locked Test | 380 | 533 | 尚未開啟的最終偵測盲測 |
| **合計** | **2,686** | **3,823** | |

YOLO11 偵測正式比較採共同設定 `imgsz=640`、`batch=4`、最大 150 epochs、patience 20、deterministic、workers 0，並比較 YOLO11n、YOLO11s、YOLO11m。D-Arch-01 設定檔中的 5 epochs/patience 3 是 pipeline smoke/baseline 設定；正式多模型結果必須以註冊的完整 benchmark 為準。Locked Test 在開發期間維持未讀取狀態，因此目前不能把偵測器 mAP 宣稱為盲測結果。

### 5. 端到端落地驗證的限制

偵測器目前輸出單一 `Wound` 框，分類器輸出七類傷口類別；兩者尚未形成同一影像、同一標註版本的 paired cascade 評估集。因此，偵測 mAP 與分類 Accuracy 不能直接相乘，也不能直接宣稱端到端正確率。完成 YOLO11m 多模型、多種子 GroupKFold 後，下一步應建立固定的 detector-crop → classifier 輸入流程，並在不調參的獨立 paired test 上一次性報告 detection success rate、crop quality、classification accuracy 與整體 cascade accuracy。

### 6. 可追溯性與研究治理

每一次訓練均應保留 config、seed、資料版本、split manifest、模型權重雜湊、訓練曲線、評估 JSON 與錯誤案例。WSNet 公開資料集來源為 [Hugging Face wseg_dataset](https://huggingface.co/datasets/subbareddyoota/wseg_dataset)，固定 revision `bdcc1e1cef607bfd7a4f98fc28f31a638a0e0ca9`，頁面標示授權為 CC BY-NC 4.0；程式碼來源另記錄於 [WSNET repository](https://github.com/subbareddy248/WSNET)。臨床影像及其他第三方資料在 consent、去識別化、augmentation lineage 與授權證據補齊前，僅列為 provenance audit，不納入正式可公開 benchmark。
