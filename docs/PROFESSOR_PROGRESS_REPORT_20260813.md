# 博士論文研究進度書面報告

## 智慧型傷口分級與照護對應系統

**報告對象：** 指導教授
**報告日期：** 2026-08-13
**報告範圍：** 分類模型封版、系統架構調整、YOLO11 物件偵測實驗進度與後續決策

## 一、報告摘要

本研究已完成七類傷口影像分類模型的正式封版流程，並將物件偵測擴充為獨立、可稽核的實驗管線。架構上不再把「偵測結果」與「分類結果」直接相乘，而是先確認偵測器的定位能力，再建立同一影像配對的 detector-crop → classifier 測試流程，最後才評估端到端效能。

分類模型目前已完成正式結果：YOLOv8n-cls（C-Arch-05）在 5 個 random seeds × 5 個 MD5-aware StratifiedGroupKFold partitions 的 25 次零資料洩漏評估中，Top-1 Accuracy 為 87.40% ± 2.78%，Macro-F1 為 87.36% ± 3.11%。48 張封存 Blind Test 只執行一次，Accuracy 為 95.83%（46/48），Macro-F1 為 94.76%，Macro ROC-AUC 為 0.9982。

YOLO11 偵測實驗目前仍處於 development benchmark 階段。正式計畫為 YOLO11n、YOLO11s、YOLO11m 三種模型，5 folds × 5 seeds，共 75 runs。截至本報告紀錄，YOLO11n 第 1 fold 已完成 4 個有效 runs；第 2 fold / seed 42 正在訓練，約第 31/150 epochs。1 個未產生完整驗證摘要的 run 已標記為 `ABORTED_PARTIAL`，不納入統計。偵測 locked test 380 張尚未開啟，因此目前不宣稱最終偵測盲測成績或端到端正確率。

## 二、專案架構改動與研究理由

### 2.1 原始架構與目前架構

原始做法是以分類模型直接接收影像並輸出七類傷口類別。此方法可以評估分類器本身，但無法確認輸入影像中的傷口位置，也無法處理背景過多、傷口太小或影像中存在多個候選區域的情況。

目前改為「偵測 → ROI 裁切 → 分類 → 照護對應」的兩階段方向，並保留人工覆核：

```text
輸入影像
  ↓
YOLO11 物件偵測器（定位 Wound ROI）
  ├─ 定位可靠 → ROI crop → YOLOv8n-cls 七類分類 → 照護建議
  ├─ 定位不確定 → 全圖分類輔助 → REVIEW_REQUIRED
  └─ 未偵測到 → fallback／人工覆核
```

這個架構符合臨床場域的安全需求：模型不是在所有情況下強制輸出高信心答案，而是在偵測失敗或定位不確定時交由護理師判斷。後續也可保留護理長的建議紀錄，作為日後規則調整與 LLM 輔助知識整理的輸入，但不會以未驗證的自動學習取代臨床覆核。

### 2.2 為何目前不直接宣稱端到端效能

目前 WSNet 偵測資料是單一 `Wound` detection class，而分類模型是七類傷口分類；兩者尚未形成同一影像、同一標註版本的 paired dataset。因此：

- 不能把 detector mAP 與 classifier Accuracy 相乘。
- 不能把獨立資料集的結果稱為端到端正確率。
- 必須另外計算 detection recall、valid crop rate、conditional classification accuracy 與 overall cascade accuracy。

## 三、分類模型正式成果

### 3.1 分類資料分布與平衡策略

分類資料集共 768 張影像：Development set 720 張，Blind Test 48 張。訓練集七個類別各 97 張，共 679 張；Validation 與 Blind Test 沒有再人工補樣，以保留實際分布，避免測試效能因人為平衡而被高估。

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

訓練設定為 `imgsz=224`、`batch=16`、`epochs=150`、`optimizer=auto`、`lr0=0.01`、`weight_decay=0.0005`。訓練階段使用 `fliplr=0.5`、`flipud=0.5`、`degrees=45`、`scale=0.5`、`erasing=0.4`；驗證與 Blind Test 不使用訓練增強。

### 3.2 分類資料洩漏控制

720 張 Development images 經 MD5 exact-content grouping 後形成 383 個 unique content groups：177 個 singleton groups，以及 206 個 duplicate groups（涵蓋 543 個檔案）。每個 duplicate group 的冗餘複本為 337 張。GroupKFold 將完整 MD5 group 放在同一 fold，最終稽核為 0 個跨 fold MD5 overlap。

### 3.3 分類結果

| 指標 | Development / 25 runs | Blind Test / 48 張 |
|---|---:|---:|
| Top-1 Accuracy | 87.40% ± 2.78% | 95.83%（46/48） |
| Top-5 Accuracy | — | 100.00% |
| Macro-F1 | 87.36% ± 3.11% | 94.76% |
| Weighted-F1 | 87.42%（bootstrap mean） | 96.23% |
| Macro ROC-AUC | — | 0.9982 |

Development pooled validation predictions 使用 `B=2,000` bootstrap；Accuracy 95% CI 為 `[86.32%, 88.48%]`。Blind Test 僅使用一次，未回頭調參，也未與交叉驗證結果重新平均。

## 四、YOLO11 物件偵測實驗架構

### 4.1 正式資料集

目前可重現的公開候選資料採官方 WSNet image-mask split。資料總量為 2,686 張影像與 3,823 個 bounding boxes，偵測類別為單一 `Wound`。

| Split | Images | Bounding boxes | 用途 |
|---|---:|---:|---|
| Train | 1,894 | 2,681 | 模型訓練 |
| Validation | 412 | 609 | 開發期評估與早停 |
| Locked Test | 380 | 533 | 最終一次性盲測，尚未開啟 |
| **合計** | **2,686** | **3,823** | |

資料以 exact image hash 進行群組隔離，並保留影像、mask、YOLO label 的檔案雜湊與版本紀錄。FUSC、Medetec、Roboflow 及臨床來源資料仍依授權、consent、去識別化與 augmentation lineage 證據狀態分開管理，不在證據未完成前宣稱為正式公開 benchmark。

### 4.2 偵測實驗 phases

| Phase | 內容 | 狀態 |
|---|---|---|
| D1 | 設定檔、trainer、evaluator、logger | 已建立，最終版尚未封版 |
| D2 | Image/label、MD5、split isolation 稽核 | PASS |
| D3 | YOLO11m smoke test | 已完成，僅驗證流程，不作科學結果 |
| D4 | Exact-hash GroupKFold development | 進行中 |
| D5 | YOLO11n/s/m × 5 seeds × 5 folds | 進行中 |
| D6 | 偵測論文表格與 bootstrap CI | 待完整 benchmark |
| D7 | 380 張 locked detection test | 尚未開啟 |

### 4.3 目前訓練進度

正式 benchmark 規劃為 75 runs。最新註冊狀態如下：

| 項目 | 目前狀態 |
|---|---|
| YOLO11n Fold 1 / Seed 42 | 有效完成，mAP@50 0.4909 |
| YOLO11n Fold 1 / Seed 123 | 有效完成，mAP@50 0.4943 |
| YOLO11n Fold 1 / Seed 3407 | 有效完成，mAP@50 0.4426 |
| YOLO11n Fold 1 / Seed 999 | 有效完成，mAP@50 0.4634 |
| YOLO11n Fold 1 / Seed 2026 | `ABORTED_PARTIAL`，排除統計 |
| YOLO11n Fold 2 / Seed 42 | 訓練中，約 epoch 31/150 |
| YOLO11s、YOLO11m 正式 benchmark | 尚未完成 |
| Detection locked test | 未使用 |

上述 mAP 數字是 development-fold 結果，不是 locked test 結果；在三模型、多種子、多 fold 完成前，不進行最終模型排名或部署宣稱。

## 五、目前研究風險與控制措施

1. **偵測與分類 domain shift：** WSNet 的影像分布可能與臨床影片不同，因此必須在 paired clinical validation 或明確標示的外部驗證資料上檢查泛化。
2. **偵測框錯誤會影響分類：** 需記錄 missed detection、wrong crop、background contamination 與 no-detection rate。
3. **資料授權與臨床治理：** 未完成來源授權、IRB／consent、去識別化與 augmentation lineage 的資料只保留稽核紀錄，不納入公開結果。
4. **模型版本漂移：** 正式部署前須建立 model registry，明確記錄 detector、classifier、config、weights hash 與資料版本，避免 backend 載入未註冊的舊權重。

## 六、下一階段工作

1. 完成 YOLO11n Fold 2 至 Fold 5，並依序完成 YOLO11s、YOLO11m 的正式 multi-seed GroupKFold。
2. 產生偵測模型的統一表格、平均值、標準差與 bootstrap confidence intervals。
3. 建立 paired cascade validation set：同一影像同時具有 detection box 與七類 classification label。
4. 固定 detector threshold、ROI padding 與 no-detection fallback，禁止使用 paired blind test 調參。
5. 以獨立 paired test 報告 detection recall、valid crop rate、conditional Macro-F1、overall cascade Macro-F1、no-detection rate、wrong-crop rate 與 P50/P95 latency。
6. 完成以上步驟後，才開啟 WSNet 380 張 locked detection test，執行一次性最終評估。

## 七、請教授確認的研究決策

1. 是否同意採用「YOLO11 偵測 + YOLOv8n 分類 + 人工覆核」作為系統的主要落地架構？
2. 是否同意在來源證據完整前，以 WSNet 作為正式偵測 benchmark，其他來源維持 quarantine／provenance audit？
3. 是否同意完成 75-run development benchmark 與 paired cascade validation 後，再執行 380 張 detection locked test？

## 八、可追溯文件與來源

- [教授詳細實驗報告](./PROFESSOR_DETAILED_EXPERIMENT_REPORT.md)
- [教授審閱索引](./PROFESSOR_REVIEW_INDEX.md)
- [資料平衡與詳細訓練設定](./SHAREPOINT_INSERT_DATA_BALANCE.md)
- [系統架構建議](../outputs/deployment_architecture_recommendation_20260812.md)
- [端到端風險與驗收標準](../outputs/cascade_deployment_risk_and_acceptance_20260812.md)
- [WSNet 官方資料集](https://huggingface.co/datasets/subbareddyoota/wseg_dataset)
- [WSNET 程式碼 repository](https://github.com/subbareddy248/WSNET)

**報告結論：** 分類模型已完成正式封版；物件偵測已建立可追溯的正式實驗流程，目前正在進行 development benchmark。偵測 locked test 與端到端 cascade accuracy 尚未宣稱，待資料配對、模型比較與治理證據完成後再進行一次性驗證。
