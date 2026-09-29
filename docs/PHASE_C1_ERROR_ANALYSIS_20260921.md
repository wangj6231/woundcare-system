# Phase C1：封版預測之受控錯誤分析與失敗分類

日期：2026-09-21。範圍：FUSeg development validation，191 張影像。狀態：COMPLETE。

## 1. 結論與執行邊界

最明確的系統性問題是**極小傷口漏檢，並與多傷口影像的覆蓋不足重疊**。32 個 FN 中 28 個為 small；其中 20 個 bbox 面積低於影像的 0.25%。多傷口影像的 instance recall 為 77.78%，低於單傷口的 92.05%；裁切失敗率分別為 34.29% 與 4.64%。

這些是相對既有 GT 與封版 matching 的描述性發現，不是臨床診斷、標註錯誤判定，亦不是改進已經成功的證據。下一個第一優先訓練候選是「僅依 training GT 幾何資訊進行 small-object-aware image sampling」，狀態 **CANDIDATE_ONLY，未授權、未執行**。

本階段只讀取 Phase C 已保存之結果、GT metadata、191 個預測 mask artifacts，以及封版的 191 張 validation 影像。原圖只用於檢視圖與不依賴模型的品質描述。未載入候選模型、未重新推論、未訓練、未調整 confidence/NMS/IoU/crop/imgsz，未使用 locked test 或 CO2Wounds，未執行 bootstrap/CI/multi-seed，未替換 App 模型。App 的 `classification_source = full_image` 與既有裁切邏輯保持原狀。

Phase C 的 development gate 仍為 **FAIL**。C1 的分析完成，不等於模型達標、可部署或取得新訓練許可。

## 2. 不可變證據與驗證

分析來源：[Phase C 結果](../experiments/results/isic_fuseg_colorfix_v1/result.json)、[逐張預測](../experiments/results/isic_fuseg_colorfix_v1/per_image_predictions.json)、同目錄 `per_image/`、`masks/` 及封版 protocol/manifest。

| 項目 | SHA256／核對結果 |
|---|---|
| Protocol | `16718f5956548e002320ebd2ade83c5ecafdf77e7c79dc1f1d2b63818b7e6df3` |
| Validation manifest | `ea58c4e7d36b699d2f1e232d23fea8df9a6010674d08769e5a2ca3aca66ab069` |
| Checkpoint | `cc2955d088928dc00c90d0ba8cab0aefeb49d4cc45da172b030c150ab0dce97a` |
| result.json | `d797561841227617ad3c8652164d12daa03d63924cada9739c9ef473007ca79e` |
| per_image_predictions.json | `1197dea64a089f9c86d0e710ab19fe4bbb199f1995a73441eb6520a147f5e9e4` |
| 預測 mask artifact SHA256 | 191/191 通過 |
| 逐張 JSON 與彙整預測一致性 | 191/191 通過 |
| 前後內容雜湊不變 | 403 個保護檔案，changed = 0 |
| 新增分析可重建性 | 由同一封版預測重建之 27 個衍生欄位一致 |

403 個保護檔案包括 Phase C 既有輸出、protocol/manifest、checkpoint、評估及裁切相關程式與 Phase C 報告；只新增 `phase_c1_*` 分析成果、C1 程式/測試及本報告。Checkpoint 僅計算檔案雜湊，不反序列化或載入模型。

完整前後快照：[phase_c1_integrity.json](../experiments/results/isic_fuseg_colorfix_v1/phase_c1_integrity.json)。逐項產物驗證：[phase_c1_artifact_verification.json](../experiments/results/isic_fuseg_colorfix_v1/phase_c1_artifact_verification.json)。

## 3. 分母、單位及既有結果

| 項目 | 數量／結果 |
|---|---:|
| Images | 191：positive 186、negative 5 |
| GT instances | 241：small 137、medium 90、large 14 |
| TP / FP / FN | 209 / 36 / 32 |
| Precision / Recall / F1 | 85.31% / 86.72% / 86.01% |
| Small / Medium / Large Recall | 109/137 = 79.56% / 86/90 = 95.56% / 14/14 = 100.00% |
| Positive crop complete ≥95% | 167/186 = 89.78% |
| Crop incomplete | 19 張：無 ROI 4 張，有 ROI 15 張 |

P/R/F1 沿用 Phase C 的 instance matching；不是 mAP，也不是七類傷口分類準確率。C1 不重新配對或修改指標。額外計算 bbox IoU 只用來描述「已判定 FP」與 GT 的幾何關係。

FN 有 32 個 instance、涉及 29 張圖；FP 有 36 個 prediction、涉及 31 張圖；crop failure 有 19 張 positive image。三者影像聯集為 49 張，不可把 32+36+19 當作 87 個獨立案例。

| 互斥影像組合 | 張數 |
|---|---:|
| 僅 FP | 19 |
| 僅 FN | 3 |
| FN + FP，不含 crop failure | 8 |
| FN + crop failure，不含 FP | 14 |
| FN + FP + crop failure | 4 |
| 僅 crop failure | 1 |
| FP + crop failure，不含 FN | 0 |
| 三組皆無 | 142 |

18/19 crop failure 同時有 FN；另 1 張（0679）無 FN 但 GT pixel retention 仍未達 95%。反過來說，FN 與 crop failure 也不是同一指標：部分未成功匹配的 GT 仍可落在較大的裁切區內。

## 4. FN 與極小傷口分析

Primary small 定義維持 `GT bbox area / image area <1%`。所有此次封版 validation 影像尺寸經查為 512×512；這是資料集影像座標，不推稱為拍攝裝置原始解析度。Secondary bins 為左閉右開區間，不更動 primary size 定義。

| Small bbox 面積占比 | Support | TP | FN | Recall |
|---|---:|---:|---:|---:|
| <0.10% | 27 | 13 | 14 | 48.15% |
| 0.10–<0.25% | 22 | 16 | 6 | 72.73% |
| 0.25–<0.50% | 38 | 33 | 5 | 86.84% |
| 0.50–<0.75% | 30 | 29 | 1 | 96.67% |
| 0.75–<1.00% | 20 | 18 | 2 | 90.00% |
| 全部 small | 137 | 109 | 28 | 79.56% |

低於 0.25% 的 49 個 GT 只占 small support 的 35.77%，卻占 small FN 的 20/28 = 71.43%。低於 0.10% 者占 small FN 的 14/28 = 50.00%。這是優先研究極小目標的依據；不以小分組結果宣稱統計顯著或泛化效果。

### 4.1 Detected 與 missed 幾何比較

以下為中位數；完整 mean、median、P25、P75、min、max 與逐 instance 紀錄皆在分析 JSON/CSV。

| 特徵 | 全 GT detected，n=209 | 全 GT missed，n=32 | Small detected，n=109 | Small missed，n=28 |
|---|---:|---:|---:|---:|
| bbox 面積占比 | 0.9502% | 0.1339% | 0.4257% | 0.0963% |
| bbox 面積，pixel² | 2,491 | 351 | 1,116 | 252.50 |
| bbox 寬，pixel | 47 | 18 | 34 | 16.50 |
| bbox 高，pixel | 47 | 21 | 33 | 18.50 |
| 寬／高比 | 0.9714 | 0.9268 | 1.0000 | 0.9268 |
| 最近邊界 normalized distance | 0.3008 | 0.2783 | 0.3145 | 0.2900 |
| 所在影像 GT 數 | 1 | 2 | 1 | 2 |

Small detected/missed 的 width ratio 中位數為 0.06641 / 0.03223，height ratio 為 0.06445 / 0.03613；面積占比平均數為 0.4475% / 0.2261%。尺寸差異比單純「看起來貼近邊界」更明顯。

邊界距離定義為 `min(x1/W, (W-x2)/W, y1/H, (H-y2)/H)`。Small missed 最小距離為 0.04297，detected 最小距離為 0.00586；不能据此把漏檢主因判為影像邊緣截斷。邊界比較是描述性，未排除 size、構圖、多 GT 等混雜。

未提供逐 GT instance mask occupancy：既有 publisher mask 為整張影像的 union，不能杜撰 instance mask 對應。bbox 面積不可寫成傷口實際像素面積。

## 5. 單傷口、多傷口與 instance-count strata

| 指標 | Single GT | Multi GT |
|---|---:|---:|
| Positive images | 151 | 35 |
| GT instances | 151 | 90 |
| TP / FN | 139 / 12 | 70 / 20 |
| Instance recall | 92.05% | 77.78% |
| 至少一個 FN 的影像 | 12/151 = 7.95% | 17/35 = 48.57% |
| Crop pass | 144/151 = 95.36% | 23/35 = 65.71% |
| Crop fail | 7/151 = 4.64% | 12/35 = 34.29% |
| Mean retained GT pixel fraction | 96.50% | 91.42% |
| FP / image | 25/151 = 0.166 | 10/35 = 0.286 |

Multi 的 instance recall 低 **14.28 個百分點**，crop fail rate 高 **29.65 個百分點**。這是組間觀察差異，不是多傷口造成失敗的因果估計。5 張 negative images 不列入這個正樣本比較。

| 每張 GT 數 | Images | GT | TP | FN | Recall | Crop pass | Mean retained fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 151 | 151 | 139 | 12 | 92.05% | 144/151，95.36% | 96.50% |
| 2 | 22 | 44 | 33 | 11 | 75.00% | 14/22，63.64% | 88.18% |
| 3+ | 13 | 46 | 37 | 9 | 80.43% | 9/13，69.23% | 96.90% |

3+ 組並非每項都比 2 組差；不可寫成「傷口數越多，表現必然線性下降」。GT 數、尺寸分布和 pixel-weighted coverage 不是同一件事。

## 6. Crop failure severity 與逐張分解

Crop retention 是保存的 GT 像素保留比例，完全沿用 Phase C；不是重新以 bbox 面積推算。表內 retention 全以百分比呈現，分母為 19 張失敗影像；空 bin 的統計不適用。

| Retained fraction bin | Count | 占 19 張 | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 4 | 21.05% | 0 | 0 | 0 | 0 |
| >0–25% | 2 | 10.53% | 14.77 | 14.77 | 13.54 | 16.01 |
| >25–50% | 1 | 5.26% | 42.82 | 42.82 | 42.82 | 42.82 |
| >50–75% | 4 | 21.05% | 70.76 | 72.17 | 64.18 | 74.55 |
| >75–90% | 1 | 5.26% | 81.56 | 81.56 | 81.56 | 81.56 |
| >90–<95% | 7 | 36.84% | 93.15 | 93.82 | 90.88 | 94.94 |
| ≥95% | 0 | 0% | — | — | — | — |

整體 19 張 mean 57.32%、median 73.37%、range 0–94.94%。若將 ≤25% 作為本次描述性的嚴重失敗區間，則有 6/19 張；另 7 張介於 90% 與 95%，不能只報「接近過關」。全體 positive 中 ≥95% 的 167 張是 pass，未放進此 failure cohort。

### 6.1 機器輔助規則

互斥 primary 優先順序：NO_ROI → LOCALIZED_WRONG_REGION → MISSED_SECOND_INSTANCE → PARTIAL_GT_COVERAGE → OTHER。

- NO_ROI：保存的 crop 為空。
- LOCALIZED_WRONG_REGION：有預測、有 ROI，但保存 matching 的 TP=0。**此名稱不代表與 GT 完全無重疊**；例如 0549 retention 仍有 70.97%，屬定位未達 matching 要求。
- MISSED_SECOND_INSTANCE：多 GT 影像中有未匹配 GT bbox 未完全包含於保存 crop。表示漏掉其他 instance 的幾何證據，不限定只漏「第二個」，也不是逐 instance mask 因果歸屬。
- PARTIAL_GT_COVERAGE：其餘有 ROI 但 GT pixels 保留未達 95%；亦作為其他有 ROI failure 的 secondary tag。
- OTHER：此次 0 張。

### 6.2 全部 19 張，無篩選展示

下表以檔名 `fuseg__NNNN.png` 的 NNNN 簡記。

| ID | GT | TP/FP/FN | Retained fraction | Primary type | Secondary |
|---|---:|---|---:|---|---|
| 0089 | 2 | 1/0/1 | 90.88% | MISSED_SECOND_INSTANCE | PARTIAL_GT_COVERAGE |
| 0164 | 3 | 2/1/1 | 81.56% | MISSED_SECOND_INSTANCE | PARTIAL_GT_COVERAGE |
| 0192 | 2 | 1/0/1 | 64.18% | MISSED_SECOND_INSTANCE | PARTIAL_GT_COVERAGE |
| 0349 | 2 | 1/2/1 | 13.54% | MISSED_SECOND_INSTANCE | PARTIAL_GT_COVERAGE |
| 0456 | 1 | 0/0/1 | 0.00% | NO_ROI | — |
| 0516 | 4 | 2/0/2 | 92.07% | MISSED_SECOND_INSTANCE | PARTIAL_GT_COVERAGE |
| 0548 | 2 | 1/0/1 | 42.82% | MISSED_SECOND_INSTANCE | PARTIAL_GT_COVERAGE |
| 0549 | 1 | 0/1/1 | 70.97% | LOCALIZED_WRONG_REGION | PARTIAL_GT_COVERAGE |
| 0591 | 1 | 0/0/1 | 0.00% | NO_ROI | — |
| 0604 | 3 | 2/0/1 | 93.82% | MISSED_SECOND_INSTANCE | PARTIAL_GT_COVERAGE |
| 0679 | 1 | 1/0/0 | 94.75% | PARTIAL_GT_COVERAGE | — |
| 0738 | 2 | 1/0/1 | 91.59% | MISSED_SECOND_INSTANCE | PARTIAL_GT_COVERAGE |
| 0798 | 1 | 0/1/1 | 16.01% | LOCALIZED_WRONG_REGION | PARTIAL_GT_COVERAGE |
| 0816 | 1 | 0/0/1 | 0.00% | NO_ROI | — |
| 0937 | 2 | 1/0/1 | 73.37% | MISSED_SECOND_INSTANCE | PARTIAL_GT_COVERAGE |
| 0964 | 3 | 2/0/1 | 94.94% | MISSED_SECOND_INSTANCE | PARTIAL_GT_COVERAGE |
| 0971 | 2 | 1/0/1 | 74.55% | MISSED_SECOND_INSTANCE | PARTIAL_GT_COVERAGE |
| 0973 | 1 | 0/0/1 | 0.00% | NO_ROI | — |
| 1009 | 2 | 1/0/1 | 93.99% | MISSED_SECOND_INSTANCE | PARTIAL_GT_COVERAGE |

有 ROI 的 15 張中，12 張（80%）有未匹配的其他 GT bbox 在 crop 外；2 張（13.33%）有預測但 TP=0；1 張（6.67%）匹配成功但 coverage 不足。這是主要觀察型態，不代表已確認根本原因。

4 張無 ROI：0456、0816、0973 為 small；0591 為 medium。因此 3/4 為 small，但不能說無 ROI 只發生於 small。

距離原 gate 90.32% 為 **1 張 positive image**（167→168）；這僅是距離描述，不挑選最容易救回的案例、不改規則、不宣告通過。

### 6.3 Crop failure × small presence

| 正樣本群組 | Images | Pass | Fail | Complete rate |
|---|---:|---:|---:|---:|
| 至少一個 small | 114 | 98 | 16 | 85.96% |
| 無 small | 72 | 69 | 3 | 95.83% |
| 單一 small | 81 | 75 | 6 | 92.59% |
| 多 GT，全部 small | 11 | 8 | 3 | 72.73% |
| Small + medium/large | 22 | 15 | 7 | 68.18% |

後三列為「至少一個 small」的細分，不能與前兩列加總。16/19 failures 含 small，12/19 為 multi；兩者重疊，不可相加成可歸因比例。

## 7. FP 幾何分類及 confidence

全部 36 個 FP 以既有 matching 判定，再套用互斥主分類。高重疊可能重複定義為 bbox IoU≥0.50 且該 GT 已被另一預測匹配；低 IoU 區間為 0<max IoU<0.50；max IoU=0 僅表示 bbox 不相交，不代表醫學上的背景或確定漏標。

| Primary taxonomy | 數量 | 占 FP | 解讀 |
|---|---:|---:|---|
| FP_ON_POSITIVE_IMAGE_OTHER_REGION | 18 | 50.00% | Positive image、max IoU=0；tag FP_SPATIALLY_DISTANT |
| FP_LOCALIZATION_MISMATCH | 15 | 41.67% | 0<max IoU<0.50；tag FP_NEAR_GT_LOW_IOU |
| FP_DUPLICATE_NEAR_GT | 2 | 5.56% | 與已匹配 GT 高重疊；tag FP_POSSIBLE_DUPLICATE |
| FP_NEGATIVE_IMAGE | 1 | 2.78% | GT-negative image 上的 prediction |
| FP_UNCLEAR | 0 | 0% | 其他 |

15 個低 IoU FP 再細分：(0,0.10) 有 2 個；[0.10,0.25) 有 4 個；[0.25,0.50) 有 9 個。35/36 FP 位於 positive images；不得概稱為「36 個背景誤判」。Duplicate-like 也不得直接改稱標註錯誤。

| Confidence（0–1） | TP，n=209 | FP，n=36 |
|---|---:|---:|
| Mean | 0.9081 | 0.5251 |
| Median | 0.9419 | 0.5208 |
| P25 | 0.9128 | 0.2052 |
| P75 | 0.9574 | 0.8134 |
| Min | 0.1325 | 0.1407 |
| Max | 0.9713 | 0.9422 |

中央區間並非高度重疊：FP P75 低於 TP P25。但 FP 整體 range 位於 TP range 內，仍有高 confidence FP 及低 confidence TP，不能完美分離。固定寬 0.10 的 histogram overlap coefficient 為 0.2929（各 bin 的兩組相對頻率取最小後加總）；這依賴分箱，並非分類準確率或校準指標。2/36 FP 位於 TP IQR。

Histogram/ECDF 只描述凍結 `confidence=0.10` 下已保存預測的條件分布，無法說明被更低 threshold 排除的預測。**沒有 threshold sweep、最佳值或建議 threshold**；只支持未來另行預先登錄 threshold 實驗的候選假設。

## 8. 不依賴模型的影像品質描述

單位為整張影像，不是 wound ROI；positive images 分為全部 GT 成功偵測（157 張）和至少一個 FN（29 張），每張只算一次。5 張 negative 的特徵保留於 CSV，但不混入這兩組比較。

| 描述量 | 定義 | 全 GT detected：mean / median | Any FN：mean / median |
|---|---|---:|---:|
| Brightness | cv2 RGB2GRAY uint8 平均值，0–255 | 93.00 / 93.70 | 97.04 / 96.46 |
| Contrast | 同灰階 pixels 的母體標準差 | 76.32 / 77.91 | 77.02 / 80.43 |
| Sharpness | cv2 Laplacian，CV_64F，ksize=1 的 variance | 100.02 / 92.72 | 101.73 / 96.11 |
| Saturation | cv2 RGB2HSV 的 S 平均值／255 | 0.1714 / 0.1697 | 0.1655 / 0.1668 |

Missed 組並未呈現簡單一致的「更暗、對比更低、更模糊」平均值模式。兩組分布重疊；黑色填邊、背景、拍攝構圖及多 GT 均可能影響整圖指標。不能寫成「低對比造成漏檢」或「影像品質已排除」。未進行 formal p-value、bootstrap 或 CI。

## 9. 分開計數的 Pareto 摘要

每個表只使用互斥 primary category；secondary tags 不加進累計。

| FN instance primary category | Count / 32 | % | Cumulative % |
|---|---:|---:|---:|
| Small FN，multi-GT | 18 | 56.25 | 56.25 |
| Small FN，single-GT | 10 | 31.25 | 87.50 |
| Medium FN，multi-GT | 2 | 6.25 | 93.75 |
| Medium FN，single-GT | 2 | 6.25 | 100.00 |

| FP prediction primary category | Count / 36 | % | Cumulative % |
|---|---:|---:|---:|
| Positive image other region | 18 | 50.00 | 50.00 |
| Localization mismatch | 15 | 41.67 | 91.67 |
| Duplicate-like | 2 | 5.56 | 97.22 |
| Negative image | 1 | 2.78 | 100.00 |

| Crop image primary category | Count / 19 | % | Cumulative % |
|---|---:|---:|---:|
| Missed other instance | 12 | 63.16 | 63.16 |
| No ROI | 4 | 21.05 | 84.21 |
| Localization without TP | 2 | 10.53 | 94.74 |
| Partial coverage after matching | 1 | 5.26 | 100.00 |

Small FN 解釋 87.5% 的 FN；multi-GT 上的 FN 為 20/32 = 62.5%；multi-GT crop failures 為 12/19 = 63.16%。最大群組因「計數單位」不同而不同，不能宣稱某一個原因解釋全部 87 筆失敗紀錄。

## 10. Evidence → candidate intervention（未執行）

| Failure evidence | Possible intervention | 新實驗？ | 狀態 |
|---|---|---|---|
| 28/32 FN 為 small；20/28 small FN <0.25% | Training-GT-only small-object-aware image sampling；唯一第一優先訓練候選 | YES | CANDIDATE_ONLY |
| Multi 的 crop fail 34.29%；12/15 ROI-present failures 有漏掉的其他 GT bbox 在 crop 外 | 獨立的 crop-policy／multi-ROI 研究 | YES，與 training 分開 | CANDIDATE_ONLY |
| 15/36 FP 低 IoU | 定位品質研究問題，暫不選為另一個優先訓練方案 | YES，未排程 | CANDIDATE_ONLY |
| FP 中位 confidence 較低，但與 TP range 重疊 | 預先登錄的 threshold 研究 | YES，無新 threshold 建議 | CANDIDATE_ONLY |

### 唯一第一優先新訓練實驗

名稱：**只改 training image sampling distribution 的小目標感知抽樣**。

選擇理由：極小 bbox 是這批封版 development evidence 中最集中的漏檢型態。以 training GT 的 size 資訊增加含小目標影像的抽樣機會，合理對應此問題；不需要 test/external outcome 或 validation 個別 failure ID 參與訓練樣本選擇。

未來若另外獲准，必須先登錄抽樣分類、權重、對照組與 advancement gates；沿用相同的 pre-FUSeg 初始化 lineage、architecture、seed、imgsz、loss、optimizer、augmentation、每 epoch 抽樣張數、訓練預算與既有評估/裁切設定。僅更動抽樣分布，保留 Phase C 為固定 comparator；不能同時增加解析度、epoch、修改 margin 或 threshold。不得直接從 validation 失敗圖複製樣本到 training。

本報告沒有聲稱資料不足或模型容量不足已被證明，也不保證抽樣能改善 Precision、crop gate 或外部泛化。資料可能存在多 GT 依賴與類似影像，下一階段仍須沿用資料隔離及來源保護。

### Crop-policy 是另一個問題

YES，需要獨立研究問題，但未執行。Multi-ROI 或既有預測聯集不會自動救回「完全未被偵測」的傷口；本次 12 個相關 crop cases 同時存在未匹配 GT，不能只靠放大 crop 就認定解決。也有 0679 這種匹配成功但 pixel retention 不足的不同型態。訓練實驗與 crop-policy 實驗必須分開，不能利用「只差 1 張」救 gate。

## 11. 16 個指定問題的明確回答

1. **32 FN 的主要 pattern？** Small 28、medium 4、large 0；multi-GT 上 20 個 FN，其中 18 個 small。
2. **28 small FN 是否集中更小區間？** 是，20/28 在 <0.25%，其中 14/28 在 <0.10%。
3. **Single 與 multi recall 差？** 92.05% vs 77.78%，相差 14.28 percentage points。
4. **Multi 是否更容易 crop incomplete？** 觀察上是：12/35=34.29% vs 7/151=4.64%，不作因果判定。
5. **19 crop failures 各類型？** 全部逐張見 §6.2；12 missed-other-instance、4 no-ROI、2 localization-without-TP、1 partial coverage。
6. **15 個有 ROI 者主要型態？** 12/15 有未匹配的其他 GT bbox 在 crop 外；這是幾何分解而非已證實根因。
7. **FP 各類幾個？** Negative 1、near-GT low-IoU 15、duplicate-like 2、positive far-from-GT 18、unclear 0。
8. **TP/FP confidence 高度重疊？** 中央 IQR 不重疊，但 range 與尾端有重疊，不能完美分離；固定分箱 overlap 29.29%。
9. **Missed 幾何特徵？** bbox 通常更小、所在圖 GT 數較多；邊界距離略低但有重疊，未證實 edge truncation。
10. **No-ROI 是否主要 small？** 3/4 small、1/4 medium，樣本很少。
11. **哪種解釋最多？** FN 是 small（87.5%）；crop 是 missed-other-instance（63.16%）；FP 是 positive-other-region（50%）。分母不同不能合併。
12. **下一個單一訓練實驗？** Training-GT-only small-object-aware image sampling，CANDIDATE_ONLY。
13. **獨立 crop-policy 問題？** YES，與訓練分開，且不假設可救回未偵測目標。
14. **需要改 threshold？** 目前沒有改動依據或授權；值得另立預先登錄的研究假設，不提供新值。
15. **現在替換 App model？** NO。Development gate 仍 FAIL，沒有新增獨立部署證據。
16. **需要 multi-seed？** C1 未授權、未執行；不能把本次探索性分析當作啟動許可。

## 12. 人工複核與視覺化交付

[人工複核表](../experiments/results/isic_fuseg_colorfix_v1/phase_c1_review_sheet.csv) 涵蓋 191 張，含 GT/TP/FP/FN、size counts、crop retention/pass、no-ROI、多傷口、machine tags、human status/tags 與 notes。`human_review_status`、`human_failure_tags` 全部保持 UNKNOWN，未自動填入 annotation_error 或 clinical_error。

檢視圖共有 15 張，圖例：綠色 GT、紅色預測、青色保存的 ROI、橘色本格聚焦的 GT 或 FP。每個 grid 都有檔名、TP/FP/FN、GT 數及 crop retention；一圖多 FN/FP 時會重複顯示該影像，但聚焦不同 instance，不是新增独立樣本。

| 圖組 | 完整覆蓋 | 頁數 |
|---|---:|---:|
| Small wound misses | 28 個 FN instances | 3 |
| All FN | 32 個 FN instances | 3 |
| Multi-wound misses | 17 張 images | 2 |
| FP by taxonomy | 36 個 predictions | 3 |
| Crop failures | 19 張 images | 2 |
| Confidence histogram + ECDF | TP/FP 全部保存 confidence | 1 |
| Small detected vs missed geometry | 137 個 small GT | 1 |

全部格位的 `(sample_id, instance_id)` 與來源清單逐項核對，沒有遺漏或用重複案例替代。0456、0816、0349、0548、0192、0798 全數保留。完成 crop 全 19 張、代表性 small/FP 頁面及統計圖的版面檢查；這不是專業人員的醫學標註審核。

- [全部 crop failures，第 1 頁](../experiments/results/isic_fuseg_colorfix_v1/phase_c1_figures/all_crop_failures_01.png)
- [全部 crop failures，第 2 頁](../experiments/results/isic_fuseg_colorfix_v1/phase_c1_figures/all_crop_failures_02.png)
- [Confidence histogram / ECDF](../experiments/results/isic_fuseg_colorfix_v1/phase_c1_figures/confidence_histogram_ecdf.png)
- [Small geometry](../experiments/results/isic_fuseg_colorfix_v1/phase_c1_figures/small_detected_missed_geometry.png)

影像 grids 為本機研究檢視產物，不自動上傳 GitHub 或公開分享。

## 13. 輸出、測試與停止點

機器可讀完整分析：[phase_c1_error_analysis.json](../experiments/results/isic_fuseg_colorfix_v1/phase_c1_error_analysis.json)。

| CSV 檔案（同輸出目錄） | Rows | 單位 |
|---|---:|---|
| phase_c1_failure_taxonomy.csv | 87 | 混合型紀錄，保留 unit 欄位，非獨立病例總數 |
| phase_c1_small_wound_analysis.csv | 137 | Small GT instance |
| phase_c1_fp_analysis.csv | 36 | FP prediction |
| phase_c1_fn_analysis.csv | 32 | FN instance |
| phase_c1_crop_failure_analysis.csv | 19 | Positive crop-failure image |
| phase_c1_review_sheet.csv | 191 | Image |
| phase_c1_all_gt_geometry.csv | 241 | GT instance |
| phase_c1_small_size_bins.csv | 5 | Diagnostic size bin |
| phase_c1_crop_severity.csv | 7 | Severity bin |
| phase_c1_image_quality.csv | 191 | Image |

測試：C1 targeted 26 項；連同 Phase C execution/postflight、C0 guards、Phase A/A.5/B1 與既有 synthetic regression，共 **188 passed、4 warnings**。警告為既有 thop 的 distutils 版本比較棄用提示；圖表產生另有 Matplotlib `labels`→`tick_labels` 的相容性提醒，均非計算失敗。測試使用 synthetic/mocked 路徑，沒有載入實驗模型權重或執行新 inference。

產物驗證 PASS：計數與 Phase C 一致、27 個純衍生欄位可重建、10 個 CSV row counts 正確、15 張 PNG 可讀、全部 failure grid 覆蓋一致、191 個 human-review 狀態 UNKNOWN、403 個保護檔案未變。程式、測試、報告及新增成果均未提交或推送 GitHub。

```text
PHASE_C1_STATUS = COMPLETE
MODEL_INFERENCE_PERFORMED = false
TRAINING_PERFORMED = false
LOCKED_TEST_USED = false
CO2WOUNDS_USED = false
PRIMARY_FAILURE_MODE = VERY_SMALL_GT_MISSES_WITH_MULTI_INSTANCE_COVERAGE_OVERLAP
RECOMMENDED_NEXT_SINGLE_EXPERIMENT = TRAINING_GT_ONLY_SMALL_OBJECT_AWARE_IMAGE_SAMPLING
RECOMMENDATION_STATUS = CANDIDATE_ONLY
SEPARATE_CROP_POLICY_EXPERIMENT_NEEDED = YES
NEW_TRAINING_AUTHORIZED = NO
MULTI_SEED_AUTHORIZED = NO
APP_MODEL_REPLACEMENT_AUTHORIZED = NO
DEVELOPMENT_GATE = FAIL_UNCHANGED
```

**停止於 Phase C1。** 不自動展開下一個訓練、threshold 或 crop-policy 實驗。
