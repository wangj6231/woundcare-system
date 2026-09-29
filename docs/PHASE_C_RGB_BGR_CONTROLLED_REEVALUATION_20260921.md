# Phase C：受控 RGB/BGR 重評估結果報告

日期：2026-09-21（Asia/Taipei）
實驗：`ISIC_FUSEG_COLORFIX_V1`
執行開始：2026-09-21 14:58:05（UTC+08:00）
最終狀態：**PHASE_C_STATUS = COMPLETE**
開發門檻：**FAIL_DEVELOPMENT_GATE**

## 1. 本次回答的問題

本次固定使用歷史 ISIC→FUSeg seed-42 gate 的 checkpoint、191 張 FUSeg validation、推論設定、配對規則、mask 計算與裁切程式，只修正傳入 Ultralytics 的 NumPy 色彩通道：PIL RGB → NumPy RGB → 明確 RGB-to-BGR → contiguous uint8 BGR。

正式執行一次，191 張全部完成，沒有跳過困難樣本或選擇性重跑。三次合成黑色影像暖機沿用歷史規則，不計入 191 張 development image inference。此比較量化的是 **evaluation correction impact（修正評估輸入錯誤的影響）**，不是重新訓練帶來的模型改善，也不是獨立外部或臨床效能證據。

## 2. 結果與原始錯誤輸入之比較

**Historical value contains confirmed RGB/BGR input-contract defect.**

| 指標 | 歷史錯誤色彩輸入 | 修正色彩輸入 | 絕對差異（百分點） |
|---|---:|---:|---:|
| Precision | 45.59% | **85.31%** | +39.72 |
| Recall | 12.86% | **86.72%** | +73.86 |
| F1 | 20.06% | **86.01%** | +65.94 |
| 裁切完整率（保留 ≥95% GT 傷口像素） | 14.52% | **89.78%** | +75.27 |
| 正樣本平均 union Mask IoU | 11.41% | **76.66%** | +65.25 |
| 正樣本平均 union Dice | 13.18% | **84.43%** | +71.25 |

上述比例由完整數值計算差異後再四捨五入，不能直接用表內四捨五入數字相減取代。

| 計數／分母 | 數值 |
|---|---:|
| 全部影像 | 191 |
| 有 GT 傷口像素的正樣本 | 186 |
| 無 GT 傷口像素的負樣本 | 5 |
| GT polygon instances | 241 |
| TP | 209 |
| FP | 36 |
| FN | 32 |
| 裁切完整的正樣本 | 167 / 186 |
| 正樣本無 ROI | 4 / 186 |
| 有預測的負樣本 | 1 / 5 |

Precision = 209 / (209+36)；Recall = 209 / (209+32)；F1 = 2×209 / (2×209+36+32)。

正樣本平均裁切保留 GT 傷口像素比例為 **95.55%**，平均保留 **3,240.88 pixels/image**。平均保留比例與「每張至少保留 95%」的成功率是不同指標：前者 95.55%，後者 89.78%。所有正樣本均納入分母，4 張無 ROI 樣本的保留比例為 0。

本歷史 fixed-point gate 不支援 Mask mAP50／mAP50–95，因此本次未另跑 mAP 評估。以上 Precision、Recall、F1 亦不是七類傷口分類 Accuracy。

## 3. 尺寸分層

沿用封版程式的 **GT bounding-box area / 原圖 area**，而不是 mask 前景像素面積：Small <1%、Medium ≥1% 且 <5%、Large ≥5%。未因本次結果更動分界。

| 尺寸 | Support | TP | FN | 修正後 Recall | 歷史錯誤輸入 Recall |
|---|---:|---:|---:|---:|---:|
| Small | 137 | 109 | 28 | **79.56%** | 11.68% |
| Medium | 90 | 86 | 4 | **95.56%** | 13.33% |
| Large | 14 | 14 | 0 | **100.00%** | 21.43% |
| 合計 | 241 | 209 | 32 | **86.72%** | 12.86% |

三個尺寸層級的量測結果均改變；剩餘 32 個 FN 中，28 個（87.5%）屬 Small。這是單一 `Wound` 類別的尺寸分層，不是七類傷口或疾病類別分析。

## 4. 封版 acceptance gate

Gate 延續歷史規則：百分比四捨五入到小數點後兩位再比較。

| 條件 | 封版門檻 | 本次結果 | 判定 |
|---|---:|---:|---|
| Precision | ≥87.18% | 85.31% | **FAIL** |
| Recall | ≥84.65% | 86.72% | PASS |
| F1 | ≥85.89% | 86.01% | PASS |
| Crop complete ≥95% | ≥90.32% | 89.78% | **FAIL** |
| GPU mean latency | ≤50 ms | 41.60 ms | 數值低於門檻；可比性未驗證 |

整體為 **FAIL_DEVELOPMENT_GATE**：Precision 與 crop completeness 兩項未通過。裁切成功數為 167，該分母下至少 168 張成功才可達到既有裁切門檻。這只是描述與固定門檻的距離，不構成修改閾值或重跑的理由。

## 5. 推論速度與可比性

| 速度指標 | 本次量測 |
|---|---:|
| GPU mean | 41.60 ms/image |
| GPU median | 39.16 ms/image |
| GPU P95 | 53.49 ms/image |
| 由 mean 換算序列 FPS | 24.04 |

量測沿用原函式：3 次合成暖機、GPU batch=1、包含 predict 與 box/mask 轉到 CPU，排除讀檔、保存診斷、HTTP 與分類器。因此它不是完整 App 的端到端延遲。

目前為 NVIDIA GeForce RTX 4060 Laptop GPU、CUDA 12.4、PyTorch 2.5.1+cu124、Ultralytics 8.3.53、Python 3.12.4。NumPy 2.0.1、Pillow 10.4.0、OpenCV 4.10.0.84 亦與歷史記錄一致。

歷史訓練日誌記錄相同 GPU 型號，但歷史 gate 本身缺少完整 GPU／driver／負載執行記錄，無法嚴格證明當時量測環境完全一致。故標示 **LATENCY_COMPARABILITY = NOT_VERIFIED**、`latency_gate_passed = null`。41.60 ms 是本次有效觀察值，不據此宣告完整 latency gate PASS。

## 6. 錯誤分析與視覺查核

### 可由 GT／計數確認

- **CONFIRMED_FROM_GT：** Small 漏檢 28、Medium 漏檢 4、Large 漏檢 0。
- **CONFIRMED_FROM_GT：** 36 個 FP 中，35 個位於正樣本，1 個位於無 GT 傷口的負樣本 `fuseg__0417.png`。正樣本上的 unmatched prediction 不應一律稱為背景誤報；可能包含定位不符或重複預測。
- **CONFIRMED_FROM_GT：** 17 張含多個 GT instance 的影像仍有漏檢。
- **OBSERVED_FAILURE：** 正樣本無 ROI 共 4 張：`0456`、`0591`、`0816`、`0973`（完整檔名皆為 `fuseg__XXXX.png`）。其中 3 個 Small、1 個 Medium。
- **OBSERVED_FAILURE：** 另有 15 張正樣本雖產生 ROI，但裁切未保留 ≥95% GT 傷口像素；合計 19 張裁切不完整。

### 已檢視既有 12 格失敗案例圖

圖中綠框為 GT bbox、紅框為保留 prediction bbox、青框為 padded ROI。該圖為既有 metric 模組產生，縮圖僅供檢視，未參與評分。

| 案例 | 可觀察到的失敗 | 證據類別 |
|---|---|---|
| `0456`、`0816` | 小尺寸 GT 傷口未產生 ROI | GT 尺寸與漏檢為 CONFIRMED_FROM_GT；可見局部範圍很小為視覺觀察 |
| `0349` | 兩處 GT 中僅匹配一處；ROI 集中於足趾，另一處未涵蓋，裁切保留率 13.54% | OBSERVED_FAILURE |
| `0548` | 兩個 medium GT 中匹配一個，第二處未納入 crop；保留率 42.82% | OBSERVED_FAILURE |
| `0192` | 兩個 medium GT 中匹配一個；crop 未涵蓋另一處，保留率 64.18% | OBSERVED_FAILURE |
| `0798` | prediction 僅覆蓋 GT 傷口附近的一小部分；TP=0、FP=1、FN=1，IoU 6.87%、crop coverage 16.01% | OBSERVED_FAILURE |

低對比、背景紋理、邊界不明或邊緣截斷仍屬 **SUSPECTED_CAUSE**，目前沒有受控因果證據；本次檢視不足以認定標註噪音或臨床診斷錯誤。下一步可針對保存結果深化 error analysis，但本次不調參、不改標註、不另跑實驗。

## 7. 封版輸入與執行完整性

| 封版項目 | SHA-256 |
|---|---|
| C0 protocol | `16718f5956548e002320ebd2ade83c5ecafdf77e7c79dc1f1d2b63818b7e6df3` |
| 同一 checkpoint | `cc2955d088928dc00c90d0ba8cab0aefeb49d4cc45da172b030c150ab0dce97a` |
| 191 筆 validation manifest | `ea58c4e7d36b699d2f1e232d23fea8df9a6010674d08769e5a2ca3aca66ab069` |
| Frozen metric script | `e62ea05d8cb231cbd697d0a114e8bd81b5f46484ea18fa72535fcb2cef4dd980` |
| Corrected gate script | `b6f6aec84419ed10beb0a0a834add274a45126808fe432eda818686abce56097` |
| 本次 execution adapter | `fa1fd947726b30fc00175697a2d45236fbad52d294ac7d045a26331e024140c6` |
| FUSeg source evidence | `c91ad4b43fabd0ef02242c2c76adca3807f5f98bf61f26df44fba540bc8a69dc` |

Checkpoint 路徑固定為 `outputs/isic_fuseg_formal_seed42_20260914/runs/fuseg_finetune_formal/weights/best.pt`，沒有替換權重。

全部 **191／191** 筆 image、mask、label 各自與封版雜湊一致（共 573 個檔案校驗），載入 checkpoint 前通過；執行後再次通過。全部資料角色為 FUSeg／`DEVELOPMENT_VALIDATION`／`val`。

Source evidence 沿用官方 repository：<https://github.com/uwm-bigdata/wound-segmentation>，commit `42a272dfe0679f20675e826385925cb7562934b6`；既有紀錄為 CC BY NC，沒有新增 license version 宣稱。本次用途為非商業離線學術研究。

推論設定固定：primary confidence 0.10、prediction floor 0.01、NMS IoU 0.70、bbox matching IoU 0.50、imgsz 768、batch 1、half=false、retina_masks=false、max_det=300、augment=false、class 0 Wound。評估 canvas 512×512、mask nearest-neighbor resize、15% per-side crop margin 與 metric implementation 均沿用封版版本。

一次性輸出目錄於開始前不存在，建立 immutable `execution.lock` 後才載入 checkpoint。該 lock 的 RUNNING 是開始時狀態；最終完成狀態以 `execution_completion.json` 為準。已存在的輸出目錄會阻止再次執行，沒有 resume/retry 流程。

57 個受保護 artifacts 執行前後 SHA-256 全部一致，包含歷史 gate result/protocol/predictions、候選與既有 D-Seg-03 checkpoint、B1 封版 artifacts、舊 Tables、封版協定與相關程式。CO2Wounds 歷史結果 artifact 依本階段要求只比對檔案雜湊，沒有讀取 CO2Wounds 資料集或重新評估。

## 8. 結果核查及一項檢查器修正

正式保存的 191 筆紀錄均包含 sample/image hash、GT instance 數、prediction 數、TP/FP/FN、GT size group、confidence、bbox、mask NPZ reference、IoU/Dice、crop、GT/retained pixels、crop success、no ROI 與 latency。

第一次額外的 mask 存檔檢查發現 3 個零 instance 陣列 dtype 為 float64。原因是**封版歷史程式**使用 `np.asarray([]).reshape(-1,512,512)`，沒有 instance 時會產生這種空陣列。初版檢查器即使對零元素也強制要求 boolean，因而誤拒。受影響的 `0128`、`0483`、`0869` 均沒有任何 mask 元素，非模型或 metric defect。

原執行程式、協定、結果、mask 檔均保留原樣。新增獨立 postflight verifier，只接受形狀精確為 `(0,512,512)` 的零 instance 陣列；非空 mask 仍須 boolean。4 項專用合成測試驗證此規則。

獨立 postflight 最終結果為 **PASS**，並完成：

- 191 份 mask 壓縮檔的 SHA、形狀、confidence、bbox、retained indices 核對；
- 從保存的 bbox/confidence 與原 GT 重建相同 matching，TP/FP/FN/pairs 完全一致；
- 從保存 mask 與原 GT mask 重算相同 pixel metrics、crop retained pixels，全數一致；
- 從保存計數重算 aggregate metrics 與 gate decision，全數一致；
- `209 + 32 = 241`；`137 + 90 + 14 = 241`；全部必要比例在合法範圍；
- 重核 573 個 validation file hashes 與 57 個受保護 artifact hashes；
- 本核查程序模型載入數 0、推論數 0。

完整相關合成／回歸測試：**162 passed、0 failed、0 skipped、4 warnings**。Warnings 為既有 `thop` 的 distutils deprecation，不影響測試通過。

## 9. 交付物

主目錄：`experiments/results/isic_fuseg_colorfix_v1/`

| 檔案 | 用途 |
|---|---|
| `result.json` | 全部正式指標、gate、來源雜湊與安全狀態 |
| `per_image_predictions.json` | 191 筆完整預測與評估紀錄 |
| `per_image/000.json` 至 `190.json` | 逐張保存紀錄，與合併檔一致 |
| `masks/000.npz` 至 `190.npz` | 原次推論保存的 masks／boxes／confidences 與索引 |
| `execution_manifest.json` | runtime file hashes、環境與受保護檔案前置快照 |
| `execution.lock` | 不可覆寫的執行開始記錄 |
| `execution_completion.json` | 完成狀態與 result/predictions SHA |
| `historical_comparison.md` | 歷史錯誤輸入與 corrected 比較 |
| `historical_integrity.json` | 57 個受保護檔案前後一致證據 |
| `postflight_verification.json` | 獨立保存結果／matching／pixels 重算核查與檢查器說明 |
| `error_analysis.json` | 分類後的漏檢／無 ROI／不完整 crop 紀錄 |
| `error_cases.json`、`worst_crop_cases.png` | 沿用歷史輸出方式的 12 個最差 crop 案例 |

這些含原影像的視覺資料保留於本機，未發布或上傳。

## 10. 最終研究決策

輸入色彩契約修正後的量測結果大幅改變，說明原本 defective evaluation 不能代表候選模型在正確输入下的效能。然而固定 gate 仍有 Precision 與 crop completeness 未通過，因此本輪結論為完成受控重評估、保留失敗判定，下一個研究討論應聚焦 small-wound misses、正樣本 FP 與多傷口 crop 不完整的錯誤分析。

原 App 模型維持既有狀態。本輪沒有訓練、調參、metric 修改、locked-test 使用、CO2Wounds 資料集使用、multi-seed、bootstrap/CI/p-value，亦未自動進入下一階段。

```text
PHASE_C_STATUS = COMPLETE
DEVELOPMENT_GATE = FAIL
LATENCY_COMPARABILITY = NOT_VERIFIED
READY_FOR_ERROR_ANALYSIS = YES
READY_FOR_NEXT_RESEARCH_DECISION = YES
MULTI_SEED_AUTHORIZED = NO
APP_MODEL_REPLACEMENT_AUTHORIZED = NO
EXTERNAL_TEST_AUTHORIZED = NO
development_images_inferred = 191
test_images_used = 0
CO2Wounds_used = false
```

**STOP AFTER PHASE C。**
