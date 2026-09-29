# Phase E0：Very-small failure mechanism audit

日期：2026-09-25。範圍僅限已封存的 FUSeg development validation 與 D2 訓練紀錄。

## 狀態

```ini
PHASE_E0_STATUS = COMPLETE
PRIMARY_VERY_SMALL_FAILURE_PATTERN = PERSISTENT_MISSES_WITH_MULTI_GT_OVERLAP_AND_WEAK_LOCALIZATION
RECOMMENDED_NEXT_SINGLE_INTERVENTION = PATCH_BASED_TRAINING
NEW_TRAINING_AUTHORIZED = NO
APP_MODEL_REPLACEMENT_AUTHORIZED = NO
EXTERNAL_TEST_AUTHORIZED = NO
```

結論：V2 已增加 very-small anchor exposure 約 42%，但固定難例依然存在，且 paired gains/losses 不一致。可支持「單純多抽相同整圖不足以可靠解決目前難例」，不能支持已找出因果機制或已證明下一個 intervention 有效。

## 研究問題與邊界

本研究僅追查：增加 very-small / small training exposure 後，為何 <0.25% GT 的偵測改善未能跨 seed 穩定出現。D2 的 sampling V2 保留為完成但未通過確認的研究結果，不修改權重、不製作 V3。

使用 seed 42、123、3407、2026、999 的 C/S 共 10 組已保存 predictions。共同母體是 191 張 FUSeg development validation、241 個 GT；本稽核涵蓋全部 49 個 very-small GT，其中 <0.10% 為 27 個，0.10–<0.25% 為 22 個。GT 是固定 polygon label 順序中的實例，不宣稱每個實例是獨立病人或經臨床確認的不同病灶。

沒有訓練、fine-tuning、新 inference、threshold sweep、resolution experiment、model crop change、外部測試或 App 替換。原 development images 只用來計算描述性影像特徵及展示原 GT。

## 分類與 matching 規則

分析前先保存 `analysis_protocol.json`。`sample_id + GT_instance_id` 是唯一鍵，GT_instance_id 為原標註的零起算索引。各模型 GT 框、順序與 validation manifest 必須一致。保存的 `pairs` 為偵測結果的唯一依據；以原 frozen matching 程式核對一致性，不改 confidence=0.10、NMS IoU=0.70、matching IoU=0.50，也不產生新模型輸出。

互斥分類依下列優先順序執行：

| 分類 | 固定規則 |
|---|---|
| CONSISTENTLY_DETECTED | 10/10 模型皆偵測到 |
| PERSISTENTLY_MISSED | 至多 1/10 模型偵測到，即至少 9/10 漏檢 |
| SAMPLING_RESPONSIVE | 非以上兩類，S 的偵測次數 > C |
| SAMPLING_HARMED | 非以上兩類，S 的偵測次數 < C |
| SEED_UNSTABLE | 其餘 C/S 總次數相等、但不是全對或幾乎全錯 |

另列嚴格 10/10 漏檢、完整 0–10 次偵測分布、逐 seed paired gain/loss，以及同一 GT 同時有 gain/loss 的旗標。分類優先順序會使「C=0、S=1」仍屬 PERSISTENTLY_MISSED，因此「所有 net improvement」「只有 S 偵測到」另行統計，不混用分類分母。SAMPLING_RESPONSIVE / HARMED 是計數描述，不是因果結論。

## 幾何及影像特徵定義

- width、height、area：原 512×512 development 影像上的 GT bbox 像素尺度；area ratio 為 bbox area / 512²，不是 mask area。
- aspect ratio：width / height。border distance 為 bbox 至最近影像邊界的距離 / 512。
- single/multi-GT：同一影像的固定 GT 數量；同圖多 GT 與重複模型觀測不獨立。
- brightness：全圖 RGB 轉 OpenCV grayscale 後的平均值，0–255。
- contrast：同一 grayscale 的 population SD。
- sharpness：全圖 `Laplacian(CV_64F, ksize=1)` 變異數，不直接當作臨床失焦判斷。
- local contrast：GT bbox 內平均灰階與固定外擴 8 像素矩形 ring 的平均灰階差絕對值；ring 排除影像上所有 GT bboxes。框用 floor 左上、ceil 右下，邊界截於原圖。
- local CNR：上述差值 / ring 灰階 SD。ring 不存在或 SD=0 時保留缺值，不補成 0。

Local contrast 是 bbox-region proxy。沒有可驗證的 per-instance tissue mask，因此不假稱是真正傷口組織與健康組織的對比。全圖特徵在同圖多 GT 會重複；特徵分組統計只是描述，不是獨立樣本檢定。

## 保存預測的漏檢結構

只在原 frozen confidence 與 IoU 定義上，逐一檢查已保存的框：

1. DETECTED：原 pairs 已配對。
2. MATCH_COMPETITION：未配對，但有保留框與本 GT IoU≥0.50；可能受一對一 matching 配對其他 GT 影響。
3. BELOW_FROZEN_CONFIDENCE：前述不成立，且已保存 prediction-floor 輸出中有 confidence 位於 [0.01,0.10)、IoU≥0.50 的框。
4. RETAINED_LOCALIZATION_BELOW_IOU：前述不成立，保留框有空間交集，但最大 IoU<0.50。
5. NO_RETAINED_OVERLAP：以上均不成立。

這些診斷不改原 TP/FN，不等於調低 confidence 後的表現。floor 輸出仍是 post-NMS，不能觀察 pre-NMS 或低於 floor 的 proposal；也不能據此斷言漏檢一定由 NMS、loss、標註噪音或解析度造成。

## Training exposure 與統計解讀

從實際消耗的 `anchors.jsonl` 重建 exposure，並核對原 training completion / telemetry。分開列出「含 very-small GT 的 anchor 抽取次數」「very-small GT-instance exposure」「small anchor exposure」「unique image coverage」「全 run 重複抽取」「同 epoch 重複抽取」；它們不是同一個單位。

Anchor exposure 不包含 mosaic companion 的完整像素曝光，也不保證小病灶經增強後仍可見。跨 seed 比較 exposure 差值與 validation very-small ΔTP，只列五組原始數據及描述性相關；常數變數的相關保留 undefined。沒有 p-value、CI、因果或 patient-level 泛化主張。

「未觀察到穩定收益」不等於已證明數學上的 diminishing-return 曲線：D2 只有 uniform 與一組固定 sampling weights，沒有多劑量、dose-response 或不同 loss/resolution 的實驗。

## 分析結果與七項回答

### 1. 有多少 very-small GT 幾乎都 miss？

49 個 GT 位於 41 張 development images。18/49（36.73%）符合至少 9/10 次漏檢，其中 16/49（32.65%）為 10/10 次完全漏檢，另外 2 個只各偵測到 1 次。22/49（44.90%）則是所有 10 個模型都偵測到。主要結構是固定難例與固定易例並存，而不是所有 GT 都因 seed 隨機波動。

| 互斥分類 | 全部 49 GT | <0.10%（27） | 0.10–<0.25%（22） |
|---|---:|---:|---:|
| PERSISTENTLY_MISSED | 18 | 12 | 6 |
| SAMPLING_RESPONSIVE | 2 | 2 | 0 |
| SAMPLING_HARMED | 3 | 1 | 2 |
| SEED_UNSTABLE | 4 | 4 | 0 |
| CONSISTENTLY_DETECTED | 22 | 8 | 14 |

完整 detected-count histogram（0 至 10 次）為：16、2、2、1、1、0、1、2、0、2、22。不能將「非 SEED_UNSTABLE」理解為完全沒有跨 seed 波動：RESPONSIVE/HARMED 也可能混合 gain/loss；另有 3 個 GT 同時出現 paired gain 與 loss，CSV 有獨立旗標。

### 2. 有多少只在 Experimental 中改善？

按偵測次數淨差計算，3 個 GT 的 S>C；其中 2 個歸入 SAMPLING_RESPONSIVE，另 1 個因 C=0、S=1 而依優先規則仍屬 PERSISTENTLY_MISSED。

若「只在 Experimental」嚴格指 C 五次都未偵測、S 至少偵測一次，只有上述 1 個 GT，而且僅成功 1/5 次，不能稱為穩定救回。

### 3. 有多少反而退化？

4 個 GT 的 S<C；其中 3 個歸入 SAMPLING_HARMED，另 1 個是 C=1、S=0 的 PERSISTENTLY_MISSED。所有淨改善或淨退化的 GT 都在 multi-GT 影像中。這是描述性共現，不能據此斷言 multi-GT 造成 sampling 退化。

### 4. 兩個 very-small bins 是否具有不同 failure structure？

有不同的觀察結構：

- <0.10%：12/27（44.44%）是 persistent misses，而且這 12 個都是 10/10 完全漏檢；只有 8/27 全部模型皆命中。另有 seed 不穩定及 gain/loss 抵消，mean ΔRecall +0.74 pp 不代表固定難例已被解決。
- 0.10–<0.25%：14/22（63.64%）全部模型皆命中；6/22（27.27%）persistent misses，其中 4 個完全漏檢。淨改善 1 個、淨退化 3 個，跨 seed 合計少 4 次命中，即 mean ΔTP −0.8、mean ΔRecall −3.64 pp。

因此 D2 的 very-small 平均下降主要由後一 bin 的少數 GT 淨退化構成，不能推論 sampling 讓全部微小 GT 普遍變差。

### 5. Multi-GT 是否仍是重要重疊因素？

| 場景 | Very-small GT | 影像數 | Persistent misses | Model×GT 漏檢觀測 | 漏檢比例 |
|---|---:|---:|---:|---:|---:|
| Single-GT | 24 | 24 | 7 | 76/240 | 31.67% |
| Multi-GT | 25 | 17 | 11 | 143/250 | 57.20% |

Multi-GT 佔 very-small GT 的 51.02%，卻佔漏檢觀測的 143/219（65.30%）；persistent misses 的 11/18（61.11%）也在 multi-GT。這是重要的共現因素，但不是唯一因素：single-GT 仍有 7 個始終漏檢。490 是同一批 49 GT 在 10 個模型中的重複觀測，不能視為獨立樣本數。

### 保存框顯示什麼？

在 219 個漏檢觀測中：

| 保存輸出的診斷 | 次數 | 漏檢觀測比例 |
|---|---:|---:|
| NO_RETAINED_OVERLAP | 121 | 55.25% |
| RETAINED_LOCALIZATION_BELOW_IOU | 67 | 30.59% |
| BELOW_FROZEN_CONFIDENCE | 31 | 14.16% |
| MATCH_COMPETITION | 0 | 0.00% |

多數漏檢並沒有可在既有 operating point 上命中的附近框；其次為有交集但定位不足。保存 floor 中的低信心候選只覆蓋一部分漏檢，因此不能簡化成「confidence 太高」。此結論不等於已驗證任何較低 threshold 的 Precision/Recall，亦不排除 post-NMS 之前未保存的訊息。

### 幾何與影像特徵

以下均為逐 GT 描述性中位數，不是患者層級的估計或顯著性比較。

| 特徵 | Persistent misses（18 GT） | Consistently detected（22 GT） |
|---|---:|---:|
| Bbox width（原圖 px） | 13.00 | 17.50 |
| Bbox height（原圖 px） | 17.00 | 19.00 |
| Bbox area / image area（%） | 0.0723 | 0.1324 |
| Aspect ratio | 0.972 | 0.893 |
| 最近 border distance / 512 | 0.296 | 0.313 |
| 全圖 brightness（0–255） | 104.93 | 92.33 |
| 全圖 contrast SD（0–255） | 82.96 | 77.55 |
| 全圖 Laplacian variance | 115.74 | 93.87 |
| Local bbox/background 灰階均值差 | 30.91 | 34.23 |
| Local CNR | 1.108 | 2.201 |

Persistent 組 bbox 較小、local CNR 較低，是值得追查的描述性差異；但全圖亮度、對比、sharpness 並未呈现一致的「更暗、更糊」方向，不能把所有難例歸因於失焦或曝光。Border distance 的中位數也沒有顯示都集中在邊緣，不能逕自認定邊界截斷為主因。全部特徵及分位數見 JSON；沒有人工新增 annotation-error 或臨床診斷標籤。

兩個 bins 的 bbox width/height 中位數分別為 12/14 px 與 19/24 px；local CNR 中位數分別為 1.625 與 1.736。分箱本就依尺寸定義，不能把箱間幾何差異當作額外獨立證據。

### 6. Sampling exposure 是否已顯示 diminishing return？

就目前固定 V2 而言，增加 exposure 沒有轉成一致 very-small detection 收益；平均 ΔTP 為 −0.6，不能再以「very-small 被抽到的次數不足」作為唯一解釋。

| Seed | C very-small anchor draws | S very-small anchor draws | C/S very-small GT exposure | Validation very-small TP：C→S | ΔTP |
|---|---:|---:|---:|---:|---:|
| 42 | 48,300 | 68,794 | 54,600 / 77,657 | 27→27 | 0 |
| 123 | 48,300 | 68,874 | 54,600 / 77,691 | 27→28 | +1 |
| 3407 | 48,300 | 68,631 | 54,600 / 77,467 | 28→24 | −4 |
| 2026 | 48,300 | 68,844 | 54,600 / 77,966 | 26→28 | +2 |
| 999 | 48,300 | 68,959 | 54,600 / 78,010 | 29→27 | −2 |

五組 very-small anchor 曝光均增加約 42.1%–42.8%，very-small GT exposure 增加約 41.9%–42.9%，但 validation ΔTP 為 0、+1、−4、+2、−2，沒有一致改善方向。

所有 C/S 都曾覆盖 771 張 unique training images，其中含 very-small 的 unique images 都是 161。每 run 均為 231,300 anchor draws，因此以 total draws−unique 定義的 repeat exposure 都是 230,529；這不代表抽樣行為相同。C 每 epoch 不重複，S 的同 epoch 重複總次數介於 88,071–88,226，顯示的是更集中反覆取樣，而非新增病例多樣性。C small-anchor draws 為 138,600，S 為 164,923–165,414；GT-instance exposure 與 anchor exposure 的差異完整保留於 CSV。

五個 paired seed 的描述性 Pearson r：Δ very-small anchor draws 對 validation ΔTP 為 0.496；Δ very-small GT exposure 為 0.462；Δ small anchor draws 為 −0.870。Unique-image 與全 run repeat 差值均為常數，相關未定義。這些係數來自同一批資料的五次隨機化，exposure 增量的範圍又很窄，不能視作穩定 dose-response、拿來調 weights，或宣稱因果。

Seed 123 與 2026 的 C/S AMP applied updates 各差 1 次，已保存而未補跑；其餘三組相等。沒有將不同實際更新次數隱藏為完全相等預算。

嚴格而言尚未證明 diminishing-return 曲線。只有 C 與單一 S 權重設定，且 anchor 次數不等於增强後有效傷口像素量。n=5 的 seed-level 描述性相關不能排除 seed、augmentation、AMP skip、影像組成與 model selection 的共同變化，不能解讀成 exposure 導致成績下降。

### 7. 唯一建議的下一個 intervention

`RECOMMENDED_NEXT_SINGLE_INTERVENTION = PATCH_BASED_TRAINING`

此為未執行的研究假說，不是已證實的改善方法。理由是：固定難例占比高、bbox 較小，增加相同整張影像的 sampling exposure 並未可靠救回；保存輸出主要為沒有重疊候選或定位不足，且 multi-GT 共現明顯。優先值得研究的單一因素是訓練時的空間呈現單位，讓局部微小目標在訓練輸入中占有較大範圍，而非繼續提高被抽取次數。

若另行授權，須另訂 paired protocol，只用 training GT 定義 patch 規則與保留條件，不得用本次 validation failure IDs 選訓練 patch；其餘初始化、更新預算、loss、sampling 權重、threshold 與既定 development 評估均須明確固定。不連帶啟用 validation tiling、higher-resolution、loss change 或 crop-policy 實驗。Patch 也可能損失上下文、增加 false positives，不能預告效果。

本次不優先選 LOSS_CHANGE：保存輸出沒有 loss 梯度或各項 loss 對錯誤的辨識證據。不優先選 HIGHER_RESOLUTION_TRAINING：目前 512×512 來源已在 768 輸入下訓練，更高輸入只改變表示尺度，不能從原圖創造新細節。這些是選擇單一待驗假說的理由，不是其他方法無效的證明。

`PRIMARY_VERY_SMALL_FAILURE_PATTERN = PERSISTENT_MISSES_WITH_MULTI_GT_OVERLAP_AND_WEAK_LOCALIZATION`

這個名稱描述保存結果中的結構，不宣稱已識別生物學或網路內部的因果機制。標註噪音、邊界歧義、原始細節不足及特徵表示等原因仍未被獨立驗證。

## 產物與重現

輸出目錄：`experiments/results/very_small_failure_audit/`。

- `persistent_failure_matrix.csv`：49 個 GT、10 組 detected flags、paired gain/loss、互斥分類及幾何欄位。
- `seed_stability.csv`：每 seed、每 very-small bin 的 C/S TP、paired gain/loss、ΔRecall。
- `geometry_analysis.csv`：逐 GT 幾何與影像特徵，缺值明確保留。
- `training_exposure.csv`：五組配對的 exposure、unique/repeat 與 validation 結果。
- `saved_miss_diagnostics.csv`：490 個 model×GT 觀測；不是 490 個獨立樣本。
- `failure_visualizations/`：全部 49 GT 的規則排序圖與 detection matrix，沒有人工挑樣。
- `source_snapshot_before.json`、`integrity.json`：輸入雜湊、歷史檔案不變證據與表格／圖片覆蓋檢查。
- `very_small_failure_summary.json`：完成後的最終摘要與授權邊界。

沒有變更 Phase B1/C/C1/D0/D0.1/D1/D2 的原始結果、程序、設定、資料與模型檔案。所有後續訓練建議必須另外預登錄及授權。

## 完成核對

- 10 組保存模型結果皆完成 checkpoint、telemetry、prediction/mask、GT 與 matching 核對；49 GT、490 model×GT 觀測與 D2 的逐 seed very-small TP 完全對上。
- 本次輸入快照涵蓋 4,318 個檔案，分析前後一致；另通過既有 D2 與歷史封版保護檢查。
- 335 項 synthetic regression tests 通過，另有 4 個既有第三方棄用警告；不是臨床有效性證明。
- CSV 採一列一實例／觀測、單位明示、保留原 detected flags 與衍生欄位；未覆寫來源資料。所有 49 個 GT 都列入圖庫，圖庫僅修正排版間距，沒有改分類或挑案例。
- 禁止載入模型套件的稽核 guard 通過；new inference=0、training=0、test_images_used=0、CO2Wounds_used=false。
- 完成後停止。沒有啟動 PATCH_BASED_TRAINING 或任何下一階段訓練。
