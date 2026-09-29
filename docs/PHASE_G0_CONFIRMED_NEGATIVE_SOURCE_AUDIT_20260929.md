# Phase G0 — 可信背景負樣本來源稽核

日期：2026-09-29。用途：非商業、離線學術／畢業研究。不是臨床部署許可。

## 結論

`PHASE_G0_STATUS=COMPLETE`；`NEGATIVE_SOURCE_FEASIBILITY=PASS`。
771 張 frozen training 重新讀取 labels：753 NONEMPTY、18 EMPTY、0 MISSING、0 INVALID。
18 張原始 mask 全零且 SHA256 與來源紀錄一致，通過 native negative semantics 與角色隔離，形成 18 張／18 unique exact contents 的名冊。
這是 **既有訓練負樣本的來源資格確認，不是新增資料、不是新模型成果**。F1.2 research gate 仍是 FAIL。

## 一、證據與語義界線

本機來源 PDF：`official_detection_sources_20260812/fuseg/repository/data/Foot Ulcer Segmentation Challenge/FootUlcerSegmentationChallenge2021.pdf`，SHA256 `f9a44fc14bc7589d03ce5c17307ad97e9dde5fcc0e2c596c8b86010a6d9547aa`。
第 9 頁明確說明少數已癒合案例沒有 wound annotation；第 11 頁把 healed/non-wound 案例的任何 wound prediction 計作 FP，正確標註／預測為全零。
第 9–10 頁記載人工標註經 wound-care specialists/nurses 審核，困難案例諮詢醫師。仍承認 annotation error 可能性，不能宣稱絕對完美 ground truth。
本次不是以「沒有 polygon」或外觀猜測無傷口：每張都核對 frozen image/label → source CSV → 原始 training image SHA256 → 原始 mask SHA256 → 全零 512×512 mask。
原始 mask 為 RGB 相同三通道；以任意非零值判前景，不因低像素值被誤當零。
只有 **negative for current FUSeg wound-localization target**，不是 clinically healthy，也不是七類診斷排除。

上游 repository commit：`42a272dfe0679f20675e826385925cb7562934b6`。Dataset release version 未標明。
使用既有 `experiments/review_v2/evidence/FUSeg_noncommercial_research_20260914.md` 來源准入；PDF 記 CC BY NC（版本未註明），本次不擴大為商業、病患照護、公開影像／權重再散布授權。
PDF 原始計畫的 610/200/200 與 README 後續增補不等於目前篩選後 inventory；本次數量完全來自 frozen 771 training，不回填舊計畫數。

候選 ID：fuseg__0092.png, fuseg__0131.png, fuseg__0223.png, fuseg__0281.png, fuseg__0361.png, fuseg__0457.png, fuseg__0502.png, fuseg__0564.png, fuseg__0600.png, fuseg__0701.png, fuseg__0725.png, fuseg__0736.png, fuseg__0778.png, fuseg__0832.png, fuseg__0899.png, fuseg__0912.png, fuseg__0931.png, fuseg__0950.png

## 二、資料隔離、重複與人工確認合約

18 張候選均有來源全零標註語義，不需要本次新增專家判讀才能取得 native-negative 資格；沒有虛構 reviewer。
Exact SHA256：18 raw files、18 unique contents、0 duplicate groups。與 191 validation hashes 及既存 200 筆 FUSeg locked-test hashes 重疊均 0。
僅比對既存 hash metadata，沒有打開 locked-test 圖片。未宣稱無近重複，也未宣稱病人層級隔離；patient/case/session identity = UNKNOWN。
分類 48 張不在本次 FUSeg locked-hash 比對範圍；未讀其資料，也不宣稱已完成 SHA256 跨任務排除。

未來人工確認必須包含 sample_id、reviewer_role、review_result、review_date、review_protocol_version。
review_result 僅允許 CONFIRMED_NO_TARGET_WOUND / TARGET_WOUND_PRESENT / UNCERTAIN；只有前者可確認資格，UNCERTAIN 一律排除。
MISSING_LABEL、INVALID_LABEL、EMPTY_LABEL_ONLY、positive image 非 GT 區域、validation FP、test/external 均不得混入 verified 名冊。
數量少不構成補造或擴增資料的理由；G0 沒有指定最低數量。

## 三、Stock pipeline 已有的負樣本監督

本次僅讀 source，未 import torch/ultralytics 或載入 checkpoint。architecture 已釘選 source hashes 全數核對；`data/utils.py` 另記本次 hash，其不在 architecture 原始 pin map，不能冒稱歷史獨立 pin。

- `data/utils.py:152–156`：empty/missing 皆產生零 target tensor，但保留不同 counter。工程可載入不等於研究上語義可信。
- `data/dataset.py:232–249` 與 `data/augment.py` Format：image 仍進 batch；空 cls/bbox/batch_idx，mask zero tensor（overlap-mask 模式為一張零 mask）。
- `utils/tal.py:64–71`：無 GT 時 foreground/target scores 為零。
- `utils/loss.py:318–350`：所有 prediction 的 cls BCE 對 zero target scores 提供背景抑制；bbox、DFL、實際 mask loss 只作用在 foreground assignments。
- `utils/loss.py:425–444`：mixed batch 中沒有 foreground 的 image 不提供實際 positive mask loss；全空情形僅保留 zero-valued mask graph terms。**不是對全圖背景執行 dense segmentation BCE 的新監督。**

`f1_v2_runner.py` 在 preprocess_batch 記錄 consumed image identities。G0 對照 300 frozen orders，C4 / H4-R 每個 epoch 771 anchors 完全一致；每張負樣本每 epoch 1 次，300 epochs 各 300 次。
因此：**Existing empty-label anchors were already part of the baseline training distribution.**
Anchor count 不是 Mosaic donor 次數、增強後仍無傷口的 batch image 次數、也不是成功 optimizer updates。

## 四、歷史 exposure 描述性核對

| Arm | Seed | 300 epochs empty-anchor draws | 每 epoch 平均 |
|---|---:|---:|---:|
| F_C4 | 42 | 5,400 | 18.000 |
| F_H4_R | 42 | 5,400 | 18.000 |
| D2_C | 42 | 5,400 | 18.000 |
| D2_S | 42 | 3,739 | 12.463 |
| D2_C | 123 | 5,400 | 18.000 |
| D2_S | 123 | 3,837 | 12.790 |
| D2_C | 3407 | 5,400 | 18.000 |
| D2_S | 3407 | 3,823 | 12.743 |
| D2_C | 2026 | 5,400 | 18.000 |
| D2_S | 2026 | 3,795 | 12.650 |
| D2_C | 999 | 5,400 | 18.000 |
| D2_S | 999 | 3,802 | 12.673 |

以上逐行重算 saved consumed-anchor telemetry，再核對既存 D2 summary；不是只搬運舊統計。
D2 S 的空標註 primary-anchor exposure 比均勻 C 少，但這與其他尺寸 sampling 一起變動，不能據此推論 FP 或 Recall 的因果機制，也不據此設定未來權重。
`existing_negative_exposure.csv` 保存各 arm × seed × epoch × candidate 的逐筆次數，包含零次。

## 五、25 個必要回答

| # | 問題 | 回答 |
|---:|---|---|
| 1 | 771 training 中空標註數 | 18 |
| 2 | Missing label 數 | 0 |
| 3 | Empty label 是否具有 dataset-native negative semantics | 有，但必須是來源可追溯的原始全零 mask；單憑空 YOLO label 不足。 |
| 4 | 證據文件 | FUSeg challenge PDF 第 9、10、11 頁；原始 train/labels masks；fuseg_only_manifest_20260821.csv；frozen training manifest。 |
| 5 | 需專家補判 candidate | 無；本次未進行新的專家判讀。 |
| 6 | 合格 raw negative files | 18 |
| 7 | Unique exact contents | 18 |
| 8 | Duplicate negative groups | 0 |
| 9 | Validation exact overlap | 0（比對 191 張 frozen image SHA256，不讀像素） |
| 10 | Locked-test exact overlap | 0（既有 FUSeg official test 的 200 筆 SHA256；未對 classification 48 張作額外保證） |
| 11 | Locked-test pixels 是否讀取 | 否。只讀已存在的 test hash CSV。 |
| 12 | Validation pixels 是否讀取 | 否。不開啟原始 validation image/mask；既有研究圖檔僅 opaque-byte 雜湊，不解碼、不檢視。 |
| 13 | 是否使用 validation FP 作負樣本 | 否。F1.3 案例僅保留為分析參考，不作 selector。 |
| 14 | Positive image 非 GT 區域是否進 pool | 否。沒有切 crop、合成或外觀篩選。 |
| 15 | C4 是否已訓練過空標註 anchors | 是。逐一核對完整 300 epochs consumed-anchor telemetry 與 frozen order。 |
| 16 | 既有 exposure 每 epoch / 全程 | 每張每 epoch 1 次、全程 300 次；18 張合計每 epoch 18 次、全程 5,400 次。 |
| 17 | Patient/case identity | UNKNOWN。18 unique contents 不代表 18 獨立病人。 |
| 18 | 來源是否完整 | 對本次 18 張 mask/image 的 native-source hash chain 完整；release version 與病人 ID 未提供，不予補造。 |
| 19 | Feasibility | PASS |
| 20 | G1 preregistration 資格 | 可提出下一階段預登錄；本次沒有建立 G1 或授權訓練。 |
| 21 | 是否訓練 | 否。 |
| 22 | 是否 inference / model loading | 否／否。 |
| 23 | 是否改 threshold/NMS/match | 否。 |
| 24 | 是否使用 external/CO2 | 否。 |
| 25 | 是否替換 App | 否。 |

## 六、下一階段界線與交付

可討論 G1 FALSE_POSITIVE_CONTROL preregistration，但本次止於 G0。未決定 oversampling、loss、budget 或新訓練資料配置。
未來 baseline 回到 train imgsz=768，不能用 1024+negatives vs768 同時改兩個因素；不得挽救性合併失敗的 scale intervention。
未來 safety priorities 同時保留 Precision/FP、Very-small Recall、Small Recall、Crop completeness，不能以 suppress detections 換取表面 Precision。
18 個舊 negative 不能稱為新增監督；是否改變 exposure 真有幫助，尚未有此實驗證據。

輸出目錄：`experiments/results/g0_negative_source_audit/`。六份 CSV 保留 flat-table typed fields、明確 NA 與來源追溯，不使用色彩代替分類；另有來源證據、摘要與 integrity JSON。
完整性：依 F1.3 保存的 source/output digests 重核 F1.2/F1.3 research artifacts、預測與 checkpoints，前後一致。原始 validation/test raster 不重新讀取；研究圖檔只當不透明 bytes 做 hash，不開啟像素。
沒有修改既有 pipeline、App、來源資料或模型；本次僅新增 G0 稽核程式、測試、報告及輸出。

```ini
PHASE_G0_STATUS = COMPLETE
NEGATIVE_SOURCE_FEASIBILITY = PASS
VERIFIED_NEGATIVE_COUNT = 18
VERIFIED_NEGATIVE_UNIQUE_CONTENT_COUNT = 18
NEEDS_EXPERT_ADJUDICATION = NO
TRAINING = false
MODEL_LOADED = false
NEW_INFERENCE = false
VALIDATION_HARD_NEGATIVE_MINING = false
LOCKED_TEST_USED = false
CO2Wounds_USED = false
EXTERNAL_TEST_USED = false
NEW_TRAINING_AUTHORIZED = NO
MULTI_SEED_AUTHORIZED = NO
APP_MODEL_REPLACEMENT_AUTHORIZED = NO
STOP AFTER G0
```
