# WoundCare Phase C0 — Controlled RGB/BGR Reevaluation Preflight Report

**日期：** 2026-09-21
**實驗 ID：** `ISIC_FUSEG_COLORFIX_V1`
**階段結論：** **PASS — FROZEN BEFORE EXECUTION**
**本階段實際執行：** 僅前置稽核、協定封版與自動化測試；未載入模型、未讀取開發影像像素、未推論、未訓練。

## 1. 研究問題與本階段邊界

先前 `ISIC → FUSeg` seed-42 development gate 將 PIL 影像轉為 NumPy RGB 後，直接傳入預期 NumPy BGR 的 Ultralytics 推論介面。這使候選模型與基準流程的輸入色彩語意不一致，因此先前比較結果可保留為歷史紀錄，但不能作為公平的模型能力比較。

Phase C0 的唯一目的，是在任何重評估發生前，將下列條件凍結：

1. 同一個 checkpoint；
2. 同一批 191 張 FUSeg validation；
3. 同一 confidence、NMS、bbox matching、resize、mask 與 crop 規則；
4. 同一套 metric implementation；
5. 唯一允許的語意變更為 `PIL RGB → contiguous NumPy BGR`；
6. 後續執行前必須驗證 protocol SHA-256，且預定輸出資料夾必須不存在。

本階段禁止訓練、fine-tuning、threshold tuning、multi-seed、Blind Test、CO2Wounds 重測與 App 權重替換。

## 2. 歷史結果之處置

歷史結果未刪除、未覆寫，也未重新命名為有效比較結果：

| 指標 | 歷史值 |
|---|---:|
| Precision | 45.59% |
| Recall | 12.86% |
| F1 | 20.06% |
| Crop complete ≥95% | 14.52% |
| Gate | `FAIL_DEVELOPMENT_GATE` |

正式解讀為：**歷史執行存在 RGB/BGR input-contract defect，因此該數值保留供稽核，但不可單獨解讀為模型真實能力或公平的 baseline comparison。**

歷史結果檔案 SHA-256：

`bdcf4e70d592158111ad4596399c095432b3bd5ac55fad35432084d97301f292`

## 3. Checkpoint 封版

| 欄位 | 封版值 |
|---|---|
| 相對路徑 | `outputs/isic_fuseg_formal_seed42_20260914/runs/fuseg_finetune_formal/weights/best.pt` |
| SHA-256 | `cc2955d088928dc00c90d0ba8cab0aefeb49d4cc45da172b030c150ab0dce97a` |
| 大小 | 45,264,156 bytes |
| 允許再訓練／微調 | 否 |

執行前若 checkpoint hash 不同，必須 fail closed；不得以「名稱相同」取代內容雜湊驗證。

## 4. Validation cohort 與來源角色

本次封版 cohort 僅為 **FUSeg 191 張 development validation**，明確角色為：

`DEVELOPMENT_VALIDATION`

產生的 manifest 為：

`experiments/protocols/ISIC_FUSEG_COLORFIX_V1_validation_manifest.json`

其內容包含每一筆的 sample ID、相對 image/mask path、image/mask/label SHA-256、來源、split 與 source role。manifest：

- 筆數：191；
- sample ID：唯一；
- image hash 與 label hash：和原 FUSeg training manifest 的 validation 集完全一致；
- 每筆均為 `source_dataset=FUSeg`、`split=val`；
- 禁止 `test`、`blind_test`、`locked_test` 與 `CO2Wounds` 路徑；
- 建立 manifest 時未開啟 image 或 mask 像素。

本階段比對的是先前已稽核之 cohort/訓練 manifest 的 sample-level 雜湊紀錄與其檔案自身的 SHA-256；因 C0 明令禁止讀取 development pixels，**沒有重新開圖計算每張 image/mask 的即時內容雜湊**。後續正式執行前應以相同封版值進行檔案完整性檢查，再開始推論。

Manifest SHA-256：

`ea58c4e7d36b699d2f1e232d23fea8df9a6010674d08769e5a2ca3aca66ab069`

## 5. FUSeg 來源與授權證據

封版來源證據：

- 官方 repository：`https://github.com/uwm-bigdata/wound-segmentation`
- 上游 checkout pin：`42a272dfe0679f20675e826385925cb7562934b6`
- 使用證據：官方答覆記錄為 `CC BY NC`
- 本專案用途：非商業、離線、學術／畢業開發與評估
- 本實驗資料角色：771 train、191 validation；官方 test 未納入
- 證據檔：`experiments/review_v2/evidence/FUSeg_noncommercial_research_20260914.md`
- 證據檔 SHA-256：`c91ad4b43fabd0ef02242c2c76adca3807f5f98bf61f26df44fba540bc8a69dc`

資料集 release version 在既有證據中未提供，因此本報告只宣告可驗證的 repository commit pin，不自行捏造版本號。

## 6. 唯一允許的程式差異

Phase C0 以 Git `HEAD` 中的歷史 gate source 與目前 worktree source 逐字比較。稽核結果：

`PASS_COLOR_CONTRACT_ONLY`

允許的兩項差異只有：

1. 匯入集中式 `validate_model_input_contract` guard；
2. 將舊的 NumPy RGB 輸入改為經明確宣告的 contiguous NumPy BGR。

未發現 threshold、NMS、resize、crop、matching 或 metric 語意變更。code-diff audit：

`experiments/protocols/ISIC_FUSEG_COLORFIX_V1_code_diff_audit.json`

SHA-256：

`d78ff089a48e0e3cad5ba9fd728aa1a097ae2f46fc6d7df53b46633cf38cff50`

## 7. 固定推論與評估協定

### 7.1 Operating point

| 項目 | 固定值 |
|---|---:|
| Primary confidence | 0.10 |
| Prediction floor `conf` | 0.01 |
| NMS IoU | 0.70 |
| Bbox match IoU | 0.50 |
| Image size | 768 |
| Batch | 1 |
| Half precision | false |
| Retina masks | false |
| Max detections | 300 |
| Test-time augmentation | false |
| Class | 0 (`Wound`) |

### 7.2 Matching 與分母

- Prediction 依 confidence 由高至低排序；
- 每個 prediction 只配對一個尚未配對且 bbox IoU 最高的 ground truth；
- 配對最低 IoU 為 0.50；
- TP、FN 與 size recall 的分母包含所有 ground-truth polygon instances；
- Small `<1%`、Medium `1%–5%`、Large `≥5%`，以 ground-truth 面積占影像比例定義。

### 7.3 Mask、resize 與 crop

- 模型輸入尺寸：768；
- 評估 canvas：512×512；
- prediction mask 以 nearest-neighbor materialize 到 512×512，再轉為 foreground boolean；
- 不新增可調式 mask threshold；
- ROI 為保留 prediction polygons 的 union bounding box；
- 每一邊增加 15% margin；
- crop complete 定義為保留至少 95% ground-truth wound pixels；
- 正樣本無 ROI 時列為失敗，不可從分母排除。

Metric implementation 完全沿用：

`experiments/review_v2/localization_benchmark.py`

SHA-256：

`e62ea05d8cb231cbd697d0a114e8bd81b5f46484ea18fa72535fcb2cef4dd980`

實際 App 裁切實作 `woundcare_inference.py` 亦依歷史 pin 核對，SHA-256：

`fe355e66b52d8d29c1a87b39426e176077bf9a1338fcbeb05b99a1b29c6793cf`

## 8. Acceptance gate

| 指標 | 最低門檻 |
|---|---:|
| Precision | 87.18% |
| Recall | 84.65% |
| F1 | 85.89% |
| Crop complete ≥95% fraction | 90.32% |
| GPU mean latency | ≤50 ms |

比較依已發表值四捨五入至小數點後兩位。任一門檻未通過時，下一步固定為 `stop_and_analyze_errors`；通過也只代表可申請下一個獨立授權階段，不會在 C0 自動開始 multi-seed、test 或 App replacement。

## 9. 預定輸出與指標定位

若後續獲得獨立執行授權，唯一允許的新輸出資料夾為：

`experiments/results/isic_fuseg_colorfix_v1/`

Phase C0 結束時此資料夾不存在。

Primary metrics 延續歷史 gate：TP、FP、FN、Precision、Recall、F1、positive mean mask IoU/Dice、crop coverage、crop-complete fraction、無 ROI 正樣本、帶 prediction 負樣本與 size recall。

GPU latency、per-image prediction、error panels 及 mean retained GT wound pixels 僅列 secondary diagnostics；Mask mAP50 與 Mask mAP50-95 並非歷史 gate 的 primary metrics，C0 不新增或重定義其計算。

## 10. Protocol 與防竄改機制

封版協定：

`experiments/protocols/ISIC_FUSEG_COLORFIX_V1_protocol.json`

Protocol SHA-256：

`16718f5956548e002320ebd2ade83c5ecafdf77e7c79dc1f1d2b63818b7e6df3`

獨立 hash 檔：

`experiments/protocols/ISIC_FUSEG_COLORFIX_V1_protocol.sha256`

未來執行入口必須先驗證 JSON bytes 與 hash 檔完全一致，且輸出資料夾不存在；任一條件不符即停止。

## 11. 自動化驗證

Phase C0 依 test-first 流程建立並通過 16 個專用測試：

1. `test_c0_requires_same_checkpoint_hash`
2. `test_c0_requires_191_validation_samples`
3. `test_c0_requires_same_validation_manifest`
4. `test_c0_rejects_unknown_source_role`
5. `test_c0_rejects_locked_test_role`
6. `test_c0_rejects_historical_external_role`
7. `test_c0_requires_source_evidence`
8. `test_c0_requires_frozen_thresholds`
9. `test_c0_requires_same_metric_script`
10. `test_c0_rejects_non_color_semantic_diff`
11. `test_c0_rgb_bgr_synthetic_equivalence`
12. `test_c0_rejects_existing_output`
13. `test_c0_protocol_hash_detects_mutation`
14. `test_c0_does_not_load_model`
15. `test_c0_does_not_read_validation_pixels`
16. `test_c0_does_not_start_multiseed`

整合回歸結果：**140 passed、0 failed、0 skipped、4 warnings**。範圍涵蓋 C0、相關來源角色／RGB/BGR guards、Phase A/A.5/B1 合成資料測試、既有 ISIC/FUSeg formal safeguards、localization metric tests、WoundCare inference 與實驗審查測試；4 項 warning 為既有第三方 `thop` 的 deprecation warnings。

## 12. Phase C0 最終執行狀態

| 安全計數 | 結果 |
|---|---:|
| `development_images_inferred` | 0 |
| `test_images_used` | 0 |
| `historical_external_images_used` | 0 |
| 模型載入 | 0 |
| 訓練／微調 | 0 |
| Multi-seed | 未啟動 |
| Future result directory | 不存在 |

本輪前後 SHA-256 再查：**28 個受保護檔案皆 unchanged**，包括歷史 ISIC gate result/protocol/predictions/checkpoint、B1 結果、舊 Table 1–5 以及 3 個 CO2Wounds 歷史結果／協定 artifact。這裡僅比對既有輸出檔的 digest，沒有存取 CO2Wounds 資料集或重新評估。

## 13. Completion gate 與決策

| 門檻 | 結論 |
|---|---|
| 歷史 checkpoint 與 SHA 身分一致 | PASS |
| 原 191 張 FUSeg validation manifest 固定 | PASS |
| 明確 development-validation role 與來源證據 | PASS |
| 歷史 operating point、推論與 crop 設定逐欄一致 | PASS |
| metric script 與 crop implementation 未改 | PASS |
| 合成 RGB/BGR boundary 測試 | PASS |
| 唯一色彩語意差異，無其他推論語意差異 | PASS |
| 新輸出不存在，protocol 與 hash 已在推論前固定 | PASS |
| 受保護歷史 artifacts 的 SHA 前後一致 | PASS — 28/28 |
| 無模型推論、開發影像像素讀取、locked test 與 CO2 使用 | PASS |
| 相關與合成測試 | PASS — 140/140 |

```text
PHASE_C0_STATUS = COMPLETE
READY_FOR_PHASE_C_CONTROLLED_REEVALUATION = YES
development_images_inferred = 0
test_images_used = 0
historical_external_images_used = 0
```

**Phase C0 到此停止。** 尚未產生任何新模型表現數字，尚未改變 App 權重，也沒有授權 Phase C execution。下一階段必須收到明確的新指令，並在實際執行入口重新檢查 protocol hash、checkpoint、manifest、metric、來源角色及輸出目錄的不存在狀態。
