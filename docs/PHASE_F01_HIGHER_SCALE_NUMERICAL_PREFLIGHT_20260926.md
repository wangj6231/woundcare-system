# Phase F0.1 — Paired AMP Numerical Feasibility & Execution-Guard Audit

日期：2026-09-26。研究線：F_HIGHER_INPUT_SCALE_V1_NUMERICAL_PREFLIGHT。

## 1. 決策與限制

```ini
PHASE_F01_STATUS = BLOCKED
NUMERICAL_FEASIBILITY_1024 = FAIL
SYNTHETIC_NUMERICAL_FIXTURE_VALIDITY = QUESTIONABLE
READY_FOR_PHASE_F1_PAIRED_HIGHER_SCALE_SEED42 = NO
NEW_TRAINING_AUTHORIZED = NO
EXTERNAL_TEST_AUTHORIZED = NO
APP_MODEL_REPLACEMENT_AUTHORIZED = NO
PATCH_BASED_TRAINING_ROUTE = STOP
STOP_AFTER_F01 = true
```

**Case D：兩組都未於8次內取得3次連續finite。** N768第8次才首次全部梯度finite；N1024第7、8次全部finite，但只有連續2次。不得將2次放寬成3次、不得延長到第9次驗證。

`PERSISTENT_NUMERICAL_INSTABILITY` 是本次「固定8次仍未符合穩定門檻」的預登錄分類，**不表示8次全為nonfinite，也不證明1024在真實訓練不安全**。兩組初始均overflow，1024不是唯一有此現象的尺度。此fixture對scale failure的判別效度QUESTIONABLE，不能直接歸咎1024。`NUMERICAL_FEASIBILITY_1024=FAIL`只指本次synthetic gate未達標，不是模型效果或正式訓練穩定性的結論。

F0仍是BLOCKED historical evidence，保留原65536→32768、第一次nonfinite觀察，以及allocated5.736GiB／reserved5.820GiB。没有追溯改成PASS。

## 2. 預先固定的比較

兩arm分別由不同CUDA subprocess執行，順序N768完整退出後才N1024。每組同checkpoint bytes、fresh model、fresh AdamW、fresh GradScaler；相同seed42、batch4、AMP、lr.0005、betas(.937,.999)、loss gains與class0。初始scale一律stock65536，不手動設定較低值。所有8次均保留，未在出現finite後提早結束。

每次zero_grad → forward/loss → loss finite檢查 → scale/backward/unscale → 每個gradient tensor檢查 → scaler.update → trainable parameter SHA驗證。禁止optimizer.step及scaler.step。模型沒有parameter updates，沒有研究epochs；BN running buffers容許在各自可拋棄RAM模型中變動，未保存。

### Fixture來源與版本紀錄

以F0封存的actual-execution source AST中純 `synthetic_source` 函式重建：4張512×512 noise images、4×4合法rectangle polygons／張、共64GT、class0。未換generator或依結果調整fixture。

Images SHA256：`8c65eed3ed2a2a6119a800eaf67b1e4b23232f918bbe4cde60bdc16936135322`

Polygons SHA256：`a9a49c42ba699167f5a18db186efac5131b134c7805e389693ecac7132d5c422`

兩組執行中重新計算的fixture hashes都與執行前封存一致。**F0當時沒有存原始fixture hash或RAM arrays**；身份證據是其實際執行程式、確定性重建與本次相同hash，不能冒稱有歷史checksum直接可比。GPU開始前發現NumPy distribution metadata2.0.1與實際runtime2.2.6不一致，已修正F0.1描述，保留initial pre-execution文件與更正receipt；正常／no-user-site模式重建hash相同，沒有改package、沒有改F0檔案、沒有GPU重跑。

512 source僅分別resize至768／1024。GT仍使用同一stock polygons2masks_overlap與mask_ratio4，沒有mask-first或patch label path。初始化SHA256仍為`1ec926026473beafafb1ef8da6aa6efcc13d38c391ec8db48006c5d793d5f2a3`；兩arm初始trainable parameter SHA一致，非接續另一arm的RAM狀態。

## 3. 每次梯度telemetry

以下loss在全部16次都finite，trainable parameter hash每次都不變。NaN/Inf欄為「含該值的gradient tensor數」，同一tensor可能同時含兩者，因此兩欄不能直接相加当作nonfinite總數。

| Arm | Attempt | Scaler before→after | Finite gradient tensors | NaN tensors | Inf tensors |
|---|---:|---|---:|---:|---:|
| N768 | 1 | 65536→32768 | 137/365 | 213 | 18 |
| N768 | 2 | 32768→16384 | 269/365 | 68 | 32 |
| N768 | 3 | 16384→8192 | 298/365 | 46 | 25 |
| N768 | 4 | 8192→4096 | 319/365 | 37 | 12 |
| N768 | 5 | 4096→2048 | 351/365 | 5 | 13 |
| N768 | 6 | 2048→1024 | 361/365 | 2 | 4 |
| N768 | 7 | 1024→512 | 364/365 | 0 | 1 |
| N768 | 8 | 512→512 | 365/365 | 0 | 0 |
| N1024 | 1 | 65536→32768 | 98/365 | 250 | 21 |
| N1024 | 2 | 32768→16384 | 292/365 | 42 | 37 |
| N1024 | 3 | 16384→8192 | 313/365 | 40 | 18 |
| N1024 | 4 | 8192→4096 | 349/365 | 4 | 16 |
| N1024 | 5 | 4096→2048 | 359/365 | 2 | 6 |
| N1024 | 6 | 2048→1024 | 362/365 | 1 | 3 |
| N1024 | 7 | 1024→1024 | 365/365 | 0 | 0 |
| N1024 | 8 | 1024→1024 | 365/365 | 0 | 0 |

逐次JSON另有nonfinite parameter names、max_abs_finite_gradient、parameter SHA及guard counters；未報告或比較loss數值以挑選scale。

## 4. Paired摘要

| 項目 | N768 | N1024 |
|---|---|---|
| Attempt1全部gradients finite | 否 | 否 |
| First finite attempt | 8 | 7 |
| First finite scale | 512.0 | 1024.0 |
| 最長連續finite | 1 | 2 |
| 至少3次連續finite | 否 | 否 |
| Numerical classification | PERSISTENT_NUMERICAL_INSTABILITY | PERSISTENT_NUMERICAL_INSTABILITY |
| Peak allocated | 3.306 GiB | 5.765 GiB |
| Peak reserved | 3.434 GiB | 6.158 GiB |
| OOM | False | False |

N768 scaler sequence：65536 → 32768 → 16384 → 8192 → 4096 → 2048 → 1024 → 512 → 512。

N1024 scaler sequence：65536 → 32768 → 16384 → 8192 → 4096 → 2048 → 1024 → 1024 → 1024。

首次finite scale皆高於1；失敗點是「連續次數不足」，不是scale跌至1以下。Stock scaler已自行降低scale並出現finite gradients，但本protocol不足以確認finite stable state。不能說已證明stock-scaler manageable transient overflow，更不能推論300epoch訓練穩定。記憶體資料只供描述，fresh optimizer沒有配置step後才產生的Adam moments。

## 5. Save／step／data guards

CPU行為測試在Ultralytics wrappers載入後安裝sentinels：假torch.save、直接.pt寫入、optimizer.step、scaler.step與image decode均拋出指定例外，temporary fake.pt始終不存在。測試使用CPU tensor與disposable optimizer，沒有GPU模型；故不再依function-object identity判斷guard。

GPU執行與CPU故意觸發測試分開統計。N768/N1024的save_attempts、checkpoint_write_attempts、optimizer_step_attempts、scaler_step_attempts、image_access_attempts **各自皆0**。每次parameter SHA不變，PARAMETER_UPDATE_COUNT=0，輸出樹無.pt/.pth。所有真實影像file-open／decode都禁止；GPU只使用RAM synthetic pixels，image_open_decode_paths為空。允許讀取只有初始化checkpoint、source與frozen configs／fixture lineage。

F0的 `update_save_guards_installed=false` 未更改；F0.1提供新的獨立behavioral evidence，不覆蓋舊觀察。

## 6. 共用training-validation override

Stock trainer的build_dataset／get_validator會使用trainer args.imgsz；不會自動讓H4 validation保持768。新增同一 `SharedValidation768` mixin供兩arm使用：train原args不動，validation使用shallow trainer proxy與args copy，把dataset與validator的imgsz都固定768。

以temporary **合成PNG/labels** 建立實際stock YOLODataset、DataLoader與SegmentationValidator，跑CPU preprocess但禁止validator.__call__／model construction／forward。

| Runtime證據 | C4 | H4 |
|---|---:|---:|
| Train dataset nominal imgsz | 768 | 1024 |
| Val dataset nominal imgsz | 768 | 768 |
| Validator args.imgsz | 768 | 768 |
| Override後trainer args.imgsz | 768 | 1024 |
| 實際synthetic val batch shape | [4,3,800,800] | [4,3,800,800] |
| Training-checkpoint validator confidence | .001 | .001 |
| NMS IoU | .70 | .70 |
| Stock AP matching IoUs | .50:.05:.95 | 相同 |

**768是stock nominal imgsz，而非保證tensor恰好768×768。** 原rect=True、pad=.5與stride32使本次square synthetic validation padding至800×800；兩組完全相同，未偷偷改padding，也不是H4以1024驗證。`BEST_CHECKPOINT_SELECTION_PARITY`在這個共同stock設定的乾式整合邊界已驗證；沒有跑191張real validation、沒有checkpoint selection結果。

這個training-time stock fitness validator與最後operational evaluation不同：最後評估仍使用封存的conf.10／floor.01／NMS.70／bbox match.50／imgsz768／crop margin15%。F01只驗證兩組共用相同checkpoint-selection基礎設施，不偷改最終matching或thresholds。

## 7. 28項完成問題

| # | 問題 | 答案 |
|---:|---|---|
| 1 | F0是否仍BLOCKED？ | 是，20個F0封存／證據檔hash未變 |
| 2 | N768 attempt1 finite？ | 否 |
| 3 | N1024 attempt1 finite？ | 否 |
| 4 | N768 scaler？ | 見完整序列，65536逐次降至512 |
| 5 | N1024 scaler？ | 見完整序列，65536逐次降至1024 |
| 6 | N768 first finite attempt？ | 8 |
| 7 | N1024 first finite attempt？ | 7 |
| 8 | N768 first finite scale？ | 512 |
| 9 | N1024 first finite scale？ | 1024 |
| 10 | N768有3次連續finite？ | 否，只有1次 |
| 11 | N1024有3次連續finite？ | 否，只有2次 |
| 12 | N768 classification？ | PERSISTENT_NUMERICAL_INSTABILITY（限8次gate定義） |
| 13 | N1024 classification？ | 同上 |
| 14 | Overflow為1024獨有？ | 否，768也有 |
| 15 | 1024已證實scaler manageable？ | 未達穩定門檻，不能確認 |
| 16 | Peak VRAM？ | 見第4節allocated/reserved |
| 17 | optimizer.step？ | 兩arm皆0 |
| 18 | scaler.step？ | 兩arm皆0 |
| 19 | checkpoint/save？ | GPU attempts0，無檔案 |
| 20 | save guard行為驗證？ | 是，例外且無file |
| 21 | Parameters unchanged？ | 全16次SHA相同 |
| 22 | Real dataset pixels？ | 0 |
| 23 | C4 train/val imgsz？ | 768/768 nominal，stock rect padding已揭露 |
| 24 | H4 train/val imgsz？ | 1024/768 nominal，同padding |
| 25 | Checkpoint-validation parity？ | shared override dry integration VERIFIED |
| 26 | Real validation inference？ | 無 |
| 27 | Locked test/CO2？ | 未讀取／未使用 |
| 28 | F1放行？ | NO |

## 8. 封存與停止

186 passed in 26.69s；TDD驗證classification、NaN/Inf分離、save/step行為、實際synthetic validator integration，以及原F0/D系列回歸。CPU tests未重跑GPU numerical arms。

兩個GPU subprocess各8次，全部原始telemetry、nonfinite names及失敗attempts都保留。初始化、參數身份、fixture hashes、separate-process順序均通過。F0 unchanged；未來C4/H4正式訓練目錄均不存在。

**停止於F0.1。** 未增加attempts、未手動改initial scale、未改batch、未試960/896/832、未啟動F1。下一階段若要處理fixture判別效度或延長觀察窗，必須另次明確預登錄與授權，不能宣稱本輪已通過。
