# Phase D0.1：配對控制設計與 AMP 更新紀錄預登錄

日期：2026-09-22。專案：WoundCare。新版本：`D_SEG_SMALL_SAMPLING_V2`。

## 1. 結論與研究範圍

**D0.1 完成；V2 可供下一次明確授權啟動 paired D1，但本次不訓練。** 最終狀態以 `D_SEG_SMALL_SAMPLING_V2_freeze.json` 為機器可核對的依據。

- `PHASE_D01_STATUS = COMPLETE`
- `READY_FOR_PHASE_D1_PAIRED_SINGLE_SEED_TRAINING = YES`
- `D1_authorized = false`
- `TRAINING_PERFORMED = false`、`MODEL_LOADED = false`、`MODEL_INFERENCE = false`
- `test_images_used = 0`、`LOCKED_TEST_USED = false`、`CO2Wounds_used = false`
- `STOP_AFTER_D01 = true`

本次新增預登錄、共用 sampler／telemetry adapter、mock 測試與封存索引。沒有建立真實模型、解碼研究影像像素、啟動 GPU 訓練、執行 validation inference 或更換 App 模型。檢查 checkpoint 只讀取 bytes 計算 SHA256，未反序列化權重。既有未提交變更保持原狀，沒有 commit 或 push。

「READY」代表本階段的設計、靜態稽核與合成測試門檻通過，**不是 GPU 實際訓練／效能改善已獲驗證**。D1 執行入口必須在另次授權後接上已封存 adapter，先驗證所有 hashes、資料角色與設定，不能悄悄另寫訓練語義。

## 2. 為何保留 V1，另建 V2

V1 停於 `BLOCKED_BY_ACTUAL_OPTIMIZER_UPDATE_COUNT_NOT_VERIFIED`。歷史訓練可確認 300 epochs、每 epoch 771 anchors、193 batches、總排程 3,741 次 optimizer opportunities，但沒有原始 AMP applied／skipped 更新紀錄。因此不能證明歷史 baseline 和未來模型的實際成功更新數完全相等。

這是 **MISSING_HISTORICAL_TELEMETRY**，不是 **TRAINING_BUG_CONFIRMED**；不推測歷史 AMP 必定沒有 skip。

全部 `D_SEG_SMALL_SAMPLING_V1_*` 保持原有內容與 BLOCKED 狀態。V2 使用兩個新訓練組共同取得 telemetry，並把公平性明確定義為相同排程、AMP 政策、scaler 實作、runner 與訓練方案；實際 applied／skipped 次數則完整觀測，不預先假定相等。

## 3. 配對實驗與唯一介入

| 項目 | Arm C：Fresh Control | Arm S：Fresh Small-aware |
|---|---|---|
| 訓練資料 | 同一份 FUSeg 771 張及 labels | 完全相同 |
| 初始化 | 同一 ISIC auxiliary best.pt | 完全相同，非 C 的訓練產物 |
| 抽樣 | uniform shuffled、無放回 | weighted categorical、有放回 |
| 每 epoch | 每張恰好一次，共 771 anchors | 共 771 draws，可重複或未抽中 |
| A/B/C weights | 1/1/1 | 2/1.5/1，沿用 V1 |
| Epoch seed | PCG64，SeedSequence([42, epoch]) | 相同 seed 規則，使用 frozen weighted choice |
| Runner / instrumentation / evaluation | 同一份程式與 hashes | 完全相同 |

唯一主要設計介入為 **ANCHOR IMAGE SAMPLING DISTRIBUTION**。Machine config diff 實際僅有 `sampling.mode`、`sampling.replacement`、`sampling.weights.A/B`，及必要的 `arm_label`、`experiment_id`、`output_path`；C 類權重兩邊均為 1，故不構成差異。任何其他 training semantic diff 均拒絕。

初始化檔重新計算 SHA256，結果為：

`1ec926026473beafafb1ef8da6aa6efcc13d38c391ec8db48006c5d793d5f2a3`

兩邊 `resume=false`，重新建立 optimizer state，不從 Phase C FUSeg final checkpoint 接續。每個 arm 開始前還必須重新核對同一初始化與 dataset hashes。

## 4. 資料與抽樣證據

沿用 V1 training manifest，重新驗證 771 組 image/label hashes；排除 FUSeg 191 validation 的 IDs 與 image hashes。Validation metadata 只用於隔離檢查，不讀取 validation outcome 來計算權重，也不把 C/C1 失敗案例加入 train。

| Train 類別 | Training-only GT bbox 面積條件 | 影像數 | S 權重 | C 原始比例 | S 預期 draw 比例 |
|---|---|---:|---:|---:|---:|
| A | 至少一個 GT <0.25% | 161 | 2.0 | 20.88% | 29.75% |
| B | 至少一個 GT <1%，但沒有 <0.25% | 301 | 1.5 | 39.04% | 41.71% |
| C | 沒有 GT <1%；含 negative images | 309 | 1.0 | 40.08% | 28.55% |

Train GT 共 965 個：small 531、medium 372、large 62；very-small 182 為 small 的子集。以上是既有 training GT 定義，不是新標註或 validation 結果。影像去重／train-val exact hash 排除不等於已證明患者獨立性；patient/case-family independence 仍未驗證。

D0.1 僅做索引模擬：兩邊各 300 個 epoch 計畫、各 231,300 draws。C 每張都恰好出現 300 次。**實際訓練消耗 anchor 數為 0**。S 與 V1 同一 weighted sampler；並未修改 weights、增加資料或使用 val outcome。

## 5. 共用訓練條件與排程

| 項目 | 兩組固定值 |
|---|---|
| Architecture / class | YOLO11m-seg；class 0 = Wound |
| Seed / size / batch | 42 / 768 / 4 |
| Epochs / patience | 300 / 80 |
| Optimizer | AdamW；lr=0.0005；betas=(0.937, 0.999) |
| Weight decay | weight group=0.0005；bias/norm groups=0 |
| Scheduler / warmup | cosine；lrf=0.01；warmup=5 epochs；warmup_bias_lr=0 |
| AMP / deterministic / workers | true / true / 2；單 GPU |
| nbs / stable accumulation | 64 / 16；沿用 historical warmup accumulation |
| Gradient clipping | norm 10.0，原始 optimizer_step 流程不變 |
| 每 epoch batches | 193，最後一批 3 張 |
| 總排程 | 57,900 batches；每 arm 3,741 optimizer opportunities |
| Loss | 原 v8SegmentationLoss；box/seg=7.5、cls=0.5、dfl=1.5；mask_ratio=4、overlap_mask=true |
| Software lineage | Ultralytics 8.3.53、torch 2.5.1+cu124、NumPy 2.0.1、Albumentations 2.0.8 |

所有其他 args、loss/architecture/runtime source hashes、augmentation 設定完整保存於兩份 config，非只比較此表的幾個超參數。原生每 epoch validation、early stopping 與 checkpoint selection 相同保留；它们不是最終 corrected operating-point evaluation，也不得成為取消 S 或修改方案的理由。

任一 arm 早停至 <300 epochs，或完整排程／telemetry 不符：`PAIRED_FIXED_BUDGET_COMPARISON=NOT_VALID`，停止並報告。不得補 epochs、改 patience、自動 resume、單獨重跑一組或刪除輸出後重試。若需改成必跑滿的 fixed-epoch 設計，另立 protocol。

## 6. AMP telemetry 如何判定

兩組共用 `make_trainer_adapter`，包住原始 `optimizer_step`，不改變其 unscale、clip、scaler.step、scaler.update、zero_grad、EMA 順序。透過 optimizer 公開 `register_step_post_hook` 觀測真正進入並正常完成的標準非 fused optimizer step：

- 原始 opportunity 正常返回、post hook 恰好一次：`applied=true`。
- 原始 opportunity 正常返回、post hook 零次：`skipped=true`、`skip_reason_if_known=AMP_STEP_SKIPPED`。
- 執行例外：記錄 `FAILED`、applied/skipped 為 null、已觀測 hook 次數與 error type，停止；不把程式錯誤偽稱 AMP skip。
- fused 或內部自管 AMP 的 optimizer 一律拒絕，避免把函式呼叫誤認為底層更新。

不以 optimizer 的 return value 判斷（正常 step 也可能回傳 None），不以 scaler 變小或數值不變猜測 applied/skip，不讀取私有 found_inf tensors，不自行宣稱 NaN/overflow 原因。

每個 opportunity 記錄 global_batch、epoch、batch_index、accumulation、attempted、scaler before/after、applied/skipped、已知原因、scheduler state、各 parameter-group LR 與 telemetry status。時間索引為零起算，completed_epochs 為完成數。每筆 JSONL 都 flush；每 epoch 檢查 anchors、batches 與預定 opportunity 數。

完成 arm 時須另存初始化／best／last SHA256、completed epochs、scheduled/applied/skipped/unknown 次數、runtime seconds 與 GPU、CUDA、torch、Ultralytics、Python。實際 GPU 和 skipped 次數必須留待 D1 記錄，現在不填造數值。

## 7. AMP 公平性與敏感度規則

不要求 realized successful updates 事前保證相等；要求相同排程、AMP/scaler 與 runner。若 sampling 改變 gradients，進而改變 skip，可視為該介入的下游執行效果，但必須揭露。

預先採保守 flag：**任何非零 `abs(control_skips - experimental_skips)` 都標記 `AMP_UPDATE_COUNT_IMBALANCE_OBSERVED`**。任一組 skip>0 就獨立列表。這不是事後挑選的有利門檻，也不是自動重訓或自動判研究 gate 失敗的規則；留待下一研究決策解讀。

若次數不同，正式文字必須表明：兩組取得相同排程優化預算，但因 AMP skip，實際成功參數更新次數不同。不得聲稱 identical realized parameter-update count。比較對象為整體 training recipe + sampling intervention，而非「嚴格等更新數」條件下的單獨機制效果。

## 8. Actual anchors 與 augmentation 限制

Sampler 計畫與 dataset index mapping 分開處理。以 `preprocess_batch` 收到的實際 `im_file` 序列核對 sample identity，而非將 DataLoader 預取計畫直接算作訓練 exposure；close_mosaic 的 loader reset 會先重設 sampler cursor，再逐批核對消耗序列。

每 epoch 保存 sample_id sequence、實際 dataset index、category。S 額外保存 weight/probability；C 保存 uniform-control identity 與 marginal probability。可從 JSONL 重建每 epoch unique/repeated anchors，summary 另提供全程 unique/repeats、A/B/C、small/very-small/medium/large GT、negative images、multi-GT anchors exposure。

兩組 augmentation algorithm、probabilities、configuration 完全相同，但 anchor sequence 不同，RNG 會作用於不同影像，mosaic companion composition 也可能不同。這是 **DOWNSTREAM_EFFECT_OF_SAMPLING_INTERVENTION**，不是第二個人工修改因素。**不宣稱 pixel-identical augmentation**。Exposure 統計是實際 anchors 附帶的未增強 GT，不是 mosaic companion 或像素層級曝光量。

## 9. 評估順序、比較層級與歷史參考

順序預先固定：訓練 C → 訓練 S → 最終 corrected evaluation C → 最終 corrected evaluation S。兩組正常完成後才能進入最終評估；不可看 C 最終分數再決定是否執行 S。Safety/integrity failure、系統中斷或上述早停規則可停止 pair，但不得偷偷重跑。

評估直接沿用 frozen Phase C FUSeg **191 validation**：corrected RGB→BGR 一次、confidence=0.10、prediction floor=0.01、NMS IoU=0.70、matching IoU=0.50、imgsz=768、crop margin=15%，相同 mask/metrics/matching/size-bin/crop 實作。Crop complete 仍是保留至少 95% GT wound pixels；positive image 無 ROI 算失敗。

| 指標 | 歷史 Phase C corrected（僅 reference） | Fresh C | Fresh S |
|---|---:|---|---|
| Precision | 209/245 = 85.31% | 尚未訓練／評估 | 尚未訓練／評估 |
| Recall | 209/241 = 86.72% | 同上 | 同上 |
| F1 | 418/486 = 86.01% | 同上 | 同上 |
| Small Recall | 109/137 = 79.56% | 同上 | 同上 |
| Medium Recall | 86/90 = 95.56% | 同上 | 同上 |
| Large Recall | 14/14 = 100.00% | 同上 | 同上 |
| Crop complete | 167/186 = 89.78% | 同上 | 同上 |
| TP / FP / FN | 209 / 36 / 32 | 同上 | 同上 |

Primary comparison 是 **fresh C vs fresh S**。歷史 corrected Phase C 只作 `HISTORICAL_REFERENCE_COMPARATOR`，不重新 inference；上表歷史值來自封存紀錄，不是本次實驗結果。Fresh C 不要求與歷史數字相等。舊 RGB/BGR 缺陷 gate 只能作方法學歷史，不得作模型優劣 comparator。

## 10. Paired advancement gate

先確認兩組完整 300 epochs、各 3,741 scheduled opportunities，applied+skipped 與排程一致、unknown=0。之後 S 必須相對 **fresh C** 同時通過：

| 條件 | 凍結判定 |
|---|---|
| Primary：Small Recall | S small TP ≥ C small TP +1；相同分母 137 |
| Precision safety | S ≥ C −0.01（1.0 percentage point） |
| F1 safety | S ≥ C −0.01 |
| Crop safety | S complete count ≥ C count −1；positive 分母 186 |
| Medium safety | S medium TP ≥ C medium TP −1；分母 90 |
| Large safety | S large TP ≥ C large TP；分母 14 |

用整數 count 與 exact rational arithmetic 判斷，拒絕負數、非整數、不一致或超出 support 的 counts，不以四捨五入百分比跨門檻。通過才為 `PASS_SMALL_SAMPLING_RESEARCH_ADVANCEMENT_GATE`。歷史 109/137 **不是**此 primary effect 的控制組數值。

另外仍報 historical development gate：P≥87.18%、R≥84.65%、F1≥85.89%、crop≥90.32% 與原 latency≤50ms 等條件；不與研究 advancement 混為一談。研究 gate 不等於 deployment readiness、App replacement、multi-seed 或盲測許可。

Secondary diagnostics 保留 small 五 bins：<0.10%、0.10–<0.25%、0.25–<0.50%、0.50–<0.75%、0.75–<1.00%；另報 single-/multi-GT instance Recall、single-/multi-GT crop failure、overall Recall、FP/FN 與 no-ROI。不得看結果改 sampling weights、confidence、NMS、解析度或 crop 規則。

## 11. 測試、保護邊界與封存

新增 D0.1 targeted tests：**45 passed**。與 D0、C/C0/C1、B1、A/A.5、localization benchmark、ISIC/FUSeg formal/warmup、experiment-review 合併回歸：**267 passed，4 個既有 thop distutils deprecation warnings**。

測試使用 synthetic metadata、mock optimizer/scaler/trainer 或非模型的假 checkpoint bytes。涵蓋設定差異拒絕、普通 applied step、scaler 數值不變的 simulated skip、例外不冒充 skip、fused 拒絕、未監控 step 拒絕、實際 anchor/index 序列、prefetch/reset、train/val/test/CO2 來源隔離、排程有效性、gate 計算、負數／越界 counts、exclusive output lock、逐筆 flush 及 best/last hash summary。

D0.1 builder 在攔截 torch、Ultralytics、PIL、cv2 imports 的獨立程序內成功完成；不建立模型、不解碼資料集影像。既有回歸中有框架模組 import 與 synthetic array 測試，不等於載入研究模型或對真實影像推論。

本次重新核對 **2,004 個保護檔案** hashes 未變，包括 V1 全部產物、D0 source/tests/report、train image/labels、必要 lineage 與 Phase C/C1 封存證據。沒有讀取 locked test 或 CO2Wounds 的資料／預測；歷史 protocol 文字中包含的來源名稱不構成重新存取該資料集。

未來專用結果目錄在 D0.1 結束時都必須**不存在**：

- `experiments/results/d_seg_small_sampling_v2_control/`
- `experiments/results/d_seg_small_sampling_v2_experimental/`

未來各組以原子建立全新目錄，再 exclusive-create `execution.lock`。重複執行、覆寫、resume、刪除後重試均禁止；中斷後留下的部分目錄也阻擋下一次執行。D0.1 只在測試的暫存目錄驗證此行為，不建立上述研究結果目錄。

## 12. 使用者指定的 23 個問題

1. **V1 為何 BLOCKED？** 缺歷史實際 AMP applied/skipped 次數，不可證明與新 run 等更新數。
2. **是否改 V1？** 沒有；仍 BLOCKED，全部 SHA256 保持。
3. **為何 fresh paired control？** 兩組採相同當前 instrumentation，直接觀測更新行為，不補造歷史資料。
4. **Primary comparator？** Fresh uniform Control C。
5. **Historical C 的角色？** Immutable historical reference；不得重新 inference。
6. **Initialization 相同？** 相同 ISIC SHA256，已重算確認；兩邊 fresh optimizer。
7. **Dataset 相同？** 同一 771 train image/label hashes，無新增、無 val 混入。
8. **Runner 相同？** 同一 adapter source 與 SHA256、相同 pinned base trainer。
9. **唯一設計差異？** Anchor sampling distribution；身分／輸出路徑差異不屬訓練因子。
10. **Applied/skip 怎麼記？** Optimizer post hook＋原始 scaler step 正常返回；逐 opportunity JSONL。
11. **要求 realized updates 相同？** 不要求事前保證；要求相同 scheduled opportunities，完整揭露實際數值。
12. **不同時如何解釋？** Sampling 下游 AMP effect；標 imbalance，不宣稱等成功更新、不自動重訓。
13. **Early stopping？** 保留 patience80；任一 <300 即 NOT_VALID，停止且不補跑。
14. **Augmentation algorithm 相同？** 是，算法、機率、設定全部相同。
15. **Pixel-identical？** 不作此宣稱；anchor/RNG/mosaic companion 可不同。
16. **Primary endpoint？** Fresh S 相對 fresh C 的 Small Recall。
17. **Gate？** Small TP +1，且 P/F1/crop/medium/large 全部 safety 條件成立，見 §10。
18. **Val outcomes 用於 sampling？** 沒有；只用 train GT geometry，val IDs/hash 僅作隔離排除。
19. **Training？** 沒有。
20. **Inference？** 沒有，亦未載入真實模型。
21. **Locked test？** 未使用，test_images_used=0。
22. **CO2？** 未使用，未重新評估。
23. **可授權 paired D1？** D0.1 設計／模擬門檻已通過，可由下一條明確指令授權；本次未授權、未啟動。

## 13. 產物與下一個停止點

全部放在 `experiments/protocols/`：

- `D_SEG_SMALL_SAMPLING_V2_protocol.json`
- `D_SEG_SMALL_SAMPLING_V2_control_config.json`
- `D_SEG_SMALL_SAMPLING_V2_experimental_config.json`
- `D_SEG_SMALL_SAMPLING_V2_telemetry_schema.json`
- `D_SEG_SMALL_SAMPLING_V2_evaluation_protocol.json`
- `D_SEG_SMALL_SAMPLING_V2_dry_audit.json`
- `D_SEG_SMALL_SAMPLING_V2_freeze.json`

Freeze index 記錄上述六份內容檔、共用 runner、D0.1 builder、test file 與本報告 SHA256；不把 freeze 自身放入其雜湊造成循環。最終 postflight 再核對全部 hashes、V1 BLOCKED 與兩個結果目錄不存在。

**停止於 D0.1。沒有 D1 訓練產物或新準確率可報告。** 下一步須另次明確授權 paired D1，且即使屆時啟動也不能使用 locked test、CO2Wounds 或執行多種子／模型替換。
