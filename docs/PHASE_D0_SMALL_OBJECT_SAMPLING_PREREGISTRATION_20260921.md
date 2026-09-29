# Phase D0：Small-Object-Aware Sampling 實驗預先登錄

文件識別日期依指示保留 20260921；跨日完成於 2026-09-22（Asia/Taipei）。

## 1. 最終狀態：BLOCKED，未啟動訓練

已完成可安全執行的 baseline／initialization／dataset audit、training-only GT 分組、300 epochs 索引抽樣模擬、單變因設定與評估／advancement gate 定義、JSON 雜湊封存及測試。

**D0 completion gate 尚未通過，不可進入 D1。** 阻擋碼：`BLOCKED_BY_ACTUAL_OPTIMIZER_UPDATE_COUNT_NOT_VERIFIED`。

歷史 baseline 使用 AMP。現存紀錄可重建 3,741 次排程上的 `optimizer_step()` 呼叫，但没有保存 GradScaler 因非有限梯度而跳過的更新次數。原始 stdout/stderr、300 列 results.csv 與 TensorBoard 均未提供實際成功更新計數。因此不能把「排程相同」直接宣稱為你要求的「實際 total optimization steps 完全相同」。

這是**證據未確認**，不是已發現 AMP 跳步或模型訓練錯誤。已知相同的配置及預算仍完整保留；沒有為了通過 gate 而关闭 AMP、改 optimizer、關閉 early stopping 或重跑基準。

```text
PHASE_D0_STATUS = BLOCKED
READY_FOR_PHASE_D1_SINGLE_SEED_TRAINING = NO
PREREGISTRATION_ARTIFACTS = FROZEN_BLOCKED
TRAINING_PERFORMED = false
MODEL_LOADED = false
MODEL_INFERENCE = false
LOCKED_TEST_USED = false
CO2WOUNDS_USED = false
TEST_IMAGES_USED = 0
D1_OUTPUT_DIRECTORY_EXISTS = false
STOP_AFTER_D0 = true
```

## 2. 研究問題、假設與限制

唯一研究問題：保持 architecture、初始化、資料集、loss、augmentation、imgsz、epochs、batch、optimizer、LR schedule 與 evaluation 不變，只改 training anchor-image sampling distribution，是否提升 small-wound Recall，且不造成不可接受的 Precision、F1 與 crop completeness 退化？

- H1：small-object-aware image sampling 改善 small Recall。
- R1：過度抽樣小傷口影像可能增加 FP 或降低 Precision。
- R2：small Recall 改善不保證 multi-instance crop completeness 改善。

本阶段無新訓練、fine-tuning、inference、model loading、threshold/NMS/IoU/crop/imgsz tuning、multi-ROI、multi-seed、test/external evaluation 或 App model replacement。沒有加入、移除或搬移任何訓練影像。C1 只提供研究假設；sampler 不讀取 validation failure IDs、confidence、FN、crop outcome。

## 3. Baseline identity 與初始化 lineage

正式 comparator 為 **Phase C 正確 RGB/BGR contract 下的 ISIC→FUSeg 結果**，不是歷史錯色結果、最新 checkpoint 或另一个模型。

| 身分項目 | 已核對值 |
|---|---|
| Baseline experiment ID | D-Formal-ISIC-FUSeg-seed42-20260914 |
| Architecture | YOLO11m-seg，單一 class id 0 = Wound |
| FUSeg training seed / completed epochs | 42 / 300 |
| Runtime architecture log | 445 layers、22,359,987 parameters；711/711 pretrained items transferred |
| FUSeg initialization | 同一份正式 ISIC auxiliary `best.pt`；不是 FUSeg 已訓練權重 |
| 初始化語意 | Fresh FUSeg optimizer，resume=False；不重跑 ISIC |
| ISIC lineage | 歷史 formal pretraining 80 epochs、best at epoch 60；僅沿用已完成初始化 |

| 雜湊項目 | SHA256 |
|---|---|
| Baseline checkpoint | `cc2955d088928dc00c90d0ba8cab0aefeb49d4cc45da172b030c150ab0dce97a` |
| Baseline runtime args.yaml | `fe0203bd780e42f6d84fde13f6778770b8e84aaf4cd1a5ca3537ed96456cd220` |
| Baseline dataset manifest | `69efe88935f1dd064b79ad9f2ceb67e2e358b0144eed985a79087f9459aa87f0` |
| Baseline／experimental 同一 initialization | `1ec926026473beafafb1ef8da6aa6efcc13d38c391ec8db48006c5d793d5f2a3` |
| Phase C corrected result | `d797561841227617ad3c8652164d12daa03d63924cada9739c9ef473007ca79e` |
| Historical runner | `1d4ab637f59a7084df42065626bd8f6685e700da7077c940fcd72b00157fa0c3` |

Continuation lock 記錄的 ISIC checkpoint hash 與現存檔案吻合；runtime args 的 model 路徑亦指向同一 ISIC checkpoint。這是前處理任務初始化，不是從 Phase C 最終 FUSeg 權重繼續訓練，故不混入額外 FUSeg fine-tuning 輪次。

所有 checkpoint 只讀 bytes 計算 SHA256；沒有 torch.load 或 YOLO model construction。

## 4. Training manifest 與來源／隔離

固定資料仍為 **771 張 FUSeg train + 191 張 FUSeg development validation**。新 manifest 不建立任何 train/val split，也不搬移影像。

Training geometry 僅解析原 train YOLO polygon labels，以 normalized polygon min/max 得 bbox area ratio，不解碼影像、不推論。每張列出 sample ID、影像／標註 hash、來源、split、GT count、最小 bbox area ratio、small/very-small flags、category/weight、每個 GT 的 size metadata。

| 檢查 | 結果 |
|---|---:|
| Training image/label 成對且雜湊符合原 manifest | 771 / 771 |
| Training GT instances | 965 |
| Small / medium / large GT | 531 / 372 / 62 |
| Very-small GT（small 子集合） | 182 |
| 至少一個 small 的 training images | 462 |
| 至少一個 very-small 的 training images | 161 |
| Single-GT / multi-GT / no-GT images | 607 / 146 / 18 |
| Validation metadata identity 與 Phase C 相同 | 191 / 191 |
| Train–validation exact image-hash overlap | 0 |
| Validation IDs 進 sampler | 0 |
| Training unique exact-content hashes | 771 |
| 模型載入、影像像素解碼 | 0 |

來源限既有 FUSeg evidence 所允許的署名、非商業、離線學術／畢業研究。沿用 upstream commit `42a272dfe0679f20675e826385925cb7562934b6` 與既存 CC BY NC 資料使用紀錄；不推定 license version 或產品部署權利。來源：[UWM FUSeg repository](https://github.com/uwm-bigdata/wound-segmentation)；本次沒有重新下載或擴大授權範圍。

Validation manifest 的 ID/hash 只供排除檢查；validation 個別預測不參與權重計算。既有 Phase C/C1 artifacts 作 hash-only 完整性保護，不將其中 failure IDs 載入抽樣器。

### Group dependency 限制

已按相同 image SHA256 建立 exact-content group，771 組均為單張，並輸出各組加總權重與 expected/observed exposure。無已知 exact-content 跨 split 洩漏。

**GROUP_LEVEL_SAMPLING_DEPENDENCE_NOT_VERIFIED**：目前 training manifest 沒有 patient/case/derived-image family IDs。本次沒有把檔名當患者 ID，也沒有新增 pHash 或臨床獨立性主張。無 exact duplicates 不代表無近重複或患者關聯。

## 5. 固定的抽樣定義

Primary small 不變：`GT bbox area / image area <0.01`。Very-small 為 secondary training diagnostic：`<0.0025`。分類依未增強的 training GT bbox 計算，以每張最小 GT 面積決定唯一 category。

| Category | 定義 | Images | Single-GT | Multi-GT | No-GT | Weight |
|---|---|---:|---:|---:|---:|---:|
| A | 至少一個 GT <0.25% | 161 | 107 | 54 | 0 | 2.0 |
| B | 有 GT <1%，但沒有 GT <0.25% | 301 | 225 | 76 | 0 | 1.5 |
| C | 沒有 GT <1%，包括空標註 | 309 | 275 | 16 | 18 | 1.0 |

沒有另加 multi-GT 權重。Empty-label negative images 保留，未刪除。只有這一組 sampling strength，沒有比較五種權重或依 validation 分數選擇權重。

Control 是既有 shuffled random sampling without replacement：每 epoch 771 個 anchor images 各一次。Experimental 是 normalized weights 的 categorical sampling **with replacement**，每 epoch 同樣 771 draws。更高重複率／更低單 epoch unique coverage 是 sampling intervention 的一部分，必須報告。

預先固定演算法：`Generator(PCG64(SeedSequence([42, zero_based_epoch]))).choice(N, size=N, replace=True, p=normalized_weights)`，NumPy 2.0.1。每個 epoch 使用獨立可重建的隨機流，不消耗 augmentation 的 global RNG。D1 必須按 frozen sample ID 映射實際 dataset index，不能假設排序恰好相同。

## 6. 300 epochs 純索引抽樣模擬

共 **231,300 draws**，沒有讀取模型或影像像素；完整 draws 的 little-endian int64 串流 hash：`20c8c51ad16479ec14c5d468cd68d00e4ebf13ab592b9d19b3220c9d05ad1245`。

| Category | 原均勻抽樣占比 | 加權預期占比 | 模擬占比 | 模擬 draws |
|---|---:|---:|---:|---:|
| A | 20.88% | 29.75% | 29.74% | 68,794 |
| B | 39.04% | 41.71% | 41.73% | 96,526 |
| C | 40.08% | 28.55% | 28.53% | 65,980 |

預先採用的保守模擬接受規則：每張與每個 exact-content group 的 expected reuse ≤2/epoch；300 epochs 觀察最大 reuse ≤3×control 的 300 次；300 epochs 不得有完全未見的 training image。這是 index-distribution engineering audit，不是模型成效門檻。

| Repetition 指標 | 結果 |
|---|---:|
| Control 每 epoch unique anchors | 771 |
| Experimental 預期 unique anchors / epoch | 477.44 |
| 模擬平均 unique anchors / epoch | 477.43 |
| 模擬平均重複 draws / epoch | 293.57 |
| 每張最大 expected reuse / epoch | 1.4245 |
| 每張最大 expected reuse / 300 epochs | 427.34 |
| 觀察到某張在單一 epoch 最大 reuse | 10 |
| 每張在 300 epochs 最大／最小 reuse | 478 / 166 |
| 300 epochs 從未抽到 | 0 |
| Simulation acceptance | PASS |

同一張某 epoch 被抽到 10 次不是每 epoch 的預期值，也不可隱藏。平均單 epoch unique coverage 明顯下降；「更多看小傷口」的代價是同一預算下少看其他影像，而不是免費增加資料。

### 6.1 Class、source 與 GT size exposure

| Anchor 所帶 GT instances | 原每 epoch | 加權預期每 epoch | 模擬 300 epochs 總量 |
|---|---:|---:|---:|
| Small | 531 | 642.80 | 192,709 |
| Medium | 372 | 318.37 | 95,212 |
| Large | 62 | 49.86 | 15,100 |
| Very-small（small 子集合，不可再加總） | 182 | 259.26 | 77,657 |
| 全部 Wound instances | 965 | 1,011.02 | 303,021 |

來源仍 100% FUSeg、所有正標註仍 class id 0 Wound，沒有傷口七分類混合比例可宣稱改善。但 positive/negative exposure 會改變：18 張 negative 的預期抽樣次數由 18/epoch 降至 12.82/epoch。Multi-GT anchor 占比亦由 18.94% 自然升至 21.99%，源於與 small 的關聯，**沒有額外 multi-GT 權重**。

因此保留 R1：降低 negative exposure 可能增加 FP；medium/large exposure 下降亦需監督。不能因 source/class ID 比例相同，就說所有資料分布都沒變。

上述 GT exposure 是「被抽中的 anchor 影像所帶原始標註」計數，不是增強後實際可見 wound pixels。原 augmentation 有 mosaic=0.1，companion images 來自原有 buffer selection；本次不模擬或改寫其像素／配對行為。D1 若另行獲准，必須完整記錄實際 anchor indices，並處理 InfiniteDataLoader prefetch/reset、close_mosaic，不能多消耗或漏掉 epoch 計畫。

## 7. Training config 與 budget freeze

不新增訓練 runner；本次只建立 manifest、純索引 sampler 與 preregistration artifacts。Historical runtime args 保留於 control/experimental 設定；只排除輸出路徑等非訓練因素，模型與資料身分另以 hash 固定。

| 配置 | Control／experimental 相同設定 |
|---|---|
| Architecture / seed | YOLO11m-seg / 42 |
| Img size / batch | 768 / 4 |
| Epochs / patience | 300 / 80 |
| Effective optimizer | AdamW，runtime log 已確認，非 optimizer=auto 推測 |
| Initial LR / betas | 0.0005 / (0.937, 0.999) |
| Weight decay | 一般 weights 0.0005；bias/norm groups 0 |
| Scheduler | Cosine，lrf=0.01 |
| Warmup | 5 epochs，warmup_bias_lr=0；全 300 列 epoch-end LR 與重建 schedule 一致 |
| AMP / deterministic / workers | true / true / 2 |
| Nominal batch nbs / stable accumulation | 64 / 16 micro-batches |
| Gradient clipping | norm=10.0 |
| Loss | 同版 v8SegmentationLoss：box=7.5、seg gain=7.5、cls=0.5、dfl=1.5 |
| Mask settings | overlap_mask=true，mask_ratio=4 |
| Architecture/runtime source | 固定本機 Ultralytics 8.3.53 相關 source SHA256；torch 2.5.1+cu124 |

Augmentation 完整固定：mosaic=0.1、close_mosaic=30、mixup=0、copy_paste=0、degrees=5、translate=0.05、scale=0.2、shear=0.5、perspective=0.0002、fliplr=0.5、flipud=0、hsv_h=0.01、hsv_s=0.4、hsv_v=0.25、bgr=0，及 runtime args 其餘 augmentation 欄位。

歷史 runtime Albumentations log：Blur/MedianBlur/ToGray/CLAHE 各 p=0.01；具體參數與目前 package identity 一併記錄。這是固定已記錄設定，不聲稱可從結果重建歷史每次隨機增強內容。

### 7.1 相同的排程預算，以及尚未確認的實際更新

| 指標 | 值 |
|---|---:|
| Anchor draws / epoch | 771 |
| Batches / epoch | ceil(771/4) = 193 |
| 最後一個 batch | 3 張，不 drop_last |
| Epochs | 300 |
| Total anchor draws | 231,300 |
| Total batches | 57,900 |
| Warmup iterations | 965 |
| 排程 optimizer calls | 3,741 |
| 實際 AMP 成功更新次數 | NOT_VERIFIED_FROM_HISTORICAL_LOGS |

前 5 epochs 排程更新次數為 94、36、23、17、13；之後為 12 或 13，完整 300 epoch schedule 存於 training config。不能把 57,900 batches 寫成 optimizer updates，也不能把 warmup 當作一直 accumulate=16。

原 patience=80 不改；如果未來 experimental 早停而未跑滿 300，必須標為嚴格實際預算比較 **NOT_COMPARABLE**，不得自行改 patience、續跑補 epochs 或把 unequal-budget 結果當成單變因成功。

### 7.2 阻擋證據與可行解鎖方向

已檢查：stdout/stderr 無 applied/skip step counter；results.csv 300 列只有 loss/metrics/LR；TensorBoard 5,711 個 events 的 summary tags 也只有 epoch 級 loss、LR、metrics，沒有 GradScaler scale、found_inf、applied-step 或 skipped-step counter。只讀 tag/step metadata，不拿其 metric values 選擇樣本。

本機 pinned torch GradScaler source 明確將非有限梯度的 optimizer.step 跳過；AMP checks passed 或 epoch loss 有限，不能反證整段訓練從未跳步。D0 禁止 model loading，沒有為補證據載入 checkpoint。

要解鎖，需另行提供可驗證的 baseline applied/skip-step 原始紀錄；若不存在，需由使用者明確决定是否接受「相同排程 optimizer calls」作為公平預算定義，或另立帶 step telemetry 的 paired control 設計。後者會涉及新訓練與實驗版本，不能在 D0 自行執行或悄悄換掉 Phase C comparator。已 frozen 的 V1 不可事後改寫，任何修訂須新版本並保留本次 blocked 證據。

## 8. Evaluation 原封繼承 Phase C

| 項目 | 凍結值 |
|---|---|
| Validation | 同一份 191 張 FUSeg manifest，SHA256 unchanged |
| Confidence / prediction floor | 0.10 / 0.01 |
| NMS IoU / bbox matching IoU | 0.70 / 0.50 |
| Evaluation imgsz | 768 |
| Crop | retained prediction polygons 的 union bbox；每邊 margin=15% |
| Crop complete | ≥95% GT pixels retained；positive 無 ROI 算失敗 |
| RGB/BGR | PIL RGB → contiguous NumPy BGR，恰好一次 |
| Size bins | small <1%；medium 1–<5%；large ≥5% |
| Metrics / crop implementation | 直接引用 Phase C 路徑與 hash |

Frozen comparator：TP/FP/FN=209/36/32；Precision=85.31%、Recall=86.72%、F1=86.01%、small=109/137=79.56%、medium=86/90=95.56%、large=14/14=100%、crop pass=167/186=89.78%。不重新 inference baseline。

Very-small diagnostic 為 29/49=59.18%，並保留 <0.10%、0.10–<0.25%、0.25–<0.50%、0.50–<0.75%、0.75–<1.00% 五 bins。D1 不得根據這些 validation bins 反過來修改已凍結 weights。

## 9. Endpoint 與 advancement gate

Primary endpoint：**Small Recall**。下列條件必須全部成立，使用未四捨五入的整數分母／分數比較，不以報表兩位小數判斷邊界。

| Gate | 凍結條件 |
|---|---|
| Small Recall 嚴格提升 | Small TP ≥110/137（≥80.29%），baseline=109/137 |
| Precision 降幅 ≤1.0 pp | ≥209/245 −0.01 = 84.306122…% |
| Overall F1 降幅 ≤1.0 pp | ≥418/486 −0.01 = 85.008230…% |
| Crop completeness 至多少一張 | ≥166/186 = 89.247312…% |
| Medium safety | TP ≥85/90；至多額外漏一個 medium |
| Large safety | TP ≥14/14；不接受額外 large FN |

Medium/large safety 是本次預先固定的保守操作定義，不是統計顯著性界線。No-ROI、FP/FN 數、overall Recall、very-small Recall、五個 small bins、single/multi-GT Recall 與 crop failure 都保留為 secondary diagnostics。

原 development gate 完整另行報告，未取代或放寬；Precision ≥87.18%、Recall ≥84.65%、F1 ≥85.89%、crop ≥90.32%、latency ≤50ms 等既定條件仍繼承。研究 advancement 與 deployment readiness 是兩件事：通過小目標研究 gate 不等於可以替換 App 模型、使用 locked test 或啟動多種子。

## 10. Baseline fairness audit

| 問題 | 判定 |
|---|---|
| Architecture 完全相同？ | YES，registered settings / lineage 相同 |
| Initialization 完全相同？ | YES，同一 ISIC checkpoint hash，fresh FUSeg optimizer |
| Training dataset 完全相同？ | YES，771 train IDs、image/label hashes 全部吻合 |
| Optimization steps 完全相同？ | **NOT_VERIFIED**：排程 3,741 calls 相同；歷史 AMP 成功更新計數缺失 |
| Augmentation 完全相同？ | YES，固定 runtime args／已記錄 augmentation 設定；未改增強方案 |
| Loss 完全相同？ | YES，固定實作與 gains |
| Optimizer 完全相同？ | YES，AdamW runtime 身分／LR／schedule 已核對 |
| Evaluation 完全相同？ | YES，直接繼承 Phase C protocol/manifest/implementations |
| 唯一 training difference 是 sampling？ | 設計上 YES；尚未實作／執行 D1 trainer，不能宣稱執行證明 |

沒有全部 YES，因此 D0 gate=BLOCKED，不輸出 READY=YES。

## 11. 19 個指定問題回答

1. **Baseline training identity？** D-Formal-ISIC-FUSeg-seed42-20260914，以 Phase C corrected evaluation 為 comparator，hash 見 §3。
2. **Experimental initialization 一致？** YES，同一正式 ISIC best.pt；不從 FUSeg final checkpoint 接續。
3. **Training dataset 一致？** YES，同一 771 張與 labels，無增刪。
4. **Small／very-small training samples？** 462／161 張含該類 GT；對應 GT instances 531／182。
5. **Weights？** A/B/C=2.0/1.5/1.0，沒有 multi-GT bonus。
6. **Expected exposure？** A/B/C 從 20.88/39.04/40.08% 變成 29.75/41.71/28.55%。
7. **Class/source 明顯改變？** 正標註仍 100% Wound、來源仍 100% FUSeg；但 negative/medium/large exposure 下降，已保留風險。
8. **每 epoch／total budget 完全相同？** 註冊排程相同；實際成功更新數未能確認，是阻擋項。
9. **唯一 difference？** 已註冊唯一因子是 anchor sampling distribution；沒有改 augmentation、loss 或 crop。
10. **Primary endpoint？** Small Recall。
11. **Advancement gate？** §9 六項全部成立，與原 deployment gate 分開。
12. **Evaluation 完全繼承 C？** YES，191 val、相同 colors/thresholds/NMS/IoU/imgsz/crop/size/metrics。
13. **Validation outcome 參與 training selection？** NO。Sampler 只接收 train GT metadata；val IDs/hash 只作隔離排除。
14. **載入模型？** NO。
15. **訓練？** NO。
16. **推論？** NO。
17. **碰 locked test？** NO，沒有讀取其圖片、manifest、predictions 或執行盲測。
18. **碰 CO2？** NO，未讀取其資料或結果。歷史 protocol 內嵌的舊保護清單不會被遍歷至 CO2 檔案。
19. **可授權 D1 single-seed？** NO；D0 gate 被 strict applied-update-count evidence 阻擋，且 D1 永遠需下一條明確授權。

## 12. 產物、雜湊與測試

全部 preregistration JSON 放在 `experiments/protocols/`，**不建立** `experiments/results/d_seg_small_sampling_v1/`。

- [Training manifest](../experiments/protocols/D_SEG_SMALL_SAMPLING_V1_training_manifest.json)
- [Sampling protocol](../experiments/protocols/D_SEG_SMALL_SAMPLING_V1_sampling_protocol.json)
- [Training config（control + experimental）](../experiments/protocols/D_SEG_SMALL_SAMPLING_V1_training_config.json)
- [Evaluation protocol](../experiments/protocols/D_SEG_SMALL_SAMPLING_V1_evaluation_protocol.json)
- [Sampling simulation（含逐圖與 exact-group exposures）](../experiments/protocols/D_SEG_SMALL_SAMPLING_V1_sampling_simulation.json)
- [主 protocol／advancement gate](../experiments/protocols/D_SEG_SMALL_SAMPLING_V1_protocol.json)
- [Audit／前後保護快照](../experiments/protocols/D_SEG_SMALL_SAMPLING_V1_audit.json)
- [Freeze hash index](../experiments/protocols/D_SEG_SMALL_SAMPLING_V1_freeze.json)

輸出以 exclusive-create 寫入；再次執行會拒絕覆寫 frozen V1。D1 output 若已存在，D0 即 fail closed。將來 D1 必須以 exclusive lock 防止覆寫／重跑，不能使用 exist_ok 或自動 resume 混入舊結果。

D0 targeted tests **38 passed**；合併 C1、C execution/postflight、C0、A/A.5/B1 與既有 synthetic regression，**226 passed，4 個既有 thop deprecation warnings**。涵蓋 training-only GT、不得讀取 validation outcomes、ID/hash exclusion、偽裝來源與 path traversal、locked/CO2 拒絕、九個公平性欄位差異拒絕、固定 weights、期待分布、epoch 重現性、small 邊界、空標註、錯誤 polygon、模型框架不載入、無 train/predict 呼叫與 result directory 不存在。

本次產生期間 **1,992 個保護檔案 hash 未变**；包含 train images/labels、baseline lineage、Phase C/C1 artifacts、必要的 source code 與 scope evidence。Binary hash 不等於解碼或檢視像素。沒有 commit/push、沒有公開 datasets/weights/影像。

## 13. Completion checklist 與停止點

PASS：baseline identity、相同初始化／771 train、training-only sampling、validation 排除、small/very-small 定義、單一 weights、sampling simulation、architecture／optimizer／loss／augmentation／imgsz／seed、Phase C evaluation、primary endpoint／advancement gate、output immutability、無 training／inference／locked test／CO2、tests。

**BLOCKED：training budget 的「歷史實際成功 optimization steps」沒有原始證據，不能確認 exact equality。** Patient/case family 依賴另標 NOT_VERIFIED，未轉換成患者獨立性聲明。

**停止於 D0。** 已封存提案，不啟動 D1；不自行降低公平性條件，也不重跑 control 來填補歷史紀錄。
