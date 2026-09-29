# Phase F1.1：中斷訓練與 Checkpoint 完整性鑑識

日期：2026-09-28（Asia/Taipei）
範圍：F1 V2 seed42，C4 train768 / H4 train1024；只讀原始產物，不做模型效果評估。

## 一、結論

**F1.1 鑑識完成；原 F1 仍為 INVALID_OR_INTERRUPTED，不是有效的固定預算配對實驗。**

C4 完成 300 epochs；H4 可確認完成 297 epochs。H4 第 298 epoch 的 193 個訓練 batches、771 個 anchor IDs 與 12 次 optimizer opportunities 均有保存證據，但沒有完整的 epoch 結束紀錄；不得把「training loop 已走完」當成「validation、checkpoint 與 epoch 結束已完整保存」。

H4 `last.pt` 確認是 0 bytes，未嘗試反序列化，也未刪除或替換。H4 `best.pt` 是可讀且保存 tensors 有限的研究產物，對應 **epoch index 292 / 第 293 epoch**，不是第 297 epoch 的最後訓練狀態。它缺少完整 exact-resume 必要狀態。因此分類為 **B：VALID_BEST_CHECKPOINT_ONLY_NOT_EXACT_RESUME**。

本階段沒有訓練、續訓、重試訓練、forward、inference、final operational evaluation 或 GPU 研究執行；沒有計算任何 768 vs 1024 的 Recall / mAP 效果差異。

## 二、證據保存與方法

先建立 `experiments/results/f_higher_scale_v2_interruption_audit/source_snapshot.json`，收錄 C4、H4、pair summary、preflight 四個目錄，共 **92 個原始檔案**的絕對路徑、大小、建立時間、修改時間及 SHA256。另保存 **2,674 個 protected inventory entries**（包含重複參照），涵蓋歷史封版、設定、資料、初始化權重及執行來源。共 2,766 次來源／protected 前後雜湊核對，沒有來源內容變更、刪除或新增。

這是**唯讀的內容雜湊與中繼資料清冊，不是完整位元組備份，也未更改原檔案的唯讀屬性**。F0／F0.1／F0.2／F1 不重新封版或改寫。最終校驗明細在 `integrity.json`；F1 protocol 與授權原文另與 execution freeze 的雜湊交叉核對。

鑑識方法遵循 diagnosing-bugs 的證據與反例檢查，但依本階段限制，明確跳過訓練重現、修復與因果干預。檢查依據為保存檔案及 installed stock writer 的程式語義，而非重新執行模型。

Checkpoint 檢查順序：檔案大小與 SHA256 → ZIP container CRC → 隔離 CPU 程序反序列化 → metadata/state/tensor finite 檢查。子程序封鎖 GPU 初始化、model call、backward、optimizer step、save、持久化檔案寫入、網路及子程序啟動。`last.pt=0` 在反序列化前就排除。

初期子程序在依賴套件的 NUL／暫存目錄探查及既有目錄 `mkdir(exist_ok=True)` 上被唯讀保護擋下；僅調整新的 F1.1 reader，允許無持久化副作用的探查。Ultralytics 對 settings/cache 的寫入嘗試仍遭封鎖。最終成功程序的 stdout 曾夾帶「寫入遭拒」訊息，後續只是重解析已保存 stdout 的 JSON，沒有為此再載入 checkpoint。所有前次鑑識嘗試均保留在 `checkpoint_integrity.json`，沒有修補任何模型。

## 三、Epoch 與部分進度交叉核對

| 證據 | C4 | H4 | 限制 |
|---|---:|---:|---|
| 完整 epoch ledger | index 0–299，300 epochs | index 0–296，297 epochs | 不能以 partial 批次替代完整結束紀錄 |
| results.csv 有效列 | 300 | 297 | H4 第 299 實體行含 177 NUL bytes，不當作 epoch 298 |
| TensorBoard 有效 step | 至 300 | 至 297 | H4 offset 696222 後 CRC 不合法；尾部 347 bytes 不解析為事件 |
| training completion | 300，存在 | 不存在 | H4 不具 completion receipt |
| 最新 pair status | — | completed_epochs=297 | 最後 at=08:20:07 UTC；舊 TRAINING_H4 不是仍存活的證據 |
| Console 最新進度 | 300/300，193/193 | 298/300，193/193 | H4 後接 validation 0/24；其後有 4,000 NUL bytes |
| best checkpoint | 完成後已 strip，epoch=-1 | index 292，第 293 epoch | metadata 不是 completed epoch count |

C4 `best_epoch=281` 與 `last_epoch=300` 來自 training completion。其 best/last 內 `epoch=-1`、optimizer/EMA/update counter 清空，符合 stock `strip_optimizer` 行為；不能據此說 C4 未訓練或 checkpoint 損壞。

### H4 epoch 298（index 297）

- 已保存 193 筆 raw-loss batch 紀錄，index 0–192；anchor batch 同為 193 筆。
- 保存的 anchor IDs 共 771 個，逐一符合該 epoch 預先凍結 sequence 的 prefix；這次已保存 prefix 長度恰等於完整 771 個 IDs。
- Console 193/193 支持 training loop 已走完；最後可觀察的訓練 batch index 為 192（第 193 批）。這不保證完整 epoch 或磁碟寫入交易完成。
- 12 次 optimizer opportunities：batch indices 5、21、37、53、69、85、101、117、133、149、165、181。
- 12 applied、0 skipped、0 unknown。最後 optimizer opportunity：batch index 181、global_batch=57502，scale_before=scale_after=128。
- raw-loss/anchor/optimizer 單筆紀錄沒有 wall-clock timestamp；精確最後事件時間為 **UNKNOWN / NOT_RECORDED**，不能拿檔案 mtime 代替。
- partial 整段之模型、梯度累積及 optimizer/scaler/RNG 狀態沒有完整原子快照；未保存區段仍屬 UNKNOWN。

## 四、Checkpoint 完整性

| 檔案 | bytes | ZIP/結構 | Tensor 檢查 | 判定 |
|---|---:|---|---|---|
| C4 root best.pt | 45,264,092 | 可讀、CRC 通過 | 715 tensors，CPU、全部 finite | VALID |
| C4 root last.pt | 45,264,092 | 可讀、CRC 通過 | 715 tensors，CPU、全部 finite | VALID |
| H4 training/weights/best.pt | 135,025,462 | 可讀、CRC 通過 | 1,810 tensors，CPU、全部 finite | VALID |
| H4 training/weights/last.pt | 0 | 未嘗試讀取 container 或反序列化 | 不適用 | INVALID_ZERO_BYTE_FILE |

SHA256：

```text
C4 best: 62b0d871b583f029d649fd905872cfdfc9ec920f1c0f7c76171eda130ff26b60
C4 last: b5028d2b53063d7528705d33a7471ab9f1ced214d00f75764483460ee9e3f67f
H4 best: d2e49a3ad5a47b9004916101771cc03204eb76b27573b7a982acd40ba6372e98
H4 last: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

C4 兩個 checkpoint 的 SHA256 與完成時 receipt 一致。H4 last 的雜湊是空檔案雜湊，不能用來宣稱 checkpoint 正常。

### H4 best metadata/state

| 項目 | 結果 |
|---|---|
| epoch | index 292 → 第 293 epoch |
| best_fitness metadata | 1.39673，僅記錄來源 metadata，不作模型效果比較 |
| model | None |
| EMA | 存在；有效有限值權重 |
| optimizer | 存在；保存 tensors 亦全部有限 |
| scheduler | checkpoint 中沒有；外部 telemetry 有觀察值，不等於完整一致的恢復快照 |
| GradScaler | 沒有 |
| train_args | 存在，imgsz1024、batch4、epochs300、seed42、AdamW、amp=true、resume=false |
| updates | 存在，3657；是 checkpoint writer 的 EMA update counter，不等於本 audit 的 applied 計數 |
| RNG / sampler / augmentation stochastic state | 皆未保存於此 checkpoint |
| embedded date | 2026-09-28T16:13:27.921147，沒有 timezone；不自行當成 UTC |

**VALID_RESEARCH_ARTIFACT ≠ EXACT_RESUME_CAPABLE。** 依本次要求的嚴格合約，現存 artifacts 不足以 exact resume，因此判 **NO（就現存證據）**。即使以已知 seed、frozen orders、scheduler telemetry 嘗試重建，也不能補回未保存的 scaler growth tracker、各 RNG、augmentation／worker 狀態及中斷當下梯度。這不是原 F1 續訓授權。

## 五、Anchor 與 AMP / 數值安全

| 範圍 | Anchor order | Scheduled observed | Applied | Skipped | Unknown |
|---|---|---:|---:|---:|---:|
| C4 完整 300 epochs | 全部 YES | 3741 | 3731 | 10 | 0 |
| H4 完整 297 epochs | 全部 YES | 3705 | 3695 | 10 | 0 |
| H4 partial epoch298 | Prefix YES，771 IDs | 12 | 12 | 0 | 0 |
| H4 保存紀錄合計 | 同上 | 3717 | 3707 | 10 | 0 |

完整 epochs 的 observed opportunities 數與 frozen per-epoch budget 相符。保存的 optimizer events 無重複、順序單調，`applied + skipped = scheduled`；unknown=0 僅指完整保存的 events，不代表未保存尾段確定無事。

C4 最大連續 skip=8、最低 scaler=128；H4 最大連續 skip=8、最低 scaler=64。以凍結的純計數／布林 `NumericalSafety` 合約重播保存紀錄，沒有觸發 fail；此重播不是 optimizer step 或模型執行。H4 optimizer JSONL 無截斷事件。CSV、console、TensorBoard 損壞尾段保留原狀，不補為 skip，也不當作有效訓練事件。Parser 測試另外驗證 malformed JSONL 回傳 `TRUNCATED_EVENT`。

保存的 raw-loss 紀錄沒有 nonfinite loss；保存的參數檢查沒有 nonfinite parameters；console 與 numerical/optimizer 紀錄未發現指定 hard-stop 字串或 CUDA OOM / allocator failure。

```text
NO_RECORDED_PREREGISTERED_NUMERICAL_HARD_STOP_BEFORE_INTERRUPTION
RECORDED_OOM = NO
```

以上不是「數值完美」，也不是未保存區段絕無異常。AMP 跳過的 10 次更新保留為事實，不把 skipped update 自動歸因為此次關機。

## 六、主機與時間軸

| 事件 | UTC（2026-09-28） | Asia/Taipei | 證據類型 |
|---|---|---|---|
| H4 best.pt 修改 | 08:13:28.940751 | 16:13:28.940751 | 檔案 mtime |
| 6008 所述「上次意外關機」 | 08:17:45（依本機 UTC+8 解讀） | 16:17:45 | 事件訊息所述時間，與後續資料衝突 |
| H4 completed297 pair status | 08:20:07.453222 | 16:20:07.453222 | status 內 at |
| H4 最後 CRC-valid TB summary step297 | 08:20:07.601850 | 16:20:07.601850 | TensorBoard wall_time |
| H4 anchor log 修改 | 08:21:36.653113 | 16:21:36.653113 | mtime，非最後 batch 精確時間 |
| H4 numerical log 修改 | 08:21:36.804879 | 16:21:36.804879 | mtime |
| H4 console log 修改 | 08:21:44.860716 | 16:21:44.860716 | mtime |
| H4 results.csv 修改 | 08:21:46.500695 | 16:21:46.500695 | mtime，檔案有 NUL 尾段 |
| H4 zero-byte last.pt 修改 | 08:21:47.426803 | 16:21:47.426803 | mtime，不是已證實的損壞發生瞬間 |
| OS LastBootUpTime | 12:58:57.500000 | 20:58:57.500000 | Windows 系統查詢 |
| Kernel-Power event41 | 12:59:08.220036 | 20:59:08.220036 | 重新啟動後記錄先前未正常關機 |
| EventLog event6008 | 12:59:27.538336 | 20:59:27.538336 | 記錄先前意外關機 |
| EventLog service start6005 | 12:59:27.539237 | 20:59:27.539237 | 事件記錄服务啟動 |

原 supervisor PID25652 / H4 PID12604 已不存在；亦未找到命令列符合 F1 runner 的 Python 程序。不能僅看 PID 或舊 status 判斷仍在訓練。

**可以描述為「有系統證據支持主機曾非正常中斷／重啟」；不能說已確定電力、GPU、過熱、硬體損壞或人工操作是原因。** 6008 提到的 16:17:45 與存活 artifacts 的 16:20–16:21 紀錄不一致，保留此矛盾，精確 process termination timestamp 為 UNKNOWN。

正式 cause 欄位維持 `UNVERIFIED_HOST_OR_PROCESS_INTERRUPTION`，並另記 `host_interruption_supported=true`。不以單一修改時間或 Event41 排定事故機制。

0-byte last.pt 屬 **CHECKPOINT_ARTIFACT_CORRUPTION=YES**；不推出 **MODEL_STATE_CORRUPTION_DURING_TRAINING=YES**。Stock writer 將序列化資料寫入 last 再可能更新 best；沒有 crash dump／原子寫入證據，不能據此斷言確切在哪個寫入階段斷電。

## 七、隔離性與驗證

本 audit 沒有讀取／解碼研究圖片來執行模型；只 hash 既有 protected 檔案，解析已保存紀錄及 checkpoint tensor。F1 persisted status 的 `test_images_used=0`、LOCKED_TEST_USED=false、CO2Wounds_used=false、EXTERNAL_TEST_USED=false 保留。

保存的 image-access paths 再與 frozen training/validation allowlist、各 arm 生成目錄及明確 framework assets 核對；未發現越界。H4 共 3276 個已保存 access entries（3275 FROZEN_DEVELOPMENT、1 framework asset；含跨 worker 重複，不是圖片數）。此結論限於保存紀錄，不把遺失事件推算為已驗證。

沒有 final evaluation lock、paired comparison 或 pair validity 完成產物；也沒有執行任何補評估。C4 已完成但 operational evaluation 仍須保持 NOT_PERFORMED。

新增 CPU-only 鑑識測試 **32 passed**；涵蓋使用者要求的 23 類檢查及解析反例、時區、唯讀／網路／子程序封鎖、禁止覆写原路徑、資料 allowlist。結果和最後 source rehash 記錄於 `integrity.json`。初次測試有 4 個 Windows cp950 stdout 解碼失敗，修正僅在新的 F1.1 test harness，沒有碰封版訓練程式。

過程中的一則初步訊息曾誤把 H4 anchor/numerical logs 說成 0 bytes；正式 snapshot 與逐筆 parser 證明兩者有有效資料，該初步描述已更正。正確大小分別為 9,364,280 與 9,628,165 bytes。不得沿用誤讀來宣稱 anchor evidence 遺失。

## 八、32 項必答事項

| # | 問題 | 正式回答 |
|---:|---|---|
| 1 | C4 completed epochs | 300 |
| 2 | H4 completed epochs | 297 |
| 3 | H4 partial epoch | 第298，zero-based index297 |
| 4 | Partial batches | 保存193個batch的anchor/raw-loss與console193/193；epoch完整結束未保存 |
| 5 | Partial anchors | 771，實際保存IDs，非以預設數量倒推 |
| 6 | Partial opportunities | 12；applied12/skipped0/unknown0 |
| 7 | C4 best/last | 皆VALID；hash吻合完成receipt，finite tensors |
| 8 | H4 last 0bytes | YES，未反序列化 |
| 9 | H4 best完整 | VALID_RESEARCH_ARTIFACT |
| 10 | H4 best epoch | index292，第293epoch，不是297 |
| 11 | Optimizer state | best中存在 |
| 12 | Scheduler state | checkpoint無；telemetry觀察不能替代原子快照 |
| 13 | GradScaler state | checkpoint無 |
| 14 | Exact resume足夠 | NO，現存artifact缺必要完整狀態 |
| 15 | 完整epochs anchor順序 | H4 297個epochs全部YES |
| 16 | Partial anchor prefix | YES，已保存771 IDs |
| 17 | Applied/skipped/unknown | H4保存合計3707/10/0；完整297epochs為3695/10/0 |
| 18 | 最大連續skip | H4 8 |
| 19 | 最低scaler | H4 64 |
| 20 | Nonfinite loss | 保存紀錄未見，未保存區段UNKNOWN |
| 21 | Nonfinite parameters | 保存檢查未見；checkpoint tensors亦有限 |
| 22 | OOM | RECORDED_OOM=NO，非絕對排除 |
| 23 | Numerical hard stop | NO_RECORDED_PREREGISTERED_NUMERICAL_HARD_STOP_BEFORE_INTERRUPTION |
| 24 | Last0byte時間 | mtime16:21:47.426803+08；損壞實際瞬間UNKNOWN |
| 25 | 系統證據 | Event41/6008支持非正常關機；精確原因未驗證 |
| 26 | 是否host interruption | 可附限定敘述主機非正常中斷；不定性為特定硬體／供電原因 |
| 27 | Final evaluation | 未執行，包括C4 |
| 28 | 768vs1024效果 | NOT_AVAILABLE，未比較 |
| 29 | 原F1狀態 | INVALID_OR_INTERRUPTED不變 |
| 30 | 原F1 resume | NO，技術可讀不改變規範 |
| 31 | Recovery分類 | B：VALID_BEST_CHECKPOINT_ONLY_NOT_EXACT_RESUME |
| 32 | 下一步選項 | A/B/C如下，均只分析，未選定或執行 |

## 九、下一階段可預登錄的三種選項（尚未選擇）

| 選項 | 設計 | 方法學定位與限制 |
|---|---|---|
| A | Fresh C4 + fresh H4 | 最乾淨的 fresh paired replication；成本最高。保留原中斷紀錄，不回填原F1。 |
| B | Reuse frozen completed C4 + 一次 fresh H4 replacement | 節省C4成本，但屬 asymmetric replacement execution；新協定必須披露第一輪H4中斷、凍結環境／設定／資料／比較規則，避免依結果選擇替代。 |
| C | 由已驗證checkpoint restart/resume H4，另立版本 | 方法學最複雜。現有best只到第293epoch且缺exact狀態；不是「從297接到300」，更不是exact mid-epoch resume。必須先處理optimizer/scaler/RNG／重複exposure等新合約，不能混回原F1。 |

單從方法學，A的fresh pair最乾淨，B是成本折衷，C連續性問題最多；**本階段不選方案，不核准或啟動任何一項。** 若將來另案選擇恢復，需要新的預登錄與明確授權，並保留本次負面工程事件和固定預算無效的事實。

## 十、交付物與最終狀態

目錄：`experiments/results/f_higher_scale_v2_interruption_audit/`

- `source_snapshot.json`：原始來源與protected清冊。
- `timeline.json`：UTC/Taipei時間軸與未知時間、時間矛盾。
- `checkpoint_integrity.json`：檔案、ZIP、state與tensor檢查及鑑識程序歷次嘗試。
- `telemetry_integrity.json`：完整／partial epochs、optimizer、loss、CSV、console、TensorBoard、資料存取證據。
- `anchor_integrity.json`：每個完整epoch與partial prefix的實際ID比對。
- `system_interruption_evidence.json`：Windows唯讀事件、程序盤點與原因限制。
- `recovery_feasibility.json`：嚴格exact-resume判定、A/B/C方案與未選擇狀態。
- `integrity.json`：前後hash、CPU鑑識測試、final status與交付物hash。

```ini
PHASE_F11_STATUS = COMPLETE
ORIGINAL_F1_STATUS = INVALID_OR_INTERRUPTED
PAIR_FIXED_BUDGET_VALID = NO
C4_TRAINING_COMPLETE = YES
H4_TRAINING_COMPLETE = NO
FINAL_OPERATIONAL_EVALUATION_PERFORMED = false
HIGHER_SCALE_EFFECT_CONCLUSION = NOT_AVAILABLE
H4_LAST_CHECKPOINT_INTEGRITY = INVALID_ZERO_BYTE_FILE
H4_BEST_CHECKPOINT_INTEGRITY = VALID
EXACT_RESUME_TECHNICALLY_POSSIBLE = NO
ORIGINAL_F1_RESUME_AUTHORIZED = NO
RECOVERY_STATE = VALID_BEST_CHECKPOINT_ONLY_NOT_EXACT_RESUME
NEW_TRAINING_AUTHORIZED = NO
```

**STOP AFTER F1.1。沒有自動開始下一個 recovery protocol 或訓練。**
