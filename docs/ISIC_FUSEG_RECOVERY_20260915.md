# ISIC → FUSeg 實驗接續紀錄（2026-09-15）

最新狀態：FUSeg 300 輪已完成，固定工作點驗收為 `FAIL_DEVELOPMENT_GATE`，五種子未啟動。本次另發現候選驗收直接傳入 RGB NumPy，而基準採 BGR，需先修正評估再判斷效能；不能把這份低分當公平比較結論。完整數值、程式依據及後續驗收條件見[最新報告第6節](PROGRESS_AND_METHODS_20260915.md#64-最新低分出現後我核對到了什麼)。以下保留原接續過程。

## 中斷原因與修正

ISIC 輔助預訓練已完成第 1–80 epoch，最佳 checkpoint 對應第 60 epoch；連續 20 epochs 未改善，符合原定 patience=20。
2026-09-15 02:35:27（台灣時間），訓練返回後的程式將原本從 1 開始的 CSV epoch 再加 1，誤判為不連續並停止。因此 FUSeg 尚未啟動。

修正僅涉及新實驗的紀錄檢查與復原／評估流程，原 Phase 1–9 及原結果檔保留。原 failure.json、execution.lock 保留作為異常證據；本次新增 continue_fuseg.lock 和獨立日誌，不重新訓練 ISIC。

## 接續實驗

- 初始模型：已完成的 ISIC formal best.pt，SHA256 `1ec926026473beafafb1ef8da6aa6efcc13d38c391ec8db48006c5d793d5f2a3`。
- 來源：FUSeg；771 張 train、191 張 validation；class 0 = Wound。
- 設定：seed=42、最多 300 epochs、patience=80、imgsz=768、batch=4；其餘超參數沿用已確認 protocol。
- 已逐檔比對 ISIC 2,000 張與 FUSeg 962 張影像及其 polygon labels 的既有 SHA256；各資料集 train/val 的 exact-content overlap=0，ISIC exposure 與 FUSeg val exact-content overlap=0。
- 此核對不代表已證明跨資料集 pHash 或病人層級隔離。ISIC 分割為既有排序前 1,800 張／後 200 張，僅為輔助預訓練，不作 wound performance 指標。
- 逐檔核對及 sealed snapshot 通過後，才啟動接續。

## 訓練完成後的自動驗證

使用同一組 191 張 FUSeg development validation 與既有評估程式，固定 confidence=0.10、NMS IoU=0.70、bbox match IoU=0.50、裁切每側增加 15% 邊距。

|指標|已確認的最低門檻|
|---|---:|
|Precision|87.18%|
|Recall|84.65%|
|F1|85.89%|
|保留至少 95% 傷口像素的正樣本比例|90.32%|
|GPU batch=1 平均定位耗時|≤50 ms/image|

百分比按已確認的兩位小數精度比較，避免相同基準因四捨五入差異被錯誤判定失敗。速度涵蓋定位推論與結果 materialization，排除磁碟讀檔、HTTP 及分類器。

未全部通過：停止擴跑，保存逐圖預測、error cases、錯誤案例圖與門檻報告。全部通過：標記可進入五種子穩定性階段；此 runner 不直接啟動五種子或部署 App。

## 安全與稽核

- 僅開啟固定 train/val；test_images_used=0。48 張分類盲測與已評估 CO2Wounds 的影像均不作本輪輸入。
- 保存封版檔案 hash 與既有 CO2Wounds 報告 hash 只用於確認未變，不重新評估該來源。
- 保留舊模型；ISIC checkpoint 僅作 lineage；候選資格須由 FUSeg 最終 checkpoint 接受開發門檻驗證。
- 尚未完成新來源測試；development 結果不構成跨來源泛化或部署通過證據。

## 驗證與查閱

已重現原 epoch 檢查錯誤，修正後實際 80 行歷史通過；82 項測試通過（含 9 項新回歸測試），涵蓋 epoch 缺號／重複／小數、nonfinite loss、YAML test 路徑、檔案變更、未登錄圖片與 exact-content overlap。

輸出根目錄：`outputs/isic_fuseg_formal_seed42_20260914/`。

- `status.json`：程序 PID、更新時間、階段與完成 epoch。
- `continue_fuseg.lock`：復原證據、逐檔驗證摘要、runner hash。
- `continue_fuseg.stdout.log`、`continue_fuseg.stderr.log`：持續訓練日誌。
- `runs/fuseg_finetune_formal/results.csv`：訓練歷史；`weights/best.pt`：候選權重。
- `result.json`：正式 validation 結果，須完成後才產生。
- `development_gate/實驗結果報告.md`：自動開發門檻報告，須完成評估後才產生。

日誌出現 `/48` 表示 191 張 validation、batch=4 的 **48 個批次**，不是使用 48 張盲測。
