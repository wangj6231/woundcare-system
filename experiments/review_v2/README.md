# 模型實驗審查 v2（獨立增補，不改封版管線）

2026-09-14 首次稽核報告位於：
`outputs/model_review_20260914/專案模型實驗審查與改進報告.md`。

## 可直接執行的功能

在專案根目錄執行：

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m unittest discover -s tests -p test_experiment_review.py -v
python tests/test_woundcare_inference.py

# 必須使用尚未存在的新輸出目錄。
python -m experiments.review_v2.audit_project --output outputs/model_review_NEW_ID
python -m experiments.review_v2.finalize_review --audit outputs/model_review_NEW_ID
```

不會 import 舊訓練調度器、載入模型、執行推論、打開 test 影像或重新訓練。
讀取既有 test 結果 JSON／計算其檔案雜湊只是保護與報告，不是再次測試。
所有輸出以 exclusive create 寫到新資料夾；中途失敗保留證據，不自動清理重跑。

## 模組與限制

- `audit_project.py`：清查保存的訓練 CSV／args／摘要、重算分類描述值、檢查明確允許的開發資料。
- `finalize_review.py`：祖先訓練集與目前 val 比對、學習率落差、App 預設權重 SHA256、離線測試與中文报告。
- `metrics.py`：固定類別與拒判分母、配對 ID、條件式 group-cluster accuracy CI。舊預測沒有 identity 時不得捏造 group。條件式 CI 不包含重新訓練及選模的不確定性。
- `guards.py`：供**新的**訓練入口整合的來源／角色／群組／祖先／輸出目錄／optimizer 檢查函式。不是能攔截任意舊脚本的沙箱，也不是法律授權證明。
- `run_development.py`：已整合來源／資料／祖先／optimizer guard 的專用開發入口，預設只做 preflight；訓練或評估失敗明確記 FAIL。沒有 test／blind 模式。
- `recipes/D-Seg-08R_warmup_control.yaml`：只改 bias warmup 的開發控制實驗設計；**尚未訓練**。不得直接傳入巢狀 schema 不相容的 `experiments/run_experiments.py`。

```powershell
# 目前會正確 BLOCKED：Yasin 原圖使用權證據尚未補齐，不讀像素也不訓練。
python -m experiments.review_v2.run_development --source-admission experiments/review_v2/source_admission.json --output outputs/dseg08r_preflight_NEW_ID
```

`source_admission.json` 是 fail-closed 狀態登錄，不是授權聲明。需逐來源核對實際用途、原始條款文件和 SHA256 才可改成 allowed。之後明確指定 `--train` 才啟動開發訓練；乾跑與訓練須使用不同的新目錄。不要只改布林值繞過來源證據。

## 後續訓練必要條件

1. 各來源的實際允許用途與原始證據可查；UNKNOWN 不等於 allowed。Redscar 無回覆不等於核准。
2. 固定 image／label manifest 及 checkpoint SHA256，確認實體架構、類別順序、全部祖先訓練資料。
3. 重新 Group CV 必須從可追溯 generic pretraining 開始，不使用已看過新 val 的傷口微調模型。
4. train-only sampling／augmentation；保存不同內容數與有效抽樣筆數，保持 val 原樣。
5. 訓練或評估 exception 必須記 FAIL，不得由 `log_done` 覆蓋。不得沿用已存在的 run 目錄。
6. 只用 development 選 checkpoint／threshold；48 張分類盲測、CO2Wounds 封存結果不再用來選擇新模型。

本次沒有改變 App 權重／啟動服務／推送 GitHub，也沒有聲稱模型正確率已提升。
# 最新完成：固定定位與裁切對照（2026-09-14）

`localization_benchmark.py` 在既有 FUSeg validation 191 張上完成 D-Seg-03 與 D-Seg-03R 的 28 個預先固定操作點。沒有訓練、沒有 test 存取，也沒有替換 App 模型。

主設定 conf=0.10 / NMS=0.70：舊模型 Recall 84.65%、裁切完整率 90.32%；新模型 Recall 82.99%、裁切完整率 88.17%。新模型不應只因歷史 Mask mAP50 微幅上升而自動被採用。

詳見 `docs/LOCALIZATION_EXPERIMENT_DECISION_20260914.md` 與 `outputs/localization_benchmark_20260914/實驗結果報告.md`。`verify_localization_results.py` 查核全部 764 筆預測與 5,348 筆逐圖指標。本文件的早期 review 記錄保留作歷史背景，不代表這輪仍未執行。
