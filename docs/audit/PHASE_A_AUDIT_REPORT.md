# WoundCare Phase A — 研究完整性稽核報告

日期：2026-09-20。起始版本：`7097445f1d71f19ed6257646ca6caa9952e61e60`。

**PHASE_STATUS = BLOCKED_BY_EXTERNAL_EVIDENCE**
**READY_FOR_NEXT_PHASE = NO**

本輪已完成可由現有檔案支持的程式／資料稽核、建立稽核工具及測試；但未滿足全部 Phase A acceptance criteria，**不宣稱 PHASE_A_COMPLETE**。未啟動訓練、模型推論、Bootstrap 重算或盲測。保留歷史結果與 App 預設。

## 1. 本輪範圍與證據

依使用者提供的兩份 audit/loop 指示，限定 Phase A。全檔詞彙掃描涵蓋241份版本控制與本機主要研究程式/文件（包括忽略於Git的 `work` 腳本）；對下列關鍵路徑人工追讀：

- README、PROJECT_STRUCTURE、PROGRESS_AND_METHODS、PUBLICATION_CHECK、FUSEG_MODEL_CARD、LOCALIZATION_EXPERIMENT_DECISION、ISIC_FUSEG_RECOVERY、DEPLOYMENT_CASCADE_ROI_FALLBACK。
- experiments README、run_experiments、evaluate、cross_validation_group、multi_seed、statistical_analysis、execute_bootstrap、build_all_phase9_tables、Table1–4。
- review_v2 guards、run_development、isic_fuseg_gate、localization/highres/tiled 推論路徑；woundcare_inference、woundcare_safety、API decode。
- 歷史 cascade 顏色轉換、OOF folds reconstruction、直接 test runner、split audit 與 CO2 external runner。
- 25份現存 OOF JSON、multiseed summary、bootstrap summary、raw run args/目錄內容；目前 development train/val 檔案摘要。

[完整掃描範圍、檔案摘要與命中行號](../../experiments/results/audit/phase_a_20260920/repository_source_inventory.json)。這是全檔靜態搜尋與關鍵路徑審查，**不是全專案每一行均已人工驗證，也不是所有動態匯入／未記錄執行歷史的形式化證明**。不讀資料庫、金鑰、test/CO2圖片；第三方套件僅讀相關輸入程式與做合成載入測試。

## 2. 3622 對 3600：目前確切知道什麼

| Seed | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | 合計 |
|---|---:|---:|---:|---:|---:|---:|
|42|150|133|145|147|145|720|
|123|145|155|141|133|146|720|
|3407|147|140|152|142|139|720|
|2026|155|146|140|141|138|720|
|999|137|133|148|139|163|720|
|總计||||||3600|

以上皆由現存JSON陣列實際計數，並與各fold儲存的val_size相符。每seed的y_true類別數都是105、109、102、102、100、103、99。

直接證據：

1. `C-Arch-05-MS_bootstrap_ci_report.json` 的 `total_predictions` 和 `bootstrap_results.total_samples` 均寫3622。
2. `build_all_phase9_tables.py:71` **直接寫死** `pooled n=3622`；CI也為literal，不是由canonical資料計算。
3. 現行 `statistical_analysis.py:246` 寫 `len(y_true)`，並於256行輸出 `fold_details`；現存舊report沒有這個欄位。因此**無法用現存輸入與程式解釋舊report的完整生成過程**。
4. 可見Git歷史只保留重置後的匯入版本，沒有可追溯原始生成3622的完整執行版本／輸入快照。

**判定：這是「儲存的統計摘要與實際預測列數矛盾」，不是已發現22筆可以刪除的重複列。** 表格如何傳播3622已定位；原summary最初為何寫3622仍 UNKNOWN，不能推論有人造假、確有22筆多餘預測，或擅自修成3600就算核銷。本輪沒有刪除任何列。

## 3. Canonical OOF 是否可重建

目前 development 再驗證結果：679 train＋41 val＝720檔；383 MD5群組；177 singletons；206 duplicate groups涵蓋543檔，337額外複本。

採**檔案×seed**為row unit，預期canonical為720×5＝3600列；不是383×5。相同內容的不同檔案仍需各自有身份，再由群組資訊支援相依抽樣。若改為內容組等權估計，必須另定estimand，不能默默換分母。

然而所有25份舊export都只含 `y_true/y_pred/y_prob/class_names/metrics`，沒有image path/ID/hash、逐列checkpoint。`evaluate.py` 的影像glob未排序，且只回傳陣列；split collector亦未排序，fold暫存於結束後刪除。保留的args只記暫存資料路徑，權重存在不能還原每個prediction row的身分。

舊41張cascade工具以**當前**資料列舉重跑SGKF，再映射MD5到fold；這不是原始720張ordered prediction manifest，也不能據此證明每個歷史row的順序。即使y_true順序與fold大小相符，相同類別內仍可能有多種對應。

| 驗證項目 | 實際狀態 |
|---|---|
|25份檔案／每seed720列|CONFIRMED_COUNT_ONLY|
|陣列長度、類別順序、有效label／probability、argmax、n_samples|PASS_NUMERIC_ONLY；機率和容差0.002|
|每image每seed恰一次|UNKNOWN，缺row→image證據|
|duplicate/missing/unknown prediction IDs|**null／不可確認，不是0**|
|歷史每fold train/val群組成員|只有摘要聲稱0 overlap；本輪不能逐成員重驗|
|canonical schema|已產生|
|canonical master CSV|**未產生**，避免捏造|

[OOF機器報告](../../experiments/results/audit/phase_a_20260920/classification_oof_audit.json)、[OOF摘要](../../experiments/results/audit/phase_a_20260920/classification_oof_audit.md)、[schema](../../experiments/results/audit/phase_a_20260920/classification_oof_schema.json)、[目前檔案身分清單](../../experiments/results/audit/phase_a_20260920/sample_identity_manifest.csv)。此清單不能當作歷史OOF master。

## 4. 歷史 Bootstrap 方法與正式裁定

現存程式將25份預測合併，固定 `default_rng(42)`，每次以 `rng.integers(0,n_samples,size=n_samples)` 有放回抽**列**，B=2000；用percentile取95%上下界。報告mean是bootstrap分布平均，std是該分布的標準差。現存實作沒有BCa校正。

沒有保持MD5群組完整，也沒有把同影像跨seed預測綁在一起；不同fold訓練資料重疊也不能變成25個獨立資料集。因此舊CI不能當作正式群組泛化不確定性。

**Verdict：`SUPERSEDED_STATISTICS / HISTORICAL ROW-WISE BOOTSTRAP`。** 此為新增稽核登錄，不覆寫舊JSON/Table3；目前**沒有**新版CI可以取代。不能把本輪裁定寫成「grouped Bootstrap 已完成」。保留舊87.40±2.78等歷史值，不重新選模型；±fold/run SD、seed SD、bootstrap CI及patient independence必須分開。

後續只有先取得canonical mapping，才可執行Phase B：固定主要estimand，計算每seed完整OOF、共用群組重抽樣保留seed相依性、B≥5000（可採10000）及固定bootstrap seed。MD5 cluster CI仍只條件於已觀察內容群組／已訓練模型，不能自動涵蓋新病人或重訓流程的所有變異。

## 5. RGB/BGR 完整已確認路徑

已安裝Ultralytics：`data/loaders.py` 的 `LoadPilAndNumpy` 對PIL做RGB→BGR、對NumPy保留；`engine/predictor.py:126` 再BGR→RGB；分類predictor也在46行以BGR→RGB轉換。以三個不同通道值的合成影像，PIL與正確BGR NumPy一致，直接RGB NumPy不同；無載入權重或模型推論。

**主缺陷：** `isic_fuseg_gate.execute_gate`（96–98）→ PIL RGB → NumPy RGB → `localization_benchmark.predict_materialized`（269起、275呼叫model）→ `model.predict` → loader保留NumPy → predictor再當BGR反轉。baseline在387／420行先RGB2BGR，與candidate不公平。

| 路徑 | 色彩輸入稽核判斷 |
|---|---|
|最新ISIC→FUSeg development gate|CONFIRMED mismatch，原FAIL保留但公平比較未完成|
|localization benchmark GPU/CPU、highres、tiled|有顯式RGB2BGR，未發現同一缺陷|
|API `backend_main.py:656` → inference|cv2.imdecode產生BGR；維持BGR，未發現同一缺陷|
|分類Phase7 `evaluate.predict_yolo`|傳影像檔名字串，不是此RGB NumPy缺陷|
|歷史`evaluate_dseg06_carch05_oof_cascade.py`|full image、ROI與segmenter有RGB NumPy直送，受影響|
|`evaluate_dseg09b_carch05_oof_cascade.py`|全圖/ROI顯式`[:,:,::-1]`；segmenter用檔名，與舊D06需區分|
|`evaluate_classifier_full_image_val.py`、`evaluate_segmentation_cascade_fallback.py`|RGB NumPy直送classifier，受影響|
|`run_automatic_cascade_eval.py`、`run_clinical_bbox_cascade_inference.py`、`run_segmentation_cascade_eval.py`|ROI RGB NumPy直送classifier，受影響|
|`run_cascade_threshold_sweep.py`|crop/full image RGB NumPy直送，受影響|
|production cascade val、clinical seg cascade|read_image使用cv2.imdecode BGR，未發現同一缺陷|
|CO2 historical external evaluator|model.predict以image path輸入；未因本次發現直接判為同一色彩缺陷|

判定範圍是已追讀的呼叫路徑，不代表其他指標、權重或ground truth都已正確。舊cascade結果可能已有後續修訂版，不可將所有版本一起判成失效。原ISIC候選45.59/12.86/20.06%低分與FAIL不刪除；本輪未測得色彩修正能改善多少，不能保證會過gate。

**需要重訓嗎？** 現有證據不支持為這個評估色彩缺陷重訓；Phase C應固定checkpoint、191張val、所有threshold/NMS/IoU/crop規則，另建結果目錄只修輸入契約，再量測影響。本輪沒有执行此項。

## 6. Locked 48 是否仍完全隔離

本輪明確為 `test_images_used=0`，沒有列舉／讀取封存影像，也沒有開啟原始431全集。這**不能倒推所有歷史執行都未越界**。

| 程式入口 | 已確認風險／界線 |
|---|---|
|run_experiments CLI|雙旗標只保護此CLI；`run_single(eval_split='test')`可直接呼叫，且同一流程會先train|
|`experiments/scripts/evaluate.py`|直接CLI/API接受任意data_dir；`split`只命名，沒有雙鎖或一次性token|
|`test_cls_model.py`|直接model.val(split='test')，無雙鎖；本輪禁止執行|
|`run_experiments --audit` → audit_split|即使development模式也會呼叫test isolation hashing；會讀test bytes，不能在本輪使用|
|舊classification source/data CLI|任意資料路徑，沒有全域已封存content hash角色封鎖|
|新固定FUSeg/ISIC runners|硬指定train/val並核對manifest；比generic library更嚴，但不是全域sandbox|

結論：本輪未使用test；**無法宣稱所有非final code path不會引用test，也無完整歷史存取日誌可證明永未誤用**。新validator支援保護路徑及既有trusted hashes；沒有trusted inventory時明列NOT_VERIFIED，沒有為此讀test圖片。

## 7. CO2Wounds 是否已完全封存

研究角色固定為 **HISTORICAL_EXTERNAL_BENCHMARK**，不得development/selection/tuning。文件及現行FUSeg/ISIC專用流程排除它。

但generic `review_v2.guards.validate_new_development_run` 只信任 caller supplied `development_allowed`＋存在的evidence file。使用暫存**合成**證據檔，傳入CO2名稱可以通過generic guard，已實際重現。這不代表現行專用runner真的讀取CO2；專用runner另有固定資料來源及hash檢查。

本輪新增validator會拒絕CO2名稱／路徑、historical role及傳入的protected hashes，但**未接線到所有歷史runner**，不是操作系統存取控制，也無法識破無可信hash可比對的任意改名拷貝。故「CO2絕不可能進development」尚未驗收，不能用新測試通過替代全域封存。

## 8. 其他方法／來源發現

- Table1把增補後768檔稱為Total Raw Images，與原始431說法不同；337是額外複本，不是所有重複群組內檔案543。保留原表，正式發表前需使用有版本的新更正說明。
- 目前720檔仍是平衡後的評估分布；組內重複檔會影響weighting，不等於自然就診分布。
- 原分類發布者/URL/license/version/download date/archive hash缺原始證據。[來源稽核](../CLASSIFICATION_DATASET_PROVENANCE.md)。非商業不是自動授權。
- Patient/case/video/derived lineage/perceptual group仍未知；不能將MD5=0 overlap改寫為patient-independent。
- 舊orchestrator/multiseed可在evaluate失敗後仍log_done；`execute_bootstrap.py`及`build_all_phase9_tables.py`有import即執行的副作用，可能覆寫歷史輸出。本輪未import或執行它們。
- OOD、small wound、dual-view、臨床ground truth與新external source仍是後續研究，不因本輪工程測試通過而完成。

## 9. 實作與驗證

新增的5個模組：OOF稽核、manifest isolation validator、Phase A artifact builder、Phase A engineering verifier、saved-metric consistency audit。最終新增18項測試，無既有檔案內容改動；沒有更新原CI、模型、訓練設定或App。再稽核發現新validator對完全缺少內容摘要的row可能誤判通過，已先重現failure再拒絕；使用者要求繼續後，又查出歷史resume混用recall欄位（第12節）。verification、r2、r3均保留。

| 驗證 | 結果 |
|---|---|
|targeted audit tests|18 passed / 0 failed / 0 skipped；2則第三方棄用警告|
|related review module tests|38 passed / 0 failed / 0 skipped；0警告輸出|
|完整synthetic tests目錄|100 passed / 0 failed / 0 skipped；0警告輸出|
|standalone synthetic cascade|exit0；不是另一個已計數unittest集合|
|歷史設定／程式／OOF／統計／表格／相關權重快照|207份，前後變更0|

各test集合互相重疊，**不可相加**。完整性快照只涵蓋列出的207份，不表示所有硬碟資料皆已hash。`test_cls_model.py`因會讀盲測故不執行；`experiments/test_phase1.py`會修改shared experiment log故不執行，並非隱藏略過失敗。

[最終測試原文及audit code hashes（r3）](../../experiments/results/audit/phase_a_20260920/verification_tests_r3.json)、[保留的r2驗證](../../experiments/results/audit/phase_a_20260920/verification_tests_r2.json)、[保留的第一輪驗證](../../experiments/results/audit/phase_a_20260920/verification_tests.json)、[安全核對](../../experiments/results/audit/phase_a_20260920/safety_verification.json)、[前快照](../../experiments/results/audit/phase_a_20260920/historical_before.json)、[後快照](../../experiments/results/audit/phase_a_20260920/historical_after.json)、[loop紀錄](LOOP_PROGRESS.md)。

重現時從專案根目錄執行（只能給全新目錄，不覆寫本次產物）：

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m experiments.phase_a_audit --output experiments/results/audit/phase_a_NEW_ID
python -m experiments.phase_a_verify --audit experiments/results/audit/phase_a_NEW_ID
```

只讀development bytes與歷史metadata、執行合成測試；不是允許重跑模型或test的指令。

## 10. Acceptance 與停止理由

| 必要條件 | 狀態 |
|---|---|
|實際3600、3622在報告／表格的所在位置|已確認|
|原3622生成的確切輸入／版本／原因|**BLOCKED**，缺原執行證據|
|每seed逐image coverage，duplicate/missing/unknown IDs|**BLOCKED**，缺ordered row mapping|
|canonical schema|完成；master不得臆造|
|歷史抽樣方法與CI有效性裁定|完成；新版CI未執行|
|主RGB/BGR路徑及已知受影響入口|已定位；修復與受控重評估留Phase C|
|locked48所有歷史存取／所有動態入口完全隔離|未證明；已列繞過風險|
|CO2全域禁止development|未完成全入口enforcement；新validator通過不等於全域完成|
|來源／病人／派生關係證據|**BLOCKED**，不從檔名猜測|
|報告、機器產物、相關工程測試|已產生並驗證|

遵照STOP-B，在必要原始mapping/provenance無法從保留檔案恢復後停止；同時明列尚未接線的程式風險，不把它們偽裝成已修復。繼續反覆抽樣或重訓不能補回遺失的historical identity。

## 11. 最小下一步（未執行）

1. 取得原始有順序的每fold validation清單、train/val群組成員、checkpoint mapping，以及bootstrap report生成輸入快照；來源原始證據另外補齊。若無法找回，正式保留「historical OOF不可重建」，不可聲稱已修正原CI。
2. 為**新流程**統一固定source-role registry與train/val檔案allowlist、內容hash封鎖、不可覆写輸出；歷史scripts只保留證據，避免從通用入口啟動。既有分類test不再解鎖。
3. 證據通過後才授權Phase B的新統計；Phase C另建版本，以同checkpoint及固定development cohort修正顏色契約。均不需要先重訓，亦不得進5 seeds或換App模型。

### 研究結論變動與不變

變動：舊CI正式被本稽核標示不適合作為群組正式CI；「多22列」改為「摘要數字矛盾，未找到22實際多餘列」；ISIC gate與部分舊cascade需色彩輸入更正後才能公平比較。

不變：Naive CV的失效紀錄、既有GroupCV點估計、一次性48張結果、CO2歷史external結果、原FAIL與所有權重均保留。本輪沒有產生任何更高的準確率、替換模型、臨床安全或外部泛化通過宣稱。

## 12. 使用者要求繼續後：新增刺傷 Recall 欄位混用發現

本節是後續Phase A forensic audit，不是Phase B的新OOF統計，也不是重新跑模型。它更新前述「數值未更動」的解讀：歷史檔案仍原樣保留，但**原Stab_wound Recall彙總不得再當作已核驗的刺傷指標引用**。

### 12.1 逐份陣列算術核對

新增 `experiments/audit_classification_metrics.py`，從保存的y_true/y_pred核對各fold的Accuracy、Macro Precision/Recall/F1、Weighted-F1與各類Precision/Recall/F1。固定類別、沿用歷史zero_division=0與兩位小數語義；不重新推論，不依據這些數字選模，不產生CI或seed-level OOF性能報告。

| 核對層級 | 結果 |
|---|---:|
|25份prediction JSON內指標 vs 同檔保存陣列|不一致0份（限上述檢查指標，不含ROC-AUC）|
|multiseed summary的每fold Accuracy/Macro-F1/Weighted-F1 vs 陣列|不一致0項|
|multiseed summary的每fold Stab_wound Recall vs 陣列|**17/25項不一致**|
|這17個錯值是否等於同fold的Macro Recall|**17/17是**|

受影響：seed42的fold1–5、seed123的fold1–5、seed3407的fold1–5、seed2026的fold1–2；seed2026後3fold與seed999這一欄沒有發現同樣差異。

具體例子：seed42/fold1保存的刺傷GT支援數17，正確12，故刺傷Recall=70.59%；summary卻寫84.18%，這正是該fold的Macro Recall。不可把macro平均當成單一類別。

### 12.2 程式根因與證據界線

- `experiments/scripts/experiment_logger.py:41,84`：CSV欄位`recall`明定／儲存的是`macro_recall`。
- `experiments/scripts/multi_seed.py:127`：Resume從CSV重建已完成fold時，卻將`float(row.get('recall') or 0)`指定給`stab_wound_recall`，註解為`best proxy`。
- 同程式新完成fold的280行則正確讀`metrics.get('stab_wound_recall', 0)`，導致摘要可能混合两種不同定義。

這是一條明確的欄位語義錯誤，並與17份保留資料的錯值一致。原始每一次resume的執行歷程仍未完整還原；本輪不假定其啟動時間或操作人。

**判定：歷史 Stab_wound Recall 88.98% ± 9.30% 及依其彙總衍生的分類主張，標為 `HISTORICAL_METRIC_SEMANTICS_ERROR / NOT_VERIFIED_AS_STAB_RECALL`。** 保留原數字／檔案供追溯，不用新的猜測數值取代。原有CI仍維持不得正式採用的判定。

### 12.3 產物與下一步

[逐fold、逐指標、整数分子分母、來源SHA256與差異清單](../../experiments/results/audit/phase_a_20260920/saved_metric_consistency.json)。26份輸入（25預測＋summary）前後SHA256相同。缺row/image mapping仍不允許canonical OOF宣稱。

下一步最小程式修正應是：新版本resume只接受有明確per-class欄位／經驗證的同fold prediction metrics；缺欄位就拒絕，禁止以macro proxy或0填補。修訂summary需新檔名、保留舊版並列差異；目前限定Phase A，未修改封版multi_seed或重算正式統計。
