# Phase A audit loop — append-only log

本輪授權：audit only；不執行 Phase B/C、不训练、不推論、不讀 locked 48 或 CO2Wounds 圖片。開始時 Git 工作樹乾淨，HEAD `7097445f1d71f19ed6257646ca6caa9952e61e60`。

## Iteration 1 — 2026-09-20

- Issue / Priority：P0 OOF 身分與 3622/3600 記錄矛盾。
- Evidence：25 份 JSON 合計 3600；歷史 bootstrap summary 記 3622。預測只含 y_true/y_pred/y_prob/class_names/metrics，沒有逐圖 ID。
- Root cause：exporter 丟失影像身分；Table 3 builder 將 3622 與 CI 寫死。原 bootstrap artifact 如何形成仍缺原始執行證據。
- Minimal fix / Files：新增 `experiments/audit_classification_oof.py`、`tests/test_phase_a_audit.py`，以公開稽核介面區分數量與身分完整性。
- Expected unchanged：舊 exporter、預測、weights、CI、表格全部保留；不默默刪除 22 列。
- Acceptance：無 ID 時 duplicate/missing/unknown 必須為 null；有 ID 的合成 fixture 要找出重複、遺漏、未知與 group 跨 fold。
- Tests：兩次新增功能先因介面不存在失敗，再通過；本輪結束 2 passed / 0 failed / 0 skipped。
- Result：稽核語義 RESOLVED；真實 OOF coverage 仍 BLOCKED。
- Research impact：不能再把「筆數相同」當每圖一次的證明；不更改模型點估計。
- Safety：locked_test_used=false；CO2Wounds_used_for_tuning=false；historical_results_deleted=false。
- Remaining risk：無 ordered historical image manifest，不能憑當前 glob/SGKF 重建可信逐列 mapping。
- Next issue：known-identity isolation、protected paths、source role。

## Iteration 2 — 2026-09-20

- Issue / Priority：P0 test/external 的保護不覆蓋所有程式入口；P1 source/identity 欠件。
- Evidence：`evaluate.py` 接受任意 data_dir；split 只命名；舊 orchestrator 的雙鎖只存在 CLI；generic guard 信任傳入 source_gates。
- Root cause：保護依賴呼叫端；沒有全專案不可繞過的資料角色／存取層。
- Minimal fix / Files：新增 `experiments/validate_dataset_isolation.py` 與 `experiments/phase_a_audit.py`，在新稽核介面拒絕 known CO2、test path、allowlist escape、protected hash 與已知跨 split 身分。
- Expected unchanged：本輪不接管舊 runner，不修改封版程式，不啟動實驗。新 validator 通過不等於全專案已強制隔離。
- Acceptance：明列未知病人 ID；受保護內容可由已凍結的摘要清單比對，不為取得摘要而讀 test。
- Tests：CO2 新介面先 red，再 green；擴充至 12 passed / 0 failed / 0 skipped，包含「test sentinel 絕不可被開啟」。
- Result：產生720列目前 development 身分清單；383 groups；207份歷史程式/設定/結果/權重前後摘要相同。
- Research impact：現有 development counts 再驗證；來源仍 UNVERIFIED。不是重建歷史 folds，也不是證明病人獨立。
- Safety：locked_test_used=false；CO2Wounds_used_for_tuning=false；historical_results_deleted=false。
- Remaining risk：原始來源／授權、patient/derived family/pHash groups、歷史 mapping 欠件；未全域接線的舊入口仍需後續隔離。
- Next issue：RGB/BGR 路徑與統計有效性正式裁定、完整工程測試。

## Iteration 3 — 2026-09-20

- Issue / Priority：P0 evaluation correctness / statistical correctness。
- Evidence：candidate gate 將 PIL RGB 轉 NumPy 後直接送 predict；baseline 先 RGB2BGR。安裝版本 loader 的 NumPy 原樣保留、PIL 反轉通道；合成三通道實測重現差異。歷史 bootstrap 對 pooled rows 逐列抽樣。
- Root cause：不同輸入型別契約未统一；統計抽樣單位忽略 exact duplicates 與跨 seed 重複評估。
- Minimal fix / Files：新增色彩契約稽核測試、`experiments/phase_a_verify.py`；不修模型推論路徑、不產生新版CI。正式裁定與影響面寫入 Phase A 報告。
- Expected unchanged：原 FAIL、原 CI、App default、thresholds 與 checkpoints 不變。
- Tests：targeted 13 passed；related 38 passed；整個 `tests/test_*.py` 95 passed；各組0 failed / 0 skipped。另 standalone synthetic cascade exit0。這些是重疊集合，不相加成146項。
- Warnings：targeted 2 則 thop/distutils DeprecationWarning；related/full/standalone 沒有警告輸出。完整 stdout/stderr 已保留。
- Result：PASS_ENGINEERING_ONLY；207歷史檔案前後未變。合成舊 guard probe 接受虛構 CO2 admission，確認其不是全域封存政策；沒有讀取CO2資料或訓練。
- Research impact：舊CI定為 HISTORICAL/SUPERSEDED_STATISTICS，無新版CI；原ISIC FAIL不能當公平色彩輸入下的最終比較。
- Safety：locked_test_used=false；CO2Wounds_used_for_tuning=false；historical_results_deleted=false。
- Remaining risk：真實 canonical OOF 無法由缺ID陣列還原；原3622報告生成紀錄與來源證據不可用。RGB/BGR實際修復及重評估是後續Phase C。
- Next issue：整理接受條件與外部證據阻擋，做最終只讀複核。

## Iteration 4 — 2026-09-20

- Issue / Priority：P0 再稽核新validator對缺乏exact-content identity的row可能給出passed。
- Evidence：新增 `test_missing_content_identity_cannot_pass_isolation_audit`，先重現1 failure（其餘13通過）；有source/path但沒有MD5/SHA256時不應通過。
- Root cause：第一版檢查已知群組重疊，卻未要求至少一個可比對內容身分。
- Minimal fix / Files：`validate_dataset_isolation.py`新增MISSING_EXACT_CONTENT_IDENTITY拒絕；verifier以iteration命名保留舊結果，不覆寫第一次verification。
- Expected unchanged：實際720列已全部有MD5/SHA256，先前實際數量與隔離稽核結論不变；歷史實驗不變。
- Acceptance：缺exact identity必須失敗；已知內容／病人／影片／家族與test/source拒絕仍通過回歸測試。
- Tests：最終targeted14 passed、related38 passed、完整synthetic tests96 passed；全部0 failed / 0 skipped。targeted2則第三方棄用警告，其餘無警告輸出；standalone synthetic cascade exit0。
- Result：RESOLVED_NEW_AUDIT_GAP；207歷史檔案摘要仍無變更。舊版與r2測試輸出皆保存。
- Research impact：只強化新稽核器，不改模型分數／CI；沒有資料時不虛構canonical master。
- Safety：locked_test_used=false；CO2Wounds_used_for_tuning=false；historical_results_deleted=false。
- Remaining risk：原3622產物生成證據、historical row/image mapping、來源授權及patient/lineage缺原始證據；既有未接新validator入口仍列為未解風險。
- Next issue：STOP-B：BLOCKED_BY_EXTERNAL_EVIDENCE；READY_FOR_NEXT_PHASE=NO。先取得原始證據與版本化後續修正範圍，不自行執行Phase B/C或訓練。

## Iteration 5 — 2026-09-20（使用者追加「繼續」）

- Issue / Priority：P0 stored metric semantics；繼續Phase A可獨立驗證的逐fold算術，不跨入Phase B/C。
- Evidence：seed42/fold1 prediction裡Stab Recall=70.59%，summary卻84.18%，恰為Macro Recall。擴查25fold後17fold同樣混用，全部17錯值等於Macro Recall。
- Root cause：logger欄位recall是macro_recall；multi_seed.py的resume第127行將它指定成stab_wound_recall，新完成fold則使用真per-class欄位。程式確有語義bug，保存數字符合此機制；不捏造原resume執行時間。
- Minimal fix / Files：新增saved-metric forensic audit與4項測試；不是修改封版resume，不產生正式替代CI／模型成績。新增report第12節與版本化verification r3。
- Expected unchanged：25prediction、multiseed summary、log與所有weights保留，不用新的猜測彙總替換88.98%。
- Acceptance：合成fixture中macro與class Recall不同時必須報錯；輸出錯值、正確陣列算術、每類整數分子/分母與來源摘要；無test或推論。
- Tests：新介面先red再green；最終targeted18 passed、related38 passed、完整synthetic tests100 passed，全部0 failed / 0 skipped。targeted2則第三方棄用警告；standalone synthetic cascade exit0。
- Result：25份prediction內已核對指標與保存陣列一致；summary中的Accuracy/Macro-F1/Weighted-F1一致，Stab Recall17/25不一致。26份此次來源檔摘要未變，整體207份歷史snapshot仍未變。
- Research impact：新增HISTORICAL_METRIC_SEMANTICS_ERROR；原88.98%±9.30%不能繼續當已核驗的刺傷Recall。這是報告語義更正，不是模型改善，也不取代缺identity的canonical OOF。
- Safety：locked_test_used=false；CO2Wounds_used_for_tuning=false；historical_results_deleted=false。
- Remaining risk：原OOF mapping/provenance缺件；不能把逐陣列算術一致宣稱逐影像完整性。ROC-AUC未在此新增稽核器重算。
- Next issue：維持BLOCKED_BY_EXTERNAL_EVIDENCE與READY_FOR_NEXT_PHASE=NO；補原始證據後，以新版本修復resume/統計和評估契約，不進訓練。

## Iteration 6 — 2026-09-20（Phase A.5 correction and guard hardening）

- Issue / Priority：P0 修正已確認的resume指標語義錯誤、歷史輸出覆寫與資料角色／顏色輸入契約風險；不做模型訓練或推論。
- Evidence：25份val prediction各seed 720列，共3600列；歷史17/25 Stab summary值誤取Macro Recall；原3622 CI與row identity缺件維持未核驗。
- Root cause：`experiment_logger.recall=macro_recall`，舊`multi_seed` resume誤作Stab；部分歷史表格／Bootstrap wrapper import-time執行；ISIC gate將PIL RGB轉RGB NumPy送入預期BGR的模型路由。
- Fix / Files：加入`resume_metrics.py`、`data_roles.py`與保存陣列v2重算器；修正multi-seed、分類訓練／評估直接入口、正式development guard、ISIC gate；歷史生成器改main guard並禁止覆寫／row-wise CI。
- Tests：Phase A.5合成回歸18 passed；完整synthetic suite 118 passed／0 failed／0 skipped；版本化摘要產出，未用模型。
- Verification：25舊prediction、歷史統計及Table1–4共35檔SHA256不變；54既有權重與Phase A snapshot一致。舊Phase A audit產物未覆寫。
- Research impact：真正Stab seed-pooled mean=91.92%、sample SD=4.52pp；原88.98±9.30只保留為錯誤歷史宣稱。Accuracy87.40及Macro-F1 87.36仍由25fold描述平均支持，非grouped CI或獨立資料集推論。
- Safety：test_images_used=0；CO2Wounds_used_for_tuning=false；new_model_training=false；舊輸出未刪除。資料來源權限／舊row identity仍缺，未知來源與受保護角色在已整合入口fail closed。
- Result：PHASE_A5_STATUS=COMPLETE；READY_FOR_PHASE_C_CONTROLLED_REEVALUATION=NO（需新版本化pin/output）；READY_FOR_FULL_PHASE_B_GROUPED_BOOTSTRAP=NO。

## Iteration 7 — Phase B1 statistical reconstruction（2026-09-20開始，2026-09-21驗證）

- Issue / Priority：以五個seed各自合併五折作正式development描述主統計；歷史25fold只留secondary。保留舊Table2/Table2b，不生成CI。
- Evidence：固定val路徑讀25份JSON，5×5皆完整，720列/seed與3600列總數；七類順序及概率shape/finite/range/sum/argmax全部通過。無row→image/group/patient mapping。
- Test-first / Minimal implementation：新增`experiments/phase_b1.py`與17項synthetic回歸；版本化JSON、Table2c、七類MD/CSV及seed/pooled混淆矩陣表，以exclusive create拒絕覆寫。
- Result：seed-pooled Accuracy 87.39±0.79pp、Macro-F1 87.47±0.78pp、Weighted-F1 87.43±0.79pp、Stab_wound Recall 91.92±4.52pp（sample SD，ddof=1）；25fold描述平均Accuracy87.40%、Macro-F1 87.36%。
- Re-audit：本輪請求的Stab逐seed標籤將3407與999互換；保存檔顯示3407=99/99、999=89/99。平均及SD相同，已在freeze/report明確註記，不自行更動來源檔。
- Verification：B1 targeted17/17，related36/36，full synthetic135/135；0 failures/skips。related僅2則第三方thop distutils棄用警告；91個受保護檔案SHA256前後完全相同（含25預測、舊統計、Table1–5、54權重）。
- Safety：無訓練、推論、圖片讀取、48張locked test或CO2Wounds存取、bootstrap CI、病人層分析或模型選擇；舊結果未覆寫。
- Status：PHASE_B1_STATUS=COMPLETE；CLASSIFICATION_RESULT_FREEZE=COMPLETE；READY_FOR_FULL_PHASE_B2_GROUPED_BOOTSTRAP=NO；READY_FOR_PHASE_C0=YES僅限新版本化規約的規劃與前置稽核，不代表准許評估。
