# 2026 年 10 月 27 日進度與驗證更新

本次更新整理既有模型證據，重新執行 App/API 軟體測試，並同步逢甲推甄使用的傷口畢業專題報告。沒有新增模型訓練、真實模型推論、盲測或 App 權重替換。模型實作與研究數字以既有 commit `37d8b3684c728b0aa15308e2cf0c794fe5fc433a` 及[公開彙總證據](evidence/progress_20261024.json)為基準；不把文件更新日期當成新實驗日期。

後續補充：[2026-10-28 網頁端實機展示與五張操作截圖](WEB_DEMO_20261028.md)。使用實際瀏覽器與合成隔離資料，未載入模型；不改寫本頁 10 月 27 日測試的歷史範圍。

## 目前完成的工作

| 工作 | 已有證據 | 可支持的結論 |
|---|---|---|
| 七類分類 | 720 張 development、383 個 MD5 群組；5 seeds × 5 folds；保存預測列 3,600 筆 | 五 seed pooled Accuracy 87.39%、Macro-F1 87.47%；不是 25 套獨立資料 |
| 分類一次性保留測試 | 48 張中 46 張正確，95.83%；本次未重開 | 只代表固定內部保留集，不代表臨床泛化 |
| FUSeg 分割訓練 | 771 train／191 development validation；300 epochs | Mask mAP50 90.32%、mAP50–95 68.08%；仍是開發結果 |
| 三模型探索比較 | 同批 191 張，其中 186 正、5 空 GT | 正樣本 Macro-IoU A/B/C＝12.63%／76.66%／76.50%；B 對 C 逐圖 89 勝／93 負／4 平，不支持穩定冠軍或自動換模型 |
| API Response V2 | 七類原始分數、原生 mask/RLE、框、polygon、ROI、座標與模型身分 | 可追溯輸出；不等於模型品質、OOD 或實體面積驗收 |
| App 上傳隔離 | input generation／request ID；新選圖與錯誤清空舊結果 | 防止晚到回應沿用到新影像或其他病人 |

## 2026 年 10 月 27 日實際重驗

| 指令 | 結果 |
|---|---|
| `python -m unittest discover -s tests -p test_api_response_v2.py -v` | 14/14 通過；thop 對 distutils 有既有 DeprecationWarning，無失敗測試 |
| `npm test`（`wound_nurse_app/`） | 23/23 通過 |
| `npm run lint` | 通過 |
| `npm run build` | TypeScript 與 Vite production build 通過 |

以上為軟體契約與建置驗證，不是最新瀏覽器端到端驗收；沒有真實模型推論，不使用患者影像、分類 locked test 48 張或 FUSeg official test 200 張。不同日期的歷史 264 項回歸與本次 14／23 項不可直接相加。

## 尚未完成與下一個驗收標準

1. 分類資料的來源、版本、授權與逐圖 lineage 尚不完整；找回可信 row→image/group 身分前，不製造 grouped bootstrap CI。
2. 目前主要分類分數仍以完整影像為輸入，ROI 是另行保存的定位證據。正式 cascade 正確率需要同一影像的定位真值與七類標籤，以及預先固定的評估規格；尚未完成。
3. 191 張 FUSeg validation 已多次參與開發；下一步須取得合法、未見、經獨立標註 QC 的資料，先凍結候選與門檻再確認，不能在原 validation 繼續選冠軍。
4. 新版上傳安全後的瀏覽器、手機、慢網路、取消、病人切換與真模型 HTTP 回歸待完成；一般單元測試不取代這些工作。
5. 照護對應仍為 `NOT_CONFIGURED`，OOD 與實體面積為 `NOT_VALIDATED`；人工覆核不可自動解除。
6. 隔離環境 R14 診斷因 controller 收尾錯誤仍為 `BLOCKED`；本次沒有修復、重跑、放寬政策或宣告部署完成。

## 推甄文件與公開邊界

Drive 原稿已先備份；傷口報告保留 Phase 1–9、資料張數、切分、模型案例、個人貢獻與老師簽名空欄，將三模型比較及 App/API 更新整合進對應章節。合併報告中的交易個人專案部分保持原內容。個人推甄文件留在私人 Drive，不納入公開 GitHub。

公開提交只包含本更新、README 與索引調整；不包含資料集、患者照片、模型權重、資料庫、密鑰、原始逐圖預測、私人環境診斷或推甄文件。

## 方法與證據連結

- [完整研究進度與方法對照](PROJECT_PROGRESS_REPORT_20260929.md)
- [三模型與 App/API 公開彙總 JSON](evidence/progress_20261024.json)
- [分類五 seed 最終開發摘要](../experiments/results/tables/Table2c_Final_Classification_Development_Summary.md)
- [FUSeg 官方來源](https://github.com/uwm-bigdata/wound-segmentation)
- [Group-aware split 的官方實作定義](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [Ultralytics 輸入影像來源與格式契約](https://docs.ultralytics.com/modes/predict/#inference-sources)
- [Response V2 後端契約測試](../tests/test_api_response_v2.py)
- [前端上傳與呈現安全測試](../wound_nurse_app/tests/ux-safety.test.tsx)
