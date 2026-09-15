# WoundCare+ 專案工程交付與後續驗收

日期：2026-09-14。定位：非商業畢業專題／研究展示原型，不是已通過臨床效能、安全或醫材驗證的產品。

## 1. 這次實際完成什麼

| 區塊 | 已完成 | 限制 |
|---|---|---|
| 分割訓練 | FUSeg 771 train／191 val，YOLO11m-seg 300 epochs 完成；最終 Mask mAP50 90.32%、mAP50–95 68.08% | 內部 development 成績，不是獨立外部測試 |
| 評估儲存 | 加入真正可勾選的人工覆核確認；伺服器仍強制驗證；修正回傳 EMR ID 錯取 reports ID 的問題 | 不替護理人員作判斷 |
| 病人切換 | 清除前一病人的影像、處置與結果；忽略舊分析／病歷請求回應；時間軸再按 patient_id 過濾 | 未聲稱所有多分頁／網路故障情境都完成驗證 |
| RAG 知識 | 覆核＋去識別化雙確認、原始 EMR 追溯、覆核者與時間、停用／重新覆核 | 是規則文字加關鍵詞檢索，不是持續微調 LLM |
| 舊 RAG 記錄 | 非破壞式增加欄位，舊筆保留且預設未確認；管理者／護理長可檢查後啟用 | 尚未實際遷移本機真實資料庫 |
| 預覽防護 | 必須明確指定獨立 DB 與 key；阻擋預設 clinical DB／同檔 hardlink；production 禁止本機 key fallback | 獨立路徑不能證明資料內容一定是虛構；仍需管理者確認 |
| 系統狀態 | API／畫面區分「服務可用」與「模型已載入」；不再把 health=ok 說成所有模型就緒 | loaded 也不代表模型已經 clinical validated |
| 分類建議防護 | 無可靠 ROI／分類信心不足／定位 fallback 時不自動輸出類別處置與 RAG | 原始分類仍供人工覆核；不是健康／非傷口偵測器 |
| 交付包 | 明確 allowlist 的原始碼 ZIP 與逐檔 SHA256 清冊 | 不含資料集、weights、DB、key 或外網服務 |

模型詳情與公開來源：[FUSeg 模型卡](FUSEG_MODEL_CARD_20260914.md)。

## 2. 本機驗證（安全的示範模式）

在完整專案或解壓後的 `woundcare-project` 根目錄執行。既有電腦已具備 Python 3.12 與 Node.js；新電腦先自行安裝對應環境。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# API 測試用依賴，不是連接真實照護資料
.\.venv\Scripts\python.exe -m pip install httpx
Push-Location wound_nurse_app
npm ci
npm run lint
npm run build
Pop-Location
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_startup_safety.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_professor_preview_api.py -v
.\.venv\Scripts\python.exe tests/test_woundcare_inference.py
.\.venv\Scripts\python.exe tests/run_local_preview_smoke.py
```

最後一行開啟 `http://127.0.0.1:8097` 的**拋棄式測試環境**：帳號 `admin`，公開的合成測試密碼 `SyntheticPreviewOnly!2026`。這組帳密只能用於 loopback UI fixture，不得用於真實部署、手機 LAN 或外網通道。按 Ctrl+C 停止。服務不載入任何模型，畫面應顯示「系統可用 · 模型未就緒」；可以測操作與 RAG，但不能拿來看影像推論準確率。

每次啟動都會建立新的暫存 DB 與隨機 key，正常停止時清理。強制殺掉程序可能留下系統暫存資料夾，內容僅能有本次合成測試資料。請勿輸入真實個資或上傳病人照片。

`requirements.txt` 仍未完整鎖定依賴版本；本機通過不等於全新電腦或 Docker build 已通過。新環境必須重跑上述驗收，不能直接沿用成功結論。

## 3. RAG 操作

1. 用管理者／護理長角色進入「覆核與稽核」。一般護理師不能新增、啟用或檢視待覆核知識。
2. 填標題、適用類別、建議、依據、注意事項及來源文件名稱。先刪去姓名、床號、病歷號等識別資訊；系統不會自動保證匿名化。
3. 若選來源評估紀錄，該 EMR 必須存在且已完成人工覆核。伺服器保存 source_emr_id，不只留一段文字。
4. 勾選「已覆核」及「已去識別化」後才可儲存。伺服器不接受跳過勾選直接送 API。
5. 舊筆、停用筆不出現在一般護理師檢索結果；可在管理介面重新確認後啟用。不確定或包含個資時不要核准，改建立一份補正且去識別化的新內容。
6. 停用會保留歷史。建議文字不會自動變成醫囑、寫入 EMR、訓練 LLM 或修改影像模型權重。

## 4. 真實資料庫與 Docker 的操作邊界

本輪只在暫存 DB 驗證，沒有啟動既有 `healthcare.db`、旋轉 key 或修改其內容。真正切換服務前必須先做離線加密備份與還原驗證，再安排停機遷移。首次啟動新程式會非破壞式新增 RAG 欄位；舊知識先排除檢索，直到重新明確覆核。

Dockerfile 已加入新的 `woundcare_safety.py`，建置 context 增補排除秘密、DB、封存資料與輸出。現有 Docker compose 仍保留；**本轮未啟動 Docker 容器或外網通道，未宣稱 Docker 端到端部署通過**。

若之後要教授外網預覽，仍需獨立示範 DB、獨立 key、唯一強密碼、host allowlist、TLS 與邀請制。不要使用上節合成 fixture 的公開密碼。詳見 [教授預覽部署規範](PROFESSOR_PREVIEW_DEPLOYMENT.md)。

## 5. 驗收證據與未完成事項

本輪通過 60 項 unittest 與 4 項獨立合成推論測試（合計 64 項），frontend lint／build 通過。最終紀錄見本機 `outputs/project_delivery_candidate_guard_20260914/verification.json`。瀏覽器已在合成環境實際檢查：模型缺席提示、未勾覆核遭阻擋、勾選後儲存、RAG 雙確認後儲存與覆核時間顯示；沒有使用真實照片做推論，也沒有驗證手機硬體速度。

另實跑 2 个合成影像的真實候選權重相容性測試，發現雜訊亦可能產生 99.32% 的七類分類信心。已補入應用層建議阻擋與 API 回歸測試，細節見模型卡。這些防護不會把分類信心重新校正，也不會把新候選模型自動部署。

仍待驗收而不是假裝完成：

- 新候選模型的來源與預處理／閾值設定經 App 端到端驗證後，才決定是否替换預設權重；本輪没有替换。
- 取得新的未見且可合法使用的 cohort 後，再執行预先固定規約的泛化評估。Redscar 核准未到，不能把未回信當同意。
- 臨床部署、病人層級隔離、專業標註、合適的陰性樣本／錯誤分析與護理流程驗收，不能由內部 mAP 或勾選方塊代替。
- GitHub 舊秘密永久視為外洩；歷史 Support purge 狀態本輪沒有向 GitHub 重新查核，不聲稱已完全清除所有伺服器物件。
- 本輪沒有推送 GitHub；新增工作檔先保持本機，避免把既有未核對修改一併發布。

## 6. 原始碼打包

```powershell
python scripts/build_project_handoff.py --output outputs/woundcare_source_HANDOFF_NEW_ID.zip
```

輸出檔必須尚不存在。包內 `SOURCE_MANIFEST.json` 有逐檔 SHA256；只收明確列出的 App 原始碼、操作文件、設定範例和合成測試。研究影像、weights、真實資料、訓練 outputs、node_modules、Git 歷史與 `.env` 實值均不在交付包。
