# 智慧型傷口分級與照護對應系統

這是一個「先找出傷口，再協助分類，最後交給護理師確認」的研究型原型。它的目標不是取代護理判斷，而是把影像整理、模型提示、人工覆核與照護建議放在同一個流程中。

## 目前進度（2026-09-06）

- 分類流程已完成：431 張原始影像，保留 48 張鎖定測試影像只使用一次。
- 分類模型在鎖定測試中答對 46/48 張（95.83%）；這是內部一次性結果，不代表臨床驗證。
- 影像分割流程已完成多輪開發比較；目前最好的模型候選只在開發資料上比較，尚未開啟官方鎖定測試。
- 最新外部資料檢查使用 607 張 CO2Wounds-V2 影像，結果已封存，不再拿來調整模型。
- 下一階段 D-Seg-10 等待 Redscar 官方個別存取核准；在授權與資料稽核完成前不會開始訓練。

## 公開內容原則

這個公開儲存庫只放程式碼、設定範例、方法說明與可重現的摘要。患者影像、標註資料、鎖定測試資料、資料庫、密鑰與大型模型權重不放在 GitHub；需要資料時，請依文件中的官方來源與授權流程取得。

## 建議閱讀順序

1. [目前進度白話報告](docs/PORTFOLIO_PROGRESS_20260906.md)
2. [GitHub 公開版安全稽核](docs/GITHUB_SECURITY_AUDIT_20260906.md)
3. [教授版完整實驗報告](docs/PROFESSOR_INTEGRATED_EXPERIMENT_REPORT_20260817.md)
4. [分類結果總表](experiments/results/tables/Table5_Final_Blind_Test_Generalization.md)
5. [專案結構](PROJECT_STRUCTURE_20260812.md)

## 本機啟動

請先複製 `.env.example`，自行產生新的 Fernet key 與管理者密碼；不要使用或提交本機的 `secret.key`、`healthcare.db`。資料集與模型權重也必須放在本機忽略目錄。

```powershell
Copy-Item .env.example .env
python -m pip install -r requirements.txt
python backend_main.py
```

這是畢業專題與研究原型，不是已完成醫療器材認證的臨床決策系統。
