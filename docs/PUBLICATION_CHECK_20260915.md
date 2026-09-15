# 2026-09-15 公開提交檢查

本次發布包含 App、實驗 runner、測試、方法報告與彙總數字，不啟動訓練、不重新執行模型測試、不開臨床資料庫、不替換 App 權重。

## 本次即時確認

- 工程驗證：82 項 unittest 通過；獨立合成推論、前端 lint、build 均 exit 0。權重 SHA256 與封版 metadata 未改變，`test_images_used=0`。
- 發布前掃描當時 Git 索引列出的 175 個工作檔；掃描後新增本檢查文件。常見 GitHub token、OpenAI key、AWS access ID、私鑰標頭，以及目前本機 key 精確比對：0 命中。不輸出 key 值。
- 敏感副檔名檢查唯一影像命中為既有 `wound_nurse_app/src/assets/hero.png`；已目視確認為抽象幾何 UI 素材，不是病人或資料集影像。
- 沒有加入資料庫、實值 `.env`、私鑰、模型權重、資料集、逐圖預測或臨床圖片。公開 JSON 僅抽取指定彙總欄位及來源報告 SHA256。
- `git diff --cached --check` 通過；新主報告的本機相對連結均存在。
- GitHub：公開 repository、預設 main；Secret Scanning 與 Push Protection 啟用；目前 API 回傳 secret-scanning alerts 空清單；Deploy Keys 為 0，協作者名單只有擁有者（admin）。
- Dependabot Security Updates 仍為 disabled；本輪沒有擅自變更平台設定。

掃描範圍是即將發布的檔案樹與上述格式，**不是所有可能秘密的保證，也不是完整滲透測試**。工程測試用的是合成資料與暫存 DB，不等於既有資料庫遷移或生產部署驗收。

## 舊歷史仍須另處理

9/6 的本機換 key、重新加密與公開分支歷史重置有既有紀錄；Support #4732924 已送出。尚無本次可驗證的伺服器清除完成證據，故不能宣稱舊 SHA／PR ref／快取已全部移除。API 沒有 secret alerts 也不能證明舊資料已消失。舊 key 必須永久視為已外洩，部署端不得復用。

## 可查閱的交付

- [完整方法與問題處理報告](PROGRESS_AND_METHODS_20260915.md)
- [聚合數據與工程驗證](evidence/progress_20260915.json)
- [安全歷史紀錄與更正](GITHUB_SECURITY_AUDIT_20260906.md)

本輪採正常提交及快轉推送，不再次強制改寫 Git 歷史。最終提交 SHA 由 GitHub commit 頁提供。
