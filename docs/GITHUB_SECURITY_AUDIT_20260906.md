# GitHub 公開版安全稽核

稽核日期：2026-09-06
Repository：`https://github.com/wangj6231/woundcare-system`
目前可見性：Public

## 結論

目前不能把公開儲存庫標記為「已完成安全處理」。Git 歷史仍包含 `secret.key` 與 `healthcare.db`；本機資料庫也含有帳號、個案與評估紀錄。即使最新提交刪除檔案，公開 Git 歷史仍可取得舊版本，因此必須先輪替密鑰、確認資料庫內容不再需要，並由儲存庫管理者清除公開歷史後，才能宣稱安全。

## 已核對的證據

- 遠端儲存庫是 `wangj6231/woundcare-system`，可推送且目前為公開儲存庫。
- `origin/main` 的檔案樹仍包含 `secret.key` 與 `healthcare.db`；目前分支也仍可在舊提交找到這兩個檔案。
- 本機 `secret.key` 存在，但本報告不記錄其內容；本機 `healthcare.db` 大小約 60 KB。
- 本機資料庫表格包含 `users`、`patients`、`emr_records`、`sessions`、`audit_log` 等；目前計數為 4 個帳號、3 個個案、3 筆評估、4 個工作階段。這些資料按醫療資料處理，不能上傳公開儲存庫。
- `.env`、資料庫、密鑰、模型權重、資料集與本機產出已加入忽略規則；仍需在提交前以 Git index 和提交內容再次掃描。

## 必須完成的處置

1. 在部署環境產生新的 Fernet key，停用／替換舊 key；若舊 key 曾在公開歷史出現，視為已外洩，不得繼續使用。
2. 確認本機 `healthcare.db` 是否含真實個資；若不再需要，移至受控備份或安全刪除，絕不加入 Git。
3. 使用受信任的歷史清理工具移除 `secret.key`、`healthcare.db`、患者影像與不應公開的權重，再以受保護方式更新 GitHub 遠端歷史。
4. 清除歷史後重新執行 secret scan、檔案名稱掃描、Git LFS／大檔掃描與權限檢查；確認 main 分支與所有可見 tag／branch 都沒有敏感檔案。
5. 部署時只使用環境變數或秘密管理服務：`WOUNDCARE_FERNET_KEY`、`WOUNDCARE_BOOTSTRAP_PASSWORD`、資料庫路徑與允許來源；不要把實值寫進 `.env.example`。
6. 若要保留研究結果，公開摘要、圖表與方法說明即可；影像、遮罩、患者資料、鎖定測試資料與模型權重放在受控儲存，不放 public GitHub。

## 本次整理範圍

本次只新增公開版說明與安全稽核文件，並加強 `.gitignore`。不會擅自把資料集、患者資料或模型權重加入 Git，也不會把密鑰內容貼在報告或提交中。歷史重寫與舊密鑰失效是高影響操作，應在確認備份與部署切換後再執行。
