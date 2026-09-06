# GitHub 公開版安全稽核

稽核日期：2026-09-06  
Repository：`https://github.com/wangj6231/woundcare-system`  
可見性：Public

## 結論

已完成本次憑證處理與公開 Git 歷史重置。曾出現於公開歷史的舊 `secret.key` 已停用；本機 `healthcare.db` 中的加密欄位已改用新密鑰重新加密。遠端 `main` 與 `agent/professor-review-package` 已指向同一個不含舊歷史的新根提交。

從 GitHub 全新下載的裸倉庫已完成二次掃描：可達歷史只有 1 個提交，`secret.key`、`healthcare.db`、`.env` 實值檔、私鑰檔與常見 API token 格式均為 0 命中。

## 已完成處置

1. 在倉庫外建立改寫前備份：舊密鑰、資料庫及完整 Git bundle，不上傳 GitHub。
2. 產生新 Fernet key，並在 SQLite transaction 內重新加密 27 個非空白欄位，包含 6 個舊版明文欄位。
3. 驗證 27/27 加密欄位均能使用新密鑰解密，SQLite `integrity_check` 為 `ok`。
4. 以當前已去除敏感檔的檔案樹建立新根提交，強制更新兩個公開分支。
5. 確認遠端無 tag，兩個 branch 與既有 Pull Request ref 均只指向清理後提交。
6. 確認 GitHub Secret Scanning 與 Push Protection 為啟用狀態。

## 權限與秘密設定檢查

- 倉庫目前為 Public，預設分支為 `main`。
- 目前唯一列出的協作者為倉庫擁有者 `wangj6231`，具有 admin 權限。
- GitHub Deploy Keys：0。
- GitHub Actions repository secrets：0；variables：0。
- Secret Scanning：Enabled；Push Protection：Enabled。
- Dependabot Security Updates：Disabled，建議後續在 GitHub 設定中啟用。

## 持續安全規則

- `secret.key`、`healthcare.db`、`.env` 實值檔、患者影像、鎖定測試集、資料集與模型權重不得加入公開 Git。
- 部署環境只透過環境變數或秘密管理服務提供 `WOUNDCARE_FERNET_KEY`、`WOUNDCARE_BOOTSTRAP_PASSWORD` 與其他實值。
- 舊密鑰備份僅用於緊急還原，不可回填到任何部署環境。
- 此次清理保證 GitHub 現有 branch、tag 與 Pull Request ref 不再可達舊歷史；GitHub 平台的快取或已被第三方複製的舊物件不在本地操作可驗證範圍內，因此舊密鑰必須永久視為已外洩。
