# 無網域教授預覽：短期 HTTPS 分享

本方案使用 Cloudflare Quick Tunnel 產生隨機的 `https://*.trycloudflare.com` 網址，適合一次教授展示或短期測試。網址只在 tunnel 程序持續執行時有效，每次重啟都會更換；請勿將其視為正式或長期部署。

## 使用限制

- 只用於 `WOUNDCARE_PROFESSOR_PREVIEW=true` 的去識別化示範資料。
- 僅將網址給受邀教授；任何拿到網址的人都可抵達登入畫面。
- 不可上傳病人資料、未授權影像、資料集或封存 Blind Test。
- 電腦需在測試期間保持開機、不中斷網路；關閉 tunnel 視窗即可停止外網存取。

## 首次準備

1. 安裝 Docker Desktop；另外安裝 `cloudflared`，或保留 Node.js 的 `npx`（啟動腳本會自動使用可用的一種）。
2. 建立 `models/`，只放封版權重，並依 `.env.quick-preview.example` 的檔名設定環境變數。
3. 複製 `.env.quick-preview.example` 成 `.env.quick-preview`，填入新的強密碼與 Fernet key。不可沿用開發資料庫或 `secret.key`。

## 啟動

在專案根目錄執行：

```powershell
./scripts/start_quick_professor_preview.ps1
```

終端會顯示一個隨機的 `https://...trycloudflare.com` 網址。開啟該網址後，使用 bootstrap 管理者建立每位教授的獨立帳號；不要分享管理者密碼。

## 收尾

教授測試結束後停止 tunnel，然後執行：

```powershell
docker compose -f docker-compose.quick-preview.yml down
```

先匯出需要保留的去識別化回饋與稽核摘要，再依照研究保留政策處理 `professor_preview_data/`。不要刪除其他環境的資料庫。
