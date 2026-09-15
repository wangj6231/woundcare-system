# WoundCare+ 前端

React / TypeScript / Vite 的護理流程研究原型。延續既有看板、評估、覆核與管理頁面，不以模型輸出自動開立處置。

## 安裝與建置

```powershell
npm ci
npm run lint
npm run build
```

建置結果在 dist，供根目錄 FastAPI 提供。開發模式 npm run dev 會使用 vite.config.ts 中的 API proxy；預設開發設定可供 LAN 存取，請勿搭配真實病人資料或測試 fixture 公開帳密。

## 安全驗收

- 模型缺席時顯示「系統可用 · 模型未就緒」。
- 已覆核評估必須另勾人工確認；選擇「已覆核」本身不等同勾選。
- 更換病人或影像時重置評估，過期影像分析不可套用到新的照護對象。
- RAG 新增與重新啟用需要明確覆核／去識別化確認；停用或未核准內容不給一般護理師檢索。
- RAG 是文字檢索，不是自動訓練 LLM；所有真實權限與確認條件由 API 再驗證。

從專案根目錄按 [交付说明](../docs/PROJECT_HANDOFF_20260914.md) 啟動拋棄式 localhost UI fixture。正式權重、病人 DB、key 及外網分享都不在此測試範圍。
