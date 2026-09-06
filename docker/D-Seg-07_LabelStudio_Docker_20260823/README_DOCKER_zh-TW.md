# D-Seg-07 Label Studio Docker 詳細操作手冊

## 1. 套件目的與資料邊界

本套件用於 D-Seg-07 傷口分割人工標註，固定以 Label Studio 1.23.0 執行。它只包含：

- 383 張 development set 的唯一 MD5 代表影像（train 342、validation 41）。
- 383 筆 Label Studio 任務。
- 單一人工標註類別 `Wound`。
- 720 張原始 development 檔案到 383 個唯一內容群組的 lineage 紀錄。

本套件**不包含** 48 張封存 Test、Blind Test、臨床帳號、既有 Label Studio 資料庫、模型權重或任何預先捏造的 polygon。`test_images_used=0`、`blind_test_used=false`。

## 2. 資料夾用途

| 路徑 | 用途 | 是否可修改 |
|---|---|---|
| `annotation_package/images/` | 383 張待標註代表影像 | 否，Docker 內唯讀 |
| `annotation_package/label_studio_tasks.json` | 383 筆匯入任務 | 否 |
| `annotation_package/label_config.xml` | `Wound` polygon 標註介面 | 否 |
| `annotation_package/group_manifest.csv` | 383 個 MD5 群組與 train/val 歸屬 | 否 |
| `annotation_package/lineage_720_to_383.csv` | 720 檔案到代表影像的追溯表 | 否 |
| `label_studio_data/` | 帳號、專案與標註進度資料庫 | 是，由容器寫入 |
| `exports/` | 手動放置匯出的 JSON | 是 |
| `docker_package_audit.json` | 封裝稽核與 SHA-256 摘要 | 否 |

不要重新命名、搬動或覆寫 `annotation_package/images/` 內的影像，否則任務 URL 與 lineage 會失去一致性。

## 3. 執行前需求

Windows 建議使用 Docker Desktop，並確認 Docker Engine 已啟動。PowerShell 執行：

```powershell
docker --version
docker compose version
docker info
```

前兩個指令應顯示版本；`docker info` 必須同時顯示 Client 與 Server。預設網址是 `http://127.0.0.1:8083`，只允許本機瀏覽器存取。若 8083 已被占用，先停止占用服務；不要為了方便把位址改成 `0.0.0.0`。

## 4. Windows 一鍵啟動

在解壓縮後的套件根目錄開啟 PowerShell：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start_dseg07_docker.ps1
```

啟動腳本會依序：

1. 確認 `annotation_package/images/` 恰有 383 張影像。
2. 拒絕含有 Test/Blind 影像的套件。
3. 驗證 `compose.yaml`。
4. 拉取鎖定版本的官方映像。
5. 啟動容器並等待健康檢查通過。

看到下列訊息才算啟動成功：

```text
D-Seg-07 Label Studio is ready: http://127.0.0.1:8083
test_images_used=0; blind_test_used=false
```

接著用瀏覽器開啟 <http://127.0.0.1:8083>。第一次拉取映像可能需要數分鐘；這不是重新訓練模型。

## 5. 第一次建立本機帳號與專案

1. 在登入頁選擇建立帳號，使用本機專用帳號與密碼。套件沒有預設密碼。
2. 登入後選擇 **Create Project**。
3. Project Name 填入 `D-Seg-07 Wound Segmentation`。
4. 進入 **Labeling Setup**，選擇 **Custom Template** 或程式碼編輯模式。
5. 開啟 `annotation_package/label_config.xml`，全選並複製內容到 Labeling Interface，儲存設定。
6. 進入 **Data Import**，上傳 `annotation_package/label_studio_tasks.json`。
7. 匯入完成後確認總任務數為 **383**，並隨機開啟 train 與 val 任務，確認影像可正常顯示。

若先匯入任務再設定介面也可以，但在正式標註前必須確認介面中只有 `Wound` polygon 與 `quality_note`，不得新增七類診斷標籤。

## 6. Wound polygon 標註規範

每張影像的目標是描繪「可見傷口區域」，不是判斷傷口診斷名稱。

1. 選擇紅色 `Wound` 工具。
2. 沿可見傷口外緣逐點建立 polygon；頂點至少 3 個。
3. 邊界轉折處增加節點，平滑直線處不需要過密節點。
4. 同一影像若有彼此不相連的多個傷口區域，可各畫一個 `Wound` polygon。
5. 排除健康皮膚、尺規、紗布、貼布、文字、浮水印、陰影與背景。
6. 不得依原始類別名稱猜測傷口邊界，也不得把整張影像框成傷口。
7. 邊界模糊、遮擋或畫面不足時，在 `quality_note` 記錄原因。
8. 若完全看不到可判讀的傷口，不要捏造 polygon；填寫 `quality_note` 並交由 QA 決定。這類任務會維持訓練閘門阻擋狀態。
9. 完成後按 **Submit**。不要用 Skip 或 Cancel 取代正式標註。

建議標註者以 100%–200% 縮放檢查邊界。每完成一批 25–50 張即執行一次進度與視覺抽查，避免到最後才發現規則不一致。

## 7. 標註品質檢查清單

正式匯出前逐項確認：

- Project 顯示 383/383 任務已完成。
- 每個任務只有一份已提交 annotation；同一份 annotation 可含多個 polygon。
- 每張影像至少一個有效 `Wound` polygon，除非已列入待裁決清單。
- 每個 polygon 至少 3 個不重複頂點，沒有零面積或自相交的明顯錯誤。
- 未建立 Abrasions、Bruises、Burns、Cut、Ingrown_nails、Laceration 或 Stab_wound 等新 polygon 類別。
- train/val 分割沒有被人工重新分配。
- Test/Blind Test 仍未匯入或檢視。

建議第二位檢查者抽查至少 10%，特別查看小傷口、低對比、遮擋、影像邊緣與多傷口案例。若只有一位標註者，報告中應如實記載為 single-annotator annotation，而非宣稱專家共識。

## 8. 匯出標註 JSON

1. 在 Label Studio 專案右上角選擇 **Export**。
2. 建立新的 export snapshot。
3. 格式選擇 **JSON**，不要選只含影像 URL 的簡化格式。
4. 下載後重新命名為 `label_studio_export.json`。
5. 將檔案放入本套件的 `exports/`：

```text
exports/label_studio_export.json
```

不要以新的檔案覆蓋舊匯出；建議先另存一份含日期的備份，例如 `label_studio_export_20260823.json`。

## 9. 回到主專案執行 QA 與建立 YOLO-seg 資料集

本 Docker 包負責標註服務，不會在匯出時自動開始訓練。將完成的 JSON 複製回主專案後，在主專案根目錄執行：

```powershell
python work/import_dseg07_label_studio_export.py `
  --package dseg07_yasin_wound_annotation_20260822 `
  --export docker/D-Seg-07_LabelStudio_Docker_20260823/exports/label_studio_export.json `
  --dataset yolo_dataset_dseg07_yasin_wound_seg_v1 `
  --gate outputs/dseg07_annotation_gate_20260823.json
```

只有下列條件全部通過才會建立訓練資料集：383 個 group_id 齊全、每個任務恰有一份完成標註、標籤只能是 `Wound`、polygon 座標有效且面積非零。預期通過狀態為：

```text
PASS_DSEG07_MANUAL_ANNOTATION_QA
training_allowed=true
test_images_used=0
```

若狀態是 `BLOCKED_PENDING_OR_INVALID_MANUAL_ANNOTATION`，查看 gate JSON 的 `issues`，回 Label Studio 修正指定 group_id 後重新匯出。不要略過 QA 閘門直接訓練。

## 10. 查看狀態、停止與重新啟動

查看狀態：

```powershell
.\status_dseg07_docker.ps1
```

停止服務但保留帳號、專案與標註：

```powershell
.\stop_dseg07_docker.ps1
```

再次啟動仍使用同一個 `label_studio_data/`，所以進度不會消失。不要使用 `docker compose down -v`；本套件雖使用 bind mount 而不是命名 volume，但刪除 `label_studio_data/` 仍會永久失去本機標註進度。

## 11. 備份與還原

備份前先停止容器，避免複製到寫入中的資料庫：

```powershell
.\stop_dseg07_docker.ps1
Copy-Item .\label_studio_data .\backup_label_studio_data_20260823 -Recurse
Copy-Item .\exports .\backup_exports_20260823 -Recurse
```

還原時先停止容器，再把備份目錄內容放回 `label_studio_data/`，然後重新執行啟動腳本。ZIP 交付檔預設不含任何已建立帳號或執行中的資料庫，避免傳遞密碼與個人資料。

## 12. Linux / macOS 啟動

Linux 首次執行先處理 UID 1001 寫入權限：

```bash
chmod +x prepare_linux_permissions.sh
./prepare_linux_permissions.sh
docker compose pull
docker compose up -d
docker compose ps
```

macOS 通常不需要 `chown`，可直接 `docker compose up -d`。完成後同樣開啟 <http://127.0.0.1:8083>。

## 13. 常見問題排除

### Docker daemon 未啟動

若出現 `Cannot connect to the Docker daemon` 或 named pipe 錯誤，先開啟 Docker Desktop，等 Engine 顯示 Running，再重跑啟動腳本。

### 8083 埠被占用

```powershell
Get-NetTCPConnection -LocalPort 8083 -ErrorAction SilentlyContinue
```

先停止占用 8083 的舊服務。若確實必須更換本機埠，可修改 `.env` 的 `LABEL_STUDIO_PORT`，但要同步記錄實際埠號；不得改成對外網卡綁定。

### 任務顯示但影像空白

先確認檔案仍在 `annotation_package/images/train/` 或 `images/val/`，再執行：

```powershell
docker compose logs --tail 100 label-studio
docker compose restart label-studio
```

任務 URL 必須維持 `/data/local-files/?d=images/...`，不要改成 Windows 絕對路徑。

### 容器一直是 starting 或 unhealthy

```powershell
docker compose ps
docker compose logs --tail 200 label-studio
```

確認磁碟空間足夠、`label_studio_data/` 可寫入，且安全軟體沒有封鎖 Docker bind mount。

### 匯出 QA 未通過

開啟 `outputs/dseg07_annotation_gate_20260823.json`，依 `issues` 中的 group_id 與原因修正。常見原因包括缺少任務、重複 annotation、沒有 polygon、錯誤標籤、少於 3 個頂點或退化面積。

## 14. 固定版本與安全設定

- Official image: `heartexlabs/label-studio:1.23.0@sha256:20cec817e63144adec9f23d699bf4f33ce4249a56eb183ae4853d20fbd10fd93`
- 映像同時鎖定版本標籤與 amd64 manifest digest，避免日後同名標籤漂移。
- 僅綁定 `127.0.0.1:8083`，不直接暴露區域網路或網際網路。
- `annotation_package` 以唯讀方式掛載；資料庫與匯出目錄分離持久化。
- 容器啟用 `no-new-privileges`。
- 不預設帳號、不在 ZIP 內保存密碼。
- 未登入時本機影像 API 會回傳 HTTP 401；這是預期的存取控制，登入後由 Label Studio 介面載入影像。
- Test images used: 0；Blind test used: false；External test status: `LOCKED_NOT_ACCESSED`。

## 15. 教授查核重點

可直接檢查 `docker_package_audit.json`、`annotation_package/package_audit.json`、`group_manifest.csv` 與 `lineage_720_to_383.csv`。本流程將完全相同內容的重複影像以 MD5 群組化，只標註 383 張代表影像，避免 337 張冗餘複本在訓練中被過度加權；train 與 validation 保持既定 group split，Test 持續封存。
