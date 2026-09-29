# 七類分類資料來源與身分稽核（2026-09-20）

狀態：**UNVERIFIED_SOURCE / BLOCKED_BY_EXTERNAL_EVIDENCE**。本文件不新增使用權、不重新授權既有資料，也不將歷史訓練視為來源核准。

## 可確認的本機事實

本輪只讀取 `yolo_wound_cls_dataset_v3/train` 與 `val` 的位元組，未列舉或開啟 `test` 圖片。現況為 train 679、val 41，合計 720 個檔案；383 個 MD5 內容群組，其中 177 個 singleton、206 個重複群組涵蓋 543 個檔案，額外複本 337。

這是**目前 development 快照**，不是歷史 OOF 的原始排序清單，也不是 720 名病人。原始 431 張與 locked test 48 張來自既有文件；本轮不開啟原始全集來重新驗證，避免接觸其中的封存內容。

| 證據項目 | 本輪判定 |
|---|---|
| 目前 development 路徑、SHA256、MD5、類別 | 已記錄 720 列 |
| 原始發布者、URL、下載版本、日期 | UNVERIFIED |
| 原始封裝檔 SHA256 | UNKNOWN |
| 原始授權條款、可使用範圍 | UNVERIFIED；不能因非商業就假定獲准 |
| 原始影像 → 現有檔案的逐圖來源對照 | 未取得可驗證證據 |
| 病人／病例／影片／機構 ID | UNKNOWN，不由檔名推測 |
| pHash 相似群組與增強血統 | 本輪未重建，保留空欄位 |
| 歷史 prediction row → image ID | 缺失；不能建立 canonical OOF master |

## 清單與限制

- [機器可讀來源紀錄](evidence/classification_dataset_provenance.json)
- [720 列 development 身分清單](../experiments/results/audit/phase_a_20260920/sample_identity_manifest.csv)
- [檔案／群組計數](../experiments/results/audit/phase_a_20260920/development_inventory.json)

`sample_id` 是目前資料根目錄下的相對檔名，僅識別檔案；MD5/SHA256 識別 exact content。相同摘要不代表相同病人，摘要不同也不代表不同病人。`derived_from` 不依 `_aug` 或複製檔名填寫。未知欄位留空。

清單 SHA256：`6db2930ead92faf03bd0b6c7ec6bc4b112cf711550e5120a69f3dad5e83bfa5d`。清單含本機路徑與逐檔資訊，只供本機稽核，沒有推送至 GitHub。

## 解除阻擋需要的原始證據

1. 原始下載頁／發布者、資料版本、當時授權文件及原封裝檔摘要。
2. 原始檔案到現有 development 檔案的可核對 mapping；若曾複製、裁切或增強，需其實際產生紀錄。
3. 歷史 25 runs 的 ordered validation manifest，能逐列對應現存預測、訓練／驗證成員與 checkpoint。
4. 病人／病例 ID 若無法取得，永久保留 `PATIENT_LEVEL_INDEPENDENCE_NOT_VERIFIED`，不能用程式生成替代 ID。

沒有上述證據時，既有結果仍保留為附限制的歷史研究紀錄；不得重新包裝成來源清楚、病人獨立的新驗證。
