# D-Seg-03R：FUSeg 非商業研究模型卡

狀態：已完成 development 訓練與驗證；尚未作為 App 預設權重，也不是臨床部署核准。

## 模型與資料

- 模型：YOLO11m-seg，單類 `Wound`；定位／分割傷口，不負責七類診斷。
- 初始化：通用 COCO 預訓練，不接續先前傷口模型。
- FUSeg：771 張 train、191 張 val；test 使用量 0。
- 768 輸入尺寸、batch 4、AdamW、lr0=0.0005、seed=42、warmup 5 epochs。
- 300 epochs 上限全部完成；patience=80。本輪只將 warmup_bias_lr 由 0.1 改成 0.0，其餘設定盡量匹配歷史 D-Seg-03。
- train／val exact-content overlap=0；pHash 距離≤4 的跨 split 近似對=0。不宣稱病人層級獨立。
- 實體複製資料，避免修改 hardlink 影響歷史版本。

## 最終同一 checkpoint 的驗證結果

來源為 `outputs/fuseg_warmup_revision_20260914/result.json` 的最終 development validation，不拼接訓練歷程峰值。

| 指標 | 本輪 |
|---|---:|
| Box Precision | 91.58% |
| Box Recall | 82.57% |
| Box mAP50 | 90.35% |
| Box mAP50–95 | 70.59% |
| Mask Precision | 91.58% |
| Mask Recall | 82.57% |
| Mask mAP50 | 90.32% |
| Mask mAP50–95 | 68.08% |

相較歷史 D-Seg-03，Mask mAP50 約 +0.25 百分點、mAP50–95 約 +0.85 百分點、Recall 約 +1.24 百分點，Precision 約 −0.87 百分點。未做顯著性檢定，不宣稱因果或全面改善。

P／R 是 Ultralytics AP 驗證流程報出的值，不能當成已鎖定 App confidence 工作點的操作性能。191 張 val 已反覆參與選模，不是獨立測試。51.42 ms／張是該次 GPU 驗證中的模型推論量測，不是手機、CPU 或 App 端到端延遲保證。

## 權重身分

本機位置：`outputs/fuseg_warmup_revision_20260914/formal/weights/best.pt`。

SHA256：`2a2d66e7036f2d8eb94e9da8907554f0fcd55484e138c25b024b14fbc6de678d`。

程式來源交付 ZIP 不包含此權重、病人影像、標註或資料集。完整訓練參數與來源證據雜湊保留於本機 `protocol.json`。

## 來源與適用範圍

- [UWM FUSeg 官方資料庫](https://github.com/uwm-bigdata/wound-segmentation)
- [FUSeg 論文](https://arxiv.org/abs/2201.00414)
- [固定版本的官方競賽文件](https://github.com/uwm-bigdata/wound-segmentation/blob/42a272dfe0679f20675e826385925cb7562934b6/data/Foot%20Ulcer%20Segmentation%20Challenge/FootUlcerSegmentationChallenge2021.pdf)

官方競賽文件資料使用欄列為「CC BY NC」，該處未註明版本；本輪保留來源標示並限定非商業、離線研究。使用者已確認專案為非商業，但這不能替授權不明的 Yasin 或未核准 Redscar 補上使用權，也不等於臨床部署驗證。

## 尚未完成的驗收

2026-09-14 追加：已用這份分割權重及 C-Arch-05 分類權重，對零矩陣與固定種子雜訊兩個合成輸入實跑 CPU 推論，架構／類別順序／JSON 相容性通過。此測試沒有讀取任何資料集影像。

其中雜訊影像未被分割器定位為傷口，但七類分類器仍以 99.32% 信心輸出 Bruises。這是閉集合模型的分布外反例，**不是**臨床假陽性率估計；也說明不能把 high confidence 當成「有傷口」的證據。

App 新增防護：無可靠定位、分類信心不足或定位 fallback 時，保留分類提示供覆核，但不自動給出類別處置／RAG 建議。這是應用層風險降低，不是模型已學會健康／非傷口拒判。模型權重與 App 預設均未替換。

證據：本機 `outputs/candidate_integration_20260914/integration.json`。CPU 合成測試約 1.17 秒（首輪）與 0.58 秒（次輪），不是完整裝置／網路／影像分布的速度評測。

新未見來源的泛化評估、固定工作點的 FP／FN 分析、陰性樣本覆蓋與實際 App／CPU 延遲測試仍待完成。不得拿分類 48 張或已封存 CO2Wounds 結果回頭調參；不得把更換權重本身當成完成落地。
