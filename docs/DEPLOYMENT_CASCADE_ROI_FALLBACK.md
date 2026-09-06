# Cascade ROI 推論與安全 fallback

目前 API 的預設影像分析流程為：

```text
Upload image
    ↓
YOLO11m-seg ROI proposal
    ↓
原圖 C-Arch-05 分類（full_image_primary）
    ↓
ROI 僅用於畫面標示與人工覆核，不影響分類輸入

若未來確認臨床 ROI 標註與 domain adaptation 足夠，才可切換：

```text
WOUNDCARE_CLASSIFICATION_SOURCE=segmentation_crop
```
```

API 會在回應中提供：

- `inference_mode`: `full_image_primary`、`segmentation_crop`、`full_image_fallback` 或 `legacy_detection`
- `fallback_used`: 是否回退至原圖分類
- `segment_confidence`、`class_confidence`
- `crop_box`（只有可信 ROI 才提供）
- `requires_human_review: true`（固定保留）

## 模型與環境變數

預設模型為：

- Segmentation：`experiments/results/segmentation/YOLO11m_WSNet_seg_aug_v5_20260818/weights/best.pt`
- Classification：`runs/classify/wound_classifier_v32/weights/best.pt`

可用環境變數覆寫：

```text
WOUNDCARE_SEG_MODEL
WOUNDCARE_CLS_MODEL
WOUNDCARE_SEG_CONFIDENCE=0.10
WOUNDCARE_CLS_CONFIDENCE=0.60
WOUNDCARE_SEG_IMGSZ=384
WOUNDCARE_SEG_DEVICE=cpu
WOUNDCARE_CLASSIFICATION_SOURCE=full_image
```

8 GB GPU 環境建議 segmentation 使用 `cpu`，避免 segmentation prototype 在 640 解析度推論時超出顯存；正式部署前應以目標硬體重新量測延遲。

## 目前驗證界線

Development val 41 張的工程性結果：

- 純原圖分類：Accuracy **73.17%**、Macro-F1 **70.79%**
- 僅 ROI cascade：端到端 Accuracy 21.95%
- 加入原圖 fallback：端到端 Accuracy 63.41%
- fallback 後 Macro-F1：67.73%

因此目前 production 預設採「原圖分類＋ROI 視覺標示」，避免公開 WSNet 與臨床影像 domain shift 造成裁切後分類退化。上述評估的 ground truth 是分類資料夾名稱，沒有臨床 bbox/mask 配對標註，因此不能視為正式偵測準確率。API 不會讀取 48 張 blind test，且所有結果都必須經護理師人工覆核。
