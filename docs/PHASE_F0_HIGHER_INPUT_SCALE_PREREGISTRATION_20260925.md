# Phase F0 — Higher Input Scale Preregistration & Resource Feasibility

版本：F_HIGHER_INPUT_SCALE_V1。實際執行日期：2026-09-26（Asia/Taipei）；檔名20260925保留使用者指定名稱。

## 1. 正式結論

```ini
PHASE_F0_STATUS = BLOCKED
RESOURCE_FEASIBILITY = PASS
NUMERICAL_FEASIBILITY = FAIL_NONFINITE_GRADIENTS
READY_FOR_PHASE_F1_PAIRED_HIGHER_SCALE_SEED42 = NO
PATCH_BASED_TRAINING_ROUTE = STOP
NEW_TRAINING_AUTHORIZED = NO
APP_MODEL_REPLACEMENT_AUTHORIZED = NO
EXTERNAL_TEST_AUTHORIZED = NO
STOP_AFTER_F0 = true
```

F0的預登錄、資料核對、唯一一次合成資源測試及封存已做完；但不能放行F1。**記憶體可行性通過，不等於數值安全或正式訓練可行性通過。** 合成backward出現非有限梯度；本次smoke前封存的安全條件明列nonfinite阻擋readiness。沒有為了通過而降低GradScaler起始scale、改batch或重跑。

這不代表1024訓練必然失敗，也沒有模型效果結論。AMP初始scale過大是一種可能解釋，但一次反向、沒有optimizer.step，不足以確認原因或後續是否會穩定。

## 2. 獨立研究線與原始資料

E_PATCH V1/V2/V3維持封存與STOP，僅為historical feasibility evidence。F0不使用canonical labels、mask-first labels或任何patch adapter。資料回到D2原始manifest的771張FUSeg training images、965個GT；影像及標籤SHA256全部符合，未增刪或修復invalid topology。僅讀PNG header核對512×512與檔案雜湊，不解碼training pixels；GPU smoke也完全沒有讀取真實資料。

兩組回到STOCK_YOLODataset及原始YOLO polygon augmentation/loss路徑。原始invalid topology既然是baseline一部分，本階段不修、不移除。191張development validation只繼承既有protocol；未推論、未看新的validation結果。Locked test、CO2與外部資料皆未使用。

## 3. Fresh C4/H4設計

| 欄位 | C4 | H4 |
|---|---|---|
| Training imgsz | 768 | 1024 |
| Final evaluation imgsz | 768 | 768 |
| 訓練中checkpoint-selection validation imgsz | 768 | 768 |
| Batch | 4 | 4 |
| Source canvas | 512×512 | 512×512 |
| 模型 | YOLO11m-seg / class0 Wound | 相同 |
| Sampling | Uniform without replacement | 相同 |
| Anchor orders | 300×771，共同seed42順序 | 相同 |
| Initialization | 相同ISIC checkpoint，fresh optimizer | 相同 |

唯一training semantic config difference為 `training_args.imgsz`，另有arm、experiment_id、output_path。Machine diff已核對。既有Phase C/D2模型只作REFERENCE_ONLY；主因果比較只可用Fresh C4 vs Fresh H4。

初始化SHA256：`1ec926026473beafafb1ef8da6aa6efcc13d38c391ec8db48006c5d793d5f2a3`，本次前後核對皆相同。resume=false。

固定300epochs、patience80、AdamW lr0=.0005、betas(.937,.999)、weight-decay .0005（bias/norm0）、cosine lrf=.01、warmup5、warmup_bias_lr0、AMP、deterministic、workers2、nbs64與gradient clip10。不做micro-batch或accumulation補償。

Loss維持v8SegmentationLoss：box/seg7.5、cls.5、dfl1.5、mask_ratio4、overlap_mask=true。Augmentation維持mosaic.1、close_mosaic30、mixup/copy_paste0、degrees5、translate.05、scale.2、shear.5、perspective.0002、fliplr.5、flipud0、HSV(.01,.4,.25)、bgr0，及原Albumentations Blur/MedianBlur/ToGray/CLAHE。未修改任何權重或gain。

### Scale的正確解釋

512→768為1.5×，512→1024為2×；相對baseline線性extent為4/3，面積為16/9≈1.7778×。只能說物件在model input／feature maps占更多像素範圍；**不是增加native resolution、創造新細節或恢復臨床資訊**。

Normalized augmentation設定相同不代表pixel-identical augmentation；其幾何像素差異屬於input scale的下游效果。沒有將其當作第二個人工改動。

## 4. Evaluation及未來執行順序

Evaluation protocol與D2 byte-identical：191張原始FUSeg development validation、full-image、RGB→BGR corrected、confidence.10、floor.01、NMS.70、bbox match.50、imgsz768、crop margin15%，原matching/mask/crop/size定義不改。

未來若重新取得明確授權，順序為C4 train768→H4 train1024→pair validity→C4 eval768→H4 eval768。不可看C4 final evaluation後決定是否訓練H4。H4不得用eval1024作primary comparison。

訓練中validation也事先明定兩組768，以免stock trainer自動沿用H4的1024造成不同checkpoint selection。這是兩組共同約束；F0未建置／執行正式trainer，F1啟動前仍須驗證validation loader與validator args都確實768。

Primary endpoint為very-small Recall<0.25%，support49；key secondary為<0.10% Recall，support27。另報Small<1%、Medium/Large、Precision/Recall/F1、Crop complete、TP/FP/FN、No ROI與single/multi-GT Recall。

Advancement gate全部須通過：H4 very-small TP≥C4+2，small/medium TP≥C4−1，Precision/F1≥C4−1pp，large TP≥C4，crop-complete≥C4−1。+2/49=+4.081633pp是operational research threshold，不是統計顯著；不得事後改+1。

Scheduled optimizer opportunities固定3741，applied+skipped=3741且unknown0；AMP skips不同需如實報告，不重訓。任何arm不到300epochs則paired fixed budget invalid、不補跑。F1若OOM須中斷，不能改batch續跑。

## 5. 唯一一次合成GPU測試

| 項目 | 結果 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Total VRAM | 7.996 GiB |
| Training-shaped input | 1024×1024，batch4，AMP true |
| 合成資料 | Seed42 noise，4張512 source，4×4合法矩形／張，共64GT |
| Mask target | [4,256,256]，stock overlap polygon raster |
| 執行範圍 | 1次warmup forward+loss；1次固定forward+loss+backward |
| Forward / Backward | True / True |
| OOM | False |
| Peak allocated | 5.736 GiB（6159095808 bytes） |
| Peak reserved | 5.820 GiB（6249512960 bytes） |
| Nonfinite gradients | True，NaN/Inf未進一步細分 |
| GradScaler scale | 65536.0 → 32768.0 |
| Model parameters unchanged | True |
| Optimizer steps / checkpoint saves | 0 / 0 |
| Real dataset pixels / research inference | 0 / 無 |
| Research accuracy、mAP、Recall | 未計算 |

Warmup與stress loss皆通過finite guard；非有限值出現在反向後梯度。沒有保存或比較loss數值來選scale。GradScaler只做scale/backward/unscale/update，未呼叫scaler.step或optimizer.step；fresh AdamW state仍為空，parameter值前後hash相同。

峰值只代表CUDA allocator中此合成工作負載，不涵蓋所有driver／其他程序顯存。因step禁止，Adam lazy moment states未配置；沒有真實Mosaic、DataLoader與長期fragmentation，因此不能保證300epochs能跑完。BatchNorm buffers可能在這個可拋棄RAM模型中變化，未保存，亦未將其當研究權重。

### Guard診斷與來源保留

原始結果中的 `update_save_guards_installed=false` **完整保留**。讀碼查明Ultralytics匯入時以wrapper替換torch.save，wrapper持有先前的禁止save函式，因此直接函式identity的合取檢查不能等同完整guard驗證。沒有宣稱該probe PASS，也未重寫原結果；source沒有step／真實image decode呼叫，參數不變且無權重寫出。

實際執行版本保存在 `executed_resource_smoke.py`，SHA256符合execution.lock。之後只以CPU測試補強「workspace外影像／權重檔也拒絕讀取」的存取政策；新版本未重跑GPU。這項hardening不追溯冒稱第一次使用了較新的guard。

## 6. 25項完成問題

| # | 問題 | 答案 |
|---:|---|---|
| 1 | Patch STOP？ | 是，V1/V2/V3未變 |
| 2 | 使用E_PATCH labels？ | 否 |
| 3 | 回到original771？ | 是，影像／標籤hash符合 |
| 4 | 只差training imgsz？ | 是，除必要arm/id/output欄位 |
| 5 | C4 imgsz？ | 768 |
| 6 | H4 imgsz？ | 1024 |
| 7 | Batch都4？ | 是 |
| 8 | Accumulation compensation？ | 無 |
| 9 | Sampling相同？ | 是，uniform without replacement |
| 10 | Anchor ordering相同？ | 是，300×771已封存 |
| 11 | Initialization相同？ | 是，相同hash、fresh optimizer |
| 12 | Optimizer/loss/augmentation相同？ | 是 |
| 13 | 1024×4 resource PASS？ | 記憶體PASS，但數值安全FAIL |
| 14 | Peak VRAM？ | Allocated 5.736、reserved 5.820 GiB |
| 15 | Smoke用real pixels？ | 否 |
| 16 | Smoke optimizer.step？ | 否 |
| 17 | Primary endpoint？ | <0.25% Very-small Recall，49GT |
| 18 | Advancement？ | +2 very-small TP及全部safety gates |
| 19 | Final eval imgsz？ | 兩組768 |
| 20 | F1 primary eval1024？ | 禁止 |
| 21 | Locked test？ | 未讀取／未使用 |
| 22 | CO2？ | 未讀取／未使用 |
| 23 | 正式training？ | 無；只有授權的合成forward/backward |
| 24 | Research-data inference？ | 無 |
| 25 | 可以放行F1？ | 否，本次F0 BLOCKED，待另次明確決策 |

## 7. 驗證與封存

156 passed in 3.63s。依TDD先驗證config差異、資料角色、合成輸入與存取邊界，再補齊共同recipe和既有D系列回歸；pytest未重跑GPU smoke。事前6份protocol/config/order/evaluation檔案雜湊未變。

1646個繼承保護檔案前後hash相同。未來C4/H4輸出目錄均不存在。F0新檔案、報告、raw smoke result、executed source及test evidence皆列入 `F_HIGHER_SCALE_V1_freeze.json`。詳細數值以resource_smoke_result.json為準，未把失敗遮蔽成PASS。

**停止在F0。** 本階段未測960/896/832、未降batch、未作accumulation補償、未重新跑synthetic backward、未開始F1。下一個決策應先釐清AMP非有限梯度及guard探針的驗證方式；須另行明確授權，不能自動延伸實驗。
