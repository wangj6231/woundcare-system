# Phase E1.2 — Mask-Authoritative Training Representation 可行性稽核

日期：2026-09-25。版本：E_PATCH_V3_MASK_FIRST。

## 1. 正式決策

```ini
PHASE_E12_STATUS = BLOCKED
MASK_FIRST_TRAINING_REPRESENTATION = NOT_FEASIBLE
BASELINE_AUGMENTATION_PARITY = NOT_VERIFIED
PATCH_BASED_TRAINING_ROUTE = STOP
READY_FOR_E2_MASK_FIRST_PAIRED_PATCH_TRAINING = NO
NEW_TRAINING_AUTHORIZED = NO
APP_MODEL_REPLACEMENT_AUTHORIZED = NO
EXTERNAL_TEST_AUTHORIZED = NO
```

本階段稽核工作已完成，但未達訓練放行條件。NOT_FEASIBLE 指「依目前封存條件及已驗證實作，不能放行」，不是證明所有 mask-first 方法皆不可能。未訓練、未載入模型、未 forward/inference、未用 GPU；test_images_used=0，未開啟 CO2Wounds 或外部測試。V1/V2 原樣保留。

## 2. 已通過：全訓練集的 source mask authority

771 張 FUSeg training images，共 965 個 annotation instances，全部重建兩次並逐 mask 比對，965/965 deterministic。每個原 annotation line ID 對應一個 mask，不拆 component、不合併 siblings。原始標註與影像都未重寫。

權威路徑嚴格沿用 E1.1：normalized polygon → float32 → pinned per-image resample_segments（1000，超過1000則max+1）→ ×512 → polygon2mask（int32 / fillPoly，ratio1）。其後只處理 mask，不作 repair、clipping polygon 或 re-rasterization。mask hash 同時包含 shape、dtype 與像素。

全771張共有 **36個 invalid source polygons**；原 eligible 圖片子集為17個，與 E1/E1.1 相符，其餘19個來自先前不需裁切的圖片，並非偷偷改寫舊結果。所有 source masks 為一個8-connected component，其中1個 instance有hole；洞與其 instance ID 都保留。這是 representation audit，不保證原醫療標註正確。

詳見 `experiments/protocols/E_PATCH_V3_training_mask_manifest.json`：逐 instance 記錄面積、4/8-connectivity、holes、bbox、polygon topology、image/label/mask SHA256。

## 3. Eligibility 明確變更

```ini
ELIGIBILITY_IDENTITY = CHANGED
```

| 定義 | Eligible images | Eligible GT |
|---|---:|---:|
| E1 原 polygon bbox | 161 | 182 |
| V3 authoritative-mask tight bbox | 142 | 161 |

移除21個 GT、無新增。完整身份差異列於 mask_authority_contract.json。新規則為 positive pixels 的 `[xmin,ymin,xmax+1,ymax+1]`，bbox area / 512² 嚴格小於0.0025。這是新的 V3 eligibility contract，**不能宣稱與 E1 intervention 完全相同**。未利用 validation 選擇定義。

256×256 source crop：中心取mask bbox，left/top=floor(center−128)再clamp至[0,256]，影像與所有instance masks共用整數slice。多目標按原GT ID排序，以zero-based epoch modulo目標數round-robin；generation不影響目標。

## 4. 182個歷史目標與新161個目標的裁切稽核

| 項目 | 原182個目標 | V3 161個目標 |
|---|---:|---:|
| Selected mask 完整保留 | 182/182 | 161/161 |
| Empty selected patches | 0 | 0 |
| Other GT FULL（cycle-observations） | 91 | 78 |
| Other GT PARTIAL（cycle-observations） | 27 | 27 |
| Other GT DROPPED（cycle-observations） | 38 | 36 |
| Selected保留但至少一個other GT被丟棄的cycles | 29 | 27 |
| Aggregate other-GT pixels retention | 67.51% | 65.47% |

0 unresolved source-topology blockers、0 vector/raster round-trip、0 raster-mismatch blockers。這裡的零差異是直接切權威mask，不是將不同representation的差異藏起來。Other-GT統計按每次cycle觀察計數，**不是unique GT數**；像素比例是總保留other pixels / 總source other pixels，不是每cycle比例的平均。Context損失仍然存在，不能因selected retention通過就忽略。

## 5. Augmentation及loss邊界：為何仍BLOCKED

以本機固定Ultralytics8.3.53源碼及SHA256查核，未import Ultralytics或建構YOLO。

| Stage | 分類 | 證據與限制 |
|---|---|---|
| Resize | MASK_ADAPTER_POSSIBLE | Image linear、mask nearest，bbox從當前mask重算 |
| Mosaic | MASK_ADAPTER_POSSIBLE | 4 tile幾何與ID唯一性通過；未驗證原buffer/companion RNG分布 |
| RandomPerspective | MASK_ADAPTER_POSSIBLE | 固定matrix共用image/mask；完整隨機matrix組合與box_candidates語義未整合驗證 |
| Flip | MASK_ADAPTER_POSSIBLE | H/V exact geometry通過；完整隨機順序未重播，recipe flipud仍為0 |
| HSV | MASK_ADAPTER_POSSIBLE | 固定gain LUT通過，mask不變 |
| Albumentations | MASK_ADAPTER_POSSIBLE | 原Blur/MedianBlur/ToGray/CLAHE未從recipe移除；diagnostic未執行完整鏈 |
| Format | MASK_ADAPTER_POSSIBLE | Native要求segments；新mask-native排序與index schema通過 |
| v8SegmentationLoss targets | MASK_NATIVE | Loss原本消費mask tensor；source-level與CPU batch結構可相容，未執行loss/model |

原生stock loader的Instances/Format仍以polygon segments為介面，不能直接宣稱它接受任意權威mask。自訂原語展示了不用polygon round-trip的技術路徑，但尚不是完整訓練adapter。最關鍵缺口：Mosaic companion buffer/RNG、RandomPerspective原box_candidates（wh>2、seg area_thr=.01、aspect<100）對mask-derived boxes的篩選語義、Albumentations鏈及native InfiniteDataLoader close/reset整合。

依使用者第15/38條，無法驗證baseline augmentation semantics就必須BLOCK。沒有偷偷關閉Mosaic、Perspective、Flip，也沒有把固定synthetic操作當作完整baseline replay。現階段停止patch路線，不延伸polygon heuristic。

### 最終target schema

768×768影像，mask_ratio=4，192×192 overlap mask。mask resize/downsampling一律INTER_NEAREST；不同於stock polygon2mask的uint8 INTER_LINEAR，**明確揭露而非宣稱byte-equivalent**。依downsample mask area降序排序，cls、normalized xywh bbox與instance ID同步；background=0，instance index=sorted row+1，後寫小mask覆蓋重疊像素。

Full-resolution空mask會有紀錄地移除；低解析度消失或overlap完全遮蔽仍保留target row並標記，不靜默創造或拆分GT。Loss中 `masks_i == (target_gt_idx+1)` 的index關係已由batch契約核對。這是無模型的schema驗證，不聲稱觀察到actual network prototype或跑過segmentation loss。

## 6. Control path characterization

合法simple synthetic polygon比較，完整紀錄見 `experiments/results/e_patch_v3_feasibility_audit/control_characterization.json`。

| Operation | Full-resolution XOR pixels | ratio4 XOR pixels |
|---|---:|---:|
| Identity | 0 | 37 |
| Resize512→768 | 300 | 72 |
| Horizontal flip | 0 | 45 |
| Fixed affine | 117 | 48 |

比較是pinned polygon raster primitive經固定座標transform，對上raster-native操作，不是完整歷史augmentation replay。這些差異只是characterization；未為匹配歷史而改動mask authority。C3/P3會共同使用新label view；歷史C/D2結果只能REFERENCE_ONLY。

## 7. 真實CPU DataLoader診斷

Synthetic與4張實際training samples均以Torch DataLoader、workers=0/2執行，prefetch_factor=2，pin_memory=false。Training subset：fuseg__0020、0022、0012、0011，依training labels決定invalid／eligible／multi-GT／其餘代表，不看validation outcomes。

兩個arm均驗證：image `[4,3,768,768]`、mask `[4,192,192]`、cls `[N,1]`、bbox `[N,4]`、batch_idx `[N]`；workers0/2的tensor hashes及metadata完全相同。Immutable FetchToken(sample_id,epoch_token,generation)避免prefetch讀取mutable epoch。診斷強制epoch269走Mosaic、270關閉；generation1建立新iterator/worker後，像素與generation0同epoch一致、無舊token混入。

**此forced branch不是production p=.1 RNG，new iterator reset也不是已驗證native InfiniteDataLoader reset。** 全程MODEL_LOADED=false、FORWARD_PASS=false，僅CPU tensors與影像處理。

## 8. 凍結的C3/P3設計（不可執行）

兩者同YOLO11m-seg、class0 Wound、seed42、同ISIC initialization SHA256、768/batch4、AdamW lr.0005、相同loss gains與全部augmentation數值；300epochs、patience80、scheduled optimizer calls3741，applied+skipped=3741且unknown=0。若未完成300epochs則NOT_VALID、不resume或補跑。

771 IDs/epoch，uniform without replacement，300×771共同anchor orders已凍結。不同欄位只有arm/id/output與representation：C3 full512；P3 mask-eligible patch256，否則full512。兩份config都execution_authorized=false、training_entrypoint=null。

未來endpoint仍Very-small Recall（49GT），P3相對C3 very-small TP≥+2；small/medium TP≥−1；Precision/F1≥−1pp；large TP不得下降；crop-complete≥−1image。191張development validation只維持原label與full-image評価：conf.10、floor.01、NMS.70、match.50、768、crop margin15%、RGB/BGR corrected。Evaluation protocol與V2 **byte-identical**；本階段未跑validation或改label。

## 9. 測試、完整性及停止

222 passed in 35.01s。TDD以mask/geometry、overlap target與真實DataLoader等使用者指定介面逐步驗證，不以mock網路產生訓練結果。

1615個繼承保護檔案hash全數未變（包含V1/V2及既有source snapshot）。Hash核對不等於checkpoint載入。Training image pixels僅讀4張；validation image pixels=0、test_images_used=0。兩個未來training output目錄均不存在。

所有JSON、程式與報告SHA256列在 `experiments/protocols/E_PATCH_V3_freeze.json`。報告不是performance改善證據；沒有新權重、沒有新mAP、沒有App replacement。

下一階段可討論 **HIGHER_INPUT_SCALE** 作單一研究介入，理由是保留full-image context且避免本次patch representation整合缺口；目前未驗證效果、未設定新門檻、未執行任何實驗。亦可另外討論feature-pyramid或loss/assignment，但不得同時混入。**本階段到此停止，E2未授權。**
