# Phase E1.1 — Polygon Topology / Raster Contract 與 Paired Patch 預登錄

日期：2026-09-25。正式狀態：**PHASE_E11_STATUS = BLOCKED**。

本階段完成的是 model-free 稽核、候選表示的驗證、source-stage adapter 與 paired 設計的封存，不是成功解除全部 blockers。沒有開始 C2/P2 訓練，沒有模型載入、推論、App replacement 或 external test。

## 1. 結果摘要

| 必要條件 | 本次結果 | 判定 |
|---|---:|---|
| E1 的17個問題 source polygons 全數重新 audit | 17/17 | 已完成檢查 |
| 找到合法且 exact raster-equivalent 的 canonical representation | **1/17** | FAIL，尚有16未解決 |
| 全部182個 target cycles 重新檢查 | 182/182 | 未沿用V1的PASS |
| 完整通過 vector + raster checks | **123/182** | FAIL |
| source尚未解決而阻擋 | 18 cycles | 不能使用原invalid polygon替代 |
| 有source view但patch raster不完全一致 | **41 cycles** | FAIL，不以IoU近似放行 |
| 可做raster檢查的cycles中，selected source-mask pixels完整保留 | 164/164 | 不等於transformed polygon raster等價 |
| 可檢查cycles的invalid transformed polygons / empty patches | 0 / 0 | 限164個分母；其餘18未驗證 |
| 最小等價adapter、兩個worker、companion token測試 | PASS | 非GPU訓練／完整Torch DataLoader測試 |
| 新增與選定回歸測試 | **210 passed** | 包含31個E1.1 tests |
| 受保護來源檔案未變 | **1,594** | 包含全部E_PATCH_V1封存產物與原training files |

**沒有將XOR=1當成「足夠接近」，沒有填洞、丟棄component或拆成多個wound。** 16個未解決表示「本次明列的確定性候選未通過」，不是數學上證明不存在任何無損表示，更不是宣稱這些傷口標註具有臨床錯誤。

## 2. 權威source raster語義

以本機固定的 Ultralytics **8.3.53** source code 查核，不import套件或載入wound model。關鍵函式由該檔案AST擷取，以NumPy/OpenCV獨立執行；完整source SHA256存入topology contract。

固定鏈：

```text
original normalized annotation coordinates
→ np.float32（與verify_image_label相同）
→ segmentation resample_segments
→ denormalize到512 source canvas
→ polygon2mask：int32轉換 → cv2.fillPoly → ratio1 resize
→ per-instance binary M_source
```

`update_labels_info` 的resample數為1000；只有該影像最大vertex count >1000才用max+1。保留原instance order與identity；若candidate增加vertex count，會對**整張影像所有siblings**重做raster比較，不能只驗candidate而漏掉共享resample count造成的改變。

這是**baseline-compatible source-stage reference**，不是聲稱等同歷史訓練所有random augmentation後的mask tensors。原訓練loader會resize到768、做augmentation；最後Format denormalize、polygon2mask、mask_ratio=4，overlap_mask=true會依面積排序並合成instance-index mask。那些downstream設定沒有在E1.1執行或替換。

以source mask而非Shapely有效性當標註語義權威；Shapely只作candidate／clipping的可表示性檢查。mask hash包含shape、dtype、contiguous bytes，避免只比浮點vector area。

前景connected-components固定採8-connectivity，另附4-connectivity結果；hole採RETR_TREE奇數depth計數。本次17個source masks均為8-connected component=1、holes=0，但其中4個在4-connectivity下有2或3個components。**8-connected不保證一定能表成一個合法simple polygon**，也不能用這個數字略過instance restriction。

## 3. Canonicalization候選與結果

原本合法simple polygon保持identity，不簡化、不重畫。E1明列的17個invalid eligible source polygons才進候選流程；候選順序固定為：

1. authoritative raster的external contour，CHAIN_APPROX_NONE，不簡化。
2. 原source的`make_valid`候選。
3. 原source的`buffer(0)`候選。

後兩者**不是ground-truth repair權威**。候選需為單一、無hole、正面積、float32後仍有效的polygon，再通過source及siblings的exact binary comparison；否則保留拒絕原因，不寫入可用canonical欄位。MultiPolygon不拆instance；GeometryCollection不默默捨棄line/point；沒有convex hull或fill-hole處理。

在候選契約封存前，另曾做training-only的contour offset/subpixel buffer(.125)可行性探查，17例均未達exact raster equivalence，故未晉級成label view。這不是validation調參，亦不隱藏失敗的候選探索。

| sample_id | GT ID | Source raster area(px) | 本次結果 |
|---|---:|---:|---|
| fuseg__0071.png | 0 | 22 | unresolved：candidate topology／multipart |
| fuseg__0110.png | 0 | 164 | unresolved：合法候選仍差1或5px |
| fuseg__0179.png | 0 | 79 | unresolved：candidate topology／multipart |
| fuseg__0181.png | 2 | 113 | unresolved：candidate topology／multipart |
| fuseg__0314.png | 0 | 150 | unresolved：candidate topology／multipart |
| fuseg__0400.png | 0 | 317 | unresolved：合法候選仍差2px |
| fuseg__0589.png | 1 | 204 | unresolved：合法候選仍差1或2px |
| fuseg__0605.png | 0 | 249 | unresolved：candidate topology／multipart |
| fuseg__0708.png | 0 | 372 | unresolved：合法候選仍差1px |
| fuseg__0749.png | 0 | 109 | unresolved：合法候選仍差2px |
| fuseg__0750.png | 0 | 1398 | unresolved：合法候選仍差2px；不是very-small target，亦有納入 |
| fuseg__0863.png | 0 | 248 | unresolved：candidate topology／multipart |
| fuseg__0870.png | 0 | 210 | unresolved：topology／collection／multipart |
| fuseg__0923.png | 0 | 257 | unresolved：合法候選仍差2px |
| fuseg__0945.png | 0 | 207 | unresolved：合法候選仍差1px |
| fuseg__0966.png | 3 | 5864 | unresolved：candidate topology／multipart；不是very-small target，亦有納入 |
| **fuseg__0996.png** | **0** | **124** | **EXACT_RASTER_EQUIVALENT：XOR=0、intersection=union=124、IoU=1** |

`fuseg__0996.png`的accepted方法是RASTER_EXTERNAL_CHAIN_NONE，candidate polygon、source/canonical raster SHA256、面積與完整lineage均已保存。未接受的16筆canonical_polygon與accepted-result XOR欄位為null，**不是0**；每個可rasterize的候選有自己的XOR／intersection／union／IoU紀錄。

### Label view範圍與不可忽略的限制

沒有改source dataset。新view放在`experiments/protocols/E_PATCH_V2_canonical_labels/view.json`，每個source label SHA256對應每個parsed source/canonical polygon hash。未解決instance用null，adapter對C2與P2都拒絕載入，不能拿舊invalid polygon當fallback。

對非eligible影像仍完整保留source labels，並記錄其中另有**19個topology-invalid source polygons**為`IDENTITY_NONELIGIBLE_UNMODIFIED_SOURCE`。它們不在E1的17個patch-source問題清單，不被單方修補、不做patch；此identity政策兩組相同。**本次不能宣稱整套training label view的所有polygon都已變成simple polygons**。若未來要擴大修補範圍，須另行預登錄，不可靜默改成另一套資料。

## 4. Integer patch contract與182-cycle重算

Source固定512×512，patch固定256×256。Selected target為原training GT eligibility/order決定的instance；center取該instance canonical bbox。規則：

```text
left = clamp(floor(cx - 128), 0, 256)
top  = clamp(floor(cy - 128), 0, 256)
right = left + 256
bottom = top + 256
image_patch = image[top:bottom, left:right].copy()
```

不改用round/ceil、不random shift、不先resize768再crop。流程是source image → canonical source label → integer source patch → patch-local normalized labels →原resize/augmentation。單一instance clipping出多個正面積parts或holes時fail closed，不產生新的wound IDs。

逐GT比較：`M_source[top:bottom,left:right]`與transformed local polygon經同一resampling/raster primitive產生的`M_vector_patch`，要求XOR=0。保留selected source mask所有pixels只是必要條件，並不足以保證後者的量化／重採樣結果相同。

此次重新執行全部182個unique target cycles，沒有繼承V1的163個PASS：

- 18個cycles因16個source polygons仍unresolved，未進clipping；mask retention等欄位為unknown。
- 其餘164個cycles已做vector及raster audit，selected authoritative mask全部完整保留。
- 其中**41個cycles、50次GT-instance-in-cycle比較出現raster mismatch**；僅123個完整通過。
- 每次mismatch為1–11px，合计147個XOR pixel observations；12次是selected GT，38次是其他GT。這些是重複觀察，不是147個獨立lesions。
- 50次不符中36次source-mask狀態為FULL、14次為PARTIAL；故問題不只發生在被邊界切開的GT。
- 沒有因差異小就放寬容忍度，沒有改primitive、重採樣數或浮點精度來追求PASS。

例：`fuseg__0067.png / target0 / GT0`差1px；`fuseg__0037.png / target0 / GT1`差3px。這是本次source-space vector→raster契約的觀察，**不是新模型預測結果或臨床性能下降**。

## 5. Multi-GT context loss仍存在

Eligible multi-GT影像仍為54張。可執行raster audit的164cycles中原GT共有299次，authoritative source-mask visibility為FULL241／PARTIAL23／DROPPED35；扣掉selected targets後，其他GT為FULL77／PARTIAL23／DROPPED35。

64個可稽核的multi-GT cycles：30全部完整保留；15有drop無partial；8有partial無drop；11兩者都有。**26個cycles selected target完整保留但另一個lesion dropped**。

上述context計數涵蓋有raster mismatch的cycles，是source-mask slice的描述，不能把它們都叫做可訓練patch。18個source-blocked cycles不補零。增加target尺度可能損失其他傷口／全局context的風險並未因canonicalization而消失。

## 6. Source-stage adapter與runtime contract

新增`experiments/patch_dataset_adapter.py`，是dependency-injected最小等價source-stage adapter，沒有training driver。C2與P2共用同一label view與adapter infrastructure：

- C2：full512 source image + 相同canonical labels，representation identity。
- P2：eligible training sample crop一次，其餘full image。
- Anchor與Mosaic companion走同一fetch；每次都帶immutable `FetchToken(sample_id, epoch_token, generation)`。
- Companion由per-anchor scoped view繼承parent epoch，不用global mutable epoch。
- 同sample+epoch跨role／worker／reset generation都得到同patch；next epoch依original target-list round-robin。
- Prefetched舊generation結果丟棄，不以當下global epoch重新解讀；generation不影響patch內容。
- 拒絕double patch、768-before-source-crop、non-training、locked／CO2 roles與unresolved labels。

Synthetic驗證採workers0與multiprocessing spawn workers2，預先提交14個跨epoch／reset-generation jobs再收取，輸出完全一致；模擬epoch269→270的close_mosaic reset語義。另直接擷取固定版本`BaseMixTransform.__call__`驗證真實companion fetch route，幾何混合部分使用recorder stub。測試用p=1只為強制經過分支，**未更改未來訓練的mosaic=0.1**。

重要限制：**沒有執行真正Torch DataLoader、完整Mosaic幾何變換或GPU trainer**；不能把minimal-equivalent驗證報成整套training runtime已跑通。未來獨立授權E2仍需在保留原companion selector、buffer identity、RNG、resize與augmentation設定的前提下接線，並驗證實際consumed telemetry。目前geometry gates未過，沒有可執行的E2訓練入口。

## 7. Fresh paired C2/P2設計

Primary comparator改為**Fresh Canonical Full-Image C2 vs Fresh Canonical Patch P2**，兩者fresh optimizer，共用同一label view。Historical D2 C seed42與Historical Phase C corrected僅作reference，不能拿歷史27/49直接決定V2 gate。

共同設定完整複製固定recipe：YOLO11m-seg、Wound class0、seed42、imgsz768、batch4、epochs300、patience80、AdamW lr0=.0005、betas(.937,.999)、WD weight=.0005 / bias&norm=0、cosine schedule、5-epoch warmup、AMP、workers2、nbs64 accumulation schedule、gradient clip10、相同loss與全部augmentation probabilities。

共同ISIC auxiliary initialization SHA256：
`1ec926026473beafafb1ef8da6aa6efcc13d38c391ec8db48006c5d793d5f2a3`。

已重新產生並封存**300×771**未來sample-ID orders：uniform、without replacement、PCG64 SeedSequence([42,epoch])，不依賴historical process state。兩組引用完全相同order檔及hash。僅representation欄位有主要語義差異；其餘不同只有arm label、experiment ID與output routing。

每arm必須300 completed epochs、3741 scheduled optimizer opportunities、applied+skipped=3741、unknown=0。早停／crash為NOT_VALID，不補跑或延長；AMP realized update差異必須另報。不可看C2結果後再決定是否跑P2；未來執行應完成兩組後再讀固定operating-point比較結果，除非安全／完整性／budget gate觸發停止。

## 8. Evaluation與相對gate

兩組未來evaluation皆為原FUSeg191 development validation **full images**；validation labels不canonicalize、不patch、不tile。RGB→BGR corrected，confidence.10、prediction floor.01、NMS IoU.70、match IoU.50、imgsz768、crop margin15%，原metrics、size definitions與crop policy不變。Evaluation JSON逐byte等同原sealed protocol。

Primary為Very-small Recall <.25%，support49；最低改善仍+2GT（4.081633 pp），只是operational research criterion，不是statistical significance。以下全部相對**未來新C2的實測counts**：

| Gate | 事前門檻 |
|---|---|
| Very-small TP | P2 ≥ C2+2 |
| Small TP | P2 ≥ C2−1 |
| Precision | P2 ≥ C2−.01 |
| F1 | P2 ≥ C2−.01 |
| Medium TP | P2 ≥ C2−1 |
| Large TP | P2 ≥ C2 |
| Crop-complete count | P2 ≥ C2−1 |

使用exact rational counts，不用rounded percentage決策；不事後改+1。固定報key secondary <.10%/27與其他Small/Medium/Large、P/R/F1、TP/FP/FN、NoROI、single/multi-GT及crop metrics。歷史27/49只有reference意義，新的C2結果目前**不存在**。

即使未來seed42 PASS，`MULTI_SEED_PATCH_AUTHORIZED=NO`，App replacement/external test仍NO；不自動開下一階段。

## 9. 產物與完整性

`experiments/protocols/` 已新增：

- `E_PATCH_V2_topology_contract.json`
- `E_PATCH_V2_canonicalization_audit.json`
- `E_PATCH_V2_canonical_label_manifest.json`
- `E_PATCH_V2_raster_contract.json`
- `E_PATCH_V2_runtime_adapter_contract.json`
- `E_PATCH_V2_patch_geometry_audit.json`
- `E_PATCH_V2_control_config.json`
- `E_PATCH_V2_experimental_config.json`
- `E_PATCH_V2_evaluation_protocol.json`
- `E_PATCH_V2_freeze.json`

另有candidate label view、model-free實作、31項E1.1 tests與`experiments/results/e_patch_v2_preregistration_audit/`的anchor orders、source snapshot、integrity與verification證據。所有V1檔案保持原SHA256。新view明列`BLOCKED_NOT_TRAINING_READY`，null instance不能當作empty negative label。

沒有讀training原圖像素；有從training labels生成binary raster，並使用synthetic source image pixels測adapter。沒有validation/test/external影像讀取；沒有weight deserialization。Checkpoint僅讀bytes核對SHA256。

以下兩個未來result目錄**均不存在**：

```text
experiments/results/e_patch_v2_seed42_control/
experiments/results/e_patch_v2_seed42_experimental/
```

`FROZEN_BEFORE_TRAINING`只表示本次規則、候選範圍與BLOCKED evidence已封存，不代表執行許可。

## 10. 停止與下一個待決策項目

本階段停止，不擴大candidate search、不引入polygon tolerance、不修source labels、不啟動GPU。若再授權新的representation研究，必須同時解決：

1. 16個source masks如何在維持單一instance的前提下無損表示。
2. 為何source-mask slice與vector再rasterize在41cycles不同，以及是否需要另立mask-first／不同label representation的paired contract。

改變label rasterizer、segmentation resampling或直接改成mask-first representation都屬新契約，不可偷偷替換本次V2。這裡沒有證明patch提升性能，也沒有證明patch本身無效；目前缺少的是可安全執行的label/raster表示。

```ini
PHASE_E11_STATUS = BLOCKED
READY_FOR_PHASE_E2_PAIRED_PATCH_SEED42_TRAINING = NO
INVALID_SOURCE_POLYGONS_EXACTLY_RESOLVED = 1/17
PATCH_CYCLES_PASS = 123/182
MULTI_SEED_PATCH_AUTHORIZED = NO
TRAINING = false
MODEL_LOADING = false
INFERENCE = false
APP_MODEL_REPLACEMENT_AUTHORIZED = NO
EXTERNAL_TEST_AUTHORIZED = NO
STOP_AFTER_E11 = YES
```
