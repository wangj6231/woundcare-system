# Phase E1 — Patch-Based Training 預登錄與幾何可行性稽核

日期：2026-09-25。狀態：**BLOCKED，不得開始 E2**。

本文件封存的是 V1 預登錄、幾何模擬結果及阻擋證據，不是訓練完成報告，也不是部署許可。没有新訓練、fine-tuning、模型載入或推論；沒有讀取 validation / locked test / CO2Wounds / external test 的影像像素。既有 control 的 aggregate 結果僅用於鎖定比較門檻，未參與 patch 選擇。

## 1. 結論與立即停止原因

固定 training manifest 共 771 張，重新核對為 **161 張 eligible images、182 個 very-small targets**。182 個 target 的 bbox 均可完整放進 256×256 patch，**但不能因此聲稱 182 個 polygon 已安全保留**。

Shapely 2.0.6 的幾何有效性檢查在 eligible images 中發現 **17 個自相交／自接觸 polygon，分布於 17 張影像**。其中 15 個本身為 very-small GT，另 2 個為 eligible 影像中的其他 GT。逐一走完全部 182 個 round-robin target cycles：

| 檢查 | 結果 | 解讀 |
|---|---:|---|
| bbox 完整包含 | 182/182 | 只是 bbox 的空間可行性 |
| polygon transform 全部通過的 cycles | 163/182 | 89.56%；所選 target retention = 1.0 |
| 被阻擋的 cycles | 19/182 | 10.44%；不能輸出可用的完整 patch labels |
| 成功 cycles 的 invalid output polygons | 0 | 分母限 163，不能外推為全數通過 |
| 成功 cycles 的 empty eligible patches | 0 | 其餘 19 為未通過／未驗證，不記作 0 |

同一張 `fuseg__0181.png` 有 3 個 eligible targets，該圖一個 invalid polygon 使三個 cycles 都被擋下，故 17 張影像對應 19 個 blocked cycles。

這是**幾何表示不符合本次 strict clipping contract**，不等同臨床標註錯誤、資料集不可用，亦不自動推翻歷史模型結果。Raster mask 的填充語義與 simple polygon topology 並非同一件事。本次沒有採用 `buffer(0)`、`make_valid`、輪廓簡化、拆分 GT、刪除影像或重寫 labels。依使用者的 fail-closed 規則停止，不能透過 silent repair 宣稱 V1 PASS。

## 2. 研究問題與證據邊界

唯一比較：**FULL_IMAGE vs GT_CENTERED_PATCH_REPRESENTATION**。architecture、initialization、771 training sample IDs、uniform anchor sampling、optimizer、loss、augmentation configuration、scheduled budget 與 evaluation 固定。

E0：49 個 validation very-small GT 中 18 個至少 9/10 模型漏檢，16 個全部漏檢、22 個全部偵測；219 個 miss observations 中 121 無 retained overlap、67 有 overlap 但 IoU 不足、31 在 frozen confidence 以下、0 matching competition。Sampling V2 的 very-small anchor exposure 增加約 42%，仍未穩定改善。

以上僅支持 patch 是 **TESTABLE_NEXT_HYPOTHESIS**，不是 PROVEN_SOLUTION。不得用 E0 persistent IDs、validation confidence、failure outcomes 或人工挑選「相似失敗案例」來產生 training patches。原始 182 個 cycles 全部被稽核，沒有只選成功案例。

## 3. Comparator identity 與順序

Primary 是 **D2 Fresh Uniform Control seed42**，即 D2 原封不動沿用的 D1 seed42 C，不是 Sampling S，也不是重新訓練的 control。

- Control best SHA256：`0d153bb44d9f1d26df72a7d0c450d095d440236a875f41d7138ade87d10dadd7`。
- 共同 initialization 是已完成的 ISIC auxiliary checkpoint，SHA256：`1ec926026473beafafb1ef8da6aa6efcc13d38c391ec8db48006c5d793d5f2a3`；不是从 FUSeg-trained weights 接續。
- Secondary：Historical Phase C corrected model。
- Context only：D2 Sampling Experimental；禁止拿它當 patch control。

已逐列讀取 historical control 的 consumed-anchor telemetry，重建 **300 epochs × 771 = 231,300 次 sample-ID ordering**。每 epoch 使用同一 canonical manifest order 與 `PCG64 SeedSequence([42, zero_based_epoch])` permutation，全部與保存 order 一致。

`EXACT_ANCHOR_ORDER_PARITY = VERIFIED` 的範圍是「historical order 可精確重建」，**不是尚不存在的 E2 consumption 已被驗證**。將來仍需逐 batch 記錄實際 consumed IDs。取樣每 epoch 無放回、每個 ID 一次；不得沿用 A/B/C = 2/1.5/1 加權取樣。

## 4. 固定 patch 規則

1. 僅使用 frozen training GT，bbox area / 512² < 0.0025 才是 eligible target。
2. 保持原始 nonempty annotation instance order。target = eligible_GT_list[zero_based_epoch mod list_length]。
3. 尺寸固定 256×256，不比較其他大小、不根據 validation 調整。
4. 中心為 selected bbox center；left = clamp(cx−128, 0, 256)，top 同理，right/bottom 加 256；不 random shift。
5. 所有 GT 與 rectangle 做幾何 clipping，轉成 patch-local coordinates，再除 256 正規化。不能保留舊 512 座標。
6. Selected target 要完整保留，使用 rectangle covers 原 polygon 的檢查，不以近似 area retention 接受被切掉的 target。
7. 其他 GT 的非零合法面積交集保留；fully retained、partially clipped、dropped 全部按原 GT identity 記錄。純 line/point 邊界接觸面積為 0，不視為可見 lesion 面積。
8. 若交集成為多個正面積 polygon 或有 hole，不擅自改變 GT instance identity，回報尚未預登錄的表示問題。本次 19 個 blockers 均為 source polygon validity，不是 multipart blocker。
9. 不符合 eligibility 的 610 張保持 full image：**18 negatives + 592 non-eligible positives**。影像與標籤均未改寫。

尺度意義：512→768 的線性尺度為 1.5；256→768 為 3。Selected target 的 representation 線性尺度相對 2 倍、bbox area ratio 4 倍。這不是 resolution experiment、不是新增影像細節，也不是預測性能改善。

### 尚未解決的 raster / runtime 契約

目前實作是**連續座標的幾何模擬器，不是可掛載訓練的 adapter**。bbox center 可能包含半像素／小數（也含 normalized annotation 的有限精度）。直接把 left/top 取整、用整數 slice 或指定插值，皆會加入未明訂的 raster alignment 決策。本次保留精確公式，不暗中選擇。

從固定版本 Ultralytics 8.3.53 的本機 source 查核：`BaseDataset.get_image_and_label` 會先載入／resize image 再 `update_labels_info`，後者會 resample segments；Mosaic companion 又會呼叫 `get_image_and_label`。因此未來 patch 必須在原始 source coordinates、label resampling 與 augmentation 之前處理，anchor 和 companion 都要恰好一次，不能在 768 影像上直接切 source-space 256 方框。

已記錄相關 source SHA256；**沒有掛載／驗證 runtime Mosaic adapter**。未來 workers=2、prefetch、close_mosaic reset 下，epoch 必須隨樣本帶 immutable token，不能只用可能提前更新的 global epoch。這些 integration 問題不能用 synthetic polygon tests 通過來取代。

## 5. Multi-GT 與 context loss

Eligible 161 張中 **54 張為 multi-GT（33.54%）**。以下按 unique target cycles 統計，不是 300 epochs exposure，也不是 182 個獨立影像：

成功的 163 cycles 包含 99 個 single-GT source cycles、64 個 multi-GT source cycles；58 個輸出 patches 仍含 ≥2 GT。

| 成功 cycles 中的其他 GT | 次數 |
|---|---:|
| fully retained | 77 |
| partially clipped | 23 |
| dropped | 35 |
| 合計 | 135 |

加上 163 個 selected targets，成功 cycles 中原 GT 共 298 次：240 fully retained、23 partially clipped、35 dropped。這是重複的 GT-in-cycle 觀察，不代表 298 個獨立 lesions。

| 成功的 64 個 multi-GT cycles | 次數 | 比例 |
|---|---:|---:|
| 全部 GT 完整保留 | 30 | 46.88% |
| 無 partial、但有 dropped | 15 | 23.44% |
| 有 partial、無 dropped | 8 | 12.50% |
| 同時有 partial 與 dropped | 11 | 17.19% |

以 54 張 multi-GT 影像為分母：24 張所有 cycles 都完整保留（44.44%）；12 張至少一個成功 cycle 有 clipping（22.22%）；15 張至少一個成功 cycle 有 drop（27.78%）；9 張有 blocked cycle（16.67%）。這些項目可以重疊，不能加成 100%。Blocked cycles 的完整 clipping / drop 計數不確定，不補零。

成功 cycles selected bbox 到四個 patch boundaries 的最小距離：最小 0 px、中位 116.50 px；source bbox area ratio 中位 0.140381%，patch 中為 0.561523%。邊界 target 可完整保留但沒有外側 context；retention=1 不等於 context 完整。

## 6. 未來 training recipe（只預登錄，不執行）

YOLO11m-seg，Wound class 0，seed42，imgsz768，batch4，epochs300，patience80；AdamW lr0=0.0005、betas=(0.937,0.999)、decay weight group=0.0005／bias與norm=0；nbs64、gradient clip norm10、AMP true。Cosine scheduler，warmup5 epochs、warmup momentum0.8、bias lr0。

完整 training_args、architecture/source hashes、augmentation、loss、runtime_optimizer、budget 已逐欄複製 frozen control config 到 `E_PATCH_V1_protocol.json`。包括 workers2、原 accumulation schedule、box7.5/cls0.5/dfl1.5、mask_ratio4、overlap_mask true，以及既有 augmentation probabilities（例如 mosaic0.1、close_mosaic30、mixup0、copy_paste0）。不新增 crop、copy-paste、scale jitter、loss weighting 或改變負例頻率。

這份 recipe 保留 historical runner 路徑是為可稽核性，**不是可直接執行的 E2 config**。舊 control runner 不具備 patch adapter，不能直接拿來聲稱已進行 patch intervention。

Control 已完成 300 epochs，scheduled3741、applied3731、skipped10、unknown0。未來 Patch 必須完成300、scheduled3741、applied+skipped3741、unknown0。若早停或 crash：FIXED_BUDGET_COMPARISON=NOT_VALID，不補跑、不延長、不重跑；AMP skips 不同要揭露，不能宣稱成功更新數完全相同。

## 7. 固定 evaluation 與 advancement gate

Evaluation protocol 是 D2 的 byte-identical JSON 副本（final freeze 再驗證 SHA256）。未來僅 FUSeg 191 development validation full images、imgsz768、corrected RGB→BGR contract、prediction floor0.01、confidence0.10、NMS IoU0.70、match IoU0.50、crop margin15%，保留原 union ROI 與 ≥95% GT pixels crop completeness 定義。禁止 validation tiling / patches / multi-crop / test-time ensemble。

Primary：Very-small Recall <0.25%，support49，control27/49（55.10%）。Key secondary：<0.10%，support27，control11/27。其餘必報 Small <1%、Precision/Recall/F1、Medium/Large、crop、TP/FP/FN、No ROI、single-/multi-GT recall。

| Gate（全部必須通過） | Control | Patch 最低要求 |
|---|---:|---:|
| Very-small TP | 27/49 | **29/49** |
| Small TP | 104/137 | ≥103 |
| Precision | 203/233 | ≥203/233 − 0.01 |
| F1 | 406/474 | ≥406/474 − 0.01 |
| Medium TP | 85/90 | ≥84 |
| Large TP | 14/14 | ≥14 |
| Crop-complete positives | 165/186 | ≥164 |

門檻採 exact rational arithmetic，不以 rounded percentage 判決。+2/49 = +4.081633 percentage points 是事前 operational improvement criterion，**不是 statistical significance**，不得看結果後降到 +1。Gate PASS 也不代表 App replacement 或 external test 已獲許可。

未來才可將 E0 的 18 / 16 persistent cases 作 post-hoc descriptive diagnostic；其 IDs 不進 training patch generation。先 seed42，任何 multi-seed confirmation 都需另立 protocol 與授權。

## 8. Invalid source polygon 清單

GT instance ID 為原 annotation 的 zero-based index；全部 source label hashes 與每個 cycle 原因都在 geometry audit JSON。沒有刪除／修復以下標註。

| sample_id | GT instance | Very-small target? |
|---|---:|---|
| fuseg__0071.png | 0 | 是 |
| fuseg__0110.png | 0 | 是 |
| fuseg__0179.png | 0 | 是 |
| fuseg__0181.png | 2 | 是 |
| fuseg__0314.png | 0 | 是 |
| fuseg__0400.png | 0 | 是 |
| fuseg__0589.png | 1 | 是 |
| fuseg__0605.png | 0 | 是 |
| fuseg__0708.png | 0 | 是 |
| fuseg__0749.png | 0 | 是 |
| fuseg__0750.png | 0 | 否 |
| fuseg__0863.png | 0 | 是 |
| fuseg__0870.png | 0 | 是 |
| fuseg__0923.png | 0 | 是 |
| fuseg__0945.png | 0 | 是 |
| fuseg__0966.png | 3 | 否 |
| fuseg__0996.png | 0 | 是 |

## 9. 測試、資料存取與保護證據

- 28 個新增 E1 synthetic tests 通過；連同 E0/D2/D1/D0.1/D0 選定回歸套件共 **179 passed**。這不是全專案測試聲明。
- 六組 synthetic 512×512 圖：center、corner、edge、multiple GT、cross-boundary secondary、tiny polygon；均已逐圖檢視。Tiny case 視覺上很小，以 exact geometry assertions 補充，不以放大外觀判定有效。
- 未抽取原 training pixels 做20例視覺圖：這是可選步驟，source geometry blocker 已成立，且 raster 契約尚未確定。`TRAINING_PIXELS_USED_FOR_PATCH_AUDIT=false`。
- 只讀771張 training image headers 驗證512×512，並讀原 training labels；training image/label bytes 僅用於 SHA256。Validation manifest 的 IDs/hashes 僅作 exclusion check，不開 validation images/labels。
- E0 aggregate summary 與 control aggregate results 是報告／門檻輸入，不是 patch generator 的輸入。Generator API 不接受 predictions 或 failure IDs。
- 稽核／finalization 程序拒絕 import torch、ultralytics、tensorflow、onnxruntime。讀 Ultralytics 原始碼不等於載入模型。
- 建立前後 **1,570 個受保護來源檔案 SHA256 未變**；涵蓋 training files、frozen recipe、initialization、control weights、telemetry 等。此數字不是聲稱整個工作區所有檔案都被掃描。
- 新增檔案採 exclusive creation，不覆寫既有 research artifacts。`experiments/results/e_patch_v1_seed42/` 必須不存在；本次沒有建立該目錄。
- 沒有更換 App weights、沒有重跑 control、沒有讀取 locked test / CO2 / FUSeg official test。

## 10. 使用者要求的25項回答

| # | 問題 | 結論 |
|---|---|---|
| 1 | Eligible training images? | 161 |
| 2 | Eligible very-small GT? | 182 |
| 3 | 256 patch 100%保留targets? | bbox182/182；polygon僅163/182已驗證，故完整gate不通過 |
| 4 | Invalid polygons? | source17；19cycles被阻擋；成功163cycles無invalid輸出 |
| 5 | Multi-GT images? | eligible群中54 |
| 6 | 其他GT完整／partial／drop? | 成功cycles77／23／35；blocked部分不假造 |
| 7 | Empty patch? | 成功cycles0；全部182未能通過檢查 |
| 8 | Negative full image? | 是，18張不變 |
| 9 | Non-eligible positive? | 是，592張不變 |
| 10 | Uniform sampling? | 是，計畫固定；不是加權sampler |
| 11 | Validation outcomes用於patch? | 否；仅aggregate control數據用於gate |
| 12 | Anchor frequency改變? | 否，771個IDs各一次/epoch；其他GT的可見exposure會改變且已揭露 |
| 13 | 理論空間尺度? | 線性2倍、area ratio4倍；非performance gain |
| 14 | imgsz改變? | 否，768 |
| 15 | loss改變? | 否 |
| 16 | augmentation config改變? | 否；runtime mosaic接線尚未驗證 |
| 17 | optimizer改變? | 否，完整recipe沿用 |
| 18 | evaluation改變? | 否，full image固定protocol |
| 19 | Primary endpoint? | Very-small Recall <0.25%, support49 |
| 20 | Advancement gate? | 表7全部符合，primary至少29/49；且先滿足完整性與固定budget |
| 21 | 讀locked test? | 否 |
| 22 | 讀CO2? | 否 |
| 23 | 訓練? | 否 |
| 24 | inference? | 否 |
| 25 | 可授權E2? | **否：geometry未全數通過，runtime/raster契約亦未完整驗證** |

## 11. 封存產物與下一步邊界

`experiments/protocols/` 中六份：`E_PATCH_V1_protocol.json`、`E_PATCH_V1_training_transform.json`、`E_PATCH_V1_training_manifest.json`、`E_PATCH_V1_geometry_audit.json`、`E_PATCH_V1_evaluation_protocol.json`、`E_PATCH_V1_freeze.json`。

`experiments/results/e_patch_v1_preregistration_audit/`：六組 synthetic figures、source snapshot、integrity、verification。這是 E1 audit 目錄，不是 E2 result 目錄。

Freeze 的 `status=FROZEN_BEFORE_TRAINING` 表示本次規則與**失敗證據**已不可變地封存；真正允許晉級的狀態仍是 `PHASE_E1_STATUS=BLOCKED`、`READY...=NO`。不得只看 FROZEN 字樣就執行。

若另獲授權，下一步應是**另立不訓練的 V2 topology/raster 契約審查**，先決定如何在不改 GT 身分／可見面積語義的前提下表示這17個輪廓，並評估任何 label normalization 是否也必須反映於 control 以維持可比性；不能悄悄修 E1 frozen檔案。還須明確定義 fractional raster alignment、Mosaic companion transform、epoch token 的 dependency-injected 測試。此文件沒有代替使用者核准上述變更，也沒有開始 V2。

```ini
PHASE_E1_STATUS = BLOCKED
READY_FOR_PHASE_E2_PATCH_SEED42_TRAINING = NO
PRIMARY_BLOCKER = FAIL_PATCH_LABEL_INVALID
NEW_TRAINING_AUTHORIZED = NO
APP_MODEL_REPLACEMENT_AUTHORIZED = NO
EXTERNAL_TEST_AUTHORIZED = NO
STOP_AFTER_E1 = YES
```
