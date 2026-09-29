# Phase F1.3 — 配對偵測轉換、誤報與裁切稽核

PHASE_F13_STATUS = COMPLETE。僅使用 F1.2 已保存結果，未載入模型、未重新推論、未訓練。

## 方法與可追溯性

F1.2 原 gate=FAIL 保持不變。本次是 single-seed、asymmetric replacement recovery、同一 development validation 的 paired descriptive error analysis；不是統計確認、泛化估計或臨床驗證。不計算 p-value 或 bootstrap CI。

191 個 sample_id、241 個 GT 身分以相同影像與凍結 polygon 非空行的零起算 index 對齊。逐行 bbox 與 SHA256 均核對，保存 matching 為權威；固定規則重算只作一致性驗證。

Confidence=.10、prediction floor=.01、NMS=.70、match IoU=.50、推論尺寸768、15% ROI margin 不變。floor 候選已經過既有 NMS，不能拿來判斷 pre-NMS 是否曾有候選。

FP 跨模型 pairing：同 image、IoU≥.50，依 IoU 由高到低；同值依兩框較低 confidence 由高到低、較高 confidence 由高到低、C4 index、H4 index 由小到大。SHARED_FP 只是此幾何規則配對，不代表已確認相同組織。

每個成功 GT 的 bbox IoU/confidence 可追溯到保存的 pair。逐 GT mask IoU/Dice=NA：GT 原始 mask 是語義聯集，未保存逐 polygon 對應的權威 instance mask，不能將影像 union mask 分數充當 GT instance 分數。

Crop 直接核對保存 ROI 在原 GT union mask 中保留≥95%像素，不重建或修改 ROI。原規則是所有保留 polygon 的 union ROI，不是 primary lesion ROI，所以 PRIMARY_ROI_MISSES_OTHER_GT 標 NA；另提供 multi-GT 中 unmatched bbox 未完全包含於 ROI 的幾何旗標，不當作逐實例像素證明。

## 核心轉換

| BOTH_DETECTED | C4_ONLY | H4_ONLY | BOTH_MISSED |
| --- | --- | --- | --- |
| 195 | 8 | 8 | 30 |

總 TP 均203，但有 8 個 C4-only 與 8 個 H4-only，另有 30 個共同漏偵測。數量相等已逐身分核實。

| size_bin | support | BOTH_DETECTED | C4_ONLY | H4_ONLY | BOTH_MISSED |
| --- | --- | --- | --- | --- | --- |
| <0.10% | 27 | 8 | 2 | 1 | 16 |
| 0.10-<0.25% | 22 | 15 | 1 | 1 | 5 |
| 0.25-<0.50% | 38 | 33 | 0 | 2 | 3 |
| 0.50-<0.75% | 30 | 24 | 4 | 0 | 2 |
| 0.75-<1.00% | 20 | 16 | 1 | 2 | 1 |
| Medium 1-<5% | 90 | 85 | 0 | 2 | 3 |
| Large >=5% | 14 | 14 | 0 | 0 | 0 |


## 21 個問題的回答

1. C4_ONLY：8 個。
2. H4_ONLY：8 個。
3. 兩者相等，且 BOTH_DETECTED+任一 only=203；總身分=241，並非僅 aggregate 算術假設。
4. 所有 size-bin 四格 counts 見上表與 size_bin_transition_summary.csv。
5. Very-small：失去 3、得到 2，共同偵測 23、共同漏掉 21，淨 -1。
6. <0.10% 的全部 gained/lost 身分如下（沒有人工挑例）：

| sample_id | GT_instance_id | size_ratio | transition |
| --- | --- | --- | --- |
| fuseg__0712.png | 0 | 0.0007438662402344007 | H4_ONLY |
| fuseg__0867.png | 0 | 0.00032043445312499865 | C4_ONLY |
| fuseg__1009.png | 1 | 0.00064086921875 | C4_ONLY |

7. 0.25-<0.50%：gain 2、loss 0，所以淨 +2 TP。機制見 miss 診斷，不宣稱因果。

| sample_id | GT_instance_id | size_ratio | transition |
| --- | --- | --- | --- |
| fuseg__0964.png | 0 | 0.0035400387499999984 | H4_ONLY |
| fuseg__0971.png | 0 | 0.004463194667968799 | H4_ONLY |

8. 0.50-<0.75%：gain 0、loss 4，所以淨 -4 TP。機制見 miss 診斷，不宣稱因果。

| sample_id | GT_instance_id | size_ratio | transition |
| --- | --- | --- | --- |
| fuseg__0180.png | 2 | 0.005798339062499996 | C4_ONLY |
| fuseg__0603.png | 1 | 0.0065689091015624975 | C4_ONLY |
| fuseg__0781.png | 0 | 0.006008148974609404 | C4_ONLY |
| fuseg__0880.png | 0 | 0.005004883125000002 | C4_ONLY |

9. Single/multi GT 交換：

| group | BOTH_DETECTED | C4_ONLY | H4_ONLY | BOTH_MISSED |
| --- | --- | --- | --- | --- |
| single | 135 | 3 | 2 | 11 |
| multi | 60 | 5 | 6 | 19 |

10–11. 固定優先順序的雙向 miss 診斷：

| direction | MATCH_COMPETITION | BELOW_FROZEN_CONFIDENCE | RETAINED_LOCALIZATION_BELOW_IOU | NO_RETAINED_OVERLAP |
| --- | --- | --- | --- | --- |
| C4_ONLY | 0 | 1 | 6 | 1 |
| H4_ONLY | 0 | 4 | 1 | 3 |

C4→H4-R 的8次 loss 中，6次有保留框但定位 IoU 不足；例如 fuseg__0604.png 的 GT1 IoU=0.4999786039，嚴格低於0.50，不可四捨五入成 TP。反方向8次 gain 中，4次在 C4 保存了低於 operating threshold 的候選。
BELOW_FROZEN_CONFIDENCE 僅表示保存候選存在於 frozen operating threshold 以下，不表示降低門檻會改善整體表現。

12. FP 類型與淨差：

| category | C4 | H4 | delta |
| --- | --- | --- | --- |
| NEGATIVE_IMAGE_FP | 2 | 2 | 0 |
| DUPLICATE_OR_MATCH_COMPETITION_FP | 1 | 0 | -1 |
| LOCALIZATION_MISMATCH_FP | 18 | 19 | 1 |
| OTHER_REGION_FP | 9 | 14 | 5 |

OTHER_REGION_FP +5 是主要淨增來源；localization +1 與 duplicate/competition −1 抵銷。OTHER_REGION 指與凍結 GT bbox 的 IoU=0，不等於經醫護確認為正常皮膚，也不能排除標註未涵蓋的病灶。
13. H4_ONLY_FP=19。
14. C4_ONLY_FP=14；SHARED_FP=16。+5 是新增與消失抵銷後的淨差，不是只有五個新 FP。
15. Multi-GT FP 7→11（+4）；single 21→22（+1）；negative 2→2（0）。淨增加的4/5在 multi images，但僅為描述性關聯。實際 unshared FP 按分組見附表。
16. Negative-image FP 是否同一批：True。全部五張的數量與置信度：

| sample_id | C4_prediction_count | H4_prediction_count | C4_confidences | H4_confidences |
| --- | --- | --- | --- | --- |
| fuseg__0128.png | 0 | 0 | [] | [] |
| fuseg__0417.png | 1 | 1 | [0.8645570874214172] | [0.8964060544967651] |
| fuseg__0483.png | 0 | 0 | [] | [] |
| fuseg__0533.png | 1 | 1 | [0.6970624327659607] | [0.9257096648216248] |
| fuseg__0869.png | 0 | 0 | [] | [] |

17. Crop gain=6、loss=7；其餘見下表。

| group | BOTH_CROP_PASS | C4_ONLY_CROP_PASS | H4_ONLY_CROP_PASS | BOTH_CROP_FAIL |
| --- | --- | --- | --- | --- |
| all | 162 | 7 | 6 | 11 |
| single | 143 | 4 | 1 | 3 |
| multi | 19 | 3 | 5 | 8 |

18. single crop gain=1、loss=4，淨 -3；逐張 retained fraction、NO_ROI/不足95%與 bbox exclusion 見 crop_transition_matrix.csv。
19. multi crop gain=5、loss=3，淨 +2；逐張 retained fraction、NO_ROI/不足95%與 bbox exclusion 見 crop_transition_matrix.csv。
20. MIXED_PATTERN: DETECTION_REDISTRIBUTION + FP_INFLATION。總 TP 未增加，不支持 global sensitivity gain。
21. 唯一建議：FALSE_POSITIVE_CONTROL。建議下一階段僅預登錄「train-only、來源可追溯且已確認非傷口的背景負樣本監督」作單一因素，固定模型、尺度、loss、threshold及crop；不得把這191張validation的FP搬進training或拿未標註區域逕當負樣本。依據是 OTHER_REGION_FP 9→14，且multi-GT淨FP +4。這不保證能解決6次定位型loss或提升very-small TP，必須保留小傷口recall與crop安全門檻，未具備可信負樣本前不得啟動。

## 配對品質（H4-R − C4）

| metric | n | mean | median | q25 | q75 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| delta_bbox_iou | 195 | 0.0026132869742487744 | -0.0021003981205208744 | -0.022236562603841714 | 0.019301822116398304 | -0.1803061383730764 | 0.28142663988913497 |
| delta_confidence | 195 | -0.021963737713984955 | -0.007633626461029053 | -0.016115427017211914 | -0.0018222332000732422 | -0.6634614318609238 | 0.6260826289653778 |
| delta_mask_iou | 0 | NA | NA | NA | NA | NA | NA |
| delta_mask_dice | 0 | NA | NA | NA | NA | NA | NA |


mask NA 的原因見方法。image-union mask 品質另存 JSON，只能作影像層次描述。

## FP churn 置信度分布

| pairing | arm | n | mean | median | q25 | q75 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SHARED_FP | C4 | 16 | 0.7118055643513799 | 0.8560924530029297 | 0.632413700222969 | 0.8949932754039764 | 0.14990973472595215 | 0.9476008415222168 |
| SHARED_FP | H4 | 16 | 0.7580230608582497 | 0.8466977179050446 | 0.7528354078531265 | 0.9195316284894943 | 0.1482430100440979 | 0.9420601725578308 |
| C4_ONLY_FP | C4 | 14 | 0.325666107237339 | 0.2437693029642105 | 0.1651211827993393 | 0.4796522632241249 | 0.10412620007991791 | 0.9131607413291931 |
| C4_ONLY_FP | H4 | 0 | NA | NA | NA | NA | NA | NA |
| H4_ONLY_FP | C4 | 0 | NA | NA | NA | NA | NA | NA |
| H4_ONLY_FP | H4 | 19 | 0.4041722726665045 | 0.4151846766471863 | 0.21068371832370758 | 0.5031526833772659 | 0.10056985169649124 | 0.9155611991882324 |


空組 n=0，其統計值 NA，不補零。未做 threshold sweep 或提出 confidence threshold。

## FP churn 按影像組別

| group | SHARED_FP | C4_ONLY_FP | H4_ONLY_FP |
| --- | --- | --- | --- |
| single | 13 | 8 | 9 |
| multi | 1 | 6 | 10 |
| negative | 2 | 0 | 0 |


## 規則化視覺化

全部符合條件的 35 張唯一影像均保存，每張左右分別 C4/H4-R。index.json 列出納入理由及 SHA256。以下只展示 sample_id 排序的前3張，不以好壞選圖。

![fuseg__0003.png](C:/Users/milo9/Desktop/智慧型傷口分級與照護對應系統/experiments/results/f_higher_scale_v2_paired_error_audit/paired_error_visualizations/fuseg__0003.png)

![fuseg__0007.png](C:/Users/milo9/Desktop/智慧型傷口分級與照護對應系統/experiments/results/f_higher_scale_v2_paired_error_audit/paired_error_visualizations/fuseg__0007.png)

![fuseg__0046.png](C:/Users/milo9/Desktop/智慧型傷口分級與照護對應系統/experiments/results/f_higher_scale_v2_paired_error_audit/paired_error_visualizations/fuseg__0046.png)

## 交付與邊界

10 份 CSV、audit_summary.json、integrity.json 與全部規則化圖表位於 experiments/results/f_higher_scale_v2_paired_error_audit/。CSV 一列一個分析單位，NA 為無資料／不適用，原始 F1.2 不覆寫。

工程紀錄：首次新稽核介面漏做凍結規則中的相同RGB通道轉灰階，於輸出分析表之前停止。只修正F1.3介面並新增回歸測試；未更改GT或F1.2，原停止紀錄保留 initial_engineering_attempt.json。完整重新核對通過後才完成報告。

```ini
PHASE_F13_STATUS = COMPLETE
PAIRED_ERROR_ANALYSIS = COMPLETE
NEW_INFERENCE = false
MODEL_LOADED = false
TRAINING = false
THRESHOLD_CHANGED = false
NMS_CHANGED = false
MATCH_IOU_CHANGED = false
LOCKED_TEST_USED = false
CO2Wounds_USED = false
EXTERNAL_TEST_USED = false
test_images_used = 0
NEW_TRAINING_AUTHORIZED = NO
APP_MODEL_REPLACEMENT_AUTHORIZED = NO
MULTI_SEED_AUTHORIZED = NO
STOP_AFTER_F13 = true
PRIMARY_FAILURE_STRUCTURE = MIXED_PATTERN: DETECTION_REDISTRIBUTION + FP_INFLATION
RECOMMENDED_NEXT_SINGLE_INTERVENTION = FALSE_POSITIVE_CONTROL
```

STOP AFTER F1.3。建議不是執行授權；不訓練、不重跑、不 multi-seed、不換尺度、不調門檻、不改 crop、不 external test、不替換 App。
