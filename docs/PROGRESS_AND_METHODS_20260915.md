# 傷口照護專案：完整進度、實驗方法與證據

資料核對日期：2026-09-15。用途：進度報告、畢業專題與履歷查閱。

數字以本機設定、既有結果 JSON 與本次檔案計數交叉核對。[公開彙總證據](evidence/progress_20260915.json)包含來源報告的 SHA256（檔案指紋）與 22 個資料版本的數量；[匯出程式](../scripts/build_public_progress_evidence.py)說明如何產生。它只公開總數與指標，完整影像、逐圖清單與權重留在受控本機。

## 1. 目前進度

我已完成七類傷口分類、傷口定位／分割的多輪開發，以及護理師操作原型。分類回答「照片較像哪一類傷口」；定位回答「傷口在哪裡」；分割再描出傷口像素範圍。把定位後的小圖送入分類器，就是本專案的串接流程（cascade）。

目前最重要的發現是：分類、定位與串接要分別驗收。分割 mAP 很高，不能直接推論裁切完整，也不能推論後面的分類會更準。

| 工作 | 已有證據 | 目前判斷 |
|---|---|---|
| 七類分類 C-Arch-05 | 歷史 25 次開發評估平均 Accuracy 87.40%；一次性鎖定測試 46/48 正確 | 封版保存；來源完整性與統計彙總仍有下文列出的限制 |
| FUSeg D-Seg-03R | 771 train／191 val，300 輪完成；Mask mAP50 90.32% | 內部開發候選；固定工作點的裁切完整率未勝過舊模型 |
| ISIC→FUSeg 新候選 | ISIC 80 輪後早停，FUSeg 300 輪完成；Mask mAP50 90.20% | 訓練已完成；固定工作點驗收記錄為 FAIL，且本次讀碼發現色彩順序錯誤，需更正驗收後再判斷模型 |
| 串接分類 | 41 張開發影像：全圖分類 90.24%，先裁切再分類 58.54% | 有實際退步證據；38 張也參與分割選模，不能當獨立測試 |
| 外部 CO2Wounds-V2 | 607 張有標註影像；Mask mAP50 40.63% | 跨來源性能仍不足；結果已看過，不能反覆拿來選模型 |
| App 與 RAG | 病人切換、覆核、防誤建議、來源追溯、隔離啟動已加測試 | 工程功能可展示；新候選尚未自動替換 App 權重 |

本文的「完成訓練」只指既定訓練程序結束；「通過工程測試」只指程式在該測試情境運作正確。兩者都需要再與模型效能、使用情境分開解釋。

## 2. 資料來源

### 2.1 來源登錄與使用決策

表中的「已知來源數」與「本專案使用數」刻意分開：網站宣稱的總量，不等於我已下載、檢查並用於訓練的量。同一照片複製或產生多個版本，也不能當成多名病人。

| 來源與直接連結 | 已知來源／本機數量 | 本專案用途及目前處理 |
|---|---|---|
| 七類分類原始資料 `Wound_dataset`；舊流程稱 Yasin | 原始 431 張；準備後 768 個檔案 | 原始發布頁、下載版本與完整授權證據目前未找到，**不能提供已核實的原始來源連結**。保留歷史結果；未核實前不發布影像、不擴充新的同來源訓練 |
| [FUSeg 官方 UWM repository](https://github.com/uwm-bigdata/wound-segmentation)，[論文](https://arxiv.org/abs/2201.00414) | 開發 962 張＝771 train＋191 val；另有官方 test 200 張的本機入庫紀錄 | 本輪只用 962 張開發影像；200 張 test 使用 0。固定來源 commit `42a272dfe0679f20675e826385925cb7562934b6` |
| [FUSeg 競賽原始條款 PDF](https://github.com/uwm-bigdata/wound-segmentation/blob/42a272dfe0679f20675e826385925cb7562934b6/data/Foot%20Ulcer%20Segmentation%20Challenge/FootUlcerSegmentationChallenge2021.pdf) | 同上，不另加總 | 2026-09-14 證據記錄為 CC BY NC，該處沒有明示版本；目前用途限定非商業研究。詳見[模型卡](FUSEG_MODEL_CARD_20260914.md) |
| [AZH 原始資料與作者說明](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/wound_dataset)，[原論文](https://doi.org/10.1038/s41598-020-78799-w) | 官方 train 831 對，653 個 parent groups；排除 14 對後 817，再排除 1 個無法形成 polygon 的 3 像素標註，實際 816 | 加入歷史 D-Seg-04/05/06 訓練；固定 val 仍為 FUSeg 191 張。不得把 FUSeg 競賽條款自動套到 AZH 全部資料；新 runner 的 AZH 准入仍待證據 |
| [BUBT Lower Limb and Feet v2](https://data.mendeley.com/datasets/hsj38fwnvr/2) | 核實 healthy 2,757 張：female 前綴 776、male 前綴 1,981 | D-Seg-05 選 312 張空標註陰性圖；D-Seg-06 再加 108，累計 420；D-Seg-08 抽 82。來源 CC BY 4.0；未有病人 ID，所以只進 train |
| [WSNet/WOUNDSEG 官方資料](https://huggingface.co/datasets/subbareddyoota/wseg_dataset)，[作者程式](https://github.com/subbareddy248/WSNET) | 2,686 張；官方本機版本 1,894 train／412 val／380 test | 資料 revision `bdcc1e1cef607bfd7a4f98fc28f31a638a0e0ca9`，資料卡 CC BY-NC 4.0。已做歷史偵測／分割，後續未重新引入當新來源；BUBT 壓縮包中的 2,686 張 wound 也不能再次加總 |
| [ISIC 2017 Task 1 官方頁](https://challenge.isic-archive.com/data/)，[訓練影像 ZIP](https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Data.zip) | 2,000 影像／2,000 masks；本輪 1,800 train＋200 val | CC0。這是皮膚病灶資料，用於輔助預訓練；不能算作 2,000 張傷口，也不進傷口效能分母 |
| [CO2Wounds-V2 官方 v1 頁](https://data.mendeley.com/datasets/s2w7rjwz49/1) | 本地封存有標註 cohort 607 張；157 張官方無標註 test 未用；607 張形成 576 個內容群組 | 607 張已做一次跨資料來源評估；後續不作 train、選閾值或選權重。網站版本 v1 與本地資料標示 v0.0.2 是不同版本命名，不應混寫 |
| 場域／医院提供影片（無公開下載連結） | bbox 177 張：130 train／40 val／7 locked；polygon 候選 170 張：130／40，共 220 polygons | 原始影像隔離保存；命名可以識別抽幀家族，不能證明 consent、病人隔離或專業標註已完成。使用者提供用途聲明仍須附原始授權與去識別化紀錄 |
| [Kaggle 舊彙整包](https://www.kaggle.com/datasets/leoscode/wound-segmentation-images) | 歷史 SHA256 配對：FUSC 1,186、Medetec 374、WSNet 1,176，共 2,736 列 | 這是彙整包內資料列，並非三份新獨立資料集。下載版號欠缺；依先前決策退出新訓練，改走官方來源 |
| [Medetec 原始網站／使用說明](https://medetec.co.uk/files/medetec-images.html) | 舊彙整包內 374 列；不是網站全量 | 學術用途條件及衍生標註使用仍須原始條款確認；本輪沒有使用 |
| Roboflow 舊匯出 | 先前口述約 180 張，沒有核實的 project URL、version、license | 正確張數與上游 URL 都列為缺件，不將約數改寫成確定數字；本輪使用 0 |

原始 431 張分類資料的來源缺件是本專案目前明確問題之一。即使找得到名稱近似的網頁，也不能在沒有逐圖對應與下載紀錄時把它寫成已核實來源。

### 2.2 候選／被排除來源也要交代

| 候選來源連結 | 網站或既有稽核記錄量 | 已納入本輪訓練 | 處理原因 |
|---|---:|---:|---|
| [Redscar](https://redscar.uib.es/dataset.html) | 394 張 | 0 | 個別存取核准未有完整證據；D-Seg-10 停在來源關卡 |
| [WoundsDB](https://chronicwounddatabase.eu/)，[條款](https://chronicwounddatabase.eu/Terms) | 歷史候選清單列 188 wound records、79 visits、47 patients；不是 188 個獨立人 | 0 | 先前遇憑證錯誤及註冊限制；本次未重新驗證可下載性 |
| [WoundTissue](https://github.com/akabircs/WoundTissue) | 宣稱 147 張，本地 checkout 只核實 13 對 | 0 | 13 對不能寫成取得完整 147 張 |
| [DFUTissueSegNet](https://github.com/uwm-bigdata/DFUTissueSegNet) | 宣稱標註 110 張 | 0 | 資料授權待核實，且同 AZH 來源，不能當獨立外部證據 |
| [SurgWound](https://huggingface.co/datasets/xuxuxuxuxu/SurgWound) | 候選清單列 686 張 | 0 | 沒有傷口分割 masks，來源收集方式也須核實 |
| [Pressure-ulcer Figshare](https://figshare.com/articles/dataset/images_of_pressure_ulcer_2_/17206940) | 20 張 | 0 | 規模太小；最多當另行規劃的定性案例 |
| DFUC2022 | 本輪 0；完整量未核實 | 0 | 既有授權审查未通過，詳見[候選稽核](DATASET_CANDIDATE_AUDIT_20260821.md)；不補寫未核實下載連結 |
| LUTSeg | 本輪 0；完整量未核實 | 0 | 既有來源審查指向 CO2Wounds 血統，會重複使用已看過的外部來源 |

這份表不是全部來源都已獲授權的宣告。網頁可瀏覽、程式開源、論文可以下載，都不能替代影像使用條款；每份來源要分別核對。歷史 `source_admission.json` 是新 runner 的預設准入表，FUSeg 本輪另有[專用原始條款證據](../experiments/review_v2/evidence/FUSeg_noncommercial_research_20260914.md)，兩者不能混當全專案授權狀態。

### 2.3 實際訓練資料版本與張數

以下 train／val 為 2026-09-15 檔案計數；test 為目錄計數或既有清單，不是本輪重新讀圖評分。空白寫「無此目錄」，不推論原始來源沒有 test。

| 版本／階段 | Train 檔案數 | Validation 檔案數 | Test／額外資料 | 為什麼這樣組成 |
|---|---:|---:|---|---|
| 分類 v3 | 679 | 41 | locked 48 | 先切原圖，再只補齊 train |
| WSNet official detection | 1,894 | 412 | locked 380 | 官方取得後的早期版本 |
| WSNet unionbox v1／seg v1 | 1,845 | 461 | train/val 版無 test 目錄；另存 380 test-GT | 重組開發資料；不得與 1,894/412 混為同版 |
| WSNet seg aug v3、v4 | 各 7,380 | 各 461 | 無此 test 目錄 | 1,845 原圖＋每圖 3 種變體；兩版本非新獨立來源 |
| D-Seg-02 public combined | 2,394 | 595 | 本輪使用 0 | 歷史混合來源開發版；不能與後來固定 191 張 val 直接排名 |
| D-Seg-03／03R FUSeg | 771 | 191 | 官方 200 未用 | 固定來源作比較基準 |
| D-Seg-04 FUSeg＋AZH | 1,587 | 191 | 使用 0 | 771＋816，先增加正樣本来源 |
| D-Seg-05 ＋BUBT | 1,899 | 191 | 使用 0 | 1,587＋312，控制陰性圖數量 |
| D-Seg-06 small/multi | 2,439 | 191 | 使用 0 | 1,899＋432 重点複本＋108 新陰性圖 |
| D-Seg-07 polygon v1 | 329 | 38 | 使用 0 | 歷史版；後续 v2 重作隔離清理 |
| D-Seg-07 polygon v2 | 326 | 38 | 383 groups 中保留 364，QC 排除 19 | 零 exact/pHash 跨切分；影像來源授權仍待補 |
| D-Seg-08 replay | 734 | 38 | 另用 FUSeg 191 張保留驗證 | 326 manual＋326 replay＋82 healthy |
| 臨床 bbox／polygon | 各 130 | 各 40 | bbox locked 7；polygon 未納入 test | 同場域抽幀版，專業複核與治理仍待補 |
| 新 ISIC 預訓練 | 1,800 | 200 | 使用 0 | 只在官方 training 2,000 張內分配 |
| 新 FUSeg 微調 | 771 | 191 | 使用 0 | 與 FUSeg 基準保持相同傷口開發 cohort |

另存的 WSNet aug v1 只有 1 張 train、v2 只有 55 張；FUSeg＋AZH `failed_build` 有 1,405 train／191 val。它們是未完整建置／失敗中間產物，不計入正式完成資料量。完整逐版本數量可查[機器可讀清單](evidence/progress_20260915.json)。不能把各版本相加，說成有數萬张不同傷口照片。

## 3. 分類資料平衡、分割

### 3.1 先分原圖，再只複製訓練資料

我使用[prepare_cls_dataset.py](../prepare_cls_dataset.py)逐類洗牌，以 seed=42 固定亂數，取 `int(n×0.8)` 為 train、`int(n×0.1)` 為 val、剩餘為 test。切分結果不是每類剛好 80/10/10，因為圖片張數必須是整數。seed 固定仍受 `os.listdir` 輸入次序影響；正式重現應依既有 split manifest，而非在另一台電腦重跑洗牌就假設完全相同。

| 類別 | 原始 | 初始 train | 補齊後 train | val | locked test | dev 唯一內容（初始 train＋val） |
|---|---:|---:|---:|---:|---:|---:|
| Abrasions 擦傷 | 85 | 68 | 97 | 8 | 9 | 76 |
| Bruises 瘀傷 | 122 | 97 | 97 | 12 | 13 | 109 |
| Burns 燒傷 | 59 | 47 | 97 | 5 | 7 | 52 |
| Cut 切傷 | 50 | 40 | 97 | 5 | 5 | 45 |
| Ingrown_nails 嵌甲 | 31 | 24 | 97 | 3 | 4 | 27 |
| Laceration 撕裂傷 | 61 | 48 | 97 | 6 | 7 | 54 |
| Stab_wound 刺傷 | 23 | 18 | 97 | 2 | 3 | 20 |
| 合計 | 431 | 342 | 679 | 41 | 48 | 383 |

最大類別的初始 train 有 97 張，所以其餘 train 補到 97；程式是循序複製影像，不是旋轉、生成新照片或 loss class weight。目的是增加小類別被訓練讀到的頻率。例如 18 張刺傷反覆出現，仍只有 18 張原始刺傷訓練圖。部分小類別內個別影像會多複製一次，類別平衡不等於類別內每張權重完全相等。

最後 679＋41＋48＝768 個檔案。把 train 和 val 合為開發池得到 720 個檔案；48 張 test 永遠不進此開發池。這樣做的依據是訓練用於學參數、驗證用於選設定、測試用於最終估計；若測試也拿來選設定，就造成資料洩漏。[scikit-learn 官方資料洩漏說明](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

### 3.2 MD5 

MD5 是把檔案位元組算成固定摘要的「內容指紋」。改檔名不會改指紋；完全一樣的照片複本可因此放在同組。但重新壓縮、旋轉或改像素會改摘要，所以零 MD5 重複不代表沒有近似影像或同病人的連拍。MD5 也不適合抵抗惡意碰撞的安全用途；本專案後續用 SHA256 紀錄檔案完整性。[Python hashlib 官方說明](https://docs.python.org/3/library/hashlib.html)

720 個開發檔案形成 383 組：177 組只有 1 張；206 組合計有 543 個檔案。若每組只算 1 個代表，額外複本就是 543−206＝337；383＋337＝720。不能說有 720 個獨立觀察。

早期一般分折將同一張照片的複本分到 train 與 val，稽核發現 178/206 重複群組跨折，所以原 Accuracy 98.34% 已失效。重新分折時，同一 MD5 組全部一起移動，得到歷史 87.40%。10.94 個百分點是兩套協定的觀察差距，不是嚴格控制其他條件後的唯一洩漏因果效應。

完整隔離應有四層：檔案摘要找完全複本；pHash 找外觀近似候選；原圖→裁切／增強／影片的家族關係；病人／影片來源分組。pHash 距離≤4 是本專案選定的檢查門檻，不是醫學界保證獨立的標準。缺病人 ID 時，要明確說病人隔離未證明。[ImageHash 實作來源](https://github.com/JohannesBuchner/imagehash)

### 3.3 5×5 

5 folds 是把 383 組分成 5 份，每次 4 份學習、1 份驗證，輪流做 5 次。再使用 42、123、3407、2026、999 五個 seed 重做流程，總共 25 次。seed 影響亂數過程，不會把 383 組變成新的 383 組。[StratifiedGroupKFold 官方 API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)說明同組不可跨折，並盡量維持類別比例。

例如 seed=42 的已存報告，fold 1–5 的 train/val 分別為 570/150、587/133、575/145、573/147、575/145。val 總和是 720；份數不必完全等大，因為一個複本群組不能拆開。此 CV 是在**已平衡的 720 instances** 上分組，所以驗證也會有整組複本；它估計的是這個平衡後評估分布，不能直接當真實就診比例。

25 次不是 25 個完全獨立資料集；不同折共享訓練資料，不同 seed 又使用同一開發池。跨 seed 變動小，可以说明該協定較穩定，但不能取代更多病人與來源。[PyTorch 可重現性說明](https://docs.pytorch.org/docs/stable/notes/randomness.html)

### 3.4 48 張test

48 張來自同一原始來源的保留份，分布是 9、13、7、5、4、7、3。全資料去重稽核可將它們與 383 組一起比對，理想結果是 431 個唯一內容且 dev/test 不重疊；若加 48 張仍只有 383 組，反而意味 test 全是既有內容，需要追查，不是正常目標。

已存一次性結果：Accuracy 46/48＝95.83%、Top-5 100%、Macro-F1 94.76%、Weighted-F1 96.23%、Macro ROC-AUC 0.9982。Bruises 有 1 張、Laceration 有 1 張誤判為 Stab_wound；刺傷 Recall 3/3＝100%，Precision 卻為 3/5＝60%。只有 3 張的類別，錯 1 張就降 33.33 個百分點，因此不能只秀 100% Recall。[封版逐類結果表](../experiments/results/tables/Table5_Final_Blind_Test_Generalization.md)

### 3.5 新發現的統計紀錄限制

封版彙總記錄 Accuracy 87.40%±2.78%、Macro-F1 87.36%±3.11%；±代表 run 間離散程度，不是每張都落在這個範圍。既有 Bootstrap 報告寫 Accuracy 95% CI 86.32–88.48%、B=2,000，但目前[統計程式](../experiments/scripts/statistical_analysis.py)把各 seed/fold 預測合併後按**預測列**抽樣，沒有按 MD5 group 重抽樣，會忽略重複影像和跨 seed 相依性。

此外報告總預測列為 3,622，理論上完整五 seed 的每張一次折外預測應為 720×5＝3,600，差 22 列尚未核銷。本輪保留原始檔，將這組 CI 定位為「待重新核對的歷史統計」，不宣稱已達嚴格群組推論。下一步先核對每 seed/fold 的影像 ID 與摘要，再另出修訂版群組抽樣報告；不需要重訓模型，也不讀 test。[SciPy Bootstrap 文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)提供方法定義，但資料相依性仍須由本專案決定抽樣單位。

## 4. Phase 1–9 實際完成與應如何解讀

| 階段 | 目的 | 證據／完成邊界 |
|---|---|---|
| Phase 1 Logger/Schema | 用一致欄位記錄模型、seed、資料角色、結果路徑，避免不同實驗混在同一數字 | [logger](../experiments/scripts/experiment_logger.py)；既有 experiment_log 本機封存 |
| Phase 2 YOLO Trainer | 把分類超參數放入設定檔；以 train 更新權重、val 選 checkpoint | [C-Arch-05 config](../experiments/configs/C-Arch-05_yolov8n_cls.yaml)；imgsz=224、batch=16、epochs 上限150、optimizer=auto、lr0=0.01 |
| Phase 3 PyTorch Trainer | 比較不同架構，不把「有設定檔」誤當「已訓練完成」 | 歷史 MobileNetV3-Large 固定 val 92.68%；ResNet50、EfficientNet、ViT 等設定檔不代表全部有正式結果；見[歷史整合報告](PROFESSOR_INTEGRATED_EXPERIMENT_REPORT_20260817.md) |
| Phase 4 Evaluation | 輸出 Accuracy、F1、每類 Recall 與混淆方向，知道哪類錯 | [指標定義](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)；不能只看整體命中率 |
| Phase 5 Orchestrator | 統一調度與隔離檢查，避免用錯 split 或重開 test | [run_experiments.py](../experiments/run_experiments.py) |
| Phase 6 naive folds | 首次五折，但跨折有複本，98.34% 失效 | [洩漏影響表](../experiments/results/tables/Table4_Data_Leakage_Impact_Analysis.md) |
| Phase 6.5 audit | 找出178個跨折重複群組，保留失效證據 | 重複≠新照片；不能只移除報告中的不佳結果 |
| Phase 6.6 grouped folds | MD5整組分配，seed42平均87.92% | [公開證據 JSON](evidence/progress_20260915.json)保留五折張數與0 overlap |
| Phase 7 multi-seed | 五seed×五fold，描述穩定性 | 平均87.40%；seed平均值SD0.76個百分點；共享資料的25 runs |
| Phase 8 statistics | 既有B=2,000 CI | 完成歷史產出，但本次發現分母與抽樣單位待修訂，不能簡寫全部PASS |
| Phase 9 paper tables | 自動匯出資料、結果、CI及洩漏對照表 | [tables](../experiments/results/tables/)；本次報告補上限制，保留封版原始檔 |

`optimizer=auto` 時，指定 lr0 並不等於 optimizer 最終真的採用該值；實際值應查各 run 的 args／log。這是[Ultralytics 訓練參數](https://docs.ultralytics.com/modes/train/)所描述的自動選擇機制。

## 5. 定位／分割怎麼訓練，為什麼換方法

### 5.1 從框位置轉成描輪廓

早期偵測器學一個矩形框 bbox；公開來源有 masks 時，可以從前景像素轉成 polygon，再訓練 YOLO11m-seg。單類 `Wound` 只需判斷傷口範圍；裁切則取所有預測傷口的聯集外接框，四側各多留 15%，減少把邊緣切掉。15% 是本專案工作設定，尚未證明全場景最佳。[Ultralytics segmentation 格式](https://docs.ultralytics.com/datasets/segment/)提供 polygon 座標定義。

有框、有 mask，都不會自動產生七類診斷標籤。若公開資料只標「Wound」，只能測定位／分割。歷史41張七類開發資料可以測串接分類，但沒有全部獨立專業 mask，來源與選模重疊也要揭露。

### 5.2 歷史循序實驗

以下皆為保存的同一 checkpoint 最終驗證指標；不同 validation 欄的數字不做直接高低排名。mAP 是跨信心門檻的偵測排序評分；mAP50 使用 IoU≥0.50，mAP50–95 用0.50到0.95多個重疊門檻平均，不是分類正確率。[Ultralytics 驗證文件](https://docs.ultralytics.com/modes/val/)

| 階段 | 學習資料／目的 | 驗證量 | Mask mAP50 | Mask mAP50–95 |
|---|---|---:|---:|---:|
| D-Seg-02 | 混合公開來源2,394，來源基準 | 595 | 55.76% | 29.79% |
| D-Seg-03 | FUSeg771，建立單來源基準 | 191 | 90.07% | 67.22% |
| D-Seg-04 | ＋AZH816，擴充正樣本 | 191 | 93.54% | 73.00% |
| D-Seg-05 | ＋BUBT312，加入背景控制 | 191 | 93.00% | 74.46% |
| D-Seg-06 | 重讀小傷口／多傷口432份，再加108陰性 | 191 | 94.15% | 76.89% |
| D-Seg-07 | 326張自補polygon，來源權利待補 | 38 | 65.11% | 31.55% |
| D-Seg-08 | 734張replay，避免只學新資料而忘掉舊資料 | 38主驗證／191保留驗證 | 59.98%／88.10% | 28.81%／66.00% |
| D-Seg-03R | COCO重新初始化，warmup修正版 | 191 | 90.32% | 68.08% |
| ISIC→FUSeg | 先皮膚病灶，再傷口微調 | FUSeg191 | 90.20% | 67.99% |

D-Seg-03→04→05→06 是接續權重的學習路徑，不是各輪從相同初始點獨立訓練。D-Seg-03R 改成 COCO 通用預訓練起點，避免接續其他傷口来源，另將 warmup_bias_lr 由0.1改為0.0。[FUSeg模型卡](FUSEG_MODEL_CARD_20260914.md)

FUSeg＋AZH＋BUBT 的平衡也不同於七類分類：D-Seg-05 有1,519張正樣本、原有68張空標註，再加312張healthy，陰性總數380，是正樣本的25.02%。healthy選取seed42、female/male**檔名前綴**各156、每個保守pHash群組取1張；不能把檔名前綴平衡當人口統計公平性證據。陰性比例上限用來避免大量背景讓模型全判空白，是本專案假設，仍需消融比較。[歷史建置紀錄](SEGMENTATION_EXPANSION_EXECUTION_20260821.md)

### 5.3 最新 ISIC→FUSeg 完整設定

ISIC在官方training的2,000張內按既有排序取前1,800張train、後200張val；不是病人分组或分層抽樣，因此只把這200張用來觀察輔助預訓練。之後清空optimizer狀態，用ISIC最佳checkpoint初始化FUSeg。遷移學習的直覺是先學影像形狀再適應目標資料，但皮膚病灶與傷口不同，是否有益必须實驗驗證。[PyTorch遷移學習教學](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

| 設定 | 固定值與意思 |
|---|---|
| 架構 | YOLO11m-seg，ISIC輔助類別切換為FUSeg Wound |
| ISIC上限／早停 | 最多100 epochs；20輪未改善則停；實際80輪，最佳第60輪 |
| FUSeg上限／早停 | 最多300；patience80；實際300輪完成 |
| imgsz／batch | 768／4；影像縮放到模型輸入，每批4張更新 |
| optimizer／lr0／lrf | AdamW／0.0005／0.01；控制參數更新與最後學習率比例 |
| warmup | 5輪；warmup_bias_lr=0；讓起初更新逐步進入正式設定 |
| seed／deterministic | 42／true；降低可控亂數差異，仍須保存硬體與版本 |
| 旋轉／平移／縮放 | degrees5、translate0.05、scale0.2 |
| shear／perspective | 0.5／0.0002 |
| 左右／上下翻轉 | fliplr0.5／flipud0 |
| hsv_h／s／v | 0.01／0.4／0.25；改變顏色亮度，標籤隨幾何變換同步 |
| mosaic／mixup／copy_paste | 0.1／0／0；最後30輪關mosaic |
| AMP／workers | true／2；混合精度降低記憶體負擔，2資料載入workers |

以上是本專案記錄值，不是文獻證明最佳值；參數語義依[Ultralytics training](https://docs.ultralytics.com/modes/train/)。資料增強只在train發生，val保持固定。每輪train產生梯度更新；val只推論、量測、決定最佳checkpoint和是否早停。

2026-09-15 02:35曾因CSV epoch本來由1開始、程式又加1而停止銜接；修正連續性檢查後保留舊failure/lock，從已完成ISIC接續FUSeg，沒有重訓ISIC。[復原紀錄](ISIC_FUSEG_RECOVERY_20260915.md)與[runner](../experiments/review_v2/isic_fuseg_formal.py)有實作。

## 6. 驗證真的找對、裁對

### 6.1 把判斷規則寫在看結果前

工作點固定confidence0.10（高於此信心才採用）、NMS IoU0.70（去掉太重疊預測）、GT matching IoU0.50（預測與正確標註重疊至少一半才算配對）。這三個數字用途不同。IoU＝交集面積÷聯集面積；每個正確物件與預測最多配對一次。

TP是配對正確、FP是多報的框、FN是漏掉的標註；Precision＝TP/(TP+FP)，Recall＝TP/(TP+FN)，F1綜合兩者。沒有可窮舉的「不是傷口的所有可能框」，因此本專案物件配對矩陣不虛構TN。[COCO官方評估程式](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)

裁切另外計算「保留原標註傷口像素的比例」。保留≥95%才算完整；沒有ROI也算失敗；分母包含全部186張正樣本。只有算「有輸出框的照片」會把漏偵測藏掉。191張val含186正樣本、5陰性、241個polygon衍生GT框，三種分母不能混寫。

### 6.2 新舊模型同工作點比較

| 指標 | 舊D-Seg-03 | 新D-Seg-03R |
|---|---:|---:|
| TP／FP／FN | 204／30／37 | 200／27／41 |
| Precision／Recall／F1 | 87.18／84.65／85.89% | 88.11／82.99／85.47% |
| ≥95%裁切完整率 | 168/186＝90.32% | 164/186＝88.17% |
| 正圖未產生ROI | 3/186 | 7/186 |
| 負圖有預測 | 3/5 | 3/5 |

新模型41個漏偵測有36個是小傷口（GT框低於原圖1%）；小傷口命中101/137，Recall73.72%。因此接下來優先研究小目標，而非直接再加300輪。只有5張陰性，仍不足估算日常拍攝的真實誤報率。[完整定位決策](LOCALIZATION_EXPERIMENT_DECISION_20260914.md)

### 6.3 做了哪些改善，結果如何

1024全圖推論小傷口Recall只增加0.73個百分點、F1下降1.24；四切片推論小傷口Recall增加1.46，但FP27→84、F1 85.47→76.89%、平均GPU定位耗時約47.83→146.79ms。因未過預設門檻，沒有升為App預設，也沒有接續切片重訓。[切片結果](TILED_INFERENCE_EXPERIMENT_20260914.md)

切片試驗的參考概念是將小目標放大再合併結果，[SAHI原論文](https://arxiv.org/abs/2202.06934)是方法依據；本專案實作固定4個重疊視窗，不等於完整重現該論文或已使用其全部合併方法。512方圖的切片384×384，起點(0,0)/(128,0)/(0,128)/(128,128)，相邻重疊256像素；反覆看到同一傷口容易多報框。

### 6.4 最新低分出現後，核對到了什麼

ISIC→FUSeg固定驗收存檔：TP31／FP37／FN210，Precision45.59%、Recall12.86%、F1 20.06%、完整裁切27/186＝14.52%，GPU39.72ms，狀態`FAIL_DEVELOPMENT_GATE`。五種子尚未開始。

本次讀碼發現[isic_fuseg_gate.py](../experiments/review_v2/isic_fuseg_gate.py)先`Image.open(...).convert("RGB")`再轉NumPy，直接呼叫`predict_materialized`；而[基準評估器](../experiments/review_v2/localization_benchmark.py)會先`cv2.cvtColor(..., COLOR_RGB2BGR)`。Ultralytics對PIL輸入要求RGB，對NumPy輸入要求BGR；因此目前新候選與基準的輸入不一致。[官方輸入格式表](https://docs.ultralytics.com/modes/predict/#inference-sources)

這是已找到的實作缺陷，**尚未量測它能解釋多少低分**。目前不能將12.86%當成公平比较下的最終模型能力，也不能宣稱修正後必然通過。本輪是報告／發布整理，沒有修改此實驗程式或重跑模型；下一步先補色彩通道回歸測試，再保留原FAIL檔、另產生修正版固定val驗收。權重、資料、confidence、NMS、IoU與裁切邊距全部固定，才能隔離程式修正的影響。

## 7. 串接可能拖累分類

41張開發影像中，全圖分類37/41＝90.24%；先裁切再分類24/41＝58.54%，其中定位37/41有ROI，4張漏定位仍算錯。不能把有ROI的37張當成新的全部樣本，隱藏4張失敗。加入分類信心0.60門檻後，只回答28張，其中20張答對：回答率68.29%、已回答正確率71.43%、全體正確率48.78%。

分類器原本看整張照片，裁小後形狀、周圍皮膚與影像比例改變，可能造成輸入分布改變；漏定位則直接讓後端無法分類。這是需用同圖配對試驗評估的設計，不能將定位Recall與分類Accuracy簡單相乘就宣稱已測得端到端正確率。

分類端使用每張圖在各seed中未參與訓練的fold權重，將5份機率平均（OOF，折外預測）。但41張中38張與分割器的primary validation重疊，故仍是選模後開發診斷；不能說整條系統對這些圖都從未見過。公開彙總見[evidence](evidence/progress_20260915.json)。

下一步要取得有來源與權利證據的「同張圖具七類標籤＋定位標註」資料，預先固定全圖、裁切、雙視圖等比較規則。在此之前保留人工覆核與完整率、回答率說明。

## 8. 外部測試

2026-08-31已在CO2Wounds-V2有標註607張做一次跨資料集評估：Mask mAP50 40.63%、mAP50–95 16.87%；固定Mask IoU0.5配對TP434、FP328、FN537，Precision56.96%、Recall44.70%、F1 50.09%。此歷史外部測試的confidence=0.25、NMS IoU=0.40，與第6節後來的定位驗收規則不同，不能混作同工作點比較。資料與完整開發血統2,562個唯一內容的SHA256及pHash≤4交集皆0；仍未有病人層級ID證據。

607張中有23個完全相同內容群組，但各組官方mask不完全一致；pHash連通群組共576。推論前固定每組代表形成576張敏感度分析，mAP50 40.91%；這個结果不能取代完整607張主結果。原報告按576群組做B=2,000 Bootstrap，固定工作點F1 CI46.68–53.31%。這比把同組照片當獨立列更符合現有內容相依性，但仍不是病人群組信賴區間。

我因此發現跨來源適應與標註一致性是主要未解問題。607張已被看過，今後只保留為歷史測試，不能拿其分數來決定D-Seg-10。Redscar若日後將train用於開發，其官方test只能稱「同來源保留測試」，不能因資料夾名test就稱完全獨立外部來源。[CO2Wounds原始來源](https://data.mendeley.com/datasets/s2w7rjwz49/1)

## 9. App實際改動與驗證

| 問題 | 本次處理 | 驗證方式與尚未完成部分 |
|---|---|---|
| 切換病人後，舊影像／晚到回應可能殘留 | 清除舊結果、忽略過期請求，時間軸按patient_id過濾 | 合成API／前端操作驗收；不代表所有多分頁故障已涵蓋 |
| 儲存EMR回傳ID可能拿到report ID | 保留正確插入cursor ID | API回歸測試 |
| 沒勾人工覆核也可能試圖送API | 前端確認＋後端強制驗證 | 跳過確認遭拒絕的合成測試 |
| RAG紀錄缺追溯／覆核 | 加入來源EMR、覆核者、時間、去識別化確認、停用與重新覆核 | 未覆核／停用記錄排除檢索；舊資料庫實際遷移仍待切換驗收 |
| 合成雜訊被七類模型以99.32%判Bruises | 沒可靠定位／低信心／fallback時擋自動類別處置與RAG | 是合成反例，不能當臨床FP率；模型尚無健康／非傷口拒判能力 |
| health=ok誤解為模型已載入 | 分開顯示服務與模型就緒 | 合成環境確認「模型未就緒」 |
| 預覽可能開到真實DB或共用key | 獨立DB/key路徑及同檔檢查，production拒本機key fallback | 暫存DB測試，不自動改既有healthcare.db |

目前RAG是經覆核文字加關鍵詞檢索，不是線上微調LLM；「護理長建議累積」是知識庫更新，不是模型參數自動持續學習。[RAG原始論文](https://arxiv.org/abs/2005.11401)是檢索輔助概念依據，本專案尚非完整實作其中稠密向量檢索系統。

2026-09-15重新執行[verify_project_delivery.py](../scripts/verify_project_delivery.py)：82項unittest通過，另有合成推論、前端lint、前端build皆exit0；權重摘要及封版metadata一致，結果`PASS_ENGINEERING_ONLY`。證據已摘錄到[公開JSON](evidence/progress_20260915.json)。此結果未涵蓋手機推論、Docker全新建置、TLS、多人併發或專業標註效能。

## 10. 問題清單、解法與完成標準

本次 GitHub 發布範圍、即時掃描及權限檢查另見[公開提交檢查](PUBLICATION_CHECK_20260915.md)；公開內容不含資料庫、密鑰或資料集影像。

| 優先序／問題 | 現在處理進度 | 怎麼解決 | 何時完成 |
|---|---|---|---|
| 1：新驗收RGB/BGR不一致 | 已定位程式差異，保留FAIL產物 | 先補帶顏色合成圖的輸入契約測試，再只重算固定191張開發驗收 | 新舊模型同輸入規則、完整分母與checkpoint身份一致，另產修訂報告 |
| 2：分類統計3,622 vs3,600 | 已標示22列差額及逐列Bootstrap限制 | 核對OOF逐圖清單，另出群組／seed相依性適當處理的CI | 每seed覆蓋正確、無漏列重列、抽樣單位明確 |
| 3：431張分類／Yasin來源欠件 | 精確張數已保留；無可核實原始URL | 找原發布頁、版本、原始授權與逐圖對應；若無法補齊則只保留歷史研究證據 | 原始證據可追溯，而不是用同名網站替代 |
| 4：小傷口漏偵測 | 已量化36/41 FN是small，切片及高解析已試 | 先完成評估修復，再研究訓練取樣、合併去重、陰性控制，一次改一項 | 同固定val的Recall／FP／完整裁切及速度共同達標 |
| 5：來源轉換後性能低 | CO2Wounds607已封存；Redscar未准入 | 取得核准新資料，先做重複／血統／病人資訊稽核 | 有獨立於選模的新來源評估，不能重用607張選模 |
| 6：crop使分類退步 | 41張配對測試已有明確退步 | 合法同圖標註資料；預先規劃全圖／crop／融合比較 | 報全體正確率、回答率、拒答、各類錯誤與選模重疊 |
| 7：場域治理與標註 | 資料隔離、命名家族可查；原始文件／複核待補 | 取得文件與去識別化紀錄、專業標註複核 | 原始證據和標註版本一致；不以命名推定授權 |
| 8：GitHub舊秘密歷史 | 9/6本機換key、重加密27欄位、重置公開分支；Support #4732924已送出 | 等平台清理後，以舊SHA不可讀作驗證；部署端key切換亦需核對 | 工單送出不等於伺服器已清除；本輪不宣稱全平台purge完成 |

依序完成單一seed的有效驗收後，再考慮五seed穩定性與模型間比較。D-Seg-10計畫中的SegFormer-B2、DeepLabV3+ ResNet50仍是候選，不能當已完成訓練。時間估計須等修正版驗收與新資料准入結果，不能直接用「再跑25次」保證準確率提升。

## 11. 每一步究竟依據什麼

以下外部文獻／官方文件提供方法定義；具體資料切分、閾值與效果以本專案清單／程式／結果為準。**本次補上的方法文獻是回溯對照，不冒稱所有文獻都曾在原始實驗開始前閱讀或據此預註冊。**

| 步驟 | 外部依據 | 我在本專案的落實與限制 |
|---|---|---|
| 資料隔離 | [scikit-learn data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) | test不調參；所有資料版本要保留來源與角色 |
| 群組切分 | [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) | 同MD5家族不能跨fold；仍缺病人群組 |
| 完全相同檔案稽核 | [hashlib](https://docs.python.org/3/library/hashlib.html) | MD5歷史去重，SHA256後續凍結內容 |
| 近似影像稽核 | [ImageHash作者實作](https://github.com/JohannesBuchner/imagehash) | pHash≤4為專案工作門檻，不保證所有相似都被找到 |
| 初始權重與seg格式 | [YOLO11](https://docs.ultralytics.com/models/yolo11/)、[seg datasets](https://docs.ultralytics.com/datasets/segment/) | 單類Wound及polygon來源要能對回原mask |
| 訓練、增強、early stopping | [Ultralytics train](https://docs.ultralytics.com/modes/train/) | 凍結protocol、train更新、val選best、test使用0 |
| seed與可重現性 | [PyTorch randomness](https://docs.pytorch.org/docs/stable/notes/randomness.html) | 固定seed、版本和清單；不誇大跨硬體完全一致 |
| 預訓練／微調 | [PyTorch transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) | ISIC只輔助，FUSeg才作傷口驗收；受益由實測決定 |
| bbox／mask AP | [COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)、[Ultralytics val](https://docs.ultralytics.com/modes/val/) | box IoU與mask IoU不可混用，mAP不可稱分類Accuracy |
| 資料輸入顏色 | [Ultralytics predict格式表](https://docs.ultralytics.com/modes/predict/#inference-sources) | NumPy=BGR、PIL=RGB，這是本次程式核對依據 |
| 統計不確定性 | [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) | 本專案需按內容群組處理相依性；歷史分類CI待修訂 |
| 小目標切片 | [SAHI論文](https://arxiv.org/abs/2202.06934) | 固定四切片概念試驗未過門檻；不等於重現整篇論文 |
| 檢索知識 | [RAG論文](https://arxiv.org/abs/2005.11401) | 目前關鍵詞檢索與人工覆核，未做LLM持續微調 |
| 公開歷史安全 | [GitHub敏感資料清除](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository) | 換key、清分支、再掃描、Support平台purge需分開驗證 |
