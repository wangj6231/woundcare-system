# 博士論文／專案教授報告：分類、PyTorch 與物件偵測整合進度

**報告日期：** 2026-08-21（資料截止：2026-08-20）
**專案：** 智慧型傷口分級與照護對應系統
**用途：** 依實驗流程直接說明每一批資料的數量與用途、分類模型、PyTorch 架構比較、MD5 稽核、YOLO11 偵測／分割，以及 cascade 的目前證據與限制。

## 一、目前結論

1. 分類模型的正式 Development 結果採用 **YOLOv8n-cls（C-Arch-05）** 的 5 seeds × 5 MD5-aware GroupKFold，共 25 次 leakage-free evaluation runs。
2. Development 結果為 **Top-1 Accuracy 87.40% ± 2.78%**、**Macro-F1 87.36% ± 3.11%**；Bootstrap 95% CI 分別為 86.32–88.48% 與 86.16–88.56%。
3. 48 張測試影像是同一來源資料的逐類封存 holdout，不是第 8 類，也不是 5×5 重新切出的資料。Blind Test 一次性結果為 **46/48 = 95.83% Top-1 Accuracy**。
4. 48 張結果應解讀為「同源封存測試集上的高表現」，不能直接宣稱為外部臨床泛化能力；原因包括樣本數小、類別數量不均，以及與 Development Set 同源。
5. PyTorch Phase 3 已有一個完成且登錄的架構比較結果：**MobileNetV3-Large（C-Arch-03）**，固定 validation 的 Top-1 Accuracy 為 92.68%。這是單次 architecture-ablation 結果，不取代最後的 25-run GroupKFold 主結果。
6. WSNet YOLO11m-seg 多風格候選完成 development-val 驗證：train 7,380 張、val 461 張、`test_images_used=0`；同解析度 `imgsz=384` 下 Box mAP50=48.78%、Mask mAP50=44.56%，作為公開來源的跨域基線。
7. 臨床 bbox development run 使用 177 張 field-frame images，依 frame group 分成 train 130、val 40、locked test 7；test 實際使用 0 張。YOLO11m 在 val 的 Precision=71.60%、Recall=38.00%、mAP50=44.77%、mAP50-95=16.43%。
8. 2026-08-20 匯入的臨床 polygon candidate 共 170 張（train 130、val 40、220 個 polygon instances），標註狀態仍為 `IMPORTED_PENDING_REVIEW`。YOLO11m-seg development val 的 Box mAP50=90.98%、Mask mAP50=90.75%，但在標註複核及臨床治理文件完成前，只能視為 development evidence，不能稱為正式臨床效能。
9. 最新臨床 segmentation cascade 在 40 張 val 都產生 ROI（40/40），但該資料只有 `class 0=Wound` 定位標籤，沒有七類分類 ground truth，因此只能報 ROI coverage 與推論速度，不能計算或宣稱七類分類正確率。

## 二、分類資料與 48 張測試集來源

### 2.1 原始資料與準備後資料的差異

目前 `Wound_dataset` 原始資料共有 **431 張**。資料建立程式先逐類切分，再只對訓練集 oversampling：

```text
原始資料：431 張
├─ 初始 train：342 張
├─ validation：41 張
└─ locked test：48 張

訓練集 oversampling 後：679 張（7 類 × 97）
準備後資料總數：679 + 41 + 48 = 768 個 image instances
```

因此，768 應稱為 **prepared dataset instances**，不應稱為 768 張原始影像。

本文的 **image instance** 是資料夾中實際餵給程式的一個影像檔，不等於一名獨立病人或一張獨立原始照片；oversampling 產生的複本也會增加 instance 數。Train 會參與梯度更新；Validation 只用來選最佳 epoch、比較設定與觀察過擬合；Locked Test 在模型與參數定案後才開啟一次，不參與訓練或調參。

### 2.2 48 張如何挑選

程式 `prepare_cls_dataset.py` 的實際規則為：

- 每一類使用 `random.seed(42)` 後洗牌。
- 前 80% 放入初始 train。
- 接續 10% 放入 validation。
- 剩餘影像放入 test。
- Validation 與 Test 直接複製，沒有 oversampling。
- 只有 Train 會補到每類 97 張。

這樣切割是為了同時保留「可學習的訓練資料」、「可重複查看的開發驗證資料」與「不能回頭調參的最終測試資料」。Oversampling 只放在 Train，目的是減少模型被 Bruises 等大類別主導；Validation 與 Test 維持原始 support，否則複製測試影像會讓評估樣本被人為放大。

| 類別 | 原始張數 | 初始 Train | Val | Locked Test |
|---|---:|---:|---:|---:|
| Abrasions | 85 | 68 | 8 | 9 |
| Bruises | 122 | 97 | 12 | 13 |
| Burns | 59 | 47 | 5 | 7 |
| Cut | 50 | 40 | 5 | 5 |
| Ingrown_nails | 31 | 24 | 3 | 4 |
| Laceration | 61 | 48 | 6 | 7 |
| Stab_wound | 23 | 18 | 2 | 3 |
| **合計** | **431** | **342** | **41** | **48** |

48 張不是一個類別，而是七類影像的合計測試切分。模型仍然只輸出七個傷口類別。

### 2.3 MD5 穩定性與隔離稽核

本次對 `yolo_wound_cls_dataset_v3` 的 768 張影像重複計算 MD5：

| 檢查 | 結果 |
|---|---:|
| 重複計算檔案數 | 768 |
| 同一檔案兩次 MD5 不一致 | **0** |
| Train–Val overlap | **0** |
| Train–Test overlap | **0** |
| Val–Test overlap | **0** |
| Development files | 720 |
| Development unique MD5 groups | 383 |
| Development singleton groups | 177 |
| Development duplicate groups | 206 |
| Duplicate groups 涵蓋檔案 | 543 |
| Development redundant instances | 337 |

所以可以清楚區分：

```text
Development：720 個 instances → 383 個 MD5 groups → 用於 5×5 GroupKFold
Blind Test：48 個獨立測試影像 → 不加入 GroupKFold
```

全資料若只做統計，可得到 383 + 48 = 431 個唯一內容 groups；但實驗上不能將它們合併後重新做 fold。48 張維持獨立封存。

383 groups 不是 383 張平均大小相同的資料。177 個 singleton groups 各只有一個檔案；206 個 duplicate groups 合計涵蓋 543 個檔案，所以超出每群一個代表檔的冗餘複本為 543 − 206 = 337。GroupKFold 會整群分配，目標是讓每 fold 的影像數和類別盡量平衡，不是單純計算 383 ÷ 5。

## 三、實驗原理、資料稽核與黑盒模型證據鏈

### 3.1 MD5 是什麼，以及本研究為什麼使用它

MD5（Message-Digest Algorithm 5）是一種把檔案位元組內容轉換成固定長度摘要的雜湊函數。對本研究而言，影像檔案的位元組內容 `x` 會被轉換成 128-bit（通常以 32 個十六進位字元表示）的摘要 `MD5(x)`。它不會理解傷口、辨識類別，也不會改變影像；它只提供一個可重現的「內容指紋」。

```text
相同檔案內容（即使檔名不同）  → 相同 MD5 → 視為同一個 exact-content group
檔案位元組不同               → 通常不同 MD5 → 視為不同內容候選
```

這裡的「相同」是 **exact-content identical**，不是「肉眼看起來相似」。重新壓縮、改變 EXIF、調整一個像素，甚至某些 augmentation，都可能產生不同 MD5。因此 MD5 是本研究的第一層 exact-duplicate 稽核工具，不是完整的語意相似度或影像品質檢查器。

本研究使用 MD5 的原因有三個：

1. 原始資料包含複本、重新命名檔案，以及同一內容可能出現多個檔名的情況；只靠檔名無法確認是否為同一影像。
2. 如果同一內容的複本被分到 train 與 validation，模型可能只是在記住畫素、背景或拍攝條件，而不是學到可泛化的傷口特徵。
3. MD5 可以在訓練前建立可重現的 group manifest，讓每一次 split、audit 與教授查閱都能回溯到檔案內容，而不是依賴人工猜測。

MD5 不是安全性證明，也不能排除惡意碰撞；本研究的使用情境是非對抗性的資料去重與洩漏稽核。若未來要進一步提高證據強度，可在同一 manifest 上增加 SHA-256 或 byte-level comparison，但不能把這項後續強化誤寫成目前已完成的結果。

### 3.2 為什麼一定要做 group-aware split

機器學習真正要估計的是「模型遇到未看過的內容時能否正確預測」。若同一內容的複本同時出現在 train 和 validation，validation 就不再是獨立模擬，而會形成資料洩漏（data leakage）。此時模型可能透過近乎相同的輪廓、背景、壓縮痕跡或影像記憶得到很高分，造成樂觀偏差。

本研究的直接證據是：原本 image-level Naive StratifiedKFold 得到 98.34%，稽核後發現 **178/206 duplicate groups 跨 fold**，因此該結果標記為 `INVALIDATED`。修正後，GroupKFold 將完整 MD5 group 綁在同一個 fold，Development 的跨 fold MD5 overlap 為 0，正式主結果降為較保守且可辯護的 87.40% ± 2.78%。這個下降不是模型突然失效，而是去除了原先由複本洩漏造成的估計膨脹。

### 3.3 383 groups、48 張 test 與全資料稽核的關係

本研究同時做「全資料隔離稽核」和「Development 交叉驗證」，兩者不能混為一談：

| 用途 | 納入資料 | 結果／規則 |
|---|---|---|
| 全資料 MD5 隔離稽核 | 720 Development + 48 Blind Test | 檢查任何 group 是否跨 Development/Test；目前 overlap = 0 |
| 5×5 GroupKFold | 僅 720 Development | 720 instances 形成 383 groups，完整 group 不跨 fold |
| 最終盲測 | 僅 48 Blind Test | 選模後一次性評估，不參與訓練、調參或 fold construction |

因此，48 張 **要放進全資料 MD5 稽核**，但不能放進 383 groups 重新做 5×5。若 48 張彼此也都是獨立內容，統計上的全資料 group 數可寫成 383 + 48 = 431；這個數字只用來說明全資料內容群組，不是新的交叉驗證母體。

### 3.4 「5×5」的統計意義

5×5 代表 **5 個 random seeds × 5 個 GroupKFold partitions = 25 次 evaluation runs**，不是 25 個互相獨立的資料集。每個 partition 都使用相同的 Development pool，但每次把不同的完整 groups 放到 validation。因而報告時使用 mean、between-seed standard deviation 與 bootstrap confidence interval，而不使用「25 independent datasets」這種過度宣稱的說法。

### 3.5 黑盒模型如何用證據解釋

深度模型的內部參數數量很大，不能僅靠一個 accuracy 宣稱它「理解」了傷口，也不能把相關性結果說成臨床因果。這裡採用的是可稽核的 evidence chain：

| 教授可能追問 | 可提供的證據 | 可以下的結論 | 不能下的結論 |
|---|---|---|---|
| 你怎麼知道沒有資料洩漏？ | 原始 manifest、MD5 group 清單、Train/Val/Test overlap matrix、178/206 invalidation audit | 本實驗協定下未發現 exact-content 跨分割 | 不能說完全排除所有近似影像或病人層級相依性 |
| 87.40% 是否只是某一次運氣？ | 5 seeds × 5 folds、25-run mean/SD、Bootstrap 95% CI | 在目前 Development protocol 下具有可重現性證據 | 不能說 25 runs 是 25 個獨立資料集 |
| 48 張 95.83% 是否代表臨床泛化？ | locked test manifest、一次性 log、test support、confusion matrix | 同源封存 holdout 的 internal one-shot benchmark | 不能直接代表跨醫院、跨病人泛化 |
| 模型到底看到了什麼？ | per-class recall、confusion matrix、錯誤影像與後續 ROI/cascade audit | 可描述哪些類別較容易混淆、哪些條件造成錯誤 | 不能僅用 accuracy 證明模型依賴正確的臨床特徵 |
| 偵測與分類能否直接相乘？ | paired detector-crop/classification-label set、valid crop rate、wrong-crop rate、end-to-end accuracy | 可分別量化定位失敗與分類失敗的貢獻 | 不能用 detector mAP × classifier accuracy 代替 cascade 測試 |

因此，本研究對「黑盒」的處理不是假裝模型透明，而是把資料、切分、執行、指標、錯誤與限制全部留下可重做的證據。解釋範圍只延伸到資料與實驗支持的程度，不把統計關聯誇大成生理機制或臨床因果。

### 3.6 教授口頭報告可直接使用的說法

> 我使用 MD5 不是把它當成模型，而是把每張影像轉成可重現的內容指紋，先確認同一份內容不能同時出現在訓練和驗證。原本 98.34% 的結果在稽核後發現 178/206 個 duplicate groups 跨 fold，因此判定為資料洩漏造成的樂觀偏差；修正後以完整 group 做 5 seeds × 5 folds，得到 87.40% ± 2.78%。48 張 test 只加入全資料隔離稽核，不加入 5×5，最後才做一次盲測。對黑盒模型，我不只報一個 accuracy，而是用 split manifest、overlap audit、seed stability、confidence interval、confusion matrix、per-class recall 和錯誤案例共同說明模型在什麼條件下可靠、在哪些條件下仍有限制。

## 四、分類實驗 Phase 1–9

| Phase | 工作 | 狀態 | 說明 |
|---|---|---|---|
| Phase 1 | Logger / schema | PASS | 統一記錄模型、seed、fold、資料集與指標 |
| Phase 2 | YOLO classification trainer | PASS | 完成 Ultralytics 分類訓練流程 |
| Phase 3 | PyTorch classification trainer | PASS | 完成 ResNet、EfficientNet、MobileNetV3、ViT 的獨立訓練介面 |
| Phase 4 | Evaluation | PASS | Accuracy、Top-5、Precision、Recall、Macro-F1、Weighted-F1、ROC-AUC、confusion matrix |
| Phase 5 | Orchestrator | PASS | 串接 config、training、evaluation、logging |
| Phase 6 | Naive StratifiedKFold | INVALIDATED | 原 98.34% 結果發現 duplicate groups 跨 fold，禁止作正式結果 |
| Phase 6.5 | Data integrity audit | PASS | MD5 grouping 與跨切分 overlap audit |
| Phase 6.6 | MD5-aware GroupKFold | PASS | 完整 MD5 group 不跨 fold |
| Phase 7 | Multi-seed GroupKFold | SEALED | 5 seeds × 5 folds = 25 runs |
| Phase 8 | Bootstrap CI | PASS | B=2,000 非參數 bootstrap |
| Phase 9 | Paper tables | SEALED | 自動產生 Markdown／CSV 表格 |

### 4.1 一次分類訓練中 Train、Validation 與 Test 的作用

一次訓練可以用「練習、模擬考、正式考試」理解。Train 影像會前向計算預測、與真實標籤計算 loss，再透過 optimizer 反向更新神經網路權重；同一批資料會重複學習多個 epochs。Validation 只做推論與計分，不反向更新權重，用來選最佳 checkpoint 及判斷是否過擬合。Model selection 完成後才使用 Locked Test，測試結果寫入 JSON 與 experiment log，不能拿來改參數後重考。

C-Arch-05 每次 run 使用 `imgsz=224`、`batch=16`、最多 150 epochs、`lr0=0.01`、`weight_decay=0.0005`。5×5 指 5 個 random seeds 乘以 5 個 MD5-aware folds，共 25 次 evaluation runs；它們來自同一 Development pool，不是 25 個獨立資料集。每一 fold 的 train/validation instance 數會因完整 group 大小不同而略有差異。

### 4.2 正式 Development 結果

最終分類模型為 **C-Arch-05 / YOLOv8n-cls**：

- `imgsz=224`、`batch=16`、`epochs=150`
- `optimizer=auto`、`lr0=0.01`、`weight_decay=0.0005`
- seeds：42、123、3407、2026、999
- 5 個 MD5-aware GroupKFold partitions

| 指標 | 25-run 結果 |
|---|---:|
| Top-1 Accuracy | 87.40% ± 2.78% |
| Macro-F1 | 87.36% ± 3.11% |
| Weighted-F1 | 87.42% |
| Stab_wound Recall | 88.98% ± 9.30% |
| Between-seed Accuracy SD | 0.76% |
| Between-seed Macro-F1 SD | 0.81% |

Accuracy 是全部影像答對的比例；Precision 是模型預測為某類時有多少真正屬於該類；Recall 是某類真實影像有多少被找回；Macro-F1 讓七類各占相同權重，較能看到少數類別；Weighted-F1 依各類影像數加權，較受大類別影響。因此正式結果不能只看單一 Accuracy。

### 4.3 Bootstrap 95% CI

以 pooled validation predictions 進行 B=2,000 percentile bootstrap：

| 指標 | Mean | 95% CI |
|---|---:|---:|
| Top-1 Accuracy | 87.40% | 86.32–88.48% |
| Macro-F1 | 87.36% | 86.16–88.56% |
| Weighted-F1 | 87.42% | 86.32–88.52% |
| Stab_wound Recall | 88.98% | 85.37–92.59% |

### 4.4 48 張 Blind Test 結果

| 指標 | 結果 |
|---|---:|
| Top-1 Accuracy | 95.83%（46/48） |
| Top-5 Accuracy | 100.00% |
| Macro-F1 | 94.76% |
| Weighted-F1 | 96.23% |
| Macro ROC-AUC | 0.9982 |
| 錯誤數 | 2/48 |

此結果只作為 **internal one-shot benchmark**。48 張與 Development Set 來源相同，且各類 support 不均，不能取代外部醫院或跨病人測試。

#### Locked Test 的類別不平衡與錯誤解讀

Locked Test 不是每一類相同數量的 balanced test set，而是依照原始資料逐類封存後得到的 48 張。因此每個類別的 support 不同，單一錯誤對 Recall 的影響也不同：

| 類別 | Test support (n) | 正確 | 錯誤／預測類別 | Recall | 若多 1 個錯誤，Recall 約下降 |
|---|---:|---:|---|---:|---:|
| Abrasions | 9 | 9 | 無 | 100.00% | 11.11 個百分點 |
| Bruises | 13 | 12 | 1 → Stab_wound | 92.31% | 7.69 個百分點 |
| Burns | 7 | 7 | 無 | 100.00% | 14.29 個百分點 |
| Cut | 5 | 5 | 無 | 100.00% | 20.00 個百分點 |
| Ingrown_nails | 4 | 4 | 無 | 100.00% | 25.00 個百分點 |
| Laceration | 7 | 6 | 1 → Stab_wound | 85.71% | 14.29 個百分點 |
| Stab_wound | 3 | 3 | 無 False Negative；但收到 2 個 False Positive | 100.00% | 33.33 個百分點 |

本次 48 張測試的兩個 Top-1 錯誤為：

1. **Bruises → Stab_wound：1 張**。因此 Bruises Recall 為 12/13 = 92.31%。
2. **Laceration → Stab_wound：1 張**。因此 Laceration Recall 為 6/7 = 85.71%。

`Stab_wound` 的 Recall 仍為 3/3 = 100%，但 Precision 只有 60.00%，因為它實際只有 3 張，卻額外接收了 Bruises 與 Laceration 各 1 張的錯誤預測：`3 / (3 + 2) = 60%`。這說明小 support 類別的單一樣本會大幅改變指標，不能只看該類別的 100% Recall 就宣稱辨識穩定。

因此，本次結果應同時報告：整體 Accuracy、Macro-F1、Weighted-F1、每類 support、每類 Recall/Precision 與混淆方向。Accuracy 95.83%（46/48）是 48 張同源封存資料的結果；Macro-F1 以各類別等權重呈現，Weighted-F1 則會較受 Bruises（n=13）等大 support 類別影響。由於最小類別只有 3–4 張，這組 locked-test 數值適合作為一次性內部 benchmark，不足以單獨證明跨病人或跨醫院泛化。

## 五、PyTorch Phase 3 詳細結果

### 5.1 C-Arch-03／MobileNetV3-Large

此結果來自 `experiments/experiment_log.csv` 中唯一完成並標記 `done` 的獨立 PyTorch run：

- Framework：PyTorch／Torchvision
- Architecture：MobileNetV3-Large，ImageNet pretrained
- Seed：42，fold=0
- Train：679，Validation：41
- Image size：224，batch size：32
- Epoch 上限：150；best epoch：49
- Optimizer：Adam，`lr=0.001`，`weight_decay=0.0005`
- Training augmentation：horizontal/vertical flip、rotation、scale、ColorJitter、RandomErasing

| 指標 | Validation 結果 |
|---|---:|
| Top-1 Accuracy | 92.68% |
| Precision | 93.65% |
| Recall | 89.29% |
| Macro-F1 | 90.10% |
| Weighted-F1 | 92.41% |
| ROC-AUC | 0.9884 |

![PyTorch C-Arch-03 validation metrics](figures/pytorch_carch03_validation_metrics.png)

**圖說：** PyTorch C-Arch-03 在固定 validation split（n=41）的已登錄指標摘要。這不是重新訓練產生的曲線；目前程式只保留 best epoch 與最終指標，沒有保存逐 epoch loss/accuracy CSV，因此報告不捏造 training curve。

此 PyTorch 結果是單次固定 train/val architecture ablation，與最終 25-run MD5-aware GroupKFold 不可直接等價比較；正式主結果仍以 C-Arch-05 的 25-run 結果為準。

目前沒有完成並登錄的 C-Arch-01 ResNet50、C-Arch-02 EfficientNet-B0 或 C-Arch-04 ViT 多次 benchmark 結果，因此不在報告中虛構排名。

## 六、YOLO11 物件偵測、分割與 cascade 進度

分類回答「這張圖是哪一類」；物件偵測回答「傷口在哪裡」，以 bounding box 表示；分割則對每個傷口像素畫出 mask。Cascade 是把定位模型與七類分類器串在一起。這三個任務的 ground truth 和指標不同，不能把 mAP、分類 Accuracy 與 ROI coverage 當成同一個分數。

### 6.1 WSNet 公開來源基線

WSNet branch 提供 image-mask pairs，可同時評估 box 與 pixel mask。原始 union-box 資料共 2,686 張、3,823 個 boxes：train 1,894 張／2,681 boxes，validation 412 張／609 boxes，locked detection test 380 張／533 boxes；test/380 至今未用於 development 選模。

後續 segmentation branch 以 1,845 張 WSNet train originals 產生三種可追溯變體（angle/perspective、illumination、sensor style），形成 7,380 張 train；461 張 validation 保持原始、不做擴充，避免驗證資料被人工變體污染。`imgsz=384` 的公平 development-val 結果如下：

| 指標 | 結果 | 白話解讀 |
|---|---:|---|
| Box Precision | 58.90% | 預測出的 boxes 中，符合配對條件的比例傾向 |
| Box Recall | 46.97% | 真實傷口 boxes 被找回的比例傾向 |
| Box mAP50 | 48.78% | IoU 0.50 下的平均 precision-recall 表現 |
| Box mAP50-95 | 22.39% | IoU 0.50–0.95 多個嚴格門檻的平均，更重視定位精準度 |
| Mask Precision | 55.59% | 預測 masks 的 precision |
| Mask Recall | 44.09% | 真實 masks 被找回的 recall |
| Mask mAP50 | 44.56% | mask IoU 0.50 下的平均表現 |
| Mask mAP50-95 | 16.84% | 多門檻 mask 定位表現 |

這些是公開來源 development validation，不是臨床 test。`mAP50-95` 低於 `mAP50`，表示模型在「大致找到傷口」後，邊界貼合度仍有改善空間。

### 6.2 臨床 bbox development 資料與結果

臨床 field-frame bbox 資料共 177 張，依 frame／augmentation group 隔離成 train 130、validation 40、locked test 7，group split overlap=0。Train 用來更新 YOLO11m 權重；Validation 用來選 checkpoint 與計算偵測指標；7 張 test 保持封存，`test_images_used=0`。

| Split | Images | 本次用途 |
|---|---:|---|
| Train | 130 | 更新 YOLO11m bbox detector 權重 |
| Validation | 40 | 開發期選模與 mAP 評估 |
| Locked Test | 7 | 保留；本次未讀取、未評估 |
| **Total** | **177** | 同一臨床來源的 development-only 資料 |

150-epoch development run 的 Validation 結果為 Precision 71.60%、Recall 38.00%、mAP50 44.77%、mAP50-95 16.43%。Precision 較高而 Recall 偏低，代表模型一旦報出傷口通常較保守，但仍漏掉不少真實傷口；因此不能只看 Precision 就說定位已可用。

此資料夾的 filename lineage 可追溯，但目前治理狀態仍為 `DEVELOPMENT_ONLY_PENDING_CLINICAL_GOVERNANCE_RECORD`；現有紀錄不能取代 IRB、consent、去識別化與發表權證據。

### 6.3 臨床 segmentation development candidate

Label Studio 匯入後的 segmentation candidate 共 170 張：train 130、validation 40，合計 220 個 polygon instances；test 使用 0 張。`annotation_status=IMPORTED_PENDING_REVIEW` 表示 polygon 已轉入訓練格式，但尚未完成專業人工複核，因此結果只能用來判斷工程可行性。

| 指標 | 40-image Validation 結果 |
|---|---:|
| Box Precision | 94.22% |
| Box Recall | 73.33% |
| Box mAP50 | 90.98% |
| Box mAP50-95 | 51.48% |
| Mask Precision | 94.22% |
| Mask Recall | 73.33% |
| Mask mAP50 | 90.75% |
| Mask mAP50-95 | 42.33% |

這組數字明顯高於 WSNet 跨域基線，但不能直接解讀為正式臨床效能：train 與 val 來自同一 field-frame 資料來源、標註仍待覆核，而且同一原始 frame 的多個 augmentation 必須持續以 group 為單位稽核。正式結論須等 polygon review、治理文件與 locked test 協定完成。

### 6.4 Cascade 推論目前能證明什麼

```text
YOLO11 detector／segmenter：在完整影像找到傷口 ROI
        ↓
YOLOv8n-cls：將裁切 ROI 預測為七類之一
        ↓
護理師人工複核 → 照護建議 → 經核准內容才進入 RAG 知識庫
```

臨床 bbox detector 在 40 張 val 找到 28 張，ROI coverage=70%，平均 latency=88.20 ms。臨床 segmenter 在同一批 40 張都產生 ROI，coverage=100%，平均 latency=79.03 ms。Coverage 只表示程式有輸出 ROI，不等於每個 ROI 都與真實傷口正確重疊；正式定位正確率仍要以人工核准的 bbox／mask ground truth 計算 IoU。

40 張 clinical validation labels 只有 `class 0=Wound`，沒有 Abrasions、Bruises、Burns、Cut、Ingrown_nails、Laceration、Stab_wound 的七類 ground truth。因此目前只能列出分類器的預測分布，不能計算 classification accuracy、Macro-F1 或 end-to-end cascade accuracy。任何七類正確率都必須等同一影像同時具備人工核准的 ROI 與七類標籤後再測。

### 6.5 Detection／Segmentation phase status

| 階段 | 目前狀態 | 尚缺證據 |
|---|---|---|
| Config、trainer、evaluator、logger | PASS | 持續保留版本與 checkpoint lineage |
| WSNet development baseline | PASS | locked test/380 尚未開啟 |
| Clinical bbox development | PASS_DEVELOPMENT_ONLY | test/7 未用；治理文件待完成 |
| Clinical segmentation candidate | PASS_DEVELOPMENT_ONLY | polygon 專業複核與治理文件待完成 |
| Clinical cascade inference | PASS_INFERENCE_ONLY | 缺同影像七類 ground truth，不能算分類正確率 |
| 多模型／多 seed benchmark | 尚未封版 | 需相同 split、相同門檻與 CI |

## 七、教授報告時的重點說法

可以用以下四句話總結：

1. **分類主結果不是 98.34%，而是經 MD5 group-aware audit 後的 25-run 結果：87.40% ± 2.78%。**
2. **48 張是同源、逐類、seed=42 的封存測試集；95.83% 是一次性內部 holdout benchmark，不宣稱外部臨床泛化。**
3. **臨床 bbox 與 segmentation 已完成 development runs，但 test 均未使用；segmentation 標註仍待專業複核，治理文件亦未完成。**
4. **目前 cascade 已能量測 ROI coverage 與 latency，但臨床資料缺七類 ground truth，所以不能宣稱七類 end-to-end accuracy。**

## 八、目前限制與下一步

- 不再修改已封版的分類 Phase 1–9，也不重跑分類 Blind Test。
- 完成 YOLO11n／YOLO11s／YOLO11m 的同一 WSNet split、相同評估規則之模型比較。
- Detection test/380 仍維持封存，待 detector selection 完成後只評估一次。
- Clinical bbox test/7 仍維持封存；在治理與標註複核完成前不把 development metrics 當正式臨床結果。
- 由專業人員覆核 170 張 clinical segmentation polygons，並補齊 governance record。
- 建立同一影像同時包含核准 bbox／mask 與七類標籤的 paired set，正式驗證 cascade，而不是用 mAP 與 Accuracy 直接相乘。
- 若要證明臨床泛化，仍需不同病人／不同場域或具來源證據的外部測試資料；目前 48 張不能承擔此結論。

## 九、主要證據檔案

- 分類資料建立規則：`prepare_cls_dataset.py`
- PyTorch trainer：`experiments/scripts/train_cls_torch.py`
- PyTorch 指標紀錄：`experiments/experiment_log.csv`
- 分類結果表：`experiments/results/tables/`
- Blind Test JSON：`experiments/results/statistics/C-Arch-05_final_blind_test_report.json`
- YOLO11m development report：`outputs/detection_unionbox/YOLO11m_WSNet_unionbox_refine_noaug_20260817_professor_report.md`
- YOLO11m training curve：`outputs/detection_unionbox/YOLO11m_WSNet_unionbox_refine_noaug_20260817_progress_curve.png`
- YOLO11m-seg latest summary：`outputs/segmentation/YOLO11m_WSNet_seg_aug_v5_20260818_summary.json`
- YOLO11m-seg fair 384 evaluation：`outputs/segmentation/YOLO11m_WSNet_seg_aug_v5_20260818_eval384.json`
- YOLO11m-seg cascade validation：`outputs/segmentation/YOLO11m_WSNet_seg_aug_v5_20260818_production_cascade_val.json`
- Cascade architecture decision：`outputs/segmentation/cascade_architecture_decision_report_20260818.json`
- Clinical annotation gate：`outputs/clinical_seg_annotation_gate_20260818.json`
- Clinical bbox 150-epoch summary：`outputs/clinical_detection/YOLO11m_clinical_bbox_dev_20260819_e150_summary.json`
- Clinical bbox cascade inference：`outputs/clinical_detection/YOLO11m_clinical_bbox_cascade_val_inference_20260819_e150/summary.json`
- Clinical segmentation candidate：`clinical_seg_candidate_20260820/candidate_summary.json`
- Clinical segmentation model summary：`outputs/segmentation/YOLO11m_clinical_seg_dev_20260820_summary.json`
- Clinical segmentation cascade inference：`outputs/clinical_detection/YOLO11m_clinical_seg_cascade_val_inference_20260820/summary.json`
