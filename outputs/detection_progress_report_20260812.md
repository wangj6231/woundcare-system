# YOLO11 物件偵測研究進度報告

日期：2026-08-12  
研究主題：傷口物件偵測與偵測後分類串接驗證

## 一、目前研究決策

分類模型 Phase 1–9 已封版，分類資料與結果均保留且未被本次偵測工作修改。物件偵測部分已刪除未能完成來源／授權證據的 Kaggle 與 Roboflow 副本，臨床影片資料則隔離保存，不與公開資料混用。

目前可重現的公開來源實驗先採用官方 WSNet 固定版本；FUSeg 僅取得官方 repository 的測試影像（沒有可供本研究使用的標註），Medetec 已下載並完成檔案雜湊，但其條款仍需在正式訓練或發表前完成用途確認，因此兩者未進入模型訓練。

## 二、資料與隔離稽核

| 項目 | 狀態 | 證據 |
|---|---|---|
| WSNet 官方版本 | PASS（內部非商業研究） | Hugging Face revision `bdcc1e1cef607bfd7a4f98fc28f31a638a0e0ca9` |
| WSNet train/val/test | PASS | 1,894 / 412 / 380 images |
| WSNet exact image-hash 跨 split | PASS | 0 個跨 split exact-hash group |
| WSNet 影像與 mask 配對 | PASS | 2,686 / 2,686 |
| Kaggle / Roboflow 偵測資料 | 已移除 | 刪除報告與保留分類資料核對 |
| 臨床資料 | 隔離保存 | `clinical_detection_isolated_20260812/`，未與 WSNet 混合 |
| Test split 使用情形 | PASS | 所有目前 baseline 與 screening 均未使用 test |

## 三、YOLO11 開發期多模型比較

三個模型均使用 WSNet 同一個 source-specific split、image size 640、seed 42、完整 train split 與 5 epochs。為排除 batch 差異，已完成共同 batch=4 的控制跑。

| 模型 | Batch | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 4 | 0.4087 | 0.3542 | 0.2922 | 0.1085 |
| YOLO11s | 4 | 0.3278 | 0.3120 | 0.2416 | 0.0772 |
| YOLO11m | 4 | 0.2469 | 0.2791 | 0.1648 | 0.0515 |

### 解讀限制

以上是 5-epoch、single-seed、single-validation-split 的開發期 screening，不是正式論文結果，也不能據此宣稱 YOLO11n 已經是最終模型。這組結果只用於縮小後續正式 benchmark 的候選範圍；test split 尚未開封。

## 四、目前階段判定

### 已完成

- D1：設定、訓練與驗證流程可執行。
- D2：WSNet source-specific 轉換、hash grouping 與 train/val/test 隔離稽核完成。
- D3：YOLO11m WSNet smoke test 完成，確認 train → validation → artifact export 流程。
- D4（開發期 screening）：YOLO11n、YOLO11s、YOLO11m 共同 batch=4 比較完成。
- D4.5（GroupKFold protocol pilot）：YOLO11n fold 1 / seed 42 完成，確認新的 ASCII image-list、label resolution、訓練與驗證 artifact 流程。
- D5.1（Formal first run）：YOLO11n fold 1 / seed 42 已完成；Precision 0.6490、Recall 0.4378、mAP@50 0.4909、mAP@50–95 0.2283。另有一個重複啟動的 auto-suffixed run 已保留但排除統計。
- D5.2（Formal second run）：YOLO11n fold 1 / seed 123 已完成；Precision 0.5964、Recall 0.4568、mAP@50 0.4943、mAP@50–95 0.2252；early stopping at 135 epochs。
- 來源與檔案雜湊、權重、results.csv、JSON summary 均已保存。

### 尚未完成，不得提前宣稱

- 正式 5-fold × 5-seed GroupKFold 多模型 benchmark。
- Bootstrap 95% CI、模型間統計比較與 inference latency 的正式報告。
- WSNet test split 的一次性盲測。
- 偵測→分類 cascade 的端到端正確率。

GroupKFold pilot 的 5-epoch 數值只用於流程驗證，不列入正式統計；完整 benchmark 仍須依預先鎖定的 epoch、early-stopping 與每一 fold/seed 紀錄執行。

## 五、偵測→分類串接目前的科學限制

現有 WSNet 標註是單一 `Wound` 偵測類別；既有分類資料是七類傷口分類，兩者沒有同一影像同時具備 bbox 與七類 class label 的已驗證 paired set。因此目前不能把偵測框裁切後直接宣稱為端到端分類正確率。下一個必要資料工作是建立並封存 paired bbox + seven-class protocol，且要補齊臨床 IRB／consent／去識別化／frame extraction／augmentation lineage 證據後，才可做臨床 cascade test。

## 六、下一步執行順序

1. 以共同 batch=4、固定 augmentation、固定 optimizer policy，建立 YOLO11n/s/m 的正式 benchmark runner。
2. 在 development data 執行 5 folds × 5 seeds；保存每一 run 的 seed、fold、hash manifest、weights、metrics 與 latency。
3. 只用 development 結果預先決定 final detector；在決策前不開啟任何 test split。
4. 對選定 detector 執行一次 WSNet locked test evaluation，另行保存，不與 CV 平均混合。
5. 建立 paired bbox + seven-class cascade set，完成一次 detection→classification evaluation；cascade 測試影像不得回饋調參。

## 七、主要成果檔案

- 多模型比較：`outputs/wsnet_official_multimodel_development_comparison_20260812.md`
- 多模型 JSON：`outputs/wsnet_official_multimodel_development_comparison_20260812.json`
- YOLO11n common batch=4：`outputs/wsnet_yolo11n_common_batch4_summary_20260812.json`
- YOLO11s common batch=4：`outputs/wsnet_yolo11s_common_batch4_summary_20260812.json`
- YOLO11m baseline：`outputs/wsnet_official_baseline_summary_20260812.json`
- WSNet 轉換摘要：`outputs/wsnet_official_yolo_build_summary_20260812.json`
- WSNet 來源 intake：`outputs/wsnet_official_intake_report_20260812.md`
- GroupKFold manifest audit：`outputs/wsnet_groupkfold_manifest_report_20260812.md`
- GroupKFold pilot：`outputs/wsnet_groupkfold_pilot_report_20260812.md`
- Formal 75-run plan：`outputs/wsnet_formal_benchmark_plan_20260812.md`
- Formal run manifest：`experiments/manifests/wsnet_formal_benchmark_runs_20260812.csv`
- YOLO11n fold 1 / seed 42 report：`outputs/wsnet_yolo11n_formal_fold01_seed42_report_20260812.md`
- YOLO11n fold 1 / seed 123 report：`outputs/wsnet_yolo11n_formal_fold01_seed123_report_20260812.md`

## 八、官方來源連結

- [WSNet dataset revision](https://huggingface.co/datasets/EthanYin98/WSNet/tree/bdcc1e1cef607bfd7a4f98fc28f31a638a0e0ca9)
- [WSNet authors’ code](https://github.com/EthanYin98/WSNet)
- [FUSeg official repository](https://github.com/uwm-bigdata/wound-segmentation)
- [Medetec official image database](https://medetec.co.uk/files/medetec-image-databases.html)
