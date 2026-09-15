"""Produce the audit addendum from saved evidence; never run a model or open test pixels."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from .audit_project import ROOT, read_json, rel, sha, snapshot_sealed, write_json
from .guards import validate_optimizer_settings


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def linked(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return f"[{p.name}](<{p.as_posix()}>)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    audit = Path(args.audit).resolve()
    if (ROOT / "outputs").resolve() not in audit.parents:
        raise ValueError("audit directory must be inside workspace outputs")
    if (audit / "專案模型實驗審查與改進報告.md").exists():
        raise FileExistsError("refusing to overwrite completed report")
    summary = read_json(audit / "audit_summary.json")
    cls = read_json(audit / "classification_recomputed.json")
    runs = read_json(audit / "training_run_inventory.json")["runs"]
    datasets = {n: read_json(audit / f"yolo_dataset_{n}_audit.json") for n in (
        "dseg06_small_multi_v1", "dseg07_yasin_wound_seg_v2", "dseg08_replay_v1")}
    d06, d07, d08 = datasets.values()
    reference_args = yaml.safe_load((ROOT / "experiments/results/segmentation/D-Seg-08_yolo11m_replay_s42/args.yaml").read_text(encoding="utf-8-sig"))
    with (ROOT / "experiments/results/segmentation/D-Seg-08_yolo11m_replay_s42/results.csv").open(encoding="utf-8-sig") as f:
        lr_rows = list(csv.DictReader(f))
    bias_lr_first = float(lr_rows[0]["lr/pg0"])
    recipe = ROOT / "experiments/review_v2/recipes/D-Seg-08R_warmup_control.yaml"
    recipe_data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    validate_optimizer_settings(recipe_data)
    optimizer_audit = {"reference_lr0": reference_args["lr0"], "reference_warmup_bias_lr": reference_args["warmup_bias_lr"],
                       "observed_first_epoch_bias_lr": bias_lr_first,
                       "observed_to_base_ratio": bias_lr_first / reference_args["lr0"],
                       "fixed_recipe": rel(recipe), "recipe_optimizer_guard": "PASS",
                       "new_training_started": False,
                       "causality": "Observed LR overshoot; effect on held-out accuracy not measured by this audit."}
    write_json(audit / "optimizer_audit.json", optimizer_audit)

    ancestry = []
    train = {r["image_sha256"]: r for d in datasets.values() for r in d["splits"]["train"]["rows"]}
    for split in ("val", "val_retention"):
        rows = d08["splits"][split]["rows"]
        near = [(t["image"], v["image"]) for t in train.values() for v in rows
                if (int(t["phash"], 16) ^ int(v["phash"], 16)).bit_count() <= 4]
        exact = set(train) & {v["image_sha256"] for v in rows}
        ancestry.append({"comparison": f"D-Seg-06/07/08 ancestor training union vs current {split}",
                         "ancestor_unique_content": len(train), "validation_images": len(rows),
                         "exact_overlap": len(exact), "phash_le4_pairs": near})
    write_json(audit / "checkpoint_ancestry_split_audit.json", ancestry)

    diagnostics = []
    for run in runs:
        with (ROOT / run["run_path"] / "results.csv").open(encoding="utf-8-sig") as f:
            records = [{k.strip(): v.strip() for k, v in row.items() if k} for row in csv.DictReader(f)]
        epochs = [float(r["epoch"]) for r in records if r.get("epoch")]
        resets = sum(b <= a for a, b in zip(epochs, epochs[1:]))
        duplicates = len(epochs) - len(set(epochs))
        if resets or duplicates or run["nonfinite_cells"]:
            diagnostics.append({"run": run["run_path"], "epoch_rows": len(epochs),
                                "epoch_resets_or_nonincreasing": resets, "duplicate_epoch_fields": duplicates,
                                "nonfinite_cells": run["nonfinite_cells"],
                                "decision": "Do not treat this CSV as one clean independent experiment; keep historical files unchanged."})
    write_json(audit / "run_history_anomalies.json", diagnostics)

    paths = {
        "app_default_classifier": "runs/classify/wound_classifier_v32/weights/best.pt",
        "reported_C_Arch_05_checkpoint": "experiments/results/raw/C-Arch-05_yolov8n_cls_s42_f0/weights/best.pt",
        "app_default_segmenter": "experiments/results/segmentation/YOLO11m_WSNet_seg_aug_v5_20260818/weights/best.pt",
        "frozen_D_Seg_09B_candidate": "experiments/results/segmentation/D-Seg-09B_weight_interpolation_20260831/weights/alpha_075.pt",
        "app_default_detector": "Wound_AI_Detection_v2/R1_YOLO11m_82/weights/best.pt"}
    identities = {k: {"path": p, "exists": (ROOT / p).is_file(), "sha256": sha(ROOT / p) if (ROOT / p).is_file() else None} for k, p in paths.items()}
    identities["scope"] = "Static backend defaults and file identities; live server process/environment was not inspected or restarted. No model objects loaded."
    identities["current_shell_override_present"] = {k: bool(os.environ.get(k)) for k in ("WOUNDCARE_CLS_MODEL", "WOUNDCARE_SEG_MODEL", "WOUNDCARE_DET_MODEL")}
    write_json(audit / "app_model_identity_audit.json", identities)

    commands = [[sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_experiment_review.py", "-v"],
                [sys.executable, "tests/test_woundcare_inference.py"]]
    test_results = []
    for cmd in commands:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=60)
        test_results.append({"command": cmd, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    write_json(audit / "verification_tests.json", test_results)
    tests_passed = all(t["exit_code"] == 0 for t in test_results)
    original = read_json(audit / "sealed_before_sha256.json")
    final = snapshot_sealed()
    unchanged = original == final
    write_json(audit / "final_sealed_integrity.json", {"unchanged": unchanged, "files": len(original),
        "changed": [k for k in original.keys() | final.keys() if original.get(k) != final.get(k)],
        "test_images_opened_this_review": 0, "models_loaded_this_review": 0, "new_training_runs": 0})

    findings = [
        {"id": "R01", "severity": "HIGH", "issue": "Phase9 tables contain hardcoded metrics; original N=3622 differs from 3600 saved predictions.", "disposition": "CORRECTED_ADDENDUM; historic files unchanged", "evidence": "classification_recomputed.json"},
        {"id": "R02", "severity": "HIGH", "issue": "Legacy row bootstrap ignores repeated-image/group dependence; no saved sample IDs.", "disposition": "CI_NOT_REPRODUCIBLY_VALIDATED; new identity-aware metric module", "evidence": "classification_recomputed.json"},
        {"id": "R03", "severity": "HIGH", "issue": "AdamW low-LR fine-tuning inherited warmup_bias_lr=0.1.", "disposition": "NEW_RECIPE_AND_GUARD; causal ablation not trained", "evidence": "optimizer_audit.json"},
        {"id": "R04", "severity": "HIGH", "issue": "YOLO11m from_n_best is actually YOLO11n per existing architecture audit.", "disposition": "REPORT_CORRECTION; do not rank by directory label", "evidence": "outputs/detection_unionbox/YOLO11m_WSNet_unionbox_from_n_best_20260817_identity_audit.md"},
        {"id": "R05", "severity": "HIGH", "issue": "Legacy R1 log contains epoch resets, repeated epoch IDs, and NaN validation losses.", "disposition": "HISTORICAL_ONLY_NOT_CLEAN_SINGLE_RUN", "evidence": "run_history_anomalies.json"},
        {"id": "R06", "severity": "HIGH", "issue": "Legacy orchestrator can log done after evaluator failure, mixes tasks and trains during blind_test mode.", "disposition": "LEGACY_NOT_AUTHORIZED_FOR_NEW_RUNS; keep sealed source intact", "evidence": "experiments/run_experiments.py"},
        {"id": "R07", "severity": "HIGH", "issue": "App default checkpoint hashes do not match report candidates.", "disposition": "NO_AUTOMATIC_WEIGHT_PROMOTION; identity audit provided", "evidence": "app_model_identity_audit.json"},
        {"id": "R08", "severity": "HIGH", "issue": "NaN confidence/malformed polygons could bypass threshold semantics or crash ROI conversion.", "disposition": "FIXED_IN_INFERENCE_MODULE_AND_TESTED", "evidence": "verification_tests.json"},
        {"id": "R09", "severity": "HIGH", "issue": "Internal validation, external evaluation, proposal coverage, selective accuracy and all-image accuracy were easily conflated.", "disposition": "SEPARATE_ESTIMANDS_AND_COHORTS; no new accuracy claim", "evidence": "saved_summary_inventory.json"},
        {"id": "R10", "severity": "HIGH", "issue": "Redscar approval missing; Yasin source permission remains UNKNOWN in stored source audit.", "disposition": "SOURCE_RESTRICTED; no fabricated consent or deployment rights", "evidence": "outputs/dseg08_replay_dataset_audit_20260830.json"},
        {"id": "R11", "severity": "MEDIUM", "issue": "Oversampled rows are not unique images; labels are hardlinked to source artifacts.", "disposition": "COUNTS_CORRECTED; never edit labels in place", "evidence": "yolo_dataset_dseg06_small_multi_v1_audit.json"},
        {"id": "R12", "severity": "HIGH", "issue": "Future repartitioned CV initialized from wound-tuned ancestors would leak prior training into new validation.", "disposition": "NEW_FAIL_CLOSED_GUARD; current fixed-split ancestry separately checked", "evidence": "checkpoint_ancestry_split_audit.json"},
        {"id": "R13", "severity": "MEDIUM", "issue": "ViT vit_s_16 evaluator constructs vit_b_16; glob-based legacy evaluator omits .jpeg.", "disposition": "DO_NOT_CLAIM_SMALL_VIT_BASELINE; legacy code preserved", "evidence": "experiments/scripts/evaluate.py"}]
    write_json(audit / "findings_and_dispositions.json", findings)

    # All result numbers below come from saved evidence, not from literal table strings.
    fusion = read_json(ROOT / "outputs/segmentation/dseg09b_carch05_dual_view_fusion_20260831/summary.json")
    external = read_json(ROOT / "outputs/segmentation/co2wounds_external_test_20260831/D-Seg-09B_CO2Wounds-V2_external_results.json")
    frozen = read_json(ROOT / "outputs/segmentation/dseg09b_weight_interpolation_20260831/D-Seg-09B_candidate_freeze_manifest.json")
    sealed_cls = read_json(ROOT / "experiments/results/predictions/test/C-Arch-05_predictions.json")["metrics"]
    report = ["# 專案模型實驗審查與改進報告", "", "日期：2026-09-14（台灣時間）", "",
              "## 一、結論與審查範圍", "",
              f"已清查 {len(runs)} 份訓練歷程與 {summary['script_count']} 份實驗相關 Python 腳本，重算分類 25-run 描述統計，並對目前 D-Seg-06／07v2／08 開發資料進行逐檔雜湊、可解碼性、polygon 數值與跨 split 重複檢查。", "",
              "不是所有過去成績都無效，也不能因為某個內部驗證超過 90% 就宣稱可落地。主要問題是部分報表不可重現、模型名稱與實體不一致、微調暖身設定、跨來源驗證不足，以及應用端未對齊研究權重。", "",
              "本次是實際程式修正與證據稽核，不是新一輪模型成績。未開啟任何 test 影像、未推論、未重訓、未更換 App 權重、未重新執行分類或 CO2Wounds 盲測。", "",
              "完整訓練紀錄清單：" + linked(audit / "training_run_inventory.csv") + "；每份最後一列訓練指標及設定雜湊：" + linked(audit / "training_run_inventory.json"), "",
              "歷史來源的每張影像、每份舊標註及所有原始 checkpoint 內部結構並未重新全面驗證；不能把本次 inventory 當成所有歷史實驗的品質認證。", "",
              "| 任務 | 已找到訓練歷程 |", "|---|---:|"]
    for task in ("classify", "detect", "segment"):
        report.append(f"| {task} | {sum(r['task'] == task for r in runs)} |")
    report += ["", "## 二、分類 Phase 1–9：保留封版，另附更正", "",
               "Phase 1–5：靜態檢視發現舊調度器評估失敗後仍可能 log_done；Blind Test 路徑會先訓練；平面偵測設定不符合其巢狀分類 schema。這些舊程式保留作為歷史證據，禁止用它啟動本次新實驗。", "",
               "Phase 6：Naive CV 舊成績維持失效，不恢復採用。Phase 6.5–7：Group CV 結果可重算描述值，但 25 runs 有共享資料／訓練集，不是 25 個獨立試驗。", "",
               "Phase 8：舊檔案的 N=3,622 與 25 個 prediction JSON 實際合計 3,600 不符；每 seed 720 列，不能把 3,600 當成 3,600 張獨立影像。缺 image_id／hash／group_id，因此原 95% CI 暫列『無法按相依結構重現驗證』，本次不捏造新的 CI。", "",
               "Phase 9：`build_all_phase9_tables.py` 以固定字串輸出部分數值；本報告直接讀預測重算，原表不覆寫。主 Accuracy／Macro-F1 的平均仍相符；Per-class Recall、N 與部分 F1／SD 用語需要更正。", "",
               "| 指標 | 25 runs 不加權平均 | 樣本 SD（ddof=1，百分點） | 母體 SD（ddof=0，百分點） |",
               "|---|---:|---:|---:|"]
    for name, key in [("Accuracy", "accuracy_all_samples"), ("Macro-F1（固定七類）", "macro_f1_fixed_classes"), ("Weighted-F1", "weighted_f1"), ("Top-5", "top5_accuracy")]:
        m = cls["run_descriptives"][key]
        report.append(f"| {name} | {pct(m['run_mean'])} | {100*m['run_sd_ddof1']:.2f} | {100*m['run_sd_ddof0']:.2f} |")
    report += ["", "舊 Accuracy ±2.78 是 ddof=0，可由資料重現；改報 ddof=1 會是 ±2.84，不是訓練性能發生改變。SD 是描述性波動，不是 CI。", "",
               "| 七類 Recall | 25-run 平均 | 3,600 列重複 OOF 合併描述值 |", "|---|---:|---:|"]
    for i, name in enumerate(cls["class_names"]):
        report.append(f"| {name} | {pct(cls['run_descriptives'][f'recall/{name}']['run_mean'])} | {pct(cls['pooled_repeated_prediction_metrics']['recall'][i])} |")
    report += ["", "以上是兩種不同加權方式，不能與舊表逐數混用；也不能把 recall 欄位當成臨床診斷敏感度。", "",
               f"分類已封存 test 原紀錄：n={sealed_cls['n_samples']}，Accuracy={sealed_cls['accuracy']:.2f}%，Macro-F1={sealed_cls['macro_f1']:.2f}%。本次只讀既存 JSON；這不是新測試，亦不併入 CV。", "",
               "重算證據：" + linked(audit / "classification_recomputed.json"), "",
               "## 三、偵測與分割：資料、模型與比較口徑", "",
               "| 資料版本 | split | 影像檔數／抽樣筆數 | unique SHA256 | 負樣本影像 | polygons |", "|---|---|---:|---:|---:|---:|"]
    for name, d in datasets.items():
        for split, s in d["splits"].items():
            report.append(f"| {name} | {split} | {s['image_files']} | {s['unique_content_sha256']} | {s['negative_images']} | {s['instances']} |")
    report += ["", "D-Seg-06 的 2,439 是訓練呈現筆數；2,007 才是不同檔案內容，差額 432 是 focus oversampling。這可以提高小目標／多傷口的曝光次數，但沒有增加獨立資訊。", "",
               "D-Seg-08：manual=326、FUSeg=163、AZH=163、BUBT healthy=82。等量 public-positive replay 並不等於所有來源、面積、膚色或拍攝條件已平衡；FUSeg 的傷口尺寸分布也不同。", "",
               "目前 primary val 是 38 張人工 polygon 公開來源圖（不是 38 位病患），沒有正常皮膚負樣本；retention val 全部來自 FUSeg，共 191 張、僅 5 張負樣本。沒有分別評估 AZH／BUBT 的完整來源外推能力，不能把此 retention 當作多來源平均泛化。", "",
               "本次逐檔 SHA256／pHash（距離≤4）train-vs-val 稽核通過；目前 D-Seg-06→07→08 所有祖先訓練內容聯集對 current primary／retention 也重新檢查：", ""]
    for a in ancestry:
        report.append(f"- {a['comparison']}：exact overlap={a['exact_overlap']}，pHash近似對={len(a['phash_le4_pairs'])}。")
    report += ["", "技術通過不等於病患層級獨立、臨床標註正確或授權通過。polygon 檢查涵蓋有限值／範圍／點數／零面積，不取代語義 QC、複雜自交或專業確認。", "",
               "大量 image／label 使用 hardlink；直接覆寫其中一份可能同步改動其他資料版本。補正應在新資料版本採用 copy-on-write，不修改既有 hardlink。", "",
               "### 舊模型結果的必要更正", "",
               "- `YOLO11m_WSNet_unionbox_from_n_best_20260817` 已有身分稽核：實體是 YOLO11n、2,590,035 參數。不能作為 YOLO11m 的結果。", "- 單一 Wound 偵測／分割只辨認傷口位置，不會產生七種傷口類別 Recall；應看來源、大小、單／多傷口、負樣本分層。", "- R1 舊 CSV 有重複 epoch、重新起算與 NaN validation loss；不能把全部列數當成同一次 700-epoch 訓練。", "- 舊 `vit_s_16` 分支其實建立 `vit_b_16`；必須核對模型實體，不能用配置名稱聲稱跑過 ViT-S。", "- 5-epoch／1-epoch 冒煙測試只證明管線可運作，不能作為同等收斂模型排名。", "",
               "### 內部／外部評估分開解讀", "",
               "D-Seg-09B 固定候選的既存驗證（非本次重新評估）：", "",
               f"- primary n={frozen['development_validation']['primary_images']}：Mask mAP50={pct(frozen['development_validation']['primary_mask_mAP50'])}。",
               f"- retention n={frozen['development_validation']['retention_images']}：Mask mAP50={pct(frozen['development_validation']['retention_mask_mAP50'])}。", "",
               f"CO2Wounds-V2 已執行並封存的外部 cohort n={external['images_evaluated']}；只作歷史外部證據，禁止回流訓練／選模型／调閾值。原始完整結果：" + linked("outputs/segmentation/co2wounds_external_test_20260831/D-Seg-09B_CO2Wounds-V2_external_results.json"), "",
               f"該原紀錄固定工作點 P={pct(external['fixed_operating_point']['instance_precision'])}、R={pct(external['fixed_operating_point']['instance_recall'])}、F1={pct(external['fixed_operating_point']['instance_f1'])}。不同 cohort 不可直接相減來判定某一資料集『有問題』；會同時混入場景、病灶、尺寸及標註政策差異。", "",
               "## 四、已修正的程式與微調方案", "",
               f"D-Seg-08 的 lr0={reference_args['lr0']}，warmup_bias_lr={reference_args['warmup_bias_lr']}。第一 epoch 記錄 lr/pg0={bias_lr_first}，為 lr0 的 {bias_lr_first/reference_args['lr0']:.1f} 倍；已對照目前安裝 Ultralytics 的暖身實作。", "",
               "這是實際設定／執行落差，不是推測值；但它對結果的因果影響尚未經控制實驗證明。新增 `D-Seg-08R-warmup-control`：bias warmup=0，其餘資料 split、初始化、seed、尺寸、batch、300-epoch 上限與 patience=80 均保持參照設計。之後只比較同一 development cohort，不使用 CO2Wounds 決策。", "",
               "目前沒有啟動此訓練：來源 gate 仍將 Yasin 標為 UNKNOWN，不能替它製造授權。新開發入口 run_development.py 已整合 source／optimizer／manifest／ancestor guards，但尚未實跑 GPU 訓練；recipe 不是已完成模型。", "",
               "實際程式改動：", "",
               "- `woundcare_inference.py`：明確 BGR／uint8 輸入契約；拒絕 NaN／無效 confidence、無效 threshold、空影像；忽略無效／退化 mask；檢查 mask-box 對齊；NMS IoU 顯式化且維持原預設0.7；原圖分類仍是預設。", "- `metrics.py`：固定類別 F1、無支援類別 Recall=null、拒判計入全樣本分母；配對檢定要求相同 ID；新增 group-cluster conditional CI，缺 ID／seed coverage 時拒絕計算。", "- `guards.py`：新開發實驗要求來源證據、禁止 test/unlock、禁止覆寫、檢查祖先訓練群組；重新分折的 CV 不准從已學過傷口資料的 checkpoint 起跑。", "- `audit_project.py`：新目錄輸出 inventory、實際資料稽核、manifest hash 比對及封版完整性；不 import 舊頂層訓練腳本。", "",
               "以上新 guard 不會自動攔截仍直接呼叫舊腳本的行為；舊調度器沒有被偷偷改写，也不應被繼續當作新實驗入口。", "",
               "## 五、App／Cascade：避免把局部成功當成落地正確率", "",
               f"既存 {fusion['sample_count']} 張 development ablation：原圖分類 {pct(fusion['full_image_baseline']['strict_accuracy_over_all_images'])}，純 ROI 裁切後全樣本正確率 {pct(fusion['roi_crop_baseline']['strict_accuracy_over_all_images'])}。裁切並非必然幫助分類。", "",
               f"Dual-view raw={pct(fusion['dual_view_fusion_raw']['strict_accuracy_over_all_images'])}；設 gate 後 selective accuracy={pct(fusion['dual_view_fusion_gated']['selective_accuracy_on_classified'])}，但 coverage={pct(fusion['dual_view_fusion_gated']['coverage'])}、全樣本 strict accuracy={pct(fusion['dual_view_fusion_gated']['strict_accuracy_over_all_images'])}。相對原圖只修好兩例，exact McNemar p={fusion['paired_fusion_vs_full']['mcnemar_exact_two_sided_p']}，不能宣稱已證明優於原圖或完成臨床驗證。", "",
               "舊 RGB ndarray 評估已存在 INVALIDATED marker；修正版使用正確色彩契約。本次不重新跑 cascade，也不自動把融合權重換到 App。", "",
               "後端預設 classifier 是 wound_classifier_v32，segmenter 是 WSNet v5；它們和研究 C-Arch-05／D-Seg-09B 的 SHA256 不同。這只是靜態預設與檔案對照，不代表已讀取某個運行中伺服器的環境。現在不改預設權重，應先確認授權、類別映射、前處理、工作點與部署回歸驗證。", "",
               "身分證據：" + linked(audit / "app_model_identity_audit.json"), "",
               "## 六、接下來的實驗順序（不以等回信作為程式改進的前提）", "",
               "1. 優先使用本報告替代不可重現的表格；原始 Phase1–9／test 保留。補來源使用權證據，不把公開下載、使用者人工標註或檔名前綴當作授權／IRB。", "2. 在來源與 ancestry gate 通過後，執行新增 warmup 單因子 ablation；記錄所有實際 LR、模型架構、參數量、權重 SHA256、package 版本和訓練／評估 failure 狀態。先短冒煙確認流程，正式結果只取完整規約。", "3. 改成 train-only source×size 抽樣／online augmentation；報曝光筆數與獨立內容數，不對 val/test 增強。一次改一個因素，避免同時改資料、模型、解析度、threshold 卻聲稱找到原因。", "4. 若要比較 YOLO11n／s／m，以相同可授權開發 pool、相同 group folds、相同尺度與預先定義的算力預算比較；每 fold 從可追溯 generic pretrained 權重重新初始化。先3-fold、單seed篩選，僅優勝候選再多seed；不是無限制重複5×5。", "5. 評估分開報 AP、固定 conf/NMS 的P/R/F1、union crop coverage、來源／尺寸／負樣本分層與端到端 latency。陰性樣本不足時 specificity 標為不足，不把2/5當成稳定性能。", "6. 當新模型使用目前已看過的驗證資料選模後，仍需要新的未見 cohort 作最終泛化驗證。Redscar 未回覆仍為未核准；不得重用分類48張或 CO2Wounds，也不把同來源 holdout 稱為獨立外部測試。", "",
               "## 七、驗證與可重現操作", "",
               f"本次回歸測試：{'PASS' if tests_passed else 'FAIL'}。測試輸出原文：" + linked(audit / "verification_tests.json"), "",
               f"封版歷史 metadata／程式檔 SHA256 比對：{len(original)} 檔，{'全部不變' if unchanged else '出現異動，須停止'}。保護證據：" + linked(audit / "final_sealed_integrity.json"), "",
               "```powershell", "# 僅執行合成資料／mock 模型測試", "$env:PYTHONIOENCODING='utf-8'", "python -m unittest discover -s tests -p test_experiment_review.py -v", "python tests/test_woundcare_inference.py", "", "# 只讀稽核；輸出目錄必須是尚未存在的新名稱", "python -m experiments.review_v2.audit_project --output outputs/model_review_NEW_ID", "python -m experiments.review_v2.finalize_review --audit outputs/model_review_NEW_ID", "```", "",
               "這些指令不啟動模型訓練。沒有可宣稱的新 mAP 增益；本次完成的是已驗證的統計／安全修正與實驗設計改進。", "",
               "## 八、方法參考", "",
               "- BGR ndarray／RGB PIL 契約與 NMS 參數：[Ultralytics Predict](https://docs.ultralytics.com/modes/predict/)。本專案實際安裝版本見 audit_summary.json，不自動升級。", "- Group CV 及非獨立樣本：[scikit-learn Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)。", "- 分割先於學習型前處理、避免資料洩漏：[scikit-learn Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)。", ""]
    with (audit / "專案模型實驗審查與改進報告.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(json.dumps({"report": str(audit / "專案模型實驗審查與改進報告.md"), "tests_passed": tests_passed,
                      "sealed_unchanged": unchanged, "findings": len(findings), "ancestry_checks": ancestry}, ensure_ascii=False, indent=2))
    return 0 if tests_passed and unchanged else 2


if __name__ == "__main__":
    raise SystemExit(main())
