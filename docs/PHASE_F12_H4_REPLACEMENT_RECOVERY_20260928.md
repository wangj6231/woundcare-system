# Phase F1.2 — Fresh H4 Replacement Recovery

更新時間（UTC）：2026-09-29T01:58:08.279538+00:00

PHASE_F12_STATUS = COMPLETE

## 不可變歷史與研究設計

The original higher-scale execution was interrupted before completing the fixed training budget and was excluded from performance comparison. A separately preregistered fresh higher-scale replacement execution was paired with the already completed, unevaluated frozen 768 control.

C4 and H4-R were not executed as one uninterrupted fresh temporal pair. This is an asymmetric replacement design and a single-seed recovery paired development experiment; no statistical significance, population confidence interval or clinical generalization is claimed.

正式採Option B；C4不重訓、不重選checkpoint，原H4不續訓、不評估。原F1固定預算pair維持無效；F1.1鑑識結果不變。

## 配方與前置條件

H4-R：YOLO11m-seg / Wound，seed42，train1024，batch4，300 epochs，patience80；ISIC原初始化，fresh optimizer/scaler。771張原始FUSeg training、965 polygon instances，191 development validation。所有loss、augmentation、AdamW、AMP、accumulation與16連續skip硬停均沿用封版規範。checkpoint選擇用SharedValidation768，預期actual shape遵循stock rect/pad，不把nominal768誤稱固定768×768。

通過300epochs/3741opportunities、unknown0、checkpoint/CSV/TensorBoard/telemetry完整性後，才建立recovery pair並依序做C4@768、H4-R@768。評估不寫入原C4目錄；全部放在新recovery pair/eval_C4與eval_H4_R。conf .10、floor .01、NMS .70、match .50、crop margin15%均不變。

原訓練/安全/validator/evaluator來源不修改；新的協調層只重綁檔案路徑與授權，研究語義沿用原凍結函數。preflight 的CPU小型合成測試不是新增research smoke，不載入傷口影像訓練；真實runtime binding會在第一批前再次檢查。

## 環境揭露

COMPATIBLE_WITH_DISCLOSED_DIFFERENCES。Original F1 receipts omit several platform fields. Current values recorded, historical equality UNKNOWN, not EXACT. Recorded critical versions and pinned training implementation hashes agree; no known semantic drift.

原F1缺OS/driver/OpenCV等完整欄位，不補造歷史值；新版記錄current值，已記錄的關鍵套件版本及active training source hashes均一致。

## 40項研究問題

1. Original F1狀態："INVALID_OR_INTERRUPTED"

2. Original H4始終未evaluation："YES; engineering evidence only"

3. C4 hash與F1.1一致："PASS"

4. C4 epochs：300

5. H4-R fresh initialization："ISIC original hash; resume=false; runtime_binding.json required"

6. 是否使用original H4 checkpoint："NO"

7. H4-R epochs：300

8. H4-R scheduled/applied/skipped/unknown：{"scheduled": 3741, "applied": 3732, "skipped": 9, "unknown": 0}

9. 最大連續skips：8

10. 最低scaler：128.0

11. Nonfinite loss：0

12. Nonfinite parameters：0

13. OOM：false

14. 第二次interruption：false

15. 300 epochs actual anchor parity："PASS"

16. SharedValidation768實際生效：{"C4": {"status": "PASS", "anchor_batches": 57900, "raw_loss_batches": 57900, "scheduled": 3741, "actual_anchor_order_matches_frozen": true, "validation": {"dataset_nominal_imgsz": 768, "first_batch_shape": [8, 3, 800, 800], "rule": "stock rect/pad/stride; nominal768 does not guarantee768x768 tensor", "shared_override": "experiments.f01_validator.SharedValidation768", "train_nominal_imgsz": 768, "val_nominal_imgsz": 768, "validator_args": {"agnostic_nms": false, "amp": true, "augment": false, "auto_augment": "randaugment", "batch": 4, "bgr": 0.0, "box": 7.5, "cache": false, "cfg": null, "classes": null, "close_mosaic": 30, "cls": 0.5, "conf": 0.001, "copy_paste": 0.0, "copy_paste_mode": "flip", "cos_lr": true, "crop_fraction": 1.0, "data": "C:\\Users\\milo9\\Desktop\\智慧型傷口分級與照護對應系統\\outputs\\isic_fuseg_pretrain_smoke_20260914\\fuseg_dataset\\dataset.yaml", "degrees": 5.0, "deterministic": true, "device": "0", "dfl": 1.5, "dnn": false, "dropout": 0.0, "dynamic": false, "embed": null, "epochs": 300, "erasing": 0.4, "exist_ok": false, "fliplr": 0.5, "flipud": 0.0, "format": "torchscript", "fraction": 1.0, "freeze": null, "half": true, "hsv_h": 0.01, "hsv_s": 0.4, "hsv_v": 0.25, "imgsz": 768, "int8": false, "iou": 0.7, "keras": false, "kobj": 1.0, "line_width": null, "lr0": 0.0005, "lrf": 0.01, "mask_ratio": 4, "max_det": 300, "mixup": 0.0, "mode": "train", "model": "C:\\Users\\milo9\\Desktop\\智慧型傷口分級與照護對應系統\\outputs\\isic_fuseg_formal_seed42_20260914\\runs\\isic_auxiliary_formal\\weights\\best.pt", "momentum": 0.937, "mosaic": 0.1, "multi_scale": false, "name": "training", "nbs": 64, "nms": false, "opset": null, "optimize": false, "optimizer": "AdamW", "overlap_mask": true, "patience": 80, "perspective": 0.0002, "plots": false, "pose": 12.0, "pretrained": true, "profile": false, "project": "C:\\Users\\milo9\\Desktop\\智慧型傷口分級與照護對應系統\\experiments\\results\\f_higher_scale_v2_seed42_control768", "rect": false, "resume": false, "retina_masks": false, "save": true, "save_conf": false, "save_crop": false, "save_frames": false, "save_hybrid": false, "save_json": false, "save_period": -1, "save_txt": false, "scale": 0.2, "seed": 42, "shear": 0.5, "show": false, "show_boxes": true, "show_conf": true, "show_labels": true, "simplify": true, "single_cls": false, "source": null, "split": "val", "stream_buffer": false, "task": "segment", "time": null, "tracker": "botsort.yaml", "translate": 0.05, "val": true, "verbose": false, "vid_stride": 1, "visualize": false, "warmup_bias_lr": 0.0, "warmup_epochs": 5, "warmup_momentum": 0.8, "weight_decay": 0.0005, "workers": 2, "workspace": null}}}, "H4-R": {"status": "PASS", "anchor_batches": 57900, "raw_loss_batches": 57900, "scheduled": 3741, "actual_anchor_order_matches_frozen": true, "validation": {"dataset_nominal_imgsz": 768, "first_batch_shape": [8, 3, 800, 800], "rule": "stock rect/pad/stride; nominal768 does not guarantee768x768 tensor", "shared_override": "experiments.f01_validator.SharedValidation768", "train_nominal_imgsz": 1024, "val_nominal_imgsz": 768, "validator_args": {"agnostic_nms": false, "amp": true, "augment": false, "auto_augment": "randaugment", "batch": 4, "bgr": 0.0, "box": 7.5, "cache": false, "cfg": null, "classes": null, "close_mosaic": 30, "cls": 0.5, "conf": 0.001, "copy_paste": 0.0, "copy_paste_mode": "flip", "cos_lr": true, "crop_fraction": 1.0, "data": "C:\\Users\\milo9\\Desktop\\智慧型傷口分級與照護對應系統\\outputs\\isic_fuseg_pretrain_smoke_20260914\\fuseg_dataset\\dataset.yaml", "degrees": 5.0, "deterministic": true, "device": "0", "dfl": 1.5, "dnn": false, "dropout": 0.0, "dynamic": false, "embed": null, "epochs": 300, "erasing": 0.4, "exist_ok": false, "fliplr": 0.5, "flipud": 0.0, "format": "torchscript", "fraction": 1.0, "freeze": null, "half": true, "hsv_h": 0.01, "hsv_s": 0.4, "hsv_v": 0.25, "imgsz": 768, "int8": false, "iou": 0.7, "keras": false, "kobj": 1.0, "line_width": null, "lr0": 0.0005, "lrf": 0.01, "mask_ratio": 4, "max_det": 300, "mixup": 0.0, "mode": "train", "model": "C:\\Users\\milo9\\Desktop\\智慧型傷口分級與照護對應系統\\outputs\\isic_fuseg_formal_seed42_20260914\\runs\\isic_auxiliary_formal\\weights\\best.pt", "momentum": 0.937, "mosaic": 0.1, "multi_scale": false, "name": "training", "nbs": 64, "nms": false, "opset": null, "optimize": false, "optimizer": "AdamW", "overlap_mask": true, "patience": 80, "perspective": 0.0002, "plots": false, "pose": 12.0, "pretrained": true, "profile": false, "project": "C:\\Users\\milo9\\Desktop\\智慧型傷口分級與照護對應系統\\experiments\\results\\f_higher_scale_v2_recovery_seed42_h4_1024", "rect": false, "resume": false, "retina_masks": false, "save": true, "save_conf": false, "save_crop": false, "save_frames": false, "save_hybrid": false, "save_json": false, "save_period": -1, "save_txt": false, "scale": 0.2, "seed": 42, "shear": 0.5, "show": false, "show_boxes": true, "show_conf": true, "show_labels": true, "simplify": true, "single_cls": false, "source": null, "split": "val", "stream_buffer": false, "task": "segment", "time": null, "tracker": "botsort.yaml", "translate": 0.05, "val": true, "verbose": false, "vid_stride": 1, "visualize": false, "warmup_bias_lr": 0.0, "warmup_epochs": 5, "warmup_momentum": 0.8, "weight_decay": 0.0005, "workers": 2, "workspace": null}}}}

17. H4-R best/last完整："PASS"

18. Environment parity："COMPATIBLE_WITH_DISCLOSED_DIFFERENCES"

19. Recovery pair有效："YES"

20. C4 eval只在pair valid之後："RECOVERY_PAIR_VALID -> C4@768 -> H4-R@768 -> gate"

21. H4-R eval只在pair valid之後："RECOVERY_PAIR_VALID -> C4@768 -> H4-R@768 -> gate"

22. C4 Very-small TP/49：{"TP": 26, "recall": 0.5306122448979592, "support": 49}

23. H4-R Very-small TP/49：{"TP": 25, "recall": 0.5102040816326531, "support": 49}

24. Very-small delta TP：-1

25. <0.10%兩組：{"C4": {"TP": 10, "recall": 0.37037037037037035, "support": 27}, "H4-R": {"TP": 9, "recall": 0.3333333333333333, "support": 27}}

26. Small safety：false

27. Precision safety：false

28. F1 safety：true

29. Medium safety：true

30. Large safety：true

31. Crop safety：true

32. Recovery research gate："FAIL"

33. Historical development gate：{"C4": "FAIL", "H4-R": "FAIL"}

34. H4-R@1024 final eval："NO"

35. Locked test："NO"

36. CO2/external："NO"

37. Multi-seed："NO"

38. App replacement："NO"

39. 原H4用途："INTERRUPTED_ENGINEERING_EVIDENCE; no performance endpoints"

40. 下一階段建議："保留負結果，未通過項目：very_small, small, precision。下一步應先利用本次已保存prediction/matching做配對錯誤分析，區分very-small新增TP是否伴隨其他尺寸或crop退化；不重推論、不改threshold、不試960或H4@1024。待分析確認失敗結構後再預登錄單一介入；本次不自動執行。"

## 已完成配對數據

```json
{
  "PHASE_F12_STATUS": "COMPLETE",
  "RECOVERY_OPTION": "B",
  "RECOVERY_PAIR_VALID": "YES",
  "C4_SOURCE": "ORIGINAL_COMPLETED_F1_CONTROL",
  "H4_SOURCE": "FRESH_REPLACEMENT_EXECUTION",
  "ORIGINAL_H4_PERFORMANCE_USED": "NO",
  "RECOVERY_RESEARCH_GATE": "FAIL",
  "MULTI_SEED_AUTHORIZED": "NO",
  "APP_MODEL_REPLACEMENT_AUTHORIZED": "NO",
  "EXTERNAL_TEST_AUTHORIZED": "NO",
  "FINAL_OPERATIONAL_EVALUATION_PERFORMED": true,
  "test_images_used": 0,
  "LOCKED_TEST_USED": false,
  "CO2Wounds_used": false,
  "EXTERNAL_TEST_USED": false,
  "H4_R_final_eval1024": false,
  "training": {
    "C4": {
      "CO2Wounds_used": false,
      "OOM": false,
      "anchors_consumed": 231300,
      "applied": 3731,
      "best_checkpoint_sha256": "62b0d871b583f029d649fd905872cfdfc9ec920f1c0f7c76171eda130ff26b60",
      "best_epoch": 281,
      "completed_epochs": 300,
      "consecutive_skipped": 0,
      "failure": null,
      "last_checkpoint_sha256": "b5028d2b53063d7528705d33a7471ab9f1ced214d00f75764483460ee9e3f67f",
      "last_epoch": 300,
      "max_consecutive_skips": 8,
      "minimum_scaler": 128.0,
      "nonfinite_loss_events": 0,
      "nonfinite_parameter_events": 0,
      "peak_GPU_allocated": 3977489408,
      "peak_GPU_reserved": 4989124608,
      "raw_batches": 57900,
      "runtime_classification": null,
      "runtime_exceptions": 0,
      "safety_adapter_sha256": "7f87e8edfdfed69b58b206ff1cd23731673d89d3625f314a2d7cc421d65bc43e",
      "scheduled": 3741,
      "skipped": 10,
      "telemetry_complete": true,
      "test_images_used": 0,
      "train_first_batch_shape": [
        4,
        3,
        768,
        768
      ],
      "train_nominal_imgsz": 768,
      "unknown": 0,
      "val_nominal_imgsz": 768
    },
    "H4-R": {
      "CO2Wounds_used": false,
      "OOM": false,
      "anchors_consumed": 231300,
      "applied": 3732,
      "best_checkpoint_sha256": "15bb26b7cef2b18282318b5019bd294ad0bb1dfdb978a1cc58b08c8f1400f28e",
      "best_epoch": 256,
      "completed_epochs": 300,
      "consecutive_skipped": 0,
      "failure": null,
      "last_checkpoint_sha256": "40a4c6a62d937f8d8753be5b1bf404357b3ce6ca78885be797abbbfb71af54ea",
      "last_epoch": 300,
      "max_consecutive_skips": 8,
      "minimum_scaler": 128.0,
      "nonfinite_loss_events": 0,
      "nonfinite_parameter_events": 0,
      "peak_GPU_allocated": 6446971392,
      "peak_GPU_reserved": 7065305088,
      "raw_batches": 57900,
      "runtime_classification": null,
      "runtime_exceptions": 0,
      "safety_adapter_sha256": "7f87e8edfdfed69b58b206ff1cd23731673d89d3625f314a2d7cc421d65bc43e",
      "scheduled": 3741,
      "skipped": 9,
      "telemetry_complete": true,
      "test_images_used": 0,
      "train_first_batch_shape": [
        4,
        3,
        1024,
        1024
      ],
      "train_nominal_imgsz": 1024,
      "unknown": 0,
      "val_nominal_imgsz": 768
    }
  },
  "evaluations": {
    "C4": {
      "CO2Wounds_used": false,
      "candidate": {
        "crop_area_positive_mean": 0.050501669606854836,
        "crop_complete95_fraction": 0.9086021505376344,
        "crop_complete95_images": 169,
        "crop_coverage_positive_mean": 0.9663889626584785,
        "f1": 0.8565400843881856,
        "fn": 38,
        "fp": 30,
        "images": 191,
        "mask_dice_positive_mean": 0.8546518737406277,
        "mask_iou_positive_mean": 0.777509417836296,
        "negative_images": 5,
        "negative_images_with_predictions": 2,
        "positive_images": 186,
        "positive_without_roi": 2,
        "precision": 0.871244635193133,
        "recall": 0.8423236514522822,
        "size_recall": {
          "large": {
            "gt": 14,
            "matched": 14,
            "recall": 1.0
          },
          "medium": {
            "gt": 90,
            "matched": 85,
            "recall": 0.9444444444444444
          },
          "small": {
            "gt": 137,
            "matched": 104,
            "recall": 0.7591240875912408
          }
        },
        "tp": 203
      },
      "checkpoint_sha256": "62b0d871b583f029d649fd905872cfdfc9ec920f1c0f7c76171eda130ff26b60",
      "diagnostics": {
        "multi_GT": {
          "FN": 25,
          "FP": 7,
          "FP_per_image": 0.2,
          "GT": 90,
          "TP": 65,
          "crop_complete_rate": 0.6285714285714286,
          "crop_fail": 13,
          "crop_pass": 22,
          "image_any_miss_rate": 0.5428571428571428,
          "image_count": 35,
          "images_with_any_miss": 19,
          "instance_recall": 0.7222222222222222,
          "mean_retained_fraction": 0.8970507359798918
        },
        "single_GT": {
          "FN": 13,
          "FP": 21,
          "FP_per_image": 0.1390728476821192,
          "GT": 151,
          "TP": 138,
          "crop_complete_rate": 0.9735099337748344,
          "crop_fail": 4,
          "crop_pass": 147,
          "image_any_miss_rate": 0.08609271523178808,
          "image_count": 151,
          "images_with_any_miss": 13,
          "instance_recall": 0.9139072847682119,
          "mean_retained_fraction": 0.9824607370541775
        },
        "small_bins": {
          "0.10-0.25%": {
            "TP": 16,
            "recall": 0.7272727272727273,
            "support": 22
          },
          "0.25-0.50%": {
            "TP": 33,
            "recall": 0.868421052631579,
            "support": 38
          },
          "0.50-0.75%": {
            "TP": 28,
            "recall": 0.9333333333333333,
            "support": 30
          },
          "0.75-1.00%": {
            "TP": 17,
            "recall": 0.85,
            "support": 20
          },
          "<0.10%": {
            "TP": 10,
            "recall": 0.37037037037037035,
            "support": 27
          }
        },
        "very_small": {
          "TP": 26,
          "recall": 0.5306122448979592,
          "support": 49
        }
      },
      "final_operational_imgsz": 768,
      "historical_development_gate": "FAIL",
      "integrity": {
        "metrics_recomputed_from_saved_counts": true,
        "num_gt_instances": 241,
        "num_images": 191,
        "size_support_equals_total_gt": true,
        "status": "PASS"
      },
      "latency": {
        "mean_ms": 33.26189633514013,
        "median_ms": 31.049200006236788,
        "n": 191,
        "p95_ms": 39.30495000167866,
        "serial_fps_from_mean": 30.064431381909277
      },
      "latency_comparability": "NOT_VERIFIED; descriptive only",
      "performance_checks": {
        "crop_complete95_fraction": true,
        "f1": false,
        "precision": false,
        "recall": false
      },
      "predictions_sha256": "2ee2cc68c9e9413ca8446fe27f1ad8e794b655a3486ba1fa7db0e607ac39f683",
      "test_images_used": 0
    },
    "H4-R": {
      "CO2Wounds_used": false,
      "candidate": {
        "crop_area_positive_mean": 0.05752719345913138,
        "crop_complete95_fraction": 0.9032258064516129,
        "crop_complete95_images": 168,
        "crop_coverage_positive_mean": 0.9639446958498171,
        "f1": 0.8475991649269311,
        "fn": 38,
        "fp": 35,
        "images": 191,
        "mask_dice_positive_mean": 0.8495841070178649,
        "mask_iou_positive_mean": 0.7705472456973158,
        "negative_images": 5,
        "negative_images_with_predictions": 2,
        "positive_images": 186,
        "positive_without_roi": 3,
        "precision": 0.8529411764705882,
        "recall": 0.8423236514522822,
        "size_recall": {
          "large": {
            "gt": 14,
            "matched": 14,
            "recall": 1.0
          },
          "medium": {
            "gt": 90,
            "matched": 87,
            "recall": 0.9666666666666667
          },
          "small": {
            "gt": 137,
            "matched": 102,
            "recall": 0.7445255474452555
          }
        },
        "tp": 203
      },
      "checkpoint_sha256": "15bb26b7cef2b18282318b5019bd294ad0bb1dfdb978a1cc58b08c8f1400f28e",
      "diagnostics": {
        "multi_GT": {
          "FN": 24,
          "FP": 11,
          "FP_per_image": 0.3142857142857143,
          "GT": 90,
          "TP": 66,
          "crop_complete_rate": 0.6857142857142857,
          "crop_fail": 11,
          "crop_pass": 24,
          "image_any_miss_rate": 0.5428571428571428,
          "image_count": 35,
          "images_with_any_miss": 19,
          "instance_recall": 0.7333333333333333,
          "mean_retained_fraction": 0.9227999409666924
        },
        "single_GT": {
          "FN": 14,
          "FP": 22,
          "FP_per_image": 0.1456953642384106,
          "GT": 151,
          "TP": 137,
          "crop_complete_rate": 0.9536423841059603,
          "crop_fail": 7,
          "crop_pass": 144,
          "image_any_miss_rate": 0.09271523178807947,
          "image_count": 151,
          "images_with_any_miss": 14,
          "instance_recall": 0.9072847682119205,
          "mean_retained_fraction": 0.9734815595644488
        },
        "small_bins": {
          "0.10-0.25%": {
            "TP": 16,
            "recall": 0.7272727272727273,
            "support": 22
          },
          "0.25-0.50%": {
            "TP": 35,
            "recall": 0.9210526315789473,
            "support": 38
          },
          "0.50-0.75%": {
            "TP": 24,
            "recall": 0.8,
            "support": 30
          },
          "0.75-1.00%": {
            "TP": 18,
            "recall": 0.9,
            "support": 20
          },
          "<0.10%": {
            "TP": 9,
            "recall": 0.3333333333333333,
            "support": 27
          }
        },
        "very_small": {
          "TP": 25,
          "recall": 0.5102040816326531,
          "support": 49
        }
      },
      "final_operational_imgsz": 768,
      "historical_development_gate": "FAIL",
      "integrity": {
        "metrics_recomputed_from_saved_counts": true,
        "num_gt_instances": 241,
        "num_images": 191,
        "size_support_equals_total_gt": true,
        "status": "PASS"
      },
      "latency": {
        "mean_ms": 33.05612303665839,
        "median_ms": 30.3524999981164,
        "n": 191,
        "p95_ms": 40.816949996951735,
        "serial_fps_from_mean": 30.251581496445475
      },
      "latency_comparability": "NOT_VERIFIED; descriptive only",
      "performance_checks": {
        "crop_complete95_fraction": true,
        "f1": false,
        "precision": false,
        "recall": false
      },
      "predictions_sha256": "af47361faf4e4c77b35dab342db0fc6a06138d49edc9edc895f012b77a9305e0",
      "test_images_used": 0
    }
  },
  "gate": {
    "status": "FAIL",
    "checks": {
      "very_small": false,
      "small": false,
      "precision": false,
      "f1": true,
      "medium": true,
      "large": true,
      "crop": true
    },
    "delta_very_small_TP": -1,
    "meaning": "single-seed paired operational advancement criterion, not statistical significance"
  },
  "saved_integrity": {
    "C4": {
      "status": "PASS",
      "num_images": 191,
      "num_gt_instances": 241,
      "metrics_recomputed_from_saved_counts": true,
      "size_support_equals_total_gt": true
    },
    "H4-R": {
      "status": "PASS",
      "num_images": 191,
      "num_gt_instances": 241,
      "metrics_recomputed_from_saved_counts": true,
      "size_support_equals_total_gt": true
    }
  },
  "historical_integrity": "PASS",
  "shared_validation": {
    "C4": {
      "status": "PASS",
      "anchor_batches": 57900,
      "raw_loss_batches": 57900,
      "scheduled": 3741,
      "actual_anchor_order_matches_frozen": true,
      "validation": {
        "dataset_nominal_imgsz": 768,
        "first_batch_shape": [
          8,
          3,
          800,
          800
        ],
        "rule": "stock rect/pad/stride; nominal768 does not guarantee768x768 tensor",
        "shared_override": "experiments.f01_validator.SharedValidation768",
        "train_nominal_imgsz": 768,
        "val_nominal_imgsz": 768,
        "validator_args": {
          "agnostic_nms": false,
          "amp": true,
          "augment": false,
          "auto_augment": "randaugment",
          "batch": 4,
          "bgr": 0.0,
          "box": 7.5,
          "cache": false,
          "cfg": null,
          "classes": null,
          "close_mosaic": 30,
          "cls": 0.5,
          "conf": 0.001,
          "copy_paste": 0.0,
          "copy_paste_mode": "flip",
          "cos_lr": true,
          "crop_fraction": 1.0,
          "data": "C:\\Users\\milo9\\Desktop\\智慧型傷口分級與照護對應系統\\outputs\\isic_fuseg_pretrain_smoke_20260914\\fuseg_dataset\\dataset.yaml",
          "degrees": 5.0,
          "deterministic": true,
          "device": "0",
          "dfl": 1.5,
          "dnn": false,
          "dropout": 0.0,
          "dynamic": false,
          "embed": null,
          "epochs": 300,
          "erasing": 0.4,
          "exist_ok": false,
          "fliplr": 0.5,
          "flipud": 0.0,
          "format": "torchscript",
          "fraction": 1.0,
          "freeze": null,
          "half": true,
          "hsv_h": 0.01,
          "hsv_s": 0.4,
          "hsv_v": 0.25,
          "imgsz": 768,
          "int8": false,
          "iou": 0.7,
          "keras": false,
          "kobj": 1.0,
          "line_width": null,
          "lr0": 0.0005,
          "lrf": 0.01,
          "mask_ratio": 4,
          "max_det": 300,
          "mixup": 0.0,
          "mode": "train",
          "model": "C:\\Users\\milo9\\Desktop\\智慧型傷口分級與照護對應系統\\outputs\\isic_fuseg_formal_seed42_20260914\\runs\\isic_auxiliary_formal\\weights\\best.pt",
          "momentum": 0.937,
          "mosaic": 0.1,
          "multi_scale": false,
          "name": "training",
          "nbs": 64,
          "nms": false,
          "opset": null,
          "optimize": false,
          "optimizer": "AdamW",
          "overlap_mask": true,
          "patience": 80,
          "perspective": 0.0002,
          "plots": false,
          "pose": 12.0,
          "pretrained": true,
          "profile": false,
          "project": "C:\\Users\\milo9\\Desktop\\智慧型傷口分級與照護對應系統\\experiments\\results\\f_higher_scale_v2_seed42_control768",
          "rect": false,
          "resume": false,
          "retina_masks": false,
          "save": true,
          "save_conf": false,
          "save_crop": false,
          "save_frames": false,
          "save_hybrid": false,
          "save_json": false,
          "save_period": -1,
          "save_txt": false,
          "scale": 0.2,
          "seed": 42,
          "shear": 0.5,
          "show": false,
          "show_boxes": true,
          "show_conf": true,
          "show_labels": true,
          "simplify": true,
          "single_cls": false,
          "source": null,
          "split": "val",
          "stream_buffer": false,
          "task": "segment",
          "time": null,
          "tracker": "botsort.yaml",
          "translate": 0.05,
          "val": true,
          "verbose": false,
          "vid_stride": 1,
          "visualize": false,
          "warmup_bias_lr": 0.0,
          "warmup_epochs": 5,
          "warmup_momentum": 0.8,
          "weight_decay": 0.0005,
          "workers": 2,
          "workspace": null
        }
      }
    },
    "H4-R": {
      "status": "PASS",
      "anchor_batches": 57900,
      "raw_loss_batches": 57900,
      "scheduled": 3741,
      "actual_anchor_order_matches_frozen": true,
      "validation": {
        "dataset_nominal_imgsz": 768,
        "first_batch_shape": [
          8,
          3,
          800,
          800
        ],
        "rule": "stock rect/pad/stride; nominal768 does not guarantee768x768 tensor",
        "shared_override": "experiments.f01_validator.SharedValidation768",
        "train_nominal_imgsz": 1024,
        "val_nominal_imgsz": 768,
        "validator_args": {
          "agnostic_nms": false,
          "amp": true,
          "augment": false,
          "auto_augment": "randaugment",
          "batch": 4,
          "bgr": 0.0,
          "box": 7.5,
          "cache": false,
          "cfg": null,
          "classes": null,
          "close_mosaic": 30,
          "cls": 0.5,
          "conf": 0.001,
          "copy_paste": 0.0,
          "copy_paste_mode": "flip",
          "cos_lr": true,
          "crop_fraction": 1.0,
          "data": "C:\\Users\\milo9\\Desktop\\智慧型傷口分級與照護對應系統\\outputs\\isic_fuseg_pretrain_smoke_20260914\\fuseg_dataset\\dataset.yaml",
          "degrees": 5.0,
          "deterministic": true,
          "device": "0",
          "dfl": 1.5,
          "dnn": false,
          "dropout": 0.0,
          "dynamic": false,
          "embed": null,
          "epochs": 300,
          "erasing": 0.4,
          "exist_ok": false,
          "fliplr": 0.5,
          "flipud": 0.0,
          "format": "torchscript",
          "fraction": 1.0,
          "freeze": null,
          "half": true,
          "hsv_h": 0.01,
          "hsv_s": 0.4,
          "hsv_v": 0.25,
          "imgsz": 768,
          "int8": false,
          "iou": 0.7,
          "keras": false,
          "kobj": 1.0,
          "line_width": null,
          "lr0": 0.0005,
          "lrf": 0.01,
          "mask_ratio": 4,
          "max_det": 300,
          "mixup": 0.0,
          "mode": "train",
          "model": "C:\\Users\\milo9\\Desktop\\智慧型傷口分級與照護對應系統\\outputs\\isic_fuseg_formal_seed42_20260914\\runs\\isic_auxiliary_formal\\weights\\best.pt",
          "momentum": 0.937,
          "mosaic": 0.1,
          "multi_scale": false,
          "name": "training",
          "nbs": 64,
          "nms": false,
          "opset": null,
          "optimize": false,
          "optimizer": "AdamW",
          "overlap_mask": true,
          "patience": 80,
          "perspective": 0.0002,
          "plots": false,
          "pose": 12.0,
          "pretrained": true,
          "profile": false,
          "project": "C:\\Users\\milo9\\Desktop\\智慧型傷口分級與照護對應系統\\experiments\\results\\f_higher_scale_v2_recovery_seed42_h4_1024",
          "rect": false,
          "resume": false,
          "retina_masks": false,
          "save": true,
          "save_conf": false,
          "save_crop": false,
          "save_frames": false,
          "save_hybrid": false,
          "save_json": false,
          "save_period": -1,
          "save_txt": false,
          "scale": 0.2,
          "seed": 42,
          "shear": 0.5,
          "show": false,
          "show_boxes": true,
          "show_conf": true,
          "show_labels": true,
          "simplify": true,
          "single_cls": false,
          "source": null,
          "split": "val",
          "stream_buffer": false,
          "task": "segment",
          "time": null,
          "tracker": "botsort.yaml",
          "translate": 0.05,
          "val": true,
          "verbose": false,
          "vid_stride": 1,
          "visualize": false,
          "warmup_bias_lr": 0.0,
          "warmup_epochs": 5,
          "warmup_momentum": 0.8,
          "weight_decay": 0.0005,
          "workers": 2,
          "workspace": null
        }
      }
    }
  },
  "interruption": false,
  "evaluation_sequence": "RECOVERY_PAIR_VALID -> C4@768 -> H4-R@768 -> gate",
  "next_view": "保留負結果，未通過項目：very_small, small, precision。下一步應先利用本次已保存prediction/matching做配對錯誤分析，區分very-small新增TP是否伴隨其他尺寸或crop退化；不重推論、不改threshold、不試960或H4@1024。待分析確認失敗結構後再預登錄單一介入；本次不自動執行。",
  "STOP_AFTER_F12": true,
  "at": "2026-09-29T01:58:06.329413+00:00"
}
```


STOP AFTER F1.2；無H4-R2、resume、multi-seed、external test、App替換或自動下一實驗。
