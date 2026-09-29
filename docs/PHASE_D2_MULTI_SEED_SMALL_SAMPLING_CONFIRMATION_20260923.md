# Phase D2 多種子配對抽樣確認

研究單位：5 paired random-seed experiments。未重新訓練或評估 seed42。

狀態：COMPLETE；Confirmation：FAIL；有效 pairs：5/5。

| Seed | C Small TP | S Small TP | Δ Small Recall (pp) | Δ Precision (pp) | Δ F1 (pp) | Δ Crop | Pair gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 104 | 108 | 2.9197 | 1.3371 | 1.5039 | 2 | PASS |
| 123 | 105 | 109 | 2.9197 | 3.8633 | 2.7477 | -5 | FAIL |
| 3407 | 107 | 103 | -2.9197 | -2.4410 | -2.0408 | 1 | FAIL |
| 2026 | 104 | 106 | 1.4599 | 2.0596 | 1.8561 | 3 | PASS |
| 999 | 108 | 108 | 0.0000 | -0.6931 | -0.3523 | 1 | FAIL |

## 平均 paired effects 與安全門檻

```json
{
  "MULTISEED_SMALL_SAMPLING_CONFIRMATION": "FAIL",
  "VALID_PAIRED_SEEDS": 5,
  "required_pairs": 5,
  "MULTISEED_CONFIRMATION_INCOMPLETE": false,
  "checks": {
    "direction_4_of_5": false,
    "mean_small_positive": true,
    "mean_precision_safety": true,
    "mean_f1_safety": true,
    "no_catastrophic_large": true,
    "all_5_pairs_valid_and_delivered": true
  },
  "small_paired_effect": {
    "mean": 0.008759124087591242,
    "sample_SD": 0.024428029971797827,
    "median": 0.014598540145985401,
    "min": -0.029197080291970802,
    "max": 0.029197080291970802,
    "positive": 3,
    "zero": 1,
    "negative": 1,
    "n_paired_seeds": 5
  },
  "mean_paired_effects": {
    "small_recall": 0.008759124087591242,
    "precision": 0.008252060765379193,
    "recall": 0.006639004149377593,
    "f1": 0.007429112462949164,
    "crop_count": 0.4,
    "very_small_TP": -0.6,
    "FP": -2.0,
    "FN": -1.6,
    "no_ROI": -0.4,
    "small_TP": 1.2,
    "medium_TP": 0.4,
    "large_TP": 0.0,
    "single_GT_recall": 0.007947019867549669,
    "multi_GT_recall": 0.0044444444444444444
  },
  "catastrophic_large_seeds": [],
  "ordinary_large_minus_one_seeds": [],
  "descriptive_summary_scope": "valid pairs only; never relabel incomplete as 5-seed confirmation",
  "exact_mean_paired_effects": {
    "small_recall": "6/685",
    "precision": "742594800278293/89989012610496675",
    "recall": "8/1205",
    "f1": "215282445176434549/28978218629762542875",
    "crop_count": "2/5",
    "very_small_TP": "-3/5",
    "FP": "-2",
    "FN": "-8/5",
    "no_ROI": "-2/5",
    "small_TP": "6/5",
    "medium_TP": "2/5",
    "large_TP": "0",
    "single_GT_recall": "6/755",
    "multi_GT_recall": "1/225"
  },
  "PHASE_D2_STATUS": "COMPLETE",
  "MULTI_SEED_TRAINING_COMPLETE": true,
  "DIRECTIONAL_POSITIVE_SEEDS": 3,
  "MEAN_PAIRED_SMALL_RECALL_DELTA": 0.008759124087591242,
  "AMP_UPDATE_IMBALANCE_SEEDS": [
    123,
    2026
  ],
  "equal_realized_update_pairs": 3,
  "small_bin_mean_paired_effects": {
    "0.10-0.25%": {
      "mean_delta_TP": -0.8,
      "mean_delta_recall": -0.03636363636363636
    },
    "0.25-0.50%": {
      "mean_delta_TP": 1.2,
      "mean_delta_recall": 0.031578947368421054
    },
    "0.50-0.75%": {
      "mean_delta_TP": 0.4,
      "mean_delta_recall": 0.013333333333333332
    },
    "0.75-1.00%": {
      "mean_delta_TP": 0.2,
      "mean_delta_recall": 0.01
    },
    "<0.10%": {
      "mean_delta_TP": 0.2,
      "mean_delta_recall": 0.007407407407407407
    }
  },
  "aborted_reason": null,
  "historical_development_gate_counts": {
    "C": {
      "PASS": 0,
      "FAIL": 5,
      "PARTIAL": 0
    },
    "S": {
      "PASS": 0,
      "FAIL": 4,
      "PARTIAL": 1
    }
  },
  "APP_MODEL_REPLACEMENT_AUTHORIZED": "NO",
  "EXTERNAL_TEST_AUTHORIZED": "NO",
  "LOCKED_TEST_USED": false,
  "CO2Wounds_used": false,
  "test_images_used": 0,
  "STOP_AFTER_D2": true
}
```

## 各 seed 五個 small bins（含未改善結果）

| Seed | Bin | C TP | S TP | Δ TP | Support |
|---:|---|---:|---:|---:|---:|
| 42 | 0.10-0.25% | 16 | 16 | 0 | 22 |
| 42 | 0.25-0.50% | 31 | 34 | 3 | 38 |
| 42 | 0.50-0.75% | 28 | 28 | 0 | 30 |
| 42 | 0.75-1.00% | 18 | 19 | 1 | 20 |
| 42 | <0.10% | 11 | 11 | 0 | 27 |
| 123 | <0.10% | 11 | 13 | 2 | 27 |
| 123 | 0.10-0.25% | 16 | 15 | -1 | 22 |
| 123 | 0.25-0.50% | 33 | 34 | 1 | 38 |
| 123 | 0.50-0.75% | 28 | 28 | 0 | 30 |
| 123 | 0.75-1.00% | 17 | 19 | 2 | 20 |
| 3407 | <0.10% | 12 | 9 | -3 | 27 |
| 3407 | 0.10-0.25% | 16 | 15 | -1 | 22 |
| 3407 | 0.25-0.50% | 32 | 33 | 1 | 38 |
| 3407 | 0.50-0.75% | 28 | 28 | 0 | 30 |
| 3407 | 0.75-1.00% | 19 | 18 | -1 | 20 |
| 2026 | <0.10% | 10 | 13 | 3 | 27 |
| 2026 | 0.10-0.25% | 16 | 15 | -1 | 22 |
| 2026 | 0.25-0.50% | 32 | 33 | 1 | 38 |
| 2026 | 0.50-0.75% | 28 | 28 | 0 | 30 |
| 2026 | 0.75-1.00% | 18 | 17 | -1 | 20 |
| 999 | <0.10% | 12 | 11 | -1 | 27 |
| 999 | 0.10-0.25% | 17 | 16 | -1 | 22 |
| 999 | 0.25-0.50% | 34 | 34 | 0 | 38 |
| 999 | 0.50-0.75% | 27 | 29 | 2 | 30 |
| 999 | 0.75-1.00% | 18 | 18 | 0 | 20 |

## AMP 與 multi-GT 詳表

```json
{
  "AMP": [
    {
      "seed": 42,
      "C_completed_epochs": 300,
      "C_scheduled_optimizer_calls": 3741,
      "C_applied_optimizer_updates": 3731,
      "C_skipped_optimizer_updates": 10,
      "S_completed_epochs": 300,
      "S_scheduled_optimizer_calls": 3741,
      "S_applied_optimizer_updates": 3731,
      "S_skipped_optimizer_updates": 10,
      "imbalance": false
    },
    {
      "seed": 123,
      "C_completed_epochs": 300,
      "C_scheduled_optimizer_calls": 3741,
      "C_applied_optimizer_updates": 3731,
      "C_skipped_optimizer_updates": 10,
      "S_completed_epochs": 300,
      "S_scheduled_optimizer_calls": 3741,
      "S_applied_optimizer_updates": 3730,
      "S_skipped_optimizer_updates": 11,
      "imbalance": true
    },
    {
      "seed": 3407,
      "C_completed_epochs": 300,
      "C_scheduled_optimizer_calls": 3741,
      "C_applied_optimizer_updates": 3732,
      "C_skipped_optimizer_updates": 9,
      "S_completed_epochs": 300,
      "S_scheduled_optimizer_calls": 3741,
      "S_applied_optimizer_updates": 3732,
      "S_skipped_optimizer_updates": 9,
      "imbalance": false
    },
    {
      "seed": 2026,
      "C_completed_epochs": 300,
      "C_scheduled_optimizer_calls": 3741,
      "C_applied_optimizer_updates": 3731,
      "C_skipped_optimizer_updates": 10,
      "S_completed_epochs": 300,
      "S_scheduled_optimizer_calls": 3741,
      "S_applied_optimizer_updates": 3732,
      "S_skipped_optimizer_updates": 9,
      "imbalance": true
    },
    {
      "seed": 999,
      "C_completed_epochs": 300,
      "C_scheduled_optimizer_calls": 3741,
      "C_applied_optimizer_updates": 3731,
      "C_skipped_optimizer_updates": 10,
      "S_completed_epochs": 300,
      "S_scheduled_optimizer_calls": 3741,
      "S_applied_optimizer_updates": 3731,
      "S_skipped_optimizer_updates": 10,
      "imbalance": false
    }
  ],
  "single_multi_GT": [
    {
      "seed": 42,
      "group": "single_GT",
      "C_recall": 0.9205298013245033,
      "S_recall": 0.9337748344370861,
      "delta_recall": 0.013245033112582781,
      "C_crop_complete": 145,
      "S_crop_complete": 145,
      "delta_crop_count": 0
    },
    {
      "seed": 42,
      "group": "multi_GT",
      "C_recall": 0.7111111111111111,
      "S_recall": 0.7333333333333333,
      "delta_recall": 0.022222222222222223,
      "C_crop_complete": 20,
      "S_crop_complete": 22,
      "delta_crop_count": 2
    },
    {
      "seed": 123,
      "group": "single_GT",
      "C_recall": 0.9072847682119205,
      "S_recall": 0.9337748344370861,
      "delta_recall": 0.026490066225165563,
      "C_crop_complete": 144,
      "S_crop_complete": 144,
      "delta_crop_count": 0
    },
    {
      "seed": 123,
      "group": "multi_GT",
      "C_recall": 0.7444444444444445,
      "S_recall": 0.7444444444444445,
      "delta_recall": 0.0,
      "C_crop_complete": 26,
      "S_crop_complete": 21,
      "delta_crop_count": -5
    },
    {
      "seed": 3407,
      "group": "single_GT",
      "C_recall": 0.9205298013245033,
      "S_recall": 0.9271523178807947,
      "delta_recall": 0.006622516556291391,
      "C_crop_complete": 143,
      "S_crop_complete": 146,
      "delta_crop_count": 3
    },
    {
      "seed": 3407,
      "group": "multi_GT",
      "C_recall": 0.7444444444444445,
      "S_recall": 0.6888888888888889,
      "delta_recall": -0.05555555555555555,
      "C_crop_complete": 23,
      "S_crop_complete": 21,
      "delta_crop_count": -2
    },
    {
      "seed": 2026,
      "group": "single_GT",
      "C_recall": 0.9271523178807947,
      "S_recall": 0.9271523178807947,
      "delta_recall": 0.0,
      "C_crop_complete": 146,
      "S_crop_complete": 145,
      "delta_crop_count": -1
    },
    {
      "seed": 2026,
      "group": "multi_GT",
      "C_recall": 0.7111111111111111,
      "S_recall": 0.7555555555555555,
      "delta_recall": 0.044444444444444446,
      "C_crop_complete": 21,
      "S_crop_complete": 25,
      "delta_crop_count": 4
    },
    {
      "seed": 999,
      "group": "single_GT",
      "C_recall": 0.9205298013245033,
      "S_recall": 0.9139072847682119,
      "delta_recall": -0.006622516556291391,
      "C_crop_complete": 144,
      "S_crop_complete": 146,
      "delta_crop_count": 2
    },
    {
      "seed": 999,
      "group": "multi_GT",
      "C_recall": 0.7666666666666667,
      "S_recall": 0.7777777777777778,
      "delta_recall": 0.011111111111111112,
      "C_crop_complete": 26,
      "S_crop_complete": 25,
      "delta_crop_count": -1
    }
  ],
  "exposure": [
    {
      "seed": 42,
      "arm": "C",
      "A": 48300,
      "B": 90300,
      "C": 92700,
      "large_GT": 18600,
      "medium_GT": 111600,
      "multi_GT_anchors": 43800,
      "negative_images": 5400,
      "small_GT": 159300,
      "very_small_GT": 54600,
      "unique_anchors": 771,
      "repeat_draws": 230529
    },
    {
      "seed": 42,
      "arm": "S",
      "A": 68794,
      "B": 96526,
      "C": 65980,
      "large_GT": 15100,
      "medium_GT": 95212,
      "multi_GT_anchors": 50840,
      "negative_images": 3739,
      "small_GT": 192709,
      "very_small_GT": 77657,
      "unique_anchors": 771,
      "repeat_draws": 230529
    },
    {
      "seed": 123,
      "arm": "C",
      "A": 48300,
      "B": 90300,
      "C": 92700,
      "large_GT": 18600,
      "medium_GT": 111600,
      "multi_GT_anchors": 43800,
      "negative_images": 5400,
      "small_GT": 159300,
      "very_small_GT": 54600,
      "unique_anchors": 771,
      "repeat_draws": 230529
    },
    {
      "seed": 123,
      "arm": "S",
      "A": 68874,
      "B": 96163,
      "C": 66263,
      "large_GT": 14982,
      "medium_GT": 95185,
      "multi_GT_anchors": 50671,
      "negative_images": 3837,
      "small_GT": 192523,
      "very_small_GT": 77691,
      "unique_anchors": 771,
      "repeat_draws": 230529
    },
    {
      "seed": 3407,
      "arm": "C",
      "A": 48300,
      "B": 90300,
      "C": 92700,
      "large_GT": 18600,
      "medium_GT": 111600,
      "multi_GT_anchors": 43800,
      "negative_images": 5400,
      "small_GT": 159300,
      "very_small_GT": 54600,
      "unique_anchors": 771,
      "repeat_draws": 230529
    },
    {
      "seed": 3407,
      "arm": "S",
      "A": 68631,
      "B": 96783,
      "C": 65886,
      "large_GT": 14941,
      "medium_GT": 95040,
      "multi_GT_anchors": 50713,
      "negative_images": 3823,
      "small_GT": 192799,
      "very_small_GT": 77467,
      "unique_anchors": 771,
      "repeat_draws": 230529
    },
    {
      "seed": 2026,
      "arm": "C",
      "A": 48300,
      "B": 90300,
      "C": 92700,
      "large_GT": 18600,
      "medium_GT": 111600,
      "multi_GT_anchors": 43800,
      "negative_images": 5400,
      "small_GT": 159300,
      "very_small_GT": 54600,
      "unique_anchors": 771,
      "repeat_draws": 230529
    },
    {
      "seed": 2026,
      "arm": "S",
      "A": 68844,
      "B": 96079,
      "C": 66377,
      "large_GT": 14877,
      "medium_GT": 96111,
      "multi_GT_anchors": 50970,
      "negative_images": 3795,
      "small_GT": 192724,
      "very_small_GT": 77966,
      "unique_anchors": 771,
      "repeat_draws": 230529
    },
    {
      "seed": 999,
      "arm": "C",
      "A": 48300,
      "B": 90300,
      "C": 92700,
      "large_GT": 18600,
      "medium_GT": 111600,
      "multi_GT_anchors": 43800,
      "negative_images": 5400,
      "small_GT": 159300,
      "very_small_GT": 54600,
      "unique_anchors": 771,
      "repeat_draws": 230529
    },
    {
      "seed": 999,
      "arm": "S",
      "A": 68959,
      "B": 96288,
      "C": 66053,
      "large_GT": 15003,
      "medium_GT": 95572,
      "multi_GT_anchors": 50960,
      "negative_images": 3802,
      "small_GT": 193001,
      "very_small_GT": 78010,
      "unique_anchors": 771,
      "repeat_draws": 230529
    }
  ]
}
```

Large 少1個：個別 seed gate FAIL，但非 catastrophic，必須揭露。少至少2個：任何有效 seed 發生即整體 FAIL。
不足5個有效 pairs 時不得縮小4/5分母；若另有 catastrophic，安全結論 FAIL，研究完整性仍標 incomplete。
統計為 paired seed effects 的平均、sample SD（ddof=1）、median、range；沒有 patient-level CI、p-values 或事後檢定。
Seed42 very-small 結果固定 C=27/49、S=27/49，未刪除或替換。Historical Phase C 僅 reference，不作 primary control。
沒有修改 weights、threshold、NMS、crop、App；沒有使用 locked test、CO2、official test 或新外部資料。
下一研究階段只能依本報告另行決定：保留已確認介入、研究 very-small failure、crop-policy 實驗或 external validation planning；本次一律停止。

## 27 項指定問題

1. **5 pairs 是否有效**：5/5

2. **各 seed Small C/S**：見 paired seed 主表

3. **各 seed Δ Small Recall**：見 paired seed 主表

4. **正向/零/負向數**：{'positive': 3, 'zero': 1, 'negative': 1}

5. **Mean paired Δ**：0.008759124087591242

6. **Sample SD**：0.024428029971797827

7. **Median**：0.014598540145985401

8. **Precision mean Δ**：0.008252060765379193

9. **F1 mean Δ**：0.007429112462949164

10. **Crop mean Δ**：0.4

11. **Medium/Large safety**：見逐 seed gates、large警示及 paired_effects.csv；不隱藏 -1

12. **Very-small 是否改善**：-0.6 mean Δ TP；seed42 Δ=0 保留

13. **<0.10%是否改善**：{'mean_delta_TP': 0.2, 'mean_delta_recall': 0.007407407407407407}

14. **改善集中 bins**：見全部五 bins 的逐 seed count 與 mean effect

15. **Multi-GT 是否一致改善**：見 single/multi-GT 逐 seed 表，不只報平均

16. **FP/FN 是否穩定**：見 paired_effects.csv 每 seed FP/FN/no-ROI

17. **AMP 是否一致**：[123, 2026]

18. **有無 invalid pair**：[]

19. **4/5方向門檻**：False

20. **整體 gate**：FAIL

21. **各 arm historical gate**：{'C': {'PASS': 0, 'FAIL': 5, 'PARTIAL': 0}, 'S': {'PASS': 0, 'FAIL': 4, 'PARTIAL': 1}}

22. **Locked test**：NO

23. **CO2**：NO

24. **Threshold 改變**：NO

25. **Crop 改變**：NO

26. **App替換**：NO

27. **下一研究阶段**：依結果另行授權，不自動進下一階段
