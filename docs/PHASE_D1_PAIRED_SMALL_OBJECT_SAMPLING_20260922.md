# Phase D1 配對抽樣實驗結果

| Metric | Historical Phase C | Fresh C | Fresh S | S−C |
|---|---:|---:|---:|---:|
| Precision | 0.853061 | 0.871245 | 0.884615 | 0.013371 |
| Recall | 0.867220 | 0.842324 | 0.858921 | 0.016598 |
| F1 | 0.860082 | 0.856540 | 0.871579 | 0.015039 |
| Small Recall | 0.795620 | 0.759124 | 0.788321 | 0.029197 |
| Very-small Recall | 0.591837 | 0.551020 | 0.551020 | 0.000000 |
| Medium Recall | 0.955556 | 0.944444 | 0.944444 | 0.000000 |
| Large Recall | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| Crop complete | 0.897849 | 0.887097 | 0.897849 | 0.010753 |
| TP | 209.000000 | 203.000000 | 207.000000 | 4.000000 |
| FP | 36.000000 | 30.000000 | 27.000000 | -3.000000 |
| FN | 32.000000 | 38.000000 | 34.000000 | -4.000000 |
| No ROI | 4.000000 | 4.000000 | 3.000000 | -1.000000 |

| Training metric | Fresh C | Fresh S |
|---|---:|---:|
| completed_epochs | 300 | 300 |
| scheduled_optimizer_calls | 3741 | 3741 |
| applied_optimizer_updates | 3731 | 3731 |
| skipped_optimizer_updates | 10 | 10 |
| anchor_draws | 231300 | 231300 |
| unique_anchors_seen | 771 | 771 |
| repeat_draws_across_run | 230529 | 230529 |

## 診斷與完整決策

```json
{
  "gates": {
    "status": "PASS_SMALL_SAMPLING_RESEARCH_ADVANCEMENT_GATE",
    "checks": {
      "small": true,
      "precision": true,
      "f1": true,
      "crop": true,
      "medium": true,
      "large": true
    }
  },
  "AMP": {
    "separate_skip_report_required": true,
    "absolute_skip_difference": 0,
    "AMP_UPDATE_COUNT_IMBALANCE_OBSERVED": false,
    "threshold_frozen": "any nonzero difference flags imbalance; never auto-retrain",
    "identical_realized_updates_claim_allowed": true
  },
  "C_diagnostics": {
    "small_bins": {
      "<0.10%": {
        "support": 27,
        "TP": 11,
        "recall": 0.4074074074074074
      },
      "0.10-0.25%": {
        "support": 22,
        "TP": 16,
        "recall": 0.7272727272727273
      },
      "0.25-0.50%": {
        "support": 38,
        "TP": 31,
        "recall": 0.8157894736842105
      },
      "0.50-0.75%": {
        "support": 30,
        "TP": 28,
        "recall": 0.9333333333333333
      },
      "0.75-1.00%": {
        "support": 20,
        "TP": 18,
        "recall": 0.9
      }
    },
    "very_small": {
      "support": 49,
      "TP": 27,
      "recall": 0.5510204081632653
    },
    "single_GT": {
      "image_count": 151,
      "GT": 151,
      "TP": 139,
      "FN": 12,
      "instance_recall": 0.9205298013245033,
      "images_with_any_miss": 12,
      "image_any_miss_rate": 0.07947019867549669,
      "crop_pass": 145,
      "crop_fail": 6,
      "crop_complete_rate": 0.9602649006622517,
      "mean_retained_fraction": 0.9603280665856576,
      "FP": 18,
      "FP_per_image": 0.11920529801324503
    },
    "multi_GT": {
      "image_count": 35,
      "GT": 90,
      "TP": 64,
      "FN": 26,
      "instance_recall": 0.7111111111111111,
      "images_with_any_miss": 21,
      "image_any_miss_rate": 0.6,
      "crop_pass": 20,
      "crop_fail": 15,
      "crop_complete_rate": 0.5714285714285714,
      "mean_retained_fraction": 0.8792432375080153,
      "FP": 9,
      "FP_per_image": 0.2571428571428571
    }
  },
  "S_diagnostics": {
    "small_bins": {
      "<0.10%": {
        "support": 27,
        "TP": 11,
        "recall": 0.4074074074074074
      },
      "0.10-0.25%": {
        "support": 22,
        "TP": 16,
        "recall": 0.7272727272727273
      },
      "0.25-0.50%": {
        "support": 38,
        "TP": 34,
        "recall": 0.8947368421052632
      },
      "0.50-0.75%": {
        "support": 30,
        "TP": 28,
        "recall": 0.9333333333333333
      },
      "0.75-1.00%": {
        "support": 20,
        "TP": 19,
        "recall": 0.95
      }
    },
    "very_small": {
      "support": 49,
      "TP": 27,
      "recall": 0.5510204081632653
    },
    "single_GT": {
      "image_count": 151,
      "GT": 151,
      "TP": 141,
      "FN": 10,
      "instance_recall": 0.9337748344370861,
      "images_with_any_miss": 10,
      "image_any_miss_rate": 0.06622516556291391,
      "crop_pass": 145,
      "crop_fail": 6,
      "crop_complete_rate": 0.9602649006622517,
      "mean_retained_fraction": 0.9751648730525103,
      "FP": 16,
      "FP_per_image": 0.10596026490066225
    },
    "multi_GT": {
      "image_count": 35,
      "GT": 90,
      "TP": 66,
      "FN": 24,
      "instance_recall": 0.7333333333333333,
      "images_with_any_miss": 20,
      "image_any_miss_rate": 0.5714285714285714,
      "crop_pass": 22,
      "crop_fail": 13,
      "crop_complete_rate": 0.6285714285714286,
      "mean_retained_fraction": 0.90965053737196,
      "FP": 8,
      "FP_per_image": 0.22857142857142856
    }
  },
  "historical_diagnostics": {
    "small_bins": {
      "<0.10%": {
        "support": 27,
        "TP": 13,
        "recall": 0.48148148148148145
      },
      "0.10-0.25%": {
        "support": 22,
        "TP": 16,
        "recall": 0.7272727272727273
      },
      "0.25-0.50%": {
        "support": 38,
        "TP": 33,
        "recall": 0.868421052631579
      },
      "0.50-0.75%": {
        "support": 30,
        "TP": 29,
        "recall": 0.9666666666666667
      },
      "0.75-1.00%": {
        "support": 20,
        "TP": 18,
        "recall": 0.9
      }
    },
    "very_small": {
      "support": 49,
      "TP": 29,
      "recall": 0.5918367346938775
    },
    "single_GT": {
      "image_count": 151,
      "GT": 151,
      "TP": 139,
      "FN": 12,
      "instance_recall": 0.9205298013245033,
      "images_with_any_miss": 12,
      "image_any_miss_rate": 0.07947019867549669,
      "crop_pass": 144,
      "crop_fail": 7,
      "crop_complete_rate": 0.9536423841059603,
      "mean_retained_fraction": 0.9650242012567489,
      "FP": 25,
      "FP_per_image": 0.16556291390728478
    },
    "multi_GT": {
      "image_count": 35,
      "GT": 90,
      "TP": 70,
      "FN": 20,
      "instance_recall": 0.7777777777777778,
      "images_with_any_miss": 17,
      "image_any_miss_rate": 0.4857142857142857,
      "crop_pass": 23,
      "crop_fail": 12,
      "crop_complete_rate": 0.6571428571428571,
      "mean_retained_fraction": 0.9141771484721,
      "FP": 10,
      "FP_per_image": 0.2857142857142857
    }
  },
  "exposure": [
    {
      "metric": "A",
      "C": 48300,
      "S": 68794
    },
    {
      "metric": "B",
      "C": 90300,
      "S": 96526
    },
    {
      "metric": "C",
      "C": 92700,
      "S": 65980
    },
    {
      "metric": "large_GT",
      "C": 18600,
      "S": 15100
    },
    {
      "metric": "medium_GT",
      "C": 111600,
      "S": 95212
    },
    {
      "metric": "multi_GT_anchors",
      "C": 43800,
      "S": 50840
    },
    {
      "metric": "negative_images",
      "C": 5400,
      "S": 3739
    },
    {
      "metric": "small_GT",
      "C": 159300,
      "S": 192709
    },
    {
      "metric": "very_small_GT",
      "C": 54600,
      "S": 77657
    }
  ]
}
```

兩組完整 300 epochs；scheduled/applied/skipped 見表。未 resume/retry，未修改 threshold/crop/metrics。未使用 locked test、CO2、external test；未啟動 multi-seed、未替換 App。

Under this single-seed paired development experiment, small-object-aware sampling improved observed small-wound recall relative to the fresh uniform-sampling control.

本結果不代表統計顯著性或臨床泛化；未新增 CI、p-value 或 McNemar。Latency 僅描述，歷史 gate 若性能全通過則標 PARTIAL，未假稱硬體可比。

下一研究決策可考慮另次授權 multi-seed confirmation；現在停止。

## 27 個指定問題

1. **Control 完整300 epochs**：300

2. **Experimental 完整300**：300

3. **Scheduled opportunities**：C=3741; S=3741

4. **Applied updates**：C=3731; S=3731

5. **Skipped updates**：C=10; S=10

6. **AMP imbalance**：False

7. **Sampling delivered**：YES; 實際序列與各 frozen epoch plan 相符，exposure 差異見 JSON/CSV

8. **Control Small Recall**：{'gt': 137, 'matched': 104, 'recall': 0.7591240875912408}

9. **Experimental Small Recall**：{'gt': 137, 'matched': 108, 'recall': 0.7883211678832117}

10. **Small TP 改變**：4

11. **Precision 改變**：0.013370749422251582

12. **F1 改變**：0.015038862980235379

13. **Crop complete 改變**：2 張

14. **Medium/Large safety**：{'medium': True, 'large': True}

15. **Very-small bins**：見五 bins diagnostics；不事後改權重

16. **Multi-GT Recall**：見 C/S single_GT、multi_GT diagnostics

17. **FP/FN/no-ROI**：見主表與 candidate summary

18. **Research gate**：PASS

19. **Historical gate**：C=FAIL; S=FAIL

20. **Fresh C vs historical C**：見主表；不因兩者不同就認定 bug

21. **Crash/resume/retry**：NO

22. **Threshold/crop/metrics 改變**：NO

23. **Locked test**：NO

24. **CO2Wounds**：NO

25. **Multi-seed**：NO

26. **App replacement**：NO

27. **下一研究決策**：另次授權 multi-seed confirmation
