# Phase D2 多種子配對確認：執行前定義

日期：2026-09-23。本文件與 D2 JSON protocol 在新 seed 訓練前封存。

## 範圍與順序

Seed42 的 D1 Control／Experimental、評估與報告全部唯讀，重新核對 hashes 後引用，不重訓、不重新推論。

只新增四組配對：123 → 3407 → 2026 → 999。每組固定 C 訓練 → S 訓練 → pair validity → C evaluation → S evaluation。正常完成的負結果不會導致取消後續 seed。

所有組使用同一 ISIC auxiliary checkpoint、771 FUSeg train、191 development val、YOLO11m-seg、768、batch4、300 epochs、patience80、AdamW、lr0.0005、既定 AMP/accumulation/loss/augmentation。S 權重固定 2/1.5/1；不加入 validation failures 或其他資料。

## 兩層 Large 安全門檻（使用者明確指定）

- Seed-level：`S_large_TP >= C_large_TP`。少至少 1 個 Large TP，該 seed 的 D1-style gate 即 FAIL。
- D2 catastrophic：`S_large_TP <= C_large_TP - 2`。任一有效 paired seed 少至少 2/14 個 Large GT（至少 14.285714… pp），D2 整體安全門檻 FAIL。
- 正好少 1 個不是 catastrophic，但必須列入報告，且個別 seed gate 仍 FAIL。

## D2 整體確認門檻

五組全部有效且 intervention delivered；至少 4/5 組 Small Recall 的 S−C >0；mean paired Small Recall Δ >0；mean Δ Precision、mean Δ F1 各 ≥−0.01；沒有任何 catastrophic Large regression。

不要求 5/5 individual gates 都 PASS，不降低 4/5 分母，不以 rounded percentages 判斷門檻。不足五組有效配對則 INCOMPLETE；若已有 catastrophic，安全結論為 FAIL，同時保留研究完整性不足標記。

每組預期 3,741 scheduled opportunities、applied+skipped=3,741、unknown=0、完整300 epochs。AMP skip 不同時如實標 imbalance，不補更新、不重訓。每組實際 consumed anchors 都核對 frozen epoch plan，S 的 A／small／very-small exposure 必須增加。

## 執行與失敗邊界

每個 arm 專用新目錄與不可覆写 execution.lock。不得 resume、重跑、刪除失敗目錄或換 seed。單一 seed 早停／OOM／crash 保留 INVALID，跳過該無效 pair 的剩餘 arm，不補跑；只在共用安全條件仍通過時嘗試其他已登錄 seed。共用 protocol／資料／環境完整性失敗則停止全部。

D1／D0.1 程式原檔不修改。D2 在獨立 module namespace 使用已驗證的原始程式，僅替換 seed-aware index planner 與路由；train/AMP/loader/metrics 程式內容不變。用 seed42 全300 epoch的 C/S index 計畫一致性測試證明 sampler 演算法保持。

## 統計解讀與停止

研究單位為 5 paired random-seed experiments，包含已觀察的 seed42；同用191張 development val，不是 n=10 independent models，也不是患者層級或未見外部資料的確認。報 mean、sample SD（ddof=1）、median、min/max、正/零/負方向數；不新增 CI、p-value 或事後選擇檢定。

完整保留五個 small bins、seed42 very-small 27/49→27/49 的未改善結果、各 seed single/multi-GT、FP/FN/no-ROI、crop、AMP 與 exposure。原 development gate 與 sampling confirmation 分開報告。

不使用 locked48、FUSeg official test、CO2Wounds 或其他 external data；不調 threshold/NMS/crop，不更換 App。D2 完成後無論 PASS/FAIL/INCOMPLETE 都停止。
