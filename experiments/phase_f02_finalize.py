"""CPU test receipt and write-once F0.2 protocol freeze; no training capability."""
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from experiments.phase_f02 import (ROOT, PRO, OUT, PREFIX, REQUEST, FUTURE, read, sha, save,
    require, install_execution_guard, historical_audit, MEDIA_ALLOWLIST, validate_pair, future_absent)

REPORT = ROOT / "docs/PHASE_F02_HIGHER_SCALE_V2_PROTOCOL_REVISION_20260926.md"


def verify():
    require(not (OUT / "verification.json").exists(), "FAIL_TEST_RECEIPT_EXISTS")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTHONDONTWRITEBYTECODE="1", PYTEST_ADDOPTS="", PYTEST_PLUGINS="")
    command = [sys.executable, "-B", "-m", "pytest", "tests/test_phase_f02.py", "-q"]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    save(OUT / "verification.json", {"command": command, "exit_code": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr, "created_at": datetime.now(timezone.utc).isoformat(),
        "test_sha256": sha(ROOT / "tests/test_phase_f02.py"),
        "scope": "stdlib/CPU event and artifact tests only; no old GPU or validator dry audits rerun"})
    print(result.stdout)
    require(result.returncode == 0, "FAIL_TESTS")


def finalize():
    guard = install_execution_guard()
    require(not REPORT.exists() and not (PRO / f"{PREFIX}_freeze.json").exists(), "FAIL_F02_ALREADY_FROZEN")
    verification = read(OUT / "verification.json")
    require(verification["exit_code"] == 0 and "60 passed" in verification["stdout"], "FAIL_TESTS")
    require(verification["test_sha256"] == sha(ROOT / "tests/test_phase_f02.py"), "FAIL_TEST_SOURCE_CHANGED")
    prepared = read(OUT / "prepared_seal.json")
    for relative, digest in prepared["artifacts_sha256"].items():
        require(sha(ROOT / relative) == digest, "FAIL_PREPARED_ARTIFACT_CHANGED: " + relative)
    require(prepared["request_sha256"] == sha(REQUEST), "FAIL_REQUEST_CHANGED")
    audit = read(OUT / "audit.json")
    for path, digest in audit["protected_sha256"].items():
        MEDIA_ALLOWLIST.add(__import__("pathlib").Path(path).resolve())
        require(sha(path) == digest, "FAIL_PROTECTED_HASH: " + path)
    require(historical_audit({}) == audit["historical"], "FAIL_HISTORY_OR_ATTEMPTS_CHANGED")
    c = read(PRO / f"{PREFIX}_control_config.json")
    h = read(PRO / f"{PREFIX}_experimental_config.json")
    require(validate_pair(c, h) == audit["paired_config_differences"], "FAIL_PAIR_CONFIG")
    future_absent()
    p = read(PRO / f"{PREFIX}_protocol.json")
    require(p["advancement_gate"] == read(PRO / "F_HIGHER_SCALE_V1_protocol.json")["advancement_gate"], "FAIL_ADVANCEMENT_CHANGED")
    require(sha(PRO / f"{PREFIX}_evaluation_protocol.json") == c["evaluation"]["sha256"], "FAIL_EVALUATION_CHANGED")
    status = {
        "PHASE_F02_STATUS": "COMPLETE",
        "READY_FOR_PHASE_F1_V2_PAIRED_HIGHER_SCALE_SEED42": "YES",
        "READINESS_SCOPE": "protocol/source audit complete; separate explicit authorization and shared runtime binding required before F1",
        "F0_STATUS": "BLOCKED_UNCHANGED", "F01_STATUS": "BLOCKED_UNCHANGED",
        "HISTORICAL_NUMERICAL_FEASIBILITY_1024": "FAIL_UNCHANGED",
        "V2_ACTUAL_TRAINING_NUMERICAL_SAFETY": "NOT_YET_OBSERVED",
        "SYNTHETIC_NUMERICAL_GATE_DISCRIMINATIVE_VALIDITY": "NOT_ESTABLISHED",
        "NEW_TRAINING_AUTHORIZED": "NO", "EXTERNAL_TEST_AUTHORIZED": "NO",
        "APP_MODEL_REPLACEMENT_AUTHORIZED": "NO", "STOP_AFTER_F02": True,
        "TRAINING": False, "MODEL_LOADING": False, "MODEL_FORWARD": False,
        "GPU_EXECUTION": False, "RESEARCH_INFERENCE": False, "test_images_used": 0,
        "CO2Wounds_used": False, "new_synthetic_attempts": 0,
        "protected_files_verified": len(audit["protected_sha256"]), "future_outputs_absent": FUTURE,
    }
    save(OUT / "status.json", status)
    save(OUT / "finalization_guard.json", guard)
    with REPORT.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(render(audit, verification, status))
    files = [*PRO.glob(PREFIX + "_*.json"), *[p for p in OUT.iterdir() if p.is_file()], REPORT,
             ROOT / "experiments/phase_f02.py", ROOT / "experiments/f02_safety.py",
             ROOT / "experiments/phase_f02_finalize.py", ROOT / "tests/test_phase_f02.py"]
    freeze = dict(status, created_at=datetime.now(timezone.utc).isoformat(), request_sha256=sha(REQUEST),
        artifacts_sha256={str(p.relative_to(ROOT)): sha(p) for p in files})
    save(PRO / f"{PREFIX}_freeze.json", freeze)
    print(json.dumps(status, ensure_ascii=False))


def render(audit, verification, status):
    questions = [
        ("F0 是否仍 BLOCKED？", "是，原 freeze／報告／證據雜湊不變。"),
        ("F0.1 是否仍 BLOCKED？", "是；原 numerical FAIL 與兩 arm 分類均保留。"),
        ("是否新增第9次 synthetic attempt？", "否，N768／N1024 各維持原8次。"),
        ("是否重新跑 GPU？", "否。F0.2 僅檔案雜湊、原始碼解析與 CPU 純事件測試。"),
        ("為何不再使用舊 synthetic readiness hard gate？", "同一規則也拒絕已有300 epochs完成證據的768 recipe；判別效度未建立，不足以單獨判定實際訓練 readiness。"),
        ("是否把1024 historical 改成PASS？", "否。F0.1 FAIL保持；V2真實訓練數值安全尚未觀測。"),
        ("V2 numerical hard stops？", "非有限loss、非有限參數、scale<1、連續16次skip、runtime exception；OOM另記FAIL_RESOURCE_RUNTIME。"),
        ("AMP skip 是否自動FAIL？", "否；單次／零星skip保留與報告，不能以總次數單獨判FAIL。"),
        ("連續多少次skip hard stop？", "16；跨epoch延續，只能由一次確認的applied update歸零。"),
        ("Scaler低於多少 hard stop？", "嚴格小於1.0；等於1.0本身不觸發此條件。"),
        ("Loss nonfinite如何處理？", "每batch raw loss／各loss component在backward前檢查；立即FAIL_NONFINITE_LOSS並停止整個pair。"),
        ("Parameter nonfinite如何處理？", "初始化及每次applied update後檢查；FAIL_NONFINITE_PARAMETERS，在下一batch／val／save前停止。"),
        ("Batch是否仍4？", "兩組都是4；不降batch，不以accumulation補償。"),
        ("Initial scaler policy兩邊相同？", "是，fresh stock GradScaler預設；source default65536，並非另傳init_scale覆寫。"),
        ("是否為H4手調scale？", "否；不依F0.1的512／1024設定初始值。"),
        ("C4 train／val scale？", "768／768 nominal，兩組同SharedValidation768與future runner。"),
        ("H4 train／val scale？", "1024／768 nominal，非H4@1024 validation。"),
        ("Final evaluation scale？", "兩組768；沿用完全相同的封存operational evaluation。"),
        ("Dataset是否771 original？", "是，771影像／965 GT原始polygon，所有SHA256重驗通過；191張dev val另核对。"),
        ("是否使用patch／canonical／mask-first labels？", "否；舊E檔僅雜湊保全，未作資料輸入，未修復topology。"),
        ("Sampling是否uniform？", "是，without replacement，每epoch771 IDs；沿用同300×771封存order。"),
        ("Advancement gate是否改變？", "否；Very-small+2、Small≥-1、P/F1≥-1pp、Medium≥-1、Large≥0、crop_complete≥-1，全部必須滿足。"),
        ("是否正式訓練？", "否。"),
        ("是否research inference？", "否；test_images_used=0，CO2Wounds未使用。"),
        ("是否可進入Future F1 V2 paired training？", "YES僅代表protocol已就緒；本輪不授權執行。下一次明確授權後仍須接入並驗證同一V2 runtime safety adapter。"),
    ]
    qtable = "\n".join(f"| {i} | {q} | {a} |" for i, (q, a) in enumerate(questions, 1))
    sources = "\n".join(f"- `{r['path']}`:{r['line']}–{r['end_line']}，`{r['function']}`" for r in audit["source"]["source_functions"])
    return f'''# Phase F0.2 — Higher-Input-Scale V2 Numerical Safety Protocol Revision

實際完成日期：2026-09-27（Asia/Taipei）；UTC封存時間見JSON。檔名依使用者指定保留20260926。

## 結論

```ini
PHASE_F02_STATUS = COMPLETE
READY_FOR_PHASE_F1_V2_PAIRED_HIGHER_SCALE_SEED42 = YES
SYNTHETIC_NUMERICAL_GATE_DISCRIMINATIVE_VALIDITY = NOT_ESTABLISHED
V2_ACTUAL_TRAINING_NUMERICAL_SAFETY = NOT_YET_OBSERVED
NEW_TRAINING_AUTHORIZED = NO
EXTERNAL_TEST_AUTHORIZED = NO
APP_MODEL_REPLACEMENT_AUTHORIZED = NO
STOP_AFTER_F02 = true
```

這是**新版本研究規範完成**，不是1024數值可行性PASS，也不是模型效果提升或F1已執行。F0與F0.1仍BLOCKED，原失敗觀察未改寫，沒有第9次attempt或重新GPU測試。

## 1. 修訂理由與保留的不利證據

F0：單次1024合成檢查的resource PASS、nonfinite gradients與numerical FAIL均保留。F0.1：3次連續finite／最多8次的門檻原封不動；N768首次finite在8、streak1、scale512；N1024首次finite在7、streak2、scale1024。兩組的PERSISTENT_NUMERICAL_INSTABILITY分類保留。N1024第7、8次僅為DESCRIPTIVE_EVIDENCE_OF_SCALER_BACKOFF_PROGRESS，不升格PASS。

本輪查核既有D1 fresh control：完成300 epochs，3741 scheduled、3731 applied、10 skipped、unknown0，telemetry integrity PASS；其training recipe與F0 control一致。這是768配方已有成功完成的經驗證據，不是保證所有未来seed穩定。原synthetic gate連這個配方也拒絕，因此其作為真實training-readiness判定的discriminative validity未建立。V2改的是**觀測安全性的方式**，不是降低模型、batch或performance gate。

舊fixture仍可描述overflow/backoff、記憶體與scale行為，不再當V2硬性readiness gate。synthetic不包含完整optimizer moments等長訓練狀態，F0 memory PASS不是300 epochs不會OOM的保證。

## 2. 固定paired設計

| 項目 | Fresh C4 | Fresh H4 |
|---|---|---|
| Training nominal imgsz | 768 | 1024 |
| Checkpoint-selection val nominal imgsz | 768 | 768 |
| Final operational eval imgsz | 768 | 768 |
| Model／seed | YOLO11m-seg／42 | 相同 |
| Batch／epochs／patience | 4／300／80 | 相同 |
| Optimizer | fresh AdamW，lr0=.0005，betas(.937,.999)，decay=.0005 | 相同 |
| Scheduler／warmup | cosine，lrf=.01，warmup5 | 相同 |
| AMP | true，fresh stock GradScaler defaults | 相同 |
| Sampling | uniform without replacement，771 anchors/epoch | 相同凍結順序 |
| Budget | 3741 scheduled opportunities，unknown=0 | 相同 |
| Validation infrastructure | SharedValidation768＋同一future V2 runner | 相同 |

workers2、deterministic=true、nbs64、warmup accumulation與其後16、clip10、loss／augmentation完整繼承V1。逐欄差異只允許training_args.imgsz，以及arm、experiment_id、output_path等識別路由。初始化為同一ISIC checkpoint，SHA256 `{audit['data']['initialization_sha256']}`，fresh model／optimizer／scaler，resume=false。

不換成D2 weighted sampler、不用patch／canonical／mask-first label、不修復原標註topology、不降低batch、不補update、不加epoch、不改patience、不試960/896/832。

## 3. 實際runtime numerical contract

| 事件 | 停止狀態 |
|---|---|
| 未scale的loss或component為NaN/Inf | FAIL_NONFINITE_LOSS |
| 初始化／applied update後參數為NaN/Inf | FAIL_NONFINITE_PARAMETERS |
| opportunity前或後scale<1.0 | FAIL_AMP_SCALE_COLLAPSE |
| 16次連續scheduled opportunity均確認skip | FAIL_PERSISTENT_AMP_UPDATE_SKIPS |
| CUDA／backward／optimizer等runtime exception | TRAINING_NUMERICAL_RUNTIME_INVALID |
| OOM | FAIL_RESOURCE_RUNTIME，並標runtime invalid |
| 計數不明、缺欄或矛盾telemetry | FAIL_TELEMETRY_INCOMPLETE，不能冒稱AMP skip |

任一arm觸發立即停止**整個pair**，保留日誌與失敗，不啟動另一arm或後續evaluation，不resume／補跑。

每batch在scaled backward前檢查raw loss，不能只在累積後的optimizer opportunity檢查，以免漏掉中間的nonfinite。每opportunity記錄epoch、batch、global_batch、scale before/after、loss finite、attempted/applied/skipped、可觀測gradient nonfinite、parameter finite。另保留lr、scheduler、accumulation、實際post-hook次數與exception。

更新判斷沿用D系列optimizer post-hook：0=skip、1=applied，多於1或exception=invalid；不能用scaler.step回傳None推斷skip。只支援已查核的non-fused AdamW；stock順序unscale→clip10→scaler.step→scaler.update→zero_grad→EMA不更改。gradient有限值若無法直接取得，明確null/not_observed，不捏造觀測。

連續skip跨epoch延續，只有applied update可歸零；total skips僅描述。兩組skip不同要標AMP_UPDATE_COUNT_IMBALANCE_OBSERVED=YES，不補實際更新，也不能宣稱realized update counts相等。

兩組都完成300 epochs、3741 scheduled、applied+skipped=scheduled、unknown0、完整telemetry且無hard failure，固定預定預算才有效。例如10 vs14 skips可以有效，但須揭露實際更新數不同。patience80不變：若早停於300之前，即PAIRED_FIXED_BUDGET_VALID=NO，不補epochs。

## 4. Source audit與實作邊界

新增純Python事件狀態機`experiments/f02_safety.py`，測試以上數值／記錄規則，**不是訓練runner**。D1/D2 legacy runner已提供post-hook accounting，但仍缺每batch raw-loss guard、update後parameter scan、跨epoch skip streak、scale-collapse與stop-pair傳遞。這些已凍結為F1共同adapter的接入前置條件；未宣稱實際runtime binding已驗證。未修改任何舊runner。

原始碼hash與AST來源位置（只讀，未import torch／Ultralytics）：

{sources}

第一次F02純查核因誤假設torch位於system site-packages而停止，當時未建立V2 artifacts且未執行GPU；改為讀取top-level package spec定位到user-site後完成雜湊查核。沒有更換依賴或重跑數值實驗。

NumPy版本仍明確區分：繼承software欄位是distribution metadata2.0.1，F0.1曾觀測實際runtime2.2.6；本輪不import NumPy、不改package，F1需記錄實際版本與來源，不能把metadata當作runtime identity。

## 5. 資料與封存核對

- 771原始FUSeg training images／965 polygon GT：image與label SHA256全部重驗，class0；不修復、不刪除標註。
- 191張development validation：image、polygon label、reference mask，以及training-validation副本雜湊核對一致；僅byte hashing，不解碼／不推論。
- train/val exact-content hash overlap=0；此處不是新做patient-level隔離證明。
- 原300×771 anchor order逐epoch檢查唯一ID、全集與hash，原order SHA `{audit['data']['order_sha256']}`；未重新抽樣。
- 初始化權重只hash，未deserialize。F0／F0.1／E三版封存檔均保留，共{status['protected_files_verified']}個歷史／資料／來源檔核對。
- `test_images_used=0`；未使用locked test、CO2Wounds或external test；未替換App模型。

## 6. 驗證與performance gate不變

沿用F0.1已封存的SharedValidation768乾式證據，不重跑。兩arm經同一mixin，val nominal768，stock rect/pad/stride保留；歷史square synthetic batch實為800×800，不保證每批768×768。兩邊用相同規則，不能C4走stock／H4走custom不同infra。

Training checkpoint validator維持stock conf .001、NMS .70、AP matching .50:.05:.95。另行final operational evaluation是191張full-image corrected RGB→BGR、confidence .10／prediction floor .01／NMS .70／match .50／imgsz768／crop margin15%，規範直接保留V1 bytes/hash。

Primary endpoint為Very-small Recall<0.25%（support49），唯一因果比較Fresh C4 vs Fresh H4；既有C、D1/D2與sampling結果都REFERENCE_ONLY。Gate全數不變：very-small TP≥C4+2、Small TP≥C4−1、Precision與F1各≥C4−1.0pp、Medium TP≥C4−1、Large TP≥C4、crop complete≥C4−1。

## 7. 25項完成問題

| # | 問題 | 回答 |
|---:|---|---|
{qtable}

## 8. Tests、方法寫法與停止

`{verification['stdout'].strip().splitlines()[-1]}`。TDD先確認失敗測試，再實作純事件guard與設定邊界，涵蓋16次／15次、epoch跨界、applied reset、累計30次零星skip、scale=1 vs .5、unknown／duplicate事件、早停與budget不一致、import/process guards、封存資料與設定。沒有重跑F0/F01 GPU，也沒有以舊suite觸發synthetic validator。

方法章應保留F0/F0.1失敗並使用未來式，因V2尚未實際訓練：

> Because the synthetic gate also rejected the empirically completed 768 training recipe, it was not retained as the hard readiness criterion in the separately versioned V2 protocol. Future paired training will retain stock AMP behavior and use preregistered online numerical safety stops. No V2 training has yet been executed.

未來結果目錄`f_higher_scale_v2_seed42_control768/`與`f_higher_scale_v2_seed42_train1024/`均不存在。READY=YES僅表示此次protocol revision completion gate通過。**STOP AFTER F0.2；下一次明確授權前，不開始Fresh C4或H4。**
'''


if __name__ == "__main__":
    if sys.argv[1:] == ["verify"]:
        verify()
    elif sys.argv[1:] == ["finalize"]:
        finalize()
    else:
        raise SystemExit("Only verify or finalize; no training command")
