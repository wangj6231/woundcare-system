"""Seal F0.1 observed synthetic evidence; never authorize execution of F1."""
from datetime import datetime,timezone
import json
from experiments.phase_f01 import ROOT,PRO,OUT,PREFIX,read,save,sha,require,classify,fixture_hashes,future_absent


def run():
    tests=read(OUT/'verification.json');require(tests['exit_code']==0,'FAIL_TESTS')
    protected=read(OUT/'protected_f0_snapshot.json')
    for p,h in protected.items():require(sha(p)==h,'FAIL_F0_EVIDENCE_CHANGED: '+p)
    for p,h in read(OUT/'pre_execution_freeze.json')['artifacts_sha256'].items():require(sha(ROOT/p)==h,'FAIL_F01_EXECUTION_CONTRACT_CHANGED: '+p)
    future_absent();require(not list(OUT.rglob('*.pt')) and not list(OUT.rglob('*.pth')),'FAIL_CHECKPOINT_WRITTEN')
    arms=[read(OUT/a/'summary.json') for a in ['N768','N1024']]
    require(arms[0]['pid']!=arms[1]['pid'],'FAIL_PROCESS_ISOLATION')
    require(read(OUT/'N768_process.json')['exited_at']<=read(OUT/'N1024_process.json')['started_at'],'FAIL_ARM_OVERLAP')
    require(arms[0]['fixture']==arms[1]['fixture']==fixture_hashes(),'FAIL_FIXTURE_IDENTITY')
    require(arms[0]['initial_trainable_parameter_sha256']==arms[1]['initial_trainable_parameter_sha256'],'FAIL_INITIAL_STATE_PARITY')
    for arm in arms:
        require(arm['completed_attempts']==8,'FAIL_INCOMPLETE_ARM')
        require(classify(arm['attempts'])['classification']==arm['classification'],'FAIL_CLASSIFICATION')
        require(all(r['parameters_unchanged'] and r['loss_finite'] for r in arm['attempts']),'FAIL_PARAMETER_OR_LOSS_SAFETY')
        require(all(arm['guards'][k]==0 for k in ['save_attempts','checkpoint_write_attempts','optimizer_step_attempts','scaler_step_attempts','image_access_attempts']),'FAIL_GUARD')
    guards=read(OUT/'guard_evidence.json');dry=read(OUT/'validator_dry_audit.json')
    require(guards['returncode']==dry['returncode']==0,'FAIL_CPU_AUDIT')
    g=guards['evidence'];v=dry['evidence']
    require(all(g[k+'_raised'] for k in ['save','write','optimizer','scaler','decode']) and not g['file_exists'],'FAIL_GUARD_BEHAVIOR')
    require(v['BEST_CHECKPOINT_SELECTION_PARITY'] and not v['validation_inference'] and v['real_validation_images_loaded']==0,'FAIL_VALIDATOR_PARITY')
    stable=arms[1]['NUMERICALLY_STABLE']
    both_persistent=all(a['classification']=='PERSISTENT_NUMERICAL_INSTABILITY' for a in arms)
    result={'PHASE_F01_STATUS':'COMPLETE' if stable else 'BLOCKED',
        'NUMERICAL_FEASIBILITY_1024':'PASS' if stable else 'FAIL',
        'NUMERICAL_FEASIBILITY_SCOPE':'preregistered synthetic gate only; not a claim of real-training safety or failure',
        'SYNTHETIC_NUMERICAL_FIXTURE_VALIDITY':'QUESTIONABLE' if both_persistent else 'NO_CASE_D_FLAG',
        'READY_FOR_PHASE_F1_PAIRED_HIGHER_SCALE_SEED42':'YES' if stable else 'NO',
        'case':'D' if both_persistent else 'OTHER',
        'interpretation':'Both arms fail to reach3 consecutive finite within8; startup overflow not unique to1024; cannot attribute failure to1024 or claim future stability.' if both_persistent else 'See paired telemetry; synthetic preflight only.',
        'N768_classification':arms[0]['classification'],'N1024_classification':arms[1]['classification'],
        'F0_historical_status':'BLOCKED_UNCHANGED','PATCH_BASED_TRAINING_ROUTE':'STOP',
        'BEST_CHECKPOINT_SELECTION_PARITY':'VERIFIED_NOMINAL768_SHARED_STOCK_RECT_PIPELINE',
        'SAVE_STEP_GUARDS':'BEHAVIORALLY_VERIFIED','PARAMETER_UPDATE_COUNT':0,
        'test_images_used':0,'CO2Wounds_used':False,'research_data_pixels_used':0,
        'research_training':False,'research_validation_inference':False,
        'NEW_TRAINING_AUTHORIZED':'NO','EXTERNAL_TEST_AUTHORIZED':'NO','APP_MODEL_REPLACEMENT_AUTHORIZED':'NO',
        'STOP_AFTER_F01':True,'F0_protected_files_unchanged':len(protected),'tests':tests['summary']}
    comparison=[]
    for a in arms:
        consecutive=longest=0
        for r in a['attempts']:
            consecutive=consecutive+1 if r['loss_finite'] and r['nonfinite_gradient_tensor_count']==0 else 0
            longest=max(longest,consecutive)
        comparison.append({k:a.get(k) for k in ['arm','classification','FIRST_FINITE_GRADIENT_ATTEMPT','FIRST_FINITE_GRADIENT_SCALE','stable_scale','peak_allocated_bytes','peak_reserved_bytes','total_VRAM_bytes','pid']}|
            {'longest_consecutive_finite':longest,'scaler_sequence':[a['attempts'][0]['scaler_before']]+[r['scaler_after'] for r in a['attempts']]})
    save(OUT/'paired_comparison.json',{'arms':comparison,'status':result,
        'F0_memory_preserved':read(ROOT/'experiments/results/f_higher_scale_v1_resource_audit/resource_smoke_result.json')})
    save(OUT/'status.json',result)
    report=ROOT/'docs/PHASE_F01_HIGHER_SCALE_NUMERICAL_PREFLIGHT_20260926.md'
    with report.open('x',encoding='utf8',newline='\n') as f:f.write(render(result,arms,comparison,v,tests))
    files=[*PRO.glob(PREFIX+'*.json'),*[p for p in OUT.rglob('*') if p.is_file()],report,
        ROOT/'experiments/phase_f01.py',ROOT/'experiments/f01_numerical.py',ROOT/'experiments/f01_guards.py',ROOT/'experiments/f01_validator.py',
        ROOT/'experiments/phase_f01_finalize.py',ROOT/'tests/test_phase_f01.py']
    freeze=dict(result,created_at=datetime.now(timezone.utc).isoformat(),
        artifacts_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        request_sha256=sha('C:/Users/milo9/.codex/attachments/d095e76a-a20f-4670-bc90-9214b6d82e64/貼上的文字.txt'),
        numerical_comparison=comparison,freeze_scope='F0.1 synthetic-only observations and shared validation dry audit; no F1 launch')
    save(PRO/f'{PREFIX}_freeze.json',freeze)
    print(json.dumps(result,ensure_ascii=False))


def render(status,arms,comparison,validator,tests):
    a,b=comparison
    sequence=lambda r:' → '.join(str(int(x)) for x in r['scaler_sequence'])
    telemetry=[]
    for arm in arms:
        for r in arm['attempts']:
            telemetry.append(f"| {arm['arm']} | {r['attempt']} | {int(r['scaler_before'])}→{int(r['scaler_after'])} | {r['finite_gradient_tensor_count']}/{r['gradient_tensor_count']} | {r['nan_gradient_tensor_count']} | {r['inf_gradient_tensor_count']} |")
    table='\n'.join(telemetry)
    return f'''# Phase F0.1 — Paired AMP Numerical Feasibility & Execution-Guard Audit

日期：2026-09-26。研究線：F_HIGHER_INPUT_SCALE_V1_NUMERICAL_PREFLIGHT。

## 1. 決策與限制

```ini
PHASE_F01_STATUS = {status['PHASE_F01_STATUS']}
NUMERICAL_FEASIBILITY_1024 = {status['NUMERICAL_FEASIBILITY_1024']}
SYNTHETIC_NUMERICAL_FIXTURE_VALIDITY = {status['SYNTHETIC_NUMERICAL_FIXTURE_VALIDITY']}
READY_FOR_PHASE_F1_PAIRED_HIGHER_SCALE_SEED42 = {status['READY_FOR_PHASE_F1_PAIRED_HIGHER_SCALE_SEED42']}
NEW_TRAINING_AUTHORIZED = NO
EXTERNAL_TEST_AUTHORIZED = NO
APP_MODEL_REPLACEMENT_AUTHORIZED = NO
PATCH_BASED_TRAINING_ROUTE = STOP
STOP_AFTER_F01 = true
```

**Case D：兩組都未於8次內取得3次連續finite。** N768第8次才首次全部梯度finite；N1024第7、8次全部finite，但只有連續2次。不得將2次放寬成3次、不得延長到第9次驗證。

`PERSISTENT_NUMERICAL_INSTABILITY` 是本次「固定8次仍未符合穩定門檻」的預登錄分類，**不表示8次全為nonfinite，也不證明1024在真實訓練不安全**。兩組初始均overflow，1024不是唯一有此現象的尺度。此fixture對scale failure的判別效度QUESTIONABLE，不能直接歸咎1024。`NUMERICAL_FEASIBILITY_1024=FAIL`只指本次synthetic gate未達標，不是模型效果或正式訓練穩定性的結論。

F0仍是BLOCKED historical evidence，保留原65536→32768、第一次nonfinite觀察，以及allocated5.736GiB／reserved5.820GiB。没有追溯改成PASS。

## 2. 預先固定的比較

兩arm分別由不同CUDA subprocess執行，順序N768完整退出後才N1024。每組同checkpoint bytes、fresh model、fresh AdamW、fresh GradScaler；相同seed42、batch4、AMP、lr.0005、betas(.937,.999)、loss gains與class0。初始scale一律stock65536，不手動設定較低值。所有8次均保留，未在出現finite後提早結束。

每次zero_grad → forward/loss → loss finite檢查 → scale/backward/unscale → 每個gradient tensor檢查 → scaler.update → trainable parameter SHA驗證。禁止optimizer.step及scaler.step。模型沒有parameter updates，沒有研究epochs；BN running buffers容許在各自可拋棄RAM模型中變動，未保存。

### Fixture來源與版本紀錄

以F0封存的actual-execution source AST中純 `synthetic_source` 函式重建：4張512×512 noise images、4×4合法rectangle polygons／張、共64GT、class0。未換generator或依結果調整fixture。

Images SHA256：`{arms[0]['fixture']['images_sha256']}`

Polygons SHA256：`{arms[0]['fixture']['polygons_sha256']}`

兩組執行中重新計算的fixture hashes都與執行前封存一致。**F0當時沒有存原始fixture hash或RAM arrays**；身份證據是其實際執行程式、確定性重建與本次相同hash，不能冒稱有歷史checksum直接可比。GPU開始前發現NumPy distribution metadata2.0.1與實際runtime2.2.6不一致，已修正F0.1描述，保留initial pre-execution文件與更正receipt；正常／no-user-site模式重建hash相同，沒有改package、沒有改F0檔案、沒有GPU重跑。

512 source僅分別resize至768／1024。GT仍使用同一stock polygons2masks_overlap與mask_ratio4，沒有mask-first或patch label path。初始化SHA256仍為`{arms[0]['initialization_sha256']}`；兩arm初始trainable parameter SHA一致，非接續另一arm的RAM狀態。

## 3. 每次梯度telemetry

以下loss在全部16次都finite，trainable parameter hash每次都不變。NaN/Inf欄為「含該值的gradient tensor數」，同一tensor可能同時含兩者，因此兩欄不能直接相加当作nonfinite總數。

| Arm | Attempt | Scaler before→after | Finite gradient tensors | NaN tensors | Inf tensors |
|---|---:|---|---:|---:|---:|
{table}

逐次JSON另有nonfinite parameter names、max_abs_finite_gradient、parameter SHA及guard counters；未報告或比較loss數值以挑選scale。

## 4. Paired摘要

| 項目 | N768 | N1024 |
|---|---|---|
| Attempt1全部gradients finite | 否 | 否 |
| First finite attempt | {a['FIRST_FINITE_GRADIENT_ATTEMPT']} | {b['FIRST_FINITE_GRADIENT_ATTEMPT']} |
| First finite scale | {a['FIRST_FINITE_GRADIENT_SCALE']} | {b['FIRST_FINITE_GRADIENT_SCALE']} |
| 最長連續finite | {a['longest_consecutive_finite']} | {b['longest_consecutive_finite']} |
| 至少3次連續finite | 否 | 否 |
| Numerical classification | {a['classification']} | {b['classification']} |
| Peak allocated | {a['peak_allocated_bytes']/1024**3:.3f} GiB | {b['peak_allocated_bytes']/1024**3:.3f} GiB |
| Peak reserved | {a['peak_reserved_bytes']/1024**3:.3f} GiB | {b['peak_reserved_bytes']/1024**3:.3f} GiB |
| OOM | {arms[0]['OOM']} | {arms[1]['OOM']} |

N768 scaler sequence：{sequence(a)}。

N1024 scaler sequence：{sequence(b)}。

首次finite scale皆高於1；失敗點是「連續次數不足」，不是scale跌至1以下。Stock scaler已自行降低scale並出現finite gradients，但本protocol不足以確認finite stable state。不能說已證明stock-scaler manageable transient overflow，更不能推論300epoch訓練穩定。記憶體資料只供描述，fresh optimizer沒有配置step後才產生的Adam moments。

## 5. Save／step／data guards

CPU行為測試在Ultralytics wrappers載入後安裝sentinels：假torch.save、直接.pt寫入、optimizer.step、scaler.step與image decode均拋出指定例外，temporary fake.pt始終不存在。測試使用CPU tensor與disposable optimizer，沒有GPU模型；故不再依function-object identity判斷guard。

GPU執行與CPU故意觸發測試分開統計。N768/N1024的save_attempts、checkpoint_write_attempts、optimizer_step_attempts、scaler_step_attempts、image_access_attempts **各自皆0**。每次parameter SHA不變，PARAMETER_UPDATE_COUNT=0，輸出樹無.pt/.pth。所有真實影像file-open／decode都禁止；GPU只使用RAM synthetic pixels，image_open_decode_paths為空。允許讀取只有初始化checkpoint、source與frozen configs／fixture lineage。

F0的 `update_save_guards_installed=false` 未更改；F0.1提供新的獨立behavioral evidence，不覆蓋舊觀察。

## 6. 共用training-validation override

Stock trainer的build_dataset／get_validator會使用trainer args.imgsz；不會自動讓H4 validation保持768。新增同一 `SharedValidation768` mixin供兩arm使用：train原args不動，validation使用shallow trainer proxy與args copy，把dataset與validator的imgsz都固定768。

以temporary **合成PNG/labels** 建立實際stock YOLODataset、DataLoader與SegmentationValidator，跑CPU preprocess但禁止validator.__call__／model construction／forward。

| Runtime證據 | C4 | H4 |
|---|---:|---:|
| Train dataset nominal imgsz | 768 | 1024 |
| Val dataset nominal imgsz | 768 | 768 |
| Validator args.imgsz | 768 | 768 |
| Override後trainer args.imgsz | 768 | 1024 |
| 實際synthetic val batch shape | [4,3,800,800] | [4,3,800,800] |
| Training-checkpoint validator confidence | .001 | .001 |
| NMS IoU | .70 | .70 |
| Stock AP matching IoUs | .50:.05:.95 | 相同 |

**768是stock nominal imgsz，而非保證tensor恰好768×768。** 原rect=True、pad=.5與stride32使本次square synthetic validation padding至800×800；兩組完全相同，未偷偷改padding，也不是H4以1024驗證。`BEST_CHECKPOINT_SELECTION_PARITY`在這個共同stock設定的乾式整合邊界已驗證；沒有跑191張real validation、沒有checkpoint selection結果。

這個training-time stock fitness validator與最後operational evaluation不同：最後評估仍使用封存的conf.10／floor.01／NMS.70／bbox match.50／imgsz768／crop margin15%。F01只驗證兩組共用相同checkpoint-selection基礎設施，不偷改最終matching或thresholds。

## 7. 28項完成問題

| # | 問題 | 答案 |
|---:|---|---|
| 1 | F0是否仍BLOCKED？ | 是，20個F0封存／證據檔hash未變 |
| 2 | N768 attempt1 finite？ | 否 |
| 3 | N1024 attempt1 finite？ | 否 |
| 4 | N768 scaler？ | 見完整序列，65536逐次降至512 |
| 5 | N1024 scaler？ | 見完整序列，65536逐次降至1024 |
| 6 | N768 first finite attempt？ | 8 |
| 7 | N1024 first finite attempt？ | 7 |
| 8 | N768 first finite scale？ | 512 |
| 9 | N1024 first finite scale？ | 1024 |
| 10 | N768有3次連續finite？ | 否，只有1次 |
| 11 | N1024有3次連續finite？ | 否，只有2次 |
| 12 | N768 classification？ | PERSISTENT_NUMERICAL_INSTABILITY（限8次gate定義） |
| 13 | N1024 classification？ | 同上 |
| 14 | Overflow為1024獨有？ | 否，768也有 |
| 15 | 1024已證實scaler manageable？ | 未達穩定門檻，不能確認 |
| 16 | Peak VRAM？ | 見第4節allocated/reserved |
| 17 | optimizer.step？ | 兩arm皆0 |
| 18 | scaler.step？ | 兩arm皆0 |
| 19 | checkpoint/save？ | GPU attempts0，無檔案 |
| 20 | save guard行為驗證？ | 是，例外且無file |
| 21 | Parameters unchanged？ | 全16次SHA相同 |
| 22 | Real dataset pixels？ | 0 |
| 23 | C4 train/val imgsz？ | 768/768 nominal，stock rect padding已揭露 |
| 24 | H4 train/val imgsz？ | 1024/768 nominal，同padding |
| 25 | Checkpoint-validation parity？ | shared override dry integration VERIFIED |
| 26 | Real validation inference？ | 無 |
| 27 | Locked test/CO2？ | 未讀取／未使用 |
| 28 | F1放行？ | NO |

## 8. 封存與停止

{tests['summary']}；TDD驗證classification、NaN/Inf分離、save/step行為、實際synthetic validator integration，以及原F0/D系列回歸。CPU tests未重跑GPU numerical arms。

兩個GPU subprocess各8次，全部原始telemetry、nonfinite names及失敗attempts都保留。初始化、參數身份、fixture hashes、separate-process順序均通過。F0 unchanged；未來C4/H4正式訓練目錄均不存在。

**停止於F0.1。** 未增加attempts、未手動改initial scale、未改batch、未試960/896/832、未啟動F1。下一階段若要處理fixture判別效度或延長觀察窗，必須另次明確預登錄與授權，不能宣稱本輪已通過。
'''


if __name__=='__main__':run()
