"""Recovery-only reporting; original F1 report/results never written here."""
import json
from datetime import datetime
from experiments.phase_f12 import (ROOT, PRE, PAIR, OUT, C4, F11, REPORT, SCHEMA, read, save, sha,
    require, now, status, verify_execution, bind_runtime)

DISCLOSURE = ('The original higher-scale execution was interrupted before completing the fixed training budget and was excluded '
    'from performance comparison. A separately preregistered fresh higher-scale replacement execution was paired with the already '
    'completed, unevaluated frozen 768 control.')
LIMITATION = ('C4 and H4-R were not executed as one uninterrupted fresh temporal pair. '
    'This is an asymmetric replacement design and a single-seed recovery paired development experiment; '
    'no statistical significance, population confidence interval or clinical generalization is claimed.')


def answers(stage,data):
    t=data.get('training',{}).get('H4-R',{})
    e=data.get('evaluations',{})
    gate=data.get('gate',{})
    pending='NOT_OBSERVED / NOT_EVALUATED'
    def tp(arm): return e.get(arm,{}).get('diagnostics',{}).get('very_small',pending)
    def sub(arm): return e.get(arm,{}).get('diagnostics',{}).get('small_bins',{}).get('<0.10%',pending)
    def safety(name): return gate.get('checks',{}).get(name,pending)
    return [
        ('Original F1狀態','INVALID_OR_INTERRUPTED'),('Original H4始終未evaluation','YES; engineering evidence only'),
        ('C4 hash與F1.1一致',data.get('historical_integrity','PASS_AT_PREFLIGHT; rechecked before execution')),
        ('C4 epochs',300),('H4-R fresh initialization','ISIC original hash; resume=false; runtime_binding.json required'),
        ('是否使用original H4 checkpoint','NO'),('H4-R epochs',t.get('completed_epochs',pending)),
        ('H4-R scheduled/applied/skipped/unknown',{k:t.get(k,pending) for k in ['scheduled','applied','skipped','unknown']}),
        ('最大連續skips',t.get('max_consecutive_skips',pending)),('最低scaler',t.get('minimum_scaler',pending)),
        ('Nonfinite loss',t.get('nonfinite_loss_events',pending)),('Nonfinite parameters',t.get('nonfinite_parameter_events',pending)),
        ('OOM',t.get('OOM',pending)),('第二次interruption',data.get('interruption',False if stage=='COMPLETE' else pending)),
        ('300 epochs actual anchor parity','PASS' if stage=='COMPLETE' else pending),
        ('SharedValidation768實際生效',data.get('shared_validation',pending)),
        ('H4-R best/last完整','PASS' if stage=='COMPLETE' else pending),
        ('Environment parity',read(PRE/'environment_parity.json')['ENVIRONMENT_PARITY']),
        ('Recovery pair有效',data.get('RECOVERY_PAIR_VALID','NOT_YET_ESTABLISHED')),
        ('C4 eval只在pair valid之後',data.get('evaluation_sequence','NOT_PERFORMED')),
        ('H4-R eval只在pair valid之後',data.get('evaluation_sequence','NOT_PERFORMED')),
        ('C4 Very-small TP/49',tp('C4')),('H4-R Very-small TP/49',tp('H4-R')),
        ('Very-small delta TP',gate.get('delta_very_small_TP',pending)),('<0.10%兩組',{'C4':sub('C4'),'H4-R':sub('H4-R')}),
        ('Small safety',safety('small')),('Precision safety',safety('precision')),('F1 safety',safety('f1')),
        ('Medium safety',safety('medium')),('Large safety',safety('large')),('Crop safety',safety('crop')),
        ('Recovery research gate',gate.get('status','NOT_EVALUATED')),
        ('Historical development gate',{a:r['historical_development_gate'] for a,r in e.items()} or pending),
        ('H4-R@1024 final eval','NO'),('Locked test','NO'),('CO2/external','NO'),('Multi-seed','NO'),('App replacement','NO'),
        ('原H4用途','INTERRUPTED_ENGINEERING_EVIDENCE; no performance endpoints'),
        ('下一階段建議',data.get('next_view','等待本次唯一H4-R執行；不自動擴充實驗。'))]


def write_report(stage,data):
    text = '# Phase F1.2 — Fresh H4 Replacement Recovery\n\n'
    text += f'更新時間（UTC）：{now()}\n\nPHASE_F12_STATUS = {stage}\n\n'
    text += '## 不可變歷史與研究設計\n\n'+DISCLOSURE+'\n\n'+LIMITATION+'\n\n'
    text += '正式採Option B；C4不重訓、不重選checkpoint，原H4不續訓、不評估。原F1固定預算pair維持無效；F1.1鑑識結果不變。\n\n'
    text += '## 配方與前置條件\n\n'
    text += ('H4-R：YOLO11m-seg / Wound，seed42，train1024，batch4，300 epochs，patience80；ISIC原初始化，fresh optimizer/scaler。'
        '771張原始FUSeg training、965 polygon instances，191 development validation。所有loss、augmentation、AdamW、AMP、accumulation與16連續skip硬停均沿用封版規範。'
        'checkpoint選擇用SharedValidation768，預期actual shape遵循stock rect/pad，不把nominal768誤稱固定768×768。\n\n'
        '通過300epochs/3741opportunities、unknown0、checkpoint/CSV/TensorBoard/telemetry完整性後，才建立recovery pair並依序做C4@768、H4-R@768。'
        '評估不寫入原C4目錄；全部放在新recovery pair/eval_C4與eval_H4_R。conf .10、floor .01、NMS .70、match .50、crop margin15%均不變。\n\n'
        '原訓練/安全/validator/evaluator來源不修改；新的協調層只重綁檔案路徑與授權，研究語義沿用原凍結函數。'
        'preflight 的CPU小型合成測試不是新增research smoke，不載入傷口影像訓練；真實runtime binding會在第一批前再次檢查。\n\n')
    parity = read(PRE/'environment_parity.json')
    text += '## 環境揭露\n\n'+parity['ENVIRONMENT_PARITY']+'。'+parity['disclosure']+'\n\n'
    text += '原F1缺OS/driver/OpenCV等完整欄位，不補造歷史值；新版記錄current值，已記錄的關鍵套件版本及active training source hashes均一致。\n\n'
    text += '## 40項研究問題\n\n'
    text += '\n\n'.join(f'{i}. {q}：{json.dumps(a,ensure_ascii=False)}' for i,(q,a) in enumerate(answers(stage,data),1))
    if data.get('evaluations'):
        text += '\n\n## 已完成配對數據\n\n```json\n'+json.dumps(data,ensure_ascii=False,indent=2)+'\n```\n'
    elif data.get('error'):
        text += '\n\n## 停止原因\n\n'+data['error']+'\n'
    text += '\n\nSTOP AFTER F1.2；無H4-R2、resume、multi-seed、external test、App替換或自動下一實驗。\n'
    REPORT.write_text(text,encoding='utf-8')
    return text


def finish():
    verify_execution()
    old=bind_runtime()
    import experiments.f1_v2_results as fr
    fr.OUTPUTS={'C4':PAIR/'eval_C4','H4':PAIR/'eval_H4_R'}
    saved={arm:fr.verify_saved_evaluation(key) for arm,key in [('C4','C4'),('H4-R','H4')]}
    e={arm:read(path/'final_evaluation.json') for arm,path in [('C4',PAIR/'eval_C4'),('H4-R',PAIR/'eval_H4_R')]}
    t={'C4':read(C4/'training_completion.json'),'H4-R':read(OUT/'training_completion.json')}
    gate=fr.advancement(e['C4'],e['H4-R'])
    runtime={a:old.audit_arm(k) for a,k in [('C4','C4'),('H4-R','H4')]}
    # Never read original interrupted H4 performance; only prior forensic engineering summaries.
    historical=read(F11/'telemetry_integrity.json')['H4']['optimizer']
    from experiments.phase_d1 import csv_output
    keys=['completed_epochs','scheduled','applied','skipped','unknown','max_consecutive_skips','minimum_scaler','nonfinite_loss_events','nonfinite_parameter_events','OOM','peak_GPU_allocated','peak_GPU_reserved']
    interrupted_runtime={k:historical.get(k,'NOT_SAVED') for k in keys}
    interrupted_runtime.update(completed_epochs=297,scheduled=historical['observed_scheduled'],role='INTERRUPTED_ENGINEERING_EVIDENCE; includes partial epoch298')
    runtime_rows=[dict(metric=k,C4=t['C4'].get(k),Original_H4_engineering=interrupted_runtime.get(k),H4_R=t['H4-R'].get(k)) for k in keys]
    csv_output(PAIR/'numerical_safety_comparison.csv',runtime_rows)
    csv_output(PAIR/'training_telemetry_comparison.csv',[dict(arm=a,**{k:r.get(k) for k in keys}) for a,r in t.items()]+[dict(arm='Original H4 (partial)',**{k:interrupted_runtime.get(k) for k in keys})])
    duration=(datetime.fromisoformat(read(PRE/'training_process_exit.json')['at'])-datetime.fromisoformat(read(PRE/'H4_R_process.json')['at'])).total_seconds()
    save(PAIR/'runtime_duration_disclosure.json',dict(H4_R_worker_wall_seconds=duration,
        scope='Whole worker lifetime including setup and stock validation, not GPU kernel-only timing',
        C4_comparable_duration='NOT_RECONSTRUCTED',original_H4_duration='INTERRUPTED; not a fixed-budget duration comparison'))
    bins=[]
    for name,c in e['C4']['diagnostics']['small_bins'].items():
        h=e['H4-R']['diagnostics']['small_bins'][name]
        bins.append(dict(bin=name,support=c['support'],C4_TP=c['TP'],H4_R_TP=h['TP'],delta=h['TP']-c['TP']))
    csv_output(PAIR/'size_bin_comparison.csv',bins)
    groups=[]
    for group in ['single_GT','multi_GT']:
        for metric,c in e['C4']['diagnostics'][group].items():
            if isinstance(c,(int,float)): groups.append(dict(group=group,metric=metric,C4=c,H4_R=e['H4-R']['diagnostics'][group][metric]))
    csv_output(PAIR/'single_multi_gt_comparison.csv',groups)
    disclosure=dict(text=DISCLOSURE,limitation=LIMITATION,design='asymmetric replacement execution',original_H4_performance_used=False,
        environment=read(PRE/'environment_parity.json'),original_F1_status='INVALID_OR_INTERRUPTED')
    save(PAIR/'recovery_design_disclosure.json',disclosure)
    outcome=dict(PHASE_F12_STATUS='COMPLETE',RECOVERY_OPTION='B',RECOVERY_PAIR_VALID='YES',
        C4_SOURCE='ORIGINAL_COMPLETED_F1_CONTROL',H4_SOURCE='FRESH_REPLACEMENT_EXECUTION',
        ORIGINAL_H4_PERFORMANCE_USED='NO',RECOVERY_RESEARCH_GATE=gate['status'],
        MULTI_SEED_AUTHORIZED='NO',APP_MODEL_REPLACEMENT_AUTHORIZED='NO',EXTERNAL_TEST_AUTHORIZED='NO',
        FINAL_OPERATIONAL_EVALUATION_PERFORMED=True, test_images_used=0,LOCKED_TEST_USED=False,CO2Wounds_used=False,EXTERNAL_TEST_USED=False,
        H4_R_final_eval1024=False,training=t,evaluations=e,gate=gate,saved_integrity=saved,
        historical_integrity='PASS',shared_validation=runtime,interruption=False,
        evaluation_sequence='RECOVERY_PAIR_VALID -> C4@768 -> H4-R@768 -> gate',
        next_view=fr.next_view(gate,t,e),STOP_AFTER_F12=True,at=now())
    verify_execution()
    save(PAIR/'paired_comparison.json',outcome)
    text=write_report('COMPLETE',outcome)
    with (PAIR/'paired_comparison.md').open('x',encoding='utf-8') as stream: stream.write(text)
    status('COMPLETE',research_gate=gate['status'],report=str(REPORT))


def interrupted(error):
    partial={}
    if (OUT/'interruption.json').exists(): partial=read(OUT/'interruption.json')
    elif (OUT/'training_completion.json').exists(): partial=read(OUT/'training_completion.json')
    try: verify_execution(); historical='PASS'
    except Exception as exc: historical='FAIL: '+str(exc)
    eval_started=any((PAIR/n/'evaluation.lock').exists() for n in ['eval_C4','eval_H4_R'])
    value=dict(PHASE_F12_STATUS='INVALID_OR_INTERRUPTED',RECOVERY_PAIR_VALID='NO',
        FINAL_OPERATIONAL_EVALUATION_PERFORMED=eval_started, evaluation_attempt_started=eval_started,
        HIGHER_SCALE_EFFECT_CONCLUSION='NOT_AVAILABLE',NEW_RETRY_AUTHORIZED='NO',
        ORIGINAL_F1_STATUS='INVALID_OR_INTERRUPTED',historical_integrity=historical,
        error=str(error),interruption=True,partial=partial,at=now(),STOP_AFTER_F12=True)
    if partial:
        value['training']={'H4-R':dict(partial.get('partial',partial),
            completed_epochs=partial.get('completed_epochs','UNKNOWN'))}
    save(PRE/'interruption.json',value)
    write_report('INVALID_OR_INTERRUPTED',value)
    status('INVALID_OR_INTERRUPTED',error=str(error),report=str(REPORT))
