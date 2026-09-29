"""Frozen operational evaluation and evidence-based F1 report; never trains."""
from fractions import Fraction
import json
import subprocess
import sys
import os
from experiments.phase_f1_v2 import (ROOT, PRO, PREFIX, PAIR, OUTPUTS, REPORT, read, save, sha,
    require, verify_execution, verify_frozen, audit_arm, status, now, TEST_FILES, EXEC_FREEZE)


def advancement(control, higher):
    c, h = control['candidate'], higher['candidate']
    cd, hd = control['diagnostics'], higher['diagnostics']
    for x, d in ((c, cd), (h, hd)):
        require(d['very_small']['support'] == 49, 'FAIL_PRIMARY_SUPPORT')
        require(x['tp']+x['fn'] == 241 and x['positive_images'] == 186, 'FAIL_DENOMINATOR')
        require([x['size_recall'][s]['gt'] for s in ('small','medium','large')] == [137,90,14], 'FAIL_SIZE_SUPPORT')
    p = lambda x: Fraction(x['tp'], x['tp']+x['fp']) if x['tp']+x['fp'] else Fraction(0)
    f = lambda x: Fraction(2*x['tp'], 2*x['tp']+x['fp']+x['fn'])
    checks = dict(very_small=hd['very_small']['TP'] >= cd['very_small']['TP']+2,
        small=h['size_recall']['small']['matched'] >= c['size_recall']['small']['matched']-1,
        precision=p(h) >= p(c)-Fraction(1,100), f1=f(h) >= f(c)-Fraction(1,100),
        medium=h['size_recall']['medium']['matched'] >= c['size_recall']['medium']['matched']-1,
        large=h['size_recall']['large']['matched'] >= c['size_recall']['large']['matched'],
        crop=h['crop_complete95_images'] >= c['crop_complete95_images']-1)
    return dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
        delta_very_small_TP=hd['very_small']['TP']-cd['very_small']['TP'],
        meaning='single-seed paired operational advancement criterion, not statistical significance')


def evaluate(arm, token):
    require(read(PAIR/'execution.lock')['token'] == token, 'FAIL_EXECUTION_AUTHORIZATION')
    require(read(PAIR/'pair_validity.json')['PAIRED_FIXED_BUDGET_VALID'] == 'YES', 'FAIL_PAIR_BEFORE_EVAL')
    verify_execution(); verify_frozen()
    audit_arm('C4'); audit_arm('H4')
    from experiments.phase_d1 import verify_validation, diagnostics
    from experiments.phase_c_execution import validate_result_integrity
    from experiments.data_roles import validate_model_input_contract
    from experiments.review_v2 import localization_benchmark as loc
    from experiments.review_v2.isic_fuseg_gate import decide
    from PIL import Image
    import numpy as np
    from ultralytics import YOLO
    from experiments.f1_v2_data_guard import install
    ep, cohort = verify_validation(ROOT)
    require(sha(PRO/f'{PREFIX}_evaluation_protocol.json') == sha(PRO/'D_SEG_SMALL_SAMPLING_V2_evaluation_protocol.json'), 'FAIL_EVAL_PROTOCOL_CHANGED')
    require(ep['prediction']['imgsz'] == 768, 'FAIL_EVAL_SCALE')
    out = OUTPUTS[arm]
    completion = read(out/'training_completion.json')
    require(not (out/'evaluation.lock').exists(), 'FAIL_EVAL_ALREADY_STARTED')
    save(out/'evaluation.lock', dict(at=now(), arm=arm, imgsz=768, checkpoint_sha256=completion['best_checkpoint_sha256'],
        evaluation_protocol_sha256=sha(PRO/f'{PREFIX}_evaluation_protocol.json'), pair_validity_sha256=sha(PAIR/'pair_validity.json')))
    install(out, 'evaluation')
    model = YOLO(str(out/'best.pt'))
    require(model.task == 'segment' and model.names == {0:'Wound'}, 'FAIL_EVAL_MODEL')
    settings = dict(ep['prediction'], iou=ep['operating_point']['nms_iou'])
    for _ in range(3):
        loc.predict_materialized(model, np.zeros((512,512,3), np.uint8), settings)
    (out/'masks').mkdir(); (out/'per_image').mkdir()
    rows = []
    for index, original in enumerate(cohort):
        with Image.open(loc.safe_path(loc.BUNDLE, original['image'])) as image:
            require(image.size == (512,512), 'FAIL_VAL_SIZE')
            bgr = validate_model_input_contract(image.convert('RGB'), source_type='PIL_RGB')
        prediction, masks, ms = loc.predict_materialized(model, bgr, settings)
        row = loc.assess(original, prediction, masks, ep['operating_point']['confidence'])
        scores = prediction.boxes.conf.cpu().numpy()
        keep = np.flatnonzero(scores >= ep['operating_point']['confidence'])
        maskrel = f'masks/{index:03d}.npz'
        with (out/maskrel).open('xb') as file:
            np.savez_compressed(file, masks=masks, confidences=scores, boxes=prediction.boxes.xyxy.cpu().numpy(), retained_indices=keep)
        matched = {p['gt'] for p in row['pairs']}
        row.update(sample_id=original['image_id'], num_gt_instances=len(row['gt_boxes']), num_predictions=len(row['pred_boxes']),
            per_instance_gt=[dict(index=i, bbox=box, size_group=loc.size_name(box), matched=i in matched) for i,box in enumerate(row['gt_boxes'])],
            prediction_mask_reference=maskrel, prediction_mask_sha256=sha(out/maskrel), retained_mask_indices=keep.tolist(),
            floor_prediction_count=len(scores), floor_confidences=scores.tolist(), floor_bboxes=prediction.boxes.xyxy.cpu().numpy().tolist(),
            retained_gt_wound_pixels=int(round(row['crop_coverage']*row['gt_pixels'])) if row['gt_pixels'] else 0, inference_latency_ms=ms)
        save(out/'per_image'/f'{index:03d}.json', row); rows.append(row)
    save(out/'per_image_predictions.json', rows)
    summary = loc.summarize(rows)
    integrity = validate_result_integrity(rows, summary, [r['image_id'] for r in cohort])
    latency = loc.latency_summary([r['inference_latency_ms'] for r in rows])
    decision = decide(summary, latency)
    performance = {k: decision['checks'][k] for k in ep['acceptance_gate']['minima_percent']}
    result = dict(candidate=summary, diagnostics=diagnostics(rows), integrity=integrity, latency=latency,
        historical_development_gate='PARTIAL_LATENCY_UNVERIFIED' if all(performance.values()) else 'FAIL',
        performance_checks=performance, latency_comparability='NOT_VERIFIED; descriptive only',
        checkpoint_sha256=sha(out/'best.pt'), predictions_sha256=sha(out/'per_image_predictions.json'),
        test_images_used=0, CO2Wounds_used=False, final_operational_imgsz=768)
    save(out/'final_evaluation.json', result)


def csv_write(name, rows):
    from experiments.phase_d1 import csv_output
    csv_output(PAIR/name, rows)


def smallest_bin(diagnostics):
    result = diagnostics['small_bins']['<0.10%']
    require(result['support'] == 27, 'FAIL_SMALLEST_BIN_SUPPORT')
    return result


def next_view(gate, completions, results):
    if gate['status'] == 'PASS':
        return ('本次只支持seed42配對development改善；下一步最合理是另立多種子配對確認規範，保留同一共同validation尺度與原gate，'
                '並事先固定seed與整體決策。不直接替換App、不宣稱臨床泛化，也不開啟external test。尚未授權或啟動。')
    failed = ', '.join(k for k,v in gate['checks'].items() if not v)
    return ('保留負結果，未通過項目：'+failed+'。下一步應先利用本次已保存prediction/matching做配對錯誤分析，'
            '區分very-small新增TP是否伴隨其他尺寸或crop退化；不重推論、不改threshold、不試960或H4@1024。'
            '待分析確認失敗結構後再預登錄單一介入；本次不自動執行。')


def verify_saved_evaluation(arm):
    import numpy as np
    from experiments.phase_c_execution import validate_result_integrity
    out = OUTPUTS[arm]
    result = read(out/'final_evaluation.json'); rows = read(out/'per_image_predictions.json')
    require(sha(out/'per_image_predictions.json') == result['predictions_sha256'], 'FAIL_SAVED_PREDICTIONS')
    ids = [r['sample_id'] for r in read(PRO/'ISIC_FUSEG_COLORFIX_V1_validation_manifest.json')['samples']]
    integrity = validate_result_integrity(rows, result['candidate'], ids)
    for i, row in enumerate(rows):
        require(read(out/'per_image'/f'{i:03d}.json') == row, 'FAIL_PER_IMAGE_RECORD')
        require(sha(out/row['prediction_mask_reference']) == row['prediction_mask_sha256'], 'FAIL_SAVED_MASK_HASH')
        with np.load(out/row['prediction_mask_reference'], allow_pickle=False) as masks:
            require(masks['masks'].shape == (row['floor_prediction_count'],512,512), 'FAIL_MASK_SHAPE')
            require(masks['retained_indices'].tolist() == row['retained_mask_indices'], 'FAIL_MASK_RETAINED')
    return integrity


def finish():
    from experiments.phase_d1 import diagnostics
    training = {a: read(p/'training_completion.json') for a,p in OUTPUTS.items()}
    evaluations = {a: read(p/'final_evaluation.json') for a,p in OUTPUTS.items()}
    audits = {a: audit_arm(a) for a in OUTPUTS}
    # Audit both common validator rules and observed batch shape, not training imgsz.
    cv, hv = [audits[a]['validation'] for a in ('C4','H4')]
    for key in ('val_nominal_imgsz','dataset_nominal_imgsz','first_batch_shape','shared_override'):
        require(cv[key] == hv[key], 'FAIL_CHECKPOINT_SELECTION_PARITY')
    for key in ('imgsz','conf','iou','rect','half','max_det','agnostic_nms','augment'):
        require(cv['validator_args'][key] == hv['validator_args'][key], 'FAIL_VALIDATOR_ARGS_PARITY')
    saved = {a: verify_saved_evaluation(a) for a in OUTPUTS}
    gate = advancement(evaluations['C4'], evaluations['H4'])
    validity = read(PAIR/'pair_validity.json')
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_ADDOPTS='', PYTEST_PLUGINS='')
    # Historical F02 assertions that future outputs do not yet exist apply only at F02/preflight.
    proc = subprocess.run([sys.executable,'-B','-m','pytest',*TEST_FILES,'-q','-k','not future_outputs_absent'],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
    save(PAIR/'postflight_tests.json', dict(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
        excluded_scope='F02 future-output-absence assertions only; F1 output existence separately audited'))
    require(proc.returncode == 0, 'FAIL_POSTFLIGHT_TESTS')
    verify_frozen(); verify_execution()
    outcome = dict(PHASE_F1_STATUS='COMPLETE', PAIRED_FIXED_BUDGET_VALID='YES',
        RUNTIME_NUMERICAL_SAFETY_CONTROL='PASS', RUNTIME_NUMERICAL_SAFETY_HIGHER_SCALE='PASS',
        HIGHER_SCALE_RESEARCH_GATE=gate['status'], AMP_UPDATE_COUNT_IMBALANCE_OBSERVED=validity['AMP_UPDATE_COUNT_IMBALANCE_OBSERVED'],
        NEGATIVE_RESULT_PRESERVED='YES' if gate['status'] == 'FAIL' else 'NOT_APPLICABLE',
        LOCKED_TEST_USED=False, CO2Wounds_used=False, EXTERNAL_TEST_USED=False, test_images_used=0,
        MULTI_SEED_AUTHORIZED='NO', APP_MODEL_REPLACEMENT_AUTHORIZED='NO', STOP_AFTER_F1=True,
        training=training, evaluations=evaluations, gate=gate, saved_integrity=saved, runtime_integrity=audits,
        final_operational_evaluation_performed=True, H4_at1024_evaluation=False,
        next_research_view=next_view(gate,training,evaluations), finished_at=now())
    historical = read(ROOT/'experiments/results/isic_fuseg_colorfix_v1/result.json')
    hd = diagnostics(read(ROOT/'experiments/results/isic_fuseg_colorfix_v1/per_image_predictions.json'))
    d42 = read(ROOT/'experiments/results/d_seg_small_sampling_v2_control/final_evaluation.json')
    groups = [(historical['candidate'],hd),(d42['candidate'],d42['diagnostics'])] + [(evaluations[a]['candidate'],evaluations[a]['diagnostics']) for a in ('C4','H4')]
    metrics = [('Precision',lambda x,d:x['precision']),('Recall',lambda x,d:x['recall']),('F1',lambda x,d:x['f1']),
        ('Very-small Recall',lambda x,d:d['very_small']['recall']),('<0.10% Recall',lambda x,d:smallest_bin(d)['recall']),
        *[(s+' Recall',lambda x,d,s=s:x['size_recall'][s]['recall']) for s in ('small','medium','large')],
        ('Crop complete',lambda x,d:x['crop_complete95_fraction']),('TP',lambda x,d:x['tp']),('FP',lambda x,d:x['fp']),
        ('FN',lambda x,d:x['fn']),('No ROI',lambda x,d:x['positive_without_roi'])]
    table = ['| Metric | Historical Phase C | D2 Fresh C42 | Fresh C4 | Fresh H4 | H4−C4 |','|---|---:|---:|---:|---:|---:|']
    for label, fn in metrics:
        values = [fn(x,d) for x,d in groups]
        table.append('| '+label+' | '+' | '.join(f'{v:.6f}' for v in [*values,values[3]-values[2]])+' |')
    numeric_keys = ['completed_epochs','scheduled','applied','skipped','max_consecutive_skips','minimum_scaler',
        'nonfinite_loss_events','nonfinite_parameter_events','OOM','runtime_exceptions','unknown','peak_GPU_allocated','peak_GPU_reserved']
    numerical = [dict(metric=k,C4=training['C4'][k],H4=training['H4'][k]) for k in numeric_keys]
    csv_write('numerical_safety_comparison.csv', numerical)
    csv_write('training_telemetry_comparison.csv',[dict(arm=a,**{k:training[a][k] for k in ('completed_epochs','scheduled','applied','skipped','best_epoch','last_epoch')}) for a in OUTPUTS])
    size_rows = []
    for name in evaluations['C4']['diagnostics']['small_bins']:
        c,h = [evaluations[a]['diagnostics']['small_bins'][name] for a in ('C4','H4')]
        size_rows.append(dict(bin=name,support=c['support'],C4_TP=c['TP'],H4_TP=h['TP'],delta_TP=h['TP']-c['TP'],C4_recall=c['recall'],H4_recall=h['recall']))
    csv_write('size_bin_comparison.csv',size_rows)
    gt_rows=[]
    for group in ('single_GT','multi_GT'):
        for metric, value in evaluations['C4']['diagnostics'][group].items():
            if isinstance(value,(int,float)):
                gt_rows.append(dict(group=group,metric=metric,C4=value,H4=evaluations['H4']['diagnostics'][group][metric]))
    csv_write('single_multi_gt_comparison.csv',gt_rows)
    numeric_table=['| Runtime metric | C4 | H4 |','|---|---:|---:|']+[f"| {r['metric']} | {r['C4']} | {r['H4']} |" for r in numerical]
    text = '# Phase F1 V2 — Seed42 配對訓練結果\n\n'+ '\n'.join(table)+'\n\n比例以0–1表示；H4−C4為比例差，非百分點。只有Fresh C4 vs Fresh H4為因果比較，歷史兩欄REFERENCE_ONLY。\n\n'+ '\n'.join(numeric_table)
    text += '\n\n## Gate與尺寸／single-multi diagnostics\n\n```json\n'+json.dumps(dict(gate=gate,validity=validity,size_bins=size_rows,single_multi=gt_rows),ensure_ascii=False,indent=2)+'\n```\n'
    text += '\n## 下一步見解\n\n'+outcome['next_research_view']+'\n\n'
    text += ('Under this preregistered single-seed paired development experiment, training at nominal imgsz 1024 improved observed very-small-wound recall relative to the fresh imgsz-768 control while satisfying the predefined safety gates.\n\n' if gate['status']=='PASS' else '本次未通過預定higher-scale advancement gate，完整保留負結果。\n\n')
    text += '這不是統計顯著性、臨床驗證或泛化證明；未新增p-value／CI。Latency僅描述，historical development gate若四項性能均達標仍標PARTIAL_LATENCY_UNVERIFIED。\n\n'
    text += 'F0／F0.1 not superseded as observations：原synthetic overflow與FAIL保留；只被superseded as V2 readiness-gate methodology。GradScaler<1是本研究engineering safety stop，不是PyTorch要求。兩組stock AMP、batch4、300 epochs、相同原始labels與共享validator；val為nominal768，actual shape見runtime JSON。\n\n'
    text += 'Scheduled opportunities相同不等於realized applied updates相同；差異與AMP skips均見表，不補跑。No locked test／CO2／external／multi-seed／App replacement；H4@1024 final eval未執行。\n\n'
    answers = completion_answers(training,evaluations,gate,outcome)
    text += '## 33項完成問題\n\n'+'\n\n'.join(f'{i}. {q}：{a}' for i,(q,a) in enumerate(answers,1))+'\n\nSTOP AFTER F1。\n'
    save(PAIR/'paired_comparison.json',outcome)
    for path in (PAIR/'paired_comparison.md',REPORT):
        with path.open('x',encoding='utf-8') as f: f.write(text)
    status('COMPLETE',research_gate=gate['status'],report=str(REPORT),next_research_view=outcome['next_research_view'])


def completion_answers(t,e,g,o):
    return [('同一runtime safety source/hash',str(t['C4']['safety_adapter_sha256']==t['H4']['safety_adapter_sha256'])),
        ('C4 epochs',t['C4']['completed_epochs']),('H4 epochs',t['H4']['completed_epochs']),
        ('C4 scheduled/applied/skipped',[t['C4'][k] for k in ('scheduled','applied','skipped')]),
        ('H4 scheduled/applied/skipped',[t['H4'][k] for k in ('scheduled','applied','skipped')]),
        ('AMP imbalance',o['AMP_UPDATE_COUNT_IMBALANCE_OBSERVED']),
        ('最大連續skips',{a:t[a]['max_consecutive_skips'] for a in t}),('最低scaler',{a:t[a]['minimum_scaler'] for a in t}),
        ('Nonfinite loss',{a:t[a]['nonfinite_loss_events'] for a in t}),('Nonfinite parameters',{a:t[a]['nonfinite_parameter_events'] for a in t}),
        ('OOM',{a:t[a]['OOM'] for a in t}),('Unknown',{a:t[a]['unknown'] for a in t}),
        ('C4 train/val nominal','768/768'),('H4 train/val nominal','1024/768'),('SharedValidation768實際生效','YES; runtime evidence保存'),
        ('Actual anchor order','兩組300×771與凍結順序一致'),('C4 Very-small TP/49',e['C4']['diagnostics']['very_small']),
        ('H4 Very-small TP/49',e['H4']['diagnostics']['very_small']),('Very-small ΔTP',g['delta_very_small_TP']),
        ('<0.10%',{a:smallest_bin(e[a]['diagnostics']) for a in e}),
        ('Small safety',g['checks']['small']),('Precision safety',g['checks']['precision']),('F1 safety',g['checks']['f1']),
        ('Medium/Large safety',{k:g['checks'][k] for k in ('medium','large')}),('Crop safety',g['checks']['crop']),
        ('Research gate',g['status']),('Historical development gate',{a:e[a]['historical_development_gate'] for a in e}),
        ('H4@1024 final eval','NO'),('Locked test','NO'),('CO2/external','NO'),('Multi-seed','NO'),('App replacement','NO'),('下一研究決策',o['next_research_view'])]


def interrupted(error):
    records = {}
    for arm, out in OUTPUTS.items():
        if (out/'interruption.json').exists():
            records[arm] = read(out/'interruption.json')
        elif (out/'training_completion.json').exists():
            records[arm] = read(out/'training_completion.json')
        else:
            records[arm] = {'started':out.exists()}
    try:
        verify_frozen(); verify_execution()
        protected = {'status':'PASS', 'historical_data_initialization_and_execution_unchanged':True}
    except Exception as exc:
        protected = {'status':'FAIL', 'error':str(exc)}
    value = dict(PHASE_F1_STATUS='INVALID_OR_INTERRUPTED', PAIRED_FIXED_BUDGET_VALID='NO',
        error=str(error), partial=records, H4_TRAINING_STARTED=OUTPUTS['H4'].exists(),
        FINAL_OPERATIONAL_EVALUATION_PERFORMED=any((p/'evaluation.lock').exists() for p in OUTPUTS.values()),
        LOCKED_TEST_USED=False, CO2Wounds_used=False, EXTERNAL_TEST_USED=False,
        retry_allowed=False, MULTI_SEED_AUTHORIZED='NO', APP_MODEL_REPLACEMENT_AUTHORIZED='NO', STOP_AFTER_F1=True,
        HIGHER_SCALE_RESEARCH_GATE='NOT_EVALUATED_INVALID_PAIR', postflight_integrity=protected)
    save(PAIR/'interruption.json',value)
    text='# Phase F1 V2 中斷／無效報告\n\n未形成完整有效的配對比較，不判定1024優劣，不重跑、不resume。\n\n```json\n'+json.dumps(value,ensure_ascii=False,indent=2)+'\n```\n\n'
    text+='## 下一步見解\n\n先根據已保存日誌辨識中斷屬於數值、資源、早停還是工程接線問題；本次不調參、不補跑。只有另次預登錄與明確授權，才處理後续實驗。若C4本身中斷，不能歸因於1024。F0/F0.1/F0.2原結果保持不變。\n'
    partial = {a:r.get('partial', r) for a,r in records.items()}
    def metric(key):
        return {a:partial[a].get(key,'NOT_OBSERVED') for a in records}
    binding = {a:read(p/'runtime_binding.json') if (p/'runtime_binding.json').exists() else 'NOT_REACHED'
               for a,p in OUTPUTS.items()}
    observations = [
        ('共同adapter source/hash', {a:sha(ROOT/'experiments/f1_v2_safety.py') if r.get('started',True) else 'NOT_STARTED' for a,r in records.items()}),
        ('C4 epochs',records['C4'].get('completed_epochs','NOT_STARTED')),
        ('H4 epochs',records['H4'].get('completed_epochs','NOT_STARTED')),
        ('C4 scheduled/applied/skipped',{k:partial['C4'].get(k,'NOT_OBSERVED') for k in ('scheduled','applied','skipped')}),
        ('H4 scheduled/applied/skipped',{k:partial['H4'].get(k,'NOT_OBSERVED') for k in ('scheduled','applied','skipped')}),
        ('AMP imbalance','UNDETERMINED_INCOMPLETE_PAIR'),('最大連續skips',metric('max_consecutive_skips')),
        ('最低scaler',metric('minimum_scaler')),('Nonfinite loss',metric('nonfinite_loss_events')),
        ('Nonfinite parameters',metric('nonfinite_parameter_events')),('OOM',metric('OOM')),('Unknown',metric('unknown')),
        ('C4實際train/val nominal',binding['C4']),('H4實際train/val nominal',binding['H4']),
        ('SharedValidation768 runtime',binding),('完整anchor parity','NOT_ESTABLISHED_INCOMPLETE_PAIR')]
    observations += [(q,'NOT_EVALUATED_INVALID_PAIR') for q in
        ('C4 Very-small TP/49','H4 Very-small TP/49','Very-small delta','<0.10% C4/H4',
         'Small safety','Precision safety','F1 safety','Medium/Large safety','Crop safety','Research gate','Historical development gate')]
    observations += [('H4@1024 final evaluation',False),('Locked test',False),('CO2/external',False),
        ('Multi-seed',False),('App replacement',False),('下一研究決策','停止；分析已保存中斷證據，不自動重跑')]
    text+='\n## 33項完成問題（中斷時未觀察者明確標記）\n\n'+'\n\n'.join(
        f'{i}. {q}：{json.dumps(a, ensure_ascii=False)}' for i,(q,a) in enumerate(observations,1))+'\n'
    if not REPORT.exists():
        with REPORT.open('x',encoding='utf-8') as f:f.write(text)
    status('INVALID_OR_INTERRUPTED',error=str(error),report=str(REPORT))
