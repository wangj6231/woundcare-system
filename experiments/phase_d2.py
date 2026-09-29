"""Frozen multi-seed paired confirmation. One attempt per new seed/arm only."""
from __future__ import annotations
import argparse
from copy import deepcopy
from fractions import Fraction
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import types
import uuid

from experiments import phase_d1 as d1
from experiments import paired_sampling_multiseed as seeded

ROOT=Path(__file__).resolve().parents[1]
PREFIX='D_SEG_SMALL_SAMPLING_V2_D2'
MASTER=Path('experiments/results/d_seg_small_sampling_v2_multiseed_summary')
REPORT=Path('docs/PHASE_D2_MULTI_SEED_SMALL_SAMPLING_CONFIRMATION_20260923.md')
NEW_SEEDS=(123,3407,2026,999)
ALL_SEEDS=(42,)+NEW_SEEDS
REQUEST=Path('C:/Users/milo9/.codex/attachments/e4ee94bc-9766-4942-99df-e052166bd74e/貼上的文字.txt')
SOURCE_FILES=['experiments/phase_d2.py','experiments/paired_sampling_multiseed.py','tests/test_phase_d2.py',
 'docs/PHASE_D2_PREREGISTRATION_20260923.md']
require=d1.require
read=d1.d0.read
sha=d1.d0.sha
write=d1.write


def prefix(seed):return f'{PREFIX}_seed{seed}'
def artifact(root,suffix):return root/'experiments/protocols'/f'{PREFIX}_{suffix}.json'
def seed_artifact(root,seed,suffix):return root/'experiments/protocols'/f'{prefix(seed)}_{suffix}.json'


def output_guard(root):
    require(not (root/MASTER).exists() and not (root/REPORT).exists(),'FAIL_OUTPUT_ALREADY_EXISTS')
    for seed in NEW_SEEDS:
        for rel in seeded.outputs(seed).values():require(not (root/rel).exists(),'FAIL_OUTPUT_ALREADY_EXISTS: '+rel)


def seed42_snapshot(root):
    paths=[]
    for rel in [*d1.paired.OUTPUTS.values(),d1.PAIR]:
        paths.extend(p for p in (root/rel).rglob('*') if p.is_file())
    paths += [root/d1.REPORT,root/'experiments/phase_d1.py',root/'tests/test_phase_d1.py']
    return {str(p):sha(p) for p in sorted(set(paths))}


def seed42_result(root,verify=True):
    c,s,_=d1.validate_freeze(root)
    lock=read(root/d1.PAIR/'execution.lock')
    require(sha(root/'experiments/phase_d1.py')==lock['driver_sha256'],'FAIL_D1_DRIVER_MUTATED')
    pair=read(root/d1.PAIR/'pair_summary.json')
    require(pair['PHASE_D1_STATUS']=='COMPLETE' and pair['PAIR_FIXED_BUDGET_VALID']=='YES','FAIL_SEED42_PREREQUISITE')
    for arm,cfg in [('C',c),('S',s)]:
        if verify:d1.verify_saved_arm(root,arm,cfg)
        out=root/d1.paired.OUTPUTS[arm]
        require(pair['training'][arm]==read(out/'training_completion.json'),'FAIL_SEED42_COMPLETION')
        require(pair['evaluations'][arm]==read(out/'final_evaluation.json'),'FAIL_SEED42_EVALUATION')
    return {'seed':42,'valid':True,'intervention_delivered':True,'origin':'IMMUTABLE_D1_READ_ONLY',
      'training':pair['training'],'evaluations':pair['evaluations'],'paired_gate':pair['paired_gate'],
      'amp':pair['amp'],'seed_gate':pair['SMALL_SAMPLING_RESEARCH_GATE']}


def config_seed_only(base,candidate,seed):
    normalized=deepcopy(candidate)
    require(candidate['training_args']['seed']==candidate['sampling']['seed']==seed,'FAIL_SEED_IDENTITY')
    normalized['training_args']['seed']=base['training_args']['seed']
    normalized['sampling']['seed']=base['sampling']['seed']
    normalized['sampling']['seed_policy']=base['sampling']['seed_policy']
    for key in ['output_path','experiment_id','runner']:normalized[key]=base[key]
    require(normalized==base,'FAIL_TRAINING_DIFF_BEYOND_SEED')
    return True


def build(root):
    output_guard(root)
    require(not list((root/'experiments/protocols').glob(PREFIX+'*')),'D2_PROTOCOL_ALREADY_EXISTS')
    c,s,_=d1.validate_freeze(root);seed42_result(root)
    d1.verify_data(root,c);d1.verify_validation(root)
    snap=seed42_snapshot(root)
    extra=read(root/d1.PAIR/'preflight.json')['additional_protected'];d1.verify_snapshot(extra)
    protocol={'experiment_id':PREFIX,'status':'FROZEN_BEFORE_EXECUTION','new_seed_order':list(NEW_SEEDS),
      'all_paired_seeds':list(ALL_SEEDS),'seed42_policy':'READ_ONLY; no retraining, reevaluation or duplicate output',
      'within_seed_order':['TRAIN_C','TRAIN_S','PAIR_VALIDITY','EVALUATE_C','EVALUATE_S'],
      'unit':'5 paired random-seed experiments; not 10 independent models; not patients',
      'primary':'within-seed S minus C Small Recall; support 137',
      'initialization':c['initialization'],'train_images':771,'development_validation_images':191,
      'sampling_weights':{'A':2.0,'B':1.5,'C':1.0},'training_args_except_seed':{k:v for k,v in c['training_args'].items() if k!='seed'},
      'new_training_semantics_vs_D1':['training_args.seed','sampling.seed'],
      'implementation':'Private verified D1/D0.1 module namespaces; same source bytes; rebind sampler seed and output/config routing only',
      'seeded_runner_sha256':sha(root/'experiments/paired_sampling_multiseed.py'),
      'base_runner_sha256':seeded.BASE_SHA,'base_driver_sha256':sha(root/'experiments/phase_d1.py'),
      'large_regression':{'support':14,'seed_level_safety':'S_large_TP >= C_large_TP; loss >=1 makes individual gate FAIL',
        'catastrophic':'S_large_TP <= C_large_TP - 2','catastrophic_loss_min_instances':2,
        'loss_2_percentage_points':100*2/14,'loss_exactly_1':'NOT catastrophic; MUST report; individual gate FAIL',
        'authorization':'Explicit user clarification on 2026-09-23 before D2 training'},
      'confirmation_gate':{'required_valid_pairs':5,'minimum_positive_deltas':4,'mean_small_delta':'strictly >0',
        'mean_precision_delta_min':-0.01,'mean_f1_delta_min':-0.01,'catastrophic_large_allowed':False,
        'counts_and_gate_arithmetic':'exact Fraction; no rounded gate comparisons',
        'missing_pairs':'INCOMPLETE; never replace 4/5 by 3/4',
        'precedence':'An observed catastrophic loss in any valid pair is FAIL even if study incomplete; completeness reported separately'},
      'failure_policy':{'local_early_stop_OOM_crash':'preserve invalid seed, no retry or remaining arm of invalid pair; attempt later registered seeds once',
        'common_freeze_data_environment_failure':'stop entire study, preserve evidence, report incomplete',
        'negative_valid_pair':'continue all remaining seeds regardless of gate outcome'},
      'budget':{'epochs':300,'patience':80,'scheduled_opportunities_per_arm':3741,'unknown_updates':0},
      'manipulation_check':'S exposure must exceed C for A, small_GT and very_small_GT; actual consumed plans must match',
      'descriptive_statistics':['mean paired effect','sample SD ddof=1','median','min','max','positive/zero/negative counts'],
      'inferential_statistics_authorized':False,'threshold_tuning_authorized':False,'crop_change_authorized':False,
      'APP_MODEL_REPLACEMENT_AUTHORIZED':'NO','EXTERNAL_TEST_AUTHORIZED':'NO','LOCKED_TEST_USED':False,
      'CO2Wounds_used':False,'D2_authorized':True,'STOP_AFTER_D2':True,'request_sha256':sha(REQUEST)}
    files=[]
    for seed in NEW_SEEDS:
        for arm,base,label in [('C',c,'control'),('S',s,'experimental')]:
            cfg=deepcopy(base);cfg['training_args']['seed']=seed;cfg['sampling']['seed']=seed
            cfg['sampling']['seed_policy']=f'Independent PCG64 SeedSequence([{seed},zero_based_epoch]); original permutation/choice algorithms'
            cfg['output_path']=seeded.outputs(seed)[arm];cfg['experiment_id']=prefix(seed)+'_'+label.upper()
            cfg['runner'].update(path='experiments/paired_sampling_multiseed.py',sha256=protocol['seeded_runner_sha256'],
              base_runner_sha256=seeded.BASE_SHA,adapter='make_runtime(seed).make_trainer_adapter')
            config_seed_only(base,cfg,seed)
            path=seed_artifact(root,seed,label+'_config');write(path,cfg);files.append(path)
        ep=seed_artifact(root,seed,'evaluation_protocol');write(ep,d1.load(root,'evaluation_protocol'));files.append(ep)
        sp=seed_artifact(root,seed,'protocol');write(sp,{'seed':seed,'master_protocol':artifact(root,'protocol').relative_to(root).as_posix(),
          'sampling_intervention':'unchanged V2','training_semantic_change_vs_D1':'SEED_ONLY','test_images_used':0});files.append(sp)
    protocol_path=artifact(root,'protocol');write(protocol_path,protocol);files.append(protocol_path)
    snapshot=artifact(root,'historical_snapshot');write(snapshot,{'seed42_files':snap,'other_protected':extra});files.append(snapshot)
    # Algebraic/sampler proof only, no models or image decoding.
    rows=read(root/c['manifest']['path'])['samples'];dry={}
    for seed in ALL_SEEDS:
        counts={'C':{},'S':{}};repeat=[]
        for epoch in range(300):
            for arm in ['C','S']:
                idx=seeded.indices(rows,arm,epoch,seed)
                require(len(idx)==771,'FAIL_DRY_LENGTH')
                if arm=='C':require(len(set(idx.tolist()))==771,'FAIL_DRY_CONTROL_REPLACEMENT')
                if seed==42:require((idx==d1.paired.plan_indices(rows,arm,epoch)).all(),'FAIL_SEED42_ALGORITHM_EQUIVALENCE')
                for i in idx:
                    r=rows[int(i)];counts[arm][r['sampling_category']]=counts[arm].get(r['sampling_category'],0)+1
                if arm=='S':repeat.append(771-len(set(idx.tolist())))
        dry[str(seed)]={'ABC_draws':counts,'S_mean_repeats_per_epoch':statistics.mean(repeat),'index_only_draws_per_arm':231300}
    dp=artifact(root,'dry_audit');write(dp,{'seed42_all_300_plans_identical':True,'seed_plans':dry,'models_loaded':False,'training_performed':False});files.append(dp)
    d1.verify_snapshot(snap);d1.validate_freeze(root)
    files += [root/p for p in SOURCE_FILES]
    freeze={'status':'FROZEN_BEFORE_TRAINING','artifacts_sha256':{p.relative_to(root).as_posix():sha(p) for p in files},
      'test_images_used':0,'seed42_read_only':True,'D2_authorized':True,'catastrophic_loss_min_instances':2,'STOP_AFTER_D2':True}
    write(artifact(root,'freeze'),freeze)
    return {'status':'FROZEN','artifacts':len(files),'seed42_files_protected':len(snap),'freeze_sha256':sha(artifact(root,'freeze'))}


def verify(root):
    f=read(artifact(root,'freeze'))
    for path,h in f['artifacts_sha256'].items():require(sha(root/path)==h,'FAIL_D2_PROTOCOL_MUTATED: '+path)
    d1.validate_freeze(root)
    snapshot=read(artifact(root,'historical_snapshot'))
    d1.verify_snapshot(snapshot['seed42_files']);d1.verify_snapshot(snapshot['other_protected'])
    return read(artifact(root,'protocol'))


def configs(root,seed):
    require(seed in NEW_SEEDS,'SEED42_RETRAIN_FORBIDDEN')
    c,s=[read(seed_artifact(root,seed,k+'_config')) for k in ['control','experimental']]
    d1.paired.config_diff(c,s)
    for cfg,label in [(c,'control'),(s,'experimental')]:config_seed_only(d1.load(root,label+'_config'),cfg,seed)
    return c,s


def state(root,stage,**extras):
    path=root/MASTER/'status.json';tmp=path.with_suffix('.tmp')
    with tmp.open('w',encoding='utf8') as f:json.dump({'at':d1.now(),'stage':stage,'test_images_used':0,'LOCKED_TEST_USED':False,'CO2Wounds_used':False,**extras},f,indent=2)
    os.replace(tmp,path)


def delegated_driver(root,seed):
    require(seed in NEW_SEEDS,'SEED42_RETRAIN_FORBIDDEN')
    protocol=verify(root);base=root/'experiments/phase_d1.py'
    require(sha(base)==protocol['base_driver_sha256'],'FAIL_BASE_DRIVER_MUTATED')
    m=types.ModuleType(f'experiments._d2_driver_{seed}');m.__file__=__file__
    exec(compile(base.read_text(encoding='utf8'),str(base),'exec'),m.__dict__)
    m.paired=seeded.make_runtime(root,seed);m.PREFIX=prefix(seed);m.PAIR=MASTER/f'seed_{seed}'
    def guarded(r):
        verify(r);c,s=configs(r,seed);return c,s,{}
    m.validate_freeze=guarded
    m.status=lambda r,stage,**extra:state(r,stage,seed=seed,**extra)
    return m


def worker(root,seed,arm,token):
    protocol=verify(root);lock=read(root/MASTER/'execution.lock')
    require(lock['token']==token and lock['driver_sha256']==sha(Path(__file__)),'FAIL_EXECUTION_AUTHORIZATION')
    require(lock['freeze_sha256']==sha(artifact(root,'freeze')),'FAIL_FREEZE_INDEX_CHANGED')
    delegated_driver(root,seed).train_arm(root,arm,token)


def effects(pair):
    c,s=[pair['evaluations'][a]['candidate'] for a in ['C','S']]
    cd,sd=[pair['evaluations'][a]['diagnostics'] for a in ['C','S']]
    def p(x):return Fraction(x['tp'],x['tp']+x['fp']) if x['tp']+x['fp'] else Fraction(0)
    def f(x):return Fraction(2*x['tp'],2*x['tp']+x['fp']+x['fn'])
    require(c['size_recall']['large']['gt']==s['size_recall']['large']['gt']==14,'FAIL_LARGE_SUPPORT')
    require(c['size_recall']['small']['gt']==s['size_recall']['small']['gt']==137,'FAIL_SMALL_SUPPORT')
    result={'small_recall':Fraction(s['size_recall']['small']['matched']-c['size_recall']['small']['matched'],137),
      'precision':p(s)-p(c),'recall':Fraction(s['tp']-c['tp'],241),'f1':f(s)-f(c),
      'crop_count':s['crop_complete95_images']-c['crop_complete95_images'],
      'very_small_TP':sd['very_small']['TP']-cd['very_small']['TP'],
      'FP':s['fp']-c['fp'],'FN':s['fn']-c['fn'],'no_ROI':s['positive_without_roi']-c['positive_without_roi']}
    for size in ['small','medium','large']:result[size+'_TP']=s['size_recall'][size]['matched']-c['size_recall'][size]['matched']
    for group in ['single_GT','multi_GT']:result[group+'_recall']=Fraction(sd[group]['TP'],sd[group]['GT'])-Fraction(cd[group]['TP'],cd[group]['GT'])
    return result


def confirmation(pairs):
    require(len({p['seed'] for p in pairs})==len(pairs) and all(p['seed'] in ALL_SEEDS for p in pairs),'FAIL_DUPLICATE_OR_UNKNOWN_SEED')
    valid=[p for p in pairs if p.get('valid') and p.get('intervention_delivered')]
    for pair in valid:
        require(d1.paired.pair_validity(pair['training']['C'],pair['training']['S'])=='VALID_SCHEDULED_BUDGET','FAIL_SUMMARY_PAIR_VALIDITY')
        require(all(pair['training']['S']['exposure'][k]>pair['training']['C']['exposure'][k] for k in ['A','small_GT','very_small_GT']),'FAIL_SUMMARY_INTERVENTION')
    deltas=[effects(p) for p in valid]
    catastrophic=[p['seed'] for p,e in zip(valid,deltas) if e['large_TP']<=-2]
    ordinary=[p['seed'] for p,e in zip(valid,deltas) if e['large_TP']==-1]
    positive=sum(e['small_recall']>0 for e in deltas);complete=len(valid)==5
    means={k:sum((e[k] for e in deltas),Fraction(0))/len(deltas) for k in deltas[0]} if deltas else {}
    checks={'direction_4_of_5':complete and positive>=4,'mean_small_positive':complete and means['small_recall']>0,
      'mean_precision_safety':complete and means['precision']>=Fraction(-1,100),
      'mean_f1_safety':complete and means['f1']>=Fraction(-1,100),'no_catastrophic_large':not catastrophic,
      'all_5_pairs_valid_and_delivered':complete}
    outcome='FAIL' if catastrophic else 'INCOMPLETE' if not complete else 'PASS' if all(checks.values()) else 'FAIL'
    vals=[float(e['small_recall']) for e in deltas]
    stats={'mean':float(means['small_recall']) if vals else None,'sample_SD':statistics.stdev(vals) if len(vals)>1 else None,
      'median':statistics.median(vals) if vals else None,'min':min(vals) if vals else None,'max':max(vals) if vals else None,
      'positive':positive,'zero':sum(v==0 for v in vals),'negative':sum(v<0 for v in vals),'n_paired_seeds':len(vals)}
    return {'MULTISEED_SMALL_SAMPLING_CONFIRMATION':outcome,'VALID_PAIRED_SEEDS':len(valid),'required_pairs':5,
      'MULTISEED_CONFIRMATION_INCOMPLETE':not complete,'checks':checks,'small_paired_effect':stats,
      'mean_paired_effects':{k:float(v) for k,v in means.items()},'catastrophic_large_seeds':catastrophic,
      'ordinary_large_minus_one_seeds':ordinary,'descriptive_summary_scope':'valid pairs only; never relabel incomplete as 5-seed confirmation',
      'exact_mean_paired_effects':{k:str(v) for k,v in means.items()}}


def save_report(root,pairs,aborted=None):
    out=root/MASTER;result=confirmation(pairs);valid=[p for p in pairs if p.get('valid') and p.get('intervention_delivered')]
    paired_rows=[];effect_rows=[];telemetry=[];exposures=[];bins=[];multi=[]
    for p in valid:
        seed=p['seed'];ef=effects(p);c,s=[p['evaluations'][a]['candidate'] for a in ['C','S']]
        paired_rows.append({'seed':seed,'C_small_TP':c['size_recall']['small']['matched'],'S_small_TP':s['size_recall']['small']['matched'],
          'delta_small_recall':float(ef['small_recall']),'delta_precision':float(ef['precision']),'delta_f1':float(ef['f1']),
          'delta_crop_count':ef['crop_count'],'pair_gate':p['seed_gate']})
        effect_rows.append({'seed':seed,**{k:float(v) for k,v in ef.items()}})
        t=p['training'];telemetry.append({'seed':seed,**{a+'_'+k:t[a][k] for a in ['C','S'] for k in ['completed_epochs','scheduled_optimizer_calls','applied_optimizer_updates','skipped_optimizer_updates']},
          'imbalance':t['C']['skipped_optimizer_updates']!=t['S']['skipped_optimizer_updates']})
        for a in ['C','S']:
            exposures.append({'seed':seed,'arm':a,**t[a]['exposure'],'unique_anchors':t[a]['unique_anchors_seen'],'repeat_draws':t[a]['repeat_draws_across_run']})
        cd,sd=[p['evaluations'][a]['diagnostics'] for a in ['C','S']]
        for name in cd['small_bins']:
            cb,sb=cd['small_bins'][name],sd['small_bins'][name]
            bins.append({'seed':seed,'bin':name,'support':cb['support'],'C_TP':cb['TP'],'S_TP':sb['TP'],'delta_TP':sb['TP']-cb['TP'],
              'delta_recall':(sb['TP']-cb['TP'])/cb['support']})
        for group in ['single_GT','multi_GT']:
            cg,sg=cd[group],sd[group]
            multi.append({'seed':seed,'group':group,'C_recall':cg['instance_recall'],'S_recall':sg['instance_recall'],
              'delta_recall':float(ef[group+'_recall']),'C_crop_complete':cg['crop_pass'],'S_crop_complete':sg['crop_pass'],
              'delta_crop_count':sg['crop_pass']-cg['crop_pass']})
    for name,rows in [('paired_seed_metrics',paired_rows),('paired_effects',effect_rows),('training_telemetry_summary',telemetry),
      ('sampling_exposure_summary',exposures),('small_bin_summary',bins),('multi_gt_summary',multi)]:
        if rows:d1.csv_output(out/(name+'.csv'),rows)
    bin_means={name:{'mean_delta_TP':statistics.mean(r['delta_TP'] for r in bins if r['bin']==name),
      'mean_delta_recall':statistics.mean(r['delta_recall'] for r in bins if r['bin']==name)} for name in dict.fromkeys(r['bin'] for r in bins)}
    result.update(PHASE_D2_STATUS='INTERRUPTED' if aborted else 'COMPLETE',MULTI_SEED_TRAINING_COMPLETE=not aborted and len(valid)==5,
      DIRECTIONAL_POSITIVE_SEEDS=result['small_paired_effect']['positive'],MEAN_PAIRED_SMALL_RECALL_DELTA=result['small_paired_effect']['mean'],
      AMP_UPDATE_IMBALANCE_SEEDS=[r['seed'] for r in telemetry if r['imbalance']],equal_realized_update_pairs=sum(not r['imbalance'] for r in telemetry),
      small_bin_mean_paired_effects=bin_means,pairs=pairs,aborted_reason=aborted,
      historical_development_gate_counts={a:{status:sum(p['evaluations'][a]['historical_development_gate']==status for p in valid) for status in ['PASS','FAIL','PARTIAL']} for a in ['C','S']},
      APP_MODEL_REPLACEMENT_AUTHORIZED='NO',EXTERNAL_TEST_AUTHORIZED='NO',LOCKED_TEST_USED=False,CO2Wounds_used=False,test_images_used=0,STOP_AFTER_D2=True)
    write(out/'multiseed_summary.json',result)
    lines=['# Phase D2 多種子配對抽樣確認','','研究單位：5 paired random-seed experiments。未重新訓練或評估 seed42。',
      '',f"狀態：{result['PHASE_D2_STATUS']}；Confirmation：{result['MULTISEED_SMALL_SAMPLING_CONFIRMATION']}；有效 pairs：{len(valid)}/5。",'',
      '| Seed | C Small TP | S Small TP | Δ Small Recall (pp) | Δ Precision (pp) | Δ F1 (pp) | Δ Crop | Pair gate |',
      '|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in paired_rows:lines.append(f"| {r['seed']} | {r['C_small_TP']} | {r['S_small_TP']} | {100*r['delta_small_recall']:.4f} | {100*r['delta_precision']:.4f} | {100*r['delta_f1']:.4f} | {r['delta_crop_count']} | {r['pair_gate']} |")
    lines += ['','## 平均 paired effects 與安全門檻','','```json',json.dumps({k:v for k,v in result.items() if k not in ['pairs']},ensure_ascii=False,indent=2),'```','',
      '## 各 seed 五個 small bins（含未改善結果）','','| Seed | Bin | C TP | S TP | Δ TP | Support |','|---:|---|---:|---:|---:|---:|']
    for r in bins:lines.append(f"| {r['seed']} | {r['bin']} | {r['C_TP']} | {r['S_TP']} | {r['delta_TP']} | {r['support']} |")
    lines += ['','## AMP 與 multi-GT 詳表','','```json',json.dumps({'AMP':telemetry,'single_multi_GT':multi,'exposure':exposures},ensure_ascii=False,indent=2),'```','',
      'Large 少1個：個別 seed gate FAIL，但非 catastrophic，必須揭露。少至少2個：任何有效 seed 發生即整體 FAIL。',
      '不足5個有效 pairs 時不得縮小4/5分母；若另有 catastrophic，安全結論 FAIL，研究完整性仍標 incomplete。',
      '統計為 paired seed effects 的平均、sample SD（ddof=1）、median、range；沒有 patient-level CI、p-values 或事後檢定。',
      'Seed42 very-small 結果固定 C=27/49、S=27/49，未刪除或替換。Historical Phase C 僅 reference，不作 primary control。',
      '沒有修改 weights、threshold、NMS、crop、App；沒有使用 locked test、CO2、official test 或新外部資料。',
      '下一研究階段只能依本報告另行決定：保留已確認介入、研究 very-small failure、crop-policy 實驗或 external validation planning；本次一律停止。']
    questions=['5 pairs 是否有效','各 seed Small C/S','各 seed Δ Small Recall','正向/零/負向數','Mean paired Δ','Sample SD','Median',
      'Precision mean Δ','F1 mean Δ','Crop mean Δ','Medium/Large safety','Very-small 是否改善','<0.10%是否改善','改善集中 bins',
      'Multi-GT 是否一致改善','FP/FN 是否穩定','AMP 是否一致','有無 invalid pair','4/5方向門檻','整體 gate','各 arm historical gate',
      'Locked test','CO2','Threshold 改變','Crop 改變','App替換','下一研究阶段']
    answers=[f'{len(valid)}/5','見 paired seed 主表','見 paired seed 主表',str({k:result['small_paired_effect'][k] for k in ['positive','zero','negative']}),
      str(result['small_paired_effect']['mean']),str(result['small_paired_effect']['sample_SD']),str(result['small_paired_effect']['median']),
      str(result['mean_paired_effects'].get('precision')),str(result['mean_paired_effects'].get('f1')),str(result['mean_paired_effects'].get('crop_count')),
      '見逐 seed gates、large警示及 paired_effects.csv；不隱藏 -1',str(result['mean_paired_effects'].get('very_small_TP'))+' mean Δ TP；seed42 Δ=0 保留',
      str(bin_means.get('<0.10%')),'見全部五 bins 的逐 seed count 與 mean effect','見 single/multi-GT 逐 seed 表，不只報平均',
      '見 paired_effects.csv 每 seed FP/FN/no-ROI',str(result['AMP_UPDATE_IMBALANCE_SEEDS']),str([p['seed'] for p in pairs if not p.get('valid')]),
      str(result['checks']['direction_4_of_5']),result['MULTISEED_SMALL_SAMPLING_CONFIRMATION'],str(result['historical_development_gate_counts']),
      'NO','NO','NO','NO','NO','依結果另行授權，不自動進下一階段']
    lines += ['','## 27 項指定問題','']+[f'{i}. **{q}**：{a}\n' for i,(q,a) in enumerate(zip(questions,answers),1)]
    with (root/REPORT).open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
    return result


def run_seed(root,seed,token,environment):
    m=delegated_driver(root,seed);pairdir=root/MASTER/f'seed_{seed}';pairdir.mkdir()
    write(pairdir/'execution.lock',{'seed':seed,'token':token,'at':d1.now(),'driver_sha256':sha(Path(__file__)),'environment':environment})
    t={};ev={};cfgs=configs(root,seed)
    for arm in ['C','S']:
        state(root,'STARTING_'+arm,seed=seed)
        with (pairdir/f'train_{arm}.log').open('x',encoding='utf8') as f:
            proc=subprocess.run([sys.executable,'-u','-m','experiments.phase_d2','--seed',str(seed),'--arm',arm,'--pair-token',token],cwd=root,stdout=f,stderr=subprocess.STDOUT)
        verify(root)  # common-integrity failure stops entire study rather than continuing.
        if proc.returncode:
            outcome={'seed':seed,'valid':False,'intervention_delivered':False,'status':'SEED_PAIR_INVALID','reason':f'{arm} exit {proc.returncode}',
              'retry_performed':False,'evaluation_performed':False}
            write(pairdir/'seed_pair_invalid.json',outcome);return outcome
        t[arm]=read(root/seeded.outputs(seed)[arm]/'training_completion.json')
        require(d1.paired.pair_validity(t[arm],t[arm])=='VALID_SCHEDULED_BUDGET','FAIL_TRAINING_COMPLETION')
        m.verify_saved_arm(root,arm,cfgs[0 if arm=='C' else 1])
    delivered=all(t['S']['exposure'][k]>t['C']['exposure'][k] for k in ['A','small_GT','very_small_GT'])
    if not delivered:
        outcome={'seed':seed,'valid':False,'intervention_delivered':False,'status':'FAIL_INTERVENTION_NOT_DELIVERED','training':t}
        write(pairdir/'seed_pair_invalid.json',outcome);return outcome
    validity=d1.paired.pair_validity(t['C'],t['S']);require(validity=='VALID_SCHEDULED_BUDGET','FAIL_PAIR_VALIDITY')
    for arm in ['C','S']:
        state(root,'EVALUATING_'+arm,seed=seed);ev[arm]=m.evaluate_arm(root,arm)
        m.verify_saved_arm(root,arm,cfgs[0 if arm=='C' else 1])
    gate=d1.paired.paired_advancement(*[d1.gate_counts(ev[a]['candidate']) for a in ['C','S']],validity)
    result={'seed':seed,'valid':True,'intervention_delivered':True,'training':t,'evaluations':ev,'paired_gate':gate,
      'seed_gate':'PASS' if gate['status'].startswith('PASS') else 'FAIL','amp':d1.paired.amp_sensitivity(t['C'],t['S'])}
    write(pairdir/'pair_summary.json',result);return result


def execute(root,expected_freeze):
    require(sha(artifact(root,'freeze'))==expected_freeze,'FAIL_FREEZE_INDEX_CHANGED')
    verify(root);output_guard(root);c,_,_=d1.validate_freeze(root)
    d1.verify_data(root,c);d1.verify_validation(root);environment=d1.runtime(c)
    pairs=[seed42_result(root)];out=root/MASTER;out.mkdir();token=str(uuid.uuid4())
    write(out/'execution.lock',{'token':token,'at':d1.now(),'driver_sha256':sha(Path(__file__)),
      'freeze_sha256':expected_freeze,'environment':environment,'D2_authorized':True,'seed42_retraining_authorized':False})
    aborted=None
    try:
        for seed in NEW_SEEDS:
            pairs.append(run_seed(root,seed,token,environment))
            verify(root)
        state(root,'POSTFLIGHT')
        with (out/'postflight_tests.log').open('x',encoding='utf8') as f:
            proc=subprocess.run([sys.executable,'-m','pytest','tests/test_phase_d2.py',*[f'tests/{t}.py' for t in d1.TESTS],'-q'],cwd=root,stdout=f,stderr=subprocess.STDOUT)
        require(proc.returncode==0,'FAIL_POSTFLIGHT_TESTS');verify(root)
    except BaseException as exc:
        aborted=f'{type(exc).__name__}: {exc}'
        write(out/'interruption.json',{'error':aborted,'at':d1.now(),'retry_allowed':False,'test_images_used':0})
        if 'seed' in locals() and seed not in {p['seed'] for p in pairs}:
            pairs.append({'seed':seed,'valid':False,'intervention_delivered':False,'status':'SEED_PAIR_INVALID','reason':aborted})
    for seed in NEW_SEEDS:
        if seed not in {p['seed'] for p in pairs}:
            pairs.append({'seed':seed,'valid':False,'intervention_delivered':False,'status':'NOT_ATTEMPTED_AFTER_COMMON_FAILURE'})
    result=save_report(root,pairs,aborted)
    state(root,result['PHASE_D2_STATUS'],confirmation=result['MULTISEED_SMALL_SAMPLING_CONFIRMATION'],valid_pairs=result['VALID_PAIRED_SEEDS'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build',action='store_true');parser.add_argument('--execute-authorized',action='store_true')
    parser.add_argument('--expected-freeze-sha256');parser.add_argument('--seed',type=int,choices=NEW_SEEDS);parser.add_argument('--arm',choices=['C','S']);parser.add_argument('--pair-token')
    args=parser.parse_args()
    if args.build:print(json.dumps(build(ROOT),indent=2))
    elif args.arm:worker(ROOT,args.seed,args.arm,args.pair_token)
    elif args.execute_authorized:execute(ROOT,args.expected_freeze_sha256)
    else:parser.error('Explicit build or authorized execution required; no resume option')
