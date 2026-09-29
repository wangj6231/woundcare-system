"""F0 higher-scale preregistration only. No research execution entrypoint."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PRO=ROOT/'experiments/protocols'
OUT=ROOT/'experiments/results/f_higher_scale_v1_resource_audit'
PREFIX='F_HIGHER_SCALE_V1'
INIT_SHA='1ec926026473beafafb1ef8da6aa6efcc13d38c391ec8db48006c5d793d5f2a3'
FUTURE=['experiments/results/f_higher_scale_v1_seed42_control768','experiments/results/f_higher_scale_v1_seed42_train1024']


def require(ok,message):
    if not ok:raise ValueError(message)


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def leaves(value,prefix=''):
    if not isinstance(value,dict) or not value:return {prefix:value}
    result={}
    for k,v in value.items():result.update(leaves(v,f'{prefix}.{k}' if prefix else k))
    return result


def make_pair():
    base=read(PRO/'D_SEG_SMALL_SAMPLING_V2_control_config.json')
    keys=('architecture','augmentation','budget','initialization','loss','runtime_optimizer','software','training_args','manifest','dataset_manifest_sha256','data_yaml_path')
    common={k:deepcopy(base[k]) for k in keys}
    common.update(schema='F_HIGHER_INPUT_SCALE_V1',execution_authorized=False,
        label_path='ORIGINAL_STOCK_YOLO_POLYGON',dataset_adapter='STOCK_YOLODataset',
        representation='FULL_IMAGE',historical_comparators='REFERENCE_ONLY',
        primary_comparison='Fresh C4 train768/eval768 vs Fresh H4 train1024/eval768',
        sampling={'mode':'uniform','replacement':False,'epoch_anchor_draws':771,'weights':'all1','seed':42,
            'order_path':f'experiments/protocols/{PREFIX}_anchor_orders.json',
            'sequence':'same frozen sample-ID order in both arms; actual consumed sequence must be logged and match every epoch',
            'RNG':'PCG64 SeedSequence([42, zero_based_epoch])'},
        evaluation={'path':f'experiments/protocols/{PREFIX}_evaluation_protocol.json','imgsz':768,
            'sha256':base['evaluation_protocol_sha256'],'after':'both arms training complete AND pair validity checked'},
        checkpoint_selection={'both_arms_validation_imgsz':768,'labels':'original validation labels unchanged',
            'criterion':'same pinned stock segmentation fitness in both arms',
            'runtime_requirement':'F1 must pin both validation loader image size and validator args to768; no implicit H4 validation at1024',
            'F0_validation_execution':False},
        budget_contract={'scheduled':3741,'applied_plus_skipped':3741,'unknown':0,
            'early_stop':'completed_epochs<300 -> PAIRED_FIXED_BUDGET_VALID=NO; no resume/top-up/retry',
            'OOM':'INTERRUPTED_RESOURCE_FAILURE; no lower batch or scale retry',
            'AMP_imbalance':'report skips and applied counts; never auto-retrain',
            'gradient_accumulation_compensation':False},
        future_runtime='new F1 binding of original stock labels and existing telemetry; must not call the D1 driver with these configs')
    c=dict(deepcopy(common),arm='C4',experiment_id=PREFIX+'_C4',output_path=FUTURE[0])
    h=dict(deepcopy(common),arm='H4',experiment_id=PREFIX+'_H4',output_path=FUTURE[1])
    c['training_args']['imgsz']=768;h['training_args']['imgsz']=1024
    return c,h


def validate_pair(c,h):
    a,b=leaves(c),leaves(h);missing=object()
    changed=sorted(k for k in a.keys()|b.keys() if a.get(k,missing)!=b.get(k,missing))
    require(changed==['arm','experiment_id','output_path','training_args.imgsz'],'FAIL_NON_SCALE_CONFIG_DIFF')
    require(c['training_args']['imgsz']==768 and h['training_args']['imgsz']==1024,'FAIL_SCALE')
    for cfg in (c,h):
        require(cfg['training_args']['batch']==4,'FAIL_BATCH')
        require(cfg['evaluation']['imgsz']==cfg['checkpoint_selection']['both_arms_validation_imgsz']==768,'FAIL_EVAL_SCALE')
        require(cfg['initialization']['sha256']==INIT_SHA and not cfg['initialization']['resume'],'FAIL_INIT')
        require(cfg['dataset_adapter']=='STOCK_YOLODataset' and cfg['label_path']=='ORIGINAL_STOCK_YOLO_POLYGON','FAIL_LABEL_PATH')
        require(not cfg['execution_authorized'],'FAIL_F1_NOT_AUTHORIZED')
    return changed


def admit_training_row(row):
    require((row.get('source'),row.get('split'),row.get('role'))==('FUSeg','train','wound_finetuning'),'FAIL_ORIGINAL_TRAINING_ONLY')
    for kind,folder in [('image','images'),('label','labels')]:
        path=row[kind+'_path'].replace('\\','/')
        prefix=f'outputs/isic_fuseg_pretrain_smoke_20260914/fuseg_dataset/{folder}/train/'
        require(path.startswith(prefix) and len(path[len(prefix):].split('/'))==1 and '..' not in path.split('/'),'FAIL_ORIGINAL_TRAINING_ONLY')
    return True


def future_absent(root=ROOT):
    require(all(not (root/p).exists() for p in FUTURE),'FAIL_FUTURE_OUTPUT_EXISTS')
    return True


def completion_status(smoke,*,tests_pass):
    passed=(tests_pass and smoke.get('RESOURCE_FEASIBILITY')=='PASS' and not smoke.get('NaN_Inf',True)
            and smoke.get('parameters_unchanged',False) and smoke.get('update_save_guards_installed',False))
    return {'PHASE_F0_STATUS':'COMPLETE' if passed else 'BLOCKED',
        'READY_FOR_PHASE_F1_PAIRED_HIGHER_SCALE_SEED42':'YES' if passed else 'NO',
        'RESOURCE_FEASIBILITY':smoke.get('RESOURCE_FEASIBILITY','NOT_EXECUTED'),
        'NUMERICAL_FEASIBILITY':smoke.get('NUMERICAL_FEASIBILITY','NOT_VERIFIED'),
        'PATCH_BASED_TRAINING_ROUTE':'STOP','NEW_TRAINING_AUTHORIZED':'NO',
        'APP_MODEL_REPLACEMENT_AUTHORIZED':'NO','EXTERNAL_TEST_AUTHORIZED':'NO',
        'STOP_AFTER_F0':True,'official_training':False,'test_images_used':0,'CO2Wounds_used':False}


def prepare():
    import importlib.metadata
    import importlib.util
    import struct
    require(not OUT.exists() and not list(PRO.glob(PREFIX+'*')),'FAIL_F0_OUTPUT_EXISTS')
    future_absent()
    # Prior patch artifacts are hash-only historical evidence, never label inputs.
    e3=read(PRO/'E_PATCH_V3_freeze.json')
    require(e3['PHASE_E12_STATUS']=='BLOCKED' and e3['PATCH_BASED_TRAINING_ROUTE']=='STOP','FAIL_PATCH_NOT_STOPPED')
    protected=read(ROOT/'experiments/results/e_patch_v3_feasibility_audit/source_snapshot.json')
    protected.update({str(ROOT/p):h for p,h in e3['artifacts_sha256'].items()})
    protected[str(PRO/'E_PATCH_V3_freeze.json')]=sha(PRO/'E_PATCH_V3_freeze.json')
    for p,h in protected.items():require(sha(p)==h,'FAIL_SEALED_HASH_CHANGED: '+p)
    c,h=make_pair();validate_pair(c,h)
    manifest=ROOT/c['manifest']['path'];require(sha(manifest)==c['manifest']['sha256'],'FAIL_D2_MANIFEST_HASH')
    rows=read(manifest)['samples'];require(len(rows)==771 and len({r['sample_id'] for r in rows})==771,'FAIL_TRAIN_IDS')
    for row in rows:
        admit_training_row(row)
        for kind in ('image','label'):
            path=ROOT/row[kind+'_path'];require(sha(path)==row[kind+'_hash'],'FAIL_ORIGINAL_HASH')
            protected[str(path)]=row[kind+'_hash']
        with (ROOT/row['image_path']).open('rb') as f:header=f.read(24)
        require(header[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',header[16:24])==(512,512),'FAIL_SOURCE_DIMENSIONS')
    init=ROOT/c['initialization']['path'];require(sha(init)==INIT_SHA,'FAIL_INITIALIZATION_SHA256')
    protected[str(init)]=INIT_SHA
    package=Path('C:/Python312/Lib/site-packages/ultralytics')
    for rel,expected in c['architecture']['source_hashes'].items():
        path=Path(importlib.util.find_spec('torch').origin).parent/rel.removeprefix('torch/') if rel.startswith('torch/') else package/rel
        require(sha(path)==expected,'FAIL_RUNTIME_PIN: '+rel);protected[str(path)]=expected
    for name,version in c['software'].items():require(importlib.metadata.version(name)==version,'FAIL_SOFTWARE_VERSION: '+name)
    for p in [manifest,PRO/'D_SEG_SMALL_SAMPLING_V2_control_config.json',PRO/'D_SEG_SMALL_SAMPLING_V2_evaluation_protocol.json']:
        protected[str(p)]=sha(p)
    save(OUT/'source_snapshot.json',protected)
    save(OUT/'data_audit.json',{'original_train_images':771,'original_GT_instances':sum(r['GT_count'] for r in rows),
        'sample_ids':[r['sample_id'] for r in rows],'all_image_hashes_match':True,'all_label_hashes_match':True,
        'dimensions':'771 PNG headers512x512; no image decode','image_pixels_decoded':0,
        'labels_repaired_or_removed':0,'training_manifest':str(manifest.relative_to(ROOT)),
        'training_manifest_sha256':sha(manifest),'E_PATCH_labels_used':False,'PATCH_BASED_TRAINING_ROUTE':'STOP',
        'initialization_sha256':sha(init),'test_images_used':0,'CO2Wounds_used':False})
    ids=[r['sample_id'] for r in rows]
    orders=[[ids[int(j)] for j in np.random.Generator(np.random.PCG64(np.random.SeedSequence([42,e]))).permutation(771)] for e in range(300)]
    save(PRO/f'{PREFIX}_anchor_orders.json',{'seed':42,'epochs':300,'per_epoch':771,'orders':orders,
        'epoch_sha256':[hashlib.sha256(json.dumps(o,separators=(',',':')).encode()).hexdigest() for o in orders],
        'RNG':'PCG64 SeedSequence([42,zero_based_epoch])','replacement':False,'weights':'all1'})
    for cfg in (c,h):cfg['sampling']['order_sha256']=sha(PRO/f'{PREFIX}_anchor_orders.json')
    validate_pair(c,h)
    save(PRO/f'{PREFIX}_control_config.json',c);save(PRO/f'{PREFIX}_experimental_config.json',h)
    shutil.copyfile(PRO/'D_SEG_SMALL_SAMPLING_V2_evaluation_protocol.json',PRO/f'{PREFIX}_evaluation_protocol.json')
    require(sha(PRO/f'{PREFIX}_evaluation_protocol.json')==c['evaluation']['sha256'],'FAIL_EVALUATION_PROTOCOL_CHANGED')
    protocol={'schema':'F_HIGHER_INPUT_SCALE_V1','created_at':datetime.now(timezone.utc).isoformat(),
        'research_question':'Only training input768 vs1024; original stock full-image labels, same fresh initialization; both final eval768',
        'scale_interpretation':{'source':[512,512],'source_to_control':1.5,'source_to_experimental':2.,'relative_linear':4/3,'relative_area':16/9,
            'meaning':'larger object pixel/feature-map extent at model input; no new native detail or recovered clinical information'},
        'PATCH_BASED_TRAINING_ROUTE':'STOP','patch_artifacts_role':'HISTORICAL_FEASIBILITY_EVIDENCE_ONLY',
        'historical_models_role':'REFERENCE_ONLY','primary_comparator':'Fresh C4 vs Fresh H4',
        'training_order':['C4_train768','H4_train1024','pair_validity','C4_eval768','H4_eval768'],
        'primary_endpoint':{'name':'Very-small Recall','bbox_area_ratio_lt':.0025,'support':49},
        'key_secondary':{'name':'Recall <0.10%','bbox_area_ratio_lt':.001,'support':27},
        'other_endpoints':['Small Recall<1%','Medium Recall','Large Recall','Precision','Recall','F1','Crop complete','TP','FP','FN','No ROI','single-GT Recall','multi-GT Recall'],
        'advancement_gate':{'all_required':True,'very_small_TP':'H4 >= C4 +2','small_TP':'H4 >= C4 -1',
            'precision':'H4 >= C4 -.01','F1':'H4 >= C4 -.01','medium_TP':'H4 >= C4 -1','large_TP':'H4 >= C4','crop_complete':'H4 >= C4 -1',
            'primary_delta_pp':200/49,'interpretation':'operational criterion; not statistical significance; no post-hoc +1 relaxation'},
        'sampling':'771 unique IDs per epoch, uniform without replacement, shared300 orders; record actual consumed sequence',
        'augmentation_interpretation':'same normalized configuration; pixel geometry differs as DOWNSTREAM_EFFECT_OF_INPUT_SCALE_INTERVENTION, not pixel-identical',
        'checkpoint_selection':'both within-training validation loaders/validator args768; same stock fitness criterion; F1 must verify runtime before execution',
        'budget':c['budget_contract'],'F1_training_authorized':False,'STOP_AFTER_F0':True,'future_outputs_absent':FUTURE,
        'forbidden':['patch','canonicalization','mask-first label path','new data','label repair','lower batch','accumulation compensation','scale sweep','validation inference in F0','test','CO2','external data']}
    save(PRO/f'{PREFIX}_protocol.json',protocol)
    smoke={'kind':'SYNTHETIC_RESOURCE_SMOKE','attempts_allowed':1,'network':'YOLO11m-seg','imgsz':1024,'batch':4,
        'AMP':True,'seed':42,'initialization':c['initialization'],'architecture':c['architecture'],
        'warmup_forward_loss_iterations':1,'fixed_backward_stress_batches':1,
        'synthetic_source_size':512,'synthetic_instances_per_image':16,'synthetic_generator':'seeded noise image with4x4 valid disjoint rectangle polygons, no real pixel access',
        'target_pipeline':'stock polygons2masks_overlap, mask_ratio4; original YOLO loss contract',
        'optimizer_step_allowed':False,'checkpoint_save_allowed':False,'research_metrics_allowed':False,
        'GradScaler':'initial65536; scale(loss).backward, unscale, inspect finite gradients, update scaler only; never scaler.step or optimizer.step',
        'optimizer_state_caveat':'fresh AdamW exists only for unscale; no lazy Adam moments allocated without step, so not proof of complete training memory fit',
        'memory_scope':'peak CUDA allocator allocation/reservation including model, loss and backward; not all driver/other-process VRAM',
        'pass':'one1024x4 fixed batch forward/backward completes withoutOOM; nonfinite loss/gradients separately block readiness',
        'failure':'BLOCK, no retry/scale/batch adjustment',
        'read_access':'synthetic arrays in memory; only approved initialization checkpoint may be deserialized; no real dataset opens in smoke',
        'forbidden_real_data':['FUSeg train','FUSeg validation','locked test','CO2Wounds','external data'],
        'official_training':False,'no_epochs':True,'F1_guarantee':False}
    save(PRO/f'{PREFIX}_resource_smoke_protocol.json',smoke)
    save(OUT/'pre_smoke_seal.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'artifacts_sha256':{str(p.relative_to(ROOT)):sha(p) for p in PRO.glob(PREFIX+'*.json')},
        'original_files_verified':len(protected),'future_outputs_absent':FUTURE})
    print(json.dumps({'preflight':'PASS','original_train':771,'initialization_sha256':INIT_SHA,'patch_route':'STOP','next':'single synthetic resource smoke'}))


def finalize():
    import ast
    smoke=read(OUT/'resource_smoke_result.json');tests=read(OUT/'verification.json')
    require(tests['exit_code']==0,'FAIL_TESTS')
    lock=read(OUT/'execution.lock')
    require(sha(OUT/'executed_resource_smoke.py')==lock['code_sha256'],'FAIL_EXECUTED_SOURCE_PROVENANCE')
    pre=read(OUT/'pre_smoke_seal.json')
    for p,h in pre['artifacts_sha256'].items():require(sha(ROOT/p)==h,'FAIL_PRE_SMOKE_CONTRACT_CHANGED')
    c=read(PRO/f'{PREFIX}_control_config.json');h=read(PRO/f'{PREFIX}_experimental_config.json')
    diff=validate_pair(c,h);future_absent()
    protected=read(OUT/'source_snapshot.json')
    for p,expected in protected.items():require(sha(p)==expected,'FAIL_PROTECTED_CHANGED: '+p)
    source=OUT/'executed_resource_smoke.py'
    tree=ast.parse(source.read_text(encoding='utf8'))
    forbidden_calls=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            if node.func.attr in {'step','imread','imdecode','open_video','fit'}:forbidden_calls.append(node.func.attr)
    require(not forbidden_calls,'FAIL_SMOKE_HAS_FORBIDDEN_CALLS')
    wrapper=Path('C:/Python312/Lib/site-packages/ultralytics/utils/patches.py')
    result=completion_status(smoke,tests_pass=True)
    blockers=[]
    if smoke['NaN_Inf']:blockers.append('NONFINITE_GRADIENTS_IN_ONE_SHOT_SYNTHETIC_BACKWARD; no numerical readiness claim')
    if not smoke.get('update_save_guards_installed'):blockers.append('COMPOSITE_GUARD_IDENTITY_PROBE_FALSE; keep original raw evidence; no all-guards-PASS claim')
    if smoke['RESOURCE_FEASIBILITY']!='PASS':blockers.append('RESOURCE_SMOKE_FAILED')
    result.update(blockers=blockers,tests=tests['summary'],protected_files_unchanged=len(protected),
                  resource_smoke_attempts=1,semantic_diff=diff)
    audit={'protected_files_checked':len(protected),'protected_files_changed':0,
        'pre_smoke_protocols_unchanged':True,'execution_source_sha256':sha(source),
        'current_source_sha256':sha(ROOT/'experiments/higher_scale_resource_smoke.py'),
        'source_revision_note':'Executed source preserved byte-for-byte. Post-smoke CPU-only hardening rejects image/weight files outside workspace as well; no second GPU smoke.',
        'post_smoke_cpu_guard_hardening_test':'test_f0_external_pixels_rejected_even_outside_workspace',
        'composite_guard_probe':smoke.get('update_save_guards_installed'),
        'guard_probe_interpretation':'Ultralytics __init__ replaces torch.save with patches.torch_save; patches captured the prior forbidden save as _torch_save. A direct function-identity conjunction is therefore false; raw result not rewritten. No optimizer update or checkpoint save occurred.',
        'guard_interpretation_basis':'static source inspection, not a second model run',
        'wrapper_source_sha256':sha(wrapper),'executed_source_forbidden_step_or_image_decode_calls':forbidden_calls,
        'parameter_values_unchanged':smoke.get('parameters_unchanged'),
        'synthetic_only_basis':'executed source generates all input arrays with seeded RNG; no real-data input argument or image decode call',
        'checkpoint_sha256_unchanged':sha(ROOT/c['initialization']['path']),
        'future_outputs_absent':FUTURE,'test_images_used':0,'research_data_inference':False,
        'research_training':False,'formal_model_outputs_created':False}
    save(OUT/'postflight.json',audit);save(OUT/'status.json',result)
    report=ROOT/'docs/PHASE_F0_HIGHER_INPUT_SCALE_PREREGISTRATION_20260925.md'
    with report.open('x',encoding='utf8',newline='\n') as f:f.write(render_report(result,smoke,tests,audit))
    artifacts=[*PRO.glob(PREFIX+'*.json'),*OUT.glob('*.json'),OUT/'execution.lock',source,report,
               ROOT/'experiments/phase_f0.py',ROOT/'experiments/higher_scale_resource_smoke.py',ROOT/'tests/test_phase_f0.py']
    freeze=dict(result,created_at=datetime.now(timezone.utc).isoformat(),
        artifacts_sha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts},
        future_outputs_absent=FUTURE,resource_smoke=smoke,
        request_sha256=sha('C:/Users/milo9/.codex/attachments/16d3ead9-8fe7-41a2-93e6-993fd7ad1bef/貼上的文字.txt'),
        freeze_scope='F0 preregistration and one-shot resource evidence; not F1 authorization')
    save(PRO/f'{PREFIX}_freeze.json',freeze)
    print(json.dumps(result,ensure_ascii=False))


def render_report(status,smoke,tests,audit):
    allocated=smoke.get('max_memory_allocated_bytes',0)/1024**3
    reserved=smoke.get('max_memory_reserved_bytes',0)/1024**3
    return f'''# Phase F0 — Higher Input Scale Preregistration & Resource Feasibility

版本：F_HIGHER_INPUT_SCALE_V1。實際執行日期：2026-09-26（Asia/Taipei）；檔名20260925保留使用者指定名稱。

## 1. 正式結論

```ini
PHASE_F0_STATUS = {status['PHASE_F0_STATUS']}
RESOURCE_FEASIBILITY = {status['RESOURCE_FEASIBILITY']}
NUMERICAL_FEASIBILITY = {status['NUMERICAL_FEASIBILITY']}
READY_FOR_PHASE_F1_PAIRED_HIGHER_SCALE_SEED42 = {status['READY_FOR_PHASE_F1_PAIRED_HIGHER_SCALE_SEED42']}
PATCH_BASED_TRAINING_ROUTE = STOP
NEW_TRAINING_AUTHORIZED = NO
APP_MODEL_REPLACEMENT_AUTHORIZED = NO
EXTERNAL_TEST_AUTHORIZED = NO
STOP_AFTER_F0 = true
```

F0的預登錄、資料核對、唯一一次合成資源測試及封存已做完；但不能放行F1。**記憶體可行性通過，不等於數值安全或正式訓練可行性通過。** 合成backward出現非有限梯度；本次smoke前封存的安全條件明列nonfinite阻擋readiness。沒有為了通過而降低GradScaler起始scale、改batch或重跑。

這不代表1024訓練必然失敗，也沒有模型效果結論。AMP初始scale過大是一種可能解釋，但一次反向、沒有optimizer.step，不足以確認原因或後續是否會穩定。

## 2. 獨立研究線與原始資料

E_PATCH V1/V2/V3維持封存與STOP，僅為historical feasibility evidence。F0不使用canonical labels、mask-first labels或任何patch adapter。資料回到D2原始manifest的771張FUSeg training images、965個GT；影像及標籤SHA256全部符合，未增刪或修復invalid topology。僅讀PNG header核對512×512與檔案雜湊，不解碼training pixels；GPU smoke也完全沒有讀取真實資料。

兩組回到STOCK_YOLODataset及原始YOLO polygon augmentation/loss路徑。原始invalid topology既然是baseline一部分，本階段不修、不移除。191張development validation只繼承既有protocol；未推論、未看新的validation結果。Locked test、CO2與外部資料皆未使用。

## 3. Fresh C4/H4設計

| 欄位 | C4 | H4 |
|---|---|---|
| Training imgsz | 768 | 1024 |
| Final evaluation imgsz | 768 | 768 |
| 訓練中checkpoint-selection validation imgsz | 768 | 768 |
| Batch | 4 | 4 |
| Source canvas | 512×512 | 512×512 |
| 模型 | YOLO11m-seg / class0 Wound | 相同 |
| Sampling | Uniform without replacement | 相同 |
| Anchor orders | 300×771，共同seed42順序 | 相同 |
| Initialization | 相同ISIC checkpoint，fresh optimizer | 相同 |

唯一training semantic config difference為 `training_args.imgsz`，另有arm、experiment_id、output_path。Machine diff已核對。既有Phase C/D2模型只作REFERENCE_ONLY；主因果比較只可用Fresh C4 vs Fresh H4。

初始化SHA256：`{INIT_SHA}`，本次前後核對皆相同。resume=false。

固定300epochs、patience80、AdamW lr0=.0005、betas(.937,.999)、weight-decay .0005（bias/norm0）、cosine lrf=.01、warmup5、warmup_bias_lr0、AMP、deterministic、workers2、nbs64與gradient clip10。不做micro-batch或accumulation補償。

Loss維持v8SegmentationLoss：box/seg7.5、cls.5、dfl1.5、mask_ratio4、overlap_mask=true。Augmentation維持mosaic.1、close_mosaic30、mixup/copy_paste0、degrees5、translate.05、scale.2、shear.5、perspective.0002、fliplr.5、flipud0、HSV(.01,.4,.25)、bgr0，及原Albumentations Blur/MedianBlur/ToGray/CLAHE。未修改任何權重或gain。

### Scale的正確解釋

512→768為1.5×，512→1024為2×；相對baseline線性extent為4/3，面積為16/9≈1.7778×。只能說物件在model input／feature maps占更多像素範圍；**不是增加native resolution、創造新細節或恢復臨床資訊**。

Normalized augmentation設定相同不代表pixel-identical augmentation；其幾何像素差異屬於input scale的下游效果。沒有將其當作第二個人工改動。

## 4. Evaluation及未來執行順序

Evaluation protocol與D2 byte-identical：191張原始FUSeg development validation、full-image、RGB→BGR corrected、confidence.10、floor.01、NMS.70、bbox match.50、imgsz768、crop margin15%，原matching/mask/crop/size定義不改。

未來若重新取得明確授權，順序為C4 train768→H4 train1024→pair validity→C4 eval768→H4 eval768。不可看C4 final evaluation後決定是否訓練H4。H4不得用eval1024作primary comparison。

訓練中validation也事先明定兩組768，以免stock trainer自動沿用H4的1024造成不同checkpoint selection。這是兩組共同約束；F0未建置／執行正式trainer，F1啟動前仍須驗證validation loader與validator args都確實768。

Primary endpoint為very-small Recall<0.25%，support49；key secondary為<0.10% Recall，support27。另報Small<1%、Medium/Large、Precision/Recall/F1、Crop complete、TP/FP/FN、No ROI與single/multi-GT Recall。

Advancement gate全部須通過：H4 very-small TP≥C4+2，small/medium TP≥C4−1，Precision/F1≥C4−1pp，large TP≥C4，crop-complete≥C4−1。+2/49=+4.081633pp是operational research threshold，不是統計顯著；不得事後改+1。

Scheduled optimizer opportunities固定3741，applied+skipped=3741且unknown0；AMP skips不同需如實報告，不重訓。任何arm不到300epochs則paired fixed budget invalid、不補跑。F1若OOM須中斷，不能改batch續跑。

## 5. 唯一一次合成GPU測試

| 項目 | 結果 |
|---|---|
| GPU | {smoke.get('GPU_name','unavailable')} |
| Total VRAM | {smoke.get('total_VRAM_bytes',0)/1024**3:.3f} GiB |
| Training-shaped input | 1024×1024，batch4，AMP true |
| 合成資料 | Seed42 noise，4張512 source，4×4合法矩形／張，共64GT |
| Mask target | [4,256,256]，stock overlap polygon raster |
| 執行範圍 | 1次warmup forward+loss；1次固定forward+loss+backward |
| Forward / Backward | {smoke['forward_success']} / {smoke['backward_success']} |
| OOM | {smoke['OOM']} |
| Peak allocated | {allocated:.3f} GiB（{smoke.get('max_memory_allocated_bytes',0)} bytes） |
| Peak reserved | {reserved:.3f} GiB（{smoke.get('max_memory_reserved_bytes',0)} bytes） |
| Nonfinite gradients | {smoke['NaN_Inf']}，NaN/Inf未進一步細分 |
| GradScaler scale | {smoke.get('GradScaler_scale_before')} → {smoke.get('GradScaler_scale_after')} |
| Model parameters unchanged | {smoke.get('parameters_unchanged')} |
| Optimizer steps / checkpoint saves | 0 / 0 |
| Real dataset pixels / research inference | 0 / 無 |
| Research accuracy、mAP、Recall | 未計算 |

Warmup與stress loss皆通過finite guard；非有限值出現在反向後梯度。沒有保存或比較loss數值來選scale。GradScaler只做scale/backward/unscale/update，未呼叫scaler.step或optimizer.step；fresh AdamW state仍為空，parameter值前後hash相同。

峰值只代表CUDA allocator中此合成工作負載，不涵蓋所有driver／其他程序顯存。因step禁止，Adam lazy moment states未配置；沒有真實Mosaic、DataLoader與長期fragmentation，因此不能保證300epochs能跑完。BatchNorm buffers可能在這個可拋棄RAM模型中變化，未保存，亦未將其當研究權重。

### Guard診斷與來源保留

原始結果中的 `update_save_guards_installed=false` **完整保留**。讀碼查明Ultralytics匯入時以wrapper替換torch.save，wrapper持有先前的禁止save函式，因此直接函式identity的合取檢查不能等同完整guard驗證。沒有宣稱該probe PASS，也未重寫原結果；source沒有step／真實image decode呼叫，參數不變且無權重寫出。

實際執行版本保存在 `executed_resource_smoke.py`，SHA256符合execution.lock。之後只以CPU測試補強「workspace外影像／權重檔也拒絕讀取」的存取政策；新版本未重跑GPU。這項hardening不追溯冒稱第一次使用了較新的guard。

## 6. 25項完成問題

| # | 問題 | 答案 |
|---:|---|---|
| 1 | Patch STOP？ | 是，V1/V2/V3未變 |
| 2 | 使用E_PATCH labels？ | 否 |
| 3 | 回到original771？ | 是，影像／標籤hash符合 |
| 4 | 只差training imgsz？ | 是，除必要arm/id/output欄位 |
| 5 | C4 imgsz？ | 768 |
| 6 | H4 imgsz？ | 1024 |
| 7 | Batch都4？ | 是 |
| 8 | Accumulation compensation？ | 無 |
| 9 | Sampling相同？ | 是，uniform without replacement |
| 10 | Anchor ordering相同？ | 是，300×771已封存 |
| 11 | Initialization相同？ | 是，相同hash、fresh optimizer |
| 12 | Optimizer/loss/augmentation相同？ | 是 |
| 13 | 1024×4 resource PASS？ | 記憶體PASS，但數值安全FAIL |
| 14 | Peak VRAM？ | Allocated {allocated:.3f}、reserved {reserved:.3f} GiB |
| 15 | Smoke用real pixels？ | 否 |
| 16 | Smoke optimizer.step？ | 否 |
| 17 | Primary endpoint？ | <0.25% Very-small Recall，49GT |
| 18 | Advancement？ | +2 very-small TP及全部safety gates |
| 19 | Final eval imgsz？ | 兩組768 |
| 20 | F1 primary eval1024？ | 禁止 |
| 21 | Locked test？ | 未讀取／未使用 |
| 22 | CO2？ | 未讀取／未使用 |
| 23 | 正式training？ | 無；只有授權的合成forward/backward |
| 24 | Research-data inference？ | 無 |
| 25 | 可以放行F1？ | 否，本次F0 BLOCKED，待另次明確決策 |

## 7. 驗證與封存

{tests['summary']}。依TDD先驗證config差異、資料角色、合成輸入與存取邊界，再補齊共同recipe和既有D系列回歸；pytest未重跑GPU smoke。事前6份protocol/config/order/evaluation檔案雜湊未變。

{audit['protected_files_checked']}個繼承保護檔案前後hash相同。未來C4/H4輸出目錄均不存在。F0新檔案、報告、raw smoke result、executed source及test evidence皆列入 `F_HIGHER_SCALE_V1_freeze.json`。詳細數值以resource_smoke_result.json為準，未把失敗遮蔽成PASS。

**停止在F0。** 本階段未測960/896/832、未降batch、未作accumulation補償、未重新跑synthetic backward、未開始F1。下一個決策應先釐清AMP非有限梯度及guard探針的驗證方式；須另行明確授權，不能自動延伸實驗。
'''


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--finalize',action='store_true');args=p.parse_args()
    finalize() if args.finalize else prepare()
