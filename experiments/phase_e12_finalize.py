"""Finalize E1.2 evidence only. Block unverified training semantics; never launch E2."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import json
import shutil
from pathlib import Path
from experiments.phase_e12 import ROOT,PRO,OUT,PACKAGE,read,save
from experiments.patch_raster import file_sha
from experiments.patch_geometry import require,model_guard


def feasibility(*,source_ok,retention_ok,loader_ok,augmentation_parity):
    passed=source_ok and retention_ok and loader_ok and augmentation_parity=='VERIFIED'
    return {'PHASE_E12_STATUS':'COMPLETE' if passed else 'BLOCKED',
        'MASK_FIRST_TRAINING_REPRESENTATION':'FEASIBLE' if passed else 'NOT_FEASIBLE',
        'PATCH_BASED_TRAINING_ROUTE':'AWAIT_SEPARATE_E2_AUTHORIZATION' if passed else 'STOP',
        'READY_FOR_E2_MASK_FIRST_PAIRED_PATCH_TRAINING':'YES' if passed else 'NO',
        'BASELINE_AUGMENTATION_PARITY':augmentation_parity,'NEW_TRAINING_AUTHORIZED':'NO',
        'APP_MODEL_REPLACEMENT_AUTHORIZED':'NO','EXTERNAL_TEST_AUTHORIZED':'NO',
        'MODEL_LOADING':False,'FORWARD_PASS':False,'TRAINING':False,'INFERENCE':False,
        'GPU_TRAINING':False,'test_images_used':0,'CO2Wounds_used':False,'STOP_AFTER_E12':True}


def stage(name,classification,current,verified,gaps):
    return {'stage':name,'classification':classification,'stock_8_3_53_behavior':current,
            'verified_in_E12':verified,'not_verified':gaps}


def draft_contract(path,data):
    # Safe continuation after an interrupted build: never replace divergent content.
    require(not (PRO/'E_PATCH_V3_freeze.json').exists(),'FAIL_ALREADY_FROZEN')
    if path.exists():require(read(path)==data,'FAIL_DRAFT_DIVERGED: '+str(path))
    else:save(path,data)


def build_contracts():
    model_guard()
    source=read(PRO/'E_PATCH_V3_mask_authority_contract.json');stats=source['stats']
    patch=read(PRO/'E_PATCH_V3_patch_audit.json');smoke=read(OUT/'real_dataloader_smoke.json')
    stages=[
        stage('resize','MASK_ADAPTER_POSSIBLE','image INTER_LINEAR; Instances retain polygon coordinates',
              'image/mask same target size; binary masks INTER_NEAREST; mask-derived bbox',[]),
        stage('Mosaic','MASK_ADAPTER_POSSIBLE','_mosaic4 placement; _cat_labels concatenates/clips Instances.segments',
              'four tiles, shared placements, occurrence-qualified IDs, no merging; fixed branch CPU smoke',
              ['stochastic p=.1 branch and buffered companion selection distribution/RNG parity',
               'native remove_zero_area_boxes versus raster visibility filtering parity']),
        stage('RandomPerspective','MASK_ADAPTER_POSSIBLE','shared homography image; apply_segments and box_candidates on vector-derived boxes',
              'fixed shared affine matrix, image linear/mask nearest, binary mask and derived tight bbox',
              ['full T@S@R@P@C parameter RNG pipeline not integrated',
               'box_candidates width/height>2, area_thr=.01, ar_thr100 versus mask-derived box/filter semantics']),
        stage('flip','MASK_ADAPTER_POSSIBLE','image flip and continuous Instances bbox/segment flips',
              'horizontal/vertical exact raster geometry; production flipud remains0',
              ['baseline stochastic flip sampling not replayed']),
        stage('HSV','MASK_ADAPTER_POSSIBLE','BGR HSV LUT transform with fixed gains; no mask usage',
              'pinned LUT formula, same input/output mask hash; fixed gains diagnostic',
              ['full RNG ordering across transforms not replayed']),
        stage('Albumentations photometric','MASK_ADAPTER_POSSIBLE','Blur .01, MedianBlur .01, ToGray .01, CLAHE .01; present in frozen recipe',
              'source inspection only; NOT silently removed from prospective recipe',
              ['full Albumentations execution and random-state parity not integrated in diagnostic adapter']),
        stage('Format','MASK_ADAPTER_POSSIBLE','expects polygons and invokes polygons2masks_overlap after augmentation',
              'mask-native area sort, overlap IDs1..N, same reordered cls/bboxes/IDs; 192x192 tensor',
              ['native polygon raster + INTER_LINEAR ratio4 differs from proposed NN mask downsampling',
               'schema compatibility does not certify trainer adapter integration']),
        stage('v8SegmentationLoss targets','MASK_NATIVE','consumes batch masks; overlap target mask is masks_i == target_gt_idx+1',
              'source inspection and CPU batch schema/index consistency; no model construction or loss forward',
              ['no actual trainer/model integration tested by authorization boundary']),
    ]
    blockers=[
        'BASELINE_AUGMENTATION_PARITY=NOT_VERIFIED: diagnostic primitives are not the complete frozen augmentation pipeline',
        'Mosaic buffering/companion RNG and RandomPerspective filtering with mask-derived boxes are not baseline-verified',
        'photometric Albumentations pipeline and close_mosaic InfiniteDataLoader integration not replayed',
    ]
    audit={'software':'Ultralytics 8.3.53','stages':stages,
        'sources':{str(p.relative_to(PACKAGE)):{'path':str(p),'sha256':file_sha(p)} for p in [PACKAGE/'data/augment.py',PACKAGE/'utils/loss.py',PACKAGE/'data/dataset.py',PACKAGE/'data/base.py',PACKAGE/'utils/instance.py',PACKAGE/'data/build.py']},
        'MASK_FIRST_NATIVE_CONTRACT':'NOT_FEASIBLE_WITH_UNMODIFIED_STOCK_SEGMENTATION_LOADER',
        'stock_reason':'stock Instances/Format require segments; arbitrary masks cannot pass through as sole GT without a custom adapter',
        'custom_adapter_scope':'deterministic isolated geometry/Format primitives and diagnostic DataLoader only; not a registered trainer adapter',
        'BASELINE_AUGMENTATION_PARITY':'NOT_VERIFIED','blockers':blockers,
        'no_silent_removal':True,'future_recipe_unchanged':True,
        'characterization_path':str((OUT/'control_characterization.json').relative_to(ROOT)),
        'no_byte_equivalence_requirement':'C3/P3 would share mask-first labels; historical pixel differences are characterization, not a requirement to change authority',
        'mask_ratio_contract':{'input':[768,768],'mask_ratio':4,'target_shape':[192,192],
            'downsample':'cv2.INTER_NEAREST; no bilinear threshold',
            'stock_difference':'polygon2mask uses cv2.resize default INTER_LINEAR on uint8; explicitly different',
            'overlap_mask':True,'ordering':'np.argsort(-per-instance downsampled area); same permutation for mask, cls, bbox and source IDs',
            'overlap_encoding':'background0, foreground sorted row+1; later/smaller instances overwrite overlapping pixels',
            'empty_policy':'fully empty full-resolution transformed masks removed and logged; low-resolution empty or fully occluded retained/logged; not silently made new GT',
            'bbox':'positive full-resolution transformed mask tight bbox, exclusive max; normalized xywh',
            'compatibility':'batch target schema and per-image local index verified; no actual network/prototype shape observation',
            'loss_if_proto_shape_differs':'pinned loss uses nearest interpolation; no claim of observing a forward pass'}}
    draft_contract(PRO/'E_PATCH_V3_augmentation_audit.json',audit)
    draft_contract(PRO/'E_PATCH_V3_dataloader_contract.json',{
        'status':'DIAGNOSTIC_SMOKE_PASS_FULL_TRAINER_NOT_VERIFIED','runtime_evidence':str((OUT/'real_dataloader_smoke.json').relative_to(ROOT)),
        'runtime_evidence_sha256':file_sha(OUT/'real_dataloader_smoke.json'),
        'guard_recheck_evidence':'experiments/results/e_patch_v3_feasibility_audit/real_dataloader_smoke_guard_verified.json',
        'guard_recheck_sha256':file_sha(OUT/'real_dataloader_smoke_guard_verified.json'),
        'FetchToken':['sample_id','epoch_token','generation'],'epoch_convention':'zero-based',
        'immutable_tokens':True,'target_rule':'sorted eligible_GT_ids[epoch_token % count] independent of worker/generation',
        'mosaic_diagnostic':'forced ON at269/OFF at270 to cover branches, NOT actual p=.1 distribution',
        'production_mosaic_recipe':{'p':.1,'close_mosaic':30,'epochs':300},
        'reset':'fresh workers/iterator; generation-tagged results; prior iterator exhausted/discarded, no shared mutable epoch',
        'production_reset_gap':'native InfiniteDataLoader reset/prefetch integration NOT_VERIFIED',
        'training_images_read':len(smoke['training_sample_ids']),'training_sample_ids':smoke['training_sample_ids'],
        'batch':{'img':'uint8 Bx3x768x768 BGR-to-RGB; trainer float/255 step not run',
                 'masks':'int32 Bx192x192 overlap instance index','cls':'float32 Nx1 class0',
                 'bboxes':'float32 Nx4 normalized xywh','batch_idx':'int64 N','instance_ids':'per-image list in target row order'},
        'workers':[0,2],'prefetch_factor':2,'pin_memory':False,'worker_parity':smoke['worker_parity'],
        'no_model_guard':'torch.load, torch.jit.load, nn.Module construction/call, Optimizer construction, CUDA lazy initialization blocked; ultralytics import blocked',
        'MODEL_LOADED':False,'FORWARD_PASS':False,'TRAINING':False,'test_images_used':0,'validation_images_used':0,'CO2Wounds_used':False})
    base=read(PRO/'E_PATCH_V2_control_config.json')
    common={k:deepcopy(base[k]) for k in ('architecture','augmentation','budget','fixed_budget_contract','initialization','loss','runtime_optimizer','software','training_args')}
    common.update(schema='E_PATCH_V3_MASK_FIRST',status='BLOCKED_FEASIBILITY_NOT_EXECUTABLE',execution_authorized=False,
        training_entrypoint=None,primary_comparison='Fresh C3 vs Fresh P3',historical_roles={'Phase C':'REFERENCE_ONLY','D2 C':'REFERENCE_ONLY','D2 S':'REFERENCE_ONLY'},
        shared_authority={'path':'experiments/protocols/E_PATCH_V3_training_mask_manifest.json','sha256':file_sha(PRO/'E_PATCH_V3_training_mask_manifest.json')},
        augmentation_contract='experiments/protocols/E_PATCH_V3_augmentation_audit.json',
        dataloader_contract='experiments/protocols/E_PATCH_V3_dataloader_contract.json',
        eligibility_contract='experiments/protocols/E_PATCH_V3_mask_authority_contract.json',
        sampling={'mode':'uniform_shuffled','replacement':False,'weights':'all1','seed':42,'epoch_anchor_draws':771,
                  'order_path':str((OUT/'anchor_orders.json').relative_to(ROOT)),'order_sha256':file_sha(OUT/'anchor_orders.json')},
        advancement_gate={k:(v.replace('C2','C3').replace('P2','P3') if isinstance(v,str) else v) for k,v in base['advancement_gate'].items()},
        evaluation_protocol='experiments/protocols/E_PATCH_V3_evaluation_protocol.json',
        evaluation_protocol_sha256=base['evaluation_protocol_sha256'],MULTI_SEED_PATCH_AUTHORIZED='NO')
    for arm,name,representation in [('C3','control','FULL512_MASK_FIRST'),('P3','experimental','MASK_BBOX_ELIGIBLE_PATCH256_ELSE_FULL512')]:
        config=dict(common,experiment_id=f'E_PATCH_V3_{arm}',arm_label=arm,representation=representation,
                    output_path=f'experiments/results/e_patch_v3_seed42_{name}')
        draft_contract(PRO/f'E_PATCH_V3_{name}_config.json',config)
    draft_contract(PRO/'E_PATCH_V3_evaluation_protocol.json',read(PRO/'E_PATCH_V2_evaluation_protocol.json'))
    # Exact inherited bytes, not a serializer reformat. Only this verified-equal,
    # newly generated, not-yet-frozen V3 output is replaced.
    shutil.copyfile(PRO/'E_PATCH_V2_evaluation_protocol.json',PRO/'E_PATCH_V3_evaluation_protocol.json')
    require(file_sha(PRO/'E_PATCH_V3_evaluation_protocol.json')==file_sha(PRO/'E_PATCH_V2_evaluation_protocol.json'),'FAIL_EVAL_CHANGED')
    result=feasibility(source_ok=stats['deterministic_masks']==stats['GT_instances']==965,
        retention_ok=patch['original182']['selected_retained']==182,loader_ok=smoke['worker_parity'],augmentation_parity='NOT_VERIFIED')
    result.update(blockers=blockers,scope_of_not_feasible='Not releasable under current frozen augmentation contract/evidence; not a claim that raster-mask learning is universally impossible',
                  completed_audit=True,stats=stats,original182=patch['original182'],V3_membership=patch['V3_membership'])
    save(OUT/'status.json',result)
    print(json.dumps(result,ensure_ascii=False))


def seal():
    model_guard();status=read(OUT/'status.json');tests=read(OUT/'verification.json')
    require(tests['exit_code']==0,'FAIL_VERIFICATION')
    guarded=read(OUT/'real_dataloader_smoke_guard_verified.json')
    require(guarded['guard_checks_passed'] and not guarded['CUDA_initialized'],'FAIL_CPU_GUARDS')
    original=read(OUT/'real_dataloader_smoke.json')
    require(all(guarded[k]==v for k,v in original.items()),'FAIL_DIAGNOSTIC_RECHECK_CHANGED')
    protected=read(OUT/'source_snapshot.json')
    for p,h in protected.items():require(file_sha(p)==h,'FAIL_PROTECTED_CHANGED: '+p)
    absent=[]
    for arm in ('control','experimental'):
        p=OUT.parent/f'e_patch_v3_seed42_{arm}';require(not p.exists(),'FAIL_TRAINING_OUTPUT_CREATED');absent.append(str(p.relative_to(ROOT)))
    c=read(PRO/'E_PATCH_V3_control_config.json');p=read(PRO/'E_PATCH_V3_experimental_config.json')
    differences=[k for k in c if c[k]!=p[k]]
    require(set(differences)=={'experiment_id','arm_label','representation','output_path'},'FAIL_PAIRED_CONFOUND')
    save(OUT/'integrity.json',{'protected_files_checked':len(protected),'protected_files_changed':0,
        'paired_config_differences':differences,'future_outputs_absent':absent,'test_images_used':0,
        'validation_image_pixels_read':0,'training_pixels_read_unique_images':4,
        'hash_only_reads':'protected snapshot includes inherited data/artifact/weight hashes; no checkpoint deserialization or evaluation'})
    report=ROOT/'docs/PHASE_E12_MASK_FIRST_FEASIBILITY_20260925.md'
    report_text=render_report(status,tests,len(protected))
    with report.open('x',encoding='utf8',newline='\n') as f:f.write(report_text)
    artifacts=[*PRO.glob('E_PATCH_V3_*.json'),*OUT.glob('*.json'),report,
        ROOT/'experiments/mask_first.py',ROOT/'experiments/mask_first_loader.py',ROOT/'experiments/phase_e12.py',
        ROOT/'experiments/phase_e12_finalize.py',ROOT/'tests/test_phase_e12.py']
    freeze=dict(status,created_at=datetime.now(timezone.utc).isoformat(),
        protected_files_unchanged=len(protected),future_outputs_absent=absent,
        artifacts_sha256={str(x.relative_to(ROOT)):file_sha(x) for x in artifacts},
        verification=tests,freeze_scope='audit evidence and blocked prospective C3/P3 design; NOT training authorization')
    save(PRO/'E_PATCH_V3_freeze.json',freeze)
    print(json.dumps({'status':status['PHASE_E12_STATUS'],'artifacts':len(artifacts),'protected_unchanged':len(protected)}))


def render_report(status,tests,protected_count):
    a=status['original182'];b=status['V3_membership']
    return f'''# Phase E1.2 — Mask-Authoritative Training Representation 可行性稽核

日期：2026-09-25。版本：E_PATCH_V3_MASK_FIRST。

## 1. 正式決策

```ini
PHASE_E12_STATUS = BLOCKED
MASK_FIRST_TRAINING_REPRESENTATION = NOT_FEASIBLE
BASELINE_AUGMENTATION_PARITY = NOT_VERIFIED
PATCH_BASED_TRAINING_ROUTE = STOP
READY_FOR_E2_MASK_FIRST_PAIRED_PATCH_TRAINING = NO
NEW_TRAINING_AUTHORIZED = NO
APP_MODEL_REPLACEMENT_AUTHORIZED = NO
EXTERNAL_TEST_AUTHORIZED = NO
```

本階段稽核工作已完成，但未達訓練放行條件。NOT_FEASIBLE 指「依目前封存條件及已驗證實作，不能放行」，不是證明所有 mask-first 方法皆不可能。未訓練、未載入模型、未 forward/inference、未用 GPU；test_images_used=0，未開啟 CO2Wounds 或外部測試。V1/V2 原樣保留。

## 2. 已通過：全訓練集的 source mask authority

771 張 FUSeg training images，共 965 個 annotation instances，全部重建兩次並逐 mask 比對，965/965 deterministic。每個原 annotation line ID 對應一個 mask，不拆 component、不合併 siblings。原始標註與影像都未重寫。

權威路徑嚴格沿用 E1.1：normalized polygon → float32 → pinned per-image resample_segments（1000，超過1000則max+1）→ ×512 → polygon2mask（int32 / fillPoly，ratio1）。其後只處理 mask，不作 repair、clipping polygon 或 re-rasterization。mask hash 同時包含 shape、dtype 與像素。

全771張共有 **36個 invalid source polygons**；原 eligible 圖片子集為17個，與 E1/E1.1 相符，其餘19個來自先前不需裁切的圖片，並非偷偷改寫舊結果。所有 source masks 為一個8-connected component，其中1個 instance有hole；洞與其 instance ID 都保留。這是 representation audit，不保證原醫療標註正確。

詳見 `experiments/protocols/E_PATCH_V3_training_mask_manifest.json`：逐 instance 記錄面積、4/8-connectivity、holes、bbox、polygon topology、image/label/mask SHA256。

## 3. Eligibility 明確變更

```ini
ELIGIBILITY_IDENTITY = CHANGED
```

| 定義 | Eligible images | Eligible GT |
|---|---:|---:|
| E1 原 polygon bbox | 161 | 182 |
| V3 authoritative-mask tight bbox | 142 | 161 |

移除21個 GT、無新增。完整身份差異列於 mask_authority_contract.json。新規則為 positive pixels 的 `[xmin,ymin,xmax+1,ymax+1]`，bbox area / 512² 嚴格小於0.0025。這是新的 V3 eligibility contract，**不能宣稱與 E1 intervention 完全相同**。未利用 validation 選擇定義。

256×256 source crop：中心取mask bbox，left/top=floor(center−128)再clamp至[0,256]，影像與所有instance masks共用整數slice。多目標按原GT ID排序，以zero-based epoch modulo目標數round-robin；generation不影響目標。

## 4. 182個歷史目標與新161個目標的裁切稽核

| 項目 | 原182個目標 | V3 161個目標 |
|---|---:|---:|
| Selected mask 完整保留 | {a['selected_retained']}/182 | {b['selected_retained']}/161 |
| Empty selected patches | 0 | 0 |
| Other GT FULL（cycle-observations） | {a['other_GT_states']['FULL']} | {b['other_GT_states']['FULL']} |
| Other GT PARTIAL（cycle-observations） | {a['other_GT_states']['PARTIAL']} | {b['other_GT_states']['PARTIAL']} |
| Other GT DROPPED（cycle-observations） | {a['other_GT_states']['DROPPED']} | {b['other_GT_states']['DROPPED']} |
| Selected保留但至少一個other GT被丟棄的cycles | {a['selected_retained_other_dropped_cycles']} | {b['selected_retained_other_dropped_cycles']} |
| Aggregate other-GT pixels retention | {a['aggregate_other_pixel_retention_ratio']:.2%} | {b['aggregate_other_pixel_retention_ratio']:.2%} |

0 unresolved source-topology blockers、0 vector/raster round-trip、0 raster-mismatch blockers。這裡的零差異是直接切權威mask，不是將不同representation的差異藏起來。Other-GT統計按每次cycle觀察計數，**不是unique GT數**；像素比例是總保留other pixels / 總source other pixels，不是每cycle比例的平均。Context損失仍然存在，不能因selected retention通過就忽略。

## 5. Augmentation及loss邊界：為何仍BLOCKED

以本機固定Ultralytics8.3.53源碼及SHA256查核，未import Ultralytics或建構YOLO。

| Stage | 分類 | 證據與限制 |
|---|---|---|
| Resize | MASK_ADAPTER_POSSIBLE | Image linear、mask nearest，bbox從當前mask重算 |
| Mosaic | MASK_ADAPTER_POSSIBLE | 4 tile幾何與ID唯一性通過；未驗證原buffer/companion RNG分布 |
| RandomPerspective | MASK_ADAPTER_POSSIBLE | 固定matrix共用image/mask；完整隨機matrix組合與box_candidates語義未整合驗證 |
| Flip | MASK_ADAPTER_POSSIBLE | H/V exact geometry通過；完整隨機順序未重播，recipe flipud仍為0 |
| HSV | MASK_ADAPTER_POSSIBLE | 固定gain LUT通過，mask不變 |
| Albumentations | MASK_ADAPTER_POSSIBLE | 原Blur/MedianBlur/ToGray/CLAHE未從recipe移除；diagnostic未執行完整鏈 |
| Format | MASK_ADAPTER_POSSIBLE | Native要求segments；新mask-native排序與index schema通過 |
| v8SegmentationLoss targets | MASK_NATIVE | Loss原本消費mask tensor；source-level與CPU batch結構可相容，未執行loss/model |

原生stock loader的Instances/Format仍以polygon segments為介面，不能直接宣稱它接受任意權威mask。自訂原語展示了不用polygon round-trip的技術路徑，但尚不是完整訓練adapter。最關鍵缺口：Mosaic companion buffer/RNG、RandomPerspective原box_candidates（wh>2、seg area_thr=.01、aspect<100）對mask-derived boxes的篩選語義、Albumentations鏈及native InfiniteDataLoader close/reset整合。

依使用者第15/38條，無法驗證baseline augmentation semantics就必須BLOCK。沒有偷偷關閉Mosaic、Perspective、Flip，也沒有把固定synthetic操作當作完整baseline replay。現階段停止patch路線，不延伸polygon heuristic。

### 最終target schema

768×768影像，mask_ratio=4，192×192 overlap mask。mask resize/downsampling一律INTER_NEAREST；不同於stock polygon2mask的uint8 INTER_LINEAR，**明確揭露而非宣稱byte-equivalent**。依downsample mask area降序排序，cls、normalized xywh bbox與instance ID同步；background=0，instance index=sorted row+1，後寫小mask覆蓋重疊像素。

Full-resolution空mask會有紀錄地移除；低解析度消失或overlap完全遮蔽仍保留target row並標記，不靜默創造或拆分GT。Loss中 `masks_i == (target_gt_idx+1)` 的index關係已由batch契約核對。這是無模型的schema驗證，不聲稱觀察到actual network prototype或跑過segmentation loss。

## 6. Control path characterization

合法simple synthetic polygon比較，完整紀錄見 `experiments/results/e_patch_v3_feasibility_audit/control_characterization.json`。

| Operation | Full-resolution XOR pixels | ratio4 XOR pixels |
|---|---:|---:|
| Identity | 0 | 37 |
| Resize512→768 | 300 | 72 |
| Horizontal flip | 0 | 45 |
| Fixed affine | 117 | 48 |

比較是pinned polygon raster primitive經固定座標transform，對上raster-native操作，不是完整歷史augmentation replay。這些差異只是characterization；未為匹配歷史而改動mask authority。C3/P3會共同使用新label view；歷史C/D2結果只能REFERENCE_ONLY。

## 7. 真實CPU DataLoader診斷

Synthetic與4張實際training samples均以Torch DataLoader、workers=0/2執行，prefetch_factor=2，pin_memory=false。Training subset：fuseg__0020、0022、0012、0011，依training labels決定invalid／eligible／multi-GT／其餘代表，不看validation outcomes。

兩個arm均驗證：image `[4,3,768,768]`、mask `[4,192,192]`、cls `[N,1]`、bbox `[N,4]`、batch_idx `[N]`；workers0/2的tensor hashes及metadata完全相同。Immutable FetchToken(sample_id,epoch_token,generation)避免prefetch讀取mutable epoch。診斷強制epoch269走Mosaic、270關閉；generation1建立新iterator/worker後，像素與generation0同epoch一致、無舊token混入。

**此forced branch不是production p=.1 RNG，new iterator reset也不是已驗證native InfiniteDataLoader reset。** 全程MODEL_LOADED=false、FORWARD_PASS=false，僅CPU tensors與影像處理。

## 8. 凍結的C3/P3設計（不可執行）

兩者同YOLO11m-seg、class0 Wound、seed42、同ISIC initialization SHA256、768/batch4、AdamW lr.0005、相同loss gains與全部augmentation數值；300epochs、patience80、scheduled optimizer calls3741，applied+skipped=3741且unknown=0。若未完成300epochs則NOT_VALID、不resume或補跑。

771 IDs/epoch，uniform without replacement，300×771共同anchor orders已凍結。不同欄位只有arm/id/output與representation：C3 full512；P3 mask-eligible patch256，否則full512。兩份config都execution_authorized=false、training_entrypoint=null。

未來endpoint仍Very-small Recall（49GT），P3相對C3 very-small TP≥+2；small/medium TP≥−1；Precision/F1≥−1pp；large TP不得下降；crop-complete≥−1image。191張development validation只維持原label與full-image評価：conf.10、floor.01、NMS.70、match.50、768、crop margin15%、RGB/BGR corrected。Evaluation protocol與V2 **byte-identical**；本階段未跑validation或改label。

## 9. 測試、完整性及停止

{tests['summary']}。TDD以mask/geometry、overlap target與真實DataLoader等使用者指定介面逐步驗證，不以mock網路產生訓練結果。

{protected_count}個繼承保護檔案hash全數未變（包含V1/V2及既有source snapshot）。Hash核對不等於checkpoint載入。Training image pixels僅讀4張；validation image pixels=0、test_images_used=0。兩個未來training output目錄均不存在。

所有JSON、程式與報告SHA256列在 `experiments/protocols/E_PATCH_V3_freeze.json`。報告不是performance改善證據；沒有新權重、沒有新mAP、沒有App replacement。

下一階段可討論 **HIGHER_INPUT_SCALE** 作單一研究介入，理由是保留full-image context且避免本次patch representation整合缺口；目前未驗證效果、未設定新門檻、未執行任何實驗。亦可另外討論feature-pyramid或loss/assignment，但不得同時混入。**本階段到此停止，E2未授權。**
'''


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');a=p.parse_args()
    seal() if a.seal else build_contracts()
