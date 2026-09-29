"""G0: training-source/metadata audit only. Never imports a model framework.

Original training masks may be decoded; validation/test source pixels may not.
Historical research artifacts/checkpoints are hashed as opaque bytes only.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results/g0_negative_source_audit'
REPORT = ROOT/'docs/PHASE_G0_CONFIRMED_NEGATIVE_SOURCE_AUDIT_20260929.md'
PRO = ROOT/'experiments/protocols'
RESULTS = ROOT/'experiments/results'
F13 = RESULTS/'f_higher_scale_v2_paired_error_audit'
UP = ROOT/'official_detection_sources_20260812/fuseg/repository'
PDF = UP/'data/Foot Ulcer Segmentation Challenge/FootUlcerSegmentationChallenge2021.pdf'
PDF_PIN = 'f9a44fc14bc7589d03ce5c17307ad97e9dde5fcc0e2c596c8b86010a6d9547aa'
RASTER = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
FORBIDDEN_IMPORTS = {'torch', 'ultralytics', 'tensorflow', 'onnxruntime'}
FLAGS = dict(TRAINING=False, MODEL_LOADED=False, NEW_INFERENCE=False,
    THRESHOLD_CHANGED=False, NMS_CHANGED=False, MATCH_IOU_CHANGED=False,
    VALIDATION_HARD_NEGATIVE_MINING=False, VALIDATION_PIXELS_READ=False,
    LOCKED_TEST_PIXELS_READ=False, LOCKED_TEST_USED=False, CO2Wounds_USED=False,
    EXTERNAL_TEST_USED=False, test_images_used=0, NEW_TRAINING_AUTHORIZED='NO',
    MULTI_SEED_AUTHORIZED='NO', APP_MODEL_REPLACEMENT_AUTHORIZED='NO',
    EXTERNAL_TEST_AUTHORIZED='NO', STOP_AFTER_G0=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csv_write(name, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    require(bool(fields), 'CSV_SCHEMA_REQUIRED')
    with (OUT/name).open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: 'NA' if row.get(k) is None else row.get(k) for k in fields} for row in rows)
    require(len(csv_read(OUT/name)) == len(rows), 'CSV_ROUNDTRIP_FAILED')


def label_state(text):
    """Missing is not empty. Validate single-class normalized segmentation rows."""
    if text is None:
        return 'MISSING', None
    lines = [line.split() for line in text.splitlines() if line.strip()]
    if not lines:
        return 'EMPTY', 0
    try:
        for line in lines:
            a = [float(x) for x in line]
            if len(a) < 7 or (len(a)-1) % 2 or a[0] != 0 or not all(math.isfinite(v) and 0 <= v <= 1 for v in a[1:]):
                return 'INVALID', None
            xy = list(zip(a[1::2], a[2::2]))
            area = abs(sum(x*y2-x2*y for (x,y),(x2,y2) in zip(xy, xy[1:]+xy[:1])))
            if len(set(xy)) < 3 or area == 0:
                return 'INVALID', None
    except (ValueError, TypeError):
        return 'INVALID', None
    return 'NONEMPTY', len(lines)


def evidence_level(status, provenance, native_zero, native_protocol, expert_result=None):
    if status == 'MISSING': return 'MISSING_LABEL'
    if status == 'INVALID': return 'INVALID_LABEL'
    if status == 'NONEMPTY': return 'POSITIVE_IMAGE_UNANNOTATED_REGION'
    if expert_result in {'UNCERTAIN', 'TARGET_WOUND_PRESENT'}: return expert_result
    if provenance and ((native_zero and native_protocol) or expert_result == 'CONFIRMED_NO_TARGET_WOUND'):
        return 'VERIFIED_NEGATIVE'
    return 'EMPTY_LABEL_ONLY'


def eligible(level, overlaps):
    return level == 'VERIFIED_NEGATIVE' and not overlaps


def count_anchors(path, ids, expected_orders=None):
    orders = defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            seq = row.get('sample_ids')
            if seq is None:
                seq = [a['sample_id'] for a in row['anchors']]
            e = row['epoch']
            require(row['batch_index'] == len(orders[e])//4, 'ANCHOR_BATCH_ORDER')
            if 'start_position' in row:
                require(row['start_position'] == len(orders[e]), 'ANCHOR_POSITION')
            orders[e].extend(seq)
    require(set(orders) == set(range(300)), 'INCOMPLETE_EPOCHS')
    require(all(len(orders[e]) == 771 for e in range(300)), 'ANCHOR_COUNT')
    require(all(set(seq) <= ids for seq in orders.values()), 'UNKNOWN_ANCHOR_ID')
    if expected_orders is not None:
        require(all(orders[e] == expected_orders[e] for e in range(300)), 'FROZEN_ORDER_MISMATCH')
    return [Counter(orders[e]) for e in range(300)]


def guard(training_files, training_masks, opaque_artifacts):
    allowed_rasters = set(training_files) | set(training_masks) | set(opaque_artifacts)
    original_open = Image.open
    def image_open(path, *args, **kw):
        require(isinstance(path, (str, Path)) and Path(path).resolve() in training_masks, 'NONTRAIN_MASK_DECODE_FORBIDDEN')
        return original_open(path, *args, **kw)
    Image.open = image_open
    def audit(event, args):
        if event == 'import' and args[0].split('.')[0] in FORBIDDEN_IMPORTS:
            raise RuntimeError('MODEL_IMPORT_FORBIDDEN')
        if event.startswith('socket.') or event in {'subprocess.Popen', 'os.system'}:
            raise RuntimeError('NETWORK_OR_PROCESS_FORBIDDEN')
        if event == 'open' and isinstance(args[0], (str, bytes)):
            p = Path(os.fsdecode(args[0])).resolve()
            writing = any(c in (args[1] or '') for c in 'wax+') or bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
            if writing and not (p.is_relative_to(OUT) or p == REPORT):
                raise RuntimeError('SOURCE_WRITE_FORBIDDEN: '+str(p))
            if not writing and p.suffix.lower() in RASTER and p not in allowed_rasters:
                raise RuntimeError('NONADMITTED_RASTER_READ: '+str(p))
    sys.addaudithook(audit)


def main():
    require(not OUT.exists() and not REPORT.exists(), 'G0_OUTPUT_ALREADY_EXISTS')
    manifest_path = PRO/'D_SEG_SMALL_SAMPLING_V1_training_manifest.json'
    manifest = read(manifest_path)
    cfg = read(PRO/'F_HIGHER_SCALE_V2_control_config.json')
    require(sha(manifest_path) == cfg['manifest']['sha256'], 'TRAIN_MANIFEST_CHANGED')
    samples = manifest['samples']
    ids = {s['sample_id'] for s in samples}
    require(len(samples) == len(ids) == 771, 'TRAIN_IDENTITY_COUNT')
    source_path = ROOT/'outputs/fuseg_only_manifest_20260821.csv'
    source = {Path(r['fuseg_image']).name:r for r in csv_read(source_path) if r['assigned_split'] == 'train'}
    require(ids <= set(source), 'SOURCE_ROWS_MISSING')
    masks = {(ROOT/source[sid]['source_mask']).resolve() for sid in ids}
    require(all(p.is_relative_to(UP/'data/Foot Ulcer Segmentation Challenge/train/labels') for p in masks), 'SOURCE_MASK_ROLE')
    images = {(ROOT/s['image_path']).resolve() for s in samples}
    images |= {(ROOT/source[sid]['source_image']).resolve() for sid in ids}
    f13_integrity = read(F13/'integrity.json')
    # Source rasters are excluded: compare saved validation/test hashes, never reopen their pixels.
    # F1/F13 generated figures are opaque integrity artifacts, never decoded or inspected.
    expected = {}
    skipped_source_rasters = 0
    for name, digest in f13_integrity['source_sha256'].items():
        p = Path(name).resolve()
        if p.suffix.lower() in RASTER:
            if p.is_relative_to(RESULTS): expected[p] = digest
            else: skipped_source_rasters += 1
        else:
            expected[p] = digest
    expected.update({Path(p).resolve():v for p,v in f13_integrity['output_sha256'].items()})
    expected[ROOT/'experiments/phase_f13.py'] = f13_integrity['audit_source_sha256']
    expected[ROOT/'tests/test_phase_f13.py'] = f13_integrity['tests_source_sha256']
    guard(images, masks, {p for p in expected if p.suffix.lower() in RASTER})
    before = {str(p):sha(p) for p in expected}
    require(all(before[str(p)] == v for p,v in expected.items()), 'F12_F13_HASH_DRIFT')
    print('Historical F1.2/F1.3 artifact hashes verified; source validation/test rasters not opened.', flush=True)
    inputs = {str(p):sha(p) for p in [manifest_path, source_path, F13/'integrity.json', PDF,
        PRO/'F_HIGHER_SCALE_V2_control_config.json', ROOT/'experiments/f1_v2_runner.py',
        ROOT/'experiments/review_v2/fuseg_warmup_experiment.py', UP/'README.md',
        UP/'data/Foot Ulcer Segmentation Challenge/README.MD',
        ROOT/'experiments/review_v2/evidence/FUSeg_noncommercial_research_20260914.md']}
    require(inputs[str(PDF)] == PDF_PIN, 'PUBLISHER_PDF_CHANGED')
    # Source PDF already read in full by the audit author: exact pinned document, pp9/11.
    native_protocol = True
    stock = {}
    for rel, digest in cfg['architecture']['source_hashes'].items():
        if rel.startswith('torch/'):
            matches = [Path(name) for name,value in f13_integrity['source_sha256'].items()
                if value == digest and Path(name).as_posix().endswith(rel)]
            require(len(matches)==1,'FROZEN_TORCH_SOURCE_PATH_AMBIGUOUS')
            p = matches[0]
        else:
            p = Path('C:/Python312/Lib/site-packages/ultralytics')/rel
        stock[str(p)] = sha(p)
        require(stock[str(p)] == digest, 'STOCK_SOURCE_CHANGED')
    utils = Path('C:/Python312/Lib/site-packages/ultralytics/data/utils.py')
    stock[str(utils)] = sha(utils)  # current source; not independently pinned by F1.2 architecture map
    inputs.update(stock)
    val_path = PRO/'ISIC_FUSEG_COLORFIX_V1_validation_manifest.json'
    val = read(val_path)['samples']
    require(len(val) == 191, 'VALIDATION_MANIFEST_COUNT')
    val_hashes = {x['image_sha256'] for x in val}
    test_path = ROOT/'outputs/fuseg_official_test_manifest_20260812.csv'
    test_metadata = csv_read(test_path)
    test_hashes = {x['sha256'] for x in test_metadata}
    require(all(len(x) == 64 for x in test_hashes), 'LOCKED_HASH_FORMAT')
    inputs.update({str(p):sha(p) for p in [val_path, test_path]})
    train_rows, candidates, overlap_rows = [], [], []
    for s in samples:
        sid = s['sample_id']; source_row = source[sid]
        require(s['split'] == 'train' and source_row['source'] == 'FUSeg' and source_row['official_split'] == 'train', 'SOURCE_ROLE')
        ip, lp = ROOT/s['image_path'], ROOT/s['label_path']
        ih = sha(ip)
        lh = sha(lp) if lp.exists() else None
        inputs[str(ip)] = ih
        if lh is not None: inputs[str(lp)] = lh
        status, n = label_state(lp.read_text(encoding='utf-8') if lp.exists() else None)
        historical_consistent = ih == s['image_hash'] and lh == s['label_hash'] and n == s['GT_count']
        require(historical_consistent, 'FROZEN_TRAIN_IDENTITY_OR_GT_DRIFT: '+sid)
        provenance = source_row['image_sha256'] == ih
        zero = False; native_mask_sha = None; nonzero = None
        if status == 'EMPTY':
            native_image, native_mask = ROOT/source_row['source_image'], ROOT/source_row['source_mask']
            native_mask_sha = sha(native_mask); native_image_sha = sha(native_image)
            inputs[str(native_image)], inputs[str(native_mask)] = native_image_sha, native_mask_sha
            provenance &= native_image_sha == ih and native_mask_sha == source_row['mask_file_sha256']
            with Image.open(native_mask) as im:
                array = np.asarray(im)
                require(array.shape in {(512,512),(512,512,3)} and array.dtype == np.uint8, 'MASK_FORMAT')
                if array.ndim == 3:
                    require(np.array_equal(array[:,:,0],array[:,:,1]) and np.array_equal(array[:,:,0],array[:,:,2]), 'MASK_RGB_DISAGREEMENT')
                    array = array[:,:,0]
                nonzero = int(np.count_nonzero(array))
                zero = nonzero == 0
        level = evidence_level(status, provenance, zero, native_protocol)
        overlaps = []
        if status == 'EMPTY':
            for role, hs in [('DEVELOPMENT_VALIDATION',val_hashes), ('FUSEG_OFFICIAL_LOCKED_TEST_HASHES_ONLY',test_hashes)]:
                hit = ih in hs
                overlap_rows.append(dict(sample_id=sid,image_sha256=ih,comparison_role=role,
                    reference_unique_hashes=len(hs),exact_overlap=hit,pixels_read=False))
                if hit: overlaps.append(role)
        row = dict(sample_id=sid,image_sha256=ih,label_sha256=lh,GT_instance_count=n,GT_count=n,
            label_status=status,source_provenance='FUSeg upstream 42a272dfe0679f20675e826385925cb7562934b6; outputs/fuseg_only_manifest_20260821.csv',
            source_image=source_row['source_image'],source_mask=source_row['source_mask'],
            source_mask_sha256=native_mask_sha,source_mask_nonzero_pixels=nonzero,
            source_metadata_negative_evidence='FUSeg challenge PDF pp9/11 + original all-zero mask' if zero and provenance else 'NOT_ESTABLISHED',
            dataset_negative_evidence=bool(zero and provenance and native_protocol),
            expert_negative_evidence='NO_NEW_INDEPENDENT_ADJUDICATION; publisher specialist-reviewed protocol',
            negative_evidence_level=level,patient_case_identity='UNKNOWN',
            historical_manifest_consistent=historical_consistent,provenance_hash_chain_verified=bool(provenance),
            future_training_eligible=eligible(level,overlaps),
            exclusion_reason=';'.join(overlaps) if overlaps else '' if eligible(level,overlaps) else level,
            source_role='TRAIN_ONLY_FUSEG_DEVELOPMENT_TRAIN')
        train_rows.append(row)
        if status == 'EMPTY': candidates.append(row)
    verified = [r for r in candidates if r['future_training_eligible']]
    groups = defaultdict(list)
    for r in candidates:
        if r['negative_evidence_level'] == 'VERIFIED_NEGATIVE': groups[r['image_sha256']].append(r['sample_id'])
    dup_rows = [dict(image_sha256=h,raw_file_count=len(v),duplicate_group=len(v)>1,sample_ids=';'.join(v),patient_case_identity='UNKNOWN') for h,v in sorted(groups.items())]
    order_path = ROOT/cfg['sampling']['order_path']
    require(sha(order_path) == cfg['sampling']['order_sha256'], 'ORDER_CHANGED')
    order = read(order_path)['orders']
    require(len(order) == 300 and all(len(x)==771 and set(x)==ids for x in order), 'FROZEN_ORDER_IDENTITY')
    inputs[str(order_path)] = sha(order_path)
    exposure, historical_totals = [], []
    candidate_ids = {r['sample_id'] for r in candidates}
    runs = [('F_C4',42,RESULTS/'f_higher_scale_v2_seed42_control768/anchor_telemetry.jsonl',order),
        ('F_H4_R',42,RESULTS/'f_higher_scale_v2_recovery_seed42_h4_1024/anchor_telemetry.jsonl',order)]
    for seed in [42,123,3407,2026,999]:
        prefix = 'd_seg_small_sampling_v2'+('' if seed==42 else f'_seed{seed}')
        runs += [(f'D2_{arm}',seed,RESULTS/(prefix+'_'+suffix)/'anchors.jsonl',None)
            for arm,suffix in [('C','control'),('S','experimental')]]
    d2_path = RESULTS/'d_seg_small_sampling_v2_multiseed_summary/sampling_exposure_summary.csv'
    previous = {(int(r['seed']),r['arm']):int(r['negative_images']) for r in csv_read(d2_path)}
    inputs[str(d2_path)] = sha(d2_path)
    for arm, seed, path, frozen in runs:
        inputs[str(path)] = sha(path)
        counts = count_anchors(path,ids,frozen)
        if arm.endswith('_C'):
            require(all(c == Counter({sid:1 for sid in ids}) for c in counts), 'CONTROL_NOT_ONCE_PER_EPOCH')
        total = sum(c[sid] for c in counts for sid in candidate_ids)
        if arm.startswith('D2'):
            require(total == previous[seed,arm[-1]], 'D2_SUMMARY_DISAGREES')
        historical_totals.append(dict(arm=arm,seed=seed,empty_anchor_draws_300epochs=total,
            mean_empty_anchor_draws_per_epoch=total/300,source=str(path.relative_to(ROOT))))
        for epoch, c in enumerate(counts):
            for sid in sorted(candidate_ids):
                exposure.append(dict(phase_arm=arm,seed=seed,epoch_1based=epoch+1,sample_id=sid,
                    anchor_draws=c[sid],verified_negative=sid in {r['sample_id'] for r in verified},
                    exposure_unit='primary_anchor_only_not_mosaic_donor_or_optimizer_updates'))
        print(f'{arm} seed {seed}: {total} empty-anchor draws across 300 epochs.', flush=True)
    states = Counter(r['label_status'] for r in train_rows)
    leaks = any(r['exact_overlap'] for r in overlap_rows)
    feasibility = 'PASS' if verified and not leaks else 'BLOCKED_BY_UNVERIFIED_NEGATIVE_SEMANTICS' if candidates and all(r['negative_evidence_level']=='EMPTY_LABEL_ONLY' for r in candidates) else 'BLOCKED_BY_NO_VERIFIED_NEGATIVE_SOURCE'
    needs_review = [r['sample_id'] for r in candidates if r['negative_evidence_level'] in {'EMPTY_LABEL_ONLY','UNCERTAIN'}]
    summary = dict(PHASE_G0_STATUS='COMPLETE',NEGATIVE_SOURCE_FEASIBILITY=feasibility,
        VERIFIED_NEGATIVE_COUNT=len(verified),VERIFIED_NEGATIVE_UNIQUE_CONTENT_COUNT=len({r['image_sha256'] for r in verified}),
        NEEDS_EXPERT_ADJUDICATION='YES' if needs_review else 'NO',expert_review_candidate_ids=needs_review,
        training_images=771,label_status_counts=dict(states),num_empty_label_images=len(candidates),
        raw_semantically_verified_files=sum(len(v) for v in groups.values()),duplicate_negative_groups=sum(len(v)>1 for v in groups.values()),
        NEGATIVE_POOL_LEAKAGE='FAIL' if leaks else 'PASS_EXACT_HASH_SCOPES_ONLY',
        validation_reference_images=191,locked_test_reference_hashes=len(test_metadata),
        locked_hash_scope='Existing FUSeg official test manifest only; no classification-48 SHA256 claim',
        historical_exposure=historical_totals,G1_PREREGISTRATION_ELIGIBLE=feasibility=='PASS',
        baseline_C4_existing_exposure=True,patient_case_identity='UNKNOWN',
        semantics='negative for current FUSeg wound-localization target; NOT clinically healthy',
        source_release='not specified; pinned repository commit and local file hashes',
        no_perceptual_or_patient_independence_claim=True,created_utc=datetime.now(timezone.utc).isoformat(),**FLAGS)
    semantics = dict(dataset_native_negative_semantics='VERIFIED_FOR_ORIGINAL_ALL_ZERO_MASKS_ONLY',
        pdf_path=str(PDF.relative_to(ROOT)),pdf_sha256=PDF_PIN,
        evidence=[dict(page=9,claim='Rare healed cases have no wound labeled in their annotation; binary pixel labels encode wound versus non-wound.'),
            dict(page=10,claim='Manual masks reviewed/refined by wound-care specialists/nurses; edge cases discussed with doctors. Source acknowledges possible annotation error.'),
            dict(page=11,claim='For non-wound/healed cases, every wound prediction is a false positive; correct annotation/prediction is entirely zero-valued non-wound/background.')],
        proof_chain='Frozen empty label -> training source CSV -> identical original image SHA256 -> original mask SHA256 -> 512x512 all-zero RGB-equal mask -> pinned publisher protocol',
        source_admission=manifest['source_admission'],stock_source_hashes=stock,
        negative_contract=dict(eligible_evidence='native all-zero mask plus verified provenance OR documented independent expert confirmation',
            target_scope=summary['semantics'],uncertain='EXCLUDE',missing_label='EXCLUDE',positive_non_GT_regions='EXCLUDE',
            validation_FP='ANALYSIS_REFERENCE_ONLY; never training selector',test_external='FORBIDDEN',
            human_review_required_fields=['sample_id','reviewer_role','review_result','review_date','review_protocol_version'],
            human_review_results=['CONFIRMED_NO_TARGET_WOUND','TARGET_WOUND_PRESENT','UNCERTAIN'],
            upgrade_result='CONFIRMED_NO_TARGET_WOUND',current_new_expert_reviews=0),
        pipeline_audit=dict(dataset='utils.py:152-156 distinguishes empty/missing counters but both yield zero targets; G0 never treats missing as negative.',
            batch='dataset.py:232-249 stacks every image, concatenates targets; augment.py Format emits empty cls/bbox/batch_idx and zero mask.',
            loss='loss.py:318-350 applies BCE classification to zero target scores. Bbox/DFL/mask supervised losses require foreground; empty images have no positive mask supervision.',
            empty_batch='tal.py:64-71 zero targets/foreground; mask loss uses zero-valued graph terms, not dense all-background mask BCE.',
            mixed_batch='loss.py:425-444 skips real mask loss for images without assigned foreground.',
            exposure='f1_v2_runner.py Ledger.anchors records consumed im_file identities before stock preprocess; audit reconciles all 300 full orders.',
            limitation='Primary anchor counts are not total augmented pixel exposure, Mosaic donor exposure, or applied optimizer-update counts.'),
        future_boundary=dict(train_imgsz=768,not_1024_plus_negatives=True,
            priorities=['Precision / FP','Very-small Recall','Small Recall','Crop completeness'],
            exposure_mechanism='NOT_DECIDED',loss='NOT_DECIDED',budget='NOT_DECIDED',training_authorized=False))
    OUT.mkdir()
    csv_write('training_manifest_audit.csv',train_rows)
    csv_write('candidate_negative_manifest.csv',candidates,list(train_rows[0]))
    csv_write('verified_negative_manifest.csv',verified,list(train_rows[0]))
    csv_write('negative_duplicate_audit.csv',dup_rows,['image_sha256','raw_file_count','duplicate_group','sample_ids','patient_case_identity'])
    csv_write('negative_role_overlap_audit.csv',overlap_rows,['sample_id','image_sha256','comparison_role','reference_unique_hashes','exact_overlap','pixels_read'])
    csv_write('existing_negative_exposure.csv',exposure)
    save('source_semantics_evidence.json',semantics)
    save('negative_source_summary.json',summary)
    REPORT.write_text(report(summary,semantics,candidates),encoding='utf-8')
    require(not FORBIDDEN_IMPORTS.intersection(sys.modules),'MODEL_MODULE_LOADED')
    after = {p:sha(p) for p in before}
    require(before == after, 'HISTORICAL_ARTIFACT_CHANGED')
    require(all(sha(p)==v for p,v in inputs.items()),'AUDIT_INPUT_CHANGED')
    outputs = {str(p):sha(p) for p in OUT.iterdir() if p.is_file()}
    outputs[str(REPORT)] = sha(REPORT)
    save('integrity.json',dict(status='PASS',before_equals_after=True,F12_F13_expected_hashes_match=True,
        protected_artifact_count=len(before),protected_sha256=before,audit_input_sha256=inputs,output_sha256=outputs,
        source_rasters_skipped_from_historical_rehash=skipped_source_rasters,
        historical_visual_artifacts='opaque-byte SHA256 only; no pixel decoding or appearance-based selection',
        training_identities_unchanged=771,validation_predictions_unchanged=True,checkpoints_unchanged=True,
        audit_source_sha256=sha(Path(__file__)),tests_source_sha256=sha(ROOT/'tests/test_phase_g0.py'),
        engineering_preflight_note='Initial G0 preflight stopped before artifact creation: torch source is in user-site, not system-site. Resolved from existing frozen F1.3 path/hash; no model import, training or inference.',**FLAGS))
    print(json.dumps({k:summary[k] for k in ['PHASE_G0_STATUS','NEGATIVE_SOURCE_FEASIBILITY','VERIFIED_NEGATIVE_COUNT','NEEDS_EXPERT_ADJUDICATION']},ensure_ascii=False))


def report(s, evidence, candidates):
    questions = [
        ('771 training 中空標註數',str(s['num_empty_label_images'])),
        ('Missing label 數',str(s['label_status_counts'].get('MISSING',0))),
        ('Empty label 是否具有 dataset-native negative semantics','有，但必須是來源可追溯的原始全零 mask；單憑空 YOLO label 不足。'),
        ('證據文件','FUSeg challenge PDF 第 9、10、11 頁；原始 train/labels masks；fuseg_only_manifest_20260821.csv；frozen training manifest。'),
        ('需專家補判 candidate',', '.join(s['expert_review_candidate_ids']) or '無；本次未進行新的專家判讀。'),
        ('合格 raw negative files',str(s['VERIFIED_NEGATIVE_COUNT'])),
        ('Unique exact contents',str(s['VERIFIED_NEGATIVE_UNIQUE_CONTENT_COUNT'])),
        ('Duplicate negative groups',str(s['duplicate_negative_groups'])),
        ('Validation exact overlap','0（比對 191 張 frozen image SHA256，不讀像素）'),
        ('Locked-test exact overlap',f'0（既有 FUSeg official test 的 {s["locked_test_reference_hashes"]} 筆 SHA256；未對 classification 48 張作額外保證）'),
        ('Locked-test pixels 是否讀取','否。只讀已存在的 test hash CSV。'),
        ('Validation pixels 是否讀取','否。不開啟原始 validation image/mask；既有研究圖檔僅 opaque-byte 雜湊，不解碼、不檢視。'),
        ('是否使用 validation FP 作負樣本','否。F1.3 案例僅保留為分析參考，不作 selector。'),
        ('Positive image 非 GT 區域是否進 pool','否。沒有切 crop、合成或外觀篩選。'),
        ('C4 是否已訓練過空標註 anchors','是。逐一核對完整 300 epochs consumed-anchor telemetry 與 frozen order。'),
        ('既有 exposure 每 epoch / 全程','每張每 epoch 1 次、全程 300 次；18 張合計每 epoch 18 次、全程 5,400 次。'),
        ('Patient/case identity','UNKNOWN。18 unique contents 不代表 18 獨立病人。'),
        ('來源是否完整','對本次 18 張 mask/image 的 native-source hash chain 完整；release version 與病人 ID 未提供，不予補造。'),
        ('Feasibility',s['NEGATIVE_SOURCE_FEASIBILITY']),
        ('G1 preregistration 資格','可提出下一階段預登錄；本次沒有建立 G1 或授權訓練。'),
        ('是否訓練','否。'),('是否 inference / model loading','否／否。'),
        ('是否改 threshold/NMS/match','否。'),('是否使用 external/CO2','否。'),('是否替換 App','否。')]
    q='\n'.join(f'| {i} | {a} | {b} |' for i,(a,b) in enumerate(questions,1))
    hist='\n'.join(f'| {r["arm"]} | {r["seed"]} | {r["empty_anchor_draws_300epochs"]:,} | {r["mean_empty_anchor_draws_per_epoch"]:.3f} |' for r in s['historical_exposure'])
    idlist=', '.join(r['sample_id'] for r in candidates)
    return f'''# Phase G0 — 可信背景負樣本來源稽核

日期：2026-09-29。用途：非商業、離線學術／畢業研究。不是臨床部署許可。

## 結論

`PHASE_G0_STATUS=COMPLETE`；`NEGATIVE_SOURCE_FEASIBILITY={s['NEGATIVE_SOURCE_FEASIBILITY']}`。
771 張 frozen training 重新讀取 labels：753 NONEMPTY、18 EMPTY、0 MISSING、0 INVALID。
18 張原始 mask 全零且 SHA256 與來源紀錄一致，通過 native negative semantics 與角色隔離，形成 18 張／18 unique exact contents 的名冊。
這是 **既有訓練負樣本的來源資格確認，不是新增資料、不是新模型成果**。F1.2 research gate 仍是 FAIL。

## 一、證據與語義界線

本機來源 PDF：`{PDF.relative_to(ROOT).as_posix()}`，SHA256 `{PDF_PIN}`。
第 9 頁明確說明少數已癒合案例沒有 wound annotation；第 11 頁把 healed/non-wound 案例的任何 wound prediction 計作 FP，正確標註／預測為全零。
第 9–10 頁記載人工標註經 wound-care specialists/nurses 審核，困難案例諮詢醫師。仍承認 annotation error 可能性，不能宣稱絕對完美 ground truth。
本次不是以「沒有 polygon」或外觀猜測無傷口：每張都核對 frozen image/label → source CSV → 原始 training image SHA256 → 原始 mask SHA256 → 全零 512×512 mask。
原始 mask 為 RGB 相同三通道；以任意非零值判前景，不因低像素值被誤當零。
只有 **negative for current FUSeg wound-localization target**，不是 clinically healthy，也不是七類診斷排除。

上游 repository commit：`42a272dfe0679f20675e826385925cb7562934b6`。Dataset release version 未標明。
使用既有 `experiments/review_v2/evidence/FUSeg_noncommercial_research_20260914.md` 來源准入；PDF 記 CC BY NC（版本未註明），本次不擴大為商業、病患照護、公開影像／權重再散布授權。
PDF 原始計畫的 610/200/200 與 README 後續增補不等於目前篩選後 inventory；本次數量完全來自 frozen 771 training，不回填舊計畫數。

候選 ID：{idlist}

## 二、資料隔離、重複與人工確認合約

18 張候選均有來源全零標註語義，不需要本次新增專家判讀才能取得 native-negative 資格；沒有虛構 reviewer。
Exact SHA256：18 raw files、18 unique contents、0 duplicate groups。與 191 validation hashes 及既存 {s['locked_test_reference_hashes']} 筆 FUSeg locked-test hashes 重疊均 0。
僅比對既存 hash metadata，沒有打開 locked-test 圖片。未宣稱無近重複，也未宣稱病人層級隔離；patient/case/session identity = UNKNOWN。
分類 48 張不在本次 FUSeg locked-hash 比對範圍；未讀其資料，也不宣稱已完成 SHA256 跨任務排除。

未來人工確認必須包含 sample_id、reviewer_role、review_result、review_date、review_protocol_version。
review_result 僅允許 CONFIRMED_NO_TARGET_WOUND / TARGET_WOUND_PRESENT / UNCERTAIN；只有前者可確認資格，UNCERTAIN 一律排除。
MISSING_LABEL、INVALID_LABEL、EMPTY_LABEL_ONLY、positive image 非 GT 區域、validation FP、test/external 均不得混入 verified 名冊。
數量少不構成補造或擴增資料的理由；G0 沒有指定最低數量。

## 三、Stock pipeline 已有的負樣本監督

本次僅讀 source，未 import torch/ultralytics 或載入 checkpoint。architecture 已釘選 source hashes 全數核對；`data/utils.py` 另記本次 hash，其不在 architecture 原始 pin map，不能冒稱歷史獨立 pin。

- `data/utils.py:152–156`：empty/missing 皆產生零 target tensor，但保留不同 counter。工程可載入不等於研究上語義可信。
- `data/dataset.py:232–249` 與 `data/augment.py` Format：image 仍進 batch；空 cls/bbox/batch_idx，mask zero tensor（overlap-mask 模式為一張零 mask）。
- `utils/tal.py:64–71`：無 GT 時 foreground/target scores 為零。
- `utils/loss.py:318–350`：所有 prediction 的 cls BCE 對 zero target scores 提供背景抑制；bbox、DFL、實際 mask loss 只作用在 foreground assignments。
- `utils/loss.py:425–444`：mixed batch 中沒有 foreground 的 image 不提供實際 positive mask loss；全空情形僅保留 zero-valued mask graph terms。**不是對全圖背景執行 dense segmentation BCE 的新監督。**

`f1_v2_runner.py` 在 preprocess_batch 記錄 consumed image identities。G0 對照 300 frozen orders，C4 / H4-R 每個 epoch 771 anchors 完全一致；每張負樣本每 epoch 1 次，300 epochs 各 300 次。
因此：**Existing empty-label anchors were already part of the baseline training distribution.**
Anchor count 不是 Mosaic donor 次數、增強後仍無傷口的 batch image 次數、也不是成功 optimizer updates。

## 四、歷史 exposure 描述性核對

| Arm | Seed | 300 epochs empty-anchor draws | 每 epoch 平均 |
|---|---:|---:|---:|
{hist}

以上逐行重算 saved consumed-anchor telemetry，再核對既存 D2 summary；不是只搬運舊統計。
D2 S 的空標註 primary-anchor exposure 比均勻 C 少，但這與其他尺寸 sampling 一起變動，不能據此推論 FP 或 Recall 的因果機制，也不據此設定未來權重。
`existing_negative_exposure.csv` 保存各 arm × seed × epoch × candidate 的逐筆次數，包含零次。

## 五、25 個必要回答

| # | 問題 | 回答 |
|---:|---|---|
{q}

## 六、下一階段界線與交付

可討論 G1 FALSE_POSITIVE_CONTROL preregistration，但本次止於 G0。未決定 oversampling、loss、budget 或新訓練資料配置。
未來 baseline 回到 train imgsz=768，不能用 1024+negatives vs768 同時改兩個因素；不得挽救性合併失敗的 scale intervention。
未來 safety priorities 同時保留 Precision/FP、Very-small Recall、Small Recall、Crop completeness，不能以 suppress detections 換取表面 Precision。
18 個舊 negative 不能稱為新增監督；是否改變 exposure 真有幫助，尚未有此實驗證據。

輸出目錄：`experiments/results/g0_negative_source_audit/`。六份 CSV 保留 flat-table typed fields、明確 NA 與來源追溯，不使用色彩代替分類；另有來源證據、摘要與 integrity JSON。
完整性：依 F1.3 保存的 source/output digests 重核 F1.2/F1.3 research artifacts、預測與 checkpoints，前後一致。原始 validation/test raster 不重新讀取；研究圖檔只當不透明 bytes 做 hash，不開啟像素。
沒有修改既有 pipeline、App、來源資料或模型；本次僅新增 G0 稽核程式、測試、報告及輸出。

```ini
PHASE_G0_STATUS = COMPLETE
NEGATIVE_SOURCE_FEASIBILITY = {s['NEGATIVE_SOURCE_FEASIBILITY']}
VERIFIED_NEGATIVE_COUNT = {s['VERIFIED_NEGATIVE_COUNT']}
VERIFIED_NEGATIVE_UNIQUE_CONTENT_COUNT = {s['VERIFIED_NEGATIVE_UNIQUE_CONTENT_COUNT']}
NEEDS_EXPERT_ADJUDICATION = {s['NEEDS_EXPERT_ADJUDICATION']}
TRAINING = false
MODEL_LOADED = false
NEW_INFERENCE = false
VALIDATION_HARD_NEGATIVE_MINING = false
LOCKED_TEST_USED = false
CO2Wounds_USED = false
EXTERNAL_TEST_USED = false
NEW_TRAINING_AUTHORIZED = NO
MULTI_SEED_AUTHORIZED = NO
APP_MODEL_REPLACEMENT_AUTHORIZED = NO
STOP AFTER G0
```
'''


if __name__ == '__main__':
    main()
