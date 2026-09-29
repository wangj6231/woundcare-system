"""Seal E1 blocked evidence; never grant training authorization."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from experiments import phase_e1 as e


def finalize():
    e.g.model_guard();e.require_future_absent(e.ROOT)
    e.g.require(not e.output('freeze').exists(),'FAIL_FREEZE_EXISTS')
    audit=e.read(e.output('geometry_audit'));protocol=e.read(e.output('protocol'))
    e.g.require(audit['PHASE_E1_STATUS']=='BLOCKED' and protocol['PHASE_E1_STATUS']=='BLOCKED','FAIL_EXPECTED_BLOCKED')
    e.g.require(protocol['READY_FOR_PHASE_E2_PATCH_SEED42_TRAINING']=='NO','FAIL_AUTHORIZATION')
    manifest=e.read(e.output('training_manifest'))['samples']
    e.g.require(len(manifest)==771 and sum(bool(r['eligible_GT_ids']) for r in manifest)==161
        and sum(len(r['eligible_GT_ids']) for r in manifest)==182,'FAIL_MANIFEST_TOTALS')
    patches=audit['patches'];passed=[p for p in patches if p['status']=='PASS']
    e.g.require(len(patches)==182 and len(passed)==163 and len(audit['original_invalid_polygons_in_eligible_images'])==17,'FAIL_AUDIT_TOTALS')
    keys=[(p['sample_id'],p['target_GT_id']) for p in patches]
    e.g.require(len(keys)==len(set(keys))==182,'FAIL_CYCLE_COVERAGE')
    for p in passed:
        e.g.require(p['selected_target_retention']==1 and p['retained_GT_count']>0,'FAIL_RETENTION')
        e.g.require(p['original_GT_count']==sum(p[k] for k in ['fully_retained_GT_count','partially_clipped_GT_count','dropped_GT_count']),'FAIL_GT_ACCOUNTING')
        e.g.require(p['transformed_label_hash']==e.g.digest(p['records']),'FAIL_LABEL_HASH')
    e.g.require(e.sha(e.output('evaluation_protocol'))==e.sha(e.ROOT/e.PROTOCOLS/'D_SEG_SMALL_SAMPLING_V2_evaluation_protocol.json'),'FAIL_EVAL_CHANGED')
    from PIL import Image
    for rel in audit['synthetic_visualizations']:
        with Image.open(e.ROOT/rel) as im:im.verify()
    tests=['tests/test_phase_e1.py','tests/test_phase_e0.py','tests/test_phase_d2.py',
           'tests/test_phase_d1.py','tests/test_phase_d01.py','tests/test_phase_d0.py']
    p=subprocess.run([sys.executable,'-m','pytest',*tests,'-q'],cwd=e.ROOT,capture_output=True,text=True,encoding='utf8')
    e.g.require(p.returncode==0 and '179 passed' in p.stdout,'FAIL_TESTS: '+p.stdout+p.stderr)
    verification={'command':[sys.executable,'-m','pytest',*tests,'-q'],'exit_code':p.returncode,
        'stdout':p.stdout,'stderr':p.stderr,'passed':179,'new_E1_tests':28,'synthetic_figures_reviewed':6,
        'no_real_image_visual_review':True,'model_loaded':False,'training':False,'inference':False}
    snapshot=e.read(e.ROOT/e.AUDIT/'source_snapshot_before.json')
    e.g.require(all(e.sha(p)==h for p,h in snapshot.items()),'FAIL_PROTECTED_SOURCE_CHANGED')
    report=e.ROOT/e.REPORT
    e.g.require(report.exists() and 'BLOCKED' in report.read_text(encoding='utf8'),'FAIL_REPORT')
    e.write(e.ROOT/e.AUDIT/'verification.json',verification)
    files=[e.output(s) for s in ['protocol','training_transform','training_manifest','geometry_audit','evaluation_protocol']]
    files+=list((e.ROOT/e.AUDIT).glob('*'))
    files +=[report,e.ROOT/'experiments/patch_geometry.py',e.ROOT/'experiments/phase_e1.py',e.ROOT/'experiments/phase_e1_finalize.py',e.ROOT/'tests/test_phase_e1.py']
    hashes={p.relative_to(e.ROOT).as_posix():e.sha(p) for p in files if p.is_file()}
    e.require_future_absent(e.ROOT)
    freeze={'status':'FROZEN_BEFORE_TRAINING','freeze_scope':'BLOCKED_V1_PREREGISTRATION_AND_FAILURE_EVIDENCE_NOT_EXECUTION_AUTHORIZATION',
        'created_at':datetime.now(timezone.utc).isoformat(),'PHASE_E1_STATUS':'BLOCKED',
        'READY_FOR_PHASE_E2_PATCH_SEED42_TRAINING':'NO','PRIMARY_BLOCKER':'FAIL_PATCH_LABEL_INVALID',
        'additional_unverified_requirements':['fractional_raster_alignment','runtime_mosaic_and_worker_epoch_adapter'],
        'artifacts_sha256':hashes,'source_snapshot_sha256':e.sha(e.ROOT/e.AUDIT/'source_snapshot_before.json'),
        'protected_files_unchanged':len(snapshot),'verification':verification,'future_output_absent':True,
        'NEW_TRAINING_AUTHORIZED':'NO','APP_MODEL_REPLACEMENT_AUTHORIZED':'NO','EXTERNAL_TEST_AUTHORIZED':'NO',
        'MODEL_LOADED':False,'TRAINING_PERFORMED':False,'MODEL_INFERENCE':False,'LOCKED_TEST_USED':False,
        'CO2Wounds_used':False,'test_images_used':0,'validation_pixels_used':0,'TRAINING_PIXELS_USED_FOR_PATCH_AUDIT':False,'STOP_AFTER_E1':True}
    e.write(e.output('freeze'),freeze)
    e.g.require(all(e.sha(e.ROOT/p)==h for p,h in hashes.items()),'FAIL_FINAL_ARTIFACT_HASH')
    print(json.dumps({k:freeze[k] for k in ['PHASE_E1_STATUS','READY_FOR_PHASE_E2_PATCH_SEED42_TRAINING','PRIMARY_BLOCKER','protected_files_unchanged','future_output_absent']},indent=2))


if __name__=='__main__':finalize()
