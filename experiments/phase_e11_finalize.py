"""Seal the E1.1 BLOCKED audit without any training authorization."""
from datetime import datetime, timezone
import json
import subprocess
import sys

from experiments import phase_e11 as e
from experiments.paired_sampling_runner import leaves


def finalize():
    e.r.model_guard();e.future_absent()
    e.r.require(not e.artifact('freeze').exists(),'FAIL_V2_ALREADY_FROZEN')
    ca=e.read(e.artifact('canonicalization_audit'));ga=e.read(e.artifact('patch_geometry_audit'))
    e.r.require(ca['resolved_exact_count']==1 and ca['unresolved_count']==16,'FAIL_CANONICAL_COUNTS')
    keys=[(x['sample_id'],x['GT_instance_id']) for x in ca['records']]
    v1=e.read(e.P/'E_PATCH_V1_geometry_audit.json')
    e.r.require(len(keys)==len(set(keys))==17 and set(keys)=={(x['sample_id'],x['GT_instance_id']) for x in v1['original_invalid_polygons_in_eligible_images']},'FAIL_CANONICAL_COVERAGE')
    accepted=[x for x in ca['records'] if x['canonicalization_status']=='EXACT_RASTER_EQUIVALENT']
    e.r.require(len(accepted)==1 and accepted[0]['xor_pixels']==0 and accepted[0]['raster_IoU']==1
        and accepted[0]['source_raster_sha256']==accepted[0]['canonical_raster_sha256'],'FAIL_EXACT_CANDIDATE')
    cycles=ga['cycles'];cyclekeys=[(x['sample_id'],x['target_GT_id']) for x in cycles]
    e.r.require(len(cycles)==len(set(cyclekeys))==182 and sum(x['status']=='PASS' for x in cycles)==123,'FAIL_CYCLE_COUNTS')
    e.r.require(sum('FAIL_RASTER_PATCH_MISMATCH' in x['failures'] for x in cycles)==41
        and sum(x['records'] is None for x in cycles)==18,'FAIL_BLOCKER_COUNTS')
    c=e.read(e.artifact('control_config'));p=e.read(e.artifact('experimental_config'))
    left,right=leaves(c),leaves(p)
    changed=sorted(k for k in left.keys()|right.keys() if left.get(k)!=right.get(k))
    e.r.require(changed==['arm_label','experiment_id','output_path','representation'],'FAIL_PAIRED_CONFIG_PARITY')
    e.r.require(c['primary_comparison']=='FRESH_CANONICAL_C2_vs_FRESH_CANONICAL_P2'
        and c['historical_D2_C_role']=='HISTORICAL_REFERENCE_ONLY','FAIL_COMPARATOR')
    order=e.read(e.OUT/'anchor_orders.json')
    ids=[x['sample_id'] for x in e.read(e.P/'E_PATCH_V1_training_manifest.json')['samples']]
    for epoch,plan in enumerate(order['epochs']):
        e.r.require(plan==e.uniform_order(ids,epoch) and e.r.digest(plan)==order['epoch_sha256'][epoch],'FAIL_ANCHOR_PLAN')
    e.r.require(len(order['epochs'])==300,'FAIL_EPOCH_COUNT')
    e.r.require(e.sha(e.artifact('evaluation_protocol'))==e.sha(e.P/'E_PATCH_V1_evaluation_protocol.json'),'FAIL_EVALUATION_CHANGED')
    tests=['tests/test_phase_e11.py','tests/test_phase_e1.py','tests/test_phase_e0.py','tests/test_phase_d2.py',
           'tests/test_phase_d1.py','tests/test_phase_d01.py','tests/test_phase_d0.py']
    result=subprocess.run([sys.executable,'-m','pytest',*tests,'-q'],cwd=e.ROOT,capture_output=True,text=True,encoding='utf8')
    e.r.require(result.returncode==0 and '210 passed' in result.stdout,'FAIL_TESTS: '+result.stdout+result.stderr)
    verification={'passed':210,'new_E11_tests':31,'command':[sys.executable,'-m','pytest',*tests,'-q'],
        'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'paired_config_differences':changed,
        'uniform_order_epochs_verified':300,'source_exact_candidate_count':1,'cycles_passed':123,
        'minimal_adapter_runtime_probe':'PASS','actual_torch_DataLoader_or_full_Mosaic':'NOT_EXECUTED'}
    snapshot=e.read(e.OUT/'source_snapshot.json')
    e.r.require(all(e.sha(path)==h for path,h in snapshot.items()),'FAIL_SOURCE_CHANGED')
    e.r.require(e.REPORT.exists() and 'PHASE_E11_STATUS = BLOCKED' in e.REPORT.read_text(encoding='utf8'),'FAIL_REPORT')
    e.write(e.OUT/'verification.json',verification)
    files=[p for p in e.P.glob('E_PATCH_V2*') if p.is_file()]+list((e.P/'E_PATCH_V2_canonical_labels').glob('*.json'))
    files+=list(e.OUT.glob('*'))+[e.REPORT]
    files+=[e.ROOT/p for p in ['experiments/patch_raster.py','experiments/patch_dataset_adapter.py','experiments/phase_e11.py','experiments/phase_e11_finalize.py','tests/test_phase_e11.py']]
    hashes={p.relative_to(e.ROOT).as_posix():e.sha(p) for p in files if p.is_file()}
    e.future_absent()
    f={'status':'FROZEN_BEFORE_TRAINING','freeze_scope':'BLOCKED_V2_CONTRACT_AND_FAILURE_EVIDENCE_NOT_AUTHORIZATION',
       'created_at':datetime.now(timezone.utc).isoformat(),'PHASE_E11_STATUS':'BLOCKED',
       'READY_FOR_PHASE_E2_PAIRED_PATCH_SEED42_TRAINING':'NO',
       'blockers':{'unresolved_source_polygons':16,'source_blocked_cycles':18,'raster_mismatch_cycles':41},
       'artifacts_sha256':hashes,'source_snapshot_sha256':e.sha(e.OUT/'source_snapshot.json'),
       'protected_files_unchanged':len(snapshot),'verification':verification,'future_outputs_absent':True,
       'TRAINING':False,'MODEL_LOADING':False,'INFERENCE':False,'test_images_used':0,'CO2Wounds_used':False,
       'MULTI_SEED_PATCH_AUTHORIZED':'NO','NEW_TRAINING_AUTHORIZED':'NO','APP_MODEL_REPLACEMENT_AUTHORIZED':'NO',
       'EXTERNAL_TEST_AUTHORIZED':'NO','STOP_AFTER_E11':True}
    e.write(e.artifact('freeze'),f)
    e.r.require(all(e.sha(e.ROOT/path)==h for path,h in hashes.items()),'FAIL_FINAL_HASH')
    print(json.dumps({k:f[k] for k in ['PHASE_E11_STATUS','READY_FOR_PHASE_E2_PAIRED_PATCH_SEED42_TRAINING','blockers','protected_files_unchanged','future_outputs_absent']},indent=2))


if __name__=='__main__':finalize()
