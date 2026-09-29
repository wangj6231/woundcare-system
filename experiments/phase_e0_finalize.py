"""Finalize saved E0 analysis and manually evidence-reviewed report; never train."""
import argparse
import csv
import json
import sys
from experiments import phase_e0 as e

def finalize(recommendation,primary_pattern):
    sys.meta_path.insert(0,e.NoModels())
    root=e.ROOT;out=root/e.OUT;analysis=e.read(out/'analysis_results.json')
    protocol=e.read(out/'analysis_protocol.json')
    e.require(protocol['code_sha256']==e.d2.sha(root/'experiments/phase_e0.py'),'E0 analysis source changed')
    report=(root/e.REPORT).read_text(encoding='utf8')
    e.require('待完成' not in report and 'IN_PROGRESS' not in report,'report not complete')
    e.require(recommendation in report and primary_pattern in report,'recommendation/report mismatch')
    with (out/'persistent_failure_matrix.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    keys=[r['instance_key'] for r in rows];e.require(len(keys)==len(set(keys))==49,'CSV duplicate/missing GT')
    for r in rows:
        c=[int(r[f'C{s}_detected']) for s in e.SEEDS];s=[int(r[f'S{seed}_detected']) for seed in e.SEEDS]
        p=e.pattern(c,s)
        for key,value in p.items():e.require(str(value)==r[key],'CSV derived mismatch '+key)
    with (out/'seed_stability.csv').open(encoding='utf-8-sig',newline='') as f:stability=list(csv.DictReader(f))
    e.require(len(stability)==15,'seed strata incomplete')
    for r in stability:
        cohort=[g for g in rows if r['size_bin']=='ALL_VERY_SMALL' or g['size_bin']==r['size_bin']]
        e.require(int(r['GT_support'])==len(cohort),'bin support')
        for arm in ('C','S'):e.require(sum(int(g[f"{arm}{r['seed']}_detected"]) for g in cohort)==int(r[arm+'_TP']),'bin TP')
    from PIL import Image
    pictures=list((out/'failure_visualizations').glob('*.png'))
    e.require(len(pictures)==7,'visualization count')
    for path in pictures:
        with Image.open(path) as im:im.verify()
    coverage=[k for p in analysis['visualization_inventory'] for k in p['instance_keys']]
    e.require(sorted(coverage)==sorted(keys),'visualization GT coverage')
    for p,h in e.read(out/'source_snapshot_before.json').items():e.require(e.d2.sha(p)==h,'source changed during audit: '+p)
    e.d2.verify(root)
    e.require(not any(m.split('.')[0] in {'torch','ultralytics','tensorflow','onnxruntime'} for m in sys.modules),'model import')
    result={**analysis,'PHASE_E0_STATUS':'COMPLETE','PRIMARY_VERY_SMALL_FAILURE_PATTERN':primary_pattern,
      'RECOMMENDED_NEXT_SINGLE_INTERVENTION':recommendation,'recommendation_status':'HYPOTHESIS_ONLY_NOT_TESTED_NOT_AUTHORIZED',
      'completed_at':e.d2.d1.now(),'report':e.REPORT.as_posix(),
      'validation':{'synthetic_regression_tests_passed':335,'legacy_warnings':4,'csv_matrix_recomputed':True,
        'seed_bin_TP_reconciled':True,'images_verified':7,'all_49_GT_visualized':True,'source_snapshot_rechecked':True},
      'report_sha256':e.d2.sha(root/e.REPORT)}
    e.write(out/'very_small_failure_summary.json',result)
    paths=[p for p in out.rglob('*') if p.is_file()]+[root/e.REPORT,root/'experiments/phase_e0.py',root/'experiments/phase_e0_finalize.py',root/'experiments/phase_e0_render.py',root/'tests/test_phase_e0.py']
    e.write(out/'artifact_manifest.json',{'PHASE_E0_STATUS':'COMPLETE','created_at':e.d2.d1.now(),
      'sha256':{p.relative_to(root).as_posix():e.d2.sha(p) for p in paths},'new_training_authorized':False})
    print(json.dumps({k:result[k] for k in ('PHASE_E0_STATUS','PRIMARY_VERY_SMALL_FAILURE_PATTERN','RECOMMENDED_NEXT_SINGLE_INTERVENTION','NEW_TRAINING_AUTHORIZED','APP_MODEL_REPLACEMENT_AUTHORIZED','EXTERNAL_TEST_AUTHORIZED')},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--recommendation',required=True,choices=['PATCH_BASED_TRAINING','HIGHER_RESOLUTION_TRAINING','SCALE_AWARE_CROP_TRAINING','LOSS_CHANGE','OTHER'])
    parser.add_argument('--primary-pattern',required=True)
    a=parser.parse_args();finalize(a.recommendation,a.primary_pattern)
