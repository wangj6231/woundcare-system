"""F0.1 numerical decisions and behavioral boundaries, no research inference."""
import unittest


class F01Tests(unittest.TestCase):
    def protocol(self):
        from experiments.phase_f01 import PRO,PREFIX,read
        return read(PRO/f'{PREFIX}_protocol.json')
    def arms(self):
        from experiments.phase_f01 import OUT,read
        return [read(OUT/x/'summary.json') for x in ('N768','N1024')]
    def validator(self):
        from experiments.phase_f01 import OUT,read
        return read(OUT/'validator_dry_audit.json')['evidence']

    def test_f01_preserves_f0_blocked(self):
        from experiments.phase_f01 import PRO,read,OUT,sha
        self.assertEqual(read(PRO/'F_HIGHER_SCALE_V1_freeze.json')['PHASE_F0_STATUS'],'BLOCKED')
        self.assertTrue(all(sha(p)==h for p,h in read(OUT/'protected_f0_snapshot.json').items()))
    def test_f01_same_synthetic_fixture(self):
        from experiments.phase_f01 import fixture_hashes
        a,b=self.arms();self.assertEqual(a['fixture'],b['fixture']);self.assertEqual(a['fixture'],fixture_hashes())
    def test_f01_same_initialization(self):
        a,b=self.arms();self.assertEqual(a['initialization_sha256'],b['initialization_sha256'])
        self.assertEqual(a['initial_trainable_parameter_sha256'],b['initial_trainable_parameter_sha256'])
    def test_f01_fresh_model_per_arm(self):
        self.assertTrue(all(a['fresh_model'] and a['fresh_optimizer'] and a['fresh_GradScaler'] for a in self.arms()))
    def test_f01_separate_processes(self):
        from experiments.phase_f01 import OUT,read
        a,b=self.arms();self.assertNotEqual(a['pid'],b['pid'])
        self.assertLessEqual(read(OUT/'N768_process.json')['exited_at'],read(OUT/'N1024_process.json')['started_at'])
    def test_f01_control_768(self):self.assertEqual(self.arms()[0]['imgsz'],768)
    def test_f01_experimental_1024(self):self.assertEqual(self.arms()[1]['imgsz'],1024)
    def test_f01_batch4_both(self):self.assertEqual([a['batch'] for a in self.arms()],[4,4])
    def test_f01_initial_scale_same(self):self.assertEqual([a['attempts'][0]['scaler_before'] for a in self.arms()],[65536,65536])
    def test_f01_no_manual_scale_override(self):
        for a in self.arms():
            for prev,current in zip(a['attempts'],a['attempts'][1:]):self.assertEqual(prev['scaler_after'],current['scaler_before'])
        self.assertIn('no manual override',self.protocol()['initial_scale_policy'])
    def test_f01_max_attempts_8(self):
        self.assertEqual(self.protocol()['MAX_ATTEMPTS'],8)
        self.assertEqual([a['completed_attempts'] for a in self.arms()],[8,8])
    def test_f01_no_optimizer_step(self):self.assertEqual([a['guards']['optimizer_step_attempts'] for a in self.arms()],[0,0])
    def test_f01_no_scaler_step(self):self.assertEqual([a['guards']['scaler_step_attempts'] for a in self.arms()],[0,0])
    def test_f01_parameters_unchanged(self):
        for a in self.arms():
            self.assertTrue(all(r['parameters_unchanged'] for r in a['attempts']))
            self.assertEqual({r['trainable_parameter_sha256'] for r in a['attempts']},{a['initial_trainable_parameter_sha256']})
    def test_f01_no_checkpoint_written(self):
        from experiments.phase_f01 import OUT
        self.assertEqual(list(OUT.rglob('*.pt')),[])
        self.assertTrue(all(a['guards']['save_attempts']==a['guards']['checkpoint_write_attempts']==0 for a in self.arms()))
    def test_f01_no_real_image_decode(self):
        self.assertTrue(all(a['guards']['image_access_attempts']==0 and not a['guards']['image_open_decode_paths'] for a in self.arms()))
    def test_f01_control_train768_val768(self):
        a=self.validator()['arms'][0];self.assertEqual((a['train_imgsz'],a['val_imgsz'],a['validator_imgsz']),(768,768,768))
    def test_f01_experimental_train1024_val768(self):
        a=self.validator()['arms'][1];self.assertEqual((a['train_imgsz'],a['val_imgsz'],a['validator_imgsz']),(1024,768,768))
    def test_f01_shared_validator_override(self):
        self.assertEqual([a['shared_override_class'] for a in self.validator()['arms']],['SharedValidation768']*2)
    def test_f01_validator_thresholds_same(self):
        self.assertEqual([a['conf'] for a in self.validator()['arms']],[.001,.001])
    def test_f01_validator_NMS_same(self):
        self.assertEqual([a['NMS_iou'] for a in self.validator()['arms']],[.7,.7])
    def test_f01_validator_matching_contract_same(self):
        a,b=self.validator()['arms'];self.assertEqual(a['matching_IoU_vector'],b['matching_IoU_vector'])
    def test_f01_no_validation_inference(self):self.assertFalse(self.validator()['validation_inference'])
    def test_f01_locked_test_rejected(self):
        from experiments.f01_guards import Guards,ForbiddenImageAccess
        from pathlib import Path
        g=Guards(Path.cwd(),Path.cwd()/'init.pt',Path.cwd()/'out')
        with self.assertRaises(ForbiddenImageAccess):g.audit('open',(str(Path.cwd()/'data/test/a.png'),'r',0))
    def test_f01_co2_rejected(self):
        from experiments.f01_guards import Guards,ForbiddenImageAccess
        from pathlib import Path
        g=Guards(Path.cwd(),Path.cwd()/'init.pt',Path.cwd()/'out')
        with self.assertRaises(ForbiddenImageAccess):g.image('CO2Wounds/test/a.png')
    def test_f01_future_training_dirs_absent(self):
        from experiments.phase_f0 import future_absent
        self.assertTrue(future_absent())
    def test_f01_shared_validator_runtime_dry_audit(self):
        import subprocess,sys,json
        p=subprocess.run([sys.executable,'-m','experiments.f01_validator','--dry-audit'],capture_output=True,text=True,encoding='utf8')
        self.assertEqual(p.returncode,0,p.stderr)
        r=json.loads(p.stdout.strip().splitlines()[-1])
        self.assertTrue(r['BEST_CHECKPOINT_SELECTION_PARITY'])
        self.assertEqual([(a['train_imgsz'],a['val_imgsz'],a['validator_imgsz']) for a in r['arms']],[(768,768,768),(1024,768,768)])
        self.assertEqual(r['real_validation_images_loaded'],0);self.assertFalse(r['validation_inference'])

    def test_f01_gradient_nan_inf_separated(self):
        import numpy as np
        from experiments.f01_numerical import gradient_stats
        s=gradient_stats([('a',np.array([1.,-3.])),('b',np.array([np.nan,2.])),('c',np.array([np.inf,4.]))])
        self.assertEqual([s[k] for k in ['gradient_tensor_count','finite_gradient_tensor_count','nonfinite_gradient_tensor_count','nan_gradient_tensor_count','inf_gradient_tensor_count']],[3,1,2,1,1])
        self.assertEqual(s['nonfinite_parameter_names'],['b','c']);self.assertEqual(s['max_abs_finite_gradient'],4.)

    def test_f01_save_step_and_decode_guards_behavior(self):
        import subprocess,sys,json
        p=subprocess.run([sys.executable,'-m','experiments.f01_guards','--self-test'],capture_output=True,text=True,encoding='utf8')
        self.assertEqual(p.returncode,0,p.stderr)
        r=json.loads(p.stdout.strip().splitlines()[-1])
        self.assertTrue(r['save_raised']);self.assertTrue(r['write_raised']);self.assertFalse(r['file_exists'])
        self.assertTrue(r['optimizer_raised']);self.assertTrue(r['scaler_raised']);self.assertTrue(r['decode_raised'])
        self.assertFalse(r['CUDA_initialized'])

    def test_f01_three_consecutive_finite_rule(self):
        from experiments.phase_f01 import classify
        def row(i,finite,scale):
            return dict(attempt=i,loss_finite=True,nonfinite_gradient_tensor_count=0 if finite else 2,
                        gradient_tensor_count=10,scaler_before=scale,scaler_after=scale if finite else scale/2,
                        parameters_unchanged=True)
        rows=[row(1,False,65536)]+[row(i,True,32768) for i in range(2,9)]
        r=classify(rows)
        self.assertEqual(r['classification'],'SCALER_MANAGED_TRANSIENT_OVERFLOW')
        self.assertEqual(r['FIRST_FINITE_GRADIENT_ATTEMPT'],2)
        self.assertEqual(r['FIRST_FINITE_GRADIENT_SCALE'],32768)
        self.assertEqual(r['stable_window'],[2,3,4])
        self.assertEqual(classify([row(i,i%2==0,65536/2**i) for i in range(1,9)])['classification'],'PERSISTENT_NUMERICAL_INSTABILITY')


if __name__=='__main__':unittest.main()
