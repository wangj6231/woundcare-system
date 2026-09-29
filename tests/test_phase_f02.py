"""CPU/stdlib-only V2 contract tests; no models, framework imports or inference."""
import pytest
import json
import subprocess
import sys
from experiments.f02_safety import NumericalSafety, compare_completed
from experiments import phase_f02 as f02


def event(i=0, applied=True, **changes):
    row = dict(epoch=i // 193, batch=i % 193, global_batch=i,
               scale_before=65536.0, scale_after=65536.0, loss_finite=True,
               optimizer_opportunity_attempted=True, optimizer_update_applied=applied,
               optimizer_update_skipped=not applied, gradient_nonfinite=None,
               gradient_observation="not_observed", parameter_finite=True)
    row.update(changes)
    return row


def test_f02_16_consecutive_skips_hard_stop():
    safety = NumericalSafety()
    for i in range(15):
        assert safety.observe(event(i, False)) is None
    assert safety.observe(event(15, False)) == "FAIL_PERSISTENT_AMP_UPDATE_SKIPS"
    assert safety.summary()["consecutive_skipped"] == 16


def test_f02_nonfinite_loss_hard_stop():
    safety = NumericalSafety()
    assert safety.check_loss(False) == "FAIL_NONFINITE_LOSS"
    assert safety.summary()["scheduled"] == 0
    with pytest.raises(RuntimeError, match="PAIR_ALREADY_STOPPED"):
        safety.observe(event())


def test_f02_nonfinite_parameter_hard_stop():
    assert NumericalSafety().observe(event(parameter_finite=False)) == "FAIL_NONFINITE_PARAMETERS"


def test_f02_scale_below_one_hard_stop():
    assert NumericalSafety().observe(event(scale_after=0.5)) == "FAIL_AMP_SCALE_COLLAPSE"
    assert NumericalSafety().observe(event(scale_after=1.0)) is None


def test_f02_total_skips_not_auto_fail():
    safety = NumericalSafety()
    for i in range(60):
        assert safety.observe(event(i, applied=i % 2 == 1)) is None
    assert safety.summary()["skipped"] == 30
    assert safety.summary()["consecutive_skipped"] == 0


def completed(skips=0, **changes):
    result = dict(completed_epochs=300, scheduled=3741, applied=3741-skips,
                  skipped=skips, unknown=0, failure=None, telemetry_complete=True)
    result.update(changes)
    return result


def test_f02_update_imbalance_reported():
    result = compare_completed(completed(10), completed(14))
    assert result["PAIRED_FIXED_BUDGET_VALID"] == "YES"
    assert result["AMP_UPDATE_COUNT_IMBALANCE_OBSERVED"] == "YES"
    assert result["identical_realized_update_counts"] is False
    assert result["retry_or_compensation_allowed"] is False


@pytest.mark.parametrize("changes", [dict(completed_epochs=299), dict(unknown=1),
    dict(scheduled=3740), dict(applied=3740), dict(telemetry_complete=False),
    dict(failure="FAIL_NONFINITE_LOSS")])
def test_f02_incomplete_budget_is_invalid(changes):
    assert compare_completed(completed(), completed(**changes))["PAIRED_FIXED_BUDGET_VALID"] == "NO"


@pytest.mark.parametrize("changes", [dict(optimizer_update_applied=None),
    dict(optimizer_update_skipped=True), dict(parameter_finite=None),
    dict(scale_after=float("nan")), dict(loss_finite=None), dict(global_batch=-1)])
def test_f02_unknown_telemetry_fails_closed(changes):
    safety = NumericalSafety()
    assert safety.observe(event(**changes)) == "FAIL_TELEMETRY_INCOMPLETE"
    assert safety.summary()["unknown"] == 1


def test_f02_skip_streak_crosses_epoch_boundary():
    safety = NumericalSafety()
    for i in range(185, 200):
        assert safety.observe(event(i, False)) is None
    assert safety.observe(event(200, False)) == "FAIL_PERSISTENT_AMP_UPDATE_SKIPS"


def test_f02_runtime_exception_stops_pair():
    safety = NumericalSafety()
    assert safety.runtime_exception("OOM", opportunity_attempted=True) == "FAIL_RESOURCE_RUNTIME"
    assert safety.summary()["unknown"] == 1
    assert safety.summary()["runtime_classification"] == "TRAINING_NUMERICAL_RUNTIME_INVALID"
    assert NumericalSafety().runtime_exception("backward") == "TRAINING_NUMERICAL_RUNTIME_INVALID"


def test_f02_only_training_imgsz_differs():
    c, h = f02.make_pair()
    assert f02.validate_pair(c, h) == ["arm", "experiment_id", "output_path", "training_args.imgsz"]
    h["training_args"]["lr0"] = .001
    with pytest.raises(ValueError):
        f02.validate_pair(c, h)


@pytest.mark.parametrize("field,value", [("batch", 2), ("resume", True), ("amp", False), ("epochs", 299)])
def test_f02_shared_recipe_mutation_rejected(field, value):
    c, h = f02.make_pair()
    c["training_args"][field] = h["training_args"][field] = value
    with pytest.raises(ValueError):
        f02.validate_pair(c, h)


@pytest.mark.parametrize("source,split,role,path", [
    ("FUSeg", "test", "test", "outputs/test.png"),
    ("CO2Wounds", "train", "wound_finetuning", "outputs/co2.png"),
    ("FUSeg", "train", "wound_finetuning", "outputs/../locked/test.png"),
    ("FUSeg", "train", "wound_finetuning", "outputs/e_patch_v3/canonical.png")])
def test_f02_locked_test_rejected(source, split, role, path):
    row = dict(source=source, split=split, role=role, image_path=path, label_path=path)
    with pytest.raises(ValueError):
        f02.admit_training_row(row)


def test_f02_future_outputs_absent(tmp_path):
    assert f02.future_absent(tmp_path)
    (tmp_path / f02.FUTURE[0]).mkdir(parents=True)
    with pytest.raises(ValueError):
        f02.future_absent(tmp_path)


def test_f02_protocol_separates_readiness_from_authority():
    contracts = f02.contracts()
    p = contracts["protocol"]
    assert p["schema"] == "F_HIGHER_INPUT_SCALE_V2"
    assert p["F1_training_authorized"] is False
    assert p["STOP_AFTER_F02"] is True
    assert p["SYNTHETIC_NUMERICAL_GATE_DISCRIMINATIVE_VALIDITY"] == "NOT_ESTABLISHED"
    assert contracts["numerical_safety_contract"]["consecutive_skip_limit"] == 16
    assert contracts["numerical_safety_contract"]["scale_minimum_inclusive"] == 1.0
    assert p["advancement_gate"] == f02.read(f02.PRO / "F_HIGHER_SCALE_V1_protocol.json")["advancement_gate"]


@pytest.fixture(scope="module")
def audit():
    return f02.read(f02.OUT / "audit.json")


def test_f02_preserves_f0_blocked(audit):
    assert audit["historical"]["states"]["F_HIGHER_SCALE_V1"]["PHASE_F0_STATUS"] == "BLOCKED"
    assert f02.read(f02.PRO / "F_HIGHER_SCALE_V1_freeze.json")["PHASE_F0_STATUS"] == "BLOCKED"


def test_f02_preserves_f01_blocked(audit):
    assert audit["historical"]["states"]["F_HIGHER_SCALE_V1_NUMERICAL_PREFLIGHT"]["PHASE_F01_STATUS"] == "BLOCKED"
    assert f02.read(f02.PRO / "F_HIGHER_SCALE_V1_NUMERICAL_PREFLIGHT_freeze.json")["NUMERICAL_FEASIBILITY_1024"] == "FAIL"


def test_f02_no_extra_synthetic_attempts(audit):
    for arm, expected in audit["historical"]["synthetic_attempt_files"].items():
        files = sorted(p.name for p in (f02.ROOT / "experiments/results/f_higher_scale_v1_numerical_preflight" / arm).glob("attempt_*.json"))
        assert files == expected == [f"attempt_{i:02d}.json" for i in range(1, 9)]


def test_f02_no_gpu_execution():
    # Exercise real import/process boundaries in a disposable CPU-only interpreter.
    script = '''
import sys, subprocess, json
from experiments.phase_f02 import install_execution_guard
g = install_execution_guard()
blocked = []
for name in ('torch', 'ultralytics', 'cv2', 'PIL', 'numpy'):
    try: __import__(name)
    except RuntimeError: blocked.append(name)
try: subprocess.run([sys.executable, '-c', 'pass'])
except RuntimeError: blocked.append('subprocess')
print(json.dumps(dict(blocked=blocked, imported=[n for n in ('torch','ultralytics','cv2','PIL','numpy') if n in sys.modules])))
'''
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=f02.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual["blocked"] == ["torch", "ultralytics", "cv2", "PIL", "numpy", "subprocess"]
    assert actual["imported"] == []
    guard = f02.read(f02.OUT / "execution_guard.json")
    assert guard["framework_imports"] == []
    assert guard["subprocess_launches"] == guard["network_calls"] == 0
    assert not any(guard[k] for k in ("GPU_EXECUTION", "MODEL_LOADING", "MODEL_FORWARD", "TRAINING", "RESEARCH_INFERENCE"))


def test_f02_same_dataset(audit):
    assert audit["data"]["train_images"] == 771
    assert audit["data"]["original_polygon_GT"] == 965
    assert audit["data"]["development_validation_images"] == 191
    assert audit["data"]["all_image_label_hashes_match"]
    assert audit["data"]["exact_train_val_hash_overlap"] == 0


def test_f02_original_labels(audit):
    assert audit["data"]["training_representation"] == "ORIGINAL_STOCK_YOLO_POLYGON"
    assert audit["data"]["label_repairs"] == 0


def test_f02_no_patch_assets(audit):
    assert not audit["data"]["patch_labels_used"]
    assert not audit["data"]["mask_first_labels_used"]
    for cfg in f02.make_pair():
        assert cfg["dataset_adapter"] == "STOCK_YOLODataset"


def test_f02_same_initialization(audit):
    assert audit["data"]["initialization_sha256"] == "1ec926026473beafafb1ef8da6aa6efcc13d38c391ec8db48006c5d793d5f2a3"
    assert f02.make_pair()[0]["initialization"] == f02.make_pair()[1]["initialization"]
    assert audit["data"]["model_deserialized"] is False


def test_f02_same_batch4():
    assert [c["training_args"]["batch"] for c in f02.make_pair()] == [4, 4]


def test_f02_same_optimizer():
    c, h = f02.make_pair()
    assert c["runtime_optimizer"] == h["runtime_optimizer"]
    assert c["runtime_optimizer"]["name"] == "AdamW"
    assert c["runtime_optimizer"]["betas"] == [.937, .999]
    assert c["runtime_optimizer"]["gradient_clip_norm"] == 10


def test_f02_same_loss():
    c, h = f02.make_pair()
    assert c["loss"] == h["loss"] == f02.read(f02.PRO / "F_HIGHER_SCALE_V1_control_config.json")["loss"]


def test_f02_same_augmentation():
    c, h = f02.make_pair()
    assert c["augmentation"] == h["augmentation"] == f02.read(f02.PRO / "F_HIGHER_SCALE_V1_control_config.json")["augmentation"]


def test_f02_control_train768():
    assert f02.make_pair()[0]["training_args"]["imgsz"] == 768


def test_f02_experimental_train1024():
    assert f02.make_pair()[1]["training_args"]["imgsz"] == 1024


def test_f02_both_checkpoint_val768():
    for cfg in f02.make_pair():
        assert cfg["checkpoint_selection"]["both_arms_validation_imgsz"] == 768
        assert cfg["checkpoint_selection"]["shared_override"] == "experiments.f01_validator.SharedValidation768"


def test_f02_both_final_eval768():
    for cfg in f02.make_pair():
        assert cfg["evaluation"]["imgsz"] == 768
    assert f02.sha(f02.PRO / "F_HIGHER_SCALE_V2_evaluation_protocol.json") == "32a745eb328a7ddfebbf0430a11d5e14d47172a427abb4712b15d11c9dd8323d"


def test_f02_amp_same_initial_policy():
    c, h = f02.make_pair()
    assert c["amp_policy"] == h["amp_policy"]
    assert c["amp_policy"]["fresh_scaler"] and c["amp_policy"]["enabled"]


def test_f02_no_manual_scaler_override():
    for cfg in f02.make_pair():
        assert cfg["amp_policy"]["manual_initial_scale"] is None
        assert "init_scale" not in cfg["training_args"]


def test_f02_no_resume():
    for cfg in f02.make_pair():
        assert cfg["training_args"]["resume"] is False
        assert cfg["initialization"]["resume"] is False


def test_f02_no_batch_reduction():
    for cfg in f02.make_pair():
        assert cfg["budget_contract"]["gradient_accumulation_compensation"] is False
        assert "no retry, batch/scale reduction" in cfg["budget_contract"]["OOM"]


def test_f02_co2_rejected(audit):
    assert audit["data"]["CO2Wounds_used"] is False
    with pytest.raises(ValueError):
        f02.admit_training_row(dict(source="CO2Wounds", split="train", role="wound_finetuning"))


def test_f02_frozen_orders_uniform(audit):
    assert audit["data"]["anchor_epochs"] == 300
    assert audit["data"]["anchors_per_epoch"] == 771
    c, h = f02.make_pair()
    assert c["sampling"] == h["sampling"]
    assert c["sampling"]["mode"] == "uniform"
    assert c["sampling"]["replacement"] is False


def test_f02_all_prepared_artifacts_unchanged():
    seal = f02.read(f02.OUT / "prepared_seal.json")
    for relative, digest in seal["artifacts_sha256"].items():
        assert f02.sha(f02.ROOT / relative) == digest, relative


def test_f02_all_historical_and_data_hashes_unchanged(audit):
    for path, digest in audit["protected_sha256"].items():
        assert f02.sha(path) == digest, path


def test_f02_real_future_outputs_absent():
    assert f02.future_absent()


def test_f02_source_audit_is_not_runtime_training_evidence(audit):
    assert audit["source"]["future_training_integration_tested"] is False
    assert audit["source"]["historical_768_evidence"]["skipped"] == 10
    assert audit["source"]["missing_future_adapter_features"]


def test_f02_exact_requested_guard_boundary_does_not_invalidate_allowed_skips():
    safety = NumericalSafety()
    for i in range(15):
        assert safety.observe(event(i, False, scale_after=2.0)) is None
    assert safety.observe(event(15, True, scale_after=2.0)) is None
    for i in range(16, 31):
        assert safety.observe(event(i, False, scale_after=1.0)) is None
    assert safety.summary()["skipped"] == 30


def test_f02_incomplete_event_and_duplicate_are_not_silent_skips():
    s = NumericalSafety()
    assert s.observe({}) == "FAIL_TELEMETRY_INCOMPLETE"
    s = NumericalSafety()
    assert s.observe(event()) is None
    assert s.observe(event()) == "FAIL_TELEMETRY_INCOMPLETE"


def test_f02_gradient_observation_can_report_overflow_without_failure():
    assert NumericalSafety().observe(event(applied=False, gradient_nonfinite=True,
                                          gradient_observation="observed", scale_after=32768.0)) is None
