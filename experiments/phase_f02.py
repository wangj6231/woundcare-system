"""F0.2 protocol/source/hash audit only. Standard library; no training entrypoint."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import ast
import sys
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
PRO = ROOT / "experiments/protocols"
OUT = ROOT / "experiments/results/f_higher_scale_v2_protocol_revision"
PREFIX = "F_HIGHER_SCALE_V2"
INIT_SHA = "1ec926026473beafafb1ef8da6aa6efcc13d38c391ec8db48006c5d793d5f2a3"
FUTURE = ["experiments/results/f_higher_scale_v2_seed42_control768",
          "experiments/results/f_higher_scale_v2_seed42_train1024"]
REQUEST = Path("C:/Users/milo9/.codex/attachments/9612a151-f34c-45db-866a-cc6eef8604f9/貼上的文字.txt")
MEDIA_ALLOWLIST = set()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def leaves(value, prefix=""):
    if not isinstance(value, dict) or not value:
        return {prefix: value}
    result = {}
    for key, item in value.items():
        result.update(leaves(item, f"{prefix}.{key}" if prefix else key))
    return result


def make_pair():
    pair = []
    for arm, kind, output in zip(("C4", "H4"), ("control", "experimental"), FUTURE):
        cfg = deepcopy(read(PRO / f"F_HIGHER_SCALE_V1_{kind}_config.json"))
        cfg.update(schema="F_HIGHER_INPUT_SCALE_V2", experiment_id=f"{PREFIX}_{arm}", output_path=output,
                   future_runtime="shared V2 runner binding REQUIRED in separately authorized F1; no F1 runner in F0.2")
        cfg["evaluation"]["path"] = f"experiments/protocols/{PREFIX}_evaluation_protocol.json"
        cfg["checkpoint_selection"].pop("F0_validation_execution", None)
        cfg["checkpoint_selection"].update(shared_override="experiments.f01_validator.SharedValidation768",
            runtime_requirement="both arms same shared V2 runner and shared override; nominal768 with stock rect/pad/stride",
            F02_validation_execution=False)
        cfg["budget_contract"]["OOM"] = "FAIL_RESOURCE_RUNTIME; stop pair; no retry, batch/scale reduction or compensation"
        cfg["numerical_safety_contract"] = f"experiments/protocols/{PREFIX}_numerical_safety_contract.json"
        cfg["amp_policy"] = {"enabled": True, "initialization": "stock GradScaler defaults, same both arms",
                             "manual_initial_scale": None, "fresh_scaler": True, "fresh_optimizer": True}
        cfg["software_provenance_note"] = ("software fields inherited distribution metadata; NumPy metadata2.0.1 != "
            "F0.1 observed runtime2.2.6. No package change here; future F1 must record actual imported runtime and source identity.")
        pair.append(cfg)
    return tuple(pair)


def validate_pair(control, experimental):
    a, b = leaves(control), leaves(experimental)
    changed = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    require(changed == ["arm", "experiment_id", "output_path", "training_args.imgsz"], "FAIL_NON_SCALE_DIFFERENCE")
    require(control["training_args"]["imgsz"] == 768 and experimental["training_args"]["imgsz"] == 1024, "FAIL_SCALE")
    for cfg, kind in zip((control, experimental), ("control", "experimental")):
        frozen = read(PRO / f"F_HIGHER_SCALE_V1_{kind}_config.json")
        for key in ("architecture", "augmentation", "loss", "runtime_optimizer", "training_args", "budget",
                    "initialization", "manifest", "dataset_manifest_sha256", "data_yaml_path", "sampling"):
            require(cfg[key] == frozen[key], "FAIL_FROZEN_RECIPE: " + key)
        require(cfg["evaluation"]["imgsz"] == cfg["checkpoint_selection"]["both_arms_validation_imgsz"] == 768, "FAIL_EVAL_SCALE")
        require(cfg["checkpoint_selection"]["shared_override"] == "experiments.f01_validator.SharedValidation768", "FAIL_SHARED_VALIDATION")
        require(cfg["amp_policy"] == make_amp_policy(), "FAIL_STOCK_AMP")
        require(cfg["label_path"] == "ORIGINAL_STOCK_YOLO_POLYGON" and cfg["dataset_adapter"] == "STOCK_YOLODataset", "FAIL_ORIGINAL_LABELS")
        require(cfg["execution_authorized"] is False and not cfg["initialization"]["resume"], "FAIL_EXECUTION_AUTHORITY")
    return changed


def make_amp_policy():
    return {"enabled": True, "initialization": "stock GradScaler defaults, same both arms",
            "manual_initial_scale": None, "fresh_scaler": True, "fresh_optimizer": True}


def admit_training_row(row):
    require((row.get("source"), row.get("split"), row.get("role")) ==
            ("FUSeg", "train", "wound_finetuning"), "FAIL_ORIGINAL_TRAIN_ONLY")
    for kind, folder in (("image", "images"), ("label", "labels")):
        prefix = f"outputs/isic_fuseg_pretrain_smoke_20260914/fuseg_dataset/{folder}/train/"
        path = row[kind + "_path"].replace("\\", "/")
        require(path.startswith(prefix) and "/" not in path[len(prefix):] and ".." not in path.split("/"), "FAIL_ORIGINAL_TRAIN_ONLY")
        require((ROOT / path).resolve().parent == (ROOT / prefix).resolve(), "FAIL_PATH_ESCAPE")
    return True


def future_absent(root=ROOT):
    require(all(not (root / p).exists() for p in FUTURE), "FAIL_FUTURE_OUTPUT_EXISTS")
    return True


def contracts():
    c, h = make_pair()
    p = deepcopy(read(PRO / "F_HIGHER_SCALE_V1_protocol.json"))
    p.pop("created_at", None)
    p.pop("STOP_AFTER_F0", None)
    p.update(schema="F_HIGHER_INPUT_SCALE_V2", STOP_AFTER_F02=True, future_outputs_absent=FUTURE,
        SYNTHETIC_NUMERICAL_GATE_DISCRIMINATIVE_VALIDITY="NOT_ESTABLISHED",
        synthetic_fixture_role="DESCRIPTIVE_ONLY_NOT_TRAINING_READINESS_HARD_GATE",
        rationale="Same synthetic gate rejected empirically completed768 recipe; change observation method, not training recipe or performance gates",
        historical_failures={"F0": "BLOCKED", "F01": "BLOCKED", "N768": "PERSISTENT_NUMERICAL_INSTABILITY",
            "N1024": "PERSISTENT_NUMERICAL_INSTABILITY", "NUMERICAL_FEASIBILITY_1024": "FAIL_UNCHANGED",
            "N1024_attempts7_8": "DESCRIPTIVE_EVIDENCE_OF_SCALER_BACKOFF_PROGRESS"},
        scope={k: False for k in ("TRAINING", "MODEL_LOADING", "MODEL_FORWARD", "GPU_EXECUTION", "RESEARCH_INFERENCE")},
        readiness_meaning="Protocol ready for separately authorized F1; not1024 numerical PASS, performance PASS, or execution permission",
        checkpoint_selection="SharedValidation768 on BOTH arms via same future runner; nominal768, stock rect/pad/stride unchanged",
        numerical_safety_contract=f"experiments/protocols/{PREFIX}_numerical_safety_contract.json",
        validation_override_contract=f"experiments/protocols/{PREFIX}_validation_override_contract.json",
        F1_training_authorized=False, budget=deepcopy(c["budget_contract"]))
    p["forbidden"] = ["additional synthetic attempts", "synthetic rerun", "F02 model load/forward/backward/GPU",
        "patch/canonical/mask-first labels", "label topology repair", "D2 weighted sampling", "batch reduction",
        "manual scaler override", "accumulation compensation", "resume/top-up/retry", "960/896/832 scale",
        "locked test", "CO2Wounds", "external test", "App replacement"]
    safety = {
        "schema": "F_HIGHER_INPUT_SCALE_V2_NUMERICAL_SAFETY",
        "scope": "Future actual stock AMP optimizer opportunities; pure event contract tested in F02 only",
        "amp_policy": make_amp_policy(), "stock_initial_scale_observed_in_source": 65536,
        "initial_scale_source_note": "record stock default; not a custom init_scale argument; never set differently by arm",
        "consecutive_skip_limit": 16, "scale_minimum_inclusive": 1.0,
        "hard_stops": {"nonfinite_raw_loss": "FAIL_NONFINITE_LOSS",
            "nonfinite_parameters_after_update": "FAIL_NONFINITE_PARAMETERS",
            "scale_below_1": "FAIL_AMP_SCALE_COLLAPSE",
            "16_consecutive_scheduled_skips": "FAIL_PERSISTENT_AMP_UPDATE_SKIPS",
            "runtime_exception": "TRAINING_NUMERICAL_RUNTIME_INVALID", "OOM": "FAIL_RESOURCE_RUNTIME"},
        "OOM_secondary_classification": "TRAINING_NUMERICAL_RUNTIME_INVALID",
        "incomplete_telemetry": "FAIL_TELEMETRY_INCOMPLETE; pair invalid, not evidence of a scale-specific numerical failure",
        "stop_action": "flush all observed evidence; stop entire pair immediately; do not launch next arm/evaluation; no automatic retry",
        "multiple_failure_flags": "retain all observed symptoms in raw telemetry; primary loss > parameters > scale > skip; runtime exceptions separate",
        "opportunity_definition": "scheduled BaseTrainer.optimizer_step call after frozen accumulation; NOT every batch",
        "per_opportunity_fields": ["epoch", "batch", "global_batch", "scale_before", "scale_after", "loss_finite",
            "optimizer_opportunity_attempted", "optimizer_update_applied", "optimizer_update_skipped",
            "gradient_nonfinite", "gradient_observation", "parameter_finite"],
        "additional_durable_fields": ["arm", "opportunity_index", "learning_rate", "scheduler_state",
            "accumulation", "observed_optimizer_post_hooks", "exception_type", "hard_stop_flags"],
        "loss_observation": "check raw aggregate loss and loss components EVERY batch before scaler.scale(loss).backward; rejectNaN/Inf immediately even between opportunities",
        "update_observation": "D0.1/D1/D2 optimizer post-hook:0=skipped,1=applied,>1=invalid; supported non-fused AdamW only; never infer from scaler.step return or scale change alone",
        "gradient_observation": "observe after stock unscale before clipping when available; otherwise null+not_observed; no extra unscale/clip/backward, no fake finite state",
        "parameter_observation": "check all model parameters initially and after each applied update, before another batch/validation/save; retain finite state on skip; no parameter mutation",
        "scale_observation": "both before and after every opportunity; NaN/Inf scale is invalid telemetry/runtime; either value<1 stops",
        "streak": "increment on confirmed skip only; reset only by confirmed applied update; carry across epoch boundaries; unknown is invalid not a skip",
        "total_skip_count_is_hard_gate": False,
        "accounting": {"scheduled": 3741, "completed_epochs": 300, "applied_plus_skipped_equals_scheduled": True, "unknown": 0},
        "imbalance": "C4 skipped != H4 skipped -> AMP_UPDATE_COUNT_IMBALANCE_OBSERVED=YES; report realized counts differ; no compensation/retrain",
        "early_stop": "completed_epochs<300 -> PAIRED_FIXED_BUDGET_VALID=NO regardless of reason; patience80 unchanged; no resume/top-up",
        "future_integration_status": "REQUIRED_NOT_EXECUTED_IN_F02",
        "reference_implementation": "experiments.f02_safety.NumericalSafety",
        "legacy_telemetry_source": "experiments/paired_sampling_runner.py",
        "integration_requirement": "same V2 runner and guards BOTH arms; wrap/observe stock steps without replacing scaler policy, loss, clip or scheduler; bind tested pure contract before any future training",
        "legacy_limitations": "legacy runner provides post-hook accounting but lacks full raw-loss/parameter/streak stops; do not call it unchanged as V2 runner",
    }
    validation = deepcopy(read(PRO / "F_HIGHER_SCALE_V1_NUMERICAL_PREFLIGHT_validator_contract.json"))
    validation.update(schema="F_HIGHER_INPUT_SCALE_V2_VALIDATION", audit_scope="F02 source/hash only; inherited F01 dry evidence, not rerun",
        common_runner="same future V2 trainer class mixing SharedValidation768 for C4 and H4",
        inherited_evidence={"path": "experiments/results/f_higher_scale_v1_numerical_preflight/validator_dry_audit.json",
            "sha256": sha(ROOT / "experiments/results/f_higher_scale_v1_numerical_preflight/validator_dry_audit.json")},
        implementation={"path": "experiments/f01_validator.py", "sha256": sha(ROOT / "experiments/f01_validator.py")},
        val_nominal_imgsz=768, historical_synthetic_batch_shape=[4, 3, 800, 800],
        actual_future_tensor_shape="record at runtime; no fixed768x768 promise; common stock rect/pad/stride",
        model_forward_in_F02=False)
    return {"protocol": p, "control_config": c, "experimental_config": h,
            "numerical_safety_contract": safety, "validation_override_contract": validation,
            "evaluation_protocol": read(PRO / "F_HIGHER_SCALE_V1_evaluation_protocol.json")}


def install_execution_guard():
    """No frameworks, decode, process launch or network in this protocol-only process."""
    denied = {"torch", "ultralytics", "cv2", "PIL", "numpy", "tensorflow", "cupy", "onnxruntime"}
    require(not denied.intersection(sys.modules), "FAIL_FRAMEWORK_ALREADY_IMPORTED")
    evidence = {"framework_imports": [], "subprocess_launches": 0, "network_calls": 0,
                "media_reads_hash_only": 0, "GPU_EXECUTION": False, "MODEL_LOADING": False,
                "MODEL_FORWARD": False, "TRAINING": False, "RESEARCH_INFERENCE": False}
    def guard(event, args):
        if event == "import" and args[0].split(".")[0] in denied:
            evidence["framework_imports"].append(args[0])
            raise RuntimeError("F02_FRAMEWORK_IMPORT_FORBIDDEN")
        if event in {"subprocess.Popen", "os.system", "os.spawn", "os.posix_spawn"}:
            evidence["subprocess_launches"] += 1
            raise RuntimeError("F02_PROCESS_LAUNCH_FORBIDDEN")
        if event in {"socket.connect", "socket.bind"}:
            evidence["network_calls"] += 1
            raise RuntimeError("F02_NETWORK_FORBIDDEN")
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pt", ".pth"}:
                require(path in MEDIA_ALLOWLIST, "F02_UNAPPROVED_MEDIA_ACCESS")
                mode = args[1] or ""
                require(not any(m in mode for m in "wax+"), "F02_MEDIA_WRITE_FORBIDDEN")
                evidence["media_reads_hash_only"] += 1
    sys.addaudithook(guard)
    return evidence


def historical_audit(protected):
    states = {}
    seals = ["F_HIGHER_SCALE_V1", "F_HIGHER_SCALE_V1_NUMERICAL_PREFLIGHT",
             "E_PATCH_V1", "E_PATCH_V2", "E_PATCH_V3"]
    for prefix in seals:
        path = PRO / f"{prefix}_freeze.json"
        seal = read(path)
        protected[str(path)] = sha(path)
        for relative, digest in seal["artifacts_sha256"].items():
            target = (ROOT / relative).resolve()
            require(target.is_relative_to(ROOT), "FAIL_SEAL_PATH_ESCAPE")
            MEDIA_ALLOWLIST.add(target)  # Some old E-phase synthetic figures, hash only; never labels.
            require(sha(target) == digest, "FAIL_HISTORICAL_ARTIFACT_CHANGED: " + relative)
            protected[str(target)] = digest
        states[prefix] = {k: v for k, v in seal.items() if k.endswith("STATUS") or k == "PATCH_BASED_TRAINING_ROUTE"}
    require(states[seals[0]]["PHASE_F0_STATUS"] == "BLOCKED", "FAIL_F0_STATUS")
    require(states[seals[1]]["PHASE_F01_STATUS"] == "BLOCKED", "FAIL_F01_STATUS")
    attempts = {}
    base = ROOT / "experiments/results/f_higher_scale_v1_numerical_preflight"
    for arm in ("N768", "N1024"):
        names = sorted(p.name for p in (base / arm).glob("attempt_*.json"))
        require(names == [f"attempt_{i:02d}.json" for i in range(1, 9)], "FAIL_EXTRA_OR_MISSING_ATTEMPT")
        summary = read(base / arm / "summary.json")
        require(summary["completed_attempts"] == 8 and summary["classification"] == "PERSISTENT_NUMERICAL_INSTABILITY", "FAIL_HISTORICAL_CLASSIFICATION")
        attempts[arm] = names
    return {"states": states, "synthetic_attempt_files": attempts,
            "SYNTHETIC_NUMERICAL_GATE_DISCRIMINATIVE_VALIDITY": "NOT_ESTABLISHED"}


def dataset_audit(cfg, protected):
    path = ROOT / cfg["manifest"]["path"]
    require(sha(path) == cfg["manifest"]["sha256"], "FAIL_TRAIN_MANIFEST_HASH")
    protected[str(path)] = sha(path)
    rows = read(path)["samples"]
    require(len(rows) == len({r["sample_id"] for r in rows}) == 771, "FAIL_771_ORIGINAL_IMAGES")
    actual_gt = 0
    for row in rows:
        admit_training_row(row)
        for kind in ("image", "label"):
            file = ROOT / row[kind + "_path"]
            MEDIA_ALLOWLIST.add(file.resolve())
            require(sha(file) == row[kind + "_hash"], "FAIL_TRAIN_HASH: " + row["sample_id"])
            protected[str(file)] = row[kind + "_hash"]
        lines = [s.split() for s in (ROOT / row["label_path"]).read_text(encoding="utf-8").splitlines() if s.strip()]
        require(len(lines) == row["GT_count"] and all(r[0] == "0" for r in lines), "FAIL_GT_COUNT_CLASS")
        actual_gt += len(lines)
    require(actual_gt == 965, "FAIL_965_GT")
    evaluation = read(PRO / "F_HIGHER_SCALE_V1_evaluation_protocol.json")
    manifest_path = ROOT / evaluation["validation_manifest"]["path"]
    require(sha(manifest_path) == evaluation["validation_manifest"]["sha256"], "FAIL_VAL_MANIFEST")
    protected[str(manifest_path)] = sha(manifest_path)
    val_rows = read(manifest_path)["samples"]
    require(len(val_rows) == len({r["sample_id"] for r in val_rows}) == 191, "FAIL_VAL191")
    require(not set(r["image_hash"] for r in rows) & set(r["image_sha256"] for r in val_rows), "FAIL_EXACT_TRAIN_VAL_OVERLAP")
    for row in val_rows:
        require((row["source_dataset"], row["source_role"], row["split"]) == ("FUSeg", "DEVELOPMENT_VALIDATION", "val"), "FAIL_VAL_ROLE")
        image = ROOT / row["relative_image_path"]
        expected_parent = ROOT / "outputs/fuseg_warmup_revision_20260914/dataset/images/val"
        require(image.resolve().parent == expected_parent.resolve(), "FAIL_VAL_IMAGE_PATH")
        mask = ROOT / row["relative_mask_path"]
        expected_mask_parent = ROOT / "official_detection_sources_20260812/fuseg/repository/data/Foot Ulcer Segmentation Challenge/validation/labels"
        require(mask.resolve().parent == expected_mask_parent.resolve(), "FAIL_VAL_MASK_PATH")
        label = image.parent.parent.parent / "labels/val" / (image.stem + ".txt")
        training_val_image = ROOT / "outputs/isic_fuseg_pretrain_smoke_20260914/fuseg_dataset/images/val" / row["sample_id"]
        training_val_label = training_val_image.parent.parent.parent / "labels/val" / (image.stem + ".txt")
        for file, expected in ((image, row["image_sha256"]), (mask, row["mask_sha256"]),
                               (label, row["label_sha256"]), (training_val_image, row["image_sha256"]),
                               (training_val_label, row["label_sha256"])):
            MEDIA_ALLOWLIST.add(file.resolve())
            require(sha(file) == expected, "FAIL_VAL_FILE_HASH: " + str(file))
            protected[str(file)] = expected
    init = ROOT / cfg["initialization"]["path"]
    MEDIA_ALLOWLIST.add(init.resolve())
    require(sha(init) == INIT_SHA, "FAIL_INITIALIZATION_HASH")
    protected[str(init)] = INIT_SHA
    order_path = ROOT / cfg["sampling"]["order_path"]
    require(sha(order_path) == cfg["sampling"]["order_sha256"], "FAIL_ORDER_HASH")
    order = read(order_path)
    require(len(order["orders"]) == 300 and order["replacement"] is False, "FAIL_ORDER_CONTRACT")
    ids = {r["sample_id"] for r in rows}
    for seq, digest in zip(order["orders"], order["epoch_sha256"], strict=True):
        require(len(seq) == len(set(seq)) == 771 and set(seq) == ids, "FAIL_ORDER_IDS")
        require(hashlib.sha256(json.dumps(seq, separators=(",", ":")).encode()).hexdigest() == digest, "FAIL_ORDER_SEQUENCE_HASH")
    protected[str(order_path)] = sha(order_path)
    data_yaml = ROOT / cfg["data_yaml_path"]
    text = data_yaml.read_text(encoding="utf-8")
    require("train: images/train" in text and "val: images/val" in text and "test:" not in text, "FAIL_DATA_YAML_SPLITS")
    protected[str(data_yaml)] = sha(data_yaml)
    return {"train_images": 771, "original_polygon_GT": actual_gt, "development_validation_images": 191,
        "training_validation_copy_hash_match": True, "exact_train_val_hash_overlap": 0,
        "all_image_label_hashes_match": True, "label_repairs": 0, "patch_labels_used": False,
        "mask_first_labels_used": False, "training_representation": "ORIGINAL_STOCK_YOLO_POLYGON",
        "initialization_sha256": INIT_SHA, "model_deserialized": False, "images_decoded": 0,
        "anchor_epochs": 300, "anchors_per_epoch": 771, "order_sha256": sha(order_path),
        "test_images_used": 0, "CO2Wounds_used": False}


def source_audit(cfg, protected):
    site = Path("C:/Python312/Lib/site-packages")
    # Resolve the top-level package location WITHOUT importing torch (user-site install).
    torch_root = Path(importlib.util.find_spec("torch").origin).parent
    source_entries = dict(cfg["architecture"]["source_hashes"])
    source_entries.update(read(PRO / "F_HIGHER_SCALE_V1_NUMERICAL_PREFLIGHT_validator_contract.json")["sources"])
    for rel, expected in source_entries.items():
        file = torch_root / rel.removeprefix("torch/") if rel.startswith("torch/") else site / "ultralytics" / rel
        require(sha(file) == expected, "FAIL_RUNTIME_SOURCE_PIN: " + rel)
        protected[str(file)] = expected
    evidence = []
    selections = {
        site / "ultralytics/engine/trainer.py": ["optimizer_step", "_setup_train", "_do_train"],
        torch_root / "amp/grad_scaler.py": ["__init__", "_maybe_opt_step", "step", "update"],
        ROOT / "experiments/paired_sampling_runner.py": ["bind_optimizer", "opportunity", "end_epoch"],
        ROOT / "experiments/f01_validator.py": ["build_dataset", "get_validator"],
    }
    for file, names in selections.items():
        source = file.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
                segment = ast.get_source_segment(source, node)
                evidence.append({"path": str(file), "function": node.name, "line": node.lineno,
                                 "end_line": node.end_lineno, "segment_sha256": hashlib.sha256(segment.encode()).hexdigest()})
        protected[str(file)] = sha(file)
    trainer = (site / "ultralytics/engine/trainer.py").read_text(encoding="utf-8")
    require('torch.amp.GradScaler("cuda", enabled=self.amp)' in trainer, "FAIL_STOCK_SCALER_SOURCE")
    scaler = (torch_root / "amp/grad_scaler.py").read_text(encoding="utf-8")
    require("init_scale: float = 2.0**16" in scaler and "growth_interval: int = 2000" in scaler, "FAIL_SCALER_DEFAULT")
    historical = ROOT / "experiments/results/d_seg_small_sampling_v2_control"
    completion = read(historical / "training_completion.json")
    integrity = read(historical / "telemetry_integrity.json")
    require(completion["completed_epochs"] == 300 and completion["scheduled_optimizer_calls"] == 3741
            and completion["applied_optimizer_updates"] == 3731 and completion["skipped_optimizer_updates"] == 10
            and completion["unknown_optimizer_opportunities"] == 0 and integrity["status"] == "PASS", "FAIL_HISTORICAL_RUNTIME_EVIDENCE")
    require(completion["runner_sha256"] == sha(ROOT / "experiments/paired_sampling_runner.py"), "FAIL_HISTORICAL_RUNNER_PIN")
    legacy_cfg = PRO / "D_SEG_SMALL_SAMPLING_V2_control_config.json"
    legacy = read(legacy_cfg)
    require(legacy["training_args"] == read(PRO / "F_HIGHER_SCALE_V1_control_config.json")["training_args"], "FAIL_768_RECIPE_LINEAGE")
    for file in (historical / "training_completion.json", historical / "telemetry_integrity.json", legacy_cfg):
        protected[str(file)] = sha(file)
    require(sum(cfg["budget"]["scheduled_optimizer_calls_per_epoch"]) == 3741, "FAIL_SCHEDULE_BUDGET")
    return {"source_functions": evidence,
        "stock_order": ["unscale", "clip_norm10", "scaler.step", "scaler.update", "zero_grad", "EMA"],
        "stock_scaler_defaults": {"init_scale": 65536, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000},
        "historical_768_evidence": {"path": str(historical.relative_to(ROOT)), "completed_epochs": 300,
            "scheduled": 3741, "applied": 3731, "skipped": 10, "unknown": 0, "telemetry_integrity": "PASS",
            "scope": "empirical completion of inherited recipe, not proof all future runs stable; not V2 causal comparator"},
        "existing_post_hook_telemetry_reusable": True,
        "missing_future_adapter_features": ["every-batch raw-loss guard before backward", "post-update parameter finite scan",
            "cross-epoch consecutive skip state", "scale-collapse guard", "durable stop-pair propagation", "shared V2 trainer binding"],
        "future_training_integration_tested": False,
        "numpy_provenance": "inherited metadata2.0.1; F01 runtime2.2.6 observed; not imported in F02, no package changed"}


def inspect():
    future_absent()
    c, h = make_pair()
    diff = validate_pair(c, h)
    protected = {}
    historical = historical_audit(protected)
    data = dataset_audit(c, protected)
    sources = source_audit(c, protected)
    for file in (PRO / "F_HIGHER_SCALE_V1_evaluation_protocol.json",):
        evaluation = read(file)
        for spec in (evaluation["metric_implementation"],):
            target = ROOT / spec["path"]
            require(sha(target) == spec["sha256"], "FAIL_EVALUATOR_PIN")
            protected[str(target)] = spec["sha256"]
    return {"historical": historical, "data": data, "source": sources, "paired_config_differences": diff,
            "protected_sha256": protected, "future_outputs_absent": FUTURE}


def prepare():
    guard = install_execution_guard()
    require(not OUT.exists() and not list(PRO.glob(PREFIX + "_*.json")), "FAIL_F02_ALREADY_EXISTS")
    audit = inspect()
    bundle = contracts()
    for name, value in bundle.items():
        path = PRO / f"{PREFIX}_{name}.json"
        if name == "evaluation_protocol":
            # Exact bytes retain the already frozen evaluation-protocol SHA.
            with path.open("xb") as dest:
                dest.write((PRO / "F_HIGHER_SCALE_V1_evaluation_protocol.json").read_bytes())
        else:
            save(path, value)
    save(OUT / "audit.json", audit)
    save(OUT / "execution_guard.json", guard)
    save(OUT / "prepared_seal.json", {"created_at": datetime.now(timezone.utc).isoformat(),
        "request_sha256": sha(REQUEST), "artifacts_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
            [*PRO.glob(PREFIX + "_*.json"), OUT / "audit.json", OUT / "execution_guard.json",
             ROOT / "experiments/phase_f02.py", ROOT / "experiments/f02_safety.py"]}})
    print(json.dumps({"prepared": True, "dataset": audit["data"], "GPU_EXECUTION": False}, ensure_ascii=False))


if __name__ == "__main__":
    require(sys.argv[1:] == ["prepare"], "ONLY_PROTOCOL_PREPARE_ALLOWED")
    prepare()
