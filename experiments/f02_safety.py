"""Pure event contract, NOT a training runner or proof of runtime integration."""
from math import isfinite


class NumericalSafety:
    def __init__(self):
        self.scheduled = self.applied = self.skipped = self.unknown = 0
        self.consecutive_skipped = 0
        self.failure = None
        self.runtime_classification = None
        self.last_global_batch = -1

    def observe(self, row):
        if self.failure:
            raise RuntimeError("PAIR_ALREADY_STOPPED")
        self.scheduled += 1
        required = {"epoch", "batch", "global_batch", "scale_before", "scale_after",
                    "loss_finite", "optimizer_opportunity_attempted", "optimizer_update_applied",
                    "optimizer_update_skipped", "gradient_nonfinite", "gradient_observation", "parameter_finite"}
        valid = required <= row.keys()
        if valid:
            valid = (all(type(row[k]) is bool for k in ("loss_finite", "optimizer_opportunity_attempted",
                     "optimizer_update_applied", "optimizer_update_skipped", "parameter_finite"))
                     and row["optimizer_opportunity_attempted"]
                     and row["optimizer_update_applied"] != row["optimizer_update_skipped"]
                     and all(type(row[k]) is int and row[k] >= 0 for k in ("epoch", "batch", "global_batch"))
                     and row["batch"] < 193
                     and row["global_batch"] == row["epoch"] * 193 + row["batch"]
                     and row["global_batch"] > self.last_global_batch
                     and all(type(row[k]) in (int, float) and isfinite(row[k]) for k in ("scale_before", "scale_after"))
                     and ((type(row["gradient_nonfinite"]) is bool and row["gradient_observation"] == "observed")
                          or (row["gradient_nonfinite"] is None and row["gradient_observation"] == "not_observed")))
        if not valid:
            self.unknown += 1
            self.failure = "FAIL_TELEMETRY_INCOMPLETE"
            return self.failure
        self.last_global_batch = row["global_batch"]
        if row["optimizer_update_applied"]:
            self.applied += 1
            self.consecutive_skipped = 0
        else:
            self.skipped += 1
            self.consecutive_skipped += 1
        if row["loss_finite"] is False:
            self.failure = "FAIL_NONFINITE_LOSS"
        elif row["parameter_finite"] is False:
            self.failure = "FAIL_NONFINITE_PARAMETERS"
        elif min(row["scale_before"], row["scale_after"]) < 1.0:
            self.failure = "FAIL_AMP_SCALE_COLLAPSE"
        elif self.consecutive_skipped >= 16:
            self.failure = "FAIL_PERSISTENT_AMP_UPDATE_SKIPS"
        return self.failure

    def check_loss(self, finite):
        """Future adapter must call on EVERY raw loss before scaled backward."""
        if self.failure:
            raise RuntimeError("PAIR_ALREADY_STOPPED")
        if finite is not True:
            self.failure = "FAIL_NONFINITE_LOSS" if finite is False else "FAIL_TELEMETRY_INCOMPLETE"
        return self.failure

    def summary(self):
        return dict(scheduled=self.scheduled, applied=self.applied, skipped=self.skipped,
                    unknown=self.unknown, consecutive_skipped=self.consecutive_skipped,
                    failure=self.failure, runtime_classification=self.runtime_classification)

    def runtime_exception(self, kind, *, opportunity_attempted=False):
        if self.failure:
            raise RuntimeError("PAIR_ALREADY_STOPPED")
        if opportunity_attempted:
            self.scheduled += 1
            self.unknown += 1
        self.runtime_classification = "TRAINING_NUMERICAL_RUNTIME_INVALID"
        self.failure = "FAIL_RESOURCE_RUNTIME" if kind == "OOM" else self.runtime_classification
        return self.failure


def compare_completed(control, experimental):
    """Budget validity only; never a performance PASS or execution authorization."""
    def valid(s):
        keys = ("completed_epochs", "scheduled", "applied", "skipped", "unknown")
        return (all(type(s.get(k)) is int and s[k] >= 0 for k in keys)
                and s["completed_epochs"] == 300 and s["scheduled"] == 3741
                and s["applied"] + s["skipped"] == s["scheduled"] and s["unknown"] == 0
                and s.get("telemetry_complete") is True and s.get("failure", "MISSING") is None)
    complete = valid(control) and valid(experimental)
    return {"PAIRED_FIXED_BUDGET_VALID": "YES" if complete else "NO",
            "AMP_UPDATE_COUNT_IMBALANCE_OBSERVED": ("YES" if control["skipped"] != experimental["skipped"] else "NO") if complete else "UNDETERMINED",
            "identical_realized_update_counts": complete and control["applied"] == experimental["applied"],
            "retry_or_compensation_allowed": False}
