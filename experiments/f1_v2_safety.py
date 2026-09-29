"""Observational V2 runtime bindings; no change to stock optimizer/scaler policy."""
from experiments.f02_safety import NumericalSafety
from copy import deepcopy
import math


class SafetyStop(RuntimeError):
    pass


def finite(tensors):
    import torch
    values = list(tensors)
    return bool(values) and bool(torch.stack([torch.isfinite(p.detach()).all() for p in values]).all().item())


class RuntimeSafety:
    def __init__(self, sink):
        self.sink = sink
        self.state = NumericalSafety()
        self.epoch = self.batch = -1
        self.raw_batches = 0
        self.nonfinite_loss_events = self.nonfinite_parameter_events = 0
        self.parameter_finite = None
        self.active = False
        self.post_hooks = 0
        self.handle = None
        self.minimum_scaler = None
        self.max_consecutive_skips = 0
        self.runtime_exceptions = 0
        self.oom = False
        self.gradient_nonfinite = None
        self.loss_finite = None

    def stop(self, reason):
        self.state.failure = self.state.failure or reason
        self.sink('numerical', dict(event='stop', reason=reason, stop_pair=True,
                                   epoch=self.epoch, batch=self.batch))
        raise SafetyStop(reason)

    def set_batch(self, epoch, batch):
        if self.state.failure:
            self.stop(self.state.failure)
        self.epoch, self.batch = epoch, batch
        self.loss_finite = None

    def bind_optimizer(self, trainer):
        self.trainer = trainer
        opt = trainer.optimizer
        if self.handle is not None or opt.defaults.get('fused') or getattr(opt, '_step_supports_amp_scaling', False):
            self.stop('FAIL_RUNTIME_SAFETY_ADAPTER_BINDING')
        def after_step(optimizer, args, kwargs):
            if not self.active:
                self.stop('FAIL_TELEMETRY_INCOMPLETE')
            self.post_hooks += 1
        self.handle = opt.register_step_post_hook(after_step)

    def bind_scaler(self, trainer):
        self.trainer = trainer
        self.parameter_finite = finite(p for p in trainer.model.parameters() if p.requires_grad)
        if not self.parameter_finite:
            self.nonfinite_parameter_events += 1
            self.stop('FAIL_NONFINITE_PARAMETERS')
        original = trainer.scaler.scale
        def guarded_scale(loss, *args, **kwargs):
            if loss is not trainer.loss:
                self.stop('FAIL_RUNTIME_SAFETY_ADAPTER_BINDING')
            total_ok = finite([loss])
            components_ok = finite([trainer.loss_items])
            self.loss_finite = total_ok and components_ok
            self.raw_batches += 1
            self.sink('numerical', dict(event='raw_loss', epoch=self.epoch, batch=self.batch,
                raw_total_loss_finite=total_ok, raw_components_finite=components_ok))
            if self.state.check_loss(total_ok and components_ok):
                self.nonfinite_loss_events += 1
                self.stop('FAIL_NONFINITE_LOSS')
            return original(loss, *args, **kwargs)
        trainer.scaler.scale = guarded_scale
        original_unscale = trainer.scaler.unscale_
        def observed_unscale(optimizer):
            if not self.active or optimizer is not trainer.optimizer:
                self.stop('FAIL_RUNTIME_SAFETY_ADAPTER_BINDING')
            result = original_unscale(optimizer)
            gradients = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
            self.gradient_nonfinite = not finite(gradients) if gradients else None
            return result
        trainer.scaler.unscale_ = observed_unscale

    def opportunity(self, trainer, stock_step):
        if self.state.failure or self.active or self.handle is None or self.loss_finite is not True:
            self.stop(self.state.failure or 'FAIL_RUNTIME_SAFETY_ADAPTER_BINDING')
        before = float(trainer.scaler.get_scale())
        if not math.isfinite(before):
            self.stop('FAIL_TELEMETRY_INCOMPLETE')
        if before < 1:
            self.state.failure = 'FAIL_AMP_SCALE_COLLAPSE'
            self.stop(self.state.failure)
        row = dict(epoch=self.epoch, batch=self.batch, batch_index=self.batch,
            global_batch=self.epoch * 193 + self.batch, accumulation=int(trainer.accumulate),
            scale_before=before, loss_finite=True, optimizer_opportunity_attempted=True,
            learning_rate=[float(g['lr']) for g in trainer.optimizer.param_groups],
            scheduler_state=deepcopy(trainer.scheduler.state_dict()))
        self.active = True
        self.post_hooks = 0
        self.gradient_nonfinite = None
        accounted = False
        try:
            result = stock_step()
            after = float(trainer.scaler.get_scale())
            row.update(scale_after=after, optimizer_post_hook_count=self.post_hooks)
            if self.post_hooks not in (0, 1):
                raise SafetyStop('FAIL_TELEMETRY_INCOMPLETE')
            applied = self.post_hooks == 1
            if applied:
                self.parameter_finite = finite(p for p in trainer.model.parameters() if p.requires_grad)
            row.update(optimizer_update_applied=applied, optimizer_update_skipped=not applied,
                unknown=0, gradient_nonfinite=self.gradient_nonfinite,
                gradient_observation='not_observed' if self.gradient_nonfinite is None else 'observed',
                parameter_finite=self.parameter_finite)
            failure = self.state.observe(row)
            accounted = True
            self.minimum_scaler = min([before, after] + ([] if self.minimum_scaler is None else [self.minimum_scaler]))
            self.max_consecutive_skips = max(self.max_consecutive_skips, self.state.consecutive_skipped)
            if self.parameter_finite is False:
                self.nonfinite_parameter_events += 1
            row.update(consecutive_skips=self.state.consecutive_skipped, stop_pair=bool(failure), failure=failure)
            self.sink('optimizer', row)
            self.sink('numerical', dict(event='optimizer_opportunity', **row))
        except BaseException as exc:
            self.runtime_exceptions += 1
            self.oom = 'out of memory' in str(exc).lower() or type(exc).__name__ == 'OutOfMemoryError'
            reason = 'FAIL_RESOURCE_RUNTIME' if self.oom else ('FAIL_TELEMETRY_INCOMPLETE' if accounted or str(exc) == 'FAIL_TELEMETRY_INCOMPLETE' else 'TRAINING_NUMERICAL_RUNTIME_INVALID')
            if not accounted:
                self.state.scheduled += 1
                self.state.unknown += 1
            self.state.failure = reason
            self.state.runtime_classification = 'TRAINING_NUMERICAL_RUNTIME_INVALID'
            if not accounted:
                row.update(optimizer_update_applied=None, optimizer_update_skipped=None, unknown=1)
            row.update(optimizer_post_hook_count=self.post_hooks, error_type=type(exc).__name__, error=str(exc),
                failure=reason, stop_pair=True)
            try:
                if not accounted:
                    self.sink('optimizer', row)
                self.sink('numerical', dict(event='runtime_failure', **row))
            finally:
                raise SafetyStop(reason) from exc
        finally:
            self.active = False
        if failure:
            self.stop(failure)
        return result

    def summary(self):
        return dict(self.state.summary(), raw_batches=self.raw_batches,
            max_consecutive_skips=self.max_consecutive_skips, minimum_scaler=self.minimum_scaler,
            nonfinite_loss_events=self.nonfinite_loss_events, nonfinite_parameter_events=self.nonfinite_parameter_events,
            OOM=self.oom, runtime_exceptions=self.runtime_exceptions)
