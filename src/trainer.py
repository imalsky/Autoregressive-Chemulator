#!/usr/bin/env python3
"""
trainer.py - LightningModule + Trainer factory.

Training styles:

1) One-jump
   - training.rollout_steps must be 1
   - loss on the first transition only

2) Autoregressive rollout (long unroll)
   - Total rollout steps K := training.rollout_steps
   - Skip/warmup steps M := training.autoregressive_training.skip_steps
       * warmup is run under torch.no_grad() (pushforward trick input generation)
       * warmup steps are excluded from the loss via masking
   - Gradients: BPTT across steps [M..K-1] (but never through warmup), every trained step weighted equally
   - The state is carried in precision.input_dtype: float64 carries it in float64 (forward_step runs the
     network in float32 and adds the increment in float64); the shards stay float32.

Long-jump term (training.loss.jump_weight > 0; one-jump or BPTT):
- every training batch carries jump pairs (z(t), z(t + D)) from the raw-trajectory store, D log-uniform
  in [jump_min_seconds, jump_max_seconds] (dataset.py sample_jump_pairs), and each step adds
  jump_weight * HybridLoss(forward_step(z(t), D), z(t + D)) to its loss. D may exceed the trained dt
  range: the relaxation head's network never sees dt. The states are float64; forward_step runs the
  network in float32 and adds the increment in float64.

Optional curriculum (training only):
- training.curriculum ramps training rollout steps from start_steps -> end_steps
  (main.py requires end_steps == rollout_steps) over ramp_epochs.
- Validation always uses rollout_steps (fixed horizon for comparability).
- Early stopping starts counting at the first epoch after the LR warmup and the curriculum ramp.

LR schedule (training.scheduler):
- Linear warmup over warmup_epochs, then a cosine from the peak (training.optimizer.lr) down to
  min_lr_fraction x peak at the run's last step (max_epochs x train_batches_per_epoch), advanced once
  per optimizer step (warmup_cosine_lr). With early stopping on, a run that stops early ends before
  the LR has decayed to the floor.

Logging:
- Uses Lightning CSVLogger. All self.log(...) metrics are written to metrics.csv.
- LearningRateMonitor logs LR automatically.
- train_loss (and its two components) is the window loss only; with the long-jump term on,
  train_jump_loss is the epoch mean of the unweighted jump loss, and each step optimizes
  train_loss + jump_weight * train_jump_loss.
- Persistence diagnostics (validation): val_persistence_loss is the val_loss definition (same
  weights, mask and windows) scored on y_next = y, the window's first state carried forward;
  val_skill = 1 - val_loss / val_persistence_loss from the two epoch means. Neither is monitored.
- Per-epoch timing uses time.perf_counter at three device-synchronized hook boundaries (epoch
  start, validation start, epoch end; no sync per step):
    epoch_time_sec            training plus the end-of-epoch validation pass (checkpoint writes
                              come after it). Training time = epoch_time_sec - perf_val_time_s.
    perf_val_time_s           the validation pass. The EMA weight swap-in runs just before this
                              module's on_validation_start (callback hooks first), so it counts as
                              training time; it takes milliseconds.
    perf_train_step_ms        training time per training step.
    perf_train_samples_per_s  training samples (steps x batch_size) per second of training time.
    perf_train_tflops         model matmul FLOP/s over training time: a forward is
                              2*sum(in*out) over the nn.Linear layers per sample-step, a step with
                              backward counts 3 forwards, a no-grad skip (pushforward) step 1,
                              and the long-jump term 3 x jump_batch_size / batch_size.
    perf_peak_mem_gb          CUDA only: peak allocated memory over the epoch (training and
                              validation), 1e9 bytes.
  Epoch 0 also contains the torch.compile and CUDA-graph warm-up.
- One INFO line per epoch in the job log: epoch, K_train, the LR at epoch start and now (after
  this epoch's last optimizer step), train/val losses and epoch_time_sec.

Batch contract (from dataset.py):
- y:  [B, K_full, S]   (states in z-space)
- dt: [B, K_full-1]    (per-transition normalized dt)
- g:  [B, G]           (G = 2 mandatory: P, T)
- training batches with the long-jump term on also hold jump_z0, jump_z1 [Bj, S] float64,
  jump_dt_norm [Bj] float32 and jump_g [Bj, G] float32 (Bj = training.loss.jump_batch_size)

Note on "pushforward trick":
- skip_steps warmup runs A(u) forward without grad, and training loss is computed on later steps.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from model import autoregressive_rollout, normalize_dt_shape

from utils import PrecisionConfig, normalization_sha256
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.core.optimizer import LightningOptimizer

from ema_callback import EMACallback
from lightning.pytorch.loggers import CSVLogger


# The one monitored metric and its direction, shared by top-k checkpointing (best.ckpt) and early
# stopping (no config key: any other value would desynchronize them).
_MONITOR_METRIC = "val_loss"
_MONITOR_MODE = "min"
# last.ckpt must hold the latest completed epoch (auto-resume starts from it).
LAST_CKPT_EVERY_N_EPOCHS = 1
# Checkpoint key holding the val_loss of the epoch that saved it: Lightning's callback state keeps only
# best/top-k scores, and main.py publishes a run that died after its final last.ckpt from this value.
CKPT_EPOCH_VAL_LOSS_KEY = "epoch_val_loss"
# One validation batch before epoch 0 runs the validation step (and its compiled path) once, so a
# compile or shape failure surfaces in minutes instead of after the first ~2 h epoch.
_SANITY_VAL_BATCHES = 1
# Per-epoch perf metrics (see Logging in the module docstring).
_FLOPS_PER_MAC = 2              # one multiply-add
_TRAIN_FLOPS_PER_FORWARD = 3    # forward + backward (input and weight gradients, ~2 forwards)
_BYTES_PER_GB = 1e9
_MS_PER_S = 1e3
_FLOPS_PER_TFLOP = 1e12
# Float format of the per-epoch INFO line (val_loss values of order 1e-3 and below need more than the
# 3 decimals EarlyStopping prints).
_EPOCH_LOG_FLOAT_FMT = "%.6e"
log = logging.getLogger(__name__)

# ==============================================================================
# Optimizer parameter groups
# ==============================================================================

# The model's only normalization layer (model.py uses nn.LayerNorm and nothing else).
_NORM_MODULE_TYPES = (nn.LayerNorm,)


def _build_optimizer_param_groups(
    module: nn.Module,
    *,
    weight_decay: float,
    exclude_norm_and_bias: bool,
) -> List[Dict[str, Any]]:
    """Create optimizer param groups, optionally excluding norm/bias from weight decay."""
    wd = float(weight_decay)
    named = [(n, p) for n, p in module.named_parameters() if p.requires_grad]

    if wd == 0.0 or not exclude_norm_and_bias:
        return [{"params": [p for _, p in named], "weight_decay": wd}]

    norm_param_names: set[str] = set()
    for mod_name, mod in module.named_modules():
        if isinstance(mod, _NORM_MODULE_TYPES):
            for pn, _ in mod.named_parameters(recurse=False):
                full = f"{mod_name}.{pn}" if mod_name else pn
                norm_param_names.add(full)

    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for name, p in named:
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "bias" or name in norm_param_names:
            no_decay.append(p)
        else:
            decay.append(p)

    groups: List[Dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": wd})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def warmup_cosine_lr(step: int, total: int, peak: float, warmup: int, min_frac: float) -> float:
    """LR of optimizer step `step` (0-based): a linear warmup to `peak` over `warmup` steps, then a cosine
    from `peak` down to min_frac x peak at step `total`, held at that floor after it."""
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


class WarmupCosineLR(torch.optim.lr_scheduler.LRScheduler):
    """warmup_cosine_lr of the optimizer's base LR (the peak), advanced once per optimizer step.

    The LR is a pure function of the steps taken (last_epoch). The schedule's steps and floor are
    scheduler state: a resume restores them with last_epoch, so it continues the schedule it started
    with (a resume that raises training.max_epochs trains the added epochs at the floor).
    """

    def __init__(self, optimizer: torch.optim.Optimizer, *, warmup_steps: int, total_steps: int,
                 min_lr_fraction: float) -> None:
        # Set before super().__init__, whose initial step() sets the LR of step 0.
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr_fraction = float(min_lr_fraction)
        super().__init__(optimizer)

    @property
    def warmup_active(self) -> bool:
        """True while the next optimizer step is still inside the linear warmup."""
        return self.last_epoch < self.warmup_steps

    def get_lr(self) -> List[float]:
        """LR of optimizer step last_epoch for every param group."""
        return [
            warmup_cosine_lr(self.last_epoch, self.total_steps, base_lr, self.warmup_steps, self.min_lr_fraction)
            for base_lr in self.base_lrs
        ]


# ==============================================================================
# Loss
# ==============================================================================


class HybridLoss(nn.Module):
    """Two-term loss: log10-space MAE and z-space MSE.

    z is normalized log10-space:
        z = (log10(y_phys) - log_mean) / log_std
    """

    def __init__(
        self,
        log_means: torch.Tensor,  # [1,1,S]
        log_stds: torch.Tensor,   # [1,1,S]
        *,
        lambda_log10_mae: float,
        lambda_z_mse: float,
        loss_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.loss_dtype = loss_dtype
        self.register_buffer("log_means", log_means.clone().detach().to(dtype=loss_dtype))
        self.register_buffer("log_stds", log_stds.clone().detach().to(dtype=loss_dtype))
        self.lambda_log10_mae = float(lambda_log10_mae)
        self.lambda_z_mse = float(lambda_z_mse)

    def forward(
        self,
        pred_z: torch.Tensor,  # [B,K,S]
        true_z: torch.Tensor,  # [B,K,S]
        *,
        step_weights: Optional[torch.Tensor] = None,  # [K]
    ) -> Dict[str, torch.Tensor]:
        """Compute the hybrid loss over [B,K,S] predictions.

        Returns loss_total = lambda_log10_mae * log10_MAE + lambda_z_mse * z_MSE, plus the two
        components. Optional per-step weights mask/scale the K rollout steps (None = uniform).
        """
        if pred_z.shape != true_z.shape:
            raise ValueError(f"Shape mismatch: pred_z={tuple(pred_z.shape)} true_z={tuple(true_z.shape)}")
        if pred_z.ndim != 3:
            raise ValueError(f"Expected [B,K,S], got {tuple(pred_z.shape)}")

        B, K, S = pred_z.shape

        w: Optional[torch.Tensor] = None
        denom: Optional[torch.Tensor] = None
        if step_weights is not None:
            if step_weights.ndim != 1 or int(step_weights.shape[0]) != int(K):
                raise ValueError(f"step_weights must have shape [K={K}], got {tuple(step_weights.shape)}")
            w = step_weights.to(device=pred_z.device, dtype=self.loss_dtype).view(1, K, 1)
            # w.sum() >= 1: FlowMapRolloutModule._step_weights refuses an all-masked horizon.
            denom = w.sum() * float(B * S)

        diff_z = (pred_z - true_z).to(self.loss_dtype)
        if w is None:
            z_mse = diff_z.square().mean()
        else:
            z_mse = diff_z.square().mul(w).sum(dtype=self.loss_dtype) / denom  # type: ignore[arg-type]

        # log10(y_phys) = z * log_std + log_mean, so the log-mean cancels in differences:
        #   |pred_log10 - true_log10| = |(pred_z - true_z) * log_std|
        diff_log10 = diff_z.abs().mul(self.log_stds)

        if w is None:
            log10_mae = diff_log10.mean()
        else:
            log10_mae = diff_log10.mul(w).sum(dtype=self.loss_dtype) / denom  # type: ignore[arg-type]

        loss_total = self.lambda_log10_mae * log10_mae + self.lambda_z_mse * z_mse
        return {"loss_total": loss_total, "loss_log10_mae": log10_mae, "loss_z_mse": z_mse}


def build_loss_buffers(
    normalization_manifest: Mapping[str, Any],
    species_variables: Sequence[str],
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create [1,1,S] log10_mean/log10_std tensors matching species_variables order."""
    if "per_key_stats" not in normalization_manifest:
        raise KeyError("Normalization manifest missing 'per_key_stats'.")
    stats = normalization_manifest["per_key_stats"]
    if not isinstance(stats, Mapping):
        raise TypeError("normalization manifest 'per_key_stats' must be a mapping.")

    means = [float(stats[name]["log_mean"]) for name in species_variables]
    stds = [float(stats[name]["log_std"]) for name in species_variables]
    log_means = torch.tensor(means, device=device, dtype=dtype).view(1, 1, -1)
    log_stds = torch.tensor(stds, device=device, dtype=dtype).view(1, 1, -1)
    return log_means, log_stds


# ==============================================================================
# torch.compile helper
# ==============================================================================


def _try_torch_compile(fn: Any, *, cfg: Mapping[str, Any]) -> Any:
    """Wrap a callable with torch.compile (strict; required keys only; the caller checks `enabled`).

    torch.compile is lazy: compilation happens at the first call, so compile errors surface there
    (the Trainer's sanity-validation batch runs the compiled eval path before epoch 0).
    """
    tc = cfg["runtime"]["torch_compile"]
    backend = str(tc["backend"])
    mode = str(tc["mode"])
    dynamic = bool(tc["dynamic"])
    fullgraph = bool(tc["fullgraph"])

    return torch.compile(fn, backend=backend, mode=mode, dynamic=dynamic, fullgraph=fullgraph)  # type: ignore[misc]


# ==============================================================================
# Curriculum
# ==============================================================================


@dataclass(frozen=True)
class RolloutCurriculum:
    """Ramps training rollout steps linearly from start_steps -> end_steps over ramp_epochs."""

    enabled: bool
    start_steps: int
    end_steps: int
    ramp_epochs: int

    def steps(self, epoch: int) -> int:
        """Return the rollout horizon for `epoch`, ramping start_steps -> end_steps over ramp_epochs.

        Called only with the curriculum enabled (_train_rollout_steps handles enabled=false).
        """
        end_k = int(self.end_steps)
        start_k = int(self.start_steps)
        if self.ramp_epochs <= 0 or start_k >= end_k:
            return start_k

        # Exact integer arithmetic (ramp_epochs >= 1 here): no float rounding at the ramp points.
        e = int(max(0, epoch))
        ramp = int(self.ramp_epochs)
        return start_k + ((end_k - start_k) * min(e, ramp)) // ramp

    def first_full_horizon_epoch(self) -> int:
        """First epoch trained at end_steps: ramp_epochs while the ramp is active, else 0.

        For epoch < ramp_epochs, ((end - start) * epoch) // ramp < end - start, so steps() first
        returns end_steps at epoch == ramp_epochs.
        """
        if self.enabled and self.ramp_epochs > 0 and self.start_steps < self.end_steps:
            return int(self.ramp_epochs)
        return 0


def _build_curriculum(tcfg: Mapping[str, Any]) -> RolloutCurriculum:
    """RolloutCurriculum from the (schema-required) training.curriculum block."""
    cur_cfg = dict(tcfg["curriculum"])
    return RolloutCurriculum(
        enabled=bool(cur_cfg["enabled"]),
        start_steps=int(cur_cfg["start_steps"]),
        end_steps=int(cur_cfg["end_steps"]),
        ramp_epochs=int(cur_cfg["ramp_epochs"]),
    )


class CurriculumAwareEarlyStopping(EarlyStopping):
    """EarlyStopping that skips the curriculum ramp and the LR warmup.

    An epoch is skipped while it is before `start_epoch` (the first full-horizon epoch of the
    rollout curriculum) or while the WarmupCosineLR warmup is still active (read from the
    scheduler). In a skipped epoch a finite metric leaves best/wait untouched, so ES starts
    counting at the first epoch after the warmup and the ramp; the non-finite stop and the
    missing-metric error stay active from epoch 0.
    Overrides private EarlyStopping methods (Lightning 2.6.x).
    """

    def __init__(self, *, start_epoch: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.start_epoch = int(start_epoch)
        self._skip_epoch = False

    def _run_early_stopping_check(self, trainer: pl.Trainer) -> None:
        in_warmup = any(
            isinstance(c.scheduler, WarmupCosineLR) and c.scheduler.warmup_active
            for c in trainer.lr_scheduler_configs
        )
        self._skip_epoch = int(trainer.current_epoch) < self.start_epoch or in_warmup
        super()._run_early_stopping_check(trainer)

    def _evaluate_stopping_criteria(self, current: torch.Tensor) -> Tuple[bool, Optional[str]]:
        if self._skip_epoch and bool(torch.isfinite(current).all()):
            return False, None
        return super()._evaluate_stopping_criteria(current)


# ==============================================================================
# Lightning module
# ==============================================================================


class FlowMapRolloutModule(pl.LightningModule):
    """LightningModule wrapper around the underlying model (must implement forward_step)."""

    def __init__(
        self,
        model: nn.Module,
        *,
        cfg: Mapping[str, Any],
        normalization_manifest: Mapping[str, Any],
        species_variables: Sequence[str],
        precision: PrecisionConfig,
    ) -> None:
        """Wire the model, loss, precision, curriculum, AR settings, and optional torch.compile from cfg."""
        super().__init__()
        self.model = model
        self.normalization_manifest = dict(normalization_manifest)
        self.species_variables = list(species_variables)

        # Precision (centralized): controls dataset/model/loss dtypes and Lightning AMP settings.
        self.precision_config = precision
        self.input_dtype = self.precision_config.input_dtype
        self.loss_dtype = self.precision_config.loss_dtype

        tcfg = dict(cfg["training"])
        mcfg = dict(cfg["model"])

        self.rollout_steps = int(tcfg["rollout_steps"])
        if self.rollout_steps < 1:
            raise ValueError(f"training.rollout_steps must be >= 1, got {self.rollout_steps}")

        # The full autoregressive_training block is schema-required (enabled, skip_steps); read it
        # directly. When AR is disabled skip_steps is inert (the one-jump training path ignores it).
        ar_cfg = dict(tcfg["autoregressive_training"])
        self.autoregressive_training = bool(ar_cfg["enabled"])
        self.ar_skip_steps = int(ar_cfg["skip_steps"])
        if self.ar_skip_steps < 0:
            raise ValueError(f"training.autoregressive_training.skip_steps must be >= 0, got {self.ar_skip_steps}")

        # Gradient-norm clip threshold (None = no clipping). Lightning forbids
        # Trainer(gradient_clip_val=...) with manual optimization, so in AR mode the module clips
        # with this value in _maybe_step_optimizer (build_lightning_trainer withholds it from the
        # Trainer); in one-jump mode the Trainer passes the same value to configure_gradient_clipping.
        runtime_cfg = dict(cfg["runtime"])
        clip_raw = runtime_cfg["gradient_clip_val"]  # required; null = no clipping (main.py rejects <= 0)
        self._grad_clip_val: Optional[float] = (
            float(clip_raw) if clip_raw is not None and float(clip_raw) > 0.0 else None
        )

        if not self.autoregressive_training and self.rollout_steps != 1:
            raise ValueError(
                "One-jump training requires training.rollout_steps == 1. "
                f"Got rollout_steps={self.rollout_steps} with autoregressive_training.enabled=False."
            )

        if self.autoregressive_training:
            self.automatic_optimization = False

        # Curriculum (training only). The full block is schema-required; read directly.
        self.curriculum = _build_curriculum(tcfg)

        # Loss
        loss_cfg = dict(tcfg["loss"])
        log_means, log_stds = build_loss_buffers(
            self.normalization_manifest,
            self.species_variables,
            device=torch.device("cpu"),
            dtype=self.loss_dtype,
        )
        self.criterion = HybridLoss(
            log_means,
            log_stds,
            lambda_log10_mae=float(loss_cfg["lambda_log10_mae"]),
            lambda_z_mse=float(loss_cfg["lambda_z_mse"]),
            loss_dtype=self.loss_dtype,
        )
        # Long-jump term (0 = off): jump_weight * criterion(forward_step(z(t), D), z(t + D)) on the batch's
        # jump pairs, added to every training step's loss (main.py checks the keys' ranges).
        self.jump_weight = float(loss_cfg["jump_weight"])
        self._jump_batch_size = int(loss_cfg["jump_batch_size"])

        # Optimizer steps per epoch (one per training batch): the train loader's length, which is
        # dataset.train_batches_per_epoch by construction (dataset.py _PreloadedStochasticBatchStream).
        self._train_batches_per_epoch = int(cfg["dataset"]["train_batches_per_epoch"])

        # Validate model interface once at init (RelaxationHead provides forward_step and delta).
        if not hasattr(self.model, "forward_step"):
            raise RuntimeError("Model must implement forward_step(y_t, dt, g) -> y_{t+1}.")

        # Optional compilation (torch_compile block is schema-required).
        runtime = dict(cfg["runtime"])
        tc = dict(runtime["torch_compile"])
        self.log_grad_norm = bool(runtime["log_grad_norm"])
        self._compiled_forward_step: Optional[Any] = None
        if bool(tc["enabled"]) and bool(tc["compile_forward_step"]):
            self._compiled_forward_step = _try_torch_compile(self.model.forward_step, cfg=cfg)  # type: ignore[attr-defined]

        # Epoch timing: perf_counter at epoch start and at validation start (None outside an epoch,
        # e.g. the sanity batch); the LR the epoch started with.
        self._epoch_t0: Optional[float] = None
        self._val_t0: Optional[float] = None
        self._epoch_lr_start: Optional[float] = None
        # Matmul FLOPs of one forward_step per sample (nn.Linear holds every matmul of the model);
        # perf_train_tflops and testing/profile_training.py count FLOPs from it.
        self.fwd_flops_per_sample_step = sum(
            _FLOPS_PER_MAC * m.in_features * m.out_features for m in self.model.modules() if isinstance(m, nn.Linear)
        )
        # AR training horizon of the current epoch (set by the curriculum); logged once per epoch.
        self._epoch_k_train: Optional[int] = None
        # Per-epoch device sum of [loss_total, loss_log10_mae, loss_z_mse] and step count; the epoch
        # means are logged once in on_train_epoch_end instead of three self.log calls per step.
        # A per-step mean equals Lightning's per-sample mean only because every training batch is
        # full-size (the training stream yields only full batches, dataset.py); the perf metrics'
        # sample count (steps x batch_size) relies on the same fact.
        self._train_loss_sum: Optional[torch.Tensor] = None
        self._train_loss_n = 0
        # Per-epoch device sum of the unweighted long-jump loss (one term per training step; None when off).
        self._train_jump_sum: Optional[torch.Tensor] = None
        # The latest epoch's val_loss (a float), stored in every checkpoint (CKPT_EPOCH_VAL_LOSS_KEY).
        self._epoch_val_loss: Optional[float] = None
        # runtime.log_grad_norm: per-epoch device sum of [pre-clip total grad norm, clip fired] over
        # optimizer steps, logged once per epoch as grad_norm (mean) and grad_clip_frac.
        self._grad_stats_sum: Optional[torch.Tensor] = None
        self._grad_stats_n = 0

        # Preserve for checkpoint reproducibility. Deep-copy so later config mutations
        # (or Lightning internals) can't alias the live training config. norm_sha256 ties every
        # checkpoint to the normalization it was trained in (checked on resume, weights_only and export).
        self.save_hyperparameters({
            "training": copy.deepcopy(dict(tcfg)),
            "model": copy.deepcopy(dict(mcfg)),
            "norm_sha256": normalization_sha256(self.normalization_manifest),
        })

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Resume: refuse a checkpoint trained in a different normalization (z-space) than this run's."""
        hparams = checkpoint.get("hyper_parameters")
        saved = hparams.get("norm_sha256") if isinstance(hparams, Mapping) else None
        if saved != self.hparams["norm_sha256"]:
            raise ValueError(
                f"resume checkpoint norm_sha256={saved!r} does not match this processed_dir's normalization.json "
                f"({self.hparams['norm_sha256']}); the checkpoint was trained on other normalization statistics"
            )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Store the saving epoch's val_loss (already a host float; CKPT_EPOCH_VAL_LOSS_KEY)."""
        checkpoint[CKPT_EPOCH_VAL_LOSS_KEY] = self._epoch_val_loss

    # ------------------------
    # Fit-time note about metric comparability
    # ------------------------

    def on_fit_start(self) -> None:
        """Log a note (once per fit) on how train_loss and val_loss differ."""
        if bool(self.hparams["training"]["ema"]["enabled"]):
            weights = "val_loss uses the EMA weights"
        else:
            weights = "val_loss uses the end-of-epoch live weights"
        weights += ", train_loss is the epoch mean at each batch's live weights"
        dropout = float(self.hparams["model"]["dropout"])
        dropout_note = f"; dropout ({dropout}) is active only in training" if dropout > 0.0 else ""
        jump_note = (
            f"; the optimized loss is train_loss + {self.jump_weight:g} x train_jump_loss (the long-jump term, "
            "logged separately)" if self.jump_weight > 0.0 else ""
        )
        if self.autoregressive_training:
            msg = (
                "train_loss is the autoregressive rollout loss over the curriculum horizon "
                "(train_rollout_steps); val_loss is the open-loop rollout over rollout_steps; both weight "
                "every step >= skip_steps equally. They also differ in weights "
                f"({weights}) and data (training vs validation windows){dropout_note}{jump_note}."
            )
        else:
            msg = (
                "train_loss and val_loss are the same one-step loss; they differ in weights "
                f"({weights}) and data (training vs validation windows){dropout_note}{jump_note}."
            )
        log.info(msg)

    def on_train_start(self) -> None:
        """Log the peak LR in force (after a resume has restored the optimizer and scheduler state)."""
        sched = self._get_scheduler()
        peak = float(sched.base_lrs[0]) if sched is not None else self._current_lr()
        log.info("peak LR (warmup target, cosine start) %.6g; training.optimizer.lr %.6g; LR now %.6g",
                 peak, float(self.hparams["training"]["optimizer"]["lr"]), self._current_lr())

    # ------------------------
    # Epoch timing
    # ------------------------

    def _sync_device(self) -> None:
        """Wait for the queued GPU work (epoch-boundary timing only; never called per step)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _current_lr(self) -> float:
        """LR of the first param group (both groups share it)."""
        return float(self.trainer.optimizers[0].param_groups[0]["lr"])

    def on_train_epoch_start(self) -> None:
        # The training stream draws this epoch's windows from (seed, epoch): resume-safe.
        self.trainer.train_dataloader.dataset.set_epoch(int(self.current_epoch))
        self._sync_device()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._epoch_lr_start = self._current_lr()
        self._val_t0 = None
        self._epoch_t0 = time.perf_counter()
        self._train_loss_sum = None
        self._train_loss_n = 0
        self._train_jump_sum = None
        self._grad_stats_sum = None
        self._grad_stats_n = 0

    def on_validation_start(self) -> None:
        """Mark the end of this epoch's training steps (skipped outside an epoch: sanity batch, init validation)."""
        if self._epoch_t0 is None:
            return
        self._sync_device()
        self._val_t0 = time.perf_counter()

    def train_forwards_per_sample(self) -> float:
        """Forward-equivalents per training sample of this epoch's steps (after its first step).

        A trained step (forward + backward) counts _TRAIN_FLOPS_PER_FORWARD, a no-grad skip step
        (pushforward warmup) 1; one-jump training is one trained step. The long-jump term adds one
        trained step on jump_batch_size pairs per batch_size samples.
        """
        k, skip = (self._epoch_k_train, self.ar_skip_steps) if self.autoregressive_training else (1, 0)
        forwards = skip + _TRAIN_FLOPS_PER_FORWARD * (k - skip)
        if self.jump_weight > 0.0:
            forwards += _TRAIN_FLOPS_PER_FORWARD * self._jump_batch_size / int(self.hparams["training"]["batch_size"])
        return forwards

    def _log_epoch_perf(self) -> float:
        """Log epoch_time_sec and the perf_* metrics (module docstring); return epoch_time_sec."""
        self._sync_device()
        t_end = time.perf_counter()
        epoch_s = t_end - self._epoch_t0
        self.log("epoch_time_sec", epoch_s, on_step=False, on_epoch=True)
        train_s = epoch_s
        if self._val_t0 is not None:
            self.log("perf_val_time_s", t_end - self._val_t0, on_step=False, on_epoch=True)
            train_s = self._val_t0 - self._epoch_t0
        steps = self._train_loss_n
        if steps > 0 and train_s > 0.0:
            samples = steps * int(self.hparams["training"]["batch_size"])
            flops = samples * self.train_forwards_per_sample() * self.fwd_flops_per_sample_step
            self.log("perf_train_step_ms", _MS_PER_S * train_s / steps, on_step=False, on_epoch=True)
            self.log("perf_train_samples_per_s", samples / train_s, on_step=False, on_epoch=True)
            self.log("perf_train_tflops", flops / train_s / _FLOPS_PER_TFLOP, on_step=False, on_epoch=True)
        if self.device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(self.device) / _BYTES_PER_GB
            self.log("perf_peak_mem_gb", peak_gb, on_step=False, on_epoch=True)
        return epoch_s

    def _log_epoch_line(self, *, k_train: int, train_loss: float, epoch_s: float) -> None:
        """One INFO line per epoch in the job log (one .item() per metric, once per epoch)."""
        nan = float("nan")
        cm = self.trainer.callback_metrics

        def metric(name: str) -> float:
            v = cm.get(name)
            return float(v) if v is not None else nan

        fields = (
            ("lr_start", self._epoch_lr_start if self._epoch_lr_start is not None else nan),
            ("lr_now", self._current_lr()),
            ("train_loss", train_loss),
            ("val_loss", metric("val_loss")),
            ("val_loss_log10_mae", metric("val_loss_log10_mae")),
            ("val_loss_z_mse", metric("val_loss_z_mse")),
            ("epoch_time_sec", epoch_s),
        )
        log.info(
            "epoch %d: K_train=%d " + " ".join(f"{name}={_EPOCH_LOG_FLOAT_FMT}" for name, _ in fields),
            int(self.current_epoch), k_train, *(v for _, v in fields),
        )

    def on_train_epoch_end(self) -> None:
        epoch_s = self._log_epoch_perf() if self._epoch_t0 is not None else float("nan")
        k_train = self._epoch_k_train if self._epoch_k_train is not None else self.rollout_steps
        if self._epoch_k_train is not None:
            self.log("train_rollout_steps", float(self._epoch_k_train), on_step=False, on_epoch=True)
            self._epoch_k_train = None
        train_loss = float("nan")
        if self._train_loss_n > 0:
            # Device-tensor divisor: tensor / Python number would take CUDA's multiply-by-reciprocal
            # path and could differ from Lightning's epoch mean by 1 ulp.
            m = self._train_loss_sum / self._train_loss_sum.new_full((), float(self._train_loss_n))
            self.log("train_loss", m[0], on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_loss_log10_mae", m[1], on_step=False, on_epoch=True)
            self.log("train_loss_z_mse", m[2], on_step=False, on_epoch=True)
            if self._train_jump_sum is not None:
                jump = self._train_jump_sum / self._train_jump_sum.new_full((), float(self._train_loss_n))
                self.log("train_jump_loss", jump, on_step=False, on_epoch=True)
            train_loss = float(m[0])
            self._train_loss_sum, self._train_jump_sum, self._train_loss_n = None, None, 0
        if self._grad_stats_n > 0:
            g = self._grad_stats_sum / self._grad_stats_sum.new_full((), float(self._grad_stats_n))
            self.log("grad_norm", g[0], on_step=False, on_epoch=True)
            if self._grad_clip_val is not None:
                self.log("grad_clip_frac", g[1], on_step=False, on_epoch=True)
            self._grad_stats_sum, self._grad_stats_n = None, 0
        if self.trainer.is_global_zero:
            self._log_epoch_line(k_train=k_train, train_loss=train_loss, epoch_s=epoch_s)
        self._epoch_t0 = None
        self._val_t0 = None

    # ------------------------
    # Forward utilities
    # ------------------------

    def _forward_step(self, y_t: torch.Tensor, dt: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Single model step y_t -> y_{t+1}, using the compiled step when available."""
        fn = self._compiled_forward_step if self._compiled_forward_step is not None else self.model.forward_step  # type: ignore[attr-defined]
        return fn(y_t, dt, g)  # type: ignore[misc]

    def _coerce_g(self, g: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Move globals g [B, G] to `like`'s device/dtype (the model's forward_step checks the shape)."""
        return g.to(device=like.device, dtype=like.dtype)

    def _train_rollout_steps(self, epoch: int, *, transitions: int) -> int:
        """Training horizon for this epoch: min(rollout_steps, available transitions, curriculum step)."""
        if not self.curriculum.enabled:
            return int(min(self.rollout_steps, transitions))
        k = int(self.curriculum.steps(epoch))
        return int(min(self.rollout_steps, transitions, k))

    def _eval_rollout_steps(self, *, transitions: int) -> int:
        """Fixed eval horizon: min(rollout_steps, available transitions) (no curriculum)."""
        return int(min(self.rollout_steps, transitions))

    def _cast_batch_tensors(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        g: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cast batch tensors to the configured input dtype.

        Lightning moves tensors to the target device; this function only adjusts dtype.
        """
        target = self.input_dtype
        if y.dtype != target:
            y = y.to(dtype=target)
        if dt.dtype != target:
            dt = dt.to(dtype=target)
        if g.dtype != target:
            g = g.to(dtype=target)
        return y, dt, g

    @staticmethod
    def _step_weights(
        k: int,
        skip_steps: int,
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Per-step loss weights (float64): 0 for warmup steps, 1 after.

        Returns None (uniform, no masking) when no step is skipped. Raises when every step would
        be masked.
        """
        k = int(k)
        skip = int(skip_steps)
        if skip >= k:
            raise ValueError(f"skip_steps ({skip}) must be < the rollout horizon K={k} (every step would be masked).")
        if skip <= 0:
            return None
        w = torch.ones(k, device=device, dtype=torch.float64)
        w[:skip] = 0.0
        return w

    def _open_loop_unroll(
        self,
        *,
        y0: torch.Tensor,     # [B,S]
        dt_bk: torch.Tensor,  # [B,K]
        g: torch.Tensor,      # [B,G]
    ) -> torch.Tensor:
        """Open-loop rollout: feed predictions back in (no teacher forcing)."""
        g_t = self._coerce_g(g, y0)
        step_fn = self._compiled_forward_step if self._compiled_forward_step is not None else self.model.forward_step  # type: ignore[attr-defined]
        return autoregressive_rollout(step_fn, y0, dt_bk, g_t)

    # ------------------------
    # Manual optimization helpers (autoregressive mode)
    # ------------------------

    def _get_single_optimizer(self) -> LightningOptimizer:
        """Return the single optimizer (configure_optimizers builds exactly one AdamW)."""
        opt = self.optimizers()
        if not isinstance(opt, LightningOptimizer):
            raise TypeError(f"expected one optimizer, got {type(opt).__name__}")
        return opt

    def _get_scheduler(self) -> Optional[WarmupCosineLR]:
        """Return the single LR scheduler, or None.

        None when training.scheduler.enabled=false, and outside fit (the weights_only init
        Trainer.validate pass sets up no optimizers or schedulers).
        """
        sched = self.lr_schedulers()
        if sched is not None and not isinstance(sched, WarmupCosineLR):
            raise TypeError(f"expected one WarmupCosineLR, got {type(sched).__name__}")
        return sched

    def _clip_and_record_grad_norm(self, params: List[torch.nn.Parameter], clip_val: Optional[float]) -> None:
        """Clip the total gradient norm to clip_val (None: no clip) and record it when log_grad_norm.

        Same clip_grad_norm_ call as Lightning's fp32 clip, called directly: Lightning's bf16-mixed
        plugin refuses to clip for a fused optimizer (a GradScaler guard that does not apply to bf16).
        The pre-clip norm and a clip-fired flag are summed on the device (no per-step host sync).
        Reading .grad here is within the step, which CUDA-graph replay allows.
        """
        if clip_val is not None:
            total = torch.nn.utils.clip_grad_norm_(params, clip_val)
        elif self.log_grad_norm:
            total = torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None])
        else:
            return
        if not self.log_grad_norm:
            return
        norm = total.detach().float()
        fired = (norm > clip_val).float() if clip_val is not None else torch.zeros_like(norm)
        stats = torch.stack([norm, fired])
        self._grad_stats_sum = stats if self._grad_stats_sum is None else self._grad_stats_sum + stats
        self._grad_stats_n += 1

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        """Automatic-optimization clip hook (one-jump mode), called before every optimizer step."""
        if gradient_clip_algorithm not in (None, "norm"):
            super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
            return
        clip_val = float(gradient_clip_val) if gradient_clip_val is not None and float(gradient_clip_val) > 0.0 else None
        params = [p for group in optimizer.param_groups for p in group["params"]]
        self._clip_and_record_grad_norm(params, clip_val)

    def _maybe_step_optimizer(self, opt: LightningOptimizer) -> None:
        """Optimizer step after every batch: clip (and optional grad-norm record), step, zero_grad.

        Manual-optimization path only (autoregressive mode; gradient accumulation is unsupported,
        main.py rejects accumulate_grad_batches != 1). Then advances the LR schedule by one step.
        """
        # The Trainer is constructed WITHOUT gradient_clip_val in AR mode (Lightning forbids it).
        params = [p for group in opt.optimizer.param_groups for p in group["params"]]
        self._clip_and_record_grad_norm(params, self._grad_clip_val)

        opt.step()
        opt.zero_grad(set_to_none=True)

        sched = self._get_scheduler()
        if sched is None:
            return
        sched.step()

    def lr_scheduler_step(self, scheduler: Any, metric: Optional[Any]) -> None:
        """No-op: the module advances the schedule after every optimizer step in both optimization
        modes (optimizer_step, _maybe_step_optimizer); Lightning's per-epoch step would add one more."""

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        """Automatic-optimization hook: after the normal step, advance the LR schedule by one step."""
        # Lightning calls this hook only in automatic optimization (stage 1); lr_scheduler_step is a
        # no-op, so the schedule is advanced here.
        super().optimizer_step(*args, **kwargs)

        sched = self._get_scheduler()
        if sched is None:
            return
        sched.step()

    # ------------------------
    # Training / eval
    # ------------------------

    def _jump_term(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Unweighted long-jump loss: criterion(forward_step(z(t), D), z(t + D)) on the batch's jump pairs.

        The pair states stay float64 (_cast_batch_tensors never sees them): forward_step runs the network
        in float32 and adds the increment in float64. dt_norm and g go in as the stream made them (float32;
        dt_norm unclamped). The term's device sum is kept for train_jump_loss.
        """
        z0, z1 = batch["jump_z0"], batch["jump_z1"]
        if z0.dtype != torch.float64 or z1.dtype != torch.float64:
            raise TypeError(f"jump_z0/jump_z1 must be float64 in the training step, got {z0.dtype}/{z1.dtype}")
        pred = self._forward_step(z0, batch["jump_dt_norm"], batch["jump_g"])
        loss = self.criterion(pred.unsqueeze(1), z1.unsqueeze(1), step_weights=None)["loss_total"]
        s = loss.detach().to(torch.promote_types(loss.dtype, torch.float32))
        self._train_jump_sum = s if self._train_jump_sum is None else self._train_jump_sum + s
        return loss

    def _training_step_one_jump(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """One-jump training: single transition y0 -> y1, loss on that one step (automatic optimization).

        With training.loss.jump_weight > 0 the returned (optimized) loss adds jump_weight x _jump_term.
        """
        y = batch["y"]
        dt = batch["dt"]
        g = batch["g"]
        y, dt, g = self._cast_batch_tensors(y, dt, g)

        B, K_full, _ = y.shape
        transitions = int(K_full) - 1
        dt_full = normalize_dt_shape(dt, batch_size=B, seq_len=transitions, context="train/one_jump")

        if K_full < 2:
            raise ValueError("one-jump training requires K_full>=2 states per window.")

        y0 = y[:, 0, :]
        y1 = y[:, 1, :]
        dt0 = dt_full[:, 0]
        g_t = self._coerce_g(g, y0)

        y_pred1 = self._forward_step(y0, dt0, g_t)
        losses = self.criterion(y_pred1.unsqueeze(1), y1.unsqueeze(1), step_weights=None)

        s = torch.stack([losses["loss_total"].detach(), losses["loss_log10_mae"].detach(), losses["loss_z_mse"].detach()])
        s = s.to(torch.promote_types(s.dtype, torch.float32))  # sum in fp32 or wider, as Lightning does
        self._train_loss_sum = s if self._train_loss_sum is None else self._train_loss_sum + s
        self._train_loss_n += 1

        if self.jump_weight > 0.0:
            return losses["loss_total"] + self.jump_weight * self._jump_term(batch)
        return losses["loss_total"]


    def _training_step_autoregressive(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Autoregressive rollout training (manual optimization).

        Runs skip_steps warmup under no_grad (pushforward), then trains steps [skip..k_train) by BPTT,
        every trained step weighted equally. Curriculum sets k_train per epoch. Handles its own
        backward/step via _maybe_step_optimizer. The state is carried in the batch's dtype
        (precision.input_dtype; float64 carries it in float64 while forward_step runs the network in
        float32). With training.loss.jump_weight > 0 it backpropagates the rollout loss plus
        jump_weight x _jump_term; the returned loss is the rollout loss only.
        """
        y = batch["y"]
        dt = batch["dt"]
        g = batch["g"]
        y, dt, g = self._cast_batch_tensors(y, dt, g)

        B, K_full, S = y.shape
        transitions = int(K_full) - 1

        dt_full = normalize_dt_shape(dt, batch_size=B, seq_len=transitions, context="train/autoregressive")
        k_train = self._train_rollout_steps(int(self.current_epoch), transitions=transitions)

        dt_train = dt_full[:, :k_train]
        y_true = y[:, 1 : 1 + k_train, :]
        # contiguous: the strided slice would give the compiled forward_step its own graph variant.
        y0 = y[:, 0, :].contiguous()
        g_t = self._coerce_g(g, y0)

        skip = int(self.ar_skip_steps)
        if skip >= k_train:
            cur = f"curriculum.start_steps={self.curriculum.start_steps}; " if self.curriculum.enabled else ""
            raise ValueError(
                f"skip_steps ({skip}) must be < the epoch-{self.current_epoch} training horizon K={k_train} "
                f"({cur}rollout_steps={self.rollout_steps})."
            )

        opt = self._get_single_optimizer()

        # Warmup once: pushforward M steps under torch.no_grad() (the outputs carry no graph).
        y_prev = y0
        if skip > 0:
            with torch.no_grad():
                for k in range(skip):
                    y_prev = self._forward_step(y_prev, dt_train[:, k], g_t)

        # BPTT across the trained portion (after warmup). No gradients through warmup.
        y_pred = y0.new_zeros((B, k_train, S))

        for k in range(skip, k_train):
            y_prev = self._forward_step(y_prev, dt_train[:, k], g_t)
            y_pred[:, k, :] = y_prev

        step_weights = self._step_weights(k_train, skip, device=y_pred.device)
        losses = self.criterion(y_pred, y_true, step_weights=step_weights)

        loss_total = losses["loss_total"]
        loss_log10_mae = losses["loss_log10_mae"].detach()
        loss_z_mse = losses["loss_z_mse"].detach()

        if self.jump_weight > 0.0:
            self.manual_backward(loss_total + self.jump_weight * self._jump_term(batch))
        else:
            self.manual_backward(loss_total)
        self._maybe_step_optimizer(opt)

        s = torch.stack([loss_total.detach(), loss_log10_mae.detach(), loss_z_mse.detach()])
        s = s.to(torch.promote_types(s.dtype, torch.float32))  # sum in fp32 or wider, as Lightning does
        self._train_loss_sum = s if self._train_loss_sum is None else self._train_loss_sum + s
        self._train_loss_n += 1
        self._epoch_k_train = int(k_train)

        return loss_total

    def _eval_step(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Open-loop rollout evaluation for val: unroll k_eval steps and log the loss, uniform over steps >= skip_steps."""
        y = batch["y"]
        dt = batch["dt"]
        g = batch["g"]
        y, dt, g = self._cast_batch_tensors(y, dt, g)

        B, K_full, _ = y.shape
        transitions = int(K_full) - 1
        k_eval = self._eval_rollout_steps(transitions=transitions)

        dt_full = normalize_dt_shape(dt, batch_size=B, seq_len=transitions, context="val/eval")
        dt_eval = dt_full[:, :k_eval]

        y0 = y[:, 0, :].contiguous()  # as in the AR training step (one forward_step graph variant fewer)
        y_true = y[:, 1 : 1 + k_eval, :]

        y_pred = self._open_loop_unroll(y0=y0, dt_bk=dt_eval, g=g)

        # Keep loss definition consistent by masking skip_steps the same way in val.
        step_weights = None
        if self.autoregressive_training:
            step_weights = self._step_weights(k_eval, self.ar_skip_steps, device=y_pred.device)

        losses = self.criterion(y_pred, y_true, step_weights=step_weights)
        loss_total = losses["loss_total"]

        self.log("val_loss", loss_total, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss_log10_mae", losses["loss_log10_mae"], on_step=False, on_epoch=True)
        self.log("val_loss_z_mse", losses["loss_z_mse"], on_step=False, on_epoch=True)
        # Persistence (y_next = y: the window's first state carried forward) under the same loss,
        # step weights and mask: the model-free baseline of val_skill (a diagnostic; never monitored).
        persistence = self.criterion(y0.unsqueeze(1).expand_as(y_true), y_true, step_weights=step_weights)
        self.log("val_persistence_loss", persistence["loss_total"], on_step=False, on_epoch=True)

        return loss_total

    def training_step(self, batch: Mapping[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        """Dispatch to the autoregressive or one-jump training step based on config."""
        if self.autoregressive_training:
            return self._training_step_autoregressive(batch)
        return self._training_step_one_jump(batch)

    def validation_step(self, batch: Mapping[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        """Open-loop rollout validation step (logs val_loss)."""
        return self._eval_step(batch)

    def on_validation_epoch_end(self) -> None:
        """Abort on a non-finite training loss or val_loss, then record val_loss and log val_skill.

        This hook runs once val_loss is final and before ModelCheckpoint/EarlyStopping act on it
        (they run in on_train_epoch_end), so a diverged epoch writes no checkpoint.
        """
        # A non-finite training loss is caught here, at the end of its epoch, not at the step: the
        # check reads the epoch's device-side loss sum once (any non-finite step loss makes it
        # non-finite), so training adds no per-step host sync. Zero steps: sanity batch, init validation.
        if self._train_loss_n > 0 and not bool(torch.isfinite(self._train_loss_sum).all()):
            raise FloatingPointError(
                f"training loss is non-finite in epoch {self.current_epoch} (sum of [loss_total, "
                f"loss_log10_mae, loss_z_mse] over its {self._train_loss_n} steps = "
                f"{self._train_loss_sum.tolist()}; global_step {self.global_step}): training diverged; "
                "caught at the end of the epoch, not at the step; aborting before this epoch's "
                "checkpoint is written."
            )
        if self._train_jump_sum is not None and not bool(torch.isfinite(self._train_jump_sum)):
            raise FloatingPointError(
                f"long-jump loss is non-finite in epoch {self.current_epoch} (sum over its {self._train_loss_n} "
                f"steps = {self._train_jump_sum.item()}; global_step {self.global_step}): training diverged; "
                "caught at the end of the epoch, not at the step; aborting before this epoch's "
                "checkpoint is written."
            )
        metric = self.trainer.callback_metrics.get(_MONITOR_METRIC)
        if metric is None:
            return
        metric_val = float(metric.detach().cpu().item()) if isinstance(metric, torch.Tensor) else float(metric)
        if not math.isfinite(metric_val):
            raise FloatingPointError(
                f"val_loss={metric_val} is non-finite at epoch {self.current_epoch} "
                f"(global_step {self.global_step}): training diverged; aborting before this "
                "epoch's checkpoint is written."
            )
        self._epoch_val_loss = metric_val
        # val_skill = 1 - val_loss / val_persistence_loss from the two epoch means (a diagnostic, also
        # logged by the weights_only init validation; a zero persistence mean gives -inf or nan).
        persistence = self.trainer.callback_metrics.get("val_persistence_loss")
        if persistence is not None:
            self.log("val_skill", 1.0 - metric / persistence, on_step=False, on_epoch=True)


    # ------------------------
    # Optimizer / scheduler
    # ------------------------

    def configure_optimizers(self) -> Any:
        """Build the AdamW optimizer (with norm/bias WD exclusion) and the LR scheduler.

        The scheduler is WarmupCosineLR (the only scheduler; peak training.optimizer.lr), stepped by
        the module after every optimizer step. Returns the Lightning optimizer/scheduler dict.
        """
        tcfg = dict(self.hparams["training"])
        opt_cfg = dict(tcfg["optimizer"])

        lr = float(opt_cfg["lr"])
        weight_decay = float(opt_cfg["weight_decay"])
        exclude_nd = bool(opt_cfg["exclude_norm_and_bias_from_weight_decay"])
        param_groups = _build_optimizer_param_groups(self, weight_decay=weight_decay, exclude_norm_and_bias=exclude_nd)

        betas = opt_cfg["betas"]
        b0, b1 = float(betas[0]), float(betas[1])
        eps = float(opt_cfg["eps"])

        use_fused = bool(opt_cfg["fused"])
        use_foreach = bool(opt_cfg["foreach"])

        # Weight decay is handled by param groups, so pass 0 here.
        opt_wd_default = 0.0

        opt_kwargs: Dict[str, Any] = {
            "params": param_groups,
            "lr": lr,
            "betas": (b0, b1),
            "eps": eps,
            "weight_decay": opt_wd_default,
        }
        if use_fused:
            opt_kwargs["fused"] = True
        else:
            opt_kwargs["foreach"] = use_foreach
        opt = torch.optim.AdamW(**opt_kwargs)

        sched_cfg = dict(tcfg["scheduler"])
        if not bool(sched_cfg["enabled"]):
            return opt

        # Warmup and run length in optimizer steps, the warmup capped at the run (main.py requires
        # warmup_epochs >= 0 and max_epochs >= 1). These equal Lightning's step counts because
        # accumulate_grad_batches == 1, the Trainer gets no max_steps or limit_train_batches, and the
        # run is single-process (main.py).
        max_epochs = int(tcfg["max_epochs"])
        warmup_steps = min(int(sched_cfg["warmup_epochs"]), max_epochs) * self._train_batches_per_epoch
        sched = WarmupCosineLR(
            opt,
            warmup_steps=warmup_steps,
            total_steps=max_epochs * self._train_batches_per_epoch,
            min_lr_fraction=float(sched_cfg["min_lr_fraction"]),
        )

        # The interval only matters to Lightning's own step, which lr_scheduler_step turns off.
        lr_scheduler: Dict[str, Any] = {
            "scheduler": sched,
            "interval": "epoch",
            "frequency": 1,
            "name": "lr",
        }
        return {"optimizer": opt, "lr_scheduler": lr_scheduler}


# ==============================================================================
# Trainer factory
# ==============================================================================


def build_lightning_trainer(
    cfg: Mapping[str, Any],
    *,
    work_dir: Path,
    precision_config: PrecisionConfig,
    train_batches_per_epoch: int,
) -> pl.Trainer:
    """Construct the Lightning Trainer from the (validated) config.

    Wires the CSV logger, LR monitor, and the optional EMA / early-stopping /
    checkpoint callbacks, plus precision and gradient clipping. All runtime and
    checkpointing keys are schema-required and read directly (no silent defaults);
    the EMA and early-stopping blocks are required too, each gated by its own
    explicit ``enabled`` flag.
    """
    tcfg = cfg["training"]
    runtime = cfg["runtime"]

    max_epochs = int(tcfg["max_epochs"])
    devices = runtime["devices"]
    accelerator = runtime["accelerator"]
    lightning_precision = precision_config.lightning_precision
    # Every metric is logged per epoch (on_epoch=True, LearningRateMonitor per epoch), so the step
    # logging interval is only Lightning's bookkeeping: one interval per epoch logs nothing extra
    # and keeps Lightning's "interval larger than the epoch" warning quiet.
    if int(train_batches_per_epoch) <= 0:
        raise ValueError(f"train_batches_per_epoch must be > 0, got {train_batches_per_epoch}")
    log_every_n_steps = int(train_batches_per_epoch)

    ckpt_cfg = dict(runtime["checkpointing"])
    enable_ckpt = bool(ckpt_cfg["enabled"])
    ckpt_every_n_epochs = int(ckpt_cfg["every_n_epochs"])
    save_top_k = int(ckpt_cfg["save_top_k"])
    save_last = bool(ckpt_cfg["save_last"])

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # CSVLogger: keep outputs rooted directly under work_dir.
    csv_logger = CSVLogger(save_dir=str(work_dir), name=".", version="")

    callbacks: List[pl.Callback] = [LearningRateMonitor(logging_interval="epoch")]

    ema_cfg = tcfg["ema"]
    if bool(ema_cfg["enabled"]):
        callbacks.append(EMACallback(decay=float(ema_cfg["decay"])))

    if enable_ckpt:
        ckpt_dir = work_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Top-k by _MONITOR_METRIC (val_loss). It must stay the first checkpoint callback: main.py
        # publishes best.ckpt from the callback whose monitor is val_loss.
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="epoch{epoch:03d}-val{val_loss:.6f}",
                auto_insert_metric_name=False,
                monitor=_MONITOR_METRIC,
                save_top_k=save_top_k,
                mode=_MONITOR_MODE,
                every_n_epochs=ckpt_every_n_epochs,
                save_last=False,
            )
        )
        if save_last:
            # last.ckpt from its own monitor-free callback: since Lightning 2.6 the top-k
            # callback's save_last fires only in epochs that enter the top-k, so last.ckpt
            # could lag the final epoch. enable_version_counter=False overwrites last.ckpt
            # in place (never last-v1.ckpt, which a resume from last.ckpt would not see).
            callbacks.append(
                ModelCheckpoint(
                    dirpath=str(ckpt_dir),
                    filename="last",
                    monitor=None,
                    save_top_k=1,
                    every_n_epochs=LAST_CKPT_EVERY_N_EPOCHS,
                    enable_version_counter=False,
                )
            )

    ar_cfg = dict(tcfg["autoregressive_training"])
    es_cfg = tcfg["early_stopping"]
    if bool(es_cfg["enabled"]):
        # ES skips the curriculum ramp (start_epoch; 0 without one) and the LR warmup.
        es_start_epoch = _build_curriculum(tcfg).first_full_horizon_epoch() if bool(ar_cfg["enabled"]) else 0
        callbacks.append(
            CurriculumAwareEarlyStopping(
                start_epoch=es_start_epoch,
                monitor=_MONITOR_METRIC,
                mode=_MONITOR_MODE,
                patience=int(es_cfg["patience"]),
                min_delta=float(es_cfg["min_delta"]),
                check_finite=True,
                verbose=True,
            )
        )

    grad_clip = runtime["gradient_clip_val"]
    gradient_clip_val = float(grad_clip) if grad_clip is not None else None

    # Lightning raises if gradient_clip_val is set while the module uses manual
    # optimization (the autoregressive path). The module clips in-loop instead
    # (FlowMapRolloutModule._grad_clip_val), so withhold it here.
    if bool(ar_cfg["enabled"]) and gradient_clip_val is not None:
        log.info(
            "Autoregressive (manual optimization): gradient clipping (%.3g) is applied "
            "in-module; not passing gradient_clip_val to the Lightning Trainer.",
            gradient_clip_val,
        )
        gradient_clip_val = None

    strategy = runtime["strategy"]
    accumulate_grad_batches = int(runtime["accumulate_grad_batches"])

    trainer_kwargs = dict(
        default_root_dir=str(work_dir),
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=lightning_precision,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
        logger=csv_logger,
        enable_progress_bar=bool(runtime["enable_progress_bar"]),
        enable_model_summary=False,
        num_sanity_val_steps=_SANITY_VAL_BATCHES,
        enable_checkpointing=bool(enable_ckpt),
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=bool(runtime["deterministic"]),
    )

    return pl.Trainer(**trainer_kwargs)
