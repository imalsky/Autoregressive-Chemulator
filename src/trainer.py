#!/usr/bin/env python3
"""
trainer.py - LightningModule + Trainer factory.

Training styles (burn-in removed):

1) One-jump
   - training.rollout_steps must be 1
   - loss on the first transition only

2) Autoregressive rollout
   - Total rollout steps K := training.rollout_steps
   - Skip/warmup steps M := training.autoregressive_training.skip_steps
       * warmup is run under torch.no_grad(), then detached (pushforward trick input generation)
       * warmup steps are excluded from the loss via masking (and/or by construction in detached-per-step mode)
   - Gradients:
       * detach_between_steps=True  : stop-grad between steps (no BPTT across time)
       * detach_between_steps=False : BPTT across steps [M..K-1] (but never through warmup)

Optional curriculum (training only):
- training.curriculum ramps training rollout steps from start_steps -> rollout_steps.
- Validation/test always use rollout_steps (fixed horizon for comparability).

Logging:
- Uses Lightning CSVLogger. All self.log(...) metrics are written to metrics.csv.
- LearningRateMonitor logs LR automatically.
- epoch_time_sec is logged manually (so we do not need Timer).

Batch contract (from dataset.py):
- y:  [B, K_full, S]   (states in z-space)
- dt: [B, K_full-1]    (per-transition normalized dt)
- g:  [B, G]           (G = 2 mandatory: P, T)

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

from utils import PrecisionConfig, parse_precision_config
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from ema_callback import EMACallback
from lightning.pytorch.loggers import CSVLogger
from lightning_utilities.core.rank_zero import rank_zero_warn


_LOSS_DENOM_EPS = 1e-3
_LOG_GRAD_NORM_DEFAULT = False
# Uniform per-step rollout weighting (no discount). Only consulted when the
# optional training.autoregressive_training.loss_discount_gamma key is absent.
_DEFAULT_LOSS_DISCOUNT_GAMMA = 1.0
log = logging.getLogger(__name__)

# ==============================================================================
# Optimizer parameter groups
# ==============================================================================

_NORM_MODULE_TYPES = (
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)
if hasattr(nn, "SyncBatchNorm"):
    _NORM_MODULE_TYPES = _NORM_MODULE_TYPES + (nn.SyncBatchNorm,)


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


class WarmupReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """ReduceLROnPlateau with linear warmup over optimizer steps."""

    def __init__(self, optimizer: torch.optim.Optimizer, *, warmup_steps: int, **kwargs: Any) -> None:
        """Capture base LRs and start at LR=0 if warming up; kwargs pass through to ReduceLROnPlateau."""
        self.warmup_steps = int(max(0, warmup_steps))
        self.warmup_step_count = 0
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        super().__init__(optimizer, **kwargs)
        if self.warmup_steps > 0:
            self._set_warmup_lrs(step=0)

    @property
    def warmup_active(self) -> bool:
        """True while still inside the linear warmup ramp."""
        return self.warmup_step_count < self.warmup_steps

    def _set_warmup_lrs(self, *, step: int) -> None:
        """Set each param group's LR to base_lr * (step / warmup_steps)."""
        scale = float(step) / float(max(1, self.warmup_steps))
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = float(base_lr) * scale
        self._last_lr = [float(group["lr"]) for group in self.optimizer.param_groups]

    def step_warmup(self) -> None:
        """Advance the warmup ramp by one optimizer step (no-op once warmup is complete)."""
        if not self.warmup_active:
            return
        self.warmup_step_count += 1
        self._set_warmup_lrs(step=self.warmup_step_count)

    def step(self, metrics: Any, epoch: Optional[int] = None) -> None:
        """Plateau step on the monitored metric; suppressed while warmup is still active."""
        if self.warmup_active:
            return
        super().step(metrics, epoch=epoch)


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
            denom = (w.sum() * float(B * S)).clamp_min(_LOSS_DENOM_EPS)

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


def _try_torch_compile(fn: Any, *, cfg: Mapping[str, Any], context: str) -> Any:
    """Optionally wrap a callable with torch.compile (strict; required keys only)."""
    runtime = cfg["runtime"]
    tc = runtime["torch_compile"]
    if not bool(tc["enabled"]):
        return fn

    if not hasattr(torch, "compile"):
        raise RuntimeError(f"runtime.torch_compile.enabled=True but torch.compile is unavailable ({context}).")

    backend = str(tc["backend"])
    mode = str(tc["mode"])
    dynamic = bool(tc["dynamic"])
    fullgraph = bool(tc["fullgraph"])
    accelerator = str(runtime["accelerator"]).lower().strip()

    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    targeting_mps = accelerator == "mps" or (accelerator == "auto" and mps_available and not torch.cuda.is_available())
    if backend == "inductor" and targeting_mps:
        log.info("Skipping torch.compile for %s (inductor backend on MPS).", context)
        return fn

    try:
        return torch.compile(fn, backend=backend, mode=mode, dynamic=dynamic, fullgraph=fullgraph)  # type: ignore[misc]
    except Exception as e:
        raise RuntimeError(f"torch.compile failed for {context}: {e}") from e


# ==============================================================================
# Curriculum
# ==============================================================================


@dataclass(frozen=True)
class RolloutCurriculum:
    """Ramps training rollout steps from start_steps -> end_steps over ramp_epochs."""

    enabled: bool
    mode: str
    start_steps: int
    end_steps: int
    ramp_epochs: int

    def steps(self, epoch: int) -> int:
        """Return the rollout horizon for `epoch`, ramping start_steps -> end_steps over ramp_epochs."""
        end_k = int(self.end_steps)
        if not self.enabled:
            return end_k

        start_k = int(self.start_steps)
        if self.ramp_epochs <= 0 or start_k >= end_k:
            return start_k

        e = int(max(0, epoch))
        t = min(1.0, float(e) / float(max(1, int(self.ramp_epochs))))
        mode = str(self.mode).lower()

        if mode == "linear":
            f = t
        elif mode == "cosine":
            f = 0.5 * (1.0 - math.cos(math.pi * t))
        else:
            raise ValueError(f"Unknown curriculum.mode '{self.mode}' (expected: linear|cosine)")

        k = start_k + int((end_k - start_k) * f)
        return int(min(end_k, k))


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
        precision: Optional[PrecisionConfig] = None,
    ) -> None:
        """Wire the model, loss, precision, curriculum, AR settings, and optional torch.compile from cfg."""
        super().__init__()
        self.model = model
        self.normalization_manifest = dict(normalization_manifest)
        self.species_variables = list(species_variables)

        # Precision (centralized): controls dataset/model/loss dtypes and Lightning AMP settings.
        self.precision_config = precision if precision is not None else parse_precision_config(cfg)
        self.model_dtype = self.precision_config.model_dtype
        self.input_dtype = self.precision_config.input_dtype
        self.loss_dtype = self.precision_config.loss_dtype

        tcfg = dict(cfg["training"])
        mcfg = dict(cfg["model"])

        self.max_epochs = int(tcfg["max_epochs"])
        self.rollout_steps = int(tcfg["rollout_steps"])
        if self.rollout_steps < 1:
            raise ValueError(f"training.rollout_steps must be >= 1, got {self.rollout_steps}")

        # The full autoregressive_training block is schema-required (enabled, skip_steps,
        # detach_between_steps, backward_per_step); read them directly. When AR is disabled
        # these values are inert (the one-jump training path ignores them).
        ar_cfg = dict(tcfg["autoregressive_training"])
        self.autoregressive_training = bool(ar_cfg["enabled"])
        self.ar_skip_steps = int(ar_cfg["skip_steps"])
        if self.ar_skip_steps < 0:
            raise ValueError(f"training.autoregressive_training.skip_steps must be >= 0, got {self.ar_skip_steps}")
        self.ar_detach_between_steps = bool(ar_cfg["detach_between_steps"])
        self.ar_backward_per_step = bool(ar_cfg["backward_per_step"])
        # Per-step discount gamma^(k-skip) on the rollout loss. 1.0 = uniform (legacy).
        # Optional key; validation/checkpoint selection (_eval_step) intentionally stays
        # uniformly weighted so val_loss is comparable across gamma settings.
        self.ar_loss_discount_gamma = float(ar_cfg.get("loss_discount_gamma", _DEFAULT_LOSS_DISCOUNT_GAMMA))
        if not (0.0 < self.ar_loss_discount_gamma <= 1.0):
            raise ValueError(
                "training.autoregressive_training.loss_discount_gamma must be in (0, 1], "
                f"got {self.ar_loss_discount_gamma}"
            )
        if not self.autoregressive_training and self.ar_loss_discount_gamma != 1.0:
            raise ValueError(
                "training.autoregressive_training.loss_discount_gamma has no effect when "
                "autoregressive_training.enabled=false; remove it or set it to 1.0."
            )

        # Manual-optimization gradient clipping: Lightning forbids Trainer(gradient_clip_val=...)
        # with manual optimization, so in AR mode the module clips via this value instead
        # (build_lightning_trainer withholds it from the Trainer in that case).
        runtime_cfg = dict(cfg["runtime"])
        clip_raw = runtime_cfg["gradient_clip_val"]  # required; may be 0.0 (disabled)
        self._manual_grad_clip_val: Optional[float] = float(clip_raw) if clip_raw is not None else None

        if not self.autoregressive_training and self.rollout_steps != 1:
            raise ValueError(
                "One-jump training requires training.rollout_steps == 1. "
                f"Got rollout_steps={self.rollout_steps} with autoregressive_training.enabled=False."
            )

        if self.autoregressive_training:
            self.automatic_optimization = False

        # Curriculum (training only). The full block is schema-required; read directly.
        cur_cfg = dict(tcfg["curriculum"])
        self.curriculum = RolloutCurriculum(
            enabled=bool(cur_cfg["enabled"]),
            mode=str(cur_cfg["mode"]),
            start_steps=int(cur_cfg["start_steps"]),
            end_steps=int(cur_cfg["end_steps"]),
            ramp_epochs=int(cur_cfg["ramp_epochs"]),
        )

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

        self.sched_cfg = dict(tcfg["scheduler"])

        # Validate model interface once at init (both FlowMapMLP and FlowMapAutoencoder provide this).
        if not hasattr(self.model, "forward_step"):
            raise RuntimeError("Model must implement forward_step(y_t, dt, g) -> y_{t+1}.")

        # Optional compilation (torch_compile block is schema-required).
        runtime = dict(cfg["runtime"])
        tc = dict(runtime["torch_compile"])
        # log_grad_norm is an optional perf/debug knob (named-constant default).
        self.log_grad_norm = bool(runtime.get("log_grad_norm", _LOG_GRAD_NORM_DEFAULT))
        self._compiled_forward_step: Optional[Any] = None
        self._compiled_open_loop_unroll: Optional[Any] = None
        if bool(tc["enabled"]):
            if bool(tc["compile_forward_step"]):
                self._compiled_forward_step = _try_torch_compile(
                    self.model.forward_step, cfg=cfg, context="FlowMapRolloutModule.model.forward_step"  # type: ignore[attr-defined]
                )
            if bool(tc["compile_open_loop_unroll"]):
                self._compiled_open_loop_unroll = _try_torch_compile(
                    self._open_loop_unroll_raw_step, cfg=cfg, context="FlowMapRolloutModule._open_loop_unroll_raw_step"
                )

        # Dataloader wiring (kept for compatibility; main.py can also pass dataloaders directly)
        self._train_dl = None
        self._val_dl = None
        self._test_dl = None

        # Epoch timing (logged; CSVLogger writes it)
        self._epoch_t0: Optional[float] = None

        # UX warnings
        self._warned_metric_mismatch = False

        # Preserve for checkpoint reproducibility. Deep-copy so later config mutations
        # (or Lightning internals) can't alias the live training config.
        self.save_hyperparameters({"training": copy.deepcopy(dict(tcfg)), "model": copy.deepcopy(dict(mcfg))})

    # ------------------------
    # Dataloader wiring
    # ------------------------

    def train_dataloader(self) -> Any:
        if self._train_dl is None:
            raise RuntimeError("train_dataloader not set. Pass dataloaders to Trainer.fit(...).")
        return self._train_dl

    def val_dataloader(self) -> Any:
        return self._val_dl

    def test_dataloader(self) -> Any:
        return self._test_dl

    # ------------------------
    # Fit-time warning about metric comparability
    # ------------------------

    def on_fit_start(self) -> None:
        """Log a one-time note that train_loss and val_loss use different rollout procedures."""
        if self._warned_metric_mismatch:
            return
        self._warned_metric_mismatch = True

        if self.autoregressive_training:
            msg = (
                "train_loss (autoregressive training) may not be directly comparable to val_loss/test_loss "
                "(open-loop rollout evaluation), especially when using detach_between_steps=True and/or a rollout curriculum. "
                "Use the same split/stage for apples-to-apples comparisons, and log train_rollout_steps to track curriculum."
            )
        else:
            msg = (
                "train_loss (one-jump training) is not directly comparable to val_loss/test_loss (open-loop rollout evaluation). "
                "Compare within the same rollout procedure."
            )
        log.info(msg)

    # ------------------------
    # Epoch timing
    # ------------------------

    def on_train_epoch_start(self) -> None:
        self._epoch_t0 = time.time()

    def on_train_epoch_end(self) -> None:
        if self._epoch_t0 is not None:
            self.log("epoch_time_sec", float(time.time() - self._epoch_t0), on_step=False, on_epoch=True)

    # ------------------------
    # Forward utilities
    # ------------------------

    def _forward_step(self, y_t: torch.Tensor, dt: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Single model step y_t -> y_{t+1}, using the compiled step when available."""
        fn = self._compiled_forward_step if self._compiled_forward_step is not None else self.model.forward_step  # type: ignore[attr-defined]
        return fn(y_t, dt, g)  # type: ignore[misc]

    def _coerce_g(self, g: torch.Tensor, batch_size: int, like: torch.Tensor) -> torch.Tensor:
        """Broadcast globals g to [batch_size, G] and move to `like`'s device/dtype (strict shape checks)."""
        if g.ndim == 1:
            g = g.view(1, -1).expand(batch_size, -1)
        elif g.ndim == 2:
            if int(g.shape[0]) == 1 and batch_size > 1:
                g = g.expand(batch_size, -1)
            elif int(g.shape[0]) != batch_size:
                raise ValueError(f"g batch dim mismatch: got {int(g.shape[0])}, expected {batch_size}")
        else:
            raise ValueError(f"g must have shape [G] or [B,G], got {tuple(g.shape)}")

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
        gamma: float = 1.0,
    ) -> Optional[torch.Tensor]:
        """Per-step loss weights: 0 for warmup steps, gamma^(k-skip) after.

        Returns None (uniform, no masking) when there is nothing to weight —
        preserving the legacy fast path.
        """
        skip = int(skip_steps)
        g = float(gamma)
        if skip <= 0 and g == 1.0:
            return None
        w = torch.ones((int(k),), device=device, dtype=torch.float32)
        if skip > 0:
            w[: min(skip, int(k))] = 0.0
        if g != 1.0:
            offsets = torch.arange(int(k), device=device, dtype=torch.float32) - float(skip)
            w = w * torch.pow(torch.tensor(g, device=device), offsets.clamp_min(0.0))
        return w

    def _open_loop_unroll(
        self,
        *,
        y0: torch.Tensor,     # [B,S]
        dt_bk: torch.Tensor,  # [B,K]
        g: torch.Tensor,      # [B,G]
    ) -> torch.Tensor:
        """Open-loop rollout: feed predictions back in (no teacher forcing)."""
        g_t = self._coerce_g(g, int(dt_bk.shape[0]), y0)
        step_fn = self._compiled_forward_step if self._compiled_forward_step is not None else self.model.forward_step  # type: ignore[attr-defined]
        return autoregressive_rollout(step_fn, y0, dt_bk, g_t)

    def _open_loop_unroll_raw_step(
        self,
        *,
        y0: torch.Tensor,     # [B,S]
        dt_bk: torch.Tensor,  # [B,K]
        g: torch.Tensor,      # [B,G]
    ) -> torch.Tensor:
        """Open-loop rollout using the raw (uncompiled) model step.

        This is used when compiling the unroll itself (compile_open_loop_unroll=True) so the
        compiler can see the full step-to-step dependency chain and fuse across steps.
        """
        g_t = self._coerce_g(g, int(dt_bk.shape[0]), y0)
        return autoregressive_rollout(self.model.forward_step, y0, dt_bk, g_t)  # type: ignore[attr-defined]

    # ------------------------
    # Manual optimization helpers (autoregressive mode)
    # ------------------------

    def _get_single_optimizer(self) -> torch.optim.Optimizer:
        """Return the single optimizer (this project never configures more than one)."""
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            if len(opt) != 1:
                raise RuntimeError("Expected exactly one optimizer.")
            return opt[0]
        return opt

    def _accumulate_grad_batches(self) -> int:
        """Return the trainer's accumulate_grad_batches as an int >= 1 (handles the schedule-dict form)."""
        acc = getattr(self.trainer, "accumulate_grad_batches", 1)
        if isinstance(acc, Mapping):
            acc = int(acc.get(0, list(acc.values())[0]))
        return int(max(1, int(acc)))

    def _maybe_step_optimizer(self, opt: torch.optim.Optimizer, batch_idx: int) -> None:
        """Step the optimizer on accumulation boundaries: unscale, optional grad-norm log, clip, step, zero.

        Manual-optimization path only (autoregressive mode). Also advances non-plateau schedulers
        and the warmup ramp; plateau schedulers are stepped on validation end instead.
        """
        acc = self._accumulate_grad_batches()
        n_batches = getattr(self.trainer, "num_training_batches", batch_idx + 1)
        # num_training_batches can be float('inf') for unsized iterables; fall back to accumulation-only stepping.
        try:
            n_batches_f = float(n_batches)
        except (TypeError, ValueError):
            n_batches_f = float("inf")
        if math.isfinite(n_batches_f):
            is_last = (batch_idx + 1) >= int(n_batches_f)
        else:
            is_last = False
        should_step = (((batch_idx + 1) % acc) == 0) or is_last
        if not should_step:
            return

        # Use the underlying torch optimizer for inspection/unscale when available.
        torch_opt = opt.optimizer if hasattr(opt, "optimizer") else opt

        # Unscale gradients so logged grad_norm is in true scale.
        # Only needed when a GradScaler is active (fp16 AMP); bf16/fp32 do not use one.
        _plugin = getattr(self.trainer.strategy, "precision_plugin", None) or getattr(self.trainer, "precision_plugin", None)
        if _plugin is not None and hasattr(_plugin, "unscale_gradients"):
            _plugin.unscale_gradients(torch_opt)

        if self.log_grad_norm:
            grads = [p.grad.detach() for pg in torch_opt.param_groups for p in pg["params"] if p.grad is not None]
            if grads:
                # torch._foreach_norm is a single fused kernel over the full list, much cheaper than a Python loop.
                per_tensor = torch._foreach_norm(grads, 2)  # type: ignore[attr-defined]
                gn2 = torch.zeros((), device=self.device, dtype=torch.float32)
                for n in per_tensor:
                    nf = n.float()
                    gn2 = gn2 + nf * nf
                self.log("grad_norm", torch.sqrt(gn2), on_step=False, on_epoch=True)

        # Manual-optimization path: clip from the module's own config value (the Trainer
        # is constructed WITHOUT gradient_clip_val in AR mode — Lightning forbids it).
        clip_val = self._manual_grad_clip_val
        if clip_val is not None and float(clip_val) > 0:
            self.clip_gradients(
                opt,
                gradient_clip_val=float(clip_val),
                gradient_clip_algorithm="norm",
            )

        opt.step()
        opt.zero_grad(set_to_none=True)

        # Step non-plateau schedulers here; plateau is stepped on validation end.
        sched = self.lr_schedulers()
        if sched is None:
            return
        sched_list = list(sched) if isinstance(sched, (list, tuple)) else [sched]
        for s in sched_list:
            sch = s.scheduler if hasattr(s, "scheduler") else s
            if isinstance(sch, WarmupReduceLROnPlateau):
                sch.step_warmup()
                continue
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                continue
            sch.step()

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        """Automatic-optimization hook: after the normal step, hand-advance the warmup ramp only."""
        # Automatic-optimization path: Lightning already drives step/epoch LR schedulers
        # (cosine LambdaLR with interval="step", plateau on validation_end). The only
        # scheduler we must hand-advance is the custom WarmupReduceLROnPlateau's
        # per-step warmup ramp, which is not driven by either of those mechanisms.
        super().optimizer_step(*args, **kwargs)
        if not self.automatic_optimization:
            return

        sched = self.lr_schedulers()
        if sched is None:
            return

        sched_list = list(sched) if isinstance(sched, (list, tuple)) else [sched]
        for s in sched_list:
            sch = s.scheduler if hasattr(s, "scheduler") else s
            if isinstance(sch, WarmupReduceLROnPlateau):
                sch.step_warmup()

    def _scheduler_steps_per_epoch(self) -> int:
        """Estimate optimizer steps per epoch (batches / accumulation), for sizing the warmup ramp."""
        if self.trainer is None:
            raise RuntimeError("Trainer must be attached before configure_optimizers for scheduler warmup.")

        num_training_batches = getattr(self.trainer, "num_training_batches", None)
        try:
            num_batches = float(num_training_batches)
        except (TypeError, ValueError):
            num_batches = 0.0

        if math.isfinite(num_batches) and num_batches > 0.0:
            acc = self._accumulate_grad_batches()
            return int(max(1, math.ceil(num_batches / float(acc))))

        max_steps = int(self.trainer.estimated_stepping_batches)
        if max_steps <= 0:
            raise RuntimeError(f"estimated_stepping_batches={max_steps}; cannot build scheduler warmup.")

        max_epochs = max(1, int(self.max_epochs))
        return int(max(1, math.ceil(float(max_steps) / float(max_epochs))))

    def _scheduler_warmup_steps(self, warmup_epochs: int) -> int:
        """Convert a warmup duration in epochs into optimizer steps, clamped to [0, max_steps]."""
        if warmup_epochs <= 0:
            return 0
        if self.trainer is None:
            raise RuntimeError("Trainer must be attached before configure_optimizers for scheduler warmup.")

        max_steps = int(self.trainer.estimated_stepping_batches)
        if max_steps <= 0:
            raise RuntimeError(f"estimated_stepping_batches={max_steps}; cannot build scheduler warmup.")

        steps_per_epoch = self._scheduler_steps_per_epoch()
        warmup_steps = int(warmup_epochs) * max(1, steps_per_epoch)
        return int(max(0, min(warmup_steps, max_steps)))


    # ------------------------
    # Training / eval
    # ------------------------

    def _training_step_one_jump(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """One-jump training: single transition y0 -> y1, loss on that one step (automatic optimization)."""
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
        g_t = self._coerce_g(g, B, y0)

        y_pred1 = self._forward_step(y0, dt0, g_t)
        losses = self.criterion(y_pred1.unsqueeze(1), y1.unsqueeze(1), step_weights=None)

        self.log("train_loss", losses["loss_total"].detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_log10_mae", losses["loss_log10_mae"], on_step=False, on_epoch=True)
        self.log("train_loss_z_mse", losses["loss_z_mse"], on_step=False, on_epoch=True)

        self.log("train_rollout_steps", float(1.0), on_step=False, on_epoch=True)

        self.log("train_skip_steps", float(0.0), on_step=False, on_epoch=True)
        self.log("train_detach_between_steps", float(1.0), on_step=False, on_epoch=True)

        return losses["loss_total"]


    def _training_step_autoregressive(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Autoregressive rollout training (manual optimization).

        Runs skip_steps warmup under no_grad (pushforward), then trains steps [skip..k_train) either
        with stop-grad-per-step (detached) or full BPTT, applying the gamma^(k-skip) per-step weights.
        Curriculum sets k_train per epoch. Handles its own backward/step via _maybe_step_optimizer.
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
        y0 = y[:, 0, :]
        g_t = self._coerce_g(g, B, y0)

        skip = int(self.ar_skip_steps)
        if skip >= k_train:
            raise ValueError(f"skip_steps ({skip}) must be < rollout_steps ({k_train}).")

        opt = self._get_single_optimizer()
        acc = self._accumulate_grad_batches()
        if (batch_idx % acc) == 0:
            opt.zero_grad(set_to_none=True)

        # Warmup once: pushforward M steps without grad, then detach.
        y_prev = y0
        if skip > 0:
            with torch.no_grad():
                for k in range(skip):
                    y_prev = self._forward_step(y_prev, dt_train[:, k], g_t)
            y_prev = y_prev.detach()

        gamma = float(self.ar_loss_discount_gamma)

        if self.ar_detach_between_steps:
            # Stop-grad between steps: no BPTT across time (but still trained on pushforward states).
            # Per-step weights gamma^(k-skip); gamma=1.0 reproduces the legacy uniform 1/num_steps.
            weights = [gamma ** (k - skip) for k in range(skip, k_train)]
            weight_sum = float(sum(weights))

            loss_total_sum = y_prev.new_zeros(())
            log10_mae_sum = y_prev.new_zeros(())
            z_mse_sum = y_prev.new_zeros(())

            for k in range(skip, k_train):
                y_next = self._forward_step(y_prev.detach(), dt_train[:, k], g_t)
                y_tgt = y_true[:, k, :]

                step_losses = self.criterion(y_next.unsqueeze(1), y_tgt.unsqueeze(1), step_weights=None)
                w_k = weights[k - skip]

                if self.ar_backward_per_step:
                    self.manual_backward(step_losses["loss_total"] * (w_k / (weight_sum * acc)))
                else:
                    loss_total_sum = loss_total_sum + step_losses["loss_total"] * w_k

                log10_mae_sum = log10_mae_sum + step_losses["loss_log10_mae"].detach() * w_k
                z_mse_sum = z_mse_sum + step_losses["loss_z_mse"].detach() * w_k

                y_prev = y_next.detach()

            if not self.ar_backward_per_step:
                self.manual_backward(loss_total_sum / (weight_sum * acc))

            self._maybe_step_optimizer(opt, batch_idx)

            loss_log10_mae = log10_mae_sum / weight_sum
            loss_z_mse = z_mse_sum / weight_sum
            loss_total = self.criterion.lambda_log10_mae * loss_log10_mae + self.criterion.lambda_z_mse * loss_z_mse

        else:
            # BPTT across the trained portion (after warmup). No gradients through warmup.
            y_pred = y0.new_zeros((B, k_train, S))

            for k in range(skip, k_train):
                y_prev = self._forward_step(y_prev, dt_train[:, k], g_t)
                y_pred[:, k, :] = y_prev

            step_weights = self._step_weights(k_train, skip, device=y_pred.device, gamma=gamma)
            losses = self.criterion(y_pred, y_true, step_weights=step_weights)

            loss_total = losses["loss_total"]
            loss_log10_mae = losses["loss_log10_mae"].detach()
            loss_z_mse = losses["loss_z_mse"].detach()

            self.manual_backward(loss_total / float(acc))
            self._maybe_step_optimizer(opt, batch_idx)

        self.log("train_loss", loss_total.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_log10_mae", loss_log10_mae, on_step=False, on_epoch=True)
        self.log("train_loss_z_mse", loss_z_mse, on_step=False, on_epoch=True)
        self.log("train_rollout_steps", float(k_train), on_step=False, on_epoch=True)
        self.log("train_skip_steps", float(skip), on_step=False, on_epoch=True)
        self.log("train_detach_between_steps", float(self.ar_detach_between_steps), on_step=False, on_epoch=True)

        return loss_total

    def _eval_step(self, batch: Mapping[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Open-loop rollout evaluation for val/test: unroll k_eval steps and log the (uniformly weighted) loss."""
        y = batch["y"]
        dt = batch["dt"]
        g = batch["g"]
        y, dt, g = self._cast_batch_tensors(y, dt, g)

        B, K_full, _ = y.shape
        transitions = int(K_full) - 1
        k_eval = self._eval_rollout_steps(transitions=transitions)

        dt_full = normalize_dt_shape(dt, batch_size=B, seq_len=transitions, context=f"{stage}/eval")
        dt_eval = dt_full[:, :k_eval]

        y0 = y[:, 0, :]
        y_true = y[:, 1 : 1 + k_eval, :]

        fn = self._compiled_open_loop_unroll if self._compiled_open_loop_unroll is not None else self._open_loop_unroll
        y_pred = fn(y0=y0, dt_bk=dt_eval, g=g)

        # Keep loss definition consistent by masking skip_steps the same way in val/test.
        step_weights = None
        if self.autoregressive_training:
            step_weights = self._step_weights(k_eval, self.ar_skip_steps, device=y_pred.device)

        losses = self.criterion(y_pred, y_true, step_weights=step_weights)
        loss_total = losses["loss_total"]

        if stage == "val":
            self.log("val_loss", loss_total, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_loss_log10_mae", losses["loss_log10_mae"], on_step=False, on_epoch=True)
            self.log("val_loss_z_mse", losses["loss_z_mse"], on_step=False, on_epoch=True)
        elif stage == "test":
            self.log("test_loss", loss_total, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_loss_log10_mae", losses["loss_log10_mae"], on_step=False, on_epoch=True)
            self.log("test_loss_z_mse", losses["loss_z_mse"], on_step=False, on_epoch=True)
        else:
            raise ValueError(f"Unknown eval stage: {stage}")

        return loss_total

    def training_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Dispatch to the autoregressive or one-jump training step based on config."""
        if self.autoregressive_training:
            return self._training_step_autoregressive(batch, batch_idx)
        return self._training_step_one_jump(batch)

    def validation_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Open-loop rollout validation step (logs val_loss)."""
        return self._eval_step(batch, stage="val")

    def test_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Open-loop rollout test step (logs test_loss)."""
        return self._eval_step(batch, stage="test")

    def on_validation_epoch_end(self) -> None:
        """Step ReduceLROnPlateau schedulers when using manual optimization."""
        if not self.autoregressive_training:
            return
        if not bool(self.sched_cfg["enabled"]):
            return

        metrics = getattr(self.trainer, "callback_metrics", {}) or {}
        # configure_optimizers enforces scheduler.monitor == "val_loss" for the plateau path.
        monitor = "val_loss"
        metric = metrics.get(monitor, None)
        if metric is None:
            return

        metric_val = float(metric.detach().cpu().item()) if isinstance(metric, torch.Tensor) else float(metric)

        sched = self.lr_schedulers()
        if sched is None:
            return

        sched_list = list(sched) if isinstance(sched, (list, tuple)) else [sched]
        for s in sched_list:
            sch = s.scheduler if hasattr(s, "scheduler") else s
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step(metric_val)


    # ------------------------
    # Optimizer / scheduler
    # ------------------------

    def configure_optimizers(self) -> Any:
        """Build the AdamW optimizer (with norm/bias WD exclusion) and the configured LR scheduler.

        Supports reduce_on_plateau (wrapped with linear warmup) and cosine_with_warmup. Returns the
        Lightning optimizer/scheduler dict; scheduler.monitor is required to be "val_loss".
        """
        tcfg = dict(self.hparams["training"])
        opt_cfg = dict(tcfg["optimizer"])

        name = str(opt_cfg["name"]).lower()
        lr = float(opt_cfg["lr"])
        weight_decay = float(opt_cfg["weight_decay"])
        exclude_nd = bool(opt_cfg["exclude_norm_and_bias_from_weight_decay"])
        param_groups = _build_optimizer_param_groups(self, weight_decay=weight_decay, exclude_norm_and_bias=exclude_nd)

        if name != "adamw":
            raise ValueError("Unsupported optimizer; only 'adamw' is allowed")

        betas = opt_cfg["betas"]
        b0, b1 = float(betas[0]), float(betas[1])
        eps = float(opt_cfg["eps"])

        use_fused = bool(opt_cfg["fused"])
        use_foreach = bool(opt_cfg["foreach"])
        if use_fused and use_foreach:
            raise ValueError("optimizer config invalid: fused and foreach cannot both be true")

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

        sched_type_raw = str(sched_cfg["type"]).strip()
        sched_type = sched_type_raw.lower()

        if sched_type in {"reducelronplateau", "reduce_on_plateau"}:
            factor = float(sched_cfg["factor"])
            patience = int(sched_cfg["patience"])
            threshold = float(sched_cfg["threshold"])
            min_lr = float(sched_cfg["min_lr"])
            mode = str(sched_cfg["mode"])
            monitor = str(sched_cfg["monitor"])
            warmup_steps = self._scheduler_warmup_steps(int(sched_cfg["warmup_epochs"]))
            if monitor != "val_loss":
                raise ValueError("scheduler.monitor must be 'val_loss'")

            sched = WarmupReduceLROnPlateau(
                opt,
                warmup_steps=warmup_steps,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
                min_lr=min_lr,
            )

            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": monitor,
                    "name": "lr",
                },
            }

        if sched_type != "cosine_with_warmup":
            raise ValueError(
                f"Unsupported scheduler type '{sched_type_raw}' "
                "(supported: reduce_on_plateau, ReduceLROnPlateau, cosine_with_warmup)"
            )

        # Cosine schedule with optional warmup (step-based).
        if self.trainer is None:
            raise RuntimeError("Trainer must be attached before configure_optimizers for cosine scheduler.")
        max_steps = int(self.trainer.estimated_stepping_batches)
        if max_steps <= 0:
            raise RuntimeError(f"estimated_stepping_batches={max_steps}; cannot build cosine schedule.")

        warmup_epochs = int(sched_cfg["warmup_epochs"])
        min_lr_ratio = float(sched_cfg["min_lr_ratio"])
        warmup_steps = self._scheduler_warmup_steps(warmup_epochs)

        def lr_lambda(step: int) -> float:
            s = int(step)
            if warmup_steps > 0 and s < warmup_steps:
                return float(s) / float(max(1, warmup_steps))

            if max_steps <= warmup_steps:
                return float(min_lr_ratio)

            progress = float(s - warmup_steps) / float(max(1, max_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            },
        }


# ==============================================================================
# Trainer factory
# ==============================================================================


def build_lightning_trainer(
    cfg: Mapping[str, Any],
    *,
    work_dir: Path,
    precision_config: Optional[PrecisionConfig] = None,
    train_batches_per_epoch: Optional[int] = None,
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
    prec = precision_config if precision_config is not None else parse_precision_config(cfg)
    lightning_precision = prec.lightning_precision
    log_every_n_steps = int(runtime["log_every_n_steps"])
    if log_every_n_steps <= 0:
        raise ValueError("runtime.log_every_n_steps must be > 0")
    if train_batches_per_epoch is not None:
        if int(train_batches_per_epoch) <= 0:
            raise ValueError("train_batches_per_epoch must be > 0 when provided")
        log_every_n_steps = min(log_every_n_steps, int(train_batches_per_epoch))

    ckpt_cfg = dict(runtime["checkpointing"])
    enable_ckpt = bool(ckpt_cfg["enabled"])
    ckpt_every_n_epochs = int(ckpt_cfg["every_n_epochs"])
    save_top_k = int(ckpt_cfg["save_top_k"])
    save_last = bool(ckpt_cfg["save_last"])
    monitor = str(ckpt_cfg["monitor"])
    if monitor != "val_loss":
        raise ValueError("runtime.checkpointing.monitor must be 'val_loss'")

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # CSVLogger: keep outputs rooted directly under work_dir.
    csv_logger = CSVLogger(save_dir=str(work_dir), name=".", version="")

    # Warn if Lightning still chooses a subdirectory (best-effort; does not crash training).
    try:
        log_dir = Path(csv_logger.log_dir).resolve()
        if log_dir != work_dir.resolve():
            rank_zero_warn(f"CSVLogger log_dir is {log_dir}, not {work_dir}. metrics.csv will be written under log_dir.")
    except Exception:
        pass

    callbacks: List[pl.Callback] = [LearningRateMonitor(logging_interval="epoch")]

    ema_cfg = tcfg["ema"]
    if bool(ema_cfg["enabled"]):
        callbacks.append(EMACallback(decay=float(ema_cfg["decay"])))

    if enable_ckpt:
        ckpt_dir = work_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="epoch{epoch:03d}-val{val_loss:.6f}",
                monitor=monitor,
                save_top_k=save_top_k,
                mode="min",
                every_n_epochs=ckpt_every_n_epochs,
                save_last=save_last,
            )
        )

    es_cfg = tcfg["early_stopping"]
    if bool(es_cfg["enabled"]):
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
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
    # (FlowMapRolloutModule._manual_grad_clip_val), so withhold it here.
    ar_cfg = dict(tcfg["autoregressive_training"])
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
        num_sanity_val_steps=0,
        enable_checkpointing=bool(enable_ckpt),
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=bool(runtime["deterministic"]),
    )

    return pl.Trainer(**trainer_kwargs)
