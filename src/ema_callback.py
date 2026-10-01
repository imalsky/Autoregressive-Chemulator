"""Exponential moving average of model weights as a Lightning callback.

The shadow EMA of the parameters is updated after every training batch
(buffers are copied, not averaged) and is swapped into the live module at
validation time so that val_loss reflects EMA-weights
performance. Live weights are restored at the end of validation. The EMA
state is also serialized into checkpoints under the key
``ema_state_dict`` so it can be loaded by stage-2 ``weights_only`` resumes
or by post-training inference.

Hook order matters. The callback must appear BEFORE ``ModelCheckpoint``
in the callback list so that live weights are restored before
``ModelCheckpoint.on_validation_end`` saves; EMA weights are then
attached to the saved checkpoint via ``on_save_checkpoint``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch

# The EMA update runs mul_ then add_ per group of at most ~20 MB, so the add_ pass finds the group's
# freshly written EMA tensors in L2 (96 MB on RTX 6000 Ada, 50 MB on H100) with 2 launches per group.
_GROUP_BYTES = 20_000_000
# Floating shadow tensors are kept at >= float32: at bf16, (1 - decay) * (live - ema) falls below the
# weight's resolution and the shadow never moves. float64 shadows are kept, never downcast.
EMA_SHADOW_MIN_DTYPE = torch.float32


def _shadow_copy(t: torch.Tensor) -> torch.Tensor:
    """Detached copy for the EMA shadow: floating tensors promoted to >= EMA_SHADOW_MIN_DTYPE, others cloned."""
    t = t.detach()
    if t.dtype.is_floating_point:
        return t.to(dtype=torch.promote_types(t.dtype, EMA_SHADOW_MIN_DTYPE), copy=True)
    return t.clone()


class EMACallback(pl.Callback):
    """Maintain EMA-of-weights and swap them into the module for validation/checkpointing."""

    def __init__(self, *, decay: float) -> None:
        """Store the EMA decay (must be in (0, 1)); the shadow is created in setup()."""
        super().__init__()
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0,1), got {decay}")
        self.decay = float(decay)
        self.ema_sd: Optional[Dict[str, torch.Tensor]] = None
        self._stash: Optional[Dict[str, torch.Tensor]] = None
        self._cache = None  # cached (ema, live) tensor groups; reset whenever ema_sd is replaced

    def _init_ema_from_module(self, pl_module: pl.LightningModule) -> None:
        """Initialize the EMA shadow as a detached copy of the module's current state_dict."""
        self.ema_sd = {k: _shadow_copy(v) for k, v in pl_module.state_dict().items()}
        self._cache = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Create the EMA shadow from the module.

        A resume replaces it in on_load_checkpoint, which Lightning runs after setup().
        """
        if self.ema_sd is None:
            self._init_ema_from_module(pl_module)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Migrate the EMA shadow onto the module's final device once Lightning has placed it."""
        # Lightning's strategy moves the module to the accelerator AFTER Callback.setup()
        # runs, so the shadow cloned in setup() can live on CPU while the live module is
        # on GPU. Migrate the shadow now that the module is on its final device.
        target = next(pl_module.parameters()).device
        for k, v in list(self.ema_sd.items()):
            if v.device != target:
                self.ema_sd[k] = v.to(target)
        self._cache = None

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        _outputs: Any,
        _batch: Any,
        _batch_idx: int,
    ) -> None:
        """Update the EMA shadow weights after every training batch."""
        if self._cache is None:
            live = pl_module.state_dict(keep_vars=True)  # live Parameter/buffer objects
            groups, other, size = [], [], 0
            for k, ema_t in self.ema_sd.items():
                # Only parameters are averaged. Buffers (loss statistics, Fourier frequencies, any
                # non-float entry) are copied: averaging a constant drifts it (d*x + (1-d)*x != x
                # in floating point).
                if not (isinstance(live[k], torch.nn.Parameter) and ema_t.dtype.is_floating_point):
                    other.append((ema_t, live[k]))
                    continue
                nbytes = ema_t.numel() * ema_t.element_size()
                if not groups or size + nbytes > _GROUP_BYTES:
                    groups.append(([], []))
                    size = 0
                groups[-1][0].append(ema_t)
                groups[-1][1].append(live[k])
                size += nbytes
            self._cache = (groups, other)
        groups, other = self._cache
        for ema_g, live_g in groups:
            torch._foreach_mul_(ema_g, self.decay)
            torch._foreach_add_(ema_g, live_g, alpha=1.0 - self.decay)
        for ema_t, new_t in other:
            ema_t.copy_(new_t)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Stash live weights and load the EMA weights so val_loss reflects EMA performance."""
        self._stash = {k: v.detach().clone() for k, v in pl_module.state_dict().items()}
        pl_module.load_state_dict(self.ema_sd, strict=True)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Restore the stashed live weights after validation."""
        pl_module.load_state_dict(self._stash, strict=True)
        self._stash = None

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Serialize the EMA shadow into the checkpoint under 'ema_state_dict'."""
        checkpoint["ema_state_dict"] = {k: v.detach().clone() for k, v in self.ema_sd.items()}

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Restore the EMA shadow from the checkpoint, or re-initialize it if absent."""
        ema = checkpoint.get("ema_state_dict")
        if isinstance(ema, dict) and ema:
            self.ema_sd = {k: _shadow_copy(v) for k, v in ema.items()}
        else:
            self._init_ema_from_module(pl_module)
        self._cache = None
