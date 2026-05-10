"""Exponential moving average of model weights as a Lightning callback.

The shadow EMA is updated after every training batch and is swapped into
the live module at validation time so that val_loss reflects EMA-weights
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


class EMACallback(pl.Callback):
    def __init__(self, decay: float = 0.9995) -> None:
        super().__init__()
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0,1), got {decay}")
        self.decay = float(decay)
        self.ema_sd: Optional[Dict[str, torch.Tensor]] = None
        self._stash: Optional[Dict[str, torch.Tensor]] = None

    def _init_ema_from_module(self, pl_module: pl.LightningModule) -> None:
        self.ema_sd = {k: v.detach().clone() for k, v in pl_module.state_dict().items()}

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if self.ema_sd is None:
            self._init_ema_from_module(pl_module)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Lightning's strategy moves the module to the accelerator AFTER Callback.setup()
        # runs, so the shadow cloned in setup() can live on CPU while the live module is
        # on GPU. Migrate the shadow now that the module is on its final device.
        if self.ema_sd is None:
            return
        target = next(pl_module.parameters()).device
        for k, v in list(self.ema_sd.items()):
            if v.device != target:
                self.ema_sd[k] = v.to(target)

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.ema_sd is None:
            self._init_ema_from_module(pl_module)
            return
        d = self.decay
        live = pl_module.state_dict()
        for k, ema_t in self.ema_sd.items():
            new_t = live[k]
            if ema_t.device != new_t.device:
                ema_t = ema_t.to(new_t.device)
                self.ema_sd[k] = ema_t
            if ema_t.dtype.is_floating_point:
                ema_t.mul_(d).add_(new_t.detach(), alpha=1.0 - d)
            else:
                ema_t.copy_(new_t)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.ema_sd is None:
            return
        self._stash = {k: v.detach().clone() for k, v in pl_module.state_dict().items()}
        pl_module.load_state_dict(self.ema_sd, strict=True)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._stash is None:
            return
        pl_module.load_state_dict(self._stash, strict=True)
        self._stash = None

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        if self.ema_sd is not None:
            checkpoint["ema_state_dict"] = {k: v.detach().clone() for k, v in self.ema_sd.items()}

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        ema = checkpoint.get("ema_state_dict")
        if isinstance(ema, dict) and ema:
            self.ema_sd = {k: v.detach().clone() for k, v in ema.items()}
        else:
            self._init_ema_from_module(pl_module)
