#!/usr/bin/env python3
"""
model.py - The relaxation head for flow-map rollout prediction.

This module defines the neural network used by the trainer. The core task is
one-step evolution under variable timestep:

    y_{t+1} = F(y_t, dt_t, g)

where:
  - y_t: state vector at time t in z-space (standardized log10, shape [B, S])
  - dt_t: normalized timestep for that step (shape [B])
  - g: global conditioning parameters (shape [B, G])

RelaxationHead is the only model. Each species relaxes exponentially toward a bounded target:

    z' = z + a (q - z),   a = -expm1(-k dt),   q = clamp(z + r(z, g), zlo, zhi),
    log10 k = c + h tanh(s / h)

A residual MLP trunk maps (z, g) to the target offset r and the rate logit s; it never sees dt,
which enters only through a. So the update vanishes as dt -> 0, the fixed point (r = 0) does not
depend on dt, and since a lies in [0, 1] and q in [zlo, zhi], the physical box
[zlo, zhi] (epsilon <= y <= 1) is invariant up to rounding. zlo, zhi and the dt range come
from normalization.json.

Design goals:
  - Deterministic behavior: no silent fallbacks.
  - Strict interfaces: ambiguous shapes/config values are rejected early.
  - Stable rollouts: a bounded target, a bounded rate and a near-identity output init.
  - Export-safe: no int() on dynamic tensor dimensions; uses torch._check for asserts.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from utils import as_bool, as_float

ActivationFactory = Callable[[], nn.Module]

# Output-layer init std of the relaxation head's trunk: a near-identity start (r ~ 0, so q ~ z, and
# log10 k ~ c, the middle of the rate range). The std is absolute, not scaled by fan-in, so the
# initial outputs grow as sqrt(width).
_OUT_INIT_STD = 1e-3
# Range of log10 k (k in 1/s): log10 k = c + h tanh(s / h) with c, h the range's midpoint and half-width.
_LOG10_RATE_RANGE = (-12.0, 3.0)
# Fixed global set, in this order. src/main.py and testing/export.py
# import it from here; the standalone processing/preprocessing.py and testing/predictions.py keep a copy.
_REQUIRED_GLOBALS = ("P", "T")


# ==============================================================================
# Shape utilities (strict, unambiguous, export-safe)
# ==============================================================================

def normalize_dt_shape(
        dt: torch.Tensor,
        batch_size: torch.SymInt,
        seq_len: torch.SymInt,
        *,
        context: str,
) -> torch.Tensor:
    """Return dt as shape [B, K]; any other shape is an error.

    Only [B, K] is accepted (no squeezing or broadcasting that could mask dataset /
    collation bugs). The rank and K checks are plain Python raises (the trainer passes a
    concrete K); only batch_size can be symbolic (torch.SymInt), checked with torch._check.
    """
    if dt.ndim != 2 or dt.shape[1] != seq_len:
        raise ValueError(f"{context}: dt must have shape [B, {seq_len}], got {tuple(dt.shape)}.")
    torch._check(dt.shape[0] == batch_size)
    return dt


def normalize_dt_step(dt_norm: torch.Tensor, batch_size: torch.SymInt, *, context: str) -> torch.Tensor:
    """Return a single-step dt [B] as shape [B, 1]; any other shape is an error.

    The rank check is a plain Python raise (ndim is concrete under torch.export);
    batch_size can be symbolic (torch.SymInt) for export compatibility.
    """
    if dt_norm.ndim != 1:
        raise ValueError(f"{context}: dt must have shape [B], got {tuple(dt_norm.shape)}.")
    torch._check(dt_norm.shape[0] == batch_size)
    return dt_norm.reshape(-1, 1)


def _validate_g_shape(g: torch.Tensor, batch_size: torch.SymInt, global_dim: int, context: str) -> None:
    """Validate g shape is exactly [B, G].

    The rank and G checks are plain Python raises with a message (ndim and G are concrete under
    torch.export and torch.compile). Only the batch comparison uses a message-less torch._check:
    batch_size can be symbolic (torch.SymInt), and a message callable breaks strict export and
    fullgraph compile in torch 2.8.
    """
    if g.ndim != 2 or g.shape[1] != global_dim:
        raise ValueError(f"{context}: g must have shape [B, {global_dim}], got {tuple(g.shape)}.")
    torch._check(g.shape[0] == batch_size)


def autoregressive_rollout(
        forward_step: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        y0: torch.Tensor,      # [B, S]
        dt_bk: torch.Tensor,   # [B, K]
        g: torch.Tensor,       # [B, G]
) -> torch.Tensor:
    """Roll a per-step forward_step K times, feeding predictions back in.

    Used by the trainer's open-loop evaluation unroll.
    """
    B = y0.shape[0]
    K = dt_bk.shape[1]
    S = y0.shape[-1]
    y_pred = torch.empty(B, K, S, device=y0.device, dtype=y0.dtype)
    state = y0
    # range(K) specializes K at trace time (torch.export guards K to a constant); only the
    # batch dim can stay dynamic. Deployment exports the one-step forward_step instead.
    for t in range(K):
        state = forward_step(state, dt_bk[:, t], g)
        y_pred[:, t, :] = state
    return y_pred


# ==============================================================================
# Config coercion (strict)
# ==============================================================================

def _as_int_list(value: Any, *, name: str) -> List[int]:
    """Coerce a config value into list[int].

    Supported input types:
      - int: interpreted as a single hidden layer
      - list/tuple of ints

    Any other type is an error, and every width must be >= 1.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name}: expected int or list[int].")
    if isinstance(value, int):
        value = [value]
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{name}: must be non-empty.")
        out: List[int] = []
        for v in value:
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"{name}: must contain ints.")
            if v < 1:
                raise ValueError(f"{name}: every width must be >= 1, got {v}.")
            out.append(int(v))
        return out
    raise TypeError(f"{name}: expected int or list[int].")


# ==============================================================================
# Building blocks
# ==============================================================================

def get_activation(name: str) -> ActivationFactory:
    """Return an activation factory by name."""
    key = str(name).lower().strip()
    # SiLU only: every shipped configuration uses it.
    factories: Dict[str, ActivationFactory] = {
        "silu": nn.SiLU,
    }
    if key not in factories:
        raise ValueError(f"activation unsupported: {key}")
    return factories[key]


def _init_linear_xavier(linear: nn.Linear, *, scale: float = 1.0) -> None:
    """Variance-preserving init; optionally scales weights."""
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    if scale != 1.0:
        with torch.no_grad():
            linear.weight.mul_(float(scale))


class ResidualMLP(nn.Module):
    """Residual MLP with per-layer skip connections.

    Per hidden layer:
        h = LN( proj(h) + Dropout(Act(Linear(h))) )
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int],
            output_dim: int,
            activation_factory: ActivationFactory,
            dropout_p: float = 0.0,
            *,
            layer_norm: bool,
            layer_norm_eps: float,
    ):
        super().__init__()

        dims = [int(input_dim)] + [int(d) for d in hidden_dims]
        h_dims = [int(d) for d in hidden_dims]
        use_ln = bool(layer_norm)

        self.linears = nn.ModuleList()
        self.projs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.drops = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(len(h_dims)):
            in_d, out_d = dims[i], dims[i + 1]
            self.linears.append(nn.Linear(in_d, out_d))
            self.acts.append(activation_factory())
            self.drops.append(nn.Dropout(p=float(dropout_p)) if float(dropout_p) > 0.0 else nn.Identity())
            self.projs.append(nn.Identity() if in_d == out_d else nn.Linear(in_d, out_d, bias=False))
            self.norms.append(nn.LayerNorm(out_d, eps=float(layer_norm_eps)) if use_ln else nn.Identity())

        self.out = nn.Linear(dims[-1], int(output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:  # type: ignore[override]
        # Xavier init throughout + conservative scaling on the residual branch.
        n_layers = max(1, len(self.linears))
        residual_scale = 1.0 / math.sqrt(float(n_layers))

        for lin in self.linears:
            _init_linear_xavier(lin, scale=residual_scale)

        for proj in self.projs:
            if isinstance(proj, nn.Linear):
                _init_linear_xavier(proj)

        _init_linear_xavier(self.out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Under autocast (bf16-mixed) the input block, which reads the raw state, P and T, and the
        # output layer run in fp32. Without autocast (fp32, torch.export) the plain loop runs.
        fp32_islands = torch.is_autocast_enabled(x.device.type)
        h = x
        for i, (lin, act, drop, proj, norm) in enumerate(zip(self.linears, self.acts, self.drops, self.projs, self.norms)):
            if fp32_islands and i == 0:
                with torch.autocast(device_type=h.device.type, enabled=False):
                    h = h.float()
                    h = norm(proj(h) + drop(act(lin(h))))
                continue
            h = proj(h) + drop(act(lin(h)))
            h = norm(h)
        if fp32_islands:
            with torch.autocast(device_type=h.device.type, enabled=False):
                return self.out(h.float())
        return self.out(h)


# ==============================================================================
# Model
# ==============================================================================

class RelaxationHead(nn.Module):
    """Per-species exponential relaxation toward a bounded target, in z-space.

        z' = z + a (q - z),  a = -expm1(-k dt),  q = clamp(z + r(z, g), zlo, zhi),  log10 k = c + h tanh(s / h)

    The trunk maps (z, g) to [r, s] (2S outputs) and never sees dt. dt_norm maps linearly to log10 dt
    (seconds) between dt_log10_min and dt_log10_max and is not clamped (dt_norm -> -inf gives the identity).
    """

    def __init__(
            self,
            *,
            state_dim: int,
            global_dim: int,
            hidden_dims: Sequence[int],
            activation_name: str,
            dropout: float,
            layer_norm: bool,
            layer_norm_eps: float,
            zlo: torch.Tensor,
            zhi: torch.Tensor,
            dt_log10_min: float,
            dt_log10_max: float,
    ):
        super().__init__()

        self.S, self.G = int(state_dim), int(global_dim)
        self.trunk = ResidualMLP(
            self.S + self.G,
            [int(d) for d in hidden_dims],
            2 * self.S,
            get_activation(activation_name),
            float(dropout),
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
        )
        # Near-identity start (r ~ 0, log10 k ~ c), drawn after the trunk's Xavier init: a seed gives
        # the same weights only while this order holds.
        with torch.no_grad():
            nn.init.normal_(self.trunk.out.weight, std=_OUT_INIT_STD)
            nn.init.zeros_(self.trunk.out.bias)
        # The box in z-space, persistent (checkpoints carry it; norm_sha256 ties it to normalization.json).
        self.register_buffer("zlo", zlo.float().clone())
        self.register_buffer("zhi", zhi.float().clone())
        lo, hi = _LOG10_RATE_RANGE
        # Plain Python floats, not buffers: the state_dict holds only the trunk and the box.
        self.c, self.h = 0.5 * (lo + hi), 0.5 * (hi - lo)
        self.dt_log10_min, self.dt_log10_max = float(dt_log10_min), float(dt_log10_max)

    def delta(self, z: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Change of z over one step: fp32 z [B, S], dt_norm [B], g [B, G] -> fp32 [B, S]."""
        o = self.trunk(torch.cat([z, g], dim=-1))
        q = torch.clamp(z + o[:, :self.S], self.zlo, self.zhi)
        log10k = self.c + self.h * torch.tanh(o[:, self.S:] / self.h)
        ldt = self.dt_log10_min + (self.dt_log10_max - self.dt_log10_min) * dt_norm
        return -torch.expm1(-torch.pow(10.0, log10k + ldt.reshape(-1, 1))) * (q - z)

    def forward_step(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """One step y_i -> y_i + delta: the network always runs in fp32, the add in y_i's dtype (fp32 or fp64)."""
        B = y_i.shape[0]  # Keep as symbolic
        _validate_g_shape(g, B, self.G, context="RelaxationHead.forward_step")
        dt = normalize_dt_step(dt_norm, B, context="RelaxationHead.forward_step").reshape(-1)
        return y_i + self.delta(y_i.float(), dt.float(), g.float()).to(y_i.dtype)


# ==============================================================================
# Factory
# ==============================================================================

def create_model(config: Dict[str, Any], normalization_manifest: Mapping[str, Any]) -> nn.Module:
    """Construct the relaxation head from config (dims from config.data) and the normalization manifest.

    Notes:
      - model.type must be "relax" (the only model); any other value is a hard error.
      - The box is epsilon <= y <= 1 in z-space: zlo = (log10(epsilon) - mu) / sd and zhi = (0 - mu) / sd,
        with mu, sd the species' log_mean, log_std, computed in float64 and stored as float32. The dt
        log range is the manifest's dt.log_min, dt.log_max.
    """
    log = logging.getLogger(__name__)

    if not isinstance(config, dict):
        raise TypeError("config must be dict.")

    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise KeyError("data required.")

    species_vars = data_cfg.get("species_variables")
    if not isinstance(species_vars, list) or not species_vars:
        raise KeyError("data.species_variables required.")
    global_vars = data_cfg.get("global_variables")
    if not isinstance(global_vars, list):
        raise TypeError("data.global_variables must be list.")
    if list(global_vars) != list(_REQUIRED_GLOBALS):
        raise ValueError(f"data.global_variables must be exactly {list(_REQUIRED_GLOBALS)}")

    S, G = len(species_vars), len(global_vars)
    log.info("Model dims: S=%d, G=%d", S, G)

    mcfg = config.get("model")
    if not isinstance(mcfg, dict):
        raise KeyError("model required.")

    model_type_raw = mcfg.get("type")
    if model_type_raw is None:
        raise KeyError("model.type required.")
    model_type = str(model_type_raw).lower().strip()
    if model_type != "relax":
        raise ValueError(f"model.type unsupported: {model_type}")

    # Architecture knobs are schema-required (main.py checks presence only);
    # types are validated here, at the point of consumption.
    activation = str(mcfg["activation"]).lower().strip()
    dropout = as_float(mcfg["dropout"], "model.dropout")
    layer_norm = as_bool(mcfg["layer_norm"], "model.layer_norm")
    layer_norm_eps = as_float(mcfg["layer_norm_eps"], "model.layer_norm_eps")
    # Positive-form range checks (as_float already rejects NaN/Inf).
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"model.dropout must be in [0, 1), got {dropout}")
    if not layer_norm_eps > 0.0:
        raise ValueError(f"model.layer_norm_eps must be > 0, got {layer_norm_eps}")
    mlp_cfg = mcfg.get("mlp")
    if not isinstance(mlp_cfg, dict):
        raise KeyError("model.mlp required.")
    hidden_dims = _as_int_list(mlp_cfg["hidden_dims"], name="model.mlp.hidden_dims")

    stats = normalization_manifest["per_key_stats"]
    mu = torch.tensor([stats[name]["log_mean"] for name in species_vars], dtype=torch.float64)
    sd = torch.tensor([stats[name]["log_std"] for name in species_vars], dtype=torch.float64)
    zlo = (np.log10(float(normalization_manifest["epsilon"])) - mu) / sd
    zhi = (0.0 - mu) / sd
    dt_range = normalization_manifest["dt"]
    return RelaxationHead(
        state_dim=S,
        global_dim=G,
        hidden_dims=hidden_dims,
        activation_name=activation,
        dropout=dropout,
        layer_norm=layer_norm,
        layer_norm_eps=layer_norm_eps,
        zlo=zlo,
        zhi=zhi,
        dt_log10_min=dt_range["log_min"],
        dt_log10_max=dt_range["log_max"],
    )
