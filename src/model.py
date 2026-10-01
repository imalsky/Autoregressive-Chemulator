#!/usr/bin/env python3
"""
model.py - Model architecture for flow-map rollout prediction.

This module defines the neural network used by the trainer. The core task is
one-step evolution under variable timestep:

    y_{t+1} = F(y_t, dt_t, g)

where:
  - y_t: state vector at time t (shape [B, S])
  - dt_t: normalized timestep for that step (shape [B])
  - g: global conditioning parameters (shape [B, G])

FlowMapMLP is the only model: a direct MLP (residual or plain trunk) that predicts
y(t+dt) from concatenated (y(t), dt, g), with an optional fixed Fourier embedding of dt.

Design goals:
  - Deterministic behavior: no silent fallbacks.
  - Strict interfaces: ambiguous shapes/config values are rejected early.
  - Stable rollouts: optional delta prediction and conservative head init.
  - Export-safe: no int() on dynamic tensor dimensions; uses torch._check for asserts.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from utils import as_bool, as_float, as_int

ActivationFactory = Callable[[], nn.Module]

# Small output init for delta heads. This reduces early rollout blow-ups. Inherited (docs/DECISIONS.md).
# The std is absolute, not scaled by fan-in, so the initial per-step delta std is about
# 1e-3 * sqrt(fan_in) * RMS(head input): ~0.045 z-units for the production residual+LN trunk at width
# 2048 (unit-RMS LayerNorm output) and ~0.027 for the non-residual fallback (post-SiLU RMS ~0.6;
# audit measurement). A width change therefore rescales the initial step as sqrt(width).
_DELTA_OUT_INIT_STD = 1e-3
# Fixed global set, in this order (spec.md section 5, items 4-6). src/main.py and testing/export.py
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

class FourierDtFeatures(nn.Module):
    """Fixed random-Fourier embedding of the normalized scalar dt (Tancik et al. 2020).

    dt_norm [..., 1] -> [..., 1 + 2F]: the raw dt is kept (strict superset of the
    plain input) and sin/cos(2*pi*b_f*dt) features are appended. The projection is
    a seeded, non-trainable buffer (learned frequencies do not help for a 1-D
    conditioner and add seed variance). Export-safe: fixed shapes, registered buffer.
    """

    def __init__(self, *, num_freqs: int, sigma: float, seed: int) -> None:
        super().__init__()
        # Positive-form checks so NaN fails too.
        if not int(num_freqs) >= 1:
            raise ValueError(f"fourier_dt.num_freqs must be >= 1, got {num_freqs}")
        if not (math.isfinite(float(sigma)) and float(sigma) > 0.0):
            raise ValueError(f"fourier_dt.sigma must be finite and > 0, got {sigma}")
        gen = torch.Generator().manual_seed(int(seed))
        freqs = torch.randn(int(num_freqs), generator=gen) * float(sigma)
        self.register_buffer("freqs", freqs)
        self.num_freqs = int(num_freqs)
        self.out_dim = 1 + 2 * int(num_freqs)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """Embed dt [..., 1] -> [..., 1 + 2F] as [dt, sin(2*pi*B*dt), cos(2*pi*B*dt)]."""
        torch._check(dt.shape[-1] == 1)
        # Compute the projection and sin/cos in at least float32 (bf16 is upcast, float64 is kept):
        # under bf16 the phase 2*pi*b_f*dt loses ~8 mantissa bits, aliasing the high-frequency
        # features to noise. Cast back to the input dtype at the end.
        ct = torch.promote_types(dt.dtype, torch.float32)
        dtc = dt.to(ct)
        proj = 2.0 * math.pi * dtc * self.freqs.to(ct)
        feats = torch.cat([dtc, torch.sin(proj), torch.cos(proj)], dim=-1)
        return feats.to(dt.dtype)


def build_fourier_dt(fourier_cfg: Any) -> Optional[FourierDtFeatures]:
    """Construct FourierDtFeatures from a model.fourier_dt config dict (or None).

    Absent config or enabled=False -> None (plain scalar dt, current behavior).
    """
    if fourier_cfg is None:
        return None
    if not isinstance(fourier_cfg, dict):
        raise TypeError("model.fourier_dt must be a dict.")
    if "enabled" not in fourier_cfg:
        raise KeyError(
            "model.fourier_dt present but missing 'enabled' — set it explicitly "
            "(a silent default would be a silent architecture change)."
        )
    enabled = fourier_cfg["enabled"]
    if not isinstance(enabled, bool):
        raise TypeError("model.fourier_dt.enabled must be a bool.")
    if not enabled:
        return None
    for key in ("num_freqs", "sigma", "seed"):
        if key not in fourier_cfg:
            raise KeyError(f"model.fourier_dt.{key} required when fourier_dt.enabled=true.")
    return FourierDtFeatures(
        num_freqs=as_int(fourier_cfg["num_freqs"], "model.fourier_dt.num_freqs"),
        sigma=as_float(fourier_cfg["sigma"], "model.fourier_dt.sigma"),
        seed=as_int(fourier_cfg["seed"], "model.fourier_dt.seed"),
    )


def get_activation(name: str) -> ActivationFactory:
    """Return an activation factory by name."""
    key = str(name).lower().strip()
    # SiLU only: the production trunk and both documented fallbacks use it (docs/DECISIONS.md).
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


class MLP(nn.Module):
    """Plain MLP: Linear -> (LayerNorm) -> Act -> (Dropout) -> ... -> Linear."""

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

        dims = [int(input_dim)] + [int(d) for d in hidden_dims] + [int(output_dim)]
        use_ln = bool(layer_norm)

        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            in_d, out_d = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.LayerNorm(out_d, eps=float(layer_norm_eps)) if use_ln else nn.Identity())
            # Dropout shares the activation's slot, so parameter keys do not depend on dropout (a
            # separate Dropout entry would renumber every later Linear and break strict loading).
            act = activation_factory()
            layers.append(nn.Sequential(act, nn.Dropout(p=float(dropout_p))) if float(dropout_p) > 0.0 else act)

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:  # type: ignore[override]
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _init_linear_xavier(m)

    def last_linear(self) -> nn.Linear:
        # Final layer in the Sequential is always the output Linear.
        out = self.network[-1]
        if not isinstance(out, nn.Linear):
            raise RuntimeError("MLP: last layer is not Linear.")
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


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

    def last_linear(self) -> nn.Linear:
        return self.out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Under autocast (bf16-mixed) the input block, which reads the raw state, dt, P and T, and the
        # delta head run in fp32. Without autocast (fp32, bf16-true, torch.export) the plain loop runs.
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
# Full models
# ==============================================================================

class FlowMapMLP(nn.Module):
    """Direct MLP flow-map: (y, dt, g) -> y_next."""

    def __init__(
            self,
            *,
            state_dim: int,
            global_dim: int,
            hidden_dims: Sequence[int],
            activation_name: str,
            dropout: float = 0.0,
            residual: bool,
            predict_delta: bool,
            layer_norm: bool,
            layer_norm_eps: float,
            fourier_dt: Optional[FourierDtFeatures] = None,
    ):
        super().__init__()

        self.S, self.G = int(state_dim), int(global_dim)
        self.predict_delta = bool(predict_delta)
        self.fourier_dt = fourier_dt

        act_factory = get_activation(activation_name)
        dt_dim = fourier_dt.out_dim if fourier_dt is not None else 1
        input_dim = self.S + dt_dim + self.G

        net_cls = ResidualMLP if bool(residual) else MLP
        self.network = net_cls(
            input_dim,
            [int(d) for d in hidden_dims],
            self.S,
            act_factory,
            float(dropout),
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
        )

        if self.predict_delta:
            self._init_delta_head()

    def _init_delta_head(self) -> None:
        """Small-output init for stable delta rollouts."""
        with torch.no_grad():
            out = self.network.last_linear()
            nn.init.zeros_(out.bias)
            nn.init.normal_(out.weight, mean=0.0, std=_DELTA_OUT_INIT_STD)

    def forward_step(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """One step y_i -> y_next from concatenated (y_i, embedded dt, g) (+ y_i if predict_delta)."""
        B = y_i.shape[0]  # Keep as symbolic
        _validate_g_shape(g, B, self.G, context="FlowMapMLP.forward_step")

        dt = normalize_dt_step(dt_norm, B, context="FlowMapMLP.forward_step").to(y_i.dtype)
        dt_feat = self.fourier_dt(dt) if self.fourier_dt is not None else dt
        x = torch.cat([y_i, dt_feat, g], dim=-1)

        y_pred = self.network(x)
        return y_pred + y_i if self.predict_delta else y_pred


# ==============================================================================
# Factory
# ==============================================================================

def create_model(config: Dict[str, Any]) -> nn.Module:
    """Construct the model from config (dims from config.data).

    Notes:
      - Model selection is driven ONLY by model.type; "mlp" is the only supported type.
      - Unknown model.type is a hard error.
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

    # Architecture knobs are schema-required (main.py checks presence only);
    # types are validated here, at the point of consumption.
    activation = str(mcfg["activation"]).lower().strip()
    dropout = as_float(mcfg["dropout"], "model.dropout")
    predict_delta = as_bool(mcfg["predict_delta"], "model.predict_delta")
    layer_norm = as_bool(mcfg["layer_norm"], "model.layer_norm")
    layer_norm_eps = as_float(mcfg["layer_norm_eps"], "model.layer_norm_eps")
    # Positive-form range checks (as_float already rejects NaN/Inf).
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"model.dropout must be in [0, 1), got {dropout}")
    if not layer_norm_eps > 0.0:
        raise ValueError(f"model.layer_norm_eps must be > 0, got {layer_norm_eps}")
    fourier_dt = build_fourier_dt(mcfg.get("fourier_dt"))  # optional feature block
    if fourier_dt is not None:
        log.info(
            "fourier_dt enabled: num_freqs=%d sigma=%s (dt input dim 1 -> %d)",
            fourier_dt.num_freqs, mcfg["fourier_dt"]["sigma"], fourier_dt.out_dim,
        )

    if model_type == "mlp":
        mlp_cfg = mcfg.get("mlp")
        if not isinstance(mlp_cfg, dict):
            raise KeyError("model.mlp required.")
        hidden_dims = _as_int_list(mlp_cfg["hidden_dims"], name="model.mlp.hidden_dims")
        residual = as_bool(mlp_cfg["residual"], "model.mlp.residual")
        return FlowMapMLP(
            state_dim=S,
            global_dim=G,
            hidden_dims=hidden_dims,
            activation_name=activation,
            dropout=dropout,
            residual=residual,
            predict_delta=predict_delta,
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
            fourier_dt=fourier_dt,
        )

    raise ValueError(f"model.type unsupported: {model_type}")
