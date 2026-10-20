"""The closed relaxation head (config head "relaxclosedr"): one model call advances the normalized log10 state z of the
entry set by a time step dt at fixed T and P, and returns delta = z' - z (fp32); the caller adds it to the state in fp64.

Per call: a residual MLP trunk maps (z, g) (g = the normalized globals: log10 P min-max, T min-max) to a target offset r and a rate s per entry;
the relaxed state is z~ = z + a (q - z) with q = clamp(z + r, zlo, zhi) (the physical box 1e-30 <= y <= 1),
a = -expm1(-k dt) and log10 k = c + h tanh(s / h) in [-12, 3]; dt enters only through a (a -> 0 as dt -> 0). z~ is formed in
fp64 and then closed: it is moved to the input's element ratios to H (C/H, N/H, O/H, He/H over every entry) by
y' = y~ exp(A lam) with lam from a damped Newton solve (elemproj.project_ln, or project_ln_fused under torch.compile in
training: same mathematics), and normalized so that the molecules sum to 1 (atom reservoirs C_res, N_res, O_res, H_res
excluded). A row whose solve does not converge (max |log-ratio residual| > 1e-4, or NaN) is re-solved by
Levenberg-Marquardt when fallback is "lm"; otherwise (fallback "keep", or LM fails too) the row keeps its input.

The entry set (S, SPECIES) is set by train.py and lhbench.py from the normalization file before a model is built.
"""
import math

import torch
import torch.nn as nn

S, G = 12, 2                        # entries, globals (P, T); S is reset from the normalization file
SPECIES = None                      # entry names, set with S
DT_LOG_MIN, DT_LOG_MAX = -1.0, 3.0  # normalization.json "dt": dt_norm = (log10 dt - DT_LOG_MIN) / (DT_LOG_MAX - DT_LOG_MIN)
EPS_LOG10 = -30.0                   # log10(normalization.epsilon)
LN10 = math.log(10.0)


def dt_log10(dt_norm):
    return DT_LOG_MIN + (DT_LOG_MAX - DT_LOG_MIN) * dt_norm


def mu_sd(zlo, zhi):
    """Per-entry log10 mean and std recovered from the box: zlo = (EPS_LOG10 - mu) / sd, zhi = -mu / sd (fp32-rounded,
    as the model was trained)."""
    sd = -EPS_LOG10 / (zhi.double() - zlo.double())
    return (-zhi.double() * sd).float(), sd.float()


def _init_linear_xavier(linear, *, scale=1.0):
    """Variance-preserving init; optionally scales weights."""
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    if scale != 1.0:
        with torch.no_grad():
            linear.weight.mul_(float(scale))


class ResidualMLP(nn.Module):
    """Residual MLP with per-layer skip connections (the trunk of the 12-species model, tag paper-v1, src/model.py).

    Per hidden layer:
        h = LN( proj(h) + Dropout(Act(Linear(h))) )
    """

    def __init__(self, input_dim, hidden_dims, output_dim, activation_factory, dropout_p=0.0, *, layer_norm, layer_norm_eps):
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

    def reset_parameters(self):
        # Xavier init throughout + conservative scaling on the residual branch.
        n_layers = max(1, len(self.linears))
        residual_scale = 1.0 / math.sqrt(float(n_layers))

        for lin in self.linears:
            _init_linear_xavier(lin, scale=residual_scale)

        for proj in self.projs:
            if isinstance(proj, nn.Linear):
                _init_linear_xavier(proj)

        _init_linear_xavier(self.out)

    def forward(self, x):
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


class RelaxHead(nn.Module):
    """The trunk and the relaxation constants: ResidualMLP (S + G) -> hidden... -> 2S (offsets r, rates s), SiLU,
    LayerNorm; near-identity start (output weights N(0, 1e-3), bias 0: r ~ 0, log10 k ~ mid-range); the box [zlo, zhi]
    as buffers; log10 k = c + h tanh(s / h) over logk_range."""

    def __init__(self, hidden, zlo, zhi, logk_range=(-12.0, 3.0), **_):
        super().__init__()
        self.trunk = ResidualMLP(S + G, hidden, 2 * S, nn.SiLU, 0.0, layer_norm=True, layer_norm_eps=1e-5)
        with torch.no_grad():  # near-identity start: r ~ 0, log10 k ~ mid-range
            nn.init.normal_(self.trunk.out.weight, std=1e-3)
            nn.init.zeros_(self.trunk.out.bias)
        self.register_buffer("zlo", zlo.float().clone())
        self.register_buffer("zhi", zhi.float().clone())
        self.c = 0.5 * (logk_range[0] + logk_range[1])
        self.h = 0.5 * (logk_range[1] - logk_range[0])


class RelaxClosedHead(RelaxHead):
    """The element-projection constants: the fp32-rounded normalization (mu, sd) in fp64, H atoms and (C, N, O, He)
    atoms per entry, and the Newton iterations. Non-persistent buffers (not in the checkpoint)."""

    def __init__(self, hidden, zlo, zhi, newton=12, **kw):
        super().__init__(hidden, zlo, zhi, **kw)
        import elemproj as EP
        mu, sd = mu_sd(zlo, zhi)
        self.register_buffer("mu", mu.double(), persistent=False)
        self.register_buffer("sd", sd.double(), persistent=False)
        nh, na = EP.stoich(SPECIES)
        self.register_buffer("nh", nh, persistent=False)
        self.register_buffer("na", na, persistent=False)
        self.newton = newton


class RelaxClosedRHead(RelaxClosedHead):
    """The closed relaxation call (module docstring). fused: training's compiled projection (project_ln_fused); the
    benches, validation and the EMA copy use project_ln. fallback: 'keep' | 'lm' for rows whose Newton solve fails.
    rowcount: optional int64 [B, 2] that a bench allocates; each call adds 1 where the Newton solve failed (column 0) and
    where the row kept its input (column 1)."""

    def __init__(self, hidden, zlo, zhi, newton=12, fallback="keep", fused=False, **kw):
        super().__init__(hidden, zlo, zhi, newton=newton, **kw)
        import elemproj as EP
        self.register_buffer("mol", EP.molidx(SPECIES), persistent=False)
        self.fused = bool(fused)
        self.fallback, self.lm_iters, self.rowcount = fallback, 32, None

    def delta(self, z, dtn, g):
        import elemproj as EP  # not stored on the module: the EMA copy deep-copies it
        o = self.trunk(torch.cat([z, g], dim=-1)).double()
        z64 = z.double()
        q = torch.maximum(torch.minimum(z64 + o[:, :S], self.zhi.double()), self.zlo.double())
        log10k = self.c + self.h * torch.tanh(o[:, S:] / self.h)
        kdt = torch.pow(10.0, log10k + dt_log10(dtn).double().reshape(-1, 1))
        zr = z64 - torch.expm1(-kdt) * (q - z64)
        ln0 = (z64 * self.sd + self.mu) * LN10
        lt = (zr * self.sd + self.mu) * LN10
        lr = EP.log_ratios(ln0, self.na, self.nh)
        w, res = (EP.project_ln_fused if self.fused else EP.project_ln)(lt, lr, self.na, self.nh, iters=self.newton, mol=self.mol)
        ok = (res < 1e-4).unsqueeze(-1)                                      # NaN residual -> not ok -> no change
        ok1 = ok
        if self.fallback == "lm":
            w2, res2 = EP.project_lm(lt, lr, self.na, self.nh, iters=self.lm_iters, mol=self.mol)
            w = torch.where(ok, w, w2)
            ok = ok | (res2 < 1e-4).unsqueeze(-1)
        if self.rowcount is not None:
            self.rowcount += torch.logical_not(torch.cat([ok1, ok], -1)).long()
        zp = torch.where(ok, (w / LN10 - self.mu) / self.sd, z64)
        zp = torch.maximum(torch.minimum(zp, self.zhi.double()), self.zlo.double())
        return (zp - z64).float()


def build(head, hidden, zlo, zhi, newton=12, fallback="keep", fused=False):
    """The model of a config: head must be "relaxclosedr"; hidden = the trunk widths; newton, fallback, fused: train.py's
    closed_iters, closed_fallback, fused_proj."""
    if head != "relaxclosedr":
        raise ValueError(f"head {head!r}: only 'relaxclosedr' is supported")
    return RelaxClosedRHead(hidden, zlo, zhi, newton=newton, fallback=fallback, fused=fused)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def flops_per_call(m):
    """Multiply-adds x 2 of the Linear layers for one call of delta."""
    return sum(2 * mod.in_features * mod.out_features for mod in m.modules() if isinstance(mod, nn.Linear))
