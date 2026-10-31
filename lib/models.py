"""The closed relaxation head (config head "relaxclosedr", or "relaxclosedrdt" with dt as a trunk input): one model call advances the normalized log10 state z of the
entry set by a time step dt at fixed T and P, and returns delta = z' - z (fp32); the caller adds it to the state in fp64.

Per call: a residual MLP trunk maps (z, g) (g = the normalized globals: log10 P min-max, T min-max) to a target offset r and a rate s per entry;
the relaxed state is z~ = z + a (q - z) with q = clamp(z + r, zlo, zhi) (the physical box 1e-30 <= y <= 1),
a = -expm1(-k dt) and log10 k = c + h tanh(s / h) in [-12, 3]; dt enters only through a (a -> 0 as dt -> 0). z~ is formed in
fp64 and then closed: it is moved to the input's element ratios to H (C/H, N/H, O/H, He/H over every entry) by
y' = y~ exp(A lam) with lam from a damped Newton solve (elemproj.project_ln, or project_ln_fused under torch.compile in
training: same mathematics), and normalized so that the molecules sum to 1 (atom reservoirs C_res, N_res, O_res, H_res
excluded). A row whose solve does not converge (max |log-ratio residual| > 1e-4, or NaN) is re-solved by
Levenberg-Marquardt when fallback is "lm" (then, with the head's cont set, by continuation from the input,
elemproj.project_cont); otherwise (fallback "keep", or every solve fails) the row keeps its input.

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


class FlowMapTrunk(nn.Module):
    """The flow-map trunk (head "relaxclosedfm"; the encoder-dynamics-decoder of the 2026 paper): input [z, g, dt_norm];
    h = E(z, g) (latent width), h' = h + F(h, g, dt_norm), output D(h') (2S: target offsets and rates). E, F, D are
    ResidualMLPs of the hidden widths (SiLU, LayerNorm). out = D's output layer (the head's near-identity start)."""

    def __init__(self, hidden, latent):
        super().__init__()
        self.enc = ResidualMLP(S + G, hidden, latent, nn.SiLU, 0.0, layer_norm=True, layer_norm_eps=1e-5)
        self.dyn = ResidualMLP(latent + G + 1, hidden, latent, nn.SiLU, 0.0, layer_norm=True, layer_norm_eps=1e-5)
        self.dec = ResidualMLP(latent, hidden, 2 * S, nn.SiLU, 0.0, layer_norm=True, layer_norm_eps=1e-5)

    @property
    def out(self):
        return self.dec.out

    def forward(self, x):
        h = self.enc(x[:, :S + G])
        h = h + self.dyn(torch.cat([h, x[:, S:]], dim=-1))
        return self.dec(h)


class FlowMapEqTrunk(nn.Module):
    """The hybrid trunk (head "relaxclosedfmeq"): h = E(z, g); the dt-free target offsets r_inf = Dq(h); h' = h + F(h, g,
    clamp(dt_norm, 0, 1)) (the trunk sees dt clamped to the normalization's 0.1-1000 s; the relaxation uses the true dt); D(h')
    -> 2S (rates s, transient directions c). Output [B, 3S]: r_inf, s, c. out = D's output layer, qout = Dq's."""

    def __init__(self, hidden, latent):
        super().__init__()
        self.enc = ResidualMLP(S + G, hidden, latent, nn.SiLU, 0.0, layer_norm=True, layer_norm_eps=1e-5)
        self.qdec = ResidualMLP(latent, hidden, S, nn.SiLU, 0.0, layer_norm=True, layer_norm_eps=1e-5)
        self.dyn = ResidualMLP(latent + G + 1, hidden, latent, nn.SiLU, 0.0, layer_norm=True, layer_norm_eps=1e-5)
        self.dec = ResidualMLP(latent, hidden, 2 * S, nn.SiLU, 0.0, layer_norm=True, layer_norm_eps=1e-5)

    @property
    def out(self):
        return self.dec.out

    @property
    def qout(self):
        return self.qdec.out

    def forward(self, x):
        h = self.enc(x[:, :S + G])
        h2 = h + self.dyn(torch.cat([h, x[:, S:S + G], torch.clamp(x[:, S + G:], 0.0, 1.0)], dim=-1))
        return torch.cat([self.qdec(h), self.dec(h2)], dim=-1)


def unit_upper(o, iu, bound=0.1):
    """U = I + N and U^-1 for N strictly upper with entries bound tanh(o / bound) at the indices iu ([B, S(S-1)/2] -> two
    [B, S, S], fp32). With bound 0.1 the growth of |z~ - q| within a call is at most ~3.4x for S = 25 (0.25 allows ~80x). N is nilpotent (N^S = 0, S <= 32), so U^-1 = (I - N)(I + N^2)(I + N^4)(I + N^8)(I + N^16) exactly;
    matrix products only (CUDA-graph safe). o = 0 gives U = U^-1 = I."""
    B, n = o.shape[0], iu.shape[1]
    S_ = int(round((1 + math.sqrt(1 + 8 * n)) / 2))
    N = o.new_zeros(B, S_, S_)
    N[:, iu[0], iu[1]] = bound * torch.tanh(o / bound)
    eye = torch.eye(S_, dtype=o.dtype, device=o.device).expand(B, S_, S_)
    Ui, P = eye - N, N @ N
    for _ in range(4):   # (I + N^2)(I + N^4)(I + N^8)(I + N^16)
        Ui = Ui @ (eye + P)
        P = P @ P
    return eye + N, Ui


def skew_exp(o, iu, bound=4.0, squarings=8, order=10):
    """exp(X) for X = K - K^T, K strictly upper with entries bound tanh(o / bound) at the indices iu ([B, S(S-1)/2] ->
    [B, S, S], fp32): a fixed-length Taylor series on X / 2^squarings, then squarings squarings; matrix products only, so
    it runs inside a captured CUDA graph (torch.linalg.matrix_exp does not). |X| <= bound (S - 1), so the series' term
    after order is below fp32 rounding. o = 0 gives the identity exactly."""
    B, n = o.shape[0], iu.shape[1]
    S_ = int(round((1 + math.sqrt(1 + 8 * n)) / 2))
    K = o.new_zeros(B, S_, S_)
    K[:, iu[0], iu[1]] = bound * torch.tanh(o / bound)
    X = (K - K.transpose(1, 2)) / 2.0 ** squarings
    eye = torch.eye(S_, dtype=o.dtype, device=o.device).expand(B, S_, S_)
    E = eye
    for j in range(order, 0, -1):   # Horner: I + X/1 (I + X/2 (I + ... X/order))
        E = eye + (X @ E) / j
    for _ in range(squarings):
        E = E @ E
    return E


class RelaxHead(nn.Module):
    """The trunk and the relaxation constants: ResidualMLP (S + G) -> hidden... -> 2S (offsets r, rates s), SiLU,
    LayerNorm; near-identity start (output weights N(0, 1e-3), bias 0: r ~ 0, log10 k ~ mid-range); the box [zlo, zhi]
    as buffers; log10 k = c + h tanh(s / h) over logk_range."""

    def __init__(self, hidden, zlo, zhi, logk_range=(-12.0, 3.0), dt_input=False, dt_rate=False, flowmap=False, latent=512,
                 flowmap_eq=False, vmix=False, vnn=False, **_):
        super().__init__()
        self.vmix = bool(vmix) or bool(vnn)   # the mixed-mode heads: S(S-1)/2 more trunk outputs set the mode mixing (delta)
        self.vnn = bool(vnn)                  # non-normal modes: S(S-1)/2 more for the triangular factor U (delta)
        self.dt_input = bool(dt_input) or bool(flowmap) or bool(flowmap_eq)   # normalized log dt is a trunk input too
        self.eq = bool(flowmap_eq)       # the hybrid head: dt-free target, bounded dt-dependent transient (delta)
        self.beta = 0.5                  # the hybrid's transient bound (beta < 1: no fixed point besides q_inf's)
        self.rate_dt = None              # the rate-only dt head: a small branch (z, g, dt) -> a rate offset per entry
        if dt_rate:
            self.rate_dt = nn.Sequential(nn.Linear(S + G + 1, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, S))
            with torch.no_grad():    # zero output: the same function as the head without the branch at the start
                nn.init.zeros_(self.rate_dt[-1].weight)
                nn.init.zeros_(self.rate_dt[-1].bias)
        if flowmap_eq:
            self.trunk = FlowMapEqTrunk(hidden, latent)
        elif flowmap:
            self.trunk = FlowMapTrunk(hidden, latent)
        else:
            self.trunk = ResidualMLP(S + G + int(self.dt_input), hidden,
                                     2 * S + (S * (S - 1) // 2) * (int(self.vmix) + int(self.vnn)), nn.SiLU, 0.0,
                                     layer_norm=True, layer_norm_eps=1e-5)
        with torch.no_grad():  # near-identity start: r ~ 0, log10 k ~ mid-range
            nn.init.normal_(self.trunk.out.weight, std=1e-3)
            nn.init.zeros_(self.trunk.out.bias)
            if flowmap_eq:   # the target decoder too (r_inf ~ 0); the transient starts at ~0 through c ~ 0
                nn.init.normal_(self.trunk.qout.weight, std=1e-3)
                nn.init.zeros_(self.trunk.qout.bias)
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
        if self.eq:   # the hybrid: the projector off the directions the element projection absorbs, in z units
            na, sd = self.na.cpu(), self.sd.cpu()   # (on the CPU; the model is moved to its device after build)
            W = torch.cat([na, torch.ones_like(na[:, :1])], 1) / (LN10 * sd.reshape(-1, 1))   # [S, 5]
            self.register_buffer("pi", torch.eye(S, dtype=torch.float64) - W @ torch.linalg.pinv(W), persistent=False)
        if self.vmix:
            self.register_buffer("iu", torch.triu_indices(S, S, 1), persistent=False)
        self.fused = bool(fused)
        self.fallback, self.lm_iters, self.rowcount = fallback, 32, None
        self.cont = False   # True: after LM, rows still unsolved get elemproj.project_cont (benches set it)

    def delta(self, z, dtn, g):
        import elemproj as EP  # not stored on the module: the EMA copy deep-copies it
        o = self.trunk(torch.cat([z, g, dtn.reshape(-1, 1).to(z.dtype)] if self.dt_input else [z, g], dim=-1)).double()
        z64 = z.double()
        q = torch.maximum(torch.minimum(z64 + o[:, :S], self.zhi.double()), self.zlo.double())
        s = o[:, S:2 * S] if (self.eq or self.vmix) else o[:, S:]
        if self.rate_dt is not None:
            s = s + self.rate_dt(torch.cat([z, g, dtn.reshape(-1, 1).to(z.dtype)], dim=-1)).double()
        log10k = self.c + self.h * torch.tanh(s / self.h)
        kdt = torch.pow(10.0, log10k + dt_log10(dtn).double().reshape(-1, 1))
        if self.eq:   # z~ = z + a v + beta a (1 - a) RMS(v) tanh(c), v = pi (q_inf - z): the target offset without the
            a = -torch.expm1(-kdt)    # components the projection absorbs (W = [na, 1] / (ln10 sd)), which the loss cannot see
            dq = (q - z64) @ self.pi  # at a = 1; the long-step result is unchanged. The transient vanishes at v = 0, as dt -> 0
            rho = torch.sqrt((dq * dq).mean(-1, keepdim=True) + 1e-12) - 1e-6   # and for a -> 1; stationary before the
            # projection needs |v_j| <= beta (1 - a_j) RMS(v) for every entry, so beta < 1 forces v = 0 (RMS smooth at 0)
            zr = z64 + a * dq + self.beta * a * (1.0 - a) * rho * torch.tanh(o[:, 2 * S:])
        elif self.vmix:   # z~ = z + V diag(a) V^T (q - z), V = exp(K - K^T) orthogonal (K strictly upper, from the trunk): the
            # relaxation to q along S mixed modes, each with its own dt-free rate (V = I: the per-entry relaxation). z~ = z
            # exactly when q = z, at every dt, so long steps and 1-500 s steps share the settled state
            P_ = S * (S - 1) // 2
            V = skew_exp(o[:, 2 * S:2 * S + P_].float(), self.iu).double()
            a = -torch.expm1(-kdt)
            y = V.transpose(1, 2) @ (q - z64).unsqueeze(-1)
            if self.vnn:   # z~ = z + V U diag(a) U^-1 V^T (q - z): modes along the columns of V U, not orthogonal, so |z~ - q|
                # can grow within a call before it decays (the eigenvalues stay a; q = z still gives z~ = z at every dt)
                U, Ui = unit_upper(o[:, 2 * S + P_:].float(), self.iu)
                y = U.double() @ (a.unsqueeze(-1) * (Ui.double() @ y))
            else:
                y = a.unsqueeze(-1) * y
            zr = z64 + (V @ y).squeeze(-1)
        else:
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
            if self.cont:
                w3, res3 = EP.project_cont(lt, ln0, lr, self.na, self.nh, mol=self.mol)
                w = torch.where(ok, w, w3)
                ok = ok | (res3 < 1e-4).unsqueeze(-1)
        if self.rowcount is not None:
            self.rowcount += torch.logical_not(torch.cat([ok1, ok], -1)).long()
        zp = torch.where(ok, (w / LN10 - self.mu) / self.sd, z64)
        zp = torch.maximum(torch.minimum(zp, self.zhi.double()), self.zlo.double())
        return (zp - z64).float()


HEADS = ("relaxclosedr", "relaxclosedrdt", "relaxclosedrk", "relaxclosedfm", "relaxclosedfmeq", "relaxclosedv",
         "relaxclosedvn")


def build(head, hidden, zlo, zhi, newton=12, fallback="keep", fused=False, latent=512):
    """The model of a config: head "relaxclosedr"; "relaxclosedrdt" (normalized log dt as an extra trunk input, so the
    target and the rate depend on dt); or "relaxclosedrk" (the rate alone depends on dt, through a small zero-initialized
    branch; the target does not, so long steps still go to it); or "relaxclosedfm" (FlowMapTrunk: encoder, latent dynamics
    with dt, decoder; latent = its latent width); or "relaxclosedfmeq" (the hybrid, FlowMapEqTrunk: a dt-free target from
    the encoder, the rate and a bounded transient from the dt-aware latent dynamics; RelaxClosedRHead.delta); or
    "relaxclosedv" (dt-free like "relaxclosedr", with S mixed relaxation modes: RelaxClosedRHead.delta); or "relaxclosedvn"
    (the same with non-normal modes). dt still
    enters the relaxation through a. hidden = the trunk widths; newton, fallback, fused: train.py's closed_iters,
    closed_fallback, fused_proj."""
    if head not in HEADS:
        raise ValueError(f"head {head!r}: one of {HEADS}")
    return RelaxClosedRHead(hidden, zlo, zhi, newton=newton, fallback=fallback, fused=fused,
                            dt_input=head == "relaxclosedrdt", dt_rate=head == "relaxclosedrk",
                            flowmap=head == "relaxclosedfm", latent=latent, flowmap_eq=head == "relaxclosedfmeq",
                            vmix=head == "relaxclosedv", vnn=head == "relaxclosedvn")


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def flops_per_call(m):
    """Multiply-adds x 2 of the Linear layers for one call of delta."""
    return sum(2 * mod.in_features * mod.out_features for mod in m.modules() if isinstance(mod, nn.Linear))
