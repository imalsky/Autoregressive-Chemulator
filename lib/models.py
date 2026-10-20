"""Model heads for the architecture tests. Every head exposes delta(z, dt_norm, g) -> [B, S] fp32:
the change of the standardized log10 state over one call. The caller adds it to the state in the
precision it carries (fp64 or fp32).

H0 delta        production FlowMapMLP (imported): z + MLP(z, dt, g).
H1 relax        z + a * (q - z), a = -expm1(-k dt), q = clamp(z + r(z, g), zlo, zhi), log10 k = c + h tanh(s / h).
                F(z, 0) = z; the fixed point (r = 0) does not depend on dt; a in [0, 1] and q in the
                physical box, so the box [zlo, zhi] (1e-30 <= y <= 1) is invariant.
H2 latentfix    latent relaxation on the common trunk, linear decoder applied to the latent displacement:
                W[alpha * (h_eq - h0)], alpha = -expm1(-k dt) (exactly 0 at dt = 0, well conditioned for tiny k dt).
H3 siblin       matched control for H2 (same trunk, rates and decoder): W h(dt) + b (nonzero at dt = 0).
H3x sibling     port of the sibling LatentLinearFlowMap as is (feedforward SiLU encoder, three linear heads,
                MLP decoder, Xavier init, decoder output x 0.1, rate bias U(-5, 5)): z + D(h(dt)).
relaxproj       relax, then if sum_12 y' > 1: log10 y' -= log10 sum_12 y', clamped at the floor (in-model projection).
relaxsimplex    one relaxation factor per state shared by all species; target from a 13-slot log-softmax
                (12 species + rest) clamped at the floor. sum_12 y <= 1 is invariant (weighted AM-GM).
relax2(r)       two exponentials per species: z + w a_f (q_f - z) + (1 - w) a_s (q_s - z); relax2r shares one target.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
sys.path.insert(0, str(paths.SNAP_SRC))  # pinned snapshot (model.MLP, FlowMapMLP, ResidualMLP)
from model import MLP, FlowMapMLP, ResidualMLP  # noqa: E402

S, G = 12, 2
DT_LOG_MIN, DT_LOG_MAX = -1.0, 3.0  # normalization.json "dt"
EPS_LOG10 = -30.0                   # log10(normalization.epsilon)
LN10 = math.log(10.0)


def dt_log10(dt_norm):
    return DT_LOG_MIN + (DT_LOG_MAX - DT_LOG_MIN) * dt_norm


def mu_sd(zlo, zhi):
    """Per-species log10 mean and std recovered from the box: zlo = (EPS_LOG10 - mu) / sd, zhi = -mu / sd."""
    sd = -EPS_LOG10 / (zhi.double() - zlo.double())
    return (-zhi.double() * sd).float(), sd.float()


class DeltaHead(nn.Module):
    """H0: the production FlowMapMLP; delta = its network output (forward_step adds it to z)."""

    def __init__(self, hidden, **_):
        super().__init__()
        self.m = FlowMapMLP(state_dim=S, global_dim=G, hidden_dims=hidden, activation_name="silu", dropout=0.0,
                            residual=True, predict_delta=True, layer_norm=True, layer_norm_eps=1e-5)

    def delta(self, z, dtn, g):
        return self.m.network(torch.cat([z, dtn.reshape(-1, 1), g], dim=-1))


class RelaxHead(nn.Module):
    """H1: per-species exponential relaxation toward a bounded target q at rate k (both dt-independent)."""

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

    def parts(self, z, g):
        o = self.trunk(torch.cat([z, g], dim=-1))
        q = torch.clamp(z + o[:, :S], self.zlo, self.zhi)
        log10k = self.c + self.h * torch.tanh(o[:, S:] / self.h)
        return q, log10k

    def delta(self, z, dtn, g):
        q, log10k = self.parts(z, g)
        kdt = torch.pow(10.0, log10k + dt_log10(dtn).reshape(-1, 1))
        return -torch.expm1(-kdt) * (q - z)


class RelaxTHead(RelaxHead):
    """Sprint 2026-10-08 arm T: relax plus a zero-initialised side input u [hidden0, 2] on the first layer's pre-activation,
    fed min-max 1000/T and ln T from g's T channel (production normalization; refused under HARNESS_NORM). u is created
    after the trunk without RNG draws, so a same-seed init and the forward at u = 0 equal relax's; a relax checkpoint
    loads strictly once a zero u is added (review_1007/verify-exp-6/relaxT_proto.py)."""

    def __init__(self, hidden, zlo, zhi, **kw):
        import os
        assert not os.environ.get("HARNESS_NORM"), "relaxT reads T from the production normalization"
        import data as D
        super().__init__(hidden, zlo, zhi, **kw)
        st = D.NORM["per_key_stats"]["T"]
        self.tmin, self.tmax = float(st["min"]), float(st["max"])
        self.u1 = (1000.0 / self.tmax, 1000.0 / self.tmin)
        self.u2 = (math.log(self.tmin), math.log(self.tmax))
        self.u = nn.Parameter(torch.zeros(int(hidden[0]), 2))

    def feats(self, g):
        T = self.tmin + (self.tmax - self.tmin) * g[:, 1:2]
        return torch.cat([(1000.0 / T - self.u1[0]) / (self.u1[1] - self.u1[0]),
                          (torch.log(T) - self.u2[0]) / (self.u2[1] - self.u2[0])], -1)

    def parts(self, z, g):
        t = self.trunk
        h = torch.cat([z, g], -1)
        for i, (lin, act, drop, proj, norm) in enumerate(zip(t.linears, t.acts, t.drops, t.projs, t.norms)):
            pre = lin(h) if i else lin(h) + self.feats(g) @ self.u.t()
            h = norm(proj(h) + drop(act(pre)))       # ResidualMLP.forward without autocast
        o = t.out(h)
        q = torch.clamp(z + o[:, :S], self.zlo, self.zhi)
        return q, self.c + self.h * torch.tanh(o[:, S:] / self.h)


class RelaxProjHead(RelaxHead):
    """relax followed by the sum projection: if sum_12 y' > 1, subtract log10 sum_12 y' from every species, then clamp
    at the floor. Same parameters as relax (a relax checkpoint loads as is). The box stays invariant; F(z, 0) = z holds
    only for inputs with sum_12 <= 1, and the fixed point can depend on dt when the target's sum exceeds 1."""

    def __init__(self, hidden, zlo, zhi, **kw):
        super().__init__(hidden, zlo, zhi, **kw)
        mu, sd = mu_sd(zlo, zhi)
        self.register_buffer("mu", mu, persistent=False)
        self.register_buffer("sd", sd, persistent=False)

    def delta(self, z, dtn, g):
        d = super().delta(z, dtn, g)
        zn = z + d
        s = torch.pow(10.0, zn * self.sd + self.mu).sum(-1, keepdim=True)
        zp = torch.maximum(zn - torch.log10(s) / self.sd, self.zlo)
        return torch.where(s > 1.0, zp - z, d)


class RelaxSimplexHead(nn.Module):
    """z' = z + a (q - z) with one relaxation factor a = -expm1(-k dt) per state, shared by all species. The target
    is a log-softmax over 13 slots (12 species + the untracked rest), clamped at the floor; the logits are residual,
    ln y_i + r_i and ln max(1 - sum_12 y, 1e-30) + r_rest, so r = 0 gives q = y / max(sum_12 y, 1) (near-identity
    start). sum_12 q <= 1 (+ at most 12e-30 from the floor clamp), and by weighted AM-GM
    sum_i y_i^(1-a) q_i^a <= (1-a) sum_12 y + a sum_12 q, so sum_12 y <= 1 is invariant. The box is invariant,
    F(z, 0) = z, and the fixed point (q = z) does not depend on dt. The rest slot's dependence on z is detached
    (forward unchanged): its gradient 1/(1 - sum_12 y) is unbounded near sum 1."""

    def __init__(self, hidden, zlo, zhi, logk_range=(-12.0, 3.0), **_):
        super().__init__()
        self.trunk = ResidualMLP(S + G, hidden, S + 2, nn.SiLU, 0.0, layer_norm=True, layer_norm_eps=1e-5)
        with torch.no_grad():
            nn.init.normal_(self.trunk.out.weight, std=1e-3)
            nn.init.zeros_(self.trunk.out.bias)
        self.register_buffer("zlo", zlo.float().clone())
        self.register_buffer("zhi", zhi.float().clone())
        mu, sd = mu_sd(zlo, zhi)
        self.register_buffer("mu", mu, persistent=False)
        self.register_buffer("sd", sd, persistent=False)
        self.c = 0.5 * (logk_range[0] + logk_range[1])
        self.h = 0.5 * (logk_range[1] - logk_range[0])

    def parts(self, z, g):
        o = self.trunk(torch.cat([z, g], dim=-1))
        ly = z * self.sd + self.mu
        rest = torch.log(torch.clamp(1.0 - torch.pow(10.0, ly).sum(-1, keepdim=True), min=1e-30)).detach()
        lq = torch.log_softmax(torch.cat([ly * LN10, rest], dim=-1) + o[:, :S + 1], dim=-1)[:, :S] / LN10
        q = torch.clamp((lq - self.mu) / self.sd, self.zlo, self.zhi)
        log10k = self.c + self.h * torch.tanh(o[:, S + 1:] / self.h)       # [B, 1]: one rate per state
        return q, log10k

    def delta(self, z, dtn, g):
        q, log10k = self.parts(z, g)
        kdt = torch.pow(10.0, log10k + dt_log10(dtn).reshape(-1, 1))
        return -torch.expm1(-kdt) * (q - z)


class Relax2Head(nn.Module):
    """Two exponentials per species: z' = z + w a_f (q_f - z) + (1 - w) a_s (q_s - z), w = sigmoid(.),
    a = -expm1(-k dt), log10 k_s = log10 k_f - softplus(.), targets clamped to the box; relax2r uses one target
    (q_s = q_f). z' is a convex combination of z, q_f and q_s, so the box is invariant; F(z, 0) = z. The fixed point
    is dt-free for relax2r, not in general for relax2."""

    def __init__(self, hidden, zlo, zhi, shared_q=False, logk_range=(-12.0, 3.0), **_):
        super().__init__()
        self.shared_q = bool(shared_q)
        self.trunk = ResidualMLP(S + G, hidden, (4 if self.shared_q else 5) * S, nn.SiLU, 0.0, layer_norm=True,
                                 layer_norm_eps=1e-5)
        with torch.no_grad():
            nn.init.normal_(self.trunk.out.weight, std=1e-3)
            nn.init.zeros_(self.trunk.out.bias)
        self.register_buffer("zlo", zlo.float().clone())
        self.register_buffer("zhi", zhi.float().clone())
        self.c = 0.5 * (logk_range[0] + logk_range[1])
        self.h = 0.5 * (logk_range[1] - logk_range[0])

    def delta(self, z, dtn, g):
        o = self.trunk(torch.cat([z, g], dim=-1))
        qf = torch.clamp(z + o[:, :S], self.zlo, self.zhi)
        j = S if self.shared_q else 2 * S
        qs = qf if self.shared_q else torch.clamp(z + o[:, S:2 * S], self.zlo, self.zhi)
        lkf = self.c + self.h * torch.tanh(o[:, j:j + S] / self.h)
        lks = lkf - nn.functional.softplus(o[:, j + S:j + 2 * S])
        w = torch.sigmoid(o[:, j + 2 * S:j + 3 * S])
        ldt = dt_log10(dtn).reshape(-1, 1)
        af = -torch.expm1(-torch.pow(10.0, lkf + ldt))
        as_ = -torch.expm1(-torch.pow(10.0, lks + ldt))
        return w * af * (qf - z) + (1.0 - w) * as_ * (qs - z)


class LatentHead(nn.Module):
    """H2 (fixed=True) and its matched control H3 siblin (fixed=False): latent relaxation
    h(dt) = h_eq + exp(-k dt)(h0 - h_eq) on the common trunk with a linear decoder."""

    def __init__(self, hidden, latent, fixed, rate_seed=0, **_):
        super().__init__()
        self.L, self.fixed = int(latent), bool(fixed)
        self.trunk = ResidualMLP(S + G, hidden, 3 * self.L, nn.SiLU, 0.0, layer_norm=True, layer_norm_eps=1e-5)
        self.W = nn.Linear(self.L, S, bias=not self.fixed)
        with torch.no_grad():
            gen = torch.Generator().manual_seed(int(rate_seed))  # sibling: rate bias U(-5, 5), seeded
            self.trunk.out.bias[2 * self.L:] = torch.empty(self.L).uniform_(-5.0, 5.0, generator=gen)
            nn.init.normal_(self.W.weight, std=1e-3)
            if self.W.bias is not None:
                nn.init.zeros_(self.W.bias)

    def delta(self, z, dtn, g):
        o = self.trunk(torch.cat([z, g], dim=-1))
        h0, heq, log10k = o[:, :self.L], o[:, self.L:2 * self.L], o[:, 2 * self.L:]
        kdt = torch.pow(10.0, torch.clamp(log10k + dt_log10(dtn).reshape(-1, 1), max=3.0))  # sibling clamp
        if self.fixed:
            return self.W(-torch.expm1(-kdt) * (heq - h0))
        return self.W(heq + torch.exp(-kdt) * (h0 - heq))


class LatentBoxHead(LatentHead):
    """H4: latentfix with the physical box made invariant. The decoded per-mode amplitudes c_sm = W_sm a_m
    (a = h_eq - h0) are scaled so that their positive parts sum to at most zhi - z and their negative parts to at
    most z - zlo; since every relaxation factor alpha_m lies in [0, 1], z + delta stays inside [zlo, zhi]."""

    def __init__(self, hidden, latent, zlo, zhi, rate_seed=0, **_):
        super().__init__(hidden, latent, fixed=True, rate_seed=rate_seed)
        self.register_buffer("zlo", zlo.float().clone())
        self.register_buffer("zhi", zhi.float().clone())

    def delta(self, z, dtn, g):
        o = self.trunk(torch.cat([z, g], dim=-1))
        h0, heq, log10k = o[:, :self.L], o[:, self.L:2 * self.L], o[:, 2 * self.L:]
        alpha = -torch.expm1(-torch.pow(10.0, torch.clamp(log10k + dt_log10(dtn).reshape(-1, 1), max=3.0)))
        c = self.W.weight.unsqueeze(0) * (heq - h0).unsqueeze(1)          # [B, S, L]
        cp, cn = torch.relu(c), torch.relu(-c)
        P, N = cp.sum(-1), cn.sum(-1)
        up, dn = torch.clamp(self.zhi - z, min=0.0), torch.clamp(z - self.zlo, min=0.0)
        sp = torch.where(P > up, up / torch.clamp(P, min=1e-30), torch.ones_like(P))
        sn = torch.where(N > dn, dn / torch.clamp(N, min=1e-30), torch.ones_like(N))
        a = alpha.unsqueeze(1)
        return (a * cp).sum(-1) * sp - (a * cn).sum(-1) * sn


class SiblingAsIs(nn.Module):
    """H3x: the sibling LatentLinearFlowMap (ref_Chemulator src/model.py) with predict_delta, MLP decoder."""

    def __init__(self, enc_hidden, latent, dec_hidden, rate_seed=0, **_):
        super().__init__()
        self.L = int(latent)
        layers, d = [], S + G
        for w in enc_hidden:
            layers += [nn.Linear(d, w), nn.SiLU()]
            d = w
        self.encoder_trunk = nn.Sequential(*layers)
        self.head_h0, self.head_heq, self.head_rate = nn.Linear(d, self.L), nn.Linear(d, self.L), nn.Linear(d, self.L)
        dl, d = [], self.L
        for w in dec_hidden:
            dl += [nn.Linear(d, w), nn.SiLU()]
            d = w
        dl.append(nn.Linear(d, S))
        self.decoder = nn.Sequential(*dl)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        with torch.no_grad():
            gen = torch.Generator().manual_seed(int(rate_seed))
            self.head_rate.bias.uniform_(-5.0, 5.0, generator=gen)
            self.decoder[-1].weight.mul_(0.1)

    def delta(self, z, dtn, g):
        f = self.encoder_trunk(torch.cat([z, g], dim=-1))
        h0, heq, log10k = self.head_h0(f), self.head_heq(f), self.head_rate(f)
        decay = torch.exp(-torch.pow(10.0, torch.clamp(log10k + dt_log10(dtn).reshape(-1, 1), max=3.0)))
        return self.decoder(heq + decay * (h0 - heq))


class _RelaxOut(nn.Module):
    """Shared relaxation output for E10 trunks: o [B, S, 2] -> delta, same law as RelaxHead."""

    def __init__(self, zlo, zhi, logk_range=(-12.0, 3.0)):
        super().__init__()
        self.register_buffer("zlo", zlo.float().clone())
        self.register_buffer("zhi", zhi.float().clone())
        self.c = 0.5 * (logk_range[0] + logk_range[1])
        self.h = 0.5 * (logk_range[1] - logk_range[0])

    def forward(self, z, o, dtn):
        q = torch.clamp(z + o[..., 0], self.zlo, self.zhi)
        log10k = self.c + self.h * torch.tanh(o[..., 1] / self.h)
        return -torch.expm1(-torch.pow(10.0, log10k + dt_log10(dtn).reshape(-1, 1))) * (q - z)


class _Block(nn.Module):
    """Pre-LN transformer block over the 12 species tokens (4 heads, SiLU MLP of width 2d)."""

    def __init__(self, d, nhead=4):
        super().__init__()
        self.nhead = nhead
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.nhead, d // self.nhead).permute(2, 0, 3, 1, 4)
        a = nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(a)
        return x + self.ff(self.ln2(x))


class RelaxAttnHead(nn.Module):
    """E10 relaxattn: the relaxation law with a species-token transformer trunk. Token i = W z_i + e_i + W_g g
    (species embedding e_i, globals added to every token); hidden = [d] * L gives L blocks of width d; a per-token
    linear map gives (r_i, s_i). Same box invariance, F(z, 0) = z and dt-free fixed point as relax."""

    def __init__(self, hidden, zlo, zhi, **_):
        super().__init__()
        d = int(hidden[0])
        self.zin, self.gin = nn.Linear(1, d), nn.Linear(G, d)
        self.emb = nn.Parameter(0.02 * torch.randn(S, d))
        self.blocks = nn.ModuleList(_Block(d) for _ in hidden)
        self.lnf, self.out = nn.LayerNorm(d), nn.Linear(d, 2)
        with torch.no_grad():
            nn.init.normal_(self.out.weight, std=1e-3)
            nn.init.zeros_(self.out.bias)
        self.relax = _RelaxOut(zlo, zhi)

    def delta(self, z, dtn, g):
        x = self.zin(z.unsqueeze(-1)) + self.emb + self.gin(g).unsqueeze(1)
        for blk in self.blocks:
            x = blk(x)
        return self.relax(z, self.out(self.lnf(x)), dtn)


class _RBFKAN(nn.Module):
    """FastKAN layer (Li 2024, arXiv 2405.06721): LayerNorm, Gaussian RBF basis on a fixed grid per input, a linear map
    of the basis plus a linear map of SiLU(x)."""

    def __init__(self, din, dout, n_basis=8, lo=-2.0, hi=2.0):
        super().__init__()
        self.ln = nn.LayerNorm(din)
        self.register_buffer("c", torch.linspace(lo, hi, n_basis))
        self.inv_h = (n_basis - 1) / (hi - lo)
        self.spline = nn.Linear(din * n_basis, dout, bias=False)
        self.base = nn.Linear(din, dout)

    def forward(self, x):
        u = self.ln(x)
        phi = torch.exp(-((u.unsqueeze(-1) - self.c) * self.inv_h) ** 2).flatten(-2)
        return self.spline(phi) + self.base(nn.functional.silu(x))


class RelaxKANHead(nn.Module):
    """E10 relaxkan: the relaxation law with a FastKAN trunk, widths (S + G) -> hidden... -> 2S."""

    def __init__(self, hidden, zlo, zhi, **_):
        super().__init__()
        dims = [S + G] + [int(h) for h in hidden]
        self.layers = nn.ModuleList(_RBFKAN(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = _RBFKAN(dims[-1], 2 * S)
        with torch.no_grad():
            nn.init.normal_(self.out.spline.weight, std=1e-3)
            nn.init.normal_(self.out.base.weight, std=1e-3)
            nn.init.zeros_(self.out.base.bias)
        self.relax = _RelaxOut(zlo, zhi)

    def delta(self, z, dtn, g):
        x = torch.cat([z, g], dim=-1)
        for lay in self.layers:
            x = lay(x)
        return self.relax(z, self.out(x).view(-1, 2, S).transpose(1, 2), dtn)


class RelaxFiLMHead(RelaxHead):
    """E11 relaxfilm(c): the relax head with FiLM conditioning on the globals (Perez et al. 2018, arXiv 1709.07871).
    Each hidden layer's pre-activation is scaled and shifted per unit, act((1 + gamma_l(g)) * (W h + b) + beta_l(g)),
    with (gamma, beta) for all layers from one SiLU MLP of g (2 -> emb -> emb -> 2 sum(hidden)) whose last layer starts
    at zero, so the model starts as the plain trunk. relaxfilm feeds the trunk z only (FiLM replaces the g concat);
    relaxfilmc keeps the concat (same trunk init as relax for the same seed)."""

    def __init__(self, hidden, zlo, zhi, concat=False, emb=32, **_):
        super().__init__(hidden, zlo, zhi)
        self.concat = concat
        if not concat:
            self.trunk = ResidualMLP(S, hidden, 2 * S, nn.SiLU, 0.0, layer_norm=True, layer_norm_eps=1e-5)
            with torch.no_grad():
                nn.init.normal_(self.trunk.out.weight, std=1e-3)
                nn.init.zeros_(self.trunk.out.bias)
        self.widths = [int(h) for h in hidden]
        self.cond = nn.Sequential(nn.Linear(G, emb), nn.SiLU(), nn.Linear(emb, emb), nn.SiLU(),
                                  nn.Linear(emb, 2 * sum(self.widths)))
        with torch.no_grad():
            nn.init.zeros_(self.cond[-1].weight)
            nn.init.zeros_(self.cond[-1].bias)

    def parts(self, z, g):
        t, gb = self.trunk, self.cond(g)
        h, o = (torch.cat([z, g], dim=-1) if self.concat else z), 0
        for lin, act, proj, norm, n in zip(t.linears, t.acts, t.projs, t.norms, self.widths):
            h = norm(proj(h) + act(lin(h) * (1.0 + gb[:, o:o + n]) + gb[:, o + n:o + 2 * n]))
            o += 2 * n
        out = t.out(h)
        q = torch.clamp(z + out[:, :S], self.zlo, self.zhi)
        log10k = self.c + self.h * torch.tanh(out[:, S:] / self.h)
        return q, log10k


SPECIES = None  # E13: the species names (set with S by train.py / lhbench.py from the normalization file)


class RelaxClosedHead(RelaxHead):
    """E13 relaxclosed: relax (same parameters), then the update is closed. The relaxed state y~ is moved to the
    input's element ratios to H (C/H, N/H, O/H, He/H over every tracked species) by y' = y~ exp(A lam), lam from a
    damped Newton solve (elemproj.project_ln: 12 unrolled fp64 iterations, differentiable), and normalised, so sum
    y' = 1 and the input's element ratios are kept (up to the floor clamp and fp32 updates) call after call. Rows whose
    solve has not converged (max |log-ratio residual| > 1e-4) take the normalised y~ instead (sum 1, ratios not
    enforced). Meant for a complete species set (all 53 network species, He included); with r = 0 the input is
    returned (lam = 0). (Not a minimum-KL projection: exp(A lam) is the multiplicative family that meets the ratios.)"""

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

    def delta(self, z, dtn, g):
        import elemproj as EP  # not stored on the module: the EMA copy deep-copies it
        zr = z + super().delta(z, dtn, g)
        ln0 = (z.double() * self.sd + self.mu) * LN10                       # input, ln y
        lt = (zr.double() * self.sd + self.mu) * LN10                       # relaxed, ln y~
        lr = EP.log_ratios(ln0, self.na, self.nh)                            # [B, 4] the input's ratios
        w, res = EP.project_ln(lt, lr, self.na, self.nh, iters=self.newton)
        ok = (res < 1e-4).unsqueeze(-1)                                      # NaN residual -> not ok
        w = torch.where(ok, w, lt - torch.logsumexp(lt, -1, keepdim=True))
        zp = (w / LN10 - self.mu) / self.sd
        zp = torch.maximum(torch.minimum(zp, self.zhi.double()), self.zlo.double())
        return (zp - z.double()).float()


class RelaxClosedRHead(RelaxClosedHead):
    """E13 relaxclosedr: relaxclosed with (1) the relaxed state formed in fp64 (relaxclosed adds the relax increment to
    the fp32 input before projecting, which erased increments below the fp32 spacing in about half the slowly moving
    entries: Codex review, 2026-10-06 evening), (2) sum 1 over molecules only (elemproj.molidx; reservoir sets carry
    atom reservoirs C_res, N_res, O_res, H_res and the untracked molecule count M_res), (3) rows whose projection has
    not converged keep their input (no change) instead of the unprojected relaxed state. Same parameters as relax."""

    def __init__(self, hidden, zlo, zhi, newton=12, fallback="keep", fused=False, **kw):
        super().__init__(hidden, zlo, zhi, newton=newton, **kw)
        import elemproj as EP
        self.register_buffer("mol", EP.molidx(SPECIES), persistent=False)
        self.fused = bool(fused)  # 2026-10-17 (fuse study): elemproj.project_ln_fused instead of project_ln (same mathematics)
        # 2026-10-12 (fixes phase): fallback 'lm' re-solves rows whose Newton projection fails with elemproj.project_lm
        # (same solution family; converged rows bitwise unchanged); 'keep' = the old rule (the row keeps its input, which
        # froze sealed H2 jobs 606 / 218 for the whole run). rowcount: optional int64 [B, 2] that the bench allocates; per
        # call it adds 1 where the Newton solve failed (column 0) and where the row still kept its input (column 1).
        self.fallback, self.lm_iters, self.rowcount = fallback, 32, None

    def delta(self, z, dtn, g):
        import elemproj as EP
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


def build(head, hidden, zlo, zhi, latent=None, dec_hidden=None, newton=12, fallback="keep", fused=False):
    """newton (2026-10-09, train.py closed_iters): the relaxclosed / relaxclosedr projection's Newton iterations.
    fallback (2026-10-12, train.py closed_fallback): relaxclosedr's rule for rows whose projection fails ('keep' | 'lm').
    fused (2026-10-17, train.py fused_proj): relaxclosedr's Newton projection as elemproj.project_ln_fused."""
    if fused and head != "relaxclosedr":
        raise ValueError("fused_proj applies to the relaxclosedr head only")
    if head == "delta":
        return DeltaHead(hidden)
    if head == "relaxclosed":
        return RelaxClosedHead(hidden, zlo, zhi, newton=newton)
    if head == "relaxclosedr":
        return RelaxClosedRHead(hidden, zlo, zhi, newton=newton, fallback=fallback, fused=fused)
    if head == "relax":
        return RelaxHead(hidden, zlo, zhi)
    if head == "relaxT":  # sprint 2026-10-08 arm T
        return RelaxTHead(hidden, zlo, zhi)
    if head in ("latentfix", "siblin"):
        return LatentHead(hidden, latent, fixed=(head == "latentfix"))
    if head == "sibling":
        return SiblingAsIs(hidden, latent, dec_hidden)
    if head == "latentbox":
        return LatentBoxHead(hidden, latent, zlo, zhi)
    if head == "relaxproj":
        return RelaxProjHead(hidden, zlo, zhi)
    if head == "relaxsimplex":
        return RelaxSimplexHead(hidden, zlo, zhi)
    if head in ("relax2", "relax2r"):
        return Relax2Head(hidden, zlo, zhi, shared_q=(head == "relax2r"))
    if head == "relaxattn":
        return RelaxAttnHead(hidden, zlo, zhi)
    if head == "relaxkan":
        return RelaxKANHead(hidden, zlo, zhi)
    if head in ("relaxfilm", "relaxfilmc"):
        return RelaxFiLMHead(hidden, zlo, zhi, concat=(head == "relaxfilmc"))
    raise ValueError(head)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def flops_per_call(m):
    """Multiply-adds x 2 of the Linear layers for one call of delta (latentfix decodes twice)."""
    return sum(2 * mod.in_features * mod.out_features for mod in m.modules() if isinstance(mod, nn.Linear))


def _check():
    zlo, zhi = -torch.ones(S) * 5, torch.ones(S)
    for hd in ("delta", "relax", "latentfix", "siblin", "sibling"):
        m = build(hd, [64, 64], zlo, zhi, latent=8, dec_hidden=[32])
        z, g = torch.zeros(4, S), torch.rand(4, G)
        d = m.delta(z, torch.rand(4), g)
        assert d.shape == (4, S), hd
    m = build("relax", [64, 64], zlo, zhi)
    with torch.no_grad():
        m.trunk.out.weight.normal_(std=1.0)
    z = torch.rand(256, S) * 6 - 5
    d = m.delta(z, torch.zeros(256), torch.rand(256, G))
    zz = z + d
    assert bool(((zz >= zlo - 1e-5) & (zz <= zhi + 1e-5)).all()), "relax box"
    m = build("latentbox", [64, 64], zlo, zhi, latent=8)
    with torch.no_grad():
        m.W.weight.normal_(std=3.0)
        m.trunk.out.weight.normal_(std=1.0)
    z = torch.rand(512, S) * 6 - 5
    for dn in (0.0, 0.5, 1.0, 2.0):
        zz = z + m.delta(z, torch.full((512,), dn), torch.rand(512, G)).detach()
        assert bool(((zz >= zlo - 1e-5) & (zz <= zhi + 1e-5)).all()), "latentbox box"
    m = build("latentfix", [64, 64], zlo, zhi, latent=8)
    with torch.no_grad():
        m.trunk.out.bias[16:] = -12.0  # tiny rates: the update must still scale with dt, and be 0 at dt_norm -> -inf
    z, g = torch.rand(8, S), torch.rand(8, G)
    d1 = m.delta(z, torch.zeros(8), g).detach()
    d2 = m.delta(z, torch.full((8,), 0.25), g).detach()
    assert float((d2 / d1).mean()) > 9.0, "latentfix tiny-rate scaling"
    _check_constraint_heads()
    print("models ok", math.isfinite(float(d.detach().abs().max())))


def _simplex_states(n, gen):
    """n log10 compositions with sum_12 <= 1 spread over the box: 13 slots log-uniform in [-30, 0], normalized; a
    quarter with the rest slot at the floor (sum_12 ~ 1), a quarter dominated by one species."""
    ly = torch.rand(n, S + 1, generator=gen, dtype=torch.float64) * 30.0 - 30.0
    ly[: n // 4, S] = -30.0
    ly[n // 4: n // 2, torch.randint(0, S, (1,), generator=gen)] = 0.0
    ly = ly - torch.log10(torch.pow(10.0, ly).sum(-1, keepdim=True))
    return torch.clamp(ly[:, :S], min=EPS_LOG10)


def _check_constraint_heads(n=100_000):
    """relaxsimplex: sum_12 <= 1 on n random states with random weights at dt_norm 0, 0.5, 1, 2 and random (fp64 head:
    exact arithmetic up to rounding; fp32 head as deployed, fp64 state as P3); box invariance on any box state;
    identity as dt_norm -> -10. relaxproj: box, sum after the update, equality with relax where no projection fires.
    relax2 / relax2r: box and identity. Uses the production normalization."""
    import data as D
    mu, sd, zlo, zhi = D.norm_consts(torch.device("cpu"))
    gen = torch.Generator().manual_seed(0)
    hid = [64, 64]
    ly = _simplex_states(n, gen)
    z = (ly - mu) / sd
    g = torch.rand(n, G, generator=gen)
    zbox = zlo + (zhi - zlo) * torch.rand(n, S, generator=gen, dtype=torch.float64)   # any box state (sum may be > 1)
    dts = [torch.full((n,), v) for v in (0.0, 0.5, 1.0, 2.0)] + [torch.rand(n, generator=gen) * 2.0]
    rep = {}
    for std in (1.0, 3.0):
        torch.manual_seed(int(std))
        m = build("relaxsimplex", hid, zlo, zhi)
        with torch.no_grad():
            m.trunk.out.weight.normal_(std=std)
            m.trunk.out.bias.normal_(std=std)
        m64 = build("relaxsimplex", hid, zlo, zhi).double()
        m64.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
        m64.mu, m64.sd = mu.clone(), sd.clone()
        with torch.no_grad():
            for i, dtn in enumerate(dts):
                z64 = z + m64.delta(z, dtn.double(), g.double())
                z32 = z + m.delta(z.float(), dtn, g).double()
                s64 = torch.pow(10.0, z64 * sd + mu).sum(-1)
                s32 = torch.pow(10.0, z32 * sd + mu).sum(-1)
                zb = zbox + m.delta(zbox.float(), dtn, g).double()
                rep[f"std{std:g}_dt{i}"] = {"max_sum_minus_1_fp64": float(s64.max() - 1.0), "max_sum_minus_1_fp32": float(s32.max() - 1.0),
                                            "box_over": float(torch.maximum(zb - zhi, zlo - zb).max())}
                assert float(s64.max()) <= 1.0 + 1e-10, ("relaxsimplex sum fp64", float(s64.max()))
                assert float(s32.max()) <= 1.0 + 1e-4, ("relaxsimplex sum fp32", float(s32.max()))
                assert bool(((zb >= zlo - 1e-5) & (zb <= zhi + 1e-5)).all()), "relaxsimplex box"
            d0 = m.delta(z.float(), torch.full((n,), -10.0), g)
            assert float(d0.abs().max()) < 1e-6, "relaxsimplex identity at dt -> 0"
    # relaxproj: same weights as a relax head
    r = build("relax", hid, zlo, zhi)
    with torch.no_grad():
        r.trunk.out.weight.normal_(std=1.0)
    p = build("relaxproj", hid, zlo, zhi)
    p.load_state_dict(r.state_dict())
    with torch.no_grad():
        for dtn in dts:
            dp, dr = p.delta(zbox.float(), dtn, g), r.delta(zbox.float(), dtn, g)
            zp = zbox + dp.double()
            sp = torch.pow(10.0, zp * sd + mu).sum(-1)
            sr = torch.pow(10.0, (zbox + dr.double()) * sd + mu).sum(-1)
            assert bool(((zp >= zlo - 1e-5) & (zp <= zhi + 1e-5)).all()), "relaxproj box"
            assert float(sp.max()) <= 1.0 + 1e-4, ("relaxproj sum", float(sp.max()))
            below = sr < 1.0 - 1e-5                   # clearly no projection: the delta must equal relax's bit for bit
            assert bool(below.any()) and bool((dp[below] == dr[below]).all()), "relaxproj != relax without projection"
            rep.setdefault("relaxproj_fired_frac", []).append(float((sr > 1.0).double().mean()))
            rep.setdefault("relaxproj_max_sum_minus_1", []).append(float(sp.max() - 1.0))
    for hd in ("relax2", "relax2r"):
        m = build(hd, hid, zlo, zhi)
        with torch.no_grad():
            m.trunk.out.weight.normal_(std=1.0)
            for dtn in dts:
                zb = zbox + m.delta(zbox.float(), dtn, g).double()
                assert bool(((zb >= zlo - 1e-5) & (zb <= zhi + 1e-5)).all()), f"{hd} box"
            assert float(m.delta(zbox.float(), torch.full((n,), -10.0), g).abs().max()) < 1e-6, f"{hd} identity"
    print("constraint heads ok", n, "cases", rep)


if __name__ == "__main__":
    _check()
