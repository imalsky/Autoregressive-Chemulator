"""Long-horizon benchmark (frozen before comparisons; see the plan).

    python lhbench.py <run_dir|identity|oracle> <out_dir> <level S|F> [--ckpt stage2.pt] [--arms A,B,C,D]

Rows are test-split sources (stratified by T x log P, uncut trajectories only) x start ages. Every
row is advanced in three precision modes inside one batched network call (network fp32, TF32 off):
  P1  host state y fp32; each call z = (log10 max(y, eps) - mu) / sd, z' = z + delta, y = 10^(z' sd + mu)
      (the production export's arithmetic, testing/export.py normalize/denormalize_species);
  P2  host state y fp32; z as P1; x = delta * sd * ln 10, yb = max(y, eps); y' = yb + yb * expm1(x) for
      |x| < 1, else yb * exp(x) (no round trip of the state; exact passthrough at a zero update for y >= eps);
  P3  host state z fp64; z' = z + fp64(delta).
A guard runs in every mode: a candidate state that is non-finite or outside log10 y in [-40, 0.05]
is rejected (the row keeps its previous state) and counted. The state of every row is recorded at
log-spaced checkpoints; scoring is offline (score.py).

Evaluation-only extensions (all off by default; the default arms are unchanged):
  --modes P2P3P4  P4 = host state y fp64 with the P2 multiplicative update in fp64 (states slot 3; slot 0 NaN).
  --renorm        after every call (P3 only): if sum_12 y > 1, subtract log10 sum_12 y from every species, then
                  clamp at the floor (host-side projection); per-row counts are saved as 'nren'.
  --substeps n    every host call of dt is made as n model calls of dt / n (arms A, B, C, E, V, H; not D, G).
  model 'ensemble:<run1>,<run2>'  delta = mean of the members' deltas (both loaded with --ckpt).
  model 'ensmed:<run1>,<run2>,...'  per-entry median of the members' next states, projected once onto the input's
                  element ratios (relaxclosedr members; --closed-iters applies to every member and the projection).
  arm V           A-arm rows and starts, dt changing per call: --v-schedule rand (log-uniform in [1, 100] s per row
                  and call, seeded in-graph) | alt (1, 10, 100 s cycled) | const<dt>; run to --v-horizon seconds
                  of mean simulated time (default the level's A dt 10 horizon); per-row elapsed time saved as 'tcum'.
  arm H           starts from a VULCAN tag's own states (--h-tag: a name under vulcan/ or an absolute directory
                  holding jobs.json and <i>.npz) at --h-starts, fixed dt --h-dt, --h-calls calls (default the
                  level's A cap); g from the jobs' P, T; truth = the same tag (score.py).
  arm G           stress dose-response: arm D's perturbation with per-group amplitudes (blend f uniform in
                  [0.1 b, b], P/T walk s.d. s per call, zero-injection rate z; see GROUPS), --g-calls stress calls,
                  then --g-recover calls with perturbation off and g frozen, plus a clean twin of every perturbed
                  row (same start, g fixed at the row's final g, same host scheme, no perturbation). --g-cache <dir>
                  (2026-10-09): the arm's sources come from that cache's test split (default: data.CACHE, as before).
  --elemproj m    (2026-10-09, P3 only) after every call, elemproj12.project of the 12 tracked species onto the element
                  ratios of that call's input (m = ratio: C/H, O/H, N/H; ratio_he: also H / (1 - sum12)), clamped at the
                  floor; per-row counts of calls whose residual exceeds 1e-6 are saved as 'nproj_bad'. Arm G applies it
                  before its own host step (tracking, renormalization).
  arm P mixing    (2026-10-09) paths whose npz has mix_k (paths_v3): before call k, the recorded GCM-like mix of the model
                  state, y <- (1 - f) y + f y_partner in linear mixing ratio (train.py's chain term uses the same rule,
                  chains.mix: P3 state through y, floored at eps; fp32 / fp64 host states blended directly, floored at
                  eps); the call's input (--elemproj target) is the mixed state; meta.json records it ('mixing').
"""
import argparse
import json
import os
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import data as D  # noqa: E402
import models as M  # noqa: E402
import chains as C  # noqa: E402
M.S, M.SPECIES = len(D.SHORT), list(D.SHORT)  # E13: the species set follows the normalization file

ENV_LO, ENV_HI = -40.0, 0.05            # physical envelope (log10 mixing ratio)
T_EDGES = [300, 800, 1300, 1800, 2300, 3001]
LOGP_EDGES = [0.0, 2.5, 5.0, 7.5, 10.01]
STARTS = (1e-3, 1e3, 1e6)               # start ages (s)
HORIZON_S = 8.64e7                      # 1000 model days
SAMPLE_SEED = 20261001
LEVELS = {  # calls per arm; transient horizons are min(cap, horizon / dt)
    "Q": {"n_src": 500, "A": {10.0: 20000}, "B": 20000, "C": 1000, "D": 20000},  # timing only
    "ST": {"A": {1.0: 2000, 10.0: 2000, 100.0: 2000}, "B": 2000, "C": 1000, "D": 2000},  # S sources, code test only
    "T": {"n_src": 100, "A": {1.0: 3000, 10.0: 3000, 100.0: 3000}, "B": 3000, "C": 1000, "D": 3000},  # code test only
    "S": {"n_src": 512, "A": {1.0: 100_000, 10.0: 1_000_000, 100.0: 864_000}, "B": 1_000_000, "C": 100_000, "D": 200_000},
    "SP": {"A": {1.0: 100_000, 10.0: 1_000_000, 100.0: 864_000}, "B": 1_000_000, "C": 100_000, "D": 200_000},
    "F": {"n_src": 1024, "A": {1.0: 10_000_000, 10.0: 8_640_000, 100.0: 864_000}, "B": 10_000_000, "C": 100_000, "D": 1_000_000},
    # production-size finalists: the SP sources with the F horizons
    "SPF": {"A": {1.0: 10_000_000, 10.0: 8_640_000, 100.0: 864_000}, "B": 10_000_000, "C": 100_000, "D": 1_000_000},
}
BLOCK = int(os.environ.get("LHB_BLOCK", "250"))  # calls per CUDA graph; 2026-10-11: LHB_BLOCK lowers it (ensmed H benches ran out of memory)
B_START_S = 1e8                         # late-hold start time (arm B)
V_SEED = SAMPLE_SEED + 11               # arm V 'rand' schedule
G_SEED = SAMPLE_SEED + 1                # arm G (as arm D)
# arm G groups: (blend max b, P/T walk s.d. per call, zero-injection rate, renormalize); G4 = campaign arm D
GROUPS = {"G0": (0.0, 0.0, 0.0, False), "G1": (1e-3, 2e-5, 0.0, False), "G2": (1e-2, 2e-4, 0.0, False),
          "G3": (0.10, 2e-3, 0.0, False), "G4": (0.10, 2e-3, 1e-3, False), "G5": (1e-2, 2e-4, 1e-5, False),
          "G2r": (1e-2, 2e-4, 0.0, True), "G3r": (0.10, 2e-3, 0.0, True), "G4r": (0.10, 2e-3, 1e-3, True)}


def _strat(test, per, seed, exclude=None):
    rng = np.random.default_rng(seed)
    T, lP = test.T, np.log10(test.P)
    ok = test.end.cpu().numpy() == 99
    if exclude is not None:
        ok[exclude] = False
    pick = []
    for a in range(len(T_EDGES) - 1):
        for b in range(len(LOGP_EDGES) - 1):
            cell = np.flatnonzero(ok & (T >= T_EDGES[a]) & (T < T_EDGES[a + 1]) & (lP >= LOGP_EDGES[b]) & (lP < LOGP_EDGES[b + 1]))
            pick.append(rng.choice(cell, size=min(per, cell.size), replace=False))
    return np.sort(np.concatenate(pick))


def select_sources(test, level):
    """Stratified, seeded, uncut (end == 99) test sources per (T, log P) cell. The S set (25 per cell) is a
    subset of the F set (S plus 26 more per cell), so the VULCAN references of S serve both."""
    if level == "T":
        return _strat(test, 5, SAMPLE_SEED)
    base = _strat(test, 25, SAMPLE_SEED)
    if level in ("SP", "SPF"):  # production-size models: 5 of the 25 S sources per cell (a subset, so S scores compare)
        rng = np.random.default_rng(SAMPLE_SEED + 3)
        T, lP = test.T[base], np.log10(test.P[base])
        pick = []
        for a in range(len(T_EDGES) - 1):
            for b in range(len(LOGP_EDGES) - 1):
                cell = base[(T >= T_EDGES[a]) & (T < T_EDGES[a + 1]) & (lP >= LOGP_EDGES[b]) & (lP < LOGP_EDGES[b + 1])]
                pick.append(rng.choice(cell, size=min(5, cell.size), replace=False))
        return np.sort(np.concatenate(pick))
    if level != "F":
        return base
    return np.sort(np.concatenate([base, _strat(test, 26, SAMPLE_SEED + 7, exclude=base)]))


VREF = paths.VREF


def vref_index(tag):
    """{job index: job} for jobs whose output exists and reached the requested runtime (end_case 1 or 2 at
    t >= runtime; with trun_min = runtime the convergence stop cannot fire earlier). tag may be an absolute
    directory (VREF / absolute path is that path)."""
    jobs = json.loads((VREF / tag / "jobs.json").read_text())
    out = {}
    for i, j in enumerate(jobs):
        f = VREF / tag / f"{i}.npz"
        if f.exists() and float(np.load(f)["t"][-1]) >= float(j["cfg"]["runtime"]) * (1 - 1e-9):
            out[i] = j
    return out


def vref_logy(tag, i, times):
    """log10 of the 12 tracked mixing ratios (Auto-Chem order) at times (s), linear in t between VULCAN steps."""
    r = np.load(VREF / tag / f"{i}.npz")
    sp, t, x = list(r["species"]), r["t"], r["x"]  # read each array once (r["x"] decompresses on every access)
    x = D.take(np.stack([np.interp(times, t, x[:, k]) for k in range(len(sp))], -1), sp)  # E13: reservoir sets
    return np.log10(np.maximum(x, D.EPS))


def mix_logy(mix):
    """log10 of the 12 tracked species of a VULCAN start composition after VULCAN's normalization."""
    tot = sum(mix.values())
    return np.log10(np.maximum(np.array([mix[s] / tot for s in D.SHORT]), D.EPS))


class Identity(torch.nn.Module):
    def delta(self, z, dtn, g):
        return torch.zeros_like(z)


class Oracle(torch.nn.Module):
    """delta = z_true(t0 + (n + 1) dt) - z for every mode block (arm A only)."""

    def __init__(self, test, ridx, t0, dt):
        super().__init__()
        self.test, self.ridx, self.t0, self.dt, self.engine = test, ridx, t0, dt, None

    def delta(self, z, dtn, g):
        n = self.engine.ncall.double()
        tt = (self.t0 + (n + 1.0) * self.dt).reshape(-1, 1)
        zt = self.test.z_of(self.test.logy_at(self.ridx, tt))[:, 0].float()
        return zt.repeat(3, 1) - z


class Ensemble(torch.nn.Module):
    """delta = mean of the members' deltas (each member keeps the box invariant, so the mean does too)."""

    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def delta(self, z, dtn, g):
        d = self.members[0].delta(z, dtn, g)
        for m in self.members[1:]:
            d = d + m.delta(z, dtn, g)
        return d / len(self.members)


class EnsembleMedian(torch.nn.Module):
    """2026-10-11 (author: "average of experts"; Codex review): per-entry median of the members' next states (closed
    heads, same entry set), then one projection onto the input's element ratios with the first member's machinery
    (relaxclosedr's rule: Newton in fp64, molecules sum to 1, a row whose solve does not converge keeps its input).
    self.nfail counts those rows (a buffer, so CUDA-graph replays accumulate it)."""

    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)
        self.register_buffer("nfail", torch.zeros((), dtype=torch.int64))

    def delta(self, z, dtn, g):
        import elemproj as EP
        h = self.members[0]
        z64 = z.double()
        zs = torch.stack([z64 + m.delta(z, dtn, g).double() for m in self.members])
        zm = zs.median(0).values
        ln0 = (z64 * h.sd + h.mu) * M.LN10
        lt = (zm * h.sd + h.mu) * M.LN10
        lr = EP.log_ratios(ln0, h.na, h.nh)
        w, res = EP.project_ln(lt, lr, h.na, h.nh, iters=h.newton, mol=h.mol)
        ok = (res < 1e-4).unsqueeze(-1)
        ok1 = ok
        if h.fallback == "lm":  # 2026-10-12: the members' fallback rule applies to the median's projection too
            w2, res2 = EP.project_lm(lt, lr, h.na, h.nh, iters=h.lm_iters, mol=h.mol)
            w = torch.where(ok, w, w2)
            ok = ok | (res2 < 1e-4).unsqueeze(-1)
        if getattr(self, "rowcount", None) is not None:
            self.rowcount += torch.logical_not(torch.cat([ok1, ok], -1)).long()
        self.nfail += torch.logical_not(ok).sum()
        zp = torch.where(ok, (w / M.LN10 - h.mu) / h.sd, z64)
        zp = torch.maximum(torch.minimum(zp, h.zhi.double()), h.zlo.double())
        return (zp - z64).float()


def load_model(spec, ckpt, dev, newton=None):
    if spec in ("identity", "oracle"):
        return Identity().to(dev), {"head": spec}
    if spec.startswith("ensmed:"):
        parts = [load_model(s, ckpt, dev, newton) for s in spec[len("ensmed:"):].split(",")]
        assert all(p[1].get("head") == "relaxclosedr" for p in parts), "ensmed: relaxclosedr members only"
        return EnsembleMedian([p[0] for p in parts]).to(dev).eval(), {"head": "ensmed", "members": [p[1] for p in parts]}
    if spec.startswith("ensemble:"):
        parts = [load_model(s, ckpt, dev) for s in spec[len("ensemble:"):].split(",")]
        return Ensemble([p[0] for p in parts]).eval(), {"head": "ensemble", "members": [p[1] for p in parts]}
    run = Path(spec)
    if ckpt.endswith(".ckpt"):  # production Lightning checkpoint: EMA weights, 'model.' prefix (testing/export.py)
        blob = torch.load(run / ckpt, map_location=dev, weights_only=True)
        hp = blob["hyper_parameters"]
        hidden = hp["model"]["mlp"]["hidden_dims"]
        m = M.DeltaHead(hidden).to(dev)
        sd = {k[len("model."):]: v for k, v in blob["ema_state_dict"].items() if k.startswith("model.")}
        m.m.load_state_dict(sd, strict=True)
        return m.eval(), {"head": "delta", "hidden": hidden, "source": str(run / ckpt)}
    blob = torch.load(run / ckpt, map_location=dev)
    cfg = blob["cfg"]
    _, _, zlo, zhi = D.norm_consts(dev)
    m = M.build(cfg["head"], cfg["hidden"], zlo, zhi, cfg.get("latent"), cfg.get("dec_hidden"), newton=cfg.get("closed_iters", 12) if newton is None else newton,
                fallback=cfg.get("closed_fallback", "keep")).to(dev)
    m.load_state_dict(blob["state_dict"])
    return m.eval(), cfg


class Engine:
    """Static-shape rollout of R rows in three modes, CUDA-graph captured in blocks."""

    def __init__(self, model, z0, dtn, g, dev, perturb=None, nm=3, dtsched=None, post=None, p4=False, keep_in=False):
        self.model, self.dev, self.nm = model, dev, nm   # nm = 1: P3 only (screening; P1/P2 recorded as NaN)
        mu, sd, _, _ = D.norm_consts(dev)
        self.mu64, self.sd64 = mu, sd
        self.mu32, self.sd32 = mu.float(), sd.float()
        self.eps32 = torch.tensor(D.EPS, dtype=torch.float32, device=dev)
        R = z0.shape[0]
        self.R = R
        self.z3 = z0.clone()                                           # P3 state (fp64 z)
        y0 = torch.pow(10.0, z0 * sd + mu).float()                     # host fp32 mixing ratios
        self.y1, self.y2 = y0.clone(), y0.clone()                      # P1, P2 states
        self.dtn3 = dtn.repeat(nm).contiguous()
        self.g3 = g.repeat(nm, 1).contiguous()
        self.zlo_env = ((ENV_LO - mu) / sd)
        self.zhi_env = ((ENV_HI - mu) / sd)
        self.p4 = p4                                                   # blocks P2, P3, P4 (nm = 3); slot 0 unused
        nslot = 4 if p4 else 3
        self.nint = torch.zeros(nslot, R, dtype=torch.int64, device=dev)   # guard interventions per mode/row
        self.first = torch.full((nslot, R), -1, dtype=torch.int64, device=dev)
        self.ncall = torch.zeros((), dtype=torch.int64, device=dev)
        self.perturb = perturb                                         # callable(engine) for arm D, in-graph
        self.dtsched = dtsched                                         # callable(engine): sets dtn3, advances t
        self.post = post                                               # callable(engine) after each call (host repair)
        self.t = torch.zeros(R, dtype=torch.float64, device=dev)       # per-row elapsed time (dtsched only)
        self.nren = torch.zeros(R, dtype=torch.int64, device=dev)      # renormalizations per row (post only)
        self.xfirst = torch.full((R,), -1, dtype=torch.int64, device=dev)  # first call with sum_12 > 1.01 (post)
        self.xcount = torch.zeros(R, dtype=torch.int64, device=dev)    # calls with sum_12 > 1.01 (post)
        self.nproj_bad = torch.zeros(R, dtype=torch.int64, device=dev)  # calls with an element-projection residual > 1e-6
        self.zin = z0.clone() if keep_in else None                     # the call's input (P3; --elemproj targets)
        self.extra = [self.t, self.dtn3, self.nren, self.xfirst, self.xcount, self.nproj_bad] + ([self.zin] if keep_in else [])
        if p4:
            self.eps64 = torch.tensor(D.EPS, dtype=torch.float64, device=dev)
            self.y4 = torch.pow(10.0, z0 * sd + mu)                    # host fp64 mixing ratios (P4)
            self.extra.append(self.y4)
        self.graphs = {}
        self.adaptive, self.counters, self.nredo = [], [], 0   # 2026-10-12: --lm-adaptive (closed_counters sets them)

    def _z_of_y(self, y):
        return (torch.log10(torch.maximum(y, self.eps32)) - self.mu32) / self.sd32

    def closed_counters(self, B):
        """2026-10-12: per-row projection-failure counters [B, 2] for every closed module (relaxclosedr heads and the
        ensemble median), kept in self.extra so the graph warmup / capture side effects are undone. Returns the list."""
        mods = [m for m in self.model.modules() if isinstance(m, (M.RelaxClosedRHead, EnsembleMedian))]
        cnt = []
        for m in mods:
            m.rowcount = torch.zeros(B, 2, dtype=torch.int64, device=self.dev)
            cnt.append(m.rowcount)
        self.extra += cnt
        self.counters = cnt
        if ADAPTIVE and self.perturb is None:
            self.adaptive = [m for m in mods if isinstance(m, M.RelaxClosedRHead) and m.fallback == "lm"]
        return mods, cnt

    def _ok_z(self, z):
        return torch.isfinite(z).all(-1) & (z >= self.zlo_env).all(-1) & (z <= self.zhi_env).all(-1)

    def _ok_y(self, y):
        ly = torch.log10(y)
        return torch.isfinite(ly).all(-1) & (ly >= ENV_LO).all(-1) & (ly <= ENV_HI).all(-1)

    def _guard(self, mode, ok, new, old):
        bad = ~ok
        self.nint[mode] += bad.long()
        self.first[mode] = torch.where(bad & (self.first[mode] < 0), self.ncall + 1, self.first[mode])
        return torch.where(ok.unsqueeze(-1), new, old)

    @torch.no_grad()
    def step(self):
        R = self.R
        if self.perturb is not None:
            self.perturb(self)
        if self.dtsched is not None:
            self.dtsched(self)
        if self.nm == 1:
            if self.zin is not None:
                self.zin.copy_(self.z3)
            d3 = self.model.delta(self.z3.float(), self.dtn3, self.g3)
            z3n = self.z3 + d3.double()
            self.z3.copy_(self._guard(2, self._ok_z(z3n), z3n, self.z3))
            if self.post is not None:
                self.post(self)
            self.ncall += 1
            return
        if self.p4:
            self._step_p4()
            self.ncall += 1
            return
        z1, z2 = self._z_of_y(self.y1), self._z_of_y(self.y2)
        zin = torch.cat([z1, z2, self.z3.float()], 0)
        d = self.model.delta(zin, self.dtn3, self.g3)
        d1, d2, d3 = d[:R], d[R:2 * R], d[2 * R:]
        # P1: z round trip in fp32 (export arithmetic)
        y1n = torch.pow(10.0, (z1 + d1) * self.sd32 + self.mu32)
        self.y1.copy_(self._guard(0, self._ok_y(y1n) & torch.isfinite(d1).all(-1), y1n, self.y1))
        # P2: multiplicative update of the fp32 host state
        yb = torch.maximum(self.y2, self.eps32)
        x = d2 * (self.sd32 * math.log(10.0))
        y2n = torch.where(x.abs() < 1.0, yb + yb * torch.expm1(x), yb * torch.exp(x))
        self.y2.copy_(self._guard(1, self._ok_y(y2n) & torch.isfinite(d2).all(-1), y2n, self.y2))
        # P3: fp64 z state
        z3n = self.z3 + d3.double()
        self.z3.copy_(self._guard(2, self._ok_z(z3n), z3n, self.z3))
        self.ncall += 1

    def _step_p4(self):
        """Blocks P2 (fp32 host, multiplicative), P3 (fp64 z), P4 (fp64 host, multiplicative, expm1 branch)."""
        R = self.R
        z2 = self._z_of_y(self.y2)
        z4 = (torch.log10(torch.maximum(self.y4, self.eps64)) - self.mu64) / self.sd64
        d = self.model.delta(torch.cat([z2, self.z3.float(), z4.float()], 0), self.dtn3, self.g3)
        d2, d3, d4 = d[:R], d[R:2 * R], d[2 * R:]
        yb = torch.maximum(self.y2, self.eps32)
        x = d2 * (self.sd32 * math.log(10.0))
        y2n = torch.where(x.abs() < 1.0, yb + yb * torch.expm1(x), yb * torch.exp(x))
        self.y2.copy_(self._guard(1, self._ok_y(y2n) & torch.isfinite(d2).all(-1), y2n, self.y2))
        z3n = self.z3 + d3.double()
        self.z3.copy_(self._guard(2, self._ok_z(z3n), z3n, self.z3))
        yb4 = torch.maximum(self.y4, self.eps64)
        x4 = d4.double() * (self.sd64 * math.log(10.0))
        y4n = torch.where(x4.abs() < 1.0, yb4 + yb4 * torch.expm1(x4), yb4 * torch.exp(x4))
        self.y4.copy_(self._guard(3, self._ok_y(y4n) & torch.isfinite(d4).all(-1), y4n, self.y4))

    def _graph(self, n, mode=None):
        """mode (2026-10-12, --lm-adaptive): capture with the lm modules' fallback set to mode ('keep' | 'lm')."""
        if (n, mode) in self.graphs:
            return self.graphs[(n, mode)]
        if mode is not None:
            for m_ in self.adaptive:
                m_.fallback = mode
        keep = (self.z3, self.y1, self.y2, self.nint, self.first, self.ncall, self.g3, *self.extra)
        snap = [t.clone() for t in keep]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                self.step()
        for t, v in zip(keep, snap):
            t.copy_(v)  # undo the warmup/capture side effects
        if mode is not None:
            for m_ in self.adaptive:
                m_.fallback = "lm"
        self.graphs[(n, mode)] = g
        return g

    def advance(self, n):
        for blk in (BLOCK, 10, 1):
            k, n = divmod(n, blk)
            if k and not self.adaptive:
                g = self._graph(blk)
                for _ in range(k):
                    g.replay()
            elif k:  # 2026-10-12 --lm-adaptive: each block runs with fallback keep; a block in which any Newton solve
                # failed is undone and replayed with fallback lm. Equal to running lm throughout (lm changes only the
                # failed rows' result, and those blocks are redone); one host sync per block.
                gk, gl = self._graph(blk, "keep"), self._graph(1, "lm")  # lm redo call by call: a small graph pool
                st = (self.z3, self.y1, self.y2, self.nint, self.first, self.ncall, self.g3, *self.extra)
                for _ in range(k):
                    snap = [t.clone() for t in st]
                    n0 = sum(c[:, 0].sum() for c in self.counters)
                    gk.replay()
                    if int(sum(c[:, 0].sum() for c in self.counters) - n0) > 0:
                        for t, v in zip(st, snap):
                            t.copy_(v)
                        for _ in range(blk):
                            gl.replay()
                        self.nredo += 1

    def states(self):
        """[3, R, S] fp64 log10 mixing ratio for the three modes."""
        l3 = self.z3 * self.sd64 + self.mu64
        if self.nm == 1:
            nan = torch.full_like(l3, float("nan"))
            return torch.stack([nan, nan, l3], 0)
        if self.p4:  # [4, R, S]: P1 slot NaN
            return torch.stack([torch.full_like(l3, float("nan")), torch.log10(self.y2.double()), l3,
                                torch.log10(self.y4)], 0)
        l1 = torch.log10(self.y1.double())
        l2 = torch.log10(self.y2.double())
        return torch.stack([l1, l2, l3], 0)


def ckpt_calls(n_total, dt):
    """Calls at which states are recorded: log-spaced physical times plus 0.9 n_total (terminal drift)."""
    taus = np.unique(np.concatenate([np.logspace(0, np.log10(max(1.0, n_total * dt)), 33), [0.9 * n_total * dt]]))
    calls = np.unique(np.clip(np.round(taus / dt).astype(np.int64), 1, n_total))
    return calls


NM = 3  # set from --modes in main
P4 = False      # --modes P2P3P4
SUBSTEPS = 1    # --substeps
ADAPTIVE = False  # --lm-adaptive (2026-10-12): fallback lm only in the graph blocks where a Newton solve failed
STARTUP = None  # --startup-substeps n:k (2026-10-12): the first k calls of a run are made as n model calls of dt / n
POST = None     # --renorm: post-call host projection for every arm


def run_arm(model, z0, dtn, g, n_total, dt, dev, perturb=None, dtsched=None, post=None, extra_calls=None,
            want_g=False):
    post = post if post is not None else POST
    ep = getattr(post, "elemproj", None)
    eng = Engine(model, z0, dtn, g, dev, perturb, nm=NM, dtsched=dtsched, post=post, p4=P4, keep_in=ep is not None)
    cmods, cnt = eng.closed_counters(z0.shape[0] * (1 if NM == 1 else 3))
    if isinstance(model, Oracle):
        model.engine = eng
    calls = ckpt_calls(n_total, dt)
    if extra_calls is not None:
        calls = np.unique(np.concatenate([calls, np.asarray(extra_calls, dtype=np.int64)]))
    rec, rec_t, done, t0 = [], [], 0, time.time()
    st_left = 0
    if STARTUP is not None and dtsched is None:  # fixed-dt arms only (dtsched arms set dtn3 every call)
        st_n, st_left = STARTUP
        assert float(dt) / st_n >= 10.0 ** D.NORM["dt"]["log_min"] * (1 - 1e-9), "startup substep below the trained dt floor"
        dtn_full = eng.dtn3.clone()
        dtn_sub = D.dt_norm_of(torch.full(dtn_full.shape, float(dt) / st_n, dtype=torch.float64, device=dev)).float()
    for c in calls:
        todo = int(c - done)
        if st_left:
            k = min(st_left, todo)
            eng.dtn3.copy_(dtn_sub)
            eng.advance(k * st_n)
            eng.dtn3.copy_(dtn_full)
            st_left, todo = st_left - k, todo - k
        if SUBSTEPS > 1:
            eng.advance(todo * SUBSTEPS)
        else:
            eng.advance(todo)
        done = int(c)
        rec.append(eng.states().cpu().numpy())
        if dtsched is not None:
            rec_t.append(eng.t.cpu().numpy())
    torch.cuda.synchronize()
    out = {"calls": calls, "states": np.stack(rec), "nint": eng.nint.cpu().numpy(), "first": eng.first.cpu().numpy(),
           "seconds": time.time() - t0}
    if cnt:  # 2026-10-12: [modules, rows (x modes), 2] projection failures: Newton failed / row kept its input
        out["projfail"] = np.stack([c.cpu().numpy() for c in cnt])
        for m_ in cmods:
            m_.rowcount = None
        pf = out["projfail"]
        out["nredo"] = eng.nredo
        print(f"projection failures (calls x rows, summed over {len(cnt)} closed modules): Newton {int(pf[..., 0].sum())} on "
              f"{int((pf[..., 0] > 0).any(0).sum())} rows; kept input {int(pf[..., 1].sum())} on {int((pf[..., 1] > 0).any(0).sum())} rows",
              flush=True)
    if dtsched is not None:
        out["tcum"] = np.stack(rec_t)                                  # [C, R] elapsed simulated time (s)
    if post is not None:
        out.update(nren=eng.nren.cpu().numpy(), xfirst=eng.xfirst.cpu().numpy(), xcount=eng.xcount.cpu().numpy())
    if ep is not None:
        out["nproj_bad"] = eng.nproj_bad.cpu().numpy()
    if SUBSTEPS > 1:
        out["substeps"] = SUBSTEPS
    if want_g:
        out["g_final"] = eng.g3[:eng.R].cpu().numpy()
    return out


def make_post(dev, renorm_rows=None, renorm_all=False, track=False):
    """Post-call host step (P3 state only). renorm_all / renorm_rows (bool [R]): if sum_12 y > 1, subtract log10 sum
    from every species, then clamp at the floor (log10 eps). track: first call and number of calls with sum_12 > 1.01
    (measured on the model's output, before any renormalization)."""
    lf = math.log10(D.EPS)

    def post(e):
        assert e.nm == 1, "post-call host step is implemented for P3 only"
        ly = e.z3 * e.sd64 + e.mu64
        s = torch.pow(10.0, ly).sum(-1)
        if track:
            exc = s > 1.01
            e.xfirst.copy_(torch.where(exc & (e.xfirst < 0), e.ncall + 1, e.xfirst))
            e.xcount += exc.long()
        if renorm_all or renorm_rows is not None:
            fix = s > 1.0
            if renorm_rows is not None:
                fix = fix & renorm_rows
            lyn = torch.clamp(ly - torch.log10(s).unsqueeze(-1), min=lf)
            e.z3.copy_(torch.where(fix.unsqueeze(-1), (lyn - e.mu64) / e.sd64, e.z3))
            e.nren += fix.long()
    return post


def make_elemproj_post(dev, mode, then=None):
    """2026-10-09 (--elemproj): post-call step (P3 only): project the 12 tracked species onto the element ratios of the
    call's input (elemproj12.project, mode ratio or ratio_he), clamp at the floor (log10 eps), count per row the calls
    whose residual exceeds 1e-6 (nproj_bad); then the arm's own post step, if any (then)."""
    import elemproj12 as EP12
    assert len(D.SHORT) == 12, "--elemproj needs a 12-species model"
    A = EP12.atoms(D.SHORT, dev)
    lf = math.log10(D.EPS)

    def post(e):
        assert e.nm == 1 and e.zin is not None, "--elemproj is implemented for P3 only"
        ly, res = EP12.project(e.z3 * e.sd64 + e.mu64, e.zin * e.sd64 + e.mu64, A, mode, floor=lf)
        e.z3.copy_((ly - e.mu64) / e.sd64)
        e.nproj_bad += (res > 1e-6).long()
        if then is not None:
            then(e)
    post.elemproj = mode
    return post


def run_H(model, tag, starts, dt, n, dev, g=None, sched=None, unclamped=False, extra_calls=None):
    """Arm H: rows = the tag's finished jobs (sorted) x starts (start-major), start state = the tag's own log10 y at
    each start, g = norm_globals(job P, T) unless given, fixed dt, n calls. Returns the npz fields.
    Sprint 2026-10-08: sched = a make_dtsched kind (per-row, per-call dt; the npz then holds tcum, the elapsed time per
    row at each recorded call; dt only sets the recorded calls); unclamped = dt_norm without the [0, 1] clamp (dt above
    the trained 1e3 s range: the relax rate extrapolates in log dt)."""
    mu64, sd64, _, _ = D.norm_consts(dev)
    ref = vref_index(tag)
    keys = sorted(ref)
    job = np.array([k for _ in starts for k in keys])
    t0 = np.array([s for s in starts for _ in keys], dtype=np.float64)
    ly0 = np.concatenate([np.stack([vref_logy(tag, i, np.array([s]))[0] for i in keys]) for s in starts])
    z0 = (torch.as_tensor(ly0, dtype=torch.float64, device=dev) - mu64) / sd64
    P = np.array([ref[i]["P"] for i in job])
    T = np.array([ref[i]["T"] for i in job])
    if g is None:
        g = torch.from_numpy(D.norm_globals(P, T)).to(dev)
    dts = dt / SUBSTEPS if SUBSTEPS > 1 else dt
    dtn = D.dt_norm_of(torch.full((job.size,), dts, dtype=torch.float64, device=dev)).float()
    if unclamped:
        lo, hi = D.NORM["dt"]["log_min"], D.NORM["dt"]["log_max"]
        dtn = ((torch.log10(torch.full((job.size,), dts, dtype=torch.float64, device=dev)) - lo) / (hi - lo)).float()
    dtsched = None if sched is None else make_dtsched(sched, job.size, NM, dev)[0]
    r = run_arm(model, z0, dtn, g, n, dt, dev, dtsched=dtsched, extra_calls=extra_calls)
    extra = {} if sched is None and not unclamped else {"sched": str(sched), "unclamped": bool(unclamped)}
    return dict(job=job, t0=t0, dt=dt, tag=str(tag), P=P, T=T, ridx=np.array([ref[i].get("row", -1) for i in job]), **extra, **r)


def make_dtsched(kind, R, nm, dev):
    """Per-call dt for arm V: returns (callable(engine), mean dt in s). Sets dtn3 in place and advances e.t."""
    if kind == "rand":
        def sched(e):
            dt = torch.pow(10.0, 2.0 * torch.rand(R, dtype=torch.float64, device=dev))   # log-uniform [1, 100] s
            e.dtn3.copy_(D.dt_norm_of(dt).float().repeat(nm))
            e.t += dt
        return sched, 99.0 / math.log(100.0)
    if kind == "alt":
        tab = torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64, device=dev)
        tabn = D.dt_norm_of(tab).float()

        def sched(e):
            k = torch.remainder(e.ncall, 3).reshape(1)
            e.dtn3.copy_(tabn.index_select(0, k).expand(e.dtn3.shape[0]))
            e.t += tab.index_select(0, k)
        return sched, 37.0
    if kind == "rand10_500":  # sprint 2026-10-08 (R-early b): log-uniform [10, 500] s per row and call
        def sched(e):
            dt = torch.pow(10.0, 1.0 + math.log10(50.0) * torch.rand(R, dtype=torch.float64, device=dev))
            e.dtn3.copy_(D.dt_norm_of(dt).float().repeat(nm))
            e.t += dt
        return sched, 490.0 / math.log(50.0)
    if kind == "startup":     # sprint 2026-10-08 (R-early d): 10 s per call until 1e3 s have elapsed, then 500 s
        d10 = torch.full((R,), 10.0, dtype=torch.float64, device=dev)
        d500 = torch.full((R,), 500.0, dtype=torch.float64, device=dev)

        def sched(e):
            dt = torch.where(e.t < 1e3, d10, d500)
            e.dtn3.copy_(D.dt_norm_of(dt).float().repeat(nm))
            e.t += dt
        return sched, 500.0
    if kind.startswith("const"):
        dtc = float(kind[len("const"):])
        dtn = D.dt_norm_of(torch.full((R,), dtc, dtype=torch.float64, device=dev)).float().repeat(nm)

        def sched(e):
            e.dtn3.copy_(dtn)
            e.t += dtc
        return sched, dtc
    raise ValueError(kind)


def run_paths(model, p_dir, dev):
    """Arm P (E13): every path of a paths folder from its VULCAN start, T, P (and dt, paths v2) changing every call,
    every call recorded (2026-10-09: a function, so train.py's path readout runs the same rollout). Returns the run_arm
    result, the path files, their arrays, the number of calls, the dt (per path, or [R, n]) and the number of recorded
    mixes applied (None for paths without mix_k). 2026-10-09: paths with mix_k get their mixes before the calls (chains.mix)."""
    files = sorted(Path(p_dir).glob("[0-9]*.npz"), key=lambda f: int(f.stem))
    Ps = [np.load(f) for f in files]
    n = min(len(p["T"]) for p in Ps)
    mu64, sd64, _, _ = D.norm_consts(dev)
    ly0 = np.log10(np.maximum(np.stack([D.take(p["x"][0], list(p["species"])) for p in Ps]), D.EPS))
    z0 = (torch.as_tensor(ly0, dtype=torch.float64, device=dev) - mu64) / sd64
    sched_g = torch.from_numpy(np.stack([D.norm_globals(p["Pbar"][:n] * 1e6, p["T"][:n]) for p in Ps], 1)).to(dev)
    if any(np.ndim(p["dt"]) > 0 for p in Ps):  # sprint 2026-10-08 (paths v2): per-call dt [n] per path
        dmat = torch.as_tensor(np.stack([np.asarray(p["dt"], np.float64)[:n] for p in Ps], 1), device=dev)  # [n, R]
        dtn_mat = D.dt_norm_of(dmat).float()
        dtn = dtn_mat[0].contiguous()
        dts = dmat.T.cpu().numpy()

        def psched(e):                   # call ncall uses T[k], P[k] and dt[k]
            k = torch.clamp(e.ncall, max=n - 1).reshape(1)
            e.g3.copy_(torch.index_select(sched_g, 0, k).squeeze(0).repeat(e.nm, 1))
            e.dtn3.copy_(torch.index_select(dtn_mat, 0, k).squeeze(0).repeat(e.nm))
            e.t += torch.index_select(dmat, 0, k).squeeze(0)
    else:
        dts = np.array([float(p["dt"]) for p in Ps])
        dtn = D.dt_norm_of(torch.as_tensor(dts, dtype=torch.float64, device=dev)).float()

        dts_t = torch.as_tensor(dts, dtype=torch.float64, device=dev)

        def psched(e):                       # in-graph: the globals of call ncall (path call k uses T[k], P[k]);
            k = torch.clamp(e.ncall, max=n - 1).reshape(1)  # index_select with a device index: no host sync
            e.g3.copy_(torch.index_select(sched_g, 0, k).squeeze(0).repeat(e.nm, 1))
            e.t += dts_t

    nmix = None
    if any("mix_k" in p.files for p in Ps):  # 2026-10-09 (paths_v3): per-call f [n, R] and partner y [n, R, S] on the device
        fy = [C.mixing_arrays(p, n) for p in Ps]
        fmat = torch.as_tensor(np.stack([a for a, _ in fy], 1), device=dev)
        ymat = torch.as_tensor(np.stack([b for _, b in fy], 1), device=dev)
        nmix = int(sum((a > 0).sum() for a, _ in fy))
        base = psched

        def psched(e):                       # the call's globals and dt, then the recorded mix of call ncall (in-graph)
            base(e)
            k = torch.clamp(e.ncall, max=n - 1).reshape(1)
            f = torch.index_select(fmat, 0, k).squeeze(0)
            yp = torch.index_select(ymat, 0, k).squeeze(0)
            e.z3.copy_(C.mix(e.z3, f, yp, e.mu64, e.sd64))
            if e.nm == 3:                    # host mixing-ratio states (P1, P2; with P2P3P4: P2, P4), floored at eps
                fb = f.unsqueeze(-1)
                for y in ((e.y2, e.y4) if e.p4 else (e.y1, e.y2)):
                    yb = torch.clamp((1.0 - fb.to(y.dtype)) * y + fb.to(y.dtype) * yp.to(y.dtype), min=D.EPS)
                    y.copy_(torch.where(fb > 0, yb, y))
    r = run_arm(model, z0, dtn, sched_g[0], n, 1.0, dev, dtsched=psched, extra_calls=np.arange(1, n + 1))
    return r, files, Ps, n, dts, nmix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("level")
    ap.add_argument("--ckpt", default="stage2.pt")
    ap.add_argument("--arms", default="A,B,C,D,E")
    ap.add_argument("--dts", default="1,10,100")
    mf = paths.BENCH_MODES   # campaign-wide default, read at start
    ap.add_argument("--modes", default=mf.read_text().strip() if mf.exists() else "all", choices=["all", "P3", "P2P3P4"])
    ap.add_argument("--renorm", action="store_true")
    ap.add_argument("--substeps", type=int, default=1)
    ap.add_argument("--v-schedule", default="rand")
    ap.add_argument("--v-horizon", type=float, default=None)
    ap.add_argument("--h-tag", default="ref_a")
    ap.add_argument("--h-starts", default="1e-3,1e3,1e6")
    ap.add_argument("--h-dt", type=float, default=10.0)
    ap.add_argument("--h-calls", type=int, default=None)
    ap.add_argument("--h-sched", default=None)   # sprint 2026-10-08: per-call dt schedule for arm H (make_dtsched kind)
    ap.add_argument("--h-dt-unclamped", action="store_true")  # sprint 2026-10-08: dt_norm without the clamp (dt > 1e3 s)
    ap.add_argument("--h-times", default=None)  # 2026-10-14: .npy of elapsed times (s, multiples of --h-dt) also recorded
    ap.add_argument("--p-dir", default=None)  # E13 arm P: a paths.py folder (T and P change every call)
    ap.add_argument("--g-groups", default="G0,G1,G2,G3,G4,G5,G2r,G3r,G4r")
    ap.add_argument("--g-calls", type=int, default=200_000)
    ap.add_argument("--g-recover", type=int, default=100_000)
    ap.add_argument("--g-cache", default=None)    # 2026-10-09: arm G's sources from this cache's test split (default data.CACHE)
    ap.add_argument("--elemproj", default=None, choices=["ratio", "ratio_he"])  # 2026-10-09: element projection after every call
    ap.add_argument("--closed-iters", type=int, default=None)  # 2026-10-09: override the closed heads' Newton iterations
    ap.add_argument("--closed-fallback", default=None, choices=["keep", "lm"])  # 2026-10-12: override relaxclosedr's fallback
    ap.add_argument("--lm-adaptive", action="store_true")  # 2026-10-12: see ADAPTIVE (needs --closed-fallback lm)
    ap.add_argument("--startup-substeps", default=None)  # 2026-10-12: n:k, the first k calls as n calls of dt / n (fixed-dt arms)
    a = ap.parse_args()
    global NM, P4, SUBSTEPS, POST, STARTUP, ADAPTIVE
    ADAPTIVE = bool(a.lm_adaptive)
    NM = 1 if a.modes == "P3" else 3
    P4 = a.modes == "P2P3P4"
    SUBSTEPS = int(a.substeps)
    assert SUBSTEPS >= 1
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = torch.device("cuda")
    lev = LEVELS[a.level]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    test = D.RawSplit("test", dev)
    src = select_sources(test, a.level)
    model, cfg = load_model(a.model, a.ckpt, dev, a.closed_iters)
    meta = {"model": a.model, "ckpt": a.ckpt, "cfg": cfg, "level": a.level, "n_src": int(src.size), "modes": a.modes,
            "sources": [str(x) for x in test.name[src]], "starts": STARTS, "notes": []}
    if a.closed_iters is not None:
        meta["closed_iters"] = a.closed_iters
    if a.startup_substeps:
        assert SUBSTEPS == 1, "--startup-substeps without --substeps"
        STARTUP = tuple(int(v) for v in a.startup_substeps.split(":"))
        meta["startup_substeps"] = a.startup_substeps
    if a.closed_fallback is not None:
        for m_ in model.modules():
            if isinstance(m_, M.RelaxClosedRHead):
                m_.fallback = a.closed_fallback
        meta["closed_fallback"] = a.closed_fallback
    idx = torch.from_numpy(src).to(dev)
    arms = a.arms.split(",")
    mu64, sd64, _, _ = D.norm_consts(dev)
    if a.renorm:
        POST = make_post(dev, renorm_all=True)
        meta["renorm"] = True
    if a.elemproj:
        assert NM == 1 and not a.renorm, "--elemproj runs in P3 mode (--modes P3), without --renorm"
        POST = make_elemproj_post(dev, a.elemproj)
        meta["elemproj"] = a.elemproj
    if SUBSTEPS > 1:
        meta["substeps"] = SUBSTEPS
        assert "D" not in arms and "G" not in arms, "substeps would perturb once per substep"
    assert not (P4 and ("D" in arms or "G" in arms)), "the stress perturbation does not update the P4 state"

    def z_of_logy(ly):
        return (torch.as_tensor(ly, dtype=torch.float64, device=dev) - mu64) / sd64

    def dtn_of(dt, n):
        if SUBSTEPS > 1:
            dt = dt / SUBSTEPS
        return D.dt_norm_of(torch.full((n,), dt, dtype=torch.float64, device=dev)).float()

    if "A" in arms:  # transient from three start ages; truth: VULCAN reference (a) where present, else raw
        for dt in [float(x) for x in a.dts.split(",")]:
            ridx = idx.repeat(len(STARTS))
            rt0 = torch.cat([torch.full_like(idx, 0, dtype=torch.float64) + t0 for t0 in STARTS])
            z0 = test.z_of(test.logy_at(ridx, rt0.reshape(-1, 1)))[:, 0]
            n = int(min(lev["A"][dt], round(HORIZON_S / dt)))
            m_arm = Oracle(test, ridx, rt0, dt) if a.model == "oracle" else model
            r = run_arm(m_arm, z0, dtn_of(dt, ridx.numel()), test.g[ridx], n, dt, dev)
            np.savez(out / f"A_dt{dt:g}.npz", ridx=ridx.cpu().numpy(), t0=rt0.cpu().numpy(), dt=dt, **r)
            print(f"A dt={dt:g} n={n} rows={ridx.numel()} {r['seconds']:.0f}s interventions={r['nint'].sum(1).tolist()}", flush=True)

    ref_a = vref_index("ref_a")
    row_to_a = {j["row"]: i for i, j in ref_a.items()}
    if "B" in arms:  # late hold at dt 10 s: start from VULCAN's state at 1e8 s, truth = VULCAN's own slow evolution
        rows = [int(r) for r in src if int(r) in row_to_a]
        if rows:
            jobs = [row_to_a[r] for r in rows]
            ridx = torch.tensor(rows, device=dev)
            z0 = z_of_logy(np.stack([vref_logy("ref_a", i, np.array([B_START_S]))[0] for i in jobs]))
            r_ = run_arm(model, z0, dtn_of(10.0, len(rows)), test.g[ridx], lev["B"], 10.0, dev)
            np.savez(out / "B_hold.npz", ridx=np.array(rows), job=np.array(jobs), t_start=B_START_S, **r_)
            print(f"B n={lev['B']} rows={len(rows)} {r_['seconds']:.0f}s interventions={r_['nint'].sum(1).tolist()}", flush=True)
        else:
            meta["notes"].append("B skipped: no VULCAN reference (a)")

    for arm, tag, dt, ncall, gkeys in (("C", "ref_p", 100.0, lev["C"], ("P", "T")), ("E", "ref_c", 10.0, 100_000, ("P", "T"))):
        if arm not in arms:
            continue
        try:
            ref = vref_index(tag)
        except FileNotFoundError:
            meta["notes"].append(f"{arm} skipped: no VULCAN reference {tag}")
            continue
        if not ref:
            meta["notes"].append(f"{arm} skipped: no finished VULCAN reference {tag}")
            continue
        keys = sorted(ref)
        z0 = z_of_logy(np.stack([mix_logy(ref[i]["mix"]) for i in keys]))
        g = torch.from_numpy(D.norm_globals([ref[i]["P"] for i in keys], [ref[i]["T"] for i in keys])).to(dev)
        r_ = run_arm(model, z0, dtn_of(dt, len(keys)), g, ncall, dt, dev)
        np.savez(out / f"{arm}_{tag}.npz", job=np.array(keys), ridx=np.array([ref[i]["row"] for i in keys]), dt=dt,
                 P=np.array([ref[i]["P"] for i in keys]), T=np.array([ref[i]["T"] for i in keys]), tag=tag, **r_)
        print(f"{arm} n={ncall} rows={len(keys)} {r_['seconds']:.0f}s interventions={r_['nint'].sum(1).tolist()}", flush=True)

    if "D" in arms:                          # operator-splitting stress test at dt 10 s
        z0 = test.z_of(test.logy_at(idx, torch.full((idx.numel(), 1), 1e4, dtype=torch.float64, device=dev)))[:, 0]
        R = idx.numel()
        torch.manual_seed(SAMPLE_SEED + 1)   # the default CUDA generator is graph-safe
        partner = torch.randperm(R, device=dev)
        zfloor = (math.log10(D.EPS) - mu64) / sd64
        eps32 = torch.tensor(D.EPS, dtype=torch.float32, device=dev)

        def perturb(e):
            # host side, before each chemistry call: transport-like blending with a partner cell (1-10% in mixing
            # ratio), P/T random walk inside the box (normalized units, reflected), rare zeros / tiny negatives in
            # one species, then the host's repair (floor at eps) so the guard's fallback state is always valid
            f = 0.01 + 0.09 * torch.rand(R, 1, device=dev)
            hit = torch.rand(R, device=dev) < 1e-3
            sp = torch.randint(0, M.S, (R,), device=dev)
            val = torch.where(torch.rand(R, device=dev) < 0.5, 0.0, -1e-30)
            mask = hit.unsqueeze(1) & (torch.arange(M.S, device=dev).unsqueeze(0) == sp.unsqueeze(1))
            for y in (e.y1, e.y2):
                yb = (1 - f) * y + f * y[partner]
                yb = torch.where(mask, val.float().unsqueeze(1).expand_as(yb), yb)
                y.copy_(torch.maximum(yb, eps32))
            y3 = torch.pow(10.0, e.z3 * e.sd64 + e.mu64)
            y3 = (1 - f.double()) * y3 + f.double() * y3[partner]
            z3 = (torch.log10(torch.clamp(y3, min=D.EPS)) - e.mu64) / e.sd64
            e.z3.copy_(torch.where(mask, zfloor.expand_as(z3), z3))
            gw = e.g3[:R] + 0.002 * torch.randn(R, 2, device=dev)
            gw = torch.where(gw < 0.0, -gw, gw)
            gw = torch.where(gw > 1.0, 2.0 - gw, gw)
            e.g3.copy_(gw.repeat(e.nm, 1))

        r = run_arm(model, z0, dtn_of(10.0, R), test.g[idx], lev["D"], 10.0, dev, perturb)
        np.savez(out / "D_stress.npz", ridx=idx.cpu().numpy(), **r)
        print(f"D n={lev['D']} rows={R} {r['seconds']:.0f}s interventions={r['nint'].sum(1).tolist()}", flush=True)

    if "V" in arms:                          # A-arm rows and starts, dt changing per call
        ridx = idx.repeat(len(STARTS))
        rt0 = torch.cat([torch.full_like(idx, 0, dtype=torch.float64) + t0 for t0 in STARTS])
        z0 = test.z_of(test.logy_at(ridx, rt0.reshape(-1, 1)))[:, 0]
        R = ridx.numel()
        sched, dt_mean = make_dtsched(a.v_schedule, R, NM, dev)
        horizon = a.v_horizon if a.v_horizon is not None else lev["A"][10.0] * 10.0
        n = int(round(horizon / dt_mean))
        torch.manual_seed(V_SEED)
        dtn0 = dtn_of(dt_mean if a.v_schedule.startswith("const") else 10.0, R)
        r = run_arm(model, z0, dtn0, test.g[ridx], n, dt_mean, dev, dtsched=sched)
        np.savez(out / f"V_{a.v_schedule}.npz", ridx=ridx.cpu().numpy(), t0=rt0.cpu().numpy(), dt=dt_mean,
                 schedule=a.v_schedule, horizon=horizon, **r)
        print(f"V {a.v_schedule} n={n} rows={R} mean_t_end={r['tcum'][-1].mean():.6g} {r['seconds']:.0f}s "
              f"interventions={r['nint'].sum(1).tolist()}", flush=True)

    if "P" in arms:                          # E13: dynamic paths, T and P change every call (truth: g2/E13_gcm/paths.py)
        r, files, Ps, n, dts, nmix = run_paths(model, a.p_dir, dev)
        np.savez(out / "P_paths.npz", path=np.array([int(f.stem) for f in files]), dt=dts,
                 T=np.stack([p["T"][:n] for p in Ps]), Pbar=np.stack([p["Pbar"][:n] for p in Ps]), p_dir=str(a.p_dir), **r)
        meta["notes"].append(f"P_paths: {len(Ps)} paths from {a.p_dir}, {n} calls, every call recorded")
        if nmix is not None:
            meta["mixing"] = {"applied": True, "n_mix": nmix, "rule": "before call k: y <- (1 - f) y + f y_partner (linear, "
                              "entry set, floored at eps; chains.mix)"}
        print(f"P n={n} rows={len(Ps)} {r['seconds']:.0f}s interventions={r['nint'].sum(1).tolist()}", flush=True)

    if "H" in arms:                          # starts from a VULCAN tag's own states; truth = the same tag
        dt = float(a.h_dt)
        n = a.h_calls if a.h_calls is not None else int(min(lev["A"].get(dt, lev["A"][10.0]), round(HORIZON_S / dt)))
        starts = [float(x) for x in a.h_starts.split(",")]
        if a.h_sched is not None:
            assert a.h_calls is not None and SUBSTEPS == 1, "--h-sched needs --h-calls (and no substeps)"
            torch.manual_seed(V_SEED)
        xc = None
        if a.h_times is not None:            # 2026-10-14: record these elapsed times too (cross-dt comparisons at equal times)
            tt = np.load(a.h_times).astype(np.float64)
            tt = tt[tt <= n * dt]
            xc = np.round(tt / dt).astype(np.int64)
            assert np.allclose(xc * dt, tt, rtol=0, atol=1e-6) and (xc >= 1).all(), "--h-times must be multiples of --h-dt"
        res = run_H(model, a.h_tag, starts, dt, n, dev, sched=a.h_sched, unclamped=a.h_dt_unclamped, extra_calls=xc)
        name = f"H_{Path(a.h_tag).name}_dt{dt:g}" + (f"_{a.h_sched}" if a.h_sched else "") + ("_unclamped" if a.h_dt_unclamped else "")
        np.savez(out / f"{name}.npz", **res)
        meta["notes"].append(f"{name}: tag {a.h_tag}, {res['job'].size // len(starts)} jobs x starts {starts}, dt {dt:g}, n {n}")
        print(f"{name} n={n} rows={res['job'].size} {res['seconds']:.0f}s interventions={res['nint'].sum(1).tolist()}", flush=True)

    if "G" in arms:                          # stress dose-response + recovery + clean twins (P3 only)
        assert NM == 1 and SUBSTEPS == 1, "arm G runs in P3 mode without substeps"
        names = a.g_groups.split(",")
        gsplit, gsrc = test, idx
        if a.g_cache is not None:            # 2026-10-09: sources from another cache's test split (same selection rule)
            gsplit = D.RawSplit("test", dev, cache=a.g_cache)
            assert gsplit.logy.shape[-1] == len(D.SHORT), (a.g_cache, gsplit.logy.shape)
            gsrc = torch.from_numpy(select_sources(gsplit, a.level)).to(dev)
            meta.update(g_cache=str(a.g_cache), g_sources=[str(x) for x in gsplit.name[gsrc.cpu().numpy()]])
        R0 = gsrc.numel()
        R = R0 * len(names)
        gidx = gsrc.repeat(len(names))
        z0 = gsplit.z_of(gsplit.logy_at(gidx, torch.full((R, 1), 1e4, dtype=torch.float64, device=dev)))[:, 0]
        amp = np.array([GROUPS[k][:3] for k in names for _ in range(R0)], dtype=np.float64)    # [R, 3]
        ren = torch.tensor([GROUPS[k][3] for k in names for _ in range(R0)], device=dev)
        flo = torch.tensor(0.1 * amp[:, :1], dtype=torch.float32, device=dev)
        fspan = torch.tensor(0.9 * amp[:, :1], dtype=torch.float32, device=dev)
        sdw = torch.tensor(amp[:, 1:2], dtype=torch.float32, device=dev)
        zrate = torch.tensor(amp[:, 2], dtype=torch.float32, device=dev)
        act = torch.from_numpy((amp != 0).any(1)).to(dev)
        all_act = bool(act.all())
        torch.manual_seed(G_SEED)
        partner = torch.cat([k * R0 + torch.randperm(R0, device=dev) for k in range(len(names))])
        zfloor = (math.log10(D.EPS) - mu64) / sd64
        eps32 = torch.tensor(D.EPS, dtype=torch.float32, device=dev)
        n_stress, n_rec = int(a.g_calls), int(a.g_recover)

        def perturb_g(e):
            # arm D's perturbation with per-row amplitudes (same draws in the same order); off after n_stress calls
            f = flo + fspan * torch.rand(R, 1, device=dev)
            hit = torch.rand(R, device=dev) < zrate
            sp = torch.randint(0, M.S, (R,), device=dev)
            torch.rand(R, device=dev)        # arm D's draw for the injected value's sign (fp32 host modes only)
            mask = hit.unsqueeze(1) & (torch.arange(M.S, device=dev).unsqueeze(0) == sp.unsqueeze(1))
            y3 = torch.pow(10.0, e.z3 * e.sd64 + e.mu64)
            y3 = (1 - f.double()) * y3 + f.double() * y3[partner]
            z3 = (torch.log10(torch.clamp(y3, min=D.EPS)) - e.mu64) / e.sd64
            z3 = torch.where(mask, zfloor.expand_as(z3), z3)
            gw = e.g3[:R] + sdw * torch.randn(R, 2, device=dev)
            gw = torch.where(gw < 0.0, -gw, gw)
            gw = torch.where(gw > 1.0, 2.0 - gw, gw)
            if not all_act:
                z3 = torch.where(act.unsqueeze(1), z3, e.z3)
                gw = torch.where(act.unsqueeze(1), gw, e.g3[:R])
            if n_rec > 0:
                on = e.ncall < n_stress
                z3 = torch.where(on, z3, e.z3)
                gw = torch.where(on, gw, e.g3[:R])
            e.z3.copy_(z3)
            e.g3.copy_(gw.repeat(e.nm, 1))

        post = make_post(dev, renorm_rows=ren if bool(ren.any()) else None, track=True)
        if a.elemproj:
            post = make_elemproj_post(dev, a.elemproj, then=post)
        n_total = n_stress + n_rec
        extra = [n_stress] + ([n_stress + k for k in (1, 10, 100, 1000, 10_000, 100_000) if k <= n_rec] if n_rec else [])
        r = run_arm(model, z0, dtn_of(10.0, R), gsplit.g[gidx], n_total, 10.0, dev, perturb_g, post=post,
                    extra_calls=extra, want_g=True)
        res = dict(ridx=gidx.cpu().numpy(), groups=np.array(names), group=np.repeat(np.arange(len(names)), R0),
                   amp=amp, renorm=ren.cpu().numpy(), n_stress=n_stress, n_recover=n_rec, **r)
        print(f"G groups={names} n={n_total} rows={R} {r['seconds']:.0f}s interventions={r['nint'].sum(1).tolist()} "
              f"renorm={int(r['nren'].sum())}", flush=True)
        tw = np.flatnonzero(act.cpu().numpy())
        if n_rec > 0 and tw.size:            # clean twins: same start, g = the row's final g, same host scheme
            twi = torch.from_numpy(tw).to(dev)
            post_t = make_post(dev, renorm_rows=ren[twi] if bool(ren[twi].any()) else None, track=True)
            if a.elemproj:
                post_t = make_elemproj_post(dev, a.elemproj, then=post_t)
            gt = torch.from_numpy(r["g_final"][tw]).to(dev)
            rt = run_arm(model, z0[twi], dtn_of(10.0, tw.size), gt, n_total, 10.0, dev, post=post_t, extra_calls=extra)
            res.update(twin_rows=tw, twin_states=rt["states"], twin_nint=rt["nint"], twin_nren=rt["nren"],
                       twin_xfirst=rt["xfirst"], twin_xcount=rt["xcount"], twin_seconds=rt["seconds"])
            if a.elemproj:
                res["twin_nproj_bad"] = rt["nproj_bad"]
            print(f"G twins rows={tw.size} {rt['seconds']:.0f}s interventions={rt['nint'].sum(1).tolist()}", flush=True)
        np.savez(out / "G_stress.npz", **res)

    (out / "meta.json").write_text(json.dumps(meta, default=str))


if __name__ == "__main__":
    main()
