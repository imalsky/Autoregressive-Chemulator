"""The cache on the GPU, the normalization and the training samplers.

A cache (ARCH_CACHE; datagen/ builds it) holds raw_<split>.npz per split (logy [N, 100, S] log10 mixing ratios of the entry
set on one stored log-time grid t_grid.npy, end = last valid row, P, T, file_idx) and meta.json. The normalization file
(HARNESS_NORM, default paths.NORM12) gives the entry set (species_variables), the per-entry log10 mean and std, the globals'
ranges and the dt range; z = (log10 y - mu) / sd.
Samplers (RawSplit.sample, sample_grid, sample_jump_pairs) draw on a torch.Generator in a fixed order; one-step windows:
dt log-uniform in DT_LOG, the start anchored at the first positive stored time with probability ANCHOR_P or log-uniform
in [t_lo, t_end - FIT_STEPS dt]; optional row-0 starts (t = 0) of one-step windows and jump pairs, each row with its own
probability (RawSplit.p0_anchor, p0_jump; one extra draw per candidate, made only when set); candidates that do not fit
are dropped by a stable sort (no host sync). Values between stored rows are the log-log interpolant (log10 y linear in
log10 t), computed in fp64; for a complete entry set (He included) the interpolated labels are projected back onto each
trajectory's element ratios and sum 1 (elemproj.project_ln).
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import paths

CACHE = Path(os.environ.get("ARCH_CACHE", str(paths.ARCH_TESTS)))
NORM_PATH = Path(os.environ.get("HARNESS_NORM") or str(paths.NORM12))  # the normalization file (empty = unset)
NORM = json.loads(NORM_PATH.read_text())   # NORM_PATH is also recorded in the model manifests (train.write_manifest)
SPECIES = list(NORM["species_variables"])
SHORT = [s.replace("_evolution", "") for s in SPECIES]
CLOSED = "He" in SHORT  # a complete entry set (He included): labels are projected in RawSplit.logy_at
FIT_STEPS = 99          # every window must fit 99 transitions before the trajectory's end
ANCHOR_P = 0.25         # share of one-step windows anchored at the first positive stored time (train.py anchor_p)
DT_LOG = (-1.0, 3.0)    # window dt, log10 s; train.py sets it from dt_min_s / dt_max_s
EPS = float(NORM["epsilon"])


def take(x, names):
    """SHORT columns of linear mixing ratios x [..., len(names)] over the full network (names). Reservoir entries
    (reservoir sets): C_res, N_res, O_res, H_res = atoms of that element in the species not in SHORT, M_res = their
    molecule count."""
    import elemproj as EP
    if not any(s.endswith("_res") for s in SHORT):
        return x[..., [names.index(s) for s in SHORT]]
    h, a = (t.numpy() for t in EP.stoich(names))      # a columns: C, N, O, He
    per = {"C_res": a[:, 0], "N_res": a[:, 1], "O_res": a[:, 2], "H_res": h, "M_res": np.ones(len(names))}
    hid = np.array([s not in SHORT for s in names])
    return np.stack([x[..., hid] @ per[s][hid] if s in per else x[..., names.index(s)] for s in SHORT], -1)


def norm_consts(device, dtype=torch.float64):
    st = NORM["per_key_stats"]
    mu = torch.tensor([st[s]["log_mean"] for s in SPECIES], dtype=dtype, device=device)
    sd = torch.tensor([st[s]["log_std"] for s in SPECIES], dtype=dtype, device=device)
    zlo = (np.log10(EPS) - mu) / sd
    zhi = (0.0 - mu) / sd
    return mu, sd, zlo, zhi


def norm_globals(P, T):
    """[N, 2] fp32 exactly as preprocessing: P log-min-max, T min-max, on float32-valued P, T."""
    st = NORM["per_key_stats"]
    P = np.asarray(P, np.float64)
    T = np.asarray(T, np.float64)
    p = (np.log10(np.maximum(P, EPS)) - st["P"]["log_min"]) / (st["P"]["log_max"] - st["P"]["log_min"])
    t = (T - st["T"]["min"]) / (st["T"]["max"] - st["T"]["min"])
    return np.stack([p, t], axis=1).astype(np.float32)


def dt_norm_of(dt_s):
    lo, hi = NORM["dt"]["log_min"], NORM["dt"]["log_max"]
    return ((torch.log10(dt_s) - lo) / (hi - lo)).clamp(0.0, 1.0)


class RawSplit:
    """One split of the cache on the device: logy [N, 100, S] fp64, end row, globals g [N, 2] (fp32), the stored times t."""

    def __init__(self, split, device):
        d = np.load(CACHE / f"raw_{split}.npz")
        self.n = d["end"].shape[0]
        self.device = device
        self.logy = torch.from_numpy(d["logy"]).to(device)
        self.end = torch.from_numpy(d["end"].astype(np.int64)).to(device)
        self.P = d["P"]
        self.T = d["T"]
        self.file_idx = d["file_idx"] if "file_idx" in d.files else None  # the raw file of each row (meta.json 'files')
        self.g = torch.from_numpy(norm_globals(self.P, self.T)).to(device)
        t = np.load(CACHE / "t_grid.npy")
        self.t = torch.from_numpy(t).to(device)
        self.logt = torch.log10(self.t[1:])            # positive rows 1..99
        self.mu, self.sd, self.zlo, self.zhi = norm_consts(device)
        self.closed = CLOSED and self.logy.shape[-1] == len(SHORT)
        if self.closed:  # labels of a complete entry set keep each trajectory's element ratios and sum 1 (logy_at)
            import elemproj as EP
            self.EP = EP
            self.nh, self.na = (t.to(device) for t in EP.stoich(SHORT))
            self.mol = EP.molidx(SHORT).to(device)  # sum 1 over molecules (reservoir sets: not over the atom reservoirs)
            self.lr = EP.log_ratios(self.logy[:, 0] * math.log(10.0), self.na, self.nh)   # [N, 4], from the start row

    w = None   # per-trajectory draw weights [n] (train.py traj_weights_file); None = uniform
    p0_anchor = None  # fp32 [n]: each row's probability that a one-step window starts at row 0 (t = 0); None = never
    p0_jump = None    # fp32 [n]: the same for jump pairs (train.py x0_anchor / x0_jump, or x0_file with x0_*_p)

    def draw(self, m, gen):
        """m trajectory indices: uniform (the original randint draw) unless self.w is set."""
        if self.w is None:
            return torch.randint(0, self.n, (m,), device=self.device, generator=gen)
        return torch.multinomial(self.w, m, replacement=True, generator=gen)

    def logy_at(self, idx, times):
        """Production interpolant: idx [B], times [B, M] (s, fp64) -> log10 y [B, M, S] fp64."""
        lt = torch.log10(times)
        n_valid = self.end[idx]                       # valid rows are 1..end (n_valid of them)
        j = torch.searchsorted(self.logt.expand(lt.shape[0], -1).contiguous(), lt.contiguous(), side="left")
        j = torch.minimum(torch.clamp(j, min=1), (n_valid - 1).reshape(-1, 1))
        i0 = j            # valid index j-1 -> raw row j
        i1 = j + 1        # valid index j   -> raw row j+1
        t0 = self.logt[j - 1]
        t1 = self.logt[j]
        w = ((lt - t0) / (t1 - t0)).clamp(0.0, 1.0)
        b = idx.reshape(-1, 1).expand_as(j)
        y0 = self.logy[b, i0]
        y1 = self.logy[b, i1]
        ly = (1.0 - w).unsqueeze(-1) * y0 + w.unsqueeze(-1) * y1
        if self.closed:  # per-species log interpolation breaks sum 1 and the ratios; restore both
            ln10 = math.log(10.0)
            lw, _ = self.EP.project_ln(ly * ln10, self.lr[idx].unsqueeze(1).expand(-1, ly.shape[1], -1), self.na, self.nh,
                                        mol=self.mol)
            ly = torch.clamp(lw / ln10, min=math.log10(EPS))
        return ly

    def z_of(self, logy):
        return (logy - self.mu) / self.sd

    def sample_grid(self, B, K, gen, t_min, end_frac=0.0):
        """B chains of K steps along the stored log-time grid, starting at a row with t >= t_min; labels are the stored
        rows (no interpolation), rows past a trajectory's end row repeat it (the samplers' constant-tail convention).
        end_frac = fraction of chains that end at the trajectory's last row (0: none; draws nothing extra). Returns z fp64
        [B, K+1, S], step lengths dt [B, K] (s), g [B, G], idx."""
        dev = self.device
        last = self.t.numel() - 1
        rlo = self.__dict__.setdefault("_r_lo", {})   # first stored row at t >= t_min, found once per t_min (no host sync
        if t_min not in rlo:                           # on later calls)
            rlo[t_min] = int(torch.searchsorted(self.t, torch.tensor([t_min], dtype=self.t.dtype, device=dev)).item())
        r_lo = rlo[t_min]
        assert r_lo + K <= last, (r_lo, K)
        idx = self.draw(B, gen)
        top = torch.clamp(torch.clamp(self.end[idx], max=last) - K, min=r_lo)
        u = torch.rand(B, device=dev, generator=gen, dtype=torch.float64)
        r0 = r_lo + torch.floor(u * (top - r_lo + 1).double()).long()
        if end_frac > 0:
            r0 = torch.where(torch.rand(B, device=dev, generator=gen) < end_frac, top, r0)
        rows = r0.reshape(-1, 1) + torch.arange(K + 1, device=dev).reshape(1, -1)
        dt = self.t[rows[:, 1:]] - self.t[rows[:, :-1]]
        lab = torch.minimum(rows, self.end[idx].reshape(-1, 1))
        return self.z_of(self.logy[idx.reshape(-1, 1).expand_as(lab), lab]), dt, self.g[idx], idx

    def sample(self, B, K, gen):
        """B windows of K transitions (K+1 states) at one dt each. Returns z fp64 [B, K+1, S], dt_norm fp32 [B],
        g fp32 [B, G], idx [B]."""
        dev = self.device
        t_lo = self.t[1]
        m = B + B // 4 + 256  # about 1% of draws do not fit; a stable sort puts fits first (no host sync)
        idx = self.draw(m, gen)
        ldt = DT_LOG[0] + (DT_LOG[1] - DT_LOG[0]) * torch.rand(m, device=dev, generator=gen, dtype=torch.float64)
        dt = torch.pow(10.0, ldt)
        t_hi = self.t[self.end[idx]] - max(FIT_STEPS, K) * dt  # windows longer than 99 transitions must fit too
        ok = t_hi > t_lo
        u = torch.rand(m, device=dev, generator=gen, dtype=torch.float64)
        lo, hi = torch.log10(t_lo), torch.log10(torch.clamp(t_hi, min=t_lo * (1 + 1e-12)))
        ts = torch.pow(10.0, lo + u * (hi - lo))
        anchor = torch.rand(m, device=dev, generator=gen) < ANCHOR_P
        ts = torch.where(anchor, t_lo.expand(m), ts)
        a0 = None
        if self.p0_anchor is not None:  # windows from row 0 (t = 0) at each row's probability; drawn after the default draws
            a0 = torch.rand(m, device=dev, generator=gen) < self.p0_anchor[idx]
            ok = torch.where(a0, self.t[self.end[idx]] - max(FIT_STEPS, K) * dt > 0, ok)
            ts = torch.where(a0, torch.zeros_like(ts), ts)
        torch._assert_async(ok.sum() >= B, "sample: fewer than B windows fit")   # on the device, no host sync
        order = torch.sort(torch.logical_not(ok).to(torch.int8), stable=True).indices[:B]
        idx, ts, dt = idx[order], ts[order], dt[order]
        if a0 is not None:
            a0 = a0[order]
        k = torch.arange(K + 1, device=dev, dtype=torch.float64)
        times = ts.reshape(-1, 1) + k.reshape(1, -1) * dt.reshape(-1, 1)
        times = torch.minimum(times, self.t[self.end[idx]].reshape(-1, 1))
        if a0 is not None:  # the interpolant needs t >= t[1]: a placeholder time, then the input is row 0 itself
            times[:, 0] = torch.where(a0, t_lo, times[:, 0])
        z = self.z_of(self.logy_at(idx, times))
        if a0 is not None:
            z[:, 0] = torch.where(a0.unsqueeze(1), self.z_of(self.logy[idx, 0]), z[:, 0])
        return z, dt_norm_of(dt).float(), self.g[idx], idx


def sample_jump_pairs(split, B, gen, d_min, d_max):
    """B pairs (z(t), z(t + D)) with D log-uniform in [d_min, d_max] s and t log-uniform in [t_lo, t_end - D]
    (pairs that do not fit are dropped via a stable sort; about 2B candidates are drawn). Returns z0, z1 (fp64),
    the unclamped dt_norm of D (may exceed 1: the head's network does not see dt), g."""
    dev = split.device
    m = 2 * B + 256
    idx = split.draw(m, gen)
    lD = np.log10(d_min) + (np.log10(d_max) - np.log10(d_min)) * torch.rand(m, device=dev, generator=gen, dtype=torch.float64)
    Dt = torch.pow(10.0, lD)
    t_lo = split.t[1]
    t_hi = split.t[split.end[idx]] - Dt
    ok = t_hi > t_lo
    u = torch.rand(m, device=dev, generator=gen, dtype=torch.float64)
    lo, hi = torch.log10(t_lo), torch.log10(torch.clamp(t_hi, min=t_lo * (1 + 1e-12)))
    t0 = torch.pow(10.0, lo + u * (hi - lo))
    a0 = None
    if split.p0_jump is not None:  # pairs from row 0 (t = 0) at each row's probability; drawn after the default draws
        a0 = torch.rand(m, device=dev, generator=gen) < split.p0_jump[idx]
        ok = torch.where(a0, t_hi >= 0, ok)
        t0 = torch.where(a0, torch.zeros_like(t0), t0)
    torch._assert_async(ok.sum() >= B, "sample_jump_pairs: fewer than B pairs fit")   # on the device, no host sync
    order = torch.sort(torch.logical_not(ok).to(torch.int8), stable=True).indices[:B]
    idx, t0, Dt = idx[order], t0[order], Dt[order]
    if a0 is None:
        zz = split.z_of(split.logy_at(idx, torch.stack([t0, t0 + Dt], 1)))
    else:  # placeholder time for row-0 starts (the interpolant needs t >= t[1]), then the input is row 0 itself
        a0 = a0[order]
        zz = split.z_of(split.logy_at(idx, torch.stack([torch.where(a0, t_lo, t0), t0 + Dt], 1)))
        zz[:, 0] = torch.where(a0.unsqueeze(1), split.z_of(split.logy[idx, 0]), zz[:, 0])
    lo_, hi_ = NORM["dt"]["log_min"], NORM["dt"]["log_max"]
    dtn = ((torch.log10(Dt) - lo_) / (hi_ - lo_)).float()
    return zz[:, 0], zz[:, 1], dtn, split.g[idx]


def cache_val(n_windows, k_states, device, seed=20261006):
    """Validation windows from the cache's validation split: z fp32 [N, k_states, S], dt_norm fp32 [N], g fp32 [N, G];
    a fixed generator, so every run sees the same windows."""
    split = RawSplit("validation", device)
    gen = torch.Generator(device=device).manual_seed(seed)
    z, dtn, g, _ = split.sample(n_windows, k_states - 1, gen)
    return z.float(), dtn, g

