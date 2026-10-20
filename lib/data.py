"""GPU-resident raw trajectories, the production normalization and a production-law window sampler.

Windows follow processing/preprocessing.py: dt log-uniform in [0.1, 1e3] s; t_start is anchored at
the first positive raw time with probability 1/4 (draw 0 of 4) or log-uniform in
[t_lo, t_end - FIT_STEPS * dt]; windows that do not fit are redrawn. Values are the production
log-log interpolant (log10 y linear in log10 t between valid raw rows), computed in fp64.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import paths

REPO = paths.SNAPSHOT
CACHE = Path(os.environ.get("ARCH_CACHE", str(paths.ARCH_TESTS)))  # E8: alternative cache via ARCH_CACHE
NORM_PATH = Path(os.environ.get("HARNESS_NORM") or str(paths.NORM12))  # empty = unset  # E13: other species sets
NORM = json.loads(NORM_PATH.read_text())   # 2026-10-19: NORM_PATH kept for the model manifests (train.write_manifest)
SPECIES = list(NORM["species_variables"])
SHORT = [s.replace("_evolution", "") for s in SPECIES]
CLOSED = "He" in SHORT  # E13: a complete species set (He included): labels are projected in RawSplit.logy_at
FIT_STEPS = 99          # preprocessing n_steps - 1: every window must fit 99 transitions
ANCHOR_P = 0.25         # draw 0 of samples_per_source_trajectory = 4 is anchored
DT_LOG = (-1.0, 3.0)    # window dt, log10 s (production law [0.1, 1e3]); train.py sets it from dt_min_s / dt_max_s
X0_ANCHOR_P = 0.0       # sprint 2026-10-08 (arm M): share of the windows of rows flagged in RawSplit.x0 that start at row 0
X0_JUMP_P = 0.0         # sprint 2026-10-08 (arm M): share of the jump pairs of flagged rows that start at row 0
EPS = float(NORM["epsilon"])


def take(x, names):
    """E13: SHORT columns of linear mixing ratios x [..., len(names)] over the full network (names). Reservoir entries
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
    """One split of the cache on the GPU: logy [N, 100, S] fp64, end row, converged-tail row, globals."""

    def __init__(self, split, device, limit=None, frac=1.0, frac_seed=0, rows=None, cache=None):
        root = CACHE if cache is None else Path(cache)  # 2026-10-09: another cache's split (lhbench --g-cache)
        d = np.load(root / f"raw_{split}.npz")
        n_all = d["end"].shape[0]
        sel = np.arange(n_all if limit is None else min(limit, n_all))
        if frac < 1.0:  # data-scaling test: a seeded random subset of the trajectories
            sel = np.sort(np.random.default_rng(frac_seed).choice(n_all, size=int(round(frac * n_all)), replace=False))
        if rows is not None:  # E13 (2026-10-07): an explicit list of the split's trajectories (train.py train_rows)
            sel = np.asarray(rows, np.int64)
        n = sel.size
        self.n = n
        self.device = device
        self.logy = torch.from_numpy(d["logy"][sel]).to(device)
        self.end = torch.from_numpy(d["end"][sel].astype(np.int64)).to(device)
        self.conv = torch.from_numpy(d["conv"][sel].astype(np.int64)).to(device)
        self.P = d["P"][sel]
        self.T = d["T"][sel]
        self.seed = d["seed"][sel]
        self.lo = d["lo"][sel]
        self.name = d["name"][sel]
        self.file_idx = d["file_idx"][sel] if "file_idx" in d.files else None  # sprint 2026-10-08: raw file of each row
        self.g = torch.from_numpy(norm_globals(self.P, self.T)).to(device)
        # E13 review (2026-10-07): log10 total number density (cm^-3; P in barye) per trajectory, for the scored mask
        self.lnd = torch.from_numpy(np.log10(np.asarray(self.P, np.float64) / (1.380649e-16 * np.asarray(self.T, np.float64)))).to(device)
        t = np.load(root / "t_grid.npy")
        self.t = torch.from_numpy(t).to(device)
        self.logt = torch.log10(self.t[1:])            # positive rows 1..99
        self.mu, self.sd, self.zlo, self.zhi = norm_consts(device)
        self.closed = CLOSED and self.logy.shape[-1] == len(SHORT)  # lhbench also loads the 12-species test cache
        if self.closed:  # E13: labels of a complete species set keep each trajectory's element ratios and sum 1 (logy_at)
            import elemproj as EP
            self.EP = EP
            self.nh, self.na = (t.to(device) for t in EP.stoich(SHORT))
            self.mol = EP.molidx(SHORT).to(device)  # sum 1 over molecules (reservoir sets: not over the atom reservoirs)
            self.lr = EP.log_ratios(self.logy[:, 0] * math.log(10.0), self.na, self.nh)   # [N, 4], from the start row

    w = None  # E13 (2026-10-07): per-trajectory sampling weights [n] (train.py traj_w); None = uniform
    x0 = None  # sprint 2026-10-08 (arm M): bool [n], rows whose row 0 (the start, t = 0) can be a training input

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
        if self.closed:  # per-species log interpolation breaks sum 1 and the ratios; restore both (E13, second review)
            ln10 = math.log(10.0)
            lw, _ = self.EP.project_ln(ly * ln10, self.lr[idx].unsqueeze(1).expand(-1, ly.shape[1], -1), self.na, self.nh,
                                        mol=self.mol)
            ly = torch.clamp(lw / ln10, min=math.log10(EPS))
        return ly

    def z_of(self, logy):
        return (logy - self.mu) / self.sd

    def sample_grid(self, B, K, gen, t_min, stride=1, end_frac=0.0):
        """E13 (Fable review 2026-10-06, proposal 1): B chains of K steps along the stored log-time grid, starting at a
        row with t >= t_min; labels are the stored rows (no interpolation), rows past a trajectory's end row repeat it
        (the sampler's constant-tail convention). Returns z fp64 [B, K+1, S], step lengths dt [B, K] (s), g [B, G], idx.
        E13 review (2026-10-07): stride = stored rows per step (chains span stride x K rows); end_frac = fraction of
        chains that end at the trajectory's last row (0: none; draws nothing extra)."""
        dev = self.device
        last = self.t.numel() - 1
        rlo = self.__dict__.setdefault("_r_lo", {})   # 2026-10-19: first stored row at t >= t_min, found once per t_min
        if t_min not in rlo:                           # (the .item() was a host sync on every call)
            rlo[t_min] = int(torch.searchsorted(self.t, torch.tensor([t_min], dtype=self.t.dtype, device=dev)).item())
        r_lo = rlo[t_min]
        assert r_lo + stride * K <= last, (r_lo, stride, K)
        idx = self.draw(B, gen)
        top = torch.clamp(torch.clamp(self.end[idx], max=last) - stride * K, min=r_lo)
        u = torch.rand(B, device=dev, generator=gen, dtype=torch.float64)
        r0 = r_lo + torch.floor(u * (top - r_lo + 1).double()).long()
        if end_frac > 0:
            r0 = torch.where(torch.rand(B, device=dev, generator=gen) < end_frac, top, r0)
        rows = r0.reshape(-1, 1) + stride * torch.arange(K + 1, device=dev).reshape(1, -1)
        dt = self.t[rows[:, 1:]] - self.t[rows[:, :-1]]
        lab = torch.minimum(rows, self.end[idx].reshape(-1, 1))
        return self.z_of(self.logy[idx.reshape(-1, 1).expand_as(lab), lab]), dt, self.g[idx], idx

    def sample(self, B, K, gen, steady_frac=0.0):
        """B windows of K transitions (K+1 states) at one dt each. Returns z fp64 [B, K+1, S],
        dt_norm fp32 [B], g fp32 [B, G], idx [B]. steady_frac oversamples windows that start inside
        a converged tail (fixed-point supervision)."""
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
        if X0_ANCHOR_P > 0 and self.x0 is not None:  # arm M: windows from row 0 (t = 0); drawn after the default draws
            a0 = self.x0[idx] & (torch.rand(m, device=dev, generator=gen) < X0_ANCHOR_P)
            ok = torch.where(a0, self.t[self.end[idx]] - max(FIT_STEPS, K) * dt > 0, ok)
            ts = torch.where(a0, torch.zeros_like(ts), ts)
        torch._assert_async(ok.sum() >= B, "sample: fewer than B windows fit")   # 2026-10-19: on the device, no host sync
        order = torch.sort(torch.logical_not(ok).to(torch.int8), stable=True).indices[:B]
        idx, ts, dt = idx[order], ts[order], dt[order]
        if a0 is not None:
            a0 = a0[order]
        if steady_frac > 0:
            ns = int(round(steady_frac * B))
            cand = torch.nonzero(self.conv >= 1).squeeze(1)
            pick = cand[torch.randint(0, cand.numel(), (ns,), device=dev, generator=gen)]
            ldt = DT_LOG[0] + (DT_LOG[1] - DT_LOG[0]) * torch.rand(ns, device=dev, generator=gen, dtype=torch.float64)
            dts = torch.pow(10.0, ldt)
            tc = self.t[self.conv[pick]]
            # start inside the converged tail; times past t_end are held at the last row (constant tail)
            idx[:ns], ts[:ns], dt[:ns] = pick, tc * 1.0001, dts
            if a0 is not None:
                a0[:ns] = False
        k = torch.arange(K + 1, device=dev, dtype=torch.float64)
        times = ts.reshape(-1, 1) + k.reshape(1, -1) * dt.reshape(-1, 1)
        times = torch.minimum(times, self.t[self.end[idx]].reshape(-1, 1))
        if a0 is not None:  # the interpolant needs t >= t[1]: a placeholder time, then the input is row 0 itself
            times[:, 0] = torch.where(a0, t_lo, times[:, 0])
        z = self.z_of(self.logy_at(idx, times))
        if a0 is not None:
            z[:, 0] = torch.where(a0.unsqueeze(1), self.z_of(self.logy[idx, 0]), z[:, 0])
        return z, dt_norm_of(dt).float(), self.g[idx], idx


def sample_jump_pairs(split, B, gen, d_min, d_max, with_idx=False):
    """B pairs (z(t), z(t + D)) with D log-uniform in [d_min, d_max] s and t log-uniform in [t_lo, t_end - D]
    (pairs that do not fit are dropped via a stable sort; about 2B candidates are drawn). Returns z0, z1 (fp64),
    the unclamped dt_norm of D (may exceed 1: only for heads whose network does not see dt), g."""
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
    if X0_JUMP_P > 0 and split.x0 is not None:  # arm M: pairs from row 0 (t = 0); drawn after the default draws
        a0 = split.x0[idx] & (torch.rand(m, device=dev, generator=gen) < X0_JUMP_P)
        ok = torch.where(a0, t_hi >= 0, ok)
        t0 = torch.where(a0, torch.zeros_like(t0), t0)
    torch._assert_async(ok.sum() >= B, "sample_jump_pairs: fewer than B pairs fit")   # 2026-10-19: on the device, no sync
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
    return (zz[:, 0], zz[:, 1], dtn, split.g[idx]) + ((idx,) if with_idx else ())


def cache_val(n_windows, k_states, device, seed=20261006):
    """E13: validation windows from the cache's validation split (for species sets without production shards):
    z fp32 [N, k_states, S], dt_norm fp32 [N], g fp32 [N, G]; a fixed generator, so every run sees the same windows."""
    split = RawSplit("validation", device)
    gen = torch.Generator(device=device).manual_seed(seed)
    z, dtn, g, _ = split.sample(n_windows, k_states - 1, gen)
    return z.float(), dtn, g


def load_val_shards(n_windows, k_states, device, split="validation"):
    """The first n_windows rows of the production shards (fp32 z), first k_states states."""
    root = paths.SHARDS / split
    ys, ds, gs, got = [], [], [], 0
    for p in sorted(root.glob("shard_*.npz")):
        with np.load(p) as z:
            ys.append(z["y_mat"][:, :k_states])
            ds.append(z["dt_norm_mat"][:, 0])
            gs.append(z["globals"])
        got += ys[-1].shape[0]
        if got >= n_windows:
            break
    y = torch.from_numpy(np.concatenate(ys)[:n_windows]).to(device)
    dtn = torch.from_numpy(np.concatenate(ds)[:n_windows]).to(device)
    g = torch.from_numpy(np.concatenate(gs)[:n_windows]).to(device)
    return y, dtn, g
