"""E13 conservation phase (2026-10-09, plan Day 1 item 7): GPU-resident changing-condition chains (paths_v3 sets) for
train.py's chain term, in the entry set of the normalization file (the 12 species, or a reservoir set under
HARNESS_NORM), and the mixing rule that the training rollout and lhbench's arm P share.

A paths_v3 path of n calls (gen/paths_v3.py): x [n+1, 53], x[k] = the state before call k (after call k - 1, before any
mix of call k). Call k (0-based) first mixes if k is in mix_k (x <- (1 - f) x + f mix_x), then runs at T[k], Pbar[k] for
dt[k]; its truth is x[k + 1]. The store keeps z of every state x[0..n] (data.take, floored at EPS, normalized) and, per
call: g (data.norm_globals of the call's P, T), dt_norm (data.dt_norm_of), log10 number density (cm^-3, the scored mask),
the mixing fraction f (0 if none) and the partner in the entry set as LINEAR values (data.take of the 53-species
partner: linear for species and reservoirs alike, so a blend in the entry set equals take() of the 53-species blend).

    store = ChainStore([set dir, ...], device, seed)   finished paths only (<p>.npz with an integer stem; the generator
                                                       writes <p>.tmp.npz, then renames); store.info: paths per set
    w = store.windows(B, L)   B windows of L calls, (path, k0) uniform over all windows that fit (k0 + L <= n), drawn
                              with the store's own torch.Generator (store.gen) on the device. Call-major, so every call's
                              slice is contiguous: z0 [B, S] (truth before call k0), zt [L, B, S] (truth after each call),
                              g [L, B, 2], dtn [L, B], lnd [L, B], f [L, B], yp [L, B, S]; path [B], k0 [B]
    mix(z, f, yp, mu, sd)     the mixing rule on a z state (fp64): y = 10^(z sd + mu), y <- (1 - f) y + f yp, back to z
                              with y floored at EPS; rows with f = 0 are returned unchanged (bit for bit)
"""
import json
from pathlib import Path

import numpy as np
import torch

import paths
import data as D

KB = 1.380649e-16
REPORTED = [s.replace("_evolution", "") for s in json.loads(paths.NORM12.read_text())["species_variables"]]  # the 12 scored


def mix(z, f, yp, mu, sd):
    fb = f.unsqueeze(-1)
    zb = (torch.log10(torch.clamp((1.0 - fb) * torch.pow(10.0, z * sd + mu) + fb * yp, min=D.EPS)) - mu) / sd
    return torch.where(fb > 0, zb, z)


def path_files(d):
    return sorted((f for f in Path(d).glob("*.npz") if f.stem.isdigit()), key=lambda f: int(f.stem))


def mixing_arrays(p, n):
    """f [n] and partner [n, S] (linear, entry set) of the first n calls of a loaded path file (zeros without mix_k)."""
    f, yp = np.zeros(n), np.zeros((n, len(D.SHORT)))
    if "mix_k" in p.files:
        k = np.asarray(p["mix_k"], np.int64)
        sel = k < n
        f[k[sel]] = p["mix_f"][sel]
        yp[k[sel]] = D.take(np.asarray(p["mix_x"], np.float64).reshape(-1, len(p["species"]))[sel], list(p["species"]))
    return f, yp


class ChainStore:
    def __init__(self, dirs, device, seed):
        mu, sd, _, _ = D.norm_consts(device)
        ly, g, dt, lnd, f, yp, n, self.info = [], [], [], [], [], [], [], {}
        for d in dirs:
            fl = path_files(d)
            self.info[str(d)] = len(fl)
            for fp in fl:
                with np.load(fp) as p:
                    m = len(p["T"])
                    ly.append(np.log10(np.maximum(D.take(np.asarray(p["x"], np.float64), list(p["species"])), D.EPS)))
                    g.append(D.norm_globals(p["Pbar"] * 1e6, p["T"]))
                    dt.append(np.asarray(p["dt"], np.float64))
                    lnd.append(np.log10(p["Pbar"] * 1e6 / (KB * p["T"])))
                    fk, yk = mixing_arrays(p, m)
                    f.append(fk)
                    yp.append(yk)
                    n.append(m)
        assert n, f"no finished paths in {dirs}"
        self.device, self.nmax = device, max(n)
        self.n = torch.tensor(n, dtype=torch.int64, device=device)
        self.soff = torch.cumsum(self.n + 1, 0) - (self.n + 1)          # first state row of each path
        self.coff = torch.cumsum(self.n, 0) - self.n                    # first call row of each path
        self.z = (torch.from_numpy(np.concatenate(ly)).to(device) - mu) / sd
        self.g = torch.from_numpy(np.concatenate(g)).to(device)
        self.dtn = D.dt_norm_of(torch.from_numpy(np.concatenate(dt)).to(device)).float()
        self.lnd = torch.from_numpy(np.concatenate(lnd)).to(device)
        self.f = torch.from_numpy(np.concatenate(f)).to(device)
        self.yp = torch.from_numpy(np.concatenate(yp)).to(device)
        self.gen = torch.Generator(device=device).manual_seed(int(seed))

    def windows(self, B, L):
        assert 1 <= L <= self.nmax, (L, self.nmax)
        dev = self.device
        cnt = torch.clamp(self.n - L + 1, min=0)                       # windows per path
        cum = torch.cumsum(cnt, 0)
        r = torch.floor(torch.rand(B, device=dev, generator=self.gen, dtype=torch.float64) * cum[-1]).long()
        r = torch.minimum(r, cum[-1] - 1)
        p = torch.searchsorted(cum, r, right=True)
        k0 = r - (cum[p] - cnt[p])
        ar = torch.arange(L + 1, device=dev)
        # contiguous [L, B] indices: a gather keeps its index's layout, and a call's slice must not have an L-dependent
        # stride (torch.compile guards on strides: every new window length would recompile train.py's delta_fn)
        c = (self.coff[p].unsqueeze(1) + k0.unsqueeze(1) + ar[:L]).T.contiguous()       # [L, B] call rows
        s = self.soff[p].unsqueeze(1) + k0.unsqueeze(1) + ar                              # [B, L + 1] state rows
        return {"z0": self.z[s[:, 0]], "zt": self.z[s[:, 1:].T.contiguous()], "g": self.g[c], "dtn": self.dtn[c], "lnd": self.lnd[c],
                "f": self.f[c], "yp": self.yp[c], "path": p, "k0": k0}
