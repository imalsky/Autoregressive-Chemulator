"""The whole-call CUDA kernel (lib/kincuda.py) against the library path on one GPU; prints PASS / FAIL per case.

Library = the production call: TorchRates + rodas3_step(pivot=False) under torch.compile + advance_compact(rtol 1e-3,
atol 1e-18, order 3). Each solver advances its own state from the same inputs.
(a) Parity. Pass: worst floored difference <= 1e-6 dex on the 12 scored gases (floors 1e-10 H2O CO CO2 CH4 NH3 HCN
    C2H2, 1e-8 H OH O, 1e-6 H2 N2), accepted-step counts identical on all but <= 1e-4 of the cell-calls, no non-finite
    state, no unfinished call, no cell recomputed by the guard.
    - yardstick cold calls: 4,096 recorded starts at their T, P, dt 1 and 500 s, starting step 1e-10 s;
    - scrambled starts: the same kind of starts, each species times 10^(sigma N(0, 1)), sigma 0 / 1 / 2 dex, at
      dt 0.01 / 1 / 50 / 500 s;
    - a 2,000-call chain: decision panel P's 256 paths x 4 offsets (1,024 cells), T, P and dt of the recorded forcing
      every call (mixing events not applied), step sizes carried from call to call.
(b) Guard. A cell with a NaN input is counted and left non-finite; a cell forced to fail (max_steps 10 for a cold start
    at dt 500 s among settled cells) and, with the rounding allowance DRIFT set to 0, the cells whose element shares
    move by more than the clamping allowance are recomputed: their kernel result is replaced, within 1e-6 dex of a
    separate library call (the library path is not bitwise reproducible: index_add accumulates with atomics); a cell
    below 300 K raises.
(c) Paired timing: GCM-like forcing (panel P's paths at staggered offsets) at 768 and 32,768 cells, 150 library warm-up
    calls, then 50 calls in which both solvers get the same state, T, P, dt and h (state advanced by the library's
    output); whole calls including rates, the guard and the host sync.
    taskset -c <cpus> $PY tests/kincuda_check.py <VULCAN thermo folder: NCHO_thermo_network.txt, NASA9/>
(about 2 min on an idle GPU with a warm compile cache). Exit 0 = every case PASS."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib"))
import kinbatch as KB  # noqa: E402
import kincuda as KC  # noqa: E402
import kinnet as KN  # noqa: E402
import kinvulcan as KV  # noqa: E402
import paths  # noqa: E402

YARD = paths.DATA / "benches/yardstick/design"
PANEL = paths.DESIGNS / "decision/P"
GASES = ["H2O", "CO", "CO2", "CH4", "NH3", "HCN", "C2H2", "H", "OH", "O", "H2", "N2"]
FLOORS = [1e-10] * 7 + [1e-8] * 3 + [1e-6] * 2
TOL, MISMATCH = 1e-6, 1e-4
dev = torch.device("cuda")
results = []


def tt(a):
    return torch.tensor(np.asarray(a), dtype=torch.float64, device=dev)


def full(B, v):
    return torch.full((B,), v, dtype=torch.float64, device=dev)


def floored(a, b, gi):
    """Per cell: worst |log10 max(a, S) - log10 max(b, S)| over the scored gases."""
    fl = torch.log10(torch.tensor(FLOORS, dtype=torch.float64, device=dev))
    la = torch.maximum(torch.log10(a[:, gi].clamp(min=1e-300)), fl)
    lb = torch.maximum(torch.log10(b[:, gi].clamp(min=1e-300)), fl)
    return (la - lb).abs().max(1).values.nan_to_num(nan=np.inf)


def verdict(name, ok, text):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {text}", flush=True)


class Tally:
    """Parity counts over the calls of one case."""

    def __init__(self):
        self.worst, self.calls, self.mis, self.nf, self.unf, self.redo = 0.0, 0, 0, [0, 0], [0, 0], 0

    def add(self, gi, lib, mod, kc):
        (xa, _, da, na), (xb, _, db, nb) = lib, mod
        self.worst = max(self.worst, float(floored(xa, xb, gi).max()))
        self.calls += xa.shape[0]
        self.mis += int((na != nb).sum())
        for i, (x, d) in enumerate(((xa, da), (xb, db))):
            self.nf[i] += int(torch.logical_not(torch.isfinite(x).all(1)).sum())
            self.unf[i] += int(torch.logical_not(d).sum())
        self.redo += kc.fallback["recomputed"]

    def verdict(self, name):
        ok = (self.worst <= TOL and self.mis <= MISMATCH * self.calls and self.nf == [0, 0] and self.unf == [0, 0]
              and self.redo == 0)
        verdict(name, ok, f"worst {self.worst:.2e} dex; step-count mismatches {self.mis} of {self.calls} cell-calls; "
                          f"non-finite lib/kernel {self.nf[0]}/{self.nf[1]}; unfinished {self.unf[0]}/{self.unf[1]}; "
                          f"recomputed by the guard {self.redo}")


def yardstick(B, seed, sigma):
    """B recorded starts (x, T, P), each species scaled by 10^(sigma N(0, 1)), renormalized."""
    u, r = np.load(YARD / "units.npz"), np.load(YARD / "rows.npz")
    row_of_unit = {int(q): i for i, q in enumerate(r["unit"])}
    rng = np.random.default_rng(seed)
    units = rng.choice(len(u["t0"]), B, replace=True)
    idx = np.array([row_of_unit[int(q)] for q in units])
    x = u["x53"][units]
    if sigma:
        x = x * 10.0 ** (sigma * rng.standard_normal(x.shape))
    return tt(x / x.sum(1, keepdims=True)), tt(r["T"][idx]), tt(r["P_barye"][idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vulcan_thermo", help="folder with NCHO_thermo_network.txt and NASA9/")
    a = ap.parse_args()
    sp = [str(s) for s in np.load(YARD / "units.npz")["species53"]]
    net = KV.VulcanNetwork(f"{a.vulcan_thermo}/NCHO_thermo_network.txt", f"{a.vulcan_thermo}/NASA9", species=sp)
    gi = [sp.index(g) for g in GASES]
    m, rates = KB.Mech(net, dev), KV.TorchRates(net, dev)
    step = torch.compile(lambda N, h, kf, kr, nd: KB.rodas3_step(m, N, kf, kr, nd, h, pivot=False), dynamic=True)
    lib = lambda x, T, P, dt, h, **kw: KN.advance_compact(step, x, rates(T, P), dt, h, order=3, **kw)
    t0 = time.time()
    kc = KC.CudaSolver(net, dev)
    print(f"kernel built and compiled in {time.time() - t0:.1f} s; {kc.wpb} warps per block, {kc.smem} B shared "
          f"memory, {kc.max_blocks} resident blocks", flush=True)

    # (a) parity
    B = 4096
    x0, T, P = yardstick(B, 0, 0.0)
    for dt in (1.0, 500.0):
        t = Tally()
        args = (x0, T, P, full(B, dt))
        t.add(gi, lib(*args, full(B, 1e-10)), kc(*args, full(B, 1e-10)), kc)
        t.verdict(f"yardstick cold calls, dt {dt:g} s")
    for sigma in (0.0, 1.0, 2.0):
        x0, T, P = yardstick(B, int(10 + sigma), sigma)
        for dt in (0.01, 1.0, 50.0, 500.0):
            t = Tally()
            args = (x0, T, P, full(B, dt))
            t.add(gi, lib(*args, full(B, 1e-10)), kc(*args, full(B, 1e-10)), kc)
            t.verdict(f"scrambled starts sigma {sigma:g} dex, dt {dt:g} s")
    r, s, F = (np.load(PANEL / f"{f}.npz") for f in ("rows", "starts", "forcing"))
    assert [str(q) for q in s["species53"]] == sp
    Tm, Pm, dtm = tt(F["T"]), tt(F["Pbar"] * 1e6), tt(F["dt"])
    B, NC = 1024, 2000
    path = np.arange(B) % 256
    pt, of = torch.tensor(path, device=dev), torch.tensor((np.arange(B) // 256) * 1500, device=dev)
    xa = tt(s["x53"][r["unit"][path]] / s["x53"][r["unit"][path]].sum(1, keepdims=True))
    xb, ha, hb = xa.clone(), full(B, 1e-10), full(B, 1e-10)
    t, ta, tb = Tally(), 0.0, 0.0
    for k in range(NC):
        Tk, Pk, dk = Tm[pt, of + k], Pm[pt, of + k], dtm[pt, of + k]
        torch.cuda.synchronize(); c0 = time.perf_counter()
        ra = lib(xa, Tk, Pk, dk, ha)
        torch.cuda.synchronize(); c1 = time.perf_counter()
        rb = kc(xb, Tk, Pk, dk, hb)
        torch.cuda.synchronize(); c2 = time.perf_counter()
        ta, tb = ta + c1 - c0, tb + c2 - c1
        t.add(gi, ra, rb, kc)
        (xa, ha), (xb, hb) = ra[:2], rb[:2]
    t.verdict(f"panel P chain, {NC} calls x {B} cells (each solver on its own state; ms per call library "
              f"{ta / NC * 1e3:.2f}, kernel {tb / NC * 1e3:.2f}, GPU may be shared)")

    # (b) guard
    B = 64
    x0, T, P = yardstick(B, 1, 0.0)
    xs, hs, _, _ = lib(x0, T, P, full(B, 100.0), full(B, 1e-10))        # settled states
    keep = torch.arange(B, device=dev) != 3
    x = xs.clone()
    x[3, 5] = float("nan")
    y, _, d, _ = kc(x, T, P, full(B, 1.0), hs)
    fb = dict(kc.fallback)
    ya = lib(x[keep], T[keep], P[keep], full(B - 1, 1.0), hs[keep])[0]
    e = float(floored(y[keep], ya, gi).max())
    ok = (fb["bad_input"] == 1 and fb["recomputed"] == 0 and not bool(torch.isfinite(y[3]).all()) and not bool(d[3])
          and e <= TOL and bool(d[keep].all()))
    verdict("guard: NaN input", ok, f"counts {fb}; row 3 non-finite and unfinished, the other {B - 1} cells within "
                                    f"{e:.1e} dex of the library")

    x, h, dt = xs.clone(), hs.clone(), full(B, 1.0)
    x[7], h[7], dt[7] = x0[7], 1e-10, 500.0                             # cell 7: cold start at 500 s
    yk = kc.launch(x, T, P, dt, h, max_steps=10)[0]
    y, _, d, n = kc(x, T, P, dt, h, max_steps=10)
    fb = dict(kc.fallback)
    i = torch.tensor([7], device=dev)
    yr, _, dr, nr = kc.library(x[i], T[i], P[i], dt[i], h[i], max_steps=10)
    e = float(floored(y[i], yr, gi).max())
    ok = (fb["unfinished"] == 1 and fb["recomputed"] == 1 and not bool(torch.equal(y[i], yk[i])) and e <= TOL
          and bool(torch.equal(n[i], nr)) and bool(torch.equal(d[i], dr))
          and bool(d[torch.arange(B, device=dev) != 7].all()))
    verdict("guard: forced failure (max_steps 10)", ok,
            f"counts {fb}; cell 7 replaced by the library path's result ({e:.1e} dex from a separate library call, "
            f"{int(nr)} accepted steps, finished {bool(dr)})")

    drift0 = KC.DRIFT
    KC.DRIFT = 0.0                                   # every rounding change of an element share is flagged
    kz = KC.CudaSolver(net, dev)
    KC.DRIFT = drift0
    x0, T, P = yardstick(256, 2, 1.0)
    args = (x0, T, P, full(256, 500.0), full(256, 1e-10))
    yk, _, _, _, redo = kz.launch(*args)
    y, _, _, _ = kz(*args)
    fb = dict(kz.fallback)
    i = redo.nonzero().squeeze(1)
    yr = kz.library(x0[i], T[i], P[i], full(len(i), 500.0), full(len(i), 1e-10))[0]
    e = float(floored(y[i], yr, gi).max())
    ok = (fb["drift"] > 0 and fb["recomputed"] == fb["drift"] and e <= TOL
          and not bool((y[i] == yk[i]).all(1).any())
          and bool(torch.equal(y[torch.logical_not(redo)], yk[torch.logical_not(redo)])))
    verdict("guard: element drift (allowance 0)", ok, f"counts {fb}; the flagged cells replaced by the library path's "
                                                      f"result ({e:.1e} dex from a separate library call), the others "
                                                      f"keep the kernel's result")

    T2 = T.clone()
    T2[0] = 250.0
    try:
        kc(*((x0, T2) + args[2:]))
        verdict("guard: T < 300 K raises", False, "no error")
    except ValueError as e:
        verdict("guard: T < 300 K raises", True, f"ValueError: {e}")

    # (c) paired timing
    for B in (768, 32768):
        path = np.arange(B) % 256
        pt, of = torch.tensor(path, device=dev), torch.tensor((np.arange(B) // 256) * 37 % 9000, device=dev)
        x = tt(s["x53"][r["unit"][path]] / s["x53"][r["unit"][path]].sum(1, keepdims=True))
        h = full(B, 1e-10)
        for k in range(150):
            x, h, _, _ = lib(x, Tm[pt, of + k], Pm[pt, of + k], dtm[pt, of + k], h)
        tim, redo, NC = [0.0, 0.0], 0, 50
        for k in range(150, 150 + NC):
            Tk, Pk, dk = Tm[pt, of + k], Pm[pt, of + k], dtm[pt, of + k]
            out = []
            for i, f in enumerate((lib, kc)):
                torch.cuda.synchronize(); c0 = time.perf_counter()
                out.append(f(x, Tk, Pk, dk, h.clone()))
                torch.cuda.synchronize(); tim[i] += time.perf_counter() - c0
            redo += kc.fallback["recomputed"]
            x, h = out[0][:2]
        ms = [q / NC * 1e3 for q in tim]
        print(f"TIME  {B} cells, GCM-like forcing, {NC} calls (paired, GPU may be shared): library "
              f"{ms[0] / B * 1e3:.3f} us per cell-call ({ms[0]:.2f} ms per call), kernel {ms[1] / B * 1e3:.3f} us per "
              f"cell-call ({ms[1]:.2f} ms per call), ratio {ms[0] / ms[1]:.1f}; recomputed by the guard {redo}",
              flush=True)

    print(f"{sum(results)} of {len(results)} cases PASS", flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
