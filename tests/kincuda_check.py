"""The whole-call CUDA kernel (lib/kincuda.py) against the stored reference on one GPU; prints PASS / FAIL per case.

Reference = the call computed by batched fp64 Rodas3 in torch with the kernel's numerics (rates from T and P every call,
exact Jacobian, LU without row exchanges, rtol 1e-3, atol 1e-18, the step control of lib/kincuda.py), stored for
exactly these inputs (tests/libref.py, paths.LIBREF; a check of the inputs' digest refuses others). Each solver
advances its own state from the same inputs.
(a) Parity. Pass: worst floored difference <= 1e-6 dex on the 12 scored gases (floors 1e-10 H2O CO CO2 CH4 NH3 HCN
    C2H2, 1e-8 H OH O, 1e-6 H2 N2), accepted-step counts identical on all but <= 1e-4 of the cell-calls, no non-finite
    state, no unfinished call, no cell recomputed by the guard.
    - yardstick cold calls: 4,096 recorded starts at their T, P, dt 1 and 500 s, starting step 1e-10 s;
    - scrambled starts: the same kind of starts, each species times 10^(sigma N(0, 1)), sigma 0 / 1 / 2 dex, at
      dt 0.01 / 1 / 50 / 500 s;
    - a 2,000-call chain: decision panel P's 256 paths x 4 offsets (1,024 cells), T, P and dt of the recorded forcing
      every call (mixing events not applied), step sizes carried from call to call.
(b) Guard and recovery. A cell with a NaN input is counted, not recomputed, and left non-finite; a cell forced to fail
    (max_steps 10 for a cold start at dt 500 s among the reference's settled cells) and, with the rounding allowance
    DRIFT set to 0, the cells whose element shares move by more than the clamping allowance are recomputed on the
    device: within 1e-6 dex of the reference call with the recovery's tries (REC_STEPS x max_steps; the reference is
    not bitwise the kernel: its sums run in another order), counted as recovered or still failed, the other cells
    bitwise those of a solver that flags none; a cell below 300 K is counted and check() raises.
(c) Layouts: one cell per block and the packed cells per block (forced through CudaSolver.small) bitwise equal (x, h,
    done, steps, flag, counts) on the yardstick cold calls, the scrambled starts, a forced-recovery case (max_steps 30)
    and a 300-call panel-P chain; 128 cells alone (one cell per block, the automatic choice for small batches) bitwise
    the same cells inside 8,192 (packed), and 8,192 cells in a permuted order bitwise the same per cell.
    taskset -c <cpus> $PY tests/kincuda_check.py <VULCAN thermo folder: NCHO_thermo_network.txt, NASA9/>
(about 1 min on an idle GPU). Exit 0 = every case PASS."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib"))
import kincuda as KC  # noqa: E402
import kinvulcan as KV  # noqa: E402
import libref as LR  # noqa: E402
import paths  # noqa: E402

YARD = paths.YARD / "design"
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


def floored(a, b):
    """Per cell: worst |log10 max(a, S) - log10 max(b, S)| over the scored gases (columns in GASES order)."""
    fl = torch.log10(torch.tensor(FLOORS, dtype=torch.float64, device=dev))
    la = torch.maximum(torch.log10(a.clamp(min=1e-300)), fl)
    lb = torch.maximum(torch.log10(b.clamp(min=1e-300)), fl)
    return (la - lb).abs().max(1).values.nan_to_num(nan=np.inf)


def tensors(e):
    """A stored reference call as tensors: x on the scored gases, accepted steps, finished, finite."""
    return dict(x=tt(e["x"]), steps=tt(e["steps"]), done=torch.tensor(e["done"], device=dev),
                finite=torch.tensor(e["finite"], device=dev))


def verdict(name, ok, text):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {text}", flush=True)


class Tally:
    """Parity counts over the calls of one case."""

    def __init__(self):
        self.worst, self.calls, self.mis, self.nf, self.unf, self.redo = 0.0, 0, 0, [0, 0], [0, 0], 0

    def add(self, gi, ref, mod, kc):
        xb, _, db, nb = mod
        self.worst = max(self.worst, float(floored(ref["x"], xb[:, gi]).max()))
        self.calls += xb.shape[0]
        self.mis += int((ref["steps"] != nb).sum())
        for i, (fin, d) in enumerate(((ref["finite"], ref["done"]), (torch.isfinite(xb).all(1), db))):
            self.nf[i] += int(torch.logical_not(fin).sum())
            self.unf[i] += int(torch.logical_not(d).sum())
        self.redo += kc.fallback["recomputed"]

    def verdict(self, name):
        ok = (self.worst <= TOL and self.mis <= MISMATCH * self.calls and self.nf == [0, 0] and self.unf == [0, 0]
              and self.redo == 0)
        verdict(name, ok, f"worst {self.worst:.2e} dex; step-count mismatches {self.mis} of {self.calls} cell-calls; "
                          f"non-finite ref/kernel {self.nf[0]}/{self.nf[1]}; unfinished {self.unf[0]}/{self.unf[1]}; "
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
    nf, nasa = f"{a.vulcan_thermo}/NCHO_thermo_network.txt", f"{a.vulcan_thermo}/NASA9"
    net = KV.VulcanNetwork(nf, nasa, species=sp)
    sig = LR.netsig(nf, nasa, sp)
    gi = [sp.index(g) for g in GASES]
    t0 = time.time()
    kc = KC.CudaSolver(net, dev)
    print(f"kernel built and compiled in {time.time() - t0:.1f} s; {kc.wpb} warps per block, {kc.smem} B shared "
          f"memory, {kc.max_blocks} resident blocks", flush=True)

    # (a) parity
    B = 4096
    x0, T, P = yardstick(B, 0, 0.0)
    for dt in (1.0, 500.0):
        t = Tally()
        args = (x0, T, P, full(B, dt), full(B, 1e-10))
        t.add(gi, tensors(LR.entry(f"call/cold{dt:g}", sig, *args)), kc(*args), kc)
        t.verdict(f"yardstick cold calls, dt {dt:g} s")
    for sigma in (0.0, 1.0, 2.0):
        x0, T, P = yardstick(B, int(10 + sigma), sigma)
        for dt in (0.01, 1.0, 50.0, 500.0):
            t = Tally()
            args = (x0, T, P, full(B, dt), full(B, 1e-10))
            t.add(gi, tensors(LR.entry(f"call/scr{sigma:g}_{dt:g}", sig, *args)), kc(*args), kc)
            t.verdict(f"scrambled starts sigma {sigma:g} dex, dt {dt:g} s")
    r, s, F = (np.load(PANEL / f"{f}.npz") for f in ("rows", "starts", "forcing"))
    assert [str(q) for q in s["species53"]] == sp
    B, NC = 1024, 2000
    path, of = np.arange(B) % 256, (np.arange(B) // 256) * 1500
    Tc, Pc, dc = (np.stack([np.asarray(q, np.float64)[path, of + k] for k in range(NC)])
                  for q in (F["T"], F["Pbar"] * 1e6, F["dt"]))
    xb = tt(s["x53"][r["unit"][path]] / s["x53"][r["unit"][path]].sum(1, keepdims=True))
    hb = full(B, 1e-10)
    R = tensors(LR.entry("chain", sig, xb, hb, Tc, Pc, dc))
    t, tb = Tally(), 0.0
    for k in range(NC):
        torch.cuda.synchronize(); c0 = time.perf_counter()
        rb = kc(xb, tt(Tc[k]), tt(Pc[k]), tt(dc[k]), hb)
        torch.cuda.synchronize(); tb += time.perf_counter() - c0
        t.add(gi, {q: v[k] for q, v in R.items()}, rb, kc)
        xb, hb = rb[:2]
    t.verdict(f"panel P chain, {NC} calls x {B} cells (the kernel on its own state; ms per call kernel "
              f"{tb / NC * 1e3:.2f}, GPU may be shared)")

    # (b) guard
    B = 64
    x0, T, P = yardstick(B, 1, 0.0)
    G = LR.entry("guard", sig, x0, T, P)
    xs, hs = tt(G["xs"]), tt(G["hs"])                                     # the reference's settled states (100 s)
    keep = torch.arange(B, device=dev) != 3
    x = xs.clone()
    x[3, 5] = float("nan")
    y, _, d, _ = kc(x, T, P, full(B, 1.0), hs)
    fb = dict(kc.fallback)
    ya = tensors(LR.arrays("guard/nan"))["x"][keep]                     # the reference from xs, hs at dt 1 s
    e = float(floored(y[keep][:, gi], ya).max())
    ok = (fb["bad_input"] == 1 and fb["recomputed"] == 0 and not bool(torch.isfinite(y[3]).all()) and not bool(d[3])
          and e <= TOL and bool(d[keep].all()))
    verdict("guard: NaN input", ok, f"counts {fb}; row 3 non-finite and unfinished, the other {B - 1} cells within "
                                    f"{e:.1e} dex of the reference")

    x, h, dt = xs.clone(), hs.clone(), full(B, 1.0)
    x[7], h[7], dt[7] = x0[7], 1e-10, 500.0                             # cell 7: cold start at 500 s
    y, _, d, n, flag = kc.launch(x, T, P, dt, h, max_steps=10)
    fb = dict(kc.fallback)
    i = torch.tensor([7], device=dev)
    R = tensors(LR.entry("guard/forced", sig, x[i], T[i], P[i], dt[i], h[i], np.array(10 * KC.REC_STEPS)))
    yr, dr, nr = R["x"], R["done"], R["steps"]
    e = float(floored(y[i][:, gi], yr).max())
    ok = (fb["unfinished"] == 1 and fb["recomputed"] == 1 and fb["recovered"] + fb["still_failed"] == 1 and e <= TOL
          and bool(torch.equal(n[i], nr)) and bool(torch.equal(d[i], dr)) and int(flag[7]) == (1 if bool(dr) else 2)
          and bool(d[torch.arange(B, device=dev) != 7].all()))
    verdict("guard: forced failure (max_steps 10)", ok,
            f"counts {fb}; cell 7 recomputed with {10 * KC.REC_STEPS} tries ({e:.1e} dex from the reference call "
            f"with those tries, {int(nr)} accepted steps, finished {bool(dr)}, flag {int(flag[7])})")

    kz = KC.CudaSolver(net, dev, drift=0.0)          # every rounding change of an element share is flagged
    x0, T, P = yardstick(256, 2, 1.0)
    args = (x0, T, P, full(256, 500.0), full(256, 1e-10))
    y, _, _, _, flag = kz.launch(*args)
    fb = dict(kz.fallback)
    yk, _, _, _, flag0 = kc.launch(*args)                               # the same call, nothing flagged
    i, o = (flag > 0).nonzero().squeeze(1), flag == 0
    yr = tensors(LR.entry("guard/drift", sig, *args, np.array(4000 * KC.REC_STEPS)))["x"][i]
    e = float(floored(y[i][:, gi], yr).max())
    ok = (fb["drift"] > 0 and fb["recomputed"] == fb["drift"] and fb["recovered"] + fb["still_failed"] == fb["drift"]
          and e <= TOL and int(flag0.sum()) == 0 and bool(torch.equal(y[o], yk[o])))
    verdict("guard: element drift (allowance 0)", ok,
            f"counts {fb}; the flagged cells recomputed on the device ({e:.1e} dex from the reference call; still "
            f"failing an allowance of 0 is expected), the others bitwise the kernel's result")

    T2 = T.clone()
    T2[0] = 250.0
    y, _, d, _ = kc(*((x0, T2) + args[2:]))
    try:
        kc.check()
        verdict("guard: T < 300 K raises in check()", False, "no error")
    except ValueError as e:
        verdict("guard: T < 300 K raises in check()", kc.fallback["cold"] == 1 and not bool(d[0]),
                f"cell 0 not stepped; ValueError: {e}")

    # (c) layouts
    small, bad = kc.small, []

    def both(*a, **kw):
        out = []
        for lim in (0, 1 << 40):                                          # packed, then one cell per block
            kc.small = lim
            out.append((kc.launch(*a, **kw), kc.count.tolist()[1:9]))
        kc.small = small
        return all(torch.equal(p, q) for p, q in zip(out[0][0], out[1][0])) and out[0][1] == out[1][1]

    B = 4096
    cases = []
    x0, T, P = yardstick(B, 0, 0.0)
    cases += [(f"cold dt {dt:g}", (x0, T, P, full(B, dt), full(B, 1e-10)), {}) for dt in (1.0, 500.0)]
    cases.append(("forced recovery", (x0, T, P, full(B, 500.0), full(B, 1e-10)), dict(max_steps=30)))
    for sigma in (0.0, 1.0, 2.0):
        x0, T, P = yardstick(B, int(10 + sigma), sigma)
        cases += [(f"scrambled {sigma:g} dt {dt:g}", (x0, T, P, full(B, dt), full(B, 1e-10)), {})
                  for dt in (0.01, 1.0, 50.0, 500.0)]
    bad = [n for n, a, kw in cases if not both(*a, **kw)]
    B = 1024
    path = np.arange(B) % 256
    xa = tt(s["x53"][r["unit"][path]] / s["x53"][r["unit"][path]].sum(1, keepdims=True))
    st, chain = [(xa, full(B, 1e-10)), (xa, full(B, 1e-10))], True
    for k in range(300):                                     # the panel-P chain's paths, offsets and forcing
        fo = (tt(Tc[k]), tt(Pc[k]), tt(dc[k]))
        o = []
        for i, lim in enumerate((0, 1 << 40)):
            kc.small = lim
            o.append(kc.launch(st[i][0], *fo, st[i][1]))
        kc.small = small
        chain = chain and all(torch.equal(p, q) for p, q in zip(*o))
        st = [q[:2] for q in o]
    x0, T, P = yardstick(8192, 7, 1.0)
    a = (x0, T, P, full(8192, 50.0), full(8192, 1e-10))
    big, sel = kc.launch(*a), torch.arange(0, 8192, 64, device=dev)
    perm = torch.randperm(8192, generator=torch.Generator().manual_seed(3)).to(dev)
    shuf = kc.launch(*(q[perm] for q in a))
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(8192, device=dev)
    alone = 128 <= kc.small < 8192 and all(torch.equal(p[sel], q)
                                           for p, q in zip(big, kc.launch(*(q[sel] for q in a))))
    order = all(torch.equal(p, q[inv]) for p, q in zip(big, shuf))
    verdict("layouts: one cell per block and packed", not bad and chain and alone and order,
            f"{len(cases)} cases x 4096 cells bitwise equal except {bad or 'none'}; 300-call chain x {B} cells "
            f"{chain}; 128 cells alone (one cell per block, small = {kc.small} cells) bitwise inside 8,192 (packed) "
            f"{alone}; 8,192 cells permuted bitwise {order}")

    print(f"{sum(results)} of {len(results)} cases PASS", flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
