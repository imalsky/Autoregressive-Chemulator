"""E13 conservation phase (2026-10-09): projection of a 12-species state onto element ratios taken from the call's input,
for use inside the rollout (lhbench --elemproj) or, later, inside the model call. torch, fp64, no data-dependent control
flow and no host transfers (CUDA-graph safe). Tracked atoms A [12, 4] (C, H, O, N per molecule) as train.py STOICH.

  ratio     y_i' = y_i exp(sum_{e in C, O, N} A_ie lam_e): lam [3] such that log C/H, O/H, N/H of y' equal those of the
            input (H is not scaled; species holding H with C, O or N move with them). Newton on the 3 unknowns in
            ln-ratio space, a fixed number of iterations, steps clipped at +-clip; a singular Jacobian gives a zero step.
  ratio_he  ratio, then one uniform scale s = tH / (E_H + tH S), tH = E_H(in) / (1 - sum12(in)), S = sum12 and E_H = H atoms
            of the ratio-projected state, so that E_H / (1 - sum12) (H to everything untracked, mostly He) keeps its input
            value; the ratios are unchanged by a uniform scale. 1 - sum12(in) is floored at 1e-30 (an input with sum12 >= 1
            then gives sum12' ~ 1, never a negative or NaN state; its H/rest target is undefined).

    project(ly, ly_in, A, mode, iters=8, clip=2.0, floor=None) -> (ly', res)
ly, ly_in: log10 mixing ratios [..., 12] (fp64); A: atoms(names) on the same device; floor: optional log10 floor applied
to ly' (the harness clamps states at log10 eps) before the residual. res [...]: max over the fixed quantities of
|log10 q(y') - log10 q(target)| in dex (ratio: C/H, O/H, N/H; ratio_he: also E_H / (1 - sum12)).
Ill-conditioned case (unit checks): when one species holds most of two elements (HCN for C and N, CO2 for C and O), their
ratio can only move through trace species and 8 steps may not reach it; the residual reports it (lhbench: nproj_bad).

    python elemproj12.py     unit checks (path states, m17fs_s0 states, random stress states), idempotence, cost
"""
import math

import torch

STOICH = {"C2H2": (2, 2, 0, 0), "CH4": (1, 4, 0, 0), "CO2": (1, 0, 2, 0), "CO": (1, 0, 1, 0), "H2O": (0, 2, 1, 0),
          "H2": (0, 2, 0, 0), "HCN": (1, 1, 0, 1), "H": (0, 1, 0, 0), "N2": (0, 0, 0, 2), "NH3": (0, 3, 0, 1),
          "OH": (0, 1, 1, 0), "O": (0, 0, 1, 0)}
LN10 = math.log(10.0)


def atoms(names, device=None):
    """[S, 4] fp64 atoms (C, H, O, N) per molecule, in the order of names (the 12 tracked species)."""
    return torch.tensor([STOICH[s] for s in names], dtype=torch.float64, device=device)


def _lnE(lny, la):
    """ln of the element totals [..., 4] of natural-log states lny [..., S]; la = ln A (-inf where A = 0)."""
    return torch.logsumexp(lny.unsqueeze(-1) + la, -2)


def project(ly, ly_in, A, mode="ratio", iters=8, clip=2.0, floor=None):
    assert mode in ("ratio", "ratio_he"), mode
    A4 = torch.cat([A[:, 0:1], A[:, 2:4], A[:, 1:2]], 1)                 # columns C, O, N, H (slices only: graph-safe)
    la, Ac = torch.log(A4), A4[:, :3]
    lny, lnin = ly * LN10, ly_in * LN10
    lEi = _lnE(lnin, la)
    tgt = lEi[..., :3] - lEi[..., 3:]                                     # ln C/H, O/H, N/H of the input
    lam = torch.zeros(tgt.shape, dtype=ly.dtype, device=ly.device)
    for _ in range(iters):
        w = lny + lam @ Ac.T
        lE = _lnE(w, la)
        F = lE[..., :3] - lE[..., 3:] - tgt
        p = torch.exp(w.unsqueeze(-1) + la - lE.unsqueeze(-2))             # [..., S, 4] share of element e's atoms in species i
        D = torch.einsum("...se,sf->...ef", p, Ac)                         # d ln E_e / d lam_f
        J = D[..., :3, :] - D[..., 3:, :]
        step = torch.linalg.solve_ex(J, -F.unsqueeze(-1))[0].squeeze(-1)  # no error check: no host sync (CUDA graphs)
        lam = lam + torch.clamp(torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0), -clip, clip)
    w = lny + lam @ Ac.T
    if mode == "ratio_he":
        tH = torch.exp(lEi[..., 3]) / torch.clamp(1.0 - torch.exp(lnin).sum(-1), min=1e-30)
        EH, S = torch.exp(_lnE(w, la)[..., 3]), torch.exp(w).sum(-1)
        w = w + torch.log(tH / (EH + tH * S)).unsqueeze(-1)
    out = w / LN10
    if floor is not None:
        out = torch.clamp(out, min=floor)
    lE = _lnE(out * LN10, la)
    res = (lE[..., :3] - lE[..., 3:] - tgt).abs().amax(-1)
    if mode == "ratio_he":
        rest = torch.clamp(1.0 - torch.pow(10.0, out).sum(-1), min=1e-30)
        res = torch.maximum(res, (lE[..., 3] - torch.log(rest) - torch.log(tH)).abs())
    return out, res / LN10


def _check():
    """Unit checks (2026-10-09): ratios to 1e-10 dex, idempotence (dex change on a second projection), cost at batch 4096.
    Inputs: VULCAN states of the 58-day paths (paths_v2x, every 10th call) with outputs = input + N(0, sd) dex per species;
    m17fs_s0's own consecutive states on those paths (bench PX_e13_m17fs_s0; ratio_he only where sum12(in) < 1, its H/rest
    target being undefined otherwise); call 10000 projected onto call 1's ratios; and, as a stress case, log-uniform random
    compositions (log10 y in [-12, -2.5], H2 0.8-0.9), where one species can hold most of two elements."""
    import json
    import sys
    import time
    from pathlib import Path
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paths
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nm = [s.replace("_evolution", "") for s in json.loads(paths.NORM12.read_text())["species_variables"]]   # harness order
    A = atoms(nm, dev)
    g = torch.Generator(device="cpu").manual_seed(0)

    def report(lab, lout, lin, mode, keep=None):
        ly, r = project(lout, lin, A, mode, floor=-30.0)
        ly2, _ = project(ly, lin, A, mode, floor=-30.0)
        if keep is not None:
            ly, ly2, r = ly[keep], ly2[keep], r[keep]
        print(f"{lab:46s} {mode:8s} rows {r.numel():7d}  max residual {float(r.max()):.1e} dex  share > 1e-10 "
              f"{float((r > 1e-10).double().mean()):.1e}  > 1e-6 {float((r > 1e-6).double().mean()):.1e}  idempotence "
              f"{float((ly2 - ly).abs().max()):.1e} dex", flush=True)

    tru = []
    for f in sorted(paths.PATHS_V2X.glob("[0-9]*.npz"), key=lambda f: int(f.stem)):
        q = np.load(f)
        sp = list(q["species"])
        tru.append(np.log10(np.maximum(q["x"][::10][:, [sp.index(k) for k in nm]], 1e-30)))
    tin = torch.as_tensor(np.concatenate(tru), device=dev)
    for sd in (0.01, 0.1, 1.0):
        tout = torch.clamp(tin + sd * torch.randn(tin.shape, generator=g, dtype=torch.float64).to(dev), max=0.0)
        for mode in ("ratio", "ratio_he"):
            report(f"VULCAN path states + N(0, {sd:g}) dex", tout, tin, mode)
    st = torch.as_tensor(np.load(paths.E / "PX_e13_m17fs_s0" / "P_paths.npz")["states"][:, 2], device=dev)  # [10000, 16, 12]
    lin, lout = st[:-1].reshape(-1, 12), st[1:].reshape(-1, 12)
    report("m17fs_s0 consecutive calls", lout, lin, "ratio")
    ok = torch.pow(10.0, lin).sum(-1) < 1.0
    report("m17fs_s0 consecutive calls, sum12(in) < 1", lout, lin, "ratio_he", keep=ok)
    print(f"  (rows with sum12(in) >= 1, H/rest target undefined: {int(ok.numel() - ok.sum())} of {ok.numel()})")
    for mode in ("ratio", "ratio_he"):
        report("m17fs_s0 call 10000 onto call 1", st[-1], st[0], mode)
    for sd in (0.01, 0.1, 1.0):
        lin = -12.0 + 9.5 * torch.rand(100_000, 12, generator=g, dtype=torch.float64)
        lin[:, nm.index("H2")] = torch.log10(0.8 + 0.1 * torch.rand(100_000, generator=g, dtype=torch.float64))
        lout = torch.clamp(lin + sd * torch.randn(100_000, 12, generator=g, dtype=torch.float64), max=0.0)
        report(f"stress: log-uniform random + N(0, {sd:g}) dex", lout.to(dev), lin.to(dev), "ratio")
    for n, mode in ((4096, "ratio"), (4096, "ratio_he"), (65536, "ratio")):   # cost of one call, eager and CUDA graph
        if dev.type != "cuda":
            break
        lin, lout = st[:-1].reshape(-1, 12)[:n].clone(), st[1:].reshape(-1, 12)[:n].clone()
        for _ in range(3):
            project(lout, lin, A, mode)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(100):
            project(lout, lin, A, mode)
        torch.cuda.synchronize()
        te = (time.perf_counter() - t) / 100
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                project(lout, lin, A, mode)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            o = project(lout, lin, A, mode)
        gr.replay()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(200):
            gr.replay()
        torch.cuda.synchronize()
        tg = (time.perf_counter() - t) / 200
        same = bool(torch.equal(o[0], project(lout, lin, A, mode)[0]))
        print(f"cost {mode:8s} batch {n} ({torch.cuda.get_device_name(0)}): eager {te * 1e6:.0f} us/call, CUDA graph "
              f"{tg * 1e6:.0f} us/call ({tg / n * 1e9:.1f} ns per cell); graph output == eager: {same}")


if __name__ == "__main__":
    _check()
