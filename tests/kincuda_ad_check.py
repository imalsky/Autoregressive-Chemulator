"""The derivative kernels (lib/kincuda_ad.py, ADSolver) on one GPU; prints PASS / FAIL per case (INFO lines are not
judged).
(a) Primal parity: the jvp and vjp kernels' (x, h, done, steps) bitwise equal to CudaSolver.launch on yardstick cold
    calls (dt 1, 500 s), scrambled starts (sigma 0 / 1 / 2 dex at dt 0.01 / 1 / 50 / 500 s) and a 300-call panel-P
    chain.
(b) Against the torch-autograd reference (paths.AD_REF: the library call differentiated by torch, contracts F and
    A): tangents of ln x_out (error / the cell's largest |d ln x|, species above 1e-25) and every input cotangent
    (error / the cell's largest |reference entry|; the log-multiplier and Arrhenius cotangents on reactions away from
    balance, see away_from_balance). Pass: step counts equal, every class median <= 1e-9 and p90 <= 1e-7 over cells
    (the maxima are rounding-dominated cells, printed). h_out's adaptive tangent is judged in (c) (the reference's is
    noisier).
(c) Adaptive contract against central differences of the kernel's own primal, per input class (x, T, P, dt, h, lm, rc;
    one random direction each, relative for x, T, P, dt, h): steps 1e-3 to 1e-8 of the direction, kept where the
    accepted-step count is unchanged and the difference agrees with the next step's to 1e-2 (a plateau); the smallest
    error over those steps counts. Only cells whose output depends on the input are judged (largest |d ln x| > 1e-6,
    |d h_out / h_out| > 1e-6): elsewhere both sides are rounding. Cases: GCM-like (panel P after 150 calls, the next
    call), cold starts dt 500 s, scrambled sigma 1 dt 50 s, GCM-like states at 1000 +- 2 K (NASA-9 switch). Pass: p90
    over cells <= 1e-5 for every input, 1e-4 for the starting step size (its derivative runs through the whole chain of
    controller steps of a cold start; the torch reference's own difference test reads the same, p90 6e-5 at dt 500 s).
(d) Dot products <c, J v> vs <J^T c, v> per cell, both contracts, log-weighted cotangents (c_i = N(0, 1) / x_i),
    scale = the larger of the two sums of absolute terms; pass: p99 over cells <= 1e-6 with every input class (the
    multipliers on reactions away from balance). Printed, not judged: the multipliers on every reaction and the
    Arrhenius table (batch total).
(e) Determinism: the same call twice gives the same bits (every output incl. the Arrhenius reduction); cells permuted,
    and cells run alone, give the same per-cell bits.
(f) Several tangent directions in one jvp call: bitwise equal to one call per direction.
(g) Status: a NaN input, a cell stopped by max_steps and a dt = 0 cell among ordinary cells (counts, every other
    derivative finite); a cell below 300 K raises.
(h) A 30-call panel-P window: carried tangents (x, h; adaptive) against central differences of the whole window.
(i) The same kind of window in reverse mode (checkpoints of x and h at the call boundaries, each call's vjp with the
    cotangents of x and h carried backwards) against the carried tangents: dot products per path, both contracts.
    taskset -c <cpus> $PY tests/kincuda_ad_check.py      (GPU1; about 5 min idle, 10-15 min on a shared GPU)"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import kinbatch as KB  # noqa: E402
import kincuda as KC  # noqa: E402
import kincuda_ad as KA  # noqa: E402
import kinvulcan as KV  # noqa: E402
import paths  # noqa: E402

YARD = paths.YARD / "design"
PANEL = paths.DESIGNS / "decision/P"
dev = torch.device("cuda")


def tt(a):
    return torch.tensor(np.asarray(a), dtype=torch.float64, device=dev)


def full(B, v):
    return torch.full((B,), v, dtype=torch.float64, device=dev)


def network():
    sp = [str(s) for s in np.load(YARD / "units.npz")["species53"]]
    th = paths.VULCAN_THERMO
    return KV.VulcanNetwork(f"{th}/NCHO_thermo_network.txt", f"{th}/NASA9", species=sp), sp


def yardstick(B, seed, sigma):
    """B recorded starts (x, T, P), each species scaled by 10^(sigma N(0, 1)), renormalized (kincuda_check)."""
    u, r = np.load(YARD / "units.npz"), np.load(YARD / "rows.npz")
    row_of_unit = {int(q): i for i, q in enumerate(r["unit"])}
    rng = np.random.default_rng(seed)
    units = rng.choice(len(u["t0"]), B, replace=True)
    idx = np.array([row_of_unit[int(q)] for q in units])
    x = u["x53"][units]
    if sigma:
        x = x * 10.0 ** (sigma * rng.standard_normal(x.shape))
    return tt(x / x.sum(1, keepdims=True)), tt(r["T"][idx]), tt(r["P_barye"][idx])


def panel():
    """Panel P: (rows, starts, forcing) npz files."""
    return [np.load(PANEL / f"{f}.npz") for f in ("rows", "starts", "forcing")]


results = []
net, sp = network()
ad = KA.ADSolver(net, dev)
kc = KC.CudaSolver(net, dev)
gen = torch.Generator(device=dev).manual_seed(7)
rn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64, device=dev)
CLS = ("x", "T", "P", "dt", "h", "lm", "rc")


def verdict(name, ok, text):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {text}", flush=True)


def pct(a):
    a = np.asarray(a, float)
    return f"median {np.median(a):.1e} p90 {np.quantile(a, 0.9):.1e} max {a.max():.1e}"


def zeros_tan(x):
    zb = torch.zeros(x.shape[0], dtype=torch.float64, device=dev)
    return torch.zeros_like(x), zb, zb, zb, zb


# ---------------------------------------------------------------------------------------------------------------- (a)
def parity():
    B = 4096
    cases = []
    x0, T, P = yardstick(B, 0, 0.0)
    cases += [(f"cold dt {dt:g}", (x0, T, P, full(B, dt), full(B, 1e-10))) for dt in (1.0, 500.0)]
    for sigma in (0.0, 1.0, 2.0):
        x0, T, P = yardstick(B, int(10 + sigma), sigma)
        cases += [(f"scrambled {sigma:g} dex dt {dt:g}", (x0, T, P, full(B, dt), full(B, 1e-10)))
                  for dt in (0.01, 1.0, 50.0, 500.0)]
    bad = []
    for name, a in cases:
        ref = kc.launch(*a)[:4]
        j = ad.jvp(*a, *zeros_tan(a[0]))[:4]
        v = ad.vjp(*a, torch.zeros_like(a[0]))[:4]
        if not all(torch.equal(p, q) and torch.equal(p, w) for p, q, w in zip(ref, j, v)):
            bad.append(name)
    verdict("primal parity, single calls", not bad,
            f"{len(cases)} cases x {B} cells: jvp and vjp outputs bitwise equal to CudaSolver.launch except "
            f"{bad or 'none'}")
    r, s, F = panel()
    Tm, Pm, dtm = tt(F["T"]), tt(F["Pbar"] * 1e6), tt(F["dt"])
    B, NC = 1024, 300
    path = np.arange(B) % 256
    pt, of = torch.tensor(path, device=dev), torch.tensor((np.arange(B) // 256) * 1500, device=dev)
    xa = tt(s["x53"][r["unit"][path]] / s["x53"][r["unit"][path]].sum(1, keepdims=True))
    st = [(xa.clone(), full(B, 1e-10)) for _ in range(3)]
    same = True
    for k in range(NC):
        a = (Tm[pt, of + k], Pm[pt, of + k], dtm[pt, of + k])
        o = [kc.launch(st[0][0], *a, st[0][1]), ad.jvp(st[1][0], *a, st[1][1], *zeros_tan(xa)),
             ad.vjp(st[2][0], *a, st[2][1], torch.zeros_like(xa))]
        same = same and all(torch.equal(o[0][q], o[i][q]) for i in (1, 2) for q in range(4))
        st = [(q[0], q[1]) for q in o]
    verdict("primal parity, panel P chain", same, f"{NC} calls x {B} cells, each solver on its own state: "
                                                  f"{'bitwise equal throughout' if same else 'differs'}")


# ---------------------------------------------------------------------------------------------------------------- (b)
KEYS = {"lnA0": 0, "B0": 1, "C0": 2, "lnAi": 6, "Bi": 7, "Ci": 8}
RATES, MECH = KV.TorchRates(net, dev), KB.Mech(net, dev)


def away_from_balance(x, T, P, q=1e-3):
    """[B, R]: reactions whose forward and reverse fluxes at the state x differ by more than q of their sum. A balanced
    fast reaction's multiplier changes nothing in exact arithmetic; its computed sensitivity is rounding of the order of
    1e-16 x its one-way flux, on any implementation."""
    kf, kr, nd = RATES(T, P)
    c3 = torch.where(MECH.tb[None], nd[:, None], torch.ones_like(kf))
    (ri, _), (pi, _) = MECH.idx
    ne = torch.cat([x * nd[:, None], torch.ones_like(nd)[:, None]], 1)
    a, b = c3 * kf * ne[:, ri].prod(-1), c3 * kr * ne[:, pi].prod(-1)
    return (a - b).abs() > q * (a + b)


def vs_reference():
    D = np.load(paths.AD_REF)
    assert [str(q) for q in D["species"]] == sp
    keep = torch.ones(ad.nr, dtype=torch.bool, device=dev)
    keep[[j for j, r in enumerate(net.reac) if r["kind"] == 6]] = False      # the reference's special law is fixed
    for c in ("gcm", "cold", "scr"):
        g = lambda k: tt(D[f"{c}/{k}"])
        x, T, P, dt, h = (g(k) for k in ("x", "T", "P", "dt", "h"))
        B = x.shape[0]
        cx, ch, xr = g("cx"), g("ch"), g("x_out")
        v = {k: g(f"v_{k}") for k in ("x", "T", "P", "dt", "h", "lm")}
        varr = {k: g(f"v_arr.{k}") for k in KEYS}
        for con in ("F", "A"):
            adapt = con == "A"
            far = away_from_balance(x, T, P)
            tx_err, nk_eq = [], 0
            g_err = {k: [] for k in ("x", "T", "P", "dt", "h", "lm", "arr", "lm all", "arr all")}
            for i in range(B):
                s = slice(i, i + 1)
                drc = torch.zeros(ad.nr, 9, dtype=torch.float64, device=dev)
                for k, j in KEYS.items():
                    drc[:, j] = torch.where(keep, varr[k][i], 0.0)
                xo, ho, _, nok, dxo, dho = ad.jvp(x[s], T[s], P[s], dt[s], h[s], *(v[k][s] for k in ("x", "T", "P",
                                                  "dt", "h")), dlm=v["lm"][s], drc=drc, adapt=adapt)
                nk_eq += int(nok[0] == D[f"{c}/nok"][i])
                m = xr[i] > 1e-25
                tref = g(f"{con}/tx")[i]
                tx_err.append(float(((dxo[0] - tref).abs() / xr[i])[m].max() / (tref.abs() / xr[i])[m].max()))
                _, _, _, _, gr = ad.vjp(x[s], T[s], P[s], dt[s], h[s], cx[s], ch[s], adapt=adapt)
                rel = lambda a, b: float((a - b).abs().max() / b.abs().max()) if bool((b != 0).any()) else float(
                    (a - b).abs().max())
                for k in ("x", "T", "P", "dt", "h"):
                    g_err[k].append(rel(gr[k][0].reshape(-1), g(f"{con}/g_{k}")[i].reshape(-1)))
                ml, rl = gr["lm"][0], g(f"{con}/g_lm")[i]
                ma = torch.stack([gr["rc"][:, j] for j in KEYS.values()], 1)
                ra = torch.stack([g(f"{con}/g_arr.{k}")[i] for k in KEYS], 1)
                if bool(far[i].any()):                  # else every reaction is balanced (an equilibrated cell)
                    g_err["lm"].append(rel(ml[far[i]], rl[far[i]]))
                    g_err["arr"].append(rel(ma[keep & far[i]], ra[keep & far[i]]))
                g_err["lm all"].append(rel(ml, rl))
                g_err["arr all"].append(rel(ma[keep], ra[keep]))
            judged = [tx_err] + [q for k, q in g_err.items() if "all" not in k]
            ok = nk_eq == B and all(np.quantile(q, 0.9) <= 1e-7 and np.median(q) <= 1e-9 for q in judged)
            verdict(f"vs torch reference, {c}, contract {con}", ok,
                    f"{B} cells, step counts equal {nk_eq}/{B}; tangent of ln x {pct(tx_err)}; cotangents (error / "
                    f"largest |reference entry| per cell; lm, arr on reactions away from balance, 'all' = every "
                    f"reaction, not judged): " + "; ".join(f"{k} {pct(q)}" for k, q in g_err.items()))


# ---------------------------------------------------------------------------------------------------------------- (c)
def direction(cls, x, T, P, dt, h):
    B = x.shape[0]
    v = dict(x=torch.zeros_like(x), T=torch.zeros_like(T), P=torch.zeros_like(T), dt=torch.zeros_like(T),
             h=torch.zeros_like(T), lm=None, rc=None)
    if cls == "x":
        v["x"] = x * rn(B, ad.ns)
    elif cls in ("T", "P", "dt", "h"):
        v[cls] = dict(T=T, P=P, dt=dt, h=h)[cls] * rn(B)
        if cls == "T":
            v["T"] = torch.where(T > 301.0, v["T"], 0.0)             # stay above 300 K
    elif cls == "lm":
        v["lm"] = rn(B, ad.nr)
    else:
        v["rc"] = rn(ad.nr, 9) * torch.tensor([1.0, 0.1, 100.0] * 3, dtype=torch.float64, device=dev)
    return v


def shifted(a, v, e):
    x, T, P, dt, h = a
    return (x + e * v["x"], T + e * v["T"], P + e * v["P"], dt + e * v["dt"], h + e * v["h"],
            None if v["lm"] is None else e * v["lm"], ad.rc0 if v["rc"] is None else ad.rc0 + e * v["rc"])


def fd_case(name, a):
    x, T, P, dt, h = a
    B = x.shape[0]
    errs, hs = {}, {}
    for cls in CLS:
        v = direction(cls, *a)
        xo, ho, _, nok, dxo, dho = ad.jvp(*a, v["x"], v["T"], v["P"], v["dt"], v["h"], dlm=v["lm"], drc=v["rc"])
        fx, fh, ok = [], [], []
        for e in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
            o = []
            for sg in (1.0, -1.0):
                x_, T_, P_, dt_, h_, lm_, rc_ = shifted(a, v, sg * e)
                o.append(ad.jvp(x_, T_, P_, dt_, h_, *zeros_tan(x), lm=lm_, rc=rc_))
            fx.append((o[0][0] - o[1][0]) / (2 * e))
            fh.append((o[0][1] - o[1][1]) / (2 * e))
            ok.append((o[0][3] == nok) & (o[1][3] == nok))
        m = xo > 1e-25
        u = torch.where(m, dxo / xo.clamp(min=1e-300), torch.zeros_like(xo))
        sc = u.abs().max(1).values
        best_x, best_h = torch.full((B,), np.inf, device=dev), torch.full((B,), np.inf, device=dev)
        for k in range(len(fx) - 1):           # plateau: neighbouring steps agree; error against the JVP there
            uf = torch.where(m, fx[k] / xo.clamp(min=1e-300), torch.zeros_like(xo))
            un = torch.where(m, fx[k + 1] / xo.clamp(min=1e-300), torch.zeros_like(xo))
            stab = ok[k] & ok[k + 1]
            agree = (uf - un).abs().max(1).values / sc.clamp(min=1e-300)
            err = (u - uf).abs().max(1).values / sc.clamp(min=1e-300)
            pick = stab & (agree < 1e-2)
            best_x = torch.where(pick, torch.minimum(best_x, err), best_x)
            hr = (dho - fh[k]).abs() / dho.abs().clamp(min=1e-300)
            agh = (fh[k] - fh[k + 1]).abs() / dho.abs().clamp(min=1e-300)
            best_h = torch.where(stab & (agh < 1e-2), torch.minimum(best_h, hr), best_h)
        sens, sens_h = sc > 1e-6, dho.abs() > 1e-6 * ho      # cells whose output depends on this input
        fin = torch.isfinite(best_x) & sens
        errs[cls] = (best_x[fin].cpu().numpy(), int(fin.sum()), int(sens.sum()))
        fh_ = torch.isfinite(best_h) & sens_h
        hs[cls] = (best_h[fh_].cpu().numpy(), int(fh_.sum()), int(sens_h.sum()))
    ok = all(np.quantile(q[0], 0.9) <= (1e-4 if k == "h" else 1e-5) for k, q in errs.items() if len(q[0]))
    verdict(f"adaptive vs central differences, {name}", ok,
            f"{B} cells; error of d ln x / the cell's largest |d ln x| at the best step (cells with a plateau / cells "
            f"sensitive to the input: largest |d ln x| > 1e-6): "
            + "; ".join(f"{k} {pct(q[0])} ({q[1]}/{q[2]})" for k, q in errs.items() if len(q[0])) + " | h_out rel "
            "(|d h / h| > 1e-6): " + "; ".join(f"{k} {pct(q[0])} ({q[1]}/{q[2]})" for k, q in hs.items() if len(q[0])))


def gcm_states(B, ncalls=150, extra=0):
    r, s, F = panel()
    Tm, Pm, dtm = tt(F["T"]), tt(F["Pbar"] * 1e6), tt(F["dt"])
    path = np.arange(B) % 256
    pt, of = torch.tensor(path, device=dev), torch.tensor((np.arange(B) // 256) * 37 % 9000, device=dev)
    x = tt(s["x53"][r["unit"][path]] / s["x53"][r["unit"][path]].sum(1, keepdims=True))
    h = full(B, 1e-10)
    for k in range(ncalls):
        x, h, _, _ = kc(x, Tm[pt, of + k], Pm[pt, of + k], dtm[pt, of + k], h)
    forcing = [(Tm[pt, of + k], Pm[pt, of + k], dtm[pt, of + k]) for k in range(ncalls, ncalls + 1 + extra)]
    return x, h, forcing


def finite_differences():
    x, h, fo = gcm_states(256)
    fd_case("GCM-like (panel P, call 151)", (x, *fo[0], h))
    x0, T, P = yardstick(64, 3, 0.0)
    fd_case("cold starts dt 500 s", (x0, T, P, full(64, 500.0), full(64, 1e-10)))
    x0, T, P = yardstick(64, 4, 1.0)
    fd_case("scrambled sigma 1 dex, dt 50 s", (x0, T, P, full(64, 50.0), full(64, 1e-10)))
    x, h, fo = gcm_states(64, 150)
    T = 1000.0 + 4.0 * (torch.rand(64, generator=gen, dtype=torch.float64, device=dev) - 0.5)
    fd_case("GCM-like states at T = 1000 +- 2 K (NASA-9 range switch), dt 50 s", (x, T, fo[0][1], full(64, 50.0), h))


# ---------------------------------------------------------------------------------------------------------------- (d)
def dot_tests():
    B = 1024
    x, h, fo = gcm_states(B)
    cases = [("GCM-like", (x, *fo[0], h))]
    x0, T, P = yardstick(B, 0, 0.0)
    cases += [("cold dt 1", (x0, T, P, full(B, 1.0), full(B, 1e-10))),
              ("cold dt 500", (x0, T, P, full(B, 500.0), full(B, 1e-10)))]
    x0, T, P = yardstick(B, 11, 1.0)
    cases.append(("scrambled 1 dex dt 50", (x0, T, P, full(B, 50.0), full(B, 1e-10))))
    for name, a in cases:
        x, T, P, dt, h = a
        far = away_from_balance(x, T, P)
        for adapt in (False, True):
            lab = "adaptive" if adapt else "frozen"
            xo, ho = kc.launch(*a)[:2]
            cx, ch = rn(B, ad.ns) / xo.clamp(min=1e-30) * (xo > 1e-25), rn(B) / ho     # log-weighted cotangents
            _, _, _, _, g = ad.vjp(*a, cx, ch, adapt=adapt)
            for part, lmmask in (("x, T, P, dt, h and lm on reactions away from balance", far),
                                 ("lm on every reaction (balanced ones are rounding; not judged)", None)):
                v = dict(x=x * rn(B, ad.ns), T=T * 1e-2 * rn(B), P=P * 1e-2 * rn(B), dt=dt * 1e-2 * rn(B),
                         h=h * 1e-2 * rn(B), lm=rn(B, ad.nr) * 1e-2 * (far if lmmask is not None else 1.0))
                if lmmask is None:
                    v.update({k: v[k] * 0.0 for k in ("x", "T", "P", "dt", "h")})
                _, _, _, nok, dxo, dho = ad.jvp(*a, v["x"], v["T"], v["P"], v["dt"], v["h"], dlm=v["lm"],
                                                adapt=adapt)
                lhs = (cx * dxo).sum(1) + ch * dho
                terms = [(g[k] * v[k]).reshape(B, -1) for k in ("x", "T", "P", "dt", "h", "lm")]
                rhs = sum(q.sum(1) for q in terms)
                sc = torch.maximum((cx * dxo).abs().sum(1) + (ch * dho).abs(), sum(q.abs().sum(1) for q in terms))
                rel = ((lhs - rhs).abs() / sc).nan_to_num(0.0).cpu().numpy()
                text = (f"{B} cells, steps mean {float(nok.mean()):.2f} max {int(nok.max())}; |<c,Jv> - <J^T c,v>| / "
                        f"scale {pct(rel)}; p99 {np.quantile(rel, 0.99):.1e}")
                if lmmask is None:
                    print(f"INFO  dot products, {name}, {lab}, {part}: {text}", flush=True)
                else:
                    verdict(f"dot products, {name}, {lab}, {part}", np.quantile(rel, 0.99) <= 1e-6, text)
        for adapt in (False, True):                 # the Arrhenius direction is global: the batch total
            drc = rn(ad.nr, 9) * 1e-3
            zx, zb = torch.zeros_like(x), torch.zeros_like(T)
            xo, ho, _, _, dxo, dho = ad.jvp(*a, zx, zb, zb, zb, zb, drc=drc, adapt=adapt)
            cx, ch = rn(B, ad.ns) / xo.clamp(min=1e-30) * (xo > 1e-25), rn(B) / ho
            _, _, _, _, g = ad.vjp(*a, cx, ch, adapt=adapt)
            l_, r_ = float(((cx * dxo).sum(1) + ch * dho).sum()), float((g["rc"] * drc).sum())
            sc = float((cx * dxo).abs().sum() + (ch * dho).abs().sum())
            print(f"INFO  dot products, {name}, Arrhenius table (every reaction), {'adaptive' if adapt else 'frozen'}: "
                  f"batch total {l_:.10e} vs {r_:.10e} (difference / sum |terms| {abs(l_ - r_) / sc:.1e})", flush=True)


# ---------------------------------------------------------------------------------------------------------------- (e)
def determinism():
    B = 2048
    x0, T, P = yardstick(B, 21, 1.0)
    a = (x0, T, P, full(B, 50.0), full(B, 1e-10))
    v = (x0 * rn(B, ad.ns), T * 1e-2 * rn(B), P * 1e-2 * rn(B), full(B, 0.5) * rn(B), full(B, 1e-12) * rn(B))
    lm, dlm, drc = rn(B, ad.nr) * 1e-2, rn(B, ad.nr) * 1e-2, rn(ad.nr, 9) * 1e-3
    cx, ch = rn(B, ad.ns), rn(B)
    j1, j2 = ad.jvp(*a, *v, lm=lm, dlm=dlm, drc=drc), ad.jvp(*a, *v, lm=lm, dlm=dlm, drc=drc)
    g1, g2 = ad.vjp(*a, cx, ch, lm=lm), ad.vjp(*a, cx, ch, lm=lm)
    same = all(torch.equal(p, q) for p, q in zip(j1, j2)) and all(torch.equal(g1[4][k], g2[4][k]) for k in g1[4])
    pm = torch.randperm(B, generator=torch.Generator(device="cpu").manual_seed(3)).to(dev)
    jp = ad.jvp(*(q[pm] for q in a), *(q[pm] for q in v), lm=lm[pm], dlm=dlm[pm], drc=drc)
    gp = ad.vjp(*(q[pm] for q in a), cx[pm], ch[pm], lm=lm[pm])
    perm = all(torch.equal(p[pm], q) for p, q in zip(j1, jp)) and all(
        torch.equal(g1[4][k][pm], gp[4][k]) for k in g1[4] if k != "rc")
    sub = torch.arange(0, B, 97, device=dev)                  # every 97th cell alone, as a batch of 1
    alone = True
    for i in sub.tolist():
        s = slice(i, i + 1)
        js = ad.jvp(*(q[s] for q in a), *(q[s] for q in v), lm=lm[s], dlm=dlm[s], drc=drc)
        gs = ad.vjp(*(q[s] for q in a), cx[s], ch[s], lm=lm[s])
        alone = alone and all(torch.equal(p[s], q) for p, q in zip(j1, js)) and all(
            torch.equal(g1[4][k][s], gs[4][k]) for k in g1[4] if k != "rc")
    verdict("determinism", same and perm and alone,
            f"repeat bitwise {same}; per-cell bitwise under a permutation of the cells {perm} and for {len(sub)} cells "
            f"run alone {alone} (the Arrhenius reduction's order follows the cells)")


def directions():
    """D tangent directions in one call against D single-direction calls: bitwise."""
    B, D = 1024, 3
    x0, T, P = yardstick(B, 41, 0.0)
    xs, hs, _, _ = kc(x0, T, P, full(B, 100.0), full(B, 1e-10))
    same = True
    for a in ((x0, T, P, full(B, 500.0), full(B, 1e-10)), (xs, T, P, full(B, 10.0), hs)):
        x, T_, P_, dt, h = a
        lm = rn(B, ad.nr) * 1e-2
        v = [(x * rn(B, ad.ns), T_ * 1e-2 * rn(B), P_ * 1e-2 * rn(B), dt * 1e-2 * rn(B), h * 1e-2 * rn(B),
              rn(B, ad.nr) * 1e-2, rn(ad.nr, 9) * 1e-3) for _ in range(D)]
        st = [torch.stack([q[k] for q in v]) for k in range(7)]
        for adapt in (False, True):
            one = [ad.jvp(*a, *q[:5], lm=lm, dlm=q[5], drc=q[6], adapt=adapt) for q in v]
            many = ad.jvp(*a, *st[:5], lm=lm, dlm=st[5], drc=st[6], adapt=adapt)
            same = same and all(torch.equal(many[k], one[0][k]) for k in range(4)) and all(
                torch.equal(many[4][d], one[d][4]) and torch.equal(many[5][d], one[d][5]) for d in range(D))
    verdict("several tangent directions in one call", same,
            f"{D} directions in one jvp call bitwise equal to {D} single-direction calls (cold dt 500 s and settled "
            f"dt 10 s starts, {B} cells, both contracts): {same}")


def status():
    """A NaN input, a cell stopped by max_steps, a dt = 0 cell among ordinary ones; a cell below 300 K raises."""
    B = 64
    x0, T, P = yardstick(B, 31, 0.0)
    xs, hs, _, _ = kc(x0, T, P, full(B, 100.0), full(B, 1e-10))
    x, h, dt = xs.clone(), hs.clone(), full(B, 1.0)
    x[3, 5] = float("nan")
    x[7], h[7], dt[7] = x0[7], 1e-10, 500.0                   # cold start: stops at max_steps 10
    dt[11] = 0.0
    v = (xs * rn(B, ad.ns), T * 1e-2 * rn(B), P * 1e-2 * rn(B), full(B, 1e-2) * rn(B), h * 1e-2 * rn(B))
    cx, ch = rn(B, ad.ns), rn(B)
    ok = True
    for adapt in (False, True):
        xo, ho, done, nok, dxo, dho = ad.jvp(x, T, P, dt, h, *v, adapt=adapt, max_steps=10)
        st_j = dict(ad.status)
        _, _, _, _, g = ad.vjp(x, T, P, dt, h, cx, ch, adapt=adapt, max_steps=10)
        st_v = dict(ad.status)
        good = torch.ones(B, dtype=torch.bool, device=dev)
        good[[3, 7]] = False
        fin = bool(torch.isfinite(dxo[good]).all()) and bool(torch.isfinite(g["x"][good]).all()) and bool(
            torch.isfinite(g["lm"][good]).all()) and bool(torch.isfinite(g["rc"]).all())
        flagged = ad.flag.nonzero().squeeze(1).tolist()
        ok = ok and st_j == st_v and st_j["bad_input"] == 1 and st_j["unfinished"] == 1 and st_j["cold"] == 0 and (
            st_j["nonfinite"] == 0) and 7 in flagged and 3 not in flagged and fin and not bool(done[7]) and bool(
            done[11]) and bool(torch.isfinite(dxo[11]).all())
    T2 = T.clone()
    T2[0] = 250.0
    try:
        ad.jvp(xs, T2, P, full(B, 1.0), hs, *v)
        raised = False
    except ValueError:
        raised = True
    verdict("status: NaN input, max_steps, dt = 0, T < 300 K", ok and raised,
            f"guard counts {st_j} (jvp) / {st_v} (vjp), flagged cells {flagged}: the NaN cell counted as bad input "
            f"(not flagged), the stopped cell unfinished and flagged, every other cell's tangent and cotangents "
            f"finite, "
            f"the Arrhenius cotangent finite; the dt = 0 cell done; T < 300 K raises: {raised}")


# ---------------------------------------------------------------------------------------------------------------- (f)
def window():
    B, NC = 128, 30
    x, h, fo = gcm_states(B, 150, NC - 1)

    def run(x, h, eps=0.0, v=None, tangent=False):
        dx, dh = (v["x"], v["h"]) if tangent else (None, None)
        for k in range(NC):
            T, P, dt = fo[k]
            if tangent:
                x, h, _, _, dx, dh = ad.jvp(x, T, P, dt, h, dx, v["T"][k] * T, torch.zeros_like(T),
                                            torch.zeros_like(T), dh, dlm=v["lm"], adapt=True, lm=None)
            else:
                lm = None if v is None else eps * v["lm"]
                x, h, _, _, _, _ = ad.jvp(x, T * (1 + eps * v["T"][k]) if v is not None else T, P, dt, h,
                                          *zeros_tan(x), lm=lm)
        return x, h, dx, dh

    v = dict(x=x * rn(B, ad.ns), h=torch.zeros_like(h), T=[1e-3 * rn(B) for _ in range(NC)], lm=1e-1 * rn(B, ad.nr))
    xe, he, dxe, dhe = run(x, h, v=v, tangent=True)
    best = torch.full((B,), np.inf, device=dev)
    prev = None
    for e in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        xp_, hp_, _, _ = run(x + e * v["x"], h, e, v)
        xm_, hm_, _, _ = run(x - e * v["x"], h, -e, v)
        fd = (xp_ - xm_) / (2 * e)
        m = xe > 1e-25
        u = torch.where(m, dxe / xe.clamp(min=1e-300), torch.zeros_like(xe))
        uf = torch.where(m, fd / xe.clamp(min=1e-300), torch.zeros_like(xe))
        sc = u.abs().max(1).values.clamp(min=1e-300)
        err = (u - uf).abs().max(1).values / sc
        if prev is not None:
            agree = (uf - prev).abs().max(1).values / sc
            best = torch.where(agree < 1e-2, torch.minimum(best, err), best)
        prev = uf
    fin = torch.isfinite(best)
    b = best[fin].cpu().numpy()
    verdict(f"{NC}-call window, carried tangents (adaptive)", fin.sum() > 0 and np.quantile(b, 0.9) <= 1e-4,
            f"{B} paths, direction on x0, T of every call and lm: error of d ln x / largest |d ln x| at the best step "
            f"{pct(b)} ({int(fin.sum())} cells with a plateau)")


def window_reverse():
    """Reverse mode through a window: state and step-size checkpoints at the call boundaries, each call's vjp with the
    cotangents (x, h) carried backwards, against the tangents carried forwards: dot products per path for every
    per-path input, and the batch totals for the Arrhenius table (a global input)."""
    B, NC = 256, 30
    x, h, fo = gcm_states(B, 150, NC - 1)
    far = away_from_balance(x, *fo[0][:2])
    zb = torch.zeros_like(h)
    for adapt in (False, True):
        v = dict(x=x * rn(B, ad.ns), h=h * rn(B), T=[1e-2 * rn(B) * q[0] for q in fo],
                 P=[1e-2 * rn(B) * q[1] for q in fo], dt=[1e-2 * rn(B) * q[2] for q in fo],
                 lm=1e-2 * rn(B, ad.nr) * far, rc=1e-3 * rn(ad.nr, 9))

        def forward(table):
            """Checkpoints and the tangent carried to the end: every per-path input, or the table alone."""
            xs, hs = [x], [h]
            dx, dh = (torch.zeros_like(x), zb) if table else (v["x"], v["h"])
            for k in range(NC):
                tan = (zb, zb, zb) if table else (v["T"][k], v["P"][k], v["dt"][k])
                xo, ho, _, _, dx, dh = ad.jvp(xs[-1], *fo[k], hs[-1], dx, *tan, dh, dlm=None if table else v["lm"],
                                              drc=v["rc"] if table else None, adapt=adapt)
                xs.append(xo)
                hs.append(ho)
            return xs, hs, dx, dh

        xs, hs, dx, dh = forward(False)
        _, _, dxr, dhr = forward(True)
        cx, ch = rn(B, ad.ns), rn(B) / hs[-1]
        gx, gh, terms = cx, ch, []
        glm, grc = torch.zeros(B, ad.nr, dtype=torch.float64, device=dev), 0.0
        for k in reversed(range(NC)):
            _, _, _, _, g = ad.vjp(xs[k], *fo[k], hs[k], gx, gh, adapt=adapt)
            terms += [g["T"] * v["T"][k], g["P"] * v["P"][k], g["dt"] * v["dt"][k]]
            glm, grc = glm + g["lm"], grc + g["rc"]
            gx, gh = g["x"], g["h"]
        terms += [(gx * v["x"]).sum(1), gh * v["h"], (glm * v["lm"]).sum(1)]
        lhs, rhs = (cx * dx).sum(1) + ch * dh, sum(terms)
        sc = torch.maximum((cx * dx).abs().sum(1) + (ch * dh).abs(), sum(q.abs() for q in terms))
        rel = ((lhs - rhs).abs() / sc).cpu().numpy()
        tl, tr = float(((cx * dxr).sum(1) + ch * dhr).sum()), float((grc * v["rc"]).sum())
        trel = abs(tl - tr) / float((cx * dxr).abs().sum() + (ch * dhr).abs().sum())
        verdict(f"{NC}-call window in reverse mode, {'adaptive' if adapt else 'frozen'}",
                np.quantile(rel, 0.99) <= 1e-6 and trel <= 1e-6,
                f"{B} paths; x0, h0, T, P, dt of every call, lm away from balance: per path |<c, J v> - <J^T c, v>| "
                f"/ scale {pct(rel)}, p99 {np.quantile(rel, 0.99):.1e}; the Arrhenius table (batch totals) {tl:.10e} "
                f"vs {tr:.10e} ({trel:.1e} of the sum of |terms|)")


def main():
    t0 = time.time()
    print(f"kernels built; jvp {ad.wpb['jvp']} warps per block, vjp {ad.wpb['vjp']}", flush=True)
    for f in (parity, vs_reference, finite_differences, dot_tests, determinism, directions, status, window,
              window_reverse):
        f()
        print(f"  ({time.time() - t0:.0f} s)", flush=True)
    print(f"{sum(results)} of {len(results)} cases PASS", flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
