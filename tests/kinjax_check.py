"""KinJax (lib/kinjax.py: the JAX primitives over the call, tangent and adjoint kernels) against CudaSolver,
kincuda_ad.ADSolver and central differences, on one GPU; PASS / FAIL per case (INFO lines are not judged).
(a) Primal: KinJax without lm bitwise CudaSolver.launch (x, h, done, steps, flag); with lm = 0 bitwise the same; the
    counts equal launch's. Inputs of tests/kincuda_ad_check.py (a): yardstick cold calls dt 1 / 500 s, scrambled sigma
    0 / 1 / 2 dex x dt 0.01 / 1 / 50 / 500 s (4,096 cells each), GCM-like states (panel P after 150 calls, 1,024 cells);
    and a recovered cell (max_steps 10 for a cold start at dt 500 s among settled cells: flag and counts of the
    device-side recovery bitwise launch's).
(b) jit(jax.jvp) and jax.linearize bitwise ADSolver.jvp (outputs and tangents of x, h; every input carries a tangent,
    lm random), both contracts, same cases; a tangent on x alone (the others symbolic zeros) bitwise ADSolver with zero
    tangents and no lm / rc tangent.
(c) jax.vjp and jax.grad bitwise ADSolver.vjp (cotangents of x, T, P, dt, h, lm, rc), both contracts, same cases; grad
    with respect to x alone (no Arrhenius reduction launched) bitwise.
(d) Dot products <c, J v> = <J^T c, v> per cell through JAX (jvp and vjp), log-weighted cotangents, both contracts:
    p99 <= 1e-6 (lm on reactions away from balance, as kincuda_ad_check (d)).
(e) Adaptive contract against central differences of the JAX primal (plateau rule of kincuda_ad_check (c)), per input
    class, GCM-like states (the cold-start case and its starting-step-size limit are kincuda_ad_check's).
(f) Transforms: vmap over cells (a per-cell function; one merged call; jvp and grad bitwise the flat ADSolver call, the
    per-cell Arrhenius cotangent bitwise a one-cell ADSolver call); vmap over tangent directions (one kinjax_lin with a
    D axis in the jaxpr; D = 3 and D = 20 > 16 bitwise ADSolver); vmap over cotangents (one kinjax_adj with D); jacfwd /
    jacrev of h_out w.r.t. T (diagonal bitwise ADSolver, off-diagonal exactly 0).
(g) A jitted 30-call lax.scan with jax.checkpoint per call (panel P, 256 paths): jvp bitwise ADSolver's carried
    tangents; grad bitwise (x0, h0, T, P, dt of every call) / to rounding (lm, rc: summed over calls) ADSolver's reverse
    window; dot products forward vs reverse; checkpointed = not checkpointed; no host sync in the jitted grad.
(h) End to end: jax.grad of a loss on log mixing ratios after 10 calls with respect to per-reaction log multipliers
    (one per reaction, shared by every cell) against central differences of the loss.
(i) Status: a NaN input, a cell stopped by max_steps, a dt = 0 cell among ordinary ones: the primal's flag and counts
    equal ADSolver's, JAX derivatives bitwise ADSolver's; a cell below 300 K counted cold, other cells' derivatives
    unchanged.
(k) Forward mode in one tangent call: the jaxpr of jit(jvp) holds one kinjax_lin and no kinjax_call, and jvp's primal
    outputs (x, h, done, steps, flag, status) are bitwise the call's; linearize's primal part and grad hold one
    kinjax_call and no kinjax_lin (partial evaluation splits it off); paired timing (GPU may be shared, INFO) of jit(jvp)
    against ADSolver.jvp and the call, 4,096 GCM-like cells.
(j) Edge cases: no lm with an Arrhenius tangent / cotangent; no h_out cotangent; grad w.r.t. x, T, P, dt, h launches no
    Arrhenius reduction; vmap of vmap of grad (per-cell Arrhenius cotangents); a mapped Arrhenius table, a mapped
    Arrhenius tangent over mapped cells and reverse mode with max_steps above CAP^2 raise; empty batches.
    taskset -c <cpus> $PY tests/kinjax_check.py   (XLA_PYTHON_CLIENT_PREALLOCATE=false, CUDA_VISIBLE_DEVICES set; about
    7 min on a shared GPU, three kernel compiles included)"""
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import kinbatch as KB  # noqa: E402
import kincuda as KC  # noqa: E402
import kincuda_ad as KA  # noqa: E402
import kinjax as KJ  # noqa: E402
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


GASES = ["H2O", "CO", "CO2", "CH4", "NH3", "HCN", "C2H2", "H", "OH", "O", "H2", "N2"]
FLOORS = [1e-10] * 7 + [1e-8] * 3 + [1e-6] * 2
CLS = ("x", "T", "P", "dt", "h", "lm", "rc")
results = []
t_start = time.time()
net, sp = network()
kj = KJ.KinJax(net)
ad = KA.ADSolver(net, dev)
kc = KC.CudaSolver(net, dev)
NR, NS = ad.nr, ad.ns
RC0 = kj.rc
gen = torch.Generator(device=dev).manual_seed(11)
rn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64, device=dev)
J = lambda a: None if a is None else jnp.asarray(a.detach().cpu().numpy())
RATES, MECH = KV.TorchRates(net, dev), KB.Mech(net, dev)


def verdict(name, ok, text):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {text}", flush=True)


def info(name, text):
    print(f"INFO  {name}: {text}", flush=True)


def npy(a):
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


def eq(a, b):
    """Bitwise equality (NaN payloads included) of two arrays of the same dtype."""
    a, b = npy(a), npy(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def pct(a):
    a = np.asarray(a, float)
    return f"median {np.median(a):.1e} p90 {np.quantile(a, 0.9):.1e} max {a.max():.1e}" if a.size else "none"


def eqns(j, acc=None):
    """Every equation of a jaxpr, sub-jaxprs included."""
    acc = [] if acc is None else acc
    for e in getattr(j, "jaxpr", j).eqns:
        acc.append(e)
        for p in e.params.values():
            for q in (p if isinstance(p, (tuple, list)) else (p,)):
                if hasattr(q, "eqns") or hasattr(q, "jaxpr"):
                    eqns(q, acc)
    return acc


def prims(j):
    """(name, operand shapes) of every equation of a jaxpr, sub-jaxprs included."""
    return [(e.primitive.name, [tuple(v.aval.shape) for v in e.invars]) for e in eqns(j)]


def fn(adapt):
    """(x, T, P, dt, h, lm, rc) -> (x_out, h_out) through KinJax."""
    return lambda x, T, P, dt, h, lm, rc: kj.call(x, T, P, dt, h, lm, rc, adapt=adapt)[:2]


def dot_loss(f, cx, ch):
    """p -> <cx, x_out> + <ch, h_out> for one call of f."""
    def loss(*p):
        xo, ho = f(*p)
        return jnp.sum(cx * xo) + jnp.sum(ch * ho)
    return loss


def gcm_states(B, ncalls=150, extra=0):
    """Panel-P states after ncalls calls of the recorded forcing (kincuda_ad_check) and the forcing of the next 1 + extra calls."""
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


def cases():
    B = 4096
    out = []
    x0, T, P = yardstick(B, 0, 0.0)
    out += [(f"cold dt {dt:g}", (x0, T, P, full(B, dt), full(B, 1e-10))) for dt in (1.0, 500.0)]
    for sigma in (0.0, 1.0, 2.0):
        x0, T, P = yardstick(B, int(10 + sigma), sigma)
        out += [(f"scrambled {sigma:g} dex dt {dt:g}", (x0, T, P, full(B, dt), full(B, 1e-10)))
                for dt in (0.01, 1.0, 50.0, 500.0)]
    x, h, fo = gcm_states(1024)
    out.append(("GCM-like", (x, *fo[0], h)))
    return out


def tangents(a):
    x, T, P, dt, h = a
    B = x.shape[0]
    return (x * rn(B, NS), T * 1e-2 * rn(B), P * 1e-2 * rn(B), dt * 1e-2 * rn(B), h * 1e-2 * rn(B),
            rn(B, NR) * 1e-2, rn(NR, 9) * 1e-3)


def away_from_balance(x, T, P, q=1e-3):
    """[B, R]: reactions whose forward and reverse fluxes differ by more than q of their sum (kincuda_ad_check)."""
    kf, kr, nd = RATES(T, P)
    c3 = torch.where(MECH.tb[None], nd[:, None], torch.ones_like(kf))
    (ri, _), (pi, _) = MECH.idx
    ne = torch.cat([x * nd[:, None], torch.ones_like(nd)[:, None]], 1)
    a, b = c3 * kf * ne[:, ri].prod(-1), c3 * kr * ne[:, pi].prod(-1)
    return (a - b).abs() > q * (a + b)


# ---------------------------------------------------------------------------------------------------------------- (a)
def primal(cs):
    bad, bad_lm, bad_cnt = [], [], []
    for name, a in cs:
        B = a[0].shape[0]
        ref = kc.launch(*a)
        cnt = kc.count.tolist()[1:9]
        ja = [J(q) for q in a]
        out = kj(*ja)
        out_lm = kj(*ja, jnp.zeros((B, NR)))
        r = [ref[0], ref[1], ref[2].int(), ref[3].int(), ref[4].int()]
        if not all(eq(p, q) for p, q in zip(r, out[:5])):
            bad.append(name)
        if not all(eq(p, q) for p, q in zip(out, out_lm)):
            bad_lm.append(name)
        if np.asarray(out[5]).tolist() != cnt:
            bad_cnt.append(name)
    verdict("primal parity", not (bad or bad_lm or bad_cnt),
            f"{len(cs)} cases: KinJax bitwise CudaSolver.launch (x, h, done, steps, flag) except {bad or 'none'}; with "
            f"lm = 0 (the tangent kernel, no directions) bitwise the same except {bad_lm or 'none'}; guard counts "
            f"equal except {bad_cnt or 'none'}")
    B = 64
    x0, T, P = yardstick(B, 1, 0.0)
    xs, hs, _, _ = kc(x0, T, P, full(B, 100.0), full(B, 1e-10))          # settled states
    x, h, dt = xs.clone(), hs.clone(), full(B, 1.0)
    x[7], h[7], dt[7] = x0[7], 1e-10, 500.0                             # cell 7: cold start at 500 s
    ref = kc.launch(x, T, P, dt, h, max_steps=10)
    cnt = kc.count.tolist()[1:9]
    out = jax.jit(lambda *p: kj.call(*p, max_steps=10))(*(J(q) for q in (x, T, P, dt, h)))
    r = [ref[0], ref[1], ref[2].int(), ref[3].int(), ref[4].int()]
    ok = all(eq(p, q) for p, q in zip(r, out[:5])) and np.asarray(out[5]).tolist() == cnt and int(out[4][7]) > 0
    verdict("primal parity with a recovered cell", ok,
            f"max_steps 10, cell 7 a cold start at dt 500 s among settled cells: KinJax bitwise CudaSolver.launch "
            f"(x, h, done, steps, flag) and its counts {dict(zip(KC.COUNTS, cnt))}: {ok}; flag of cell 7 "
            f"{int(out[4][7])}")


# ------------------------------------------------------------------------------------------------------------ (b, c)
def parity(cs):
    bad_j, bad_l, bad_v, bad_g, bad_x = [], [], [], [], []
    jvps = {ad_: jax.jit(lambda p, t, f=fn(ad_): jax.jvp(f, p, t)) for ad_ in (True, False)}
    for ci, (name, a) in enumerate(cs):
        B = a[0].shape[0]
        lm = rn(B, NR) * 1e-2
        v = tangents(a)
        cx, ch = rn(B, NS), rn(B)
        P_ = tuple(J(q) for q in a) + (J(lm), RC0)
        V_ = tuple(J(q) for q in v)
        for adapt in (True, False):
            tag = f"{name}, {'A' if adapt else 'F'}"
            r = ad.jvp(*a, *v[:5], lm=lm, dlm=v[5], drc=v[6], adapt=adapt)
            (xo, ho), (dx, dh) = jvps[adapt](P_, V_)
            if not (eq(r[0], xo) and eq(r[1], ho) and eq(r[4], dx) and eq(r[5], dh)):
                bad_j.append(tag)
            g = ad.vjp(*a, cx, ch, lm=lm, adapt=adapt)[4]
            _, pull = jax.vjp(fn(adapt), *P_)
            bars = pull((J(cx), J(ch)))
            if not all(eq(g[k], b) for k, b in zip(CLS, bars)):
                bad_v.append(tag + " " + ",".join(k for k, b in zip(CLS, bars) if not eq(g[k], b)))
            if ci % 5 == 0:                                   # linearize, grad, x alone on every fifth case
                _, lin = jax.linearize(fn(adapt), *P_)
                dx2, dh2 = lin(*V_)
                if not (eq(r[4], dx2) and eq(r[5], dh2)):
                    bad_l.append(tag)
                loss = dot_loss(fn(adapt), J(cx), J(ch))
                gr = jax.jit(jax.grad(loss, argnums=tuple(range(7))))(*P_)
                if not all(eq(g[k], b) for k, b in zip(CLS, gr)):
                    bad_g.append(tag)
                zb = torch.zeros_like(a[1])
                r0 = ad.jvp(*a, v[0], zb, zb, zb, zb, lm=lm, adapt=adapt)
                _, (dx0, dh0) = jax.jvp(lambda x: fn(adapt)(x, *P_[1:]), (P_[0],), (V_[0],))
                gx_only = jax.grad(lambda x: loss(x, *P_[1:]))(P_[0])
                if not (eq(r0[4], dx0) and eq(r0[5], dh0) and eq(g["x"], gx_only)):
                    bad_x.append(tag)
    verdict("jit(jvp) bitwise ADSolver.jvp", not bad_j,
            f"{len(cs)} cases x 2 contracts, tangents on x, T, P, dt, h, lm, rc (lm random): outputs and tangents of x, "
            f"h bitwise except {bad_j or 'none'}")
    verdict("vjp bitwise ADSolver.vjp", not bad_v,
            f"{len(cs)} cases x 2 contracts: cotangents of x, T, P, dt, h, lm, rc bitwise except {bad_v or 'none'}")
    verdict("linearize and grad bitwise ADSolver", not (bad_l or bad_g or bad_x),
            f"every fifth case x 2 contracts: linearize's linear map {bad_l or 'all bitwise'}; jit(grad) of "
            f"<cx, x_out> + <ch, h_out> w.r.t. all seven inputs {bad_g or 'all bitwise'}; a tangent on x alone and "
            f"grad w.r.t. x alone {bad_x or 'all bitwise'}")


# ---------------------------------------------------------------------------------------------------------------- (d)
def dot_tests():
    B = 1024
    x, h, fo = gcm_states(B)
    cs = [("GCM-like", (x, *fo[0], h))]
    x0, T, P = yardstick(B, 0, 0.0)
    cs.append(("cold dt 500", (x0, T, P, full(B, 500.0), full(B, 1e-10))))
    x0, T, P = yardstick(B, 11, 1.0)
    cs.append(("scrambled 1 dex dt 50", (x0, T, P, full(B, 50.0), full(B, 1e-10))))
    for name, a in cs:
        far = away_from_balance(a[0], a[1], a[2])
        P_ = tuple(J(q) for q in a) + (jnp.zeros((B, NR)), RC0)
        for adapt in (False, True):
            f = fn(adapt)
            (xo, ho), pull = jax.vjp(f, *P_)
            xo_t, ho_t = tt(np.asarray(xo)), tt(np.asarray(ho))
            cx, ch = rn(B, NS) / xo_t.clamp(min=1e-30) * (xo_t > 1e-25), rn(B) / ho_t
            g = [tt(np.asarray(q)) for q in pull((J(cx), J(ch)))]
            x_, T_, P__, dt_, h_ = a
            v = [x_ * rn(B, NS), T_ * 1e-2 * rn(B), P__ * 1e-2 * rn(B), dt_ * 1e-2 * rn(B), h_ * 1e-2 * rn(B),
                 rn(B, NR) * 1e-2 * far, torch.zeros(NR, 9, dtype=torch.float64, device=dev)]
            _, (dxo, dho) = jax.jvp(f, P_, tuple(J(q) for q in v))
            dxo, dho = tt(np.asarray(dxo)), tt(np.asarray(dho))
            lhs = (cx * dxo).sum(1) + ch * dho
            terms = [(g[k] * v[k]).reshape(B, -1) for k in range(6)]
            rhs = sum(q.sum(1) for q in terms)
            sc = torch.maximum((cx * dxo).abs().sum(1) + (ch * dho).abs(), sum(q.abs().sum(1) for q in terms))
            rel = ((lhs - rhs).abs() / sc).nan_to_num(0.0).cpu().numpy()
            verdict(f"dot products, {name}, {'adaptive' if adapt else 'frozen'}", np.quantile(rel, 0.99) <= 1e-6,
                    f"{B} cells, x, T, P, dt, h and lm away from balance: |<c,Jv> - <J^T c,v>| / scale {pct(rel)}, "
                    f"p99 {np.quantile(rel, 0.99):.1e}")
            drc = rn(NR, 9) * 1e-3                               # the Arrhenius table: the batch total
            z = [jnp.zeros_like(q) for q in P_[:6]]
            _, (dxr, dhr) = jax.jvp(f, P_, tuple(z) + (J(drc),))
            l_ = float(jnp.sum(J(cx) * dxr) + jnp.sum(J(ch) * dhr))
            r_ = float((g[6] * drc).sum())
            s_ = float(jnp.sum(jnp.abs(J(cx) * dxr)) + jnp.sum(jnp.abs(J(ch) * dhr)))
            info(f"dot products, {name}, Arrhenius table, {'adaptive' if adapt else 'frozen'}",
                 f"batch total {l_:.10e} vs {r_:.10e} (difference / sum |terms| {abs(l_ - r_) / s_:.1e})")


# ---------------------------------------------------------------------------------------------------------------- (e)
def fd_case(name, a, f_all):
    """kincuda_ad_check (c) through JAX: one random direction per input class; error of d ln x at the best plateau step."""
    x, T, P, dt, h = a
    B = x.shape[0]
    P_ = tuple(J(q) for q in a) + (jnp.zeros((B, NR)), RC0)
    f = jax.jit(lambda *p: kj.call(*p)[:4])
    jv = jax.jit(lambda p, t: jax.jvp(fn(True), p, t))
    errs, hs = {}, {}
    for ci, cls in enumerate(CLS):
        v = [jnp.zeros_like(q) for q in P_]
        if cls == "x":
            v[0] = J(x * rn(B, NS))
        elif cls in ("T", "P", "dt", "h"):
            i = CLS.index(cls)
            w = a[i] * rn(B)
            if cls == "T":
                w = torch.where(T > 301.0, w, 0.0)
            v[i] = J(w)
        elif cls == "lm":
            v[5] = J(rn(B, NR))
        else:
            v[6] = J(rn(NR, 9) * torch.tensor([1.0, 0.1, 100.0] * 3, dtype=torch.float64, device=dev))
        (xo, ho), (dxo, dho) = jv(P_, tuple(v))
        nok = f(*P_)[3]
        fx, fh, ok = [], [], []
        for e in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
            o = [f(*(p + sg * e * q for p, q in zip(P_, v))) for sg in (1.0, -1.0)]
            fx.append((o[0][0] - o[1][0]) / (2 * e))
            fh.append((o[0][1] - o[1][1]) / (2 * e))
            ok.append((o[0][3] == nok) & (o[1][3] == nok))
        xo, dxo, ho, dho = (np.asarray(q) for q in (xo, dxo, ho, dho))
        m = xo > 1e-25
        xc = np.clip(xo, 1e-300, None)
        u = np.where(m, dxo / xc, 0.0)
        sc = np.abs(u).max(1)
        best_x, best_h = np.full(B, np.inf), np.full(B, np.inf)
        for k in range(len(fx) - 1):
            uf, un = (np.where(m, np.asarray(q) / xc, 0.0) for q in (fx[k], fx[k + 1]))
            stab = np.asarray(ok[k]) & np.asarray(ok[k + 1])
            agree = np.abs(uf - un).max(1) / np.clip(sc, 1e-300, None)
            err = np.abs(u - uf).max(1) / np.clip(sc, 1e-300, None)
            best_x = np.where(stab & (agree < 1e-2), np.minimum(best_x, err), best_x)
            hd = np.clip(np.abs(dho), 1e-300, None)
            hr = np.abs(dho - np.asarray(fh[k])) / hd
            agh = np.abs(np.asarray(fh[k]) - np.asarray(fh[k + 1])) / hd
            best_h = np.where(stab & (agh < 1e-2), np.minimum(best_h, hr), best_h)
        sens, sens_h = sc > 1e-6, np.abs(dho) > 1e-6 * ho
        fin = np.isfinite(best_x) & sens
        errs[cls] = (best_x[fin], int(fin.sum()), int(sens.sum()))
        fh_ = np.isfinite(best_h) & sens_h
        hs[cls] = (best_h[fh_], int(fh_.sum()), int(sens_h.sum()))
    ok = all(np.quantile(q[0], 0.9) <= (1e-4 if k == "h" else 1e-5) for k, q in errs.items() if len(q[0]))
    verdict(f"adaptive vs central differences (JAX primal), {name}", ok,
            f"{B} cells; error of d ln x / the cell's largest |d ln x| at the best step (cells with a plateau / cells "
            f"sensitive): " + "; ".join(f"{k} {pct(q[0])} ({q[1]}/{q[2]})" for k, q in errs.items() if len(q[0]))
            + " | h_out rel: " + "; ".join(f"{k} {pct(q[0])} ({q[1]}/{q[2]})" for k, q in hs.items() if len(q[0])))


def finite_differences():
    x, h, fo = gcm_states(256)
    fd_case("GCM-like (panel P, call 151), 256 cells", (x, *fo[0], h), None)


# ---------------------------------------------------------------------------------------------------------------- (f)
def transforms():
    B = 512
    x, h, fo = gcm_states(B)
    a = (x, *fo[0], h)
    lm = rn(B, NR) * 1e-2
    P_ = tuple(J(q) for q in a) + (J(lm), RC0)
    v = tangents(a)
    cx, ch = rn(B, NS), rn(B)
    for adapt in (True, False):
        lab = "A" if adapt else "F"
        f = fn(adapt)
        r = ad.jvp(*a, *v[:5], lm=lm, dlm=v[5], drc=v[6], adapt=adapt)
        g = ad.vjp(*a, cx, ch, lm=lm, adapt=adapt)[4]
        # vmap over cells: a per-cell function
        cell = lambda x, T, P, dt, h, lm, rc: kj.call(x, T, P, dt, h, lm, rc, adapt=adapt)[:2]
        vj = jax.jit(jax.vmap(lambda p, t: jax.jvp(cell, p, t), in_axes=((0,) * 6 + (None,), (0,) * 6 + (None,))))
        args = (P_, tuple(J(q) for q in v))
        (xo, ho), (dx, dh) = vj(*args)
        pj = prims(jax.make_jaxpr(vj)(*args))
        lins = [s for n, s in pj if n == "kinjax_lin"]
        ok1 = eq(r[0], xo) and eq(r[1], ho) and eq(r[4], dx) and eq(r[5], dh) and len(lins) == 1 and lins[0][0] == (
            B, NS)
        lc = lambda x, T, P, dt, h, lm, rc, cx, ch: dot_loss(cell, cx, ch)(x, T, P, dt, h, lm, rc)
        vg = jax.jit(jax.vmap(jax.grad(lc, argnums=tuple(range(7))), in_axes=(0,) * 6 + (None, 0, 0)))
        gr = vg(*P_, J(cx), J(ch))
        ok2 = all(eq(g[k], b) for k, b in zip(CLS[:6], gr[:6])) and gr[6].shape == (B, NR, 9)
        alone = True
        for i in range(0, B, 73):                             # per-cell Arrhenius cotangent = a one-cell call's
            s = slice(i, i + 1)
            gi = ad.vjp(*(q[s] for q in a), cx[s], ch[s], lm=lm[s], adapt=adapt)[4]
            alone = alone and eq(gi["rc"], gr[6][i])
        tot = float(np.abs(np.asarray(gr[6]).sum(0) - npy(g["rc"])).max() / np.abs(npy(g["rc"])).max())
        verdict(f"vmap over cells ({lab})", ok1 and ok2 and alone,
                f"{B} cells as a per-cell function: one kinjax_lin on [{B}, {NS}] in the jaxpr {len(lins) == 1}; "
                f"vmap(jvp) bitwise the flat ADSolver.jvp {ok1}; vmap(grad) x, T, P, dt, h, lm bitwise ADSolver.vjp "
                f"{ok2}; the per-cell Arrhenius cotangent bitwise a one-cell ADSolver call (every 73rd cell) {alone}; "
                f"its sum over cells vs the flat call's {tot:.1e} (relative, summation order)")
        # vmap over tangent directions
        ok3, txt = True, []
        for D in (3, 20):
            vs = [tangents(a) for _ in range(D)]
            st = [torch.stack([q[k] for q in vs]) for k in range(7)]
            vd = jax.jit(jax.vmap(lambda *t: jax.jvp(f, P_, t)[1]))
            dxs, dhs = vd(*(J(q) for q in st))
            lins = [s for n, s in prims(jax.make_jaxpr(vd)(*(J(q) for q in st))) if n == "kinjax_lin"]
            ref = [ad.jvp(*a, *(q[c:c + 16] for q in st[:5]), lm=lm, dlm=st[5][c:c + 16], drc=st[6][c:c + 16],
                          adapt=adapt) for c in range(0, D, 16)]
            rdx, rdh = torch.cat([q[4] for q in ref]), torch.cat([q[5] for q in ref])
            o = eq(rdx, dxs) and eq(rdh, dhs) and len(lins) == 1 and lins[0][NP_IDX] == (D, B, NS)
            ok3 = ok3 and o
            txt.append(f"D = {D}: one kinjax_lin with tangents [{D}, {B}, {NS}] {len(lins) == 1}, bitwise {o}")
        verdict(f"vmap over tangent directions ({lab})", ok3, "; ".join(txt) + " (ADSolver: 16 per call)")
        # vmap over cotangents
        V = 5
        cxs, chs = rn(V, B, NS), rn(V, B)
        _, pull = jax.vjp(f, *P_)
        vp = jax.jit(jax.vmap(pull))
        bars = vp((J(cxs), J(chs)))
        adjs = [s for n, s in prims(jax.make_jaxpr(vp)((J(cxs), J(chs)))) if n == "kinjax_adj"]
        ok4 = len(adjs) == 1 and adjs[0][NP_IDX] == (V, B, NS)
        for d in range(V):
            gd = ad.vjp(*a, cxs[d], chs[d], lm=lm, adapt=adapt)[4]
            ok4 = ok4 and all(eq(gd[k], b[d]) for k, b in zip(CLS, bars))
        verdict(f"vmap over cotangents ({lab})", ok4,
                f"{V} cotangents: one kinjax_adj with cotangents [{V}, {B}, {NS}] {len(adjs) == 1}; every cotangent "
                f"of x, T, P, dt, h, lm, rc bitwise ADSolver.vjp {ok4}")
        # jacfwd / jacrev of h_out w.r.t. T on 8 cells
        s = slice(0, 8)
        p8 = tuple(q[s] for q in P_[:6]) + (RC0,)
        hT = lambda T: f(p8[0], T, *p8[2:])[1]
        jf, jr = jax.jit(jax.jacfwd(hT))(p8[1]), jax.jit(jax.jacrev(hT))(p8[1])
        z8 = torch.zeros(8, dtype=torch.float64, device=dev)
        rj = ad.jvp(*(q[s] for q in a), torch.zeros_like(a[0][s]), torch.ones(8, dtype=torch.float64, device=dev),
                    z8, z8, z8, lm=lm[s], adapt=adapt)[5]
        rv = ad.vjp(*(q[s] for q in a), torch.zeros_like(a[0][s]), torch.ones(8, dtype=torch.float64, device=dev),
                    lm=lm[s], adapt=adapt)[4]["T"]
        off = ~np.eye(8, dtype=bool)
        ok5 = eq(rj, np.diag(np.asarray(jf)).copy()) and eq(rv, np.diag(np.asarray(jr)).copy()) and not np.asarray(
            jf)[off].any() and not np.asarray(jr)[off].any()
        verdict(f"jacfwd / jacrev ({lab})", ok5,
                f"d h_out / d T on 8 cells: diagonals bitwise ADSolver (jvp dT = 1, vjp gh = 1), off-diagonal "
                f"entries exactly 0: {ok5}")


NP_IDX = KJ.NP     # operand index of the first tangent / cotangent in a kinjax_lin / kinjax_adj equation


# ---------------------------------------------------------------------------------------------------------------- (g)
def window(x0, h0, Ts, Ps, dts, lm, rc, adapt, ckpt=True):
    """len(Ts) calls as one lax.scan, each call under jax.checkpoint (or not)."""
    def step(c, fo):
        xo, ho = kj.call(c[0], *fo, c[1], lm, rc, adapt=adapt)[:2]
        return (xo, ho), None
    (x, h), _ = jax.lax.scan(jax.checkpoint(step) if ckpt else step, (x0, h0), (Ts, Ps, dts))
    return x, h


def scan_window():
    B, NC = 256, 30
    x, h, fo = gcm_states(B, 150, NC - 1)
    far = away_from_balance(x, *fo[0][:2])
    lm = rn(B, NR) * 1e-2
    Ts, Ps, dts = (torch.stack([q[i] for q in fo]) for i in range(3))
    P_ = (J(x), J(h), J(Ts), J(Ps), J(dts), J(lm), RC0)
    for adapt in (False, True):
        lab = "adaptive" if adapt else "frozen"
        v = dict(x=x * rn(B, NS), h=h * rn(B), T=1e-2 * rn(NC, B) * Ts, P=1e-2 * rn(NC, B) * Ps,
                 dt=1e-2 * rn(NC, B) * dts, lm=1e-2 * rn(B, NR) * far, rc=1e-3 * rn(NR, 9))
        win = lambda *p, ck=True: window(*p, adapt=adapt, ckpt=ck)
        V_ = tuple(J(v[k]) for k in ("x", "h", "T", "P", "dt", "lm", "rc"))
        (xe, he), (dxe, dhe) = jax.jit(lambda p, t: jax.jvp(win, p, t))(P_, V_)
        xs, hs = [x], [h]                                     # ADSolver: carried tangents
        dx, dh = v["x"], v["h"]
        for k in range(NC):
            xo, ho, _, _, dx, dh = ad.jvp(xs[-1], *fo[k], hs[-1], dx, v["T"][k], v["P"][k], v["dt"][k], dh, lm=lm,
                                          dlm=v["lm"], drc=v["rc"], adapt=adapt)
            xs.append(xo)
            hs.append(ho)
        okf = eq(xs[-1], xe) and eq(hs[-1], he) and eq(dx, dxe) and eq(dh, dhe)
        cx, ch = rn(B, NS), rn(B) / hs[-1]
        loss = lambda *p, ck=True: dot_loss(lambda *q: win(*q, ck=ck), J(cx), J(ch))(*p)
        gfun = jax.jit(jax.grad(loss, argnums=tuple(range(7))))
        gj = gfun(*P_)
        gn = jax.jit(jax.grad(lambda *p: loss(*p, ck=False), argnums=tuple(range(7))))(*P_)
        same_ck = all(eq(p, q) for p, q in zip(gj, gn))
        gx, gh, gT, gP, gdt = cx, ch, [None] * NC, [None] * NC, [None] * NC   # ADSolver: reverse window
        glm, grc = torch.zeros(B, NR, dtype=torch.float64, device=dev), torch.zeros(NR, 9, dtype=torch.float64,
                                                                                    device=dev)
        for k in reversed(range(NC)):
            g = ad.vjp(xs[k], *fo[k], hs[k], gx, gh, lm=lm, adapt=adapt)[4]
            gT[k], gP[k], gdt[k] = g["T"], g["P"], g["dt"]
            glm, grc = glm + g["lm"], grc + g["rc"]
            gx, gh = g["x"], g["h"]
        okr = all(eq(p, q) for p, q in ((gx, gj[0]), (gh, gj[1]), (torch.stack(gT), gj[2]), (torch.stack(gP), gj[3]),
                                        (torch.stack(gdt), gj[4])))
        rel = lambda p, q: float(np.abs(npy(p) - np.asarray(q)).max() / max(np.abs(npy(p)).max(), 1e-300))
        r_lm, r_rc = rel(glm, gj[5]), rel(grc, gj[6])
        # dot products: per path for the per-path inputs; the Arrhenius table (global) by batch totals
        gt_ = [tt(np.asarray(q)) for q in gj]
        terms = [(gt_[0] * v["x"]).sum(1), gt_[1] * v["h"], (gt_[2] * v["T"]).sum(0), (gt_[3] * v["P"]).sum(0),
                 (gt_[4] * v["dt"]).sum(0), (gt_[5] * v["lm"]).sum(1)]
        jv = jax.jit(lambda p, t: jax.jvp(win, p, t))
        _, (dx0, dh0) = jv(P_, V_[:6] + (jnp.zeros_like(V_[6]),))
        _, (dxr, dhr) = jv(P_, tuple(jnp.zeros_like(q) for q in V_[:6]) + (V_[6],))
        cdx, cdh = cx * tt(np.asarray(dx0)), ch * tt(np.asarray(dh0))
        sc = torch.maximum(cdx.abs().sum(1) + cdh.abs(), sum(q.abs() for q in terms))
        d_rel = ((cdx.sum(1) + cdh - sum(terms)).abs() / sc).cpu().numpy()
        tl = float(jnp.sum(J(cx) * dxr) + jnp.sum(J(ch) * dhr))
        tr = float((gt_[6] * v["rc"]).sum())
        trel = abs(tl - tr) / float(jnp.sum(jnp.abs(J(cx) * dxr)) + jnp.sum(jnp.abs(J(ch) * dhr)))
        verdict(f"{NC}-call scan with jax.checkpoint, {lab}",
                okf and okr and same_ck and r_lm <= 1e-12 and r_rc <= 1e-12 and np.quantile(d_rel, 0.99) <= 1e-6
                and trel <= 1e-6,
                f"{B} paths: jit(jvp) bitwise ADSolver's carried tangents {okf}; jit(grad) of x0, h0, T, P, dt of every "
                f"call bitwise ADSolver's reverse window {okr}, lm / rc (summed over calls) relative {r_lm:.1e} / "
                f"{r_rc:.1e}; checkpointed bitwise not checkpointed {same_ck}; per-path dot products forward vs "
                f"reverse {pct(d_rel)} p99 {np.quantile(d_rel, 0.99):.1e}; Arrhenius batch totals {tl:.10e} vs "
                f"{tr:.10e} ({trel:.1e})")
        if not adapt:
            continue
        pj = prims(jax.make_jaxpr(lambda *p: jax.grad(loss, argnums=tuple(range(7)))(*p))(*P_))
        info("jaxpr of grad through the checkpointed scan", ", ".join(
            f"{n} x{sum(1 for q, _ in pj if q == n)}" for n in ("kinjax_call", "kinjax_lin", "kinjax_adj", "remat",
                                                                "checkpoint", "scan")))
        gfun(*P_)[0].block_until_ready()                      # no host sync: returns before the GPU finishes
        t0 = time.perf_counter()
        out = gfun(*P_)
        t1 = time.perf_counter()
        jax.block_until_ready(out)
        t2 = time.perf_counter()
        verdict("no host sync in the jitted grad", (t1 - t0) < 0.25 * (t2 - t0),
                f"jit(grad) of the {NC}-call window returns after {1e3 * (t1 - t0):.1f} ms, results ready after "
                f"{1e3 * (t2 - t0):.1f} ms")


# ---------------------------------------------------------------------------------------------------------------- (h)
def end_to_end():
    B, NC = 256, 10
    x, h, fo = gcm_states(B, 150, NC - 1)
    Ts, Ps, dts = (J(torch.stack([q[i] for q in fo])) for i in range(3))
    gi = jnp.array([sp.index(g) for g in GASES])
    S = jnp.array(FLOORS)
    x0, h0 = J(x), J(h)

    def final(theta):
        lm = jnp.broadcast_to(theta, (B, NR))
        return window(x0, h0, Ts, Ps, dts, lm, RC0, adapt=True)[0]

    rng = np.random.default_rng(5)
    y = jnp.log10(jax.jit(final)(jnp.asarray(0.3 * rng.standard_normal(NR)))[:, gi] + S)
    loss = jax.jit(lambda th: jnp.mean((jnp.log10(final(th)[:, gi] + S) - y) ** 2))
    th0 = jnp.zeros(NR)
    g = np.asarray(jax.jit(jax.grad(lambda th: jnp.mean((jnp.log10(final(th)[:, gi] + S) - y) ** 2)))(th0))
    dirs = [rng.standard_normal(NR) for _ in range(5)]
    top = np.argsort(-np.abs(g))[:5]
    dirs += [np.eye(NR)[i] for i in top]
    errs, lines = [], []
    for k, u in enumerate(dirs):
        d_ad = float(g @ u)
        fd = []
        for e in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
            fd.append((float(loss(th0 + e * u)) - float(loss(th0 - e * u))) / (2 * e))
        best = min((abs(fd[i] - d_ad) / abs(d_ad) for i in range(len(fd) - 1)
                    if abs(fd[i] - fd[i + 1]) < 1e-2 * abs(d_ad)), default=np.inf)
        errs.append(best)
        lines.append(f"{'random' if k < 5 else f'reaction {top[k - 5]}'} {d_ad:.6e} rel err {best:.1e}")
    verdict("end to end: grad of a log-mixing-ratio loss after 10 calls w.r.t. per-reaction log multipliers",
            max(errs) <= 1e-4,
            f"{B} paths, loss {float(loss(th0)):.4e}, |grad| {np.linalg.norm(g):.3e}; directional derivative vs "
            f"central differences (best plateau step): " + "; ".join(lines))


# ---------------------------------------------------------------------------------------------------------------- (i)
def status():
    B = 64
    x0, T, P = yardstick(B, 31, 0.0)
    xs, hs, _, _ = kc(x0, T, P, full(B, 100.0), full(B, 1e-10))
    x, h, dt = xs.clone(), hs.clone(), full(B, 1.0)
    x[3, 5] = float("nan")
    x[7], h[7], dt[7] = x0[7], 1e-10, 500.0
    dt[11] = 0.0
    lm = rn(B, NR) * 1e-2
    v = tangents((xs, T, P, dt, h))
    cx, ch = rn(B, NS), rn(B)
    ok, txt = True, []
    for adapt in (False, True):
        f = lambda *p: kj.call(*p, adapt=adapt, max_steps=10)
        P_ = tuple(J(q) for q in (x, T, P, dt, h, lm)) + (RC0,)
        out = jax.jit(f)(*P_)
        r = ad.jvp(x, T, P, dt, h, *v[:5], lm=lm, dlm=v[5], drc=v[6], adapt=adapt, max_steps=10)
        st_j, fl_j = dict(ad.status), npy(ad.flag)
        g = ad.vjp(x, T, P, dt, h, cx, ch, lm=lm, adapt=adapt, max_steps=10)[4]
        st_v = dict(ad.status)
        g2 = lambda *p: f(*p)[:2]
        _, (dx, dh) = jax.jit(lambda p, t: jax.jvp(g2, p, t))(P_, tuple(J(q) for q in v))
        _, pull = jax.vjp(g2, *P_)
        bars = pull((J(cx), J(ch)))
        stat = dict(zip(KC.COUNTS, np.asarray(out[5]).tolist()))
        o = (stat == st_j == st_v and stat["recomputed"] == 1 and eq(fl_j, out[4]) and eq(r[4], dx)
             and eq(r[5], dh) and all(eq(g[k], b) for k, b in zip(CLS, bars)))
        ok = ok and o
        txt.append(f"{'A' if adapt else 'F'}: primal status {stat}, ADSolver {st_j}, flags equal and derivatives "
                   f"bitwise {o}")
    T2 = T.clone()
    T2[0] = 250.0
    P_ = tuple(J(q) for q in (xs, T2, P, full(B, 1.0), hs, lm)) + (RC0,)
    out = kj(*P_)
    _, (dx, dh) = jax.jvp(fn(True), P_, tuple(J(q) for q in v))
    Pw = tuple(J(q) for q in (xs, T, P, full(B, 1.0), hs, lm)) + (RC0,)
    _, (dxw, dhw) = jax.jvp(fn(True), Pw, tuple(J(q) for q in v))
    cold = int(np.asarray(out[5])[1]) == 1 and eq(np.asarray(dx)[1:], np.asarray(dxw)[1:]) and bool(
        np.isfinite(np.asarray(dx)[0]).all())
    raised = False
    try:
        KJ.check(out[5])
    except ValueError:
        raised = True
    verdict("status: NaN input, max_steps, dt = 0, T < 300 K", ok and cold and raised,
            "; ".join(txt) + f"; a cell at 250 K: counted cold, not stepped, its tangent finite, the other cells' "
            f"tangents bitwise those of a call without it {cold}, check() raises {raised}")


def recovered():
    """Cells the guard flags (settled starts at dt 1000 s, max_steps 3), redone on the recovery's path: the call with lm
    = 0 bitwise CudaSolver.launch (flags, counts), with random lm bitwise ADSolver's primal; jit(jvp), vjp and grad
    bitwise ADSolver (recovered cells' derivatives, NaN on cells still failed), both contracts."""
    B, ms = 256, 3
    x0, T, P = yardstick(B, 51, 0.0)
    xs, hs, _, _ = kc(x0, T, P, full(B, 100.0), full(B, 1e-10))
    a = (xs, T, P, full(B, 1e3), hs)
    ref = kc.launch(*a, max_steps=ms)
    cnt = kc.count.tolist()[1:9]
    lm = rn(B, NR) * 1e-2
    v = tangents(a)
    cx, ch = rn(B, NS), rn(B)
    f = jax.jit(lambda *p: kj.call(*p, max_steps=ms))
    P0 = tuple(J(q) for q in a) + (jnp.zeros((B, NR)), RC0)
    P_ = tuple(J(q) for q in a) + (J(lm), RC0)
    o0, o1 = f(*P0), f(*P_)
    r = [ref[0], ref[1], ref[2].int(), ref[3].int(), ref[4]]
    ok0 = all(eq(p, q) for p, q in zip(r, o0[:5])) and np.asarray(o0[5]).tolist() == cnt
    rj = ad.jvp(*a, *v[:5], lm=lm, dlm=v[5], drc=v[6], max_steps=ms)
    st1 = dict(ad.status)
    ok1 = all(eq(p, q) for p, q in zip((rj[0], rj[1], rj[2].int(), rj[3].int(), ad.flag), o1[:5])) and dict(
        zip(KC.COUNTS, np.asarray(o1[5]).tolist())) == st1
    nf = [int((np.asarray(o1[4]) == k).sum()) for k in (1, 2)]
    bad = []
    for adapt in (True, False):
        g2 = lambda *p: kj.call(*p, adapt=adapt, max_steps=ms)[:2]  # noqa: E731
        r = ad.jvp(*a, *v[:5], lm=lm, dlm=v[5], drc=v[6], adapt=adapt, max_steps=ms)
        _, (dx, dh) = jax.jit(lambda p, t: jax.jvp(g2, p, t))(P_, tuple(J(q) for q in v))
        g = ad.vjp(*a, cx, ch, lm=lm, adapt=adapt, max_steps=ms)[4]
        _, pull = jax.vjp(g2, *P_)
        bars = pull((J(cx), J(ch)))
        gr = jax.jit(jax.grad(dot_loss(g2, J(cx), J(ch)), argnums=tuple(range(7))))(*P_)
        if not (eq(r[4], dx) and eq(r[5], dh)):
            bad.append(f"jvp {'A' if adapt else 'F'}")
        if not all(eq(g[k], b) for k, b in zip(CLS, bars)):
            bad.append(f"vjp {'A' if adapt else 'F'}")
        if not all(eq(g[k], b) for k, b in zip(CLS, gr)):
            bad.append(f"grad {'A' if adapt else 'F'}")
    verdict("recovered cells", ok0 and ok1 and nf[0] > 0 and not bad,
            f"{B} settled cells at dt 1000 s, max_steps 3: the call with lm = 0 bitwise CudaSolver.launch incl. flags "
            f"and counts {dict(zip(KC.COUNTS, cnt))}: {ok0}; with random lm bitwise ADSolver's primal, flags and "
            f"counts {ok1} ({nf[0]} recovered, {nf[1]} still failed); jit(jvp), vjp, grad bitwise ADSolver (both "
            f"contracts) except {bad or 'none'}")


def jvp_cost():
    B = 4096
    x, h, fo = gcm_states(B)
    a = (x, *fo[0], h)
    P_ = tuple(J(q) for q in a) + (jnp.zeros((B, NR)), RC0)
    v = tangents(a)
    V_ = tuple(J(q) for q in v)
    cx, ch = J(rn(B, NS)), J(rn(B))
    f = fn(True)
    jv = jax.jit(lambda p, t: jax.jvp(f, p, t))
    full_jv = jax.jit(lambda p, t: jax.jvp(lambda *q: kj.call(*q), p, t)[0])
    n_j = [q for q, _ in prims(jax.make_jaxpr(jv)(P_, V_))]
    n_l = [q for q, _ in prims(jax.make_jaxpr(lambda *p: jax.linearize(f, *p)[0])(*P_))]
    n_g = [q for q, _ in prims(jax.make_jaxpr(jax.grad(dot_loss(f, cx, ch), argnums=tuple(range(7))))(*P_))]
    cnt = lambda ns: {k: ns.count(k) for k in ("kinjax_call", "kinjax_lin", "kinjax_adj")}  # noqa: E731
    s_j = cnt(n_j) == {"kinjax_call": 0, "kinjax_lin": 1, "kinjax_adj": 0}
    s_l = cnt(n_l) == {"kinjax_call": 1, "kinjax_lin": 0, "kinjax_adj": 0}
    s_g = cnt(n_g) == {"kinjax_call": 1, "kinjax_lin": 0, "kinjax_adj": 1}
    prim_eq = all(eq(p, q) for p, q in zip(kj(*P_), full_jv(P_, V_)))
    call = jax.jit(lambda *p: f(*p))
    runs = {"jit(jvp)": lambda: jax.block_until_ready(jv(P_, V_)), "call": lambda: jax.block_until_ready(call(*P_)),
            "ADSolver.jvp": lambda: (ad.jvp(*a, *v[:5], dlm=v[5], drc=v[6]), torch.cuda.synchronize())}
    for g in runs.values():
        g()
    tm = {k: [] for k in runs}
    for k in range(10):                                       # rotated order, paired
        for name in list(runs)[k % 3:] + list(runs)[:k % 3]:
            t0 = time.perf_counter()
            runs[name]()
            tm[name].append(time.perf_counter() - t0)
    med = {k: float(np.median(q)) for k, q in tm.items()}
    verdict("forward mode in one tangent call", s_j and s_l and s_g and prim_eq,
            f"jit(jvp) jaxpr {cnt(n_j)} {s_j}; linearize's primal jaxpr {cnt(n_l)} {s_l}; grad jaxpr {cnt(n_g)} {s_g}; "
            f"jvp's primal outputs (x, h, done, steps, flag, status) bitwise the call's {prim_eq}")
    info("jvp cost (paired medians of 10 rotated runs, 4,096 GCM-like cells, GPU may be shared)",
         f"jit(jvp) {1e3 * med['jit(jvp)']:.2f} ms, ADSolver.jvp {1e3 * med['ADSolver.jvp']:.2f} ms, the call "
         f"{1e3 * med['call']:.2f} ms: jit(jvp) / ADSolver.jvp {med['jit(jvp)'] / med['ADSolver.jvp']:.2f}, "
         f"jit(jvp) / call {med['jit(jvp)'] / med['call']:.2f}")


def edges():
    """No lm with an rc tangent / cotangent; no h_out cotangent; the reduction skipped when rc's cotangent is not
    wanted; vmap of vmap of grad (groups multiply); what is refused; empty batches."""
    B = 256
    x, h, fo = gcm_states(B)
    a = (x, *fo[0], h)
    P5 = tuple(J(q) for q in a)
    drc, cx = rn(NR, 9) * 1e-3, rn(B, NS)
    zb = torch.zeros_like(h)
    r = ad.jvp(*a, torch.zeros_like(x), zb, zb, zb, zb, drc=drc)
    _, (dx, dh) = jax.jvp(lambda rc: fn(True)(*P5, None, rc), (RC0,), (J(drc),))
    g = ad.vjp(*a, cx)[4]
    xloss = lambda *p: jnp.sum(J(cx) * fn(True)(*p)[0])
    gr = jax.grad(xloss, argnums=(0, 1, 2, 3, 4, 6))(*P5, None, RC0)
    ok1 = eq(r[4], dx) and eq(r[5], dh) and all(eq(g[k], q) for k, q in zip(("x", "T", "P", "dt", "h", "rc"), gr))
    adj = [e for e in eqns(jax.make_jaxpr(jax.grad(xloss))(*P5, None, RC0)) if e.primitive.name == "kinjax_adj"]
    ok2 = len(adj) == 1 and not adj[0].params["want_rc"] and not adj[0].params["want_lm"] and adj[0].outvars[
        6].aval.shape == (0,)
    # vmap of vmap of grad: [2, 128] cells, the Arrhenius cotangent per cell
    lm = rn(B, NR) * 1e-2
    cell = lambda x, T, P, dt, h, lm, rc: kj.call(x, T, P, dt, h, lm, rc)[:2]
    lc = lambda x, T, P, dt, h, lm, rc, cx, ch: dot_loss(cell, cx, ch)(x, T, P, dt, h, lm, rc)
    ch = rn(B)
    ax = (0,) * 6 + (None, 0, 0)
    vv = jax.jit(jax.vmap(jax.vmap(jax.grad(lc, argnums=(5, 6)), in_axes=ax), in_axes=ax))
    sh = lambda q: q.reshape((2, B // 2) + q.shape[1:])
    glm2, grc2 = vv(*(sh(q) for q in P5), sh(J(lm)), RC0, sh(J(cx)), sh(J(ch)))
    gf = ad.vjp(*a, cx, ch, lm=lm)[4]
    ok3 = eq(gf["lm"], np.asarray(glm2).reshape(B, NR)) and grc2.shape == (2, B // 2, NR, 9)
    for i in (0, 77, 200):
        s = slice(i, i + 1)
        gi = ad.vjp(*(q[s] for q in a), cx[s], ch[s], lm=lm[s])[4]
        ok3 = ok3 and eq(gi["rc"], np.asarray(grc2).reshape(B, NR, 9)[i])
    # refused: a mapped Arrhenius table; a mapped rc tangent over a mapped call; reverse mode above CAP^2 steps
    refused = []
    try:
        jax.vmap(lambda rc: fn(True)(*P5, None, rc))(jnp.stack([RC0, RC0]))
    except NotImplementedError:
        refused.append("mapped rc")
    try:
        jax.vmap(lambda p, t: jax.jvp(cell, p, t), in_axes=((0,) * 6 + (None,), (0,) * 7))(
            P5 + (J(lm), RC0), tuple(jnp.zeros_like(q) for q in P5 + (J(lm),)) + (jnp.zeros((B, NR, 9)),))
    except NotImplementedError:
        refused.append("mapped rc tangent over mapped cells")
    try:
        jax.grad(lambda x: jnp.sum(kj.call(x, *P5[1:], max_steps=5000)[0]))(P5[0])
    except ValueError:
        refused.append("max_steps 5000 in reverse mode")
    # empty batches
    E = tuple(q[:0] for q in P5)
    _, (dxe, _) = jax.jvp(lambda x: fn(True)(x, *E[1:], None, RC0), (E[0],), (E[0],))
    ge = jax.grad(lambda *p: jnp.sum(fn(True)(*p)[0]), argnums=(0, 6))(*E, None, RC0)
    dxv = jax.vmap(lambda t: jax.jvp(lambda x: fn(True)(x, *E[1:], None, RC0), (E[0],), (t,))[1][0])(
        jnp.zeros((3, 0, NS)))
    ok4 = dxe.shape == (0, NS) and ge[0].shape == (0, NS) and not np.asarray(ge[1]).any() and dxv.shape == (3, 0, NS)
    verdict("edge cases", ok1 and ok2 and ok3 and len(refused) == 3 and ok4,
            f"no lm: rc tangent and x-only cotangents (h_out cotangent absent) bitwise ADSolver {ok1}; grad w.r.t. "
            f"x, T, P, dt, h only: one kinjax_adj without the Arrhenius reduction {ok2}; vmap(vmap(grad)) over "
            f"[2, {B // 2}] cells: lm cotangents bitwise the flat call, Arrhenius cotangent per cell bitwise a "
            f"one-cell call {ok3}; refused with an error: {refused}; empty batch jvp and grad {ok4}")


def main():
    print(f"kernels built ({time.time() - t_start:.0f} s); cells per block and resident blocks "
          f"{KJ._MODULES[kj.module]}", flush=True)
    cs = cases()
    for f, a in ((primal, (cs,)), (parity, (cs,)), (dot_tests, ()), (finite_differences, ()), (transforms, ()),
                 (scan_window, ()), (end_to_end, ()), (status, ()), (recovered, ()), (jvp_cost, ()),
                 (edges, ())):
        f(*a)
        print(f"  ({time.time() - t_start:.0f} s)", flush=True)
    print(f"{sum(results)} of {len(results)} cases PASS", flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
