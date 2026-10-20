"""The score: models (and mini-chem) against VULCAN on test-set designs, one number per model and a paired comparison.
NumPy only.

    python score.py --set DIR --run DIR --out DIR MODEL [OTHER ...]

--set  a design folder (manifest.json; bench/score_design.py writes it), a folder of design folders (a test set: the
       decision set holds the score panels S, L, P and the readout designs Sraw and side), or a stored yardstick-format
       design (rows.npz with dt_idx and dt_bin_weights, parents.npz with cells, meta.json; the mini-chem yardstick and
       F1), read as stored, not rebuilt.
--run  <run>/<MODEL>/<design name>/states.npz and meta.json (bench/score_run.py); for a yardstick-format design
       <run>/<MODEL>/model_states.npz and meta.json, and <run>/minichem/<cell>.npz and rows_summary.npz. Optional
       readout inputs: <run>/<MODEL>/cyc1000.json (bench/cyc1000.py), <run>/<MODEL>/speed_keep.json and speed_lm.json
       (bench/speed.py).
MODEL  the first model (A) is compared with each OTHER (B, the incumbent); 'minichem' is a solver on designs that carry
       mini-chem files. States whose recorded design hash differs from the design in use are refused.
Writes score.json, tables.txt and large_misses_<solver>_<design>.csv into --out (refuses an existing score.json).

Per saved call: the 12 reported gases (log10), validated before flooring; per-gas floors S (1e-10 H2O CO CO2 CH4 NH3 HCN
C2H2; 1e-8 H OH O; 1e-6 H2 N2); e = |max(pred, S) - max(truth, S)|; statistic = 12-gas mean (primary); worst gas,
6-major mean and 5-trace mean are readouts. A failure is an unrecovered step: a model row fails at its first kept input
(the projection failed after its LM rescue), guard rejection or non-finite state, or at an invalid saved state; LM
rescues are counted, not failures. Mini-chem fails at its driver's first failed call or a saved state whose sum differs
from 1 by more than 1e-6 (its driver floors before testing, so its count is a lower bound). From a row's first failed
call on every saved call scores 2 dex (readouts at 0.5 and 5 dex).

Row -> condition (exact quadrature): the statistic is piecewise linear in log10 t between saved calls (a restart row is
held at its first value back to t0; a t = 0 row starts at its first call) and is integrated over each decade of t in
[1, 1e8] s with the pieces split at the decade edges; zero covered log-time = untested. Case = (parent, dt slot): a
segment design pools a dt value's t = 0 row with its restarts, a long or path design has one row per case. Case score =
mean over its tested decades. Bin score of a parent = mean of its cases in that bin; parent = sum_b w_b bin_b over the
tested bins (w_b = log(b_hi / b_lo) / log 500) in segment and long designs, the row itself in path designs (its bin is
that of dt0). Slots past the bins (the 10 s rows of S) are readouts. Condition = start-weighted mean of its parents
(manifest start weights; eqt 0.55, eqf 0.45, raw starts 0), renormalised over the parents present.

Panel score = 5 % symmetric trimmed mean of the in-domain condition scores (plain mean beside it). With S, L and P
present: SCORE = (S + L + P) / 3. Comparison: D_c = A_c - B_c per condition; D = mean over panels of the trimmed means,
D_mean = of the plain means; bootstrap: conditions resampled with replacement within each (panel, bootstrap group),
panels independently, 4000 resamples, fixed seed, trim recomputed; 95 % percentile intervals. Outcome at the manifest's
tie margin delta: BETTER if D <= -delta and D's upper limit < 0, WORSE if D >= delta and its lower limit > 0, else TIE.
D_mean is not part of the outcome (flag R1).

Readouts (never the outcome): (a) lasting error, the log-time mean over [t_end / 10, t_end] of every row from t = 0 (L,
P); (b) per-bin D inside the joint bootstrap with a studentized max-|z| simultaneous band over the bins, absolute
per-bin scores, the 10 s rows of S; (c) raw starts (the Sraw design) and the late window of its t = 0 rows; (d) large
misses (rows with a non-failed saved call over 0.5 / 2 dex) with identities, failures; (e) the 1000-day cycle; (f)
speed; (g) mini-chem on its native cells (yardstick-format designs, conditions resampled within cells); (h) continuity:
the yardstick under its stored start weights (rule v3.1), the score at start weights 0.70 / 0.30 and 0.40 / 0.60; (i)
diagnostics by decade, restart age, time since restart, start, stratum, T band, P decade, Z and C/O terciles, He/H
halves, fixed / variable dt, mixed / unmixed paths, coverage; (j) snap-back on the T2 design (kicked rows minus their
identity partners, bootstrap by source job, in-domain boxes only). Flags R1-R7 are listed for reading before adoption;
none is a veto."""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np

FORMAT = "score-1 design"
GASES = ["C2H2", "CH4", "CO2", "CO", "H2O", "H2", "HCN", "H", "N2", "NH3", "OH", "O"]       # design gases12 order
FLOOR = {"H2O": 1e-10, "CO": 1e-10, "CO2": 1e-10, "CH4": 1e-10, "NH3": 1e-10, "HCN": 1e-10, "C2H2": 1e-10,
         "H": 1e-8, "OH": 1e-8, "O": 1e-8, "H2": 1e-6, "N2": 1e-6}
LOGS = np.log10(np.array([FLOOR[g] for g in GASES]))
MAJOR6 = [GASES.index(g) for g in ("H2O", "CO", "CO2", "CH4", "NH3", "N2")]
TRACE5 = [GASES.index(g) for g in ("HCN", "C2H2", "H", "OH", "O")]
STATS = ("mean12", "worst", "major6", "trace5")
SEED, NBOOT, TRIM, PEN, PENS = 20261020, 4000, 0.05, 2.0, (0.5, 5.0)
DELTA = 0.005                                                    # tie margin of the yardstick-format designs (no manifest)
EDGES = 10.0 ** np.arange(9)                                     # 1, 10, ..., 1e8 s; decade d = (10^d, 10^(d+1)]
AGES = [0, 1, 3, 10, 30, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7]
FREE_BIN = {500.0: 7, 100.0: 6, 10.0: 3}                         # yardstick free-run dt -> bin
LEGACY_CODE = {"eq-target": "eqt", "eq-far": "eqf", "raw-A": "rawA", "raw-B": "rawB", "raw-C": "rawC"}
RAW_W = {"rawA": 0.5, "rawB": 0.5}                               # the raw-start table: rawA and rawB, equal weight
ALT_W = ({"eqt": 0.70, "eqf": 0.30}, {"eqt": 0.40, "eqf": 0.60})  # continuity start weights
TBANDS = ((300, 650), (650, 1300), (1300, 2000), (2000, 3000))


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------- per-call statistics
def merge_bad(D, ff, bad):
    """First failed call per row (1-based, 0 = none) after adding the saved calls with an invalid state."""
    ff = ff.astype(np.int64).copy()
    if bad.any():
        big = np.iinfo(np.int64).max
        fb = np.full(D.R, big)
        np.minimum.at(fb, D.row[bad], D.k[bad])
        fb = np.where(fb == big, 0, fb)
        ff = np.where((ff > 0) & (fb > 0), np.minimum(ff, fb), np.maximum(ff, fb))
    return ff


def point_stats(D, logp, ffrow):
    """logp [S, 12] log10 prediction (NaN = invalid). Returns the per-point statistics, the worst-gas index, the mask
    of invalid saved calls and the first failed call per row with those calls added."""
    bad = ~np.isfinite(logp).all(1)
    out = {s: np.empty(D.S) for s in STATS}
    wg = np.empty(D.S, np.int8)
    for a in range(0, D.S, 2_000_000):
        b = min(a + 2_000_000, D.S)
        e = np.abs(np.maximum(logp[a:b], LOGS) - np.maximum(D.logt[a:b], LOGS))
        e[~np.isfinite(e)] = 0.0                              # only at invalid points, which are failed (ff) anyway
        out["mean12"][a:b] = e.mean(1)
        out["worst"][a:b] = e.max(1)
        out["major6"][a:b] = e[:, MAJOR6].mean(1)
        out["trace5"][a:b] = e[:, TRACE5].mean(1)
        wg[a:b] = e.argmax(1)
    out["wgas"] = wg
    out["bad"] = bad
    out["ff"] = merge_bad(D, ffrow, bad)
    out["nbad_saved"] = int(bad.sum())
    return out


def failed_points(D, ff):
    f = ff[D.row]
    return (f > 0) & (D.k >= f)


def load_minichem(D, mc, rtol5=False):
    """Mini-chem log10 states at every saved point (or only the rtol 1e-5 subset's) and first failed call per row. Cells
    without a mini-chem file are left out (their points stay NaN and have[] False)."""
    sp = None
    logp = np.full((D.S, 12), np.nan)
    ffrow = np.zeros(D.R, np.int64)
    files = [mc / "rtol1e-5.npz"] if rtol5 else [mc / f"{c}.npz" for c in D.cells if (mc / f"{c}.npz").exists()]
    have = np.zeros(D.S, bool)
    for f in files:
        z = np.load(f)
        sp13 = [str(x) for x in z["species13"]]
        if sp is None:
            sp = sp13
        m = [sp13.index(g) for g in GASES]
        v = z["vmr13"]
        sidx = z["sidx"]
        assert np.array_equal(z["k"].astype(np.int64), D.k[sidx])
        ok = np.isfinite(v).all(1) & (v >= 0).all(1)                    # the full 13-species state
        lv = np.full((sidx.size, 12), np.nan)
        with np.errstate(divide="ignore"):
            lv[ok] = np.log10(v[ok][:, m])
        logp[sidx] = lv
        have[sidx] = True
        rows = z["row"].astype(np.int64)
        ffrow[rows] = z["first_fail_any"]
    if not rtol5:
        rs = np.load(mc / "rows_summary.npz")
        hr = np.zeros(D.R, bool)
        hr[D.row[have]] = True
        assert np.array_equal(rs["first_fail_any"].astype(np.int64)[hr], ffrow[hr])
    return logp, ffrow, have


def load_model(path):
    """Model log10 states and first calls: literal = first LM rescue, guard rejection or non-finite state (the runner's
    first_fail); unrecovered = first kept input, guard rejection or non-finite state. 0 = none."""
    z = np.load(path)
    assert [str(g) for g in z["gases12"]] == GASES
    logp = z["logy12"]
    ff = z["first_fail"].astype(np.int64)
    ff = np.where(ff > 0, ff, 0)
    big = np.iinfo(np.int64).max
    unrec = np.stack([z["first_kept"], z["first_guard"], z["first_nonfin"]]).astype(np.int64)
    unrec = np.where(unrec > 0, unrec, big).min(0)
    unrec = np.where(unrec == big, 0, unrec)
    extra = {k: z[k].astype(np.int64) for k in ("newton_n", "kept_n", "guard_n", "nonfin_n", "first_newton",
                                                 "first_kept", "first_guard", "first_nonfin")}
    return logp, ff, unrec, extra


# ------------------------------------------------------------------------------------------------- quadrature
def pieces(D):
    """The piecewise-linear-in-log10-t pieces of every row (exact quadrature): consecutive saved calls (linear), plus
    [log10 t0, u_1] at the first call's value for restart rows. Cached on D."""
    if getattr(D, "_pc", None) is None:
        u = np.log10(D.tabs)
        for k in range(9):
            u[D.tabs == 10.0 ** k] = float(k)
        same = np.zeros(D.S, bool)
        same[1:] = D.row[1:] == D.row[:-1]
        i = np.flatnonzero(same)
        t0 = D.r["t0"]
        rs = np.flatnonzero(t0 > 0)
        f = D.r["off"][:-1][rs]
        u0 = np.log10(t0[rs])
        for k in range(9):
            u0[t0[rs] == 10.0 ** k] = float(k)
        D._pc = dict(pa=np.concatenate([i - 1, f]), pb=np.concatenate([i, f]),
                     ua=np.concatenate([u[i - 1], u0]), ub=np.concatenate([u[i], u[f]]),
                     prow=np.concatenate([D.row[i], rs]))
        assert (D._pc["ub"] >= D._pc["ua"]).all()
    return D._pc


def pieces_elapsed(D):
    """Pieces in log10 of the time since the row's own start (tau = saved tel): consecutive saved calls only (tau = 0 at
    the start has no logarithm, so coverage starts at the first saved call). Cached on D."""
    if getattr(D, "_pe", None) is None:
        tau = D.tel
        u = np.log10(tau)
        for k in range(9):
            u[tau == 10.0 ** k] = float(k)
        same = np.zeros(D.S, bool)
        same[1:] = D.row[1:] == D.row[:-1]
        i = np.flatnonzero(same)
        D._pe = dict(pa=i - 1, pb=i, ua=u[i - 1], ub=u[i], prow=D.row[i])
        assert (D._pe["ub"] >= D._pe["ua"]).all() and (D._pe["ua"] >= 0).all()
    return D._pe


def exact_sums(D, e, rowmask, rowsel, ncase, pc=None):
    """Exact integral of the piecewise-linear (log10 t) per-call statistic over each decade, pieces split at the decade
    edges; returns covered log-length and integral per (case, decade). rowsel / rowmask: rows taken (None = all); pc: the
    pieces (default pieces(D))."""
    pc = pieces(D) if pc is None else pc
    sel = np.ones(pc["prow"].size, bool) if rowsel is None else rowsel[pc["prow"]]
    if rowmask is not None:
        sel &= rowmask[pc["prow"]]
    ua, ub = pc["ua"][sel], pc["ub"][sel]
    ea, eb = e[pc["pa"][sel]], e[pc["pb"][sel]]
    case = D.case_row[pc["prow"][sel]]
    span = np.where(ub > ua, ub - ua, 1.0)
    sw = np.zeros(ncase * 8)
    swe = np.zeros(ncase * 8)
    for d in range(8):
        lo, hi = np.maximum(ua, d), np.minimum(ub, d + 1.0)
        m = hi > lo
        vl = ea[m] + (eb[m] - ea[m]) * (lo[m] - ua[m]) / span[m]
        vh = ea[m] + (eb[m] - ea[m]) * (hi[m] - ua[m]) / span[m]
        L = hi[m] - lo[m]
        key = case[m] * 8 + d
        sw += np.bincount(key, L, ncase * 8)
        swe += np.bincount(key, L * 0.5 * (vl + vh), ncase * 8)
    return sw, swe


def window_means(D, e, lo, hi, rowmask=None, pc=None):
    """Per-row log-time mean of the per-call statistic over [lo[row], hi[row]] (log10 s) with the same piecewise-linear
    quadrature; NaN where the window holds no covered log-time."""
    pc = pieces(D) if pc is None else pc
    sel = np.ones(pc["prow"].size, bool) if rowmask is None else rowmask[pc["prow"]]
    ua, ub, pr = pc["ua"][sel], pc["ub"][sel], pc["prow"][sel]
    ea, eb = e[pc["pa"][sel]], e[pc["pb"][sel]]
    span = np.where(ub > ua, ub - ua, 1.0)
    a, b = np.maximum(ua, lo[pr]), np.minimum(ub, hi[pr])
    m = b > a
    va = ea[m] + (eb[m] - ea[m]) * (a[m] - ua[m]) / span[m]
    vb = ea[m] + (eb[m] - ea[m]) * (b[m] - ua[m]) / span[m]
    L = b[m] - a[m]
    sw = np.bincount(pr[m], L, D.R)
    swe = np.bincount(pr[m], L * 0.5 * (va + vb), D.R)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sw > 0, swe / np.where(sw > 0, sw, 1.0), np.nan)


def case_means(sw, swe, ncase):
    """(case, decade) means from the covered log-length sw and integral swe; a decade with sw = 0 is untested (NaN).
    Case score = mean over its tested decades."""
    with np.errstate(invalid="ignore", divide="ignore"):
        V = np.where(sw > 0, swe / np.where(sw > 0, sw, 1.0), np.nan).reshape(ncase, 8)   # (case, decade)
    return V, _nanmean(V, 1)


def _nanwmean(V, w, axis):
    """Weighted mean over axis, weights renormalised over finite entries; NaN where none is finite."""
    V = np.moveaxis(V, axis, -1)
    ok = np.isfinite(V)
    ww = np.where(ok, w, 0.0)
    s = ww.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, (np.where(ok, V, 0.0) * ww).sum(-1) / np.where(s > 0, s, 1.0), np.nan)


def _nanmean(V, axis):
    return _nanwmean(V, np.ones(V.shape[axis]), axis)


# ------------------------------------------------------------------------------------------------ summaries, bootstrap
def trimmed(X, axis=-1):
    X = np.sort(X, axis=axis)
    n = X.shape[axis]
    g = int(np.floor(TRIM * n))
    sl = [slice(None)] * X.ndim
    sl[axis] = slice(g, n - g)
    return X[tuple(sl)].mean(axis=axis)


def tmean(X, axis=-1):
    """trimmed() over the finite entries along axis (equal to trimmed() when all are finite); NaN where none is."""
    X = np.asarray(X, float)
    if np.isfinite(X).all():
        return trimmed(X, axis)
    X = np.sort(np.moveaxis(X, axis, -1), axis=-1)                     # NaN sorts last
    n = np.isfinite(X).sum(-1)
    g = np.floor(TRIM * n).astype(np.int64)
    j = np.arange(X.shape[-1])
    m = (j >= g[..., None]) & (j < (n - g)[..., None])
    c = m.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, np.where(m, X, 0.0).sum(-1) / np.where(c > 0, c, 1), np.nan)


def nmean(X, axis=-1):
    """Plain mean over the finite entries along axis; NaN where none is."""
    X = np.asarray(X, float)
    ok = np.isfinite(X)
    c = ok.sum(axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, np.where(ok, X, 0.0).sum(axis) / np.where(c > 0, c, 1), np.nan)


_IDX = {}


def boot_idx(cells):
    """(NBOOT, n) resample indices: conditions drawn with replacement within each cell; fixed seed."""
    key = cells.tobytes()
    if key not in _IDX:
        rng = np.random.default_rng(SEED)
        cols = []
        for c in np.unique(cells):
            m = np.flatnonzero(cells == c)
            cols.append(m[rng.integers(0, m.size, size=(NBOOT, m.size))])
        _IDX[key] = np.concatenate(cols, 1)
    return _IDX[key]


def strat_idx(groups, panel_no):
    """(NBOOT, n) resample indices of one panel in the joint bootstrap: drawn with replacement within each bootstrap
    group, from the panel's own generator default_rng([SEED, panel_no]) (panels independent); column j holds a draw of
    the group of condition j, so a row of indices keeps every group's size."""
    rng = np.random.default_rng([SEED, panel_no])
    out = np.empty((NBOOT, groups.size), np.int64)
    for c in np.unique(groups):
        m = np.flatnonzero(groups == c)
        out[:, m] = m[rng.integers(0, m.size, size=(NBOOT, m.size))]
    return out


def summarize(x, cells):
    """Trimmed mean, mean, median of x with 95% percentile bootstrap intervals (resampling within cells)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return dict(n=0)
    x, cells = x[ok], cells[ok]
    X = x[boot_idx(cells)]
    o = dict(n=int(x.size), n_dropped=int((~ok).sum()), frac_neg=float((x < 0).mean()),
             degenerate=bool(np.unique(cells, return_counts=True)[1].max() == 1))
    for nm, f, fb in (("trim", trimmed(x), trimmed(X, 1)), ("mean", x.mean(), X.mean(1)),
                      ("median", np.median(x), np.median(X, 1))):
        lo, hi = np.percentile(fb, [2.5, 97.5])
        o[nm], o[nm + "_lo"], o[nm + "_hi"] = float(f), float(lo), float(hi)
    return o


def band(Xb, cells):
    """Studentized two-sided 95% max-|z| simultaneous band for the trimmed mean of each column of Xb (n, nb)."""
    ok = np.isfinite(Xb).all(1)
    Xb, cells = Xb[ok], cells[ok]
    T = trimmed(Xb, 0)
    Tb = trimmed(Xb[boot_idx(cells)], 1)                                        # (NBOOT, nb)
    se = Tb.std(0, ddof=1)
    z = np.abs(Tb - T) / np.where(se > 0, se, np.inf)
    c = float(np.percentile(z.max(1), 95))
    return dict(T=T.tolist(), se=se.tolist(), c=c, lo=(T - c * se).tolist(), hi=(T + c * se).tolist(), n=int(ok.sum()))


def fail_rates(D, ff, rows_sel):
    N = D.r["N"][rows_sel].astype(np.int64)
    f = ff[rows_sel]
    fc = np.where(f > 0, N - f + 1, 0)
    return dict(calls=int(N.sum()), failed_calls=int(fc.sum()), call_rate=float(fc.sum() / max(N.sum(), 1)),
                rows=int(N.size), failed_rows=int((f > 0).sum()), row_rate=float((f > 0).mean()) if N.size else 0.0)


def outcome(d, lo, hi, delta):
    """BETTER / TIE / WORSE of A against B from D and its 95 % interval at the tie margin delta."""
    if d <= -delta and hi < 0:
        return "BETTER"
    if d >= delta and lo > 0:
        return "WORSE"
    return "TIE"


def interval(xb):
    xb = np.asarray(xb, float)
    xb = xb[np.isfinite(xb)]
    if xb.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(xb, [2.5, 97.5])
    return float(lo), float(hi)


def joint(ds, idxs):
    """Paired summary over panels. ds: per-panel condition values (the scored conditions; NaN = untested; 1-D, or 2-D
    with one column per sub-table), idxs: per-panel (NBOOT, n) resample indices of the joint bootstrap. Returns D (mean
    over panels of the trimmed means), D_mean (of the plain means), the per-panel values and 95 % percentile intervals
    from the joint resamples; panels without a finite value are left out of the mean (per column)."""
    T = np.stack([tmean(d, 0) for d in ds])                                  # (npanel[, ncol])
    M = np.stack([nmean(d, 0) for d in ds])
    Tb = np.stack([tmean(d[i], 1) for d, i in zip(ds, idxs)])                # (npanel, NBOOT[, ncol])
    Mb = np.stack([nmean(d[i], 1) for d, i in zip(ds, idxs)])
    Dp, Dm, Db, Dmb = nmean(T, 0), nmean(M, 0), nmean(Tb, 0), nmean(Mb, 0)
    out = dict(D=Dp, D_mean=Dm, T=T, M=M, Db=Db, Dmb=Dmb, Tb=Tb, Mb=Mb)
    return out


def jsum(j, col=None):
    """JSON-ready numbers of one column of a joint() result."""
    g = (lambda a: a) if col is None else (lambda a: a[..., col])
    D, Dm, Db, Dmb = g(j["D"]), g(j["D_mean"]), g(j["Db"]), g(j["Dmb"])
    lo, hi = interval(Db)
    mlo, mhi = interval(Dmb)
    T, M, Tb, Mb = g(j["T"]), g(j["M"]), g(j["Tb"]), g(j["Mb"])
    per = []
    for p in range(T.shape[0]):
        a, b = interval(Tb[p])
        c, d = interval(Mb[p])
        per.append(dict(trim=float(T[p]), trim_lo=a, trim_hi=b, mean=float(M[p]), mean_lo=c, mean_hi=d))
    return dict(D=float(D), D_lo=lo, D_hi=hi, D_mean=float(Dm), D_mean_lo=mlo, D_mean_hi=mhi, panels=per)


# ------------------------------------------------------------------------------------------------ yardstick-format designs
class Yard:
    """A stored yardstick-format design, read as stored: segments restarted from the truth at the restart ages, with m
    dt draws per bin plus a 10 s anchor slot, and free runs from t = 0 at the dt values of meta.json 'free_runs'.
    Conditions = (cell, rank); its stored start weights (meta.json parent_weights) are rule v3.1's."""
    kind = "yard"

    def __init__(self, des):
        des = Path(des)
        self.dir, self.name = des, des.name
        r = np.load(des / "rows.npz")
        self.r = {k: r[k] for k in ("kind", "pidx", "unit", "cell", "cond", "start", "age_idx", "t0", "dt", "dt_idx",
                                    "dt_bin", "N", "T", "P_bar", "off")}
        self.cells = [str(c) for c in r["cells"]]
        self.starts = [str(s) for s in r["starts"]]
        self.bin_w = r["dt_bin_weights"].astype(float)
        self.bins = r["dt_bins"]
        s = np.load(des / "saved.npz")
        self.k, self.tabs, self.row = s["k"].astype(np.int64), s["tabs"], s["row"].astype(np.int64)
        self.tel = s["tel"]
        p = np.load(des / "parents.npz")
        self.p = {k: p[k] for k in ("cell", "rank", "start", "stratum", "T", "P_bar", "free", "free_order", "cond_key")}
        self.R, self.S, self.P = self.r["N"].size, self.k.size, self.p["cell"].size
        conv = json.loads((des / "meta.json").read_text())["conventions"]
        self.pw = np.array([conv["parent_weights"][st] for st in self.starts], float)        # stored (v3.1) weights
        self.free_dt = tuple(float(f["dt"]) for f in conv.get("free_runs", []))
        off = self.r["off"]
        assert off[-1] == self.S and np.array_equal(self.row, np.repeat(np.arange(self.R), np.diff(off)))
        assert (self.tabs > 1.0).all() and (self.tabs <= 1e8).all()
        self.ncpc = int(self.p["rank"].max()) + 1
        self.ncond = len(self.cells) * self.ncpc
        self.pcond = self.p["cell"] * self.ncpc + self.p["rank"]
        assert np.unique(self.pcond * len(self.starts) + self.p["start"]).size == self.P
        self.cond_cell = np.arange(self.ncond) // self.ncpc
        st = np.empty(self.ncond, "U1")
        st[self.pcond] = self.p["stratum"]
        self.cond_stratum = st
        Tc = np.empty(self.ncond)
        Tc[self.pcond] = self.p["T"]
        self.cond_T = Tc
        kind = self.r["kind"]
        self.seg_row = kind == 0
        seg = self.seg_row
        self.nslot = int(self.r["dt_idx"][seg].max()) + 1
        self.m = (self.nslot - 1) // 8
        q = self.r["dt_idx"][seg]
        assert np.array_equal(self.r["dt_bin"][seg], np.where(q < 8 * self.m, q // self.m, -1))
        nf = len(self.free_dt)
        fidx = np.full(self.R, -1)
        for i, d in enumerate(self.free_dt):
            fidx[(kind == 1) & (self.r["dt"] == d)] = i
        assert (fidx[kind == 1] >= 0).all()
        self.fidx = fidx
        case = np.where(kind == 0, self.r["pidx"] * self.nslot + self.r["dt_idx"],
                        self.p["free_order"][self.r["pidx"]] * max(nf, 1) + fidx)
        assert (case >= 0).all()
        self.case_row = case
        self.nfree = int(self.p["free"].sum())
        if self.nfree:
            self.fparents = np.flatnonzero(self.p["free"])[np.argsort(self.p["free_order"][self.p["free"]])]
            assert np.array_equal(self.p["free_order"][self.fparents], np.arange(self.nfree))
            self.fcond = np.unique(self.pcond[self.fparents])
            self.fcond_cell = self.fcond // self.ncpc
            self.fparent_fc = np.searchsorted(self.fcond, self.pcond[self.fparents])
            self.fparent_start = self.p["start"][self.fparents]
            self.fstarts = sorted(set(self.fparent_start.tolist()))
        t = np.load(des / "truth_saved.npz")
        assert [str(g) for g in t["gases12"]] == GASES
        self.logt = t["logx12"]
        log(f"design {des.name} (stored yardstick format) loaded: {self.R} rows, {self.S} saved points")

    def weights(self, w):
        """Start weights by code (eqt, eqf, rawA, rawB, rawC) -> array in this design's start order (missing = 0)."""
        return np.array([w.get(LEGACY_CODE[s], 0.0) for s in self.starts], float)


def yard_profile(D, st, ff, stats=("mean12", "worst"), pen=PEN):
    """Per-parent arrays of one solver on a yardstick-format design (exact quadrature): segments pscore [P] (bins
    weighted, the anchor slot out), bm [P, 8], anchor [P], pdec, adec [P, 8]; free runs cs [nfree, nf], Vd [nfree, nf,
    8]."""
    fp = failed_points(D, ff)
    out = {}
    for sname in stats:
        e = np.where(fp, pen, st[sname])
        ncase = D.P * D.nslot
        V, case = case_means(*exact_sums(D, e, None, D.seg_row, ncase), ncase)
        Vd = V.reshape(D.P, D.nslot, 8)
        cs = case.reshape(D.P, D.nslot)
        m8 = 8 * D.m
        bm = _nanmean(cs[:, :m8].reshape(D.P, 8, D.m), 2)                          # (parent, bin)
        dbm = _nanmean(Vd[:, :m8, :].reshape(D.P, 8, D.m, 8), 2)                    # (parent, bin, decade)
        res = {"seg": dict(pscore=_nanwmean(bm, D.bin_w, 1), bm=bm, anchor=cs[:, m8],
                           pdec=_nanwmean(np.moveaxis(dbm, 1, 2), D.bin_w, 2), adec=Vd[:, m8, :])}
        if D.nfree:
            nf = len(D.free_dt)
            V, case = case_means(*exact_sums(D, e, None, ~D.seg_row, D.nfree * nf), D.nfree * nf)
            res["free"] = dict(cs=case.reshape(D.nfree, nf), Vd=V.reshape(D.nfree, nf, 8))
        out[sname] = res
    return out


def _cond(D, X, w):
    """Parent array (P, ...) -> condition array (ncond, ...), parent weights w renormalised over finite parents."""
    sh = X.shape[1:]
    M = np.full((D.ncond, len(D.starts)) + sh, np.nan)
    M[D.pcond, D.p["start"]] = X
    return _nanwmean(np.moveaxis(M, 1, -1), w, -1)


def _by_start(D, X):
    M = np.full((D.ncond, len(D.starts)), np.nan)
    M[D.pcond, D.p["start"]] = X
    return M


def _fcond(D, X, w):
    sh = X.shape[1:]
    M = np.full((D.fcond.size, len(D.starts)) + sh, np.nan)
    M[D.fparent_fc, D.fparent_start] = X
    return _nanwmean(np.moveaxis(M, 1, -1), w, -1)


def _fby_start(D, X):
    M = np.full((D.fcond.size, len(D.starts)), np.nan)
    M[D.fparent_fc, D.fparent_start] = X
    return M[:, D.fstarts]


def yard_tables(D, prof, w):
    """Rule v3.1's condition tables from yard_profile() at start weights w (free runs: equal weight per dt value)."""
    out = {}
    for s, res in prof.items():
        g = res["seg"]
        t = {"seg": dict(cond=_cond(D, g["pscore"], w), bin=_cond(D, g["bm"], w), anchor=_cond(D, g["anchor"], w),
                         dec=_cond(D, g["pdec"], w), anchor_dec=_cond(D, g["adec"], w), start=_by_start(D, g["pscore"]))}
        if "free" in res:
            f = res["free"]
            ps = _nanmean(f["cs"], 1)
            t["free"] = dict(cond=_fcond(D, ps, w), bin=_fcond(D, f["cs"], w), dec=_fcond(D, _nanmean(f["Vd"], 1), w),
                             start=_fby_start(D, ps))
        out[s] = t
    return out


def yard_compare(D, PA, PB, ffA, ffB):
    """Rule v3.1's paired segment and free-run tables (12-gas mean with bins, decades, starts, anchor, cells, strata;
    worst gas D, A, B; failure rates), conditions resampled within cells."""
    res = {}
    tabs = [("seg", D.cond_cell)] + ([("free", D.fcond_cell)] if D.nfree else [])
    for tab, cells in tabs:
        t = {}
        for s in PA:
            A, Bp = PA[s][tab], PB[s][tab]
            d = A["cond"] - Bp["cond"]
            e = dict(D=summarize(d, cells), A=summarize(A["cond"], cells), B=summarize(Bp["cond"], cells))
            if s == "mean12":
                Db = A["bin"] - Bp["bin"]
                e["band"] = band(Db, cells)
                e["bins"] = [dict(D=summarize(Db[:, j], cells), A=summarize(A["bin"][:, j], cells),
                                  B=summarize(Bp["bin"][:, j], cells)) for j in range(Db.shape[1])]
                e["dec"] = [dict(D=summarize(A["dec"][:, j] - Bp["dec"][:, j], cells),
                                 A=summarize(A["dec"][:, j], cells), B=summarize(Bp["dec"][:, j], cells)) for j in range(8)]
                e["start"] = [dict(D=summarize(A["start"][:, j] - Bp["start"][:, j], cells),
                                   A=summarize(A["start"][:, j], cells), B=summarize(Bp["start"][:, j], cells))
                              for j in range(A["start"].shape[1])]
                if tab == "seg":
                    e["anchor"] = dict(D=summarize(A["anchor"] - Bp["anchor"], cells),
                                       A=summarize(A["anchor"], cells), B=summarize(Bp["anchor"], cells))
                    e["anchor_dec"] = [dict(D=summarize(A["anchor_dec"][:, j] - Bp["anchor_dec"][:, j], cells))
                                       for j in range(8)]
                    e["cell"] = []
                    for c in range(len(D.cells)):
                        m = D.cond_cell == c
                        e["cell"].append(dict(D=summarize(d[m], D.cond_cell[m]), A=summarize(A["cond"][m], D.cond_cell[m]),
                                              B=summarize(Bp["cond"][m], D.cond_cell[m])))
                    e["stratum"] = {}
                    for k in sorted(set(D.cond_stratum.tolist()), key="GCE".find):
                        m = D.cond_stratum == k
                        e["stratum"][k] = dict(D=summarize(d[m], D.cond_cell[m]), A=summarize(A["cond"][m], D.cond_cell[m]),
                                               B=summarize(Bp["cond"][m], D.cond_cell[m]))
                else:
                    e["cell_meanD"] = [float(np.mean(d[D.fcond_cell == c])) for c in range(len(D.cells))
                                       if (D.fcond_cell == c).any()]
                    st = D.cond_stratum[D.fcond]
                    e["stratum"] = {k: dict(D=summarize(d[st == k], D.fcond_cell[st == k]))
                                    for k in sorted(set(st.tolist()), key="GCE".find)}
            t[s] = e
        rows = D.seg_row if tab == "seg" else ~D.seg_row
        t["fail_A"], t["fail_B"] = fail_rates(D, ffA, rows), fail_rates(D, ffB, rows)
        res[tab] = t
    return res


def yard_score1(D, PA, PB, delta):
    """Score-1 tables of a yardstick-format design (readout (g)): segments at start weights eqt 0.55 / eqf 0.45; the
    raw starts (rawA, rawB) as their own table; free runs at 500 s and 10 s weighted by bin (eqt parents: the
    stationarity check; rawA parents: their own table). Conditions resampled within cells; outcome at delta. Cells
    without B values (no mini-chem file) drop out."""
    W1, WR = D.weights({"eqt": 0.55, "eqf": 0.45}), D.weights(RAW_W)
    out = {}
    a, b = PA["mean12"]["seg"], PB["mean12"]["seg"]
    for nm, w in (("seg", W1), ("seg_raw", WR)):
        A, B = _cond(D, a["pscore"], w), _cond(D, b["pscore"], w)
        t = dict(D=summarize(A - B, D.cond_cell), A=summarize(A, D.cond_cell), B=summarize(B, D.cond_cell))
        Ab, Bb = _cond(D, a["bm"], w), _cond(D, b["bm"], w)
        t["bins"] = [dict(D=summarize(Ab[:, j] - Bb[:, j], D.cond_cell)) for j in range(8)]
        t["anchor10"] = dict(D=summarize(_cond(D, a["anchor"], w) - _cond(D, b["anchor"], w), D.cond_cell))
        out[nm] = t
    if D.nfree:
        keep = [i for i, d in enumerate(D.free_dt) if d in (500.0, 10.0)]
        wb = D.bin_w[[FREE_BIN[D.free_dt[i]] for i in keep]]
        fa, fb = PA["mean12"]["free"]["cs"][:, keep], PB["mean12"]["free"]["cs"][:, keep]
        for nm, w in (("free_stationarity", W1), ("free_raw", WR)):
            A, B = _fcond(D, _nanwmean(fa, wb, 1), w), _fcond(D, _nanwmean(fb, wb, 1), w)
            out[nm] = dict(D=summarize(A - B, D.fcond_cell), A=summarize(A, D.fcond_cell), B=summarize(B, D.fcond_cell),
                           dt_values=[D.free_dt[i] for i in keep])
    for t in out.values():
        s = t["D"]
        t["outcome"] = outcome(s["trim"], s["trim_lo"], s["trim_hi"], delta) if s.get("n") else None
    return out


# ------------------------------------------------------------------------------------------------ score-1 designs
class Panel:
    """One score-1 design (bench/score_design.py): manifest-driven shapes, start weights, bins, bootstrap groups,
    in-domain flags and tie margin; every listed file's sha256 is checked against the manifest."""
    kind = "panel"

    def __init__(self, ddir):
        ddir = Path(ddir)
        self.dir = ddir
        man = json.loads((ddir / "manifest.json").read_text())
        assert man.get("format") == FORMAT, f"{ddir}: not a {FORMAT}"
        for f, h in man["files"].items():
            got = sha256(ddir / f)
            assert got == h, f"{ddir / f}: sha256 {got[:16]} differs from the manifest's {h[:16]}"
        self.man, self.sha = man, sha256(ddir / "manifest.json")
        self.name, self.mode, self.panel, self.role = man["name"], man["mode"], man.get("panel"), man["role"]
        c, p, r = (np.load(ddir / f) for f in ("conditions.npz", "parents.npz", "rows.npz"))
        self.c = {k: c[k] for k in c.files}
        self.p = {k: p[k] for k in p.files}
        self.r = {k: r[k] for k in r.files}
        s = np.load(ddir / "saved.npz")
        self.k, self.tabs, self.tel, self.row = s["k"].astype(np.int64), s["tabs"], s["tel"], s["row"].astype(np.int64)
        t = np.load(ddir / "truth_saved.npz")
        assert [str(g) for g in t["gases12"]] == GASES
        self.logt = t["logx12"]
        self.C, self.P, self.R, self.S = self.c["key"].size, self.p["cond"].size, self.r["N"].size, self.k.size
        off = self.r["off"]
        assert off[-1] == self.S and np.array_equal(self.row, np.repeat(np.arange(self.R), np.diff(off)))
        assert (self.tabs > 0).all() and (self.tel > 0).all()
        self.starts = list(man["start_codes"])
        self.W = np.array([man["start_weights"].get(s, 0.0) for s in self.starts], float)
        self.bin_w, self.bins = np.array(man["dt_bin_weights"], float), np.array(man["dt_bins"], float)
        self.nslot, self.m = int(man["slots"]), int(man["slots_per_bin"])
        self.delta = float(man["delta"])
        self.pcond, self.pstart = self.p["cond"].astype(np.int64), self.p["start"].astype(np.int64)
        assert np.unique(self.pcond * len(self.starts) + self.pstart).size == self.P
        self.rpar = self.r["parent"].astype(np.int64)
        self.case_row = self.rpar * self.nslot + self.r["dt_idx"].astype(np.int64)
        assert np.unique(self.case_row).size == self.case_row.size or self.mode == "seg"
        self.group = self.c["group"].astype(np.int64)
        self.scored = self.c["in_domain"].astype(bool)
        if self.mode == "path":
            assert np.array_equal(np.sort(self.rpar), np.arange(self.P)), "path designs: one row per parent"
            self.pbin = np.full(self.P, -1)
            self.pbin[self.rpar] = self.r["dt_bin"]
        log(f"design {self.name} ({self.mode}, {self.role}{', panel ' + self.panel if self.panel else ''}) loaded: "
            f"{self.C} conditions ({int(self.scored.sum())} in domain), {self.P} parents, {self.R} rows, {self.S} saved points")

    def cond(self, X, w=None):
        """Parent array (P, ...) -> condition array (C, ...), start weights w (default the manifest's) renormalised over
        the finite parents."""
        w = self.W if w is None else w
        M = np.full((self.C, len(self.starts)) + X.shape[1:], np.nan)
        M[self.pcond, self.pstart] = X
        return _nanwmean(np.moveaxis(M, 1, -1), w, -1)

    def by_start(self, X):
        M = np.full((self.C, len(self.starts)) + X.shape[1:], np.nan)
        M[self.pcond, self.pstart] = X
        return M

    def wvec(self, w):
        return np.array([w.get(s, 0.0) for s in self.starts], float)


def panel_parents(Pn, e, rowmask=None, pc=None):
    """Per-parent arrays of a per-call statistic e: pscore [P], bm [P, 8] (bin scores; a path parent fills its own bin),
    pdec [P, 8], readout slots ro [P, nro] and rod [P, nro, 8] (segment designs)."""
    ncase = Pn.P * Pn.nslot
    V, case = case_means(*exact_sums(Pn, e, rowmask, None, ncase, pc), ncase)
    Vd = V.reshape(Pn.P, Pn.nslot, 8)
    cs = case.reshape(Pn.P, Pn.nslot)
    if Pn.mode == "path":
        bm = np.full((Pn.P, 8), np.nan)
        bm[np.arange(Pn.P), Pn.pbin] = cs[:, 0]
        return dict(pscore=cs[:, 0], bm=bm, pdec=Vd[:, 0, :], ro=cs[:, 1:], rod=Vd[:, 1:, :])
    m8 = 8 * Pn.m
    bm = _nanmean(cs[:, :m8].reshape(Pn.P, 8, Pn.m), 2)
    dbm = _nanmean(Vd[:, :m8, :].reshape(Pn.P, 8, Pn.m, 8), 2)
    return dict(pscore=_nanwmean(bm, Pn.bin_w, 1), bm=bm, pdec=_nanwmean(np.moveaxis(dbm, 1, 2), Pn.bin_w, 2),
                ro=cs[:, m8:], rod=Vd[:, m8:, :])


def panel_profile(Pn, st, ff, stats=STATS, pen=PEN, w=None):
    """Condition tables of one solver on one design: cond [C], bin [C, 8], dec [C, 8], start [C, nstart] (parent alone),
    r10 [C] and r10_dec [C, 8] (the first readout slot, if any), and the per-parent arrays."""
    fp = failed_points(Pn, ff)
    out = {}
    for s in stats:
        pp = panel_parents(Pn, np.where(fp, pen, st[s]))
        t = dict(par=pp, cond=Pn.cond(pp["pscore"], w), bin=Pn.cond(pp["bm"], w), dec=Pn.cond(pp["pdec"], w),
                 start=Pn.by_start(pp["pscore"]))
        if pp["ro"].shape[1]:
            t["r10"], t["r10_dec"] = Pn.cond(pp["ro"][:, 0], w), Pn.cond(pp["rod"][:, 0, :], w)
        out[s] = t
    return out


def lasting(Pn, st, ff, pen=PEN):
    """Readout (a): per row from t = 0 the log-time mean of the 12-gas statistic over [t_end / 10, t_end] (t_end = its
    last saved time), combined to conditions as the score (bins weighted in long designs, the row in path designs)."""
    e = np.where(failed_points(Pn, ff), pen, st["mean12"])
    last = np.log10(Pn.tabs[Pn.r["off"][1:] - 1])
    rm = Pn.r["t0"] == 0
    v = window_means(Pn, e, last - 1.0, last, rm)
    X = np.full((Pn.P, Pn.nslot), np.nan)
    X[Pn.rpar[rm], Pn.r["dt_idx"][rm]] = v[rm]
    if Pn.mode == "path":
        pb = np.full((Pn.P, 8), np.nan)
        pb[np.arange(Pn.P), Pn.pbin] = X[:, 0]
        return Pn.cond(X[:, 0]), Pn.cond(pb)
    bm = _nanmean(X[:, :8 * Pn.m].reshape(Pn.P, 8, Pn.m), 2)
    return Pn.cond(_nanwmean(bm, Pn.bin_w, 1)), Pn.cond(bm)


def late_window(Pn, st, ff, pen=PEN):
    """Readout (c): every t = 0 row's late window [t_end / 10, t_end]; per (parent, bin) -> condition [C, 8] and the
    bin-weighted condition value [C]."""
    e = np.where(failed_points(Pn, ff), pen, st["mean12"])
    last = np.log10(Pn.tabs[Pn.r["off"][1:] - 1])
    rm = Pn.r["t0"] == 0
    v = window_means(Pn, e, last - 1.0, last, rm)
    X = np.full((Pn.P, Pn.nslot), np.nan)
    X[Pn.rpar[rm], Pn.r["dt_idx"][rm]] = v[rm]
    bm = _nanmean(X[:, :8 * Pn.m].reshape(Pn.P, 8, Pn.m), 2)
    return Pn.cond(_nanwmean(bm, Pn.bin_w, 1)), Pn.cond(bm)


def since_restart(Pn, st, ff, rm, pen=PEN):
    """Readout (i): per condition, the 12-gas statistic per decade of the time since the row's own start (rows rm),
    bins weighted over the bins that cover the decade (readout slots left out)."""
    e = np.where(failed_points(Pn, ff), pen, st["mean12"])
    ncase = Pn.P * Pn.nslot
    V, _ = case_means(*exact_sums(Pn, e, rm, None, ncase, pieces_elapsed(Pn)), ncase)
    m8 = 8 * Pn.m
    dbm = _nanmean(V.reshape(Pn.P, Pn.nslot, 8)[:, :m8, :].reshape(Pn.P, 8, Pn.m, 8), 2)
    return Pn.cond(_nanwmean(np.moveaxis(dbm, 1, 2), Pn.bin_w, 2))


def snapback(Pn, st, ff, pen=PEN):
    """Readout (j) per kicked row of a T2 design: log-time mean of the 12-gas statistic over tau in [1e3 s, t_end]
    minus the same for its identity partner."""
    pr = np.load(Pn.dir / "pairs.npz")
    e = np.where(failed_points(Pn, ff), pen, st["mean12"])
    last = np.log10(Pn.tabs[Pn.r["off"][1:] - 1])
    v = window_means(Pn, e, np.full(Pn.R, 3.0), last)
    return v[pr["kicked"]] - v[pr["partner"]], {k: pr[k] for k in pr.files}


def coverage(Pn):
    """Per (bin, decade): share of the cases in that bin whose decade has covered log-time (exact quadrature)."""
    ncase = Pn.P * Pn.nslot
    sw = exact_sums(Pn, np.zeros(Pn.S), None, None, ncase)[0].reshape(Pn.P, Pn.nslot, 8)
    has = sw > 0
    exists = np.zeros((Pn.P, Pn.nslot), bool)
    exists[Pn.rpar, Pn.r["dt_idx"]] = True
    out = []
    for b in range(8):
        if Pn.mode == "path":
            sel = (Pn.pbin == b)[:, None] & exists
        else:
            sel = np.zeros((Pn.P, Pn.nslot), bool)
            sel[:, b * Pn.m:(b + 1) * Pn.m] = True
            sel &= exists
        h = has[sel]
        out.append([float(h[:, d].mean()) if h.size else None for d in range(8)])
    return out


def large_misses(Pn, st, ff, solver, out_dir):
    """Rows with a non-failed saved call above 0.5 / 2 dex (12-gas mean; any single gas); identities to
    large_misses_<solver>_<design>.csv; counts by start, bin, stratum and gas. Returns the summary and the row masks."""
    fp = failed_points(Pn, ff)
    m12 = np.where(fp, -1.0, st["mean12"])
    wst = np.where(fp, -1.0, st["worst"])
    off = Pn.r["off"]
    rmax12 = np.maximum.reduceat(m12, off[:-1])
    rmaxw = np.maximum.reduceat(wst, off[:-1])
    out = {}
    for thr in (0.5, 2.0):
        out[f"rows_mean12_gt{thr}"] = int((rmax12 > thr).sum())
        out[f"rows_anygas_gt{thr}"] = int((rmaxw > thr).sum())
    par = Pn.rpar
    cond = Pn.pcond[par]
    cols = ["panel", "row", "condition", "stratum", "in_domain", "T_K", "P_bar", "Z", "C_O", "He_H", "start", "dt_s",
            "dt_bin", "t0_s", "N", "call", "t_s", "gas", "gas_err_dex", "max_mean12_dex"]
    lines = [",".join(cols)]
    for r in np.flatnonzero(rmaxw > 0.5):
        a, b = off[r], off[r + 1]
        jw = a + int(np.argmax(wst[a:b]))
        c = cond[r]
        vals = [Pn.name, r, Pn.c["key"][c], Pn.c["stratum"][c], bool(Pn.scored[c]), Pn.c["T"][c], Pn.c["P_bar"][c],
                Pn.c["Z"][c], Pn.c["C_O"][c], Pn.c["He_H"][c], Pn.starts[Pn.pstart[par[r]]], Pn.r["dt"][r],
                Pn.r["dt_bin"][r], Pn.r["t0"][r], Pn.r["N"][r], Pn.k[jw], Pn.tabs[jw], GASES[int(st["wgas"][jw])],
                float(wst[jw]), float(rmax12[r])]
        lines.append(",".join(str(v) for v in vals))
    (out_dir / f"large_misses_{solver}_{Pn.name}.csv").write_text("\n".join(lines) + "\n")
    for nm, sel in (("anygas_gt2", rmaxw > 2.0), ("mean12_gt0.5", rmax12 > 0.5)):
        bd = {"start": {s: int((sel & (Pn.pstart[par] == i)).sum()) for i, s in enumerate(Pn.starts)},
              "bin": {str(b): int((sel & (Pn.r["dt_bin"] == b)).sum()) for b in range(-1, 8)},
              "stratum": {s: int((sel & (Pn.c["stratum"][cond] == s)).sum()) for s in np.unique(Pn.c["stratum"])}}
        if nm == "anygas_gt2" and sel.any():
            g = np.array([st["wgas"][off[r] + int(np.argmax(wst[off[r]:off[r + 1]]))] for r in np.flatnonzero(sel)])
            bd["gas"] = {GASES[i]: int((g == i).sum()) for i in np.unique(g)}
        out["breakdown_" + nm] = bd
    return out, rmax12, rmaxw


def fail_detail(Pn, ff):
    """Failures (section 4) of one solver on one design: totals, by start and by bin."""
    o = {"all": fail_rates(Pn, ff, np.ones(Pn.R, bool))}
    st = Pn.pstart[Pn.rpar]
    o["by_start"] = {s: fail_rates(Pn, ff, st == i) for i, s in enumerate(Pn.starts) if (st == i).any()}
    o["by_bin"] = {str(b): fail_rates(Pn, ff, Pn.r["dt_bin"] == b) for b in range(-1, 8) if (Pn.r["dt_bin"] == b).any()}
    return o


# ------------------------------------------------------------------------------------------------ loading a set
def load_set(path):
    """A design (score-1 or yardstick format) or a folder of score-1 designs -> list of design objects."""
    path = Path(path)
    if (path / "manifest.json").exists():
        return [Panel(path)]
    if (path / "rows.npz").exists() and (path / "parents.npz").exists():
        return [Yard(path)]
    subs = sorted(d for d in path.iterdir() if (d / "manifest.json").exists())
    assert subs, f"{path}: no design found"
    return [Panel(d) for d in subs]


def model_states(D, run, name):
    """Point statistics of one solver on one design; refuses states whose recorded design hash differs."""
    if D.kind == "yard":
        if name == "minichem":
            logp, ffrow, have = load_minichem(D, run / "minichem")
            st = point_stats(D, logp, ffrow)
            st["have"] = have
            return st
        d = run / name
        meta = json.loads((d / "meta.json").read_text())
        for f, k in (("rows.npz", "rows_npz_sha256"), ("saved.npz", "saved_npz_sha256")):
            assert meta[k] == sha256(D.dir / f), f"{d}: states made on a different {f}; refused"
        logp, ff, unrec, extra = load_model(d / "model_states.npz")
    else:
        assert name != "minichem", f"{D.name}: no mini-chem files in a score-1 design"
        d = run / name / D.name
        meta = json.loads((d / "meta.json").read_text())
        assert meta["design_sha256"] == D.sha, f"{d}: states made on design {meta['design_sha256'][:16]}, not {D.sha[:16]}; refused"
        logp, ff, unrec, extra = load_model(d / "states.npz")
    st = point_stats(D, logp, ff)
    st["ff_literal"] = st["ff"]
    st["ff"] = merge_bad(D, unrec, st["bad"])                       # unrecovered steps (and invalid saved states) only
    st["extra"] = extra
    return st


# ------------------------------------------------------------------------------------------------ the score of a set
def by_condition_subsets(Pn):
    """Readout (i) condition subsets of a design: name -> boolean mask over conditions (the scored ones decide)."""
    c = Pn.c
    out = {}
    for s in np.unique(c["stratum"]):
        out[f"stratum {s}"] = c["stratum"] == s
    for lo, hi in TBANDS:
        out[f"T {lo}-{hi} K"] = (c["T"] >= lo) & (c["T"] < hi)
    lp = np.log10(c["P_bar"])
    for d in range(-5, 2):
        out[f"P 1e{d}-1e{d + 1} bar"] = (lp >= d) & (lp < d + 1)
    for nm in ("Z", "C_O"):
        v = c[nm]
        if np.isfinite(v).sum() >= 3:
            q1, q2 = np.nanquantile(v[Pn.scored], [1 / 3, 2 / 3])
            out[f"{nm} tercile 1"], out[f"{nm} tercile 2"], out[f"{nm} tercile 3"] = v <= q1, (v > q1) & (v <= q2), v > q2
    v = c["He_H"]
    if np.isfinite(v).sum() >= 2:
        med = np.nanmedian(v[Pn.scored])
        out["He/H lower half"], out["He/H upper half"] = v <= med, v > med
    if Pn.mode == "path":
        out["fixed dt"], out["variable dt"] = ~c["variable_dt"], c["variable_dt"]
        out["unmixed"], out["mixed"] = ~c["mixed"], c["mixed"]
    return out


def score_set(designs, run, models, out_dir):
    """Score every design of a set for every model and compare the first with each other."""
    out_dir.mkdir(parents=True, exist_ok=True)
    yards = [D for D in designs if D.kind == "yard"]
    panels = [D for D in designs if D.kind == "panel"]
    res = dict(score="score-1", generated=time.strftime("%F %T"), models=list(models), designs={}, absolute={},
               comparisons={}, failures={}, large_misses={}, readouts_model={})
    lines = []
    if yards:
        for D in yards:
            r, L = score_yard(D, run, models, out_dir)
            res["designs"][D.name] = r
            lines += L
    if panels:
        r, L = score_panels(panels, run, models, out_dir)
        for k, v in r.items():
            res[k] = v if not isinstance(v, dict) else {**res.get(k, {}), **v}
        lines += L
    return res, lines


def score_yard(D, run, models, out_dir):
    """A yardstick-format design: continuity tables under its stored start weights (rule v3.1) and score-1 tables vs
    each other solver (readout (g) for mini-chem)."""
    a0 = models[0]
    sts = {m: model_states(D, run, m) for m in models}
    prof = {m: yard_profile(D, sts[m], sts[m]["ff"]) for m in models}
    for m in models:
        log(f"{D.name}: {m} profile done")
    W1 = D.weights({"eqt": 0.55, "eqf": 0.45})
    delta = DELTA
    out = dict(format="stored yardstick format", rows=int(D.R), points=int(D.S), stored_start_weights=D.pw.tolist(),
               score1_start_weights=W1.tolist(), delta=delta, continuity_v31={}, score1={}, failures={}, large_misses={})
    L = [f"=== {D.name} (stored yardstick format; conditions resampled within cells) ==="]
    for m in models:
        st = sts[m]
        out["failures"][m] = {"seg": fail_rates(D, st["ff"], D.seg_row)}
        if D.nfree:
            out["failures"][m]["free"] = fail_rates(D, st["ff"], ~D.seg_row)
    for b in models[1:]:
        key = f"{a0}_vs_{b}"
        tA, tB = yard_tables(D, prof[a0], D.pw), yard_tables(D, prof[b], D.pw)
        out["continuity_v31"][key] = yard_compare(D, tA, tB, sts[a0]["ff"], sts[b]["ff"])
        out["score1"][key] = yard_score1(D, prof[a0], prof[b], delta)
        c = out["continuity_v31"][key]["seg"]["mean12"]["D"]
        L.append(f"  {a0} vs {b}: continuity (stored weights {D.pw.tolist()}) segments D {fs(c)}"
                 + (f"; free runs D {fs(out['continuity_v31'][key]['free']['mean12']['D'])}" if D.nfree else ""))
        for nm, t in out["score1"][key].items():
            L.append(f"    score-1 {nm:18s} D {fs(t['D'])}  A {fa(t['A'])}  B {fa(t['B'])}  -> {t['outcome']}")
    return out, L


def score_panels(panels, run, models, out_dir):
    """Score-1 designs: absolute scores, comparisons (the first model vs each other), readouts and flags."""
    order = {"S": 0, "L": 1, "P": 2}
    score_p = sorted([D for D in panels if D.role == "score"], key=lambda D: order[D.panel])
    full = [D.panel for D in score_p] == ["S", "L", "P"]
    tables = score_p if score_p else panels
    a0 = models[0]
    delta = tables[0].delta
    assert all(D.delta == delta for D in panels), "designs with different tie margins"
    idx = {D.name: strat_idx(D.group[D.scored], i) for i, D in enumerate(tables)}
    sts, prof = {}, {}
    for D in panels:
        for m in models:
            sts[(m, D.name)] = model_states(D, run, m)
            prof[(m, D.name)] = panel_profile(D, sts[(m, D.name)], sts[(m, D.name)]["ff"])
        log(f"{D.name}: profiles done")
    out = dict(set_kind="decision set (S, L, P)" if full else "tables (one per design)", delta=delta,
               retrain_spread=tables[0].man.get("retrain_spread"),
               designs={D.name: dict(mode=D.mode, panel=D.panel, role=D.role, manifest_sha256=D.sha,
                                     conditions=int(D.C), in_domain=int(D.scored.sum()), rows=int(D.R)) for D in panels},
               absolute={}, comparisons={}, failures={}, large_misses={}, readouts_model={})
    L = [f"=== score-1 ({out['set_kind']}; delta {delta} dex; retrain-to-retrain spread: {out['retrain_spread']}) ==="]
    rm = {}
    for m in models:
        out["failures"][m] = {D.name: fail_detail(D, sts[(m, D.name)]["ff"]) for D in panels}
        out["large_misses"][m] = {}
        for D in panels:
            lm, r12, rw = large_misses(D, sts[(m, D.name)], sts[(m, D.name)]["ff"], m, out_dir)
            out["large_misses"][m][D.name] = lm
            rm[(m, D.name)] = (r12, rw)
        j = joint([prof[(m, D.name)]["mean12"]["cond"][D.scored] for D in tables], [idx[D.name] for D in tables])
        a = jsum(j)
        ab = {}
        for D in panels:
            ab[D.name] = {s: summarize(prof[(m, D.name)][s]["cond"][D.scored], D.group[D.scored]) for s in STATS}
        out["absolute"][m] = dict(SCORE=a["D"] if full else None, SCORE_lo=a["D_lo"] if full else None,
                                  SCORE_hi=a["D_hi"] if full else None, panels=ab)
        out["readouts_model"][m] = model_readouts(run, m)
        L.append(f"{m}: SCORE " + (f"{a['D']:.4f} [{a['D_lo']:.4f}, {a['D_hi']:.4f}]" if full else "n/a (not a full set)")
                 + "; " + "; ".join(f"{D.name} {ab[D.name]['mean12'].get('trim', float('nan')):.4f}" for D in panels))
    for b in models[1:]:
        key = f"{a0}_vs_{b}"
        c = compare_panels(panels, tables, full, idx, sts, prof, a0, b, delta, rm, run)
        out["comparisons"][key] = c
        L += comparison_lines(c, a0, b)
    return out, L


def model_readouts(run, m):
    """Readouts (e) and (f) of one model when their files exist in its run folder."""
    o = {}
    f = run / m / "cyc1000.json"
    if f.exists():
        z = json.loads(f.read_text())
        kept = sum(int(r[1]) for r in z["projfail"]) if z.get("projfail") else 0
        o["cyc1000"] = dict(finite=z["finite"], rejected=int(sum(z["rejected"])), kept_input=kept,
                            last5_max=float(max(z["cycle_change_last5_max"])), cycles=z["cycles"])
        o["cyc1000"]["R6"] = bool((not z["finite"]) or o["cyc1000"]["rejected"] > 0 or kept > 0
                                  or o["cyc1000"]["last5_max"] > 0.1)
    for fl in ("keep", "lm"):
        f = run / m / f"speed_{fl}.json"
        if f.exists():
            z = json.loads(f.read_text())
            g = z["gpu"].get("100000") or z["gpu"].get(100000)
            o[f"speed_{fl}_us_per_cell_1e5"] = g["us_per_cell"] if g else None
    return o


def compare_panels(panels, tables, full, idx, sts, prof, a, b, delta, rm, run):
    """The paired comparison of a with b on score-1 designs: D, D_mean, outcome, readouts (a)-(d), (h)-(j), flags."""
    P = {D.name: D for D in panels}
    pr = lambda m, D, s="mean12": prof[(m, D.name)][s]
    ds = [pr(a, D)["cond"][D.scored] - pr(b, D)["cond"][D.scored] for D in tables]
    ix = [idx[D.name] for D in tables]
    j = joint(ds, ix)
    c = dict(a=a, b=b, tables=[D.name for D in tables], **jsum(j))
    c["outcome"] = outcome(c["D"], c["D_lo"], c["D_hi"], delta)
    c["what"] = ("the one number: D = (T_S + T_L + T_P) / 3" if full else
                 "D over the designs present (not the decision set's one number)")
    c["per_design"] = {}
    for D in panels:
        g = D.group[D.scored]
        t = {s: dict(D=summarize(pr(a, D, s)["cond"][D.scored] - pr(b, D, s)["cond"][D.scored], g),
                     A=summarize(pr(a, D, s)["cond"][D.scored], g), B=summarize(pr(b, D, s)["cond"][D.scored], g))
             for s in STATS}
        t["out_of_domain_conditions"] = int((~D.scored).sum())
        c["per_design"][D.name] = t
    rd = {}
    # (a) lasting
    lt = [D for D in tables if D.mode in ("long", "path")]
    if lt:
        la = {D.name: (lasting(D, sts[(a, D.name)], sts[(a, D.name)]["ff"]), lasting(D, sts[(b, D.name)], sts[(b, D.name)]["ff"]))
              for D in lt}
        jl = joint([la[D.name][0][0][D.scored] - la[D.name][1][0][D.scored] for D in lt], [idx[D.name] for D in lt])
        rd["a_lasting"] = dict(tables=[D.name for D in lt], **jsum(jl))
        jb = joint([la[D.name][0][1][D.scored] - la[D.name][1][1][D.scored] for D in lt], [idx[D.name] for D in lt])
        rd["a_lasting"]["per_bin"] = [jsum(jb, k) for k in range(8)]
        if "L" in [D.panel for D in lt]:
            DL = [D for D in lt if D.panel == "L"][0]
            v = la[DL.name][0][1][DL.scored, 7] - la[DL.name][1][1][DL.scored, 7]
            rd["a_lasting"]["L_200_500s_1000_days"] = summarize(v, DL.group[DL.scored])
    # (b) per bin, inside the joint bootstrap, with the simultaneous band
    jb = joint([pr(a, D)["bin"][D.scored] - pr(b, D)["bin"][D.scored] for D in tables], ix)
    rd["b_bins"] = [jsum(jb, k) for k in range(8)]
    T, Tb = jb["D"], jb["Db"]
    with warnings.catch_warnings():                                # a bin no table covers is NaN throughout
        warnings.simplefilter("ignore", RuntimeWarning)
        se = np.nanstd(Tb, 0, ddof=1)
        z = np.abs(Tb - T) / np.where(se > 0, se, np.inf)
        cc = float(np.nanpercentile(np.nanmax(z, 1), 95))
    rd["b_band"] = dict(T=T.tolist(), se=se.tolist(), c=cc, lo=(T - cc * se).tolist(), hi=(T + cc * se).tolist())
    ja = joint([pr(a, D)["bin"][D.scored] for D in tables], ix)
    jbb = joint([pr(b, D)["bin"][D.scored] for D in tables], ix)
    rd["b_absolute"] = dict(A=[float(x) for x in ja["D"]], B=[float(x) for x in jbb["D"]])
    S = [D for D in tables if D.panel == "S" or (D.mode == "seg" and "r10" in pr(a, D))]
    if S and "r10" in pr(a, S[0]):
        DS = S[0]
        g = DS.group[DS.scored]
        rd["b_10s"] = dict(D=summarize(pr(a, DS)["r10"][DS.scored] - pr(b, DS)["r10"][DS.scored], g),
                           per_decade=[summarize(pr(a, DS)["r10_dec"][DS.scored, k] - pr(b, DS)["r10_dec"][DS.scored, k], g)
                                       for k in range(8)])
    # (c) raw starts
    if "Sraw" in P:
        D = P["Sraw"]
        g = D.group[D.scored]
        wr = D.wvec(RAW_W)
        A_ = D.cond(pr(a, D)["par"]["pscore"], wr)
        B_ = D.cond(pr(b, D)["par"]["pscore"], wr)
        r = dict(D=summarize((A_ - B_)[D.scored], g), A=summarize(A_[D.scored], g), B=summarize(B_[D.scored], g))
        r["per_start"] = {s: summarize((pr(a, D)["start"][:, i] - pr(b, D)["start"][:, i])[D.scored], g)
                          for i, s in enumerate(D.starts) if (D.pstart == i).any()}
        lwa, lwb = late_window(D, sts[(a, D.name)], sts[(a, D.name)]["ff"]), late_window(D, sts[(b, D.name)], sts[(b, D.name)]["ff"])
        r["late_window_per_bin"] = [summarize((lwa[1][:, k] - lwb[1][:, k])[D.scored], g) for k in range(8)]
        hot = D.c["T"] >= 1300
        r["late_window_T_band"] = {nm: summarize((lwa[0] - lwb[0])[D.scored & msk], D.group[D.scored & msk])
                                   for nm, msk in (("T < 1300 K", ~hot), ("T >= 1300 K", hot))}
        rd["c_raw"] = r
    # (d) large misses, failures
    rd["d_large_misses"] = {}
    for D in panels:
        r12a, rwa = rm[(a, D.name)]
        r12b, rwb = rm[(b, D.name)]
        only = np.flatnonzero((rwa > 2.0) & ~(rwb > 2.0))
        rd["d_large_misses"][D.name] = dict(
            rows_anygas_gt2_A_not_B=int(only.size), rows_list=only[:50].tolist(),
            rows_mean12_gt0p5_A_not_B=int(((r12a > 0.5) & ~(r12b > 0.5)).sum()))
    fA = sum(fail_rates(D, sts[(a, D.name)]["ff"], np.ones(D.R, bool))["failed_rows"] for D in tables)
    fB = sum(fail_rates(D, sts[(b, D.name)]["ff"], np.ones(D.R, bool))["failed_rows"] for D in tables)
    rd["d_failed_rows"] = dict(A=fA, B=fB)
    # (h) continuity start weights
    rd["h_start_weights"] = {}
    for w in ALT_W:
        dd = [D.cond(pr(a, D)["par"]["pscore"], D.wvec(w))[D.scored] - D.cond(pr(b, D)["par"]["pscore"], D.wvec(w))[D.scored]
              for D in tables]
        rd["h_start_weights"][f"eqt {w['eqt']:.2f} / eqf {w['eqf']:.2f}"] = jsum(joint(dd, ix))
    # (i) diagnostics
    di = {}
    for D in tables:
        g = D.group
        A_, B_ = pr(a, D), pr(b, D)
        x = {"per_decade": [summarize((A_["dec"][:, k] - B_["dec"][:, k])[D.scored], g[D.scored]) for k in range(8)],
             "per_start": {s: summarize((A_["start"][:, i] - B_["start"][:, i])[D.scored], g[D.scored])
                           for i, s in enumerate(D.starts) if (D.pstart == i).any()},
             "subsets": {}, "coverage_bin_decade": coverage(D)}
        for nm, msk in by_condition_subsets(D).items():
            mm = msk & D.scored
            if mm.sum():
                x["subsets"][nm] = dict(n=int(mm.sum()), D=summarize((A_["cond"] - B_["cond"])[mm], g[mm]))
        if D.mode == "seg":
            ag = D.r["age_idx"]
            x["per_restart_age"] = []
            for k in range(int(ag.max()) + 1):
                rr = ag == k
                if not rr.any():
                    continue
                pa = panel_parents(D, np.where(failed_points(D, sts[(a, D.name)]["ff"]), PEN, sts[(a, D.name)]["mean12"]), rr)
                pb = panel_parents(D, np.where(failed_points(D, sts[(b, D.name)]["ff"]), PEN, sts[(b, D.name)]["mean12"]), rr)
                v = (D.cond(pa["pscore"]) - D.cond(pb["pscore"]))[D.scored]
                x["per_restart_age"].append(dict(t0=AGES[k] if k < len(AGES) else None, rows=int(rr.sum()),
                                                 D=summarize(v, g[D.scored])))
            x["since_restart"] = {}
            for nm, rr in (("restart rows (t0 > 0)", D.r["t0"] > 0), ("t0 = 0 rows", D.r["t0"] == 0)):
                sa = since_restart(D, sts[(a, D.name)], sts[(a, D.name)]["ff"], rr)
                sb = since_restart(D, sts[(b, D.name)], sts[(b, D.name)]["ff"], rr)
                x["since_restart"][nm] = [summarize((sa[:, k] - sb[:, k])[D.scored], g[D.scored]) for k in range(8)]
            if "eqf" in D.starts:
                i = D.starts.index("eqf")
                rr = D.r["t0"] > 0
                pa = panel_parents(D, np.where(failed_points(D, sts[(a, D.name)]["ff"]), PEN, sts[(a, D.name)]["mean12"]), rr)
                pb = panel_parents(D, np.where(failed_points(D, sts[(b, D.name)]["ff"]), PEN, sts[(b, D.name)]["mean12"]), rr)
                v = (D.by_start(pa["pscore"])[:, i] - D.by_start(pb["pscore"])[:, i])[D.scored]
                x["eqf_restart_rows_only"] = summarize(v, g[D.scored])
        di[D.name] = x
    if "side" in P:
        D = P["side"]
        g = D.group[D.scored]
        di["side_CO_1_2"] = dict(D=summarize((pr(a, D)["cond"] - pr(b, D)["cond"])[D.scored], g))
    for pen in PENS:
        dd = []
        for D in tables:
            pa = panel_parents(D, np.where(failed_points(D, sts[(a, D.name)]["ff"]), pen, sts[(a, D.name)]["mean12"]))
            pb = panel_parents(D, np.where(failed_points(D, sts[(b, D.name)]["ff"]), pen, sts[(b, D.name)]["mean12"]))
            dd.append((D.cond(pa["pscore"]) - D.cond(pb["pscore"]))[D.scored])
        di[f"penalty_{pen:g}"] = jsum(joint(dd, ix))
    rd["i_diagnostics"] = di
    # (j) snap-back
    t2 = [D for D in panels if (D.dir / "pairs.npz").exists()]
    if t2:
        D = t2[0]
        ea, prs = snapback(D, sts[(a, D.name)], sts[(a, D.name)]["ff"])
        eb, _ = snapback(D, sts[(b, D.name)], sts[(b, D.name)]["ff"])
        rd["j_snapback"] = snap_summary(ea, eb, prs)
    c["readouts"] = rd
    # flags
    fl = {}
    fl["R1"] = dict(fired=bool(np.sign(c["D_mean"]) != np.sign(c["D"]) or abs(c["D_mean"] - c["D"]) > delta),
                    D=c["D"], D_mean=c["D_mean"])
    if "a_lasting" in rd:
        x = rd["a_lasting"]
        fl["R2"] = dict(fired=bool(x["D"] >= delta and x["D_lo"] > 0), D=x["D"], lo=x["D_lo"])
    else:
        fl["R2"] = dict(fired=None, note="no long or path design in this set")
    weak = [k for k in range(8) if np.isfinite(T[k]) and T[k] >= delta and rd["b_band"]["lo"][k] > 0]
    r10 = rd.get("b_10s", {}).get("D", {})
    r10f = bool(r10.get("n") and r10["trim"] >= delta and r10["trim_lo"] > 0)
    fl["R3"] = dict(fired=bool(weak or r10f), weak_bins=weak, readout_10s=r10f)
    if "c_raw" in rd:
        x = rd["c_raw"]["D"]
        fl["R4"] = dict(fired=bool(x.get("n") and x["trim"] >= delta and x["trim_lo"] > 0), D=x.get("trim"), lo=x.get("trim_lo"))
    else:
        fl["R4"] = dict(fired=None, note="no Sraw design in this set")
    fl["R5"] = dict(fired=bool(fA > fB), failed_rows_A=fA, failed_rows_B=fB)
    cy = model_readouts(run, a).get("cyc1000")
    fl["R6"] = dict(fired=cy["R6"], **cy) if cy else dict(fired=None, note="no cyc1000.json for A (final pick only)")
    if "j_snapback" in rd:
        x = rd["j_snapback"]["D_snap"]
        fl["R7"] = dict(fired=bool(x["D"] >= delta and x["D_lo"] > 0), D=x["D"], lo=x["D_lo"])
    else:
        fl["R7"] = dict(fired=None, note="no T2 design in this set (final pick only)")
    c["flags"] = fl
    return c


def snap_summary(ea, eb, prs):
    """Readout (j): per class the share of kicked rows with excess <= 0.1 dex and the trimmed mean (each model), and
    D_snap = trimmed mean over the in-domain kicked rows of excess_A - excess_B, bootstrap by source job (whole source
    jobs resampled with replacement); the out-of-domain boxes are a separate line."""
    ind = prs["in_domain"].astype(bool)
    out = {"classes": {}}
    for cl in ("k1", "k2", "k3"):
        m = (prs["cls"] == cl) & ind
        out["classes"][cl] = dict(n=int(m.sum()), share_le_0p1_A=float((ea[m] <= 0.1).mean()) if m.any() else None,
                                  share_le_0p1_B=float((eb[m] <= 0.1).mean()) if m.any() else None,
                                  trim_A=float(tmean(ea[m])) if m.any() else None,
                                  trim_B=float(tmean(eb[m])) if m.any() else None)

    def clus(sel):
        d = (ea - eb)[sel]
        src = prs["src"][sel]
        us = np.unique(src)
        if us.size == 0:
            return dict(n=0)
        rows_of = [np.flatnonzero(src == s) for s in us]
        rng = np.random.default_rng([SEED, 7])
        Db = np.empty(NBOOT)
        for i in range(NBOOT):
            pick = rng.integers(0, us.size, us.size)
            Db[i] = tmean(d[np.concatenate([rows_of[p] for p in pick])])
        lo, hi = interval(Db)
        return dict(D=float(tmean(d)), D_lo=lo, D_hi=hi, n_rows=int(d.size), n_source_jobs=int(us.size))

    out["D_snap"] = clus(ind)
    out["out_of_domain"] = clus(~ind)
    return out


# ------------------------------------------------------------------------------------------------ text
def f3(x):
    return "   nan" if x is None or not np.isfinite(x) else f"{x:+.3f}" if abs(x) < 10 else f"{x:+.2f}"


def fs(s, k="trim"):
    if not s or s.get("n", 0) == 0:
        return "n/a"
    if s.get("degenerate"):
        return f"{f3(s[k])} [no interval: 1 condition per cell]"
    return f"{f3(s[k])} [{f3(s[k + '_lo'])}, {f3(s[k + '_hi'])}]"


def fa(s):
    if not s or s.get("n", 0) == 0:
        return "n/a"
    return f"{s['trim']:.3f}"


def fj(x):
    return f"{f3(x['D'])} [{f3(x['D_lo'])}, {f3(x['D_hi'])}]"


def comparison_lines(c, a, b):
    L = ["", f"--- {a} vs {b} (D = {a} - {b}, dex; negative = {a} better; {c['what']}) ---",
         f"OUTCOME: {c['outcome']}   D {fj(c)}   D_mean {f3(c['D_mean'])} [{f3(c['D_mean_lo'])}, {f3(c['D_mean_hi'])}]"]
    for nm, p in zip(c["tables"], c["panels"]):
        L.append(f"  {nm:6s} trimmed D {f3(p['trim'])} [{f3(p['trim_lo'])}, {f3(p['trim_hi'])}]  mean D {f3(p['mean'])}")
    for nm, t in c["per_design"].items():
        L.append(f"  {nm:6s} " + "  ".join(f"{s} D {fs(t[s]['D'])}" for s in ("mean12", "worst")) +
                 f"  A {fa(t['mean12']['A'])} B {fa(t['mean12']['B'])}" +
                 (f"  ({t['out_of_domain_conditions']} out-of-domain conditions left out)" if t["out_of_domain_conditions"] else ""))
    L.append("flags (read before adopting; never vetoes): " + "; ".join(
        f"{k} {'FIRED' if v['fired'] else ('n/a' if v['fired'] is None else 'quiet')}" for k, v in c["flags"].items()))
    rd = c["readouts"]
    if "a_lasting" in rd:
        L.append(f"(a) lasting D {fj(rd['a_lasting'])}" + (f"; L 200-500 s (1000 days) {fs(rd['a_lasting']['L_200_500s_1000_days'])}"
                                                           if "L_200_500s_1000_days" in rd["a_lasting"] else ""))
    L.append("(b) per bin D [95%] / simultaneous [lo, hi]: " + "; ".join(
        f"b{k} {f3(x['D'])} [{f3(x['D_lo'])}, {f3(x['D_hi'])}] / [{f3(rd['b_band']['lo'][k])}, {f3(rd['b_band']['hi'][k])}]"
        for k, x in enumerate(rd["b_bins"])))
    if "b_10s" in rd:
        L.append(f"    10 s rows of S: D {fs(rd['b_10s']['D'])}")
    if "c_raw" in rd:
        L.append(f"(c) raw starts D {fs(rd['c_raw']['D'])}; per start " +
                 ", ".join(f"{s} {fs(v)}" for s, v in rd["c_raw"]["per_start"].items()))
    L.append(f"(d) failed rows A {rd['d_failed_rows']['A']} / B {rd['d_failed_rows']['B']}; rows with a gas > 2 dex for A and "
             f"not B: " + ", ".join(f"{k} {v['rows_anygas_gt2_A_not_B']}" for k, v in rd["d_large_misses"].items()))
    L.append("(h) start weights: " + "; ".join(f"{k} D {fj(v)}" for k, v in rd["h_start_weights"].items()))
    if "j_snapback" in rd:
        x = rd["j_snapback"]
        L.append(f"(j) snap-back D_snap {fj(x['D_snap'])} ({x['D_snap'].get('n_rows')} in-domain kicked rows, "
                 f"{x['D_snap'].get('n_source_jobs')} source jobs); out of domain " +
                 (fj(x["out_of_domain"]) if x["out_of_domain"].get("n_rows") else "n/a"))
    return L


def write_json(path, obj):
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    Path(path).write_text(json.dumps(clean(obj), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", type=Path, required=True, help="design folder, folder of designs, or yardstick-format design")
    ap.add_argument("--run", type=Path, required=True, help="run folder (one folder per model; minichem/ for yardstick designs)")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("models", nargs="+", help="the first is compared with each other")
    a = ap.parse_args()
    assert len(a.models) >= 1 and len(set(a.models)) == len(a.models)
    assert not (a.out / "score.json").exists(), f"refusing to overwrite {a.out / 'score.json'}"
    t0 = time.time()
    designs = load_set(a.set)
    res, lines = score_set(designs, a.run, a.models, a.out)
    res["seconds"] = time.time() - t0
    (a.out / "tables.txt").write_text("\n".join([f"score-1, generated {res['generated']}; set {a.set}; models "
                                                 f"{', '.join(a.models)}"] + lines) + "\n")
    write_json(a.out / "score.json", res)
    log("written", a.out, f"{res['seconds']:.0f} s")


if __name__ == "__main__":
    main()
