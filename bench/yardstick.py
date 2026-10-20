"""Yardstick scorer: models and mini-chem against VULCAN on the yardstick design (histories at fixed T and P, run as
restarted segments and as free runs with constant steps), scored per condition and paired. NumPy only.

    python yardstick.py --design DIR --run DIR --out DIR [--rule v3.1|v3] MODEL [MODEL ...]

Inputs (read only): <design>/rows.npz, saved.npz, truth_saved.npz, parents.npz, units.npz; <run>/minichem/<cell>.npz,
rows_summary.npz, rtol1e-5.npz; <run>/<MODEL>/model_states.npz. Outputs in --out: score.json, tables.txt and
large_misses_<solver>.csv. Comparisons: every MODEL vs mini-chem, and the first MODEL vs each other MODEL.

Pipeline:
  per saved call: the 12 reported gases (log10), validated before flooring; per-gas floors S (1e-10 H2O CO CO2 CH4 NH3
  HCN C2H2; 1e-8 H OH O; 1e-6 H2 N2), error = |max(pred, S) - max(truth, S)|; statistics: 12-gas mean (primary), worst
  gas, 6-major mean (H2O CO CO2 CH4 NH3 N2), 5-trace mean (HCN C2H2 H OH O). A failed step penalizes the remainder of
  its row (segment or free run): from the row's first failed call on, every call scores a fixed penalty of 2 dex for
  every statistic (readouts at 0.5 and 5 dex), and the failed-call rate counts N - first + 1 calls for that row.
  per case (parent history x dt value): log10-time quadrature within each decade of absolute time (1-1e8 s), case =
  mean over covered decades. Segments: the 2 draws of a dt bin averaged, bins weighted by log length; the 10 s anchor
  is a readout. Free runs: equal weight per dt value. Condition = start-type weighted mean (eq-target 0.35, eq-far 0.25,
  raw-A 0.20, raw-B 0.20, renormalised over the starts present); D = A - B per condition.
  summaries: 5% symmetric trimmed mean, mean, median; bootstrap of conditions within cells (4000 resamples, fixed
  seed); studentized max-|z| simultaneous band over the dt bins.
  verdict per table: "beats" iff trimmed D upper < 0, mean D < 0, every bin's simultaneous upper < +0.05, worst-gas
  trimmed D upper < +0.05 and failed-call rate A <= B + 0.001; "loses" iff trimmed D lower > 0. Headline: "beats" iff
  both tables beat, "loses" iff one loses and neither beats.

Rules (--rule):
  v3.1  (default) a failure is an unrecovered step only. A model row fails at its first kept input (the projection
        failed after its LM rescue), guard rejection (candidate state non-finite or outside the envelope) or non-finite
        state; LM rescues are counted, not failures.
        Exact quadrature: the statistic, piecewise linear in log10 t between saved calls (held at the first call's
        value back to a restart's t0), is integrated over each decade with the pieces split at the decade edges; a
        decade with zero covered log-time is untested.
  v3    a model row also fails at its first LM rescue (projection fallback ran). Whole-node quadrature: each saved
        call's trapezoid weight goes to the decade of its own time (a call exactly at 10 or 100 s gives weight to a
        decade with no covered time).
  Mini-chem's failures are the same under both rules: its driver's first failed call (istate < -1, reached time short,
  non-finite or negative state) or a saved state whose sum differs from 1 by more than 1e-6; excess-work continuations
  and DLSODE warnings are counted, not failures.
  Rescue settings are those the scored states were made with (this module only reads them): model, 32 projection Newton
  iterations with the adaptive LM rescue where Newton fails, one model call per step; mini-chem, DLSODE at rtol 1e-3,
  atol 1e-30, with its driver's own restarts within a step (a fresh DLSODE start after every 50 returns, a
  continuation after an excess-work return).
  Limitation: mini-chem's driver floors (1e-30) and renormalises the state before it tests for a failure, so a failure
  that leaves a finite floored state is not detected; its failure count is a lower bound.

Readouts (never the verdict): penalties 0.5 and 5 dex; the other rule's model failure reading and quadrature (v3 also:
the scores without single calls at a decade's upper edge); per restart age; restart rows only and t0 = 0 rows only,
each through the five verdict checks; time since restart (per decade of the time since a row's own start, restart rows
and t0 = 0 rows); the representable-start subset; mini-chem's rtol 1e-5 subset; the first calls from t = 0; model inputs
at the 1e-30 floor; failures; large misses; coverage. The per-decade tables of the verdict tables are in absolute time
of a history (a decade pools the t0 = 0 row with the restarts made in it); the time-since-restart table is not."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

GASES = ["C2H2", "CH4", "CO2", "CO", "H2O", "H2", "HCN", "H", "N2", "NH3", "OH", "O"]       # design gases12 order
FLOOR = {"H2O": 1e-10, "CO": 1e-10, "CO2": 1e-10, "CH4": 1e-10, "NH3": 1e-10, "HCN": 1e-10, "C2H2": 1e-10,
         "H": 1e-8, "OH": 1e-8, "O": 1e-8, "H2": 1e-6, "N2": 1e-6}
LOGS = np.log10(np.array([FLOOR[g] for g in GASES]))
MAJOR6 = [GASES.index(g) for g in ("H2O", "CO", "CO2", "CH4", "NH3", "N2")]
TRACE5 = [GASES.index(g) for g in ("HCN", "C2H2", "H", "OH", "O")]
STATS = ("mean12", "worst", "major6", "trace5")
SEED, NBOOT, TRIM, PEN, PENS = 20261020, 4000, 0.05, 2.0, (0.5, 5.0)
EDGES = 10.0 ** np.arange(9)                                     # 1, 10, ..., 1e8 s; decade d = (10^d, 10^(d+1)]
FREE_DT = (500.0, 100.0, 10.0)
FREE_BIN = {0: 7, 1: 6, 2: 3}                                    # free dt index -> segment bin for the "together" table
PW = np.array([0.35, 0.25, 0.20, 0.20])                          # eq-target, eq-far, raw-A, raw-B
STRAT = {"G": "GCM", "C": "cooler", "E": "edge"}
AGES = [0, 1, 3, 10, 30, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7]
KSER = (1, 2, 5, 10, 2000)
ALLTABS = ("seg", "free", "tog")
RULE_TEXT = {"v3.1": "rule v3.1: failures = unrecovered steps only (LM rescues counted, not failures); exact quadrature "
                     "split at decade edges",
             "v3": "rule v3: failures from the first LM rescue on; whole-node quadrature"}


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ design (solver-free)
class Design:
    """Rows, saved calls, truth, node weights, decades, case keys, parent / condition maps, and the rule."""

    def __init__(self, des, rule):
        des = Path(des)
        self.rule, self.exact = rule, rule == "v3.1"
        r = np.load(des / "rows.npz")
        self.r = {k: r[k] for k in ("kind", "pidx", "unit", "cell", "cond", "start", "age_idx", "t0", "dt", "dt_idx",
                                    "dt_bin", "N", "T", "P_bar", "off")}
        self.cells = [str(c) for c in r["cells"]]
        self.starts = [str(s) for s in r["starts"]]
        self.bin_w = r["dt_bin_weights"].astype(float)
        self.bins = r["dt_bins"]
        s = np.load(des / "saved.npz")
        self.k, self.tabs, self.row = s["k"].astype(np.int64), s["tabs"], s["row"].astype(np.int64)
        p = np.load(des / "parents.npz")
        self.p = {k: p[k] for k in ("cell", "rank", "start", "stratum", "T", "P_bar", "free", "free_order", "cond_key")}
        self.R, self.S, self.P = self.r["N"].size, self.k.size, self.p["cell"].size
        u = np.load(des / "units.npz")
        self.representable_row = u["mc_representable"][self.r["unit"]]
        self.units = {k: u[k] for k in ("model_ly25", "pidx", "age_idx", "mc_vmr13", "mc_species13")}
        off = self.r["off"]
        assert off[-1] == self.S and np.array_equal(self.row, np.repeat(np.arange(self.R), np.diff(off)))
        self.w = self._weights()
        assert (self.tabs > 1.0).all() and (self.tabs <= 1e8).all()
        self.dec = np.searchsorted(EDGES, self.tabs, side="left") - 1
        restart = self.r["t0"][self.row] > 0
        assert np.array_equal(self.dec[restart], (self.r["age_idx"][self.row[restart]].astype(np.int64) - 1) // 2)
        # parents, conditions
        self.pcond = self.p["cell"] * 40 + self.p["rank"]
        self.ncond = 400
        self.cond_cell = np.arange(400) // 40
        st = np.empty(400, "U1")
        st[self.pcond] = self.p["stratum"]
        self.cond_stratum = st
        key = np.empty(400, object)
        key[self.pcond] = self.p["cond_key"]
        self.cond_key = key
        Tc = np.empty(400)
        Tc[self.pcond] = self.p["T"]
        self.cond_T = Tc
        # case keys: segments pidx * 17 + dt_idx; free runs free_order * 3 + free dt index
        kind = self.r["kind"]
        self.seg_row = kind == 0
        fidx = np.full(self.R, -1)
        for i, d in enumerate(FREE_DT):
            fidx[(kind == 1) & (self.r["dt"] == d)] = i
        assert (fidx[kind == 1] >= 0).all()
        self.fidx = fidx
        case = np.where(kind == 0, self.r["pidx"] * 17 + self.r["dt_idx"],
                        self.p["free_order"][self.r["pidx"]] * 3 + fidx)
        assert (case >= 0).all()
        self.case_row = case
        self.fparents = np.flatnonzero(self.p["free"])[np.argsort(self.p["free_order"][self.p["free"]])]
        assert np.array_equal(self.p["free_order"][self.fparents], np.arange(40))
        self.fcond = np.unique(self.pcond[self.fparents])                  # 20 condition ids with free runs
        self.fcond_cell = self.fcond // 40
        self.fparent_fc = np.searchsorted(self.fcond, self.pcond[self.fparents])
        self.fparent_start = self.p["start"][self.fparents]
        assert set(self.fparent_start.tolist()) == {0, 2}
        t = np.load(des / "truth_saved.npz")
        assert [str(g) for g in t["gases12"]] == GASES
        self.logt = t["logx12"]
        log("design loaded", self.R, "rows", self.S, "saved points; rule", rule)

    def pieces(self):
        """The piecewise-linear-in-log10-t pieces of every row (exact quadrature): consecutive saved calls (linear),
        plus [log10 t0, u_1] at the first call's value for restart rows. Cached."""
        if getattr(self, "_pc", None) is None:
            u = np.log10(self.tabs)
            for k in range(9):
                u[self.tabs == 10.0 ** k] = float(k)
            same = np.zeros(self.S, bool)
            same[1:] = self.row[1:] == self.row[:-1]
            i = np.flatnonzero(same)
            t0 = self.r["t0"]
            rs = np.flatnonzero(t0 > 0)
            f = self.r["off"][:-1][rs]
            u0 = np.log10(t0[rs])
            for k in range(9):
                u0[t0[rs] == 10.0 ** k] = float(k)
            self._pc = dict(pa=np.concatenate([i - 1, f]), pb=np.concatenate([i, f]),
                            ua=np.concatenate([u[i - 1], u0]), ub=np.concatenate([u[i], u[f]]),
                            prow=np.concatenate([self.row[i], rs]))
            assert (self._pc["ub"] >= self._pc["ua"]).all()
        return self._pc

    def pieces_elapsed(self):
        """Pieces in log10 of the time since the row's own start, tau = k dt (time-since-restart readout): consecutive
        saved calls only (tau = 0 at the start has no logarithm, so coverage starts at the first saved call). Cached."""
        if getattr(self, "_pe", None) is None:
            tau = self.k * self.r["dt"][self.row]
            u = np.log10(tau)
            for k in range(9):
                u[tau == 10.0 ** k] = float(k)
            same = np.zeros(self.S, bool)
            same[1:] = self.row[1:] == self.row[:-1]
            i = np.flatnonzero(same)
            self._pe = dict(pa=i - 1, pb=i, ua=u[i - 1], ub=u[i], prow=self.row[i])
            assert (self._pe["ub"] >= self._pe["ua"]).all() and (self._pe["ua"] >= 0).all()
        return self._pe

    def _weights(self):
        """Whole-node weights in dex of log10 tabs (trapezoid; a restart row's first node also holds back to t0); a
        row's weights sum to its covered log-time."""
        off, t = self.r["off"], self.tabs
        first, last = off[:-1], off[1:] - 1
        g = np.zeros(self.S)
        g[1:] = np.log10(t[1:] / t[:-1])
        g[first] = 0.0
        gn = np.zeros(self.S)
        gn[:-1] = g[1:]
        gn[last] = 0.0
        w = 0.5 * (g + gn)
        t0 = self.r["t0"]
        rs = t0 > 0
        w[first[rs]] += np.log10(t[first[rs]] / t0[rs])
        # checks: sums equal the covered log-time
        sw = np.add.reduceat(w, first)
        cov = np.where(rs, np.log10(t[last] / np.where(rs, t0, 1.0)), np.log10(t[last] / t[first]))
        assert np.allclose(sw, cov, rtol=1e-9, atol=1e-12) and (w > 0).all()
        return w


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
    """Mini-chem log10 states at every saved point (or only the rtol 1e-5 subset's) and first failed call per row."""
    sp = None
    logp = np.full((D.S, 12), np.nan)
    ffrow = np.zeros(D.R, np.int64)
    files = [mc / "rtol1e-5.npz"] if rtol5 else [mc / f"{c}.npz" for c in D.cells]
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
        assert have.all()
        rs = np.load(mc / "rows_summary.npz")
        assert np.array_equal(rs["first_fail_any"].astype(np.int64), ffrow)
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


# ----------------------------------------------------------------------------------------------------- aggregation
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


def edge_singletons(D):
    """Points that are the only saved calls of their (case, decade) and sit exactly at the decade's upper edge in a
    t = 0 row (whole-node quadrature readout)."""
    seg_pt = D.seg_row[D.row]
    key = np.where(seg_pt, D.case_row[D.row], D.P * 17 + D.case_row[D.row]) * 8 + D.dec
    edge = (D.r["t0"][D.row] == 0) & (D.tabs == EDGES[D.dec + 1])
    na = np.bincount(key)
    ne = np.bincount(key, edge.astype(float), minlength=na.size)
    return edge & (na[key] == ne[key])


def exact_sums(D, e, rowmask, tab, ncase, pc=None):
    """Exact integral of the piecewise-linear (log10 t) per-call statistic over each decade, pieces split at the decade
    edges; returns covered log-length and integral per (case, decade). pc: the pieces (default D.pieces())."""
    pc = D.pieces() if pc is None else pc
    sel = D.seg_row[pc["prow"]] if tab == "seg" else np.logical_not(D.seg_row[pc["prow"]])
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


def case_means(sw, swe, ncase):
    """(case, decade) means from the covered log-length sw and integral swe; a decade with sw = 0 is untested (NaN).
    Case score = mean over its tested decades."""
    with np.errstate(invalid="ignore", divide="ignore"):
        V = np.where(sw > 0, swe / np.where(sw > 0, sw, 1.0), np.nan).reshape(ncase, 8)   # (case, decade)
    return V, _nanmean(V, 1)


def profile(D, st, ff, pen=PEN, rowmask=None, stats=STATS, ptmask=None, exact=None):
    """Per-condition scores of one solver. st = point_stats output, ff = first failed call per row. rowmask / ptmask
    (readouts) zero the quadrature weight of excluded rows / points (ptmask: whole-node quadrature only); exact = the
    quadrature (default: the rule's)."""
    exact = D.exact if exact is None else exact
    fp = failed_points(D, ff)
    w = D.w if rowmask is None else D.w * rowmask[D.row]
    if ptmask is not None:
        w = w * ptmask
    seg_pt = D.seg_row[D.row]
    out = {}
    for sname in stats:
        e = np.where(fp, pen, st[sname])
        res = {}
        for tab in ("seg", "free"):
            sel = seg_pt if tab == "seg" else ~seg_pt
            ncase = D.P * 17 if tab == "seg" else 40 * 3
            if exact:
                sw, swe = exact_sums(D, e, rowmask, tab, ncase)
            else:
                key = D.case_row[D.row[sel]] * 8 + D.dec[sel]
                sw = np.bincount(key, w[sel], ncase * 8)
                swe = np.bincount(key, w[sel] * e[sel], ncase * 8)
            V, case = case_means(sw, swe, ncase)
            if tab == "seg":
                Vd = V.reshape(D.P, 17, 8)
                cs = case.reshape(D.P, 17)
                bm = _nanmean(cs[:, :16].reshape(D.P, 8, 2), 2)                         # (parent, bin)
                pscore = _nanwmean(bm, D.bin_w, 1)
                anchor = cs[:, 16]
                dbm = _nanmean(Vd[:, :16, :].reshape(D.P, 8, 2, 8), 2)                  # (parent, bin, decade)
                pdec = _nanwmean(np.moveaxis(dbm, 1, 2), D.bin_w, 2)                     # (parent, decade)
                adec = Vd[:, 16, :]
                res["seg"] = dict(cond=_cond(D, pscore), bin=_cond(D, bm), anchor=_cond(D, anchor),
                                  dec=_cond(D, pdec), anchor_dec=_cond(D, adec), start=_by_start(D, pscore),
                                  bin_start=None)
            else:
                Vd = V.reshape(40, 3, 8)
                cs = case.reshape(40, 3)
                pscore = _nanmean(cs, 1)
                pdec = _nanmean(Vd, 1)
                res["free"] = dict(cond=_fcond(D, pscore), bin=_fcond(D, cs), dec=_fcond(D, pdec),
                                   start=_fby_start(D, pscore))
        s, f = res["seg"], res["free"]
        tb = s["bin"][D.fcond].copy()
        for fi, bi in FREE_BIN.items():
            tb[:, bi] = 0.5 * (tb[:, bi] + f["bin"][:, fi])
        res["tog"] = dict(cond=0.5 * (s["cond"][D.fcond] + f["cond"]), bin=tb)
        out[sname] = res
    return out


def _cond(D, X):
    """Parent array (P, ...) -> condition array (400, ...), parent weights renormalised over finite parents."""
    sh = X.shape[1:]
    M = np.full((400, 4) + sh, np.nan)
    M[D.pcond, D.p["start"]] = X
    return _nanwmean(np.moveaxis(M, 1, -1), PW, -1)


def _by_start(D, X):
    M = np.full((400, 4), np.nan)
    M[D.pcond, D.p["start"]] = X
    return M


def _fcond(D, X):
    sh = X.shape[1:]
    M = np.full((20, 4) + sh, np.nan)
    M[D.fparent_fc, D.fparent_start] = X
    return _nanwmean(np.moveaxis(M, 1, -1), PW, -1)


def _fby_start(D, X):
    M = np.full((20, 4), np.nan)
    M[D.fparent_fc, D.fparent_start] = X
    return M[:, [0, 2]]


# ------------------------------------------------------------------------------------------------ summaries, bootstrap
def trimmed(X, axis=-1):
    X = np.sort(X, axis=axis)
    n = X.shape[axis]
    g = int(np.floor(TRIM * n))
    sl = [slice(None)] * X.ndim
    sl[axis] = slice(g, n - g)
    return X[tuple(sl)].mean(axis=axis)


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
    return dict(calls=int(N.sum()), failed_calls=int(fc.sum()), call_rate=float(fc.sum() / N.sum()),
                rows=int(N.size), failed_rows=int((f > 0).sum()), row_rate=float((f > 0).mean()))


def compare(D, PA, PB, ffA, ffB, label_a, label_b):
    """Paired tables for A - B on every statistic, per table, and the verdicts."""
    res = {"a": label_a, "b": label_b}
    for tab, cells in (("seg", D.cond_cell), ("free", D.fcond_cell), ("tog", D.fcond_cell)):
        t = {}
        for s in STATS:
            A, Bp = PA[s][tab], PB[s][tab]
            d = A["cond"] - Bp["cond"]
            e = dict(D=summarize(d, cells), A=summarize(A["cond"], cells), B=summarize(Bp["cond"], cells))
            if s == "mean12":
                Db = A["bin"] - Bp["bin"]
                if tab == "tog":
                    e["band"] = None
                else:
                    e["band"] = band(Db, cells)
                e["bins"] = [dict(D=summarize(Db[:, j], cells), A=summarize(A["bin"][:, j], cells),
                                  B=summarize(Bp["bin"][:, j], cells)) for j in range(Db.shape[1])]
                if tab != "tog":
                    e["dec"] = [dict(D=summarize(A["dec"][:, j] - Bp["dec"][:, j], cells),
                                     A=summarize(A["dec"][:, j], cells), B=summarize(Bp["dec"][:, j], cells))
                                for j in range(8)]
                    e["start"] = [dict(D=summarize(A["start"][:, j] - Bp["start"][:, j], cells),
                                       A=summarize(A["start"][:, j], cells), B=summarize(Bp["start"][:, j], cells))
                                  for j in range(A["start"].shape[1])]
                if tab == "seg":
                    e["anchor"] = dict(D=summarize(A["anchor"] - Bp["anchor"], cells),
                                       A=summarize(A["anchor"], cells), B=summarize(Bp["anchor"], cells))
                    e["anchor_dec"] = [dict(D=summarize(A["anchor_dec"][:, j] - Bp["anchor_dec"][:, j], cells))
                                       for j in range(8)]
                    e["cell"] = []
                    for c in range(10):
                        m = D.cond_cell == c
                        e["cell"].append(dict(D=summarize(d[m], D.cond_cell[m]), A=summarize(A["cond"][m], D.cond_cell[m]),
                                              B=summarize(Bp["cond"][m], D.cond_cell[m])))
                    e["stratum"] = {}
                    for k in ("G", "C", "E"):
                        m = D.cond_stratum == k
                        e["stratum"][k] = dict(D=summarize(d[m], D.cond_cell[m]), A=summarize(A["cond"][m], D.cond_cell[m]),
                                               B=summarize(Bp["cond"][m], D.cond_cell[m]))
                if tab == "free":
                    e["cell_meanD"] = [float(np.mean(d[D.fcond_cell == c])) for c in range(10)]
                    st = D.cond_stratum[D.fcond]
                    e["stratum"] = {k: dict(D=summarize(d[st == k], D.fcond_cell[st == k])) for k in ("G", "C", "E")}
            t[s] = e
        if tab != "tog":
            rows = D.seg_row if tab == "seg" else ~D.seg_row
            t["fail_A"], t["fail_B"] = fail_rates(D, ffA, rows), fail_rates(D, ffB, rows)
            t["verdict"] = verdict(t)
        res[tab] = t
    v = [res["seg"]["verdict"]["verdict"], res["free"]["verdict"]["verdict"]]
    if all(x == "beats" for x in v):
        hv = "beats"
    elif "loses" in v and "beats" not in v:
        hv = "loses"
    else:
        hv = "not established"
    res["headline"] = dict(verdict=hv, seg=v[0], free=v[1])
    return res


def verdict(t):
    m, w = t["mean12"], t["worst"]
    c = dict(trim_upper_lt0=m["D"]["trim_hi"] < 0, mean_lt0=m["D"]["mean"] < 0,
             bins_upper_lt005=max(m["band"]["hi"]) < 0.05, worst_trim_upper_lt005=w["D"]["trim_hi"] < 0.05,
             fail_ok=t["fail_A"]["call_rate"] <= t["fail_B"]["call_rate"] + 0.001)
    c = {k: bool(v) for k, v in c.items()}
    if all(c.values()):
        vv = "beats"
    elif m["D"]["trim_lo"] > 0:
        vv = "loses"
    else:
        vv = "not established"
    return dict(verdict=vv, **c)


# ------------------------------------------------------------------------------------------------------ readouts
def coverage(D):
    """Per case and decade: union log-length of the rows' intervals, restarts, longest span, and whether the decade is
    tested (whole-node quadrature: a saved call in it; exact quadrature: positive covered log-time)."""
    t0, dt, N = D.r["t0"], D.r["dt"], D.r["N"].astype(float)
    a = np.where(t0 > 0, t0, dt)
    b = np.where(t0 > 0, t0 + N * dt, N * dt)
    out = {}
    for tab, ncase in (("seg", D.P * 17), ("free", 120)):
        rows = np.flatnonzero(D.seg_row if tab == "seg" else ~D.seg_row)
        order = rows[np.argsort(D.case_row[rows], kind="stable")]
        cid = D.case_row[order]
        bounds = np.flatnonzero(np.diff(np.concatenate([[-1], cid, [ncase + 1]])))
        cov = np.zeros((ncase, 8))
        lng = np.zeros((ncase, 8))
        nre = np.zeros((ncase, 8), np.int64)
        for i in range(bounds.size - 1):
            rr = order[bounds[i]:bounds[i + 1]]
            c = cid[bounds[i]]
            iv = sorted(zip(a[rr].tolist(), b[rr].tolist(), (t0[rr] > 0).tolist()))
            merged = []
            for x, y, _ in iv:
                if merged and x <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], y)
                else:
                    merged.append([x, y])
            for d in range(8):
                lo, hi = EDGES[d], EDGES[d + 1]
                for x, y in merged:
                    x2, y2 = max(x, lo), min(y, hi)
                    if y2 > x2:
                        L = np.log10(y2 / x2)
                        cov[c, d] += L
                        lng[c, d] = max(lng[c, d], L)
                nre[c, d] = sum(1 for x, _, rs in iv if rs and lo <= x < hi)
        if D.exact:
            has = exact_sums(D, np.zeros(D.S), None, tab, ncase)[0].reshape(ncase, 8) > 0
        else:
            sel = D.seg_row[D.row] if tab == "seg" else ~D.seg_row[D.row]
            key = D.case_row[D.row[sel]] * 8 + D.dec[sel]
            has = np.bincount(key, minlength=ncase * 8).reshape(ncase, 8) > 0
        out[tab] = dict(cov=cov, longest=lng, restarts=nre, has=has)
    return out


def coverage_table(D, cv):
    s = cv["seg"]
    dti = np.tile(np.arange(17), D.P)
    rows = []
    for j in list(range(8)) + ["anchor"]:
        m = (dti // 2 == j) & (dti < 16) if j != "anchor" else dti == 16
        rows.append(dict(bin=j, cov=s["cov"][m].mean(0).tolist(), covered=s["has"][m].mean(0).tolist(),
                         longest=s["longest"][m].mean(0).tolist(), restarts=s["restarts"][m].mean(0).tolist()))
    f = cv["free"]
    fi = np.tile(np.arange(3), 40)
    fr = [dict(dt=FREE_DT[i], cov=f["cov"][fi == i].mean(0).tolist(), covered=f["has"][fi == i].mean(0).tolist())
          for i in range(3)]
    return dict(seg=rows, free=fr)


def large_misses(D, st, ff, name, out_dir):
    """Rows with a non-failed saved call above 0.5 / 2 dex (12-gas mean; any single gas); identities to CSV."""
    fp = failed_points(D, ff)
    m12 = np.where(fp, -1.0, st["mean12"])
    wst = np.where(fp, -1.0, st["worst"])
    off = D.r["off"]
    rmax12 = np.maximum.reduceat(m12, off[:-1])
    rmaxw = np.maximum.reduceat(wst, off[:-1])
    out = {}
    for thr in (0.5, 2.0):
        out[f"rows_mean12_gt{thr}"] = int((rmax12 > thr).sum())
        out[f"rows_anygas_gt{thr}"] = int((rmaxw > thr).sum())
        for tab, sel in (("seg", D.seg_row), ("free", ~D.seg_row)):
            out[f"{tab}_rows_mean12_gt{thr}"] = int((rmax12[sel] > thr).sum())
            out[f"{tab}_rows_anygas_gt{thr}"] = int((rmaxw[sel] > thr).sum())
    rr = np.flatnonzero(rmaxw > 0.5)
    lines = ["row,kind,cell,cond_key,stratum,T_K,P_bar,start,dt_s,dt_bin,t0_s,N,k_worst,tabs_worst_s,gas,gas_err_dex,"
             "max_mean12_dex,k_mean12,tabs_mean12_s"]
    recs = []
    for r in rr:
        a, b = off[r], off[r + 1]
        jw = a + int(np.argmax(wst[a:b]))
        jm = a + int(np.argmax(m12[a:b]))
        p = D.r["pidx"][r]
        rec = dict(row=int(r), kind=int(D.r["kind"][r]), cell=D.cells[D.r["cell"][r]], cond_key=str(D.p["cond_key"][p]),
                   stratum=str(D.p["stratum"][p]), T=float(D.r["T"][r]), P_bar=float(D.r["P_bar"][r]),
                   start=D.starts[D.r["start"][r]], dt=float(D.r["dt"][r]), dt_bin=int(D.r["dt_bin"][r]),
                   t0=float(D.r["t0"][r]), N=int(D.r["N"][r]), k_worst=int(D.k[jw]), tabs_worst=float(D.tabs[jw]),
                   gas=GASES[int(st["wgas"][jw])], gas_err=float(wst[jw]), max_mean12=float(m12[jm]),
                   k_mean12=int(D.k[jm]), tabs_mean12=float(D.tabs[jm]))
        recs.append(rec)
        lines.append(",".join(str(v) for v in rec.values()))
    (out_dir / f"large_misses_{name}.csv").write_text("\n".join(lines) + "\n")
    recs.sort(key=lambda x: -x["gas_err"])
    out["top_anygas"] = recs[:25]
    out["top_mean12"] = sorted(recs, key=lambda x: -x["max_mean12"])[:15]
    # breakdowns of rows with any gas > 2 dex and 12-gas mean > 0.5 dex
    for nm, sel in (("anygas_gt2", rmaxw > 2.0), ("mean12_gt0.5", rmax12 > 0.5)):
        bd = {}
        for fld, labels in (("cell", D.cells), ("start", D.starts)):
            v = D.r[fld][sel]
            bd[fld] = {labels[i]: int((v == i).sum()) for i in range(len(labels)) if (v == i).any()}
        dtc = np.where(D.r["kind"][sel] == 1, 9, np.where(D.r["dt_idx"][sel] == 16, 8, D.r["dt_bin"][sel]))
        bd["dt_class"] = {str(int(i)): int((dtc == i).sum()) for i in np.unique(dtc)}
        ag = D.r["age_idx"][sel]
        bd["t0"] = {f"{AGES[i]:g}": int((ag == i).sum()) for i in np.unique(ag)}
        strat = D.p["stratum"][D.r["pidx"][sel]]
        bd["stratum"] = {STRAT[k]: int((strat == k).sum()) for k in ("G", "C", "E")}
        if nm == "anygas_gt2":
            g = st["wgas"][off[:-1][sel] + np.array([int(np.argmax(wst[off[r]:off[r + 1]])) for r in np.flatnonzero(sel)],
                                                     dtype=np.int64)] if sel.any() else np.array([], int)
            bd["gas"] = {GASES[i]: int((g == i).sum()) for i in np.unique(g)}
        out["breakdown_" + nm] = bd
    return out


def row_quad_mean(D, e, rows):
    """Per-row log-time mean of a per-point statistic (whole-row trapezoid; the same under both quadratures)."""
    off = D.r["off"]
    num = np.add.reduceat(D.w * e, off[:-1])[rows]
    den = np.add.reduceat(D.w, off[:-1])[rows]
    return num / den


def rtol_readout(D, mc, st_main, ff_main, PB_main, PA):
    """Mini-chem rtol 1e-5 subset vs rtol 1e-3 on the same rows (against VULCAN)."""
    logp5, ff5, have = load_minichem(D, mc, rtol5=True)
    z = np.load(mc / "rtol1e-5.npz")
    rows = z["row"].astype(np.int64)
    sub = np.zeros(D.R, bool)
    sub[rows] = True
    assert np.array_equal(have, sub[D.row])
    logp = np.where(have[:, None], logp5, np.nan)
    st5 = point_stats(D, np.where(have[:, None], logp, 0.0), ff5)        # non-subset points are placeholders
    ff5 = st5["ff"]
    e3 = np.where(failed_points(D, ff_main), PEN, st_main["mean12"])
    e5 = np.where(failed_points(D, ff5), PEN, st5["mean12"])
    m3, m5 = row_quad_mean(D, e3, rows), row_quad_mean(D, e5, rows)
    d = m5 - m3
    q = [50, 90, 99, 99.9]
    out = dict(rows=int(rows.size), fail_rows_1e3=int((ff_main[rows] > 0).sum()), fail_rows_1e5=int((ff5[rows] > 0).sum()),
               row_mean_1e3=float(m3.mean()), row_mean_1e5=float(m5.mean()), diff_mean=float(d.mean()),
               absdiff_pct={str(p): float(np.percentile(np.abs(d), p)) for p in q}, absdiff_max=float(np.abs(d).max()),
               frac_1e5_better=float((d < 0).mean()), frac_1e5_worse=float((d > 0).mean()))
    # substituted full mini-chem: subset points take the 1e-5 states and failure status
    st_sub = {}
    for s in STATS:
        st_sub[s] = np.where(have, st5[s], st_main[s])
    st_sub["wgas"] = st_main["wgas"]
    ff_sub = np.where(sub, ff5, ff_main)
    P_sub = profile(D, st_sub, ff_sub, stats=("mean12",))
    out["substituted"] = {}
    for tab, cells in (("seg", D.cond_cell), ("free", D.fcond_cell)):
        out["substituted"][tab] = dict(
            mc_1e3=summarize(PB_main["mean12"][tab]["cond"], cells), mc_sub=summarize(P_sub["mean12"][tab]["cond"], cells),
            D_1e3=summarize(PA["mean12"][tab]["cond"] - PB_main["mean12"][tab]["cond"], cells),
            D_sub=summarize(PA["mean12"][tab]["cond"] - P_sub["mean12"][tab]["cond"], cells))
    return out


# ---------------------------------------------------------------------------------------------------------- text
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


def table_lines(D, res):
    a, b = res["a"], res["b"]
    L = []
    P = L.append
    P(f"=== {a} vs {b} (D = {a} - {b}, dex; negative = {a} better) -- {D.rule} ===")
    h = res["headline"]
    P(f"HEADLINE VERDICT: {a} {h['verdict'].upper()} vs {b} (segment table: {h['seg']}; free-run table: {h['free']})")
    for tab, nm in (("seg", "SEGMENT TABLE (400 conditions)"), ("free", "FREE-RUN TABLE (20 conditions)"),
                    ("tog", "TOGETHER (20 conditions with both; equal weight segment / free run)")):
        t = res[tab]
        P("")
        P(f"--- {nm} ---")
        if "verdict" in t:
            v = t["verdict"]
            P(f"verdict: {v['verdict']}  | trimmed upper<0: {v['trim_upper_lt0']}  mean<0: {v['mean_lt0']}  "
              f"every bin simult. upper<+0.05: {v['bins_upper_lt005']}  worst-gas trimmed upper<+0.05: "
              f"{v['worst_trim_upper_lt005']}  failure rate ok: {v['fail_ok']}")
        P(f"{'statistic':10s} {'D trimmed [95%]':>26s} {'D mean [95%]':>26s} {'D median [95%]':>26s} "
          f"{'%D<0':>5s} {a + ' trim':>10s} {b + ' trim':>10s} {a + ' mean':>10s} {b + ' mean':>10s}")
        for s in STATS:
            e = t[s]
            P(f"{s:10s} {fs(e['D']):>26s} {fs(e['D'], 'mean'):>26s} {fs(e['D'], 'median'):>26s} "
              f"{100 * e['D']['frac_neg']:5.1f} {e['A']['trim']:10.4f} {e['B']['trim']:10.4f} {e['A']['mean']:10.4f} "
              f"{e['B']['mean']:10.4f}")
        for side, lab in (("A", a), ("B", b)):
            P(f"absolute error vs VULCAN, {lab}: " + "; ".join(
                f"{s} trimmed {fs(t[s][side])} mean {fs(t[s][side], 'mean')} median {fs(t[s][side], 'median')}"
                for s in ("mean12", "worst")))
        if tab != "tog":
            for side, lab in (("fail_A", a), ("fail_B", b)):
                fr = t[side]
                P(f"failures {lab:6s}: calls {fr['failed_calls']:,} of {fr['calls']:,} ({100 * fr['call_rate']:.4f}%), "
                  f"rows {fr['failed_rows']:,} of {fr['rows']:,} ({100 * fr['row_rate']:.4f}%)")
        e = t["mean12"]
        P("per dt " + ("bin" if tab != "free" else "value") + " (primary statistic):")
        P(f"  {'bin':12s} {'D trimmed [95%]':>26s} {'simult. [lo, hi]':>20s} {'D mean':>8s} {'D median':>8s} "
          f"{a + ' trim':>10s} {b + ' trim':>10s}")
        labels = ([f"{int(D.bins[j][0])}-{int(D.bins[j][1])} s" for j in range(8)] if tab != "free"
                  else [f"{FREE_DT[j]:g} s" for j in range(3)])
        for j, bl in enumerate(labels):
            x = e["bins"][j]
            bd = e["band"]
            sim = f"[{f3(bd['lo'][j])}, {f3(bd['hi'][j])}]" if bd else ""
            P(f"  {bl:12s} {fs(x['D']):>26s} {sim:>20s} {f3(x['D'].get('mean')):>8s} {f3(x['D'].get('median')):>8s} "
              f"{fa(x['A']):>10s} {fa(x['B']):>10s}")
        if e.get("band"):
            P(f"  band critical value c = {e['band']['c']:.3f} (studentized max-|z|, two-sided 95%)")
        if tab == "seg":
            x = e["anchor"]
            P(f"10 s anchor: D trimmed {fs(x['D'])}, mean {fs(x['D'], 'mean')}, median {fs(x['D'], 'median')}; "
              f"{a} {fa(x['A'])}, {b} {fa(x['B'])}")
        if tab != "tog":
            P("per decade of absolute time (primary statistic; readout, individual 95% intervals):")
            for j in range(8):
                x = e["dec"][j]
                P(f"  1e{j}-1e{j + 1} s   D trimmed {fs(x['D']):>26s}  mean {f3(x['D'].get('mean'))}  "
                  f"n {x['D'].get('n', 0):3d}  {a} {fa(x['A'])}  {b} {fa(x['B'])}")
            P("per start type (primary statistic, that parent alone):")
            names = D.starts if tab == "seg" else ["eq-target", "raw-A"]
            for j, sn in enumerate(names):
                x = e["start"][j]
                P(f"  {sn:10s} D trimmed {fs(x['D']):>26s}  mean {fs(x['D'], 'mean'):>26s}  {a} {fa(x['A'])}  {b} {fa(x['B'])}")
        if tab == "seg":
            P("per cell (primary statistic; bootstrap inside the cell):")
            for c in range(10):
                x = e["cell"][c]
                P(f"  {D.cells[c]:10s} D trimmed {fs(x['D']):>26s}  mean {fs(x['D'], 'mean'):>26s}  "
                  f"{a} {fa(x['A'])}  {b} {fa(x['B'])}")
            P("per T stratum (primary statistic):")
            for k in ("G", "C", "E"):
                x = e["stratum"][k]
                P(f"  {STRAT[k]:7s} (n {x['D']['n']:3d}) D trimmed {fs(x['D']):>26s}  mean {fs(x['D'], 'mean'):>26s}  "
                  f"{a} {fa(x['A'])}  {b} {fa(x['B'])}")
        if tab == "free":
            P("per cell, mean D of its 2 conditions: " + ", ".join(f"{D.cells[c]} {f3(v)}" for c, v in enumerate(e["cell_meanD"])))
            P("per T stratum: " + ", ".join(f"{STRAT[k]} (n {e['stratum'][k]['D'].get('n', 0)}) {fs(e['stratum'][k]['D'])}"
                                             for k in ("G", "C", "E")))
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


# ------------------------------------------------------------------------------------------------------- solvers
def first_call_values(D, logp):
    """log10 prediction at saved calls k in KSER of every t = 0 segment row (N = 2000)."""
    rows = np.flatnonzero(D.seg_row & (D.r["t0"] == 0))
    off = D.r["off"]
    out = {}
    for kk in KSER:
        idx = np.empty(rows.size, np.int64)
        for j, r in enumerate(rows):
            a = off[r] + int(np.searchsorted(D.k[off[r]:off[r + 1]], kk))
            assert D.k[a] == kk
            idx[j] = a
        out[kk] = (idx, logp[idx].copy())
    return rows, out


def solver_mc(D, mc):
    logp, ffrow, _ = load_minichem(D, mc)
    st = point_stats(D, logp, ffrow)
    st["first_t0"] = first_call_values(D, logp)
    del logp
    rs = np.load(mc / "rows_summary.npz")
    st["recorded"] = dict(rows_excess_work=int((rs["n_excess_total"] > 0).sum()), excess_work_total=int(rs["n_excess_total"].sum()),
                          rows_dlsode_warning=int((rs["n_tdh_total"] > 0).sum()), dlsode_warning_total=int(rs["n_tdh_total"].sum()),
                          rows_masked_state=int((rs["n_masked"] > 0).sum()),
                          rows_driver_fail_ne_any=int((rs["first_fail"] != rs["first_fail_any"]).sum()))
    log("mini-chem point stats; saved points with an invalid state:", st["nbad_saved"])
    return st


def solver_model(D, run, name):
    logp, ff, unrec, extra = load_model(run / name / "model_states.npz")
    st = point_stats(D, logp, ff)
    st["ff_literal"] = st["ff"]
    st["ff_lenient"] = merge_bad(D, unrec, st["bad"])
    st["ff"] = st["ff_lenient"] if D.exact else st["ff_literal"]
    st["first_t0"] = first_call_values(D, logp)
    del logp
    st["extra"] = extra
    log(f"{name} point stats; saved points with an invalid state:", st["nbad_saved"])
    return st


def ff_reading(st, reading):
    """A solver's first failed calls under a model failure reading ('literal' = from the first LM rescue on, 'lenient'
    = unrecovered steps only); mini-chem has one reading."""
    return st.get("ff_" + reading, st["ff"])


# ------------------------------------------------------------------------------------------------------ readouts
def subset_table(D, stA, ffA, stB, ffB, rm):
    """The segment table restricted to rows rm, with the five verdict checks."""
    PA = profile(D, stA, ffA, rowmask=rm, stats=("mean12", "worst"))
    PB = profile(D, stB, ffB, rowmask=rm, stats=("mean12", "worst"))
    c = D.cond_cell
    t = {s: dict(D=summarize(PA[s]["seg"]["cond"] - PB[s]["seg"]["cond"], c), A=summarize(PA[s]["seg"]["cond"], c),
                 B=summarize(PB[s]["seg"]["cond"], c)) for s in ("mean12", "worst")}
    Db = PA["mean12"]["seg"]["bin"] - PB["mean12"]["seg"]["bin"]
    t["mean12"]["band"] = band(Db, c)
    t["fail_A"], t["fail_B"] = fail_rates(D, ffA, rm), fail_rates(D, ffB, rm)
    t["verdict"] = verdict(t)
    return t


def since_restart(D, st, ff, rm, pen=PEN):
    """Per-condition primary statistic per decade of the time since the row's own start (segment rows rm), exact split
    in log10 of that time; within a parent, the 2 draws of a bin averaged and the bins combined by log weights
    renormalised over the bins that cover the decade (the 10 s anchor left out), as the per-decade readout."""
    e = np.where(failed_points(D, ff), pen, st["mean12"])
    ncase = D.P * 17
    V, _ = case_means(*exact_sums(D, e, rm, "seg", ncase, D.pieces_elapsed()), ncase)
    dbm = _nanmean(V.reshape(D.P, 17, 8)[:, :16, :].reshape(D.P, 8, 2, 8), 2)          # (parent, bin, decade)
    return _cond(D, _nanwmean(np.moveaxis(dbm, 1, 2), D.bin_w, 2))                   # (condition, decade)


def posthoc(D, stA, stB, inputs, b_is_mc):
    """Readouts of one comparison (never the verdict): (1) per restart age, (2) restart-only and t = 0-only segment
    tables with the five checks, (3) the other rule's quadrature, (4) first-call moves from t = 0 (vs mini-chem only),
    (5) time since restart (this rule's failure reading). (1)-(3): both model failure readings; a model on side B takes
    the same reading as side A."""
    out = {}
    c = D.cond_cell
    seg, age = D.seg_row, D.r["age_idx"]
    readings = ("literal", "lenient")
    per_age = {r: [] for r in readings}
    for a in range(17):
        rm = seg & (age == a)
        pb = {}
        for r in readings:
            ffA, ffB = ff_reading(stA, r), ff_reading(stB, r)
            if id(ffB) not in pb:
                pb[id(ffB)] = profile(D, stB, ffB, rowmask=rm, stats=("mean12",))["mean12"]["seg"]["cond"]
            PB = pb[id(ffB)]
            PA = profile(D, stA, ffA, rowmask=rm, stats=("mean12",))["mean12"]["seg"]["cond"]
            per_age[r].append(dict(t0=AGES[a], rows=int(rm.sum()), D=summarize(PA - PB, c), A=summarize(PA, c),
                                   B=summarize(PB, c), fail_A=fail_rates(D, ffA, rm), fail_B=fail_rates(D, ffB, rm)))
    out["per_age"] = per_age
    out["subsets"] = {}
    for nm, rm in (("restart_only_t0>0", seg & (D.r["t0"] > 0)), ("t0_only", seg & (D.r["t0"] == 0))):
        out["subsets"][nm] = {r: subset_table(D, stA, ff_reading(stA, r), stB, ff_reading(stB, r), rm) for r in readings}
    ex = {}
    for r in readings:
        ffA, ffB = ff_reading(stA, r), ff_reading(stB, r)
        PA = profile(D, stA, ffA, exact=not D.exact, stats=("mean12", "worst"))
        PB = profile(D, stB, ffB, exact=not D.exact, stats=("mean12", "worst"))
        ex[r] = mini_compare(D, PA, PB, ffA, ffB)
    out["node_quadrature" if D.exact else "exact_quadrature"] = ex
    out["since_restart"] = {}
    for nm, rm in (("restart_only_t0>0", seg & (D.r["t0"] > 0)), ("t0_only", seg & (D.r["t0"] == 0))):
        A, B = since_restart(D, stA, stA["ff"], rm), since_restart(D, stB, stB["ff"], rm)
        out["since_restart"][nm] = [dict(D=summarize(A[:, j] - B[:, j], c), A=summarize(A[:, j], c),
                                         B=summarize(B[:, j], c)) for j in range(6)]
    if b_is_mc:
        out["first_call_moves"] = first_call_moves(D, stA, stB, inputs)
    return out


def first_call_moves(D, stA, stB, inputs):
    """First calls from t = 0 (segments), per start type: 12-gas mean of floored |log10 change| of the truth, model A
    and mini-chem (from its own mapped start), and the floored errors."""
    rows, kA = stA["first_t0"]
    rows_b, kB = stB["first_t0"]
    assert np.array_equal(rows, rows_b)
    f, mA = kA[1]
    mB = kB[1][1]
    t1 = D.logt[f]
    s0 = inputs["model_ly12"][D.r["unit"][rows]]
    s0mc = inputs["mc_ly12"][D.r["unit"][rows]]
    F = lambda x: np.maximum(x, LOGS)
    dT = np.abs(F(t1) - F(s0)).mean(1)
    dA = np.abs(F(mA) - F(s0)).mean(1)
    eA = np.abs(F(mA) - F(t1)).mean(1)
    dB = np.abs(F(mB) - F(s0mc)).mean(1)
    eB = np.abs(F(mB) - F(t1)).mean(1)
    st_ = D.r["start"][rows]
    mv = {}
    for s, sn in enumerate(D.starts):
        m = st_ == s
        big = m & (dT > 0.1)
        md, mn = np.nanmedian, np.nanmean
        mv[sn] = dict(rows=int(m.sum()), truth_move_median=float(md(dT[m])), truth_move_mean=float(mn(dT[m])),
                      model_move_median=float(md(dA[m])), model_move_mean=float(mn(dA[m])),
                      model_err_median=float(md(eA[m])), model_err_mean=float(mn(eA[m])),
                      mc_move_median=float(md(dB[m])), mc_err_median=float(md(eB[m])), mc_err_mean=float(mn(eB[m])),
                      mc_rows_invalid=int(np.isnan(eB[m]).sum()),
                      frac_model_move_lt_half_truth=float((dA[big] < 0.5 * dT[big]).mean()) if big.any() else None,
                      rows_truth_move_gt0p1=int(big.sum()))
        ser = {}
        for kk in KSER:
            ik, mk = kA[kk]
            tk = D.logt[ik]
            ser[kk] = dict(truth_move=float(md(np.abs(F(tk) - F(s0)).mean(1)[m])),
                           model_move=float(md(np.abs(F(mk) - F(s0)).mean(1)[m])),
                           model_err=float(md(np.abs(F(mk) - F(tk)).mean(1)[m])),
                           mc_err=float(md(np.abs(F(kB[kk][1]) - F(tk)).mean(1)[m])))
        mv[sn]["series_medians"] = ser
    return mv


def inputs_readout(D):
    """Entries of the 25-entry model input at the 1e-30 floor in the design's start units, per start type, t0 = 0 vs
    t0 > 0; and the start states (log10, 12 gases) of the model and of mini-chem for the first-call readout."""
    ly = D.units["model_ly25"]
    cnt = (ly <= -29.999).sum(1)
    st = D.p["start"][D.units["pidx"]]
    ag = D.units["age_idx"]
    out = {"panel": {}}
    for s, sn in enumerate(D.starts):
        for a, lab in ((0, "t0=0"), (None, "t0>0")):
            m = (st == s) & ((ag == 0) if a == 0 else (ag > 0))
            out["panel"][f"{sn} {lab}"] = dict(units=int(m.sum()), min=int(cnt[m].min()), median=float(np.median(cnt[m])),
                                               max=int(cnt[m].max()), frac_ge14=float((cnt[m] >= 14).mean()))
    mc = D.units["mc_vmr13"]
    sp = [str(x) for x in D.units["mc_species13"]]
    with np.errstate(divide="ignore"):
        mcl = np.log10(np.maximum(mc[:, [sp.index(g) for g in GASES]], 1e-300))
    return out, dict(model_ly12=ly[:, :12], mc_ly12=mcl)


def posthoc_lines(D, ph, a, b):
    prim = "lenient" if D.exact else "literal"
    L = [f"--- READOUTS ({a} vs {b}; never the verdict; failure readings: literal = from the first LM rescue on, lenient = "
         f"unrecovered steps only; this rule's reading: {prim}) ---"]
    for r in ("literal", "lenient"):
        L.append(f"(1) per restart age, segment rows of that age only, {r} failure reading: t0 s | rows | D trimmed [95%] | "
                 f"{a} trim | {b} trim | fail calls {a} / {b}")
        for x in ph["per_age"][r]:
            L.append(f"    {x['t0']:8g} {x['rows']:6d}  {fs(x['D']):>26s}  {fa(x['A']):>7s}  {fa(x['B']):>7s}  "
                     f"{100 * x['fail_A']['call_rate']:.3f}% / {100 * x['fail_B']['call_rate']:.3f}%")
    L.append("(2) segment table restricted to restart rows (t0 > 0) / to t0 = 0 rows, five verdict checks:")
    for nm, o in ph["subsets"].items():
        for r in ("literal", "lenient"):
            t = o[r]
            v = t["verdict"]
            L.append(f"    {nm:18s} {r:8s} D trimmed {fs(t['mean12']['D'])} mean {f3(t['mean12']['D']['mean'])}  "
                     f"{a} {fa(t['mean12']['A'])} {b} {fa(t['mean12']['B'])}  max bin simult. hi "
                     f"{max(t['mean12']['band']['hi']):+.3f}  worst-gas D {fs(t['worst']['D'])}  fail "
                     f"{100 * t['fail_A']['call_rate']:.3f}% / {100 * t['fail_B']['call_rate']:.3f}%  -> {v['verdict']} "
                     f"[trim<0 {v['trim_upper_lt0']}, mean<0 {v['mean_lt0']}, bins {v['bins_upper_lt005']}, worst "
                     f"{v['worst_trim_upper_lt005']}, fail {v['fail_ok']}]")
    key = "node_quadrature" if D.exact else "exact_quadrature"
    L.append("(3) " + ("whole-node quadrature (rule v3's)" if D.exact else
                       "exact interval-splitting quadrature at decade edges (zero covered log-time = untested)") +
             " instead of this rule's:")
    for r in ("literal", "lenient"):
        for tab, t in ph[key][r].items():
            L.append(f"    {r:8s} {tab:4s} D trimmed {fs(t['mean12']['D'])} mean {f3(t['mean12']['D']['mean'])}  "
                     f"{a} {fa(t['mean12']['A'])} {b} {fa(t['mean12']['B'])}  max bin hi {max(t['mean12']['band']['hi']):+.3f}"
                     f"  worst-gas D {fs(t['worst']['D'])}  -> {t['verdict']['verdict']}")
    L.append(f"(5) time since restart (segments; per decade of the time since the row's own start t0, exact split in log10 "
             f"of that time, covered from the first saved call on; this rule's failure reading): decade | restart rows "
             f"(t0 > 0) D trimmed [95%], {a} trim, {b} trim | t0 = 0 rows D trimmed [95%], {a} trim, {b} trim")
    sr = ph["since_restart"]
    for j in range(6):
        x, y = sr["restart_only_t0>0"][j], sr["t0_only"][j]
        L.append(f"    1e{j}-1e{j + 1} s  {fs(x['D']):>26s}  {fa(x['A']):>6s}  {fa(x['B']):>6s}  |  {fs(y['D']):>26s}  "
                 f"{fa(y['A']):>6s}  {fa(y['B']):>6s}")
    if "first_call_moves" not in ph:
        return L
    L.append(f"(4) first call from t = 0 (segments; 12-gas mean of floored |log10 change|): start | rows | truth move median "
             f"(mean) | {a} move median (mean) | {a} error median (mean) | {a} moves < half the truth (rows with truth move "
             f"> 0.1) | {b} move median | {b} error median (mean)")
    for sn, x in ph["first_call_moves"].items():
        fr = x["frac_model_move_lt_half_truth"]
        L.append(f"    {sn:9s} {x['rows']:5d}  {x['truth_move_median']:.3f} ({x['truth_move_mean']:.3f})  "
                 f"{x['model_move_median']:.3f} ({x['model_move_mean']:.3f})  {x['model_err_median']:.3f} "
                 f"({x['model_err_mean']:.3f})  {('%.2f' % fr) if fr is not None else 'n/a'} of {x['rows_truth_move_gt0p1']}  "
                 f"{x['mc_move_median']:.3f}  {x['mc_err_median']:.3f} ({x['mc_err_mean']:.3f})")
    L.append(f"    medians along the t = 0 rows at calls k = {', '.join(map(str, KSER))} (truth move / {a} move / {a} error / "
             f"{b} error):")
    for sn, x in ph["first_call_moves"].items():
        L.append(f"    {sn:9s} " + "  ".join(f"k {kk}: {v['truth_move']:.3f}/{v['model_move']:.3f}/{v['model_err']:.3f}/"
                                             f"{v['mc_err']:.3f}" for kk, v in x["series_medians"].items()))
    return L


def inputs_lines(io):
    L = ["model input entries (of 25) at the 1e-30 floor, design start units (min / median / max, share with >= 14):"]
    for k, x in io["panel"].items():
        L.append(f"    {k:16s} units {x['units']:5d}  {x['min']} / {x['median']:.0f} / {x['max']}  >= 14: {100 * x['frac_ge14']:.1f}%")
    return L


def sensitivity(D, stA, stB):
    """Readouts: penalties 0.5 and 5 dex; the other rule's model failure reading (v3: side A lenient; v3.1: both model
    sides literal); v3: single edge calls dropped; v3.1: whole-node quadrature."""
    ffA, ffB = stA["ff"], stB["ff"]
    out = {}
    for pen in PENS:
        PA = profile(D, stA, ffA, pen=pen, stats=("mean12", "worst"))
        PB = profile(D, stB, ffB, pen=pen, stats=("mean12", "worst"))
        out[f"penalty_{pen:g}"] = mini_compare(D, PA, PB, ffA, ffB)
    if D.exact:
        fa_, fb_ = ff_reading(stA, "literal"), ff_reading(stB, "literal")
        PA = profile(D, stA, fa_, stats=("mean12", "worst"))
        PB = profile(D, stB, fb_, stats=("mean12", "worst"))
        out["literal_model_failure"] = mini_compare(D, PA, PB, fa_, fb_)
        PA = profile(D, stA, ffA, stats=("mean12", "worst"), exact=False)
        PB = profile(D, stB, ffB, stats=("mean12", "worst"), exact=False)
        out["node_quadrature"] = mini_compare(D, PA, PB, ffA, ffB)
        return out
    fa_ = stA["ff_lenient"]
    PA = profile(D, stA, fa_, stats=("mean12", "worst"))
    PB = profile(D, stB, ffB, stats=("mean12", "worst"))
    out["lenient_model_failure"] = mini_compare(D, PA, PB, fa_, ffB)
    keep = np.logical_not(edge_singletons(D))
    PA = profile(D, stA, ffA, stats=("mean12", "worst"), ptmask=keep)
    PB = profile(D, stB, ffB, stats=("mean12", "worst"), ptmask=keep)
    out["posthoc_no_edge_singletons"] = mini_compare(D, PA, PB, ffA, ffB)
    out["posthoc_no_edge_singletons"]["points_dropped"] = int(keep.size - keep.sum())
    return out


def mini_compare(D, PA, PB, ffA, ffB):
    o = {}
    for tab, cells in (("seg", D.cond_cell), ("free", D.fcond_cell)):
        t = {s: dict(D=summarize(PA[s][tab]["cond"] - PB[s][tab]["cond"], cells), A=summarize(PA[s][tab]["cond"], cells),
                     B=summarize(PB[s][tab]["cond"], cells)) for s in ("mean12", "worst")}
        t["mean12"]["band"] = band(PA["mean12"][tab]["bin"] - PB["mean12"][tab]["bin"], cells)
        t["mean12"]["start"] = [dict(D=summarize(PA["mean12"][tab]["start"][:, j] - PB["mean12"][tab]["start"][:, j], cells),
                                     A=summarize(PA["mean12"][tab]["start"][:, j], cells))
                                for j in range(PA["mean12"][tab]["start"].shape[1])]
        t["mean12"]["dec"] = [dict(D=summarize(PA["mean12"][tab]["dec"][:, j] - PB["mean12"][tab]["dec"][:, j], cells),
                                   A=summarize(PA["mean12"][tab]["dec"][:, j], cells)) for j in range(8)]
        if tab == "seg":
            t["mean12"]["anchor"] = dict(D=summarize(PA["mean12"]["seg"]["anchor"] - PB["mean12"]["seg"]["anchor"], cells),
                                         A=summarize(PA["mean12"]["seg"]["anchor"], cells),
                                         B=summarize(PB["mean12"]["seg"]["anchor"], cells))
        rows = D.seg_row if tab == "seg" else ~D.seg_row
        t["fail_A"], t["fail_B"] = fail_rates(D, ffA, rows), fail_rates(D, ffB, rows)
        t["verdict"] = verdict(t)
        o[tab] = t
    return o


def sens_lines(sens, a, b, starts):
    L = [f"--- sensitivity ({a} vs {b}; readouts) ---"]
    for k, o in sens.items():
        for tab in ("seg", "free"):
            t = o[tab]
            L.append(f"  {k:26s} {tab:4s} D trimmed {fs(t['mean12']['D'])}  mean {fs(t['mean12']['D'], 'mean')}  "
                     f"{a} {t['mean12']['A']['trim']:.4f}  {b} {t['mean12']['B']['trim']:.4f}  max bin hi "
                     f"{max(t['mean12']['band']['hi']):+.3f}  worst-gas D trimmed {fs(t['worst']['D'])}  "
                     f"fail {a} {100 * t['fail_A']['call_rate']:.4f}% {b} {100 * t['fail_B']['call_rate']:.4f}%  -> "
                     f"{t['verdict']['verdict']}")
    for k, o in sens.items():
        if k.startswith("penalty"):
            continue
        t = o["seg"]["mean12"]
        L.append(f"  {k} segments per start: " + "; ".join(
            f"{nm} D {fs(x['D'])} ({a} {fa(x['A'])})" for nm, x in zip(starts, t["start"])))
        L.append(f"  {k} segments per decade D trimmed: " +
                 "; ".join(f"1e{j} {f3(x['D']['trim'])}" for j, x in enumerate(t["dec"])))
        L.append(f"  {k} 10 s anchor: D {fs(t['anchor']['D'])}, {a} {fa(t['anchor']['A'])}, {b} {fa(t['anchor']['B'])}")
        tf = o["free"]["mean12"]
        L.append(f"  {k} free runs per start: " + "; ".join(
            f"{nm} D {fs(x['D'])}" for nm, x in zip(["eq-target", "raw-A"], tf["start"])) +
            "; per dt value D trimmed: " + ", ".join(f"{d:g} s {f3(v)}" for d, v in zip(FREE_DT, tf["band"]["T"])))
        if "points_dropped" in o:
            L.append(f"  {k}: saved points dropped {o['points_dropped']:,}")
    return L


def repr_lines(rep, a, b):
    L = [f"--- representable-start subset ({a} vs {b}; rows whose start unit is representable in mini-chem) ---"]
    for tab in ("seg", "free"):
        for s in ("mean12", "worst"):
            x = rep[tab][s]
            L.append(f"  {tab:4s} {s:7s} D trimmed {fs(x['D'])}  mean {fs(x['D'], 'mean')}  median {fs(x['D'], 'median')}  "
                     f"{a} {fa(x['A'])}  {b} {fa(x['B'])}  (n {x['D']['n']})")
    return L


def fail_detail(D, st):
    """Failure breakdown of one solver (this rule's reading) by table, start, cell, dt class, t0; the other model
    reading and the recorded recoveries beside it."""
    ff = st["ff"]
    o = {}
    tabs = (("seg", D.seg_row), ("free", np.logical_not(D.seg_row)))
    for nm, sel in tabs:
        o[nm] = fail_rates(D, ff, sel)
    f = ff > 0
    dtc = np.where(D.r["kind"] == 1, 9, np.where(D.r["dt_idx"] == 16, 8, D.r["dt_bin"]))
    for fld, labels in (("start", D.starts), ("cell", D.cells)):
        o["rows_by_" + fld] = {labels[i]: int((f & (D.r[fld] == i)).sum()) for i in range(len(labels))}
    o["rows_by_dt_class"] = {str(i): int((f & (dtc == i)).sum()) for i in range(10)}
    o["rows_by_age_idx"] = {str(i): int((f & (D.r["age_idx"] == i)).sum()) for i in range(17)}
    st_ = D.p["stratum"][D.r["pidx"]]
    o["rows_by_stratum"] = {STRAT[k]: int((f & (st_ == k)).sum()) for k in ("G", "C", "E")}
    fc = ff[f]
    o["first_call_pct"] = {str(p): float(np.percentile(fc, p)) for p in (10, 50, 90)} if f.any() else {}
    if "ff_lenient" in st:
        alt = "literal" if D.exact else "lenient"
        o[alt] = {nm: fail_rates(D, st["ff_" + alt], sel) for nm, sel in tabs}
        ex = st["extra"]
        o["rows_newton_any"] = int((ex["newton_n"] > 0).sum())
        o["rows_kept_any"] = int((ex["kept_n"] > 0).sum())
        o["rows_guard_any"] = int((ex["guard_n"] > 0).sum())
        o["rows_nonfin_any"] = int((ex["nonfin_n"] > 0).sum())
        o["newton_calls_total"] = int(ex["newton_n"].sum())
    else:
        o["recorded"] = st["recorded"]
    return o


def fail_lines(D, fd, name):
    L = [f"failures {name} ({D.rule} reading): segments {fd['seg']['failed_rows']:,} rows ({100 * fd['seg']['row_rate']:.3f}%), "
         f"{fd['seg']['failed_calls']:,} calls ({100 * fd['seg']['call_rate']:.4f}%); free runs {fd['free']['failed_rows']} rows, "
         f"{fd['free']['failed_calls']:,} calls ({100 * fd['free']['call_rate']:.4f}%)"]
    L.append(f"  rows by start {fd['rows_by_start']}; by stratum {fd['rows_by_stratum']}")
    L.append(f"  rows by cell {fd['rows_by_cell']}")
    L.append(f"  rows by dt class (0-7 bins, 8 = 10 s anchor, 9 = free runs) {fd['rows_by_dt_class']}")
    L.append(f"  rows by restart age index (0 = start, 1 = 1 s, 2 = 3 s, ...) {fd['rows_by_age_idx']}")
    L.append(f"  first failed call percentiles (10/50/90) {fd['first_call_pct']}")
    if "rows_newton_any" in fd:
        alt = "literal" if D.exact else "lenient"
        le = fd[alt]
        L.append(f"  recorded: rows with any LM rescue (projection Newton failure) {fd['rows_newton_any']:,}, kept input "
                 f"{fd['rows_kept_any']}, guard {fd['rows_guard_any']}, non-finite {fd['rows_nonfin_any']}; LM-rescue calls "
                 f"in all {fd['newton_calls_total']:,}")
        L.append(f"  {alt.upper()} reading (readout): segments {le['seg']['failed_rows']:,} rows, {le['seg']['failed_calls']:,} "
                 f"calls ({100 * le['seg']['call_rate']:.4f}%); free {le['free']['failed_rows']} rows")
    if "recorded" in fd:
        r = fd["recorded"]
        L.append(f"  recorded, not failures: rows with excess-work continuations {r['rows_excess_work']} ({r['excess_work_total']} "
                 f"in all), rows with DLSODE warnings {r['rows_dlsode_warning']} ({r['dlsode_warning_total']:,} in all); rows "
                 f"with a masked state (failures) {r['rows_masked_state']}")
    return L


def lm_lines(lm, name):
    L = [f"large misses {name} (non-failed saved calls): rows with 12-gas mean > 0.5 dex {lm['rows_mean12_gt0.5']:,} "
         f"(segments {lm['seg_rows_mean12_gt0.5']:,}, free {lm['free_rows_mean12_gt0.5']}), > 2 dex {lm['rows_mean12_gt2.0']:,}; "
         f"rows with any gas > 0.5 dex {lm['rows_anygas_gt0.5']:,} (free {lm['free_rows_anygas_gt0.5']}), > 2 dex "
         f"{lm['rows_anygas_gt2.0']:,} (free {lm['free_rows_anygas_gt2.0']})"]
    for nm in ("anygas_gt2", "mean12_gt0.5"):
        L.append(f"  breakdown rows {nm}: {lm['breakdown_' + nm]}")
    L.append("  worst rows by single-gas error (cell, stratum, T K, P bar, start, dt s, t0 s, call k, t s, gas, error dex, row max 12-gas mean):")
    for x in lm["top_anygas"][:15]:
        L.append(f"    {x['cell']:9s} {x['stratum']} {x['T']:7.1f} {x['P_bar']:9.2e} {x['start']:9s} dt {x['dt']:8.3f} "
                 f"t0 {x['t0']:8.2g} k {x['k_worst']:5d} t {x['tabs_worst']:9.3g} {x['gas']:5s} {x['gas_err']:6.2f}  "
                 f"{x['max_mean12']:5.2f}")
    L.append("  worst rows by 12-gas mean:")
    for x in lm["top_mean12"][:10]:
        L.append(f"    {x['cell']:9s} {x['stratum']} {x['T']:7.1f} {x['P_bar']:9.2e} {x['start']:9s} dt {x['dt']:8.3f} "
                 f"t0 {x['t0']:8.2g} k {x['k_mean12']:5d} t {x['tabs_mean12']:9.3g} mean12 {x['max_mean12']:5.2f}")
    return L


def cov_lines(D, ct):
    L = ["coverage per (dt bin, decade): log-time fraction covered (mean over cases) / fraction of cases in which the "
         "decade is tested (" + ("positive covered log-time" if D.exact else ">= 1 saved call in it") + ")"]
    L.append("  " + f"{'bin':10s}" + "".join(f"{'1e%d' % d:>13s}" for d in range(8)))
    for x in ct["seg"]:
        lab = f"{int(D.bins[x['bin']][0])}-{int(D.bins[x['bin']][1])} s" if x["bin"] != "anchor" else "10 s anch"
        L.append("  " + f"{lab:10s}" + "".join(f"{c:6.3f}/{h:5.2f} " for c, h in zip(x["cov"], x["covered"])))
    for x in ct["free"]:
        L.append("  " + f"free {x['dt']:<5g}" + "".join(f"{c:6.3f}/{h:5.2f} " for c, h in zip(x["cov"], x["covered"])))
    L.append("  restarts per case in the decade (mean) and longest uninterrupted span (dex, mean):")
    for x in ct["seg"]:
        lab = f"{int(D.bins[x['bin']][0])}-{int(D.bins[x['bin']][1])} s" if x["bin"] != "anchor" else "10 s anch"
        L.append("  " + f"{lab:10s}" + "".join(f"{r:5.2f}/{g:6.3f} " for r, g in zip(x["restarts"], x["longest"])))
    return L


def rtol_lines(rt, a):
    L = [f"mini-chem rtol 1e-5 subset ({rt['rows']:,} rows): per-row log-time mean of the primary statistic vs VULCAN, "
         f"rtol 1e-3 {rt['row_mean_1e3']:.4f} vs 1e-5 {rt['row_mean_1e5']:.4f} dex (mean diff {rt['diff_mean']:+.5f}); "
         f"|diff| p50/p90/p99/p99.9 " + "/".join(f"{v:.2e}" for v in rt["absdiff_pct"].values()) +
         f", max {rt['absdiff_max']:.3f}; 1e-5 better in {100 * rt['frac_1e5_better']:.1f}% of rows, worse in "
         f"{100 * rt['frac_1e5_worse']:.1f}%; failed rows 1e-3 {rt['fail_rows_1e3']}, 1e-5 {rt['fail_rows_1e5']}"]
    for tab, o in rt["substituted"].items():
        L.append(f"  {tab}: mini-chem trimmed score {fs(o['mc_1e3'])} -> with the subset at 1e-5 {fs(o['mc_sub'])}; "
                 f"D({a}) trimmed {fs(o['D_1e3'])} -> {fs(o['D_sub'])}")
    return L


# ------------------------------------------------------------------------------------------------------- driver
def run(design, run_dir, out_dir, rule, models):
    t_start = time.time()
    mc = run_dir / "minichem"
    out_dir.mkdir(parents=True, exist_ok=True)
    D = Design(design, rule)
    ct = coverage_table(D, coverage(D))
    log("coverage done")
    stB = solver_mc(D, mc)
    PB = profile(D, stB, stB["ff"])
    log("mini-chem profile done")
    out = dict(rule=RULE_TEXT[rule], rule_id=rule, generated=time.strftime("%F %T"), coverage=ct, comparisons={},
               failures={}, large_misses={}, sensitivity={}, representable={})
    out["failures"]["minichem"] = fail_detail(D, stB)
    out["large_misses"]["minichem"] = large_misses(D, stB, stB["ff"], "minichem", out_dir)
    sts, profs = {}, {}
    for m in models:
        sts[m] = solver_model(D, run_dir, m)
        profs[m] = profile(D, sts[m], sts[m]["ff"])
        out["failures"][m] = fail_detail(D, sts[m])
        out["large_misses"][m] = large_misses(D, sts[m], sts[m]["ff"], m, out_dir)
        log(m, "profile done")
    rep_mask = D.representable_row
    PB_rep = profile(D, stB, stB["ff"], rowmask=rep_mask, stats=("mean12", "worst"))
    out["posthoc_inputs"], inputs = inputs_readout(D)
    out["posthoc"] = {}
    pairs = [(m, "minichem") for m in models] + [(models[0], m) for m in models[1:]]
    for a, b in pairs:
        stA, PA = sts[a], profs[a]
        if b == "minichem":
            stb, Pb, Pb_rep = stB, PB, PB_rep
        else:
            stb, Pb = sts[b], profs[b]
            Pb_rep = profile(D, stb, stb["ff"], rowmask=rep_mask, stats=("mean12", "worst"))
        res = compare(D, PA, Pb, stA["ff"], stb["ff"], a, "mini-chem" if b == "minichem" else b)
        key = f"{a}_vs_{b}"
        out["comparisons"][key] = res
        out["sensitivity"][key] = sensitivity(D, stA, stb)
        PA_rep = profile(D, stA, stA["ff"], rowmask=rep_mask, stats=("mean12", "worst"))
        out["representable"][key] = {tab: {s: dict(D=summarize(PA_rep[s][tab]["cond"] - Pb_rep[s][tab]["cond"], cells),
                                                   A=summarize(PA_rep[s][tab]["cond"], cells),
                                                   B=summarize(Pb_rep[s][tab]["cond"], cells)) for s in ("mean12", "worst")}
                                     for tab, cells in (("seg", D.cond_cell), ("free", D.fcond_cell))}
        out["posthoc"][key] = posthoc(D, stA, stb, inputs, b == "minichem")
        log(key, "compared:", res["headline"])
    a0 = models[0]
    out["rtol"] = rtol_readout(D, mc, stB, stB["ff"], PB, profs[a0])
    out["rtol_vs_model"] = a0
    out["representable_rows"] = dict(rows=int(rep_mask.sum()), of=int(D.R), seg=int(rep_mask[D.seg_row].sum()),
                                     free=int(rep_mask[~D.seg_row].sum()))
    out["seconds"] = time.time() - t_start
    out["models_scored"] = list(models)
    # ---- text
    L = [out["rule"], f"generated {out['generated']}; models scored: {', '.join(models)}", ""]
    L.append("SUMMARY (primary statistic, D = A - B, trimmed mean [95%]; absolute = trimmed mean of condition scores vs VULCAN)")
    for key, res in out["comparisons"].items():
        s, f = res["seg"], res["free"]
        L.append(f"  {res['a']:5s} vs {res['b']:9s}: {res['headline']['verdict'].upper():15s} segments D {fs(s['mean12']['D'])} "
                 f"({res['a']} {s['mean12']['A']['trim']:.4f}, {res['b']} {s['mean12']['B']['trim']:.4f}; fail "
                 f"{100 * s['fail_A']['call_rate']:.3f}% / {100 * s['fail_B']['call_rate']:.3f}%) | free runs D "
                 f"{fs(f['mean12']['D'])} ({f['mean12']['A']['trim']:.4f} / {f['mean12']['B']['trim']:.4f}) | together D "
                 f"{fs(res['tog']['mean12']['D'])}")
    L.append("")
    for key, res in out["comparisons"].items():
        L += table_lines(D, res)
        L.append("")
        L += sens_lines(out["sensitivity"][key], res["a"], res["b"], D.starts)
        L.append("")
        L += repr_lines(out["representable"][key], res["a"], res["b"])
        L.append("")
        L += posthoc_lines(D, out["posthoc"][key], res["a"], res["b"])
        L.append("")
    L.append("=== MODEL INPUTS AT THE FLOOR ===")
    L += inputs_lines(out["posthoc_inputs"])
    L.append("")
    L.append("=== FAILURES ===")
    L += fail_lines(D, out["failures"]["minichem"], "mini-chem")
    for m in models:
        L += fail_lines(D, out["failures"][m], m)
    L.append("")
    L.append("=== LARGE MISSES (identities in large_misses_<solver>.csv) ===")
    L += lm_lines(out["large_misses"]["minichem"], "mini-chem")
    for m in models:
        L += lm_lines(out["large_misses"][m], m)
    L.append("")
    L.append("=== MINI-CHEM TOLERANCE (rtol 1e-5 subset) ===")
    L += rtol_lines(out["rtol"], a0)
    L.append("")
    L.append(f"=== COVERAGE === representable-start rows: {out['representable_rows']}")
    L += cov_lines(D, ct)
    (out_dir / "tables.txt").write_text("\n".join(L) + "\n")
    write_json(out_dir / "score.json", out)
    log("written", out_dir, f"{out['seconds']:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--design", type=Path, required=True, help="design folder (rows, saved, truth_saved, parents, units)")
    ap.add_argument("--run", type=Path, required=True, help="run folder (minichem/ and one folder per model)")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--rule", choices=("v3.1", "v3"), default="v3.1")
    ap.add_argument("models", nargs="+", help="model folders under --run; the first is also compared with each other")
    a = ap.parse_args()
    for m in a.models:
        assert (a.run / m / "model_states.npz").exists(), f"no {a.run / m / 'model_states.npz'}"
    run(a.design, a.run, a.out, a.rule, a.models)


if __name__ == "__main__":
    main()
