"""Score a benchmark directory written by lhbench.py.

    python score.py <bench_dir>      -> <bench_dir>/score.json (+ printed headline)

Errors are |log10 y_model - log10 y_true| (dex) on solver-resolved truth values (x * n0 >= 1e-2 cm^-3, VULCAN's
atol; n0 = P / (k T), P in dyn/cm^2), pooled over species and rows within a T stratum. Truth: the VULCAN reruns
(no early stop) where present (primary), else the raw log-log interpolant (arm A secondary). Persistence (the start
state held) is scored the same way. The verdict per precision mode (pre-registered, provisional thresholds):
  eligible = zero guard interventions in the nominal arms (A, B, C, E) and every A (VULCAN truth) and E stratum at
  every checkpoint with p90 <= 0.1 and p99 <= 0.3 dex, and the hold (B) displacement p90 <= 0.02 dex in every
  stratum at every checkpoint, and cool strata (T < 1300 K) below persistence at the final A checkpoint.
  robust = zero interventions in the stress arm D (reported apart; host repair is not an intervention).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402
import lhbench as L  # noqa: E402

KB = 1.380649e-16
MODES = ("P1", "P2", "P3", "P4")    # P4 only in --modes P2P3P4 runs (4 state slots)
RAD = ["H2O", "CH4", "CO", "CO2", "NH3", "HCN"]
PASS_P90, PASS_P99, PASS_HOLD = 0.1, 0.3, 0.02
_VCACHE = {}


def vtruth(tag, job, times):
    key = (tag, int(job))
    if key not in _VCACHE:
        r = np.load(L.VREF / tag / f"{int(job)}.npz")
        sp = list(r["species"])
        _VCACHE[key] = (r["t"], r["x"][:, [sp.index(s) for s in D.SHORT]])
    t, x = _VCACHE[key]
    return np.log10(np.maximum(np.stack([np.interp(times, t, x[:, j]) for j in range(x.shape[1])], -1), D.EPS))


_CLOSABLE = None
# Atoms per molecule (C, H, O, N) for the 12 tracked species (data.SHORT order is looked up by name).
STOICH = {"C2H2": (2, 2, 0, 0), "CH4": (1, 4, 0, 0), "CO2": (1, 0, 2, 0), "CO": (1, 0, 1, 0), "H2O": (0, 2, 1, 0),
          "H2": (0, 2, 0, 0), "HCN": (1, 1, 0, 1), "H": (0, 1, 0, 0), "N2": (0, 0, 0, 2), "NH3": (0, 3, 0, 1),
          "OH": (0, 1, 1, 0), "O": (0, 0, 1, 0)}
ELEMENTS = ("C", "H", "O", "N")


class _Stoich(dict):  # E13: any other species (53-species sets) from its formula, (C, H, O, N); He -> zeros
    def __missing__(self, sp):
        import re
        n = lambda e: sum(int(k or 1) for el, k in re.findall(r"([A-Z][a-z]?)(\d*)", sp) if el == e)
        return tuple(n(e) for e in ELEMENTS)


STOICH = _Stoich(STOICH)


def element_budget(st3, tru, T):
    """Element-budget error against the truth's own totals at the same time (the tracked species are not
    element-closed, so this is not conservation drift): |E_model - E_true| / E_true per element, pooled over rows
    within each T stratum, at every checkpoint. st3 [C, R, S] log10 y (one mode), tru [R, C, S]."""
    A = np.array([STOICH[sp] for sp in D.SHORT], dtype=np.float64)          # [S, 4]
    Em = np.einsum("crs,se->rce", 10.0 ** st3, A)
    Et = np.einsum("rcs,se->rce", 10.0 ** tru, A)
    rel = np.abs(Em - Et) / np.maximum(Et, 1e-300)
    tb = tbin(T)
    out = {}
    for b, sname in enumerate(stratum_names()):
        rb = tb == b
        if not rb.any():
            continue
        out[sname] = {el: {"p90": [pct(rel[rb, c, k], 90) for c in range(rel.shape[1])], "final_p90": pct(rel[rb, -1, k], 90)}
                      for k, el in enumerate(ELEMENTS)}
    return out


def closable():
    """[5 T strata, S] bool: species whose closure floor (p90 |dlog10| between a full-state VULCAN restart and the
    same restart with the untracked species removed, max over 1e2/1e4/1e6 s; reference (b)) is <= PASS_P90. The
    floor's 3 T bins map onto the 5 benchmark strata (cool: 300-1300 K, mid: 1300-1800 K, hot: >= 1800 K)."""
    global _CLOSABLE
    if _CLOSABLE is not None:
        return _CLOSABLE
    jobs = json.loads((L.VREF / "ref_b" / "jobs.json").read_text())
    times = np.array([1e2, 1e4, 1e6])
    D, T = [], []
    for i in range(0, len(jobs), 2):
        a = jobs[i]
        la, lb = vtruth("ref_b", i, times), vtruth("ref_b", i + 1, times)
        res = la + np.log10(a["P"] / (KB * a["T"])) >= -2.0
        D.append(np.where(res, np.abs(la - lb), np.nan))
        T.append(a["T"])
    D, T = np.array(D), np.array(T)
    floor = {}
    for name, lo, hi in (("cool", 300, 1300), ("mid", 1300, 1800), ("hot", 1800, 3001)):
        m = (T >= lo) & (T < hi)
        floor[name] = np.nanmax(np.nanpercentile(D[m], 90, axis=0), axis=0)   # [S]
    by_stratum = ["cool", "cool", "mid", "hot", "hot"]
    _CLOSABLE = np.stack([floor[k] <= PASS_P90 for k in by_stratum])
    return _CLOSABLE


def pct(x, q):
    return float(np.percentile(x, q)) if x.size else float("nan")


def tbin(T):
    return np.clip(np.searchsorted(L.T_EDGES, T, side="right") - 1, 0, len(L.T_EDGES) - 2)


def stratum_names():
    return [f"{L.T_EDGES[b]}-{L.T_EDGES[b + 1]}K" for b in range(len(L.T_EDGES) - 1)]


def score_rows(st, tru, P, T, pers=None):
    """st [C, 3, R, S] model log10 y; tru [R, C, S]; P, T [R]; pers [R, C, S] persistence |error| or None."""
    res = (tru + np.log10(P / (KB * T))[:, None, None]) >= -2.0
    tb = tbin(T)
    C = tru.shape[1]
    out = {}
    for m, name in enumerate(MODES[:st.shape[1]]):
        if np.isnan(st[:, m]).all():
            continue  # mode not run (P3-only benchmark)
        sgn = st[:, m].transpose(1, 0, 2) - tru
        e = np.abs(sgn)
        mo = {"strata": {}}
        for b, sname in enumerate(stratum_names()):
            rb = tb == b
            if not rb.any():
                continue
            cell = lambda arr, c: arr[rb, c][res[rb, c]]
            p90 = [pct(cell(e, c), 90) for c in range(C)]
            p99 = [pct(cell(e, c), 99) for c in range(C)]
            cm = closable()[b]
            cellc = lambda arr, c: arr[rb, c][:, cm][res[rb, c][:, cm]]
            p90c = [pct(cellc(e, c), 90) for c in range(C)]
            s = {"n_rows": int(rb.sum()), "p50": [pct(cell(e, c), 50) for c in range(C)], "p90": p90, "p99": p99,
                 "p90_closable": p90c, "max_p90_closable": float(np.nanmax(p90c)), "closable_species": [D.SHORT[j] for j in np.flatnonzero(cm)],
                 "max_p90": float(np.nanmax(p90)), "max_p99": float(np.nanmax(p99)), "final_p90": p90[-1],
                 "final_bias_rad": {sp: (float(np.median(sgn[rb, -1, j][res[rb, -1, j]])) if res[rb, -1, j].any() else None)
                                    for j, sp in enumerate(D.SHORT) if sp in RAD}}
            if pers is not None:
                pp = [pct(cell(pers, c), 90) for c in range(C)]
                s["persist_p90"] = pp
                s["final_persist_p90"] = pp[-1]
            mo["strata"][sname] = s
        vals = list(mo["strata"].values())
        mo["worst_max_p90"] = max(v["max_p90"] for v in vals)
        mo["worst_max_p90_closable"] = max(v["max_p90_closable"] for v in vals)
        mo["worst_max_p99"] = max(v["max_p99"] for v in vals)
        mo["worst_final_p90"] = max(v["final_p90"] for v in vals)
        mo["pass_accuracy"] = bool(mo["worst_max_p90"] <= PASS_P90 and mo["worst_max_p99"] <= PASS_P99)
        if pers is not None:
            cool = [v for k, v in mo["strata"].items() if k.startswith(("300-", "800-"))]
            mo["cool_beats_persistence_final"] = bool(all(v["final_p90"] < v["final_persist_p90"] for v in cool))
        out[name] = mo
    return out


def interventions(d):
    return {"rows_with_interventions": (d["nint"] > 0).sum(1).tolist(), "nint": d["nint"].sum(1).tolist(),
            "first_call": [int(x[x >= 0].min()) if (x >= 0).any() else -1 for x in d["first"]]}


def score_A(f, test, ref_a_rows):
    d = np.load(f)
    ridx, t0, dt, calls, st = d["ridx"], d["t0"], float(d["dt"]), d["calls"], d["states"]
    times = t0[:, None] + calls[None, :] * dt
    with torch.no_grad():
        raw = test.logy_at(torch.from_numpy(ridx), torch.from_numpy(times)).numpy()
        raw0 = test.logy_at(torch.from_numpy(ridx), torch.from_numpy(t0[:, None])).numpy()[:, 0]
    P, T = test.P[ridx], test.T[ridx]
    out = {"dt": dt, "calls": calls.tolist(), **interventions(d)}
    out["raw"] = score_rows(st, raw, P, T, np.abs(raw0[:, None, :] - raw))
    have = np.array([int(r) in ref_a_rows for r in ridx])
    if have.any():
        k = np.flatnonzero(have)
        vt = np.stack([vtruth("ref_a", ref_a_rows[int(ridx[i])], times[i]) for i in k])
        v0 = np.stack([vtruth("ref_a", ref_a_rows[int(ridx[i])], t0[i:i + 1])[0] for i in k])
        out["vulcan"] = score_rows(st[:, :, k], vt, P[k], T[k], np.abs(v0[:, None, :] - vt))
        out["vulcan_rows"] = int(k.size)
        if not np.isnan(st[:, 2]).all():
            out["element_budget_P3"] = element_budget(st[:, 2][:, k], vt, T[k])
    return out


def score_B(f, test):
    d = np.load(f)
    st, calls, ridx, ts = d["states"], d["calls"], d["ridx"], float(d["t_start"])
    tru = np.stack([vtruth("ref_a", j, ts + calls * 10.0) for j in d["job"]])
    tru0 = np.stack([vtruth("ref_a", j, np.array([ts]))[0] for j in d["job"]])
    out = {"calls": calls.tolist(), "t_start": ts, **interventions(d),
           "modes": score_rows(st, tru, test.P[ridx], test.T[ridx], np.abs(tru0[:, None, :] - tru))}
    for v in out["modes"].values():
        v["pass_hold"] = bool(v["worst_max_p90"] <= PASS_HOLD)
    return out


def score_V(f):
    d = np.load(f)
    st, calls, dt, tag = d["states"], d["calls"], float(d["dt"]), str(d["tag"])
    tru = np.stack([vtruth(tag, j, calls * dt) for j in d["job"]])
    tru0 = np.stack([vtruth(tag, j, np.array([0.0]))[0] for j in d["job"]])
    return {"calls": calls.tolist(), "dt": dt, **interventions(d),
            "modes": score_rows(st, tru, d["P"], d["T"], np.abs(tru0[:, None, :] - tru))}


def score_D(f):
    d = np.load(f)
    st, calls = d["states"], d["calls"]
    i9 = int(np.argmin(np.abs(calls - 0.9 * calls[-1])))
    out = {**interventions(d), "modes": {}}
    for m, name in enumerate(MODES[:st.shape[1]]):
        s = st[:, m]
        if np.isnan(s).all():
            continue
        out["modes"][name] = {"min_log10": float(np.nanmin(s)), "max_log10": float(np.nanmax(s)),
                              "all_finite": bool(np.isfinite(s).all()),
                              "terminal_change_p90": pct(np.abs(s[-1] - s[i9]), 90)}
    return out


# ---- physical-consistency metrics of the new arms (definitions of the G2-A2 audit: 12-species sum against the
# training-data band normalization.json species_sum; element ratios log10((E_a + 1e-12) / (E_b + 1e-12)) from the
# 12 tracked species, model against truth over the same 12 species)
_SS = D.NORM["species_sum"]
SUM_UP = [("gt_datamax", _SS["max"]), ("gt_1.001", 1.001), ("gt_1.01", 1.01), ("gt_1.05", 1.05), ("gt_1.1", 1.1), ("gt_2", 2.0)]
SUM_LOW = [("lt_datamin", _SS["min"]), ("lt_q0.001", _SS["quantiles"]["0.001"]), ("lt_q0.01", _SS["quantiles"]["0.01"])]
EFLOOR = 1e-12
RATIOS = (("C", "H"), ("O", "H"), ("N", "H"), ("C", "O"))
A12 = np.array([STOICH[sp] for sp in D.SHORT], dtype=np.float64)       # [S, 4] C, H, O, N


def sum12_stats(st, late=None):
    """st [C, R, S] log10 y of one mode; late [C] bool (tau >= 1e3 s) or None."""
    s12 = (10.0 ** st).sum(-1)                                             # [C, R]
    floor = (st <= np.log10(D.EPS) + 1e-3).all(-1)
    ever = (s12 > 1.01).any(0)
    out = {"max_sum": float(s12.max()), "min_sum": float(s12.min()), "p99_sum": pct(s12.ravel(), 99),
           "final_frac_gt_1.01": float((s12[-1] > 1.01).mean()), "rows_ever_gt_1.01": int(ever.sum()), "rows": int(s12.shape[1]),
           "n_floor_rowckpt": int(floor.sum()), "rows_ever_all_floor": int(floor.any(0).sum()),
           "frac_in_band_pm0.05": float(((s12 >= _SS["min"] - 0.05) & (s12 <= _SS["max"] + 0.05)).mean())}
    for k, v in SUM_UP:
        out[f"frac_{k}"] = float((s12 > v).mean())
    for k, v in SUM_LOW:
        out[f"frac_{k}"] = float((s12 < v).mean())
    if late is not None and late.any():
        out["late_frac_gt_1.001"] = float((s12[late] > 1.001).mean())
        out["late_frac_gt_1.01"] = float((s12[late] > 1.01).mean())
        out["late_max_sum"] = float(s12[late].max())
    return out


def ratio_logs(E):
    return {f"{a}/{b}": np.log10((E[..., ELEMENTS.index(a)] + EFLOOR) / (E[..., ELEMENTS.index(b)] + EFLOOR)) for a, b in RATIOS}


def element_ratio_err(st, tru, T, late=None):
    """|ratio_model - ratio_truth| (dex), 12 species each. st [C, R, S] one mode, tru [R, C, S] log10 truth."""
    rm = ratio_logs(np.einsum("crs,se->rce", 10.0 ** st, A12))
    rt = ratio_logs(np.einsum("rcs,se->rce", 10.0 ** tru, A12))
    tb = tbin(T)
    out = {}
    for k in rm:
        d = np.abs(rm[k] - rt[k])                                          # [R, C]
        per = {}
        for b, sname in [(None, "all")] + list(enumerate(stratum_names())):
            rb = np.ones(d.shape[0], bool) if b is None else tb == b
            if not rb.any():
                continue
            p90 = np.array([pct(d[rb, c], 90) for c in range(d.shape[1])])
            per[sname] = {"final_p90": float(p90[-1]), "max_ckpt_p90": float(np.nanmax(p90)),
                          "late_max_p90": float(np.nanmax(p90[late])) if late is not None and late.any() else None,
                          "frac_gt_0.1": float((d[rb] > 0.1).mean())}
        bands = [v for kk, v in per.items() if kk != "all"]
        per["worst_band_final_p90"] = max(v["final_p90"] for v in bands)
        per["worst_band_late_max_p90"] = max(v["late_max_p90"] for v in bands) if late is not None and late.any() else None
        out[k] = per
    return out


def late_summary(mo, tau):
    """Campaign, late-max (tau >= 1e3 s), early-max and final worst-band p90 (G2-A3 definitions) per mode."""
    late = np.asarray(tau) >= 1e3
    out = {}
    for m, v in mo.items():
        g = np.array([s["p90"] for s in v["strata"].values()])            # [bands, C]
        gc = np.array([s["p90_closable"] for s in v["strata"].values()])
        out[m] = {"campaign": float(np.nanmax(g)), "late_max": float(np.nanmax(g[:, late])) if late.any() else None,
                  "early_max": float(np.nanmax(g[:, ~late])) if (~late).any() else None, "final": float(np.nanmax(g[:, -1])),
                  "late_max_closable": float(np.nanmax(gc[:, late])) if late.any() else None}
    return out


def _consistency(st, tru, T, tau):
    late = np.asarray(tau) >= 1e3
    out = {}
    for m, name in enumerate(MODES[:st.shape[1]]):
        if np.isnan(st[:, m]).all():
            continue
        out[name] = {"sum12": sum12_stats(st[:, m], late)}
        if tru is not None:
            out[name]["element_ratio"] = element_ratio_err(st[:, m], tru, T, late)
    return out


def score_truth_arm(st, tru, P, T, pers, tau):
    out = score_rows(st, tru, P, T, pers)
    return {"modes": out, "summary": late_summary(out, tau), "consistency": _consistency(st, tru, T, tau)}


def score_Vt(f, test, ref_a_rows):
    """Arm V: A-arm rows with per-row elapsed time; VULCAN (a) truth at each row's own time."""
    d = np.load(f)
    st, ridx, t0, tcum = d["states"], d["ridx"], d["t0"], d["tcum"]
    times = t0[:, None] + tcum.T                                           # [R, C]
    tau = np.median(tcum, 1)
    k = np.flatnonzero([int(r) in ref_a_rows for r in ridx])
    vt = np.stack([vtruth("ref_a", ref_a_rows[int(ridx[i])], times[i]) for i in k])
    v0 = np.stack([vtruth("ref_a", ref_a_rows[int(ridx[i])], t0[i:i + 1])[0] for i in k])
    out = {"schedule": str(d["schedule"]), "calls": d["calls"].tolist(), "tau_median": tau.tolist(),
           "tau_min": tcum.min(1).tolist(), "tau_max": tcum.max(1).tolist(), "vulcan_rows": int(k.size), **interventions(d)}
    out.update(score_truth_arm(st[:, :, k], vt, test.P[ridx[k]], test.T[ridx[k]], np.abs(v0[:, None, :] - vt), tau))
    out["by_start"] = {f"{s:g}": late_summary(score_rows(st[:, :, k[t0[k] == s]], vt[t0[k] == s], test.P[ridx[k[t0[k] == s]]],
                                                         test.T[ridx[k[t0[k] == s]]]), tau) for s in np.unique(t0[k])}
    return out


def score_H(f):
    """Arm H: start from a VULCAN tag's own states; truth = the same tag at t0 + calls * dt."""
    d = np.load(f)
    st, job, t0, dt, tag, calls = d["states"], d["job"], d["t0"], float(d["dt"]), str(d["tag"]), d["calls"]
    el = d["tcum"].T if "tcum" in d.files else calls[None, :] * dt   # 2026-10-08: a dt schedule saves per-row tcum
    times = t0[:, None] + el
    tau = np.median(np.broadcast_to(el, times.shape), 0)
    vt = np.stack([vtruth(tag, j, times[i]) for i, j in enumerate(job)])
    v0 = np.stack([vtruth(tag, j, t0[i:i + 1])[0] for i, j in enumerate(job)])
    out = {"tag": tag, "dt": dt, "calls": calls.tolist(), "rows": int(job.size), **interventions(d)}
    if "substeps" in d:
        out["substeps"] = int(d["substeps"])
    out.update(score_truth_arm(st, vt, d["P"], d["T"], np.abs(v0[:, None, :] - vt), tau))
    out["by_start"] = {f"{s:g}": late_summary(score_rows(st[:, :, t0 == s], vt[t0 == s], d["P"][t0 == s], d["T"][t0 == s]), tau)
                       for s in np.unique(t0)}
    return out


def score_G(f):
    """Arm G (P3): per group, the 12-species sum through the stress and recovery phases, the first call with
    sum_12 > 1.01, all-floor rows, guard interventions, and the distance of the recovered state from the row's clean
    twin (|dlog10 y| on species with twin y >= 1e-20; element ratios likewise)."""
    d = np.load(f)
    st, calls, names, grp = d["states"][:, 2], d["calls"], [str(x) for x in d["groups"]], d["group"]
    n_s, n_r = int(d["n_stress"]), int(d["n_recover"])
    s12 = (10.0 ** st).sum(-1)                                             # [C, R]
    stress = calls <= n_s
    i_s = int(np.flatnonzero(calls == n_s)[0])
    tw_of = {}
    if "twin_rows" in d:
        tw_of = {int(r): i for i, r in enumerate(d["twin_rows"])}
        tws = d["twin_states"][:, 2]                                       # [C, Rt, S]
    out = {"n_stress": n_s, "n_recover": n_r, "calls": calls.tolist(), "groups": {}}
    for gi, name in enumerate(names):
        rows = np.flatnonzero(grp == gi)
        xf = d["xfirst"][rows]
        go = {"amp": d["amp"][rows[0]].tolist(), "renorm": bool(d["renorm"][rows[0]]), "rows": int(rows.size),
              "rows_with_interventions": int((d["nint"][2, rows] > 0).sum()), "nint": int(d["nint"][2, rows].sum()),
              "stress": sum12_stats(st[stress][:, rows]),
              "stress_end": sum12_stats(st[i_s:i_s + 1, rows]),
              "rows_exc_1.01_during_stress": int(((xf > 0) & (xf <= n_s)).sum()),
              "median_first_exc_call": float(np.median(xf[xf > 0])) if (xf > 0).any() else None,
              "frac_calls_exc_1.01": float(d["xcount"][rows].sum() / (rows.size * (n_s + n_r))),
              "renorm_count": int(d["nren"][rows].sum()), "renorm_rows": int((d["nren"][rows] > 0).sum())}
        if n_r > 0:
            go["recovery_end"] = sum12_stats(st[-1:, rows])
            go["recovery_curve_frac_gt_1.01"] = {int(c): float((s12[ci, rows] > 1.01).mean()) for ci, c in enumerate(calls) if c >= n_s}
            # chemistry-only phase: element-ratio drift from the stress end to the recovery end (blending changes a
            # cell's element budget, so the twin distance below includes legitimate composition change; this does not)
            ra, rb = ratio_logs(10.0 ** st[i_s, rows] @ A12), ratio_logs(10.0 ** st[-1, rows] @ A12)
            go["recovery_ratio_drift_p90"] = {k: pct(np.abs(rb[k] - ra[k]), 90) for k in ra}
            go["recovery_ratio_drift_max"] = {k: float(np.abs(rb[k] - ra[k]).max()) for k in ra}
            i_t = len(calls) - 2                                           # terminal settling: last two checkpoints
            dlast = np.abs(st[-1, rows] - st[i_t, rows])
            go["recovery_terminal_change"] = {"calls": [int(calls[i_t]), int(calls[-1])], "p90": pct(dlast[st[-1, rows] >= -20.0], 90),
                                              "rowmax_p90": pct(np.where(st[-1, rows] >= -20.0, dlast, 0.0).max(-1), 90)}
        if tw_of and all(int(r) in tw_of for r in rows):
            ti = np.array([tw_of[int(r)] for r in rows])
            for lab, ci in (("stress_end", i_s), ("recovery_end", len(calls) - 1)):
                a, b = st[ci, rows], tws[ci, ti]
                m = b >= -20.0
                dd = np.where(m, np.abs(a - b), np.nan)
                with np.errstate(all="ignore"):
                    rowmax = np.nanmax(np.where(m.any(-1, keepdims=True), dd, 0.0), -1)
                ea, eb = ratio_logs(10.0 ** a @ A12), ratio_logs(10.0 ** b @ A12)
                go[f"twin_dist_{lab}"] = {"rowmax_p50": pct(rowmax, 50), "rowmax_p90": pct(rowmax, 90), "rowmax_max": float(rowmax.max()),
                                          "pooled_p90": pct(dd[m], 90), "frac_rows_rowmax_gt_0.1": float((rowmax > 0.1).mean()),
                                          "ratio_p90": {k: pct(np.abs(ea[k] - eb[k]), 90) for k in ea},
                                          "ratio_frac_gt_0.1": {k: float((np.abs(ea[k] - eb[k]) > 0.1).mean()) for k in ea}}
            go["twin_sum12_recovery_end"] = sum12_stats(tws[-1:, ti])
            ta, tb_ = ratio_logs(10.0 ** tws[i_s, ti] @ A12), ratio_logs(10.0 ** tws[-1, ti] @ A12)
            go["twin_ratio_drift_p90_same_calls"] = {k: pct(np.abs(tb_[k] - ta[k]), 90) for k in ta}
            go["twin_rows_with_interventions"] = int((d["twin_nint"][2, ti] > 0).sum())
        out["groups"][name] = go
    return out


def consistency_dir(bench, dst):
    """sum_12 and element-ratio metrics plus the G2-A3 summary for every truth-bearing standard arm of a bench
    directory (stored controls are read-only: the result goes to dst)."""
    bench = Path(bench)
    test = D.RawSplit("test", torch.device("cpu"))
    ref_a_rows = {j["row"]: i for i, j in L.vref_index("ref_a").items()}
    out = {"bench": str(bench)}
    for f in sorted(bench.glob("A_dt*.npz")):
        d = np.load(f)
        ridx, t0, dt, calls, st = d["ridx"], d["t0"], float(d["dt"]), d["calls"], d["states"]
        k = np.flatnonzero([int(r) in ref_a_rows for r in ridx])
        times = t0[:, None] + calls[None, :] * dt
        vt = np.stack([vtruth("ref_a", ref_a_rows[int(ridx[i])], times[i]) for i in k])
        v0 = np.stack([vtruth("ref_a", ref_a_rows[int(ridx[i])], t0[i:i + 1])[0] for i in k])
        r = score_truth_arm(st[:, :, k], vt, test.P[ridx[k]], test.T[ridx[k]], np.abs(v0[:, None, :] - vt), calls * dt)
        r["by_start"] = {f"{s:g}": late_summary(score_rows(st[:, :, k[t0[k] == s]], vt[t0[k] == s], test.P[ridx[k[t0[k] == s]]],
                                                           test.T[ridx[k[t0[k] == s]]]), calls * dt) for s in np.unique(t0[k])}
        r.pop("modes")
        out[f.stem] = {**r, **interventions(d), "nren": int(d["nren"].sum()) if "nren" in d else None}
    if (bench / "B_hold.npz").exists():
        d = np.load(bench / "B_hold.npz")
        st, calls, ridx, ts = d["states"], d["calls"], d["ridx"], float(d["t_start"])
        tru = np.stack([vtruth("ref_a", j, ts + calls * 10.0) for j in d["job"]])
        tru0 = np.stack([vtruth("ref_a", j, np.array([ts]))[0] for j in d["job"]])
        r = score_truth_arm(st, tru, test.P[ridx], test.T[ridx], np.abs(tru0[:, None, :] - tru), calls * 10.0)
        r.pop("modes")
        out["B_hold"] = {**r, **interventions(d), "nren": int(d["nren"].sum()) if "nren" in d else None}
    for f in sorted(bench.glob("[CE]_ref_*.npz")):
        d = np.load(f)
        st, calls, dt, tag = d["states"], d["calls"], float(d["dt"]), str(d["tag"])
        tru = np.stack([vtruth(tag, j, calls * dt) for j in d["job"]])
        tru0 = np.stack([vtruth(tag, j, np.array([0.0]))[0] for j in d["job"]])
        r = score_truth_arm(st, tru, d["P"], d["T"], np.abs(tru0[:, None, :] - tru), calls * dt)
        r.pop("modes")
        out[f.stem] = {**r, **interventions(d), "nren": int(d["nren"].sum()) if "nren" in d else None}
    if (bench / "D_stress.npz").exists():
        d = np.load(bench / "D_stress.npz")
        out["D_stress"] = {"consistency": _consistency(d["states"], None, None, d["calls"] * 10.0), **interventions(d)}
    Path(dst).write_text(json.dumps(out, indent=1))
    return out


def verdict(out):
    v = {}
    for mi, m in enumerate(MODES):
        if not any(m in out[k].get("vulcan", out[k].get("modes", {})) for k in out if k.startswith("A_dt")):
            continue  # mode not run
        nominal = sum(out[k]["rows_with_interventions"][mi] for k in out if k[:1] in "ABCE" and isinstance(out[k], dict) and "rows_with_interventions" in out[k])
        A = [out[k] for k in out if k.startswith("A_dt")]
        acc = bool(A) and all("vulcan" in a for a in A) and all(a["vulcan"][m]["pass_accuracy"] for a in A)
        cool = all(a["vulcan"][m]["cool_beats_persistence_final"] for a in A if "vulcan" in a)
        hold = out["B_hold"]["modes"][m]["pass_hold"] if "B_hold" in out else None
        sw = out["E_ref_c"]["modes"][m]["pass_accuracy"] if "E_ref_c" in out else None
        pert = out["C_ref_p"]["modes"][m]["pass_accuracy"] if "C_ref_p" in out else None  # noqa: E501
        robust = (out["D_stress"]["rows_with_interventions"][mi] == 0) if "D_stress" in out else None
        rank = max([a["vulcan"][m]["worst_max_p90"] for a in A if "vulcan" in a] or [float("nan")])
        v[m] = {"nominal_intervention_rows": int(nominal), "accuracy_A": bool(acc), "cool_skill": bool(cool),
                "hold": hold, "switch_E": sw, "perturb_C": pert, "robust_D": robust,
                "eligible": bool(nominal == 0 and acc and cool and hold is not False and sw is not False and pert is not False),
                "rank_worst_p90_A": rank}
    return v


def main(bench):
    bench = Path(bench)
    test = D.RawSplit("test", torch.device("cpu"))
    ref_a_rows = {j["row"]: i for i, j in L.vref_index("ref_a").items()}
    out = {"meta": json.loads((bench / "meta.json").read_text())}
    out["meta"].pop("sources", None)
    for f in sorted(bench.glob("A_dt*.npz")):
        out[f.stem] = score_A(f, test, ref_a_rows)
    if (bench / "B_hold.npz").exists():
        out["B_hold"] = score_B(bench / "B_hold.npz", test)
    for f in sorted(bench.glob("[CE]_ref_*.npz")):
        out[f.stem] = score_V(f)
    if (bench / "D_stress.npz").exists():
        out["D_stress"] = score_D(bench / "D_stress.npz")
    for f in sorted(bench.glob("V_*.npz")):
        out[f.stem] = score_Vt(f, test, ref_a_rows)
    for f in sorted(bench.glob("H_*.npz")):
        out[f.stem] = score_H(f)
    if (bench / "G_stress.npz").exists():
        out["G_stress"] = score_G(bench / "G_stress.npz")
    out["verdict"] = verdict(out)
    (bench / "score.json").write_text(json.dumps(out, indent=1))
    print(headline(out))


def headline(out):
    lines = [f"model {out['meta']['model']} {out['meta'].get('ckpt')}  notes {out['meta'].get('notes')}"]
    for k, v in out.items():
        if k.startswith("A_dt"):
            for m in MODES:
                for tr in ("vulcan", "raw"):
                    if tr in v and m in v[tr]:
                        mo = v[tr][m]
                        lines.append(f"  {k} {m} {tr:6s}: worst p90 {mo['worst_max_p90']:.3g} p99 {mo['worst_max_p99']:.3g} final p90 "
                                     f"{mo['worst_final_p90']:.3g} pass {mo['pass_accuracy']} cool<persist {mo['cool_beats_persistence_final']}"
                                     f" int_rows {v['rows_with_interventions'][MODES.index(m)]}")
        if k in ("B_hold",) or k.startswith(("C_ref", "E_ref")):
            for m in v["modes"]:
                mo = v["modes"][m]
                lines.append(f"  {k} {m}: worst p90 {mo['worst_max_p90']:.3g} p99 {mo['worst_max_p99']:.3g} final p90 {mo['worst_final_p90']:.3g}"
                             f" int_rows {v['rows_with_interventions'][MODES.index(m)]}")
        if k == "D_stress":
            for m in v["modes"]:
                mo = v["modes"][m]
                lines.append(f"  D {m}: range [{mo['min_log10']:.3g}, {mo['max_log10']:.3g}] finite {mo['all_finite']} int_rows {v['rows_with_interventions'][MODES.index(m)]}")
        if k.startswith(("V_", "H_")):
            for m, s in v["summary"].items():
                c = v["consistency"][m]["sum12"]
                lines.append(f"  {k} {m}: campaign {s['campaign']:.3g} late-max {s['late_max']} final {s['final']:.3g} "
                             f"int_rows {v['rows_with_interventions'][MODES.index(m)]} sum12>1.01 {c['frac_gt_1.01']:.3f} max {c['max_sum']:.3g}")
        if k == "G_stress":
            for gname, gv in v["groups"].items():
                lines.append(f"  G {gname}: stress sum12>1.01 {gv['stress']['frac_gt_1.01']:.3f} max {gv['stress']['max_sum']:.3g} "
                             f"floor {gv['stress']['n_floor_rowckpt']} int_rows {gv['rows_with_interventions']} "
                             f"twin rowmax p90 {gv.get('twin_dist_recovery_end', {}).get('rowmax_p90')}")
    for m, vv in out["verdict"].items():
        lines.append(f"  VERDICT {m}: {vv}")
    return "\n".join(lines)


if __name__ == "__main__":
    if sys.argv[1] == "--consistency":     # score.py --consistency <bench_dir> <out.json>
        consistency_dir(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1])
