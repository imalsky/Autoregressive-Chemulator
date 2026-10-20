"""The scorer (bench/score.py) on hand-made cases and on the stored yardstick inputs (CPU only, no model call; nothing
outside the work folder is written).

(a) Synthetic cases with hand values, through the scorer's own functions and, for the whole pipeline, through tiny
    design and states folders written into the work folder:
    - rule v3.1's five cases: a successful LM rescue is not a failure; a failed rescue (kept input at call 5) fails the
      row from that call on (failed calls N - 5 + 1) and every later saved call scores 2 dex; an invalid saved state is
      a failure; a piece crossing 10 s is split exactly; a decade touched only by a call exactly at its upper edge has
      zero covered log-time and is untested;
    - a gas below its floor in prediction and truth scores 0;
    - start weights 0.55 / 0.45 with raw parents at 0 (and the renormalisation when a parent is missing);
    - bin weights log(b_hi / b_lo) / log 500 (equal to the stored yardstick's), and the 10 s rows outside the bins;
    - a variable-dt path row's quadrature on its cumulative time;
    - a mixing event and a failure in a path row (the remainder penalised);
    - the lasting window [t_end / 10, t_end];
    - a T2 pair's excess;
    - three panels combining into the one number, through the files;
    - the trimmed mean of a known vector; the bootstrap fixed by its seed;
    - BETTER / TIE / WORSE at +-delta (directly and through the files);
    - the in-domain filters: an out-of-domain T2 box never enters D_snap, and an out-of-domain path condition is left
      out of its design's score and counted.
(b) Rule v3.1 reproduction on the stored yardstick inputs (mini-chem yardstick design/ and run/; CBg, FINAL and LD1 vs
    mini-chem) with the design's stored start weights and conditions resampled within cells: every leaf of
    comparisons/<model>_vs_minichem/seg/{mean12, worst, fail_A, fail_B} and of free/mean12/bins must equal the stored
    score_v31/score.json as float64 values (formatting ignored), and CBg must print segments +0.031 [+0.023, +0.038].
    The other free-run numbers differ from score-1 on purpose (rule v3.1 weights dt values equally, score-1 weights
    bins): listed, not tested.
    taskset -c <cpus> $PY tests/score_regress.py [--work DIR]   (about 2 min)
Exit 0 = PASS."""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib"))
import paths  # noqa: E402

sys.path.insert(0, str(REPO / "bench"))
import score as S  # noqa: E402

YARD = paths.PROJECT / "experiments/2026-10-19_minichem-yardstick/_build"
V31 = paths.PROJECT / "experiments/2026-10-20_t0-fix/_build/score_v31/score.json"
MODELS = ("CBg", "FINAL", "LD1")
BINS = [(1, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500)]
ROWK = ("first_newton", "first_kept", "first_guard", "first_nonfin", "first_fail", "newton_n", "kept_n", "guard_n", "nonfin_n")


def check(results, name, ok, detail=""):
    print(f"  {name}: {'ok' if ok else 'WRONG'}{' (' + detail + ')' if detail else ''}", flush=True)
    results.append(bool(ok))


def obj(**k):
    o = type("D", (), {})()
    for a, v in k.items():
        setattr(o, a, v)
    return o


def solve(D, f):
    """A model's point statistics with the scorer's failure reading (unrecovered steps and invalid saved states)."""
    logp, ff, unrec, _ = S.load_model(f)
    st = S.point_stats(D, logp, ff)
    st["ff_literal"] = st["ff"]
    st["ff"] = S.merge_bad(D, unrec, st["bad"])
    return st


def write_states(path, logy12, first=None, design_sha="x"):
    path.mkdir(parents=True, exist_ok=True)
    R = int(first["N"]) if first and "N" in first else None
    first = {k: v for k, v in (first or {}).items() if k != "N"}
    base = {k: np.full(R if R is not None else len(next(iter(first.values()))), -1 if k.startswith("first") else 0)
            for k in ROWK}
    base.update({k: np.asarray(v) for k, v in first.items()})
    np.savez(path / "states.npz", gases12=np.array(S.GASES), logy12=logy12, **base)
    (path / "meta.json").write_text(json.dumps({"design_sha256": design_sha}))


def write_design(d, name, mode, panel, role, cond, par, rows, saved, truth, slots, m=1):
    """A score-1 design folder (the files bench/score.py reads) with its manifest and sha256 values."""
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "conditions.npz", **cond)
    np.savez(d / "parents.npz", **par)
    np.savez(d / "rows.npz", **rows)
    np.savez(d / "saved.npz", **saved)
    np.savez(d / "truth_saved.npz", gases12=np.array(S.GASES), logx12=truth)
    files = ["conditions.npz", "parents.npz", "rows.npz", "saved.npz", "truth_saved.npz"]
    w = np.log(np.array([b / a for a, b in BINS])) / np.log(500.0)
    man = dict(format=S.FORMAT, name=name, mode=mode, panel=panel, role=role, start_codes=["eqt", "eqf", "rawA", "rawB"],
               start_weights={"eqt": 0.55, "eqf": 0.45, "rawA": 0.0, "rawB": 0.0}, slots=slots, slots_per_bin=m,
               dt_bins=BINS, dt_bin_weights=w.tolist(), bootstrap="stratum", delta=0.005, retrain_spread="unmeasured",
               files={f: S.sha256(d / f) for f in files})
    (d / "manifest.json").write_text(json.dumps(man))
    return S.sha256(d / "manifest.json")


def tiny(d, name, mode, panel, role, ncond=4, in_domain=None, dt_bin=0, slots=9):
    """ncond conditions x {eqt, eqf}, one row per parent from t = 0 at dt 10 s in bin dt_bin (slot dt_bin; a path
    design: slot 0), saved at calls 1, 10, 100 (t = 10, 100, 1000 s), truth log10 x = -3 for every gas."""
    C, P = ncond, 2 * ncond
    cond = dict(key=np.array([f"{name}{i}" for i in range(C)]), stratum=np.array(["G", "C"] * (C // 2)),
                group=np.array([0, 1] * (C // 2)), in_domain=np.ones(C, bool) if in_domain is None else in_domain,
                T=np.full(C, 1000.0), P_bar=np.full(C, 1.0), Z=np.ones(C), C_O=np.full(C, 0.5), He_H=np.full(C, 0.09),
                variable_dt=np.zeros(C, bool), mixed=np.zeros(C, bool))
    par = dict(cond=np.repeat(np.arange(C), 2), start=np.tile([0, 1], C), id=np.arange(P), job=np.arange(P),
               kick=np.array([""] * P), src=np.full(P, -1), T=np.full(P, 1000.0), P_bar=np.ones(P))
    slot = 0 if mode == "path" else dt_bin
    rows = dict(parent=np.arange(P), unit=np.arange(P), t0=np.zeros(P), dt=np.full(P, 10.0), N=np.full(P, 100),
                dt_idx=np.full(P, slot), dt_bin=np.full(P, dt_bin), age_idx=np.zeros(P, np.int8), nsave=np.full(P, 3),
                off=np.arange(0, 3 * P + 1, 3))
    k = np.tile([1, 10, 100], P)
    saved = dict(k=k, tabs=k * 10.0, tel=k * 10.0, row=np.repeat(np.arange(P), 3))
    return write_design(d, name, mode, panel, role, cond, par, rows, saved, np.full((3 * P, 12), -3.0),
                        1 if mode == "path" else slots)


def unit_checks(work):
    res = []
    # (1), (2): rule v3.1 failures. Three rows saved at calls 1, 2, 5, 10, 2000 (N = 2000), errors 0. Row 0: LM rescue at
    # call 2 (successful); row 1: LM rescue at 2, kept input at 5 (the rescue failed); row 2: clean.
    u = work / "unit"
    u.mkdir(parents=True, exist_ok=True)
    first = dict(first_newton=[2, 2, -1], first_kept=[-1, 5, -1], first_guard=[-1, -1, -1], first_nonfin=[-1, -1, -1],
                 first_fail=[2, 2, -1], newton_n=[1, 1, 0], kept_n=[0, 1, 0], guard_n=[0, 0, 0], nonfin_n=[0, 0, 0])
    np.savez(u / "m.npz", gases12=np.array(S.GASES), logy12=np.zeros((15, 12)), **{k: np.array(v) for k, v in first.items()})
    D = obj(R=3, S=15, row=np.repeat(np.arange(3), 5), k=np.tile([1, 2, 5, 10, 2000], 3), logt=np.zeros((15, 12)),
            r={"t0": np.zeros(3), "off": np.array([0, 5, 10, 15]), "N": np.full(3, 2000)})
    st = solve(D, u / "m.npz")
    fp = S.failed_points(D, st["ff"])
    e = np.where(fp, S.PEN, st["mean12"])
    fc = S.fail_rates(D, st["ff"], np.ones(3, bool))["failed_calls"]
    check(res, "a successful LM rescue is not a failure", st["ff"][0] == 0 and not fp[:5].any() and st["ff_literal"][0] == 2,
          f"first failed call per row {st['ff'].tolist()}, literal {st['ff_literal'].tolist()}")
    check(res, "a failed rescue penalises the remainder (failed calls N - 5 + 1)",
          st["ff"].tolist() == [0, 5, 0] and e.reshape(3, 5).tolist() == [[0.0] * 5, [0.0, 0.0, 2.0, 2.0, 2.0], [0.0] * 5]
          and fc == 1996, f"statistic {e.reshape(3, 5)[1].tolist()}, failed calls {fc}")
    np.savez(u / "m.npz", gases12=np.array(S.GASES), logy12=np.where((np.arange(15) == 13)[:, None], np.nan, 0.0),
             **{k: np.array(v) for k, v in first.items()})
    st = solve(D, u / "m.npz")
    check(res, "an invalid saved state is a failure", st["ff"].tolist() == [0, 5, 10],
          f"non-finite state at call 10 of row 2 -> first failed calls {st['ff'].tolist()} (want [0, 5, 10])")
    # (3), (4): row 0 starts at t = 0, saved at 10, 100, 1000 s with statistic 1, 3, 9; row 1 restarts at t0 = 3 s, saved
    # at 5, 20, 100 s with 2, 4, 6 (its piece 5 -> 20 s crosses 10 s); held at the first value back to t0.
    q = obj(tabs=np.array([10.0, 100.0, 1000.0, 5.0, 20.0, 100.0]), row=np.repeat(np.arange(2), 3), S=6,
            r={"t0": np.array([0.0, 3.0]), "off": np.array([0, 3, 6])}, case_row=np.array([0, 1]))
    e = np.array([1.0, 3.0, 9.0, 2.0, 4.0, 6.0])
    sw, swe = S.exact_sums(q, e, None, None, 2)
    V, case = S.case_means(sw, swe, 2)
    lg = np.log10
    a, v1 = 1.0 - lg(5.0), 2.0 + 2.0 * (1.0 - lg(5.0)) / lg(4.0)
    want_sw, want_swe = np.zeros(16), np.zeros(16)
    want_sw[[1, 2, 8, 9]] = [1.0, 1.0, lg(5.0 / 3.0) + a, lg(2.0) + lg(5.0)]
    want_swe[[1, 2, 8, 9]] = [2.0, 6.0, 2.0 * lg(5.0 / 3.0) + a * (2.0 + v1) / 2.0, lg(2.0) * (v1 + 4.0) / 2.0 + lg(5.0) * 5.0]
    check(res, "a piece crossing 10 s is split exactly",
          np.allclose(sw, want_sw, rtol=0, atol=1e-12) and np.allclose(swe, want_swe, rtol=0, atol=1e-12),
          f"row 1 covered log-time 1e0 {sw[8]:.6f} / 1e1 {sw[9]:.6f}, integrals {swe[8]:.6f} / {swe[9]:.6f}")
    check(res, "zero-duration coverage is untested", sw[0] == 0.0 and np.isnan(V[0, 0]) and case[0] == 4.0,
          f"row 0's decade 1e0 (only a call exactly at 10 s): covered {sw[0]}, value {V[0, 0]}; case {case[0]} (want 4.0)")
    # gas below its floor in both
    g = obj(R=1, S=2, row=np.zeros(2, int), k=np.array([1, 2]), logt=np.full((2, 12), -3.0))
    g.logt[:, S.GASES.index("H2O")] = -12.0
    lp = np.full((2, 12), -3.0)
    lp[0, S.GASES.index("H2O")] = -11.0
    lp[1, S.GASES.index("H2O")] = -9.0
    st = S.point_stats(g, lp, np.zeros(1, np.int64))
    check(res, "a gas below its floor in both scores 0", st["mean12"][0] == 0.0 and abs(st["mean12"][1] - 1 / 12) < 1e-15,
          f"H2O -11 vs -12 (floor -10): {st['mean12'][0]}; -9 vs -12: {st['mean12'][1]:.6f} (want 1/12)")
    # start weights
    pn = obj(C=2, starts=["eqt", "eqf", "rawA", "rawB"], pcond=np.array([0, 0, 0, 0, 1, 1]),
             pstart=np.array([0, 1, 2, 3, 0, 2]), W=np.array([0.55, 0.45, 0.0, 0.0]))
    cv = S.Panel.cond(pn, np.array([1.0, 2.0, 10.0, 20.0, 3.0, 30.0]))
    check(res, "start weights 0.55 / 0.45, raw parents at 0", cv[0] == (0.55 * 1.0 + 0.45 * 2.0) / 1.0 and cv[1] == 3.0,
          f"eqt 1, eqf 2, rawA 10, rawB 20 -> {cv[0]!r} (want 1.45); eqt 3 with no eqf, rawA 30 -> {cv[1]!r} (want 3.0)")
    # bin weights and the 10 s rows
    w = np.log(np.array([b / a for a, b in BINS])) / np.log(500.0)
    ys = np.load(YARD / "design" / "rows.npz")["dt_bin_weights"]
    sd = work / "bins"
    tiny(sd / "S", "S", "seg", "S", "readout", dt_bin=0)
    z = json.loads((sd / "S" / "manifest.json").read_text())
    pnl = S.Panel(sd / "S")
    pb = S._nanwmean(np.array([[1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3.0]]), w, 1)   # bins 0 and 7 tested
    check(res, "bin weights log(b_hi / b_lo) / log 500 (= the stored yardstick's), renormalised over tested bins",
          np.allclose(w, ys, rtol=0, atol=1e-15) and abs(w.sum() - 1.0) < 1e-12 and z["dt_bin_weights"] == w.tolist()
          and abs(pb[0] - (w[0] * 1.0 + w[7] * 3.0) / (w[0] + w[7])) < 1e-15,
          f"weights {np.round(w, 4).tolist()}, sum {w.sum():.12f}; bins 0 (1.0) and 7 (3.0) -> {pb[0]:.6f}")
    rows = np.load(sd / "S" / "rows.npz")
    r = {k: rows[k] for k in rows.files}
    r["dt_idx"] = r["dt_idx"].copy()
    r["dt_bin"] = r["dt_bin"].copy()
    r["dt_idx"][1], r["dt_bin"][1] = 8, -1          # parent 1's only row becomes a 10 s readout row
    tr = np.full((pnl.S, 12), -3.0)
    tr[3:6] = -2.0                                   # with a 1 dex error on every gas
    rewrite(sd / "S", rows=r, truth=tr)
    pnl = S.Panel(sd / "S")
    st = S.point_stats(pnl, np.full((pnl.S, 12), -3.0), np.zeros(pnl.R, np.int64))
    pr = S.panel_profile(pnl, st, st["ff"], stats=("mean12",))["mean12"]
    check(res, "the 10 s rows stay outside the bins", np.isnan(pr["par"]["pscore"][1]) and pr["par"]["ro"][1, 0] == 1.0
          and pr["cond"][0] == 0.0 and pr["r10"][0] == 1.0,
          f"parent 1 (10 s row only): score {pr['par']['pscore'][1]}, readout {pr['par']['ro'][1, 0]}; condition 0 "
          f"{pr['cond'][0]}, its 10 s readout {pr['r10'][0]}")
    # a variable-dt path row: dt 10, 30, 60 -> cumulative t 10, 40, 100 s; statistic 1, 2, 4 (one row, one parent)
    vp = obj(tabs=np.array([10.0, 40.0, 100.0]), row=np.zeros(3, int), S=3, R=1,
             r={"t0": np.zeros(1), "off": np.array([0, 3])}, case_row=np.array([0]))
    sw, swe = S.exact_sums(vp, np.array([1.0, 2.0, 4.0]), None, None, 1)
    V, case = S.case_means(sw, swe, 1)
    want = ((lg(4.0)) * 1.5 + lg(2.5) * 3.0) / (lg(4.0) + lg(2.5))
    check(res, "a variable-dt path row is integrated on its cumulative time",
          abs(case[0] - want) < 1e-14 and np.isnan(V[0, 0]) and abs(V[0, 1] - want) < 1e-14 and abs(sw[1] - 1.0) < 1e-15,
          f"case {case[0]:.12f} (want {want:.12f}; decade 1e1 only, covered {sw[1]:.15f} dex)")
    # a path design with a mixing event (truth jumps at call 2) and a kept-input failure at call 3
    pd = work / "pathmix"
    tiny(pd, "Pm", "path", "P", "readout", ncond=2, dt_bin=3)
    np.savez(pd / "forcing.npz", mix_row=np.array([0]), mix_k=np.array([2]), mix_f=np.array([0.1]),
             mix_x=np.full((1, 53), 1e-3))
    pnl = S.Panel(pd)
    tr = np.full((pnl.S, 12), -3.0)
    tr[1] = -2.5                                      # the mixed truth at row 0's second saved call
    rewrite(pd, truth=tr)
    pnl = S.Panel(pd)
    write_states(work / "runpm" / "A" / "Pm", np.full((pnl.S, 12), -3.0),
                 dict(N=pnl.R, first_kept=[-1, 3, -1, -1], first_fail=[-1, 3, -1, -1]), pnl.sha)
    st = S.model_states(pnl, work / "runpm", "A")
    pr = S.panel_profile(pnl, st, st["ff"], stats=("mean12",))["mean12"]["par"]["pscore"]
    w0 = (0.25 + 0.25) / 2.0        # row 0: 0, 0.5, 0 at t = 10, 100, 1000 s -> decades 1e1 and 1e2 both 0.25
    w1 = (1.0 + 2.0) / 2.0          # row 1: 0, 2, 2 (calls 10 and 100 after the failure at 3) -> 1.0 and 2.0
    check(res, "a mix and a failure in a path row", abs(pr[0] - w0) < 1e-15 and pr[1] == w1 and pr[2] == 0.0,
          f"row 0 (mixed truth, no failure) {pr[0]} (want {w0}); row 1 (kept input at call 3) {pr[1]} (want {w1}); "
          f"row 2 {pr[2]}")
    # lasting window [t_end / 10, t_end]: saved at 10, 100, 1000, 1e4 s with 1, 2, 4, 8 -> mean over [1e3, 1e4] = 6
    lw = obj(tabs=np.array([10.0, 100.0, 1000.0, 1e4]), row=np.zeros(4, int), S=4, R=1,
             r={"t0": np.zeros(1), "off": np.array([0, 4])})
    v = S.window_means(lw, np.array([1.0, 2.0, 4.0, 8.0]), np.array([3.0]), np.array([4.0]))
    v2 = S.window_means(lw, np.array([1.0, 2.0, 4.0, 8.0]), np.array([2.5]), np.array([4.0]))
    check(res, "the lasting window [t_end / 10, t_end]", v[0] == 6.0 and abs(v2[0] - (0.5 * 3.5 + 1.0 * 6.0) / 1.5) < 1e-15,
          f"{v[0]} (want 6); over [10^2.5, 1e4] {v2[0]:.6f}")
    # T2 pair excess: kicked row 0.3 everywhere, identity 0.1 -> 0.2
    tp = obj(tabs=np.tile([500.0, 1e3, 1e5, 8.64e7], 2), row=np.repeat([0, 1], 4), S=8, R=2,
             r={"t0": np.zeros(2), "off": np.array([0, 4, 8])})
    ex = S.window_means(tp, np.repeat([0.3, 0.1], 4), np.full(2, 3.0), np.full(2, np.log10(8.64e7)))
    check(res, "a T2 pair's excess", abs((ex[0] - ex[1]) - 0.2) < 1e-15, f"excess {ex[0] - ex[1]:.15f}")
    # trimmed mean, bootstrap seed
    x = np.arange(40.0)
    x[0], x[39] = -1e6, 1e6
    check(res, "the trimmed mean of a known vector", S.trimmed(x) == np.mean(np.arange(2.0, 38.0)) and S.tmean(np.append(x, np.nan)) == S.trimmed(x),
          f"{S.trimmed(x)} (want 19.5; 2 trimmed per side of 40); NaN ignored")
    gr = np.array([0, 1] * 30)
    i1, i2 = S.strat_idx(gr, 0), S.strat_idx(gr, 0)
    s1, s2 = S.summarize(np.sin(np.arange(60.0)), gr), S.summarize(np.sin(np.arange(60.0)), gr)
    check(res, "the bootstrap is fixed by its seed", np.array_equal(i1, i2) and s1 == s2 and not np.array_equal(i1, S.strat_idx(gr, 1))
          and (gr[i1] == gr).all(), "same indices and summaries twice; panels independent; groups kept")
    # outcome at +-delta
    oc = [S.outcome(*t, 0.005) for t in ((-0.005, -0.007, -0.001), (-0.0049, -0.007, -0.001), (-0.006, -0.009, 0.0001),
                                          (0.005, 0.001, 0.008), (0.006, -0.0001, 0.01), (0.0, -0.002, 0.002))]
    check(res, "BETTER / TIE / WORSE at +-delta", oc == ["BETTER", "TIE", "TIE", "WORSE", "TIE", "TIE"], f"{oc}")
    # snap-back in-domain filter
    prs = dict(cls=np.array(["k1", "k1", "k2", "k3"]), in_domain=np.array([True, True, True, False]), src=np.array([0, 1, 2, 3]))
    ea, eb = np.array([0.2, 0.3, 0.1, 9.0]), np.array([0.1, 0.1, 0.1, 0.0])
    sn = S.snap_summary(ea, eb, prs)
    check(res, "an out-of-domain T2 box never enters D_snap", sn["D_snap"]["n_rows"] == 3 and abs(sn["D_snap"]["D"] - 0.1) < 1e-15
          and sn["out_of_domain"]["n_rows"] == 1, f"D_snap {sn['D_snap']['D']} over {sn['D_snap']['n_rows']} rows; out of domain "
          f"{sn['out_of_domain']['D']} over {sn['out_of_domain']['n_rows']}")
    return res


def rewrite(d, rows=None, truth=None):
    """Replace rows.npz and / or truth_saved.npz of a test design and refresh its manifest hashes."""
    if rows is not None:
        np.savez(d / "rows.npz", **rows)
    if truth is not None:
        np.savez(d / "truth_saved.npz", gases12=np.array(S.GASES), logx12=truth)
    man = json.loads((d / "manifest.json").read_text())
    man["files"] = {f: S.sha256(d / f) for f in man["files"]}
    (d / "manifest.json").write_text(json.dumps(man))
    return S.sha256(d / "manifest.json")


def pipeline_checks(work):
    """Three panels through the files: SCORE and D = (S + L + P) / 3, the outcome at +-delta, an out-of-domain path
    condition left out and counted."""
    res = []
    for nm, dA, want in (("better", -0.010, "BETTER"), ("worse", 0.010, "WORSE"), ("tie", -0.003, "TIE")):
        d = work / f"set_{nm}"
        shas = {"S": tiny(d / "S", "S", "seg", "S", "score", dt_bin=0),
                "L": tiny(d / "L", "L", "long", "L", "score", dt_bin=7, slots=8),
                "P": tiny(d / "P", "P", "path", "P", "score", in_domain=np.array([True, True, True, False]), dt_bin=4)}
        offB = {"S": 0.10, "L": 0.20, "P": 0.30}
        for pnl in ("S", "L", "P"):
            n = 24
            for m, off in (("B", offB[pnl]), ("A", offB[pnl] + dA)):
                lp = np.full((n, 12), -3.0 + off)
                lp[18:] = -3.0 + 5.0 if pnl == "P" else lp[18:]       # P's last condition (out of domain): 5 dex off
                write_states(d / "run" / m / pnl, lp, dict(N=8), shas[pnl])
        out = work / f"out_{nm}"
        r, _ = S.score_set(S.load_set(d), d / "run", ["A", "B"], out)
        c = r["comparisons"]["A_vs_B"]
        sB = r["absolute"]["B"]["SCORE"]
        ok = (abs(sB - (0.10 + 0.20 + 0.30) / 3) < 1e-14 and abs(c["D"] - dA) < 1e-14 and c["outcome"] == want
              and c["per_design"]["P"]["out_of_domain_conditions"] == 1 and c["per_design"]["P"]["mean12"]["D"]["n"] == 3)
        check(res, f"three panels into the one number, outcome {want}", ok,
              f"SCORE B {sB:.6f} (want 0.2), D {c['D']:+.6f} [{c['D_lo']:+.6f}, {c['D_hi']:+.6f}] (want {dA:+.3f}), "
              f"{c['outcome']}; P: 1 out-of-domain condition left out, {c['per_design']['P']['mean12']['D']['n']} scored")
    return res


def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{p}/{i}")
    else:
        yield p, o


def get(o, p):
    for k in p.strip("/").split("/"):
        o = o[int(k)] if isinstance(o, list) else o[k]
    return o


def same(u, v):
    if isinstance(u, bool) or isinstance(v, bool) or u is None or v is None or isinstance(u, str):
        return u == v
    return np.float64(u) == np.float64(v)


def v31_checks(work):
    """(b): rule v3.1 on the stored yardstick inputs vs the stored score_v31 json."""
    res = []
    ref = json.loads(V31.read_text())["comparisons"]
    D = S.Yard(YARD / "design")
    mc = S.model_states(D, YARD / "run", "minichem")
    pb = S.yard_profile(D, mc, mc["ff"])
    tB = S.yard_tables(D, pb, D.pw)
    for m in MODELS:
        st = S.model_states(D, YARD / "run", m)
        pa = S.yard_profile(D, st, st["ff"])
        c = S.yard_compare(D, S.yard_tables(D, pa, D.pw), tB, st["ff"], mc["ff"])
        S.write_json(work / f"v31_{m}.json", c)
        new = json.loads((work / f"v31_{m}.json").read_text())
        old = ref[f"{m}_vs_minichem"]
        n, bad = 0, []
        for sub in ("seg/mean12", "seg/worst", "seg/fail_A", "seg/fail_B", "free/mean12/bins"):
            for p, v in leaves(get(old, sub), sub):
                n += 1
                try:
                    w = get(new, p)
                except (KeyError, IndexError):
                    bad.append(f"{p} missing")
                    continue
                if not same(v, w):
                    bad.append(f"{p}: {v!r} -> {w!r}")
        s1 = S.yard_score1(D, pa, pb, S.DELTA)
        check(res, f"{m} vs mini-chem: every stored leaf of seg/{{mean12, worst, fail_A, fail_B}} and free/mean12/bins equal",
              not bad, f"{n} leaves, {len(bad)} differ{': ' + '; '.join(bad[:4]) if bad else ''}; segments D "
              f"{S.fs(new['seg']['mean12']['D'])}; listed, not tested: free runs rule v3.1 (dt values equal) D "
              f"{S.fs(new['free']['mean12']['D'])}, score-1 (500 s and 10 s by bin, eqt parents) D "
              f"{S.fs(s1['free_stationarity']['D'])}")
        if m == "CBg":
            pin = S.fs(new["seg"]["mean12"]["D"])
            check(res, "CBg pinned", pin == "+0.031 [+0.023, +0.038]", f"segments {pin} (want +0.031 [+0.023, +0.038])")
        del st, pa
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    tmp = os.environ.get("TMPDIR")
    assert tmp and not tmp.startswith("/tmp"), "TMPDIR must be set to your own scratch (run/env.sh)"
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="score_regress_", dir=tmp))
    work.mkdir(parents=True, exist_ok=True)
    print(f"score_regress: work {work}", flush=True)
    t0 = time.time()
    print("(a) synthetic cases:", flush=True)
    ra = unit_checks(work) + pipeline_checks(work)
    ok_a = all(ra)
    print(f"(a) {'PASS' if ok_a else 'FAIL'} ({sum(ra)} of {len(ra)} checks)", flush=True)
    print("(b) rule v3.1 reproduction on the stored yardstick inputs:", flush=True)
    rb = v31_checks(work)
    ok_b = all(rb)
    print(f"(b) {'PASS' if ok_b else 'FAIL'} ({sum(rb)} of {len(rb)} checks)", flush=True)
    ok = ok_a and ok_b
    print(f"total {time.time() - t0:.0f} s")
    print(f"RESULT score_regress: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
