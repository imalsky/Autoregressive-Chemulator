"""Score step-size sweeps (sweep.py outputs) with the per-gas rule and combine the nine dt with equal weight.
Truth = the row's VULCAN tag at t0 + tcum (scoreboard.vtruth, as scoreboard.bench with tcum); states cut to the 12 reported gases
(scoreboard.cut12); error = scorerule.err with SCORE_RULE=phys (per-gas floors) on resolved points (truth or prediction
number density >= 1e-2 cm^-3). Per row: w(call) = worst-gas error at a recorded call (NaN -> 0);
  lasting = w at the last call N;  late = max w over recorded calls >= N / 2;  mean = mean of w over the recorded calls
  (log-spaced + the second-half grid);  at tau = w at the recorded call where the row's elapsed time equals tau (1e3, 1e4 s:
  the same elapsed time at every dt; NaN where a dt does not reach tau in N calls).
Per (model, start type, subset, dt): rows lasting > 0.5, late > 0.5, median of the per-row mean, rows lasting > 2, late > 2,
max w; rows > 0.5 at tau. Equal weight = mean over the dt of the per-dt counts; worst dt = the dt with the largest count;
ratio = model count / reference count per dt (n/0 shown as 'inf', 0/0 as '-').
Subsets (devall6 rows): scope = the target, inside the envelope (select_jobs.env_margin) AND Z 0.1-300x AND C/O 0.1-1;
inside; all;
cooler = 300 <= T < 1300 K and 1e-5 <= P <= 100 bar; fig4 = the six raw-gas cases (own tag).
    CUDA_VISIBLE_DEVICES= SCORE_RULE=phys HARNESS_NORM= python score_sweep.py <out stem> <ref label>=<sweep dir>
        [<label>=<sweep dir> ...]      -> <stem>.txt (tables), <stem>.json (all numbers), <stem>_rows.npz (per-row values)"""
import json
import os
import sys
from pathlib import Path

import numpy as np

assert os.environ.get("SCORE_RULE") == "phys", "set SCORE_RULE=phys"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bench/
import scoreboard as S  # noqa: E402  (cut12, vtruth, KB; SR = scorerule; SP = the 12 gases)
from select_jobs import env_margin  # noqa: E402

assert S.SR.RULE == "phys" and len(S.SP) == 12
TAUS = (1e3, 1e4)
STARTS = {1e-3: "raw (t0 1e-3 s)", 1e6: "evolved (t0 1e6 s)"}


def score(d):
    """Per-row values for one sweep npz."""
    st = S.cut12(d["states"]).transpose(1, 0, 2)                          # [R, C, 12]
    el = d["tcum"].T                                                       # [R, C]
    tags = [str(t) for t in d["tags"]]
    vt = np.stack([S.vtruth(tags[ti], j, t0 + e) for ti, j, t0, e in zip(d["tagi"], d["job"], d["t0"], el)])
    lnd = np.log10(d["P"] / (S.KB * d["T"]))[:, None, None]
    e = S.SR.err(vt, st, ((vt + lnd) >= -2.0) | ((st + lnd) >= -2.0), S.SP)
    w = np.nan_to_num(e, nan=0.0, posinf=np.inf).max(-1)                                  # [R, C] worst gas per recorded call
    calls = d["calls"]
    N = int(calls[-1])
    wsp = np.nan_to_num(e, nan=-1.0).argmax(-1)                            # worst gas index per call
    r = {"lasting": w[:, -1], "late": w[:, calls >= N / 2].max(1), "mean": w.mean(1), "maxw": w.max(1),
         "last_gas": np.array([S.SP[k] for k in wsp[:, -1]])}
    for tau in TAUS:
        hit = np.isclose(el, tau, rtol=0, atol=1e-6)
        r[f"at{tau:g}"] = np.where(hit.any(1), np.where(hit, w, -np.inf).max(1), np.nan)
    return r, N


def subsets(d):
    inside, _, _ = env_margin()
    T, P = d["T"], d["P"] / 1e6
    dev = d["tagi"] == 0
    ins = dev & inside(T, P)
    sc = ins & (d["Z"] >= 0.1) & (d["Z"] <= 300) & (d["CO"] >= 0.1) & (d["CO"] <= 1)   # the target (was Z 1-100x, C/O <= 1)
    cool = dev & (T >= 300) & (T < 1300) & (P >= 1e-5) & (P <= 100)
    return {"scope": sc, "inside": ins, "all": dev, "cooler": cool, "fig4": d["tagi"] == 1}


def cells(r, sub, d, dts):
    out = {}
    for t0 in STARTS:
        for sn, sm in sub.items():
            for dt in dts:
                m = sm & (d["t0"] == t0) & (d["dtrow"] == dt)
                if not m.any():
                    continue
                c = {"rows": int(m.sum()), "last05": int((r["lasting"][m] > 0.5).sum()), "late05": int((r["late"][m] > 0.5).sum()),
                     "med_mean": float(np.median(r["mean"][m])), "last2": int((r["lasting"][m] > 2).sum()),
                     "late2": int((r["late"][m] > 2).sum()), "max": float(r["maxw"][m].max())}
                for tau in TAUS:
                    v = r[f"at{tau:g}"][m]
                    c[f"at{tau:g}_05"] = int((v > 0.5).sum()) if np.isfinite(v).all() else None
                    c[f"at{tau:g}_med"] = float(np.median(v)) if np.isfinite(v).all() else None
                out[(t0, sn, dt)] = c
    return out


def fmt(v, k=0):
    if v is None:
        return "-"
    return f"{v:.{k}f}" if isinstance(v, float) else str(v)


def main():
    stem = Path(sys.argv[1])
    items = [a.split("=", 1) for a in sys.argv[2:]]
    labs = [k for k, _ in items]
    ref = labs[0]
    D, R, C = {}, {}, {}
    for lab, path in items:
        d = np.load(Path(path) / "sweep.npz")
        D[lab] = d
        R[lab], n = score(d)
        print(f"{lab}: {d['dtrow'].size} rows, N {n}", flush=True)
    d0 = D[ref]
    for lab in labs:   # same rows in the same order for every model (paired)
        for k in ("dtrow", "t0", "tagi", "job"):
            assert np.array_equal(D[lab][k], d0[k]), (lab, k)
    dts = sorted(set(d0["dtrow"].tolist()))
    sub = subsets(d0)
    for lab in labs:
        C[lab] = cells(R[lab], sub, d0, dts)
    L = []
    n_calls = int(d0["calls"][-1])
    L.append(f"Step-size sweep: {len(labs)} models, {d0['dtrow'].size} rows = {len(dts)} dt x {len(STARTS)} starts x "
             f"{d0['dtrow'].size // len(dts) // len(STARTS)} jobs; N = {n_calls} calls per row (elapsed dt x N); reference {ref}.")
    L.append("Columns: dt (s). EQ = equal-weight mean over the dt; worst = dt with the largest count (ties: smallest dt).")
    for metric, title in (("last05", "lasting > 0.5 (rows, last call)"), ("late05", "late > 0.5 (rows, second half)"),
                          ("med_mean", "median per-row mean worst-gas error (dex)"), ("last2", "lasting > 2 (readout)"),
                          ("late2", "late > 2 (readout)"), ("max", "max worst-gas error (readout, dex)"),
                          ("at1000_05", "rows > 0.5 at elapsed 1e3 s (matched time)"),
                          ("at10000_05", "rows > 0.5 at elapsed 1e4 s (matched time; '-' = dt does not reach it)"),
                          ("at1000_med", "median worst-gas error at elapsed 1e3 s (dex)"),
                          ("at10000_med", "median worst-gas error at elapsed 1e4 s (dex)")):
        L.append("")
        L.append(f"== {title}")
        for sn in sub:
            for t0, tn in STARTS.items():
                if (t0, sn, dts[0]) not in C[ref]:
                    continue
                rows = C[ref][(t0, sn, dts[0])]["rows"]
                L.append(f"-- {sn} / {tn}, {rows} rows per dt")
                L.append(f"   {'model':18s} " + " ".join(f"{dt:>7g}" for dt in dts) + "      EQ  worst")
                for lab in labs:
                    v = [C[lab][(t0, sn, dt)][metric] for dt in dts]
                    k = 3 if metric in ("med_mean", "at1000_med", "at10000_med") else (2 if metric == "max" else 0)
                    vv = [x for x in v if x is not None]
                    eq = float(np.mean(vv)) if vv else None
                    worst = dts[int(np.argmax([(-1 if x is None else x) for x in v]))] if vv and max(vv) > 0 else None
                    L.append(f"   {lab:18s} " + " ".join(f"{fmt(x, k):>7s}" for x in v) + f" {fmt(eq, 3 if k else 1):>7s}  "
                             f"{fmt(worst)}")
                if metric in ("last05", "late05") and len(labs) > 1:
                    for lab in labs[1:]:
                        rr = []
                        for dt in dts:
                            a_, b_ = C[lab][(t0, sn, dt)][metric], C[ref][(t0, sn, dt)][metric]
                            rr.append("-" if a_ == 0 and b_ == 0 else ("inf" if b_ == 0 else f"{a_ / b_:.2f}"))
                        L.append(f"   {lab + '/' + ref:18s} " + " ".join(f"{x:>7s}" for x in rr))
    L.append("")
    L.append("== headline (scope, inside, cooler): equal-weight mean over dt of rows > 0.5; raw + evolved summed; worst dt")
    head = {}
    for lab in labs:
        for sn in ("scope", "inside", "cooler", "all"):
            for metric in ("last05", "late05"):
                per = [sum(C[lab][(t0, sn, dt)][metric] for t0 in STARTS) for dt in dts]
                head[(lab, sn, metric)] = (float(np.mean(per)), dts[int(np.argmax(per))] if max(per) > 0 else float('nan'), per)
    for sn in ("scope", "inside", "cooler", "all"):
        for metric in ("last05", "late05"):
            L.append(f"   {sn:7s} {metric:7s} " + " | ".join(f"{lab} {head[(lab, sn, metric)][0]:.1f} (worst dt "
                                                             f"{head[(lab, sn, metric)][1]:g}: {max(head[(lab, sn, metric)][2])})"
                                                             for lab in labs))
    f4 = np.flatnonzero((d0["tagi"] == 1) & (d0["t0"] == 1e-3))
    names = ["1000K 1mbar", "1500K 1mbar", "2000K 1mbar", "1000K 10bar", "1500K 10bar", "2000K 10bar"]
    if f4.size:
        L.append("")
        L.append("== fig4 raw-gas cases (t0 1e-3 s), worst-gas error (dex) per case and dt: lasting / late; then at elapsed "
                 "1e3 / 1e4 s ('nan' = not reached)")
        for lab in labs:
            L.append(f"-- {lab}")
            L.append("   " + f"{'dt':>5s} " + " ".join(f"{n:>13s}" for n in names) + " || " + " ".join(f"{n:>13s}" for n in names))
            for dt in dts:
                idx = sorted([i for i in f4 if d0["dtrow"][i] == dt], key=lambda i: int(d0["job"][i]))
                L.append("   " + f"{dt:>5g} " + " ".join(f"{R[lab]['lasting'][i]:6.2f}/{R[lab]['late'][i]:<6.2f}" for i in idx)
                         + " || " + " ".join(f"{R[lab]['at1000'][i]:6.2f}/{R[lab]['at10000'][i]:<6.2f}" for i in idx))
    txt = "\n".join(L)
    Path(str(stem) + ".txt").write_text(txt + "\n")
    js = {lab: {f"{t0:g}|{sn}|{dt:g}": c for (t0, sn, dt), c in C[lab].items()} for lab in labs}
    js["_headline"] = {f"{lab}|{sn}|{m}": {"eq": v[0], "worst_dt": v[1], "per_dt": v[2]} for (lab, sn, m), v in head.items()}
    js["_meta"] = {"ref": ref, "dirs": dict(items), "N": n_calls, "dts": dts, "rule": "phys"}
    Path(str(stem) + ".json").write_text(json.dumps(js, indent=1))
    np.savez(str(stem) + "_rows.npz", **{f"{lab}__{k}": v for lab in labs for k, v in R[lab].items()},
             dtrow=d0["dtrow"], t0=d0["t0"], tagi=d0["tagi"], job=d0["job"], T=d0["T"], P=d0["P"], Z=d0["Z"], CO=d0["CO"],
             **{f"sub_{k}": v for k, v in sub.items()})
    print(txt)


if __name__ == "__main__":
    main()
