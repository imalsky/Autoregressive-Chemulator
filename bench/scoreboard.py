"""Overall-error scoreboard (2026-10-08 methods audit; author: "the target is best overall error ... any big error is a
problem"). Per run, on resolved points (truth or prediction >= 1e-2 cm^-3, as analyze_e13) with a mixing-ratio floor
(lib/scorerule.py: the per-gas floors, SCORE_RULE=phys, the default since 2026-10-19; flat12 = the old 1e-12 floor) so
that near-threshold trace noise does not dominate the tails:
  G dt500 per T band: pooled late-max p90, worst species (late-max p90) and its name, late-max p99, max, rows with any
    species > 0.5 dex at a late checkpoint; low-P cell (P < 1e-3 bar) pooled p90; first checkpoint (500 s) max and rows
    > 0.5 dex (excluded from the late window); C/O and N/O |dlog10| p99 at 1000 days (12 tracked species); max sum12.
  G+Gc below 1300 K: the same per-band cells on the pooled cold instrument.
  paths v2s / v2l per path-mean-T band and all: pool p90 (calls >= 50), p99, max, paths with any species > 0.5 dex.
Writes scoreboard_<tag>.json to paths.OUT and prints one block per run.
    python scoreboard.py <tag> <run> [<run> ...]"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
import scorerule as SR  # noqa: E402  (2026-10-11: SCORE_RULE selects the scoring rule)
E = paths.E
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_e13 as A  # noqa: E402
import score_sprint as SS  # noqa: E402

OUT = paths.OUT   # outputs (2026-10-09: were written beside this file)

FLOOR = -12.0
SP = list(A.D.SHORT)
EDGES = A.EDGES


def bench(f):
    d = np.load(f)
    st = A.cut12(d["states"][:, 2]).transpose(1, 0, 2)
    el = d["tcum"].T if "tcum" in d.files else np.broadcast_to(d["calls"][None, :] * float(d["dt"]), (d["t0"].size, d["calls"].size))
    vt = np.stack([A.SC.vtruth(str(d["tag"]), j, d["t0"][i] + el[i]) for i, j in enumerate(d["job"])])
    lnd = np.log10(d["P"] / (A.KB * d["T"]))[:, None, None]
    e = SR.err(vt, st, ((vt + lnd) >= -2.0) | ((st + lnd) >= -2.0), SP)
    late = np.median(el, 0) >= 1e3
    return {"e": e[:, late], "e1": e[:, np.logical_not(late)], "st": st, "vt": vt, "T": d["T"], "P": d["P"] / 1e6,
            "job": d["job"]}


def lmax(e, q):
    v = [np.nanpercentile(e[:, c], q) for c in range(e.shape[1]) if (~np.isnan(e[:, c])).any()]
    return float(np.max(v)) if v else float("nan")   # inf / NaN when a column holds a failed (non-finite) prediction


def cells(B, m):
    e = B["e"][m]
    per = {s: lmax(e[:, :, j:j + 1], 90) for j, s in enumerate(SP)}
    w = max(per, key=lambda s: np.nan_to_num(per[s], nan=-1))
    out = {"p90": lmax(e, 90), "worst_sp_p90": per[w], "worst_sp": w, "p99": lmax(e, 99),
           "max": float(np.nanmax(e)) if (~np.isnan(e)).any() else float("nan"),
           "rows_gt0.5": int((np.nan_to_num(e, nan=0.0).max(axis=(1, 2)) > 0.5).sum()), "rows": int(m.sum()),
           "nonfinite_rows": int((np.isinf(e).any(axis=(1, 2)) | np.isinf(B["e1"][m]).any(axis=(1, 2))).sum())}
    lp = m & (B["P"] < 1e-3)
    out["lowP_p90"] = lmax(B["e"][lp], 90) if lp.any() else float("nan")
    e1 = B["e1"][m]
    if e1.size:
        out["call1_max"] = float(np.nanmax(e1)) if (~np.isnan(e1)).any() else float("nan")
        out["call1_rows_gt0.5"] = int((np.nan_to_num(e1, nan=0.0).max(axis=(1, 2)) > 0.5).sum())
    return out


def ratios(lx):
    x = 10.0 ** lx
    g = {s: x[..., SP.index(s)] for s in SP}
    C = 2 * g["C2H2"] + g["CH4"] + g["CO2"] + g["CO"] + g["HCN"]
    O = 2 * g["CO2"] + g["CO"] + g["H2O"] + g["OH"] + g["O"]
    N = 2 * g["N2"] + g["NH3"] + g["HCN"]
    return np.log10(C / O), np.log10(N / O)


def g_board(run):
    B = bench(E / f"H_g500_{run}" / "H_gtest_dt500.npz")
    r = {f"{lo}-{hi}": cells(B, (B["T"] >= lo) & (B["T"] < hi)) for lo, hi in zip(EDGES[:-1], EDGES[1:])}
    co_m, no_m = ratios(B["st"][:, -1])
    co_t, no_t = ratios(B["vt"][:, -1])
    r["composition"] = {"CO_p99": float(np.percentile(np.abs(co_m - co_t), 99)), "NO_p99": float(np.percentile(np.abs(no_m - no_t), 99)),
                        "sum12_max": float((10.0 ** B["st"]).sum(-1).max())}
    k = np.unravel_index(np.nanargmax(B["e"]), B["e"].shape)
    r["worst"] = {"dex": float(B["e"][k]), "T": float(B["T"][k[0]]), "P_bar": float(B["P"][k[0]]), "job": int(B["job"][k[0]]),
                  "species": SP[k[2]]}
    return r, B


def pooled_cold(Bg, Bc):
    mg, mc = Bg["T"] < 1300, Bc["T"] < 1300
    P = {k: np.concatenate([Bg[k][mg], Bc[k][mc]]) for k in ("e", "e1", "T", "P")}
    P["job"] = np.concatenate([Bg["job"][mg], Bc["job"][mc] + 10 ** 6])
    return {f"{lo}-{hi}": cells(P, (P["T"] >= lo) & (P["T"] < hi)) for lo, hi in zip(EDGES[:-1], EDGES[1:]) if hi <= 1300}


def path_board(run, pat):
    f = E / pat.format(run) / "P_paths.npz"
    if not f.exists():
        return None
    d = np.load(f)                                         # as score_sprint.paths_errors, keeping the truth for the floor
    st, calls = A.cut12(d["states"][:, 2]), d["calls"]
    pdir = Path(str(d["p_dir"]))
    vt, res = [], []
    for k, p in enumerate(d["path"]):
        q = np.load(pdir / f"{int(p)}.npz")
        cols = [list(q["species"]).index(s) for s in A.D.SHORT]
        x = np.log10(np.maximum(q["x"][calls][:, cols], A.D.EPS))
        n = q["Pbar"][calls - 1] * 1e6 / (A.KB * q["T"][calls - 1])
        vt.append(x)
        res.append((x + np.log10(n)[:, None] >= -2.0) | (st[:, k] + np.log10(n)[:, None] >= -2.0))
    vt, res = np.stack(vt, 1), np.stack(res, 1)
    e = SR.err(vt, st, res, SP)
    Tm = d["T"].mean(1)
    late = calls >= 50
    out = {}
    for b, m in [("all", np.ones_like(Tm, bool))] + SS.bands(Tm):
        if not m.any():
            continue
        el = e[late][:, m]
        q = el[~np.isnan(el)]   # inf (a failed, non-finite prediction) stays in
        out[b] = {"pool_p90": float(np.percentile(q, 90)) if q.size else float("nan"),
                  "p99": float(np.percentile(q, 99)) if q.size else float("nan"),
                  "max": float(q.max()) if q.size else float("nan"),
                  "paths_gt0.5": int((np.nan_to_num(e[:, m], nan=0.0).max(axis=(0, 2)) > 0.5).sum()), "paths": int(m.sum()),
                  "nonfinite_paths": int(np.isinf(e[:, m]).any(axis=(0, 2)).sum())}
    return out


def main(tag, runs):
    res = {}
    for run in runs:
        g, Bg = g_board(run)
        fc = E / f"H_gc500_{run}" / "H_gtest_cold_dt500.npz"
        cold = pooled_cold(Bg, bench(fc)) if fc.exists() else None
        res[run] = {"G": g, "G+Gc": cold, "v2s": path_board(run, "P2s_{}"), "v2l": path_board(run, "P2l_{}")}
        print(f"== {run}")
        for b in [f"{lo}-{hi}" for lo, hi in zip(EDGES[:-1], EDGES[1:])]:
            c = g[b]
            print(f"  G {b:9s} p90 {c['p90']:.3f} | worst sp {c['worst_sp']} {c['worst_sp_p90']:.2f} | p99 {c['p99']:.3f} | max {c['max']:.2f}"
                  f" | rows>0.5 {c['rows_gt0.5']}/{c['rows']} | lowP p90 {c['lowP_p90']:.3f} | call1 max {c.get('call1_max', float('nan')):.2f},"
                  f" rows>0.5 {c.get('call1_rows_gt0.5', -1)}"
                  + (f" | NON-FINITE predictions in {c['nonfinite_rows']} rows" if c["nonfinite_rows"] else ""))
        for b, c in (cold or {}).items():
            print(f"  G+Gc {b:9s} p90 {c['p90']:.3f} | worst sp {c['worst_sp']} {c['worst_sp_p90']:.2f} | p99 {c['p99']:.3f} | max {c['max']:.2f}"
                  f" | rows>0.5 {c['rows_gt0.5']}/{c['rows']}"
                  + (f" | NON-FINITE predictions in {c['nonfinite_rows']} rows" if c["nonfinite_rows"] else ""))
        print(f"  composition C/O p99 {g['composition']['CO_p99']:.3f}, N/O p99 {g['composition']['NO_p99']:.3f}, sum12 max "
              f"{g['composition']['sum12_max']:.3f} | worst point {g['worst']}")
        for s in ("v2s", "v2l"):
            for b, c in (res[run][s] or {}).items():
                print(f"  {s} {b:9s} pool p90 {c['pool_p90']:.3f} | p99 {c['p99']:.3f} | max {c['max']:.2f} | paths>0.5 {c['paths_gt0.5']}/{c['paths']}"
                      + (f" | NON-FINITE predictions in {c['nonfinite_paths']} paths" if c["nonfinite_paths"] else ""))
        (OUT / f"scoreboard_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
