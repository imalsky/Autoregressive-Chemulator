"""Development-board readouts restricted to a region of (T, P) (scale phase, 2026-10-14). No model call, no GPU: reads the
existing P* benches of a bench label (E13_gcm/<set>_<label>) with scoreboard.bench (per-gas rule, SCORE_RULE=phys).

A mask is any function inside(T_K, P_bar) -> bool array. Two makers:
  box(P_lo, P_hi, T_lo, T_hi)                  inclusive bounds (default: 1e-5 <= P <= 100 bar, any T)
  from_hist(logp_edges, T_edges, occ, min_occ)  a 2-D occupancy histogram occ[i, j] over log10(P / bar) bin i and T bin j
                                                (e.g. GCM T-P points); inside = the point's bin holds >= min_occ; points
                                                beyond the edges are outside
Fixed-condition sets G, Gc, H2, H2c, H3, H3c: a row (job x start age) is inside if its (T, P) is. Per side (inside, outside)
and per T band inside: rows; rows with a late (elapsed >= 1e3 s) point > 0.5 / > 2 dex; first-call rows > 2; the max over
every checkpoint; "lasting" rows = error > 0.5 / > 2 dex (max over species) at the last checkpoint, 1000 days after the
start. The six sets' sides sum to board6.py's totals by construction (outside = not inside).
Path sets PD (DEV), PX (xlong), P2l, P2s, PH: a call is inside if the (T, P) it runs at is. Per side: share of calls, late
window (board6.py: PD / PX / PH calls >= 2500, P2l >= 500, P2s >= 50) p50 / p99 / max pooled over scored points of the
side's calls, share of late calls with any species > 0.5 dex, the max over every call; paths whose last call is > 0.5 / > 2.
Row summaries of the fixed sets are cached in <SCORE_MASK_CACHE>/rows_<label>.npz (default paths.T5_DESIGN; rebuilt
when a bench file is newer).

Use from Python (run with SCORE_RULE=phys, HARNESS_NORM unset, PYTHONDONTWRITEBYTECODE=1, pinned with taskset):
    import score_mask as M; t = M.table("e13s_FINw1536_s2", M.box(1e-5, 100)); print(M.fmt(t))
or from the shell (writes paths.OUT/score_mask_<name>.txt / .json):
    python score_mask.py <name> [--box P_lo P_hi T_lo T_hi] [--hist f.npz --keys logp_edges,T_edges,occ --min-occ 1] [label ...]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))   # bench/
import scoreboard as S  # noqa: E402  (puts lib/ on sys.path)
import paths  # noqa: E402
import scorerule as SR  # noqa: E402

assert os.environ.get("SCORE_RULE") == "phys" and not os.environ.get("HARNESS_NORM")
CACHE = Path(os.environ.get("SCORE_MASK_CACHE") or paths.T5_DESIGN)   # per-row caches rows_<label>.npz
E, SP = S.E, S.SP
LABS = {"FINAL": "e13s_FINw1536_s2", "F2": "e13f_F2ev19_s2_fx", "B": "e13s_FINw1024_s2", "G1": "e13s_G1w1536_s2",
        "A": "e13s_S2U2w1536_s2"}
HSETS = {"G": ("g500", "H_gtest_dt500.npz"), "Gc": ("gc500", "H_gtest_cold_dt500.npz"), "H2": ("h2", "H_h2_dt500.npz"),
         "H2c": ("h2c", "H_h2c_dt500.npz"), "H3": ("h3", "H_h3_dt500.npz"), "H3c": ("h3c", "H_h3c_dt500.npz")}
PSETS = {"PD": 2500, "PX": 2500, "P2l": 500, "P2s": 50, "PH": 2500}   # late-window start (calls), as board6.py
EDGES = [300, 800, 1300, 1800, 2300, 3001]


def box(P_lo=1e-5, P_hi=100.0, T_lo=-np.inf, T_hi=np.inf):
    return lambda T, P: (P >= P_lo) & (P <= P_hi) & (T >= T_lo) & (T <= T_hi)


def from_hist(logp_edges, T_edges, occ, min_occ=1):
    lp, te, occ = np.asarray(logp_edges, float), np.asarray(T_edges, float), np.asarray(occ)
    assert occ.shape == (lp.size - 1, te.size - 1), "occ must be [log10 P bins, T bins]"

    def inside(T, P):
        T, x = np.asarray(T, float), np.log10(np.asarray(P, float))
        i, j = np.searchsorted(lp, x, side="right") - 1, np.searchsorted(te, T, side="right") - 1
        i = np.where(x == lp[-1], lp.size - 2, i)                       # right edge belongs to the last bin
        j = np.where(T == te[-1], te.size - 2, j)
        ok = (i >= 0) & (i < lp.size - 1) & (j >= 0) & (j < te.size - 1)
        return ok & (occ[np.clip(i, 0, lp.size - 2), np.clip(j, 0, te.size - 2)] >= min_occ)
    return inside


def rows(label):
    """Per-row summaries over the six fixed sets (concatenated), cached in rows_<label>.npz."""
    files = [E / f"H_{pre}_{label}" / fn for pre, fn in HSETS.values()]
    cache = CACHE / f"rows_{label}.npz"
    if cache.exists() and cache.stat().st_mtime > max(f.stat().st_mtime for f in files):
        return dict(np.load(cache))
    out = {k: [] for k in ("set", "job", "t0", "T", "P", "late", "first", "last", "late_sp", "late_t", "last_sp")}
    for k, f in zip(HSETS, files):
        B, d = S.bench(f), np.load(f)
        e, e1 = np.nan_to_num(B["e"], nan=0.0, posinf=np.inf), np.nan_to_num(B["e1"], nan=0.0, posinf=np.inf)
        assert e1.shape[1] == 1 and d["calls"][-1] * float(d["dt"]) == 8.64e7, f   # first call alone early; end = 1000 days
        n = e.shape[0]
        flat = e.reshape(n, -1).argmax(1)
        c, q = np.unravel_index(flat, e.shape[1:])
        out["set"] += [k] * n
        out["job"].append(B["job"])
        out["t0"].append(d["t0"])
        out["T"].append(B["T"])
        out["P"].append(B["P"])                                         # bar (scoreboard.bench divides by 1e6)
        out["late"].append(e.max(axis=(1, 2)))
        out["first"].append(e1.max(axis=(1, 2)))
        out["last"].append(e[:, -1].max(1))
        out["late_sp"] += [SP[j] for j in q]
        out["late_t"].append(d["t0"] + d["calls"][1:][c] * float(d["dt"]))
        out["last_sp"] += [SP[j] for j in e[:, -1].argmax(1)]
    r = {k: np.array(v) if k in ("set", "late_sp", "last_sp") else np.concatenate(v) for k, v in out.items()}
    np.savez(cache, **r)
    return r


def paths(label, pre):
    """Per-gas error [C, paths, 12] (NaN = not scored), per-call T [paths, C] and P in bar, as scoreboard.path_board."""
    d = np.load(E / f"{pre}_{label}" / "P_paths.npz")
    st, calls = S.cut12(d["states"][:, 2]), d["calls"]
    pdir = Path(str(d["p_dir"]))
    vt, res = [], []
    for k, p in enumerate(d["path"]):
        q = np.load(pdir / f"{int(p)}.npz")
        cols = [list(q["species"]).index(s) for s in S.D.SHORT]
        x = np.log10(np.maximum(q["x"][calls][:, cols], S.D.EPS))
        n = np.log10(q["Pbar"][calls - 1] * 1e6 / (S.KB * q["T"][calls - 1]))[:, None]
        assert np.allclose(d["T"][k], q["T"][calls - 1]) and np.allclose(d["Pbar"][k], q["Pbar"][calls - 1])
        vt.append(x)
        res.append((x + n >= -2.0) | (st[:, k] + n >= -2.0))
    vt, res = np.stack(vt, 1), np.stack(res, 1)
    return SR.err(vt, st, res, SP), calls, d["T"], d["Pbar"]


def fixed_cells(r, m):
    lm, fm, la = r["late"][m], r["first"][m], r["last"][m]
    return {"rows": int(m.sum()), "late_gt05": int((lm > 0.5).sum()), "late_gt2": int((lm > 2).sum()),
            "first_gt2": int((fm > 2).sum()), "max": float(max(lm.max(), fm.max())) if m.any() else float("nan"),
            "last_gt05": int((la > 0.5).sum()), "last_gt2": int((la > 2).sum())}


def path_cells(e, calls, cin, late0):
    """e [C, paths, 12]; cin [C, paths] = call inside the side."""
    late = (calls >= late0)[:, None] & cin
    q = e[late][np.isfinite(e[late])]
    qa = e[cin][np.isfinite(e[cin])]
    lc = np.nan_to_num(e, nan=0.0, posinf=np.inf).max(-1)[late]
    return {"call_share": float(cin.mean()), "late_calls": int(late.sum()),
            "late_p50": float(np.percentile(q, 50)) if q.size else float("nan"),
            "late_p99": float(np.percentile(q, 99)) if q.size else float("nan"),
            "late_max": float(q.max()) if q.size else float("nan"),
            "late_call_share_gt05": float((lc > 0.5).mean()) if lc.size else float("nan"),
            "max_all": float(qa.max()) if qa.size else float("nan")}


def table(label, inside, path_sets=tuple(PSETS)):
    r = rows(label)
    m = inside(r["T"], r["P"])
    t = {"fixed": {"in": fixed_cells(r, m), "out": fixed_cells(r, np.logical_not(m)),
                   "all": fixed_cells(r, np.ones_like(m))},
         "fixed_in_bands": {f"{lo}-{hi}": fixed_cells(r, m & (r["T"] >= lo) & (r["T"] < hi)) for lo, hi in zip(EDGES[:-1], EDGES[1:])},
         "fixed_out_bands": {f"{lo}-{hi}": fixed_cells(r, np.logical_not(m) & (r["T"] >= lo) & (r["T"] < hi))
                             for lo, hi in zip(EDGES[:-1], EDGES[1:])},
         "per_set": {k: {"in": fixed_cells(r, m & (r["set"] == k)), "out": fixed_cells(r, np.logical_not(m) & (r["set"] == k))}
                     for k in HSETS},
         "paths": {}}
    for pre in path_sets:
        e, calls, T, P = paths(label, pre)
        cin = inside(T, P).T                                               # [C, paths]
        last = np.nan_to_num(e[-1], nan=0.0, posinf=np.inf).max(-1)                       # [paths]
        t["paths"][pre] = {"in": path_cells(e, calls, cin, PSETS[pre]), "out": path_cells(e, calls, np.logical_not(cin), PSETS[pre]),
                           "all": path_cells(e, calls, np.ones_like(cin), PSETS[pre]),
                           "paths": int(e.shape[1]), "last_call_in": int(cin[-1].sum()),
                           "last_gt05_in": int(((last > 0.5) & cin[-1]).sum()), "last_gt2_in": int(((last > 2) & cin[-1]).sum()),
                           "last_gt05_out": int(((last > 0.5) & np.logical_not(cin[-1])).sum()),
                           "last_gt2_out": int(((last > 2) & np.logical_not(cin[-1])).sum())}
    return t


def fmt(t, name=""):
    L = [f"== {name}"]
    for side in ("in", "out", "all"):
        c = t["fixed"][side]
        L.append(f"  fixed {side:3s}: rows {c['rows']:5d} | late > 2 {c['late_gt2']:3d}, > 0.5 {c['late_gt05']:4d} | first > 2 "
                 f"{c['first_gt2']} | max {c['max']:.2f} | lasting (1000 d) > 0.5 {c['last_gt05']:3d}, > 2 {c['last_gt2']}")
    for side in ("in", "out"):
        for b, c in t[f"fixed_{side}_bands"].items():
            L.append(f"    {side:3s} {b:9s} rows {c['rows']:5d} | late > 2 {c['late_gt2']:3d}, > 0.5 {c['late_gt05']:4d} | first > 2 "
                     f"{c['first_gt2']} | max {c['max']:.2f} | lasting > 0.5 {c['last_gt05']:3d}, > 2 {c['last_gt2']}")
    for pre, p in t["paths"].items():
        for side in ("in", "out", "all"):
            c = p[side]
            L.append(f"  {pre:3s} {side:3s}: calls {c['call_share']:.3f} | late p50 {c['late_p50']:.3f} p99 {c['late_p99']:.3f} max "
                     f"{c['late_max']:.2f} | late calls > 0.5 {c['late_call_share_gt05']:.4f} | max all calls {c['max_all']:.2f}")
        L.append(f"  {pre:3s} last call: in {p['last_call_in']}/{p['paths']} paths; > 0.5 in {p['last_gt05_in']}, out "
                 f"{p['last_gt05_out']}; > 2 in {p['last_gt2_in']}, out {p['last_gt2_out']}")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--box", nargs=4, type=float, metavar=("P_lo", "P_hi", "T_lo", "T_hi"))
    ap.add_argument("--hist")
    ap.add_argument("--keys", default="logp_edges,T_edges,occ")
    ap.add_argument("--min-occ", type=float, default=1)
    a = ap.parse_args()
    if a.hist:
        h = np.load(a.hist)
        inside = from_hist(*[h[k] for k in a.keys.split(",")], min_occ=a.min_occ)
    else:
        inside = box(*a.box) if a.box else box()
    res, txt = {}, []
    for lab in a.labels or LABS.values():
        res[lab] = table(lab, inside)
        txt.append(fmt(res[lab], lab))
        print(txt[-1], flush=True)
    (paths.OUT / f"score_mask_{a.name}.txt").write_text("\n".join(txt) + "\n")
    (paths.OUT / f"score_mask_{a.name}.json").write_text(json.dumps(res, indent=1))
