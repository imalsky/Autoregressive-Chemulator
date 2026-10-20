"""Sprint 2026-10-08 scorer (plan item 2 and section 6), built on E13_gcm/analyze_e13.py's load / table / p90_late (imported;
its module body runs only under __main__ since the sprint's import guard). For every run and test it reports, per T band:
the late-max p90 pooled over the three starts (the gate), per start, two corners (Z > 30 and C/O > 0.9; Z < 0.3), four
species (H, OH, O, C2H2), and a single-arm 95% job-bootstrap interval (whole jobs with all their starts resampled; the
late max recomputed inside every resample). Pairs: a paired job bootstrap (common job draws for both sides; each pair
seeded from its own name), each side the mean over its seeds, seeds matched to their own controls. Paths (arm P): the
pooled p90 over calls >= 50 (paths x calls x resolved species) and the max over calls of the per-call p90, per band of
the path's mean T, with a paired path bootstrap. Writes scores_sprint.json to paths.OUT.

    python score_sprint.py [run ...]      (default: every run in RUNS that has an output)
"""
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
E = paths.E
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_e13 as A  # noqa: E402

OUT = paths.OUT   # outputs (2026-10-09: were written beside this file)
EDGES = A.EDGES
NB = 1000
SPECIES = ("H", "OH", "O", "C2H2")
SCR = paths.SCRATCH
RUNS = ["e13_capw17_s0", "e13_capw17_s1", "e13_capw17r_s1", "e13_capw17r_s2", "e13_d17c_s0", "e13_d18c_s0", "e13_d18c_s1",
        "e13_s2long17_s0", "e13_s2long17_s1", "e13_m17_s0", "e13_m17_s1", "e13_relaxTz17_s0", "e13_relaxTz17_s1",
        "e13_big17lr_s0", "e13_l17_s0", "e13_m17f_s0", "e13_ms17_s0", "e13_m17f_s1", "e13_m17fs_s0", "e13_m17fs_s1", "e13_m64f_s0", "e13_m17ffc_s0", "e13_m18fd_s0", "e13_m17fT_s0", "e13_m17fj_s0", "e13_m17fw_s0", "e13_m17fT_s1", "e13_m18fd_s1", "e13_f0fs_s0", "e13_f0fs_s1", "e13_allGend16_s0", "e13_allGendF16_s1", "e13_big17_s0", "e13_f017_s0"]
# test: (bench folder pattern, file, truth tag override or None, jobs.json for the corners)
TESTS = {"G dt500": ("H_g500_{}", "H_gtest_dt500.npz", None, E / "gtest/jobs.json"),
         "G dt100": ("H_g100_{}", "H_gtest_dt100.npz", None, E / "gtest/jobs.json"),
         "Gc dt500": ("H_gc500_{}", "H_gtest_cold_dt500.npz", None, E / "gtest_cold/jobs.json"),
         "cold tag": ("H_ccold500_{}", "H_e13_ctruth_cold_dt500.npz", SCR / "e13_ctruth_cold", None),
         "seen tag": ("H_cseen500_{}", "H_e13_ctruth_seen_dt500.npz", SCR / "e13_ctruth_seen", None),
         # readiness (plan section 5, R-early / R-full): spin-up schedule, unclamped dt > 500 s, random dt 10-500 s
         "R startup": ("R_startup_{}", "H_gtest_dt500_startup.npz", None, E / "gtest/jobs.json"),
         "R dt1000u": ("R_dt1000u_{}", "H_gtest_dt1000_unclamped.npz", None, E / "gtest/jobs.json"),
         "R dt2000u": ("R_dt2000u_{}", "H_gtest_dt2000_unclamped.npz", None, E / "gtest/jobs.json"),
         "R dt3600u": ("R_dt3600u_{}", "H_gtest_dt3600_unclamped.npz", None, E / "gtest/jobs.json"),
         "R rand10_500": ("R_rand_{}", "H_gtest_dt125.25_rand10_500.npz", None, E / "gtest/jobs.json")}
# (arm run, its control run): seeds matched to the control with the same stage 1 and the same stage-2 generator
PAIRS = {"C2 stage-2 replicate": [("e13_capw17r_s1", "e13_capw17_s0")],
         "C2b stage-2 replicate": [("e13_capw17r_s2", "e13_capw17_s0")],
         "C1 from-scratch replicate": [("e13_capw17_s1", "e13_capw17_s0")],
         "C1 - C2 (stage-1 noise)": [("e13_capw17_s1", "e13_capw17r_s1")],
         "from-scratch pair allGendF - allGend": [("e13_allGendF16_s1", "e13_allGend16_s0")],
         "capw17 - allGend": [("e13_capw17_s0", "e13_allGend16_s0")],
         "D1a": [("e13_d17c_s0", "e13_capw17_s0")],
         "D1b": [("e13_d18c_s0", "e13_capw17_s0"), ("e13_d18c_s1", "e13_capw17r_s1")],
         "S": [("e13_s2long17_s0", "e13_capw17_s0"), ("e13_s2long17_s1", "e13_capw17r_s1")],
         "M": [("e13_m17_s0", "e13_capw17_s0"), ("e13_m17_s1", "e13_capw17r_s1")],
         "T": [("e13_relaxTz17_s0", "e13_capw17_s0"), ("e13_relaxTz17_s1", "e13_capw17r_s1")],
         "L": [("e13_l17_s0", "e13_capw17_s0")],
         "F0": [("e13_f017_s0", "e13_capw17_s0")],
         "M s0": [("e13_m17_s0", "e13_capw17_s0")], "S s0": [("e13_s2long17_s0", "e13_capw17_s0")],
         "T s0": [("e13_relaxTz17_s0", "e13_capw17_s0")], "D1b s0": [("e13_d18c_s0", "e13_capw17_s0")],
         "W": [("e13_big17lr_s0", "e13_capw17_s0")], "M from scratch": [("e13_m17f_s0", "e13_capw17_s0")],
         "M+S s0": [("e13_ms17_s0", "e13_capw17_s0")], "M+S vs S s0": [("e13_ms17_s0", "e13_s2long17_s0")],
         "M+S vs M s0": [("e13_ms17_s0", "e13_m17_s0")],
         "M from scratch 2 seeds": [("e13_m17f_s0", "e13_capw17_s0"), ("e13_m17f_s1", "e13_capw17_s1")],
         "M from scratch + S": [("e13_m17fs_s0", "e13_capw17_s0")], "M fs + S vs M fs": [("e13_m17fs_s0", "e13_m17f_s0")],
         "FiLM fs": [("e13_m17ffc_s0", "e13_m17f_s0")], "M+D fs": [("e13_m18fd_s0", "e13_m17f_s0")],
         "relaxT fs": [("e13_m17fT_s0", "e13_m17f_s0")], "jump in s1 fs": [("e13_m17fj_s0", "e13_m17f_s0")],
         "wide dt fs": [("e13_m17fw_s0", "e13_m17f_s0")], "M fs s1 - s0 (seed spread)": [("e13_m17f_s1", "e13_m17f_s0")],
         "M fs + S 2 seeds": [("e13_m17fs_s0", "e13_capw17_s0"), ("e13_m17fs_s1", "e13_capw17_s1")],
         "M64 from scratch": [("e13_m64f_s0", "e13_capw17_s0")], "M64 vs M fs": [("e13_m64f_s0", "e13_m17f_s0")],
         "M fs + S vs M fs 2 seeds": [("e13_m17fs_s0", "e13_m17f_s0"), ("e13_m17fs_s1", "e13_m17f_s1")],
         "M fs + S s1 - s0 (seed spread)": [("e13_m17fs_s1", "e13_m17fs_s0")],
         "relaxT fs 2 seeds": [("e13_m17fT_s0", "e13_m17f_s0"), ("e13_m17fT_s1", "e13_m17f_s1")],
         "M+D fs 2 seeds": [("e13_m18fd_s0", "e13_m17f_s0"), ("e13_m18fd_s1", "e13_m17f_s1")],
         "F0fs vs M fs + S 2 seeds": [("e13_f0fs_s0", "e13_m17fs_s0"), ("e13_f0fs_s1", "e13_m17fs_s1")],
         "F0fs 2 seeds": [("e13_f0fs_s0", "e13_capw17_s0"), ("e13_f0fs_s1", "e13_capw17_s1")],
         "F0fs s1 - s0 (seed spread)": [("e13_f0fs_s1", "e13_f0fs_s0")]}


def rng_for(name):
    return np.random.default_rng(zlib.crc32(name.encode()))


def bands(T):
    return [(f"{lo}-{hi}", (T >= lo) & (T < hi)) for lo, hi in zip(EDGES[:-1], EDGES[1:])]


def job_rows(X, m):
    jobs = np.unique(X["job"][m])
    return [np.flatnonzero(m & (X["job"] == j)) for j in jobs]


def ci_single(X, m, name):
    rows, r = job_rows(X, m), rng_for(name)
    v = [A.p90_late(X["e"][np.concatenate([rows[k] for k in r.integers(0, len(rows), len(rows))])]) for _ in range(NB)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def score_file(f, tag, jobs_json, name):
    X = A.load(f, tag)
    t = A.table(X)
    Z = CO = None
    if jobs_json is not None and Path(jobs_json).exists():
        J = json.loads(Path(jobs_json).read_text())
        Z = np.array([J[j]["Z"] for j in X["job"]])
        CO = np.array([J[j]["C_O"] for j in X["job"]])
    sp = list(A.D.SHORT)
    for b, m in bands(X["T"]):
        if b not in t:
            continue
        t[b]["ci95"] = ci_single(X, m, f"{name} {b}")
        # tail guards (2026-10-08, author: p90 alone hides the tail): late-max p99, and rows with any resolved species
        # off by > 0.5 dex at any late checkpoint
        em = X["e"][m]
        t[b]["late_max_p99"] = float(max(np.nanpercentile(em[:, c], 99) for c in range(em.shape[1]) if np.isfinite(em[:, c]).any()))
        t[b]["rows_gt0.5"] = int((np.nan_to_num(em, nan=0.0).max(axis=(1, 2)) > 0.5).sum())
        t[b]["by_species"] = {s: A.p90_late(X["e"][m][:, :, sp.index(s):sp.index(s) + 1]) for s in SPECIES}
        if Z is not None:
            for cn, cm in (("Z>30&CO>0.9", (Z > 30) & (CO > 0.9)), ("Z<0.3", Z < 0.3)):
                mm = m & cm
                t[b][f"corner {cn}"] = {"p90": A.p90_late(X["e"][mm]) if mm.any() else None, "jobs": int(np.unique(X["job"][mm]).size)}
    return X, t


def boot_pairs(pairs, L, name):
    """Paired job bootstrap: diff = mean over seeds k of (stat(A_k) - stat(B_k)), common job draws for every term."""
    Xs = [(L[a], L[b]) for a, b in pairs]
    X0 = Xs[0][0]
    for x, y in Xs:
        assert np.array_equal(x["job"], X0["job"]) and np.array_equal(y["job"], X0["job"]) and np.array_equal(x["t0"], X0["t0"])
    r, out = rng_for(name), {}
    for b, m in bands(X0["T"]):
        if not m.any():
            continue
        rows = job_rows(X0, m)
        d0 = np.mean([A.p90_late(x["e"][m]) - A.p90_late(y["e"][m]) for x, y in Xs])
        ds = []
        for _ in range(NB):
            idx = np.concatenate([rows[k] for k in r.integers(0, len(rows), len(rows))])
            ds.append(np.mean([A.p90_late(x["e"][idx]) - A.p90_late(y["e"][idx]) for x, y in Xs]))
        out[b] = {"diff": float(d0), "ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))], "seeds": len(Xs)}
    return out


def pool_cold(Xg, Xc):
    """The cold instrument (plan section 13, Gate 0): G and Gc rows below 1300 K in one set; Gc jobs offset by 10^6."""
    mg, mc = Xg["T"] < 1300, Xc["T"] < 1300
    assert Xg["e"].shape[1:] == Xc["e"].shape[1:]
    out = {"nint": Xg["nint"] + Xc["nint"]}
    for k, v in Xg.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == Xg["job"].shape:
            out[k] = np.concatenate([v[mg], (Xc[k] + 10 ** 6 if k == "job" else Xc[k])[mc]])
    return out


def paths_errors(f):
    """Arm P errors [calls, paths, 12] (NaN = neither truth nor prediction resolved), calls, path mean T."""
    d = np.load(f)
    st = A.cut12(d["states"][:, 2])
    calls = d["calls"]
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
    return np.where(res, np.abs(st - vt), np.nan), calls, d["T"].mean(1)


def paths_stats(e, calls, idx):
    late = calls >= 50
    v = e[late][:, idx].ravel()
    v = v[np.isfinite(v)]
    per = [np.nanpercentile(e[c, idx], 90) for c in range(e.shape[0]) if np.isfinite(e[c, idx]).any()]
    return float(np.percentile(v, 90)) if v.size else np.nan, float(max(per)) if per else np.nan


def main(only):
    out = {"tables": {}, "pairs": {}}
    runs = only or RUNS
    LS = {}
    for test, (pat, fn, tag, jj) in ({} if os.environ.get("PATHS_ONLY") else TESTS).items():  # PATHS_ONLY=1: paths sets only
        L = LS[test] = {}
        for r in runs:
            f = E / pat.format(r) / fn
            if not f.exists():
                continue
            X, t = score_file(f, tag, jj, f"{test} {r}")
            L[r] = X
            out["tables"][f"{test} {r}"] = t
            print(f"{test:9s} {r:18s} nint {X['nint']} | " + " | ".join(
                f"{b}: {v['late_max_p90']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}] (" +
                "/".join(f"{x:.3f}" for x in v["late_max_p90_by_start"].values()) + ")" for b, v in t.items()))
            print(f"{'  tail':9s} {r:18s} p99 | " + " | ".join(
                f"{b}: {v['late_max_p99']:.3f}, rows >0.5 dex {v['rows_gt0.5']}/{v['rows']}" for b, v in t.items()))
        for pn, pairs in PAIRS.items():
            if all(a in L and b in L for a, b in pairs):
                r = out["pairs"][f"{test} {pn}"] = boot_pairs(pairs, L, f"{test} {pn}")
                print(f"  {test} {pn} ({len(pairs)} seed pair(s)): " + "; ".join(
                    f"{b} {v['diff']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]" for b, v in r.items()))
    L = {r: pool_cold(LS["G dt500"][r], LS["Gc dt500"][r]) for r in LS.get("G dt500", {}) if r in LS["Gc dt500"]}
    for r, X in L.items():
        t = out["tables"][f"G+Gc dt500 {r}"] = A.table(X)
        for b, m in bands(X["T"]):
            if b in t:
                t[b]["ci95"] = ci_single(X, m, f"G+Gc dt500 {r} {b}")
        print(f"G+Gc dt500 {r:18s} " + " | ".join(f"{b}: {v['late_max_p90']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}] "
                                                 f"({v['rows']} rows)" for b, v in t.items()))
    for pn, pairs in PAIRS.items():
        if all(a in L and b in L for a, b in pairs):
            r = out["pairs"][f"G+Gc dt500 {pn}"] = boot_pairs(pairs, L, f"G+Gc dt500 {pn}")
            print(f"  G+Gc dt500 {pn} ({len(pairs)} seed pair(s)): " + "; ".join(
                f"{b} {v['diff']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]" for b, v in r.items()))
    for pset, pat in (() if os.environ.get("NO_PATHS") else (("paths", "P_{}"), ("paths_v2s", "P2s_{}"), ("paths_v2l", "P2l_{}"))):  # 64 paths; v2 short, long; NO_PATHS=1: skip
        P = {}
        out.setdefault(pset, {})
        out.setdefault(pset + "_pairs", {})
        for r in runs:
            f = E / pat.format(r) / "P_paths.npz"
            if f.exists():
                e, calls, Tm = P[r] = paths_errors(f)
                res = {}
                for b, m in [("all", np.ones_like(Tm, bool))] + bands(Tm):
                    if m.any():
                        pool, mx = paths_stats(e, calls, np.flatnonzero(m))
                        res[b] = {"pool50_p90": pool, "max_p90": mx, "paths": int(m.sum())}
                out[pset][r] = res
                print(f"{pset:9s} {r:18s} " + " | ".join(f"{b}: {v['pool50_p90']:.3f}/{v['max_p90']:.3f}" for b, v in res.items()))
        for pn, pairs in PAIRS.items():
            if all(a in P and b in P for a, b in pairs):
                rr, res = rng_for(f"{pset} {pn}"), {}
                Tm = P[pairs[0][0]][2]
                for b, m in [("all", np.ones_like(Tm, bool))] + bands(Tm):
                    ids = np.flatnonzero(m)
                    if ids.size == 0:
                        continue
                    f0 = lambda idx: np.mean([paths_stats(P[a][0], P[a][1], idx)[0] - paths_stats(P[c][0], P[c][1], idx)[0] for a, c in pairs])
                    d0 = f0(ids)
                    ds = [f0(ids[rr.integers(0, ids.size, ids.size)]) for _ in range(NB)]
                    res[b] = {"diff_pool50": float(d0), "ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]}
                out[pset + "_pairs"][pn] = res
                print(f"  {pset} {pn}: " + "; ".join(f"{b} {v['diff_pool50']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]" for b, v in res.items()))
    dst = OUT / ("scores_sprint.json" if not only else "scores_sprint_partial.json")
    if os.environ.get("PATHS_ONLY"):
        dst = OUT / "scores_sprint_paths.json"
    if os.environ.get("NO_PATHS"):
        dst = OUT / "scores_sprint_nopaths.json"
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(dst)


if __name__ == "__main__":
    main(sys.argv[1:])
