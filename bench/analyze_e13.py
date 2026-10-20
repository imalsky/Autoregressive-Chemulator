"""E13 analysis (run_e13_gpu.sh models). For every model and test, per T band, over the 12 reported species (a 53-species
model's states are cut to them): late-max p90 (the worst checkpoint at t >= 1e3 s of the per-checkpoint p90 of
|dlog10 y| over rows x resolved species; resolved: truth number density >= 1e-2 cm^-3, as analyze_e9.armH), end p90
(last checkpoint), rows > 3 dex at the end, collapses (state at the floor while the truth is above 1e-15); in brackets
the late-max p90 over species whose truth is also >= 1e-8 (the default mask admits mixing ratios down to about 1e-21).
Target (author, 2026-10-06): p90 <= 0.1 dex after a full GCM-length run in every band.
Scored mask since 2026-10-07 (author: small mixing ratios can carry high opacities): a (row, checkpoint, species) counts
when the truth OR the prediction is at >= 1e-2 cm^-3, so over-predicting a species whose truth is below the threshold
is scored too; "t" = the earlier truth-only mask, printed for comparison.
Tests: G at dt 10 (1e6 calls), 100, 500 s; U at dt 10; paths (P: T, P change every call; every call scored).
G dt100 tight: rollouts started from, and scored against, the rtol 0.005 truth (gtest_tight). "TARGET MET" is printed
only for full-length tests (G dt100, G dt500, G dt100 tight) with no guard interventions; dt 10 (116 days) is a screen.
Run without HARNESS_NORM: the reported set is the production 12 species. Since 2026-10-07 each band table also holds
late_max_p90_by_start (the same statistic for each start t0 alone) and the seen/unseen pseudo-truth tables (c<tag> <run>). Paired bootstrap between arms (resampling unit the test job, its three starts together, per band; each side the mean
over its seeds). Skips what is not written.

    python analyze_e13.py          (writes paths.OUT/scores_e13.json; prints the tables)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
E = paths.E
import score as SC  # noqa: E402
import data as D  # noqa: E402

assert len(D.SHORT) == 12, "run without HARNESS_NORM: the reported set is the production 12 species"
KB = 1.380649e-16
EDGES = [300, 800, 1300, 1800, 2300, 3001]
SP53 = [str(s) for s in np.load(paths.VREF / "ref_a" / "0.npz")["species"]]   # network order, from a VULCAN reference box
ARMS = ("phys", "all", "pp", "old", "physwide", "all53u", "all53c", "allJ", "allG", "all53cfix", "res", "allJG", "physG", "resG", "all17G", "allG0", "allGst", "allGend", "allGun", "allGan", "allGcmb", "allGr", "allGw", "allGwf", "dcold", "drand", "capw", "all17E", "capw17", "big17", "allGendF", "capw17c", "base", "e12")  # base: production e9_F10fill_s0; e12: the E12 pilot control (G dt100 only)
TESTS = {"G dt10": ("H_g10_{}", "H_gtest_dt10.npz"), "G dt100": ("H_g100_{}", "H_gtest_dt100.npz"),
         "G dt500": ("H_g500_{}", "H_gtest_dt500.npz"), "U dt10": ("H_u10_{}", "H_utest_dt10.npz"),
         "G dt100 tight": ("H_gt100_{}", "H_gtest_tight_dt100.npz")}
FULL = ("G dt100", "G dt500", "G dt100 tight")
PAIRS = [("phys", "all"), ("pp", "all"), ("phys", "pp"), ("old", "phys"), ("physwide", "phys"), ("all53u", "all"),
         ("all53c", "all"), ("all53c", "all53u"), ("phys", "base"), ("all", "base"), ("allJ", "all"), ("allG", "all"),
         ("allG", "allJ"), ("allG", "base"), ("res", "all53c"), ("all53cfix", "all53c"),
         ("allJG", "allG"), ("physG", "allG"), ("resG", "allG"), ("resG", "res"), ("all17G", "allG"),
         ("allGst", "allG0"), ("allGend", "allG0"), ("allGun", "allG0"), ("allGan", "allG0"),
         ("allGcmb", "allG0"), ("allGcmb", "allGend"), ("allGr", "allG0"),
         ("allGw", "allG0"), ("allGwf", "allG"),
         ("dcold", "drand"), ("dcold", "allGend"), ("drand", "allGend"), ("allGendF", "allGend"),
         ("capw", "allGend"), ("all17E", "allGend"), ("capw17", "capw"), ("capw17", "all17E"), ("big17", "capw17"),
         ("big17", "all17E"), ("capw17c", "capw17"),
         ("capw17", "allGend"), ("big17", "allGend"), ("allGwf", "allG0")]  # 2026-10-07: pairs quoted from an_v9/v11/v12
NB = 1000
rng = np.random.default_rng(0)


def runs(arm):
    if arm == "all17G":
        return ["e13_all17G_s0"]
    if arm in ("all17E", "capw17", "big17"):  # 2^17-cache runs (E13 v11), seed 0
        return [f"e13_{arm}_s0"]
    if arm == "capw17c":  # capw17 recipe on the cold-enriched cache arch_gcm17cold_all (E13 v13, prepared 15:45), two seeds
        return ["e13_capw17c_s0", "e13_capw17c_s1"]
    if arm == "allGendF":  # allGend recipe from scratch, seed 1, allG's rows: the from-scratch noise replicate
        return ["e13_allGendF16_s1"]
    if arm == "allGr":  # allG config, stage-2 seed 1, same stage 1 as allG s0 (training-noise replicate)
        return ["e13_allGr16_s1"]
    if arm == "allG0":  # seed 0 alone: the exact control of the 2026-10-07 review branches (same stage 1, same seed)
        return ["e13_allG16_s0"]
    if arm == "e12":
        return ["e12_mlp512x6_sob16_s0"]
    return ["e9_F10fill_s0"] if arm == "base" else [f"e13_{arm}16_s{s}" for s in (0, 1)]


RESP = paths.RES_NORM
SPRES = [s.replace("_evolution", "") for s in json.loads(RESP.read_text())["species_variables"]] if RESP.exists() else []


def cut12(st):  # 53-species and reservoir-set states -> the 12 reported species
    for sp in (SP53, SPRES):
        if st.shape[-1] == len(sp) and st.shape[-1] != len(D.SHORT):
            return st[..., [sp.index(s) for s in D.SHORT]]
    return st


def load(f, tag=None):
    """Errors over the 12 reported species for an arm-H file: e [rows, late calls, 12] (NaN = unresolved), plus
    collapse flags, job, t0, T. tag replaces the file's truth (same job indices)."""
    d = np.load(f)
    st = cut12(d["states"][:, 2])                                          # [C, R, 12]
    job, t0, dt, calls = d["job"], d["t0"], float(d["dt"]), d["calls"]
    tag = str(d["tag"]) if tag is None else str(tag)
    tag = tag.replace("Chemulator_Project/sweep_2026-10-03", "Chemulator_Project/experiments/2026-10-03_repo-sweep")
    el = d["tcum"].T if "tcum" in d.files else np.broadcast_to(calls[None, :] * dt, (t0.size, calls.size))  # [R, C] s
    times = t0[:, None] + el                                               # a dt schedule (sched) saves per-row tcum
    vt = np.stack([SC.vtruth(tag, j, times[i]) for i, j in enumerate(job)])  # [R, C, 12]
    s = np.transpose(st, (1, 0, 2))
    e = np.abs(s - vt)
    lnd = np.log10(d["P"] / (KB * d["T"]))[:, None, None]
    res = (vt + lnd) >= -2.0
    either = res | ((s + lnd) >= -2.0)
    coll = ((s <= -29.0) & (vt > -15.0)).any(-1).any(-1)
    late = np.flatnonzero(np.median(el, 0) >= 1e3)
    return {"e": np.where(either, e, np.nan)[:, late], "et": np.where(res, e, np.nan)[:, late],
            "e8": np.where(res & (vt >= -8.0), e, np.nan)[:, late],
            "job": job, "t0": t0, "T": d["T"], "coll": coll, "nint": int(d["nint"].sum())}


def p90_late(e):
    v = [np.nanpercentile(e[:, c], 90) for c in range(e.shape[1]) if np.isfinite(e[:, c]).any()]
    return float(max(v)) if v else np.nan


def p90_end(e):
    v = e[:, -1].ravel()
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 90)) if v.size else np.nan


def table(X):
    r = {}
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (X["T"] >= lo) & (X["T"] < hi)
        if m.any():
            em = np.nan_to_num(X["e"][m][:, -1], nan=0.0).max(-1)
            r[f"{lo}-{hi}"] = {"late_max_p90": p90_late(X["e"][m]), "end_p90": p90_end(X["e"][m]), "rows": int(m.sum()),
                               "rows_gt3dex_end": int((em > 3).sum()), "collapse_rows": int(X["coll"][m].sum()),
                               "late_max_p90_y1e-8": p90_late(X["e8"][m]), "end_p90_y1e-8": p90_end(X["e8"][m]),
                               "late_max_p90_truthmask": p90_late(X["et"][m]),
                               "late_max_p90_by_start": {f"{s:g}": p90_late(X["e"][m & (X["t0"] == s)])  # 2026-10-07
                                                         for s in np.unique(X["t0"][m])}}
    return r


def boot(As, Bs):
    X, r = As[0], {}
    assert all(np.array_equal(X["job"], Y["job"]) and np.array_equal(X["t0"], Y["t0"]) for Y in As + Bs)
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (X["T"] >= lo) & (X["T"] < hi)
        if not m.any():
            continue
        jobs = np.unique(X["job"][m])
        rows = [np.flatnonzero(m & (X["job"] == j)) for j in jobs]
        for name, fn in (("late", p90_late), ("end", p90_end)):
            side = lambda Ys, idx: np.mean([fn(Y["e"][idx]) for Y in Ys])
            d0 = side(As, m) - side(Bs, m)
            ds = []
            for _ in range(NB):
                idx = np.concatenate([rows[k] for k in rng.integers(0, len(jobs), len(jobs))])
                ds.append(side(As, idx) - side(Bs, idx))
            r[f"{lo}-{hi} {name}"] = {"diff": float(d0), "ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]}
    return r


def paths_score(f):
    """Arm P: every call scored against the chained VULCAN truth; per band of the path's mean T and overall: max over
    calls of the per-call p90 over paths x resolved species, and the p90 at the last call."""
    d = np.load(f)
    st = cut12(d["states"][:, 2])                                          # [C, R, 12]
    calls = d["calls"]
    pdir = Path(str(d["p_dir"]))
    vt, res = [], []
    for k, p in enumerate(d["path"]):
        q = np.load(pdir / f"{int(p)}.npz")
        cols = [list(q["species"]).index(s) for s in D.SHORT]
        x = np.log10(np.maximum(q["x"][calls][:, cols], D.EPS))            # truth after call c is x[c]
        n = q["Pbar"][calls - 1] * 1e6 / (KB * q["T"][calls - 1])
        vt.append(x)
        res.append((x + np.log10(n)[:, None] >= -2.0) | (st[:, k] + np.log10(n)[:, None] >= -2.0))  # truth or prediction
    vt, res = np.stack(vt, 1), np.stack(res, 1)                            # [C, R, 12]
    e = np.where(res, np.abs(st - vt), np.nan)
    Tm = d["T"].mean(1)
    out = {}
    for name, m in [("all", np.ones_like(Tm, bool))] + [(f"{lo}-{hi}", (Tm >= lo) & (Tm < hi)) for lo, hi in
                                                         zip(EDGES[:-1], EDGES[1:])]:
        if m.any():
            per = [np.nanpercentile(e[c, m], 90) for c in range(e.shape[0]) if np.isfinite(e[c, m]).any()]
            out[name] = {"max_p90": float(max(per)), "end_p90": float(np.nanpercentile(e[-1, m], 90)), "paths": int(m.sum())}
    return out


def main():  # sprint 2026-10-08 (plan item 2): import guard; the module body ran on import before
    out = {"tables": {}, "pairs": {}, "paths": {}}
    for test, (dname, fname) in TESTS.items():
        for tag in [None]:
            lab = test
            L = {}
            for arm in ARMS:
                for m in runs(arm):
                    f = E / dname.format(m) / fname
                    if f.exists():
                        L.setdefault(arm, []).append(load(f, tag))
                        t = out["tables"][f"{lab} {m}"] = table(L[arm][-1])
                        ok = test in FULL and L[arm][-1]["nint"] == 0 and len(t) == 5 and all(v["late_max_p90"] <= 0.1 for v in t.values())
                        print(f"{lab:14s} {m:18s} " + " | ".join(f"{k}: {v['late_max_p90']:.2f}/{v['end_p90']:.2f} "
                                                               f"t{v['late_max_p90_truthmask']:.2f} [{v['late_max_p90_y1e-8']:.2f}] ({v['rows_gt3dex_end']}, c{v['collapse_rows']})"
                                                               for k, v in t.items()) + ("  TARGET MET" if ok else ""))
            for a, b in PAIRS:
                if a in L and b in L:
                    r = out["pairs"][f"{lab} {a} vs {b}"] = boot(L[a], L[b])
                    print(f"  {lab} {a} - {b} (seeds {len(L[a])}/{len(L[b])}), late: " + "; ".join(
                        f"{k.split()[0]} {v['diff']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]" for k, v in r.items()
                        if k.endswith("late")))
    for arm in ARMS:
        for m in runs(arm):
            f = E / f"P_{m}" / "P_paths.npz"
            if f.exists():
                r = out["paths"][m] = paths_score(f)
                print(f"paths {m:18s} " + " | ".join(f"{k}: {v['max_p90']:.2f}/{v['end_p90']:.2f} (n {v['paths']})" for k, v in r.items()))
    TAG = paths.SCRATCH  # 2026-10-07: seen/unseen pseudo-truth tags (ctruth.py), as in analysis/an_v11.py, an_v12.py
    for g in ("seen", "unseen"):
        for arm in ARMS:
            for m in runs(arm):
                f = E / f"H_c{g}500_{m}" / f"H_e13_ctruth_{g}_dt500.npz"
                if f.exists():
                    t = out["tables"][f"c{g} {m}"] = table(load(f, TAG / f"e13_ctruth_{g}"))
                    print(f"c{g:6s} {m:18s} " + " | ".join(f"{k}: {v['late_max_p90']:.3f}" for k, v in t.items()))
    (paths.OUT / "scores_e13.json").write_text(json.dumps(out, indent=1))   # was E (2026-10-09 move)


if __name__ == "__main__":
    main()
