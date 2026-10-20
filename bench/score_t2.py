"""T2 scorer (rules fixed in protocol.txt before any model result). Reads the model's arm-H bench on the T2 tag (lhbench
--arms H --h-tag <t2/tag> --h-starts 1e-3 --h-dt 500, e.g. E13_gcm/H_t2tag_<label>/H_tag_dt500.npz), the tag's boxes and
the source jobs' original truths. Per-gas rule (SCORE_RULE=phys, HARNESS_NORM unset), 12 reported species, NaN (no
scored point) counts as 0. Box clock = elapsed time since the restart (the model starts at 1e-3 s).
Per kicked row (k1p, k1m, k2p, k2m, k3), with its identity partner k0 (same source job and t_r):
  s0(tau) = e(V_k, V_0) separation (primary), sO = e(V_k, O(t_r + tau)) vs the original truth, c = e(V_0, O) calibration;
  E_k = e(M_k, V_k), E_0 = e(M_0, V_0) surrogate errors (max over species); X0 = max_{tau>=1e3} E_k - max_{tau>=1e3} E_0;
  d = e(M_k, M_0) model deviation; A = d - s0 on the half-time species q (kicked species; max over 12 for k3);
  D0 = s0_q(1e-3 s); half-times (first time after which the deviation stays < D0 / 2): h_V on the kicked box's own
  steps, h_Vg and h_M on the model checkpoints; growth = max_{tau>=1e3} E_k - E_k(500 s).
Rules (protocol.txt sec. 3), verdict inside env_any+margin (region "in"); readouts by band, region, class, tier, t_r.
Writes <out stem>.txt and <out stem>.json.
    CUDA_VISIBLE_DEVICES= SCORE_RULE=phys PYTHONDONTWRITEBYTECODE=1 taskset -c 9,11 $PY score_t2.py <bench npz> <out stem>"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
assert os.environ.get("SCORE_RULE") == "phys" and not os.environ.get("HARNESS_NORM"), "SCORE_RULE=phys, HARNESS_NORM unset"
sys.path.insert(0, str(Path(__file__).resolve().parent))   # bench/
import scoreboard as SB  # noqa: E402
SR = SB.SR
SP = SB.SP
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datagen"))
import build_kicks as K  # noqa: E402  (truth loader, SETS)

KB = 1.380649e-16
NONFINITE = [0]   # bench rows with any non-finite model state (amendment 2)
T0 = 0.0  # 2026-10-14 19:4x (verifier, before any model result): the model starts at the kick itself (was 1e-3 s)
CLASS = {"k1p": "k1", "k1m": "k1", "k2p": "k2", "k2m": "k2", "k3": "k3"}
BANDS = ("300-650", "650-1300", "1300-2000", "2000-3000")


def err(a, b, lnd):
    """per-gas rule |.| between log10 arrays a, b [..., 12]; NaN (not scored) -> 0"""
    e = SR.err(a, b, ((a + lnd) >= -2.0) | ((b + lnd) >= -2.0), SP)
    return np.nan_to_num(e, nan=0.0)


def first_below(times, dev, d0):
    bad = np.flatnonzero(dev >= d0 / 2.0)
    if bad.size == 0:
        return float(times[0])
    if bad[-1] == dev.size - 1:
        return float("inf")
    return float(times[bad[-1] + 1])


def qsel(e, q):
    return e.max(-1) if q is None else e[..., q]


def main():
    f = Path(sys.argv[1])
    d = np.load(f)
    tag = str(d["tag"])
    jobs = json.loads((Path(tag) / "jobs.json").read_text())
    B = SB.bench(f)
    st, vt = B["st"], B["vt"]                                     # [R, C, 12] model, VULCAN-kicked at the checkpoints
    NONFINITE[0] = int(np.logical_not(np.isfinite(st)).any(axis=(1, 2)).sum())   # amendment 2: fail closed
    st = np.where(np.isfinite(st), st, 10.0)                     # a non-finite prediction scores as a huge error, never as 0
    el = np.asarray(d["calls"], np.float64) * float(d["dt"])
    t0 = np.asarray(d["t0"], np.float64)
    assert np.allclose(t0, T0), "T2 bench must start at T0"
    tc = T0 + el
    late = el >= 1e3
    i500 = int(np.argmin(np.abs(el - 500.0)))
    assert abs(el[i500] - 500.0) < 1e-6 and el[-1] == 8.64e7, (el[:3], el[-1])
    rowof = {int(j): r for r, j in enumerate(d["job"])}
    part = {}
    for ti, b in enumerate(jobs):
        if b["kick"] == "k0":
            part[(b["src_set"], b["src_job"], b["t_target"])] = ti
    rows, excl = [], {"no bench row (box missing)": 0, "identity partner missing": 0}
    cal = {}
    orig = {}
    for ti, b in enumerate(jobs):
        if b["kick"] == "k0":
            continue
        if ti not in rowof:
            excl["no bench row (box missing)"] += 1
            continue
        p0 = part.get((b["src_set"], b["src_job"], b["t_target"]))
        if p0 is None or p0 not in rowof:
            excl["identity partner missing"] += 1
            continue
        r, r0 = rowof[ti], rowof[p0]
        lnd = np.log10(b["P"] / (KB * b["T"]))
        q = None if b["kick"] == "k3" else SP.index(b["ksp"])
        key = (b["src_set"], b["src_job"])
        if key not in orig:
            t, x, sp, _ = K.truth(*key)
            orig[key] = (t, x[:, [sp.index(s) for s in SP]])
        to, xo = orig[key]
        lo = np.log10(np.maximum(np.stack([np.interp(b["t_r"] + tc, to, xo[:, k]) for k in range(12)], -1), 1e-30))
        # VULCAN fine grid: the kicked box's own steps in [1e-3, 1e-3 + 8.64e7]
        tk = np.load(Path(tag) / f"{ti}.npz")["t"]
        tf = np.concatenate([[T0], tk[(tk > T0) & (tk <= T0 + 8.64e7)]])
        vk_f = SB.vtruth(tag, ti, tf)
        v0_f = SB.vtruth(tag, p0, tf)
        s0_f = err(vk_f, v0_f, lnd)
        D0 = float(qsel(s0_f[0], q))
        s0 = err(vt[r], vt[r0], lnd)                              # [C, 12] at the checkpoints
        sO = err(vt[r], lo, lnd)
        if (key, b["t_target"]) not in cal:
            cal[(key, b["t_target"])] = float(err(vt[r0], lo, lnd)[late].max())
        Eks, E0s = err(st[r], vt[r], lnd), err(st[r0], vt[r0], lnd)   # [C, 12]
        Ek, E0 = Eks.max(-1), E0s.max(-1)
        xs = Eks - E0s                                            # per-species, per-time excess over the identity row
        dm = qsel(err(st[r], st[r0], lnd), q)
        s0q = qsel(s0, q)
        hV = first_below(tf, qsel(s0_f, q), D0)
        hVg = first_below(tc, s0q, D0)
        hM = first_below(tc, dm, D0)
        Amp = dm - s0q
        rows.append({"i": ti, "set": b["src_set"], "job": b["src_job"], "t_target": b["t_target"], "t_r": b["t_r"],
                     "kick": b["kick"], "class": CLASS[b["kick"]], "ksp": b["ksp"], "tier": b["tier"], "band": b["band"],
                     "region": b["region"], "stratum": b["stratum"], "T": b["T"], "P_bar": b["P"] / 1e6, "D0": D0,
                     "abs_factor": b["abs_factor"], "cal": cal[(key, b["t_target"])],
                     "s0_end": float(s0q[-1]), "s0_end12": float(s0[-1].max()), "sO_end12": float(sO[-1].max()),
                     "hV": hV, "hVg": hVg, "hM": hM,
                     "Ek_first": float(Ek[i500]), "Ek_late": float(Ek[late].max()), "Ek_end": float(Ek[-1]),
                     "Ek_late_sp": SP[int(np.unravel_index(np.argmax(err(st[r], vt[r], lnd)[late]), (late.sum(), 12))[1])],
                     "E0_late": float(E0[late].max()), "X0_rowmax": float(Ek[late].max() - E0[late].max()),
                     "X0": float(xs[late].max()), "Xc": float(xs[i500:].max()),
                     "growth": float(Ek[late].max() - Ek[i500]), "A_max": float(Amp[late].max()), "A_end": float(Amp[-1])})
    return rows, excl, cal, jobs, f


def share(m):
    return float(np.mean(m)) if len(m) else float("nan")


def rules(R):
    """protocol.txt sec. 3 on a list of rows; returns dict of rule results"""
    out = {}
    for c in ("k1", "k2", "k3"):
        rr = [r for r in R if r["class"] == c]
        ok = [r["X0"] <= 0.1 for r in rr]
        out[f"a_{c}"] = {"n": len(rr), "share": share(ok), "need": 0.95,
                         "verdict": "too few rows" if len(rr) < 20 else ("PASS" if share(ok) >= 0.95 else "FAIL"),
                         "fail_rows": [r for r in rr if r["X0"] > 0.1]}
    rb = [r for r in R if r["hV"] < 1e4 and r["D0"] >= 0.1]
    okb = [np.isfinite(r["hM"]) and 0.5 * r["hVg"] * (1 - 1e-9) <= r["hM"] <= 2.0 * r["hVg"] * (1 + 1e-9) for r in rb]
    out["b"] = {"n": len(rb), "share": share(okb), "need": 0.90,
                "verdict": "too few rows" if len(rb) < 20 else ("PASS" if share(okb) >= 0.90 else "FAIL"),
                "fail_rows": [r for r, o in zip(rb, okb) if not o]}
    rc = [r for r in R if (r["hV"] > 1e7) and r["D0"] >= 0.1]
    okc = [r["Xc"] <= 0.3 for r in rc]
    out["c"] = {"n": len(rc), "share": share(okc), "need": 0.95,
                "verdict": "too few rows" if len(rc) < 20 else ("PASS" if share(okc) >= 0.95 else "FAIL"),
                "fail_rows": [r for r, o in zip(rc, okc) if not o]}
    v = [out[k]["verdict"] for k in ("a_k1", "a_k2", "a_k3", "b", "c")]
    out["T2"] = "PASS" if all(x == "PASS" for x in v) else ("FAIL" if "FAIL" in v else "UNDECIDED (too few rows)")
    return out


def fmt_rules(name, res):
    L = [f"  {name}: T2 {res['T2']}"]
    for k in ("a_k1", "a_k2", "a_k3", "b", "c"):
        x = res[k]
        L.append(f"    rule {k:5s}: {x['verdict']:12s} share {x['share']:.3f} (need >= {x['need']:.2f}) of {x['n']} rows; "
                 f"{len(x['fail_rows'])} failing")
    return L


def grp(R, name):
    if not R:
        return f"    {name:28s} n    0"
    X0 = np.array([r["X0"] for r in R])
    hV = np.array([r["hV"] for r in R])
    hM = np.array([r["hM"] for r in R])
    nk = np.array([r["D0"] >= 0.1 for r in R])
    Ek = np.array([r["Ek_late"] for r in R])
    Ee = np.array([r["Ek_end"] for r in R])
    g = np.array([r["growth"] for r in R])
    med = lambda v: float(np.median(v)) if v.size else float("nan")
    return (f"    {name:28s} n {len(R):4d} | X0 <= 0.1 {np.mean(X0 <= 0.1):.3f}, p50 {np.median(X0):+.3f} p95 {np.percentile(X0, 95):+.3f} "
            f"max {X0.max():+.2f} | D0 >= 0.1: {int(nk.sum())}; h_V med {med(hV[nk]):.3g} (< 500 s {np.mean(hV[nk] < 500) if nk.any() else float('nan'):.2f}, "
            f"> 1e7 / never {np.mean(hV[nk] > 1e7) if nk.any() else float('nan'):.2f}); h_M med {med(hM[nk]):.3g} (never "
            f"{np.mean(np.isinf(hM[nk])) if nk.any() else float('nan'):.2f}) | E_k late p50 {np.median(Ek):.3f} max {Ek.max():.2f}, "
            f"end p50 {np.median(Ee):.3f} max {Ee.max():.2f} | growth > 0.3: {int((g > 0.3).sum())}")


def report(rows, excl, cal, jobs, f):
    L = [f"T2 SCORE (score_t2.py; rules: protocol.txt). Bench {f}; tag {len(jobs)} boxes; kicked rows scored {len(rows)}; "
         f"excluded: " + ", ".join(f"{k} {v}" for k, v in excl.items()),
         f"Calibration (identity box vs original truth, late max, per-gas rule): (job, t_r) pairs {len(cal)}, flagged > 0.1 dex "
         f"(CAL): {sum(v > 0.1 for v in cal.values())}; max {max(cal.values()) if cal else float('nan'):.3f}", "",
         "VERDICT (inside env_any+margin):"]
    res_in = rules([r for r in rows if r["region"] == "in"])
    if excl["no bench row (box missing)"] > 0 or excl["identity partner missing"] > 0:   # amendment 2: fail closed
        res_in["T2"] = "INVALID (bench rows missing)"
    L.append(f"bench rows with a non-finite model state (scored as 10 in log10, i.e. a huge error): {NONFINITE[0]}")
    L += fmt_rules("inside env_any+margin", res_in)
    R_in = [r for r in rows if r["region"] == "in"]
    ek = np.array([r["Ek_late"] for r in R_in]); e0 = np.array([r["E0_late"] for r in R_in])
    L.append("  ABSOLUTE ACCURACY (readout; the rules are relative to the identity row, so a shared error can cancel): "
             f"kicked rows E_k late max <= 0.1 dex {np.mean(ek <= 0.1):.3f}, <= 0.5 {np.mean(ek <= 0.5):.3f}; "
             f"identity partners E_0 late max <= 0.1 {np.mean(e0 <= 0.1):.3f}, <= 0.5 {np.mean(e0 <= 0.5):.3f}")
    L.append("  Wording: PASS means FINAL responds to element-preserving kicks as VULCAN does (no excess error over its own "
             "unkicked error, matched fast recovery, no growth in slow cells), at 500 s checkpoint resolution; it does not by "
             "itself mean near-perfect trajectories (see the absolute readout).")
    L.append("READOUTS with the same rules (no verdict):")
    for nm, sub in (("cooler stratum", [r for r in rows if r["region"] == "cool"]),
                    ("outside (other)", [r for r in rows if r["region"] == "out"]),
                    ("miss jobs (any region)", [r for r in rows if r["stratum"] == "miss"]),
                    ("all rows", rows)):
        L += fmt_rules(nm, rules(sub))
    L.append("")
    L.append("GROUPS (X0 = excess late error over the identity row; h = recovery half-times, s; E_k = model vs VULCAN-kicked)")
    for reg in ("in", "cool", "out"):
        for bn in BANDS:
            for c in ("k1", "k2", "k3"):
                sub = [r for r in rows if r["region"] == reg and r["band"] == bn and r["class"] == c]
                if sub:
                    L.append(grp(sub, f"{reg} {bn} {c}"))
    for tt in K.T_TARGETS:
        L.append(grp([r for r in rows if r["t_target"] == tt], f"t_target {tt:.0e}"))
    for tr in (1, 2, 3):
        L.append(grp([r for r in rows if r["tier"] == tr and r["class"] != "k3"], f"tier {tr} (k1/k2)"))
    for lo, hi in ((-6, -4), (-4, -2), (-2, 0), (0, 3)):
        L.append(grp([r for r in rows if r["region"] == "in" and lo <= np.log10(r["P_bar"]) < hi], f"in, log P [{lo}, {hi})"))
    L.append("")
    af = np.array([abs(r["abs_factor"]) for r in rows])
    se = np.array([r["s0_end12"] for r in rows])
    L.append("sanity: separation s0 at 1000 days (max over 12) by |abs_factor| decade: " + ", ".join(
        f"[{lo:.0e}, {hi:.0e}) n {int(((af >= lo) & (af < hi)).sum())} med {np.median(se[(af >= lo) & (af < hi)]) if ((af >= lo) & (af < hi)).any() else float('nan'):.3f}"
        for lo, hi in ((0, 1e-6), (1e-6, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2))))
    L.append(f"max amplification A = d - s0 (late, half-time species): p50 {np.median([r['A_max'] for r in rows]):+.3f}, "
             f"p95 {np.percentile([r['A_max'] for r in rows], 95):+.3f}, max {max(r['A_max'] for r in rows):+.2f}")
    L.append("")
    for k in ("a_k1", "a_k2", "a_k3", "b", "c"):
        fr = res_in[k]["fail_rows"]
        L.append(f"FAILING ROWS inside, rule {k} ({len(fr)}): set job t_r kick ksp tier band P_bar D0 hV hVg hM E0_late Ek_first "
                 f"Ek_late(sp) Ek_end X0 growth")
        for r in sorted(fr, key=lambda r: -r["X0"])[:60]:
            L.append(f"    {r['set']:4s} {r['job']:5d} {r['t_r']:.3e} {r['kick']:4s} {r['ksp'] or '-':5s} {r['tier']} {r['band']:9s} "
                     f"{r['P_bar']:.2e} {r['D0']:.2f} {r['hV']:.3g} {r['hVg']:.3g} {r['hM']:.3g} {r['E0_late']:.2f} {r['Ek_first']:.2f} "
                     f"{r['Ek_late']:.2f}({r['Ek_late_sp']}) {r['Ek_end']:.2f} {r['X0']:+.2f} {r['growth']:+.2f}")
    L.append("")
    L.append("MISS JOBS one by one: set job band region | kicked rows | X0 max | E_k late max (sp) | E_0 late max | h_V / h_M of the k2 rows")
    keys = sorted({(r["set"], r["job"]) for r in rows if r["stratum"] == "miss"})
    for key in keys:
        rr = [r for r in rows if (r["set"], r["job"]) == key]
        k2 = [r for r in rr if r["class"] == "k2"]
        w = max(rr, key=lambda r: r["Ek_late"])
        L.append(f"    {key[0]:4s} {key[1]:5d} {rr[0]['band']:9s} {rr[0]['region']:4s} | {len(rr):2d} | {max(r['X0'] for r in rr):+.2f} | "
                 f"{w['Ek_late']:.2f} ({w['Ek_late_sp']}, {w['kick']}, t_r {w['t_r']:.2g}) | {max(r['E0_late'] for r in rr):.2f} | "
                 + " ".join(f"{r['hV']:.2g}/{r['hM']:.2g}" for r in k2))
    return "\n".join(L) + "\n", res_in


if __name__ == "__main__":
    assert len(sys.argv) == 3, "usage: score_t2.py <bench npz> <out stem>"
    rows, excl, cal, jobs, f = main()
    txt, res_in = report(rows, excl, cal, jobs, f)
    stem = Path(sys.argv[2])
    Path(str(stem) + ".txt").write_text(txt)
    clean = {k: {a: v for a, v in x.items() if a != "fail_rows"} | {"n_fail": len(x["fail_rows"])} if isinstance(x, dict) else x
             for k, x in res_in.items()}
    fin = lambda v: v if not isinstance(v, float) or np.isfinite(v) else ("inf" if v > 0 else "nan")   # strict JSON
    rows = [{k: fin(v) for k, v in r.items()} for r in rows]
    clean = {k: ({a: fin(v) for a, v in x.items()} if isinstance(x, dict) else x) for k, x in clean.items()}
    Path(str(stem) + ".json").write_text(json.dumps({"rules_inside": clean, "excluded": excl, "rows": rows}, allow_nan=False))
    print(txt[:6000])
