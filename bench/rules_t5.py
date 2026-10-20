"""T5 decision rules, executable and fail-closed.
Arm X vs reference R, FINAL as the G6 anchor, all benched with run/bench_dev.sh (six
fixed sets through DEV6, five path sets, T2). No model call, no GPU.
Rows: job x start age (1e-3, 1e3, 1e6 s) of G, Gc, H2, H2c, H3, H3c (score_mask.rows, per-gas rule, SCORE_RULE=phys);
late = max over checkpoints >= 1e3 s elapsed; lasting = the last checkpoint (1000 days); paired by (set, job, start age).
inside = env_any+margin (select_jobs.env_margin); outside = not inside; cooler = 300 <= T < 1300 K and 1e-5 <= P <= 100
bar (overlaps inside). "fixed" = R > 0.5 and X <= 0.5; "broken" = R <= 0.5 and X > 0.5 (same row).
Order: INVALID (a missing / failed output, row order mismatch, T2 json INVALID or not reproducible) -> INCONCLUSIVE (C0
moves lasting > 0.5 inside by >= 20 % vs FINAL: fine-tune churn) -> ADOPT (P1-P3 and G1-G7 all pass) ->
NO GAIN (lasting inside X >= 0.90 x R and T2 mean change < +0.02: the lever is closed) -> NOT ADOPTED (everything
else: partial gain 0.80-0.90 x R, snap-back only, any guard failure; failing rows read and reported).
    SCORE_RULE=phys HARNESS_NORM= python rules_t5.py <X label> <R label> [--final L] [--t2x json] [--t2r json]
        [--t2npz-x npz] [--t2npz-r npz] [--out stem]
Writes <out>.txt / .json (default paths.OUT/verdict_<X>_vs_<R>); last line T5-VERDICT: <verdict>."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_mask as M  # noqa: E402  (rows(), path_scores(), path_cells(); per-gas rule; row caches: SCORE_MASK_CACHE)
import paths  # noqa: E402
from select_jobs import env_margin  # noqa: E402
import score_t2 as ST  # noqa: E402  (rules(): the T2 protocol's five rules)

S, E = M.S, M.E
T2DIR = paths.T2
RULES5 = ("a_k1", "a_k2", "a_k3", "b", "c")
FIN_LATE2_IN, FIN_LAST05_IN = 5, 61          # G6 thresholds (FINAL inside: late > 2 = 4, lasting > 0.5 = 61)

ap = argparse.ArgumentParser()
ap.add_argument("x")
ap.add_argument("r")
ap.add_argument("--final", default="e13s_FINw1536_s2")
ap.add_argument("--t2x")
ap.add_argument("--t2r")
ap.add_argument("--t2npz-x")
ap.add_argument("--t2npz-r")
ap.add_argument("--out")
a = ap.parse_args()
X, R, F = a.x, a.r, a.final
t2j = {X: Path(a.t2x) if a.t2x else T2DIR / f"score_t2_{X}.json", R: Path(a.t2r) if a.t2r else T2DIR / f"score_t2_{R}.json"}
t2n = {X: Path(a.t2npz_x) if a.t2npz_x else E / f"H_t2tag_{X}" / "H_tag_dt500.npz",
       R: Path(a.t2npz_r) if a.t2npz_r else E / f"H_t2tag_{R}" / "H_tag_dt500.npz"}
stem = Path(a.out) if a.out else paths.OUT / f"verdict_{X}_vs_{R}"
invalid, L, out = [], [], {"X": X, "R": R, "FINAL": F}


def hfile(k, lab):
    pre, fn = M.HSETS[k]
    return E / f"H_{pre}_{lab}" / fn


def files_of(lab):
    return [hfile(k, lab) for k in M.HSETS] + [E / f"{pre}_{lab}" / "P_paths.npz" for pre in M.PSETS]


miss = [f"missing {f}" for lab in (X, R, F) for f in files_of(lab) if not f.exists()]
invalid += miss
for lab in (X, R):
    for f in (t2j[lab], t2n[lab]):
        if not f.exists():
            invalid.append(f"missing {f}")
if miss:
    L.append("INVALID before scoring: " + "; ".join(invalid))
    (Path(str(stem) + ".txt")).write_text("\n".join(L) + "\nT5-VERDICT: INVALID\n")
    print("\n".join(L) + "\nT5-VERDICT: INVALID")
    sys.exit(0)

rows = {lab: M.rows(lab) for lab in (X, R, F)}
for lab in (R, F):
    for k in ("set", "job", "t0"):
        if not np.array_equal(rows[X][k], rows[lab][k]):
            invalid.append(f"row order differs in {k}: {X} vs {lab}")
T, P = rows[X]["T"], rows[X]["P"]
inside, _, _ = env_margin()
ins = inside(T, P)
outs = np.logical_not(ins)
cool = (T >= 300) & (T < 1300) & (P >= 1e-5) & (P <= 100)
allr = np.ones_like(ins)


def cnt(lab, m, key, thr):
    return int((m & (rows[lab][key] > thr)).sum())


def paired(m, key):
    fx = m & (rows[R][key] > 0.5) & (rows[X][key] <= 0.5)
    br = m & (rows[R][key] <= 0.5) & (rows[X][key] > 0.5)
    return int(fx.sum()), int(br.sum())


q = {}
for nm, m in (("inside", ins), ("outside", outs), ("cooler", cool), ("all", allr)):
    q[nm] = {lab: {"last05": cnt(lab, m, "last", 0.5), "last2": cnt(lab, m, "last", 2.0), "late05": cnt(lab, m, "late", 0.5),
                   "late2": cnt(lab, m, "late", 2.0), "first2": cnt(lab, m, "first", 2.0)} for lab in (X, R, F)}
    q[nm]["last_fixed"], q[nm]["last_broken"] = paired(m, "last")
    q[nm]["late_fixed"], q[nm]["late_broken"] = paired(m, "late")
    L.append(f"{nm:8s} rows {int(m.sum()):5d} | lasting > 0.5 X {q[nm][X]['last05']} R {q[nm][R]['last05']} F {q[nm][F]['last05']} "
             f"(fixed {q[nm]['last_fixed']}, broken {q[nm]['last_broken']}) | late > 0.5 X {q[nm][X]['late05']} R "
             f"{q[nm][R]['late05']} F {q[nm][F]['late05']} (fixed {q[nm]['late_fixed']}, broken {q[nm]['late_broken']}) | late > 2 "
             f"X {q[nm][X]['late2']} R {q[nm][R]['late2']} F {q[nm][F]['late2']} | lasting > 2 X {q[nm][X]['last2']} R "
             f"{q[nm][R]['last2']} | first > 2 X {q[nm][X]['first2']} R {q[nm][R]['first2']}")
out["counts"] = q

pth = {}
for pre in M.PSETS:
    pth[pre] = {}
    for lab in (X, R):
        e, calls, _, _ = M.path_scores(lab, pre)
        c = M.path_cells(e, calls, np.ones(e.shape[:2], bool), M.PSETS[pre])
        pth[pre][lab] = {"late_p99": c["late_p99"], "max_all": c["max_all"]}
    L.append(f"path {pre:3s}: late p99 X {pth[pre][X]['late_p99']:.3f} R {pth[pre][R]['late_p99']:.3f} | max X "
             f"{pth[pre][X]['max_all']:.2f} R {pth[pre][R]['max_all']:.2f}")
out["paths"] = pth

# Non-finite model states (P3), kept input, failed Newton solves (rows with >= 1).
num = {}
for lab in (X, R):
    nf = kept = newton = 0
    for f in files_of(lab) + ([t2n[lab]] if t2n[lab].exists() else []):
        d = np.load(f)
        st = d["states"][:, 2]
        nf += int(np.logical_not(np.isfinite(st)).any(axis=(0, 2)).sum())
        pf = d["projfail"] if "projfail" in d.files else None
        if pf is None:
            invalid.append(f"no projfail counters in {f}")
            continue
        kept += int((pf[..., 1].sum(0) > 0).sum())
        if "H_t2tag" not in str(f):
            newton += int((pf[..., 0].sum(0) > 0).sum())
    num[lab] = {"nonfinite_rows": nf, "kept_rows": kept, "newton_fail_rows": newton}
    L.append(f"numerics {lab}: non-finite rows {nf}, kept-input rows {kept}, failed-Newton rows (fixed + path sets) {newton}")
out["numerics"] = num

# Invented switches among X's new late > 2 rows (board6.invented's rule, per row).
Lth = np.array([M.SR.PHYS[s] for s in M.SP])


def invented_rows(lab):
    flags = []
    for k in M.HSETS:
        B = S.bench(hfile(k, lab))
        late = slice(B["st"].shape[1] - B["e"].shape[1], None)
        st, vt = np.maximum(B["st"][:, late], Lth), np.maximum(B["vt"][:, late], Lth)
        mv_m, mv_t = st.max(1) - st.min(1), vt.max(1) - vt.min(1)
        big = np.nan_to_num(B["e"], nan=0.0).max(1) > 2.0
        flags.append((big & (mv_m > 2.0) & (mv_t < 0.5)).any(1))
    return np.concatenate(flags)


newx = (rows[X]["late"] > 2.0) & (rows[R]["late"] <= 2.0)
inv = invented_rows(X)
assert inv.size == newx.size
n_inv_new = int((inv & newx).sum())
L.append(f"X's new late > 2 rows (R <= 2): {int(newx.sum())}, invented switches among them {n_inv_new}:")
for i in np.flatnonzero(newx):
    r = rows[X]
    L.append(f"    {r['set'][i]:4s} job {int(r['job'][i]):5d} t0 {r['t0'][i]:.0e} T {T[i]:7.1f} K P {P[i]:.2e} bar "
             f"{'inside' if ins[i] else 'outside'} | late {r['late'][i]:.2f} ({r['late_sp'][i]}) vs R {rows[R]['late'][i]:.2f} | "
             f"lasting {r['last'][i]:.2f} | invented {bool(inv[i])}")
out["new_gt2"] = {"n": int(newx.sum()), "invented": n_inv_new}

t2 = {}


def fl(v):
    return float(v) if isinstance(v, str) and v in ("inf", "-inf", "nan") else v


for lab in (X, R):
    if not t2j[lab].exists():
        continue
    j = json.loads(t2j[lab].read_text())
    if str(j["rules_inside"]["T2"]).startswith("INVALID") or any(v > 0 for v in j["excluded"].values()):
        invalid.append(f"T2 {lab}: {j['rules_inside']['T2']}, excluded {j['excluded']}")
    rr = [{k: fl(v) for k, v in r.items()} for r in j["rows"]]
    ri = ST.rules([r for r in rr if r["region"] == "in"])
    for k in RULES5:
        if not math.isclose(ri[k]["share"], j["rules_inside"][k]["share"], abs_tol=1e-12):
            invalid.append(f"T2 {lab}: recomputed inside share {k} {ri[k]['share']} != json {j['rules_inside'][k]['share']}")
    rc = ST.rules([r for r in rr if r["region"] == "cool"])
    t2[lab] = {"inside": {k: j["rules_inside"][k]["share"] for k in RULES5},
               "cool": {k: {"share": rc[k]["share"], "n": rc[k]["n"]} for k in RULES5}}
if X in t2 and R in t2:
    dl = {k: t2[X]["inside"][k] - t2[R]["inside"][k] for k in RULES5}
    t2mean = float(np.mean(list(dl.values())))
    L.append("T2 inside shares: " + ", ".join(f"{k} X {t2[X]['inside'][k]:.3f} R {t2[R]['inside'][k]:.3f} ({dl[k]:+.3f})" for k in RULES5)
             + f"; mean change {t2mean:+.4f}")
    L.append("T2 cooler-stratum shares: " + ", ".join(f"{k} X {t2[X]['cool'][k]['share']:.3f} R {t2[R]['cool'][k]['share']:.3f} "
                                                      f"(n {t2[X]['cool'][k]['n']}/{t2[R]['cool'][k]['n']})" for k in RULES5))
else:
    dl, t2mean = None, float("nan")
out["t2"] = t2
out["t2_mean_change"] = t2mean

I = q["inside"]
res = {}
lx, lr, lf = I[X]["last05"], I[R]["last05"], I[F]["last05"]
res["P1"] = (lx <= 0.80 * lr and I["last_fixed"] >= 2 * I["last_broken"],
             f"lasting > 0.5 inside X {lx} <= 0.80 x R {lr} = {0.8 * lr:.1f}; fixed {I['last_fixed']} >= 2 x broken {I['last_broken']}")
res["P2"] = (I[X]["late05"] <= I[R]["late05"] and I["late_fixed"] >= I["late_broken"],
             f"late > 0.5 inside X {I[X]['late05']} <= R {I[R]['late05']}; fixed {I['late_fixed']} >= broken {I['late_broken']}")
if dl is not None:
    res["P3"] = (all(v >= -0.02 for v in dl.values()) and t2mean >= 0.0,
                 f"T2 inside: every share change >= -0.02 (min {min(dl.values()):+.3f}) and mean change {t2mean:+.4f} >= 0")
else:
    res["P3"] = (False, "T2 shares missing")
res["G1"] = (q["inside"][X]["late2"] <= q["inside"][R]["late2"] + 1 and q["outside"][X]["late2"] <= q["outside"][R]["late2"] + 2
             and q["inside"][X]["last2"] <= q["inside"][R]["last2"],
             f"late > 2 inside X {q['inside'][X]['late2']} <= R + 1 = {q['inside'][R]['late2'] + 1}; outside X "
             f"{q['outside'][X]['late2']} <= R + 2 = {q['outside'][R]['late2'] + 2}; lasting > 2 inside X {q['inside'][X]['last2']} "
             f"<= R {q['inside'][R]['last2']}")
cq = q["cooler"]
g2 = [cq[X]["late05"] <= 1.10 * cq[R]["late05"], cq[X]["last05"] <= cq[R]["last05"] + 3, cq[X]["late2"] <= cq[R]["late2"] + 1]
g2t = []
if X in t2 and R in t2:
    for k in RULES5:
        if t2[X]["cool"][k]["n"] >= 20 and t2[R]["cool"][k]["n"] >= 20:
            g2t.append((k, t2[X]["cool"][k]["share"] >= t2[R]["cool"][k]["share"] - 0.05))
else:
    g2.append(False)
res["G2"] = (all(g2) and all(v for _, v in g2t),
             f"cooler late > 0.5 X {cq[X]['late05']} <= 1.10 x R = {1.1 * cq[R]['late05']:.1f}; lasting > 0.5 X {cq[X]['last05']} <= R + 3 = "
             f"{cq[R]['last05'] + 3}; late > 2 X {cq[X]['late2']} <= R + 1 = {cq[R]['late2'] + 1}; T2 cooler shares not below R - 0.05 "
             f"(rules with >= 20 rows): " + (", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in g2t) or "none"))
g3 = [(pre, pth[pre][X]["late_p99"] <= pth[pre][R]["late_p99"] + 0.03 and pth[pre][X]["max_all"] <= pth[pre][R]["max_all"] + 1.0)
      for pre in M.PSETS]
res["G3"] = (all(v for _, v in g3), "paths late p99 <= R + 0.03 and max <= R + 1.0: " + ", ".join(f"{p} {'ok' if v else 'FAIL'}" for p, v in g3))
res["G4"] = (q["all"][X]["first2"] <= q["all"][R]["first2"] + 2, f"first-call rows > 2 X {q['all'][X]['first2']} <= R + 2 = {q['all'][R]['first2'] + 2}")
res["G5"] = (num[X]["kept_rows"] == 0 and num[X]["nonfinite_rows"] == 0 and num[X]["newton_fail_rows"] <= max(10, 2 * num[R]["newton_fail_rows"]),
             f"X kept-input rows {num[X]['kept_rows']} = 0, non-finite rows {num[X]['nonfinite_rows']} = 0, failed-Newton rows "
             f"{num[X]['newton_fail_rows']} <= max(10, 2 x R {num[R]['newton_fail_rows']})")
res["G6"] = (q["inside"][X]["late2"] <= FIN_LATE2_IN and lx <= FIN_LAST05_IN,
             f"vs FINAL: late > 2 inside X {q['inside'][X]['late2']} <= {FIN_LATE2_IN}; lasting > 0.5 inside X {lx} <= {FIN_LAST05_IN} "
             f"(FINAL live: {q['inside'][F]['late2']}, {lf})")
res["G7"] = (n_inv_new < 3, f"invented switches among X's new late > 2 rows {n_inv_new} < 3 (new rows {int(newx.sum())}, listed above)")
inconcl = lf > 0 and abs(lr - lf) >= 0.20 * lf
out["rules"] = {k: {"pass": bool(v[0]), "detail": v[1]} for k, v in res.items()}
adopt = all(v[0] for v in res.values())
no_gain = lx >= 0.90 * lr and not (t2mean >= 0.02)
if invalid:
    verdict = "INVALID"
elif inconcl:
    verdict = "INCONCLUSIVE"
elif adopt:
    verdict = "ADOPT"
elif no_gain:
    verdict = "NO GAIN"
else:
    verdict = "NOT ADOPTED"
why = []
if verdict == "NOT ADOPTED":
    if 0.80 * lr < lx < 0.90 * lr:
        why.append("partial gain (lasting inside between 0.80 and 0.90 x R)")
    if not res["P1"][0] and t2mean >= 0.02:
        why.append("snap-back only (T2 mean change >= +0.02, P1 fails)")
    why += [f"{k} fails" for k, v in res.items() if not v[0]]
L.append("")
L.append(f"inconclusive check (read first): C0 lasting > 0.5 inside {lr} vs FINAL {lf}: change "
         f"{(lr - lf) / lf * 100 if lf else float('nan'):+.1f} % ({'>= 20 %: INCONCLUSIVE' if inconcl else '< 20 %'})")
for k, v in res.items():
    L.append(f"  {k}: {'PASS' if v[0] else 'FAIL'}  {v[1]}")
L.append(f"  snap-back gain readout (T2 mean change >= +0.05): {'yes' if t2mean >= 0.05 else 'no'} ({t2mean:+.4f}); note: arm D's A kicks are "
         "T2's kick family by design, so a T2 gain is expected and is not by itself evidence of generalisation")
if invalid:
    L.append("INVALID: " + "; ".join(invalid))
L.append(f"NO GAIN test: lasting inside X {lx} >= 0.90 x R = {0.9 * lr:.1f} and T2 mean change {t2mean:+.4f} < +0.02: {no_gain}")
if why:
    L.append("not adopted because: " + "; ".join(why))
out.update(verdict=verdict, invalid=invalid, inconclusive=bool(inconcl), adopt_test=bool(adopt), no_gain_test=bool(no_gain), why=why)
L.append(f"T5-VERDICT: {verdict}")
txt = f"T5 RULES (rules_t5.py; text t5/rules.txt): X = {X}, R = {R}, FINAL = {F}\n" + "\n".join(L) + "\n"
Path(str(stem) + ".txt").write_text(txt)
Path(str(stem) + ".json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(txt)
