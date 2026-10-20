"""T2 snap-back test, step 2 (scale phase, 2026-10-14): element-preserving kicks of the full 53-species truth state, written
as an arm-H tag (jobs.json in vulcan_ref.box format + extra metadata per box; the boxes' <i>.npz are written on the
generator side, training-data-generator/e13/drive_t2.py).

Restart state: the source job's stored VULCAN step nearest (in log t) to t_target = 1.5e3, 1e5, 1e7 s (all 53 species,
floored at 1e-30); t_r = that step's time (as diag/build_w1.py). VULCAN settings: the truth job's cfg with rtol 0.005
(G and Gc truths ran at the default 0.05, H2 / H2c / H3 / H3c at 0.005), runtime = trun_min = 1e8 s (covers 1000 days =
8.64e7 s after the restart; lhbench.vref_index needs t[-1] >= runtime), plus dt_var_max = DT_VAR_MAX when set (VULCAN's
step-growth cap only, for >= 10 output points per decade; decided from the smoke test, recorded in protocol.txt).

Element conservation. The re-match is match2.py (x_i = h_i exp(W_i . lam), Newton on the dual) on the four ATOM
totals C, N, O, H (cols 0-3), He held fixed; the molecule count is left free, because chemistry does not conserve it
(CH4 + H2O <-> CO + 3 H2 changes it by 2): holding it too (a 5-total match) forced trace species to absorb the
molecule-count change and dropped 2,925 of 4,482 kicks in a dry run (10-14 18:5x). Checked on the state VULCAN starts
from (floored at 1e-30, divided by its sum, as vulcan_ref.box does), against the identity restart's started state:
  preserved to <= 1e-10 (max relative difference): the element FRACTIONS, atoms of C, N, O, H, He per atom (equivalently
    every element ratio, C/He, O/H, ...): what defines the gas at fixed T, P;
  not preserved: the absolute per-volume totals, which scale by S0/S' (S = the 53-species sum before renormalisation)
    because VULCAN starts every box at n = P/kT while the kick changes the molecule count. Recorded per box as
    abs_factor = S0/S' - 1; kicks with |abs_factor| > 1e-2 are DROPPED (a > 1 % molecule-count change is not a kick).
    Within the cap this is equivalent to a <= 1 % pressure shift of the attractor (I: negligible against 0.1-dex rules).

Kicks per (job, t_r):
  k0   identity (control).
  k1p / k1m  kicked species x 10^+0.3 / x 10^-0.3;  k2p / k2m  x 10^+1 / x 10^-1. Kicked species: the job's dominant miss
       species (bench/select_jobs.py), else its band default; down the list [dom_sp, band default, CH4, C2H2, CO2, H2O] to the
       first with truth x >= 10 S (per-gas floor) at t_r; when ALL four k1/k2 kicks of that species drop, the next species
       of the list that passes the 10 S test is tried (ksp_reason "fallback after X infeasible"; X's drops stay counted).
       The kicked species is held at f x; then
       tier 1: only the 33 hidden species float (other 19 tracked + He fixed); accepted if the re-match converges
               (residual <= 1e-10) and no hidden species present (>= 1e-12 before or after) moves > 1 dex;
       tier 2: else one carrier per element floats too, distinct, chosen in the order H, C, O, N as the largest
               atoms x mixing ratio among the 19 tracked other than He, the kicked species and carriers already chosen;
               accepted if residual <= 1e-10, hidden moves <= 1 dex and every carrier moves <= 0.1 dex (per-gas floors
               for reported carriers, 1e-12 for named ones);
       tier 3: as tier 2 but a carrier moves more than 0.1 and at most |log10 f| dex (an exchange kick); recorded with
               its collateral (dcar);
       else DROPPED and counted (infeasible / not converged / hidden > 1 dex / carriers > |log10 f| / |abs_factor| cap).
  k3   random +-0.3 dex (seeded signs) on every reported species except one carrier per element (distinct, order H, C, O,
       N, largest atoms x mixing ratio among the 12; H2 for H), which float with the 33 hidden and the 7 named tracked
       species; accepted if residual <= 1e-10, hidden / named move <= 1 dex and carriers move <= the amplitude; else the
       amplitude shrinks 0.3 -> 0.2 -> 0.1, else DROPPED and counted.
Known gap: in cold gas where CH4 holds most carbon, no single-species CH4 kick exists (CH4 / 2 needs another C carrier,
e.g. CO at ~1e-6, to rise ~2 dex), and k3 treats CH4 as the C carrier, so those cells get no CH4 kick: the CH4-collapse
direction. An exchange-kick arm (CH4 <-> CO, partner allowed to move) would be a separate test; not built here.
Recorded per box (jobs.json): src_set, src_job, t_target, t_r, step, kick, ksp, ksp_reason, factor (or amp and signs
for k3), tier, carriers, d0_ksp (per-gas floored displacement of the kicked species between started states, kick vs
identity), d0_12 / d0_12_sp (max over the 12 and its species), d25 (max |dlog10| over the 25 entries present >= 1e-12:
what the model sees), dhid, dcar, res (re-match residual), elem_rel (fraction check), abs_factor, region, band,
stratum, reason.

    python build_kicks.py smoke    3 jobs x 2 kicks at t_r 1.5e3 s -> paths.T2/smoke/tag/jobs.json
    python build_kicks.py full     every selected job -> paths.T2/tag/jobs.json and kicks.txt (refuses to overwrite
                                   tag/jobs.json)
    python build_kicks.py dry      as full, summary printed only (nothing written)
Run: CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 taskset -c 9,11 $PY build_kicks.py <mode>"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "lib")]
import paths  # noqa: E402
import kick_common as C  # noqa: E402  (truth sets, 25-entry definition, per-gas floors)
import match2 as M2  # noqa: E402

SETS = dict(C.SETS)
SETS["H3c"] = (paths.HOLDOUTS["h3c"], None)
SEED = 20261014
T_TARGETS = (1.5e3, 1e5, 1e7)
KICKS = (("k1p", 10 ** 0.3), ("k1m", 10 ** -0.3), ("k2p", 10.0), ("k2m", 0.1))
KIDS = ("k0", "k1p", "k1m", "k2p", "k2m", "k3")
FALLBACK = ("CH4", "C2H2", "CO2", "H2O")
RUNTIME = "1e8"
RTOL = "0.005"
DT_VAR_MAX = None       # set after the smoke test (protocol.txt)
H_END = 8.64e7          # 1000 days after the restart
RES_MAX = 1e-10         # re-match residual (relative, atom totals); the fraction check below decides
FRAC_MAX = 1e-10        # started-state element-fraction check
ABS_MAX = 1e-2          # |S0/S' - 1| cap
ORDER = (3, 0, 2, 1)    # carrier choice order: H, C, O, N (columns of common.wmat: C, N, O, H, 1)


def truth(s, j):
    d = SETS[s][0]
    r = np.load(d / f"{j}.npz")
    return r["t"], r["x"], [str(v) for v in r["species"]], json.loads((d / "jobs.json").read_text())[j]


def started(x):
    v = np.maximum(np.asarray(x, np.float64), 1e-30)
    return v / v.sum()


def fractions(v, W, he):
    q = np.concatenate([v @ W[:, :4], [v[he]]])        # atoms C, N, O, H, He
    return q / q.sum()


def lfl(v, floor):
    return np.maximum(np.log10(np.maximum(v, 1e-30)), floor)


class Ctx:
    def __init__(self, sp):
        self.sp = sp
        self.W, self.hid = C.wmat(sp)
        self.he = sp.index("He")
        self.rep = [sp.index(s) for s in C.SP12]
        self.named = [sp.index(s) for s in C.TRACKED if s not in C.SP12 and s != "He"]
        self.trk = [sp.index(s) for s in C.TRACKED if s != "He"]
        self.floor = np.full(len(sp), -12.0)          # displacement floor: S for reported species, 1e-12 otherwise
        for s in C.SP12:
            self.floor[sp.index(s)] = C.S[s]

    def disp(self, a, b, idx):
        """per-species floored |dlog10| between linear states a, b on indices idx"""
        idx = np.asarray(idx, int)
        return np.abs(lfl(a[idx], self.floor[idx]) - lfl(b[idx], self.floor[idx]))

    def carriers(self, xs, pool, exclude):
        """one distinct carrier per element, order H, C, O, N: largest atoms x mixing ratio in pool"""
        out = []
        for e in ORDER:
            c = [i for i in pool if i not in exclude and i not in out]
            w = self.W[c, e] * xs[c]
            if w.max() > 0:
                out.append(c[int(np.argmax(w))])
        return sorted(out)


def try_match(cx, h, b, free):
    out, res = M2.match(h, cx.W, b, fixed=np.logical_not(free), iters=1000, cols=[0, 1, 2, 3])
    if out is None:
        return None, np.inf, "infeasible (an atom total <= 0 after the fixed part)"
    if not np.isfinite(res) or res > RES_MAX:
        return None, res, f"not converged (residual {res:.1e})"
    return out, res, ""


def hid_move(cx, xs, out, mask):
    idx = np.flatnonzero(mask & ((xs >= 1e-12) | (out >= 1e-12)))
    return float(cx.disp(xs, out, idx).max()) if idx.size else 0.0


def single(cx, xs, k, f):
    b = xs @ cx.W
    h = xs.copy()
    h[k] = xs[k] * f
    out, res, why = try_match(cx, h, b, cx.hid)
    if out is not None:
        dh = hid_move(cx, xs, out, cx.hid)
        if dh <= 1.0:
            return out, {"tier": 1, "res": res, "carriers": [], "dcar": 0.0, "dhid": dh}, ""
        why = f"hidden species move {dh:.2f} dex"
    car = cx.carriers(xs, cx.trk, {k})
    free = cx.hid.copy()
    free[car] = True
    out, res, why2 = try_match(cx, h, b, free)
    if out is None:
        return None, None, f"tier 1 {why}; tier 2/3 {why2}"
    dh = hid_move(cx, xs, out, cx.hid)
    if dh > 1.0:
        return None, None, f"tier 1 {why}; tier 2/3 hidden species move {dh:.2f} dex"
    dc = float(cx.disp(xs, out, car).max())
    meta = {"res": res, "carriers": [cx.sp[i] for i in car], "dcar": dc, "dhid": dh}
    if dc <= 0.1:
        return out, {"tier": 2, **meta}, ""
    if dc <= abs(np.log10(f)) + 1e-12:
        return out, {"tier": 3, **meta}, ""
    return None, None, f"tier 1 {why}; carriers {meta['carriers']} would move {dc:.2f} dex > |kick| {abs(np.log10(f)):.1f}"


def random_kick(cx, xs, rng):
    b = xs @ cx.W
    car = cx.carriers(xs, cx.rep, set())
    kk = [i for i in cx.rep if i not in car]
    sg = rng.choice([-1.0, 1.0], size=len(kk))
    free = cx.hid.copy()
    free[car] = True
    free[cx.named] = True
    mh = cx.hid.copy()
    mh[cx.named] = True
    why = []
    for amp in (0.3, 0.2, 0.1):
        h = xs.copy()
        h[kk] = xs[kk] * 10.0 ** (amp * sg)
        out, res, w = try_match(cx, h, b, free)
        if out is None:
            why.append(f"amp {amp}: {w}")
            continue
        dh = hid_move(cx, xs, out, mh)
        dc = float(cx.disp(xs, out, car).max())
        if dh > 1.0 or dc > amp + 1e-12:
            why.append(f"amp {amp}: hidden/named {dh:.2f}, carriers {dc:.2f} dex")
            continue
        return out, {"tier": 1, "res": res, "carriers": [cx.sp[i] for i in car], "dcar": dc, "dhid": dh, "amp": amp,
                     "signs": dict(zip([cx.sp[i] for i in kk], sg.tolist()))}, ""
    return None, None, "; ".join(why)


def species_list(cx, xs, e):
    """candidate kicked species at this state: the ordered list filtered by x >= 10 S"""
    order = list(dict.fromkeys(q for q in [e["dom_sp"], e["default_sp"]] + list(FALLBACK) if q is not None))
    return order, [s for s in order if xs[cx.sp.index(s)] >= 10.0 * 10.0 ** C.S[s]]


def label(s, e, order, failed):
    why = "dominant miss species" if s == e["dom_sp"] else ("band default" if s == e["default_sp"] else "fallback")
    if s != order[0]:
        why += f" (first choice {order[0]}" + (" below 10 S at t_r" if order[0] not in failed else " infeasible") + ")"
    if failed:
        why += "; fallback after " + ", ".join(failed) + " infeasible"
    return why


def build(sel, t_targets):
    boxes, drops, notes = [], [], []

    def drop(rec, why):
        drops.append({**{a: rec[a] for a in ("src_set", "src_job", "t_target", "t_r", "kick", "ksp", "band", "region")},
                      "why": why})

    for e in sel:
        s, j = e["set"], e["job"]
        t, x, sp, jb = truth(s, j)
        assert len(sp) == 53 and x.shape[1] == 53, (s, j)
        cx = Ctx(sp)
        cfg = {**jb["cfg"], "runtime": RUNTIME, "trun_min": RUNTIME, "rtol": RTOL}
        if DT_VAR_MAX is not None:
            cfg["dt_var_max"] = DT_VAR_MAX
        for ti, tt in enumerate(t_targets):
            i = int(np.argmin(np.abs(np.log10(np.maximum(t, 1e-300)) - np.log10(tt))))
            tr = float(t[i])
            assert t[-1] >= tr + H_END, (s, j, tr, t[-1])
            xs = np.maximum(x[i], 1e-30)
            v0 = started(xs)
            f0 = fractions(v0, cx.W, cx.he)
            base = {"T": jb["T"], "P": jb["P"], "cfg": cfg, "src_set": s, "src_job": j, "t_target": tt, "t_r": tr,
                    "step": i, "dlogt_step": float(abs(np.log10(tr) - np.log10(tt))), "region": e["region"],
                    "band": e["band"], "stratum": e["stratum"], "reason": e["reason"], "sel_n": e["n"]}

            def accept(kid, f, out, meta, ksp=None, kwhy=None):
                rec = {**base, "kick": kid, "ksp": ksp, "ksp_reason": kwhy, "factor": f}
                if out is None:
                    drop(rec, meta)
                    return False
                v = started(out)
                rel = float(np.max(np.abs(fractions(v, cx.W, cx.he) / f0 - 1.0)))
                absf = float(xs.sum() / out.sum() - 1.0)
                if rel > FRAC_MAX:
                    drop(rec, f"started-state element-fraction check {rel:.1e} > {FRAC_MAX:.0e}")
                    return False
                if abs(absf) > ABS_MAX:
                    drop(rec, f"molecule-count change |S0/S' - 1| = {abs(absf):.2e} > {ABS_MAX:.0e}")
                    return False
                d12 = cx.disp(v0, v, cx.rep)
                l25a, l25b = np.log10(np.maximum(C.to25(v0, sp), 1e-30)), np.log10(np.maximum(C.to25(v, sp), 1e-30))
                pres = (l25a >= -12) | (l25b >= -12)
                rec.update(meta)
                rec.update({"elem_rel": rel, "abs_factor": absf, "d0_12": float(d12.max()),
                            "d0_12_sp": C.SP12[int(np.argmax(d12))],
                            "d0_ksp": float(d12[C.SP12.index(ksp)]) if ksp else None,
                            "d25": float(np.abs(l25a - l25b)[pres].max()) if pres.any() else 0.0,
                            "mix": {a: float(max(val, 1e-30)) for a, val in zip(sp, out)}})
                boxes.append(rec)
                return True

            accept("k0", None, xs, {"tier": 0, "res": 0.0, "carriers": [], "dcar": 0.0, "dhid": 0.0})
            order, ok = species_list(cx, xs, e)
            failed = []
            for ksp in ok:
                k = sp.index(ksp)
                kwhy = label(ksp, e, order, failed)
                n_ok = 0
                for kid, f in KICKS:
                    out, meta, why = single(cx, xs, k, f)
                    n_ok += accept(kid, f, out, meta if out is not None else why, ksp, kwhy)
                if n_ok:
                    break
                failed.append(ksp)
            if not ok:
                notes.append(f"{s} {j} t_r {tr:.3g}: no kick species >= 10 S")
            rng = np.random.default_rng([SEED, list(SETS).index(s), j, ti])
            out, meta, why = random_kick(cx, xs, rng)
            accept("k3", None, out, meta if out is not None else why)
    return boxes, drops, notes


def summary(boxes, drops, notes, sel, t_targets):
    L = [f"T2 kicks (build_kicks.py): {len(sel)} source jobs x t_target {list(t_targets)}; {len(boxes)} boxes kept, "
         f"{len(drops)} kicks dropped. rtol {RTOL}, runtime = trun_min = {RUNTIME} s, dt_var_max {DT_VAR_MAX or 'VULCAN default 2'}."]
    L.append("kept / dropped per kick and t_target; tiers of kept (1 / 2 / 3):")
    for k in KIDS:
        for tt in t_targets:
            kb = [b for b in boxes if b["kick"] == k and b["t_target"] == tt]
            kd = [d for d in drops if d["kick"] == k and d["t_target"] == tt]
            tiers = [sum(b["tier"] == q for b in kb) for q in (1, 2, 3)]
            L.append(f"  {k:4s} t {tt:7.1e}: kept {len(kb):4d} dropped {len(kd):4d}; tiers {tiers[0]} / {tiers[1]} / {tiers[2]}")
    L.append("dropped per kick and band (a k1/k2 drop of a species later replaced by a fallback species is still counted):")
    for k in KIDS[1:]:
        L.append(f"  {k:4s}: " + ", ".join(f"{bn} {sum(d['kick'] == k and d['band'] == bn for d in drops)}"
                                           for bn in ("300-650", "650-1300", "1300-2000", "2000-3000")))
    cause = {}
    for d in drops:
        w = d["why"]
        c = ("molecule-count cap" if "molecule-count" in w else "fraction check" if "fraction check" in w else
             "carriers > |kick|" if "would move" in w else "hidden > 1 dex" if "hidden" in w and "tier 2/3 hidden" in w else
             "infeasible" if "tier 2/3 infeasible" in w else "not converged" if "not converged" in w else
             "k3: all amplitudes fail" if d["kick"] == "k3" else w[:40])
        cause[c] = cause.get(c, 0) + 1
    L.append("drop causes: " + ", ".join(f"{a} {v}" for a, v in sorted(cause.items(), key=lambda q: -q[1])))
    amps = [b.get("amp") for b in boxes if b["kick"] == "k3"]
    L.append(f"k3 amplitudes kept: 0.3 {amps.count(0.3)}, 0.2 {amps.count(0.2)}, 0.1 {amps.count(0.1)}")
    ks = {}
    for b in boxes:
        if b["ksp"]:
            ks[b["ksp"]] = ks.get(b["ksp"], 0) + 1
    L.append("kicked species (k1/k2 boxes): " + ", ".join(f"{a} {v}" for a, v in sorted(ks.items(), key=lambda q: -q[1])))
    L.append(f"k1/k2 boxes on a fallback species: below 10 S {sum(1 for b in boxes if b['ksp_reason'] and 'below 10 S' in b['ksp_reason'])}, "
             f"after an infeasible species {sum(1 for b in boxes if b['ksp_reason'] and 'fallback after' in b['ksp_reason'])}")
    jk = {}
    for b in boxes:
        if b["ksp"]:
            jk.setdefault((b["src_set"], b["src_job"]), set()).add(b["t_target"])
    L.append(f"source jobs with >= 1 k1/k2 box: {len(jk)} of {len(sel)}; (job, t_r) pairs with >= 1: {sum(len(v) for v in jk.values())} "
             f"of {len(sel) * len(t_targets)}")
    nb = [b for b in boxes if b["kick"] != "k0"]
    L.append(f"started-state element-fraction check (C, N, O, H, He per atom vs identity): max rel {max(b['elem_rel'] for b in boxes):.2e}; "
             f"re-match residual max {max(b['res'] for b in boxes):.2e}; |abs_factor| p50 {np.median([abs(b['abs_factor']) for b in nb]):.1e} "
             f"p90 {np.percentile([abs(b['abs_factor']) for b in nb], 90):.1e} max {max(abs(b['abs_factor']) for b in nb):.1e}; "
             f"step-time offset |dlog t| max {max(b['dlogt_step'] for b in boxes):.3f}")
    for k in KIDS[1:]:
        kb = [b for b in boxes if b["kick"] == k]
        if kb:
            d = np.array([b["d0_ksp"] if b["d0_ksp"] is not None else b["d0_12"] for b in kb])
            L.append(f"  {k}: realized start displacement ({'kicked species' if k != 'k3' else 'max over 12'}, per-gas floors) "
                     f"p10 {np.percentile(d, 10):.2f} p50 {np.median(d):.2f} p90 {np.percentile(d, 90):.2f}; "
                     f"null (< 0.05 dex) {int((d < 0.05).sum())}; d25 p50 {np.median([b['d25'] for b in kb]):.2f}; "
                     f"carrier move p90 {np.percentile([b['dcar'] for b in kb], 90):.3f}")
    L.append("Known gap: cold gas where CH4 holds most carbon has no single-species CH4 kick (its partner, e.g. CO ~1e-6, would "
             "rise ~2 dex) and k3 keeps CH4 as the C carrier: the CH4-collapse direction is not kicked; an exchange-kick arm "
             "would be a separate test.")
    L += notes
    L.append("")
    L.append("dropped kicks: set job t_target t_r kick ksp band region why")
    for d in drops:
        L.append(f"  {d['src_set']:4s} {d['src_job']:5d} {d['t_target']:7.1e} {d['t_r']:.3e} {d['kick']:4s} {d['ksp'] or '-':5s} "
                 f"{d['band']:9s} {d['region']:4s} {d['why']}")
    L.append("")
    L.append("boxes: i set job t_r kick ksp tier d0_ksp d0_12(sp) d25 dcar abs_factor carriers")
    for i, b in enumerate(boxes):
        L.append(f"  {i:5d} {b['src_set']:4s} {b['src_job']:5d} {b['t_r']:.3e} {b['kick']:4s} {b['ksp'] or '-':5s} {b['tier']} "
                 f"{(b['d0_ksp'] if b['d0_ksp'] is not None else float('nan')):5.2f} {b['d0_12']:5.2f}({b['d0_12_sp']}) "
                 f"{b['d25']:5.2f} {b['dcar']:.3f} {b['abs_factor']:+.1e} {','.join(b['carriers'])}"
                 + (f" amp {b['amp']}" if "amp" in b else ""))
    return "\n".join(L) + "\n"


def main():
    mode = sys.argv[1]
    sel = json.loads((paths.T2 / "jobs_sel.json").read_text())
    if mode == "smoke":
        pick = [("G", 638), ("Gc", 545), ("H2", 52)]
        sel = [e for e in sel if (e["set"], e["job"]) in pick]
        boxes, drops, notes = build(sel, (1.5e3,))
        keep = []
        for e in sel:
            mine = [b for b in boxes if (b["src_set"], b["src_job"]) == (e["set"], e["job"])]
            keep.append(next(b for b in mine if b["kick"] == "k0"))
            keep.append(next(b for kid in ("k2m", "k1m", "k2p", "k1p", "k3") for b in mine if b["kick"] == kid))
        out = paths.T2 / "smoke/tag"
        out.mkdir(parents=True, exist_ok=True)
        assert not any(out.glob("[0-9]*.npz")), "smoke tag already has boxes"
        (out / "jobs.json").write_text(json.dumps(keep))
        print("\n".join(summary(keep, drops, notes, sel, (1.5e3,)).splitlines()[-8:]))
        return
    assert mode in ("full", "dry"), mode
    out = paths.T2 / "tag"
    assert mode == "dry" or not (out / "jobs.json").exists(), "tag/jobs.json exists: refusing to overwrite"
    boxes, drops, notes = build(sel, T_TARGETS)
    order = sorted(range(len(boxes)), key=lambda q: (T_TARGETS.index(boxes[q]["t_target"]), boxes[q]["sel_n"],
                                                       KIDS.index(boxes[q]["kick"])))
    boxes = [boxes[q] for q in order]
    s = summary(boxes, drops, notes, sel, T_TARGETS)
    if mode == "dry":
        print("\n".join(s.splitlines()[:44]))
        return
    out.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(boxes)
    (out / "jobs.json").write_text(txt)
    s = s.replace("\n", f"\njobs.json crc32 {zlib.crc32(txt.encode()):08x}, {len(txt) / 1e6:.1f} MB\n", 1)
    (paths.T2 / "kicks.txt").write_text(s)
    print("\n".join(s.splitlines()[:44]))


if __name__ == "__main__":
    main()
