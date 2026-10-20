"""T2 snap-back test, step 1 (scale phase, 2026-10-14): source jobs from the six development sets (G, Gc, H2, H2c, H3, H3c).
Read-only on every input; writes paths.T2/jobs.txt and jobs_sel.json (refuses to overwrite them: they define the
stored T2 tag). env_margin() is the envelope every decision scorer uses.

  miss   every job of FINAL's lasting rows (error > 0.5 dex at the last checkpoint, 1000 days after the start) and of its
         rows > 2 dex (late max), from research/critic/rows_e13s_FINw1536_s2.npz (scoreboard.bench, per-gas rule).
         Dominant species: late_sp of the job's worst > 2 dex row if it has one, else last_sp of its worst lasting row.
  strat  a stratified sample inside env_any+margin (GCM bins holding >= 1 model, research/gcm/gcm_tp.npz n_models,
         dilated by +-4 T bins = +-100 K and +-2 log P bins = +-0.1 dex; the grid spans 1e-5..100 bar), miss jobs excluded:
         band quotas 300-650 K 50, 650-1300 K 55, 1300-2000 K 30, 2000-3000 K 25 (66 % below 1300 K), each split evenly over
         the log P decades that hold candidates (shortfalls of a cell go to the band's other cells, round robin).
  cool   the cooler-planet stratum: 300-1300 K, 1e-5..100 bar, outside env_any+margin, miss jobs excluded; 30 jobs split
         evenly over log P decades.
Kick species default by band (used when a job has no dominant miss species, or when that species is below 10 S at t_r;
build_kicks.py): 300-650 K CH4, 650-1300 K C2H2, 1300-2000 K CO2, 2000-3000 K H2O.
Seed 20261014 (numpy default_rng).
    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 taskset -c 9,11 $PY select_jobs.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402

SEED = 20261014
BANDS = [(300.0, 650.0), (650.0, 1300.0), (1300.0, 2000.0), (2000.0, 3000.1)]
BNAME = ["300-650", "650-1300", "1300-2000", "2000-3000"]
QUOTA = [50, 55, 30, 25]
N_COOL = 30
DEFAULT_SP = {"300-650": "CH4", "650-1300": "C2H2", "1300-2000": "CO2", "2000-3000": "H2O"}


def env_margin():
    """inside(T K, P bar) for env_any dilated by +-100 K (4 x 25 K bins) and +-0.1 dex (2 x 0.05 dex bins)."""
    g = np.load(paths.GCM_TP)
    lp, te, nm = g["logP_edges"], g["T_edges"], g["n_models"]
    m = nm >= 1
    pad = np.zeros((m.shape[0] + 4, m.shape[1] + 8), bool)
    for di in range(-2, 3):
        for dj in range(-4, 5):
            pad[2 + di:2 + di + m.shape[0], 4 + dj:4 + dj + m.shape[1]] |= m
    md = pad[2:-2, 4:-4]
    assert abs(lp[0] + 5.0) < 1e-9 and abs(lp[-1] - 2.0) < 1e-9

    def inside(T, P):
        T, x = np.asarray(T, float), np.log10(np.asarray(P, float))
        i = np.searchsorted(lp, x, side="right") - 1
        j = np.searchsorted(te, T, side="right") - 1
        i = np.where(x == lp[-1], lp.size - 2, i)
        j = np.where(T == te[-1], te.size - 2, j)
        ok = (i >= 0) & (i < lp.size - 1) & (j >= 0) & (j < te.size - 1)
        return ok & md[np.clip(i, 0, lp.size - 2), np.clip(j, 0, te.size - 2)]
    return inside, int(m.sum()), int(md.sum())


def band_of(T):
    for (lo, hi), nme in zip(BANDS, BNAME):
        if lo <= T < hi:
            return nme
    return "none"


def take_even(rng, cells, quota):
    """cells: {key: array of candidate indices}; even split, shortfalls redistributed round robin."""
    keys = sorted(k for k in cells if len(cells[k]))
    take = {k: 0 for k in keys}
    left = quota
    while left > 0:
        open_ = [k for k in keys if take[k] < len(cells[k])]
        if not open_:
            break
        for k in open_:
            if left == 0:
                break
            take[k] += 1
            left -= 1
    return {k: rng.choice(cells[k], size=take[k], replace=False) for k in keys}, take


def main():
    for f in ("jobs.txt", "jobs_sel.json"):
        if (paths.T2 / f).exists():
            sys.exit(f"refusing to overwrite {paths.T2 / f}")
    r = dict(np.load(paths.CRITIC_ROWS))
    key = [(str(s), int(j)) for s, j in zip(r["set"], r["job"])]
    jobs = sorted(set(key))
    idx = {k: [i for i, kk in enumerate(key) if kk == k] for k in jobs}
    T = {k: float(r["T"][idx[k][0]]) for k in jobs}
    P = {k: float(r["P"][idx[k][0]]) for k in jobs}           # bar
    inside, n_env, n_envm = env_margin()
    ins = {k: bool(inside(T[k], P[k])) for k in jobs}
    inbox = {k: 1e-5 <= P[k] <= 100.0 for k in jobs}

    sel = {}
    for k in jobs:
        rows = idx[k]
        big = [i for i in rows if r["late"][i] > 2.0]
        last = [i for i in rows if r["last"][i] > 0.5]
        if not big and not last:
            continue
        why = []
        if last:
            il = max(last, key=lambda i: r["last"][i])
            why.append(f"lasting {len(last)} row(s), max {r['last'][il]:.2f} {r['last_sp'][il]} (t0 {r['t0'][il]:g})")
        if big:
            ib = max(big, key=lambda i: r["late"][i])
            why.append(f"> 2 dex {len(big)} row(s), max {r['late'][ib]:.2f} {r['late_sp'][ib]} (t0 {r['t0'][ib]:g})")
        dom = str(r["late_sp"][ib]) if big else str(r["last_sp"][il])
        sel[k] = {"stratum": "miss", "reason": "; ".join(why), "dom_sp": dom}
    n_miss = len(sel)

    rng = np.random.default_rng(SEED)
    rest = [k for k in jobs if k not in sel]
    log = []
    for (lo, hi), bn, q in zip(BANDS, BNAME, QUOTA):
        cand = [k for k in rest if ins[k] and lo <= T[k] < hi]
        cells = {}
        for k in cand:
            d = int(min(np.floor(np.log10(P[k])), 1))
            cells.setdefault(d, []).append(k)
        cells = {d: np.array(sorted(v), dtype=object) for d, v in cells.items()}
        pick, take = take_even(rng, cells, q)
        for d, ks in pick.items():
            for k in ks:
                k = (str(k[0]), int(k[1]))
                sel[k] = {"stratum": "strat", "reason": f"stratified {bn} K, log P decade {d}", "dom_sp": None}
        log.append(f"  strat {bn} K: {len(cand)} candidates; taken per decade " +
                   ", ".join(f"{d}: {take[d]}/{len(cells[d])}" for d in sorted(take)))
    cand = [k for k in rest if k not in sel and inbox[k] and 300.0 <= T[k] < 1300.0 and not ins[k]]
    cells = {}
    for k in cand:
        d = int(min(np.floor(np.log10(P[k])), 1))
        cells.setdefault(d, []).append(k)
    cells = {d: np.array(sorted(v), dtype=object) for d, v in cells.items()}
    pick, take = take_even(rng, cells, N_COOL)
    for d, ks in pick.items():
        for k in ks:
            k = (str(k[0]), int(k[1]))
            sel[k] = {"stratum": "cool", "reason": f"cooler stratum 300-1300 K outside env_any+margin, log P decade {d}",
                      "dom_sp": None}
    log.append(f"  cool: {len(cand)} candidates; taken per decade " + ", ".join(f"{d}: {take[d]}/{len(cells[d])}" for d in sorted(take)))

    out, lines = [], []
    order = sorted(sel, key=lambda k: ({"miss": 0, "strat": 1, "cool": 2}[sel[k]["stratum"]], k[0], k[1]))
    for n, k in enumerate(order):
        s = sel[k]
        reg = "in" if ins[k] else ("cool" if (inbox[k] and 300.0 <= T[k] < 1300.0) else "out")
        bn = band_of(T[k])
        e = {"n": n, "set": k[0], "job": k[1], "T": T[k], "P_bar": P[k], "region": reg, "band": bn,
             "stratum": s["stratum"], "dom_sp": s["dom_sp"], "default_sp": DEFAULT_SP.get(bn), "reason": s["reason"]}
        out.append(e)
        lines.append(f"{n:4d} {k[0]:4s} {k[1]:5d} {T[k]:7.1f} {P[k]:.3e} {reg:4s} {bn:9s} {s['stratum']:5s} "
                     f"{(s['dom_sp'] or '-'):5s} {DEFAULT_SP.get(bn, '-'):5s} {s['reason']}")
    cnt = {}
    for e in out:
        cnt.setdefault((e["stratum"], e["band"], e["region"]), 0)
        cnt[(e["stratum"], e["band"], e["region"])] += 1
    head = [f"T2 source jobs (select_jobs.py, seed {SEED}): {len(out)} jobs = {n_miss} miss + "
            f"{sum(e['stratum'] == 'strat' for e in out)} stratified inside env_any+margin + "
            f"{sum(e['stratum'] == 'cool' for e in out)} cooler stratum.",
            f"env_any bins (n_models >= 1): {n_env}; after the +-100 K / +-0.1 dex margin: {n_envm} (of 140 x 176).",
            f"region: in = inside env_any+margin; cool = 300-1300 K, 1e-5..100 bar, outside it; out = anything else (miss jobs only).",
            "kick species: dom_sp (miss jobs) first, else the band default; build_kicks.py falls back down the list "
            "[dom_sp, band default, CH4, C2H2, CO2, H2O] to the first species with truth x >= 10 S at t_r.",
            "band defaults: 300-650 K CH4, 650-1300 K C2H2, 1300-2000 K CO2, 2000-3000 K H2O."] + log + \
           ["counts (stratum, band, region): " + ", ".join(f"{a}/{b}/{c} {v}" for (a, b, c), v in sorted(cnt.items())),
            "", "   n set    job       T     P_bar  reg  band      strat dom   dflt  reason"]
    (paths.T2 / "jobs.txt").write_text("\n".join(head + lines) + "\n")
    (paths.T2 / "jobs_sel.json").write_text(json.dumps(out, indent=1))
    print("\n".join(head[:8 + len(log)]))


if __name__ == "__main__":
    main()
