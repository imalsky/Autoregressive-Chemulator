"""Sprint 2026-10-08, arm M (plan item 6): small-displacement training rows. Each row k takes production design row
i = 2^20 - 1 - k (rows from the top of gcm.py's design, so they never meet the cold rows scanned upward from 219,549):
its box T, P and element budget (Z, C/O, N/O, He) as gcm.params(i), but its start is the FastChem equilibrium at a
shifted state (T + dT, P 10^dlogP), projected onto the row's element ratios (gcm.start with kind eq, one source), so
the start sits a small displacement off the box's own equilibrium. Shift modes in equal thirds (k mod 3): T only,
P only, both; random signs; |dT| log-uniform 1-300 K, |dlogP| log-uniform 0.003-1 dex; a shift that leaves the box
range (300-3000 K, 1e-6 to 1e3 bar) is redrawn, not clipped. Mode 3 (pilot controls, rows k >= ZERO_K0): no shift.
The box is generate._box (VULCAN-0D, unchanged) at rtol 0.05 (stock, as gcm.py) or a given tighter rtol, every
network species stored, mapped onto T_GRID as gcm.box. gcm.npz(i) is never written and gcm.box is never called (the
cold drivers treat those files as finished design rows); FastChem work files go under this set's folder.

    python gen_shift.py rows <set> <k0> <n> [zero]   <set>/manifest.npz (k, i, mode, dT, dlogP); refuses to overwrite
    python gen_shift.py drive <set> <rtol> <cpu,...> (one fresh process per box, 1800 s timeout; skips finished rows)
    python gen_shift.py box <set> <rtol> <k>         (internal: one box -> <set>/boxes_<rtol>/<k>.npz)
    python gen_shift.py pack <set> <rtol> <out_h5>   (gen_cold.pack's per-group format + design_row, dT, dlogP,
                                                      shift_mode, near_mode, rtol; SEED = 4,260,000,000 + k)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
sys.path.insert(0, str(paths.DATAGEN))
import gcm  # noqa: E402

G = gcm.G
HERE = paths.GEN_SHIFT   # the sets' folders (2026-10-09: was this script's folder, sprint _build/gen_shift)
M_SEED, SEED0, ZERO_K0 = 20261008, 4_260_000_000, 1_000_000
assert SEED0 > gcm.SEED_BASE + 2 ** gcm.M_MAX and SEED0 + ZERO_K0 + 2 ** 17 < 2 ** 32


def design_row(k):
    return 2 ** gcm.M_MAX - 1 - (k if k < ZERO_K0 else 2 ** 17 + (k - ZERO_K0))  # controls: rows below the M range


def shift(k, T, Pbar):
    """(mode, dT K, dlogP dex) of row k."""
    if k >= ZERO_K0:
        return 3, 0.0, 0.0
    rng = np.random.default_rng([M_SEED, k])
    mode = k % 3
    for _ in range(1000):
        dT = rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(0.0, np.log10(300.0)) if mode != 1 else 0.0
        dP = rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(np.log10(0.003), 0.0) if mode != 0 else 0.0
        if 300.0 <= T + dT <= 3000.0 and -6.0 <= np.log10(Pbar) + dP <= 3.0:
            return mode, float(dT), float(dP)
    raise RuntimeError(f"row {k}: no in-box shift")


def rows(s, k0, n, zero=0):
    d = HERE / s
    d.mkdir(exist_ok=True)
    if (d / "manifest.npz").exists():
        raise FileExistsError(d / "manifest.npz")
    ks = list(range(k0, k0 + n)) + list(range(ZERO_K0, ZERO_K0 + zero))
    out = {"k": [], "i": [], "mode": [], "dT": [], "dlogP": [], "T": []}
    for k in ks:
        i = design_row(k)
        p = gcm.params(i)
        m, dT, dP = shift(k, p["T"], p["P"] / 1e6)
        for key, v in zip(out, (k, i, m, dT, dP, p["T"])):
            out[key].append(v)
    np.savez(d / "manifest.npz", **{key: np.array(v) for key, v in out.items()})
    T, m = np.array(out["T"]), np.array(out["mode"])
    print(f"{s}: {len(ks)} rows (k {k0}..{k0 + n - 1} + {zero} zero-shift controls), design rows {min(out['i'])}.."
          f"{max(out['i'])}; modes {np.bincount(m, minlength=4).tolist()}; T {T.min():.0f}-{T.max():.0f} K")


def manifest(s):
    return {key: v for key, v in np.load(HERE / s / "manifest.npz").items()}


def boxdir(s, rtol):
    return HERE / s / f"boxes_{rtol}"


def box(s, rtol, k):
    m = manifest(s)
    j = int(np.flatnonzero(m["k"] == k)[0])
    i, dT, dP = int(m["i"][j]), float(m["dT"][j]), float(m["dlogP"][j])
    sp = gcm.species()
    p = gcm.params(i)
    gcm.W = HERE / s  # FastChem work files: <set>/fc/<j>
    ps = {**p, "kind": "eq", "src": [(p["T"] + dT, p["P"] / 1e6 * 10.0 ** dP)], "w": [1.0]}
    x0 = gcm.start(2_000_000 + k, ps, sp)
    G.SPECIES_OUT = sp
    if rtol != "0.05":
        G.CFG_OVERRIDES["rtol"] = rtol
    r = G._box((p["T"], p["P"], SEED0 + k, -25.0, None, x0))  # chdirs into the VULCAN clone
    out = boxdir(s, rtol) / f"{k}.npz"
    tmp = out.with_name(f"{k}.tmp.npz")
    np.savez(tmp, grid=G.to_grid(r["t"], r["x"]), end_case=r["end_case"], count=r["count"], seconds=r["seconds"],
             t_last=r["t"][-1], x0=np.array([x0[q] for q in sp]), i=i, k=k, dT=dT, dlogP=dP, mode=int(m["mode"][j]),
             rtol=float(rtol))
    os.replace(tmp, out)


def drive(s, rtol, cpus):
    G.build_chem_funs()  # checks the clone's commit; chem_funs.py already built
    m = manifest(s)
    bd = boxdir(s, rtol)
    (bd / "logs").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **G.THREAD_ENV}
    todo = [int(k) for k in m["k"] if not (bd / f"{int(k)}.npz").exists()]
    free, running, t0 = list(cpus), {}, time.time()
    print(f"drive {s} rtol {rtol}: {len(todo)} of {m['k'].size} rows on {len(cpus)} CPUs", flush=True)
    with open(HERE / s / f"status_{rtol}.jsonl", "a") as st:
        while running or todo:
            while todo and free:
                k, c = todo.pop(0), free.pop(0)
                log = open(bd / "logs" / f"{k}.log", "w")
                p = subprocess.Popen(["timeout", "-k", "10", str(gcm.TIMEOUT), "taskset", "-c", c, gcm.PY, str(Path(__file__).resolve()),
                                      "box", s, rtol, str(k)], env=env, stdout=log, stderr=subprocess.STDOUT)
                running[p] = (k, c, time.time(), log)
            time.sleep(0.2)
            for p in [q for q in running if q.poll() is not None]:
                k, c, ts, log = running.pop(p)
                log.close()
                free.append(c)
                ok = (bd / f"{k}.npz").exists()
                if ok:
                    (bd / "logs" / f"{k}.log").unlink()
                st.write(json.dumps({"k": k, "rc": p.returncode, "wall_s": round(time.time() - ts, 2), "cpu": c,
                                     "timeout": p.returncode in (124, 137), "npz": ok}) + "\n")
                st.flush()
    print(f"drive done: {sum((bd / f'{int(k)}.npz').exists() for k in m['k'])} of {m['k'].size} rows present, "
          f"{time.time() - t0:.0f} s", flush=True)


def pack(s, rtol, out, only_shift=True):
    """gen_cold.pack's per-group format for the finished rows of the set (zero-shift controls left out unless
    only_shift is False). Missing rows go to the file attr missing_rows; refuses > 1% missing or an existing out."""
    import h5py
    sp = gcm.species()
    m = manifest(s)
    ks = [int(k) for k in m["k"] if not (only_shift and k >= ZERO_K0)]
    bd = boxdir(s, rtol)
    have = [k for k in ks if (bd / f"{k}.npz").exists()]
    missing = [k for k in ks if not (bd / f"{k}.npz").exists()]
    if len(missing) > len(ks) // 100:
        raise RuntimeError(f"{len(missing)} of {len(ks)} rows missing (first {missing[:5]}); rerun drive")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)
    ends, drifts = {}, []
    with h5py.File(tmp, "w-", libver="v110") as f:
        f.attrs["design"] = (f"gen_shift.py (sprint 2026-10-08, arm M) set {s}, rtol {rtol}: box T, P and element budget "
                             f"from gcm.py design rows 2^20 - 1 - k; start = FastChem equilibrium at (T + dT, P 10^dlogP) "
                             f"projected onto the row's element ratios; SEED = {SEED0} + k (an ID); see the docstring")
        f.attrs["species"] = np.array(sp, dtype="S")
        f.attrs["missing_rows"] = np.array(missing, dtype=np.int64)
        for k in have:
            d = np.load(bd / f"{k}.npz")
            i = int(d["i"])
            p = gcm.params(i)
            ends[int(d["end_case"])] = ends.get(int(d["end_case"]), 0) + 1
            rr2 = gcm.ratios(np.maximum(d["grid"][[0, -1]], 0.0), sp)
            drift = float(np.abs(rr2[1] / rr2[0] - 1.0).max())
            drifts.append(drift)
            g = f.create_group(G.group_name(p["T"], p["P"], SEED0 + k))
            g.attrs["P"] = np.float64(p["P"])
            g.attrs["SEED"] = np.int64(SEED0 + k)
            g.attrs["T"] = np.float64(p["T"])
            for key in ("Z", "C_O", "f_NO"):
                g.attrs[key] = np.float64(p[key])
            g.attrs["sigma"] = np.float64(0.0)
            g.attrs["start"] = "eq"
            g.attrs["near"] = np.int64(1)
            g.attrs["sources_T_Pbar"] = np.array([(p["T"] + float(d["dT"]), p["P"] / 1e6 * 10.0 ** float(d["dlogP"]))],
                                                 dtype=np.float64)
            g.attrs["mix_weights"] = np.array([1.0], dtype=np.float64)
            g.attrs["end_case"] = np.int64(d["end_case"])
            g.attrs["t_last"] = np.float64(d["t_last"])
            g.attrs["elem_drift"] = np.float64(drift)
            g.attrs["design_row"] = np.int64(i)
            g.attrs["dT"] = np.float64(d["dT"])
            g.attrs["dlogP"] = np.float64(d["dlogP"])
            g.attrs["shift_mode"] = np.int64(d["mode"])
            g.attrs["near_mode"] = "loguniform"
            g.attrs["rtol"] = np.float64(d["rtol"])
            g.create_dataset("t_time", data=G.T_GRID, **G.H5_DSET_KW)
            for q, name in enumerate(sp):
                g.create_dataset(name + "_evolution", data=d["grid"][:, q], **G.H5_DSET_KW)
    with h5py.File(tmp, "r") as f:
        assert len(f) == len(have), (len(f), len(have))
    os.replace(tmp, out)
    dr = np.array(drifts)
    print(f"wrote {len(have)} groups to {out}; missing rows {len(missing)}; end_case counts {ends}; element-ratio drift "
          f"p50 {np.median(dr):.2e} p99 {np.percentile(dr, 99):.2e} max {dr.max():.2e}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "rows":
        rows(a[1], int(a[2]), int(a[3]), int(a[4]) if len(a) > 4 else 0)
    elif a[0] == "drive":
        drive(a[1], a[2], a[3].split(","))
    elif a[0] == "box":
        box(a[1], a[2], int(a[3]))
    elif a[0] == "pack":
        pack(a[1], a[2], a[3])
