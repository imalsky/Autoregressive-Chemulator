"""E13 cold data (2026-10-07): new gcm.py boxes for the 32,768 first production design rows >= 131072 (never run) with
gcm.params(i)["T"] < 1300 K, packed into ONE HDF5 file with gcm.pack's per-group format. Thin driver around gcm.py
(imported, unchanged): boxes run as `gcm.py box i` (npz in gcm's _build/gcm/boxes/<i // 2^16>/, kept), status in
_build/gcm/status_cold.jsonl, failed-box logs in _build/gcm/logs/.

    python gen_cold.py rows                        (rows/cold_gen_rows.npy; refuses to overwrite)
    python gen_cold.py drive <cpu,cpu,...> [n]     (the first n listed rows, default all; skips rows whose npz exists)
    python gen_cold.py pack <out_h5> [n]           (the first n listed rows that have an npz -> out_h5; missing rows in
                                                    the file attr missing_rows; refuses > 1% missing or an existing out)
    python gen_cold.py check <new_cache> <old_cache> <new_h5> [n]
                                                   (split sizes; asserts each old-cache group (blocks 0-1) has the same
                                                    split and identical entries in the new cache, and that new_h5 and
                                                    the cache account for every listed row)
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
ROWS = paths.E / "rows" / "cold_gen_rows.npy"   # 2026-10-09: was beside this script (E13_gcm/rows)
FIRST, N_ROWS, T_MAX = 2 * gcm.BLOCK, 32768, 1300.0
STATUS = gcm.W / "status_cold.jsonl"
UPPER = 2 ** gcm.M_MAX
# Sprint 2026-10-08 (plan item 3): GEN_ROWS (rows file), GEN_FIRST, GEN_N, GEN_UPPER (rows must stay below it: the
# M rows are taken from the top of the design) and GEN_STATUS override the constants above. Two drivers need disjoint
# rows files (the busy-process assert runs over the listed rows) and one status file each; pack and check read the
# full list.
ROWS = Path(os.environ.get("GEN_ROWS", ROWS))
FIRST, N_ROWS = int(os.environ.get("GEN_FIRST", FIRST)), int(os.environ.get("GEN_N", N_ROWS))
UPPER, STATUS = int(os.environ.get("GEN_UPPER", UPPER)), Path(os.environ.get("GEN_STATUS", STATUS))
T_LO, T_MAX = float(os.environ.get("GEN_TLO", 0.0)), float(os.environ.get("GEN_TMAX", T_MAX))  # rows with T_LO <= T < T_MAX


def rows():
    if ROWS.exists():
        raise FileExistsError(ROWS)
    sel, i = [], FIRST
    while len(sel) < N_ROWS:
        if T_LO <= gcm.params(i)["T"] < T_MAX:
            sel.append(i)
        i += 1
    sel = np.array(sel, np.int64)
    T = np.array([gcm.params(int(k))["T"] for k in sel])
    assert sel[0] >= FIRST and sel[-1] < UPPER and np.all(np.diff(sel) > 0) and np.all(T < T_MAX) and np.all(T >= T_LO)
    np.save(ROWS, sel)
    print(f"{sel.size} rows {sel[0]}..{sel[-1]} (scanned {i - FIRST}), T {T.min():.2f}-{T.max():.2f} K, blocks "
          f"{dict(zip(*[a.tolist() for a in np.unique(sel // gcm.BLOCK, return_counts=True)]))}")


def listed(n=None):
    r = np.load(ROWS)
    return [int(i) for i in (r if n is None else r[:n])]


def drive(cpus, n=None):
    """gcm.drive's loop over the listed rows: one fresh process per box, taskset to one CPU, gcm's timeout and env."""
    G.build_chem_funs()  # checks the clone's commit; chem_funs.py already built
    rr = listed(n)
    blocks = sorted({i // gcm.BLOCK for i in rr})
    assert not any(gcm.packed(b).exists() for b in blocks), "a block of these rows is packed (its npz were removed)"
    ps = subprocess.run(["pgrep", "-a", "-u", str(os.getuid()), "-f", "gcm.py box "], capture_output=True, text=True)
    busy = {int(ln.split()[-1]) for ln in ps.stdout.splitlines() if ln.split()[-1].isdigit()} & set(rr)
    assert not busy, f"box processes still running for {len(busy)} listed rows (e.g. {sorted(busy)[:5]}); wait for them"
    (gcm.W / "logs").mkdir(parents=True, exist_ok=True)
    for b in blocks:
        (gcm.W / "boxes" / f"{b:02d}").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **G.THREAD_ENV}
    todo = [i for i in rr if not gcm.npz(i).exists()]
    free, running, t0 = list(cpus), {}, time.time()
    print(f"drive: {len(todo)} of {len(rr)} listed rows to run on {len(cpus)} CPUs", flush=True)
    with open(STATUS, "a") as st:
        while running or todo:
            while todo and free:
                i, c = todo.pop(0), free.pop(0)
                log = open(gcm.W / "logs" / f"{i}.log", "w")
                p = subprocess.Popen(["timeout", "-k", "10", str(gcm.TIMEOUT), "taskset", "-c", c, gcm.PY,
                                      str(gcm.HERE / "gcm.py"), "box", str(i)], env=env, stdout=log,
                                     stderr=subprocess.STDOUT)
                running[p] = (i, c, time.time(), log)
            time.sleep(0.2)
            for p in [q for q in running if q.poll() is not None]:
                i, c, ts, log = running.pop(p)
                log.close()
                free.append(c)
                ok = gcm.npz(i).exists()
                if ok:
                    (gcm.W / "logs" / f"{i}.log").unlink()
                st.write(json.dumps({"i": i, "rc": p.returncode, "wall_s": round(time.time() - ts, 2), "cpu": c,
                                     "timeout": p.returncode in (124, 137), "npz": ok}) + "\n")
                st.flush()
    print(f"drive done: listed rows present {sum(gcm.npz(i).exists() for i in rr)} of {len(rr)}, "
          f"{time.time() - t0:.0f} s", flush=True)


def pack(out, n=None):
    """gcm.pack's per-group format (attrs, datasets, compression) for the listed rows that have an npz -> out (written
    as out.tmp, checked, renamed). npz kept."""
    import h5py
    sp = gcm.species()
    rr = listed(n)
    have = [i for i in rr if gcm.npz(i).exists()]
    missing = [i for i in rr if not gcm.npz(i).exists()]
    if len(missing) > len(rr) // 100:
        raise RuntimeError(f"{len(missing)} of {len(rr)} rows missing (first {missing[:5]}); rerun drive")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)
    ends, drifts = {}, []
    with h5py.File(tmp, "w-", libver="v110") as f:
        f.attrs["design"] = (f"gcm.py (2026-10-06 design), scrambled Sobol d={gcm.D} seed={gcm.QMC_SEED}, rows: the first "
                             f"{len(rr)} rows >= {FIRST} of 2^{gcm.M_MAX} with T < {T_MAX:g} K ({rr[0]}-{rr[-1]}, listed in "
                             f"{ROWS}; gen_cold.py); see the gcm.py docstring; SEED = {gcm.SEED_BASE} + row (an ID)")
        f.attrs["species"] = np.array(sp, dtype="S")
        f.attrs["missing_rows"] = np.array(missing, dtype=np.int64)
        for i in have:
            p = gcm.params(i)
            d = np.load(gcm.npz(i))
            ends[int(d["end_case"])] = ends.get(int(d["end_case"]), 0) + 1
            rr2 = gcm.ratios(np.maximum(d["grid"][[0, -1]], 0.0), sp)
            drift = float(np.abs(rr2[1] / rr2[0] - 1.0).max())
            drifts.append(drift)
            g = f.create_group(G.group_name(p["T"], p["P"], gcm.SEED_BASE + i))
            g.attrs["P"] = np.float64(p["P"])
            g.attrs["SEED"] = np.int64(gcm.SEED_BASE + i)
            g.attrs["T"] = np.float64(p["T"])
            for k in ("Z", "C_O", "f_NO", "sigma"):
                g.attrs[k] = np.float64(p[k])
            g.attrs["start"] = p["kind"]
            g.attrs["near"] = np.int64(p["near"])
            g.attrs["sources_T_Pbar"] = np.array(p["src"], dtype=np.float64)
            g.attrs["mix_weights"] = np.array(p["w"], dtype=np.float64)
            g.attrs["end_case"] = np.int64(d["end_case"])
            g.attrs["t_last"] = np.float64(d["t_last"])
            g.attrs["elem_drift"] = np.float64(drift)
            g.create_dataset("t_time", data=G.T_GRID, **G.H5_DSET_KW)
            for k, s in enumerate(sp):
                g.create_dataset(s + "_evolution", data=d["grid"][:, k], **G.H5_DSET_KW)
    with h5py.File(tmp, "r") as f:
        assert len(f) == len(have), (len(f), len(have))
    os.replace(tmp, out)
    dr = np.array(drifts)
    print(f"wrote {len(have)} groups to {out}; missing rows {len(missing)}; end_case counts {ends}; element-ratio drift "
          f"p50 {np.median(dr):.2e} p99 {np.percentile(dr, 99):.2e} max {dr.max():.2e}", flush=True)


def check(new, old, h5, n=None, rows=True):
    """Every group of the old cache (blocks 0-1) must sit in the same split of the new cache (by group name), with
    identical entries; the new file h5 must hold every listed row (or record it missing), and every one of its groups
    must be in the new cache or accounted for (step cap or a reject)."""
    import h5py
    mn = json.loads((Path(new) / "meta.json").read_text())
    mo = json.loads((Path(old) / "meta.json").read_text())
    print(f"new cache {new}: files {mn['files']}, kept_by_split {mn['kept_by_split']}, kept_total {mn['kept_total']}, "
          f"group_rejects {mn['group_rejects']}")
    print(f"old cache {old}: files {mo['files']}, kept_by_split {mo['kept_by_split']}")
    assert mn["files"][:len(mo["files"])] == mo["files"], (mn["files"], mo["files"])
    k, fresh = len(mo["files"]), []
    for s in ("train", "validation", "test"):
        za, zb = np.load(Path(new) / f"raw_{s}.npz"), np.load(Path(old) / f"raw_{s}.npz")
        na, fa, nb = za["name"], za["file_idx"], zb["name"]
        same = set(na[fa < k].tolist()) == set(nb.tolist()) and int((fa < k).sum()) == nb.size
        print(f"{s}: new {na.size} (blocks 0-{k - 1}: {int((fa < k).sum())}, new file: {int((fa >= k).sum())}); "
              f"old {nb.size}; old groups in the same split: {same}", flush=True)
        assert same and np.unique(na).size == na.size, f"split mismatch in {s}"
        ident = {key: np.array_equal(za[key][:nb.size], zb[key]) for key in zb.files}
        print(f"{s}: old entries identical as the new cache's first {nb.size} (every field: {sorted(ident)}): "
              f"{all(ident.values())}", flush=True)
        assert all(ident.values()), ident
        fresh += na[fa >= k].tolist()
    with h5py.File(h5, "r") as f:
        miss = [int(i) for i in f.attrs["missing_rows"]]
        seeds = {n: int(f[n].attrs["SEED"]) for n in f}
        cap = sum(int(f[n].attrs["end_case"]) == 3 for n in f)
    if rows:  # sprint 2026-10-08: --no-rows (a new file outside the design, e.g. arm M) skips the listed-rows check
        rr = listed(n)
        assert sorted([s - gcm.SEED_BASE for s in seeds.values()] + miss) == rr, "new file does not cover the listed rows"
    rej = sum(mn["group_rejects"].values()) - sum(mo["group_rejects"].values())
    assert set(fresh) <= set(seeds) and len(fresh) + cap + rej == len(seeds), (len(fresh), cap, rej, len(seeds))
    print(f"new file {h5}: {len(seeds)} groups + {len(miss)} missing rows; in the cache {len(fresh)}, step cap {cap}, "
          f"rejected {rej}")
    print("SPLIT-EQUALITY OK: every block-0/1 group is in the same split as in the old cache (entries identical)",
          flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "rows":
        rows()
    elif a[0] == "drive":
        drive(a[1].split(","), int(a[2]) if len(a) > 2 else None)
    elif a[0] == "pack":
        pack(a[1], int(a[2]) if len(a) > 2 else None)
    elif a[0] == "check":
        nr = "--no-rows" in a
        a = [x for x in a if x != "--no-rows"]
        check(a[1], a[2], a[3], int(a[4]) if len(a) > 4 else None, rows=not nr)
