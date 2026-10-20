"""Raw-trajectory cache builder (datagen/cache_gcm.py reuses _work, _cfg, CACHE_DIR and SPLITS for the generator's packs).

Every raw group that the 12-species preprocessing keeps (same rejects, same frozen-tail cut, same split, same
frozen_tail_no_window rule) is stored with its full raw grid as log10 mixing ratio (float64), so the GPU sampler can
interpolate windows at any (t_start, dt, K) with the exact interpolant. Uses that preprocessing's own functions
(processing/preprocessing.py and configs/stage1.json of the pinned snapshot, paths.SNAPSHOT).

Run (no multiprocessing: its spawn semaphores live on the shared RAM disk), one process per chunk:
    for i in 0..7: taskset -c <cpu_i> python rawtraj.py chunk i 8   (writes CACHE_DIR/part_<i>.pkl)
    python rawtraj.py merge 8                                        (writes CACHE_DIR/raw_<split>.npz)
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

REPO = paths.SNAPSHOT
sys.path.insert(0, str(paths.SNAP_PROCESSING))
import preprocessing as PP  # noqa: E402

CACHE_DIR = Path(os.environ.get("ARCH_CACHE", str(paths.ARCH_TESTS)))
FLOOR = os.environ.get("RAWTRAJ_FLOOR") == "1"  # keep groups with values below drop_below (floored at epsilon)
CFG_PATH = REPO / "configs" / "stage1.json"
SPLITS = ("train", "validation", "test")
LO_BY_FILE = {"run6050-result.h5": -20.0}  # sampler lower bound; -25 for the other files


def _cfg():
    return PP.parse_precfg(PP.load_json(CFG_PATH), cfg_path=CFG_PATH)


def _work(args):
    file_path, names = args
    cfg = _cfg()
    eps = float(cfg.epsilon)
    out = {sp: {"logy": [], "end": [], "conv": [], "P": [], "T": [], "seed": [], "name": []} for sp in SPLITS}
    rej = {"non_finite": 0, "non_positive_global": 0, "too_few_valid": 0, "drop_below": 0, "frozen_tail_no_window": 0}
    tref = None
    with h5py.File(file_path, "r") as fin:
        for name in names:
            grp = fin[name]
            li = PP.leaf_dataset_index(grp)
            try:
                t_raw = PP.read_time(grp, time_key=cfg.time_key, leaf_index=li)
                y_raw = PP.read_species_matrix(grp, t_len=t_raw.shape[0], species_vars=cfg.species_variables, leaf_index=li)
                g = PP.read_globals_vector(grp, global_vars=cfg.global_variables, leaf_index=li)
            except ValueError as e:
                if PP._is_non_finite_raw_error(e):
                    rej["non_finite"] += 1
                    continue
                if PP._is_non_positive_global_error(e):
                    rej["non_positive_global"] += 1
                    continue
                raise
            if tref is None:
                tref = t_raw.copy()
            if t_raw.shape != tref.shape or not np.allclose(t_raw, tref, rtol=1e-12, atol=0):
                raise RuntimeError(f"{name}: time grid differs from the file's first grid")
            valid = t_raw > 0
            if int(np.count_nonzero(valid)) < 2:
                rej["too_few_valid"] += 1
                continue
            if not FLOOR and np.any(y_raw < float(cfg.drop_below)):
                rej["drop_below"] += 1
                continue
            cut = PP.frozen_tail_cut(y_raw, epsilon=eps)
            T_len = t_raw.shape[0]
            end = T_len - 1 if cut is None else int(cut)
            if cut is not None:
                # Production rejects a cut group none of whose 4 draws fits (frozen_tail_no_window).
                v = valid & (np.arange(T_len) <= cut)
                t_valid = t_raw[v]
                rng = np.random.default_rng([int(cfg.seed), PP._seed_word(str(name))])
                ok = False
                for sidx in range(int(cfg.samples_per_source_trajectory)):
                    dt_s = PP.sample_dt(cfg, rng)
                    ts = PP.pick_t_start(t_raw=t_raw, t_valid=t_valid, dt_s=dt_s, n_steps=cfg.n_steps, t_min=cfg.t_min,
                                         rng=rng, anchor_first=(sidx == 0), max_attempts=cfg.max_chunk_attempts_per_source)
                    if ts is not None:
                        ok = True  # production would also interpolate; finite data give finite windows
                if not ok:
                    rej["frozen_tail_no_window"] += 1
                    continue
            # Converged repeated tail (kept by production): first row of the trailing identical run.
            same = np.all(y_raw == y_raw[-1], axis=1)
            differ = np.flatnonzero(~same)
            conv = -1
            if cut is None and differ.size and int(differ[-1]) + 1 < T_len - 1:
                conv = int(differ[-1]) + 1
            split = PP.split_for_trajectory(Path(file_path).name, str(name), seed=PP.SPLIT_SEED,
                                            val_frac=cfg.val_fraction, test_frac=cfg.test_fraction)
            o = out[split]
            o["logy"].append(np.log10(np.maximum(y_raw, eps)))
            o["end"].append(end)
            o["conv"].append(conv)
            o["P"].append(float(g[0]))
            o["T"].append(float(g[1]))
            o["seed"].append(int(np.asarray(grp.attrs["SEED"]).reshape(-1)[0]))
            o["name"].append(str(name))
    return Path(file_path).name, out, rej, tref


def _jobs(n):
    cfg = _cfg()
    jobs = []
    per_file = n // 4
    for fp in PP.list_raw_files(cfg.raw_dir, cfg.raw_file_patterns):
        with h5py.File(fp, "r") as f:
            names = list(f.keys())
        step = -(-len(names) // per_file)
        jobs += [(str(fp), names[k:k + step]) for k in range(0, len(names), step)]
    if len(jobs) != n:
        raise RuntimeError(f"expected {n} chunks, got {len(jobs)}")
    return jobs


def chunk(i, n):
    t0 = time.time()
    res = _work(_jobs(n)[i])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"part_{i}.pkl", "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"chunk {i}/{n} done in {time.time() - t0:.0f} s", flush=True)


def merge(n):
    cfg = _cfg()
    res = []
    for i in range(n):
        with open(CACHE_DIR / f"part_{i}.pkl", "rb") as f:
            res.append(pickle.load(f))
    rej_tot = {}
    grids = {}
    file_names = sorted({r[0] for r in res})
    for fname, _, rej, tref in res:
        for k, v in rej.items():
            rej_tot[k] = rej_tot.get(k, 0) + v
        grids.setdefault(fname, tref)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for sp in SPLITS:
        parts = [(fname, out[sp]) for fname, out, _, _ in res]
        fidx = np.concatenate([np.full(len(o["end"]), file_names.index(fn), np.int8) for fn, o in parts])
        lo = np.array([LO_BY_FILE.get(fn, -25.0) for fn in file_names])[fidx]
        cat = lambda k, dt: np.concatenate([np.asarray(o[k], dtype=dt) for _, o in parts])
        logy = np.concatenate([np.stack(o["logy"]) for _, o in parts if o["logy"]])
        np.savez(CACHE_DIR / f"raw_{sp}.npz", logy=logy, end=cat("end", np.int16), conv=cat("conv", np.int16),
                 P=cat("P", np.float64), T=cat("T", np.float64), seed=cat("seed", np.int64), file_idx=fidx, lo=lo,
                 name=cat("name", "U64"))
        counts[sp] = int(logy.shape[0])
    t_grid = grids[file_names[0]]
    np.save(CACHE_DIR / "t_grid.npy", t_grid)
    summ = json.loads((cfg.processed_dir / "preprocessing_summary.json").read_text())
    meta = {"files": file_names, "kept_by_split": counts, "kept_total": sum(counts.values()), "group_rejects": rej_tot,
            "max_grid_rel_diff_vs_first_file": {fn: float(np.max(np.abs(grids[fn] / t_grid - 1)[1:])) for fn in file_names},
            "production_groups_kept": summ.get("groups_kept"),
            "production_group_rejects": summ.get("group_rejects")}
    (CACHE_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    for i in range(n):
        (CACHE_DIR / f"part_{i}.pkl").unlink()


if __name__ == "__main__":
    if sys.argv[1] == "chunk":
        chunk(int(sys.argv[2]), int(sys.argv[3]))
    else:
        merge(int(sys.argv[2]))
