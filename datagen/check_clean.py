"""Checks of a clean 12-species training cache (cache_gcm.py) against its clean.py packs and the generator's design table
(design_map.npy). Reads only; no generator import.

    python check_clean.py file <cache dir> <raw dir> <design_map.npy> <pack name>     (one pack; run one per pack)
    GCM_FILES=<comma list> python check_clean.py cache <cache dir>

file:  - the pack's groups (rows = SEED - 4,210,000,000) and its missing_rows are disjoint and together are exactly its
         listed rows: run_clean_b<bb>-result.h5 lists rows bb x 2^16 to bb x 2^16 + 2^16 - 1, a partial pack
         run_clean_b<bb>[_o<k>]_n<n>-result.h5 the n rows from bb x 2^16 + k;
       - no group has end_case 3; the start and stratum attrs are the design's codes and their counts equal the design
         table's over the same rows;
       - the cache holds every group of the pack exactly once, and the pack's split shares are near 0.8 / 0.1 / 0.1
         (within 5 binomial standard deviations of its group count);
       - row 0 (every species): the ratios C, N, O, He to H equal the group's C_H, N_H, O_H, He_H attrs to 1e-9
         (relative), and He_H lies in [0.078, 0.100].
cache: end = 99 (the last of the 100 stored rows) for every row of every split; meta.json files = GCM_FILES, in order.
Prints CHECK-CLEAN-FILE OK <pack> / CHECK-CLEAN-CACHE OK, else ... FAIL."""
import json
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import elemproj as EP  # noqa: E402

SEED_BASE = 4_210_000_000      # clean.py: group SEED = SEED_BASE + design row
BLOCK = 2 ** 16
STARTS = ("eqt", "eqn", "eqf", "mix", "pert", "rawA", "rawB")
STRATA = ("G", "C", "F")
SPLIT_P = {"train": 0.8, "validation": 0.1, "test": 0.1}
ok = True


def chk(cond, msg):
    global ok
    ok &= bool(cond)
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)


def design_codes(path, rows):
    """Start and stratum codes of the given design rows: fields 'kind' and 'stratum' of the generator's design map
    (design_map.npy, or design_map_ext.npy for rows from 786,432: a structured array, one record per row, in row order
    from its first record's row)."""
    m = np.load(path, mmap_mode="r")
    m = m[rows - int(m["row"][0])]
    assert np.array_equal(m["row"], rows), "design map records are not in row order"
    return m["kind"].astype(str), m["stratum"].astype(str)


def check_file(cache, raw, dmap, name):
    m = re.fullmatch(r"run_clean_b(\d\d)(?:_o(\d+))?(?:_n(\d+))?-result\.h5", name)
    assert m, f"not a clean pack name: {name}"
    b, k0, n = int(m[1]), int(m[2] or 0), int(m[3] or BLOCK)
    listed = np.arange(b * BLOCK + k0, b * BLOCK + k0 + n)
    with h5py.File(raw / name, "r") as f:
        sp = [s.decode() for s in f.attrs["species"]]
        miss = np.asarray(f.attrs["missing_rows"], dtype=np.int64)
        groups = list(f)
        A = {k: [] for k in ("SEED", "end_case", "start", "stratum", "C_H", "N_H", "O_H", "He_H")}
        x0 = np.empty((len(groups), len(sp)))
        for j, g in enumerate(groups):
            G = f[g]
            for k in A:
                A[k].append(G.attrs[k])
            x0[j] = [G[s + "_evolution"][0] for s in sp]
    rows = np.array(A["SEED"], dtype=np.int64) - SEED_BASE
    start = np.array([str(v) for v in A["start"]])
    strat = np.array([str(v) for v in A["stratum"]])
    print(f"{name}: {len(groups)} groups, {miss.size} missing rows, {len(sp)} species", flush=True)
    both = np.concatenate([rows, miss])
    chk(np.unique(both).size == both.size and np.array_equal(np.sort(both), listed),
        f"groups ({rows.size}) and missing_rows ({miss.size}) are disjoint and together the listed rows "
        f"{listed[0]}-{listed[-1]}")
    ec = np.array(A["end_case"], dtype=np.int64)
    ecc = {int(v): int(c) for v, c in zip(*np.unique(ec, return_counts=True))}
    chk(not np.any(ec == 3), f"no end_case 3 (end_case counts {ecc})")
    chk(set(start) <= set(STARTS) and set(strat) <= set(STRATA),
        f"codes: start {sorted(map(str, set(start)))}, stratum {sorted(map(str, set(strat)))}")
    d_start, d_strat = design_codes(dmap, rows)
    for lab, a, d, codes in (("start", start, d_start, STARTS), ("stratum", strat, d_strat, STRATA)):
        ca = {c: int((a == c).sum()) for c in codes}
        cd = {c: int((d == c).sum()) for c in codes}
        chk(ca == cd, f"{lab} counts equal the design table's over the same rows: {ca}" + ("" if ca == cd else f" vs {cd}"))
    meta = json.loads((cache / "meta.json").read_text())
    k = meta["files"].index(name)
    per = {}
    names = []
    for s in SPLIT_P:
        with np.load(cache / f"raw_{s}.npz") as z:
            sel = z["file_idx"] == k
            per[s] = int(sel.sum())
            names += z["name"][sel].tolist()
    chk(len(names) == len(set(names)) and set(names) == set(groups),
        f"the cache holds every group once ({len(names)} entries, {len(groups)} groups)")
    tot = max(1, sum(per.values()))
    lines = []
    for s, p in SPLIT_P.items():
        sh, tol = per[s] / tot, 5.0 * np.sqrt(p * (1.0 - p) / tot)
        lines.append(abs(sh - p) <= tol)
        print(f"  {s}: {per[s]} rows, share {sh:.4f} (expected {p}, tolerance {tol:.4f})", flush=True)
    chk(all(lines), "split shares near 0.8 / 0.1 / 0.1")
    h, a = (t.numpy() for t in EP.stoich(sp))                               # a columns: C, N, O, He
    r0 = (x0 @ a) / (x0 @ h)[:, None]
    att = np.stack([np.asarray(A[e], dtype=np.float64) for e in ("C_H", "N_H", "O_H", "He_H")], -1)
    rel = np.abs(r0 / att - 1.0)
    chk(np.all(np.isfinite(rel)) and rel.max() <= 1e-9,
        f"row-0 C, N, O, He to H equal the attrs to 1e-9: max relative difference per element "
        f"{', '.join(f'{v:.2e}' for v in rel.max(0))}")
    he = att[:, 3]
    chk(np.all((he >= 0.078) & (he <= 0.100)), f"He_H inside 0.078-0.100 (min {he.min():.5f}, max {he.max():.5f})")
    print(f"CHECK-CLEAN-FILE OK {name}" if ok else f"CHECK-CLEAN-FILE FAIL {name}")


def check_cache(cache):
    meta = json.loads((cache / "meta.json").read_text())
    files = os.environ["GCM_FILES"].split(",")
    chk(meta["files"] == files, f"meta.json files = GCM_FILES ({len(files)} files)")
    chk(np.load(cache / "t_grid.npy").size == 100, "t_grid has 100 stored times")
    for s in SPLIT_P:
        with np.load(cache / f"raw_{s}.npz") as z:
            end = z["end"]
        chk(end.size > 0 and np.all(end == 99), f"{s}: end = 99 for all {end.size} rows (min {end.min() if end.size else None})")
    print("CHECK-CLEAN-CACHE OK" if ok else "CHECK-CLEAN-CACHE FAIL")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "file":
        check_file(Path(a[1]), Path(a[2]), Path(a[3]), a[4])
    elif a[0] == "cache":
        check_cache(Path(a[1]))
    else:
        sys.exit(__doc__)
