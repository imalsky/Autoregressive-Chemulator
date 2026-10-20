"""Checks of the training cache U4 against U3, the cache it extends. No writes.
  1. files: U4's first files are U3's 14 (same names, same order); the rest are run_gcm_z23n1 and run_gcm_z23n2; t_grid
     and the species list equal U3's.
  2. prefix identity: in each split, U4's first n_U3 rows equal U3's rows bitwise in every field: old rows keep their
     split, order and values, so the per-row draw weights carry over; old files' rows come first.
  3. per new file (not with --res): SEED - 4.2e9 plus the file's missing_rows = paths.GEN/rows_n1.npy / rows_n2.npy;
     start types are design types only; cache entries are distinct groups of the file; groups - entries - step-cap groups
     (end_case 3) >= 0, and summed over the new files it equals the cache's new rawtraj rejects (meta group_rejects minus
     U3's).
    python check_u4.py <U4 dir> <U3 dir> <raw dir>          (12-species or 53-species caches)
    python check_u4.py --res <U4 res dir> <U3 res dir>        (files and prefix identity only)
Prints CHECK-U4 OK / CHECK-U4-RES OK, else ... FAIL."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402

SEED_BASE = 4_200_000_000   # gcm.py's SEED_BASE: group SEED = SEED_BASE + design row
GEN = paths.GEN
NEW = ["run_gcm_z23n1-result.h5", "run_gcm_z23n2-result.h5"]
COV = {"run_gcm_z23n1-result.h5": "rows_n1.npy", "run_gcm_z23n2-result.h5": "rows_n2.npy"}
res_mode = sys.argv[1] == "--res"
a = sys.argv[2:] if res_mode else sys.argv[1:]
new, old = Path(a[0]), Path(a[1])
raw = None if res_mode else Path(a[2])
ok = True


def chk(cond, msg):
    global ok
    ok &= bool(cond)
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)


mn, mo = json.loads((new / "meta.json").read_text()), json.loads((old / "meta.json").read_text())
nf = len(mo["files"])
chk(mn["files"][:nf] == mo["files"], f"U4 files start with U3's {nf} files")
chk(mn["files"][nf:] == NEW, f"new files {mn['files'][nf:]}")
chk(np.array_equal(np.load(new / "t_grid.npy"), np.load(old / "t_grid.npy")), "t_grid equal")
if "species" in mo:
    chk(mn.get("species") == mo["species"], "species list equal")
names = {k: {} for k in range(nf, len(mn["files"]))}
for s in ("train", "validation", "test"):
    zn, zo = np.load(new / f"raw_{s}.npz"), np.load(old / f"raw_{s}.npz")
    n0 = zo["seed"].size
    chk(zn["seed"].size >= n0, f"{s}: U4 rows {zn['seed'].size} >= U3 rows {n0}")
    for key in zo.files:
        A, B = zo[key], zn[key][:n0]
        same = A.shape == B.shape and A.dtype == B.dtype and np.array_equal(A, B)
        chk(same, f"{s}: prefix {key} bitwise equal")
    fi = zn["file_idx"]
    chk(np.all(fi[:n0] < nf) and np.all(fi[n0:] >= nf), f"{s}: old files' rows first, new files' after")
    for k in names:
        names[k][s] = zn["name"][fi == k].tolist()
    print(f"  {s}: U3 {n0} rows, U4 {zn['seed'].size} (+{zn['seed'].size - n0}); per new file "
          + ", ".join(f"{mn['files'][k].replace('-result.h5', '')} {len(names[k][s])}" for k in names), flush=True)
if not res_mode:
    rej = sum(mn["group_rejects"].values()) - sum(mo["group_rejects"].values())   # rejects allowed,
    diff = 0                                                                          # but every one accounted for
    for k in names:
        f = mn["files"][k]
        allnm = names[k]["train"] + names[k]["validation"] + names[k]["test"]
        with h5py.File(raw / f, "r") as h:
            groups = list(h)
            cap = sum(int(h[g].attrs["end_case"]) == 3 for g in groups)
            miss = [int(q) for q in h.attrs["missing_rows"]]
            cov = sorted([int(h[g].attrs["SEED"]) - SEED_BASE for g in groups] + miss) == np.load(GEN / COV[f]).tolist()
            chk(cov, f"{f}: SEED - 4.2e9 plus missing_rows ({len(miss)}) = {COV[f]}")
            stt = [str(h[g].attrs["start"]) if "start" in h[g].attrs else "" for g in groups]
            chk(set(stt) <= {"eq", "mix", "pert", "broad"}, f"{f}: start types {sorted(set(stt))} (design types only)")
        chk(len(set(allnm)) == len(allnm) and set(allnm) <= set(groups), f"{f}: cache entries are distinct groups of the file")
        dk = len(groups) - len(allnm) - cap
        chk(dk >= 0, f"{f}: groups {len(groups)} - entries {len(allnm)} - step cap {cap} = {dk} (rawtraj rejects) >= 0")
        diff += dk
    chk(diff == rej, f"groups not in the cache beyond the step cap, summed over the new files: {diff} = new rejects {rej} "
                     f"({mn['group_rejects']} vs U3 {mo['group_rejects']})")
tag = "CHECK-U4-RES" if res_mode else "CHECK-U4"
print(f"{tag} OK" if ok else f"{tag} FAIL")
