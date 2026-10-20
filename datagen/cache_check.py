"""Arm M: the cache holds what the solver wrote. For every cached row of a gen_shift set (all splits): logy row 0 =
log10 max(x0, eps) of the 12 species (x0 = the box's normalized start), rows 1-99 = log10 max(grid rows 1-99, eps);
end and converged-tail classes counted.   python cache_check.py <cache_dir> <set> <rtol>"""
import json
import sys
from pathlib import Path

import numpy as np

cache, s, rtol = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
HERE = paths.GEN_SHIFT   # gen_shift sets (2026-10-09: was this script's folder)
sys.path.insert(0, str(paths.DATAGEN))
NORM = json.loads(paths.NORM12.read_text())
EPS = float(NORM["epsilon"])
SHORT = [v.replace("_evolution", "") for v in NORM["species_variables"]]
import gcm  # noqa: E402
sp = gcm.species()
cols = [sp.index(q) for q in SHORT]
SEED0 = 4_260_000_000
files = json.loads((cache / "meta.json").read_text())["files"]
d0max = d1max = 0.0
n = 0
ends, convs = {}, {"converged tail": 0, "no tail": 0}
for split in ("train", "validation", "test"):
    z = dict(np.load(cache / f"raw_{split}.npz"))   # load each array once (NpzFile re-reads on every access)
    for j in range(z["seed"].size):
        if not files[int(z["file_idx"][j])].startswith("run_gcm_m"):
            continue
        k = int(z["seed"][j]) - SEED0
        bf = HERE / s / f"boxes_{rtol}" / f"{k}.npz"
        if not bf.exists():  # an arm-M row of another set (another raw file in the same cache)
            continue
        b = np.load(bf)
        x0 = b["x0"] / b["x0"].sum()
        d0max = max(d0max, float(np.max(np.abs(z["logy"][j, 0] - np.log10(np.maximum(x0[cols], EPS))))))
        d1max = max(d1max, float(np.max(np.abs(z["logy"][j, 1:] - np.log10(np.maximum(b["grid"][1:, cols], EPS))))))
        ends[int(z["end"][j])] = ends.get(int(z["end"][j]), 0) + 1
        convs["converged tail" if z["conv"][j] >= 1 else "no tail"] += 1
        n += 1
print(f"{n} cached M rows: max |row 0 - log10 x0| {d0max:.1e}; max |rows 1-99 - log10 grid| {d1max:.1e}; end row counts "
      f"{dict(sorted(ends.items()))}; {convs}")
assert d0max < 1e-12 and d1max < 1e-12
print("CACHE = SOLVER OUTPUT: OK")
