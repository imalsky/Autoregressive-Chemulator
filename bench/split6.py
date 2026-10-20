"""Split a combined arm-H bench (datagen/combine_tags.py tag) back into the per-set files lhbench writes, so the boards read them
unchanged (fixes phase 2026-10-12; scale-phase copy 2026-10-13 adds the sealed H4 / H4c, opened once at the phase's end). Rows of set s = the rows whose job's 'orig' is 's:<id>', in combined order (start-major,
ids ascending = the per-set order); job -> the set's own id, tag -> the set's own path; states split on axis 2, nint / first /
projfail on axis 1; calls, dt, seconds, nredo copied (seconds and nredo are the combined run's). The (job, t0) order is
asserted equal to a reference per-set bench of the same set where one exists (REF below). Refuses to overwrite.
    python split6.py <combined bench dir> <label>     -> E13_gcm/H_<prefix>_<label>/<npz> + meta.json"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402

E = paths.E
SETS = {"gtest": ("g500", str(paths.GTEST)), "gtest_cold": ("gc500", str(paths.GTEST_COLD)),
        **{k: (k, str(v)) for k, v in paths.HOLDOUTS.items()}}   # h4, h4c: sealed, opened once on a frozen recipe
REF = "e13s_CBg_s2"   # per-set benches of the current pick (row order only)

src, label = Path(sys.argv[1]), sys.argv[2]
f = next(src.glob("H_*_dt*.npz"))
d = np.load(f)
jobs = json.loads((Path(str(d["tag"])) / "jobs.json").read_text())
orig = [jobs[int(k)]["orig"].split(":") for k in d["job"]]
setof = np.array([o[0] for o in orig])
oid = np.array([int(o[1]) for o in orig])
dt = float(d["dt"])
for s in dict.fromkeys(setof):
    pre, tag = SETS[s]
    m = setof == s
    r = dict(job=oid[m], t0=d["t0"][m], P=d["P"][m], T=d["T"][m], ridx=d["ridx"][m], states=d["states"][:, :, m],
             nint=d["nint"][:, m], first=d["first"][:, m], calls=d["calls"], dt=d["dt"], tag=np.array(tag),
             seconds=d["seconds"])
    if "projfail" in d.files:
        r.update(projfail=d["projfail"][:, m], nredo=d["nredo"])
    name = f"H_{Path(tag).name}_dt{dt:g}.npz"
    ref = E / f"H_{pre}_{REF}" / name
    same = None
    if ref.exists():
        q = np.load(ref)
        same = np.array_equal(q["job"], r["job"]) and np.array_equal(q["t0"], r["t0"])
        assert same, f"{s}: (job, t0) order differs from {ref}"
    out = E / f"H_{pre}_{label}"
    out.mkdir(parents=True)
    np.savez(out / name, **r)
    meta = json.loads((src / "meta.json").read_text())
    meta["split_from"] = str(f)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{s}: {int(m.sum())} rows -> {out / name}; order equal to {ref.name if ref.exists() else '-'}: {same}")
