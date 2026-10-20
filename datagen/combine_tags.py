"""Combined arm-H tag: one lhbench H run per model over several fixed-condition sets. An H bench costs about the same for
24 rows as for 3072 (measured on a shared GPU: 3866 s vs 5670 s), so the sets run as one batch and bench/split6.py splits
the result back into the per-set files.
Jobs in set order, each set's ids ascending (so every start-major block keeps the per-set row order); <k>.npz are symlinks to
the set's own files; jobs.json entries are the set's own with 'orig' = '<set>:<id>' added. Refuses to overwrite.
    python combine_tags.py <out dir> <set name>=<tag dir> [...]"""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
assert not out.exists(), f"{out} exists"
jobs = []
links = []
for a in sys.argv[2:]:
    name, tag = a.split("=", 1)
    src = json.loads((Path(tag) / "jobs.json").read_text())
    for i, j in enumerate(src):
        jobs.append(dict(j, orig=f"{name}:{i}"))
        links.append(Path(tag) / f"{i}.npz")
    print(f"{name}: {len(src)} jobs from {tag}")
out.mkdir(parents=True)
for k, f in enumerate(links):
    if f.exists():
        (out / f"{k}.npz").symlink_to(f)
(out / "jobs.json").write_text(json.dumps(jobs))
print(f"wrote {out}: {len(jobs)} jobs, {sum(f.exists() for f in links)} with output")
