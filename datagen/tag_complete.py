"""Write <tag>/COMPLETE once every job of the tag reached its runtime (lhbench.vref_index's test), with the sha256 of
jobs.json and the job count; benches of a tag wait for it (vref_index silently drops unfinished jobs).
    python tag_complete.py <tag_dir>"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

t = Path(sys.argv[1])
J = json.loads((t / "jobs.json").read_text())
ok = sum((t / f"{i}.npz").exists() and float(np.load(t / f"{i}.npz")["t"][-1]) >= float(j["cfg"]["runtime"]) * (1 - 1e-9)
         for i, j in enumerate(J))
print(f"{t}: {ok} of {len(J)} jobs finished")
if ok == len(J):
    (t / "COMPLETE").write_text(json.dumps({"jobs": len(J), "jobs_json_sha256": hashlib.sha256((t / "jobs.json").read_bytes()).hexdigest()}) + "\n")
    print("COMPLETE written")
