"""Rows of the step-size sweep: a seeded subsample of the combined development tag devall6 (G, Gc, H2, H2c, H3,
H3c; the opened fixed-condition sets) plus the six fig4 raw-gas cases. Categories (rules_t5 / late judge definitions):
scope = inside the GCM envelope (select_jobs.env_margin: env_any +- 100 K, +- 0.1 dex) AND Z 1-100x AND C/O <= 1 (the
target when the stored rows_main.json was drawn; score_sweep.py's scope is wider);
inside-other = inside, outside that composition; outside = not inside. Within each category jobs are drawn uniformly at
random (seed 20261019) from the dev sets, so set, T and P mix follow the sets' own design; the cooler stratum (300 <= T <
1300 K, 1e-5 <= P <= 100 bar) is not quota'd and is reported as a subset.
    python select_rows.py <n scope> <n inside-other> <n outside> > rows.json summary on stderr"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bench/
from select_jobs import env_margin  # noqa: E402
import paths  # noqa: E402

DEV = str(paths.DEVALL6)
FIG4 = str(paths.FIG4 / "tag")
ns, ni, no = (int(x) for x in sys.argv[1:4])
jobs = json.load(open(DEV + "/jobs.json"))
T = np.array([j["T"] for j in jobs])
P = np.array([j["P"] for j in jobs]) / 1e6
Z = np.array([j["Z"] for j in jobs])
CO = np.array([j["C_O"] for j in jobs])
inside, _, _ = env_margin()
ins = inside(T, P)
sc = ins & (Z >= 1) & (Z <= 100) & (CO <= 1)
cool = (T >= 300) & (T < 1300) & (P >= 1e-5) & (P <= 100)
rng = np.random.default_rng(20261019)
pick = np.concatenate([rng.choice(np.flatnonzero(m), n, replace=False) for m, n in
                       ((sc, ns), (ins & np.logical_not(sc), ni), (np.logical_not(ins), no))])
pick = np.sort(pick)
sets = np.array([jobs[i]["orig"].split(":")[0] for i in pick])
print(f"dev jobs {pick.size}: scope {int(sc[pick].sum())}, inside {int(ins[pick].sum())}, cooler {int(cool[pick].sum())} "
      f"(cooler & inside {int((cool & ins)[pick].sum())}); per set " + ", ".join(f"{s} {int((sets == s).sum())}" for s in dict.fromkeys(sets))
      + "; plus 6 fig4 cases", file=sys.stderr)
print(json.dumps({"tags": [DEV, FIG4], "rows": [[0, int(i)] for i in pick] + [[1, k] for k in range(6)]}))
