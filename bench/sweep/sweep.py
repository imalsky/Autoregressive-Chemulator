"""Step-size sweep runner (2026-10-19; author: "no dt gap"). One GPU run per model evaluates every dt in --dts at once: each
row = (dt, start age, tag job) keeps its own fixed dt for N calls; all rows are batched in one rollout. Arm H machinery of
lhbench.py in --lib (default: this repository's lib/; <run>/harness/lib = a run's frozen copy; imported read-only): start
state = the VULCAN tag's own log10 y at the start age (lhbench.vref_logy), g = data.norm_globals(job P, T), truth = the same tag. The per-row dt is
passed to lhbench.run_arm as a dtsched callable (a constant dt per row), so the npz holds tcum (elapsed time
per row at each recorded call). Inference policy = the P* bench flags, set exactly as lhbench.main sets its globals for
  --modes P3 --closed-iters 32 --closed-fallback lm --lm-adaptive   (no --startup-substeps, no --substeps, no --renorm)
with LHB_BLOCK=50, ARCH_CACHE=/scratch/imalsky/arch_gcm17m_res, HARNESS_NORM=<that cache>/norm.json, TF32 off.
Recorded calls: lhbench.ckpt_calls(N, 1) (33 log-spaced calls + 0.9 N + N) plus --lin evenly spaced calls over the second
half (N/2 .. N) plus --every (every call; tests only).
    python sweep.py <run dir> <out dir> <rows.json> --n N [--dts 1,2,5,10,20,50,100,200,500] [--starts 1e-3,1e6]
        [--lin 20] [--every] [--block 50] [--max-rows R] [--timing 200] [--lib DIR]
rows.json = {"tags": [tag dir, ...], "rows": [[tag index, job index], ...]}. Row order: dt-major, then start age, then the
rows.json order. Writes <out>/sweep.npz (P3 states only, [C, R, S]) and <out>/meta.json; refuses to overwrite.
--timing K: no output; prints the wall time of K calls after the graph capture (per-call cost at this batch size)."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("out")
ap.add_argument("rows")
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--dts", default="1,2,5,10,20,50,100,200,500")
ap.add_argument("--starts", default="1e-3,1e6")
ap.add_argument("--ckpt", default="stage2.pt")
ap.add_argument("--closed-iters", type=int, default=32)
ap.add_argument("--lin", type=int, default=20)
ap.add_argument("--every", action="store_true")
ap.add_argument("--times", default="1e3,1e4")            # elapsed times (s) recorded at every dt that reaches them
ap.add_argument("--block", type=int, default=50)
ap.add_argument("--max-rows", type=int, default=None)   # timing: first R rows of the dt-major order
ap.add_argument("--timing", type=int, default=0)
ap.add_argument("--lib", default=str(Path(__file__).resolve().parents[2] / "lib"))   # lhbench.py's folder
a = ap.parse_args()

CACHE = str(Path(os.environ.get("CHEM_SCRATCH") or "/scratch/imalsky") / "arch_gcm17m_res")
os.environ["LHB_BLOCK"] = str(a.block)                   # read by lhbench at import (BLOCK)
os.environ["ARCH_CACHE"] = CACHE                         # read by data at import (CACHE)
os.environ["HARNESS_NORM"] = CACHE + "/norm.json"        # read by data at import (NORM: species set, normalization)
run = Path(a.run)
lib = Path(a.lib)
sys.path.insert(0, str(lib))
import torch  # noqa: E402
import lhbench as LB  # noqa: E402

assert Path(LB.__file__).resolve().parent == lib.resolve(), LB.__file__
assert LB.BLOCK == a.block
D, M = LB.D, LB.M
out = Path(a.out)
if not a.timing:
    assert not (out / "sweep.npz").exists(), f"refusing to overwrite {out}"

# lhbench.main's globals for --lm-adaptive without startup substeps. NM = 1 (P3) matters only for an older frozen copy given
# with --lib, whose lhbench still has the P1 / P2 modes (its other mode globals default to off)
LB.NM, LB.ADAPTIVE, LB.STARTUP = 1, True, None
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
dev = torch.device("cuda")
model, cfg = LB.load_model(str(run), a.ckpt, dev, a.closed_iters)   # --closed-iters 32
nclosed = 0
for m_ in model.modules():                                          # --closed-fallback lm
    if isinstance(m_, M.RelaxClosedRHead):
        m_.fallback = "lm"
        nclosed += 1
assert nclosed >= 1, "no closed head: the P* fallback flags would not apply"

sel = json.loads(Path(a.rows).read_text())
tags = [str(t) for t in sel["tags"]]
refs = [LB.vref_index(t) for t in tags]
base = [(int(ti), int(jb)) for ti, jb in sel["rows"]]
for ti, jb in base:
    assert jb in refs[ti], f"job {jb} of {tags[ti]} is not a finished job"
dts = [float(x) for x in a.dts.split(",")]
starts = [float(x) for x in a.starts.split(",")]
rows = [(dt, s, ti, jb) for dt in dts for s in starts for ti, jb in base]
if a.max_rows:
    rows = rows[:a.max_rows]
R = len(rows)
ly = {}                                                              # (tag, job) -> {start: log10 y}, as run_H
for ti, jb in dict.fromkeys((r[2], r[3]) for r in rows):
    ly[(ti, jb)] = {s: LB.vref_logy(tags[ti], jb, np.array([s]))[0] for s in starts}
mu64, sd64, _, _ = D.norm_consts(dev)
ly0 = np.stack([ly[(ti, jb)][s] for _, s, ti, jb in rows])
z0 = (torch.as_tensor(ly0, dtype=torch.float64, device=dev) - mu64) / sd64
P = np.array([refs[ti][jb]["P"] for _, _, ti, jb in rows])
T = np.array([refs[ti][jb]["T"] for _, _, ti, jb in rows])
g = torch.from_numpy(D.norm_globals(P, T)).to(dev)
dtr = torch.as_tensor(np.array([r[0] for r in rows]), dtype=torch.float64, device=dev)
dtn = D.dt_norm_of(dtr).float()                                      # as run_H: dt_norm_of(fp64 dt).float(), clamped


def sched(e):                                                        # per-row constant dt
    e.dtn3.copy_(dtn)
    e.t += dtr


if a.timing:
    eng = LB.Engine(model, z0, dtn, g, dev, nm=1, dtsched=sched)
    eng.closed_counters(R)
    eng.advance(2 * a.block)                                         # capture + warm
    torch.cuda.synchronize()
    t0 = time.time()
    eng.advance(a.timing)
    torch.cuda.synchronize()
    s = time.time() - t0
    print(f"TIMING rows {R} calls {a.timing}: {s:.2f} s, {1e3 * s / a.timing:.2f} ms/call, {1e6 * s / a.timing / R:.3f} us/cell, "
          f"redo blocks {eng.nredo}", flush=True)
    sys.exit(0)

N = int(a.n)
extra = np.unique(np.round(np.linspace(N // 2, N, a.lin + 1)).astype(np.int64)) if a.lin else np.array([], np.int64)
if a.times:   # common elapsed times: call tau / dt for every dt that reaches tau in N calls (same calls recorded for all rows)
    xt = [round(tau / dt) for tau in (float(x) for x in a.times.split(",")) for dt in dts
          if abs(tau / dt - round(tau / dt)) < 1e-9 and 1 <= round(tau / dt) <= N]
    extra = np.unique(np.concatenate([extra, np.array(xt, np.int64)]))
if a.every:
    extra = np.arange(1, N + 1)
t_start = time.time()
r = LB.run_arm(model, z0, dtn, g, N, 1.0, dev, dtsched=sched, extra_calls=extra)
wall = time.time() - t_start
out.mkdir(parents=True, exist_ok=True)
Z = np.array([refs[ti][jb].get("Z", np.nan) for _, _, ti, jb in rows], np.float64)
CO = np.array([refs[ti][jb].get("C_O", np.nan) for _, _, ti, jb in rows], np.float64)
np.savez(out / "sweep.npz", dtrow=np.array([x[0] for x in rows]), t0=np.array([x[1] for x in rows]),
         tagi=np.array([x[2] for x in rows]), job=np.array([x[3] for x in rows]), tags=np.array(tags), P=P, T=T, Z=Z, CO=CO,
         calls=r["calls"], tcum=r["tcum"], states=r["states"][:, 2], nint=r["nint"], first=r["first"],
         projfail=r.get("projfail", np.zeros(0)), nredo=r.get("nredo", -1), seconds=r["seconds"])
meta = {"run": str(run), "ckpt": a.ckpt, "cfg": cfg, "rows_file": str(Path(a.rows).resolve()), "n_calls": N, "dts": dts,
        "starts": starts, "rows": R, "base_rows": len(base), "block": a.block, "closed_iters": a.closed_iters,
        "closed_fallback": "lm", "lm_adaptive": True, "modes": "P3", "startup_substeps": None, "arch_cache": CACHE,
        "harness_norm": os.environ["HARNESS_NORM"], "seconds_rollout": r["seconds"], "seconds_wall": wall,
        "ms_per_call": 1e3 * r["seconds"] / N, "nredo": int(r.get("nredo", -1)), "recorded_calls": int(r["calls"].size), "lib": str(lib.resolve()),
        "lib_git": subprocess.run(["git", "-C", str(lib), "describe", "--always", "--dirty", "--tags"], capture_output=True,
                                  text=True).stdout.strip() or None,
        "harness_git_head": (run / "harness" / "GIT_HEAD").read_text().strip() if (run / "harness" / "GIT_HEAD").exists() else None}
(out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
pf = r.get("projfail")
print(f"SWEEP {run.name}: rows {R} calls {N} rollout {r['seconds']:.1f} s ({meta['ms_per_call']:.2f} ms/call) wall {wall:.1f} s; "
      f"guard interventions {int(r['nint'].sum())}; Newton failures {int(pf[..., 0].sum()) if pf is not None else -1}; "
      f"kept input {int(pf[..., 1].sum()) if pf is not None else -1}; redo blocks {meta['nredo']}", flush=True)
