"""VULCAN-0D reference runs for the benchmark (CPU; datagen's .venv; datagen is imported, not modified).

    python vulcan_ref.py box <jobs.json> <i> <out_dir>    one box in a fresh process (VULCAN binds its cfg at import)
    python vulcan_ref.py drive <jobs.json> <out_dir> <cpu,cpu,...>   one serial worker process per CPU, no multiprocessing

A job is {"T": K, "P": dyn/cm2, "seed": int, "lo": float} (the recovered sampler start) or {"T", "P", "mix": {species: x}}
(a full 53-species start), plus "cfg": {vulcan_cfg overrides}. Output per box: <i>.npz with t (s), x [steps, 53]
(y_time / n0, all species), species, end_case, count, seconds.
"""
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

DATAGEN = str(paths.DATAGEN)
sys.path.insert(0, DATAGEN)
import generate as G  # noqa: E402  (constants, sampler, chem_funs location; nothing is run at import)

PY = DATAGEN + "/.venv/bin/python"


def box(job):
    over = {**G.CFG_OVERRIDES, **job.get("cfg", {}), "T_box": repr(float(job["T"])), "p_box": repr(float(job["P"]))}
    src = open(os.path.join(G.VULCAN_DIR, "vulcan_cfg.py")).read()
    for k, v in over.items():
        src, n = re.subn(r"^%s\s*=.*$" % re.escape(k), f"{k} = {v}", src, count=1, flags=re.M)
        if n != 1:
            raise KeyError(f"{k} not found in vulcan_cfg.py")
    cfg = types.ModuleType("vulcan_cfg")
    exec(compile(src, os.path.join(G.VULCAN_DIR, "vulcan_cfg.py"), "exec"), cfg.__dict__)
    sys.modules["vulcan_cfg"] = cfg
    os.chdir(G.VULCAN_DIR)
    sys.path[:0] = [G.CHEM_FUNS_DIR, G.VULCAN_DIR]
    import chem_funs
    species = chem_funs.spec_list
    if "mix" in job:
        mix = {sp: G.FLOOR for sp in species}
        mix.update(job["mix"])
    else:
        mix = {sp: G.FLOOR for sp in species}
        mix.update(G.sample_weights(int(job["seed"]), float(job["lo"])))
    tot = sum(mix.values())
    cfg.const_mix = {sp: v / tot for sp, v in mix.items()}
    import store, build_atm, op
    var, atm, para = store.Variables(), store.AtmData(), store.Parameters()
    para.start_time = time.time()
    make_atm = build_atm.Atm()
    output = op.Output()
    atm = make_atm.load_TPK(atm, output)
    make_atm.sp_sat(atm)
    rate = op.ReadRate()
    var = rate.read_rate(var, atm)
    var = rate.rev_rate(var, atm)
    var = rate.remove_rate(var)
    ini = build_atm.InitialAbun()
    var = ini.ini_y(var, atm)
    var = ini.ele_sum(var)
    solver = getattr(op, cfg.ode_solver)()
    integ = op.Integration(solver, output)
    solver.naming_solver(para)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        integ(var, atm, para, make_atm)
    t = np.concatenate([[0.0], np.array(var.t_time, dtype=np.float64)])
    y0 = np.array([cfg.const_mix[s] for s in species])[None]
    x = np.concatenate([y0, np.array(var.y_time)[:, 0, :] / float(atm.n_0)])
    return {"t": t, "x": x, "species": np.array(species), "end_case": para.end_case, "count": para.count,
            "seconds": time.time() - t0}


def main():
    mode = sys.argv[1]
    if mode == "box":
        jobs = json.loads(Path(sys.argv[2]).read_text())
        i, out = int(sys.argv[3]), Path(sys.argv[4])
        r = box(jobs[i])
        np.savez_compressed(out / f"{i}.npz", **r)
        return
    jobs_path, out, cpus = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(",")
    out.mkdir(parents=True, exist_ok=True)
    n = len(json.loads(jobs_path.read_text()))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(out / "_mpl")}
    # one serial worker per CPU: worker w runs boxes w, w + W, ... each in a fresh process
    workers = []
    for w, c in enumerate(cpus):
        todo = [i for i in range(w, n, len(cpus)) if not (out / f"{i}.npz").exists()]
        cmd = " ; ".join(f"taskset -c {c} {PY} {__file__} box {jobs_path} {i} {out} >> {out}/w{w}.log 2>&1" for i in todo)
        if cmd:
            workers.append(subprocess.Popen(["bash", "-c", cmd], env=env))
    t0 = time.time()
    for p in workers:
        p.wait()
    done = sum((out / f"{i}.npz").exists() for i in range(n))
    print(f"{done}/{n} boxes in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
