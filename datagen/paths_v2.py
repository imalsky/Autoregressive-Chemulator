"""Sprint 2026-10-08, paths v2 (plan sections 5 and 7): GCM-like reference paths with T, P AND dt changing every call.
As E13_gcm/paths.py (element budget Z 0.1-300x, C/O 0.1-2, N/O 0.3-3x solar; T(t) = Tm + dT sin(2 pi t / tau + phi),
log10 P(t) = log10 Pm + dlogP sin(2 pi t / tau + phi + pi/2), tau = 1e5 s, dT 100-600 K, dlogP 0-1 dex, clipped to
300-3000 K and 1e-6 to 1e3 bar; start = FastChem equilibrium at (T(0), P(0)) projected onto the budget), except: mean T
300-2900 K; per-call dt log-uniform 10-500 s (per-path generator); T and P of call k are taken at the elapsed time t_k
before the call. Two sets in separate folders (lhbench arm P truncates to the shortest path): short (512 paths x 200
calls, Sobol seed 20261015) and long (64 paths x 1000 calls, seed 20261016); added 2026-10-08 (methods audit: v2l
lasts ~1.5 days): xlong (16 paths x 10,000 calls at a fixed dt of 500 s = 5e6 s = 50 forcing periods, seed 20261017). Truth: one VULCAN box per call
(vulcan_ref.box, runtime = dt_k, all 53 species, rtol PATHS_RTOL or 0.05), the end state carried to the next call.
Output E13_gcm/paths_v2s or paths_v2l (never E13_gcm/paths): <p>.npz with t, x [n + 1, 53], species, T, Pbar, dt [n].
    python paths_v2.py jobs <short|long|xlong>           (<dir>/paths.json; refuses to overwrite)
    python paths_v2.py drive <short|long|xlong> <cpu,...> (one path per CPU at a time; skips finished paths)
    python paths_v2.py run <short|long|xlong> <p>        (internal)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
E = paths.E
DATAGEN = str(paths.DATAGEN)
sys.path.insert(0, DATAGEN)
import generate as G  # noqa: E402
import gcm  # noqa: E402

SETS = {"short": (E / "paths_v2s", 9, 200, 20261015), "long": (E / "paths_v2l", 6, 1000, 20261016),
        "xlong": (E / "paths_v2x", 4, 10000, 20261017),
        # 2026-10-09 truth-tolerance check: the first 3000 calls of xlong (same seed, so the same T, P, dt and start),
        # driven with PATHS_RTOL 0.01 and 0.005 respectively
        "xlong_r01": (E / "paths_v2x_r01", 4, 3000, 20261017), "xlong_r005": (E / "paths_v2x_r005", 4, 3000, 20261017)}
DTFIX = {"xlong": 500.0, "xlong_r01": 500.0, "xlong_r005": 500.0}  # fixed dt per call (s); other sets draw dt log-uniform in 10-500 s
TAU = 1e5
VREF = str(paths.LIB / "vulcan_ref.py")
RTOL = os.environ.get("PATHS_RTOL", "0.05")


def jobs(s):
    from scipy.stats import qmc
    out_dir, m, ncall, seed = SETS[s]
    out_dir.mkdir(exist_ok=True)
    assert not (out_dir / "paths.json").exists()
    u = qmc.Sobol(d=10, scramble=True, seed=seed).random_base2(m=m)
    out = []
    for i, r in enumerate(u):
        Z = 10.0 ** (-1.0 + np.log10(3000.0) * r[0])
        co = 10.0 ** (-1.0 + np.log10(20.0) * r[1])
        fno = 10.0 ** (np.log10(0.3) + r[2])
        oh = Z * 10.0 ** (gcm.SUN["O"] - 12.0)
        el = {"C": co * oh, "N": fno * Z * 10.0 ** (gcm.SUN["N"] - 12.0), "O": oh, "He": 10.0 ** (gcm.SUN["He"] - 12.0)}
        Tm, lPm = 300.0 + 2600.0 * r[3], -4.0 + 6.0 * r[4]
        dT, dlP, phi = 100.0 + 500.0 * r[5], r[6], 2.0 * np.pi * r[7]
        dt = np.full(ncall, DTFIX[s]) if s in DTFIX else 10.0 ** np.random.default_rng([seed, i]).uniform(1.0, np.log10(500.0), ncall)
        t = np.concatenate([[0.0], np.cumsum(dt)[:-1]])          # elapsed time before call k
        T = np.clip(Tm + dT * np.sin(2 * np.pi * t / TAU + phi), 300.0, 3000.0)
        Pbar = 10.0 ** np.clip(lPm + dlP * np.sin(2 * np.pi * t / TAU + phi + np.pi / 2), -6.0, 3.0)
        out.append({"path": i, "Z": Z, "C_O": co, "f_NO": fno, "el": el, "dt": dt.tolist(), "T": T.tolist(),
                    "Pbar": Pbar.tolist()})
    (out_dir / "paths.json").write_text(json.dumps(out))
    Tm_ = np.array([np.mean(j["T"]) for j in out])
    print(f"{s}: {len(out)} paths x {ncall} calls; mean dt {np.mean([np.mean(j['dt']) for j in out]):.1f} s; path mean T "
          f"per band {np.histogram(Tm_, [300, 800, 1300, 1800, 2300, 3001])[0].tolist()}")


def run(s, p):
    out_dir, m, ncall, seed = SETS[s]
    if (out_dir / f"{p}.npz").exists():
        return
    J = json.loads((out_dir / "paths.json").read_text())[p]
    sp = gcm.species()
    w = out_dir / f"w{p}"
    w.mkdir(exist_ok=True)
    gcm.W = w  # FastChem work files
    x = gcm.start(3 * 10 ** 6 + {"short": 0, "long": 10 ** 5, "xlong": 2 * 10 ** 5, "xlong_r01": 2 * 10 ** 5, "xlong_r005": 2 * 10 ** 5}[s] + p, {"el": J["el"], "src": [(J["T"][0], J["Pbar"][0])],
                                                                      "w": [1.0], "kind": "eq"}, sp)
    xs, ts, tend = [np.array([x[q] for q in sp])], [0.0], []
    for k in range(ncall):
        dt = J["dt"][k]
        job = {"T": J["T"][k], "P": J["Pbar"][k] * 1e6, "seed": 0, "lo": -25.0, "mix": x,
               "cfg": {"runtime": repr(float(dt)), "trun_min": repr(float(dt)), **({"rtol": RTOL} if RTOL != "0.05" else {})}}
        (w / "jobs.json").write_text(json.dumps([job]))
        (w / "0.npz").unlink(missing_ok=True)
        subprocess.run([gcm.PY, VREF, "box", str(w / "jobs.json"), "0", str(w)], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=1800, env={**os.environ, **G.THREAD_ENV, "MPLBACKEND": "Agg",
                                                                     "MPLCONFIGDIR": str(w / "_mpl")})
        r = np.load(w / "0.npz")
        assert list(r["species"]) == sp
        xe = np.array([np.interp(dt, r["t"], r["x"][:, j]) for j in range(len(sp))])  # held past VULCAN's last step
        tend.append(float(r["t"][-1]))  # 2026-10-08: VULCAN's last time per call, to check that the call reached dt
        xe = np.maximum(xe, G.FLOOR)
        xe /= xe.sum()
        x = dict(zip(sp, xe.tolist()))
        xs.append(xe)
        ts.append(ts[-1] + dt)
    tmp = out_dir / f"{p}.tmp.npz"
    np.savez(tmp, t=np.array(ts), x=np.array(xs), species=np.array(sp), T=np.array(J["T"]), Pbar=np.array(J["Pbar"]),
             dt=np.array(J["dt"]), rtol=float(RTOL), tend=np.array(tend))
    os.replace(tmp, out_dir / f"{p}.npz")


def drive(s, cpus):
    out_dir = SETS[s][0]
    todo = [p for p in range(len(json.loads((out_dir / "paths.json").read_text()))) if not (out_dir / f"{p}.npz").exists()]
    free, running, t0 = list(cpus), {}, time.time()
    (out_dir / "logs").mkdir(exist_ok=True)
    print(f"drive {s}: {len(todo)} paths on {len(cpus)} CPUs, rtol {RTOL}", flush=True)
    with open(out_dir / "status.jsonl", "a") as st:
        while todo or running:
            while todo and free:
                p, c = todo.pop(0), free.pop(0)
                log = open(out_dir / "logs" / f"{p}.log", "w")
                q = subprocess.Popen(["taskset", "-c", c, gcm.PY, str(Path(__file__).resolve()), "run", s, str(p)],
                                     stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **G.THREAD_ENV})
                running[q] = (p, c, time.time(), log)
            time.sleep(1.0)
            for q in [z for z in running if z.poll() is not None]:
                p, c, ts, log = running.pop(q)
                log.close()
                free.append(c)
                st.write(json.dumps({"p": p, "rc": q.returncode, "wall_s": round(time.time() - ts, 1),
                                     "npz": (out_dir / f"{p}.npz").exists()}) + "\n")
                st.flush()
    print(f"drive {s} done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "jobs":
        jobs(a[1])
    elif a[0] == "drive":
        drive(a[1], a[2].split(","))
    else:
        run(a[1], int(a[2]))
