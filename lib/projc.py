"""CPU fp64 driver of projc.c (fixes phase 2026-10-12): the closed head's element-ratio projection as one structured
Levenberg-Marquardt solve per cell (species grouped by C, N, O atom counts, He exact, scalar step cap, a cell stops once
max |F| < stop), OpenMP across cells. Measured (fixes-phase checks/solver_speed/ctest_stop.txt, 100,000 realistic relaxed
states of B6 s2, 48 iterations, stop 1e-13): 0 failures, <= 8.5e-10 dex from elemproj.project_ln where that converges,
1.15 us/cell on 1 core, 0.19-0.29 on 8, 0.14-0.17 on 16.
    P = projc.Projector(na, nh, mol)          # torch or numpy stoichiometry as in models.RelaxClosedRHead
    w, res = P(lt, lr, threads=8)             # lt [B, S] relaxed ln y, lr [B, 4] -> normalised ln y', residual [B]
The shared library is compiled on first use into lib/_build/ (gcc -O3 -march=native -fopenmp)."""
import ctypes
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SO = HERE / "_build" / "projc.so"
_dp, _ip = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)


def _lib():
    src = HERE / "projc.c"
    if not SO.exists() or SO.stat().st_mtime < src.stat().st_mtime:
        SO.parent.mkdir(exist_ok=True)
        subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-fPIC", "-shared", str(src), "-o", str(SO), "-lm"], check=True)
    lib = ctypes.CDLL(str(SO))
    lib.project.argtypes = [ctypes.c_int] * 3 + [_dp, _dp, _ip, _dp, _dp, _dp, _ip, ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_double, ctypes.c_double, _dp, _dp]
    return lib


class Projector:
    def __init__(self, na, nh, mol, iters=48, cap=4.0, stop=1e-13):
        na = np.asarray(getattr(na, "cpu", lambda: na)(), np.float64)
        self.nh = np.ascontiguousarray(np.asarray(getattr(nh, "cpu", lambda: nh)(), np.float64))
        self.S = na.shape[0]
        self.he = int(np.flatnonzero(na[:, 3] > 0)[0])
        keys = sorted({tuple(int(v) for v in na[i, :3]) for i in range(self.S) if i != self.he})
        self.K = len(keys)
        self.grp = np.array([-1 if i == self.he else keys.index(tuple(int(v) for v in na[i, :3])) for i in range(self.S)], np.int32)
        self.P = np.ascontiguousarray(np.array(keys, np.float64))
        self.mol = np.zeros(self.S, np.int32)
        self.mol[np.asarray(getattr(mol, "cpu", lambda: mol)())] = 1
        self.na = np.ascontiguousarray(na)
        self.iters, self.cap, self.stop = iters, cap, stop
        self.lib, self.gomp = _lib(), ctypes.CDLL("libgomp.so.1")

    def __call__(self, lt, lr, threads=1):
        lt = np.ascontiguousarray(np.asarray(lt, np.float64))
        lr = np.ascontiguousarray(np.asarray(lr, np.float64))
        n = lt.shape[0]
        out, res = np.empty((n, self.S)), np.empty(n)
        self.gomp.omp_set_num_threads(int(threads))
        c = lambda a, t: a.ctypes.data_as(t)
        self.lib.project(n, self.S, self.K, c(lt, _dp), c(lr, _dp), c(self.grp, _ip), c(self.P, _dp), c(self.nh, _dp),
                         c(self.na, _dp), c(self.mol, _ip), self.he, self.iters, self.cap, self.stop, c(out, _dp), c(res, _dp))
        return out, res
