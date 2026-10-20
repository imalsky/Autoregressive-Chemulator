"""Rollout benchmark of a trained model (GPU, CUDA graphs, TF32 off).

    python lhbench.py <run dir> <out dir> SP --arms H|P [--ckpt stage2.pt] [--h-tag DIR ...] [--p-dir DIR] [options]

Every row is advanced in the P3 precision mode: the host state is the fp64 normalized log10 state z, and each call does
z' = z + fp64(delta(z fp32, dt, T/P)). A candidate state that is non-finite or outside log10 y in [-40, 0.05] is rejected
(the row keeps its previous state) and counted per row (nint, first; slot 2 of the [3, rows] arrays, slots 0 and 1 are
unused). States are recorded at log-spaced calls (ckpt_calls) plus any requested calls, as [calls, 3, rows, entries]
log10 mixing ratios with slots 0 and 1 NaN. Scoring is offline (bench/).

  arm H   fixed conditions: rows = the finished jobs of a VULCAN tag (--h-tag: a folder holding jobs.json and <i>.npz, or a
          name under paths.VREF) x start ages (--h-starts, start-major); start state = the tag's own state at that age,
          P and T of the job, fixed dt (--h-dt), --h-calls calls (default min(the level's cap at that dt, 1000 days / dt));
          --h-times adds the calls at these elapsed times (multiples of --h-dt). Truth = the same tag.
  arm P   changing conditions: every path of a paths folder (--p-dir; <i>.npz with x, species, T, Pbar, dt and optionally
          mix_k / mix_f / mix_x) from its VULCAN start; T, P (and a per-call dt) change every call, the recorded mixes
          are applied before their calls (chains.mix); every call is recorded.

Inference options (the decision benches use --closed-iters 32 --closed-fallback lm --lm-adaptive, and on arm H
--startup-substeps 5:1):
  --closed-iters n        Newton iterations of the closed head's element projection (default: the checkpoint's)
  --closed-fallback lm    rows whose Newton projection fails are re-solved by Levenberg-Marquardt (elemproj.project_lm);
                          per-row counters 'projfail' [modules, rows, 2]: Newton failed / row kept its input
  --lm-adaptive           each CUDA-graph block runs with fallback 'keep'; a block in which a Newton solve failed is undone
                          and rerun with 'lm' (equal to 'lm' throughout, faster)
  --startup-substeps n:k  the first k calls of every row are made as n calls of dt / n (fixed-dt arm H)
Never change the rows per call of a bench (its set): results differ by up to 7e-4 dex across batch shapes, so bitwise
comparisons hold only at equal batch shape. LHB_BLOCK sets the calls per CUDA graph (default 250).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import data as D  # noqa: E402
import models as M  # noqa: E402
import chains as C  # noqa: E402
M.S, M.SPECIES = len(D.SHORT), list(D.SHORT)  # the entry set follows the normalization file

ENV_LO, ENV_HI = -40.0, 0.05            # physical envelope (log10 mixing ratio)
HORIZON_S = 8.64e7                      # 1000 model days
LEVELS = {"SP": {1.0: 100_000, 10.0: 1_000_000, 100.0: 864_000}}   # default arm-H calls: min(cap at dt (else at 10 s), HORIZON_S / dt)
BLOCK = int(os.environ.get("LHB_BLOCK", "250"))  # calls per CUDA graph (lower it when a large bench runs out of memory)
VREF = paths.VREF


def vref_index(tag):
    """{job index: job} for jobs whose output exists and reached the requested runtime (end_case 1 or 2 at
    t >= runtime; with trun_min = runtime the convergence stop cannot fire earlier). tag may be an absolute
    directory (VREF / absolute path is that path)."""
    jobs = json.loads((VREF / tag / "jobs.json").read_text())
    out, drop = {}, []
    for i, j in enumerate(jobs):
        f = VREF / tag / f"{i}.npz"
        if f.exists() and float(np.load(f)["t"][-1]) >= float(j["cfg"]["runtime"]) * (1 - 1e-9):
            out[i] = j
        else:
            drop.append(i)
    if drop:
        print(f"vref_index {tag}: {len(drop)} of {len(jobs)} jobs missing or short of their runtime, not used; first ids "
              f"{drop[:10]}", file=sys.stderr, flush=True)
    return out


def vref_logy(tag, i, times):
    """log10 of the entry set's mixing ratios (data.take of the tag's species) at times (s), linear in t between VULCAN
    steps, floored at data.EPS."""
    r = np.load(VREF / tag / f"{i}.npz")
    sp, t, x = list(r["species"]), r["t"], r["x"]  # read each array once (r["x"] decompresses on every access)
    x = D.take(np.stack([np.interp(times, t, x[:, k]) for k in range(len(sp))], -1), sp)
    return np.log10(np.maximum(x, D.EPS))


def load_model(spec, ckpt, dev, newton=None):
    """The EMA weights of <run dir>/<ckpt> in eval mode, and the checkpoint's config. newton overrides the closed head's
    Newton iterations. A run with a manifest.json (train.write_manifest) is refused unless the normalization file in use
    (data.NORM_PATH) is byte-identical to the one it was trained with."""
    run = Path(spec)
    man = run / "manifest.json"
    if man.exists():
        want = json.loads(man.read_text())["norm"]["sha256"]
        got = hashlib.sha256(D.NORM_PATH.read_bytes()).hexdigest()
        if got != want:
            sys.exit(f"{run}: the normalization in use, {D.NORM_PATH} (sha256 {got[:16]}), is not the one this model was "
                     f"trained with ({want[:16]}, saved as {run / 'norm.json'}); set HARNESS_NORM to that file")
    blob = torch.load(run / ckpt, map_location=dev)
    cfg = blob["cfg"]
    _, _, zlo, zhi = D.norm_consts(dev)
    m = M.build(cfg["head"], cfg["hidden"], zlo, zhi, newton=cfg.get("closed_iters", 12) if newton is None else newton,
                fallback=cfg.get("closed_fallback", "keep")).to(dev)
    m.load_state_dict(blob["state_dict"])
    return m.eval(), cfg


class Engine:
    """Static-shape P3 rollout of R rows, CUDA-graph captured in blocks. nm must be 1 (P3; kept as an argument for the
    callers' signature). dtsched: callable(engine) run before every call (sets dtn3 / g3, advances t)."""

    def __init__(self, model, z0, dtn, g, dev, nm=1, dtsched=None):
        assert nm == 1, "P3 only"
        self.model, self.dev, self.nm = model, dev, nm
        mu, sd, _, _ = D.norm_consts(dev)
        self.mu64, self.sd64 = mu, sd
        R = z0.shape[0]
        self.R = R
        self.z3 = z0.clone()                                           # P3 state (fp64 z)
        self.dtn3 = dtn.repeat(nm).contiguous()
        self.g3 = g.repeat(nm, 1).contiguous()
        self.zlo_env = ((ENV_LO - mu) / sd)
        self.zhi_env = ((ENV_HI - mu) / sd)
        self.nint = torch.zeros(3, R, dtype=torch.int64, device=dev)   # guard interventions per row (slot 2 = P3)
        self.first = torch.full((3, R), -1, dtype=torch.int64, device=dev)
        self.ncall = torch.zeros((), dtype=torch.int64, device=dev)
        self.dtsched = dtsched
        self.t = torch.zeros(R, dtype=torch.float64, device=dev)       # per-row elapsed time (dtsched only)
        self.extra = [self.t, self.dtn3]
        self.graphs = {}
        self.adaptive, self.counters, self.nredo = [], [], 0           # --lm-adaptive (closed_counters sets them)

    def closed_counters(self, B):
        """Per-row projection-failure counters [B, 2] for every closed head, kept in self.extra so the graph warmup /
        capture side effects are undone. Returns the modules and the counters."""
        mods = [m for m in self.model.modules() if isinstance(m, M.RelaxClosedRHead)]
        cnt = []
        for m in mods:
            m.rowcount = torch.zeros(B, 2, dtype=torch.int64, device=self.dev)
            cnt.append(m.rowcount)
        self.extra += cnt
        self.counters = cnt
        if ADAPTIVE:
            self.adaptive = [m for m in mods if m.fallback == "lm"]
        return mods, cnt

    def _ok_z(self, z):
        return torch.isfinite(z).all(-1) & (z >= self.zlo_env).all(-1) & (z <= self.zhi_env).all(-1)

    def _guard(self, mode, ok, new, old):
        bad = ~ok
        self.nint[mode] += bad.long()
        self.first[mode] = torch.where(bad & (self.first[mode] < 0), self.ncall + 1, self.first[mode])
        return torch.where(ok.unsqueeze(-1), new, old)

    @torch.no_grad()
    def step(self):
        if self.dtsched is not None:
            self.dtsched(self)
        d3 = self.model.delta(self.z3.float(), self.dtn3, self.g3)
        z3n = self.z3 + d3.double()
        self.z3.copy_(self._guard(2, self._ok_z(z3n), z3n, self.z3))
        self.ncall += 1

    def _graph(self, n, mode=None):
        """mode (--lm-adaptive): capture with the lm heads' fallback set to mode ('keep' | 'lm')."""
        if (n, mode) in self.graphs:
            return self.graphs[(n, mode)]
        if mode is not None:
            for m_ in self.adaptive:
                m_.fallback = mode
        keep = (self.z3, self.nint, self.first, self.ncall, self.g3, *self.extra)
        snap = [t.clone() for t in keep]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                self.step()
        for t, v in zip(keep, snap):
            t.copy_(v)  # undo the warmup/capture side effects
        if mode is not None:
            for m_ in self.adaptive:
                m_.fallback = "lm"
        self.graphs[(n, mode)] = g
        return g

    def advance(self, n):
        for blk in (BLOCK, 10, 1):
            k, n = divmod(n, blk)
            if k and not self.adaptive:
                g = self._graph(blk)
                for _ in range(k):
                    g.replay()
            elif k:  # --lm-adaptive: each block runs with fallback keep; a block in which any Newton solve failed is
                # undone and replayed with fallback lm. Equal to running lm throughout (lm changes only the failed rows'
                # result, and those blocks are redone); one host sync per block.
                gk, gl = self._graph(blk, "keep"), self._graph(1, "lm")  # lm redo call by call: a small graph pool
                st = (self.z3, self.nint, self.first, self.ncall, self.g3, *self.extra)
                for _ in range(k):
                    snap = [t.clone() for t in st]
                    n0 = sum(c[:, 0].sum() for c in self.counters)
                    gk.replay()
                    if int(sum(c[:, 0].sum() for c in self.counters) - n0) > 0:
                        for t, v in zip(st, snap):
                            t.copy_(v)
                        for _ in range(blk):
                            gl.replay()
                        self.nredo += 1

    def states(self):
        """[3, R, S] fp64 log10 mixing ratio; slots 0 and 1 NaN, slot 2 the P3 state."""
        l3 = self.z3 * self.sd64 + self.mu64
        nan = torch.full_like(l3, float("nan"))
        return torch.stack([nan, nan, l3], 0)


def ckpt_calls(n_total, dt):
    """Calls at which states are recorded: log-spaced physical times plus 0.9 n_total (terminal drift)."""
    taus = np.unique(np.concatenate([np.logspace(0, np.log10(max(1.0, n_total * dt)), 33), [0.9 * n_total * dt]]))
    calls = np.unique(np.clip(np.round(taus / dt).astype(np.int64), 1, n_total))
    return calls


ADAPTIVE = False  # --lm-adaptive: fallback lm only in the graph blocks where a Newton solve failed
STARTUP = None    # --startup-substeps n:k: the first k calls of a run are made as n model calls of dt / n


def run_arm(model, z0, dtn, g, n_total, dt, dev, dtsched=None, extra_calls=None):
    """Roll the rows z0 (fp64 z) for n_total calls of dt (dtn: dt_norm per row; g: globals per row) and record the states
    at ckpt_calls(n_total, dt) plus extra_calls. Returns the npz fields."""
    eng = Engine(model, z0, dtn, g, dev, dtsched=dtsched)
    cmods, cnt = eng.closed_counters(z0.shape[0])
    calls = ckpt_calls(n_total, dt)
    if extra_calls is not None:
        calls = np.unique(np.concatenate([calls, np.asarray(extra_calls, dtype=np.int64)]))
    rec, rec_t, done, t0 = [], [], 0, time.time()
    st_left = 0
    if STARTUP is not None and dtsched is None:  # fixed-dt arms only (dtsched arms set dtn3 every call)
        st_n, st_left = STARTUP
        assert float(dt) / st_n >= 10.0 ** D.NORM["dt"]["log_min"] * (1 - 1e-9), "startup substep below the trained dt floor"
        dtn_full = eng.dtn3.clone()
        dtn_sub = D.dt_norm_of(torch.full(dtn_full.shape, float(dt) / st_n, dtype=torch.float64, device=dev)).float()
    for c in calls:
        todo = int(c - done)
        if st_left:
            k = min(st_left, todo)
            eng.dtn3.copy_(dtn_sub)
            eng.advance(k * st_n)
            eng.dtn3.copy_(dtn_full)
            st_left, todo = st_left - k, todo - k
        eng.advance(todo)
        done = int(c)
        rec.append(eng.states().cpu().numpy())
        if dtsched is not None:
            rec_t.append(eng.t.cpu().numpy())
    torch.cuda.synchronize()
    out = {"calls": calls, "states": np.stack(rec), "nint": eng.nint.cpu().numpy(), "first": eng.first.cpu().numpy(),
           "seconds": time.time() - t0}
    if cnt:  # [modules, rows, 2] projection failures: Newton failed / row kept its input
        out["projfail"] = np.stack([c.cpu().numpy() for c in cnt])
        for m_ in cmods:
            m_.rowcount = None
        pf = out["projfail"]
        out["nredo"] = eng.nredo
        print(f"projection failures (calls x rows, summed over {len(cnt)} closed modules): Newton {int(pf[..., 0].sum())} on "
              f"{int((pf[..., 0] > 0).any(0).sum())} rows; kept input {int(pf[..., 1].sum())} on {int((pf[..., 1] > 0).any(0).sum())} rows",
              flush=True)
    if dtsched is not None:
        out["tcum"] = np.stack(rec_t)                                  # [C, R] elapsed simulated time (s)
    return out


def run_H(model, tag, starts, dt, n, dev, extra_calls=None):
    """Arm H: rows = the tag's finished jobs (sorted) x starts (start-major), start state = the tag's own log10 y at
    each start, g = norm_globals(job P, T), fixed dt, n calls. Returns the npz fields."""
    mu64, sd64, _, _ = D.norm_consts(dev)
    ref = vref_index(tag)
    keys = sorted(ref)
    job = np.array([k for _ in starts for k in keys])
    t0 = np.array([s for s in starts for _ in keys], dtype=np.float64)
    ly0 = np.concatenate([np.stack([vref_logy(tag, i, np.array([s]))[0] for i in keys]) for s in starts])
    z0 = (torch.as_tensor(ly0, dtype=torch.float64, device=dev) - mu64) / sd64
    P = np.array([ref[i]["P"] for i in job])
    T = np.array([ref[i]["T"] for i in job])
    g = torch.from_numpy(D.norm_globals(P, T)).to(dev)
    dtn = D.dt_norm_of(torch.full((job.size,), dt, dtype=torch.float64, device=dev)).float()
    r = run_arm(model, z0, dtn, g, n, dt, dev, extra_calls=extra_calls)
    return dict(job=job, t0=t0, dt=dt, tag=str(tag), P=P, T=T, ridx=np.array([ref[i].get("row", -1) for i in job]), **r)


def run_paths(model, p_dir, dev):
    """Arm P: every path of a paths folder from its VULCAN start, T, P (and dt) changing every call, every call recorded
    (also train.py's path readout). Returns the run_arm result, the path files, their arrays, the number of calls, the dt
    (per path, or [R, n]) and the number of recorded mixes applied (None for paths without mix_k)."""
    files = sorted(Path(p_dir).glob("[0-9]*.npz"), key=lambda f: int(f.stem))
    Ps = [np.load(f) for f in files]
    n = min(len(p["T"]) for p in Ps)
    mu64, sd64, _, _ = D.norm_consts(dev)
    ly0 = np.log10(np.maximum(np.stack([D.take(p["x"][0], list(p["species"])) for p in Ps]), D.EPS))
    z0 = (torch.as_tensor(ly0, dtype=torch.float64, device=dev) - mu64) / sd64
    sched_g = torch.from_numpy(np.stack([D.norm_globals(p["Pbar"][:n] * 1e6, p["T"][:n]) for p in Ps], 1)).to(dev)
    if any(np.ndim(p["dt"]) > 0 for p in Ps):  # per-call dt [n] per path
        dmat = torch.as_tensor(np.stack([np.asarray(p["dt"], np.float64)[:n] for p in Ps], 1), device=dev)  # [n, R]
        dtn_mat = D.dt_norm_of(dmat).float()
        dtn = dtn_mat[0].contiguous()
        dts = dmat.T.cpu().numpy()

        def psched(e):                   # call ncall uses T[k], P[k] and dt[k]
            k = torch.clamp(e.ncall, max=n - 1).reshape(1)
            e.g3.copy_(torch.index_select(sched_g, 0, k).squeeze(0).repeat(e.nm, 1))
            e.dtn3.copy_(torch.index_select(dtn_mat, 0, k).squeeze(0).repeat(e.nm))
            e.t += torch.index_select(dmat, 0, k).squeeze(0)
    else:
        dts = np.array([float(p["dt"]) for p in Ps])
        dtn = D.dt_norm_of(torch.as_tensor(dts, dtype=torch.float64, device=dev)).float()

        dts_t = torch.as_tensor(dts, dtype=torch.float64, device=dev)

        def psched(e):                       # in-graph: the globals of call ncall (path call k uses T[k], P[k]);
            k = torch.clamp(e.ncall, max=n - 1).reshape(1)  # index_select with a device index: no host sync
            e.g3.copy_(torch.index_select(sched_g, 0, k).squeeze(0).repeat(e.nm, 1))
            e.t += dts_t

    nmix = None
    if any("mix_k" in p.files for p in Ps):  # per-call f [n, R] and partner y [n, R, S] on the device
        fy = [C.mixing_arrays(p, n) for p in Ps]
        fmat = torch.as_tensor(np.stack([a for a, _ in fy], 1), device=dev)
        ymat = torch.as_tensor(np.stack([b for _, b in fy], 1), device=dev)
        nmix = int(sum((a > 0).sum() for a, _ in fy))
        base = psched

        def psched(e):                       # the call's globals and dt, then the recorded mix of call ncall (in-graph)
            base(e)
            k = torch.clamp(e.ncall, max=n - 1).reshape(1)
            f = torch.index_select(fmat, 0, k).squeeze(0)
            yp = torch.index_select(ymat, 0, k).squeeze(0)
            e.z3.copy_(C.mix(e.z3, f, yp, e.mu64, e.sd64))
    r = run_arm(model, z0, dtn, sched_g[0], n, 1.0, dev, dtsched=psched, extra_calls=np.arange(1, n + 1))
    return r, files, Ps, n, dts, nmix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("level", choices=sorted(LEVELS))
    ap.add_argument("--ckpt", default="stage2.pt")
    ap.add_argument("--arms", required=True)                  # H, P or H,P
    ap.add_argument("--modes", default="P3", choices=["P3"])
    ap.add_argument("--h-tag", default=None)
    ap.add_argument("--h-starts", default="1e-3,1e3,1e6")
    ap.add_argument("--h-dt", type=float, default=10.0)
    ap.add_argument("--h-calls", type=int, default=None)
    ap.add_argument("--h-times", default=None)                # .npy of elapsed times (s, multiples of --h-dt) also recorded
    ap.add_argument("--p-dir", default=None)
    ap.add_argument("--closed-iters", type=int, default=None)
    ap.add_argument("--closed-fallback", default=None, choices=["keep", "lm"])
    ap.add_argument("--lm-adaptive", action="store_true")    # needs --closed-fallback lm
    ap.add_argument("--startup-substeps", default=None)       # n:k
    a = ap.parse_args()
    global STARTUP, ADAPTIVE
    arms = a.arms.split(",")
    assert set(arms) <= {"H", "P"}, f"arms H and P only: {a.arms}"
    assert "H" not in arms or a.h_tag, "arm H needs --h-tag"
    assert "P" not in arms or a.p_dir, "arm P needs --p-dir"
    ADAPTIVE = bool(a.lm_adaptive)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(a.model, a.ckpt, dev, a.closed_iters)
    meta = {"model": a.model, "ckpt": a.ckpt, "cfg": cfg, "level": a.level, "modes": a.modes, "notes": []}
    if a.closed_iters is not None:
        meta["closed_iters"] = a.closed_iters
    if a.startup_substeps:
        STARTUP = tuple(int(v) for v in a.startup_substeps.split(":"))
        meta["startup_substeps"] = a.startup_substeps
    if a.closed_fallback is not None:
        for m_ in model.modules():
            if isinstance(m_, M.RelaxClosedRHead):
                m_.fallback = a.closed_fallback
        meta["closed_fallback"] = a.closed_fallback

    if "P" in arms:
        r, files, Ps, n, dts, nmix = run_paths(model, a.p_dir, dev)
        np.savez(out / "P_paths.npz", path=np.array([int(f.stem) for f in files]), dt=dts,
                 T=np.stack([p["T"][:n] for p in Ps]), Pbar=np.stack([p["Pbar"][:n] for p in Ps]), p_dir=str(a.p_dir), **r)
        meta["notes"].append(f"P_paths: {len(Ps)} paths from {a.p_dir}, {n} calls, every call recorded")
        if nmix is not None:
            meta["mixing"] = {"applied": True, "n_mix": nmix, "rule": "before call k: y <- (1 - f) y + f y_partner (linear, "
                              "entry set, floored at eps; chains.mix)"}
        print(f"P n={n} rows={len(Ps)} {r['seconds']:.0f}s interventions={r['nint'].sum(1).tolist()}", flush=True)

    if "H" in arms:
        dt = float(a.h_dt)
        cap = LEVELS[a.level]
        n = a.h_calls if a.h_calls is not None else int(min(cap.get(dt, cap[10.0]), round(HORIZON_S / dt)))
        starts = [float(x) for x in a.h_starts.split(",")]
        xc = None
        if a.h_times is not None:            # record these elapsed times too (cross-dt comparisons at equal times)
            tt = np.load(a.h_times).astype(np.float64)
            tt = tt[tt <= n * dt]
            xc = np.round(tt / dt).astype(np.int64)
            assert np.allclose(xc * dt, tt, rtol=0, atol=1e-6) and (xc >= 1).all(), "--h-times must be multiples of --h-dt"
        res = run_H(model, a.h_tag, starts, dt, n, dev, extra_calls=xc)
        name = f"H_{Path(a.h_tag).name}_dt{dt:g}"
        np.savez(out / f"{name}.npz", **res)
        meta["notes"].append(f"{name}: tag {a.h_tag}, {res['job'].size // len(starts)} jobs x starts {starts}, dt {dt:g}, n {n}")
        print(f"{name} n={n} rows={res['job'].size} {res['seconds']:.0f}s interventions={res['nint'].sum(1).tolist()}", flush=True)

    (out / "meta.json").write_text(json.dumps(meta, default=str))


if __name__ == "__main__":
    main()
