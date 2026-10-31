"""Model runner: one model on one design (bench/score_design.py, or a stored yardstick-format design with its batch
file), GPU, CUDA graphs. Writes the states at the design's saved calls; truth is never read.

    python score_run.py <run dir> <design dir> <out dir> [--ckpt stage2.pt] [--batches b,b,...] [--groups FILE] [--cont]

Environment (run/env.sh, then): LHB_BLOCK=50 and HARNESS_NORM = the model's normalization (its run folder's norm.json);
lhbench.load_model refuses a run whose manifest names another normalization. --groups: the batch file of a
yardstick-format design (its run folder's groups.npz). --batches runs a subset (parts that exist are skipped); when every
batch's part exists the parts are merged.

Inference policy (fixed): the P3 state (fp64 z, z' = z + fp64(delta(z fp32, dt, T/P))), 32 Newton iterations of the closed
head's element projection, the LM rescue where Newton fails run adaptively (lhbench.ADAPTIVE: a CUDA-graph block in which a
Newton solve failed is undone and rerun with the rescue), LHB_BLOCK 50, TF32 off, one model call per step, no startup
substeps; --cont adds the continuation fallback after the LM rescue (models.py, the head's cont), recorded in
meta.json as "cont". The rows of a batch run together for the batch's largest N; a row's counters are read at its own N and later
calls are neither recorded nor counted. Fixed-dt rows advance in graph blocks between saved calls; path rows advance one
call at a time with T, P, dt and the recorded mixing set in-graph before each call, as lhbench.run_paths does.

Starts: the design's 53-species start units mapped to the model's entries with data.take and the 1e-30 floor exactly as
lhbench.vref_logy (one unit at a time, [1, 53]) for constant-condition designs and as lhbench.run_paths (per path, [53])
for path designs; asserted bitwise against those functions on up to 200 units read from the design's source.

Per row (as the counters of bench/score.py expect): newton_n, kept_n (calls whose Newton projection failed / that kept
their input after the rescue), guard_n (candidate rejected by the guard: non-finite or outside log10 y [-40, 0.05]),
nonfin_n (saved calls with a non-finite entry among all entries), and the first such call of each (1-based, -1 none);
first_fail = the first of first_newton, first_guard, first_nonfin.
Outputs: <out>/parts/<batch>.npz, then states.npz (logy12 [S, 12] at every saved call, the row fields, per-batch seconds)
(model_states.npz for a yardstick-format design) and meta.json (design identity, model and normalization sha256, code
version, policy, GPU seconds, start checks)."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "lib"
GASES = ["C2H2", "CH4", "CO2", "CO", "H2O", "H2", "HCN", "H", "N2", "NH3", "OH", "O"]
ROWK = ["newton_n", "kept_n", "guard_n", "first_newton", "first_kept", "first_guard", "nonfin_n", "first_nonfin", "first_fail"]
POLICY = ("P3 state, 32 Newton iterations, adaptive LM rescue, LHB_BLOCK 50, TF32 off, one model call per step, no "
          "startup substeps")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def code_state():
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    return {"git_head": git("rev-parse", "HEAD"), "lib_bench_modified": git("status", "--porcelain", "--", "lib", "bench")}


# ------------------------------------------------------------------------------------------------ the design
class Design:
    """What the runner needs of a design: rows, saved calls, start units, batches (and forcing for paths)."""

    def __init__(self, ddir, groups=None):
        d = Path(ddir)
        self.dir = d
        if (d / "manifest.json").exists():
            man = json.loads((d / "manifest.json").read_text())
            for f in ("rows.npz", "saved.npz", "starts.npz", "groups.npz", "forcing.npz"):
                if f in man["files"]:
                    assert sha256(d / f) == man["files"][f], f"{d / f}: sha256 differs from the manifest"
            self.legacy, self.man, self.sha = False, man, sha256(d / "manifest.json")
            self.mode, self.name = man["mode"], man["name"]
            self.src = Path(man["source"]["dir"])
            assert man["start_map"] == ("run_paths" if self.mode == "path" else "vref_logy")
            r = np.load(d / "rows.npz")
            self.N, self.dt, self.unit, self.off = r["N"].astype(np.int64), r["dt"], r["unit"].astype(np.int64), r["off"]
            self.job = r["job"].astype(np.int64)
            s = np.load(d / "saved.npz")
            self.k, self.srow, self.tabs = s["k"].astype(np.int64), s["row"].astype(np.int64), s["tabs"]
            st = np.load(d / "starts.npz")
            self.sp53, self.x53 = [str(v) for v in st["species53"]], st["x53"]
            self.u_t0, self.u_job, self.u_T, self.u_P = st["t0"], st["job"].astype(np.int64), st["T"], st["P_barye"]
            g = np.load(d / "groups.npz")
            self.g_rows, self.g_off, self.g_nrun = g["rows"].astype(np.int64), g["off"], g["nrun"].astype(np.int64)
            self.phase = None
            if self.mode == "path":
                f = np.load(d / "forcing.npz")
                self.F = {k: f[k] for k in f.files}
        else:
            assert groups is not None, "a yardstick-format design needs --groups (its run folder's groups.npz)"
            self.legacy, self.mode, self.name = True, "seg", d.name
            r = np.load(d / "rows.npz")
            self.N, self.dt, self.unit, self.off = r["N"].astype(np.int64), r["dt"], r["unit"].astype(np.int64), r["off"]
            self.r = {k: r[k] for k in ("dtn32", "P_barye", "T", "kind")}
            s = np.load(d / "saved.npz")
            self.k, self.srow = s["k"].astype(np.int64), s["row"].astype(np.int64)
            u = np.load(d / "units.npz")
            self.u = {k: u[k] for k in ("model_ly25", "model_species25", "g32", "tag", "t0")}
            self.meta = json.loads((d / "meta.json").read_text())
            gp = np.load(groups)
            assert str(gp["rows_npz_sha256"]) == sha256(d / "rows.npz"), "the batch file belongs to another design"
            self.g_rows, self.g_off, self.g_nrun = gp["rows"].astype(np.int64), gp["off"], gp["nrun"].astype(np.int64)
            self.phase = [str(p) for p in gp["phase"]]
            self.groups_sha = sha256(groups)
        self.R, self.S = self.N.size, self.k.size
        assert self.off[-1] == self.S and np.array_equal(self.srow, np.repeat(np.arange(self.R), np.diff(self.off)))

    def part_name(self, bi):
        return f"{self.phase[bi]}_b{bi:02d}" if self.legacy else f"b{bi:03d}"


# ------------------------------------------------------------------------------------------------ the code
def load_lib(run, ckpt, cont=False):
    """This repository's lib (read only) with lhbench's globals set for the policy, and the model."""
    assert os.environ.get("LHB_BLOCK") == "50", "LHB_BLOCK=50"
    assert os.environ.get("HARNESS_NORM"), "HARNESS_NORM = the model's normalization file"
    sys.path.insert(0, str(LIB))
    import torch
    import lhbench as LB
    assert Path(LB.__file__).resolve().parent == LIB.resolve(), LB.__file__
    assert LB.BLOCK == 50
    LB.ADAPTIVE, LB.STARTUP = True, None
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = torch.device("cuda")
    model, cfg = LB.load_model(str(run), ckpt, dev, 32)
    n = 0
    for m_ in model.modules():
        if isinstance(m_, LB.M.RelaxClosedRHead):
            m_.fallback = "lm"
            m_.cont = cont
            assert m_.newton == 32
            n += 1
    assert n == 1, n
    for mod in ("data", "models", "elemproj", "paths", "chains"):
        assert Path(sys.modules[mod].__file__).resolve().parent == LIB.resolve(), mod

    class TEngine(LB.Engine):
        """lhbench.Engine plus a 'post' hook called after the guard, inside the captured step."""

        def __init__(self, *a, post=None, **k):
            super().__init__(*a, **k)
            self.post = post

        @torch.no_grad()
        def step(self):
            super().step()
            if self.post is not None:
                self.post(self)
    return torch, LB, model, dev, TEngine


class Tracker:
    """Engine 'post' hook: per row and rowcount column, the first call (1-based) in which the count rose: column 0 =
    Newton projection failed, column 1 = row kept its input. Runs after Engine.step() counted the call."""

    def __init__(self, torch, eng, cnt):
        self.torch, self.cnt = torch, cnt
        self.prev = torch.zeros_like(cnt)
        self.first = torch.full_like(cnt, -1)
        eng.extra += [self.prev, self.first]       # snapshot / restore with graph warmup, capture and LM redo blocks

    def __call__(self, e):
        ch = self.cnt > self.prev
        self.first.copy_(self.torch.where(ch & (self.first < 0), e.ncall, self.first))
        self.prev.copy_(self.cnt)


def rollout(torch, LB, TEngine, model, dev, z0, dtn, g, nrun, J, K, nrow, ncol=12, dtsched=None, per_call=False,
            label="", prog=600.0):
    """One batch. J [P] batch row and K [P] call (1-based) of every point to record, sorted by K; nrow [R] each row's own
    N. per_call: advance one call at a time (path rows, with dtsched). Returns host arrays."""
    R = z0.shape[0]
    holder = {}
    eng = TEngine(model, z0, dtn, g, dev, nm=1, dtsched=dtsched, post=lambda e: holder["t"](e))
    cmods, cnt = eng.closed_counters(R)
    assert len(cnt) == 1 and tuple(cnt[0].shape) == (R, 2) and eng.adaptive
    tr = Tracker(torch, eng, cnt[0])
    holder["t"] = tr
    U, us = np.unique(K, return_index=True)
    ue = np.append(us[1:], K.size)
    assert U[-1] == nrun and np.all(np.diff(K) >= 0)
    endc = {}
    for j in range(R):
        endc.setdefault(int(nrow[j]), []).append(j)
    endc = {c: torch.as_tensor(np.array(v), device=dev) for c, v in endc.items()}
    assert set(endc) <= set(U.tolist())
    Jt = torch.as_tensor(J, device=dev)
    buf = torch.empty(K.size, ncol, dtype=torch.float64, device=dev)
    tbuf = torch.empty(K.size, dtype=torch.float64, device=dev)
    nf_n = torch.zeros(R, dtype=torch.int64, device=dev)
    nf_first = torch.full((R,), -1, dtype=torch.int64, device=dev)
    cntN = torch.full((R, 2), -1, dtype=torch.int64, device=dev)
    nintN = torch.full((R,), -1, dtype=torch.int64, device=dev)
    torch.cuda.synchronize()
    t0 = time.time()
    tlast, done = t0, 0
    for i, c in enumerate(U.tolist()):
        while done < c:                            # same calls as one advance; per_call: one call per advance, as run_paths
            k = 1 if per_call else min(c - done, 20000)
            eng.advance(k)
            done += k
            if time.time() - tlast > prog:
                torch.cuda.synchronize()
                tlast = time.time()
                log(f"{label} call {done}/{nrun} {tlast - t0:.0f}s redo blocks {eng.nredo}")
        l3 = eng.z3 * eng.sd64 + eng.mu64          # = Engine.states()[2]
        jj = Jt[us[i]:ue[i]]
        sel = l3.index_select(0, jj)
        buf[us[i]:ue[i]] = sel[:, :ncol]
        tbuf[us[i]:ue[i]] = eng.t.index_select(0, jj)
        bad = torch.logical_not(torch.isfinite(sel).all(-1))
        nf_n.index_add_(0, jj, bad.long())
        nf_first[jj] = torch.where(bad & (nf_first[jj] < 0), c, nf_first[jj])
        if c in endc:
            rr = endc[c]
            cntN[rr] = cnt[0][rr]
            nintN[rr] = eng.nint[2][rr]
    torch.cuda.synchronize()
    secs = time.time() - t0
    nr = torch.as_tensor(nrow, device=dev)

    def mask(f):
        return torch.where((f > 0) & (f <= nr), f, torch.full_like(f, -1)).cpu().numpy()

    out = dict(vals=buf.cpu().numpy(), newton_n=cntN[:, 0].cpu().numpy(), kept_n=cntN[:, 1].cpu().numpy(),
               guard_n=nintN.cpu().numpy(), first_newton=mask(tr.first[:, 0]), first_kept=mask(tr.first[:, 1]),
               first_guard=mask(eng.first[2]), nonfin_n=nf_n.cpu().numpy(), first_nonfin=mask(nf_first),
               newton_total=cnt[0][:, 0].cpu().numpy(), guard_total=eng.nint[2].cpu().numpy(), seconds=secs,
               nredo=int(eng.nredo), nrun=int(nrun), rows_in_batch=int(R), t_engine=tbuf.cpu().numpy())
    assert (out["newton_n"] >= 0).all() and (out["guard_n"] >= 0).all(), "a row's end call was not recorded"
    for m_ in cmods:
        m_.rowcount = None
    del eng, buf, tr, holder
    torch.cuda.empty_cache()
    return out


def first_fail(o):
    f = np.stack([o["first_newton"], o["first_guard"], o["first_nonfin"]], 0).astype(np.int64)
    f = np.where(f > 0, f, np.iinfo(np.int64).max).min(0)
    return np.where(f == np.iinfo(np.int64).max, -1, f)


class _PathFile:
    """The mixing fields of one path, shaped as the path file chains.mixing_arrays reads."""

    def __init__(self, F, r, species):
        sel = F["mix_row"] == r
        self.files = ["mix_k", "mix_f", "mix_x", "species"] if F["has_mix"][r] else ["species"]
        self.a = {"mix_k": F["mix_k"][sel], "mix_f": F["mix_f"][sel], "mix_x": F["mix_x"][sel], "species": np.array(species)}

    def __getitem__(self, k):
        return self.a[k]


# ------------------------------------------------------------------------------------------------ starts
def start_logy(D, LB):
    """log10 y over the model's entries per start unit (constant designs) or per row (path designs)."""
    Dm = LB.D
    if D.legacy:
        assert list(D.u["model_species25"]) == list(Dm.SHORT), "the stored model starts use another entry set"
        return D.u["model_ly25"]
    if D.mode == "path":
        return np.log10(np.maximum(np.stack([Dm.take(D.x53[D.unit[r]], D.sp53) for r in range(D.R)]), Dm.EPS))
    return np.stack([np.log10(np.maximum(Dm.take(D.x53[u:u + 1], D.sp53), Dm.EPS))[0] for u in range(D.x53.shape[0])])


def check_starts(D, LB, ly, torch, dev):
    """Bitwise checks of the mapped starts (and, for a yardstick-format design, of dtn and g) on a seeded sample."""
    Dm = LB.D
    rng = np.random.default_rng(20261020)
    chk = {}
    if D.legacy:
        tag = D.dir.parent / "panel" / "tag"
        assert sha256(tag / "jobs.json") == D.meta["provenance"]["panel_jobs_sha256"], f"{tag}: not the design's tag"
        dtn = Dm.dt_norm_of(torch.as_tensor(D.dt, dtype=torch.float64, device=dev)).float().cpu().numpy()
        chk["dtn_mismatch_rows"] = int((dtn.view(np.uint32) != D.r["dtn32"].view(np.uint32)).sum())
        g = Dm.norm_globals(D.r["P_barye"], D.r["T"])
        chk["g_mismatch_rows"] = int((g.view(np.uint32) != D.u["g32"][D.unit].view(np.uint32)).any(1).sum())
        su = np.sort(rng.choice(D.u["t0"].size, 200, replace=False))
        ref = np.stack([LB.vref_logy(str(tag), int(D.u["tag"][k]), np.array([D.u["t0"][k]]))[0] for k in su])
        chk["start_sample"] = int(su.size)
        chk["start_bitwise_mismatch"] = int((ref.view(np.uint64) != ly[su].view(np.uint64)).any(1).sum())
        assert chk["dtn_mismatch_rows"] == 0 and chk["g_mismatch_rows"] == 0 and chk["start_bitwise_mismatch"] == 0, chk
        return chk
    if D.mode == "path":
        su = np.sort(rng.choice(D.R, min(200, D.R), replace=False))
        ref = []
        for r in su:
            p = np.load(D.src / f"{D.job[r]}.npz")
            x0 = p["x"][0] if "x" in p.files else p["x0"]
            ref.append(Dm.take(x0, [str(s) for s in p["species"]]))
        ref = np.log10(np.maximum(np.stack(ref), Dm.EPS))
        mine = ly[su]
    else:
        su = np.sort(rng.choice(D.x53.shape[0], min(200, D.x53.shape[0]), replace=False))
        ref = np.stack([LB.vref_logy(str(D.src), int(D.u_job[u]), np.array([D.u_t0[u]]))[0] for u in su])
        mine = ly[su]
    chk["start_sample"] = int(su.size)
    chk["start_bitwise_mismatch"] = int((ref.view(np.uint64) != mine.view(np.uint64)).any(1).sum())
    assert chk["start_bitwise_mismatch"] == 0, chk
    return chk


# ------------------------------------------------------------------------------------------------ batches
def run_batch(torch, LB, TEngine, model, dev, D, bi, ly, g32):
    Dm = LB.D
    mu64, sd64, _, _ = Dm.norm_consts(dev)
    rows = D.g_rows[D.g_off[bi]:D.g_off[bi + 1]]
    nrun = int(D.g_nrun[bi])
    R = rows.size
    cnt_ = D.off[rows + 1] - D.off[rows]
    pts = np.concatenate([np.arange(D.off[x], D.off[x + 1]) for x in rows])
    J = np.repeat(np.arange(R), cnt_)
    K = D.k[pts]
    o_ = np.argsort(K, kind="stable")
    pts, J, K = pts[o_], J[o_], K[o_]
    assert np.array_equal(D.srow[pts], rows[J])
    nrow = D.N[rows].astype(np.int64)
    label = f"{D.name} b{bi}"
    if D.mode != "path":
        units = D.unit[rows]
        z0 = (torch.as_tensor(ly[units], dtype=torch.float64, device=dev) - mu64) / sd64
        g = torch.from_numpy(g32[units]).to(dev)
        dtn = Dm.dt_norm_of(torch.as_tensor(D.dt[rows], dtype=torch.float64, device=dev)).float()
        return rows, pts, rollout(torch, LB, TEngine, model, dev, z0, dtn, g, nrun, J, K, nrow, label=label)
    # path rows, as lhbench.run_paths: per-call T, P, dt and the recorded mixes, in-graph
    F = D.F
    n = F["T"].shape[1]
    assert nrun == n and (nrow == n).all()
    z0 = (torch.as_tensor(ly[rows], dtype=torch.float64, device=dev) - mu64) / sd64
    sched_g = torch.from_numpy(np.stack([Dm.norm_globals(F["Pbar"][r][:n] * 1e6, F["T"][r][:n]) for r in rows], 1)).to(dev)
    dmat = torch.as_tensor(np.stack([np.asarray(F["dt"][r], np.float64)[:n] for r in rows], 1), device=dev)   # [n, R]
    dtn_mat = Dm.dt_norm_of(dmat).float()
    dtn = dtn_mat[0].contiguous()

    def psched(e):                   # call ncall uses T[k], P[k] and dt[k]
        k = torch.clamp(e.ncall, max=n - 1).reshape(1)
        e.g3.copy_(torch.index_select(sched_g, 0, k).squeeze(0).repeat(e.nm, 1))
        e.dtn3.copy_(torch.index_select(dtn_mat, 0, k).squeeze(0).repeat(e.nm))
        e.t += torch.index_select(dmat, 0, k).squeeze(0)
    sched = psched
    if F["has_mix"][rows].any():     # per-call f [n, R] and partner y [n, R, S] on the device
        fy = [LB.C.mixing_arrays(_PathFile(F, r, D.sp53), n) for r in rows]
        fmat = torch.as_tensor(np.stack([a for a, _ in fy], 1), device=dev)
        ymat = torch.as_tensor(np.stack([b for _, b in fy], 1), device=dev)

        def sched(e):                # the call's globals and dt, then the recorded mix of call ncall (in-graph)
            psched(e)
            k = torch.clamp(e.ncall, max=n - 1).reshape(1)
            f = torch.index_select(fmat, 0, k).squeeze(0)
            yp = torch.index_select(ymat, 0, k).squeeze(0)
            e.z3.copy_(LB.C.mix(e.z3, f, yp, e.mu64, e.sd64))
    return rows, pts, rollout(torch, LB, TEngine, model, dev, z0, dtn, sched_g[0], nrun, J, K, nrow, dtsched=sched,
                              per_call=True, label=label)


def merge(D, out, run, ckpt, meta_extra):
    parts = [out / "parts" / f"{D.part_name(i)}.npz" for i in range(len(D.g_nrun))]
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        log(f"not merged: {len(missing)} parts missing ({missing[:5]})")
        return False
    logy = np.full((D.S, 12), np.nan)
    fill = np.zeros(D.S, bool)
    ra = {k: np.full(D.R, -2, np.int64) for k in ROWK}
    batch = np.full(D.R, -1, np.int32)
    secs = np.zeros(len(parts))
    nredo = np.zeros(len(parts), np.int64)
    tdiff = 0.0
    for i, p in enumerate(parts):
        z = np.load(p)
        assert not fill[z["pts"]].any()
        logy[z["pts"]] = z["logy12"]
        fill[z["pts"]] = True
        for k in ROWK:
            ra[k][z["rows"]] = z[k]
        batch[z["rows"]] = i
        secs[i], nredo[i] = float(z["seconds"]), int(z["nredo"])
        if "t_engine" in z.files:
            tdiff = max(tdiff, float(np.max(np.abs(z["t_engine"] - D.tabs[z["pts"]]))))
    assert fill.all() and (batch >= 0).all() and all((v >= -1).all() for v in ra.values())
    fields = dict(logy12=logy, gases12=np.array(GASES), batch=batch, batch_seconds=secs, batch_nredo=nredo,
                  batch_nrun=D.g_nrun, **ra)
    meta = {"model_run": str(run), "ckpt": ckpt, "stage2_sha256": sha256(Path(run) / ckpt),
            "harness_norm": os.environ["HARNESS_NORM"], "harness_norm_sha256": sha256(os.environ["HARNESS_NORM"]),
            "design": str(D.dir), "policy": POLICY, "code": code_state(), "gpu_seconds": float(secs.sum()),
            "rows": int(D.R), "points": int(D.S), **meta_extra}
    if D.legacy:
        meta.update(rows_npz_sha256=sha256(D.dir / "rows.npz"), saved_npz_sha256=sha256(D.dir / "saved.npz"),
                    groups_sha256=D.groups_sha, batch_phase=D.phase)
        np.savez(out / "model_states.npz", batch_phase=np.array(D.phase), **fields)
    else:
        meta.update(design_sha256=D.sha, design_name=D.name, mode=D.mode)
        if D.mode == "path":
            meta["path_time_max_abs_diff_s"] = tdiff         # engine time at the saved calls vs the design's
        np.savez(out / "states.npz", **fields)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"{D.name} merged: {D.S} points, rows {D.R}, GPU s {secs.sum():.1f}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("design", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ckpt", default="stage2.pt")
    ap.add_argument("--batches", default=None)
    ap.add_argument("--groups", type=Path, default=None)
    ap.add_argument("--cont", action="store_true")
    a = ap.parse_args()
    D = Design(a.design, a.groups)
    torch, LB, model, dev, TEngine = load_lib(a.run, a.ckpt, a.cont)
    out = a.out
    (out / "parts").mkdir(parents=True, exist_ok=True)
    for f in ("states.npz", "model_states.npz"):
        assert not (out / f).exists(), f"refusing to overwrite {out / f}"
    ly = start_logy(D, LB)
    chk = check_starts(D, LB, ly, torch, dev)
    chk["stage2_sha256"] = sha256(a.run / a.ckpt)
    chk["code"] = code_state()
    (out / "checks.json").write_text(json.dumps(chk, indent=1))
    log(f"{D.name}: start checks {({k: v for k, v in chk.items() if k != 'code'})}")
    if D.legacy:
        g32 = D.u["g32"]
    elif D.mode != "path":
        g32 = LB.D.norm_globals(D.u_P, D.u_T)
    else:
        g32 = None
    todo = range(len(D.g_nrun)) if a.batches is None else [int(b) for b in a.batches.split(",")]
    tot = 0.0
    for bi in todo:
        pf = out / "parts" / f"{D.part_name(bi)}.npz"
        if pf.exists():
            log(f"{D.name} batch {bi} done earlier, skipped")
            continue
        rows, pts, o = run_batch(torch, LB, TEngine, model, dev, D, bi, ly, g32)
        tot += o["seconds"]
        ff = first_fail(o)
        tmp = out / "parts" / f"{D.part_name(bi)}.tmp.npz"
        te = o.pop("t_engine")
        extra = {"t_engine": te} if D.mode == "path" else {}
        np.savez(tmp, rows=rows.astype(np.int32), pts=pts.astype(np.int64), logy12=o.pop("vals"), first_fail=ff, **o, **extra)
        tmp.rename(pf)
        log(f"{D.name} batch {bi} done: {o['seconds']:.1f}s ({1e3 * o['seconds'] / o['nrun']:.2f} ms/call), rows {rows.size}, "
            f"redo blocks {o['nredo']}, rows with a Newton failure {int((o['newton_n'] > 0).sum())}, kept input "
            f"{int((o['kept_n'] > 0).sum())}, guard {int((o['guard_n'] > 0).sum())}, non-finite {int((o['nonfin_n'] > 0).sum())}")
    log(f"{D.name}: rollout seconds this process {tot:.1f}")
    merge(D, out, a.run, a.ckpt, {"checks": chk, "cont": a.cont})


if __name__ == "__main__":
    main()
