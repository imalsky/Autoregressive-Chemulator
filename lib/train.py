"""Train the closed relaxation head in two stages on a GPU-resident cache.

    python train.py <run_dir>     (run_dir/config.json over DEFAULTS; writes stage1.pt / stage2.pt, log.csv, model_info.json,
                                   done.json, manifest.json and norm.json)

Stage 1 (s1_*): one-step windows from the cache's training trajectories (data.RawSplit.sample: dt log-uniform in
[dt_min_s, dt_max_s], start anchored at the first stored time with probability anchor_p, optional row-0 starts of the
x0_file rows, optional per-trajectory draw weights traj_weights_file), loss on the increment.
Stage 2 (s2_*, usually branched from a stage-1 file with init_from): four terms whose model calls run in lockstep (fewer,
larger calls; lock_terms): the one-step windows; long-jump pairs (z(t), z(t + D)), D log-uniform in [jump_min_s,
jump_max_s] (weight jump_w); grid chains of grid_K stored log-time intervals from t >= grid_t_min, each made of 1..grid_sub
equal model calls (weight grid_w; grid_end_frac of the chains end at the trajectory's last row); changing-condition chains
from the chain_sets path folders (lib/chains.py): n_pre ~ U{0..pre_max} calls without gradient from the truth, then K calls
with gradient, the recorded mixing before every call (weight chain_w; K and pre_max ramp over chain_K_ramp /
chain_pre_ramp steps). Gradient clipping per row inside the BPTT rollouts (grid_row_clip, chain_row_clip), a global clip
(clip, optionally with the norm in fp64), steps with a non-finite gradient skipped (skip_nonfinite).
Loss: HybridLoss (log10 MAE + lambda_z x z-space MSE) on fp64 targets; the rollout state is carried in fp64, the network
runs in fp32 with TF32 on. AdamW (fused), linear warmup then cosine to lr_min_frac x the peak, EMA weights (s1_ema / s2_ema)
written as the stage file; stage 2 starts from stage 1's EMA weights. Validation: one-step windows of the cache's
validation split (fixed draws), EMA and live model. path_readout: at every validation, the EMA model's arm-P bench rollout
(lhbench.run_paths) on a paths folder, scored in its late window.
ckpt_every / resume: exact continuation from run_dir/ckpt_stage<N>.pt (model, EMA, optimizer, generators, accumulators).
Environment: ARCH_CACHE (the cache), HARNESS_NORM (its normalization file), HARNESS_DEVICE (default cuda).
"""
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import data as D  # noqa: E402
import models as M  # noqa: E402
import chains as C  # noqa: E402
M.S, M.SPECIES = len(D.SHORT), list(D.SHORT)  # the entry set follows the normalization file

DEFAULTS = {
    "head": "relaxclosedr",     # the only head (models.build)
    "hidden": [512] * 6,        # trunk widths
    "seed": 0,                  # model init (torch.manual_seed) and the training generators (1000 seed + stage, + 11, + 12)
    "batch": 2048,              # one-step windows per step (both stages)
    "s1_steps": 150_000, "s1_lr": 3e-4, "s1_warmup": 2000, "s1_ema": 0.9995, "s1_lambda_z": 0.1,
    "s2_steps": 25_000, "s2_lr": 3e-5, "s2_warmup": 1000, "s2_ema": 0.999, "s2_lambda_z": 0.25,
    "s2_K": 1, "s2_skip": 0,    # stage 2's windows: one-step (the only values supported)
    "jump_w": 0.0, "jump_min_s": 0.1, "jump_max_s": 1e7, "jump_batch": 1024,     # long-jump term (stage 2)
    "grid_w": 0.0, "grid_K": 20, "grid_batch": 256, "grid_t_min": 1e3, "grid_sub": 4,   # grid term (stage 2): intervals,
    "grid_end_frac": 0.0,       # chains, first time (s), sub-steps 1 + step % grid_sub; share of chains ending at the last row
    "anchor_p": 0.25,           # share of one-step windows anchored at the first stored time (data.ANCHOR_P)
    "lr_min_frac": 0.01, "wd": 1e-5, "clip": 1.0,
    "val_windows": 65536, "val_every": 10_000, "log_every": 1000,
    "init_from": None,          # a stage file to start stage 2 from (stage 1 is then skipped)
    "compile": True,            # torch.compile the model calls
    "dt_min_s": 0.1, "dt_max_s": 1e3,   # one-step window dt range (s)
    "cache": None,              # the ARCH_CACHE this run must use (asserted; None = not checked)
    "traj_weights_file": None,  # .npy of per-train-row draw weights (every sampler)
    "x0_file": None,            # raw file name(s) (cache meta.json 'files') whose rows may start at row 0 (t = 0)
    "x0_anchor_p": 0.0,         # with x0_file: share of those rows' windows that start at row 0 (data.X0_ANCHOR_P)
    "x0_jump_p": 0.0,           # with x0_file: share of those rows' jump pairs that start at row 0 (data.X0_JUMP_P)
    "ckpt_every": 0,            # every n steps save run/ckpt_stage<N>.pt (atomic); removed once stage<N>.pt is written. 0 = off
    "resume": False,            # skip a stage whose stage<N>.pt exists, continue one from its ckpt_stage<N>.pt
    "val_source": "cache",      # validation windows from the cache's validation split (data.cache_val; the only source)
    "path_readout": None,       # a paths folder: at every validation, the EMA model's arm-P rollout scored (path_readout)
    "chain_w": 0.0,             # chain term (stage 2): chain_batch windows per step from the chain_sets folders (finished
    "chain_sets": [],           # paths at the stage's start; an exact resume needs the same files)
    "chain_batch": 256,
    "chain_K_start": 8, "chain_K": 16, "chain_K_ramp": 0,               # K grows linearly over chain_K_ramp steps (0: chain_K)
    "chain_pre_max_start": 0, "chain_pre_max": 80, "chain_pre_ramp": 0,  # pre_max likewise
    "chain_tail_w": 0.0,        # + chain_tail_w x the mean over the top chain_tail_q fraction of windows of each window's max
    "chain_tail_q": 0.1,        # |dlog10 y| over its K calls and the 12 reported species (logged every 50 steps either way)
    "closed_iters": 12,         # Newton iterations of the element projection
    "closed_fallback": "keep",  # rows whose Newton projection fails: 'keep' (input) | 'lm' (Levenberg-Marquardt re-solve)
    "fused_proj": False,        # the training model's projection as elemproj.project_ln_fused (same mathematics, compiles
                                # into fewer kernels); the EMA copy, validation and the benches use project_ln
    "loss_every": 50,           # steps between the train_loss samples averaged into each log row (a host sync each)
    "lockstep": True,           # stage 2 runs its terms in lockstep (the only implementation)
    "skip_nonfinite": False,    # a step whose total gradient norm is not finite makes no optimizer step and no EMA update;
                                # counted (log rows nonfinite_skip / nonfinite_skips); stops if most of a log interval is skipped
    "grid_row_clip": 0.0,       # c > 0: after every grid-rollout call with gradient, an identity whose backward scales each
    "chain_row_clip": 0.0,      # row's incoming state gradient to norm <= c (RowClip); chain_row_clip likewise for the chain
                                # rollout. c is absolute: re-derive it when the batch, K or weight of the term change. Log:
                                # rowclip_<term> = clipped (row, call) pairs / all
    "clip_norm_fp64": False,    # the global gradient norm in fp64 (fp32 sums of squares overflow above ~1.8e19); log rows add
                                # the pre-clip norm's median (gnorm_p50, per grid sub-step count gnorm_p50_nsub), max and
                                # the steps above 1e3 (gnorm_gt1e3)
}


class HybridLoss(nn.Module):
    """Two-term loss on [B, K, S] z-space predictions (z = (log10 y - log_mean) / log_std): lambda_log10_mae x the mean
    |dlog10 y| (= |dz| log_std; the log mean cancels) + lambda_z_mse x the mean dz^2, in loss_dtype. The loss of the
    12-species model (tag paper-v1, src/trainer.py), without its per-step weights."""

    def __init__(self, log_means, log_stds, *, lambda_log10_mae, lambda_z_mse, loss_dtype=torch.float32):
        super().__init__()
        self.loss_dtype = loss_dtype
        self.register_buffer("log_means", log_means.clone().detach().to(dtype=loss_dtype))
        self.register_buffer("log_stds", log_stds.clone().detach().to(dtype=loss_dtype))
        self.lambda_log10_mae = float(lambda_log10_mae)
        self.lambda_z_mse = float(lambda_z_mse)

    def forward(self, pred_z, true_z):
        if pred_z.shape != true_z.shape:
            raise ValueError(f"Shape mismatch: pred_z={tuple(pred_z.shape)} true_z={tuple(true_z.shape)}")
        if pred_z.ndim != 3:
            raise ValueError(f"Expected [B,K,S], got {tuple(pred_z.shape)}")
        diff_z = (pred_z - true_z).to(self.loss_dtype)
        z_mse = diff_z.square().mean()
        diff_log10 = diff_z.abs().mul(self.log_stds)
        log10_mae = diff_log10.mean()
        loss_total = self.lambda_log10_mae * log10_mae + self.lambda_z_mse * z_mse
        return {"loss_total": loss_total, "loss_log10_mae": log10_mae, "loss_z_mse": z_mse}


class RowClip(torch.autograd.Function):
    """Identity in forward; in backward each row's incoming gradient (norm over the last dimension, fp64 like the state) is
    scaled to norm <= rc[0]; rc[1] (a device counter) counts the clipped rows. rc[0] is read at backward time."""

    @staticmethod
    def forward(ctx, z, rc):
        ctx.rc = rc
        return z.view_as(z)

    @staticmethod
    def backward(ctx, gr):
        s = torch.clamp(ctx.rc[0] / torch.linalg.vector_norm(gr, dim=-1, keepdim=True), max=1.0)   # norm 0: 1
        ctx.rc[1].add_((s < 1.0).sum())
        return gr * s, None


def row_clip(z, rc):
    """RowClip on a rollout state z [B, S] when rc ([c, device counter, rows seen]) is set and z carries gradient."""
    if rc is None or not z.requires_grad:
        return z
    rc[2] += z.shape[0]
    return RowClip.apply(z, rc)


def lr_at(step, total, peak, warmup, min_frac):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def param_groups(model, wd):
    decay, no = [], []
    for n, p in model.named_parameters():
        (no if (p.ndim < 2 or "norm" in n.lower()) else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no, "weight_decay": 0.0}]


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        if getattr(self.shadow, "fused", False):   # the copy only runs eager (validation, path readout): the unfused
            self.shadow.fused = False              # projection, faster there

    @torch.no_grad()
    def update(self, model):
        torch._foreach_lerp_(list(self.shadow.parameters()), list(model.parameters()), 1.0 - self.decay)


@torch.no_grad()
def validate(model, vy, vdt, vg, crit, bs=8192):
    """One-step loss of model and of persistence (no change) on the validation windows vy [N, 2, S]."""
    tot = per = n = 0.0
    for i in range(0, vy.shape[0], bs):
        y, dtn, g = vy[i:i + bs].double(), vdt[i:i + bs], vg[i:i + bs]
        d = model.delta(y[:, 0].float(), dtn, g)
        l = crit((y[:, 0] + d.double()).unsqueeze(1), y[:, 1:2])["loss_total"]
        pl = crit(y[:, 0:1], y[:, 1:2])["loss_total"]
        w = y.shape[0]
        tot += float(l) * w
        per += float(pl) * w
        n += w
    return tot / n, per / n


def finite(*ms):
    """All floating weights finite; checked before every save, so a divergence stops the run without overwriting the last
    good checkpoint."""
    return all(bool(torch.isfinite(v).all()) for m in ms for v in m.state_dict().values() if v.is_floating_point())


def save_atomic(obj, path):
    """torch.save to a temporary file beside path, fsync, then rename (a kill never leaves a partial file)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def code_info(lib_md5):
    """The code a run trained with: its lib folder, every lib file's md5, and the git commit, tags and dirty flag when lib
    is the lib/ of a git repository (None for a copy outside one)."""
    lib = Path(__file__).resolve().parent

    def git(*a):
        r = subprocess.run(["git", "-C", str(lib), *a], capture_output=True, text=True, check=False)
        return r.stdout.strip() if r.returncode == 0 else None
    out = {"lib": str(lib), "lib_md5": lib_md5, "git_commit": None}
    top = git("rev-parse", "--show-toplevel")
    if top is not None and Path(top).resolve() == lib.parent:
        out.update(git_commit=git("rev-parse", "HEAD"), git_tags=(git("tag", "--points-at", "HEAD") or "").split(),
                   git_dirty=bool(git("status", "--porcelain", "--", ".")))
    return out


def write_manifest(run, cfg, code, note=None):
    """run/manifest.json, what a model's checkpoints need at use: the sha256 of every .pt in run; a byte-identical copy of
    the normalization file in use (data.NORM_PATH) as run/norm.json, with its sha256; the entry order; the resolved config
    (DEFAULTS merged with config.json); the parent checkpoint's sha256 (init_from); the code. lhbench.load_model refuses a
    normalization file whose sha256 differs."""
    run = Path(run)
    shutil.copyfile(D.NORM_PATH, run / "norm.json")
    assert sha256(run / "norm.json") == sha256(D.NORM_PATH)
    man = {"checkpoints": {f.name: sha256(f) for f in sorted(run.glob("*.pt"))},
           "norm": {"file": "norm.json", "sha256": sha256(run / "norm.json"), "source": str(D.NORM_PATH)},
           "species": list(D.SHORT), "config": cfg,
           "parent": {"path": str(cfg["init_from"]), "sha256": sha256(cfg["init_from"])} if cfg["init_from"] else None,
           "code": code}
    if note:
        man["note"] = note
    (run / "manifest.json").write_text(json.dumps(man, indent=1))


def readout_setup(p_dir, dev):
    """path_readout: the 12 reported species' VULCAN truth after every call of a paths folder and the log10 number density
    of the scored mask, computed once. Window: calls >= 3000 (calls >= 50 for sets of at most 3000 calls)."""
    import lhbench as L
    rep = [s.replace("_evolution", "") for s in json.loads(paths.NORM12.read_text())["species_variables"]]
    Ps = [np.load(f) for f in sorted(Path(p_dir).glob("[0-9]*.npz"), key=lambda f: int(f.stem))]
    n = min(len(p["T"]) for p in Ps)
    c = np.arange(1, n + 1)                                        # truth after call c is x[c]; call c ran at T, P [c - 1]
    vt = np.stack([np.log10(np.maximum(p["x"][c][:, [list(p["species"]).index(s) for s in rep]], D.EPS)) for p in Ps], 1)
    lnd = np.stack([np.log10(p["Pbar"][c - 1] * 1e6 / (1.380649e-16 * p["T"][c - 1])) for p in Ps], 1)
    return {"L": L, "dir": str(p_dir), "vt": vt, "lnd": lnd[..., None], "cols": [D.SHORT.index(s) for s in rep],
            "late": (c >= (3000 if n > 3000 else 50))[:, None, None], "dev": dev}


def path_readout(model, pr):
    """The model's arm-P rollout on the readout paths (the bench's code and precision: P3 state, TF32 off for the rollout,
    restored after), |dlog10 y| of the 12 reported species on the scored points (truth or prediction >= 1e-2 cm^-3, and
    either >= 1e-12 mixing ratio) in the late window: p50, p90, p99, max (dex), and its seconds."""
    t = time.time()
    tf = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        st = pr["L"].run_paths(model, pr["dir"], pr["dev"])[0]["states"][:, 2][..., pr["cols"]]   # [n, R, 12] log10
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf
    vt, ld = pr["vt"], pr["lnd"]
    ok = (((vt + ld) >= -2.0) | ((st + ld) >= -2.0)) & ((vt >= -12.0) | (st >= -12.0)) & pr["late"]
    e = np.abs(st - vt)[ok]
    q = np.percentile(e, [50, 90, 99]) if e.size else [float("nan")] * 3
    return {"path_p50": float(q[0]), "path_p90": float(q[1]), "path_p99": float(q[2]),
            "path_max": float(e.max()) if e.size else float("nan"), "path_s": time.time() - t}


def _ramp(a, b, n, step):
    return b if n <= 0 else int(a + (b - a) * min(1.0, step / n))


def chain_draw(cs, cfg, step, pre_gen):
    """The chain term's draw: K and n_pre (ramped) and the windows."""
    K = _ramp(cfg["chain_K_start"], cfg["chain_K"], cfg["chain_K_ramp"], step)
    pm = _ramp(cfg["chain_pre_max_start"], cfg["chain_pre_max"], cfg["chain_pre_ramp"], step)
    n_pre = int(torch.randint(0, pm + 1, (1,), generator=pre_gen))
    return K, n_pre, cs.windows(cfg["chain_batch"], n_pre + K)


def chain_loss(crit, cfg, w, n_pre, pred, mu, sd, rep, want_tail):
    """The chain term's loss of the predictions pred [B, K, S] after the K calls with gradient, and its tail: per window
    the max |dlog10 y| over its K calls and the 12 reported species on the scored points (truth or prediction >= 1e-2
    cm^-3, and truth or prediction >= 1e-12 mixing ratio), averaged over the top chain_tail_q fraction of windows (dex);
    with gradient when chain_tail_w > 0, else computed only if want_tail (for the log), else None."""
    true = w["zt"][n_pre:].transpose(0, 1)
    lnd = w["lnd"][n_pre:].T                                             # [B, K]
    cl = crit(pred, true)["loss_total"]
    tail = None
    if cfg["chain_tail_w"] > 0 or want_tail:
        with torch.set_grad_enabled(cfg["chain_tail_w"] > 0):
            yt, yq, ld = true[..., rep] * sd[rep] + mu[rep], pred[..., rep] * sd[rep] + mu[rep], lnd.unsqueeze(-1)
            m = (((yt + ld) >= -2.0) | ((yq + ld) >= -2.0)) & ((yt >= -12.0) | (yq >= -12.0))
            e = ((yq - yt).abs() * m).amax(dim=(1, 2))                   # [B] worst scored point per window
            tail = torch.topk(e, max(1, math.ceil(cfg["chain_tail_q"] * e.numel()))).values.mean()
    return cl, tail


def grid_draw(train, cfg, step, gen):
    """The grid term's draw: chains of grid_K stored intervals, their sub-step count nsub = 1 + step % grid_sub and the
    normalized sub-step lengths (unclamped, like the jump pairs)."""
    Kg = cfg["grid_K"]
    gz, gdt, gg, _ = train.sample_grid(cfg["grid_batch"], Kg, gen, cfg["grid_t_min"], end_frac=cfg["grid_end_frac"])
    nsub = 1 + step % cfg["grid_sub"]                    # equal relax sub-steps per stored interval
    lo_, hi_ = D.NORM["dt"]["log_min"], D.NORM["dt"]["log_max"]
    gdtn = ((torch.log10(gdt / nsub) - lo_) / (hi_ - lo_)).float()
    return gz, gdtn, gg, Kg, nsub


def lock_fns(model, delta_fn, comp, mu, sd):
    """The model calls of lock_terms. delta: the one-step + jump call (delta_fn); gstep: a grid call; cstep: a chain call
    with gradient, or a call with grid and chain rows (grid rows have mix fraction 0); cstep_ng: a chain pre-roll call
    (without gradient; its own compiled function). gstep, cstep and cstep_ng run the casts, the state add and (cstep,
    cstep_ng) the mix inside the compiled call; each has its own dynamo cache."""
    def gstep(zs, dtn, g):
        return zs + model.delta(zs.float(), dtn, g).double()

    def cstep(zs, f, yp, dtn, g):
        zs = C.mix(zs, f, yp, mu, sd)
        return zs + model.delta(zs.float(), dtn, g).double()

    def cstep_ng(zs, f, yp, dtn, g):
        zs = C.mix(zs, f, yp, mu, sd)
        return zs + model.delta(zs.float(), dtn, g).double()
    c = (lambda fn: torch.compile(fn, fullgraph=True)) if comp else (lambda fn: fn)
    return {"delta": delta_fn, "gstep": c(gstep), "cstep": c(cstep), "pre": c(cstep_ng)}


def lock_rollouts(fns, gz, gdtn, gg, nsub, Kg, w, n_pre, K):
    """The grid rollout (Kg stored intervals x nsub calls, all with gradient) and the chain rollout (n_pre calls without
    gradient first, then K calls with gradient, the mix before each) in one batch per call: the K chain calls share the
    grid's last calls. The gradient is that of the separate rollouts. fns["rc_g"] / fns["rc_c"]: row_clip of the grid /
    chain state after each call with gradient (None: off). Returns the grid predictions after each interval [Bg, Kg, S]
    and the chain predictions after each gradient call [Bc, K, S]."""
    rg, rcc = fns["rc_g"], fns["rc_c"]
    zc = w["z0"]
    with torch.no_grad():
        for ci in range(n_pre):
            zc = fns["pre"](zc, w["f"][ci], w["yp"][ci], w["dtn"][ci], w["g"][ci])
    Ng = Kg * nsub
    L = max(Ng, K)
    zg = gz[:, 0]
    Bg = zg.shape[0]
    f0, y0 = w["f"].new_zeros(Bg), w["yp"].new_zeros(Bg, w["yp"].shape[-1])   # grid rows: mix fraction 0 (unchanged)
    gp, cp = [], []
    for c in range(L):
        gi, cj = c - (L - Ng), c - (L - K)                # grid call gi, chain gradient call cj (negative: not yet)
        ci = cj + n_pre                                  # the chain call (window index) of this call
        if gi >= 0 and cj >= 0:
            out = fns["cstep"](torch.cat([zg, zc]), torch.cat([f0, w["f"][ci]]), torch.cat([y0, w["yp"][ci]]),
                               torch.cat([gdtn[:, gi // nsub], w["dtn"][ci]]), torch.cat([gg, w["g"][ci]]))
            zg, zc = row_clip(out[:Bg], rg), row_clip(out[Bg:], rcc)
        elif gi >= 0:
            zg = row_clip(fns["gstep"](zg, gdtn[:, gi // nsub].contiguous(), gg), rg)
        else:
            zc = row_clip(fns["cstep"](zc, w["f"][ci], w["yp"][ci], w["dtn"][ci], w["g"][ci]), rcc)
        if gi >= 0 and (gi + 1) % nsub == 0:
            gp.append(zg)
        if cj >= 0:
            cp.append(zc)
    return torch.stack(gp, 1), torch.stack(cp, 1)


def lock_terms(fns, crit, train, cs, cfg, step, gen, pre_gen, mu, sd, rep, want_tail):
    """Stage 2's terms: every batch drawn first (one-step windows, jump pairs, grid chains on gen; chain windows on the
    store's generator and pre_gen), then the one-step windows and the jump pairs in one model call and the grid and chain
    rollouts in lock_rollouts. Returns the one-step + jump + grid loss (summed in that order), the chain loss and tail,
    n_pre and K."""
    z, dtn, g, _ = train.sample(cfg["batch"], 1, gen)
    z0 = z[:, 0]
    d_true = (z[:, 1] - z0).float()
    j0, j1, jdt, jg = D.sample_jump_pairs(train, cfg["jump_batch"], gen, cfg["jump_min_s"], cfg["jump_max_s"])
    gz, gdtn, gg, Kg, nsub = grid_draw(train, cfg, step, gen)
    K, n_pre, w = chain_draw(cs, cfg, step, pre_gen)
    B1 = z0.shape[0]
    dd = fns["delta"](torch.cat([z0, j0]).float(), torch.cat([dtn, jdt]), torch.cat([g, jg]))
    d_pred, jp = dd[:B1], j0 + dd[B1:].double()
    loss = crit(d_pred.unsqueeze(1), d_true.unsqueeze(1))["loss_total"]
    jl = crit(jp.unsqueeze(1), j1.unsqueeze(1))["loss_total"]
    loss = loss + cfg["jump_w"] * jl
    gp, cpred = lock_rollouts(fns, gz, gdtn, gg, nsub, Kg, w, n_pre, K)
    gl = crit(gp, gz[:, 1:])["loss_total"]
    loss = loss + cfg["grid_w"] * gl
    cl, tail = chain_loss(crit, cfg, w, n_pre, cpred, mu, sd, rep, want_tail)
    return loss, cl, tail, n_pre, K


def run_stage(stage, cfg, model, train, vy, vdt, vg, mu, sd, run, log, pr=None):
    dev = train.device
    s = "s1" if stage == 1 else "s2"
    steps, peak, warm = cfg[f"{s}_steps"], cfg[f"{s}_lr"], cfg[f"{s}_warmup"]
    crit = HybridLoss(sd.new_zeros(1, 1, M.S).float(), sd.float().view(1, 1, -1), lambda_log10_mae=1.0,
                      lambda_z_mse=cfg[f"{s}_lambda_z"]).to(dev)
    opt = torch.optim.AdamW(param_groups(model, cfg["wd"]), lr=peak, betas=(0.9, 0.999), eps=1e-8, fused=True)
    ema = EMA(model, cfg[f"{s}_ema"])
    delta_fn = torch.compile(model.delta, fullgraph=True) if cfg["compile"] else model.delta
    gen = torch.Generator(device=dev).manual_seed(1000 * cfg["seed"] + stage)
    t0, acc, nacc = time.time(), 0.0, 0
    lock = stage == 2
    if lock:  # own generators (windows on the device, n_pre on the host), so the other batches do not depend on this term
        cs = C.ChainStore(cfg["chain_sets"], dev, 1000 * cfg["seed"] + 11)
        pre_gen = torch.Generator().manual_seed(1000 * cfg["seed"] + 12)
        rep = torch.tensor([D.SHORT.index(s) for s in C.REPORTED], device=dev)
        cacc = [0.0, 0.0, 0, 0]  # chain loss, tail (sampled every 50 steps), samples; sum of n_pre over the log interval
        log(stage, None, chain_paths=cs.info, chain_calls=int(cs.n.sum()))
    params = list(model.parameters())
    rcs = {k: [float(cfg[f"{k}_row_clip"]), torch.zeros((), dtype=torch.int64, device=dev), 0]
           if cfg[f"{k}_row_clip"] > 0 and stage == 2 else None for k in ("grid", "chain")}
    gnl = []                                                        # clip_norm_fp64: (grid sub-steps, pre-clip norm) per step
    if lock:
        lfns = lock_fns(model, delta_fn, cfg["compile"], mu, sd)
        lfns.update(rc_g=rcs["grid"], rc_c=rcs["chain"])
    nskip = nskip_log = 0                                           # skip_nonfinite: steps skipped so far, at the last log row
    ck = run / f"ckpt_stage{stage}.pt"
    step0 = 0
    if cfg["resume"] and ck.exists():  # continue exactly where the last checkpoint left off
        blob = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(blob["model"])
        ema.shadow.load_state_dict(blob["ema"])
        opt.load_state_dict(blob["opt"])
        gen.set_state(blob["gen"].cpu())          # generator states are CPU ByteTensors (map_location moved them)
        if lock and blob.get("chain_gen") is not None:
            cs.gen.set_state(blob["chain_gen"].cpu())
            pre_gen.set_state(blob["pre_gen"].cpu())
            cacc[:] = blob["cacc"]
        step0, acc, nacc = blob["step"], blob["acc"], blob["nacc"]
        nskip = nskip_log = blob.get("nskip") or 0
        t0 -= blob["seconds"]
        log(stage, step0, resumed_from=str(ck))
    for step in range(step0, steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, steps, peak, warm, cfg["lr_min_frac"])
        if lock:   # stage 2: the one-step, jump, grid and chain terms (lock_terms)
            loss, cl, tail, n_pre, kc = lock_terms(lfns, crit, train, cs, cfg, step, gen, pre_gen, mu, sd, rep,
                                                   (step + 1) % 50 == 0)
            loss = loss + cfg["chain_w"] * cl
            if cfg["chain_tail_w"] > 0:
                loss = loss + cfg["chain_tail_w"] * tail
            cacc[3] += n_pre
            if (step + 1) % 50 == 0:
                cacc[0] += float(cl.detach())
                cacc[1] += float(tail.detach())
                cacc[2] += 1
        else:      # stage 1: one-step windows (teacher forcing)
            z, dtn, g, _ = train.sample(cfg["batch"], 1, gen)
            z0 = z[:, 0]
            d_true = (z[:, 1] - z0).float()
            d_pred = delta_fn(z0.float(), dtn, g)
            loss = crit(d_pred.unsqueeze(1), d_true.unsqueeze(1))["loss_total"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["clip_norm_fp64"]:   # as clip_grad_norm_ with the norm in fp64
            gl_ = [p_.grad for p_ in params if p_.grad is not None]
            tn = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(q_, dtype=torch.float64) for q_ in gl_]))
            torch._foreach_mul_(gl_, torch.clamp(cfg["clip"] / (tn + 1e-6), max=1.0).float())
            gnl.append((1 + step % cfg["grid_sub"], float(tn)))
        else:
            tn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        skipped = bool(cfg["skip_nonfinite"]) and not math.isfinite(float(tn))   # a host sync per step
        if skipped:
            nskip += 1
            log(stage, step + 1, nonfinite_skip=nskip, gnorm_total=float(tn))
        else:
            opt.step()
            ema.update(model)
        acc += float(loss.detach()) if (step + 1) % cfg["loss_every"] == 0 and not skipped else 0.0
        nacc += 1 if (step + 1) % cfg["loss_every"] == 0 and not skipped else 0
        if (step + 1) % cfg["log_every"] == 0:
            extra = {}
            if lock:
                extra.update(chain_loss=cacc[0] / max(1, cacc[2]), chain_tail=cacc[1] / max(1, cacc[2]), chain_K=kc,
                             chain_pre_max=_ramp(cfg["chain_pre_max_start"], cfg["chain_pre_max"], cfg["chain_pre_ramp"], step),
                             chain_pre_mean=cacc[3] / cfg["log_every"])
                cacc[:] = [0.0, 0.0, 0, 0]
            if cfg["skip_nonfinite"]:
                extra["nonfinite_skips"] = nskip
                if nskip - nskip_log > cfg["log_every"] // 2:   # most steps of the interval skipped: a breakdown, not a rare row
                    sys.exit(f"stage {stage} step {step + 1}: {nskip - nskip_log} of {cfg['log_every']} steps skipped "
                             "(non-finite gradient); stopped (last checkpoint kept)")
                nskip_log = nskip
            if gnl:
                a_ = np.array([v for _, v in gnl])
                extra.update(gnorm_p50=float(np.median(a_)), gnorm_max=float(a_.max()), gnorm_gt1e3=int((a_ > 1e3).sum()),
                             gnorm_p50_nsub=[float(np.median([v for n_, v in gnl if n_ == k])) if any(n_ == k for n_, _ in gnl)
                                             else None for k in range(1, cfg["grid_sub"] + 1)])
                gnl.clear()
            for k_, rc in rcs.items():   # clipped (row, call) pairs / all since the last row
                if rc is not None:
                    extra[f"rowclip_{k_}"] = int(rc[1]) / max(1, rc[2])
                    rc[1].zero_()
                    rc[2] = 0
            if dev.type == "cuda":   # peak allocated memory since the start (log only)
                extra["peak_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 1e9
            log(stage, step + 1, train_loss=acc / max(1, nacc), lr=opt.param_groups[0]["lr"],
                s_per_step=(time.time() - t0) / (step + 1), **extra)
            if not math.isfinite(acc / max(1, nacc)):
                sys.exit(f"stage {stage} step {step + 1}: non-finite train loss; stopped (last checkpoint kept)")
            acc, nacc = 0.0, 0
        if (step + 1) % cfg["val_every"] == 0 or step + 1 == steps:
            vl, vp = validate(ema.shadow, vy, vdt, vg, crit)
            fz = getattr(model, "fused", False)   # eager, so the unfused projection
            if fz:
                model.fused = False
            try:
                vlive, _ = validate(model, vy, vdt, vg, crit)
            finally:
                if fz:
                    model.fused = True
            log(stage, step + 1, val_loss=vl, val_loss_live=vlive, val_persistence=vp,
                **({} if pr is None else path_readout(ema.shadow, pr)))
        if cfg["ckpt_every"] > 0 and (step + 1) % cfg["ckpt_every"] == 0 and step + 1 < steps:
            if not finite(model, ema.shadow):
                sys.exit(f"stage {stage} step {step + 1}: non-finite weights; stopped (last checkpoint kept)")
            save_atomic({"model": model.state_dict(), "ema": ema.shadow.state_dict(), "opt": opt.state_dict(),
                         "gen": gen.get_state(), "chain_gen": cs.gen.get_state() if lock else None,
                         "pre_gen": pre_gen.get_state() if lock else None, "cacc": list(cacc) if lock else None,
                         "step": step + 1, "acc": acc, "nacc": nacc, "nskip": nskip,
                         "seconds": time.time() - t0, "cfg": cfg, "stage": stage}, ck)
    if not finite(model, ema.shadow):
        sys.exit(f"stage {stage}: non-finite weights at the end; stage file not written")
    save_atomic({"state_dict": ema.shadow.state_dict(), "cfg": cfg, "stage": stage,
                 "seconds": time.time() - t0}, run / f"stage{stage}.pt")
    if ck.exists():
        ck.unlink()   # the stage file supersedes the checkpoint
    model.load_state_dict(ema.shadow.state_dict())  # stage 2 starts from the EMA weights
    return time.time() - t0


def main(run):
    run = Path(run)
    user = json.loads((run / "config.json").read_text())
    unknown = sorted(set(user) - set(DEFAULTS))
    if unknown:   # a key this train.py does not know would otherwise be ignored silently
        sys.exit(f"{run / 'config.json'}: keys not in train.DEFAULTS: {unknown}")
    cfg = {**DEFAULTS, **user}
    assert cfg["cache"] in (None, str(D.CACHE)), f"config cache {cfg['cache']} but ARCH_CACHE {D.CACHE}"
    assert cfg["s2_K"] == 1 and cfg["s2_skip"] == 0, "stage 2 trains one-step windows only (s2_K 1, s2_skip 0)"
    assert cfg["val_source"] == "cache", "validation windows come from the cache (val_source 'cache')"
    assert cfg["s2_steps"] <= 0 or (cfg["lockstep"] and cfg["jump_w"] > 0 and cfg["grid_w"] > 0 and cfg["chain_w"] > 0), \
        "stage 2 runs the one-step, jump, grid and chain terms in lockstep (lockstep true, jump_w, grid_w, chain_w > 0)"
    lib_md5 = {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in sorted(Path(__file__).resolve().parent.glob("*.py"))}
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True   # training with TF32 on; the benches turn it off
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device(os.environ.get("HARNESS_DEVICE", "cuda"))
    D.DT_LOG = (math.log10(cfg["dt_min_s"]), math.log10(cfg["dt_max_s"]))
    D.ANCHOR_P = cfg["anchor_p"]
    D.X0_ANCHOR_P, D.X0_JUMP_P = cfg["x0_anchor_p"], cfg["x0_jump_p"]
    train = D.RawSplit("train", dev)
    if cfg["traj_weights_file"] is not None:
        w = np.load(cfg["traj_weights_file"]).astype(np.float64)
        assert w.shape == (train.n,) and np.all(w >= 0) and w.sum() > 0, (w.shape, train.n)
        train.w = torch.from_numpy(w).float().to(dev)   # fp32 weights
    if cfg["x0_file"] is not None:
        files = json.loads((D.CACHE / "meta.json").read_text())["files"]
        xf = cfg["x0_file"] if isinstance(cfg["x0_file"], list) else [cfg["x0_file"]]  # one name or a list
        train.x0 = torch.from_numpy(np.isin(train.file_idx, [files.index(f) for f in xf])).to(dev)
        assert bool(train.x0.any()), cfg["x0_file"]
    mu, sd, zlo, zhi = D.norm_consts(dev)
    model = M.build(cfg["head"], cfg["hidden"], zlo, zhi, newton=cfg["closed_iters"], fallback=cfg["closed_fallback"],
                    fused=cfg["fused_proj"]).to(dev)
    vy, vdt, vg = D.cache_val(cfg["val_windows"], 2, dev)
    pr = None if cfg["path_readout"] is None else readout_setup(cfg["path_readout"], dev)
    logf = open(run / "log.csv", "a")

    def log(stage, step, **kw):
        rec = {"stage": stage, "step": step, "time": time.time(), **kw}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    info = {"params": M.n_params(model), "flops_per_call": M.flops_per_call(model), "n_train_groups": train.n}
    (run / "model_info.json").write_text(json.dumps(info))
    secs = {}
    if cfg["init_from"]:
        model.load_state_dict(torch.load(cfg["init_from"], map_location=dev)["state_dict"])
    for stage, todo in ((1, not cfg["init_from"]), (2, cfg["s2_steps"] > 0)):
        if not todo:
            continue
        done = run / f"stage{stage}.pt"
        if cfg["resume"] and done.exists():  # finished before the restart; stage 2 starts from its EMA weights
            blob = torch.load(done, map_location=dev, weights_only=False)
            model.load_state_dict(blob["state_dict"])
            secs[f"s{stage}"] = blob["seconds"]
            log(stage, None, resumed_from=str(done))
            continue
        secs[f"s{stage}"] = run_stage(stage, cfg, model, train, vy, vdt, vg, mu, sd, run, log, pr)
    (run / "done.json").write_text(json.dumps({"seconds": secs, **info, "arch_cache": str(D.CACHE), "lib_md5": lib_md5}))
    write_manifest(run, cfg, code_info(lib_md5))


if __name__ == "__main__":
    main(sys.argv[1])
