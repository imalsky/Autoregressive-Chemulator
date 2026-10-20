"""Train one arm: stage 1 (one-step) then stage 2 (K-step BPTT, curriculum, pushforward skip),
with production semantics and step-based schedules. Targets are fp64 increments; the rollout
state is carried in fp64 and the network runs in fp32.

    python train.py <run_dir>     (run_dir/config.json; writes stage1.pt, stage2.pt, log.csv, done.json)

2026-10-09 (each off by default): ckpt_every / resume (run_dir/ckpt_stage<N>.pt, exact continuation), val_source (the
validation windows' source) and path_readout (an arm-P rollout of the EMA model at every validation, see path_readout).
2026-10-09 conservation phase (off by default): the chain term (chain_*; changing-condition paths_v3 chains with their
recorded mixing, lib/chains.py, see chain_term) and closed_iters (the closed heads' Newton iterations).
2026-10-13 scale phase (off by default): the stage-2 stability bundle (s2_prewarm, grid_w_ramp, grid_K_ramp, grid_sub_ramp,
grid_bptt, term_clip, gnorm_every; see DEFAULTS) and peak_mem_gb in the periodic log rows.
2026-10-18 gfix (off by default): grid_row_clip / chain_row_clip (per-row gradient clipping in the BPTT rollouts, RowClip),
clip_norm_fp64 (the global norm in fp64, logged), grid_bptt under lockstep.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
sys.path.insert(0, str(paths.SNAP_SRC))  # pinned snapshot (trainer.HybridLoss)
import data as D  # noqa: E402
import models as M  # noqa: E402
import chains as C  # noqa: E402
M.S, M.SPECIES = len(D.SHORT), list(D.SHORT)  # E13: the species set follows the normalization file
from trainer import HybridLoss  # noqa: E402

DEFAULTS = {
    "head": "delta", "hidden": [512] * 6, "latent": None, "dec_hidden": None, "seed": 0,
    "batch": 2048, "s2_batch": None,   # s2_batch: stage-2 batch if set (else batch)
    "s1_steps": 150_000, "s1_lr": 3e-4, "s1_warmup": 2000, "s1_ema": 0.9995, "s1_lambda_z": 0.1,
    "s2_steps": 25_000, "s2_lr": 3e-5, "s2_warmup": 1000, "s2_ema": 0.999, "s2_lambda_z": 0.25,
    "s2_K": 10, "s2_K_start": 2, "s2_ramp": 5000, "s2_skip": 1,
    "s2_long_unroll": 0,        # >0: pushforward this many no-grad steps at the window's dt, then BPTT K (X2)
    "s2_long_random": False,    # X2: draw the no-grad length uniformly in [0, s2_long_unroll] each step
    "semigroup_w": 0.0,         # X1: weight of |F(F(z,a),a) - F(z,2a)| (unlabeled) in both stages
    "steady_frac": 0.0,         # X3: fraction of each batch started inside converged tails
    "s2_detach": False,         # regime: stop the gradient between rollout steps (each step trained from a constant input)
    "s2_gamma": 1.0,            # regime: per-step loss weight gamma^(k - skip)
    "input_noise": 0.0,         # regime: Gaussian noise (z units) on each window's start state; targets stay clean
    "jump_w": 0.0,              # long-jump loss weight: F(z(t), D) vs z(t + D), D log-uniform in [jump_min_s, jump_max_s]
    "jump_min_s": 0.1, "jump_max_s": 1e7, "jump_batch": 1024,
    "jump_stages": [2],         # stages that add the jump loss
    "grid_w": 0.0,              # E13 flow term: roll grid_K stored log-grid intervals from t >= grid_t_min (1..grid_sub
    "grid_K": 20, "grid_batch": 256, "grid_t_min": 1e3, "grid_sub": 4,  # equal sub-steps each), loss on every row
    "grid_stages": [2],
    "grid_sq_w": 0.0,           # 2026-10-08 audit: + grid_sq_w * mean dex^2 on grid-chain points the scoreboard counts
                                # (truth or prediction >= 1e-2 cm^-3 and >= 1e-12 mixing ratio): big misses get more push
    "grid_stride": 1, "grid_end_frac": 0.0,  # E13 review 2026-10-07: rows per grid step; fraction of chains ending at the last row
    "unres_w": 1.0,             # E13 review: loss weight of pairs whose truth is below 1e-2 cm^-3 (not scored), one-step/jump/grid
    "unres_hinge": 0.0,         # E13 review: weight of relu(pred log10 n + 2) on those pairs (they must not rise into the scored range)
    "loss_floor": {},           # 2026-10-11 (author's per-gas rule): {entry: log10 mixing-ratio floor}; where the truth is below
                                # its floor only gross error counts (loss_sub_w x the error beyond loss_sub_dead dex) plus
                                # loss_cross_w per dex the prediction rises above the floor; entries not listed as before. {} = off
    "loss_sub_w": 0.1, "loss_sub_dead": 2.0, "loss_cross_w": 1.0,
    "species_w": {},            # 2026-10-11: {entry: weight} on points whose truth is at or above its loss floor (with loss_floor)
    "ema_extra": [],            # 2026-10-11: extra EMA decays tracked in stage 2 beside s2_ema (training unchanged), saved as
                                # stage2_ema<decay>.pt before stage2.pt; [] = off
    "train_rows": None,         # E13 (2026-10-07): .npy of train-split row indices to train on (replaces train_frac)
    "traj_w": None,             # E13 (2026-10-07): [T_cut K, w]: trajectories with T < T_cut drawn w times as often
    "anchor_p": 0.25,           # E13 review: fraction of one-step windows anchored at the first positive time (data.ANCHOR_P)
    "phys_sum_w": 0.0,          # weight of mean relu(log10 sum_12 y_pred) (dex) added to each supervised term
    "phys_elem_w": 0.0,         # weight of mean relu(|drift_pred| - |drift_true|) of log10 C/H, O/H, N/H (dex), same terms
    "phys_stages": [2],         # stages that add the two physics terms
    "fp_w": 0.0,                # stage-2 fixed-point term: HybridLoss(q(z_eq, g), z_eq) on converged-tail end states
    "fp_batch": 512,
    "lr_min_frac": 0.01, "wd": 1e-5, "clip": 1.0,
    "val_windows": 65536, "val_every": 10_000, "log_every": 1000,
    "init_from": None,          # path to a stage-1/2 .pt to start stage 2 from (branches); skips stage 1
    "skip_s1": False,
    "compile": True,
    "train_frac": 1.0,          # data-scaling test: train on a seeded random fraction of the training trajectories
    "train_frac_seed": 0,       # which random fraction (E12 replicate)
    "dt_min_s": 0.1, "dt_max_s": 1e3,  # window dt range (E13: the GCM radiative-transfer range)
    "cache": None,              # sprint 2026-10-08: the ARCH_CACHE this run must use (asserted; None = not checked)
    "traj_weights_file": None,  # sprint 2026-10-08: .npy of per-train-row draw weights (all samplers; excludes traj_w)
    "x0_file": None,            # sprint 2026-10-08 (arm M): raw file(s) (meta.json 'files' entries) whose rows start at row 0
    "x0_anchor_p": 0.0,         # with x0_file: share of those rows' windows that start at row 0 (data.X0_ANCHOR_P)
    "x0_jump_p": 0.0,           # with x0_file: share of those rows' jump pairs that start at row 0 (data.X0_JUMP_P)
    "ckpt_every": 0,            # 2026-10-09: every n steps save run/ckpt_stage<N>.pt (model, EMA, optimizer, generators, step,
                                # loss accumulators; atomic); removed once stage<N>.pt is written. 0 = off
    "resume": False,            # 2026-10-09: skip a stage whose stage<N>.pt exists, continue one from its ckpt_stage<N>.pt
    "val_source": "shards",     # 2026-10-09: "shards" (production shards for 12-species sets, else the cache, as before) |
                                # "cache" (the cache's validation split, data.cache_val, for every species set)
    "path_readout": None,       # 2026-10-09: a paths folder (e.g. E13_gcm/paths_v2x): at every validation, the EMA model's
                                # arm-P rollout (lhbench.run_paths, P3, TF32 off) scored as xlong_board (see path_readout)
    "chain_w": 0.0,             # 2026-10-09 conservation phase: weight of the chain term (chain_term): chain_batch windows per
    "chain_sets": [],           # step from these paths_v3 set folders (finished paths at the stage's start; an exact resume
    "chain_batch": 256,         # needs the same files), n_pre ~ U{0..pre_max} no-grad calls from the truth at the window
    "chain_stages": [2],        # start, then K calls with gradient; the recorded mixing before every call. 0 = off
    "chain_K_start": 8, "chain_K": 16, "chain_K_ramp": 0,               # K grows linearly over chain_K_ramp steps (0: chain_K)
    "chain_pre_max_start": 0, "chain_pre_max": 80, "chain_pre_ramp": 0,  # pre_max likewise
    "chain_tail_w": 0.0,        # + chain_tail_w * CVaR: the mean over the top chain_tail_q fraction of windows of each window's
    "chain_tail_q": 0.1,        # max |dlog10 y| over its K calls and the 12 reported species on the scored points (logged always)
    "closed_iters": 12,         # 2026-10-09: Newton iterations of the relaxclosed / relaxclosedr projection (models.build)
    "closed_fallback": "keep",  # 2026-10-12: relaxclosedr rows whose Newton projection fails: 'keep' (input) | 'lm' (re-solve)
    # 2026-10-13 stage-2 stability bundle (scale phase; each 0 = off, the run as before). Ramps and curricula count stage-2
    # steps from 0, prewarm steps included; the prewarm steps are part of s2_steps.
    "s2_prewarm": 0,            # first n stage-2 steps: full gradient + AdamW step at lr 0 (moments fill, weights and EMA
                                # unchanged, weight decay too); s2_warmup and the cosine then run over the remaining steps
    "grid_w_ramp": 0,           # grid term weights (grid_w and grid_sq_w) x min(1, step / grid_w_ramp)
    "grid_K_start": 5, "grid_K_ramp": 0,  # grid intervals grow linearly grid_K_start -> grid_K over grid_K_ramp steps (_ramp)
    "grid_sub_ramp": 0,         # sub-steps per interval: 1 + step % grid_sub as before, capped by a maximum that grows
                                # linearly 1 -> grid_sub over grid_sub_ramp steps (_ramp)
    "grid_bptt": 0,             # n > 0: in each grid chain only the last n model calls carry gradient (earlier calls run
                                # without it; their predictions stay in the loss value as constants)
    "term_clip": 0.0,           # c > 0: one backward per stage-2 term; each unweighted term's gradient is scaled down to at
                                # most c x the running mean (decay 0.99, of its clipped norms) of its norm, then weighted and
                                # summed (the global clip still applies after)
    "optimizer": "adamw",       # 2026-10-16 batch study: "adamw" (as before) | "lion" (Chen et al. 2023, sign of an EMA blend)
    "lion_betas": [0.9, 0.99],  # Lion's (beta1, beta2); weight decay as AdamW's decoupled wd x lr
    "gnorm_every": 0,           # n > 0: every n stage-2 steps, a log row with each term's unweighted gradient norm (one extra
                                # backward per term on those steps; the update is unchanged), the total pre-clip norm and
                                # the grid ramp state
    # 2026-10-17 fuse study (each off by default: the run as before)
    "fused_proj": False,        # relaxclosedr: the Newton projection as elemproj.project_ln_fused (same mathematics, written
                                # for kernel fusion under compile; no cuBLAS solve)
    "compile_mode": None,       # torch.compile mode of delta_fn (None: the default mode, as before), e.g. "reduce-overhead"
                                # (CUDA graphs). Stage 1: every call. Stage 2: only the calls without gradient (chain pre-roll,
                                # grid_bptt, s2_skip, long unroll); calls with gradient keep the default mode (CUDA graph
                                # trees on the varying-length BPTT rollouts: illegal memory access with project_ln, minutes
                                # of re-recording per new call pattern with project_ln_fused; fuse study report)
    "loss_every": 50,           # steps between the train_loss samples averaged into each log row (a host sync each)
    # 2026-10-17 fastt (each off by default: the run as before)
    "lockstep": False,          # stage 2 (s2_K 1 with the jump, grid and chain terms): the same terms in fewer, larger model
                                # calls (lock_terms): one-step windows and jump pairs in one call; the chain's K gradient
                                # calls share the grid's last calls (its pre-roll runs first, in its own calls without
                                # gradient), the chain mix, casts and state add compiled inside the call (lock_rollouts,
                                # lock_fns). Same loss and gradient up to rounding; same activation memory
    "skip_nonfinite": False,    # a step whose total gradient norm is not finite makes no optimizer step and no EMA update;
                                # counted (log rows nonfinite_skip / nonfinite_skips, checkpoint key nskip)
    # 2026-10-18 gfix (each off by default: the run as before)
    "grid_row_clip": 0.0,       # c > 0: after every grid-rollout model call with gradient, an identity whose backward scales
    "chain_row_clip": 0.0,      # each row's incoming state gradient to norm <= c (RowClip; per-timestep gradient clipping
                                # in BPTT), chain_row_clip likewise for the chain rollout's gradient calls. c is absolute:
                                # it scales with the term's weight / (rows x intervals x entries), so re-derive it when
                                # grid_batch, grid_K, grid_w (chain_batch, chain_K, chain_w) change. Log: rowclip_<term> =
                                # clipped (row, call) pairs / all, per log row
    "clip_norm_fp64": False,    # the global gradient norm (clipping, the non-finite guard) in fp64 (fp32 sums of squares
                                # overflow above ~1.8e19); log rows add the pre-clip norm's median (gnorm_p50, and per grid
                                # sub-step count gnorm_p50_nsub), max (gnorm_max) and the steps above 1e3 (gnorm_gt1e3)
}
TCLIP_DECAY = 0.99              # term_clip running-mean decay


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al. 2023, arXiv 2302.06675, Algorithm 2): update = sign(beta1 m + (1 - beta1) g); decoupled weight
    decay p -= lr wd p; m <- beta2 m + (1 - beta2) g. One state tensor per parameter (fp32 like the parameters)."""

    def __init__(self, params, lr, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, {"lr": lr, "betas": tuple(betas), "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                c = m.mul(b1).add_(p.grad, alpha=1.0 - b1)
                if g["weight_decay"]:
                    p.mul_(1.0 - g["lr"] * g["weight_decay"])
                p.add_(c.sign_(), alpha=-g["lr"])
                m.mul_(b2).add_(p.grad, alpha=1.0 - b2)


class RowClip(torch.autograd.Function):
    """2026-10-18 gfix (grid_row_clip, chain_row_clip): identity in forward; in backward each row's incoming gradient
    (norm over the last dimension, fp64 like the state) is scaled to norm <= rc[0]; rc[1] (a device counter) counts the
    clipped rows. rc[0] is read at backward time."""

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
        if getattr(self.shadow, "fused", False):   # 2026-10-17 fastt (fuse review issue 3): the copy only runs eager
            self.shadow.fused = False              # (validation, path readout): the unfused projection, faster there

    @torch.no_grad()
    def update(self, model):
        torch._foreach_lerp_(list(self.shadow.parameters()), list(model.parameters()), 1.0 - self.decay)


def rollout_loss(delta_fn, crit, z, dtn, g, k_train, skip, detach=False, gamma=1.0, z0=None, keep=None):
    """Rollout loss over steps skip..k_train-1 (state fp64); z [B, K+1, S] fp64 truth. BPTT by default; with
    detach, each step's gradient stops at its input. Steps are weighted gamma^(k - skip). If keep is a list, the
    predictions [B, k_train - skip, S] and the state entering step skip are appended to it."""
    zs = z[:, 0] if z0 is None else z0
    with torch.no_grad():
        for k in range(skip):
            zs = zs + delta_fn(zs.float(), dtn, g).double()
    if keep is not None:
        keep.append(zs)
    preds = []
    for k in range(skip, k_train):
        inp = zs.detach() if detach else zs
        zs = inp + delta_fn(inp.float(), dtn, g).double()
        preds.append(zs)
    pred = torch.stack(preds, dim=1)
    true = z[:, 1 + skip:1 + k_train]
    w = None if gamma == 1.0 else torch.tensor([gamma ** j for j in range(k_train - skip)], device=z.device)
    if keep is not None:
        keep.append(pred)
    return crit(pred, true, step_weights=w)


# Atoms (C, H, O, N) per molecule of the 12 tracked species (as score.py STOICH); element-total floor as g2/G2-A2.
STOICH = {"C2H2": (2, 2, 0, 0), "CH4": (1, 4, 0, 0), "CO2": (1, 0, 2, 0), "CO": (1, 0, 1, 0), "H2O": (0, 2, 1, 0),
          "H2": (0, 2, 0, 0), "HCN": (1, 1, 0, 1), "H": (0, 1, 0, 0), "N2": (0, 0, 0, 2), "NH3": (0, 3, 0, 1),
          "OH": (0, 1, 1, 0), "O": (0, 0, 1, 0)}
EFLOOR = 1e-12


def _ratios(z, mu, sd, A):
    """log10 C/H, O/H, N/H [..., 3] of standardized states z [..., S] (fp32)."""
    lE = torch.log10(torch.pow(10.0, z.float() * sd + mu) @ A + EFLOOR)
    return lE[..., [0, 2, 3]] - lE[..., 1:2]


def phys_terms(pred, true, start_m, start_t, mu, sd, A):
    """Soft physics penalties in dex for predictions pred of truth true, both [..., S] z, made from the model input
    start_m; start_t is the truth at the start. Returns (mean relu(log10 sum_12 y_pred), mean over C/H, O/H, N/H of
    relu(|r(pred) - r(start_m)| - |r(true) - r(start_t)|)): element-ratio drift beyond the truth's own drift."""
    s = torch.pow(10.0, pred.float() * sd + mu).sum(-1)
    dm = _ratios(pred, mu, sd, A) - _ratios(start_m, mu, sd, A)
    dt_ = _ratios(true, mu, sd, A) - _ratios(start_t, mu, sd, A)
    return torch.relu(torch.log10(s)).mean(), torch.relu(dm.abs() - dt_.abs()).mean()


def semigroup_term(delta_fn, z0, dtn, g):
    """|F(F(z, a), a) - F(z, 2a)| in z, with 2a in the trained dt range (dt_norm + log10(2)/4). States are
    carried in fp64 (network inputs cast to fp32), as in the rollout, so tiny increments are not erased."""
    dt2 = (dtn + math.log10(2.0) / 4.0)
    ok = dt2 <= 1.0
    one = z0 + delta_fn(z0.float(), dtn, g).double()
    two = one + delta_fn(one.float(), dtn, g).double()
    big = z0 + delta_fn(z0.float(), dt2.clamp(max=1.0), g).double()
    return ((two - big).abs().float() * ok.reshape(-1, 1)).mean()


@torch.no_grad()
def validate(model, vy, vdt, vg, crit, K, skip, bs=8192):
    tot = per = n = 0.0
    for i in range(0, vy.shape[0], bs):
        y, dtn, g = vy[i:i + bs].double(), vdt[i:i + bs], vg[i:i + bs]
        if K == 1:
            d = model.delta(y[:, 0].float(), dtn, g)
            l = crit((y[:, 0] + d.double()).unsqueeze(1), y[:, 1:2])["loss_total"]
            pl = crit(y[:, 0:1], y[:, 1:2])["loss_total"]
        else:
            zs = y[:, 0]
            preds = []
            for k in range(K):
                zs = zs + model.delta(zs.float(), dtn, g).double()
                preds.append(zs)
            pred = torch.stack(preds, 1)[:, skip:]
            l = crit(pred, y[:, 1 + skip:1 + K])["loss_total"]
            pl = crit(y[:, 0:1].expand(-1, K - skip, -1), y[:, 1 + skip:1 + K])["loss_total"]
        w = y.shape[0]
        tot += float(l) * w
        per += float(pl) * w
        n += w
    return tot / n, per / n


def finite(*ms):
    """2026-10-09 (chain-term review): all floating weights finite; checked before every save, so a divergence stops the
    run without overwriting the last good checkpoint."""
    return all(bool(torch.isfinite(v).all()) for m in ms for v in m.state_dict().values() if v.is_floating_point())


def save_atomic(obj, path):
    """2026-10-09: torch.save to a temporary file beside path, fsync, then rename (a kill never leaves a partial file)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def code_info(lib_md5):
    """2026-10-19: the code a run trained with: its lib folder, every lib file's md5, and the git commit, tags and dirty
    flag when lib is the lib/ of a git repository (None for a frozen copy outside one)."""
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
    """2026-10-19: run/manifest.json, what a model's checkpoints need at use: the sha256 of every .pt in run; a byte-identical
    copy of the normalization file in use (data.NORM_PATH) as run/norm.json, with its sha256; the entry order; the resolved
    config (DEFAULTS merged with config.json); the parent checkpoint's sha256 (init_from); the code. lhbench.load_model
    refuses a normalization file whose sha256 differs."""
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
    """2026-10-09 (path_readout): the 12 reported species' VULCAN truth after every call of a paths folder and the log10
    number density of the scored mask, computed once. Window: calls >= 3000 (the 58-day set's late window; calls >= 50
    for sets of at most 3000 calls)."""
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
    """2026-10-09: the model's arm-P rollout on the readout paths (the bench's code and precision: P3 state, TF32 off for
    the rollout, restored after), |dlog10 y| of the 12 reported species on xlong_board's points (truth or prediction
    >= 1e-2 cm^-3, and either >= 1e-12 mixing ratio) in the late window: p50, p90, p99, max (dex), and its seconds."""
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


def unres_loss(crit, pred, true, zt, zp, lnd, mu, sd, cfg):
    """E13 review (2026-10-07): HybridLoss's loss_total with weight unres_w on pairs whose truth is below the scored
    number density (log10 y + log10 n_tot < -2), plus unres_hinge * relu(log10 y_pred + log10 n_tot + 2) on those pairs;
    both summed over [B, K, S] and divided by B K S (unres_w 1, unres_hinge 0: exactly loss_total). pred, true: what
    crit would receive (z or increments); zt, zp: absolute truth and prediction z [B, K, S] (fp64); lnd [B] or [B, K]."""
    dz = (pred - true).to(crit.loss_dtype)
    ld = lnd.reshape(lnd.shape[0], -1, 1)                                  # lnd [B] (per window) or [B, K] (chains: per call)
    res = (zt * sd + mu + ld) >= -2.0
    w = torch.where(res, 1.0, cfg["unres_w"]).to(crit.loss_dtype)
    el = crit.lambda_log10_mae * dz.abs() * crit.log_stds + crit.lambda_z_mse * dz.square()
    h = torch.relu(zp * sd + mu + ld + 2.0) * torch.logical_not(res)
    if not cfg["loss_floor"]:
        return ((el * w).sum() + cfg["unres_hinge"] * h.sum().to(crit.loss_dtype)) / dz.numel()
    lf, ws = floor_tensors(cfg, zt.device)                                 # [S] log10 floor (-inf: none), [S] weight
    yt, yp = zt * sd + mu, zp * sd + mu                                    # log10 mixing ratio, fp64
    sub = yt < lf                                                          # truth below its floor
    over = torch.clamp(dz.abs() * crit.log_stds - cfg["loss_sub_dead"], min=0.0)   # dex beyond the dead zone
    el_sub = crit.lambda_log10_mae * over + crit.lambda_z_mse * (over / crit.log_stds).square()
    el = torch.where(sub, cfg["loss_sub_w"] * el_sub, ws.to(crit.loss_dtype) * el)
    cross = torch.where(sub, torch.relu(yp - lf), 0.0).sum().to(crit.loss_dtype)   # dex the prediction rises above the floor
    return ((el * w).sum() + cfg["unres_hinge"] * h.sum().to(crit.loss_dtype) + cfg["loss_cross_w"] * cross) / dz.numel()


_FLOORS = {}


def floor_tensors(cfg, dev):
    """loss_floor and species_w as [S] tensors in the entry order (data.SHORT): floor -inf and weight 1 where unlisted."""
    key = (json.dumps(cfg["loss_floor"], sort_keys=True), json.dumps(cfg["species_w"], sort_keys=True), str(dev))
    if key not in _FLOORS:
        names = list(D.SHORT)
        bad = sorted(set(cfg["loss_floor"]) - set(names)) + sorted(set(cfg["species_w"]) - set(cfg["loss_floor"]))
        assert not bad, f"loss_floor / species_w entries not in the entry set or without a floor: {bad}"
        lf = torch.tensor([float(cfg["loss_floor"].get(s, -math.inf)) for s in names], dtype=torch.float64, device=dev)
        ws = torch.tensor([float(cfg["species_w"].get(s, 1.0)) for s in names], dtype=torch.float64, device=dev)
        _FLOORS[key] = (lf, ws)
    return _FLOORS[key]


def _ramp(a, b, n, step):
    return b if n <= 0 else int(a + (b - a) * min(1.0, step / n))


def chain_term(delta_fn, crit, cs, cfg, step, pre_gen, mu, sd, rep, unres, want_tail, rc=None):
    """2026-10-09 conservation phase: rollout on changing-condition chains (lib/chains.py). chain_batch windows of n_pre + K
    calls; from the truth at the window start, n_pre ~ U{0..pre_max} model calls without gradient, then K calls with
    gradient (BPTT; state fp64, network fp32, as the grid term); before every call with a recorded mix, the bench's blend
    (chains.mix). Returns the HybridLoss over the K calls vs the truth after each (unres_loss with each call's number
    density when the unres options are on), the tail (or None), n_pre and K. Tail: per window the max |dlog10 y| over its
    K calls and the 12 reported species on the scoreboard's points (truth or prediction >= 1e-2 cm^-3, and truth or
    prediction >= 1e-12 mixing ratio), averaged over the top chain_tail_q fraction of windows (CVaR, dex); with gradient
    when chain_tail_w > 0, else only if want_tail (for the log). rc: row_clip after each gradient call (2026-10-18)."""
    K, n_pre, w = chain_draw(cs, cfg, step, pre_gen)
    zs = w["z0"]
    with torch.no_grad():
        for j in range(n_pre):
            zs = C.mix(zs, w["f"][j], w["yp"][j], mu, sd)
            zs = zs + delta_fn(zs.float(), w["dtn"][j], w["g"][j]).double()
    preds = []
    for j in range(n_pre, n_pre + K):
        zs = C.mix(zs, w["f"][j], w["yp"][j], mu, sd)
        zs = row_clip(zs + delta_fn(zs.float(), w["dtn"][j], w["g"][j]).double(), rc)
        preds.append(zs)
    pred = torch.stack(preds, 1)                                         # [B, K, S]
    cl, tail = chain_loss(crit, cfg, w, n_pre, pred, mu, sd, rep, unres, want_tail)
    return cl, tail, n_pre, K


def chain_draw(cs, cfg, step, pre_gen):
    """2026-10-17 (fastt): chain_term's draw: K and n_pre (ramped) and the windows."""
    K = _ramp(cfg["chain_K_start"], cfg["chain_K"], cfg["chain_K_ramp"], step)
    pm = _ramp(cfg["chain_pre_max_start"], cfg["chain_pre_max"], cfg["chain_pre_ramp"], step)
    n_pre = int(torch.randint(0, pm + 1, (1,), generator=pre_gen))
    return K, n_pre, cs.windows(cfg["chain_batch"], n_pre + K)


def chain_loss(crit, cfg, w, n_pre, pred, mu, sd, rep, unres, want_tail):
    """2026-10-17 (fastt): chain_term's loss and tail of the predictions pred [B, K, S] after the K calls with gradient."""
    true = w["zt"][n_pre:].transpose(0, 1)
    lnd = w["lnd"][n_pre:].T                                             # [B, K]
    cl = unres_loss(crit, pred, true, true, pred, lnd, mu, sd, cfg) if unres else crit(pred, true)["loss_total"]
    tail = None
    if cfg["chain_tail_w"] > 0 or want_tail:
        with torch.set_grad_enabled(cfg["chain_tail_w"] > 0):
            yt, yq, ld = true[..., rep] * sd[rep] + mu[rep], pred[..., rep] * sd[rep] + mu[rep], lnd.unsqueeze(-1)
            m = (((yt + ld) >= -2.0) | ((yq + ld) >= -2.0)) & ((yt >= -12.0) | (yq >= -12.0))
            e = ((yq - yt).abs() * m).amax(dim=(1, 2))                   # [B] worst scored point per window
            tail = torch.topk(e, max(1, math.ceil(cfg["chain_tail_q"] * e.numel()))).values.mean()
    return cl, tail


def grid_draw(train, cfg, step, gen):
    """2026-10-17 (fastt): the grid term's draw (run_stage, lock_terms): chains, normalized sub-step lengths, intervals,
    sub-steps, the ramped weights and the logged grid state."""
    Kg = _ramp(cfg["grid_K_start"], cfg["grid_K"], cfg["grid_K_ramp"], step)   # grid_K unless grid_K_ramp
    gw, gsw = cfg["grid_w"], cfg["grid_sq_w"]
    if cfg["grid_w_ramp"] > 0:
        gf = min(1.0, step / cfg["grid_w_ramp"])
        gw, gsw = gw * gf, gsw * gf
    gz, gdt, gg, gidx = train.sample_grid(cfg["grid_batch"], Kg, gen, cfg["grid_t_min"], cfg["grid_stride"],
                                          cfg["grid_end_frac"])
    nsub = 1 + step % cfg["grid_sub"]                    # equal relax sub-steps per stored interval
    smax = _ramp(1, cfg["grid_sub"], cfg["grid_sub_ramp"], step)   # grid_sub unless grid_sub_ramp
    nsub = min(nsub, smax)
    gstate = {"grid_w_eff": gw, "grid_K_eff": Kg, "grid_sub_max": smax}
    lo_, hi_ = D.NORM["dt"]["log_min"], D.NORM["dt"]["log_max"]
    gdtn = ((torch.log10(gdt / nsub) - lo_) / (hi_ - lo_)).float()  # unclamped (as the jump pairs)
    return gz, gdtn, gg, gidx, Kg, nsub, gw, gsw, gstate


def grid_loss(crit, cfg, train, gz, gp, gidx, mu, sd, unres, gw, gsw, loss, terms):
    """2026-10-17 (fastt): the grid term (and grid_sq) of the predictions gp [B, Kg, S] added to loss; appends to terms."""
    if unres:
        gl = unres_loss(crit, gp, gz[:, 1:], gz[:, 1:], gp, train.lnd[gidx], mu, sd, cfg)
    else:
        gl = crit(gp, gz[:, 1:])["loss_total"]
    loss = loss + gw * gl
    terms.append(("grid", gw, gl))
    if cfg["grid_sq_w"] > 0:
        ld = train.lnd[gidx].reshape(-1, 1, 1)
        yt, yp = gz[:, 1:] * sd + mu, gp * sd + mu               # log10 mixing ratio, fp64
        m = (((yt + ld) >= -2.0) | ((yp + ld) >= -2.0)) & ((yt >= -12.0) | (yp >= -12.0))
        sq = ((yp - yt).square() * m).sum().to(crit.loss_dtype)
        loss = loss + gsw * sq / m.numel()                       # same operation order as before the bundle
        terms.append(("grid_sq", gsw, sq / m.numel()))
    return loss


def lock_fns(model, delta_fn, comp, cmode, mu, sd):
    """2026-10-17 fastt (lockstep): the model calls of lock_terms. delta: the one-step + jump call (delta_fn); gstep: a
    grid call; cstep: a chain call with gradient, or a call with grid and chain rows (grid rows have mix fraction 0); pre:
    a chain pre-roll call without gradient. gstep, cstep and pre run the casts, the state add and (cstep, pre) the mix
    inside the compiled call ("fold"); each is its own function, so each has its own dynamo cache. With compile_mode set,
    pre is the mix (eager) and delta_fn's no-gradient compile (its CUDA-graph output is consumed at once)."""
    def gstep(zs, dtn, g):
        return zs + model.delta(zs.float(), dtn, g).double()

    def cstep(zs, f, yp, dtn, g):
        zs = C.mix(zs, f, yp, mu, sd)
        return zs + model.delta(zs.float(), dtn, g).double()

    def cstep_ng(zs, f, yp, dtn, g):
        zs = C.mix(zs, f, yp, mu, sd)
        return zs + model.delta(zs.float(), dtn, g).double()

    def pre_cm(zs, f, yp, dtn, g):
        zs = C.mix(zs, f, yp, mu, sd)
        return zs + delta_fn(zs.float(), dtn, g).double()
    c = (lambda fn: torch.compile(fn, fullgraph=True)) if comp else (lambda fn: fn)
    # share_pre False (fastt tests): the pre-roll in its own calls was as fast as sharing the grid's calls and needs no
    # extra activation memory (sharing: +50 % of the step's peak at nsub 4, n_pre 80)
    return {"delta": delta_fn, "gstep": c(gstep), "cstep": c(cstep), "pre": pre_cm if cmode else c(cstep_ng),
            "share_pre": False}


def lock_rollouts(fns, gz, gdtn, gg, nsub, Kg, w, n_pre, K):
    """2026-10-17 fastt (lockstep): the grid rollout (Kg stored intervals x nsub calls, all with gradient) and the chain
    rollout (n_pre calls without gradient, then K with, the mix before each) in one batch per call, aligned to end together.
    fns["share_pre"] True: of L = max(Kg nsub, n_pre + K) calls, the grid rows run in the last Kg nsub and the chain rows in
    the last n_pre + K; chain pre-roll rows in a call with grid rows go through it with gradient on (their activations are
    kept until the backward, and the backward runs over them) and the chain state is detached where its gradient part
    starts. False: the n_pre pre-roll calls run first as calls without gradient, then the K gradient calls share the grid's
    last calls (no extra activations). Either way the gradient is that of the separate rollouts (chain_term's no_grad
    pre-roll); calls with chain pre-roll rows only run without gradient. Returns the grid predictions after each interval
    [Bg, Kg, S] and the chain predictions after each gradient call [Bc, K, S], as the separate rollouts (up to rounding).
    2026-10-18 gfix: fns["bptt"] n > 0: only the grid's last n calls carry gradient (its earlier calls run alone without
    gradient, a chain call at the same position in its own call); fns["rc_g"] / fns["rc_c"]: row_clip of the grid / chain
    state after each gradient call (absent: off)."""
    rg, rcc = fns.get("rc_g"), fns.get("rc_c")
    zc, o = w["z0"], 0
    if not fns["share_pre"]:
        with torch.no_grad():
            for ci in range(n_pre):
                zc = fns["pre"](zc, w["f"][ci], w["yp"][ci], w["dtn"][ci], w["g"][ci])
        o = n_pre                                    # the chain calls left: the K gradient calls
    Ng, Nc = Kg * nsub, n_pre + K - o
    L = max(Ng, Nc)
    g0 = max(0, Ng - fns["bptt"]) if fns.get("bptt", 0) > 0 else 0   # grid calls gi < g0 run without gradient
    zg = gz[:, 0]
    Bg = zg.shape[0]
    f0, y0 = w["f"].new_zeros(Bg), w["yp"].new_zeros(Bg, w["yp"].shape[-1])   # grid rows: mix fraction 0 (unchanged)
    gp, cp = [], []
    for c in range(L):
        gi, cj = c - (L - Ng), c - (L - Nc)
        ci = cj + o                                  # the chain call (window index) of this call, if cj >= 0
        if cj >= 0 and ci == n_pre:
            zc = zc.detach()
        if gi >= g0 and cj >= 0:
            out = fns["cstep"](torch.cat([zg, zc]), torch.cat([f0, w["f"][ci]]), torch.cat([y0, w["yp"][ci]]),
                               torch.cat([gdtn[:, gi // nsub], w["dtn"][ci]]), torch.cat([gg, w["g"][ci]]))
            zg, zc = row_clip(out[:Bg], rg), out[Bg:]
            if ci >= n_pre:
                zc = row_clip(zc, rcc)
        else:
            if gi >= g0:
                zg = row_clip(fns["gstep"](zg, gdtn[:, gi // nsub].contiguous(), gg), rg)
            elif gi >= 0:
                with torch.no_grad():
                    zg = fns["gstep"](zg, gdtn[:, gi // nsub].contiguous(), gg)
            if cj >= 0 and ci >= n_pre:
                zc = row_clip(fns["cstep"](zc, w["f"][ci], w["yp"][ci], w["dtn"][ci], w["g"][ci]), rcc)
            elif cj >= 0:
                with torch.no_grad():
                    zc = fns["pre"](zc, w["f"][ci], w["yp"][ci], w["dtn"][ci], w["g"][ci])
        if gi >= 0 and (gi + 1) % nsub == 0:
            gp.append(zg)
        if cj >= 0 and ci >= n_pre:
            cp.append(zc)
    return torch.stack(gp, 1), torch.stack(cp, 1)


def lock_terms(fns, crit, train, cs, cfg, step, gen, pre_gen, mu, sd, rep, unres, want_tail, pen):
    """2026-10-17 fastt (lockstep): stage 2's one-step, jump, grid and chain terms as run_stage computes them (s2_K 1),
    with every batch drawn first (same generators, same order: one-step windows, jump pairs, grid chains on gen; chain
    windows on the store's and pre_gen), then the one-step windows and the jump pairs in one model call and the grid and
    chain rollouts in lock_rollouts. Returns the loss (one-step + jump + grid terms, summed in run_stage's order), the
    terms list, the grid state and chain_term's returns (cl, tail, n_pre, K). pen: run_stage's physics penalty or None."""
    dev = train.device
    z, dtn, g, widx = train.sample(cfg["batch"], 1, gen, cfg["steady_frac"])
    z0 = z[:, 0]
    if cfg["input_noise"] > 0:
        z0 = z0 + cfg["input_noise"] * torch.randn(z0.shape, device=dev, dtype=z0.dtype, generator=gen)
    d_true = (z[:, 1] - z0).float()
    j0, j1, jdt, jg, jidx = D.sample_jump_pairs(train, cfg["jump_batch"], gen, cfg["jump_min_s"], cfg["jump_max_s"], True)
    gz, gdtn, gg, gidx, Kg, nsub, gw, gsw, gstate = grid_draw(train, cfg, step, gen)
    K, n_pre, w = chain_draw(cs, cfg, step, pre_gen)
    B1 = z0.shape[0]
    dd = fns["delta"](torch.cat([z0, j0]).float(), torch.cat([dtn, jdt]), torch.cat([g, jg]))
    d_pred, jp = dd[:B1], j0 + dd[B1:].double()
    if unres:
        loss = unres_loss(crit, d_pred.unsqueeze(1), d_true.unsqueeze(1), z[:, 1:2], (z0 + d_pred.double()).unsqueeze(1),
                          train.lnd[widx], mu, sd, cfg)
    else:
        loss = crit(d_pred.unsqueeze(1), d_true.unsqueeze(1))["loss_total"]
    if pen is not None:
        loss = loss + pen(0, z0.float() + d_pred, z[:, 1], z0, z[:, 0])
    terms = [("main", 1.0, loss)]
    if unres:
        jl = unres_loss(crit, jp.unsqueeze(1), j1.unsqueeze(1), j1.unsqueeze(1), jp.unsqueeze(1), train.lnd[jidx], mu, sd, cfg)
    else:
        jl = crit(jp.unsqueeze(1), j1.unsqueeze(1))["loss_total"]
    if pen is not None:
        jl = jl + pen(2, jp, j1, j0, j0)
    loss = loss + cfg["jump_w"] * jl
    terms.append(("jump", cfg["jump_w"], jl))
    gp, cpred = lock_rollouts(fns, gz, gdtn, gg, nsub, Kg, w, n_pre, K)
    loss = grid_loss(crit, cfg, train, gz, gp, gidx, mu, sd, unres, gw, gsw, loss, terms)
    cl, tail = chain_loss(crit, cfg, w, n_pre, cpred, mu, sd, rep, unres, want_tail)
    return loss, terms, gstate, (cl, tail, n_pre, K)


def run_stage(stage, cfg, model, train, vy, vdt, vg, mu, sd, run, log, pr=None):
    dev = train.device
    s = "s1" if stage == 1 else "s2"
    steps, peak, warm = cfg[f"{s}_steps"], cfg[f"{s}_lr"], cfg[f"{s}_warmup"]
    crit = HybridLoss(sd.new_zeros(1, 1, M.S).float(), sd.float().view(1, 1, -1), lambda_log10_mae=1.0,
                      lambda_z_mse=cfg[f"{s}_lambda_z"]).to(dev)
    if cfg["optimizer"] == "lion":   # 2026-10-16 batch study
        opt = Lion(param_groups(model, cfg["wd"]), lr=peak, betas=cfg["lion_betas"])
    else:
        assert cfg["optimizer"] == "adamw", cfg["optimizer"]
        opt = torch.optim.AdamW(param_groups(model, cfg["wd"]), lr=peak, betas=(0.9, 0.999), eps=1e-8, fused=True)
    ema = EMA(model, cfg[f"{s}_ema"])
    extras = [EMA(model, d) for d in cfg["ema_extra"]] if stage == 2 else []
    cmode = cfg["compile_mode"] if cfg["compile"] else None          # 2026-10-17 fuse study (see DEFAULTS)
    if cmode is None:
        delta_fn = torch.compile(model.delta, fullgraph=True) if cfg["compile"] else model.delta
    elif stage == 1:
        delta_fn = torch.compile(model.delta, fullgraph=True, mode=cmode)
    else:
        f_grad = torch.compile(model.delta, fullgraph=True)
        f_nograd = torch.compile(model.delta, fullgraph=True, mode=cmode)

        def delta_fn(z, dtn, g):
            return (f_grad if torch.is_grad_enabled() else f_nograd)(z, dtn, g)
    gen = torch.Generator(device=dev).manual_seed(1000 * cfg["seed"] + stage)
    K, skip = cfg["s2_K"], cfg["s2_skip"]
    t0, acc, nacc = time.time(), 0.0, 0
    unres = cfg["unres_w"] != 1.0 or cfg["unres_hinge"] > 0 or bool(cfg["loss_floor"])
    if cfg["loss_floor"]:  # rollout_loss (stage 2 with s2_K > 1), validate, fp_w and semigroup call crit directly
        assert cfg["s2_K"] == 1 and cfg["fp_w"] == 0 and cfg["semigroup_w"] == 0, "loss_floor covers s2_K 1 runs only"
    phys = (cfg["phys_sum_w"] > 0 or cfg["phys_elem_w"] > 0) and stage in cfg["phys_stages"]
    fp = cfg["fp_w"] > 0 and stage == 2
    if fp:  # own generator, so the window and jump batches match the run without the term
        fp_cand = torch.nonzero(train.conv >= 1).squeeze(1)
        fp_gen = torch.Generator(device=dev).manual_seed(1000 * cfg["seed"] + 7)
    chain = cfg["chain_w"] > 0 and stage in cfg["chain_stages"]
    if chain:  # own generators (windows on the device, n_pre on the host), so the other batches match the run without it
        cs = C.ChainStore(cfg["chain_sets"], dev, 1000 * cfg["seed"] + 11)
        pre_gen = torch.Generator().manual_seed(1000 * cfg["seed"] + 12)
        rep = torch.tensor([D.SHORT.index(s) for s in C.REPORTED], device=dev)
        cacc = [0.0, 0.0, 0, 0]  # chain loss, tail (sampled every 50 steps), samples; sum of n_pre over the log interval
        log(stage, None, chain_paths=cs.info, chain_calls=int(cs.n.sum()))
    if phys:
        A = torch.tensor([STOICH[sp] for sp in D.SHORT], dtype=torch.float32, device=dev)
        mu32, sd32 = mu.float(), sd.float()
        pacc = [0.0] * 4     # main sum hinge, main element hinge, jump sum hinge, jump element hinge

        def pen(slot, *a):
            h, e = phys_terms(*a, mu32, sd32, A)
            if (step + 1) % 50 == 0:
                pacc[slot] += float(h)
                pacc[slot + 1] += float(e)
            return cfg["phys_sum_w"] * h + cfg["phys_elem_w"] * e
    pw = cfg["s2_prewarm"] if stage == 2 else 0                     # 2026-10-13 stability bundle (see DEFAULTS)
    tclip = cfg["term_clip"] > 0 and stage == 2
    tc_mean = {}                                                    # term_clip: running mean norm per term
    params = list(model.parameters())
    rcs = {k: [float(cfg[f"{k}_row_clip"]), torch.zeros((), dtype=torch.int64, device=dev), 0]
           if cfg[f"{k}_row_clip"] > 0 and stage == 2 else None for k in ("grid", "chain")}   # 2026-10-18 gfix (row_clip)
    gnl = []                                                        # clip_norm_fp64: (grid sub-steps, pre-clip norm) per step
    lock = bool(cfg["lockstep"]) and stage == 2                     # 2026-10-17 fastt (see DEFAULTS, lock_terms)
    if lock:
        assert (K == 1 and cfg["head"] != "delta" and cfg["jump_w"] > 0 and 2 in cfg["jump_stages"] and cfg["grid_w"] > 0
                and 2 in cfg["grid_stages"] and chain and not tclip and cfg["semigroup_w"] == 0), \
            "lockstep: stage 2 with s2_K 1 and the jump, grid and chain terms only (no term_clip, semigroup_w)"
        lfns = lock_fns(model, delta_fn, cfg["compile"], cmode, mu, sd)
        lfns.update(bptt=cfg["grid_bptt"], rc_g=rcs["grid"], rc_c=rcs["chain"])   # 2026-10-18 gfix (lock_rollouts)
    nskip = nskip_log = 0                                           # skip_nonfinite: steps skipped so far, at the last log row
    ck = run / f"ckpt_stage{stage}.pt"
    step0 = 0
    if cfg["resume"] and ck.exists():  # 2026-10-09: continue exactly where the last checkpoint left off
        blob = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(blob["model"])
        ema.shadow.load_state_dict(blob["ema"])
        for e, sd_ in zip(extras, blob.get("ema_extra") or []):
            e.shadow.load_state_dict(sd_)
        opt.load_state_dict(blob["opt"])
        gen.set_state(blob["gen"].cpu())          # generator states are CPU ByteTensors (map_location moved them)
        if fp:
            fp_gen.set_state(blob["fp_gen"].cpu())
        if phys:
            pacc[:] = blob["pacc"]
        if chain and blob.get("chain_gen") is not None:
            cs.gen.set_state(blob["chain_gen"].cpu())
            pre_gen.set_state(blob["pre_gen"].cpu())
            cacc[:] = blob["cacc"]
        if tclip and blob.get("tclip"):
            tc_mean.update(blob["tclip"])
        step0, acc, nacc = blob["step"], blob["acc"], blob["nacc"]
        nskip = nskip_log = blob.get("nskip") or 0
        t0 -= blob["seconds"]
        log(stage, step0, resumed_from=str(ck))
    for step in range(step0, steps):
        for pg in opt.param_groups:
            pg["lr"] = 0.0 if step < pw else lr_at(step - pw, steps - pw, peak, warm, cfg["lr_min_frac"])
        if cmode is not None and "overhead" in cmode:   # 2026-10-17: a new CUDA-graph generation per step
            torch.compiler.cudagraph_mark_step_begin()
        if lock:   # 2026-10-17 fastt: the one-step, jump, grid and chain terms in fewer, larger calls (lock_terms)
            loss, terms, gstate, ch = lock_terms(lfns, crit, train, cs, cfg, step, gen, pre_gen, mu, sd, rep, unres,
                                                 (step + 1) % 50 == 0, pen if phys else None)
        elif stage == 1 or K == 1:  # one-step (teacher forcing); K == 1 in stage 2 = the one-step continuation control
            z, dtn, g, widx = train.sample(cfg["batch"], 1, gen, cfg["steady_frac"])
            z0 = z[:, 0]
            if cfg["input_noise"] > 0:
                z0 = z0 + cfg["input_noise"] * torch.randn(z0.shape, device=dev, dtype=z0.dtype, generator=gen)
            d_true = (z[:, 1] - z0).float()
            d_pred = delta_fn(z0.float(), dtn, g)
            if unres:
                loss = unres_loss(crit, d_pred.unsqueeze(1), d_true.unsqueeze(1), z[:, 1:2], (z0 + d_pred.double()).unsqueeze(1),
                                  train.lnd[widx], mu, sd, cfg)
            else:
                loss = crit(d_pred.unsqueeze(1), d_true.unsqueeze(1))["loss_total"]
            if phys:
                loss = loss + pen(0, z0.float() + d_pred, z[:, 1], z0, z[:, 0])
        else:
            k_train = min(K, cfg["s2_K_start"] + (K - cfg["s2_K_start"] + 1) * step // max(1, cfg["s2_ramp"]))
            zt0 = None   # truth at the state entering the scored steps when that state is the model's (long unroll)
            if cfg["s2_long_unroll"] > 0:
                z, dtn, g, zt0 = long_unroll_batch(delta_fn, train, cfg, gen, k_train)
            else:
                z, dtn, g, _ = train.sample(_s2_batch(cfg), k_train, gen, cfg["steady_frac"])
            z0 = None
            if cfg["input_noise"] > 0:
                z0 = z[:, 0] + cfg["input_noise"] * torch.randn(z[:, 0].shape, device=dev, dtype=z.dtype, generator=gen)
            keep = [] if phys else None
            loss = rollout_loss(delta_fn, crit, z, dtn, g, k_train, skip, cfg["s2_detach"], cfg["s2_gamma"], z0, keep)["loss_total"]
            if phys:
                zin, pred = keep
                tz = z if zt0 is None else torch.cat([zt0.unsqueeze(1), z[:, 1:]], dim=1)
                inp = torch.cat([zin.unsqueeze(1), pred[:, :-1]], dim=1)
                loss = loss + pen(0, pred, tz[:, 1 + skip:1 + k_train], inp, tz[:, skip:k_train])
        if not lock:
            terms = [("main", 1.0, loss)]   # (name, weight, unweighted term): loss = sum of weight x term (term_clip, gnorm_every)
            gstate = {}
        if not lock and cfg["jump_w"] > 0 and stage in cfg["jump_stages"]:
            if cfg["head"] in ("delta",):
                raise ValueError("the long-jump loss needs a head whose network does not see dt")
            j0, j1, jdt, jg, jidx = D.sample_jump_pairs(train, cfg["jump_batch"], gen, cfg["jump_min_s"], cfg["jump_max_s"], True)
            jp = j0 + delta_fn(j0.float(), jdt, jg).double()
            if unres:
                jl = unres_loss(crit, jp.unsqueeze(1), j1.unsqueeze(1), j1.unsqueeze(1), jp.unsqueeze(1), train.lnd[jidx], mu, sd, cfg)
            else:
                jl = crit(jp.unsqueeze(1), j1.unsqueeze(1))["loss_total"]
            if phys:
                jl = jl + pen(2, jp, j1, j0, j0)
            loss = loss + cfg["jump_w"] * jl
            terms.append(("jump", cfg["jump_w"], jl))
        if not lock and cfg["grid_w"] > 0 and stage in cfg["grid_stages"]:  # E13: train the flow on the stored grid (Fable review)
            if cfg["head"] in ("delta",):
                raise ValueError("the grid flow term needs a head whose network does not see dt")
            gz, gdtn, gg, gidx, Kg, nsub, gw, gsw, gstate = grid_draw(train, cfg, step, gen)
            c0 = Kg * nsub - cfg["grid_bptt"] if cfg["grid_bptt"] > 0 else 0   # calls before c0 run without gradient
            zs, preds, c = gz[:, 0], [], 0
            for k in range(Kg):
                for _ in range(nsub):
                    with torch.set_grad_enabled(c >= c0):
                        zs = zs + delta_fn(zs.float(), gdtn[:, k].contiguous(), gg).double()
                    zs = row_clip(zs, rcs["grid"])   # 2026-10-18 gfix (no-op when off or without gradient)
                    c += 1
                preds.append(zs)
            gp = torch.stack(preds, 1)
            loss = grid_loss(crit, cfg, train, gz, gp, gidx, mu, sd, unres, gw, gsw, loss, terms)
        if chain:
            cl, tail, n_pre, kc = ch if lock else chain_term(delta_fn, crit, cs, cfg, step, pre_gen, mu, sd, rep, unres,
                                                             (step + 1) % 50 == 0, rcs["chain"])
            loss = loss + cfg["chain_w"] * cl
            terms.append(("chain", cfg["chain_w"], cl))
            if cfg["chain_tail_w"] > 0:
                loss = loss + cfg["chain_tail_w"] * tail
                terms.append(("chain_tail", cfg["chain_tail_w"], tail))
            cacc[3] += n_pre
            if (step + 1) % 50 == 0:
                cacc[0] += float(cl.detach())
                cacc[1] += float(tail.detach())
                cacc[2] += 1
        if fp:
            pick = fp_cand[torch.randint(0, fp_cand.numel(), (cfg["fp_batch"],), device=dev, generator=fp_gen)]
            zc = train.z_of(train.logy[pick, train.end[pick]]).float()
            q, _ = model.parts(zc, train.g[pick])
            fl = crit(q.unsqueeze(1), zc.unsqueeze(1))["loss_total"]
            loss = loss + cfg["fp_w"] * fl
            terms.append(("fp", cfg["fp_w"], fl))
        if cfg["semigroup_w"] > 0:
            sg = semigroup_term(delta_fn, z[:, 0], dtn, g)
            loss = loss + cfg["semigroup_w"] * sg
            terms.append(("semigroup", cfg["semigroup_w"], sg))
        opt.zero_grad(set_to_none=True)
        gn = None
        if stage == 2 and (tclip or (cfg["gnorm_every"] > 0 and (step + 1) % cfg["gnorm_every"] == 0)):
            gn, gsum, used = {}, None, [False] * len(params)   # per-term gradients (one backward each)
            for i, (name, w, t) in enumerate(terms):
                # graphs are kept for the full backward that follows (gnorm only) or for a next term on the same graph
                keep = not tclip or (i + 1 < len(terms) and terms[i + 1][0] in ("grid_sq", "chain_tail"))
                gs = torch.autograd.grad(t, params, retain_graph=keep, allow_unused=True)
                used = [u or q_ is not None for u, q_ in zip(used, gs)]
                gs = [torch.zeros_like(p) if q_ is None else q_ for p, q_ in zip(params, gs)]
                n = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(q_) for q_ in gs])))
                gn[f"gnorm_{name}"] = n
                if tclip:
                    m_ = tc_mean.get(name)
                    s_ = cfg["term_clip"] * m_ / n if m_ is not None and n > cfg["term_clip"] * m_ else 1.0
                    if math.isfinite(n):
                        tc_mean[name] = n if m_ is None else TCLIP_DECAY * m_ + (1 - TCLIP_DECAY) * min(n, cfg["term_clip"] * m_)
                    gn[f"tclip_{name}"] = s_
                    if gsum is None:
                        gsum = [q_ * (w * s_) for q_ in gs]
                    else:
                        torch._foreach_add_(gsum, gs, alpha=w * s_)
            if tclip:
                for p, q_, u in zip(params, gsum, used):
                    p.grad = q_ if u else None   # unused parameters keep no gradient, as after loss.backward()
        if not tclip:
            loss.backward()
        if cfg["clip_norm_fp64"]:   # 2026-10-18 gfix: as clip_grad_norm_ with the norm in fp64
            gl_ = [p_.grad for p_ in params if p_.grad is not None]
            tn = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(q_, dtype=torch.float64) for q_ in gl_]))
            torch._foreach_mul_(gl_, torch.clamp(cfg["clip"] / (tn + 1e-6), max=1.0).float())
            gnl.append((min(1 + step % cfg["grid_sub"], _ramp(1, cfg["grid_sub"], cfg["grid_sub_ramp"], step)), float(tn)))
        else:
            tn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        skipped = bool(cfg["skip_nonfinite"]) and not math.isfinite(float(tn))   # 2026-10-17 fastt (a host sync per step)
        if skipped:
            nskip += 1
            log(stage, step + 1, nonfinite_skip=nskip, gnorm_total=float(tn))
        if gn is not None and (cfg["gnorm_every"] > 0 and (step + 1) % cfg["gnorm_every"] == 0):
            log(stage, step + 1, gnorm_total=float(tn), lr=opt.param_groups[0]["lr"], **gn, **gstate)
        if not skipped:
            opt.step()
            ema.update(model)
            for e in extras:
                e.update(model)
        acc += float(loss.detach()) if (step + 1) % cfg["loss_every"] == 0 and not skipped else 0.0
        nacc += 1 if (step + 1) % cfg["loss_every"] == 0 and not skipped else 0
        if (step + 1) % cfg["log_every"] == 0:
            extra = {}
            if phys:
                extra = {k: v / max(1, nacc) for k, v in zip(("phys_sum", "phys_elem", "phys_sum_jump", "phys_elem_jump"), pacc)}
                pacc[:] = [0.0] * 4
            if chain:
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
            if gnl:   # 2026-10-18 gfix (clip_norm_fp64)
                a_ = np.array([v for _, v in gnl])
                extra.update(gnorm_p50=float(np.median(a_)), gnorm_max=float(a_.max()), gnorm_gt1e3=int((a_ > 1e3).sum()),
                             gnorm_p50_nsub=[float(np.median([v for n_, v in gnl if n_ == k])) if any(n_ == k for n_, _ in gnl)
                                             else None for k in range(1, cfg["grid_sub"] + 1)])
                gnl.clear()
            for k_, rc in rcs.items():   # 2026-10-18 gfix (row_clip): clipped (row, call) pairs / all since the last row
                if rc is not None:
                    extra[f"rowclip_{k_}"] = int(rc[1]) / max(1, rc[2])
                    rc[1].zero_()
                    rc[2] = 0
            if dev.type == "cuda":   # 2026-10-13: peak allocated memory since the start (log only)
                extra["peak_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 1e9
            log(stage, step + 1, train_loss=acc / max(1, nacc), lr=opt.param_groups[0]["lr"],
                s_per_step=(time.time() - t0) / (step + 1), **extra)
            if not math.isfinite(acc / max(1, nacc)):
                sys.exit(f"stage {stage} step {step + 1}: non-finite train loss; stopped (last checkpoint kept)")
            acc, nacc = 0.0, 0
        if (step + 1) % cfg["val_every"] == 0 or step + 1 == steps:
            vK = 1 if stage == 1 else K
            vl, vp = validate(ema.shadow, vy, vdt, vg, crit, vK, 0 if stage == 1 else skip)
            fz = getattr(model, "fused", False)   # 2026-10-17 fastt (fuse review issue 3): eager, so the unfused projection
            if fz:
                model.fused = False
            try:
                vlive, _ = validate(model, vy, vdt, vg, crit, vK, 0 if stage == 1 else skip)
            finally:
                if fz:
                    model.fused = True
            log(stage, step + 1, val_loss=vl, val_loss_live=vlive, val_persistence=vp,
                **({} if pr is None else path_readout(ema.shadow, pr)))
        if cfg["ckpt_every"] > 0 and (step + 1) % cfg["ckpt_every"] == 0 and step + 1 < steps:
            if not finite(model, ema.shadow):
                sys.exit(f"stage {stage} step {step + 1}: non-finite weights; stopped (last checkpoint kept)")
            save_atomic({"model": model.state_dict(), "ema": ema.shadow.state_dict(), "opt": opt.state_dict(),
                         "ema_extra": [e.shadow.state_dict() for e in extras],
                         "gen": gen.get_state(), "fp_gen": fp_gen.get_state() if fp else None,
                         "pacc": list(pacc) if phys else None, "chain_gen": cs.gen.get_state() if chain else None,
                         "pre_gen": pre_gen.get_state() if chain else None, "cacc": list(cacc) if chain else None,
                         "tclip": dict(tc_mean) if tclip else None, "step": step + 1, "acc": acc, "nacc": nacc, "nskip": nskip,
                         "seconds": time.time() - t0, "cfg": cfg, "stage": stage}, ck)
    if not finite(model, ema.shadow):
        sys.exit(f"stage {stage}: non-finite weights at the end; stage file not written")
    for d, e in zip(cfg["ema_extra"], extras):   # before stage<N>.pt, whose existence the queues read as "done"
        if finite(model, e.shadow):
            save_atomic({"state_dict": e.shadow.state_dict(), "cfg": cfg, "stage": stage, "ema": d}, run / f"stage{stage}_ema{d}.pt")
    save_atomic({"state_dict": ema.shadow.state_dict(), "cfg": cfg, "stage": stage,   # 2026-10-19: atomic, as the checkpoints
                 "seconds": time.time() - t0}, run / f"stage{stage}.pt")
    if ck.exists():
        ck.unlink()   # the stage file supersedes the checkpoint
    model.load_state_dict(ema.shadow.state_dict())  # stage 2 starts from the EMA weights (production)
    return time.time() - t0


@torch.no_grad()
def _advance(delta_fn, zs, dtn, g, n):
    for _ in range(n):
        zs = zs + delta_fn(zs.float(), dtn, g).double()
    return zs


def _s2_batch(cfg):
    return cfg["s2_batch"] or cfg["batch"]


def long_unroll_batch(delta_fn, train, cfg, gen, k_train):
    """X2 (unroll long, backprop short): run the model n_pre no-grad steps from the window start at
    the window's dt, then score the next k_train steps against the truth at those times. The
    windows are drawn to fit n_pre + k_train <= 99 transitions (the production fit rule). Also returns the truth
    at the model state that enters the scored steps."""
    n_pre = int(cfg["s2_long_unroll"])
    if cfg["s2_long_random"]:
        n_pre = int(torch.randint(0, n_pre + 1, (1,), device=train.device, generator=gen))
    z, dtn, g, idx = train.sample(_s2_batch(cfg), n_pre + k_train, gen, cfg["steady_frac"])
    zs = _advance(delta_fn, z[:, 0].clone(), dtn, g, n_pre)
    z2 = torch.cat([zs.unsqueeze(1), z[:, n_pre + 1:n_pre + 1 + k_train]], dim=1)
    return z2, dtn, g, z[:, n_pre]


def main(run):
    run = Path(run)
    user = json.loads((run / "config.json").read_text())
    unknown = sorted(set(user) - set(DEFAULTS))
    if unknown:   # 2026-10-19: a key this train.py does not know would be ignored silently (an old run's config belongs to
        sys.exit(f"{run / 'config.json'}: keys not in train.DEFAULTS: {unknown}")   # the code copy in its own folder)
    cfg = {**DEFAULTS, **user}
    assert cfg["cache"] in (None, str(D.CACHE)), f"config cache {cfg['cache']} but ARCH_CACHE {D.CACHE}"
    lib_md5 = {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in sorted(Path(__file__).resolve().parent.glob("*.py"))}
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True   # training as production (TF32 on); the benchmark turns it off
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device(os.environ.get("HARNESS_DEVICE", "cuda"))  # E10: CPU training via HARNESS_DEVICE=cpu
    D.DT_LOG = (math.log10(cfg["dt_min_s"]), math.log10(cfg["dt_max_s"]))
    D.ANCHOR_P = cfg["anchor_p"]
    D.X0_ANCHOR_P, D.X0_JUMP_P = cfg["x0_anchor_p"], cfg["x0_jump_p"]
    train = D.RawSplit("train", dev, frac=cfg["train_frac"], frac_seed=cfg["train_frac_seed"],
                       rows=None if cfg["train_rows"] is None else np.load(cfg["train_rows"]))
    if cfg["traj_w"] is not None:
        train.w = torch.where(torch.as_tensor(train.T, dtype=torch.float64) < cfg["traj_w"][0], float(cfg["traj_w"][1]), 1.0).to(dev)
    if cfg["traj_weights_file"] is not None:
        assert cfg["traj_w"] is None, "traj_w and traj_weights_file are exclusive"
        w = np.load(cfg["traj_weights_file"]).astype(np.float64)
        assert w.shape == (train.n,) and np.all(w >= 0) and w.sum() > 0, (w.shape, train.n)
        train.w = torch.from_numpy(w).float().to(dev)   # fp32 as traj_w's weights (same draws for the same weights)
    if cfg["x0_file"] is not None:
        files = json.loads((D.CACHE / "meta.json").read_text())["files"]
        xf = cfg["x0_file"] if isinstance(cfg["x0_file"], list) else [cfg["x0_file"]]  # one name or a list
        train.x0 = torch.from_numpy(np.isin(train.file_idx, [files.index(f) for f in xf])).to(dev)
        assert bool(train.x0.any()), cfg["x0_file"]
    mu, sd, zlo, zhi = D.norm_consts(dev)
    model = M.build(cfg["head"], cfg["hidden"], zlo, zhi, newton=cfg["closed_iters"], fallback=cfg["closed_fallback"],
                    fused=cfg["fused_proj"]).to(dev)
    assert cfg["val_source"] in ("shards", "cache"), cfg["val_source"]
    if len(D.SPECIES) == 12 and cfg["val_source"] == "shards":
        vy, vdt, vg = D.load_val_shards(cfg["val_windows"], cfg["s2_K"] + 1, dev)
    else:  # E13: no production shards for this species set, or val_source "cache" (2026-10-09)
        vy, vdt, vg = D.cache_val(cfg["val_windows"], cfg["s2_K"] + 1, dev)
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
    for stage, todo in ((1, not cfg["skip_s1"] and not cfg["init_from"]), (2, cfg["s2_steps"] > 0)):
        if not todo:
            continue
        done = run / f"stage{stage}.pt"
        if cfg["resume"] and done.exists():  # 2026-10-09: finished before the restart; stage 2 starts from its EMA weights
            blob = torch.load(done, map_location=dev, weights_only=False)
            model.load_state_dict(blob["state_dict"])
            secs[f"s{stage}"] = blob["seconds"]
            log(stage, None, resumed_from=str(done))
            continue
        secs[f"s{stage}"] = run_stage(stage, cfg, model, train, vy, vdt, vg, mu, sd, run, log, pr)
    (run / "done.json").write_text(json.dumps({"seconds": secs, **info, "arch_cache": str(D.CACHE), "lib_md5": lib_md5}))
    write_manifest(run, cfg, code_info(lib_md5))   # 2026-10-19


if __name__ == "__main__":
    main(sys.argv[1])
