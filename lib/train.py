"""Train the closed relaxation head in two stages on a GPU-resident cache.

    python train.py <run_dir>     (run_dir/config.json over DEFAULTS; writes stage1.pt / stage2.pt, log.csv, model_info.json,
                                   done.json, manifest.json and norm.json)

Stage 1 (s1_*): one-step windows from the cache's training trajectories (data.RawSplit.sample: dt log-uniform in
[dt_min_s, dt_max_s], start anchored at the first stored time with probability anchor_p, optional row-0 starts with a
probability per cache file (x0_anchor, x0_jump; row0_probs), optional per-trajectory draw weights traj_weights_file), loss
on the increment.
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
(lhbench.run_paths) on a paths folder, scored in its late window. t0_readout: at every validation, the EMA model rolled
from row 0 (t = 0) of the validation rows of the t0_readout_files, 10 calls at dt 10 s and 100 s, against the stored
trajectory (t0_setup, t0_readout).
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
import scorerule as SR  # noqa: E402
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
    "x0_anchor": None,          # {cache file name (meta.json 'files'): p}: share of that file's one-step windows that start
    "x0_jump": None,            # at row 0 (t = 0); x0_jump likewise for its jump pairs (row0_probs; other files: 0)
    "x0_file": None,            # older form, exclusive with x0_anchor / x0_jump: raw file name(s) whose rows may start at
    "x0_anchor_p": 0.0,         # row 0, each with probability x0_anchor_p (one-step windows) and x0_jump_p (jump pairs)
    "x0_jump_p": 0.0,
    "ckpt_every": 0,            # every n steps save run/ckpt_stage<N>.pt (atomic); removed once stage<N>.pt is written. 0 = off
    "resume": False,            # skip a stage whose stage<N>.pt exists, continue one from its ckpt_stage<N>.pt
    "val_source": "cache",      # validation windows from the cache's validation split (data.cache_val; the only source)
    "path_readout": None,       # a paths folder: at every validation, the EMA model's arm-P rollout scored (path_readout)
    "t0_readout": False,        # at every validation, the EMA model from row 0 of the validation rows of the cache files
    "t0_readout_files": [],     # named here (meta.json 'files'), scored against their stored trajectories (t0_readout;
                                # log key t0_readout, written only when on)
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
    "s1_init_from": None,       # a stage file to start stage 1 from (a fine-tune; exclusive with init_from). A dt-aware head
                                # (relaxclosedrdt) started from a head without the dt input gets that input's weight
                                # columns at zero, so it starts as the same function (load_init)
    "miss_w": 0.0, "miss_tau": 0.1, "miss_c": 1.0,   # stage 1: + miss_w x the mean over windows and the 12 scored gases
                                # of phi(relu(a - miss_tau)), a = Misses.resid (dex), phi(x) = x^2 / (2 miss_c) up to
                                # miss_c, then x - miss_c / 2: every miss gets a gradient that grows with it up to a
                                # bounded slope (a few huge misses cannot own the batch)
    "frac_w": 0.0, "frac_cap": 2.0,  # stage 1: + frac_w x the fractional error summed over a trajectory's calls and the 12
                                # scored gases, averaged over trajectories: per point 10^a - 1 (a = Misses.resid, dex),
                                # linear past frac_cap dex with the same slope (one-step windows: one call; rollouts:
                                # roll_K calls)
    "cvar_w": 0.0, "cvar_q": 0.05,    # stage 1: + cvar_w x the mean of the top cvar_q fraction of the windows' relu(worst a -
                                # miss_tau) (worst = max over the 12 gases)
    "roll_w": 0.0,              # stage 1: + roll_w x the loss of constant-condition rollouts, summed over their calls
    "roll_batch": 512,          # (roll_draw): roll_batch runs, each from row 0 (share roll_t0_frac) or a log-uniform age in
    "roll_K": 8,                # [1, 1e3] s, one dt each (log-uniform in data.DT_LOG); n_pre uniform in 0 to roll_pre_max
    "roll_pre_max": 32,         # calls without gradient, then roll_K calls with gradient, each against the stored trajectory
    "roll_t0_frac": 0.5,        # at that time (HybridLoss on the states + the miss terms when on); roll_row_clip as
    "roll_row_clip": 0.0,       # grid_row_clip, after every call with gradient
    "roll_pre0_frac": 0.0,      # > 0: n_pre = 0 (the rollout's first calls supervised) with this probability per step, else
                                # uniform in 1 to roll_pre_max (0: n_pre uniform in 0 to roll_pre_max, the original draw)
    "roll_pre_long_frac": 0.0,  # > 0: n_pre uniform in roll_pre_max + 1 to roll_pre_long_max with this probability per
    "roll_pre_long_max": 256,   # step (after the roll_pre0_frac share), so rollouts also train at states reached late
    "roll_late_frac": 0.0,      # > 0: each rollout run instead starts, with this probability (its own generator, not
    "roll_late_tmin": 1e4,      # checkpointed, like the rollout generators), at an age log-uniform in [roll_late_tmin,
                                # t_end - (n_pre + roll_K) x dt_max] s of its trajectory, so rollouts train at settled states
    "stat_w": 0.0,              # stage 1: + stat_w x the late one-call term (stat_term): stat_batch runs, dt log-uniform in
    "stat_batch": 2048,         # data.DT_LOG, a time log-uniform in [stat_tmin, t_end - dt]; per resolved non-He entry the
    "stat_tmin": 1e4,           # signed residual of the call's change in log10 y, r = (d_pred - d_true) / (stat_eps +
    "stat_eps": 1e-6,           # |d_true|) (dex; the projected update), Huber with transition stat_huber; the mean over the
    "stat_huber": 1.0,          # entries whose truth moves less than stat_eps + stat_moving_w x the mean over those that move
    "stat_moving_w": 0.0,       # more (own generator, not checkpointed; own compiled call; per-term gradient norms logged)
    "stat_ramp": 0,             # > 0: stat_w ramps linearly from 0 over the stage's first stat_ramp steps
    "stat_eps_dt": False,       # true: the floor is stat_eps x dt / dt_max, at least stat_eps_min (a per-time tolerance)
    "stat_eps_min": 3e-7,
    "stat_moving_scored": False,   # true: the moving mean over the 12 scored gases only
    "s1_jump_w": 0.0,           # stage 1: + s1_jump_w x the long-jump loss of jump_batch pairs (data.sample_jump_pairs, D in
                                # [jump_min_s, jump_max_s], as stage 2's jump term; own generator, not checkpointed, and
                                # own compiled call)
    "s2_joint": False,          # stage 2 also carries the miss / fractional terms (miss_w, frac_w) on its one-step windows
                                # and the rollout term (roll_*; stage 1's generators, not checkpointed), so the short-run and
                                # long-run terms train together
    "fm_latent": 512,           # the flow-map head's latent width (head relaxclosedfm)
    "s1_targets": 1,            # > 1: stage-1 one-step windows as batch / s1_targets anchor states with s1_targets end
                                # points each, one dt per equal log-width slice of data.DT_LOG (multi_targets)
    "resample_every": 0,        # stage 1: every n steps, per-trajectory draw weights: (1 - resample_alpha) uniform +
    "resample_alpha": 0.0,      # resample_alpha in proportion to clamp(e / miss_tau - 1, 0, resample_cap), e = the live
    "resample_cap": 10.0,       # model's worst scored error on two windows of each training run (from row 0, and from a
                                # log-uniform start in 1e-3..1e3 s; dt as the one-step windows; own generator)
                                # (miss_weights). 0 = off
    "log_misses": False,        # log rows add the sampled batch's share of one-step windows with a scored miss (> miss_tau)
                                # and a large miss (> 1 dex), its worst error, and its share of rows whose projection failed
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


class Misses:
    """Errors on the 12 scored gases of z states and the miss terms built on them (miss_w, cvar_w). zp, zt [B, S] fp64 z
    states, lnd [B] log10 number density (cm^-3).
    err: the score's per-gas error [B, 12] (dex), 0 where a point does not count, as scorerule.err with SCORE_RULE phys (for
    logs and the draw weights).
    resid: the training residual [B, 12] (dex): |p - t| where the truth t is at or above the gas's floor S, max(p - S, 0)
    where it is below, 0 on unresolved points (mask without gradient). Unlike err it keeps a gradient when a gas that is
    present is predicted below its floor (err is flat there)."""

    def __init__(self, cfg, mu, sd, dev):
        self.cols = torch.tensor([D.SHORT.index(s) for s in C.REPORTED], device=dev)
        self.mu, self.sd = mu[self.cols], sd[self.cols]
        self.floor = torch.tensor([SR.PHYS[s] for s in C.REPORTED], dtype=torch.float64, device=dev)
        self.w, self.tau, self.c, self.cw, self.q = cfg["miss_w"], cfg["miss_tau"], cfg["miss_c"], cfg["cvar_w"], cfg["cvar_q"]
        self.fw, self.fcap = cfg["frac_w"], cfg["frac_cap"]
        self.on = self.w > 0 or self.cw > 0 or self.fw > 0   # any miss-aware loss term

    def err(self, zp, zt, lnd):
        lp = zp[:, self.cols] * self.sd + self.mu
        lt = zt[:, self.cols] * self.sd + self.mu
        res = ((lt + lnd[:, None]) >= -2.0) | ((lp + lnd[:, None]) >= -2.0)
        cnt = res & ((lt >= self.floor) | (lp >= self.floor))
        return torch.where(cnt, (torch.maximum(lp, self.floor) - torch.maximum(lt, self.floor)).abs(), torch.zeros_like(lp))

    def resid(self, zp, zt, lnd):
        lp = zp[:, self.cols] * self.sd + self.mu
        lt = zt[:, self.cols] * self.sd + self.mu
        with torch.no_grad():
            res = ((lt + lnd[:, None]) >= -2.0) | ((lp + lnd[:, None]) >= -2.0)
        a = torch.where(lt >= self.floor, (lp - lt).abs(), torch.relu(lp - self.floor))
        return torch.where(res, a, torch.zeros_like(a))

    def loss(self, zp, zt, lnd, calls=1):
        """The miss-aware terms on rows of states [B, S]; calls = the calls per trajectory when the rows are a
        trajectory's calls in order (rollouts, call-minor): the fractional term is then summed over each trajectory."""
        a = self.resid(zp, zt, lnd)
        out = zp.new_zeros(())
        if self.fw > 0:
            ac = torch.clamp(a, max=self.fcap)
            f = torch.pow(10.0, ac) - 1.0 + math.log(10.0) * 10.0 ** self.fcap * torch.relu(a - self.fcap)
            out = out + self.fw * calls * f.sum(-1).mean()
        if self.w > 0:
            x = torch.relu(a - self.tau)
            out = out + self.w * torch.where(x <= self.c, x.square() / (2.0 * self.c), x - 0.5 * self.c).mean()
        if self.cw > 0:
            worst = torch.relu(a.amax(-1) - self.tau)
            out = out + self.cw * torch.topk(worst, max(1, math.ceil(self.q * worst.numel()))).values.mean()
        return out


@torch.no_grad()
def miss_weights(model, train, ms, cfg, gen, bs=16384):
    """resample_every: per-trajectory draw weights from the live model's worst scored error on two one-step windows of
    every training run, one from row 0 (t = 0) and one from a log-uniform start in [1e-3, 1e3] s, each at a log-uniform
    dt in data.DT_LOG (eager calls, the training projection). Returns weights fp32 [n] from m = clamp(e /
    miss_tau - 1, 0, resample_cap) mixed with uniform weights: (1 - resample_alpha) / n + resample_alpha m / sum m (uniform
    if no run misses), and the share of runs with e > miss_tau."""
    dev, n = train.device, train.n
    e = torch.empty(n, dtype=torch.float64, device=dev)
    lo, hi = D.DT_LOG
    for i in range(0, n, bs):
        idx = torch.arange(i, min(n, i + bs), device=dev)
        B = idx.numel()
        dt0, dt1 = (torch.pow(10.0, lo + (hi - lo) * torch.rand(B, device=dev, generator=gen, dtype=torch.float64))
                    for _ in range(2))
        ts = torch.pow(10.0, -3.0 + 6.0 * torch.rand(B, device=dev, generator=gen, dtype=torch.float64))
        ts = torch.minimum(ts, train.t[train.end[idx]] - dt1)
        z0 = train.z_of(train.logy[idx, 0])
        y = train.logy_at(idx, torch.stack([dt0, ts, ts + dt1], 1))
        zz = train.z_of(y)
        g, lnd = train.g[idx], train.lnd[idx]
        p0 = z0 + model.delta(z0.float(), D.dt_norm_of(dt0).float(), g).double()
        p1 = zz[:, 1] + model.delta(zz[:, 1].float(), D.dt_norm_of(dt1).float(), g).double()
        e[i:i + B] = torch.maximum(ms.err(p0, zz[:, 0], lnd).amax(-1), ms.err(p1, zz[:, 2], lnd).amax(-1))
    m = torch.clamp(e / cfg["miss_tau"] - 1.0, 0.0, cfg["resample_cap"])
    al = cfg["resample_alpha"] if float(m.sum()) > 0 else 0.0
    w = (1.0 - al) / n + al * m / torch.clamp(m.sum(), min=1e-300)
    return w.float(), float((e > cfg["miss_tau"]).double().mean())


def roll_draw(train, cfg, gen, pre_gen, late_gen=None):
    """The rollout term's draw: roll_batch runs (uniform, or train.w when set), start age 0 (row 0) with probability
    roll_t0_frac else log-uniform in [1, 1e3] s, one dt per run (log-uniform in data.DT_LOG), n_pre uniform in 0 to
    roll_pre_max on the host generator pre_gen; with late_gen, a run starts instead with probability roll_late_frac at an
    age log-uniform in [roll_late_tmin, t_end - (n_pre + roll_K) x dt_max] of its trajectory (draws on late_gen only).
    Returns z0 [B, S] (the state at the start age), the truth after calls n_pre + 1 to n_pre + roll_K [B, K, S]
    (RawSplit.logy_at), dt_norm [B], g [B, G], lnd [B] and n_pre."""
    dev, B, K = train.device, cfg["roll_batch"], cfg["roll_K"]
    if cfg["roll_pre0_frac"] > 0 or cfg["roll_pre_long_frac"] > 0:
        r = float(torch.rand(1, generator=pre_gen))
        if r < cfg["roll_pre0_frac"]:
            n_pre = 0
        elif r < cfg["roll_pre0_frac"] + cfg["roll_pre_long_frac"]:
            n_pre = int(torch.randint(cfg["roll_pre_max"] + 1, cfg["roll_pre_long_max"] + 1, (1,), generator=pre_gen))
        else:
            n_pre = int(torch.randint(1, cfg["roll_pre_max"] + 1, (1,), generator=pre_gen))
    else:
        n_pre = int(torch.randint(0, cfg["roll_pre_max"] + 1, (1,), generator=pre_gen))
    idx = train.draw(B, gen)
    lo, hi = D.DT_LOG
    dt = torch.pow(10.0, lo + (hi - lo) * torch.rand(B, device=dev, generator=gen, dtype=torch.float64))
    age = torch.pow(10.0, 3.0 * torch.rand(B, device=dev, generator=gen, dtype=torch.float64))
    at0 = torch.rand(B, device=dev, generator=gen) < cfg["roll_t0_frac"]
    if late_gen is not None:
        late = torch.rand(B, device=dev, generator=late_gen) < cfg["roll_late_frac"]
        lo_t = math.log10(cfg["roll_late_tmin"])
        hi_t = torch.log10(train.t[train.end[idx]].double() - (n_pre + K) * 10.0 ** hi)
        assert bool((hi_t > lo_t).all()), "roll_late_tmin leaves no room before a trajectory's end"
        age = torch.where(late, torch.pow(10.0, lo_t + (hi_t - lo_t) * torch.rand(B, device=dev, generator=late_gen,
                                                                                    dtype=torch.float64)), age)
        at0 = at0 & torch.logical_not(late)
    age = torch.where(at0, torch.zeros_like(age), age)
    k = torch.arange(n_pre + 1, n_pre + K + 1, device=dev, dtype=torch.float64)
    zt = train.z_of(train.logy_at(idx, age[:, None] + dt[:, None] * k[None]))
    z0 = train.z_of(train.logy_at(idx, torch.clamp(age, min=float(train.t[1]))[:, None].contiguous()))[:, 0]
    z0 = torch.where(at0[:, None], train.z_of(train.logy[idx, 0]), z0)
    return z0, zt, D.dt_norm_of(dt).float(), train.g[idx], train.lnd[idx], n_pre


def stat_term(sfn, train, cfg, gen, mu, sd, scol, sfloor, he):
    """The late one-call term (stat_w): stat_batch runs (train.draw), dt log-uniform in data.DT_LOG, a time t log-uniform
    in [stat_tmin, t_end - dt] of the run; z(t) and z(t + dt) from RawSplit.logy_at; one model call sfn (the projected
    update). With stat_eps_dt the floor eps is stat_eps x dt / dt_max (at least stat_eps_min) for the residual and the
    still / moving split; with stat_moving_scored the moving entries are the scored gases only. Entries count where
    resolved (truth or prediction >= 1e-2 cm^-3), the scored gases also where truth or
    prediction is at or above the gas's floor (scorerule.PHYS), He never. r = (d_pred - d_true) / (stat_eps + |d_true|) in
    dex; Huber(r) with transition stat_huber; returns the mean over the counted entries whose truth moves less than
    stat_eps + stat_moving_w x the mean over the others, both means (detached) and diagnostics (the still mean over the
    scored gases, still / moving entries per row, the median |r| of the still entries)."""
    dev, B = train.device, cfg["stat_batch"]
    idx = train.draw(B, gen)
    lo, hi = D.DT_LOG
    dt = torch.pow(10.0, lo + (hi - lo) * torch.rand(B, device=dev, generator=gen, dtype=torch.float64))
    lo_t = math.log10(cfg["stat_tmin"])
    hi_t = torch.log10(train.t[train.end[idx]].double() - dt)
    assert bool((hi_t > lo_t).all()), "stat_tmin leaves no room before a trajectory's end"
    t = torch.pow(10.0, lo_t + (hi_t - lo_t) * torch.rand(B, device=dev, generator=gen, dtype=torch.float64))
    zz = train.z_of(train.logy_at(idx, torch.stack([t, t + dt], 1)))                 # [B, 2, S] fp64
    z0 = zz[:, 0]
    d_true = (zz[:, 1] - z0) * sd                                                     # dex
    d_pred = sfn(z0.float(), D.dt_norm_of(dt).float(), train.g[idx]).double() * sd
    with torch.no_grad():
        lt, lp = zz[:, 1] * sd + mu, z0 * sd + mu + d_pred
        lnd = train.lnd[idx].reshape(-1, 1)
        cnt = ((lt + lnd) >= -2.0) | ((lp + lnd) >= -2.0)
        cnt[:, scol] &= (lt[:, scol] >= sfloor) | (lp[:, scol] >= sfloor)
        cnt[:, he] = False
        still = cnt & (d_true.abs() < (cfg["stat_eps"] if not cfg["stat_eps_dt"] else torch.clamp(
            cfg["stat_eps"] * dt / 10.0 ** hi, min=cfg["stat_eps_min"]).reshape(-1, 1)))
        moving = cnt & torch.logical_not(still)
        if cfg["stat_moving_scored"]:
            sc = torch.zeros_like(moving)
            sc[:, scol] = True
            moving = moving & sc
    eps = cfg["stat_eps"]
    if cfg["stat_eps_dt"]:
        eps = torch.clamp(cfg["stat_eps"] * dt / 10.0 ** hi, min=cfg["stat_eps_min"]).reshape(-1, 1)
    r = (d_pred - d_true) / (eps + d_true.abs())
    h, a = cfg["stat_huber"], r.abs()
    hub = torch.where(a <= h, 0.5 * r * r, h * (a - 0.5 * h))
    ls = (hub * still).sum() / still.sum().clamp(min=1)
    lm = (hub * moving).sum() / moving.sum().clamp(min=1)
    with torch.no_grad():   # diagnostics: the still mean over the scored gases, and the counted entries per row
        scm = torch.zeros_like(still)
        scm[:, scol] = True
        diag = {"still_scored": (hub * (still & scm)).sum() / (still & scm).sum().clamp(min=1),
                "n_still": still.sum() / B, "n_moving": moving.sum() / B,
                "med": a[still].median() if bool(still.any()) else a.new_zeros(())}
    return ls + cfg["stat_moving_w"] * lm, ls.detach(), lm.detach(), diag


def multi_targets(train, cfg, gen):
    """s1_targets > 1: the stage-1 one-step windows as batch / T anchor states (T = s1_targets) with T end points each,
    one dt per equal log-width slice of data.DT_LOG (uniform in log within it). Anchor starts as RawSplit.sample draws
    them: row 0 with the run's p0_anchor probability, else the first positive stored time with probability
    data.ANCHOR_P, else log-uniform in [t[1], t_end - dt_max]. Labels from RawSplit.logy_at. Returns z [B, 2, S] (anchor,
    end point), dt_norm [B], g [B, G], idx [B], anchor-major (the T end points of one anchor are adjacent)."""
    dev, T = train.device, cfg["s1_targets"]
    assert cfg["batch"] % T == 0, f"batch {cfg['batch']} is not a multiple of s1_targets {T}"
    m = cfg["batch"] // T
    idx = train.draw(m, gen)
    lo, hi = D.DT_LOG
    t1 = train.t[1]
    thi = torch.clamp(train.t[train.end[idx]] - 10.0 ** hi, min=float(t1) * (1 + 1e-12))
    u = torch.rand(m, device=dev, generator=gen, dtype=torch.float64)
    ts = torch.pow(10.0, torch.log10(t1) + u * (torch.log10(thi) - torch.log10(t1)))
    ts = torch.where(torch.rand(m, device=dev, generator=gen) < D.ANCHOR_P, t1.expand(m), ts)
    a0 = torch.zeros(m, dtype=torch.bool, device=dev)
    if train.p0_anchor is not None:
        a0 = torch.rand(m, device=dev, generator=gen) < train.p0_anchor[idx]
        ts = torch.where(a0, torch.zeros_like(ts), ts)
    j = torch.arange(T, device=dev, dtype=torch.float64)
    dt = torch.pow(10.0, lo + (hi - lo) * (j + torch.rand(m, T, device=dev, generator=gen, dtype=torch.float64)) / T)
    y0 = train.logy_at(idx, torch.where(a0, t1.expand(m), ts)[:, None].contiguous())[:, 0]   # row 0: placeholder time
    y0 = torch.where(a0[:, None], train.logy[idx, 0], y0)
    y1 = train.logy_at(idx, ts[:, None] + dt)                                                 # [m, T, S]
    z = torch.stack([train.z_of(y0).repeat_interleave(T, 0), train.z_of(y1).reshape(m * T, -1)], 1)
    return z, D.dt_norm_of(dt.reshape(-1)).float(), train.g[idx].repeat_interleave(T, 0), idx.repeat_interleave(T)


def load_init(model, path, dev):
    """A stage file's weights into model (init_from, s1_init_from). A dt-aware head started from a head without the dt input:
    the first layer's (and its projection's) weight column for the dt input is added at zero, so the start is the same
    function; likewise a mixed-mode head's added outputs from a per-entry or a mixed-mode head."""
    sdict = torch.load(path, map_location=dev)["state_dict"]
    own = model.state_dict()
    if getattr(model, "rate_dt", None) is not None and not any(k.startswith("rate_dt.") for k in sdict):
        sdict.update({k: v for k, v in own.items() if k.startswith("rate_dt.")})   # the zero-initialized rate branch
    if getattr(model, "vmix", False):   # a mixed-mode head from a per-entry or a mixed-mode head: the added outputs at zero
        for k in ("trunk.out.weight", "trunk.out.bias"):   # (V = I, U = I: the start is the stored model)
            if k in sdict and k in own and sdict[k].shape[0] in (2 * M.S, 2 * M.S + M.S * (M.S - 1) // 2) \
                    and sdict[k].shape[0] < own[k].shape[0]:
                sdict[k] = torch.cat([sdict[k], torch.zeros_like(own[k][sdict[k].shape[0]:])], 0)
    if getattr(model, "dt_input", False):
        for k in ("trunk.linears.0.weight", "trunk.projs.0.weight"):
            if k in sdict and k in own and sdict[k].shape[1] + 1 == own[k].shape[1]:
                sdict[k] = torch.cat([sdict[k], torch.zeros_like(sdict[k][:, :1])], 1)
    model.load_state_dict(sdict)


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


T0_DTS, T0_CALLS = (10.0, 100.0), 10   # t0_readout: step sizes (s) and calls per step size
T0_BANDS = {"300-650": (300, 650), "650-1300": (650, 1300), "1300-2000": (1300, 2000), "2000-3000": (2000, 3001)}   # K


def t0_setup(files, dev):
    """t0_readout: the validation rows of the named cache files (meta.json 'files'): their row-0 states z, globals, file,
    T, log10 number density and, per dt in T0_DTS, the 12 reported species' stored trajectory at the calls k dt
    (k = 1..T0_CALLS; the training interpolant RawSplit.logy_at), computed once."""
    val = D.RawSplit("validation", dev)
    names = json.loads((D.CACHE / "meta.json").read_text())["files"]
    rows = np.flatnonzero(np.isin(val.file_idx, [names.index(f) for f in files]))
    if rows.size == 0:
        sys.exit(f"t0_readout: no validation rows in {files}")
    idx = torch.from_numpy(rows).to(dev)
    cols = [D.SHORT.index(s) for s in C.REPORTED]
    k = torch.arange(1, T0_CALLS + 1, dtype=torch.float64, device=dev)
    vt = {dt: val.logy_at(idx, (k * dt).expand(rows.size, -1).contiguous())[..., cols].cpu().numpy() for dt in T0_DTS}
    return {"z0": val.z_of(val.logy[idx, 0]), "g": val.g[idx], "mu": val.mu, "sd": val.sd, "cols": cols, "vt": vt,
            "file": np.asarray(names)[val.file_idx[rows]], "T": val.T[rows], "dev": dev,
            "lnd": np.log10(val.P[rows] / (1.380649e-16 * val.T[rows]))[:, None, None]}


@torch.no_grad()
def t0_readout(model, t0, bs=8192):
    """The model rolled from row 0 of the t0_setup rows, T0_CALLS calls at each dt in T0_DTS (fp64 state z, z += fp64(
    model.delta(z fp32, dt, g)); TF32 off for the rollout, restored after), against the stored trajectory. Row error: the
    max over the calls and the 12 reported species of the per-gas scored error (scorerule.err; resolved: number density
    >= 1e-2 cm^-3; not scored = 0). Per dt, over all rows, per file and per T band (K): rows, p50 / p90 / p99 of the row
    error over the rows whose states stay finite, its max over all rows (inf if a state is not finite), share of rows
    > 0.5 dex; non-finite rows. And its seconds."""
    t = time.time()
    tf = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

    def q(e):
        f = e[np.isfinite(e)]
        p = np.percentile(f, [50, 90, 99]) if f.size else [float("nan")] * 3
        return {"rows": int(e.size), "p50": float(p[0]), "p90": float(p[1]), "p99": float(p[2]),
                "max": float(e.max()) if e.size else float("nan"), "gt05": float((e > 0.5).mean()) if e.size else float("nan")}
    out = {}
    try:
        for dt in T0_DTS:
            sts = []
            for i in range(0, t0["z0"].shape[0], bs):
                z, g = t0["z0"][i:i + bs], t0["g"][i:i + bs]
                dtn = D.dt_norm_of(torch.full((z.shape[0],), dt, dtype=torch.float64, device=t0["dev"])).float()
                st = []
                for _ in range(T0_CALLS):
                    z = z + model.delta(z.float(), dtn, g).double()
                    st.append(z)
                sts.append((torch.stack(st, 1) * t0["sd"] + t0["mu"])[..., t0["cols"]].cpu().numpy())
            st, vt, lnd = np.concatenate(sts), t0["vt"][dt], t0["lnd"]
            e = SR.err(vt, st, ((vt + lnd) >= -2.0) | ((st + lnd) >= -2.0), C.REPORTED)
            e = np.nan_to_num(e, nan=0.0, posinf=np.inf).max(axis=(1, 2))
            out[f"dt{dt:g}"] = {"all": q(e), "files": {f: q(e[t0["file"] == f]) for f in np.unique(t0["file"])},
                                "T": {b: q(e[(t0["T"] >= lo) & (t0["T"] < hi)]) for b, (lo, hi) in T0_BANDS.items()},
                                "nonfinite_rows": int(np.logical_not(np.isfinite(st)).any(axis=(1, 2)).sum())}
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf
    out["s"] = time.time() - t
    return out


def row0_probs(cfg, split):
    """Per-row probabilities (fp32 [n], on the split's device) that a one-step window and a jump pair of a training row
    start at its row 0 (t = 0): x0_anchor / x0_jump map cache file names (meta.json 'files') to p, rows of other files get
    0; the older x0_file (one name or a list) gives every named file x0_anchor_p and x0_jump_p. A term whose rows are all
    0 gets None: its sampler then makes no extra draw (the draws of a run without row-0 starts are unchanged)."""
    if cfg["x0_file"] is not None:
        xf = cfg["x0_file"] if isinstance(cfg["x0_file"], list) else [cfg["x0_file"]]
        per = ({f: cfg["x0_anchor_p"] for f in xf}, {f: cfg["x0_jump_p"] for f in xf})
    else:
        per = (cfg["x0_anchor"] or {}, cfg["x0_jump"] or {})
    if not (per[0] or per[1]):
        return None, None
    files = json.loads((D.CACHE / "meta.json").read_text())["files"]
    out = []
    for pf in per:
        p = np.zeros(split.n, np.float32)
        for f, v in pf.items():
            m = split.file_idx == files.index(f)
            assert m.any() and 0.0 <= v <= 1.0, f"x0: file {f} (training rows {int(m.sum())}), probability {v}"
            p[m] = v
        out.append(torch.from_numpy(p).to(split.device) if p.any() else None)
    return out


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
    n_pre, K and the one-step windows' (d_pred, z0, z1, idx) for s2_joint."""
    z, dtn, g, idx = train.sample(cfg["batch"], 1, gen)
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
    return loss, cl, tail, n_pre, K, (d_pred, z0, z[:, 1], idx)


def run_stage(stage, cfg, model, train, vy, vdt, vg, mu, sd, run, log, pr=None, t0r=None):
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
    macc = racc = 0.0                                               # miss / rollout term samples (as acc), logged
    rfn = gen_roll = pre_roll = late_roll = None
    if (stage == 1 or cfg["s2_joint"]) and cfg["roll_w"] > 0:   # the rollout term's calls (compiled like lock_fns' grid
        # call) and generators
        def rstep(zs, dtn, g):
            return zs + model.delta(zs.float(), dtn, g).double()

        def rstep_ng(zs, dtn, g):
            return zs + model.delta(zs.float(), dtn, g).double()
        cmp = (lambda fn: torch.compile(fn, fullgraph=True)) if cfg["compile"] else (lambda fn: fn)
        rfn = {"step": cmp(rstep), "pre": cmp(rstep_ng)}
        gen_roll = torch.Generator(device=dev).manual_seed(1000 * cfg["seed"] + 14)
        pre_roll = torch.Generator().manual_seed(1000 * cfg["seed"] + 15)
        if cfg["roll_late_frac"] > 0:
            late_roll = torch.Generator(device=dev).manual_seed(1000 * cfg["seed"] + 16)
    sfn = stat_gen = None
    sacc = [0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0]   # stat term: still / moving means, median |r|, samples, scored still, counts
    if stage == 1 and cfg["stat_w"] > 0:   # the late one-call term's own compiled call and generator
        def sstep(zs, dtn, g):
            return model.delta(zs, dtn, g)
        sfn = torch.compile(sstep, fullgraph=True) if cfg["compile"] else sstep
        stat_gen = torch.Generator(device=dev).manual_seed(1000 * cfg["seed"] + 17)
        scol = torch.tensor([D.SHORT.index(s_) for s_ in C.REPORTED], device=dev)
        sfloor = torch.tensor([SR.PHYS[s_] for s_ in C.REPORTED], dtype=torch.float64, device=dev)
        she = D.SHORT.index("He")
    jfn = jump_gen = None
    jacc = 0.0                                                      # stage-1 jump term samples (as acc), logged
    if stage == 1 and cfg["s1_jump_w"] > 0:   # the stage-1 long-jump term's own compiled call and generator
        def jstep(zs, dtn, g):
            return model.delta(zs, dtn, g)
        jfn = torch.compile(jstep, fullgraph=True) if cfg["compile"] else jstep
        jump_gen = torch.Generator(device=dev).manual_seed(1000 * cfg["seed"] + 18)
    gns = {}                                                        # stat term: per-term gradient norms at the log step
    ms = Misses(cfg if stage == 1 else {**cfg, "cvar_w": 0.0}, mu, sd, dev) if (stage == 1 or cfg["s2_joint"]) \
        and (cfg["miss_w"] > 0 or cfg["cvar_w"] > 0 or cfg["frac_w"] > 0
                                                     or cfg["resample_every"] > 0
                                                     or cfg["log_misses"]) else None
    rs_gen = torch.Generator(device=dev).manual_seed(1000 * cfg["seed"] + 13) if stage == 1 and cfg["resample_every"] > 0 \
        else None
    params = list(model.parameters())
    rcs = {k: [float(cfg[f"{k}_row_clip"]), torch.zeros((), dtype=torch.int64, device=dev), 0]
           if cfg[f"{k}_row_clip"] > 0 and stage == 2 else None for k in ("grid", "chain")}
    rcs["roll"] = [float(cfg["roll_row_clip"]), torch.zeros((), dtype=torch.int64, device=dev), 0] \
        if cfg["roll_row_clip"] > 0 and (stage == 1 or cfg["s2_joint"]) and cfg["roll_w"] > 0 else None
    gnl = []                                                        # clip_norm_fp64: (grid sub-steps, pre-clip norm) per step
    if lock:
        lfns = lock_fns(model, delta_fn, cfg["compile"], mu, sd)
        lfns.update(rc_g=rcs["grid"], rc_c=rcs["chain"])
    nskip = nskip_log = 0                                           # skip_nonfinite: steps skipped so far, at the last log row

    def roll_term():   # the rollout term: the loss of every call with gradient, summed (averaged) over them
        rz, rzt, rdtn, rg, rlnd, rpre = roll_draw(train, cfg, gen_roll, pre_roll, late_roll)
        with torch.no_grad():
            for _ in range(rpre):
                rz = rfn["pre"](rz, rdtn, rg)
        rp = []
        for _ in range(cfg["roll_K"]):
            rz = row_clip(rfn["step"](rz, rdtn, rg), rcs["roll"])
            rp.append(rz)
        rp = torch.stack(rp, 1)
        rl = crit(rp, rzt)["loss_total"]
        if ms is not None and ms.on:
            S_ = rp.shape[-1]
            rl = rl + ms.loss(rp.reshape(-1, S_), rzt.reshape(-1, S_), rlnd.repeat_interleave(cfg["roll_K"]),
                              calls=cfg["roll_K"])
        return rl
    ck = run / f"ckpt_stage{stage}.pt"
    step0 = 0
    if cfg["resume"] and ck.exists():
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
        if lock:
            loss, cl, tail, n_pre, kc, one = lock_terms(lfns, crit, train, cs, cfg, step, gen, pre_gen, mu, sd, rep,
                                                        (step + 1) % 50 == 0)
            loss = loss + cfg["chain_w"] * cl
            if cfg["chain_tail_w"] > 0:
                loss = loss + cfg["chain_tail_w"] * tail
            if cfg["s2_joint"]:   # stage 1's short-run terms beside the long-run ones
                d_pred, z0, z1, idx = one
                if ms is not None and ms.on:
                    ml = ms.loss(z0 + d_pred.double(), z1, train.lnd[idx])
                    loss = loss + ml
                    macc += float(ml.detach()) if (step + 1) % cfg["loss_every"] == 0 else 0.0
                if rfn is not None:
                    rl = roll_term()
                    loss = loss + cfg["roll_w"] * rl
                    racc += float(rl.detach()) if (step + 1) % cfg["loss_every"] == 0 else 0.0
            cacc[3] += n_pre
            if (step + 1) % 50 == 0:
                cacc[0] += float(cl.detach())
                cacc[1] += float(tail.detach())
                cacc[2] += 1
        else:
            if rs_gen is not None and (step % cfg["resample_every"] == 0 or step == step0):   # weights are not checkpointed
                train.w, rs_share = miss_weights(model, train, ms, cfg, rs_gen)
                log(stage, step, resample_miss_share=rs_share, resample_w_mean=float(train.w.mean()))
            z, dtn, g, idx = train.sample(cfg["batch"], 1, gen) if cfg["s1_targets"] <= 1 else multi_targets(train, cfg, gen)
            z0 = z[:, 0]
            d_true = (z[:, 1] - z0).float()
            d_pred = delta_fn(z0.float(), dtn, g)
            loss = crit(d_pred.unsqueeze(1), d_true.unsqueeze(1))["loss_total"]
            if ms is not None and ms.on:
                ml = ms.loss(z0 + d_pred.double(), z[:, 1], train.lnd[idx])
                loss = loss + ml
                macc += float(ml.detach()) if (step + 1) % cfg["loss_every"] == 0 else 0.0
            if rfn is not None:
                rl = roll_term()
                loss = loss + cfg["roll_w"] * rl
                racc += float(rl.detach()) if (step + 1) % cfg["loss_every"] == 0 else 0.0
            if jfn is not None:   # the stage-1 long-jump term: one call from z(t) across D to z(t + D)
                j0, j1, jdt, jg = D.sample_jump_pairs(train, cfg["jump_batch"], jump_gen, cfg["jump_min_s"],
                                                      cfg["jump_max_s"])
                jl = crit((j0 + jfn(j0.float(), jdt, jg).double()).unsqueeze(1), j1.unsqueeze(1))["loss_total"]
                loss = loss + cfg["s1_jump_w"] * jl
                jacc += float(jl.detach()) if (step + 1) % cfg["loss_every"] == 0 else 0.0
            if sfn is not None:   # the late one-call term (stat_term; its own graph through sfn)
                sl, ls_, lm_, dg_ = stat_term(sfn, train, cfg, stat_gen, mu, sd, scol, sfloor, she)
                sw = cfg["stat_w"] * (min(1.0, step / cfg["stat_ramp"]) if cfg["stat_ramp"] > 0 else 1.0)
                if (step + 1) % cfg["loss_every"] == 0:
                    sacc[0] += float(ls_)
                    sacc[1] += float(lm_)
                    sacc[2] += float(dg_["med"])
                    sacc[3] += 1
                    sacc[4] += float(dg_["still_scored"])
                    sacc[5] += float(dg_["n_still"])
                    sacc[6] += float(dg_["n_moving"])
        opt.zero_grad(set_to_none=True)
        if sfn is not None and (step + 1) % cfg["log_every"] == 0:
            # per-term gradient norms (fp64) before the global clip: the stat graph is separate, so its gradient is taken
            # alone (compiled backwards allow one pass per graph), the main loss is back-propagated, then the two are added
            gs_ = torch.autograd.grad(sw * sl, params, allow_unused=True)
            loss.backward()
            nrm = lambda gl: float(torch.linalg.vector_norm(torch.stack(
                [torch.linalg.vector_norm(q_, dtype=torch.float64) for q_ in gl if q_ is not None])))
            gns.update(gnorm_main=nrm([p_.grad for p_ in params]), gnorm_stat=nrm(gs_))
            for p_, g_ in zip(params, gs_):
                if g_ is not None:
                    p_.grad = g_ if p_.grad is None else p_.grad + g_
            loss = (loss + sw * sl).detach()
        else:
            if sfn is not None:
                loss = loss + sw * sl
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
            if ms is not None and ms.on:
                extra["miss_loss"] = macc / max(1, nacc)
                macc = 0.0
            if rfn is not None:
                extra["roll_loss"] = racc / max(1, nacc)
                racc = 0.0
            if jfn is not None:
                extra["jump_loss"] = jacc / max(1, nacc)
                jacc = 0.0
            if sfn is not None:
                n_ = max(1, sacc[3])
                extra.update(stat_still=sacc[0] / n_, stat_moving=sacc[1] / n_, stat_r_median=sacc[2] / n_,
                             stat_still_scored=sacc[4] / n_, stat_n_still=sacc[5] / n_, stat_n_moving=sacc[6] / n_, **gns)
                sacc[:] = [0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0]
            if ms is not None and cfg["log_misses"] and not lock:   # this step's batch: misses and projection failures
                with torch.no_grad():
                    model.rowcount = torch.zeros(z0.shape[0], 2, dtype=torch.long, device=dev)
                    e_ = ms.err(z0 + model.delta(z0.float(), dtn, g).double(), z[:, 1], train.lnd[idx]).amax(-1)
                    extra.update(miss_share=float((e_ > cfg["miss_tau"]).double().mean()),
                                 large_share=float((e_ > 1.0).double().mean()), worst=float(e_.max()),
                                 proj_fail_share=float((model.rowcount[:, 1] > 0).double().mean()))
                    model.rowcount = None
            elif ms is not None and cfg["log_misses"] and cfg["s2_joint"]:   # stage 2: this step's one-step windows
                with torch.no_grad():
                    d_pred, z0, z1, idx = one
                    e_ = ms.err(z0 + d_pred.double(), z1, train.lnd[idx]).amax(-1)
                    extra.update(miss_share=float((e_ > cfg["miss_tau"]).double().mean()),
                                 large_share=float((e_ > 1.0).double().mean()), worst=float(e_.max()))
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
                **({} if pr is None else path_readout(ema.shadow, pr)),
                **({} if t0r is None else {"t0_readout": t0_readout(ema.shadow, t0r)}))
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
    if bool(cfg["t0_readout"]) != bool(cfg["t0_readout_files"]):
        sys.exit(f"{run / 'config.json'}: t0_readout and t0_readout_files are set together")
    assert cfg["s2_steps"] <= 0 or (cfg["lockstep"] and cfg["jump_w"] > 0 and cfg["grid_w"] > 0 and cfg["chain_w"] > 0), \
        "stage 2 runs the one-step, jump, grid and chain terms in lockstep (lockstep true, jump_w, grid_w, chain_w > 0)"
    lib_md5 = {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in sorted(Path(__file__).resolve().parent.glob("*.py"))}
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True   # training with TF32 on; the benches turn it off
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device(os.environ.get("HARNESS_DEVICE", "cuda"))
    D.DT_LOG = (math.log10(cfg["dt_min_s"]), math.log10(cfg["dt_max_s"]))
    D.ANCHOR_P = cfg["anchor_p"]
    if (cfg["x0_file"] is not None or cfg["x0_anchor_p"] > 0 or cfg["x0_jump_p"] > 0) and \
            (cfg["x0_anchor"] is not None or cfg["x0_jump"] is not None):
        sys.exit(f"{run / 'config.json'}: x0_file / x0_anchor_p / x0_jump_p and x0_anchor / x0_jump are exclusive")
    train = D.RawSplit("train", dev)
    if cfg["traj_weights_file"] is not None:
        w = np.load(cfg["traj_weights_file"]).astype(np.float64)
        assert w.shape == (train.n,) and np.all(w >= 0) and w.sum() > 0, (w.shape, train.n)
        train.w = torch.from_numpy(w).float().to(dev)   # fp32 weights
    train.p0_anchor, train.p0_jump = row0_probs(cfg, train)
    if cfg["init_from"] and cfg["s1_init_from"]:
        sys.exit(f"{run / 'config.json'}: init_from (stage 2 only) and s1_init_from (stage 1 from a file) are exclusive")
    if cfg["s2_joint"] and cfg["s2_steps"] <= 0:
        sys.exit(f"{run / 'config.json'}: s2_joint needs a stage 2")
    if (cfg["cvar_w"] > 0 or cfg["resample_every"] > 0 or cfg["stat_w"] > 0 or cfg["s1_jump_w"] > 0
            or (not cfg["s2_joint"] and (cfg["miss_w"] > 0 or cfg["frac_w"] > 0 or cfg["roll_w"] > 0))) \
            and (cfg["init_from"] or cfg["s1_steps"] <= 0):
        sys.exit(f"{run / 'config.json'}: cvar_w, resample_every, stat_w, s1_jump_w and (without s2_joint) miss_w, frac_w and "
                 "roll_w act in stage 1 only, and this run has no stage 1")
    if cfg["resample_every"] > 0 and cfg["traj_weights_file"] is not None:
        sys.exit(f"{run / 'config.json'}: resample_every replaces the draw weights; traj_weights_file must be None")
    train.lnd = torch.from_numpy(np.log10(train.P / (C.KB * train.T))).to(dev)   # log10 number density per run (cm^-3)
    mu, sd, zlo, zhi = D.norm_consts(dev)
    model = M.build(cfg["head"], cfg["hidden"], zlo, zhi, newton=cfg["closed_iters"], fallback=cfg["closed_fallback"],
                    fused=cfg["fused_proj"], latent=cfg["fm_latent"]).to(dev)
    vy, vdt, vg = D.cache_val(cfg["val_windows"], 2, dev)
    pr = None if cfg["path_readout"] is None else readout_setup(cfg["path_readout"], dev)
    t0r = t0_setup(cfg["t0_readout_files"], dev) if cfg["t0_readout"] else None
    logf = open(run / "log.csv", "a")

    def log(stage, step, **kw):
        rec = {"stage": stage, "step": step, "time": time.time(), **kw}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    info = {"params": M.n_params(model), "flops_per_call": M.flops_per_call(model), "n_train_groups": train.n}
    (run / "model_info.json").write_text(json.dumps(info))
    secs = {}
    if cfg["init_from"] or cfg["s1_init_from"]:
        load_init(model, cfg["init_from"] or cfg["s1_init_from"], dev)
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
        secs[f"s{stage}"] = run_stage(stage, cfg, model, train, vy, vdt, vg, mu, sd, run, log, pr, t0r)
    (run / "done.json").write_text(json.dumps({"seconds": secs, **info, "arch_cache": str(D.CACHE), "lib_md5": lib_md5}))
    write_manifest(run, cfg, code_info(lib_md5))


if __name__ == "__main__":
    main(sys.argv[1])
