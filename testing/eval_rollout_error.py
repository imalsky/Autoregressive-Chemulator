#!/usr/bin/env python3
"""Percentile rollout-error curve over the test set (the paper's headline metric).

Rolls the exported 1-step model autoregressively at the trajectory's own (fixed)
dt over a batch of test trajectories and reports fractional-error percentiles vs
rollout step -- the same metric format as the sibling one-step paper
(10/50/90/95th pct @ 10 and 99 steps). Reuses predictions.py's load/denorm
helpers so the numbers are apples-to-apples with that script.

Use the SAME script later on the HPC stage-1 (one-step baseline) and stage-2
(AR) exports to produce the headline before/after figure.

Run:  conda run -n nn python Auto-Chem/testing/eval_rollout_error.py [models/v3] [K]
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "testing"))

import predictions as P  # noqa: E402  (needs the sys.path insert above; reuses its loaders/denormalizers)

N_SAMPLES_CAP = 4000          # cap trajectories pooled for the percentile estimate
ROLLOUT_START_INDEX = 1       # match predictions.py default
PCTLS = (10, 50, 90, 95)
REPORT_STEPS = (1, 5, 10, 25, 50, 99)


def _denorm_globals_batch(g_z, global_keys, manifest):
    return np.stack([P._denormalize_globals(g_z[i], global_keys, manifest) for i in range(g_z.shape[0])], axis=0)


@torch.inference_mode()
def _rollout_batch(model, y0_phys, g_phys, dt_seconds, infer_dtype, K):
    """y0_phys[B,S], g_phys[B,G], dt_seconds[B,K] -> preds[B,K,S] (physical)."""
    dev = P.INFER_DEVICE
    y = torch.from_numpy(y0_phys.astype(np.float32)).to(dev, infer_dtype)
    g = torch.from_numpy(g_phys.astype(np.float32)).to(dev, infer_dtype)
    dt = torch.from_numpy(dt_seconds.astype(np.float32)).to(dev, infer_dtype)
    out = []
    for k in range(K):
        y = model(y, dt[:, k], g)
        out.append(y.detach().to("cpu", torch.float64).numpy())
    return np.stack(out, axis=1)  # [B,K,S]


def main() -> None:
    run_dir_arg = sys.argv[1] if len(sys.argv) > 1 else "models/v3"
    K_req = int(sys.argv[2]) if len(sys.argv) > 2 else 99

    run_dir = P._resolve_repo_path(run_dir_arg)
    export_path = P._resolve_export_path(run_dir)
    model, meta, infer_dtype = P._load_export_module(export_path)
    norm_path = P._resolve_normalization_path(meta)
    manifest = P._load_json(norm_path)
    processed_dir = norm_path.parent
    species_keys = list(manifest["species_variables"])
    global_keys = list(manifest["global_variables"])
    dt = manifest["dt"]
    print(f"[model] {export_path}")
    print(f"[data]  {processed_dir}  (dt log10 [{dt['log_min']},{dt['log_max']}] s, S={len(species_keys)})")

    # pool test samples across shards up to the cap
    shard_dir = processed_dir / "test"
    shards = sorted(shard_dir.glob("shard_*.npz"))
    ys, gs, dts = [], [], []
    n = 0
    for sp in shards:
        y_mat, g_mat, dt_norm_mat = P._load_shard(sp)
        ys.append(y_mat)
        gs.append(g_mat)
        dts.append(dt_norm_mat)
        n += y_mat.shape[0]
        if n >= N_SAMPLES_CAP:
            break
    y_mat = np.concatenate(ys)[:N_SAMPLES_CAP]
    g_mat = np.concatenate(gs)[:N_SAMPLES_CAP]
    dt_norm_mat = np.concatenate(dts)[:N_SAMPLES_CAP]
    B, T, S = y_mat.shape
    K = min(K_req, (T - 1) - ROLLOUT_START_INDEX)
    print(f"[eval]  B={B} trajectories, T={T}, rolling out K={K} steps from index {ROLLOUT_START_INDEX}\n")

    # slice + denormalize to physical
    y0_z = y_mat[:, ROLLOUT_START_INDEX, :]
    y_true_z = y_mat[:, ROLLOUT_START_INDEX + 1 : ROLLOUT_START_INDEX + 1 + K, :]
    dt_seq_norm = dt_norm_mat[:, ROLLOUT_START_INDEX : ROLLOUT_START_INDEX + K]
    y0_phys = P._denormalize_species(y0_z, species_keys, manifest)
    y_true = P._denormalize_species(y_true_z, species_keys, manifest)          # [B,K,S]
    g_phys = _denorm_globals_batch(g_mat, global_keys, manifest)
    dt_seconds = P._dt_norm_to_seconds(dt_seq_norm, manifest)                  # [B,K]

    y_pred = _rollout_batch(model, y0_phys, g_phys, dt_seconds, infer_dtype, K)  # [B,K,S]

    # fractional error per (traj, step, species); pool over traj+species at each step
    eps = 1e-30
    yt = np.clip(y_true, eps, None)
    yp = np.clip(y_pred, eps, None)
    frac = np.abs(yp - yt) / (np.abs(yt) + 1e-12)        # [B,K,S]
    dex = np.abs(np.log10(yp) - np.log10(yt))            # [B,K,S]
    finite = np.isfinite(y_pred).all(axis=(1, 2)).mean()
    print(f"finite trajectories: {100*finite:.1f}%\n")

    # per-step percentiles (pool over B and S)
    print(f"{'step':>5} " + " ".join(f"frac_p{p:<2}" for p in PCTLS) + "   dex_p90")
    print("-" * 52)
    curve = []
    for k in range(K):
        fk = frac[:, k, :].ravel() * 100.0
        ps = np.percentile(fk, PCTLS)
        dk = np.percentile(dex[:, k, :].ravel(), 90)
        curve.append((k + 1, *ps, dk))
        if (k + 1) in REPORT_STEPS or (k + 1) == K:
            print(f"{k+1:>5} " + " ".join(f"{v:7.1f}%" for v in ps) + f"   {dk:6.3f}")

    # headline lines (match sibling paper @10 and @99)
    print("\n=== headline (fractional error %, pooled species x trajectories) ===")
    for target in (10, K):
        row = curve[target - 1]
        pcts = ", ".join(f"p{p}={row[i+1]:.1f}%" for i, p in enumerate(PCTLS))
        print(f"@ {target} steps: {pcts}   (sibling one-step paper @10: p90=14.5%, @99: p90=56.6%)")

    out = run_dir / "rollout_error_curve.csv"
    with open(out, "w") as f:
        f.write("step," + ",".join(f"frac_p{p}" for p in PCTLS) + ",dex_p90\n")
        for row in curve:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
