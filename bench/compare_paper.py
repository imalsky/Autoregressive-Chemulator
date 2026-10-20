"""Our G dt500 errors in the paper's metric (ApJ 10.3847/1538-4357/aea15b, Fig 6: |dlog10 x| averaged over all 12
species and all output times per trajectory, no resolved mask; floor 1e-20 as its fractional-error epsilon) next to
our resolved-point metric, for all of G and for the paper's mini-chem window (T 1000-2500 K, P 1e2-1e7 Pa).
    python compare_paper.py <run> [<run> ...]"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_e13 as A  # noqa: E402

E = paths.E
R = paths.RUNS
for run in sys.argv[1:]:
    sd = torch.load(R / run / "stage2.pt", map_location="cpu", weights_only=False)["state_dict"]
    npar = sum(v.numel() for v in sd.values())
    d = np.load(E / f"H_g500_{run}" / "H_gtest_dt500.npz")
    st = A.cut12(d["states"][:, 2]).transpose(1, 0, 2)
    times = d["t0"][:, None] + d["calls"][None, :] * float(d["dt"])
    vt = np.stack([A.SC.vtruth(str(d["tag"]), j, times[i]) for i, j in enumerate(d["job"])])
    lnd = np.log10(d["P"] / (A.KB * d["T"]))[:, None, None]
    res = ((vt + lnd) >= -2.0) | ((st + lnd) >= -2.0)
    late = d["calls"] * float(d["dt"]) >= 1e3
    e_paper = np.abs(np.maximum(st, -20.0) - np.maximum(vt, -20.0))[:, late]           # every species, floor 1e-20
    e_ours = np.where(res, np.abs(st - vt), np.nan)[:, late]
    win = (d["T"] >= 1000) & (d["T"] <= 2500) & (d["P"] / 10 >= 1e2) & (d["P"] / 10 <= 1e7)  # P dyn cm^-2 -> Pa
    print(f"{run}: {npar / 1e6:.2f} M parameters; G dt500 {d['calls'][-1]} calls (1000 days), {late.sum()} checkpoints")
    for name, m in (("all G rows", np.ones(win.size, bool)), ("paper window", win)):
        rp = e_paper[m].reshape(m.sum(), -1).mean(1)
        ro = np.nanmean(e_ours[m].reshape(m.sum(), -1), 1)
        print(f"  {name} ({m.sum()} rows): paper metric per-row mean: mean {rp.mean():.4f}, median {np.median(rp):.4f}, "
              f"p90 {np.percentile(rp, 90):.4f} | resolved-only: mean {np.nanmean(ro):.4f}, median {np.nanmedian(ro):.4f}"
              f" | fractional |dx|/x median {np.median(np.expm1(np.log(10) * e_paper[m])):.4f}")
