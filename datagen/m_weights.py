"""Arm M draw weights: the train split's M rows (raw file run_gcm_m18) get 25% of draws, the old rows uniform (capw17's
band shares among them). Prints the drawn shares (CPU, 2e6 draws through data.RawSplit.draw with the fp32 weights
train.py uses) and the row-0 shares of windows and jump pairs with x0_anchor_p 0.5 / x0_jump_p 0.25.
    ARCH_CACHE=<cache> python m_weights.py <cache> <out.npy> [w_cold]
With w_cold (arm M + D, 2026-10-08), the old rows below 1300 K are drawn w_cold times as often as the others (as traj_w
[1300, w_cold]) and the M rows still get 25% of all draws."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import data as D  # noqa: E402

cache, out = Path(sys.argv[1]), Path(sys.argv[2])
assert D.CACHE == cache, (D.CACHE, cache)
files = json.loads((cache / "meta.json").read_text())["files"]
z = np.load(cache / "raw_train.npz")
m = np.isin(z["file_idx"], [i for i, f in enumerate(files) if f.startswith("run_gcm_m18")])  # every arm-M file
n_m, n_o = int(m.sum()), int(np.logical_not(m).sum())
base = np.where(z["T"] < 1300.0, float(sys.argv[3]), 1.0) if len(sys.argv) > 3 else np.ones(m.size)
w_m = base[np.logical_not(m)].sum() / (3.0 * n_m)
w = np.where(m, w_m, base)
if out.exists():
    assert np.array_equal(np.load(out), w), "existing weights differ"
else:
    np.save(out, w)
wt = torch.from_numpy(w).float()
g = torch.Generator().manual_seed(0)
d = torch.multinomial(wt, 2_000_000, replacement=True, generator=g).numpy()
T = z["T"]
print(f"train rows: old {n_o}, M {n_m}; M weight {w_m:.4f}; drawn M share {m[d].mean():.4f} (target 0.25)")
EDG = [300, 800, 1300, 1800, 2300, 3001]
old = np.logical_not(m[d])
print(f"old-row draws below 1300 K: {(T[d][old] < 1300).mean():.4f}")
print("old-row draws by band:", np.round(np.histogram(T[d][old], EDG)[0] / old.sum(), 4).tolist(),
      "; capw17 (uniform over arch_gcm17_all train):", np.round(np.histogram(T[np.logical_not(m)], EDG)[0] / n_o, 4).tolist())
print("M-row draws by band:", np.round(np.histogram(T[d][m[d]], EDG)[0] / m[d].sum(), 4).tolist())
# Production sampling check (second opinion, codex c2_mreview): the real samplers on this cache with the fp32 weights
# and the M rows flagged, x0_anchor_p 0.5 / x0_jump_p 0.25, dt 10-500 s (capw17's window law): accepted-window M share,
# row-0 share among M windows, jump-pair M and row-0 shares, grid-chain M share; M rows whose stored trajectory ends
# early (end row < 99) and could fail the 99 x dt fit.
D.DT_LOG = (1.0, np.log10(500.0))
D.X0_ANCHOR_P, D.X0_JUMP_P = 0.5, 0.25
tr = D.RawSplit("train", torch.device("cpu"))
assert tr.n == w.size
tr.w = torch.from_numpy(w).float()
tr.x0 = torch.from_numpy(m)
gen = torch.Generator().manual_seed(3)
mw, r0 = [], []
for _ in range(10):
    zz, dtn, gg, idx = tr.sample(2048, 1, gen)
    fl = tr.x0[idx]
    mw.append(fl.double().mean())
    r0.append(((zz[:, 0] == tr.z_of(tr.logy[idx, 0])).all(1) & fl).double().sum() / max(1, int(fl.sum())))
j0, j1, jdt, jg, jidx = D.sample_jump_pairs(tr, 20000, gen, 0.1, 1e7, True)
jf = tr.x0[jidx]
jr = ((j0 == tr.z_of(tr.logy[jidx, 0])).all(1) & jf).double().sum() / max(1, int(jf.sum()))
_, _, _, gidx = tr.sample_grid(20000, 20, gen, 1e3, 1, 0.5)
end = z["end"][m]
print(f"windows (dt 10-500 s): M share {float(np.mean(mw)):.4f}; row-0 share among M windows {float(np.mean(r0)):.4f} (key 0.5)")
print(f"jump pairs: M share {float(jf.double().mean()):.4f}; row-0 share among M pairs {float(jr):.4f} (key 0.25)")
print(f"grid chains: M share {float(tr.x0[gidx].double().mean()):.4f} (no row 0 by design)")
print(f"M rows with end row < 99: {int((end < 99).sum())} of {end.size}; with t[end] < 99 x 10 s: "
      f"{int((np.load(D.CACHE / 't_grid.npy')[end] < 990.0).sum())}")
