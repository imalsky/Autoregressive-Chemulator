"""Paper-style figures (as fig_paper.py) for one model on the 58-day paths (paths_v2x: 16 paths x 10,000 calls at dt
500 s, T and P follow a 1e5 s cycle, VULCAN chained call by call; model autoregressive with an fp64 state, bench PX_<run>):
  long_paths_trajectories.png  among paths whose VULCAN abundances of the 6 shown species move >= 0.5 dex, the paths at
                               the 25th, 50th and 75th percentile of per-path mean error (calls >= 50): VULCAN (tinted solid) vs the model (dashed), 6 species, every call
  long_paths_parity.png        model vs VULCAN mixing ratio on resolved points (as xlong_board.py) in three elapsed
                               windows, 0-5.8, 5.8-17 and 17-58 days (calls 50-1000, 1000-3000, 3000-10000)
Outputs go to paths.FIGS (the phase folder's top level; was the sprint folder's).
    python fig_xlong.py <run>"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
sys.path.insert(0, str(paths.SCIPLOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sciplot import *  # noqa: E402,F401,F403
import scoreboard as S  # noqa: E402

RUN = sys.argv[1]
OUT = paths.FIGS
SP = S.SP
SHOW = ["CH4", "CO", "H2O", "CO2", "NH3", "HCN"]
NAMES = {"CH4": "CH$_4$", "CO": "CO", "H2O": "H$_2$O", "CO2": "CO$_2$", "NH3": "NH$_3$", "HCN": "HCN"}
FLOOR = -14.0  # window, as fig_paper.py

d = np.load(S.E / f"PX_{RUN}" / "P_paths.npz")
st, calls = S.A.cut12(d["states"][:, 2]), d["calls"]                 # [C, paths, 12]
vt, rs, Tp, Pp = [], [], [], []
for k, p in enumerate(d["path"]):
    q = np.load(paths.PATHS_V2X / f"{int(p)}.npz")
    cols = [list(q["species"]).index(s) for s in S.A.D.SHORT]
    x = np.log10(np.maximum(q["x"][calls][:, cols], S.A.D.EPS))
    n = q["Pbar"][calls - 1] * 1e6 / (S.A.KB * q["T"][calls - 1])
    vt.append(x)
    rs.append((x + np.log10(n)[:, None] >= -2.0) | (st[:, k] + np.log10(n)[:, None] >= -2.0))
    Tp.append(q["T"].mean())
    Pp.append(np.exp(np.log(q["Pbar"]).mean()))
vt, rs, Tp, Pp = np.stack(vt, 1), np.stack(rs, 1), np.array(Tp), np.array(Pp)
e = np.where(rs & ((vt >= S.FLOOR) | (st >= S.FLOOR)), np.abs(st - vt), np.nan)
pm = np.nanmean(e[calls >= 50].transpose(1, 0, 2).reshape(e.shape[1], -1), 1)  # per-path mean error
days = calls * 500.0 / 86400.0

fig, axs = panels(1, 3)
js = [SP.index(s) for s in SHOW]
mv = (vt[:, :, js].max(0) - vt[:, :, js].min(0)).max(-1)            # as fig_paper.py: shown species move >= 0.5 dex
rows = np.flatnonzero(mv >= 0.5)
for k, (ax, qq) in enumerate(zip(axs, (0.25, 0.5, 0.75))):
    i = rows[np.argsort(pm[rows])[int(round(qq * (rows.size - 1)))]]
    lo_y, hi_y = [], []
    for c, s in enumerate(SHOW):
        j = SP.index(s)
        ax.plot(days, 10.0 ** vt[:, i, j], color=pastel(CYC[c]), lw=1.5)
        ax.plot(days, 10.0 ** st[:, i, j], color=CYC[c], lw=0.7, **DASH_KW, label=NAMES[s])
        lo_y += [vt[:, i, j].min(), st[:, i, j].min()]
        hi_y += [vt[:, i, j].max(), st[:, i, j].max()]
    ax.set_yscale("log")
    ax.set_xlim(days[0], days[-1])
    ax.set_ylim(10.0 ** max(FLOOR, min(lo_y)), 10.0 ** max(hi_y))
    ax.set_xlabel("time (days)")
    ax.set_ylabel(f"mixing ratio (mean {Tp[i]:.0f} K, {Pp[i]:.2g} bar)")
    print(f"trajectory panel {k}: path {int(d['path'][i])}, mean T {Tp[i]:.0f} K, geometric-mean P {Pp[i]:.3g} bar, "
          f"per-path mean error {pm[i]:.4f} dex (quantile {qq} of {rows.size} evolving paths)")
axs[0].legend(loc="center", bbox_to_anchor=(0.5, 0.69), ncol=2)  # every corner sits on data: the empty band
                                                                # (placed for e13_m17fs_s0; check for another run)
from matplotlib.lines import Line2D  # noqa: E402
axs[1].legend(handles=[Line2D([], [], color=pastel(INK), lw=1.5), Line2D([], [], color=INK, lw=0.7, **DASH_KW)],
              labels=["VULCAN", "model"], loc="center", bbox_to_anchor=(0.5, 0.645))
fig.savefig(OUT / "long_paths_trajectories.png", dpi=300)

edges = np.linspace(FLOOR, 0.0, 141)
fig, axs = panels(1, 3)
Hs = []
for lo_c, hi_c in [(50, 1000), (1000, 3000), (3000, 10001)]:
    m = (calls >= lo_c) & (calls < hi_c)
    ok = np.isfinite(e[m])
    xv, yv = vt[m][ok], st[m][ok]
    keep = (xv >= FLOOR) & (yv >= FLOOR)
    H, _, _ = np.histogram2d(xv[keep], yv[keep], bins=[edges, edges])
    Hs.append(H / ok.sum())
    print(f"parity calls {lo_c}-{hi_c}: {keep.sum()} of {ok.sum()} scored points drawn (>= 1e-14)")
vmin, vmax = min(h[h > 0].min() for h in Hs), max(h.max() for h in Hs)
for ax, H, (a, b) in zip(axs, Hs, [("0", "5.8"), ("5.8", "17"), ("17", "58")]):
    im = ax.pcolormesh(10.0 ** edges, 10.0 ** edges, np.where(H > 0, H, np.nan).T, cmap="cividis",
                       norm=LogNorm(vmin=vmin, vmax=vmax), rasterized=True)
    ax.plot([10.0 ** FLOOR, 1.0], [10.0 ** FLOOR, 1.0], color=RED, lw=1.2, **DASH_KW, label="1:1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect("auto")
    ax.set_xlabel(f"VULCAN mixing ratio (days {a}-{b})")
    ax.set_ylabel("model mixing ratio")
    if ax is axs[0]:
        ax.legend(loc="upper left")  # the empty corner; one entry serves all three panels
colorbar(fig, im, axs[2], "fraction of points per bin")
fig.savefig(OUT / "long_paths_parity.png", dpi=300)
