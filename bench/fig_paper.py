"""Figures in the style of the ApJ paper's main ones (10.3847/1538-4357/aea15b, Figs 2, 4, 5, 6) for one model on the
G test set (3072 rows, autoregressive at dt 500 s for 1000 days, P3 = fp64 state):
  paper_trajectories.png  three representative fresh-start rows (t0 = 1e-3 s; among the rows of a T band whose
                          VULCAN abundances of the 6 shown species move >= 0.5 dex after the first call, the one with
                          the median error), VULCAN from t0 (tinted solid) vs the model's calls from 500 s (dashed)
  paper_parity.png        predicted vs VULCAN mixing ratio, every resolved point at checkpoints >= 1e3 s (density)
  paper_errors.png        C/O from the 12 tracked species at 1000 days, VULCAN vs model; per-row mean |dlog10 x|
                          over checkpoints >= 1e3 s (G test) and per-path mean over calls >= 50 (long GCM-like paths,
                          T and P change every call) for the model and a reference model
Outputs go to paths.FIGS (the phase folder's top level; was the sprint folder's).
    python fig_paper.py <run> <reference run>"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
sys.path.insert(0, str(paths.SCIPLOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sciplot import *  # noqa: E402,F401,F403
import analyze_e13 as A  # noqa: E402
import score_sprint as SS  # noqa: E402

RUN, REF = sys.argv[1], sys.argv[2]
E = paths.E
OUT = paths.FIGS
SP = list(A.D.SHORT)
SHOW = ["CH4", "CO", "H2O", "CO2", "NH3", "HCN"]
NAMES = {"CH4": "CH$_4$", "CO": "CO", "H2O": "H$_2$O", "CO2": "CO$_2$", "NH3": "NH$_3$", "HCN": "HCN"}
FLOOR = -14.0  # window: mixing ratios below 1e-14 are not drawn (resolved points reach about 1e-23 at 1 kbar)


def bench(run):
    d = np.load(E / f"H_g500_{run}" / "H_gtest_dt500.npz")
    st = A.cut12(d["states"][:, 2]).transpose(1, 0, 2)                    # [R, C, 12] log10 mixing ratio, fp64 state
    times = d["t0"][:, None] + d["calls"][None, :] * float(d["dt"])
    tag = str(d["tag"])
    vt = np.stack([A.SC.vtruth(tag, j, times[i]) for i, j in enumerate(d["job"])])
    lnd = np.log10(d["P"] / (A.KB * d["T"]))[:, None, None]
    res = ((vt + lnd) >= -2.0) | ((st + lnd) >= -2.0)                       # the scorer's resolved points
    late = d["calls"] * float(d["dt"]) >= 1e3
    return {"st": st, "vt": vt, "res": res, "late": late, "times": times, "tag": tag, "d": d}


def row_err(B):
    e = np.where(B["res"], np.abs(B["st"] - B["vt"]), np.nan)[:, B["late"]]
    return np.nanmean(e.reshape(e.shape[0], -1), 1)


def bar(P):
    """Pressure label from dyn cm^-2: 3 significant figures, mathtext power of ten below 0.01 bar."""
    p = P / 1e6
    if p >= 0.01:
        return f"{float(f'{p:.3g}'):g} bar"
    e = int(np.floor(np.log10(p)))
    return f"${p / 10 ** e:.2g} \\times 10^{{{e}}}$ bar"


def c_to_o(lx):
    x = 10.0 ** lx
    g = {s: x[..., SP.index(s)] for s in SP}
    C = 2 * g["C2H2"] + g["CH4"] + g["CO2"] + g["CO"] + g["HCN"]
    O = 2 * g["CO2"] + g["CO"] + g["H2O"] + g["OH"] + g["O"]
    return C / O


B, R = bench(RUN), bench(REF)
d = B["d"]
err, err_ref = row_err(B), row_err(R)
print(f"{RUN}: per-row mean |dlog10 x| median {np.median(err):.4f}, p90 {np.percentile(err, 90):.4f}; "
      f"{REF}: median {np.median(err_ref):.4f}, p90 {np.percentile(err_ref, 90):.4f}")

# trajectories: per T band, the fresh-start row with the median per-row error
fig, axs = panels(1, 3)
for k, (ax, (lo, hi)) in enumerate(zip(axs, [(300, 800), (1300, 1800), (2300, 3001)])):
    js = [SP.index(s) for s in SHOW]
    mv = np.abs(B["vt"][:, -1, js] - B["vt"][:, 0, js]).max(-1)
    rows = np.flatnonzero((d["t0"] == 1e-3) & (d["T"] >= lo) & (d["T"] < hi) & (mv >= 0.5))
    i = rows[np.argsort(err[rows])[rows.size // 2]]
    tm = B["times"][i]
    td = np.geomspace(d["t0"][i], tm[-1], 600)
    vd = A.SC.vtruth(B["tag"], d["job"][i], td)
    lo_y, hi_y = [], []
    for c, s in enumerate(SHOW):
        j = SP.index(s)
        ax.plot(td, 10.0 ** vd[:, j], color=pastel(CYC[c]), lw=2.5, solid_capstyle="butt")
        ax.plot(tm, 10.0 ** B["st"][i, :, j], color=CYC[c], lw=1.2, **DASH_KW, marker="o", ms=2.0, label=NAMES[s])
        lo_y += [vd[:, j].min(), B["st"][i, :, j].min()]
        hi_y += [vd[:, j].max(), B["st"][i, :, j].max()]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(td[0], td[-1])
    ax.set_ylim(10.0 ** max(FLOOR, min(lo_y)), 10.0 ** max(hi_y))
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"mixing ratio ({d['T'][i]:.0f} K, {bar(d['P'][i])})")
    print(f"trajectory panel {k}: row {i}, job {d['job'][i]}, T {d['T'][i]:.0f} K, P {d['P'][i] / 1e6:.3g} bar, "
          f"mean err {err[i]:.4f} dex; {rows.size} evolving rows in the band")
legend(axs[0])
from matplotlib.lines import Line2D  # noqa: E402
legend(axs[1], handles=[Line2D([], [], color=pastel(INK), lw=2.5), Line2D([], [], color=INK, lw=1.2, **DASH_KW)],
       labels=["VULCAN", "model"])
fig.savefig(OUT / "paper_trajectories.png", dpi=300)

# parity: density of (VULCAN, model) over resolved late points, every species
m = B["res"][:, B["late"]]
xv, yv = B["vt"][:, B["late"]][m], B["st"][:, B["late"]][m]
keep = (xv >= FLOOR) & (yv >= FLOOR)
edges = np.linspace(FLOOR, 0.0, 141)
H, _, _ = np.histogram2d(xv[keep], yv[keep], bins=[edges, edges])
fig, ax = square()
im = ax.pcolormesh(10.0 ** edges, 10.0 ** edges, H.T, cmap="cividis", norm=LogNorm(vmin=1, vmax=H.max()), rasterized=True)
ax.plot([10.0 ** FLOOR, 1.0], [10.0 ** FLOOR, 1.0], color=RED, lw=1.2, **DASH_KW, label="1:1")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_aspect("auto")
ax.set_xlabel("VULCAN mixing ratio")
ax.set_ylabel("model mixing ratio")
ax.legend(loc="upper left")  # the empty corner
colorbar(fig, im, ax, "points per bin")
fig.savefig(OUT / "paper_parity.png", dpi=300)
print(f"parity: {keep.sum()} of {m.sum()} resolved late points drawn (>= 1e-14)")

# errors: C/O at 1000 days, and the per-row mean error distribution
fig, axs = panels(1, 3)
cv, cm = c_to_o(B["vt"][:, -1]), c_to_o(B["st"][:, -1])
axs[0].plot(cv, cm, "o", color=INK, ms=2.0, mew=0, label=RUN.replace("e13_", ""))
lim = [min(cv.min(), cm.min()), max(cv.max(), cm.max())]
axs[0].plot(lim, lim, color=RED, lw=1.2, **DASH_KW, label="1:1")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlim(lim)
axs[0].set_ylim(lim)
axs[0].set_xlabel("C/O, VULCAN (12 tracked species, 1000 days)")
axs[0].set_ylabel("C/O, model")
legend(axs[0])
bins = np.geomspace(min(err.min(), err_ref.min()), max(err.max(), err_ref.max()), 50)
axs[1].hist(err_ref, bins=bins, histtype="step", color=INK, lw=1.5, label=REF.replace("e13_", "") + " (old baseline)")
axs[1].hist(err, bins=bins, histtype="step", color=RED, lw=1.5, label=RUN.replace("e13_", ""))
axs[1].set_xscale("log")
axs[1].set_xlim(bins[0], bins[-1])
axs[1].set_xlabel("per-row mean $|\\Delta \\log_{10} x|$ (dex)")
axs[1].set_ylabel("test rows (G, fixed T and P)")
axs[1].legend(loc="upper left")  # the empty corner


def path_err(run):
    e, calls, _ = SS.paths_errors(E / f"P2l_{run}" / "P_paths.npz")
    return np.nanmean(e[calls >= 50].transpose(1, 0, 2).reshape(e.shape[1], -1), 1)


pe, pe_ref = path_err(RUN), path_err(REF)
pb = np.geomspace(min(pe.min(), pe_ref.min()), max(pe.max(), pe_ref.max()), 31)
axs[2].hist(pe_ref, bins=pb, histtype="step", color=INK, lw=1.5, label=REF.replace("e13_", "") + " (old baseline)")
axs[2].hist(pe, bins=pb, histtype="step", color=RED, lw=1.5, label=RUN.replace("e13_", ""))
axs[2].set_xscale("log")
axs[2].set_xlim(pb[0], pb[-1])
axs[2].set_xlabel("per-path mean $|\\Delta \\log_{10} x|$ (dex)")
axs[2].set_ylabel("long GCM-like paths")
axs[2].legend(loc="upper left")  # the empty corner
print(f"paths v2l per-path mean |dlog10 x|: {RUN} median {np.median(pe):.4f}, {REF} median {np.median(pe_ref):.4f} "
      f"({pe.size} paths)")
fig.savefig(OUT / "paper_errors.png", dpi=300)
print("C/O at 1000 days: median |dlog10 C/O|", f"{np.median(np.abs(np.log10(cm / cv))):.4f},",
      f"p90 {np.percentile(np.abs(np.log10(cm / cv)), 90):.4f}")
