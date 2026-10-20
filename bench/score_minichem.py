"""The six mini-chem paper cases (Tsai et al. 2022, arXiv 2204.04201, Fig. 4: solar, raw initial gas; paths.FIG4 tag jobs
0-5 = mini-chem cases c0-c5): worst gas per call against VULCAN for the model benches and for mini-chem's stored predictions.
  model:     the bench paths.FIG4/<prefix>_dt<dt>/H_tag_dt<dt>.npz (arm H from VULCAN's state at 1e-3 s, every call recorded);
  mini-chem: paths.FIG4/mc_c<case>_*.npz (made by the mini-chem driver on the generator side), log10 mixing ratio interpolated
             linearly in log10 t onto the bench's call times (t = 0 dropped).
Truth = the bench's VULCAN tag at the call times (scoreboard.bench); error = scorerule.err (SCORE_RULE=phys: per-gas floors,
resolved points = truth or prediction number density >= 1e-2 cm^-3), 12 gases. Per case: median / max / last of the worst gas
per call, and the worst gas at the last call. No model call, no mini-chem run.
    SCORE_RULE=phys python score_minichem.py <dt> [<dt> ...]      (e.g. 1 20 500)"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoreboard as S  # noqa: E402  (puts lib/ on sys.path)
import paths  # noqa: E402

assert os.environ.get("SCORE_RULE") == "phys", "set SCORE_RULE=phys (else the old flat 1e-12 floor applies)"
F = paths.FIG4
CASE = ["1000K 1mbar", "1500K 1mbar", "2000K 1mbar", "1000K 10bar", "1500K 10bar", "2000K 10bar"]
MOD = [("FINAL", "gpu_all"), ("LD", "l_e13s_LD_s3"), ("CBg", "cbg")]


def mc_logy(case, times):
    """mini-chem's log10 mixing ratios of the 12 gases [len(times), 12], linear in log10 t between its stored times."""
    mc = np.load(next(F.glob(f"mc_c{case}_*.npz")))
    msp = list(mc["species"])
    return np.stack([np.interp(np.log10(times), np.log10(mc["t"][1:]),
                               np.log10(np.maximum(mc["vmr"][1:, msp.index(s)], 1e-30))) for s in S.SP], -1)


def err(vt, y, lnd):
    """per-gas rule error [C, 12] of log10 y against the truth vt (NaN = not scored)."""
    return S.SR.err(vt[None], y[None], (((vt + lnd) >= -2.0) | ((y + lnd) >= -2.0))[None], S.SP)[0]


def cases(f):
    """Per case 0-5 of a fig4 bench: call times, model error and mini-chem error vs VULCAN [C, 12]."""
    B, d = S.bench(f), np.load(f)
    out = []
    for job in range(6):
        r = int(np.flatnonzero(d["job"] == job)[0])
        tm = d["t0"][r] + np.asarray(d["calls"], np.float64) * float(d["dt"])
        lnd = np.log10(d["P"][r] / (S.KB * d["T"][r]))
        out.append((tm, err(B["vt"][r], B["st"][r], lnd), err(B["vt"][r], mc_logy(job, tm), lnd)))
    return out


def cell(e):
    """median / max / last of the worst gas per call, and the worst gas at the last call."""
    w = np.nanmax(e, -1)
    return float(np.nanmedian(w)), float(np.nanmax(w)), float(w[-1]), S.SP[int(np.nanargmax(np.nan_to_num(e[-1], nan=-1)))]


if __name__ == "__main__":
    for dt in sys.argv[1:]:
        print(f"dt {dt} s: worst gas per call, median / max / end (dex); worst gas at the end")
        mc_lines = []
        for nm, pre in MOD:
            f = F / f"{pre}_dt{dt}" / f"H_tag_dt{dt}.npz"
            if not f.exists():
                print(f"  {nm:5s} (no bench)")
                continue
            res = cases(f)
            for k, lab in ((1, f"  {nm:5s} "), (2, f"  minichem-at-{nm}-calls ")):
                cells = [cell(c[k]) for c in res]
                line = lab + " | ".join(f"{c}: {m:.2f}/{x:.2f}/{e:.2f} {g}" for c, (m, x, e, g) in zip(CASE, cells))
                (print(line) if k == 1 else mc_lines.append(line))
        print("\n".join(mc_lines))
