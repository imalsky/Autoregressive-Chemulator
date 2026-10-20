"""Paper-style inference cost (as the ApJ paper's Fig. 2 right): time per cell for one model call vs batch size, on the
GPU (CUDA graph of the fp64-state step, TF32 off) and the CPU (1 and 8 threads), from paths.OUT/speed_<run>.json.
Output paper_speed.png in paths.FIGS (the phase folder's top level).
    python fig_speed.py <run>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
sys.path.insert(0, str(paths.SCIPLOT))
from sciplot import *  # noqa: E402,F401,F403

RUN = sys.argv[1]
OUT = paths.FIGS
S = json.loads((paths.OUT / f"speed_{RUN}.json").read_text())   # speed.py output
fig, ax = square()
ys = []
for c, (key, lab, mk) in enumerate((("gpu", "GPU (RTX 6000 Ada)", "o"), ("cpu8", "CPU, 8 threads", "s"), ("cpu1", "CPU, 1 thread", "^"))):
    n = sorted(int(k) for k in S[key])
    y = [S[key][str(k)]["us_per_cell"] for k in n]
    ys += y
    ax.plot(n, y, color=CYC[c], marker=mk, ms=MS, label=lab)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(min(ys), max(ys))
ax.set_xlabel("batch size (cells)")
ax.set_ylabel("time per cell per call ($\\mu$s)")
legend(ax)
fig.savefig(OUT / "paper_speed.png", dpi=300)
