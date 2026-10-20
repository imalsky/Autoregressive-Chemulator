"""Candidate judge: a run X against a reference R (the rule CBg was judged by, with the run names as arguments). No model
call, no GPU.
  X vs R by rules_t5.py (fixed-condition rows of G, Gc, H2, H2c, H3, H3c; the five path sets; numerics; T2 snap-back);
  X vs FINAL by rules_t5.py with FINAL's T2 files (readout; --final names another anchor, whose T2 score is
  paths.T2/score_t2_<FINAL>.json, or score_t2.json for e13s_FINw1536_s2, which kept that name);
  small steps from evolved states (sd_score.py, dt 1 s) vs R;
  speed (paths.OUT/speed_<X>_idle_keep.json, us per cell at 1e5 cells) and the 1000-day run (paths.OUT/cyc1000_<X>.json),
  when present.
Pass rule: helps() below (rules_t5 ADOPT, or lasting > 0.5 inside X <= 0.90 x R with fixed >= 2 x broken,
late > 0.5 inside X <= R, P3 and every guard G1-G7); then SD (non-finite rows 0, rows > 0.5 inside <= 1.10 x R + 3, rows > 2
<= R + 1), speed <= 10 us per cell and a bounded 1000-day run (finite, nothing rejected, no kept input, last five cycle
changes <= 0.1 dex). The later pass rule v2 (pass_v2.py: large misses as readouts) is not wired in.
Inputs: the benches of X, R and FINAL in paths.E (run/bench_dev.sh, all sets incl. T2 and SD1), X's T2 score
paths.T2/score_t2_<X>.json. Writes <out>/verdict_<X>_vs_<R>.{txt,json}, <out>/verdict_<X>_vs_<FINAL>.{txt,json} and
<out>/judge_<X>.txt; last line VERDICT.
    SCORE_RULE=phys HARNESS_NORM= python judge.py <X> <R> [--final FINAL] [--out DIR]   (DIR default paths.OUT)"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

assert os.environ.get("SCORE_RULE") == "phys" and not os.environ.get("HARNESS_NORM")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sd_score as SD  # noqa: E402  (puts lib/ on sys.path)
import paths  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("x")
ap.add_argument("r")
ap.add_argument("--final", default="e13s_FINw1536_s2")
ap.add_argument("--out", default=str(paths.OUT))
a = ap.parse_args()
CB, LD, FIN, OUT = a.x, a.r, a.final, Path(a.out)


def r5(x, r, extra=()):
    stem = OUT / f"verdict_{x}_vs_{r}"
    js = stem.with_suffix(".json")
    if js.exists():
        js.unlink()
    subprocess.run([sys.executable, str(HERE / "rules_t5.py"), x, r, "--final", r, "--out", str(stem), *extra],
                   cwd=HERE, check=False, stdout=subprocess.DEVNULL)
    return json.loads(js.read_text()) if js.exists() else {"verdict": "INVALID"}


def helps(j, x, r):
    if j["verdict"] in ("ADOPT",):
        return True
    if j["verdict"] == "INVALID":
        return False
    c, g = j["counts"]["inside"], j["rules"]
    return (c[x]["last05"] <= 0.90 * c[r]["last05"] and c["last_fixed"] >= 2 * c["last_broken"] and c[x]["late05"] <= c[r]["late05"]
            and g["P3"]["pass"] and all(g[f"G{i}"]["pass"] for i in range(1, 8)))


L = []
jl = r5(CB, LD)
t2f = paths.T2 / f"score_t2_{FIN}.json"
t2f = t2f if t2f.exists() else paths.T2 / "score_t2.json"   # FINAL's T2 score kept the name score_t2.json
jf = r5(CB, FIN, ("--t2r", str(t2f), "--t2npz-r", str(paths.E / f"H_t2tag_{FIN}" / "H_tag_dt500.npz")))
for nm, j, r in ((f"{CB} vs {LD}", jl, LD), (f"{CB} vs {FIN}", jf, FIN)):
    if j["verdict"] == "INVALID":
        L.append(f"{nm}: INVALID")
        continue
    c = j["counts"]["inside"]
    L.append(f"{nm}: {j['verdict']}; lasting inside {c[CB]['last05']} vs {c[r]['last05']} (fixed {c['last_fixed']}, broken "
             f"{c['last_broken']}); late > 0.5 {c[CB]['late05']} vs {c[r]['late05']}; late > 2 {c[CB]['late2']} vs {c[r]['late2']}; "
             f"cooler lasting {j['counts']['cooler'][CB]['last05']} vs {j['counts']['cooler'][r]['last05']}; T2 mean "
             f"{j.get('t2_mean_change', float('nan')):+.3f}; guards "
             f"{' '.join(k + (':ok' if j['rules'][k]['pass'] else ':FAIL') for k in j['rules'] if k.startswith('G'))}")
try:
    sa, sb = SD.score(CB, "1"), SD.score(LD, "1")
    sd_ok = (sa["nonfinite_rows"] == 0 and sa["inside"]["gt05"] <= 1.10 * sb["inside"]["gt05"] + 3
             and sa["inside"]["gt2"] <= sb["inside"]["gt2"] + 1)
    L.append(f"SD dt 1 s inside: rows > 0.5 {CB} {sa['inside']['gt05']} {LD} {sb['inside']['gt05']}; > 2 {sa['inside']['gt2']} vs "
             f"{sb['inside']['gt2']} -> {'ok' if sd_ok else 'FAIL'}")
except FileNotFoundError:
    sd_ok = False
    L.append("SD dt 1 s: missing")
sp = paths.OUT / f"speed_{CB}_idle_keep.json"
speed = json.loads(sp.read_text())["gpu"]["100000"]["us_per_cell"] if sp.exists() else None
cy = paths.OUT / f"cyc1000_{CB}.json"
cyc = None
if cy.exists():
    cj = json.loads(cy.read_text())
    cyc = (cj["run"].rstrip("/").endswith(CB) and cj["days_min"] >= 1000 and bool(cj["finite"]) and sum(cj["rejected"]) == 0
           and all(r[1] == 0 for r in cj["projfail"]) and max(cj["cycle_change_last5_max"]) <= 0.1)
L.append(f"speed {speed if speed is not None else 'PENDING'} us/cell; 1000-day run {'PENDING' if cyc is None else ('PASS' if cyc else 'FAIL')}")
ok = helps(jl, CB, LD) and sd_ok
fin = ok and speed is not None and speed <= 10.0 and cyc is True
L.append(f"VERDICT: {f'{CB} is the new candidate' if fin else (f'{CB} beats {LD} (speed / 1000-day pending)' if ok and (speed is None or cyc is None) else f'{LD} stays the candidate')}")
(OUT / f"judge_{CB}.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
