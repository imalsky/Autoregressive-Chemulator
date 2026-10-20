"""Development board: per bench label (E13_gcm/<set>_<label>):
  fixed-condition sets G, Gc, H2, H2c, H3, H3c: rows (job x start age) with a late (>= 1e3 s) point > 0.5 / > 2 dex,
  first-call rows > 2, the max over every checkpoint (readout only), invented-switch rows (late error > 2 dex in a species
  whose prediction moves > 2 dex over the late checkpoints while its truth moves < 0.5 dex, floors applied), rows with a
  failed Newton solve and rows that kept their input;
  path sets PD (DEV), PX (xlong), P2l, P2s, PH: the max over every call, late-window p50 / p99, solve counters.
Totals over the six fixed sets. A missing set is an error (BOARD_SETS restricts the sets on purpose).
Run with SCORE_RULE=phys, HARNESS_NORM unset (outputs in HARNESS_OUT, default paths.OUT).
    python board6.py <label> [<label> ...]      (writes HARNESS_OUT/board6_<label>.json per label)"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # bench/
import scoreboard as S  # noqa: E402  (puts lib/ on sys.path)
import paths as PA  # noqa: E402
import scorerule as SR  # noqa: E402

assert os.environ.get("SCORE_RULE") == "phys" and not os.environ.get("HARNESS_NORM")
HB = str(Path(__file__).resolve().parent)
PY = os.environ.get("PY", sys.executable)
OLD = PA.PATHS_V3
HSETS = {"G": ("g500", "H_gtest_dt500.npz"), "Gc": ("gc500", "H_gtest_cold_dt500.npz"), "H2": ("h2", "H_h2_dt500.npz"),
         "H2c": ("h2c", "H_h2c_dt500.npz"), "H3": ("h3", "H_h3_dt500.npz"), "H3c": ("h3c", "H_h3c_dt500.npz"),
         "H4": ("h4", "H_h4_dt500.npz"), "H4c": ("h4c", "H_h4c_dt500.npz")}
PSETS = {"PD": (f"{OLD}/dev", "2500-"), "PX": (f"{S.E}/paths_v2x", "2500-"), "P2l": (f"{S.E}/paths_v2l", "500-"),
         "P2s": (f"{S.E}/paths_v2s", "50-"), "PH": (f"{OLD}/hold", "2500-"),
         "PH2": (f"{OLD}/hold2", "2500-")}   # sealed: opened once
FIXED = ("G", "Gc", "H2", "H2c", "H3", "H3c")
SEALED = ("H4", "H4c")  # sealed sets, opened once on a frozen recipe: BOARD_SETS="H4 H4c PH2", totals over H4 + H4c alone
want = os.environ.get("BOARD_SETS", " ".join(FIXED + ("PD", "PX", "P2l", "P2s", "PH"))).split()
L = np.array([SR.PHYS[s] for s in S.SP])


def invented(B):
    """Rows with a late error > 2 dex in a species the model moves > 2 dex (late checkpoints) while the truth moves < 0.5."""
    late = slice(B["st"].shape[1] - B["e"].shape[1], None)
    st, vt = np.maximum(B["st"][:, late], L), np.maximum(B["vt"][:, late], L)
    mv_m, mv_t = st.max(1) - st.min(1), vt.max(1) - vt.min(1)
    big = np.nan_to_num(B["e"], nan=0.0).max(1) > 2.0
    return (big & (mv_m > 2.0) & (mv_t < 0.5)).any(1)


for lab in sys.argv[1:]:
    r = {}
    for k in want:
        if k in HSETS:
            pre, fn = HSETS[k]
            f = S.E / f"H_{pre}_{lab}" / fn
            if not f.exists():
                sys.exit(f"MISSING {k}: {f}")
            B = S.bench(f)
            d = np.load(f)
            lm = np.nan_to_num(B["e"], nan=0.0).max(axis=(1, 2))
            fm = np.nan_to_num(B["e1"], nan=0.0).max(axis=(1, 2))
            pf = d["projfail"] if "projfail" in d.files else None
            r[k] = {"rows": int(lm.size), "late_gt05": int((lm > 0.5).sum()), "late_gt2": int((lm > 2).sum()),
                    "first_gt2": int((fm > 2).sum()), "invented_gt2": int(invented(B).sum()),
                    "max_all": float(max(lm.max(), fm.max())),
                    "newton_fail_rows": None if pf is None else int((pf[..., 0].sum(0) > 0).sum()),
                    "kept_rows": None if pf is None else int((pf[..., 1].sum(0) > 0).sum())}
        elif k in PSETS:
            pdir, late = PSETS[k]
            f = S.E / f"{k}_{lab}" / "P_paths.npz"
            if not f.exists():
                sys.exit(f"MISSING {k}: {f}")
            out = f"board6_tmp_{k}_{lab}.json"
            subprocess.run([PY, f"{HB}/path_board.py", pdir, lab, "--prefix", k, "--windows", f"0-,{late}"],
                           env={**os.environ, "PB_OUT": out}, check=True, capture_output=True)
            j = json.load(open(PA.OUT / out))[lab]
            (PA.OUT / out).unlink()
            w_all = [v for key, v in j.items() if key.startswith("calls 0-")][0]
            w_late = [v for key, v in j.items() if key.startswith("calls ") and not key.startswith("calls 0-")][0]
            d = np.load(f)
            pf = d["projfail"] if "projfail" in d.files else None
            r[k] = {"paths": int(j["paths"]) if isinstance(j["paths"], (int, float)) else len(j["paths"]),
                    "max_all": float(w_all["max"]), "late_p50": float(w_late["p50"]), "late_p99": float(w_late["p99"]),
                    "newton_fail_rows": None if pf is None else int((pf[..., 0].sum(0) > 0).sum()),
                    "kept_rows": None if pf is None else int((pf[..., 1].sum(0) > 0).sum())}
    hs = [k for k in FIXED if k in r] or [k for k in SEALED if k in r]
    tot = {key: sum(r[k][key] for k in hs) for key in ("late_gt2", "late_gt05", "first_gt2", "invented_gt2")}
    tot.update(fixed_sets=hs, max_all=max(v["max_all"] for v in r.values()), dev_late_p99=r.get("PD", {}).get("late_p99"),
               kept_rows=sum((v["kept_rows"] or 0) for v in r.values()))
    r["total"] = tot
    print(f"== {lab}: over {'+'.join(hs)}: late rows > 2 {tot['late_gt2']}, > 0.5 {tot['late_gt05']}, first-call > 2 "
          f"{tot['first_gt2']}, invented > 2 {tot['invented_gt2']}; DEV late p99 {tot['dev_late_p99']}; max "
          f"{tot['max_all']:.2f}; kept-input rows {tot['kept_rows']}")
    for k, v in r.items():
        if k != "total":
            print(f"   {k:4s} " + ", ".join(f"{a} {b:.3f}" if isinstance(b, float) else f"{a} {b}" for a, b in v.items()))
    (PA.OUT / f"board6_{lab}.json").write_text(json.dumps(r, indent=1))
