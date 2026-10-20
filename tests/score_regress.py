"""Stored-output scorers rerun against their stored results (CPU only, no model call, nothing outside the work folder
written). Every script runs from this repository; the stored results stay in the scale phase's _build (NB below).

  (i)   bench/sweep/score_sweep.py on the four stored sweeps (CBg, LD1, FINAL, LD; labels and folders as recorded in
        the stored score.json) -> score.json and score.txt byte-identical to the stored ones, score_rows.npz array-equal.
  (ii)  bench/score_minichem.py 1 20 500 on the stored fig4 benches -> median / max / end of the worst gas per call for
        CBg and FINAL in the six cases equal the numbers logged in paper-notes/notes.md (entry "2026-10-19: raw-gas cases at
        1, 20 and 500 s"), LD's two dt-20 cases in the scale phase's log.txt (2026-10-19 14:04), and mini-chem's two dt-20
        cases at CBg's calls in notes.md (entry "2026-10-19: CBg on two mini-chem paper cases at dt 20 s").
  (iii) the CBg verdict: bench/rules_t5.py run as the CBg judge ran it (CBg vs LD with --final LD; CBg vs FINAL with
        --final FINAL and FINAL's T2 files) -> verdict json and txt byte-identical to _build/p/verdict_e13s_CBg_s2_vs_*;
        score_mask's per-row caches are rebuilt in the work folder (SCORE_MASK_CACHE) and compared with the stored ones.
  (iv)  only with --boards-save DIR or --boards-check DIR: bench/scoreboard.py (tag 'regress') and bench/path_board.py
        (DEV paths, --vs LD) on CBg's stored benches, saved to DIR or compared with DIR: every value already in the saved
        json must be equal; keys only in the new json are listed.
  (v)   bench/judge.py e13s_CBg_s2 e13s_LD_s3 (out: the work folder) -> both verdicts byte-identical to the stored ones and
        judge_e13s_CBg_s2.txt equal to the stored _build/p/judge_cbg.txt with the run names written out (CB, LD, FINAL ->
        their labels; "CB-VERDICT" -> "VERDICT").
Every scorer imports the boards and scorerule from this repository (the module each one loaded is printed), with
SCORE_RULE=phys, HARNESS_NORM unset and no GPU.
    taskset -c <cpus> $PY tests/score_regress.py [--work DIR] [--boards-save DIR | --boards-check DIR]
Exit 0 = PASS.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NB = Path("/home/imalsky/Chemulator_Project/experiments/2026-10-13_e13-scale/_build")
E = Path("/home/imalsky/Chemulator_Project/experiments/2026-10-03_repo-sweep/_build/g2/E13_gcm")
DEV = Path("/home/imalsky/Chemulator_Project/experiments/2026-10-09_e13-conservation/_build/paths_v3/dev")
CB, LD, FIN = "e13s_CBg_s2", "e13s_LD_s3", "e13s_FINw1536_s2"
ENV = {"CUDA_VISIBLE_DEVICES": "", "SCORE_RULE": "phys", "HARNESS_NORM": "", "PYTHONDONTWRITEBYTECODE": "1",
       "PYTHONPATH": f"{REPO / 'bench'}:{REPO / 'lib'}"}
# run a script as `python script args` would (its folder first on sys.path), then report where the shared modules came from
RUNNER = ("import runpy, sys; s = sys.argv[1]; sys.argv = sys.argv[1:]; sys.path.insert(0, s.rsplit('/', 1)[0])\n"
          "try:\n    runpy.run_path(s, run_name='__main__')\nfinally:\n"
          "    print('MODULES', {m: getattr(sys.modules.get(m), '__file__', None) for m in ('scoreboard', 'scorerule')})")
CASES = ["1000K 1mbar", "1500K 1mbar", "2000K 1mbar", "1000K 10bar", "1500K 10bar", "2000K 10bar"]
# notes.md, 2026-10-19 "raw-gas cases at 1, 20 and 500 s": worst gas per call, median/max/end, the six cases in order
FIG4 = {("500", "CBg"): "0.02/0.03/0.02 0.08/0.53/0.12 0.00/2.11/0.00 0.07/0.88/0.07 0.04/0.69/0.04 0.01/0.09/0.01",
        ("500", "FINAL"): "0.08/0.28/0.18 0.12/0.56/0.16 0.00/2.00/0.00 0.13/0.91/0.13 0.02/0.71/0.02 0.00/0.05/0.00",
        ("20", "CBg"): "0.00/0.00/0.00 0.18/0.59/0.13 0.01/1.59/0.00 0.74/1.19/0.61 0.03/0.28/0.01 0.01/0.23/0.01",
        ("20", "FINAL"): "0.00/0.00/0.00 0.11/0.98/0.09 0.00/1.90/0.00 0.35/1.52/0.27 0.11/0.31/0.11 0.00/0.45/0.00",
        ("1", "CBg"): "0.00/0.00/0.00 0.66/1.85/0.45 1.89/4.78/0.31 1.17/1.20/1.14 0.07/1.14/0.06 0.01/2.24/0.01",
        ("1", "FINAL"): "0.00/0.00/0.00 0.32/1.92/0.26 1.44/5.10/0.17 0.33/0.53/0.37 0.22/0.60/0.14 0.00/2.44/0.00"}
FIG4_LOG = {("20", "LD", "1500K 1mbar"): "0.12/0.85/0.08", ("20", "LD", "1000K 10bar"): "0.41/1.26/0.34"}   # log.txt 14:04
# notes.md, 2026-10-19 "CBg on two mini-chem paper cases at dt 20 s": mini-chem at CBg's 5,000 calls of 20 s
FIG4_MC = {("20", "CBg", "1500K 1mbar"): "0.91/3.83/0.88", ("20", "CBg", "1000K 10bar"): "0.30/0.30/0.30"}


def run(script, args, cwd, log, env_extra=None):
    t = time.time()
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, "-c", RUNNER, str(script), *map(str, args)], cwd=cwd, stdout=f,
                           stderr=subprocess.STDOUT, env={**os.environ, **ENV, **(env_extra or {})}, check=False)
    txt = Path(log).read_text()
    mods = re.findall(r"^MODULES (.*)$", txt, re.MULTILINE)
    origin = eval(mods[-1]) if mods else {}
    wrong = {m: f for m, f in origin.items() if f is not None and not str(f).startswith(str(REPO))}
    print(f"  ran {Path(script).name} {' '.join(map(str, args))[:150]} (cwd {cwd}): exit {r.returncode}, {time.time() - t:.0f} s;"
          f" modules {origin}", flush=True)
    assert not wrong, f"{script} imported shared modules from outside {REPO}: {wrong}"
    return r.returncode, txt


def same_bytes(a, b):
    return Path(a).read_bytes() == Path(b).read_bytes()


def npz_equal(a, b):
    x, y = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    bad = sorted(set(x.files) ^ set(y.files))
    for k in set(x.files) & set(y.files):
        u, v = x[k], y[k]
        eq = u.shape == v.shape and (np.array_equal(u, v, equal_nan=True) if u.dtype.kind == "f" else np.array_equal(u, v))
        if not eq:
            bad.append(k)
    return bad


def check_sweep(work):
    d = NB / "dtsweep"
    meta = json.loads((d / "score.json").read_text())["_meta"]
    args = [work / "sweep" / "score"] + [f"{k}={v}" for k, v in meta["dirs"].items()]
    (work / "sweep").mkdir()
    rc, _ = run(REPO / "bench" / "sweep" / "score_sweep.py", args, d, work / "sweep" / "log.txt")
    ok = rc == 0
    for ext in (".json", ".txt"):
        eq = rc == 0 and same_bytes(work / "sweep" / f"score{ext}", d / f"score{ext}")
        print(f"  score{ext}: {'byte-identical' if eq else 'DIFFERS'} (stored {d / ('score' + ext)})")
        ok &= eq
    bad = npz_equal(work / "sweep" / "score_rows.npz", d / "score_rows.npz") if rc == 0 else ["(not written)"]
    print(f"  score_rows.npz: {'every array equal' if not bad else 'DIFFERS: ' + ', '.join(bad[:10])}")
    return ok and not bad


def check_fig4(work):
    (work / "fig4").mkdir()
    rc, txt = run(REPO / "bench" / "score_minichem.py", ["1", "20", "500"], REPO / "bench", work / "fig4" / "log.txt")
    got, mc, dt = {}, {}, None
    for line in txt.splitlines():
        m = re.match(r"dt (\S+) s:", line)
        if m:
            dt = m.group(1)
            continue
        m = re.match(r"\s+(\w+)\s+(1000K 1mbar: .*)$", line)
        if m and dt:
            for case, cell in re.findall(r"(\d+K \d+\w+): (\S+) \w+", m.group(2)):
                got[(dt, m.group(1), case)] = cell
        m = re.match(r"\s+minichem-at-(\w+)-calls\s+(1000K 1mbar: .*)$", line)
        if m and dt:
            for case, cell in re.findall(r"(\d+K \d+\w+): (\S+) \w+", m.group(2)):
                mc[(dt, m.group(1), case)] = cell
    want = {(dt, mod, c): v for (dt, mod), s in FIG4.items() for c, v in zip(CASES, s.split())}
    want.update(FIG4_LOG)
    bad = [(k, want[k], got.get(k)) for k in want if got.get(k) != want[k]]
    bad += [(("mini-chem",) + k, v, mc.get(k)) for k, v in FIG4_MC.items() if mc.get(k) != v]
    n = len(want) + len(FIG4_MC)
    print(f"  {n} logged cells (CBg and FINAL x 3 dt x 6 cases from notes.md; LD 2 cells from log.txt; mini-chem 2 cells "
          f"from notes.md): {n - len(bad)} equal" + ("" if not bad else f"; DIFFER: {bad[:8]}"))
    return rc == 0 and not bad


def check_verdict(work):
    design = work / "design"
    design.mkdir()
    ok = True
    for r, extra in ((LD, []), (FIN, ["--t2r", NB / "t2" / "score_t2.json", "--t2npz-r", E / f"H_t2tag_{FIN}" / "H_tag_dt500.npz"])):
        stem = work / f"verdict_{CB}_vs_{r}"
        rc, _ = run(REPO / "bench" / "rules_t5.py", [CB, r, "--final", r, "--out", stem, *extra], REPO / "bench",
                    work / f"log_rules_{r}.txt", {"SCORE_MASK_CACHE": str(design)})
        for ext in (".json", ".txt"):
            ref = NB / "p" / f"verdict_{CB}_vs_{r}{ext}"
            eq = rc == 0 and Path(str(stem) + ext).exists() and same_bytes(str(stem) + ext, ref)
            print(f"  verdict {CB} vs {r}{ext}: {'byte-identical' if eq else 'DIFFERS'} (stored {ref})")
            ok &= eq
        if rc == 0:
            j = json.loads(Path(str(stem) + ".json").read_text())
            print(f"    verdict {j['verdict']}; lasting inside {j['counts']['inside'][CB]['last05']} vs {j['counts']['inside'][r]['last05']}")
    for lab in (CB, LD, FIN):
        new, old = design / f"rows_{lab}.npz", NB / "t5" / "design" / f"rows_{lab}.npz"
        bad = npz_equal(new, old) if new.exists() else ["(not rebuilt)"]
        print(f"  rebuilt row cache rows_{lab}.npz vs stored: {'every array equal' if not bad else 'DIFFERS: ' + ', '.join(bad)}")
        ok &= not bad
    return ok


def check_judge(work):
    out = work / "judge"
    out.mkdir()
    rc, txt = run(REPO / "bench" / "judge.py", [CB, LD, "--out", out], REPO / "bench", work / "log_judge.txt")
    ok = rc == 0
    for r in (LD, FIN):
        for ext in (".json", ".txt"):
            new, ref = out / f"verdict_{CB}_vs_{r}{ext}", NB / "p" / f"verdict_{CB}_vs_{r}{ext}"
            eq = new.exists() and same_bytes(new, ref)
            print(f"  judge verdict {CB} vs {r}{ext}: {'byte-identical' if eq else 'DIFFERS'} (stored {ref})")
            ok &= eq
    want = (NB / "p" / "judge_cbg.txt").read_text().replace("CB-VERDICT", "VERDICT")
    for a, b in (("CB", CB), ("LD", LD), ("FINAL", FIN)):
        want = re.sub(rf"\b{a}\b", b, want)
    got = (out / f"judge_{CB}.txt").read_text() if (out / f"judge_{CB}.txt").exists() else ""
    print(f"  judge_{CB}.txt vs stored judge_cbg.txt (names written out): {'equal' if got == want else 'DIFFERS'}")
    if got != want:
        print("    stored:", want, "    new:", got, sep="\n")
    print("  " + (got.splitlines()[-1] if got else "(no judge output)"))
    return ok and got == want


def diff_json(old, new, path=""):
    """Paths where a value of old is missing or different in new; keys only in new are returned separately."""
    if isinstance(old, dict) and isinstance(new, dict):
        bad, extra = [], [f"{path}/{k}" for k in new if k not in old]
        for k in old:
            if k not in new:
                bad.append(f"{path}/{k} (missing)")
            else:
                b, e = diff_json(old[k], new[k], f"{path}/{k}")
                bad += b
                extra += e
        return bad, extra
    same = json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True)
    return ([] if same else [f"{path}: {old!r} -> {new!r}"]), []


def boards(work, save, check):
    out = work / "boards"
    out.mkdir()
    env = {"HARNESS_OUT": str(out)}
    run(REPO / "bench" / "scoreboard.py", ["regress", CB], REPO / "bench", out / "log_scoreboard.txt", env)
    run(REPO / "bench" / "path_board.py", [DEV, CB, "--vs", LD], REPO / "bench", out / "log_path_board.txt", env)
    files = ["scoreboard_regress.json", "path_board_PD.json"]
    if save:
        Path(save).mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copyfile(out / f, Path(save) / f)
        print(f"  boards saved to {save}: {files}")
        return True
    ok = True
    for f in files:
        bad, extra = diff_json(json.loads((Path(check) / f).read_text()), json.loads((out / f).read_text()))
        print(f"  {f} vs {Path(check) / f}: {'every saved value equal' if not bad else f'{len(bad)} DIFFER: {bad[:6]}'}"
              + (f"; new keys (information): {extra[:12]}{' ...' if len(extra) > 12 else ''}" if extra else ""))
        ok &= not bad
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--boards-save", default=None)
    g.add_argument("--boards-check", default=None)
    a = ap.parse_args()
    tmp = os.environ.get("TMPDIR")
    assert tmp and not tmp.startswith("/tmp"), "TMPDIR must be set to your own scratch (run/env.sh)"
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="score_regress_", dir=tmp))
    work.mkdir(parents=True, exist_ok=True)
    print(f"score_regress: work {work}", flush=True)
    res, t0 = {}, time.time()
    checks = [("(i) step-size sweep scorer", check_sweep), ("(ii) fig4 / mini-chem scorer", check_fig4),
              ("(iii) CBg verdict", check_verdict), ("(v) judge template", check_judge)]
    if a.boards_save or a.boards_check:
        checks.append(("(iv) boards", lambda w: boards(w, a.boards_save, a.boards_check)))
    for name, fn in checks:
        t = time.time()
        print(f"{name}:", flush=True)
        res[name] = fn(work)
        print(f"{name}: {'PASS' if res[name] else 'FAIL'} ({time.time() - t:.0f} s)", flush=True)
    ok = all(res.values())
    print(f"total {time.time() - t0:.0f} s")
    print(f"RESULT score_regress: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
