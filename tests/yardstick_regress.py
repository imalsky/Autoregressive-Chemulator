"""The yardstick scorer (bench/yardstick.py): rule v3.1 on synthetic inputs, and rule v3 rerun on the stored yardstick
inputs and compared with the stored result (CPU only, no model call; nothing outside the work folder is written).

(a) Rule v3.1 on synthetic inputs (unit_checks, through solver_model, failed_points, fail_rates, exact_sums and
    case_means): a successful LM rescue is not a failure (v3: it is); a failed rescue (kept input at call 5) fails the
    row from that call on and every later saved call scores the 2 dex penalty (failed calls N - 5 + 1); an invalid
    saved state is a failure under both rules; a piece crossing the 10 s edge is split exactly; a decade touched only by
    a call exactly at its upper edge has zero covered log-time, is untested and does not enter the case mean (v3 gives it
    0.5 dex of weight).
(b) Runs `bench/yardstick.py --rule v3 --design <stored>/design --run <stored>/run --out <work>/score CBg FINAL LD1` and
compares with <stored>/score:
  score.json: every value of the stored json (numbers, booleans, strings, lists) must be equal at the same key path in
    the new json (each leaf compared as its json text, so floats exactly), except these keys, which the module does not
    reproduce on purpose:
      rule (label text), generated and seconds (facts of the run);
      models_missing and models_segment_only (the old scorer's deadline fallback, both empty in the stored result);
      posthoc_inputs/train and posthoc_inputs/train_note (a floor histogram of the training caches: a diagnosis of the
      training data read from the caches, not part of the scorer).
    Keys only in the new json are listed (information: readouts the module adds).
  large_misses_<solver>.csv for mini-chem, CBg, FINAL and LD1: byte-identical.
  tables.txt is not compared (its labels were reworded).
(c) Rule v3.1 pin: runs `bench/yardstick.py --rule v3.1` on the same inputs; for CBg, FINAL and LD1 vs mini-chem, every
    value of the stored readout posthoc/<model>_vs_minichem/exact_quadrature/lenient/{seg,free} (unrecovered-step
    failures, exact interval splitting: D, A, B summaries of the 12-gas mean and the worst gas, the per-bin band, per
    start, per decade, the 10 s anchor, failure rates, verdict) must be equal in comparisons/<model>_vs_minichem of the
    v3.1 json; CBg's segment and free-run D (trimmed [95%]) must print as +0.031 [+0.023, +0.038] and +0.039 [+0.006,
    +0.074].
(b) and (c) run concurrently.
    taskset -c <cpus> $PY tests/yardstick_regress.py [--work DIR] [--stored DIR]   (about 8 min, 2 processes)
--stored defaults to the yardstick folder under paths.PROJECT. Exit 0 = PASS."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib"))
import paths  # noqa: E402

sys.path.insert(0, str(REPO / "bench"))
import numpy as np  # noqa: E402
import yardstick as Y  # noqa: E402

STORED = paths.PROJECT / "experiments/2026-10-19_minichem-yardstick/_build"
MODELS = ("CBg", "FINAL", "LD1")
SKIP = {"/rule", "/generated", "/seconds", "/models_missing", "/models_segment_only", "/posthoc_inputs/train",
        "/posthoc_inputs/train_note"}


def diff_json(old, new, path=""):
    """Paths where a value of old is missing or different in new (SKIP paths ignored); keys only in new separately."""
    if path in SKIP:
        return [], []
    if isinstance(old, dict) and isinstance(new, dict):
        bad, extra = [], [f"{path}/{k}" for k in new if k not in old]
        for k in old:
            if k not in new:
                if f"{path}/{k}" not in SKIP:
                    bad.append(f"{path}/{k} (missing)")
                continue
            b, e = diff_json(old[k], new[k], f"{path}/{k}")
            bad += b
            extra += e
        return bad, extra
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        bad, extra = [], []
        for i, (u, v) in enumerate(zip(old, new)):
            b, e = diff_json(u, v, f"{path}/{i}")
            bad += b
            extra += e
        return bad, extra
    same = json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True)
    return ([] if same else [f"{path}: {old!r} -> {new!r}"]), []


def unit_checks(work):
    """Rule v3.1 on synthetic inputs, through the module's own functions (independent of the stored result)."""
    ok = True
    # (1), (2): three rows saved at calls 1, 2, 5, 10, 2000 (N = 2000), errors 0 everywhere. Row 0: LM rescue at call 2
    # (a successful rescue); row 1: LM rescue at call 2, kept input at call 5 (the rescue failed); row 2: clean.
    u = work / "unit"
    (u / "m").mkdir(parents=True, exist_ok=True)
    first = dict(first_newton=[2, 2, -1], first_kept=[-1, 5, -1], first_guard=[-1, -1, -1], first_nonfin=[-1, -1, -1],
                 first_fail=[2, 2, -1], newton_n=[1, 1, 0], kept_n=[0, 1, 0], guard_n=[0, 0, 0], nonfin_n=[0, 0, 0])
    np.savez(u / "m" / "model_states.npz", gases12=np.array(Y.GASES), logy12=np.zeros((15, 12)),
             **{k: np.array(v) for k, v in first.items()})
    D = type("D", (), {})()
    D.R, D.S, D.row, D.k = 3, 15, np.repeat(np.arange(3), 5), np.tile([1, 2, 5, 10, 2000], 3)
    D.logt, D.seg_row = np.zeros((15, 12)), np.ones(3, bool)
    D.r = {"t0": np.zeros(3), "off": np.array([0, 5, 10, 15]), "N": np.full(3, 2000)}
    got = {}
    for rule in ("v3.1", "v3"):
        D.exact = rule == "v3.1"
        st = Y.solver_model(D, u, "m")
        fp = Y.failed_points(D, st["ff"])
        e = np.where(fp, Y.PEN, st["mean12"])
        got[rule] = (st["ff"].tolist(), fp.reshape(3, 5).astype(int).tolist(), e.reshape(3, 5).tolist(),
                     Y.fail_rates(D, st["ff"], np.ones(3, bool))["failed_calls"])
    want = {"v3.1": ([0, 5, 0], [[0, 0, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0]],
                     [[0.0] * 5, [0.0, 0.0, 2.0, 2.0, 2.0], [0.0] * 5], 1996),
            "v3": ([2, 2, 0], [[0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
                   [[0.0, 2.0, 2.0, 2.0, 2.0], [0.0, 2.0, 2.0, 2.0, 2.0], [0.0] * 5], 3998)}
    for rule in ("v3.1", "v3"):
        ff, fp, e, fc = got[rule]
        print(f"  {rule}: first failed call per row {ff}; penalized saved calls {fp}; per-call statistic {e}; failed calls "
              f"{fc}")
    c1 = got["v3.1"][0][0] == 0 and got["v3.1"][1][0] == [0] * 5 and got["v3"][0][0] == 2
    c2 = got["v3.1"] == want["v3.1"] and got["v3"] == want["v3"]
    # invalid saved calls are failures under both readings: row 2's call 10 non-finite
    D.exact = True
    np.savez(u / "m" / "model_states.npz", gases12=np.array(Y.GASES),
             logy12=np.where((np.arange(15) == 13)[:, None], np.nan, 0.0), **{k: np.array(v) for k, v in first.items()})
    st = Y.solver_model(D, u, "m")
    c5 = st["ff"].tolist() == [0, 5, 10] and st["ff_literal"].tolist() == [2, 2, 10]
    print(f"  a non-finite saved state at call 10 of row 2: v3.1 {st['ff'].tolist()} (want [0, 5, 10]), v3 "
          f"{st['ff_literal'].tolist()} (want [2, 2, 10])")
    # (3), (4): row 0 starts at t = 0, saved at 10, 100, 1000 s, statistic 1, 3, 9; row 1 restarts at t0 = 3 s, saved at
    # 5, 20, 100 s, statistic 2, 4, 6 (its piece 5 -> 20 s crosses the 10 s edge). Linear in log10 t between saved calls,
    # held at the first value back to t0.
    q = type("D", (), {})()
    q.tabs = np.array([10.0, 100.0, 1000.0, 5.0, 20.0, 100.0])
    q.row, q.S = np.repeat(np.arange(2), 3), 6
    q.r = {"t0": np.array([0.0, 3.0]), "off": np.array([0, 3, 6])}
    q.seg_row, q.case_row = np.array([True, True]), np.array([0, 1])
    q.pieces = lambda: Y.Design.pieces(q)
    e = np.array([1.0, 3.0, 9.0, 2.0, 4.0, 6.0])
    sw, swe = Y.exact_sums(q, e, None, "seg", 2)
    V, case = Y.case_means(sw, swe, 2)
    l = np.log10
    a, v1 = 1.0 - l(5.0), 2.0 + 2.0 * (1.0 - l(5.0)) / l(4.0)              # piece 5 -> 20 s: part below 10 s, value at 10 s
    want_sw = np.zeros(16)
    want_swe = np.zeros(16)
    want_sw[[1, 2, 8, 9]] = [1.0, 1.0, l(5.0 / 3.0) + a, l(2.0) + l(5.0)]
    want_swe[[1, 2, 8, 9]] = [2.0, 6.0, 2.0 * l(5.0 / 3.0) + a * (2.0 + v1) / 2.0, l(2.0) * (v1 + 4.0) / 2.0 + l(5.0) * 5.0]
    c3 = np.allclose(sw, want_sw, rtol=0, atol=1e-12) and np.allclose(swe, want_swe, rtol=0, atol=1e-12)
    print(f"  boundary crossing: row 1 covered log-time 1e0 {sw[8]:.6f} (want {want_sw[8]:.6f}), 1e1 {sw[9]:.6f} (want "
          f"{want_sw[9]:.6f}); integrals {swe[8]:.6f} / {swe[9]:.6f} (want {want_swe[8]:.6f} / {want_swe[9]:.6f})")
    w = Y.Design._weights(q)
    dec = np.searchsorted(Y.EDGES, q.tabs, side="left") - 1
    Vn, cn = Y.case_means(np.bincount(q.case_row[q.row] * 8 + dec, w, 16),
                          np.bincount(q.case_row[q.row] * 8 + dec, w * e, 16), 2)
    c4 = bool(sw[0] == 0.0 and np.isnan(V[0, 0]) and case[0] == 4.0 and Vn[0, 0] == 1.0 and abs(cn[0] - 4.0) > 0.1)
    print(f"  zero-duration coverage: row 0's decade 1e0 (only a call exactly at 10 s) covered {sw[0]:.1f} dex -> untested "
          f"({V[0, 0]}); case score {case[0]:.4f} = mean of 1e1 {V[0, 1]:.1f} and 1e2 {V[0, 2]:.1f} (want 4.0); the whole-node "
          f"quadrature gives that decade {w[0]:.2f} dex at value {Vn[0, 0]:.1f} and a case score {cn[0]:.4f}")
    for nm, c in (("successful rescue is not a failure", c1), ("failed rescue penalizes the remainder", c2),
                  ("invalid saved calls", c5), ("boundary crossing split exactly", c3),
                  ("zero-duration coverage untested", c4)):
        print(f"  {nm}: {'ok' if c else 'WRONG'}")
        ok &= c
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    ap.add_argument("--stored", type=Path, default=STORED)
    a = ap.parse_args()
    tmp = os.environ.get("TMPDIR")
    assert tmp and not tmp.startswith("/tmp"), "TMPDIR must be set to your own scratch (run/env.sh)"
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="yardstick_regress_", dir=tmp))
    work.mkdir(parents=True, exist_ok=True)
    out, ref = work / "score", a.stored / "score"
    print(f"yardstick_regress: work {work}; stored {a.stored}", flush=True)
    t0 = time.time()
    print("(a) rule v3.1 on synthetic inputs:", flush=True)
    ok_a = unit_checks(work)
    print(f"(a) {'PASS' if ok_a else 'FAIL'}", flush=True)
    procs = {}
    for rule, o in (("v3", out), ("v3.1", work / "score_v31")):
        cmd = [sys.executable, str(REPO / "bench" / "yardstick.py"), "--rule", rule, "--design", str(a.stored / "design"),
               "--run", str(a.stored / "run"), "--out", str(o), *MODELS]
        f = open(work / f"log_{rule}.txt", "w")
        procs[rule] = (subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}), f)
    rc = {}
    for rule, (pr, f) in procs.items():
        rc[rule] = pr.wait()
        f.close()
        print(f"  ran yardstick.py --rule {rule}: exit {rc[rule]}, {time.time() - t0:.0f} s (log {work / f'log_{rule}.txt'})",
              flush=True)
    print("(b) rule v3 on the stored inputs vs the stored result:", flush=True)
    ok = rc["v3"] == 0
    if ok:
        old, new = (json.loads((p / "score.json").read_text()) for p in (ref, out))
        bad, extra = diff_json(old, new)
        print(f"  score.json vs {ref / 'score.json'}: " + ("every stored value equal" if not bad else
                                                             f"{len(bad)} DIFFER: {bad[:8]}"))
        print(f"  skipped keys: {sorted(SKIP)}")
        print(f"  keys only in the new json (information): {len(extra)}; first {extra[:10]}")
        ok &= not bad
        for s in ("minichem",) + MODELS:
            eq = (out / f"large_misses_{s}.csv").read_bytes() == (ref / f"large_misses_{s}.csv").read_bytes()
            print(f"  large_misses_{s}.csv: {'byte-identical' if eq else 'DIFFERS'}")
            ok &= eq
    print(f"(b) {'PASS' if ok else 'FAIL'}")
    print("(c) rule v3.1 pin: v3.1 comparisons vs the stored unrecovered-failure, exact-quadrature readout:", flush=True)
    ok_c = rc["v3.1"] == 0
    if ok_c:
        old = json.loads((ref / "score.json").read_text())["posthoc"]
        new = json.loads((work / "score_v31" / "score.json").read_text())
        ok_c &= new.get("rule_id") == "v3.1"
        for m in MODELS:
            for tab in ("seg", "free"):
                bad, _ = diff_json(old[f"{m}_vs_minichem"]["exact_quadrature"]["lenient"][tab],
                                   new["comparisons"][f"{m}_vs_minichem"][tab])
                d = new["comparisons"][f"{m}_vs_minichem"][tab]["mean12"]["D"]
                print(f"  {m:5s} {tab:4s} D trimmed {Y.fs(d)}: " + ("every stored value equal" if not bad else
                                                                     f"{len(bad)} DIFFER: {bad[:6]}"))
                ok_c &= not bad
        pin = {t: Y.fs(new["comparisons"]["CBg_vs_minichem"][t]["mean12"]["D"]) for t in ("seg", "free")}
        want = {"seg": "+0.031 [+0.023, +0.038]", "free": "+0.039 [+0.006, +0.074]"}
        print(f"  CBg pinned: segments {pin['seg']} (want {want['seg']}), free runs {pin['free']} (want {want['free']})")
        ok_c &= pin == want
    print(f"(c) {'PASS' if ok_c else 'FAIL'}")
    ok &= ok_a and ok_c
    print(f"total {time.time() - t0:.0f} s")
    print(f"RESULT yardstick_regress: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
