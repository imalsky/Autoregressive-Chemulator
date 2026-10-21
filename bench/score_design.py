"""Design builder: turns a test set (generator output files) into a design that bench/score_run.py runs and
bench/score.py scores. CPU only; reads VULCAN output files, never runs VULCAN.

    python score_design.py seg  <tag dir>   <out dir> --panel S|Sraw|side [--role score|readout] [--opened NOTE]
    python score_design.py long <tag dir>   <out dir> --panel L [--role score|readout] [--opened NOTE]
    python score_design.py long <tag dir>   <out dir> --t2 [--name t2]                 (the T2 snap-back tag)
    python score_design.py path <paths dir> <out dir> [--panel P --role score] [--name NAME] [--opened NOTE]

Refuses an existing output and any source or output path containing "SEALED", "e13_holdout4" or "hold2" unless
--opened gives the author's decision note (written into the manifest). Every file of the design is listed with its
sha256 in manifest.json; the sha256 of manifest.json is the design's identity (score_run records it, score.py checks it).

Sources
  seg   a constant-condition tag: jobs.json (one entry per accepted history) and <i>.npz (t, x [rows, 53], species,
        restart_idx: the stored rows at the restart ages 1 s ... 3e7 s and 1e8 s). Entry keys: slot (condition slot),
        sobol_index (the Sobol point i; a refilled slot carries the new point's i), start (eqt, eqf, rawA, rawB), id,
        stratum (G, C), T (K), P (dyn cm^-2), Pbar, Z, C_O, He_H, dt_draws (8 values, one per bin, bin order), accepted.
        Rows: per parent, restart ages t0 in {0, 1, 3, ..., 3e7 s} x the 8 bin draws and dt = 10 s exactly (slot 8, outside
        the bins); N = 2000 at t0 = 0, else min(floor((t0_next - t0) / dt), 2000) with t0_next = 1e8 s after 3e7 s (exact on
        the binary values); N = 0 is untested (no row). Saved calls: 64 log-spaced (first and last included).
  long  a tag without restarts, stored to >= 172,800 x the largest dt: the same entry keys; per parent one row per bin from
        t = 0 at its draw, 172,800 calls, 256 saved calls. --t2: the T2 tag (entry keys T, P, src_set, src_job, t_target,
        kick k0 / k1m / k1p / k2m / k2p / k3, region), one row per box from t = 0 at 500 s for 172,800 calls; pairs.npz holds
        each kicked row's identity partner (same src_set, src_job, t_target), its class (k1 = k1m + k1p, k2 = k2m + k2p,
        k3), its source job (the bootstrap group) and its domain flag.
  path  a paths folder: <i>.npz per path and paths.json. Two layouts:
        - every call stored (x [n + 1, 53], species, T [n], Pbar [n], dt [n], optional mix_k, mix_f, mix_x): the older
          path sets; one path = one condition with start code 'path'; truth at a saved call k = x[k];
        - saved calls only (x0 [53], species, T, Pbar, dt, hold [n], mix_k, mix_f, mix_x, saved_k [m], logx_saved [m, 53];
          paths.json entries: path, slot, sobol_index, start, id, stratum, model, Z, C_O, He_H, dt0, variable_dt, mixed):
          the changing-condition panel; the two rows of a slot (eqt, eqf) must share the forcing, holds and mixing, the
          holds must be the longest runs of calls with summed dt <= 1,500 s and T, P must be constant within a hold.
        n = the shortest path's call count; 256 saved calls. A path is in the domain if every call lies inside
        1e-5-100 bar and 300-3000 K; the design keeps every path (the batch shape) and flags the others.
Every condition slot of a new panel must hold its starts with one Sobol index, one composition and one T-P point, and
history IDs base + block + 2 i (+ 1) (base 4.22e9, or 4.23e9 for the sealed twin; blocks S 0, Sraw 10,000, side 20,000,
L 30,000, P 40,000).

Design files (row r's saved points are [off[r], off[r + 1]) of saved.npz and truth_saved.npz)
  manifest.json      format, name, mode, panel, role, start codes and weights, slots, slots per bin, dt bins and weights,
                     bootstrap groups, tie margin delta, the retrain-spread statement, domain, counts, source, start map,
                     sha256 of every other file, opened note
  conditions.npz     key, stratum, group (bootstrap group), in_domain, T, P_bar, Z, C_O, He_H, variable_dt, mixed  [C]
  parents.npz        cond, start, id, job, kick, src, T, P_bar  [P]
  rows.npz           parent, unit, job, t0, dt (path rows: dt0), N, dt_idx (slot), dt_bin (-1 outside the bins), age_idx,
                     nsave, off [R + 1]
  saved.npz          k, tabs (time since the history's t = 0), tel (time since the row's start), row  [S]
  truth_saved.npz    gases12, logx12 [S, 12] = log10 max(x, 1e-30), linear in t between stored steps (constant designs)
  starts.npz         species53, x53 [U, 53] (linear mole fractions of the start units), logx53, parent, age_idx, t0, job,
                     T, P_barye  [U]
  forcing.npz        path designs: T, Pbar, dt, hold [R, n]; mix_row, mix_k, mix_f, mix_x (every recorded event); has_mix [R]
  pairs.npz          T2: kicked, partner (rows), cls, kick, src, in_domain
  groups.npz         rows (batches concatenated), off, nrun (calls per batch): segment designs as the stored yardstick
                     batches (N = 2000 rows in design order in the fewest equal chunks <= 16,384 rows, the others sorted
                     by (N, row) in chunks <= 16,384); long designs in design order in the fewest equal chunks; path
                     designs one batch in file order"""
import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

FORMAT = "score-1 design"
GASES = ["C2H2", "CH4", "CO2", "CO", "H2O", "H2", "HCN", "H", "N2", "NH3", "OH", "O"]
AGES = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7]
NEXT = AGES[1:] + [1e8]
NCAP, M_SEG, M_LONG, M_PATH, N_LONG, DT10, HOLD = 2000, 64, 256, 256, 172_800, 10.0, 1500.0
BINS = [(1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 50.0), (50.0, 100.0), (100.0, 200.0), (200.0, 500.0)]
BIN_W = [float(np.log(b / a) / np.log(500.0)) for a, b in BINS]
BMAX = 16384
STARTS = ["eqt", "eqf", "rawA", "rawB"]
W_EQ = {"eqt": 0.55, "eqf": 0.45, "rawA": 0.0, "rawB": 0.0}
W_RAW = {"eqt": 0.0, "eqf": 0.0, "rawA": 0.5, "rawB": 0.5}
BASES = {4_220_000_000: "decision set", 4_230_000_000: "sealed twin"}
BLOCK = {"S": 0, "Sraw": 10_000, "side": 20_000, "L": 30_000, "P": 40_000}
FIRST = {"eqt": 0, "eqf": 1, "rawA": 0, "rawB": 1}                # ID offset within a slot: base + block + 2 i + FIRST
DELTA = 0.005
SPREAD = "unmeasured: no retraining-noise test was run; the tie margin is fixed"
DOMAIN = {"P_bar": [1e-5, 100.0], "T_K": [300.0, 3000.0]}
SEALED = ("SEALED", "e13_holdout4", "hold2")
POLICY = ("P3 state, 32 Newton iterations, adaptive LM rescue, LHB_BLOCK 50, TF32 off, one model call per step, no "
          "startup substeps")
OUTS = ["manifest.json", "conditions.npz", "parents.npz", "rows.npz", "saved.npz", "truth_saved.npz", "starts.npz",
        "forcing.npz", "pairs.npz", "groups.npz"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def saved_calls(n, m):
    """unique(round(10 ** linspace(0, log10 n, m))), first = 1 and last = n forced."""
    k = np.unique(np.round(10.0 ** np.linspace(0.0, np.log10(n), m)).astype(np.int64))
    k[0], k[-1] = 1, n
    assert k[0] == 1 and k[-1] == n and np.all(np.diff(k) > 0)
    return k


def seg_n(t0, tn, dt):
    """N = min(floor((t0_next - t0) / dt), 2000), exact on the binary values; 2000 at t0 = 0."""
    if t0 == 0.0:
        return NCAP
    n = min(int((Fraction(tn) - Fraction(t0)) // Fraction(dt)), NCAP)
    assert n == 0 or t0 + n * dt <= tn, (t0, tn, dt, n)
    return n


def dt_bin(dt):
    """Bin index of a step: b with lo < dt <= hi (1 s goes to the first bin), -1 outside 1-500 s."""
    for b, (lo, hi) in enumerate(BINS):
        if (lo < dt or (b == 0 and dt == lo)) and dt <= hi:
            return b
    return -1


def holds(dt, h=HOLD):
    """Hold index per call: a hold is the longest run of consecutive calls whose summed dt is <= h (at least one call)."""
    out = np.empty(len(dt), np.int64)
    k, s = 0, 0.0
    for i, d in enumerate(dt):
        if i and s + d > h:
            k, s = k + 1, 0.0
        s += d
        out[i] = k
    return out


def in_domain(Pbar, T):
    return bool((np.min(Pbar) >= DOMAIN["P_bar"][0]) and (np.max(Pbar) <= DOMAIN["P_bar"][1])
                and (np.min(T) >= DOMAIN["T_K"][0]) and (np.max(T) <= DOMAIN["T_K"][1]))


def refuse_sealed(paths, opened):
    for p in paths:
        for s in (str(p), str(Path(p).resolve())):
            if any(tok in s for tok in SEALED) and not opened:
                raise SystemExit(f"refusing a sealed path ({p}): give --opened with the author's decision note")


def code_version():
    here = Path(__file__).resolve().parents[1]
    r = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True)
    d = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", "bench/score_design.py"], capture_output=True,
                       text=True)
    return {"git_head": r.stdout.strip(), "score_design_modified": bool(d.stdout.strip())}


def check_slots(jobs, panel):
    """Every slot: its starts share one Sobol index, composition and T-P point; IDs base + block + 2 i (+ 1)."""
    by = {}
    for i, j in enumerate(jobs):
        by.setdefault(int(j["slot"]), []).append(i)
    base = None
    for s, idx in sorted(by.items()):
        js = [jobs[i] for i in idx]
        ref = js[0]
        for j in js:
            for k in ("sobol_index", "Z", "C_O", "He_H", "T", "P", "stratum"):
                assert j[k] == ref[k], f"slot {s}: {k} differs between its starts ({j[k]} vs {ref[k]})"
            assert j["start"] in STARTS, f"slot {s}: unknown start {j['start']}"
            b = int(j["id"]) - BLOCK[panel] - 2 * int(j["sobol_index"]) - FIRST[j["start"]]
            assert b in BASES, f"slot {s} {j['start']}: id {j['id']} is not base + block + 2 i (+ 1) for i = {j['sobol_index']}"
            base = b if base is None else base
            assert b == base, "IDs from two bases"
        assert len({j["start"] for j in js}) == len(js), f"slot {s}: a start twice"
    return by, base


def write(out, arrays, man, opened):
    """Write the design files, then the manifest with every file's sha256; refuses to overwrite."""
    out.mkdir(parents=True, exist_ok=True)
    for f in OUTS:
        assert not (out / f).exists(), f"refusing to overwrite {out / f}"
    for name, a in arrays.items():
        np.savez(out / name, **a)
    man["files"] = {name: sha256(out / name) for name in arrays}
    man.update(format=FORMAT, delta=DELTA, retrain_spread=SPREAD, domain=DOMAIN, dt_bins=BINS, dt_bin_weights=BIN_W,
               policy=POLICY, opened=opened, code=code_version())
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"design {out}: {json.dumps(man['counts'])}; manifest sha256 {sha256(out / 'manifest.json')}")


def csr(saved):
    """(row list of k, tabs, tel, truth) -> saved.npz and truth arrays with offsets."""
    k = np.concatenate([s[0] for s in saved]).astype(np.int32)
    nsave = np.array([s[0].size for s in saved], np.int32)
    off = np.concatenate([[0], np.cumsum(nsave.astype(np.int64))])
    row = np.repeat(np.arange(len(saved), dtype=np.int32), nsave)
    return dict(k=k, tabs=np.concatenate([s[1] for s in saved]), tel=np.concatenate([s[2] for s in saved]), row=row), \
        np.concatenate([s[3] for s in saved]), nsave, off


def groups_seg(N):
    full = np.flatnonzero(N == NCAP)
    chunks = np.array_split(full, -(-full.size // BMAX)) if full.size else []
    short = np.flatnonzero(N < NCAP)
    short = short[np.lexsort((short, N[short]))]
    chunks = list(chunks) + [short[i:i + BMAX] for i in range(0, short.size, BMAX)]
    return chunks


def groups_npz(chunks, N):
    allr = np.concatenate(chunks)
    assert np.array_equal(np.sort(allr), np.arange(N.size)), "every row exactly once"
    return dict(rows=allr.astype(np.int32), off=np.concatenate([[0], np.cumsum([c.size for c in chunks])]).astype(np.int64),
                nrun=np.array([int(N[c].max()) for c in chunks], np.int64))


def truth_at(t, x, gcol, tabs):
    """log10 max(np.interp(tabs, t, x[:, gas]), 1e-30) per reported gas (stored steps, linear in t and x)."""
    return np.log10(np.maximum(np.stack([np.interp(tabs, t, x[:, c]) for c in gcol], -1), 1e-30))


# ------------------------------------------------------------------------------------------------ constant conditions
def build_tag(src, out, mode, panel, role, opened, t2=False, name=None):
    jobs = json.loads((src / "jobs.json").read_text())
    if t2:
        return build_t2(src, out, jobs, role, opened, name or "t2")
    assert all(j.get("accepted", False) for j in jobs), "the tag holds a rejected history"
    by, base = check_slots(jobs, panel)
    slots = sorted(by)
    C = len(slots)
    cpos = {s: c for c, s in enumerate(slots)}
    sp53, gcol = None, None
    par, units, rows, saved, untested = [], [], [], [], []
    for c, s in enumerate(slots):
        for i in sorted(by[s], key=lambda i: STARTS.index(jobs[i]["start"])):
            j = jobs[i]
            z = np.load(src / f"{i}.npz")
            t, x, sp = z["t"], z["x"], [str(v) for v in z["species"]]
            if sp53 is None:
                sp53, gcol = sp, [sp.index(g) for g in GASES]
            assert sp == sp53
            dts = [float(d) for d in j["dt_draws"]]
            assert len(dts) == 8 and all(dt_bin(d) == q for q, d in enumerate(dts)), f"job {i}: dt draws {dts}"
            p = len(par)
            par.append(dict(cond=c, start=STARTS.index(j["start"]), id=int(j["id"]), job=i, T=float(j["T"]),
                            P_bar=float(j["Pbar"]), P=float(j["P"])))
            tabs_p, rows_p = [], []
            if mode == "seg":
                ri = z["restart_idx"]
                assert np.array_equal(t[ri], np.array(NEXT)), f"job {i}: stored rows at the restart ages"
                xs = np.stack([np.interp(np.array(AGES), t, x[:, k]) for k in range(53)], -1)
                for a, t0 in enumerate(AGES):
                    units.append(dict(parent=p, age_idx=a, t0=t0, job=i, x=xs[a], T=float(j["T"]), P=float(j["P"])))
                for a, (t0, tn) in enumerate(zip(AGES, NEXT)):
                    u = len(units) - len(AGES) + a
                    for q, dt in enumerate(dts + [DT10]):
                        n = seg_n(t0, tn, dt)
                        if n == 0:
                            untested.append((p, a, q))
                            continue
                        rows_p.append(dict(parent=p, unit=u, job=i, t0=t0, dt=dt, N=n, dt_idx=q, dt_bin=q if q < 8 else -1,
                                           age_idx=a, m=M_SEG))
            else:
                assert float(t[-1]) >= N_LONG * max(dts), f"job {i}: stored to {t[-1]}"
                units.append(dict(parent=p, age_idx=0, t0=0.0, job=i,
                                  x=np.stack([np.interp(np.array([0.0]), t, x[:, k]) for k in range(53)], -1)[0],
                                  T=float(j["T"]), P=float(j["P"])))
                for q, dt in enumerate(dts):
                    rows_p.append(dict(parent=p, unit=len(units) - 1, job=i, t0=0.0, dt=dt, N=N_LONG, dt_idx=q, dt_bin=q,
                                       age_idx=0, m=M_LONG))
            for r in rows_p:
                k = saved_calls(r["N"], r["m"])
                tel = k * r["dt"]
                tabs = r["t0"] + k * r["dt"]
                assert np.all(np.diff(tabs) > 0) and tabs[-1] <= float(t[-1])
                saved.append([k, tabs, tel, None])
                tabs_p.append(tabs)
                rows.append(r)
            tr = truth_at(t, x, gcol, np.concatenate(tabs_p))
            o = 0
            for sv in saved[len(saved) - len(rows_p):]:
                sv[3] = tr[o:o + sv[0].size]
                o += sv[0].size
    js0 = [jobs[by[s][0]] for s in slots]
    cond = dict(key=np.array([f"{panel}:{s}:i{js0[c]['sobol_index']}" for c, s in enumerate(slots)]),
                stratum=np.array([str(j["stratum"]) for j in js0]),
                group=np.array([{"G": 0, "C": 1}[j["stratum"]] for j in js0], np.int32),
                in_domain=np.array([in_domain([float(j["Pbar"])], [float(j["T"])]) for j in js0]),
                T=np.array([float(j["T"]) for j in js0]), P_bar=np.array([float(j["Pbar"]) for j in js0]),
                Z=np.array([float(j["Z"]) for j in js0]), C_O=np.array([float(j["C_O"]) for j in js0]),
                He_H=np.array([float(j["He_H"]) for j in js0]), variable_dt=np.zeros(C, bool), mixed=np.zeros(C, bool),
                slot=np.array(slots, np.int32), sobol_index=np.array([int(j["sobol_index"]) for j in js0], np.int64))
    assert cond["in_domain"].all(), "a panel condition outside the domain"
    return finish(out, mode, panel, role, opened, name or panel, cond, par, rows, saved, units, sp53,
                  W_RAW if panel == "Sraw" else W_EQ, "stratum",
                  extra_counts=dict(untested_triples=len(untested), id_base=base, set=BASES[base]),
                  source=dict(dir=str(src), kind="tag", jobs_sha256=sha256(src / "jobs.json")))


def build_t2(src, out, jobs, role, opened, name):
    """The T2 tag as a long design: one row per box from t = 0 at 500 s, 172,800 calls, the pairs and domain flags."""
    sp53, gcol = None, None
    par, units, rows, saved = [], [], [], []
    src_id = {}
    for i, j in enumerate(jobs):
        z = np.load(src / f"{i}.npz")
        t, x, sp = z["t"], z["x"], [str(v) for v in z["species"]]
        if sp53 is None:
            sp53, gcol = sp, [sp.index(g) for g in GASES]
        assert sp == sp53 and float(t[-1]) >= N_LONG * 500.0
        g = src_id.setdefault((j["src_set"], j["src_job"]), len(src_id))
        par.append(dict(cond=i, start=0, id=i, job=i, T=float(j["T"]), P_bar=float(j["P"]) / 1e6, P=float(j["P"]),
                        kick=j["kick"], src=g))
        units.append(dict(parent=i, age_idx=0, t0=0.0, job=i,
                          x=np.stack([np.interp(np.array([0.0]), t, x[:, k]) for k in range(53)], -1)[0],
                          T=float(j["T"]), P=float(j["P"])))
        r = dict(parent=i, unit=i, job=i, t0=0.0, dt=500.0, N=N_LONG, dt_idx=7, dt_bin=7, age_idx=0, m=M_LONG)
        k = saved_calls(N_LONG, M_LONG)
        tabs = k * 500.0
        saved.append([k, tabs, tabs.copy(), truth_at(t, x, gcol, tabs)])
        rows.append(r)
    ind = np.array([in_domain([p["P_bar"]], [p["T"]]) for p in par])
    reg = np.array([j["region"] for j in jobs])
    assert np.array_equal(ind, reg != "out"), "domain flags differ from the tag's regions"
    part = {(j["src_set"], j["src_job"], j["t_target"]): i for i, j in enumerate(jobs) if j["kick"] == "k0"}
    kk = [i for i, j in enumerate(jobs) if j["kick"] != "k0"]
    partner = np.array([part[(jobs[i]["src_set"], jobs[i]["src_job"], jobs[i]["t_target"])] for i in kk])
    assert np.array_equal(ind[kk], ind[partner]), "a kicked box and its identity partner on two sides of the domain edge"
    cls = np.array([{"k1m": "k1", "k1p": "k1", "k2m": "k2", "k2p": "k2", "k3": "k3"}[jobs[i]["kick"]] for i in kk])
    pairs = dict(kicked=np.array(kk, np.int32), partner=partner.astype(np.int32), cls=cls,
                 kick=np.array([jobs[i]["kick"] for i in kk]), src=np.array([par[i]["src"] for i in kk], np.int32),
                 in_domain=ind[kk])
    C = len(jobs)
    cond = dict(key=np.array([f"{j['src_set']}:{j['src_job']}:{j['t_target']:g}:{j['kick']}" for j in jobs]),
                stratum=reg, group=np.array([p["src"] for p in par], np.int32), in_domain=ind,
                T=np.array([p["T"] for p in par]), P_bar=np.array([p["P_bar"] for p in par]), Z=np.full(C, np.nan),
                C_O=np.full(C, np.nan), He_H=np.full(C, np.nan), variable_dt=np.zeros(C, bool), mixed=np.zeros(C, bool))
    ex = dict(boxes=C, in_domain=int(ind.sum()), out_of_domain=int((~ind).sum()), kicked=len(kk),
              kicked_in_domain=int(ind[kk].sum()), source_jobs=len(src_id),
              source_jobs_in_domain=len({par[i]["src"] for i in range(C) if ind[i]}))
    return finish(out, "long", None, role, opened, name, cond, par, rows, saved, units, sp53, {"box": 1.0}, "src_job",
                  starts=["box"], pairs=pairs, extra_counts=ex,
                  source=dict(dir=str(src), kind="tag (T2)", jobs_sha256=sha256(src / "jobs.json")))


# ------------------------------------------------------------------------------------------------ paths
def build_paths(src, out, panel, role, opened, name):
    files = sorted((f for f in src.glob("[0-9]*.npz") if f.stem.isdigit()), key=lambda f: int(f.stem))
    meta = {int(e["path"]): e for e in json.loads((src / "paths.json").read_text())}
    Ps = [np.load(f) for f in files]
    n = min(len(p["T"]) for p in Ps)
    every = "x" in Ps[0].files
    assert all(("x" in p.files) == every for p in Ps), "mixed path layouts"
    sp53 = [str(s) for s in Ps[0]["species"]]
    gcol = [sp53.index(g) for g in GASES]
    m = M_PATH
    k = saved_calls(n, m)
    par, units, rows, saved, conds = [], [], [], [], []
    F = dict(T=[], Pbar=[], dt=[], hold=[])
    mix = dict(mix_row=[], mix_k=[], mix_f=[], mix_x=[])
    has_mix = []
    slot_of = {}
    for r, (f, p) in enumerate(zip(files, Ps)):
        assert [str(s) for s in p["species"]] == sp53
        e = meta[int(f.stem)]
        T, Pb, dt = (np.asarray(p[a], np.float64)[:n] for a in ("T", "Pbar", "dt"))
        assert np.ndim(p["dt"]) == 1, "per-call dt expected"
        tc = np.cumsum(dt)
        tabs = tc[k - 1]
        if every:
            x0 = np.asarray(p["x"][0], np.float64)
            tr = np.log10(np.maximum(np.asarray(p["x"], np.float64)[k][:, gcol], 1e-30))
            hold = np.full(n, -1, np.int64)
            dt0 = float(np.exp(np.mean(np.log(dt))))
            start, cid = 0, r
            ckey = f"{name}:path{f.stem}"
            mixed = bool("mix_k" in p.files and (np.asarray(p["mix_k"]) < n).any())
            conds.append(dict(key=ckey, stratum="all", group=0, in_domain=in_domain(Pb, T), T=float(T.mean()),
                              P_bar=float(10 ** np.mean(np.log10(Pb))), Z=float(e.get("Z", np.nan)),
                              C_O=float(e.get("C_O", np.nan)), He_H=float(e.get("el", {}).get("He", np.nan)),
                              variable_dt=bool(np.any(dt != dt[0])), mixed=mixed))
        else:
            x0 = np.asarray(p["x0"], np.float64)
            assert np.array_equal(np.asarray(p["saved_k"], np.int64), k), f"{f.name}: saved calls differ from {m} log-spaced"
            tr = np.maximum(np.asarray(p["logx_saved"], np.float64)[:, gcol], -30.0)
            hold = np.asarray(p["hold"], np.int64)[:n]
            assert np.array_equal(hold, holds(dt)), f"{f.name}: holds are not the longest runs with summed dt <= {HOLD:g} s"
            for a in (T, Pb):
                assert all(np.all(a[hold == h] == a[hold == h][0]) for h in np.unique(hold)), f"{f.name}: forcing varies in a hold"
            dt0, start = float(e["dt0"]), STARTS.index(e["start"])
            s = int(e["slot"])
            if s not in slot_of:
                slot_of[s] = (len(conds), r)
                conds.append(dict(key=f"{name}:{s}:i{e['sobol_index']}", stratum=e["stratum"],
                                  group={"G": 0, "C": 1}[e["stratum"]], in_domain=in_domain(Pb, T), T=float(T.mean()),
                                  P_bar=float(10 ** np.mean(np.log10(Pb))), Z=float(e["Z"]), C_O=float(e["C_O"]),
                                  He_H=float(e["He_H"]), variable_dt=bool(e["variable_dt"]), mixed=bool(e["mixed"]),
                                  e=e))
            else:
                c0, r0 = slot_of[s]
                e0 = conds[c0]["e"]
                for kk in ("sobol_index", "Z", "C_O", "He_H", "stratum", "model", "dt0"):
                    assert e[kk] == e0[kk], f"slot {s}: {kk} differs between its rows"
                for a in ("T", "Pbar", "dt", "hold", "mix_k", "mix_f", "mix_x"):
                    assert np.array_equal(np.asarray(p[a]), np.asarray(Ps[r0][a])), f"slot {s}: {a} differs between its rows"
            cid = slot_of[s][0]
            b = int(e["id"]) - BLOCK[panel] - 2 * int(e["sobol_index"]) - FIRST[e["start"]]
            assert b in BASES, f"{f.name}: id {e['id']} is not base + block + 2 i (+ 1)"
        par.append(dict(cond=cid, start=start, id=int(e.get("id", int(f.stem))), job=int(f.stem), T=float(T[0]),
                        P_bar=float(Pb[0]), P=float(Pb[0]) * 1e6))
        units.append(dict(parent=r, age_idx=0, t0=0.0, job=int(f.stem), x=x0, T=float(T[0]), P=float(Pb[0]) * 1e6))
        rows.append(dict(parent=r, unit=r, job=int(f.stem), t0=0.0, dt=dt0, N=n, dt_idx=0, dt_bin=dt_bin(dt0), age_idx=0))
        saved.append([k.copy(), tabs, tabs.copy(), tr])
        for a, v in (("T", T), ("Pbar", Pb), ("dt", dt), ("hold", hold)):
            F[a].append(v)
        hm = "mix_k" in p.files
        has_mix.append(hm)
        if hm:
            mk = np.asarray(p["mix_k"], np.int64)
            mix["mix_row"].append(np.full(mk.size, r, np.int32))
            mix["mix_k"].append(mk)
            mix["mix_f"].append(np.asarray(p["mix_f"], np.float64))
            mix["mix_x"].append(np.asarray(p["mix_x"], np.float64).reshape(-1, len(sp53)))
    for c in conds:
        c.pop("e", None)
    cond = {a: np.array([c[a] for c in conds]) for a in conds[0]}
    cond["group"] = cond["group"].astype(np.int32)
    forcing = {a: np.stack(v) for a, v in F.items()}
    forcing.update({a: (np.concatenate(v) if v else np.zeros((0, 53) if a == "mix_x" else 0)) for a, v in mix.items()})
    forcing["has_mix"] = np.array(has_mix)
    nout = int((~cond["in_domain"]).sum())
    out_calls = int(sum(int(((F["Pbar"][r] < DOMAIN["P_bar"][0]) | (F["Pbar"][r] > DOMAIN["P_bar"][1]) |
                             (F["T"][r] < DOMAIN["T_K"][0]) | (F["T"][r] > DOMAIN["T_K"][1])).sum()) for r in range(len(rows))))
    ex = dict(paths=len(rows), calls_per_path=n, out_of_domain_paths=nout, out_of_domain_calls=out_calls,
              layout="every call stored" if every else "saved calls only")
    if not every:
        assert cond["in_domain"].all(), "a panel path outside the domain"
    return finish(out, "path", panel, role, opened, name, cond, par, rows, saved, units, sp53,
                  {"path": 1.0} if every else W_EQ, "all" if every else "stratum", starts=["path"] if every else STARTS,
                  forcing=forcing, extra_counts=ex,
                  source=dict(dir=str(src), kind="paths", paths_json_sha256=sha256(src / "paths.json"),
                              files=[f.name for f in files]))


# ------------------------------------------------------------------------------------------------ common
def finish(out, mode, panel, role, opened, name, cond, par, rows, saved, units, sp53, weights, boot, starts=STARTS,
           forcing=None, pairs=None, extra_counts=None, source=None):
    sv, truth, nsave, off = csr(saved)
    R = len(rows)
    rws = {a: np.array([r[a] for r in rows]) for a in ("parent", "unit", "job", "t0", "dt", "N", "dt_idx", "dt_bin", "age_idx")}
    for a, ty in (("parent", np.int32), ("unit", np.int32), ("N", np.int32), ("dt_idx", np.int8), ("dt_bin", np.int8),
                  ("age_idx", np.int8)):
        rws[a] = rws[a].astype(ty)
    rws.update(nsave=nsave, off=off)
    P = len(par)
    pars = dict(cond=np.array([p["cond"] for p in par], np.int32), start=np.array([p["start"] for p in par], np.int8),
                id=np.array([p["id"] for p in par], np.int64), job=np.array([p["job"] for p in par], np.int64),
                kick=np.array([p.get("kick", "") for p in par]), src=np.array([p.get("src", -1) for p in par], np.int32),
                T=np.array([p["T"] for p in par]), P_bar=np.array([p["P_bar"] for p in par]))
    x53 = np.stack([u["x"] for u in units])
    with np.errstate(divide="ignore"):
        lx = np.log10(np.maximum(x53, 1e-30))
    st = dict(species53=np.array(sp53), x53=x53, logx53=lx, parent=np.array([u["parent"] for u in units], np.int32),
              age_idx=np.array([u["age_idx"] for u in units], np.int8), t0=np.array([u["t0"] for u in units]),
              job=np.array([u["job"] for u in units], np.int64), T=np.array([u["T"] for u in units]),
              P_barye=np.array([u["P"] for u in units]))
    N = rws["N"].astype(np.int64)
    if mode == "seg":
        chunks = groups_seg(N)
    elif mode == "long":
        chunks = np.array_split(np.arange(R), -(-R // BMAX))
    else:
        assert R <= BMAX, "a path design runs as one batch"
        chunks = [np.arange(R)]
    arrays = {"conditions.npz": cond, "parents.npz": pars, "rows.npz": rws, "saved.npz": sv,
              "truth_saved.npz": dict(gases12=np.array(GASES), logx12=truth), "starts.npz": st,
              "groups.npz": groups_npz(chunks, N)}
    if forcing is not None:
        arrays["forcing.npz"] = forcing
    if pairs is not None:
        arrays["pairs.npz"] = pairs
    nslot = 9 if mode == "seg" else (8 if mode == "long" else 1)
    counts = dict(conditions=int(cond["key"].size), in_domain_conditions=int(cond["in_domain"].sum()), parents=P, rows=R,
                  saved_points=int(sv["k"].size), units=len(units), calls=int(N.sum()), batches=len(chunks),
                  strata={s: int((cond["stratum"] == s).sum()) for s in np.unique(cond["stratum"])},
                  starts={s: int((pars["start"] == i).sum()) for i, s in enumerate(starts)}, **(extra_counts or {}))
    man = dict(name=name, mode=mode, panel=panel, role=role, start_codes=starts, start_weights=weights, slots=nslot,
               slots_per_bin=1, bootstrap=boot, start_map="run_paths" if mode == "path" else "vref_logy",
               counts=counts, source=source)
    write(out, arrays, man, opened)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("seg", "long", "path"))
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--panel", choices=("S", "Sraw", "side", "L", "P"), default=None)
    ap.add_argument("--role", choices=("score", "readout"), default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--t2", action="store_true")
    ap.add_argument("--opened", default=None, help="the author's decision note that opens a sealed set")
    a = ap.parse_args()
    refuse_sealed([a.src, a.out], a.opened)
    for f in OUTS:
        assert not (a.out / f).exists(), f"refusing to overwrite {a.out / f}"
    role = a.role or ("score" if a.panel in ("S", "L", "P") else "readout")
    if a.mode in ("seg", "long") and not a.t2:
        assert a.panel in (("S", "Sraw", "side") if a.mode == "seg" else ("L",)), f"--panel for {a.mode}"
        build_tag(a.src, a.out, a.mode, a.panel, role, a.opened, name=a.name)
    elif a.t2:
        assert a.mode == "long"
        build_tag(a.src, a.out, "long", None, "readout", a.opened, t2=True, name=a.name)
    else:
        assert a.panel in (None, "P")
        build_paths(a.src, a.out, a.panel, role if a.panel else "readout", a.opened, a.name or a.out.name)


if __name__ == "__main__":
    main()
