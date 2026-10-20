"""E13 generalization-gap check (2026-10-07): is allG's late-time error larger on training-split trajectories it never
saw than on the ones it trained on? Two pseudo-tags in lhbench's VULCAN-tag format, built from the 12-species cache
/scratch/imalsky/arch_gcm16_all raw_train.npz: "seen" = rows in allG's train_frac subset (data.RawSplit's seeded
choice, train_frac and train_frac_seed from allG s0's config.json), "unseen" = the rest of the train split. Both are
restricted to eq/mix starts (the start kinds of the G test; kind from the production design via the SEED) and
sampled to N jobs each (seeded). Truth = the production log-log interpolant of the stored grid rows (data.RawSplit.
logy_at; constant after a trajectory's end row), evaluated exactly at the times arm H and analyze_e13.load query
(the starts and t0 + ckpt_calls(n, dt) dt), so lhbench's linear-in-t interpolation never acts between grid rows. This
truth is not a dense VULCAN rerun: absolute numbers are not comparable with G, only seen vs unseen.

    python ctruth.py build      (/scratch/imalsky/e13_ctruth_{seen,unseen}: jobs.json, <i>.npz; refuses to overwrite)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths  # noqa: E402
SW = paths.SWEEP
sys.path.insert(0, str(paths.DATAGEN))
import data as D  # noqa: E402
import torch  # noqa: E402
import gcm  # noqa: E402
import lhbench as LH  # noqa: E402

OUT = paths.SCRATCH
CFG = json.loads((SW / "runs/e13_allG16_s0/config.json").read_text())
N, SEED, STARTS, DT, NCALL = 1024, 20261007, (1e-3, 1e3, 1e6), 500.0, 172800


def build():
    assert str(D.CACHE) == str(paths.SCRATCH / "arch_gcm16_all") and len(D.SHORT) == 12
    tr = D.RawSplit("train", "cpu")
    n_all = tr.n
    seen = np.sort(np.random.default_rng(CFG["train_frac_seed"]).choice(n_all, size=int(round(CFG["train_frac"] * n_all)),
                                                                         replace=False))
    assert seen.size == 36659, seen.size
    kind = np.array([gcm.params(int(s) - gcm.SEED_BASE)["kind"] for s in tr.seed])
    phys = np.isin(kind, ("eq", "mix"))
    unseen = np.setdiff1d(np.arange(n_all), seen)
    rng = np.random.default_rng(SEED)
    calls = LH.ckpt_calls(NCALL, DT)
    times = np.unique(np.concatenate([[s] + list(s + calls * DT) for s in STARTS]))
    for name, pool in (("seen", seen), ("unseen", unseen)):
        out = OUT / f"e13_ctruth_{name}"
        out.mkdir()
        rows = np.sort(rng.choice(pool[phys[pool]], size=N, replace=False))
        ly = tr.logy_at(torch.from_numpy(rows), torch.from_numpy(np.tile(times, (N, 1)))).numpy()  # [N, M, 12]
        jobs = []
        for i, r in enumerate(rows):
            p = gcm.params(int(tr.seed[r]) - gcm.SEED_BASE)
            np.savez(out / f"{i}.npz", t=times, x=10.0 ** ly[i], species=np.array(D.SHORT))
            jobs.append({"T": float(tr.T[r]), "P": float(tr.P[r]), "row": int(r), "seed": int(tr.seed[r]),
                         "name": str(tr.name[r]), "start": p["kind"], "Z": p["Z"], "C_O": p["C_O"], "f_NO": p["f_NO"],
                         "cfg": {"runtime": repr(float(times[-1]))}})
        (out / "jobs.json").write_text(json.dumps(jobs, indent=1))
        print(name, "pool", pool.size, "eq/mix", int(phys[pool].sum()), "jobs", N, "times", times.size, "->", out)


if __name__ == "__main__" and sys.argv[1:] == ["build"]:
    build()
