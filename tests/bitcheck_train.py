"""Bitwise training check: the current lib/ against a reference lib folder, on the same data and the same draws.

Each side runs train.main(<run dir>) in its own process, exactly as `python train.py <run dir>` does, on a short copy of
a real run's config. Recorded per side: every optimizer step's loss (the scalar passed to Tensor.backward), the gradients
the optimizer receives (after clipping), the pre-clip gradients and norm wherever clip_grad_norm_ is used, every log row
(timing and memory fields dropped) and the final stage file's weights. Tensors are compared by the sha256 of their bytes,
so PASS means bit for bit.

Modes (environment as CBg trained: ARCH_CACHE=/scratch/imalsky/arch_gcmU4_res,
HARNESS_NORM=/scratch/imalsky/arch_gcm17m_res/norm.json; TF32 on, as train.main sets it):
  s2  CBg's config: stage 2 from CBS1's stage1.pt (lockstep, fused projection, row clips, fp64 clip norm, non-finite
      guard). Reference: lib/ of git tag e13s_CBg_s2, extracted with git archive into the work folder.
  s1  CBS1's config: stage 1 from scratch. Reference: CBS1's frozen copy, runs/e13s_CBS1_s2/harness/lib.
Short-run overrides, the same on both sides: N steps (default 4); log_every 1; validation (2048 windows) at the last step
only; no checkpoints, resume or path readout; traj_weights_file None, because CUDA multinomial draws do not repeat from one
generator state (uniform randint draws do), so the weighted trajectory draw is the one path not covered. s2 also sets
chain_K_ramp = chain_pre_ramp = 1: step 0 runs the start of the curriculum (8 chain calls, no pre-roll) and later steps
its end (16 calls, 0-80 pre-roll calls); grid sub-steps cycle 1, 2, 3, 4 over steps 0-3.
Both sides compile with inductor's run-time kernel choices off (torch._inductor.config.triton.autotune_pointwise and
dynamic_scale_rblock False). With them on, each process times several block sizes for some reduction kernels at their
first launch and keeps the fastest, so two processes can sum in different orders: seen 2026-10-19 in stage 1, where the
layer-norm backward of the first block (a register-heavy reduction that dynamic_scale_rblock gives a second, smaller
block size) gave different input-layer gradients in about 1 process in 10, the same code and data each time. With one
kernel config each, the two sides run the same kernels whenever they trace the same graph. Training keeps both on, so
these runs are not bitwise those of training (nor are two training runs of one config); they test that two code trees
compute the same thing.

    source run/env.sh
    taskset -c <cpus> $PY tests/bitcheck_train.py s2|s1 [--steps N] [--ref DIR] [--repeat] [--negative] [--work DIR]

--repeat also runs the reference a second time (cross-process repeatability control). --negative also runs a copy of the
current lib whose loss weight lambda_log10_mae is scaled by (1 + 2^-20); that comparison must differ. Exit 0 = PASS.
The work folder (default under TMPDIR) keeps each side's config, log and record; stage files are deleted once hashed.
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = Path("/home/imalsky/Chemulator_Project/experiments/2026-10-13_e13-scale/_build/runs")
ENV = {"ARCH_CACHE": "/scratch/imalsky/arch_gcmU4_res", "HARNESS_NORM": "/scratch/imalsky/arch_gcm17m_res/norm.json",
       "PYTHONDONTWRITEBYTECODE": "1"}
MODES = {"s2": {"config": RUNS / "e13s_CBg_s2" / "config.json", "ref": "tag:e13s_CBg_s2",
                "over": {"s2_steps": None, "chain_K_ramp": 1, "chain_pre_ramp": 1}},
         "s1": {"config": RUNS / "e13s_CBS1_s2" / "config.json", "ref": str(RUNS / "e13s_CBS1_s2" / "harness" / "lib"),
                "over": {"s1_steps": None}}}
COMMON = {"log_every": 1, "val_every": 10 ** 9, "val_windows": 2048, "ckpt_every": 0, "resume": False,
          "path_readout": None, "traj_weights_file": None}
DROP = {"time", "s_per_step", "peak_mem_gb"}          # log fields that are not numerics
NEG = ("lambda_log10_mae=1.0,", "lambda_log10_mae=1.0 + 2.0 ** -20,")   # negative control: (old, new) in train.py


# ----------------------------------------------------------------------------------------------------------- worker
def hashes(ts):
    return [[str(t.dtype), list(t.shape), hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
             float(t.detach().double().sum())] for t in ts]


def worker(lib, run, out):
    """Run lib's train.main(run) with recorders installed; write the record to out (json)."""
    sys.path.insert(0, lib)
    import torch
    import torch._inductor.config
    from torch.optim.optimizer import register_optimizer_step_pre_hook
    torch._inductor.config.triton.autotune_pointwise = False   # one kernel config per reduction, no run-time timing
    torch._inductor.config.dynamic_scale_rblock = False         # choices (see the docstring)
    rec = {"loss": [], "clip": [], "grad": []}
    backward0, clip0 = torch.Tensor.backward, torch.nn.utils.clip_grad_norm_

    def backward(self, *a, **k):
        if self.dim() == 0:
            rec["loss"].append(float(self.detach().double()).hex())
        return backward0(self, *a, **k)

    def clip(params, max_norm, *a, **k):
        params = list(params)
        pre = hashes([p.grad for p in params if p.grad is not None])
        tn = clip0(params, max_norm, *a, **k)
        rec["clip"].append({"norm": float(tn).hex(), "pre": pre})
        return tn

    def step_hook(opt, args, kwargs):
        rec["grad"].append(hashes([p.grad for g in opt.param_groups for p in g["params"] if p.grad is not None]))

    torch.Tensor.backward = backward
    torch.nn.utils.clip_grad_norm_ = clip
    register_optimizer_step_pre_hook(step_hook)
    import train
    assert Path(train.__file__).resolve().parent == Path(lib).resolve(), train.__file__
    t0 = time.time()
    train.main(run)
    run = Path(run)
    rec["seconds"] = time.time() - t0
    rec["log"] = [{k: v for k, v in json.loads(s).items() if k not in DROP} for s in (run / "log.csv").read_text().splitlines()]
    rec["model_info"] = json.loads((run / "model_info.json").read_text())
    rec["lib_md5"] = json.loads((run / "done.json").read_text())["lib_md5"]
    rec["stage"] = {}
    for f in sorted(run.glob("stage*.pt")):
        blob = torch.load(f, map_location="cpu", weights_only=False)
        sd = blob["state_dict"]
        rec["stage"][f.name] = {"weights": dict(zip(sorted(sd), hashes([sd[k] for k in sorted(sd)]))),
                                "cfg": json.loads(json.dumps(blob["cfg"]))}
        f.unlink()                                       # hashed; 85 MB each
    Path(out).write_text(json.dumps(rec))


# ----------------------------------------------------------------------------------------------------- orchestrator
def gpu_idle():
    idx = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if idx != "1":
        sys.exit(f"CUDA_VISIBLE_DEVICES={idx!r}: source run/env.sh first (GPU1 only)")
    q = subprocess.run(["nvidia-smi", "-i", idx, "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True, check=True).stdout.split()
    if q:
        sys.exit(f"GPU{idx} holds other processes {q}: not starting")


def ref_lib(spec, work):
    if not spec.startswith("tag:"):
        return Path(spec)
    blob = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", spec[4:], "lib"], capture_output=True,
                          check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as t:
        t.extractall(work / "ref_tag", filter="data")
    return work / "ref_tag" / "lib"


def neg_lib(work):
    dst = work / "neg_lib"
    shutil.copytree(REPO / "lib", dst, ignore=shutil.ignore_patterns("__pycache__", "_build"))
    f = dst / "train.py"
    s = f.read_text()
    assert s.count(NEG[0]) == 1, f"negative control: {NEG[0]!r} found {s.count(NEG[0])} times in train.py"
    f.write_text(s.replace(NEG[0], NEG[1]))
    return dst


def run_side(label, lib, cfg, work, env):
    d = work / label
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    t = time.time()
    with open(d / "worker.log", "w") as log:
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(lib), str(d), str(d / "record.json")],
                           cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    secs = time.time() - t
    print(f"  {label}: lib {lib}, exit {r.returncode}, {secs:.0f} s (log {d / 'worker.log'})", flush=True)
    if r.returncode != 0:
        print((d / "worker.log").read_text()[-3000:])
        sys.exit(f"{label}: worker failed")
    return json.loads((d / "record.json").read_text())


def compare(name, a, b, verbose=True):
    """Number of differences between records a and b (0 = bitwise equal); prints one line per step."""
    bad = 0
    n = max(len(a["grad"]), len(b["grad"]))
    if len(a["loss"]) != len(b["loss"]) or len(a["grad"]) != len(b["grad"]) or len(a["clip"]) != len(b["clip"]):
        print(f"  {name}: different step counts: loss {len(a['loss'])}/{len(b['loss'])}, optimizer steps "
              f"{len(a['grad'])}/{len(b['grad'])}, clip calls {len(a['clip'])}/{len(b['clip'])}")
        bad += 1
    for k in range(n):
        la, lb = (x["loss"][k] if k < len(x["loss"]) else None for x in (a, b))
        ga, gb = (x["grad"][k] if k < len(x["grad"]) else [] for x in (a, b))
        ng = sum(1 for p, q in zip(ga, gb) if p[:3] != q[:3]) + abs(len(ga) - len(gb))
        ca, cb = (x["clip"][k] if k < len(x["clip"]) else None for x in (a, b))
        nc = 0 if ca == cb else 1
        bad += int(la != lb) + ng + nc
        if verbose:
            lv = float.fromhex(la) if la else float("nan")
            print(f"  {name} step {k}: loss {'EQUAL' if la == lb else 'DIFF'} ({lv!r}"
                  f"{'' if la == lb else ' vs ' + repr(float.fromhex(lb) if lb else None)}), gradient tensors {len(ga)}, "
                  f"differing {ng}" + ("" if ca is None and cb is None else f", pre-clip norm/gradients {'EQUAL' if nc == 0 else 'DIFF'}"),
                  flush=True)
            for p, q in list(zip(ga, gb))[:0 if ng == 0 else 3]:
                if p[:3] != q[:3]:
                    print(f"      shape {p[1]}: sum {p[3]!r} vs {q[3]!r}")
    la_, lb_ = a["log"], b["log"]
    nlog, only = 0, set()
    if len(la_) != len(lb_):
        nlog += 1
    for ra, rb in zip(la_, lb_):
        only |= set(ra) ^ set(rb)
        nlog += sum(ra[k] != rb[k] for k in set(ra) & set(rb))
    st_a, st_b = a["stage"], b["stage"]
    nst = int(sorted(st_a) != sorted(st_b))
    for f in set(st_a) & set(st_b):
        wa, wb = st_a[f]["weights"], st_b[f]["weights"]
        nst += int(sorted(wa) != sorted(wb)) + sum(wa[k][:3] != wb[k][:3] for k in set(wa) & set(wb))
        ca, cb = st_a[f]["cfg"], st_b[f]["cfg"]
        nst += sum(ca[k] != cb[k] for k in set(ca) & set(cb))
        if set(ca) ^ set(cb):
            print(f"  {name}: {f} cfg keys on one side only (information): {sorted(set(ca) ^ set(cb))}")
    nmi = int(a["model_info"] != b["model_info"])
    print(f"  {name}: log rows {len(la_)}/{len(lb_)}, differing values {nlog}"
          + (f" (keys on one side only, information: {sorted(only)})" if only else "")
          + f"; stage files {sorted(st_a)}: weight / cfg differences {nst}; model_info {'EQUAL' if nmi == 0 else 'DIFF'}")
    return bad + nlog + nst + nmi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=sorted(MODES))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--ref", default=None, help="reference lib folder (default: the mode's reference)")
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument("--negative", action="store_true")
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    gpu_idle()
    m = MODES[a.mode]
    tmp = os.environ.get("TMPDIR")
    assert tmp and not tmp.startswith("/tmp"), "TMPDIR must be set to your own scratch (run/env.sh)"
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix=f"bitcheck_{a.mode}_", dir=tmp))
    work.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(m["config"].read_text())
    cfg.update(COMMON)
    cfg.update({k: (a.steps if v is None else v) for k, v in m["over"].items()})
    env = {**os.environ, **ENV}
    sides = [("ref", ref_lib(a.ref or m["ref"], work)), ("cur", REPO / "lib")]
    if a.repeat:
        sides.append(("ref2", sides[0][1]))
    if a.negative:
        sides.append(("neg", neg_lib(work)))
    t0 = time.time()
    print(f"bitcheck {a.mode}: {a.steps} steps, config {m['config']} + {json.dumps({**COMMON, **m['over']})}; work {work}",
          flush=True)
    R = {lab: run_side(lab, lib, cfg, work, env) for lab, lib in sides}
    diff_md5 = sorted(k for k in set(R["ref"]["lib_md5"]) | set(R["cur"]["lib_md5"])
                      if R["ref"]["lib_md5"].get(k) != R["cur"]["lib_md5"].get(k))
    print(f"lib files differing (md5) between ref and cur (information): {diff_md5 or 'none'}")
    ok = True
    nd = compare("cur vs ref", R["ref"], R["cur"])
    print(f"cur vs ref: {'bitwise EQUAL' if nd == 0 else f'{nd} differences'}")
    ok &= nd == 0
    if a.repeat:
        nr = compare("ref2 vs ref", R["ref"], R["ref2"], verbose=False)
        print(f"ref2 vs ref (repeatability control): {'bitwise EQUAL' if nr == 0 else f'{nr} differences'}")
        ok &= nr == 0
    if a.negative:
        nn = compare("neg vs cur", R["cur"], R["neg"], verbose=False)
        print(f"neg vs cur (negative control, must differ): {'DIFFERS as expected' if nn > 0 else 'EQUAL: the check is blind'}"
              f" ({nn} differences)")
        ok &= nn > 0
    print(f"total {time.time() - t0:.0f} s")
    print(f"RESULT {a.mode}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:5])
    else:
        main()
