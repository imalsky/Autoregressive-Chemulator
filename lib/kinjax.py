"""The solver's call (lib/kincuda.py's CUDA kernels) and its derivatives (kincuda_ad's tangent and adjoint kernels) as
JAX primitives, through the XLA FFI handlers in kinjax.cc (built with g++ into paths.KINJAX at first use).

KinJax(net)(x, T, P, dt, h, lm=None, rc=None) -> (x, h, done, steps, flag, status) for x [..., S] mole fractions in the
network's species order, T [...] K, P [...] dyn cm^-2, dt [...] s, starting step sizes h [...] s, optional per-cell log
rate multipliers lm [..., R] (each multiplies a reaction's forward and reverse coefficients together) and the Arrhenius
table rc [R, 9] (default: the network's, as kincuda.tables); float64, any leading shape, flattened into one GPU call.
Without lm the call is the production call (call kernel, guard, device-side recovery: bitwise CudaSolver.launch); with
lm it is the tangent kernel with no directions, the guard and the recovery's path with the multipliers (kincuda_ad's
jvp_rec), the same arithmetic (bitwise CudaSolver.launch at lm = 0). The handlers enqueue on the stream XLA gives them
and never synchronise; every buffer a kernel writes is an output (XLA allocates it per invocation). jax_enable_x64 must
be on; on a shared GPU set XLA_PYTHON_CLIENT_PREALLOCATE=false before JAX starts.

Outputs: x (renormalised), h (the step size to carry to the next call), done (int32), steps (float64, accepted steps),
flag (int32: 0 passed the guard, 1 recovered, 2 still failed after the recovery), status (int32 [8], counts over the
whole primitive call, named by kincuda.COUNTS). The derivative kernels recompute the call bitwise, the recovery
included: a recovered cell has the derivative of the recovered result; a cell with flag 2 has NaN derivatives (and an
Arrhenius cotangent summed over it is NaN).

Derivatives of x_out and h_out with respect to x, T, P, dt, h, lm and rc, contract A (adapt=True, default: the step
control differentiated on the branch taken) or F (adapt=False: accepted step sizes frozen; no dt, h or h_out derivative),
a static argument. The primal's JVP rule binds the primal for the outputs and the linear primitive _lin_p for the
tangents (FFI target kinjax_jvp: the tangent kernel, which recomputes the call; D directions share one launch per 16);
_lin_p's transpose binds _adj_p (kinjax_vjp: the adjoint kernel, one launch per cotangent, recompute and reverse; then
the Arrhenius-table reduction, only when that cotangent is wanted). jax.jvp, linearize, vjp, grad, jacfwd / jacrev and
vmap all go through these rules. vmap: mapped cell operands are merged into the cells; (co)tangents mapped over an
unmapped call become more directions (one primal recomputation per launch for all of them); the Arrhenius cotangent of a
mapped call is reduced per mapped element (groups). The mechanism tables and rc cannot be mapped.
"""
import ctypes
import functools
import hashlib
import math
import os
import subprocess
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax._src import dispatch
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

import kincuda as KC
import kincuda_ad as KA
import paths

_SRC = Path(__file__).with_name("kinjax.cc")
NDIFF = 7    # differentiable operands: x, T, P, dt, h, lm, rc
NP = 19      # all primal operands: those and the tables perm, blob, rf, nasa, sst, jt, lu, optr, atoms, gt, xg, didx

# The Arrhenius-table cotangent per group of consecutive cells: arrh of kincuda_ad over G equal groups (grid y), the
# same summation order within a group, so G = 1 is bitwise arrh and group g is bitwise a call of that group alone.
_ARRHG = r"""
extern "C" __global__ void arrhg(const double* __restrict__ glm, const double* __restrict__ Ta,
    const double* __restrict__ Pa, const double* __restrict__ rc, const int* __restrict__ rf, double* __restrict__ grc,
    int B, int G)
{
  __shared__ double red[9][256];
  const int r = blockIdx.x, tid = threadIdx.x, fl = rf[r], Bg = B / G;
  for (int g = blockIdx.y; g < G; g += gridDim.y) {
    const double *gl = glm + (size_t)g * Bg * NR, *Tg = Ta + (size_t)g * Bg, *Pg = Pa + (size_t)g * Bg;
    double acc[9];
    #pragma unroll
    for (int j = 0; j < 9; j++) acc[j] = 0.0;
    for (int cell = tid; cell < Bg; cell += 256) {
      const double gv = gl[(size_t)cell * NR + r];
      if (gv == 0.0) continue;
      const double Tc = Tg[cell], nd = Pg[cell] / (KBOLTZ * Tc);
      double dT, dn, D[9];
      kpart(rc + 9 * r, fl, log(Tc), 1.0 / Tc, nd, dT, dn, D);
      #pragma unroll
      for (int j = 0; j < 9; j++) acc[j] += gv * D[j];
    }
    #pragma unroll
    for (int j = 0; j < 9; j++) red[j][tid] = acc[j];
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
      if (tid < s) {
        #pragma unroll
        for (int j = 0; j < 9; j++) red[j][tid] += red[j][tid + s];
      }
      __syncthreads();
    }
    if (tid < 9) grc[((size_t)g * NR + r) * 9 + tid] = red[tid][0];
    __syncthreads();
  }
}
"""


def build():
    """paths.KINJAX / libkinjax_<hash>.so (g++; the hash covers kinjax.cc and the JAX version), built if missing."""
    import nvidia.cuda_runtime
    cuda_inc = os.path.join(list(nvidia.cuda_runtime.__path__)[0], "include")
    tag = hashlib.sha256(_SRC.read_bytes() + jax.__version__.encode()).hexdigest()[:12]
    lib = paths.KINJAX / f"libkinjax_{tag}.so"
    if lib.exists():
        return lib
    lib.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=lib.parent) as tmp:   # rename into place: no half-written library is loaded
        out = os.path.join(tmp, lib.name)
        subprocess.run([os.environ.get("CXX", "g++"), "-O2", "-std=c++17", "-shared", "-fPIC", "-Wall", "-Wextra",
                        "-isystem", jax.ffi.include_dir(), "-isystem", cuda_inc, str(_SRC), "-o", out, "-ldl"],
                       check=True)
        os.replace(out, lib)
    return lib


_LIB = []


def _lib():
    """The shim (ctypes), built if needed and its three targets registered with XLA once."""
    if not _LIB:
        lib = ctypes.CDLL(str(build()))
        lib.KinjaxDevice.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int]
        lib.KinjaxLoad.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                   ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_char_p,
                                   ctypes.c_int]
        lib.KinjaxLoad.restype = ctypes.c_int64
        for name, sym in (("kinjax_call", "KinjaxCall"), ("kinjax_jvp", "KinjaxJvp"), ("kinjax_vjp", "KinjaxVjp")):
            jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(getattr(lib, sym)), platform="CUDA")
        _LIB.append(lib)
    return _LIB[0]


_MODULES = {}   # handle -> dict(ns, nr, rec (recovery scratch doubles), call / jvp / vjp: (cells per block, resident
                #                                                                                      blocks))


# --- primitives -------------------------------------------------------------------------------------------------------

def _nw(B, module, kind):
    """Cell slots of a launch of kernel kind for B cells (each owns a slice of the scratch)."""
    cpb, max_blocks = _MODULES[module][kind]
    return max(1, min(-(-B // cpb), max_blocks)) * cpb


def _has(a):
    """An optional operand or output is present: absent ones are empty vectors, present ones have a cell or direction
    axis besides their own."""
    return a.ndim > 1


def _sa(shape, dtype=jnp.float64):
    return jax.core.ShapedArray(tuple(shape), dtype)


def _call_abstract(x, T, P, dt, h, lm, rc, *tabs, module, rtol, atol, max_steps, adapt, rec_steps):
    B, ns = x.shape
    nr = _MODULES[module]["nr"]
    scr = _nw(B, module, "jvp") * (3 * nr + 2) if _has(lm) else _nw(B, module, "call") * (2 * nr + 2)
    rscr = _nw(B, module, "jvp") * ns * ns if _has(lm) else _MODULES[module]["rec"]
    i32 = jnp.int32
    return (_sa((B, ns)), _sa((B,)), _sa((B,), i32), _sa((B,)), _sa((B,), i32), _sa((10,), i32), _sa((scr,)),
            _sa((B,), i32), _sa((rscr,)))


def _lin_abstract(*args, module, rtol, atol, max_steps, adapt, rec_steps):
    x, tx = args[0], args[NP]
    D, (B, ns) = tx.shape[0], x.shape
    nr = _MODULES[module]["nr"]
    nw = _nw(B, module, "jvp")
    scr = nw * (3 * nr + 2 + min(D, KA.DMAX) * (2 * nr + ns))
    i32 = jnp.int32
    return (_sa((D, B, ns)), _sa((D, B)), _sa((B, ns)), _sa((B,)), _sa((B,), i32), _sa((B,)), _sa((B,), i32),
            _sa((10,), i32), _sa((scr,)), _sa((B,), i32), _sa((nw * ns * ns,)))


def _adj_abstract(*args, module, rtol, atol, max_steps, adapt, rec_steps, want_lm, want_rc, groups, cap):
    if max_steps > cap ** 2 or rec_steps > (2 * cap) ** 2:
        raise ValueError(f"kinjax: reverse mode needs max_steps <= {cap ** 2} (and the recovery's tries <= "
                         f"{4 * cap ** 2})")
    x, cx = args[0], args[NP]
    D, (B, ns) = cx.shape[0], x.shape
    nr = _MODULES[module]["nr"]
    nw = _nw(B, module, "vjp")
    i32 = jnp.int32
    return (_sa((D, B, ns)),) + tuple(_sa((D, B)) for _ in range(4)) + (
        _sa((D, B, nr) if want_lm else (0,)), _sa((D, groups, nr, 9) if want_rc else (0,)),
        _sa((B, ns)), _sa((B,)), _sa((B,), i32), _sa((B,)), _sa((B,), i32), _sa((10,), i32),
        _sa((nw * (12 * nr + 2),)), _sa((nw * 2 * cap * (ns + 3),)), _sa((0,) if want_lm else (B * nr,)),
        _sa((B,), i32), _sa((nw * ns * ns,)), _sa((nw * 4 * cap * (ns + 3),)))


def _lower(target, keep):
    """FFI lowering of target with the params named in keep as attributes (integers and bools as int64)."""
    rule = jax.ffi.ffi_lowering(target)

    def lower(ctx, *args, **params):
        return rule(ctx, *args, **{k: (np.int64(v) if isinstance(v, (bool, int, np.bool_, np.integer)) else v)
                                   for k, v in params.items() if k in keep})
    return lower


def _prim(name, abstract, target, keep):
    p = Primitive(name)
    p.multiple_results = True
    p.def_impl(functools.partial(dispatch.apply_primitive, p))
    p.def_abstract_eval(abstract)
    mlir.register_lowering(p, _lower(target, keep), platform="cuda")
    return p


_BASE = ("module", "rtol", "atol", "max_steps")
_call_p = _prim("kinjax_call", _call_abstract, "kinjax_call", _BASE + ("rec_steps",))
_lin_p = _prim("kinjax_lin", _lin_abstract, "kinjax_jvp", _BASE + ("adapt", "rec_steps"))
_adj_p = _prim("kinjax_adj", _adj_abstract, "kinjax_vjp", _BASE + ("adapt", "groups", "cap", "rec_steps"))


def _zero(o):
    return ad.Zero(jax.typeof(o).to_tangent_aval())


def _call_jvp(primals, tangents, **params):
    outs = _call_p.bind(*primals, **params)
    if any(type(t) is not ad.Zero for t in tangents[NDIFF:]):
        raise NotImplementedError("kinjax: no derivative with respect to the mechanism tables")
    tan = list(tangents[:NDIFF])
    if not _has(primals[5]):
        tan[5] = _zero(primals[5])                                   # lm absent
    if all(type(t) is ad.Zero for t in tan):
        return outs, [_zero(o) for o in outs]
    empty = jnp.zeros((0,), jnp.float64)
    lin = [ad.instantiate_zeros(t)[None] for t in tan[:5]] + [
        empty if type(t) is ad.Zero else t[None] for t in tan[5:]]
    dx, dh = _lin_p.bind(*primals, *lin, **params)[:2]
    return outs, [dx[0], dh[0]] + [_zero(o) for o in outs[2:]]


def _lin_transpose(cts, *args, **params):
    if any(ad.is_undefined_primal(a) for a in args[:NP]):
        raise ValueError("kinjax: _lin_p is linear only in its tangent operands")
    tans = args[NP:]
    cx = ad.instantiate_zeros(cts[0])
    ch = jnp.zeros((0,), jnp.float64) if type(cts[1]) is ad.Zero else cts[1]   # absent: h_out's cotangent is 0
    want = [ad.is_undefined_primal(t) for t in tans]
    outs = _adj_p.bind(*args[:NP], cx, ch, want_lm=want[5], want_rc=want[6], groups=1, cap=KA.CAP, **params)
    D, nr = cx.shape[0], _MODULES[params["module"]]["nr"]
    bars = list(outs[:6]) + [outs[6].reshape(D, nr, 9) if want[6] else None]
    return [None] * NP + [b if w else None for w, b in zip(want, bars)]


def _front(a, d, V):
    """vmap: a's mapped axis d moved to the front, or a broadcast along a new front axis of size V if d is None."""
    return jnp.broadcast_to(a, (V,) + a.shape) if d is None else jnp.moveaxis(a, d, 0)


def _size(args, dims):
    return next(a.shape[d] for a, d in zip(args, dims) if d is not None)


def _cells(args, dims, V):
    """vmap: the cell operands (x, T, P, dt, h, lm) with the mapped axis merged into the cell axis (an absent lm stays
    empty)."""
    out = []
    for a, d in zip(args[:6], dims[:6]):
        if len(out) == 5 and not _has(a):
            out.append(a)
            continue
        f = _front(a, d, V)
        out.append(f.reshape((-1,) + f.shape[2:]))
    return out


def _no_map(dims, what):
    if any(d is not None for d in dims):
        raise NotImplementedError(f"kinjax: {what} cannot be mapped")


def _call_batch(args, dims, **params):
    _no_map(dims[6:NP], "the Arrhenius table and the mechanism tables")
    V = _size(args, dims)
    cells = _cells(args, dims, V)
    B = cells[1].shape[0] // V
    xo, ho, done, nok, redo, count, scr, lst, rscr = _call_p.bind(*cells, *args[6:NP], **params)
    ns = xo.shape[-1]
    return ((xo.reshape(V, B, ns), ho.reshape(V, B), done.reshape(V, B), nok.reshape(V, B), redo.reshape(V, B), count,
             scr, lst, rscr), (0, 0, 0, 0, 0, None, None, None, None))


def _tan_batch(prim, args, dims, **params):
    """vmap of _lin_p / _adj_p. (Co)tangents mapped over an unmapped call become more directions [V D, ...]; a mapped
    call is merged into the cells [V B] and the (co)tangents follow it ([D, V B, ...])."""
    _no_map(dims[6:NP], "the Arrhenius table and the mechanism tables")
    V = _size(args, dims)
    cell_dims = dims[:5] + (dims[5] if _has(args[5]) else None,)
    tans, tdims = args[NP:], dims[NP:]
    adj = prim is _adj_p
    if all(d is None for d in cell_dims):
        new = [t if not _has(t) else (lambda f: f.reshape((V * f.shape[1],) + f.shape[2:]))(_front(t, d, V))
               for t, d in zip(tans, tdims)]
        outs = prim.bind(*args[:NP], *new, **params)
        nout = 7 if adj else 2
        res = [o.reshape((V, o.shape[0] // V) + o.shape[1:]) if _has(o) else o for o in outs[:nout]]
        return res + list(outs[nout:]), [0 if _has(o) else None for o in outs[:nout]] + [None] * (len(outs) - nout)
    if not adj and _has(tans[6]) and tdims[6] is not None:
        raise NotImplementedError("kinjax: a tangent of the Arrhenius table that differs between mapped calls")
    cells = _cells(args, dims, V)
    B = cells[1].shape[0] // V
    new = []
    for i, (t, d) in enumerate(zip(tans, tdims)):
        if not _has(t) or (not adj and i == 6):
            new.append(t)                                          # absent, or the rc tangent [D, R, 9] (shared)
            continue
        f = jnp.swapaxes(_front(t, d, V), 0, 1)                    # [D, V, B, ...]
        new.append(f.reshape((f.shape[0], V * B) + f.shape[3:]))
    if adj:
        params = dict(params, groups=params["groups"] * V)
    outs = prim.bind(*cells, *args[6:NP], *new, **params)
    nout = 7 if adj else 2
    res = []
    for k, o in enumerate(outs[:nout]):
        if not _has(o):
            res.append(o)
        elif k == 6:                                               # grc [D, V G, R, 9] -> [D, V, G, R, 9]
            res.append(o.reshape((o.shape[0], V, o.shape[1] // V) + o.shape[2:]))
        else:
            res.append(o.reshape((o.shape[0], V, B) + o.shape[2:]))
    return res + list(outs[nout:]), [1 if _has(o) else None for o in outs[:nout]] + [None] * (len(outs) - nout)


ad.primitive_jvps[_call_p] = _call_jvp
ad.primitive_transposes[_lin_p] = _lin_transpose
batching.primitive_batchers[_call_p] = _call_batch
batching.primitive_batchers[_lin_p] = functools.partial(_tan_batch, _lin_p)
batching.primitive_batchers[_adj_p] = functools.partial(_tan_batch, _adj_p)


# --- the solver -------------------------------------------------------------------------------------------------------

class KinJax:
    """KinJax(net)(x, T, P, dt, h, lm=None, rc=None, rtol=1e-3, atol=1e-18, max_steps=4000, adapt=True) -> (x, h, done,
    steps, flag, status): one solver call per cell on the local CUDA devices, differentiable (see the module
    docstring). The call kernel's layout is CudaSolver's (built once here through torch to choose it); the tables are
    device constants of the traced call; rtol, atol, max_steps and adapt are static."""

    def __init__(self, net):
        if not jax.config.jax_enable_x64:
            raise RuntimeError("kinjax needs jax_enable_x64 (float64 state)")
        lib = _lib()
        cs = KC.CudaSolver(net, torch.device("cuda"))         # the call kernel's warps, cells per block, source
        tab = KC.tables(net, cs.w)
        KA.supported(tab)
        ext = KA.ad_tables(net, tab)
        self.ns, self.nr = tab["ns"], tab["nr"]
        blob, _ = KC._blob(tab)
        devs, err = jax.local_devices(backend="cuda"), ctypes.create_string_buffer(512)
        props = set()
        for d in devs:
            p = (ctypes.c_int * 4)()
            if lib.KinjaxDevice(d.local_hardware_id, p, err, 512):
                raise RuntimeError(err.value.decode())
            props.add(tuple(p))
        if len(props) != 1:
            raise RuntimeError(f"kinjax: the local CUDA devices differ ({props})")
        major, minor, _, optin = props.pop()
        nl = 32 * cs.w
        rounds = [tab["sst"].shape[0], tab["jround"], ext["gt"].shape[0], nl]
        geo = {"call": [self.ns, self.nr, cs.cpb, nl * cs.cpb, cs.smem, nl, cs.smem_rec] + rounds}
        self.src = {"call": cs.src}
        for kind in ("jvp", "vjp"):
            perwa = KA.layout(tab, ext, kind)[1]
            wpb = min(32, (optin - len(blob)) // (8 * perwa))
            if wpb < 1:
                raise RuntimeError(f"the {kind} kernel's work arrays do not fit in shared memory")
            sm = 8 * wpb * perwa + len(blob)                    # the recovery's kernel: the same grid and layout
            geo[kind] = [self.ns, self.nr, wpb, 32 * wpb, sm, 32 * wpb, sm] + rounds
            self.src[kind] = KA.source_ad(tab, ext, kind, wpb) + (_ARRHG if kind == "vjp" else "")
        rec = cs.rec_blocks * (2 * self.nr + 2 + (0 if cs.info["rec_w_shared"] else self.ns ** 2))
        del cs
        handle, cells = -1, {}
        for k, kind in enumerate(("call", "jvp", "vjp")):
            cubin = KC.compile_cubin(self.src[kind], f"{major}{minor}")
            g = (ctypes.c_int * 11)(*geo[kind])
            mb = []
            for d in devs:
                b = (ctypes.c_int * 2)()
                h = lib.KinjaxLoad(handle, d.local_hardware_id, k, cubin, g, b, err, 512)
                if h < 0:
                    raise RuntimeError(err.value.decode())
                handle = h
                mb.append(tuple(b))
            cells[kind] = (geo[kind][2], min(q[0] for q in mb))
            if kind == "call" and min(q[1] for q in mb) * (2 * self.nr + 2) > rec:
                raise RuntimeError("kinjax: the devices hold more recover blocks than the recovery scratch")
        self.module = int(handle)
        _MODULES[self.module] = dict(ns=self.ns, nr=self.nr, rec=rec, **cells)
        u = lambda a, t: jnp.asarray(np.ascontiguousarray(a).view(t))  # noqa: E731
        self.rc = jnp.asarray(tab["rc"])
        self.tabs = (jnp.asarray(tab["perm"]), u(np.frombuffer(blob, np.uint32), np.uint32), jnp.asarray(tab["rf"]),
                     jnp.asarray(tab["nasa"]), u(tab["sst"], np.uint32), u(tab["jt"], np.uint64),
                     u(tab["lu"], np.uint64), jnp.asarray(tab["optr"]), jnp.asarray(tab["atoms"]),
                     u(ext["gt"], np.uint64), jnp.asarray(ext["xg"]), u(tab["didx"], np.uint16))
        self._jit = jax.jit(self.call, static_argnames=("rtol", "atol", "max_steps", "adapt"))

    def call(self, x, T, P, dt, h, lm=None, rc=None, rtol=1e-3, atol=1e-18, max_steps=4000, adapt=True):
        """The traceable call (use inside jit / vmap / grad); __call__ is its jitted form."""
        T = jnp.asarray(T, jnp.float64)
        lead = T.shape
        B = math.prod(lead)
        flat = [jnp.asarray(a, jnp.float64).reshape(s) for a, s in ((x, (B, self.ns)), (T, (B,)), (P, (B,)),
                                                                      (dt, (B,)), (h, (B,)))]
        lm = jnp.zeros((0,), jnp.float64) if lm is None else jnp.asarray(lm, jnp.float64).reshape(B, self.nr)
        rc = self.rc if rc is None else jnp.asarray(rc, jnp.float64).reshape(self.nr, 9)
        xo, ho, done, nok, redo, count = _call_p.bind(
            *flat, lm, rc, *self.tabs, module=np.int64(self.module), rtol=np.float64(rtol), atol=np.float64(atol),
            max_steps=np.int64(max_steps), adapt=bool(adapt), rec_steps=np.int64(KC.REC_STEPS * max_steps))[:6]
        return (xo.reshape(lead + (self.ns,)), ho.reshape(lead), done.reshape(lead),
                nok.astype(jnp.int32).reshape(lead), redo.reshape(lead), count[1:9])

    def __call__(self, x, T, P, dt, h, lm=None, rc=None, rtol=1e-3, atol=1e-18, max_steps=4000, adapt=True):
        return self._jit(x, T, P, dt, h, lm, rc, rtol=rtol, atol=atol, max_steps=max_steps, adapt=adapt)


def check(status):
    """Host check of a call's status (one device-to-host copy): the counts as a dict (kincuda.COUNTS); raises if any
    cell was below T_MIN."""
    n = dict(zip(KC.COUNTS, np.asarray(status).tolist()))
    if n["cold"]:
        raise ValueError(f"{n['cold']} cells below {KC.T_MIN:g} K: outside the chemistry solver's domain")
    return n


if __name__ == "__main__":   # build the shim: python kinjax.py
    print(build())
