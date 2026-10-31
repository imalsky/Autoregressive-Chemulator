"""The solver's whole call as one CUDA kernel: what kinvulcan.TorchRates + kinbatch.rodas3_step(pivot=False) +
kinnet.advance_compact(order=3) compute for every cell, in one launch.

One warp per cell. The warp computes the cell's rate coefficients from T and P (NASA-9 Gibbs energies; Arrhenius,
falloff and the special law; reverse rates from the equilibrium constants; n_tot folded into the three-body
coefficients), then takes every Rodas3 step of the call with advance_compact's step control: a step is accepted when its
RMS error is within tolerance and no species is below -atol, accepted states are clamped at 0, the next step is
0.9 err^(-1/3) times the last (within [0.2, 5], at most 0.5 after a rejection), at most max_steps tries. The exact
Jacobian J = Jn + w 1^T (Jn = d(S v)/dn, w the constant-P term) enters through an LU factor without row exchanges of
M0 = I / (gamma h) - Jn on its sparse pattern, in a minimum-degree order, and the Sherman-Morrison formula for the
rank-one term. Lane l owns row l of the sparse leading part and row K0 + l of the trailing block that the fill makes
dense; that block is factored in registers. The S v sums and the Jacobian terms run as balanced per-lane streams.
Persistent warps take cells from an atomic queue, so no cell waits for the stiffest one.

The CUDA source is generated from the network at construction and compiled once with NVRTC through ctypes (the library
of torch's CUDA wheels); the kernel runs on torch's current stream.

Guard: after the call a second small kernel checks every cell (new state finite, call finished, each element's share of
the atoms unchanged up to rounding and what the clamping of accepted small negatives can add) and counts the failures;
the host reads the counts (one small copy per call) and recomputes the flagged cells from their inputs by the library
path (TorchRates + rodas3_step + advance_compact, same arguments). The element test checks the linear solves:
E^T W = E^T / (gamma h) for the element matrix E (E^T S = 0), so every stage moves the element budget by gamma h E^T
(residual of its solve); a factor without row exchanges that has lost accuracy shows up there. Cells with non-finite
inputs are counted, not recomputed (the library returns a non-finite state for them too); cells below 300 K are not
stepped and the call raises, as TorchRates does. The counts of the last call are in CudaSolver.fallback.
"""
import ctypes
import os
import string

import numpy as np
import torch

import kinbatch as KB
import kinetics as K
import kinnet as KN
import kinvulcan as KV

T_MIN = 300.0   # K, as TorchRates
DRIFT = 1e-6    # relative change of an element's share of the atoms left to rounding (in-domain one-call worst 3e-8)


def _order(pattern, n):
    """Greedy minimum-degree elimination order on the symmetrized pattern (ties by index)."""
    adj = [set() for _ in range(n)]
    for i, j in pattern:
        if i != j:
            adj[i].add(j)
            adj[j].add(i)
    left, order = set(range(n)), []
    while left:
        k = min(left, key=lambda q: (len(adj[q] & left), q))
        order.append(k)
        nb = adj[k] & left
        for a in nb:
            adj[a] |= nb - {a}
        left.remove(k)
    return order


def _lanes(items, size):
    """Items spread over 32 lanes, longest first onto the least-loaded lane: (items per lane, rounds)."""
    lanes, load = [[] for _ in range(32)], np.zeros(32, int)
    for it in sorted(items, key=lambda q: -size(q)):
        l_ = int(np.argmin(load))
        lanes[l_].append(it)
        load[l_] += size(it)
    return lanes, int(load.max())


def _special():
    """kinvulcan's special reaction OH + CH3 + M -> CH3OH + M: k0 as two Arrhenius terms, k_inf; (A, B, C) for A T^B
    exp(-C / T)."""
    return (1.932e3, -9.88, 7544.0), (5.109e-11, -6.25, 1433.0), (1.031e-10, -0.018, -16.74)


def tables(net):
    """The kernel's tables from a VulcanNetwork: species order, rate coefficients, S v and Jacobian term streams, LU
    pattern and update operations (species in the kernel's order: perm[i] = network index of row i)."""
    S = np.asarray(net.S)
    ns, nr = S.shape
    re = [list(r["re"]) for r in net.reac]
    pr = [list(r["pr"]) for r in net.reac]
    if not 32 < ns <= 64 or 2 * nr > 1024 or max(len(q) for q in re + pr) > 3 or np.abs(S).max() > 2:
        raise ValueError("the kernel holds 33-64 species, at most 512 reactions, 3 species per side, |S| <= 2")
    live = [r for r in range(nr) if np.any(S[:, r] != 0)]
    dv = {}                                   # d v_r / d n_q: (sign, rate index (kf r, kr nr + r), other species)
    for r in live:
        for side, k, sgn in ((re[r], r, 1.0), (pr[r], nr + r, -1.0)):
            for m in range(len(side)):
                dv.setdefault((r, side[m]), []).append((sgn, k, side[:m] + side[m + 1:]))
    jn = {}                                   # entry (p, q) of Jn: (S[p, r], (r, q))
    for (r, q) in dv:
        for p in np.flatnonzero(S[:, r]):
            jn.setdefault((int(p), q), []).append((S[p, r], (r, q)))
    pat = set(jn) | {(i, i) for i in range(ns)}
    perm = _order(pat, ns)
    pos = {s: k for k, s in enumerate(perm)}
    P = {(pos[i], pos[j]) for i, j in pat}
    ops = [[] for _ in range(ns)]             # symbolic LU without exchanges: ops[k] = (i, j) updated by pivot k
    for k in range(ns):
        rows = sorted(i for (i, j) in P if j == k and i > k)
        cols = sorted(j for (i, j) in P if i == k and j > k)
        for i in rows:
            for j in cols:
                P.add((i, j))
                ops[k].append((i, j))
    slot = {e: n for n, e in enumerate(sorted(P))}          # row-major
    nnz = len(slot)
    k0 = next(k for k in range(ns) if all((i, j) in P for i in range(k, ns) for j in range(k, ns)))
    k0 = max(k0, ns - 32)
    if k0 > 32:
        raise ValueError(f"the dense trailing block starts at row {k0}; the kernel needs <= 32 sparse rows")
    # reactions: species slots (padding ns: n[ns] = 1, g[ns] = 0), rate laws
    rx = np.full((nr, 6), ns, np.uint8)
    rc = np.zeros((nr, 9))
    rf = np.zeros(nr, np.int32)
    sp0, sp1, spi = _special()
    ln = lambda a: np.log(a) if a > 0 else -np.inf
    for j, r in enumerate(net.reac):
        rx[j, :len(re[j])] = [pos[q] for q in re[j]]
        rx[j, 3:3 + len(pr[j])] = [pos[q] for q in pr[j]]
        if r["kind"] == 2:
            a0, a1, ai = (r["A"], r["B"], r["C"]), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        elif r["kind"] in (3, 5):
            a0, a1 = (r["A0"], r["B0"], r["C0"]), (0.0, 0.0, 0.0)
            ai = (r["Ai"], r["Bi"], r["Ci"]) if r["kind"] == 3 else (1.0, 0.0, 0.0)
        else:
            a0, a1, ai = sp0, sp1, spi
        rc[j] = [ln(a0[0]), a0[1], a0[2], ln(a1[0]), a1[1], a1[2], ln(ai[0]), ai[1], ai[2]]
        dmu = len(pr[j]) - len(re[j])
        rf[j] = (r["kind"] in (3, 6)) | (r["kind"] == 6) << 1 | (r["kind"] in (3, 5, 6)) << 2 | (dmu + 4) << 3
    # S v: each row's terms cut into pieces of <= ch; a row's first piece writes part[i], the others part[ns + e]
    rows = [[(r, S[perm[i], r]) for r in live if S[perm[i], r] != 0] for i in range(ns)]
    ch = -(-sum(len(t) for t in rows) // 32)
    pieces, xp, nx = [], [0], 0
    for i, t in enumerate(rows):
        for c in range(0, len(t), ch):
            pieces.append((i if c == 0 else ns + nx, t[c:c + ch]))
            nx += c > 0
        xp.append(nx)
    npart = ns + nx
    if npart > 127:
        raise ValueError("too many S v pieces")
    lanes, sround = _lanes(range(len(pieces)), lambda q: len(pieces[q][1]))
    sst = np.zeros((sround, 32), np.uint32)
    for l_, its in enumerate(lanes):
        q = 0
        for it in its:
            dst, terms = pieces[it]
            for e, (r, c) in enumerate(terms):
                sst[q, l_] = (r | (c < 0) << 9 | (abs(c) == 2) << 10 | dst << 11 | (e == len(terms) - 1) << 18
                              | 1 << 31)
                q += 1
    # Jacobian: the terms of each slot (one lane each): coefficient * k * n_a * n_b
    terms = [[] for _ in range(nnz)]
    for (p, q), lst in jn.items():
        s = slot[(pos[p], pos[q])]
        for c, (r, q_) in lst:
            for sgn, k, rest in dv[(r, q_)]:
                ab = [pos[o] for o in rest] + [ns] * (2 - len(rest))
                terms[s].append((k, c * sgn, ab[0], ab[1]))
    lanes, jround = _lanes([s for s in range(nnz) if terms[s]], lambda s: len(terms[s]))
    jt = np.zeros((jround, 32), np.uint64)
    for l_, its in enumerate(lanes):
        q = 0
        for s in its:
            for e, (k, c, a, b) in enumerate(terms[s]):
                jt[q, l_] = (int(k) | int(c < 0) << 10 | int(abs(c) == 2) << 11 | a << 12 | b << 18 | s << 24
                             | int(e == len(terms[s]) - 1) << 35 | 1 << 37)
                q += 1
    # LU of the sparse leading part (pivots < k0; their updates reach the trailing block too)
    lu = [(slot[(i, j)] | slot[(i, k)] << 11 | slot[(k, j)] << 22) for k in range(k0) for (i, j) in ops[k]]
    optr = np.cumsum([0] + [len(ops[k]) for k in range(k0)]).astype(np.int32)
    smap = np.full((ns, ns), 0xFFFF, np.uint16)
    for (i, j), s in slot.items():
        smap[i, j] = s
    rowptr = np.searchsorted([i for (i, j) in sorted(P)], np.arange(ns + 1)).astype(np.uint16)
    colof = np.array([j for (i, j) in sorted(P)], np.uint8)
    diag = np.array([slot[(i, i)] for i in range(ns)], np.uint16)
    nasa = np.concatenate([np.asarray(net.lo)[perm], np.asarray(net.hi)[perm]], 0)
    return dict(ns=ns, nr=nr, nnz=nnz, k0=k0, P=P, perm=np.array(perm, np.int32), rx=rx, rc=rc, rf=rf, nasa=nasa,
                npart=npart, xp=np.array(xp, np.uint8), sst=sst, jt=jt, lu=np.array(lu, np.uint64), optr=optr,
                smap=smap, rowptr=rowptr, colof=colof, diag=diag, atoms=np.asarray(net.atoms, float))


_SRC = r"""
#define NS $ns
#define NR $nr
#define NNZ $nnz
#define NPART $npart
#define K0 $k0
#define NT $nt
#define WPB $wpb
#define PERW $perw
#define TBYTES $tbytes
#define LPRE $lpre
#define LTAIL $ltail
#define UPRE $upre
#define UTAIL $utail
#define NEL $nel
#define ETOT $etot
#define TMIN $tmin
#define DRIFT $drift
#define KBOLTZ $kb
#define P0BAR $p0
#define O_SMAP $o_smap
#define O_ROWPTR $o_rowptr
#define O_DIAG $o_diag
#define O_COLOF $o_colof
#define O_RX $o_rx
#define O_XP $o_xp
#define FULL 0xffffffffu
typedef unsigned short u16;
typedef unsigned char u8;
typedef unsigned long long u64;

struct Tab {   // small tables, copied to shared memory once per block
  const u16 *smap, *rowptr, *diag;   // slot of (row, col) or 0xFFFF; first slot of each row; diagonal slots
  const u8 *colof, *rx, *xp;         // column of each slot; species of each reaction (3 + 3, padding NS); S v pieces
};

__device__ __forceinline__ double wsum(double v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
  return v;
}

// F = (n_tot / nd) S v(n) at the state Y (n = nd Y / n_tot, n_tot = sum Y); sv = S v unless null; returns n_tot.
__device__ __forceinline__ double rhs(const double* Y, double* F, double* sv, double* n, double* v, double* part,
                                      const double* kf, const double* kr, double nd, const Tab& T,
                                      const unsigned* __restrict__ sst, int sround, int lane) {
  double p = 0.0;
  for (int i = lane; i < NS; i += 32) p += Y[i];
  const double ntot = wsum(p), xf = nd / ntot, ixf = ntot / nd;
  for (int i = lane; i < NS; i += 32) n[i] = Y[i] * xf;
  for (int i = lane; i < NPART; i += 32) part[i] = 0.0;
  if (lane == 0) n[NS] = 1.0;
  __syncwarp();
  #pragma unroll 2
  for (int r = lane; r < NR; r += 32) {
    const u8* q = T.rx + 6 * r;
    const double a = kf[r] * n[q[0]] * n[q[1]] * n[q[2]];
    const double b = kr[r] * n[q[3]] * n[q[4]] * n[q[5]];
    v[r] = a - b;
  }
  __syncwarp();
  {   // S v as lane streams: reaction, sign, x2, destination piece, end of piece
    double acc = 0.0;
    #pragma unroll 4
    for (int q = 0; q < sround; q++) {
      const unsigned w = __ldg(sst + q * 32 + lane);
      if (w >> 31) {
        const double tv = (((w >> 10) & 1u) ? 2.0 : 1.0) * v[w & 511u];
        acc += ((w >> 9) & 1u) ? -tv : tv;
        if ((w >> 18) & 1u) { part[(w >> 11) & 127u] = acc; acc = 0.0; }
      }
    }
  }
  __syncwarp();
  for (int i = lane; i < NS; i += 32) {
    double s = part[i];
    for (int e = T.xp[i]; e < T.xp[i + 1]; e++) s += part[NS + e];
    F[i] = s * ixf;
    if (sv) sv[i] = s;
  }
  __syncwarp();
  return ntot;
}

// y <- W^-1 y = M0^-1 y + c0 sum(M0^-1 y) u (Sherman-Morrison, W = M0 - w 1^T). Lane l holds rows l (< K0) and K0 + l
// (< NS); L and U of the sparse leading columns come from shared memory, the trailing block from registers c.
// TWIN: also u <- M0^-1 u (u = w on entry) and c0 = 1 / (1 - sum u); returns c0.
template <bool TWIN>
__device__ __forceinline__ double rsolve(const double* M, const double* dinv, double* y, double* u, double c0,
                                         const u16* smap, const double (&c)[NT], int lane, int myt) {
  const bool hp = lane < K0, ht = lane < NT;
  double a0 = hp ? y[lane] : 0.0, a1 = ht ? y[K0 + lane] : 0.0, b0 = 0.0, b1 = 0.0;
  if (TWIN) { b0 = hp ? u[lane] : 0.0; b1 = ht ? u[K0 + lane] : 0.0; }
  const double d0 = hp ? dinv[lane] : 0.0, d1 = ht ? dinv[K0 + lane] : 0.0;
  #pragma unroll
  for (int k = 0; k < K0; k++) {                        // unit L, sparse columns
    if (!(((LPRE | LTAIL) >> k) & 1u)) continue;
    const double yk = __shfl_sync(FULL, a0, k), uk = TWIN ? __shfl_sync(FULL, b0, k) : 0.0;
    if (((LPRE >> k) & 1u) && hp && lane > k) {
      const unsigned s = smap[lane * NS + k];
      if (s != 0xFFFFu) { const double l = M[s]; a0 -= l * yk; b0 -= l * uk; }
    }
    if (((LTAIL >> k) & 1u) && ht) {
      const unsigned s = smap[(K0 + lane) * NS + k];
      if (s != 0xFFFFu) { const double l = M[s]; a1 -= l * yk; b1 -= l * uk; }
    }
  }
  #pragma unroll
  for (int k = 0; k < NT - 1; k++) {                    // unit L, dense trailing columns
    const double yk = __shfl_sync(FULL, a1, k), uk = TWIN ? __shfl_sync(FULL, b1, k) : 0.0;
    if (myt > k) { const double l = c[k]; a1 -= l * yk; b1 -= l * uk; }
  }
  #pragma unroll
  for (int k = NT - 1; k >= 0; k--) {                   // U, trailing columns
    if (k == 0 && !(UTAIL & 1u)) continue;
    const double zk = __shfl_sync(FULL, a1 * d1, k), wk = TWIN ? __shfl_sync(FULL, b1 * d1, k) : 0.0;
    if (((UTAIL >> k) & 1u) && hp) {
      const unsigned s = smap[lane * NS + K0 + k];
      if (s != 0xFFFFu) { const double l = M[s]; a0 -= l * zk; b0 -= l * wk; }
    }
    if (k > 0 && myt >= 0 && myt < k) { const double l = c[k]; a1 -= l * zk; b1 -= l * wk; }
  }
  #pragma unroll
  for (int k = K0 - 1; k >= 0; k--) {                   // U, sparse columns
    if (!((UPRE >> k) & 1u)) continue;
    const double zk = __shfl_sync(FULL, a0 * d0, k), wk = TWIN ? __shfl_sync(FULL, b0 * d0, k) : 0.0;
    if (lane < k) {
      const unsigned s = smap[lane * NS + k];
      if (s != 0xFFFFu) { const double l = M[s]; a0 -= l * zk; b0 -= l * wk; }
    }
  }
  a0 *= d0; a1 *= d1; b0 *= d0; b1 *= d1;
  if (TWIN) {
    c0 = 1.0 / (1.0 - wsum((hp ? b0 : 0.0) + (ht ? b1 : 0.0)));
    if (hp) u[lane] = b0;
    if (ht) u[K0 + lane] = b1;
  }
  const double s = wsum((hp ? a0 : 0.0) + (ht ? a1 : 0.0)) * c0;
  __syncwarp();
  if (hp) y[lane] = a0 + s * u[lane];
  if (ht) y[K0 + lane] = a1 + s * u[K0 + lane];
  __syncwarp();
  return c0;
}

extern "C" __global__ void __launch_bounds__(32 * WPB) call(
    const double* __restrict__ x0, const double* __restrict__ Ta, const double* __restrict__ Pa,
    const double* __restrict__ dta, double* __restrict__ hio, double* __restrict__ xout, int* __restrict__ doneo,
    double* __restrict__ noko, int* __restrict__ count, int B, double rtol, double atol, int max_steps,
    const int* __restrict__ perm, const unsigned* __restrict__ blob, const double* __restrict__ rc,
    const int* __restrict__ rf, const double* __restrict__ nasa, double* scratch, const unsigned* __restrict__ sst,
    const u64* __restrict__ jt, const u64* __restrict__ lu, const int* __restrict__ optr, int sround, int jround)
{
  extern __shared__ double smem[];
  unsigned* tb32 = (unsigned*)(smem + WPB * PERW);
  for (int i = threadIdx.x; i < TBYTES / 4; i += blockDim.x) tb32[i] = blob[i];
  __syncthreads();
  const u8* tb = (const u8*)tb32;
  Tab T;
  T.smap = (const u16*)(tb + O_SMAP); T.rowptr = (const u16*)(tb + O_ROWPTR); T.diag = (const u16*)(tb + O_DIAG);
  T.colof = tb + O_COLOF; T.rx = tb + O_RX; T.xp = tb + O_XP;
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
  const int myt = lane < NT ? lane : -1;      // this lane's row K0 + myt of the dense trailing block
  double c[NT];                               // that row: M0, then its LU factors
  #pragma unroll
  for (int j = 0; j < NT; j++) c[j] = 0.0;
  double* M = smem + wib * PERW;              // M0 by slot, then its LU factors (leading columns)
  double *dinv = M + NNZ, *n = dinv + NS, *v = n + NS + 1, *N = v + NR, *Y = N + NS, *F = Y + NS, *u = F + NS,
         *K1 = u + NS, *K2 = K1 + NS, *K3 = K2 + NS, *K4 = K3 + NS, *sv = K4 + NS, *part = sv + NS;
  double* kf = scratch + (size_t)(blockIdx.x * WPB + wib) * 2 * NR;
  double* kr = kf + NR;
  int cell = blockIdx.x * WPB + wib;
  while (cell < B) {
    const double Tc = Ta[cell], dt = dta[cell], nd = Pa[cell] / (KBOLTZ * Tc);
    double p = 0.0;
    for (int i = lane; i < NS; i += 32) { N[i] = x0[(size_t)cell * NS + perm[i]]; p += N[i]; }
    const bool fin = isfinite(wsum(p)) && isfinite(Tc) && isfinite(nd) && isfinite(dt), cold = Tc < TMIN;
    if (fin && !cold) {   // rate coefficients; three-body ones times nd
      const double lT = log(Tc), iT = 1.0 / Tc, lKT = log(KBOLTZ * Tc / P0BAR);
      double* g = v;                          // G / RT of each species
      const double* cf = nasa + (Tc < 1000.0 ? 0 : NS * 9);
      for (int i = lane; i < NS; i += 32) {
        const double* a = cf + 9 * i;
        const double T2 = Tc * Tc, T3 = T2 * Tc, T4 = T3 * Tc;
        const double hh = -a[0] * iT * iT + a[1] * lT * iT + a[2] + a[3] * Tc / 2 + a[4] * T2 / 3 + a[5] * T3 / 4
                          + a[6] * T4 / 5 + a[7] * iT;
        const double ss = -a[0] * iT * iT / 2 - a[1] * iT + a[2] * lT + a[3] * Tc + a[4] * T2 / 2 + a[5] * T3 / 3
                          + a[6] * T4 / 4 + a[8];
        g[i] = hh - ss;
      }
      if (lane == 0) g[NS] = 0.0;
      __syncwarp();
      for (int r = lane; r < NR; r += 32) {
        const double* q = rc + 9 * r;
        const int fl = rf[r];                 // 1 falloff, 2 second k0 term, 4 three-body, (dmu + 4) << 3
        double k = exp(q[0] + q[1] * lT - q[2] * iT);
        if (fl & 2) k += exp(q[3] + q[4] * lT - q[5] * iT);
        if (fl & 1) k = k / (1.0 + k * nd / exp(q[6] + q[7] * lT - q[8] * iT));
        const u8* x = T.rx + 6 * r;
        const double dg = g[x[3]] + g[x[4]] + g[x[5]] - g[x[0]] - g[x[1]] - g[x[2]];
        const double c3 = (fl & 4) ? nd : 1.0;
        kf[r] = k * c3;
        kr[r] = (k * exp(dg + (double)((fl >> 3) - 4) * lKT)) * c3;
      }
    }
    __syncwarp();
    double h = fmax(fmin(hio[cell], dt), 1e-12), t = 0.0, nok = 0.0;
    int it = (dt > 0.0 && fin && !cold) ? 0 : max_steps;   // no step for non-finite inputs or cold gas
    while (it < max_steps && t < dt * (1.0 - 1e-14)) {
      const double ha = fmin(h, dt - t), igh = 1.0 / (0.5 * ha), ih = 1.0 / ha;
      for (int s = lane; s < NNZ; s += 32) M[s] = 0.0;
      const double ntot = rhs(N, F, sv, n, v, part, kf, kr, nd, T, sst, sround, lane);
      {   // M = -Jn: lane streams of terms (rate index, sign, x2, species a, b, slot, end of slot)
        double acc = 0.0;
        #pragma unroll 4
        for (int q = 0; q < jround; q++) {
          const u64 w = __ldg(jt + q * 32 + lane);
          if ((w >> 37) & 1ull) {
            const double pr = kf[w & 1023ull] * n[(w >> 12) & 63ull] * n[(w >> 18) & 63ull];   // kr = kf + NR
            const double cf = ((w >> 11) & 1ull) ? (((w >> 10) & 1ull) ? -2.0 : 2.0)
                                                  : (((w >> 10) & 1ull) ? -1.0 : 1.0);
            acc = fma(cf, pr, acc);
            if ((w >> 35) & 1ull) { M[(w >> 24) & 2047ull] = -acc; acc = 0.0; }
          }
        }
      }
      __syncwarp();
      for (int i = lane; i < NS; i += 32) {    // u = w = S v / nd - Jn x; M = M0 = I / (gamma h) - Jn
        double acc = 0.0;
        const int s1 = T.rowptr[i + 1];
        for (int s = T.rowptr[i]; s < s1; s++) acc += M[s] * N[T.colof[s]];
        u[i] = sv[i] / nd + acc / ntot;
        M[T.diag[i]] += igh;
      }
      __syncwarp();
      for (int k = 0; k < K0; k++) {           // LU of the sparse leading columns (updates reach the trailing block)
        if (!(((LPRE | LTAIL) >> k) & 1u)) continue;
        const int oa = optr[k], ob = optr[k + 1];
        u64 op = (oa + lane < ob) ? __ldg(lu + oa + lane) : 0ull;
        const double dk = 1.0 / M[T.diag[k]];
        { const unsigned q0 = T.smap[lane * NS + k]; if (lane > k && q0 != 0xFFFFu) M[q0] *= dk; }
        if (lane + 32 < NS) { const unsigned q1 = T.smap[(lane + 32) * NS + k]; if (q1 != 0xFFFFu) M[q1] *= dk; }
        __syncwarp();
        for (int e = oa + lane; e < ob; e += 32) {
          const u64 nx = (e + 32 < ob) ? __ldg(lu + e + 32) : 0ull;
          M[op & 2047ull] -= M[(op >> 11) & 2047ull] * M[(op >> 22) & 2047ull];
          op = nx;
        }
        __syncwarp();
      }
      if (myt >= 0) {                           // dense trailing block into registers (last NT slots of the row)
        const int b = T.rowptr[K0 + myt + 1] - NT;
        #pragma unroll
        for (int j = 0; j < NT; j++) c[j] = M[b + j];
      }
      #pragma unroll
      for (int k = 0; k < NT; k++) {
        const double rp = 1.0 / __shfl_sync(FULL, c[k], k);
        const bool act = myt > k;
        const double l = act ? c[k] * rp : 0.0;
        if (act) c[k] = l;
        #pragma unroll
        for (int j = k + 1; j < NT; j++) { const double uu = __shfl_sync(FULL, c[j], k); if (act) c[j] -= l * uu; }
      }
      {
        double dd = 1.0;
        #pragma unroll
        for (int j = 0; j < NT; j++) if (myt == j) dd = c[j];
        if (myt >= 0) dinv[K0 + myt] = 1.0 / dd;
      }
      for (int i = lane; i < K0; i += 32) dinv[i] = 1.0 / M[T.diag[i]];
      for (int i = lane; i < NS; i += 32) K1[i] = F[i];
      __syncwarp();
      // Rodas3 stages (kinbatch.RODAS3, gamma = 1/2)
      const double c0 = rsolve<true>(M, dinv, K1, u, 0.0, T.smap, c, lane, myt);
      for (int i = lane; i < NS; i += 32) K2[i] = F[i] + 4.0 * ih * K1[i];
      __syncwarp();
      rsolve<false>(M, dinv, K2, u, c0, T.smap, c, lane, myt);
      for (int i = lane; i < NS; i += 32) Y[i] = N[i] + 2.0 * K1[i];
      __syncwarp();
      rhs(Y, F, nullptr, n, v, part, kf, kr, nd, T, sst, sround, lane);
      for (int i = lane; i < NS; i += 32) K3[i] = F[i] + ih * (K1[i] - K2[i]);
      __syncwarp();
      rsolve<false>(M, dinv, K3, u, c0, T.smap, c, lane, myt);
      for (int i = lane; i < NS; i += 32) Y[i] = N[i] + 2.0 * K1[i] + K3[i];
      __syncwarp();
      rhs(Y, F, nullptr, n, v, part, kf, kr, nd, T, sst, sround, lane);
      for (int i = lane; i < NS; i += 32) K4[i] = F[i] + ih * (K1[i] - K2[i] - (8.0 / 3.0) * K3[i]);
      __syncwarp();
      rsolve<false>(M, dinv, K4, u, c0, T.smap, c, lane, myt);
      // step control (kinnet.advance_compact, order 3)
      double e2 = 0.0;
      bool nn = true;
      for (int i = lane; i < NS; i += 32) {
        const double y = N[i] + 2.0 * K1[i] + K3[i] + K4[i];
        Y[i] = y;
        const double sc = atol + rtol * fmax(fabs(N[i]), fabs(y));
        const double q = K4[i] / sc;
        e2 += q * q;
        nn = nn && (y >= -atol);
      }
      const double en = sqrt(wsum(e2) / NS);
      const bool nonneg = __all_sync(FULL, nn), efin = isfinite(en), ok = (en <= 1.0) && nonneg && efin;
      if (ok) {
        for (int i = lane; i < NS; i += 32) N[i] = fmax(Y[i], 0.0);
        t += ha;
        nok += 1.0;
      }
      __syncwarp();
      double fac = efin ? 0.9 / cbrt(fmax(en, 1e-10)) : 0.1;
      fac = fmin(fmax(fac, 0.2), 5.0);
      if (!ok) fac = fmin(fac, 0.5);
      h = ha * fac;
      it++;
    }
    p = 0.0;
    for (int i = lane; i < NS; i += 32) p += N[i];
    const double s = wsum(p);
    for (int i = lane; i < NS; i += 32) xout[(size_t)cell * NS + perm[i]] = N[i] / s;
    if (lane == 0) { hio[cell] = h; doneo[cell] = (t >= dt * (1.0 - 1e-14)) ? 1 : 0; noko[cell] = nok; }
    int nxt = 0;
    if (lane == 0) nxt = gridDim.x * WPB + atomicAdd(count, 1);
    cell = __shfl_sync(FULL, nxt, 0);
    __syncwarp();
  }
}

// Checks of each cell after the call (one thread per cell, network species order); counts in count[1..6]: T < TMIN,
// else non-finite input, else: non-finite state, unfinished, element drift, recompute (redo = 1).
extern "C" __global__ void guard(const double* __restrict__ x0, const double* __restrict__ Ta,
    const double* __restrict__ Pa, const double* __restrict__ dta, const double* __restrict__ xout,
    const int* __restrict__ doneo, const double* __restrict__ noko, const double* __restrict__ atoms,
    int* __restrict__ redo, int* __restrict__ count, int B, double atol)
{
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= B) return;
  double e0[NEL], e1[NEL], p = 0.0;   // atoms of each element in the input and the new state
  #pragma unroll
  for (int e = 0; e < NEL; e++) { e0[e] = 0.0; e1[e] = 0.0; }
  bool xfin = true;
  for (int i = 0; i < NS; i++) {
    const double a = x0[(size_t)cell * NS + i], b = xout[(size_t)cell * NS + i];
    p += a;
    xfin = xfin && isfinite(b);
    #pragma unroll
    for (int e = 0; e < NEL; e++) { e0[e] += atoms[i * NEL + e] * a; e1[e] += atoms[i * NEL + e] * b; }
  }
  const double Tc = Ta[cell], nd = Pa[cell] / (KBOLTZ * Tc);
  int r = 0;
  if (Tc < TMIN) atomicAdd(count + 2, 1);
  else if (!(isfinite(p) && isfinite(Tc) && isfinite(nd) && isfinite(dta[cell]))) atomicAdd(count + 1, 1);
  else {
    double s0 = 0.0, s1 = 0.0;
    #pragma unroll
    for (int e = 0; e < NEL; e++) { s0 += e0[e]; s1 += e1[e]; }
    bool drift = false;   // share moved beyond rounding (DRIFT) and what clamping accepted negatives can add
    #pragma unroll
    for (int e = 0; e < NEL; e++)
      drift = drift || fabs(e1[e] / s1 - e0[e] / s0) > DRIFT * e0[e] / s0 + 2.0 * atol * noko[cell] * ETOT / s0;
    if (!xfin) atomicAdd(count + 3, 1);
    if (!doneo[cell]) atomicAdd(count + 4, 1);
    if (drift) atomicAdd(count + 5, 1);
    r = (!xfin || !doneo[cell] || drift) ? 1 : 0;
    if (r) atomicAdd(count + 6, 1);
  }
  redo[cell] = r;
}
"""


def _blob(tab):
    """The small tables as one byte string for shared memory, and the byte offset of each."""
    blob, offs = b"", {}
    for name in ("smap", "rowptr", "diag", "colof", "rx", "xp"):
        blob += b"\0" * (-len(blob) % 2)
        offs[name] = len(blob)
        blob += np.ascontiguousarray(tab[name]).tobytes()
    return blob + b"\0" * (-len(blob) % 8), offs


def _perw(tab):
    """Shared-memory doubles per warp: M, dinv, n, v, N, Y, F, u, K1-K4, sv, part."""
    ns = tab["ns"]
    return tab["nnz"] + ns + ns + 1 + tab["nr"] + 9 * ns + tab["npart"]


def source(tab, wpb):
    """The kernel's CUDA source for the tables of tables() and wpb warps per block."""
    ns, k0, P = tab["ns"], tab["k0"], tab["P"]
    nt = ns - k0
    bits = lambda ks: f"{sum(1 << k for k in ks)}u"
    blob, offs = _blob(tab)
    return string.Template(_SRC).substitute(
        ns=ns, nr=tab["nr"], nnz=tab["nnz"], npart=tab["npart"], k0=k0, nt=nt, wpb=wpb, perw=_perw(tab),
        tbytes=len(blob),
        lpre=bits(k for k in range(k0) if any((i, k) in P for i in range(k + 1, k0))),
        ltail=bits(k for k in range(k0) if any((i, k) in P for i in range(k0, ns))),
        upre=bits(k for k in range(k0) if any((i, k) in P for i in range(k))),
        utail=bits(j for j in range(nt) if any((i, k0 + j) in P for i in range(k0))),
        nel=tab["atoms"].shape[1], etot=repr(float(tab["atoms"].sum())), tmin=repr(T_MIN), drift=repr(DRIFT),
        kb=repr(K.KB), p0=repr(K.P0), **{f"o_{k}": v for k, v in offs.items()})


_CUDA = []


def _cuda():
    """The CUDA driver library (ctypes), loaded once."""
    if not _CUDA:
        lib = ctypes.CDLL("libcuda.so.1")
        vp, ui = ctypes.c_void_p, ctypes.c_uint
        lib.cuLaunchKernel.argtypes = [vp, ui, ui, ui, ui, ui, ui, ui, vp, ctypes.POINTER(vp), vp]
        lib.cuModuleLoadData.argtypes = [ctypes.POINTER(vp), ctypes.c_char_p]
        lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_char_p]
        lib.cuFuncSetAttribute.argtypes = [vp, ctypes.c_int, ctypes.c_int]
        lib.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, vp]
        lib.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [ctypes.POINTER(ctypes.c_int), vp, ctypes.c_int,
                                                                    ctypes.c_size_t]
        _CUDA.append(lib)
    return _CUDA[0]


def _check(r, what):
    if r != 0:
        s = ctypes.c_char_p()
        _cuda().cuGetErrorString(r, ctypes.byref(s))
        raise RuntimeError(f"CUDA driver, {what}: error {r} ({(s.value or b'').decode()})")


def compile_cubin(src, arch):
    """CUDA source -> cubin for sm_<arch> with NVRTC (the libnvrtc of the nvidia-cuda-nvrtc wheel)."""
    import nvidia.cuda_nvrtc
    nv = ctypes.CDLL(os.path.join(list(nvidia.cuda_nvrtc.__path__)[0], "lib", "libnvrtc.so.12"))
    nv.nvrtcGetErrorString.restype = ctypes.c_char_p

    def chk(r, log=""):
        if r != 0:
            raise RuntimeError(f"NVRTC: {nv.nvrtcGetErrorString(r).decode()}\n{log}")

    prog = ctypes.c_void_p()
    chk(nv.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"kincuda.cu", 0, None, None))
    opts = [f"-arch=sm_{arch}".encode(), b"-std=c++17"]
    r = nv.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts))
    n = ctypes.c_size_t()
    chk(nv.nvrtcGetProgramLogSize(prog, ctypes.byref(n)))
    log = ctypes.create_string_buffer(n.value)
    chk(nv.nvrtcGetProgramLog(prog, log))
    chk(r, log.value.decode())
    chk(nv.nvrtcGetCUBINSize(prog, ctypes.byref(n)))
    cubin = ctypes.create_string_buffer(n.value)
    chk(nv.nvrtcGetCUBIN(prog, cubin))
    chk(nv.nvrtcDestroyProgram(ctypes.byref(prog)))
    return cubin.raw


def _launch(fn, grid, block, smem, stream, args):
    """cuLaunchKernel with args: tensors (pointers) or ctypes scalars."""
    vals = [ctypes.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor) else a for a in args]
    ptrs = (ctypes.c_void_p * len(vals))(*[ctypes.addressof(v) for v in vals])
    _check(_cuda().cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, smem, stream, ptrs, None), "launch")


class CudaSolver:
    """CudaSolver(net, device)(x, T, P, dt, h) -> (x, h, done, accepted steps): the call
    advance_compact(rodas3_step(pivot=False), x, TorchRates(T, P), dt, h, order=3) for x [B, S] mole fractions in the
    network's species order, T [B] K, P [B] dyn cm^-2, dt [B] s and starting step sizes h [B] (float64 CUDA tensors).
    fallback: the counts of the last call's guard (cells with non-finite inputs; kernel cells with a non-finite state,
    an unfinished call, an element drift; cells recomputed by the library path); recomputed_total: cells recomputed
    since construction. Calls run one at a time on torch's current stream (the queue counter and the rate scratch are
    shared)."""

    def __init__(self, net, device):
        self.dev = torch.device(device)
        torch.zeros(1, device=self.dev)                       # torch's context is current from here on
        tab = tables(net)
        self.ns, self.nr = tab["ns"], tab["nr"]
        prop = torch.cuda.get_device_properties(self.dev)
        blob, _ = _blob(tab)
        self.wpb = min(32, (prop.shared_memory_per_block_optin - len(blob)) // (8 * _perw(tab)))
        self.smem = 8 * self.wpb * _perw(tab) + len(blob)
        self.src = source(tab, self.wpb)
        cu = _cuda()
        self._mod, self._fn, self._guard = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        _check(cu.cuModuleLoadData(ctypes.byref(self._mod), compile_cubin(self.src, f"{prop.major}{prop.minor}")),
               "load")
        _check(cu.cuModuleGetFunction(ctypes.byref(self._fn), self._mod, b"call"), "function")
        _check(cu.cuModuleGetFunction(ctypes.byref(self._guard), self._mod, b"guard"), "function")
        _check(cu.cuFuncSetAttribute(self._fn, 8, self.smem), "shared memory")   # MAX_DYNAMIC_SHARED_SIZE_BYTES
        occ = ctypes.c_int()
        _check(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(occ), self._fn, 32 * self.wpb, self.smem),
               "occupancy")
        if occ.value < 1:
            raise RuntimeError("the kernel does not fit on this device")
        self.max_blocks = prop.multi_processor_count * occ.value
        f = lambda a, view=None: torch.tensor(a if view is None else np.asarray(a).view(view), device=self.dev)
        self._tabs = [f(tab["perm"]), f(np.frombuffer(blob, np.int32).copy()), f(tab["rc"]), f(tab["rf"]),
                      f(tab["nasa"])]
        self._scratch = torch.empty(self.max_blocks * self.wpb * 2 * self.nr, dtype=torch.float64, device=self.dev)
        self._streams = [f(tab["sst"].ravel(), np.int32), f(tab["jt"].ravel(), np.int64), f(tab["lu"], np.int64),
                         f(tab["optr"]), ctypes.c_int(tab["sst"].shape[0]), ctypes.c_int(tab["jt"].shape[0])]
        self._atoms = f(tab["atoms"])
        self.count = torch.zeros(7, dtype=torch.int32, device=self.dev)
        self.rates, self.mech = KV.TorchRates(net, self.dev), KB.Mech(net, self.dev)
        self.fallback, self.recomputed_total = {}, 0

    def launch(self, x, T, P, dt, h, rtol=1e-3, atol=1e-18, max_steps=4000):
        """The kernel and its checks, no host sync: (x, h, done, accepted steps, recompute flag) per cell; the counters
        in self.count (queue, non-finite input, T < T_MIN, non-finite state, unfinished, drift, recompute)."""
        if not all(a.is_cuda and a.dtype == torch.float64 for a in (x, T, P, dt, h)) or x.shape[1] != self.ns:
            raise ValueError(f"x [B, {self.ns}], T, P, dt, h [B]: float64 CUDA tensors")
        B = x.shape[0]
        x, T, P, dt = (a.contiguous() for a in (x, T, P, dt))
        xo, ho = torch.empty_like(x), h.contiguous().clone()
        done, redo = (torch.empty(B, dtype=torch.int32, device=self.dev) for _ in range(2))
        nok = torch.empty(B, dtype=torch.float64, device=self.dev)
        nb = max(1, min(-(-B // self.wpb), self.max_blocks))
        self.count.zero_()
        stream = torch.cuda.current_stream(self.dev).cuda_stream
        _launch(self._fn, nb, 32 * self.wpb, self.smem, stream,
                [x, T, P, dt, ho, xo, done, nok, self.count, ctypes.c_int(B), ctypes.c_double(rtol),
                 ctypes.c_double(atol), ctypes.c_int(max_steps), *self._tabs, self._scratch, *self._streams])
        _launch(self._guard, -(-B // 128) or 1, 128, 0, stream,
                [x, T, P, dt, xo, done, nok, self._atoms, redo, self.count, ctypes.c_int(B), ctypes.c_double(atol)])
        return xo, ho, done.bool(), nok, redo.bool()

    def library(self, x, T, P, dt, h, rtol=1e-3, atol=1e-18, max_steps=4000):
        """The same call by the library path (TorchRates, rodas3_step without pivoting, advance_compact)."""
        m = self.mech
        step = lambda N, hh, kf, kr, nd: KB.rodas3_step(m, N, kf, kr, nd, hh, pivot=False)
        return KN.advance_compact(step, x, self.rates(T, P), dt, h, rtol=rtol, atol=atol, max_steps=max_steps, order=3)

    def __call__(self, x, T, P, dt, h, rtol=1e-3, atol=1e-18, max_steps=4000):
        """The guarded call: (x, h, done, accepted steps); raises if any T < T_MIN (see the class docstring)."""
        xo, ho, done, nok, redo = self.launch(x, T, P, dt, h, rtol, atol, max_steps)
        n = self.count.tolist()
        if n[2]:
            raise ValueError(f"{n[2]} cells below {T_MIN:g} K (coldest {float(T.min()):.1f} K): outside the chemistry "
                             "solver's domain")
        self.fallback = dict(zip(("bad_input", "nonfinite", "unfinished", "drift", "recomputed"), n[1:2] + n[3:]))
        if n[6]:
            i = redo.nonzero().squeeze(1)
            new = self.library(x[i], T[i], P[i], dt[i], h[i], rtol, atol, max_steps)
            for out, a in zip((xo, ho, done, nok), new):
                out[i] = a
            self.recomputed_total += n[6]
        return xo, ho, done, nok
