"""Derivatives of one solver call (lib/kincuda.py's call: TorchRates + Rodas3 + advance_compact's step control) as CUDA
kernels generated from the same tables: a tangent (JVP) kernel and an adjoint (VJP) kernel; each also returns the call's
result with the call kernel's arithmetic (bitwise). One warp per cell: each lane walks W of the call kernel's 32 W lane
streams (S v, Jacobian terms, LU updates), and the trailing block, the solves and the Sherman-Morrison steps are the call
kernel's code, so every sum of the primal is the call kernel's. Networks: at most 64 species with at most 32 leading
rows, fewer than 512 reactions, and the Arrhenius, falloff and special laws only (no Troe, efficiency or explicit
reverse laws), e.g. NCHO.

Differentiated inputs: the state x, T, P, dt, the starting step size h, per-cell log rate multipliers lm [B, R] (each
multiplies a reaction's forward and reverse coefficients together) and the Arrhenius table rc [R, 9] (ln A, B, C of k0,
of the special law's second k0 term and of k_inf, as kincuda.tables). Outputs: x_out, h_out.

Two contracts, both on the branch the call takes (accept / reject, clamped species, step-control limits):
  frozen (adapt=False): the accepted step sizes held fixed (replay): no derivative through dt, h or the error estimate,
    rejected attempts drop out, h_out has no tangent;
  adaptive (adapt=True): the step control differentiated as well: ha = min(h, dt - t), h_next = ha fac(err) with
    fac = 0.9 err^(-1/3) inside its limits, rejected attempts included, the carried h in and out.
Ties of min / max / |.| / clamps follow torch's derivative conventions (half to each side of a tie of min / max,
d|a| = 0 at 0, a clamp passes at its bound).

Tangent: for each attempt the stage tangents W Kd_i = bd_i + (hd / (gamma h^2)) K_i + Jd K_i are solved with the
attempt's factors (Sherman-Morrison as the primal); Jd K_i, the derivative of the Jacobian along the tangent applied to
K_i, comes reaction by reaction from hyper-dual products; the rate tangents from the Arrhenius, falloff, three-body and
NASA-9 derivatives.
Adjoint: the call runs forward once storing the state before every attempt (beyond CAP attempts: a checkpoint every
ceil(n / CAP) attempts, each segment recomputed), then reverses the attempts, recomputing each one except the one whose
stages are still in place: transposed solves W^-T y = M0^-T (y + c0 (u . y) 1) on the same LU (the dense block's
factors written back to shared memory); the reverse of F at the stage states Y4, Y3; then one pass at N for F(N) and
the four Jacobian cotangents z_i K_i^T (each the derivative of J^T z_i along K_i). The rate cotangents are reduced to T,
P and the per-cell log multipliers in the kernel and to the Arrhenius table by a second kernel in a fixed order. No
atomics enter the arithmetic (the cell queue only): the same inputs give the same bits on one GPU, whatever the batch.

Recovered cells: after each kernel the call kernel's guard lists the cells CudaSolver would recompute, and a second
kernel (jvp_rec, vjp_rec) redoes each listed cell on the recover kernel's path (kincuda: the same rates, streams and
step control with W = I / (gamma h) - J formed densely and factored with threshold pivoting, REC_STEPS x max_steps
tries; its result bitwise the recover kernel's) and differentiates that path: the stage tangents solved with the dense
pivoted factors, the adjoint's transposed solves W^-T = P^T L^-T U^-T with the same factors and row order (up to CAPR
stored attempts before checkpoints). Its result passes the guard's test (flag 1) or not (flag 2: the cell's tangents
and cotangents are NaN).
"""
import ctypes
import string

import numpy as np
import torch

import kincuda as KC

CAP = 64   # attempts stored per cell before the adjoint switches to segment checkpoints (calls up to CAP^2 attempts)
DMAX = 16  # tangent directions per jvp call
# per-warp work arrays of each kernel (name, length in units of NS unless given) and the arrays sharing their space
# (free while they are used); names a kernel does not use sit at offset 0.
_LAYOUT = {
    "jvp": (["M", "DINV", "N0", "N1", "V", "PART", "NN", "Y", "F", "SV", "U", "K1", "K2", "K3", "K4", "ND", "YD",
             "SV1N", "KD1", "KD2", "KD3", "KD4", "YS", "YDS"],
            {"FD": "F", "SVN": "SV", "T1": "YS", "T2": "YDS", "N2": "YD", "N12": "F"}),
    "vjp": (["M", "DINV", "N0", "N1", "V", "PART", "NN", "Y", "F", "SV", "U", "K1", "K2", "K3", "K4", "NB", "KB1",
             "KB2", "KB3", "Z", "YB", "YB2", "NBD"], {"YS": "F"}),
}
NAMES = ("M", "DINV", "N0", "N1", "N2", "N12", "V", "PART", "NN", "Y", "F", "SV", "U", "K1", "K2", "K3",
         "K4", "ND", "YD", "FD", "SVN", "SV1N", "KD1", "KD2", "KD3", "KD4", "YS", "YDS", "T1", "T2", "NB", "KB1", "KB2",
         "KB3", "Z", "YB", "YB2", "NBD")


def supported(tab):
    """Raises NotImplementedError unless the derivative kernels take the network of kincuda.tables' tab: at most 64
    species with at most 32 leading rows, fewer than 512 reactions, the Arrhenius, falloff and special laws only."""
    if tab["law"] or tab["k0"] > 32 or tab["ns"] > 64 or 2 * tab["nr"] >= 1024:
        raise NotImplementedError("the derivative kernels take networks of at most 64 species (32 leading rows), "
                                  "fewer than 512 reactions and the Arrhenius, falloff and special laws only")


def ad_tables(net, tab):
    """Tables of the derivative kernels beyond kincuda.tables: the stream of the transposed concentration Jacobian (for
    each species the terms rho_k n_a n_b of the reactions holding it, rho = per-reaction cotangent times kf or kr), its
    piece offsets, and the row masks of the transposed triangular solves."""
    ns, nr, k0, P = tab["ns"], tab["nr"], tab["k0"], tab["P"]
    pos = {int(s): k for k, s in enumerate(tab["perm"])}
    S = np.asarray(net.S)
    rows = [[] for _ in range(ns)]
    for r in range(nr):
        if not np.any(S[:, r] != 0):
            continue
        for side, k in ((net.reac[r]["re"], r), (net.reac[r]["pr"], nr + r)):
            sl = [pos[q] for q in side]
            for m in range(len(sl)):
                rest = sl[:m] + sl[m + 1:]
                rest = rest + [ns] * (2 - len(rest))
                rows[sl[m]].append((k, rest[0], rest[1]))
    ch = -(-sum(len(t) for t in rows) // 32)
    pieces, xg, nx = [], [0], 0
    for i, t in enumerate(rows):
        for c in range(0, len(t), ch):
            pieces.append((i if c == 0 else ns + nx, t[c:c + ch]))
            nx += c > 0
        xg.append(nx)
    npg = ns + nx
    if npg > 127:
        raise ValueError("too many gather pieces")
    lanes, ground = KC._lanes(range(len(pieces)), lambda q: len(pieces[q][1]), 32)
    gt = np.zeros((ground, 32), np.uint64)
    for l_, its in enumerate(lanes):
        q = 0
        for it in its:
            dst, terms = pieces[it]
            for e, (k, a, b) in enumerate(terms):
                gt[q, l_] = int(k) | a << 12 | b << 18 | dst << 24 | int(e == len(terms) - 1) << 35 | 1 << 37
                q += 1
    urow = sum(1 << k for k in range(k0) if any((k, j) in P for j in range(k + 1, ns)))
    lrow = sum(1 << k for k in range(k0) if any((k, j) in P for j in range(k)))
    return dict(gt=gt, xg=np.array(xg, np.int32), npg=npg, urow=urow, lrow=lrow)


def layout(tab, ext, kind):
    """Shared-memory offsets (doubles) of kernel kind's per-warp work arrays and their total."""
    ns, nr = tab["ns"], tab["nr"]
    size = dict(M=tab["nnz"], N0=ns + 1, N1=ns + 1, V=max(nr, ns + 1), F=ns + 1, YD=ns + 1, YB=ns + 1, YB2=ns + 1, NBD=ns + 1,
                PART=2 * tab["npart"] if kind == "jvp" else max(tab["npart"], ext["npg"]))
    names, alias = _LAYOUT[kind]
    off, o = dict.fromkeys(NAMES, 0), 0
    for k in names:
        off[k] = o
        o += size.get(k, ns)
    for k, a in alias.items():
        off[k] = off[a]
    return off, o


_SRC_AD = r"""
#define WPB $wpb
#define PERWA $perwa
#define CAP $cap
#define CAPR (2 * CAP)                         // attempts stored per recovered cell (REC_STEPS x max_steps <= CAPR^2)
#define DNAN __longlong_as_double(0x7ff8000000000000ll)   // derivatives of a cell still failed after the recovery
#define DMAX $dmax
#define REC (NS + 3)
#define NPG $npg
#define UROW $urow
#define LROW $lrow
$offsets

__device__ __forceinline__ double sgn(double a) { return a > 0.0 ? 1.0 : (a < 0.0 ? -1.0 : 0.0); }

// The call kernel's rhs on one warp: F = (n_tot / nd) S v(n) at Y, u0 = S v / nd unless null; each lane walks W of the
// call kernel's NL lane streams, so every sum is the call kernel's. Returns n_tot.
__device__ __noinline__ double rhs1(const double* Y, double* F, double* u0, double* n, double* v, double* part,
                                   const double* kf, const double* kr, double nd, const Tab& T,
                                   const unsigned* __restrict__ sst, int sround, int lane) {
  double p = 0.0;
  for (int i = lane; i < NS; i += 32) p += Y[i];
  const double ntot = wsum(p), xf = nd / ntot, ixf = ntot / nd;
  for (int i = lane; i < NS; i += 32) n[i] = Y[i] * xf;
  for (int i = lane; i < NPART; i += 32) part[i] = 0.0;
  if (lane == 0) n[NS] = 1.0;
  __syncwarp();
  #pragma unroll
  for (int qq = 0; qq < (NR + NL - 1) / NL; qq++) {
    const int na = VSA(qq), nb = VSB(qq);
    #pragma unroll
    for (int m = 0; m < W; m++) {
      const int r = lane + 32 * m + NL * qq;
      if (r < NR) {
        const u8* q = T.rx + 6 * r;
        if (na == 3 || nb == 3) {
          const double a = kf[r] * n[q[0]] * n[q[1]] * n[q[2]];
          const double b = kr[r] * n[q[3]] * n[q[4]] * n[q[5]];
          v[r] = a - b;
        } else {
          double a = kf[r] * n[q[0]], b = kr[r] * n[q[3]];
          if (na == 2) a = a * n[q[1]];
          if (nb == 2) b = b * n[q[4]];
          v[r] = __dsub_rn(a, b);
        }
      }
    }
  }
  __syncwarp();
  #pragma unroll
  for (int m = 0; m < W; m++) {
    const int cl = lane + 32 * m;
    double acc = 0.0;
    #pragma unroll 4
    for (int q = 0; q < sround; q++) {
      const unsigned w = __ldg(sst + q * NL + cl);
      if (w >> 31) {
        const double cf = ((w >> $sv_x2) & 1u) ? (((w >> $sv_sg) & 1u) ? -2.0 : 2.0)
                                               : (((w >> $sv_sg) & 1u) ? -1.0 : 1.0);
        acc = fma(cf, v[w & ${sv_r}u], acc);
        if ((w >> $sv_end) & 1u) { part[(w >> $sv_dst) & ${sv_pm}u] = acc; acc = 0.0; }
      }
    }
  }
  __syncwarp();
  for (int i = lane; i < NS; i += 32) {
    double s = part[i];
    for (int e = T.xp[i]; e < T.xp[i + 1]; e++) s += part[NS + e];
    F[i] = s * ixf;
    if (u0) u0[i] = s / nd;
  }
  __syncwarp();
  return ntot;
}

// o1 = S w1, o2 = S w2 in one pass of the S v lane streams (part holds 2 NPART doubles).
__device__ __forceinline__ void svs2(const double* w1, const double* w2, double* o1, double* o2, double* part,
                                     const Tab& T, const unsigned* __restrict__ sst, int sround, int lane) {
  for (int i = lane; i < 2 * NPART; i += 32) part[i] = 0.0;
  __syncwarp();
  for (int m = 0; m < W; m++) {
    const int cl = lane + 32 * m;
    double a1 = 0.0, a2 = 0.0;
    for (int q = 0; q < sround; q++) {
      const unsigned x = __ldg(sst + q * NL + cl);
      if (x >> 31) {
        const int r = x & ${sv_r}u;
        const double cf = (((x >> $sv_x2) & 1u) ? 2.0 : 1.0) * (((x >> $sv_sg) & 1u) ? -1.0 : 1.0);
        a1 += cf * w1[r];
        a2 += cf * w2[r];
        if ((x >> $sv_end) & 1u) {
          const int d = (x >> $sv_dst) & ${sv_pm}u;
          part[2 * d] = a1; part[2 * d + 1] = a2; a1 = 0.0; a2 = 0.0;
        }
      }
    }
  }
  __syncwarp();
  for (int i = lane; i < NS; i += 32) {
    double s1 = part[2 * i], s2 = part[2 * i + 1];
    for (int e = T.xp[i]; e < T.xp[i + 1]; e++) { s1 += part[2 * (NS + e)]; s2 += part[2 * (NS + e) + 1]; }
    o1[i] = s1;
    o2[i] = s2;
  }
  __syncwarp();
}

// Rate coefficients of the cell (the call kernel's code), times exp(lm) when per-cell log multipliers are given.
__device__ __forceinline__ void rates(double* S, double* kf, double* kr, double Tc, double nd, const double* lmc,
    const double* __restrict__ rc, const int* __restrict__ rf, const double* __restrict__ nasa, const Tab& T, int lane)
{
  const double lT = log(Tc), iT = 1.0 / Tc, lKT = log(KBOLTZ * Tc / P0BAR);
  double* g = S + A_V;                    // G / RT of each species
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
    const int fl = rf[r];
    const double k = law(rc + RCW * r, fl, Tc, lT, iT, nd);
    const u8* x = T.rx + 6 * r;
    const double dg = g[x[3]] + g[x[4]] + g[x[5]] - g[x[0]] - g[x[1]] - g[x[2]];
    const double c3 = (fl & 4) ? nd : 1.0;
    kf[r] = k * c3;
    kr[r] = $kr;
    if (lmc) { const double e = exp(lmc[r]); kf[r] *= e; kr[r] *= e; }
  }
  __syncwarp();
}

// Partial derivatives of ln kf of a reaction (rate law q = its rc row, flags fl) at (T, nd): dT = d/dT, dn = d/d ln nd,
// D[j] = d/d q[j]. ln kr adds d(sum G/RT, products - reactants)/dT + dmu / T to dT.
__device__ __forceinline__ void kpart(const double* q, int fl, double lT, double iT, double nd, double& dT, double& dn,
                                      double* D) {
  const double e0 = exp(q[0] + q[1] * lT - q[2] * iT), e1 = (fl & 2) ? exp(q[3] + q[4] * lT - q[5] * iT) : 0.0;
  const double kl = e0 + e1, w0 = kl > 0.0 ? e0 / kl : 0.0, w1 = kl > 0.0 ? e1 / kl : 0.0;
  double ph = 1.0, pc = 0.0;              // d ln k / d ln k_low, d ln k / d ln k_inf
  if (fl & 1) {
    const double rr = kl * nd / exp(q[6] + q[7] * lT - q[8] * iT);
    ph = 1.0 / (1.0 + rr);
    pc = rr / (1.0 + rr);
  }
  D[0] = ph * w0; D[1] = ph * w0 * lT; D[2] = -ph * w0 * iT;
  D[3] = ph * w1; D[4] = ph * w1 * lT; D[5] = -ph * w1 * iT;
  D[6] = pc; D[7] = pc * lT; D[8] = -pc * iT;
  dT = (ph * (w0 * (q[1] + q[2] * iT) + w1 * (q[4] + q[5] * iT)) + pc * (q[7] + q[8] * iT)) * iT;
  dn = -pc + ((fl & 4) ? 1.0 : 0.0);
}

// d(G/RT)/dT = -(H/RT)/T of each species into g[0..NS] (g[NS] = 0).
__device__ __forceinline__ void gibbs_dT(double* g, double Tc, const double* __restrict__ nasa, int lane) {
  const double lT = log(Tc), iT = 1.0 / Tc;
  const double* cf = nasa + (Tc < 1000.0 ? 0 : NS * 9);
  for (int i = lane; i < NS; i += 32) {
    const double* a = cf + 9 * i;
    const double T2 = Tc * Tc, T3 = T2 * Tc, T4 = T3 * Tc;
    const double hh = -a[0] * iT * iT + a[1] * lT * iT + a[2] + a[3] * Tc / 2 + a[4] * T2 / 3 + a[5] * T3 / 4
                      + a[6] * T4 / 5 + a[7] * iT;
    g[i] = -hh * iT;
  }
  if (lane == 0) g[NS] = 0.0;
  __syncwarp();
}

// Tangents of ln kf, ln kr (dk[r], dk[NR + r]) for the tangents of T (dTc), ln nd (dlnd), the log multipliers (dlmc,
// may be null) and the Arrhenius table (drc, may be null).
__device__ __forceinline__ void rates_tan(double* S, double* dk, double Tc, double nd, double dTc, double dlnd,
    const double* dlmc, const double* __restrict__ rc, const double* drc, const int* __restrict__ rf,
    const double* __restrict__ nasa, const Tab& T, int lane)
{
  const double lT = log(Tc), iT = 1.0 / Tc;
  double* gT = S + A_N1;
  gibbs_dT(gT, Tc, nasa, lane);
  for (int r = lane; r < NR; r += 32) {
    const int fl = rf[r];
    double dT, dn, D[9];
    kpart(rc + 9 * r, fl, lT, iT, nd, dT, dn, D);
    double d = dT * dTc + dn * dlnd;
    if (drc) for (int j = 0; j < 9; j++) d += D[j] * drc[9 * r + j];
    if (dlmc) d += dlmc[r];
    const u8* x = T.rx + 6 * r;
    const double ddg = gT[x[3]] + gT[x[4]] + gT[x[5]] - gT[x[0]] - gT[x[1]] - gT[x[2]];
    dk[r] = d;
    dk[NR + r] = d + (ddg + (double)(((fl >> 3) & 7) - 4) * iT) * dTc;
  }
  __syncwarp();
}

// y <- M0^-1 y on one warp (the call kernel's msolve, results stored).
__device__ __forceinline__ void msol(const double* M, const double* dinv, double* y, const u16* smap,
                                     const double (&c)[NT], int lane, int myt) {
  double a0[RP], a1;
  msolve(M, dinv, y, smap, c, lane, myt, a0, a1);
  __syncwarp();
  #pragma unroll
  for (int p = 0; p < RP; p++) if (lane + 32 * p < K0) y[lane + 32 * p] = a0[p];
  if (lane < NT) y[K0 + lane] = a1;
  __syncwarp();
}

// The Sherman-Morrison update of a solved stage: y <- y + c0 (sum y) u (the call kernel's arithmetic).
__device__ __forceinline__ void smup(double* y, const double* u, double c0, int lane) {
  const bool ht = lane < NT;
  double a0[RP];
  #pragma unroll
  for (int p = 0; p < RP; p++) a0[p] = lane + 32 * p < K0 ? y[lane + 32 * p] : 0.0;
  const double a1 = ht ? y[K0 + lane] : 0.0;
  double sa = a0[0];
  #pragma unroll
  for (int p = 1; p < RP; p++) sa += a0[p];
  const double s = wsum(sa + a1) * c0;
  __syncwarp();
  #pragma unroll
  for (int p = 0; p < RP; p++) if (lane + 32 * p < K0) y[lane + 32 * p] = a0[p] + s * u[lane + 32 * p];
  if (ht) y[K0 + lane] = a1 + s * u[K0 + lane];
  __syncwarp();
}

// y <- W^-1 y = M0^-1 y + c0 (1 . M0^-1 y) u with the attempt's factors.
__device__ __forceinline__ void wsolve(const double* M, const double* dinv, double* y, const double* u, double c0,
                                       const u16* smap, const double (&c)[NT], int lane, int myt) {
  msol(M, dinv, y, smap, c, lane, myt);
  smup(y, u, c0, lane);
}

// y <- W^-1 y with the attempt's factors: the sparse LU of M0 and the Sherman-Morrison step, or (DN: the recovery's
// attempt) the dense threshold-pivoted LU of W in Wd with 1 / U_ii at S + A_DINV and the row order at S + A_M (as ints).
template <bool DN> __device__ __forceinline__ void xsolve(double* S, const double* Wd, const double (&c)[NT], double c0,
                                                          double* y, const u16* smap, int lane, int myt) {
  if (DN) dense_solve(Wd, S + A_DINV, (const int*)(S + A_M), y, lane);
  else wsolve(S + A_M, S + A_DINV, y, S + A_U, c0, smap, c, lane, myt);
}

// One Rodas3 attempt of size ha from N with the call kernel's arithmetic on one warp (its NL lane streams walked W per
// lane): the LU of M0 in M (leading slots and the trailing block's factors) and c (the block), dinv, u = M0^-1 w, c0,
// K1..K4, Y = the new state; ok = accepted; returns the RMS error.
__device__ __forceinline__ double attempt(double* S, double (&c)[NT], double ha, const double* kf, const double* kr,
    double nd, double rtol, double atol, double& c0, bool& ok, const Tab& T, const unsigned* __restrict__ sst,
    const u64* __restrict__ jt, const u64* __restrict__ lu, const int* __restrict__ optr, int sround, int jround,
    int lane, int myt)
{
  double *M = S + A_M, *dinv = S + A_DINV, *n = S + A_N0, *v = S + A_V, *N = S + A_NN, *Y = S + A_Y, *F = S + A_F,
         *u = S + A_U, *K1 = S + A_K1, *K2 = S + A_K2, *K3 = S + A_K3, *K4 = S + A_K4, *part = S + A_PART;
  const double igh = 1.0 / (0.5 * ha), ih = 1.0 / ha;
  const double ntot = rhs1(N, F, u, n, v, part, kf, kr, nd, T, sst, sround, lane);
  for (int s = lane; s < NNZ; s += 32) M[s] = 0.0;
  __syncwarp();
  {   // M = -Jn: the call kernel's lane streams of 32-bit terms
    const unsigned* jw = (const unsigned*)jt;
    #pragma unroll
    for (int m = 0; m < W; m++) {
      const int cl = lane + 32 * m;
      const u16* jsl = (const u16*)(jw + jround * NL) + cl * JSMAX;
      double acc = 0.0;
      int ne = 0;
      #pragma unroll 4
      for (int q = 0; q < jround; q++) {
        const unsigned w = __ldg(jw + q * NL + cl);
        const double pr = kf[w & ${j_km}u] * n[(w >> $j_a) & ${j_sm}u] * n[(w >> $j_b) & ${j_sm}u];
        const double cf = ((w >> $j_x2) & 1u) ? (((w >> $j_sg) & 1u) ? -2.0 : 2.0)
                                              : (((w >> $j_sg) & 1u) ? -1.0 : 1.0);
        acc = fma(cf, pr, acc);
#if NFIX
        if ((w >> $j_end) & 1u) {
          const unsigned d = __ldg(jsl + ne);
          if (d < NNZ) M[d] = -acc; else Y[d - NNZ] = -acc;
          acc = 0.0;
          ne++;
        }
#else
        if ((w >> $j_end) & 1u) { M[__ldg(jsl + ne)] = -acc; acc = 0.0; ne++; }
#endif
      }
    }
  }
  __syncwarp();
#if NFIX
  if (lane == 0) {
    const u16* fx = (const u16*)((const unsigned*)jt + jround * NL) + NL * JSMAX;
    for (int e = 0; e < NFIX; e++) M[__ldg(fx + e)] += Y[e];
  }
  __syncwarp();
#endif
  for (int i = lane; i < NS; i += 32) {       // u = w = S v / nd - Jn x; M = M0 = I / (gamma h) - Jn
    double acc = 0.0;
    const int s1 = T.rowptr[i + 1];
    for (int s = T.rowptr[i]; s < s1; s++) acc += M[s] * N[T.colof[s]];
    if (i >= K0)
      for (int j = 0; j < NT; j++) acc += M[TB0 + (i - K0) * NT + j] * N[K0 + j];
    u[i] = u[i] + acc / ntot;
    M[T.diag[i]] += igh;
  }
  __syncwarp();
  for (int k = 0; k < K0; k++) {               // LU of the sparse leading columns
    const int len = T.blen[k];
    if (len) {
      for (int i = lane; i < len; i += 32) dinv[k + i] = 1.0 / M[T.diag[k + i]];
      __syncwarp();
    }
    if (!(LPRE(k) | LTAIL(k))) continue;
    const int oa = optr[k], ob = optr[k + 1];
    const double dk = dinv[k];
    for (int i = lane; i < NS; i += 32) {
      const unsigned q = T.smap[i * NS + k];
      if (i > k && q != 0xFFFFu) M[q] *= dk;
    }
    __syncwarp();
    for (int e = oa + lane; e < ob; e += 32) {
      const u64 op = __ldg(lu + e);
      M[op & ${slm}ull] -= M[(op >> $lu_b) & ${slm}ull] * M[(op >> $lu_c) & ${slm}ull];
    }
    __syncwarp();
  }
  if (myt >= 0) {                              // dense trailing block in registers; 1 / its diagonal into dinv
    #pragma unroll
    for (int j = 0; j < NT; j++) c[j] = M[TB0 + myt * NT + j];
  }
  #pragma unroll
  for (int k = 0; k < NT; k++) {
    const double rp = 1.0 / __shfl_sync(FULL, c[k], k);
    if (myt == k) dinv[K0 + k] = rp;
    const bool act = myt > k;
    const double l = act ? c[k] * rp : 0.0;
    if (act) c[k] = l;
    #pragma unroll
    for (int j = k + 1; j < NT; j++) { const double uu = __shfl_sync(FULL, c[j], k); if (act) c[j] -= l * uu; }
  }
  if (myt >= 0) {                              // the factors back into the block's slots (transposed solves)
    #pragma unroll
    for (int j = 0; j < NT; j++) M[TB0 + myt * NT + j] = c[j];
  }
  for (int i = lane; i < NS; i += 32) K1[i] = F[i];
  __syncwarp();
  msol(M, dinv, K1, T.smap, c, lane, myt);     // stage 1: K1 and u = M0^-1 w, c0 = 1 / (1 - sum u)
  {
    const bool ht = lane < NT;
    double a0[RP], a1;
    msolve(M, dinv, u, T.smap, c, lane, myt, a0, a1);
    double sb = lane < K0 ? a0[0] : 0.0;
    #pragma unroll
    for (int p = 1; p < RP; p++) sb += lane + 32 * p < K0 ? a0[p] : 0.0;
    c0 = 1.0 / (1.0 - wsum(sb + (ht ? a1 : 0.0)));
    __syncwarp();
    #pragma unroll
    for (int p = 0; p < RP; p++) if (lane + 32 * p < K0) u[lane + 32 * p] = a0[p];
    if (ht) u[K0 + lane] = a1;
    __syncwarp();
  }
  smup(K1, u, c0, lane);
  for (int i = lane; i < NS; i += 32) K2[i] = F[i] + 4.0 * ih * K1[i];
  __syncwarp();
  wsolve(M, dinv, K2, u, c0, T.smap, c, lane, myt);
  for (int i = lane; i < NS; i += 32) Y[i] = N[i] + 2.0 * K1[i];
  __syncwarp();
  rhs1(Y, F, nullptr, n, v, part, kf, kr, nd, T, sst, sround, lane);
  for (int i = lane; i < NS; i += 32) K3[i] = F[i] + ih * (K1[i] - K2[i]);
  __syncwarp();
  wsolve(M, dinv, K3, u, c0, T.smap, c, lane, myt);
  for (int i = lane; i < NS; i += 32) {
    double y = N[i] + 2.0 * K1[i];
    y = y + K3[i];
    Y[i] = y;
  }
  __syncwarp();
  rhs1(Y, F, nullptr, n, v, part, kf, kr, nd, T, sst, sround, lane);
  for (int i = lane; i < NS; i += 32) K4[i] = F[i] + ih * (K1[i] - K2[i] - (8.0 / 3.0) * K3[i]);
  __syncwarp();
  wsolve(M, dinv, K4, u, c0, T.smap, c, lane, myt);
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
  const bool nonneg = __all_sync(FULL, nn), efin = isfinite(en);
  ok = (en <= 1.0) && nonneg && efin;
  __syncwarp();
  return en;
}

// One attempt of the recover kernel (kincuda) on one warp, its arithmetic bitwise: W = I / (gamma h) - Jn - w 1^T formed
// densely in Wd from the call kernel's streams, dense_lu (1 / U_ii in dinv, the row order in pr = S + A_M as ints), the
// stages with dense_solve; K1..K4, Y = the new state; ok = accepted; returns the RMS error.
__device__ __forceinline__ double rattempt(double* S, double* Wd, double ha, const double* kf, const double* kr,
    double nd, double rtol, double atol, bool& ok, const Tab& T, const unsigned* __restrict__ sst,
    const u64* __restrict__ jt, const u16* __restrict__ didx, int sround, int jround, int lane)
{
  double *dinv = S + A_DINV, *n = S + A_N0, *v = S + A_V, *N = S + A_NN, *Y = S + A_Y, *F = S + A_F, *u = S + A_U,
         *K1 = S + A_K1, *K2 = S + A_K2, *K3 = S + A_K3, *K4 = S + A_K4, *part = S + A_PART;
  int* pr = (int*)(S + A_M);
  const double igh = 1.0 / (0.5 * ha), ih = 1.0 / ha;
  const double ntot = rhs1(N, F, u, n, v, part, kf, kr, nd, T, sst, sround, lane);
  for (int s = lane; s < NS * NS; s += 32) Wd[s] = 0.0;
  __syncwarp();
  {   // W = -Jn: the call kernel's term streams, written to the dense positions of the slots
    const unsigned* jw = (const unsigned*)jt;
    #pragma unroll
    for (int m = 0; m < W; m++) {
      const int cl = lane + 32 * m;
      const u16* jsl = (const u16*)(jw + jround * NL) + cl * JSMAX;
      double acc = 0.0;
      int ne = 0;
      #pragma unroll 4
      for (int qq = 0; qq < jround; qq++) {
        const unsigned w = __ldg(jw + qq * NL + cl);
        const double pq = kf[w & ${j_km}u] * n[(w >> $j_a) & ${j_sm}u] * n[(w >> $j_b) & ${j_sm}u];
        const double cf = ((w >> $j_x2) & 1u) ? (((w >> $j_sg) & 1u) ? -2.0 : 2.0)
                                              : (((w >> $j_sg) & 1u) ? -1.0 : 1.0);
        acc = fma(cf, pq, acc);
        if ((w >> $j_end) & 1u) {
          const unsigned d = __ldg(jsl + ne);
          if (d < NNZ) Wd[__ldg(didx + d)] = -acc; else Y[d - NNZ] = -acc;
          acc = 0.0;
          ne++;
        }
      }
    }
  }
  __syncwarp();
#if NFIX
  if (lane == 0) {                              // the extra pieces of long slots, in order
    const u16* fx = (const u16*)((const unsigned*)jt + jround * NL) + NL * JSMAX;
    for (int e = 0; e < NFIX; e++) Wd[__ldg(didx + __ldg(fx + e))] += Y[e];
  }
  __syncwarp();
#endif
  for (int i = lane; i < NS; i += 32) {         // u = w = S v / nd - Jn x (the call kernel's sums, columns in order)
    double acc = 0.0;
    for (int j = 0; j < NS; j++)
      if (T.smap[i * NS + j] != 0xFFFFu) acc += Wd[i * NS + j] * N[j];
    u[i] = u[i] + acc / ntot;
  }
  __syncwarp();
  for (int s = lane; s < NS * NS; s += 32) {    // W = I / (gamma h) - Jn - w 1^T
    const int i = s / NS;
    Wd[s] = (s == i * (NS + 1) ? Wd[s] + igh : Wd[s]) - u[i];
  }
  for (int i = lane; i < NS; i += 32) K1[i] = F[i];
  __syncwarp();
  dense_lu(Wd, dinv, pr, lane);
  dense_solve(Wd, dinv, pr, K1, lane);
  for (int i = lane; i < NS; i += 32) K2[i] = F[i] + 4.0 * ih * K1[i];
  __syncwarp();
  dense_solve(Wd, dinv, pr, K2, lane);
  for (int i = lane; i < NS; i += 32) Y[i] = N[i] + 2.0 * K1[i];
  __syncwarp();
  rhs1(Y, F, nullptr, n, v, part, kf, kr, nd, T, sst, sround, lane);
  for (int i = lane; i < NS; i += 32) K3[i] = F[i] + ih * (K1[i] - K2[i]);
  __syncwarp();
  dense_solve(Wd, dinv, pr, K3, lane);
  for (int i = lane; i < NS; i += 32) Y[i] = N[i] + 2.0 * K1[i] + K3[i];
  __syncwarp();
  rhs1(Y, F, nullptr, n, v, part, kf, kr, nd, T, sst, sround, lane);
  for (int i = lane; i < NS; i += 32) K4[i] = F[i] + ih * (K1[i] - K2[i] - (8.0 / 3.0) * K3[i]);
  __syncwarp();
  dense_solve(Wd, dinv, pr, K4, lane);
  double e2 = 0.0;
  bool nn = true;
  for (int i = lane; i < NS; i += 32) {
    const double y = N[i] + 2.0 * K1[i] + K3[i] + K4[i];
    Y[i] = y;
    const double sc = atol + rtol * fmax(fabs(N[i]), fabs(y));
    const double q4 = K4[i] / sc;
    e2 += q4 * q4;
    nn = nn && (y >= -atol);
  }
  const double en = sqrt(wsum(e2) / NS);
  const bool nonneg = __all_sync(FULL, nn), efin = isfinite(en);
  ok = (en <= 1.0) && nonneg && efin;
  __syncwarp();
  return en;
}

// An attempt of the call kernel (attempt) or, DN, of the recover kernel (rattempt; c0 = 0, unused).
template <bool DN> __device__ __forceinline__ double xattempt(double* S, double (&c)[NT], double* Wd, double ha,
    const double* kf, const double* kr, double nd, double rtol, double atol, double& c0, bool& ok, const Tab& T,
    const unsigned* __restrict__ sst, const u64* __restrict__ jt, const u64* __restrict__ lu,
    const int* __restrict__ optr, const u16* __restrict__ didx, int sround, int jround, int lane, int myt)
{
  if (DN) {
    c0 = 0.0;
    return rattempt(S, Wd, ha, kf, kr, nd, rtol, atol, ok, T, sst, jt, didx, sround, jround, lane);
  }
  return attempt(S, c, ha, kf, kr, nd, rtol, atol, c0, ok, T, sst, jt, lu, optr, sround, jround, lane, myt);
}

// Step-size factor of the controller after an attempt with RMS error en (accepted ok) and its derivative d fac / d en
// on the branch taken (0 where a limit holds).
__device__ __forceinline__ double factor(double en, bool ok, double& dfde) {
  const bool efin = isfinite(en);
  double fac = efin ? 0.9 / cbrt(fmax(en, 1e-10)) : 0.1;
  dfde = (efin && en >= 1e-10) ? -fac / (3.0 * en) : 0.0;
  if (fac < 0.2 || fac > 5.0) dfde = 0.0;
  fac = fmin(fmax(fac, 0.2), 5.0);
  if (!ok) { if (fac > 0.5) dfde = 0.0; fac = fmin(fac, 0.5); }
  return fac;
}

// One attempt with acceptance and step control (the call kernel's loop body; DN: the recover kernel's); returns ok, en =
// its RMS error, c0 = its Sherman-Morrison factor (its stages and factors stay in place).
template <bool DN> __device__ __forceinline__ bool fstep(double* S, double (&c)[NT], double* Wd, double& h, double& t,
    double& nok, double dt, const double* kf, const double* kr, double nd, double rtol, double atol, double& en,
    double& c0, const Tab& T, const unsigned* __restrict__ sst, const u64* __restrict__ jt,
    const u64* __restrict__ lu, const int* __restrict__ optr, const u16* __restrict__ didx, int sround, int jround,
    int lane, int myt)
{
  double *N = S + A_NN, *Y = S + A_Y;
  const double ha = fmin(h, dt - t);
  double dfde;
  bool ok;
  en = xattempt<DN>(S, c, Wd, ha, kf, kr, nd, rtol, atol, c0, ok, T, sst, jt, lu, optr, didx, sround, jround, lane,
                    myt);
  if (ok) {
    for (int i = lane; i < NS; i += 32) N[i] = fmax(Y[i], 0.0);
    t += ha;
    nok += 1.0;
  }
  __syncwarp();
  h = ha * factor(en, ok, dfde);
  return ok;
}

#if KIND == 0
// ---------------------------------------------------------------------------------------------------------- tangent
// The tangent of F = (n_tot / nd) S v at Y, Fd (unless null), for the state tangent Yd and the rate tangents (dk:
// d ln kf, d ln kr; dlnd: d ln nd); leaves n, n1 = the concentrations and their tangent; sv, svd = S v and its tangent
// (va: a per-reaction work array). Returns n_tot, td = sum Yd.
__device__ __forceinline__ double rhs_dual(double* S, const double* Y, const double* Yd, double* Fd, double* sv,
    double* svd, double* va, const double* kf, const double* kr, const double* dk, double nd, double dlnd, double& td,
    const Tab& T, const unsigned* __restrict__ sst, int sround, int lane)
{
  double *n = S + A_N0, *n1 = S + A_N1, *v = S + A_V, *part = S + A_PART;
  double p = 0.0, pd = 0.0;
  for (int i = lane; i < NS; i += 32) { p += Y[i]; pd += Yd[i]; }
  const double ntot = wsum(p), xf = nd / ntot;
  td = wsum(pd);
  const double rt = td / ntot;
  for (int i = lane; i < NS; i += 32) { n[i] = Y[i] * xf; n1[i] = xf * (Yd[i] - Y[i] * rt) + n[i] * dlnd; }
  if (lane == 0) { n[NS] = 1.0; n1[NS] = 0.0; }
  __syncwarp();
  for (int r = lane; r < NR; r += 32) {
    const u8* q = T.rx + 6 * r;
    const double a0 = n[q[0]], a1 = n[q[1]], a2 = n[q[2]], b0 = n[q[3]], b1 = n[q[4]], b2 = n[q[5]];
    const double P = a0 * a1 * a2, Q = b0 * b1 * b2;
    const double Pd = n1[q[0]] * a1 * a2 + a0 * n1[q[1]] * a2 + a0 * a1 * n1[q[2]];
    const double Qd = n1[q[3]] * b1 * b2 + b0 * n1[q[4]] * b2 + b0 * b1 * n1[q[5]];
    v[r] = kf[r] * P - kr[r] * Q;
    va[r] = kf[r] * (dk[r] * P + Pd) - kr[r] * (dk[NR + r] * Q + Qd);
  }
  __syncwarp();
  svs2(v, va, sv, svd, part, T, sst, sround, lane);
  if (Fd) {
    const double g = ntot / nd, gd = g * (rt - dlnd);
    for (int i = lane; i < NS; i += 32) Fd[i] = svd[i] * g + sv[i] * gd;
    __syncwarp();
  }
  return ntot;
}

// out = d/de [J(N) K] along the tangent (N + e Nd, ln k + e dk, ln nd + e dlnd): the second derivative of F, reaction
// by reaction, from hyper-dual products. n, n1 = the concentrations at N and their tangent; svN, sv1N = S v at N and
// its tangent; ntot, td = sum N, sum Nd.
__device__ __forceinline__ void jdot(double* S, const double* N, const double* Nd, const double* K, double* out,
    const double* svN, const double* sv1N, double ntot, double td, const double* kf, const double* kr,
    const double* dk, double nd, double dlnd, double* sv2, double* sv12, double* va, const Tab& T,
    const unsigned* __restrict__ sst, int sround, int lane)
{
  double *n = S + A_N0, *n1 = S + A_N1, *n2 = S + A_N2, *n12 = S + A_N12, *v = S + A_V, *part = S + A_PART;
  double p = 0.0;
  for (int i = lane; i < NS; i += 32) p += K[i];
  const double t2 = wsum(p);
  for (int i = lane; i < NS; i += 32) {
    const double x = N[i] / ntot, x1 = (Nd[i] - x * td) / ntot, x2 = (K[i] - x * t2) / ntot;
    const double x12 = -(x1 * t2 + x2 * td) / ntot;
    n2[i] = nd * x2;
    n12[i] = nd * (x12 + x2 * dlnd);
  }
  if (lane == 0) { n2[NS] = 0.0; n12[NS] = 0.0; }
  __syncwarp();
  for (int r = lane; r < NR; r += 32) {
    const u8* q = T.rx + 6 * r;
    double hp[2][4];
    #pragma unroll
    for (int s = 0; s < 2; s++) {
      double p0 = 1.0, p1 = 0.0, p2 = 0.0, p12 = 0.0;
      #pragma unroll
      for (int m = 0; m < 3; m++) {
        const int j = q[3 * s + m];
        const double a = n[j], a1 = n1[j], a2 = n2[j], a12 = n12[j];
        p12 = p12 * a + p1 * a2 + p2 * a1 + p0 * a12;
        p1 = p1 * a + p0 * a1;
        p2 = p2 * a + p0 * a2;
        p0 = p0 * a;
      }
      hp[s][0] = p0; hp[s][1] = p1; hp[s][2] = p2; hp[s][3] = p12;
    }
    v[r] = kf[r] * hp[0][2] - kr[r] * hp[1][2];
    va[r] = kf[r] * (hp[0][3] + hp[0][2] * dk[r]) - kr[r] * (hp[1][3] + hp[1][2] * dk[NR + r]);
  }
  __syncwarp();
  svs2(v, va, sv2, sv12, part, T, sst, sround, lane);
  const double g = ntot / nd, g1 = g * (td / ntot - dlnd), g2 = t2 / nd, g12 = -t2 * dlnd / nd;
  for (int i = lane; i < NS; i += 32) out[i] = sv12[i] * g + sv1N[i] * g2 + sv2[i] * g1 + svN[i] * g12;
  __syncwarp();
}

// Tangent of the attempt just taken (K1..K4, Y, the factors, u, c0 in place; DN: the recovery's dense factors) for the
// state tangent Nd, the step-size tangent had and the rate tangents: Yd = the new state's tangent; returns the RMS
// error's tangent (adapt) or 0.
template <bool DN> __device__ __forceinline__ double tstep(double* S, const double (&c)[NT], const double* Wd,
    double ha, double had, double c0, double en, int adapt, const double* kf, const double* kr, const double* dk,
    double* va, double nd, double dlnd, double rtol, double atol, const Tab& T, const unsigned* __restrict__ sst,
    int sround, int lane, int myt)
{
  double *N = S + A_NN, *Nd = S + A_ND, *Y = S + A_Y, *Yd = S + A_YD, *Fd = S + A_FD, *svN = S + A_SVN,
         *sv1N = S + A_SV1N, *Ys = S + A_YS, *Yds = S + A_YDS, *t1 = S + A_T1, *t2 = S + A_T2, *K = S + A_K1,
         *Kd = S + A_KD1;
  double *K1 = K, *K2 = K + NS, *K3 = K + 2 * NS, *K4 = K + 3 * NS;
  double *Kd1 = Kd, *Kd2 = Kd + NS, *Kd3 = Kd + 2 * NS, *Kd4 = Kd + 3 * NS;
  const double ih = 1.0 / ha, hh = had * ih * ih;
  double td, tdx;
  const double ntot = rhs_dual(S, N, Nd, nullptr, svN, sv1N, va, kf, kr, dk, nd, dlnd, td, T, sst, sround, lane);
  for (int s = 0; s < 4; s++) {
    jdot(S, N, Nd, K + s * NS, Kd + s * NS, svN, sv1N, ntot, td, kf, kr, dk, nd, dlnd, t1, t2, va, T, sst, sround,
         lane);
    for (int i = lane; i < NS; i += 32) Kd[s * NS + i] += 2.0 * hh * K[s * NS + i];
  }
  {                                                       // F's tangent at N (its space held n12 so far)
    const double g = ntot / nd, gd = g * (td / ntot - dlnd);
    for (int i = lane; i < NS; i += 32) Fd[i] = sv1N[i] * g + svN[i] * gd;
  }
  for (int i = lane; i < NS; i += 32) Kd1[i] += Fd[i];
  __syncwarp();
  xsolve<DN>(S, Wd, c, c0, Kd1, T.smap, lane, myt);
  for (int i = lane; i < NS; i += 32) Kd2[i] += Fd[i] + 4.0 * ih * Kd1[i] - 4.0 * hh * K1[i];
  __syncwarp();
  xsolve<DN>(S, Wd, c, c0, Kd2, T.smap, lane, myt);
  for (int i = lane; i < NS; i += 32) { Ys[i] = N[i] + 2.0 * K1[i]; Yds[i] = Nd[i] + 2.0 * Kd1[i]; }
  __syncwarp();
  rhs_dual(S, Ys, Yds, Fd, svN, sv1N, va, kf, kr, dk, nd, dlnd, tdx, T, sst, sround, lane);
  for (int i = lane; i < NS; i += 32) Kd3[i] += Fd[i] + ih * (Kd1[i] - Kd2[i]) - hh * (K1[i] - K2[i]);
  __syncwarp();
  xsolve<DN>(S, Wd, c, c0, Kd3, T.smap, lane, myt);
  for (int i = lane; i < NS; i += 32) { Ys[i] += K3[i]; Yds[i] += Kd3[i]; }
  __syncwarp();
  rhs_dual(S, Ys, Yds, Fd, svN, sv1N, va, kf, kr, dk, nd, dlnd, tdx, T, sst, sround, lane);
  for (int i = lane; i < NS; i += 32)
    Kd4[i] += Fd[i] + ih * (Kd1[i] - Kd2[i] - (8.0 / 3.0) * Kd3[i]) - hh * (K1[i] - K2[i] - (8.0 / 3.0) * K3[i]);
  __syncwarp();
  xsolve<DN>(S, Wd, c, c0, Kd4, T.smap, lane, myt);
  for (int i = lane; i < NS; i += 32) Yd[i] = Nd[i] + 2.0 * Kd1[i] + Kd3[i] + Kd4[i];
  __syncwarp();
  if (!adapt || !(en > 0.0)) return 0.0;
  double e = 0.0;
  for (int i = lane; i < NS; i += 32) {
    const double aN = fabs(N[i]), aY = fabs(Y[i]), sc = atol + rtol * fmax(aN, aY);
    const double dN = sgn(N[i]) * Nd[i], dY = sgn(Y[i]) * Yd[i];
    const double dsc = rtol * (aN > aY ? dN : (aN < aY ? dY : 0.5 * (dN + dY)));
    const double q = K4[i] / sc;
    e += q * (Kd4[i] - q * dsc) / sc;
  }
  return wsum(e) / (NS * en);
}

#define JVP_ARGS                                                                                                   \
    const double* __restrict__ x0, const double* __restrict__ Ta, const double* __restrict__ Pa,                         \
    const double* __restrict__ dta, const double* __restrict__ hin, const double* lm, const double* __restrict__ rc,     \
    const double* __restrict__ dx0, const double* __restrict__ dTa, const double* __restrict__ dPa,                      \
    const double* __restrict__ ddta, const double* __restrict__ dhin, const double* dlm, const double* drc, int D,       \
    double* __restrict__ xout, double* __restrict__ hout, int* __restrict__ doneo, double* __restrict__ noko,            \
    double* __restrict__ dxout, double* __restrict__ dhout, int* __restrict__ count, int B, double rtol, double atol,    \
    int max_steps, int adapt, const int* __restrict__ perm, const unsigned* __restrict__ blob,                          \
    const int* __restrict__ rf, const double* __restrict__ nasa, double* scratch, const unsigned* __restrict__ sst,     \
    const u64* __restrict__ jt, const u64* __restrict__ lu, const int* __restrict__ optr, int sround, int jround
#define JVP_PASS x0, Ta, Pa, dta, hin, lm, rc, dx0, dTa, dPa, ddta, dhin, dlm, drc, D, xout, hout, doneo, noko, dxout, \
    dhout, count, B, rtol, atol, max_steps, adapt, perm, blob, rf, nasa, scratch, sst, jt, lu, optr, sround, jround

// The call and its tangents, one warp per cell. D tangent directions share the call: direction d's inputs and outputs
// at [d * B + cell] (dx0, dxout, dlm by rows of NS or NR), drc at [d * NR * 9]; dlm, drc may be null. DN = false: every
// cell, the call kernel's attempts (queue count[0]). DN = true: the first count[6] cells of list, the recover kernel's
// attempts (dense W in wscr, one NS x NS per warp; max_steps = the recovery's tries; queue count[9]), each result then
// tested as the recover kernel tests it: redo = 1 and count[7], or redo = 2, count[8] and NaN tangents.
template <bool DN> __device__ __forceinline__ void jvp_cells(JVP_ARGS, int* redo, const int* list, double* wscr,
    const u16* __restrict__ didx, const double* __restrict__ atoms)
{
  extern __shared__ double smem[];
  unsigned* tb32 = (unsigned*)(smem + WPB * PERWA);
  for (int i = threadIdx.x; i < TBYTES / 4; i += blockDim.x) tb32[i] = blob[i];
  __syncthreads();
  const u8* tb = (const u8*)tb32;
  Tab T;
  T.smap = (const u16*)(tb + O_SMAP); T.rowptr = (const u16*)(tb + O_ROWPTR); T.diag = (const u16*)(tb + O_DIAG);
  T.colof = tb + O_COLOF; T.rx = tb + O_RX; T.xp = tb + O_XP; T.blen = tb + O_BLEN;
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
  const int myt = lane < NT ? lane : -1;
  double c[NT];
  #pragma unroll
  for (int j = 0; j < NT; j++) c[j] = 0.0;
  double* S = smem + wib * PERWA;
  double *N = S + A_NN, *Nd = S + A_ND, *Y = S + A_Y, *Yd = S + A_YD;
  const size_t wid = (size_t)(blockIdx.x * WPB + wib);
  double* kf = scratch + wid * (3 * NR + 2 + D * (2 * NR + NS));
  double *kr = kf + NR, *va = kr + NR + 2, *dk = va + NR, *Ng = dk + 2 * NR * D;   // per direction: d ln k, state tangent
  double* Wd = DN ? wscr + wid * NS * NS : nullptr;
  if (lane == 0) kf[2 * NR] = 0.0;                            // the zero rate of the Jacobian's padding words
  double hd[DMAX], tdot[DMAX], dlnd[DMAX];                    // per direction: h, t, ln nd tangents (warp-uniform)
  const int nc = DN ? count[6] : B;
  int q = blockIdx.x * WPB + wib;
  while (q < nc) {
    const int cell = DN ? list[q] : q;
    const double Tc = Ta[cell], dt = dta[cell], Pc = Pa[cell], nd = Pc / (KBOLTZ * Tc);
    double p = 0.0;
    for (int i = lane; i < NS; i += 32) { N[i] = x0[(size_t)cell * NS + perm[i]]; p += N[i]; }
    for (int d = 0; d < D; d++)
      for (int i = lane; i < NS; i += 32) Ng[d * NS + i] = dx0[((size_t)d * B + cell) * NS + perm[i]];
    const bool fin = isfinite(wsum(p)) && isfinite(Tc) && isfinite(nd) && isfinite(dt), cold = Tc < TMIN;
    if (fin && !cold) rates(S, kf, kr, Tc, nd, lm ? lm + (size_t)cell * NR : nullptr, rc, rf, nasa, T, lane);
    const double hi = hin[cell], m = fmin(hi, dt);
    for (int d = 0; d < D; d++) {
      const size_t qd = (size_t)d * B + cell;
      const double dTc = dTa[qd], ddt = adapt ? ddta[qd] : 0.0, dhi = adapt ? dhin[qd] : 0.0;
      dlnd[d] = dPa[qd] / Pc - dTc / Tc;
      if (fin && !cold)
        rates_tan(S, dk + 2 * NR * d, Tc, nd, dTc, dlnd[d], dlm ? dlm + qd * NR : nullptr, rc,
                  drc ? drc + (size_t)d * NR * 9 : nullptr, rf, nasa, T, lane);
      const double md = hi < dt ? dhi : (hi > dt ? ddt : 0.5 * (dhi + ddt));
      hd[d] = m > 1e-12 ? md : (m < 1e-12 ? 0.0 : 0.5 * md);
      tdot[d] = 0.0;
    }
    __syncwarp();
    double h = fmax(m, 1e-12), t = 0.0, nok = 0.0;
    int it = (dt > 0.0 && fin && !cold) ? 0 : max_steps;
    while (it < max_steps && t < dt * (1.0 - 1e-14)) {
      const double rem = dt - t, ha = fmin(h, rem);
      double c0, dfde;
      bool ok;
      const double en = xattempt<DN>(S, c, Wd, ha, kf, kr, nd, rtol, atol, c0, ok, T, sst, jt, lu, optr, didx, sround,
                                     jround, lane, myt);
      const double fac = factor(en, ok, dfde);
      for (int d = 0; d < D; d++) {                           // each direction's tangent on the attempt in place
        const double ddt = adapt ? ddta[(size_t)d * B + cell] : 0.0;
        const double had = !adapt ? 0.0 : (h < rem ? hd[d] : (h > rem ? ddt - tdot[d] : 0.5 * (hd[d] + ddt - tdot[d])));
        double end = 0.0;
        if (ok || (adapt && dfde != 0.0)) {
          double* ng = Ng + d * NS;
          for (int i = lane; i < NS; i += 32) Nd[i] = ng[i];
          __syncwarp();
          end = tstep<DN>(S, c, Wd, ha, had, c0, en, adapt, kf, kr, dk + 2 * NR * d, va, nd, dlnd[d], rtol, atol, T,
                          sst, sround, lane, myt);
          if (ok) for (int i = lane; i < NS; i += 32) ng[i] = Y[i] >= 0.0 ? Yd[i] : 0.0;
          __syncwarp();
        }
        if (ok) tdot[d] += had;
        hd[d] = adapt ? had * fac + ha * dfde * end : 0.0;
      }
      if (ok) {
        for (int i = lane; i < NS; i += 32) N[i] = fmax(Y[i], 0.0);
        t += ha;
        nok += 1.0;
      }
      __syncwarp();
      h = ha * fac;
      it++;
    }
    p = 0.0;
    for (int i = lane; i < NS; i += 32) p += N[i];
    const double s = wsum(p);
    for (int i = lane; i < NS; i += 32) xout[(size_t)cell * NS + perm[i]] = N[i] / s;
    for (int d = 0; d < D; d++) {
      const double* ng = Ng + d * NS;
      double pd = 0.0;
      for (int i = lane; i < NS; i += 32) pd += ng[i];
      const double sd = wsum(pd);
      for (int i = lane; i < NS; i += 32) dxout[((size_t)d * B + cell) * NS + perm[i]] = (ng[i] - N[i] / s * sd) / s;
      if (lane == 0) dhout[(size_t)d * B + cell] = hd[d];
    }
    const int dn = (t >= dt * (1.0 - 1e-14)) ? 1 : 0;
    if (lane == 0) { hout[cell] = h; doneo[cell] = dn; noko[cell] = nok; }
    if (DN) {                                                 // the recover kernel's test of the result
      __syncwarp();
      int bad = 0;
      if (lane == 0) {
        bad = check_cell(x0, Ta, Pa, dta, xout, dn, nok, atoms, atol, cell);
        redo[cell] = bad ? 2 : 1;
        atomicAdd(count + (bad ? 8 : 7), 1);
      }
      if (__shfl_sync(FULL, bad, 0))
        for (int d = 0; d < D; d++) {
          for (int i = lane; i < NS; i += 32) dxout[((size_t)d * B + cell) * NS + i] = DNAN;
          if (lane == 0) dhout[(size_t)d * B + cell] = DNAN;
        }
    }
    int nxt = 0;
    if (lane == 0) nxt = gridDim.x * WPB + atomicAdd(count + (DN ? 9 : 0), 1);
    q = __shfl_sync(FULL, nxt, 0);
    __syncwarp();
  }
}

extern "C" __global__ void __launch_bounds__(32 * WPB) jvp(JVP_ARGS) {
  jvp_cells<false>(JVP_PASS, nullptr, nullptr, nullptr, nullptr, nullptr);
}

// After jvp and the guard: the listed cells redone on the recovery's path with their tangents (redo, list: the guard's).
extern "C" __global__ void __launch_bounds__(32 * WPB) jvp_rec(JVP_ARGS, int* __restrict__ redo,
    const int* __restrict__ list, double* __restrict__ wscr, const u16* __restrict__ didx,
    const double* __restrict__ atoms) {
  if ((int)(blockIdx.x * WPB) >= count[6]) return;            // no listed cell for this block
  jvp_cells<true>(JVP_PASS, redo, list, wscr, didx, atoms);
}
#endif

#if KIND == 1
// ---------------------------------------------------------------------------------------------------------- adjoint
// y <- W^-T y = M0^-T (y + c0 (u . y) 1) (W = M0 - w 1^T, u = M0^-1 w): U^T then L^T with every factor read from M (the
// dense block's factors written back), lane l holding entries l and K0 + l.
__device__ __forceinline__ void tsolve(const double* M, const double* dinv, double* y, const double* u, double c0,
                                       const u16* smap, int lane) {
  const bool hp = lane < K0, ht = lane < NT;
  double a0 = hp ? y[lane] : 0.0, a1 = ht ? y[K0 + lane] : 0.0;
  const double al = c0 * wsum((hp ? u[lane] * a0 : 0.0) + (ht ? u[K0 + lane] * a1 : 0.0));
  if (hp) a0 += al;
  if (ht) a1 += al;
  const double d0 = hp ? dinv[lane] : 0.0, d1 = ht ? dinv[K0 + lane] : 0.0;
  #pragma unroll
  for (int k = 0; k < K0; k++) {                        // U^T, sparse rows: y_j -= U_kj w_k (j > k)
    if (!((UROW >> k) & 1ull)) continue;
    const double wk = __shfl_sync(FULL, a0 * d0, k);
    if (hp && lane > k) { const unsigned s = smap[k * NS + lane]; if (s != 0xFFFFu) a0 -= M[s] * wk; }
    if (ht) { const unsigned s = smap[k * NS + K0 + lane]; if (s != 0xFFFFu) a1 -= M[s] * wk; }
  }
  #pragma unroll
  for (int k = 0; k < NT - 1; k++) {                    // U^T, dense rows
    const double wk = __shfl_sync(FULL, a1 * d1, k);
    if (ht && lane > k) a1 -= M[smap[(K0 + k) * NS + K0 + lane]] * wk;
  }
  a0 *= d0;
  a1 *= d1;
  #pragma unroll
  for (int k = NT - 1; k >= 0; k--) {                   // L^T, dense rows: w_j -= L_kj x_k (j < k)
    const double xk = __shfl_sync(FULL, a1, k);
    if (ht && lane < k) a1 -= M[smap[(K0 + k) * NS + K0 + lane]] * xk;
    if (hp) { const unsigned s = smap[(K0 + k) * NS + lane]; if (s != 0xFFFFu) a0 -= M[s] * xk; }
  }
  #pragma unroll
  for (int k = K0 - 1; k >= 0; k--) {                   // L^T, sparse rows
    if (!((LROW >> k) & 1ull)) continue;
    const double xk = __shfl_sync(FULL, a0, k);
    if (hp && lane < k) { const unsigned s = smap[k * NS + lane]; if (s != 0xFFFFu) a0 -= M[s] * xk; }
  }
  __syncwarp();
  if (hp) y[lane] = a0;
  if (ht) y[K0 + lane] = a1;
  __syncwarp();
}

// y <- W^-T y with dense_lu's factors (W = P^T L U, P the row order pr): U^T, then L^T, then the rows back to their
// original order. Lane l holds rows l + 32 p.
__device__ __forceinline__ void dense_tsolve(const double* A, const double* dinv, const int* pr, double* y, int lane) {
  double a[RPD];
  #pragma unroll
  for (int p = 0; p < RPD; p++) a[p] = lane + 32 * p < NS ? y[lane + 32 * p] : 0.0;
  #pragma unroll
  for (int k = 0; k < NS - 1; k++) {                   // U^T: y_i -= U_ki w_k (i > k), w_k = y_k / U_kk
    const double wk = __shfl_sync(FULL, a[k >> 5] * dinv[k], k & 31);
    #pragma unroll
    for (int p = k >> 5; p < RPD; p++) {
      const int i = lane + 32 * p;
      if (i > k && i < NS) a[p] -= A[k * NS + i] * wk;
    }
  }
  #pragma unroll
  for (int p = 0; p < RPD; p++) if (lane + 32 * p < NS) a[p] *= dinv[lane + 32 * p];
  #pragma unroll
  for (int k = NS - 1; k > 0; k--) {                   // unit L^T: y_i -= L_ki y_k (i < k)
    const double xk = __shfl_sync(FULL, a[k >> 5], k & 31);
    #pragma unroll
    for (int p = 0; p <= (k >> 5); p++) {
      const int i = lane + 32 * p;
      if (i < k) a[p] -= A[k * NS + i] * xk;
    }
  }
  __syncwarp();
  #pragma unroll
  for (int p = 0; p < RPD; p++) if (lane + 32 * p < NS) y[pr[lane + 32 * p]] = a[p];
  __syncwarp();
}

// y <- W^-T y with the attempt's factors (xsolve's two cases).
template <bool DN> __device__ __forceinline__ void xtsolve(double* S, const double* Wd, double c0, double* y,
                                                           const u16* smap, int lane) {
  if (DN) dense_tsolve(Wd, S + A_DINV, (const int*)(S + A_M), y, lane);
  else tsolve(S + A_M, S + A_DINV, y, S + A_U, c0, smap, lane);
}

// Reverse of F = (n_tot / nd) S v(n), n = nd Y / n_tot, at Y for the cotangent Fb (rate coefficients fixed): Yb +=
// J(Y)^T Fb; lbk (ln k common to kf and kr; ln kr alone) and lbnd (ln nd) += their cotangents. The sums of n_q times
// the species cotangents come reaction by reaction (a product of m factors: sum over its slots = m times the product).
__device__ __forceinline__ void rhs_vjp(double* S, const double* Y, const double* Fb, double* Yb, double* lbk,
    double* rho, double& lbnd, const double* kf, const double* kr, double nd, const Tab& T,
    const u64* __restrict__ gt, const int* __restrict__ xg, int ground, int lane)
{
  double *n = S + A_N0, *part = S + A_PART;
  double p = 0.0;
  for (int i = lane; i < NS; i += 32) p += Y[i];
  const double ntot = wsum(p), xf = nd / ntot, g = ntot / nd;
  for (int i = lane; i < NS; i += 32) n[i] = Y[i] * xf;
  if (lane == 0) n[NS] = 1.0;
  for (int i = lane; i < NPG; i += 32) part[i] = 0.0;
  __syncwarp();
  double ff = 0.0, tl = 0.0;
  for (int r = lane; r < NR; r += 32) {
    const u8* q = T.rx + 6 * r;
    double sb = 0.0, mre = 0.0, mpr = 0.0;
    #pragma unroll
    for (int m = 0; m < 3; m++) {
      if (q[m] < NS) { sb -= Fb[q[m]]; mre += 1.0; }
      if (q[3 + m] < NS) { sb += Fb[q[3 + m]]; mpr += 1.0; }
    }
    const double vb = sb * g;
    const double a = kf[r] * (n[q[0]] * n[q[1]] * n[q[2]]), b = kr[r] * (n[q[3]] * n[q[4]] * n[q[5]]), ab = a - b;
    ff += vb * ab;
    tl += vb * (mre * a - mpr * b);
    rho[r] = vb * kf[r];
    rho[NR + r] = -vb * kr[r];
    lbk[r] += vb * ab;
    lbk[NR + r] -= vb * b;
  }
  __syncwarp();
  const double FF = wsum(ff), tau = wsum(tl);
  {
    double acc = 0.0;
    for (int qq = 0; qq < ground; qq++) {
      const u64 w = __ldg(gt + qq * 32 + lane);
      if ((w >> 37) & 1ull) {
        acc = fma(rho[w & 1023ull], n[(w >> 12) & 63ull] * n[(w >> 18) & 63ull], acc);
        if ((w >> 35) & 1ull) { part[(w >> 24) & 127ull] = acc; acc = 0.0; }
      }
    }
  }
  __syncwarp();
  for (int i = lane; i < NS; i += 32) {
    double s = part[i];
    for (int e = xg[i]; e < xg[i + 1]; e++) s += part[NS + e];
    Yb[i] += xf * s + (FF - tau) / ntot;
  }
  lbnd += tau - FF;
  __syncwarp();
}

// The reverse at N of F(N) for the cotangent z1 + z2 and of the four stages' Jacobian terms: Nb += J(N)^T (z1 + z2) +
// sum_i d/ds [J(N + s K_i)^T z_i] (z_i = W^-T of stage i's cotangent: z1..z3 in Kb1..Kb3, z4 in z), and the matching
// rate cotangents, in one pass over the reactions and one gather (n1_i = the concentrations' derivative along K_i).
__device__ __forceinline__ void rhs_vjp_n(double* S, double* lbk, double* rho, double& lbnd, const double* kf,
    const double* kr, double nd, const Tab& T, const u64* __restrict__ gt, const int* __restrict__ xg, int ground,
    int lane)
{
  constexpr int OZ[4] = {A_KB1, A_KB2, A_KB3, A_Z}, ON1[4] = {A_N1, A_YB, A_YB2, A_NBD};
  double *N = S + A_NN, *Nb = S + A_NB, *n = S + A_N0, *part = S + A_PART;
  const double* K = S + A_K1;
  double p = 0.0, pd[4] = {0.0, 0.0, 0.0, 0.0};
  for (int i = lane; i < NS; i += 32) {
    p += N[i];
    #pragma unroll
    for (int s = 0; s < 4; s++) pd[s] += K[s * NS + i];
  }
  const double ntot = wsum(p), xf = nd / ntot, g = ntot / nd;
  double rt[4];
  #pragma unroll
  for (int s = 0; s < 4; s++) rt[s] = wsum(pd[s]) / ntot;
  for (int i = lane; i < NS; i += 32) {
    n[i] = N[i] * xf;
    #pragma unroll
    for (int s = 0; s < 4; s++) S[ON1[s] + i] = xf * (K[s * NS + i] - N[i] * rt[s]);
  }
  if (lane == 0) {
    n[NS] = 1.0;
    #pragma unroll
    for (int s = 0; s < 4; s++) S[ON1[s] + NS] = 0.0;
  }
  for (int i = lane; i < NPG; i += 32) part[i] = 0.0;
  __syncwarp();
  double ffF = 0.0, tF = 0.0, ff[4], ffd[4], tl[4], tdl[4];
  #pragma unroll
  for (int s = 0; s < 4; s++) { ff[s] = 0.0; ffd[s] = 0.0; tl[s] = 0.0; tdl[s] = 0.0; }
  for (int r = lane; r < NR; r += 32) {
    const u8* q = T.rx + 6 * r;
    double mre = 0.0, mpr = 0.0;
    #pragma unroll
    for (int m = 0; m < 3; m++) { mre += q[m] < NS ? 1.0 : 0.0; mpr += q[3 + m] < NS ? 1.0 : 0.0; }
    const double n0 = n[q[0]], n1 = n[q[1]], n2 = n[q[2]], n3 = n[q[3]], n4 = n[q[4]], n5 = n[q[5]];
    const double a = kf[r] * (n0 * n1 * n2), b = kr[r] * (n3 * n4 * n5), ab = a - b;
    double vbF = 0.0, lm = 0.0, lr = 0.0;
    #pragma unroll
    for (int s = 0; s < 4; s++) {
      const double *z = S + OZ[s], *d = S + ON1[s];
      double sb = 0.0;
      #pragma unroll
      for (int m = 0; m < 3; m++) {
        if (q[m] < NS) sb -= z[q[m]];
        if (q[3 + m] < NS) sb += z[q[3 + m]];
      }
      const double vb = sb * g;
      const double Pd = d[q[0]] * n1 * n2 + n0 * d[q[1]] * n2 + n0 * n1 * d[q[2]];
      const double Qd = d[q[3]] * n4 * n5 + n3 * d[q[4]] * n5 + n3 * n4 * d[q[5]];
      const double ad = kf[r] * Pd, bd = kr[r] * Qd;
      lm += vb * rt[s] * ab + vb * (ad - bd);
      lr -= vb * rt[s] * b + vb * bd;
      ff[s] += vb * ab;
      ffd[s] += vb * (ad - bd);
      tl[s] += vb * (mre * a - mpr * b);
      tdl[s] += vb * ((mre - 1.0) * ad - (mpr - 1.0) * bd);
      rho[s * 2 * NR + r] = vb * kf[r];
      rho[s * 2 * NR + NR + r] = -vb * kr[r];
      if (s < 2) vbF += vb;
    }
    lm += vbF * ab;
    lr -= vbF * b;
    ffF += vbF * ab;
    tF += vbF * (mre * a - mpr * b);
    lbk[r] += lm;
    lbk[NR + r] += lr;
  }
  __syncwarp();
  const double FF_F = wsum(ffF), tau_F = wsum(tF);
  double sc = (FF_F - tau_F) / ntot, lb = tau_F - FF_F;
  #pragma unroll
  for (int s = 0; s < 4; s++) {
    const double FF = wsum(ff[s]), fd = wsum(ffd[s]), tau = wsum(tl[s]);
    const double FFd = fd + FF * rt[s], taud = rt[s] * tau + wsum(tdl[s]) + fd;
    sc += (FFd - taud) / ntot - (FF - tau) * rt[s] / ntot;
    lb += taud - FFd;
  }
  {
    double acc = 0.0;
    for (int qq = 0; qq < ground; qq++) {
      const u64 w = __ldg(gt + qq * 32 + lane);
      if ((w >> 37) & 1ull) {
        const int k = (int)(w & 1023ull), ia = (int)((w >> 12) & 63ull), ib = (int)((w >> 18) & 63ull);
        const double na = n[ia], nb = n[ib], r0 = rho[k], r1 = rho[2 * NR + k];
        acc = fma(r0 + r1, na * nb, acc);
        #pragma unroll
        for (int s = 0; s < 4; s++) {
          const double* d = S + ON1[s];
          acc = fma(s == 0 ? r0 : (s == 1 ? r1 : rho[s * 2 * NR + k]), d[ia] * nb + na * d[ib], acc);
        }
        if ((w >> 35) & 1ull) { part[(w >> 24) & 127ull] = acc; acc = 0.0; }
      }
    }
  }
  __syncwarp();
  for (int i = lane; i < NS; i += 32) {
    double s = part[i];
    for (int e = xg[i]; e < xg[i + 1]; e++) s += part[NS + e];
    Nb[i] += xf * s + sc;
  }
  lbnd += lb;
  __syncwarp();
}

// Reverse of the stages of the attempt in place (K1..K4, the full LU in M, dinv, u, c0; DN: the recovery's dense factors)
// for the cotangents Yb of the new state and z of K4: Nb += the state's cotangent, lbk, lbnd += the rate cotangents;
// returns the cotangent of the step size (adapt) or 0. The stages' W^-T solves come first (z4 in z, z3..z1 in place of
// Kb3..Kb1), with the reverse of F at Y4 and Y3 between them; the terms at N follow in one pass.
template <bool DN> __device__ __forceinline__ double sadj(double* S, const double* Wd, double ha, double c0, int adapt,
    const double* kf, const double* kr, double* lbk, double* rho, double& lbnd, double nd, const Tab& T,
    const u64* __restrict__ gt, const int* __restrict__ xg, int ground, int lane)
{
  double *N = S + A_NN, *Nb = S + A_NB, *Yb = S + A_YB, *Yb2 = S + A_YB2, *z = S + A_Z, *Ys = S + A_YS, *K1 = S + A_K1,
         *K2 = S + A_K2, *K3 = S + A_K3, *K4 = S + A_K4, *Kb1 = S + A_KB1, *Kb2 = S + A_KB2, *Kb3 = S + A_KB3;
  const double ih = 1.0 / ha;
  double ihb = 0.0, ighb = 0.0;
  for (int i = lane; i < NS; i += 32) {
    const double yb = Yb[i];
    Nb[i] += yb; Kb1[i] = 2.0 * yb; Kb2[i] = 0.0; Kb3[i] = yb; z[i] += yb;
  }
  __syncwarp();
  // stage 4: K4 = W^-1 (F(Y4) + (K1 - K2 - 8/3 K3) / h), Y4 = N + 2 K1 + K3
  xtsolve<DN>(S, Wd, c0, z, T.smap, lane);
  for (int i = lane; i < NS; i += 32) {
    Kb1[i] += ih * z[i]; Kb2[i] -= ih * z[i]; Kb3[i] -= (8.0 / 3.0) * ih * z[i];
    ihb += z[i] * (K1[i] - K2[i] - (8.0 / 3.0) * K3[i]);
    ighb -= z[i] * K4[i];
    Ys[i] = N[i] + 2.0 * K1[i] + K3[i];
    Yb2[i] = 0.0;
  }
  __syncwarp();
  rhs_vjp(S, Ys, z, Yb2, lbk, rho, lbnd, kf, kr, nd, T, gt, xg, ground, lane);
  for (int i = lane; i < NS; i += 32) { Nb[i] += Yb2[i]; Kb1[i] += 2.0 * Yb2[i]; Kb3[i] += Yb2[i]; }
  __syncwarp();
  // stage 3: K3 = W^-1 (F(Y3) + (K1 - K2) / h), Y3 = N + 2 K1
  xtsolve<DN>(S, Wd, c0, Kb3, T.smap, lane);
  for (int i = lane; i < NS; i += 32) {
    Kb1[i] += ih * Kb3[i]; Kb2[i] -= ih * Kb3[i];
    ihb += Kb3[i] * (K1[i] - K2[i]);
    ighb -= Kb3[i] * K3[i];
    Ys[i] = N[i] + 2.0 * K1[i];
    Yb2[i] = 0.0;
  }
  __syncwarp();
  rhs_vjp(S, Ys, Kb3, Yb2, lbk, rho, lbnd, kf, kr, nd, T, gt, xg, ground, lane);
  for (int i = lane; i < NS; i += 32) { Nb[i] += Yb2[i]; Kb1[i] += 2.0 * Yb2[i]; }
  __syncwarp();
  // stage 2: K2 = W^-1 (F(N) + 4 K1 / h)
  xtsolve<DN>(S, Wd, c0, Kb2, T.smap, lane);
  for (int i = lane; i < NS; i += 32) {
    Kb1[i] += 4.0 * ih * Kb2[i];
    ihb += 4.0 * Kb2[i] * K1[i];
    ighb -= Kb2[i] * K2[i];
  }
  __syncwarp();
  // stage 1: K1 = W^-1 F(N)
  xtsolve<DN>(S, Wd, c0, Kb1, T.smap, lane);
  for (int i = lane; i < NS; i += 32) ighb -= Kb1[i] * K1[i];
  __syncwarp();
  rhs_vjp_n(S, lbk, rho, lbnd, kf, kr, nd, T, gt, xg, ground, lane);
  if (!adapt) return 0.0;
  return -(wsum(ihb) + 2.0 * wsum(ighb)) / (ha * ha);
}

__device__ __forceinline__ void save(double* rp, const double* N, double h, double t, int lane) {
  for (int i = lane; i < NS; i += 32) rp[i] = N[i];
  if (lane == 0) { rp[NS] = h; rp[NS + 1] = t; }
}

#define VJP_ARGS                                                                                                   \
    const double* __restrict__ x0, const double* __restrict__ Ta, const double* __restrict__ Pa,                         \
    const double* __restrict__ dta, const double* __restrict__ hin, const double* lm, const double* __restrict__ rc,     \
    const double* __restrict__ gx, const double* gh,                                                                    \
    double* __restrict__ xout, double* __restrict__ hout, int* __restrict__ doneo, double* __restrict__ noko,            \
    double* __restrict__ gx0, double* __restrict__ gT, double* __restrict__ gP, double* __restrict__ gdt,               \
    double* __restrict__ ghin, double* __restrict__ glm, int* __restrict__ count, int B, double rtol, double atol,      \
    int max_steps, int adapt, const int* __restrict__ perm, const unsigned* __restrict__ blob,                          \
    const int* __restrict__ rf, const double* __restrict__ nasa, double* scratch, double* rec,                          \
    const unsigned* __restrict__ sst, const u64* __restrict__ jt, const u64* __restrict__ lu,                           \
    const int* __restrict__ optr, const u64* __restrict__ gt, const int* __restrict__ xg, int sround, int jround,       \
    int ground
#define VJP_PASS x0, Ta, Pa, dta, hin, lm, rc, gx, gh, xout, hout, doneo, noko, gx0, gT, gP, gdt, ghin, glm, count, B, \
    rtol, atol, max_steps, adapt, perm, blob, rf, nasa, scratch, rec, sst, jt, lu, optr, gt, xg, sround, jround, ground

// The call and the cotangents of its inputs, one warp per cell. DN = false: every cell, the call kernel's attempts
// (queue count[0]; CAP stored attempts per cell in rec). DN = true: the first count[6] cells of list, the recover
// kernel's attempts (dense W in wscr; max_steps = the recovery's tries; CAPR stored attempts; queue count[9]); its
// result tested as the recover kernel tests it: redo = 1 and count[7], or redo = 2, count[8] and NaN cotangents.
template <bool DN> __device__ __forceinline__ void vjp_cells(VJP_ARGS, int* redo, const int* list, double* wscr,
    const u16* __restrict__ didx, const double* __restrict__ atoms)
{
  constexpr int CP = DN ? CAPR : CAP;
  extern __shared__ double smem[];
  unsigned* tb32 = (unsigned*)(smem + WPB * PERWA);
  for (int i = threadIdx.x; i < TBYTES / 4; i += blockDim.x) tb32[i] = blob[i];
  __syncthreads();
  const u8* tb = (const u8*)tb32;
  Tab T;
  T.smap = (const u16*)(tb + O_SMAP); T.rowptr = (const u16*)(tb + O_ROWPTR); T.diag = (const u16*)(tb + O_DIAG);
  T.colof = tb + O_COLOF; T.rx = tb + O_RX; T.xp = tb + O_XP; T.blen = tb + O_BLEN;
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
  const int myt = lane < NT ? lane : -1;
  double c[NT];
  #pragma unroll
  for (int j = 0; j < NT; j++) c[j] = 0.0;
  double* S = smem + wib * PERWA;
  double *M = S + A_M, *N = S + A_NN, *Y = S + A_Y, *Nb = S + A_NB, *Yb = S + A_YB, *z = S + A_Z, *K4 = S + A_K4;
  const size_t wid = (size_t)(blockIdx.x * WPB + wib);
  double* kf = scratch + wid * (12 * NR + 2);
  double *kr = kf + NR, *lbk = kr + NR + 2, *rho = lbk + 2 * NR;    // rho: 4 x 2 NR
  if (lane == 0) kf[2 * NR] = 0.0;                            // the zero rate of the Jacobian's padding words
  double* recA = rec + wid * 2 * CP * REC;
  double* recB = recA + CP * REC;
  double* Wd = DN ? wscr + wid * NS * NS : nullptr;
  const int nc = DN ? count[6] : B;
  int q = blockIdx.x * WPB + wib;
  while (q < nc) {
    const int cell = DN ? list[q] : q;
    const double Tc = Ta[cell], dt = dta[cell], Pc = Pa[cell], nd = Pc / (KBOLTZ * Tc);
    const double* lmc = lm ? lm + (size_t)cell * NR : nullptr;
    double p = 0.0;
    for (int i = lane; i < NS; i += 32) { N[i] = x0[(size_t)cell * NS + perm[i]]; p += N[i]; }
    const bool fin = isfinite(wsum(p)) && isfinite(Tc) && isfinite(nd) && isfinite(dt), cold = Tc < TMIN;
    if (fin && !cold) rates(S, kf, kr, Tc, nd, lmc, rc, rf, nasa, T, lane);
    __syncwarp();
    const double hi = hin[cell], m0 = fmin(hi, dt), h0 = fmax(m0, 1e-12);
    double h = h0, t = 0.0, nok = 0.0;
    int na = 0;
    const bool run = dt > 0.0 && fin && !cold;
    double en_l = 0.0, c0_l = 0.0;                            // the last attempt run (its stages are in place)
    // forward: the state before each of the first CP attempts (and whether it was accepted)
    while (run && na < max_steps && t < dt * (1.0 - 1e-14)) {
      if (na < CP) save(recA + na * REC, N, h, t, lane);
      const bool ok = fstep<DN>(S, c, Wd, h, t, nok, dt, kf, kr, nd, rtol, atol, en_l, c0_l, T, sst, jt, lu, optr,
                                didx, sround, jround, lane, myt);
      if (na < CP && lane == 0) recA[na * REC + NS + 2] = ok ? 1.0 : 0.0;
      na++;
    }
    p = 0.0;
    for (int i = lane; i < NS; i += 32) p += N[i];
    const double s = wsum(p);
    double gs = 0.0;
    for (int i = lane; i < NS; i += 32) {
      xout[(size_t)cell * NS + perm[i]] = N[i] / s;
      gs += gx[(size_t)cell * NS + perm[i]] * N[i];
    }
    gs = wsum(gs);
    const int dn = (t >= dt * (1.0 - 1e-14)) ? 1 : 0;
    if (lane == 0) { hout[cell] = h; doneo[cell] = dn; noko[cell] = nok; }
    bool bad = false;
    if (DN) {                                                 // the recover kernel's test of the result
      __syncwarp();
      int b = 0;
      if (lane == 0) {
        b = check_cell(x0, Ta, Pa, dta, xout, dn, nok, atoms, atol, cell);
        redo[cell] = b ? 2 : 1;
        atomicAdd(count + (b ? 8 : 7), 1);
      }
      bad = __shfl_sync(FULL, b, 0) != 0;
      if (bad) na = 0;                                        // no reverse: NaN cotangents below
    }
    // reverse: x = N / sum N
    for (int i = lane; i < NS; i += 32) Nb[i] = gx[(size_t)cell * NS + perm[i]] / s - gs / (s * s);
    for (int r = lane; r < 2 * NR; r += 32) lbk[r] = 0.0;
    __syncwarp();
    double hb = (adapt && gh) ? gh[cell] : 0.0, tb = 0.0, dtb = 0.0, lbnd = 0.0;
    const int st = na <= CP ? 1 : (na + CP - 1) / CP;         // attempts per stored segment
    if (st > 1) {                                             // checkpoints every st attempts
      for (int i = lane; i < NS; i += 32) N[i] = x0[(size_t)cell * NS + perm[i]];
      __syncwarp();
      h = h0; t = 0.0; nok = 0.0;
      for (int j = 0; j < na; j++) {
        if (j % st == 0) save(recA + (j / st) * REC, N, h, t, lane);
        fstep<DN>(S, c, Wd, h, t, nok, dt, kf, kr, nd, rtol, atol, en_l, c0_l, T, sst, jt, lu, optr, didx, sround,
                  jround, lane, myt);
      }
    }
    const int seg = st > 1 ? st : na;                         // attempts per reversed segment
    for (int g = na > 0 ? (na - 1) / seg : -1; g >= 0; g--) {
      const int j0 = g * seg, j1 = min(na, j0 + seg);
      double* rb = recA;
      if (st > 1) {                                           // the segment's attempts, recomputed and stored
        __syncwarp();
        for (int i = lane; i < NS; i += 32) N[i] = recA[g * REC + i];
        h = recA[g * REC + NS]; t = recA[g * REC + NS + 1];
        __syncwarp();
        for (int j = j0; j < j1; j++) {
          save(recB + (j - j0) * REC, N, h, t, lane);
          const bool ok = fstep<DN>(S, c, Wd, h, t, nok, dt, kf, kr, nd, rtol, atol, en_l, c0_l, T, sst, jt, lu, optr,
                                    didx, sround, jround, lane, myt);
          if (lane == 0) recB[(j - j0) * REC + NS + 2] = ok ? 1.0 : 0.0;
        }
        rb = recB - j0 * REC;
      }
      for (int j = j1 - 1; j >= j0; j--) {
        __syncwarp();
        const double* rp = rb + j * REC;
        if (!adapt && rp[NS + 2] == 0.0) continue;            // frozen steps: a rejected attempt has no effect
        for (int i = lane; i < NS; i += 32) N[i] = rp[i];
        const double hj = rp[NS], tj = rp[NS + 1], rem = dt - tj, ha = fmin(hj, rem);
        __syncwarp();
        double c0 = c0_l, en = en_l, dfde;
        bool ok = rp[NS + 2] != 0.0;
        if (j < j1 - 1)                                       // else the attempt just run, still in place
          en = xattempt<DN>(S, c, Wd, ha, kf, kr, nd, rtol, atol, c0, ok, T, sst, jt, lu, optr, didx, sround, jround,
                            lane, myt);
        if (!DN && myt >= 0) {                                // the dense block's factors back into M
          #pragma unroll
          for (int k = 0; k < NT; k++) M[TB0 + myt * NT + k] = c[k];
        }
        const double fac = factor(en, ok, dfde);
        double hab = adapt ? hb * fac : 0.0;
        const double enb = adapt ? hb * ha * dfde : 0.0;
        for (int i = lane; i < NS; i += 32) {
          if (ok) { Yb[i] = Y[i] >= 0.0 ? Nb[i] : 0.0; Nb[i] = 0.0; }
          else Yb[i] = 0.0;
          z[i] = 0.0;
        }
        if (ok && adapt) hab += tb;
        if (enb != 0.0 && en > 0.0) {         // en = sqrt(mean (K4 / sc)^2), sc = atol + rtol max(|N|, |Y|)
          for (int i = lane; i < NS; i += 32) {
            const double aN = fabs(N[i]), aY = fabs(Y[i]), sc = atol + rtol * fmax(aN, aY);
            const double qq = K4[i] / sc, qb = enb * qq / (NS * en), scb = -rtol * qb * qq / sc;
            z[i] = qb / sc;
            const double wN = aN > aY ? 1.0 : (aN < aY ? 0.0 : 0.5);
            Nb[i] += wN * scb * sgn(N[i]);
            Yb[i] += (1.0 - wN) * scb * sgn(Y[i]);
          }
        }
        __syncwarp();
        if (ok || enb != 0.0)
          hab += sadj<DN>(S, Wd, ha, c0, adapt, kf, kr, lbk, rho, lbnd, nd, T, gt, xg, ground, lane);
        if (adapt) {                                          // ha = min(h, dt - t)
          const double wh = hj < rem ? 1.0 : (hj > rem ? 0.0 : 0.5);
          hb = wh * hab;
          dtb += (1.0 - wh) * hab;
          tb -= (1.0 - wh) * hab;
        } else hb = 0.0;
      }
    }
    __syncwarp();
    // h0 = max(min(hin, dt), 1e-12)
    {
      const double wm = m0 > 1e-12 ? 1.0 : (m0 < 1e-12 ? 0.0 : 0.5), wh = hi < dt ? 1.0 : (hi > dt ? 0.0 : 0.5);
      const double mb = run ? wm * hb : 0.0;
      if (lane == 0) { ghin[cell] = wh * mb; gdt[cell] = dtb + (1.0 - wh) * mb; }
    }
    for (int i = lane; i < NS; i += 32) gx0[(size_t)cell * NS + perm[i]] = Nb[i];
    // rate cotangents -> T, P and the log multipliers
    double gTl = 0.0, gnl = 0.0;
    if (run) {
      const double lT = log(Tc), iT = 1.0 / Tc;
      double* gTs = S + A_N1;
      gibbs_dT(gTs, Tc, nasa, lane);
      for (int r = lane; r < NR; r += 32) {
        const double lm = lbk[r], lr = lbk[NR + r];  // ln k (forward and reverse together), ln kr alone
        glm[(size_t)cell * NR + r] = lm;
        const int fl = rf[r];
        double dT, dn, D[9];
        kpart(rc + 9 * r, fl, lT, iT, nd, dT, dn, D);
        const u8* x = T.rx + 6 * r;
        const double ddg = gTs[x[3]] + gTs[x[4]] + gTs[x[5]] - gTs[x[0]] - gTs[x[1]] - gTs[x[2]];
        gTl += lm * dT + lr * (ddg + (double)(((fl >> 3) & 7) - 4) * iT);
        gnl += lm * dn;
      }
    } else {
      for (int r = lane; r < NR; r += 32) glm[(size_t)cell * NR + r] = 0.0;
    }
    const double gTt = wsum(gTl), lnd = wsum(gnl) + lbnd;
    if (lane == 0) { gT[cell] = run ? gTt - lnd / Tc : 0.0; gP[cell] = run ? lnd / Pc : 0.0; }
    if (DN && bad) {                                          // still failed after the recovery
      for (int i = lane; i < NS; i += 32) gx0[(size_t)cell * NS + i] = DNAN;
      for (int r = lane; r < NR; r += 32) glm[(size_t)cell * NR + r] = DNAN;
      if (lane == 0) { gT[cell] = DNAN; gP[cell] = DNAN; gdt[cell] = DNAN; ghin[cell] = DNAN; }
    }
    int nxt = 0;
    if (lane == 0) nxt = gridDim.x * WPB + atomicAdd(count + (DN ? 9 : 0), 1);
    q = __shfl_sync(FULL, nxt, 0);
    __syncwarp();
  }
}

extern "C" __global__ void __launch_bounds__(32 * WPB) vjp(VJP_ARGS) {
  vjp_cells<false>(VJP_PASS, nullptr, nullptr, nullptr, nullptr, nullptr);
}

// After vjp and the guard: the listed cells redone on the recovery's path with their cotangents (redo, list: the
// guard's; rec holds CAPR attempts per warp).
extern "C" __global__ void __launch_bounds__(32 * WPB) vjp_rec(VJP_ARGS, int* __restrict__ redo,
    const int* __restrict__ list, double* __restrict__ wscr, const u16* __restrict__ didx,
    const double* __restrict__ atoms) {
  if ((int)(blockIdx.x * WPB) >= count[6]) return;            // no listed cell for this block
  vjp_cells<true>(VJP_PASS, redo, list, wscr, didx, atoms);
}

// Cotangent of the Arrhenius table: grc[r, j] = sum over cells of glm[cell, r] d ln k_r / d rc[r, j], one block per
// reaction, a fixed summation order.
extern "C" __global__ void arrh(const double* __restrict__ glm, const double* __restrict__ Ta,
    const double* __restrict__ Pa, const double* __restrict__ rc, const int* __restrict__ rf, double* __restrict__ grc,
    int B)
{
  __shared__ double red[9][256];
  const int r = blockIdx.x, tid = threadIdx.x, fl = rf[r];
  double acc[9];
  #pragma unroll
  for (int j = 0; j < 9; j++) acc[j] = 0.0;
  for (int cell = tid; cell < B; cell += 256) {
    const double g = glm[(size_t)cell * NR + r];
    if (g == 0.0) continue;
    const double Tc = Ta[cell], nd = Pa[cell] / (KBOLTZ * Tc);
    double dT, dn, D[9];
    kpart(rc + 9 * r, fl, log(Tc), 1.0 / Tc, nd, dT, dn, D);
    #pragma unroll
    for (int j = 0; j < 9; j++) acc[j] += g * D[j];
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
  if (tid < 9) grc[r * 9 + tid] = red[tid][0];
}
#endif
"""


def source_ad(tab, ext, kind, wpb, cap=CAP):
    """CUDA source of the tangent (kind "jvp") or adjoint (kind "vjp": vjp and arrh) kernels, after the call kernel's
    source whose helpers, stream layout and descriptor widths they use."""
    off, perwa = layout(tab, ext, kind)
    w = tab["wd"]
    m = lambda b: (1 << b) - 1  # noqa: E731
    j_a = w["k"] + 2
    return KC.source(tab, 1) + string.Template(_SRC_AD).substitute(
        wpb=wpb, perwa=perwa, cap=cap, dmax=DMAX, npg=ext["npg"], urow=f"{ext['urow']}ull", lrow=f"{ext['lrow']}ull",
        offsets="\n".join([f"#define KIND {int(kind == 'vjp')}"] + [f"#define A_{k} {v}" for k, v in off.items()]),
        sv_r=m(w["r"]), sv_sg=w["r"], sv_x2=w["r"] + 1, sv_dst=w["r"] + 2, sv_pm=m(w["part"]),
        sv_end=w["r"] + 2 + w["part"], j_km=m(w["k"]), j_sg=w["k"], j_x2=w["k"] + 1, j_a=j_a, j_b=j_a + w["s"],
        j_sm=m(w["s"]), j_end=j_a + 2 * w["s"], slm=m(w["slot"]), lu_b=w["slot"], lu_c=2 * w["slot"], kr=KC._KR)


class ADSolver:
    """ADSolver(net, device): one solver call with its derivatives, x [B, S] in the network's species order, T [B] K,
    P [B] dyn cm^-2, dt [B] s, h [B] s (float64 CUDA tensors), optional per-cell log multipliers lm [B, R] and Arrhenius
    table rc [R, 9] (default: the network's, as kincuda.tables).
      jvp(...)  -> (x, h, done, accepted steps, dx, dh): the call and its tangent for the input tangents;
      vjp(...)  -> (x, h, done, accepted steps, cotangents dict x, T, P, dt, h, lm [B, R], rc [R, 9]).
    adapt selects the contract (module docstring). After each kernel the call kernel's guard lists the cells
    CudaSolver recomputes on the device, and those cells are redone on the recovery's path with their derivatives
    (module docstring), so the result is bitwise CudaSolver.launch's and the derivative is that of the result returned.
    status: the counts of kincuda.COUNTS (recomputed = flagged by the guard); flag [B] int32: 0 passed the guard, 1
    recovered, 2 still failed after the recovery (NaN derivatives; with an rc cotangent the reduction over cells is
    NaN too). Cells below 300 K raise. Calls run one at a time on torch's current stream (the rate and record scratch
    are shared)."""

    def __init__(self, net, device, cap=CAP):
        self.dev = torch.device(device)
        torch.zeros(1, device=self.dev)
        prop = torch.cuda.get_device_properties(self.dev)
        fit = lambda t: (prop.shared_memory_per_block_optin - len(KC._blob(t)[0])) // (8 * KC._perc(t))  # noqa: E731
        tab = KC.tables(net, 2)                               # the call kernel's warps per cell (CudaSolver's rule)
        if KC.warps(fit(tab)) != 2:
            tab = KC.tables(net, KC.warps(fit(tab)))
        supported(tab)
        ext = ad_tables(net, tab)
        self.ns, self.nr, self.cap = tab["ns"], tab["nr"], cap
        blob, _ = KC._blob(tab)
        cu = KC._cuda()
        self.wpb, self.smem, self.max_blocks, self.src, self._fn, self._mod = {}, {}, {}, {}, {}, {}
        self.capr = 2 * cap                                   # CAPR of the source: the recovery's stored attempts
        for kind in ("jvp", "vjp"):
            _, perwa = layout(tab, ext, kind)
            wpb = min(32, (prop.shared_memory_per_block_optin - len(blob)) // (8 * perwa))
            if wpb < 1:
                raise RuntimeError(f"the {kind} kernel's work arrays do not fit in shared memory")
            self.wpb[kind], self.smem[kind] = wpb, 8 * wpb * perwa + len(blob)
            self.src[kind] = source_ad(tab, ext, kind, wpb, cap)
            self._mod[kind] = ctypes.c_void_p()
            KC._check(cu.cuModuleLoadData(ctypes.byref(self._mod[kind]),
                                          KC.compile_cubin(self.src[kind], f"{prop.major}{prop.minor}")), "load")
            for name in (kind, kind + "_rec", "arrh" if kind == "vjp" else "guard"):
                f = ctypes.c_void_p()
                KC._check(cu.cuModuleGetFunction(ctypes.byref(f), self._mod[kind], name.encode()), "function")
                self._fn[name] = f
            for f in (self._fn[kind], self._fn[kind + "_rec"]):
                KC._check(cu.cuFuncSetAttribute(f, 8, self.smem[kind]), "shared memory")
            f = self._fn[kind]
            occ = ctypes.c_int()
            KC._check(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(occ), f, 32 * wpb, self.smem[kind]),
                      "occupancy")
            if occ.value < 1:
                raise RuntimeError(f"kernel {kind} does not fit on this device")
            self.max_blocks[kind] = prop.multi_processor_count * occ.value
        g = lambda a, view=None: torch.tensor(a if view is None else np.asarray(a).view(view), device=self.dev)
        self.rc0 = g(tab["rc"])
        self._tabs = [g(tab["perm"]), g(np.frombuffer(blob, np.int32).copy()), g(tab["rf"]), g(tab["nasa"])]
        self._nw = {k: self.max_blocks[k] * self.wpb[k] for k in ("jvp", "vjp")}
        self._scratch = torch.empty(self._nw["vjp"] * (12 * self.nr + 2), dtype=torch.float64, device=self.dev)
        self._jscr = self._scratch
        self._rec = torch.empty(self._nw["vjp"] * 2 * cap * (self.ns + 3), dtype=torch.float64, device=self.dev)
        self._rrec = torch.empty(self._nw["vjp"] * 2 * self.capr * (self.ns + 3), dtype=torch.float64, device=self.dev)
        self._wscr = torch.empty(max(self._nw.values()) * self.ns ** 2, dtype=torch.float64, device=self.dev)
        self._st = [g(tab["sst"].ravel(), np.int32), g(tab["jt"].ravel(), np.int64), g(tab["lu"], np.int64),
                    g(tab["optr"])]
        self._gt = [g(ext["gt"].ravel(), np.int64), g(ext["xg"])]
        self._rounds = [ctypes.c_int(tab["sst"].shape[0]), ctypes.c_int(tab["jround"]),
                        ctypes.c_int(ext["gt"].shape[0])]
        self._atoms, self._didx = g(tab["atoms"]), g(tab["didx"], np.int16)
        self.count = torch.zeros(10, dtype=torch.int32, device=self.dev)
        self.status, self.flag = {}, None

    def _prep(self, x, T, P, dt, h, lm, rc, max_steps):
        if not all(a.is_cuda and a.dtype == torch.float64 for a in (x, T, P, dt, h)) or x.shape[1] != self.ns:
            raise ValueError(f"x [B, {self.ns}], T, P, dt, h [B]: float64 CUDA tensors")
        if max_steps > self.cap * self.cap or KC.REC_STEPS * max_steps > self.capr ** 2:
            raise ValueError(f"max_steps above {self.cap ** 2} (CAP^2) or the recovery's tries above {self.capr ** 2}")
        B = x.shape[0]
        lm = None if lm is None else lm.contiguous()
        rc = self.rc0 if rc is None else rc.contiguous()
        if lm is not None and tuple(lm.shape) != (B, self.nr) or tuple(rc.shape) != (self.nr, 9):
            raise ValueError(f"lm [B, {self.nr}], rc [{self.nr}, 9]")
        return B, [a.contiguous() for a in (x, T, P, dt, h)], lm, rc

    def _ptr(self, a):
        return ctypes.c_void_p(0) if a is None else a

    def _guard(self, x, T, P, dt, xo, done, nok, atol, stream):
        """The call kernel's guard on the kernel's result: (redo, list) of the cells to redo on the recovery's path."""
        B = x.shape[0]
        redo, lst = (torch.empty(B, dtype=torch.int32, device=self.dev) for _ in range(2))
        KC._launch(self._fn["guard"], -(-B // 128) or 1, 128, 0, stream,
                   [x, T, P, dt, xo, done, nok, self._atoms, redo, lst, self.count, ctypes.c_int(B),
                    ctypes.c_double(atol)])
        return redo, lst

    def _finish(self, redo):
        """The counts in status, the per-cell flag; raises for cells below T_MIN."""
        n = self.count.tolist()
        if n[2]:
            raise ValueError(f"{n[2]} cells below {KC.T_MIN:g} K: outside the chemistry solver's domain")
        self.status = dict(zip(KC.COUNTS, n[1:9]))
        self.flag = redo

    def jvp(self, x, T, P, dt, h, dx, dT, dP, ddt, dh, lm=None, dlm=None, rc=None, drc=None, adapt=True, rtol=1e-3,
            atol=1e-18, max_steps=4000):
        """The call and its tangent: (x, h, done, accepted steps, dx, dh) for the tangents dx [B, S], dT, dP, ddt,
        dh [B], dlm [B, R] and drc [R, 9] (dlm, drc optional); with a leading axis of D <= DMAX directions on every
        tangent (dx [D, B, S], dT [D, B], ..., dlm [D, B, R], drc [D, R, 9]) the D tangents share one call and come
        back as dx [D, B, S], dh [D, B]."""
        B, (x, T, P, dt, h), lm, rc = self._prep(x, T, P, dt, h, lm, rc, max_steps)
        one = dx.dim() == 2
        D = 1 if one else dx.shape[0]
        lead = () if one else (D,)
        if not 1 <= D <= DMAX or tuple(dx.shape) != lead + (B, self.ns) or any(
                tuple(a.shape) != lead + (B,) for a in (dT, dP, ddt, dh)) or (
                dlm is not None and tuple(dlm.shape) != lead + (B, self.nr)) or (
                drc is not None and tuple(drc.shape) != lead + (self.nr, 9)):
            raise ValueError(f"tangents: dx [D, B, S], dT, dP, ddt, dh [D, B], dlm [D, B, R], drc [D, R, 9] "
                             f"(D <= {DMAX}), or the same without the D axis")
        tan = [a.contiguous() for a in (dx, dT, dP, ddt, dh)]
        dlm = None if dlm is None else dlm.contiguous()
        drc = None if drc is None else drc.contiguous()
        xo, dxo = torch.empty_like(x), torch.empty((D, B, self.ns), dtype=torch.float64, device=self.dev)
        ho, nok = (torch.empty(B, dtype=torch.float64, device=self.dev) for _ in range(2))
        dho = torch.empty((D, B), dtype=torch.float64, device=self.dev)
        done = torch.empty(B, dtype=torch.int32, device=self.dev)
        need = self._nw["jvp"] * (3 * self.nr + 2 + D * (2 * self.nr + self.ns))
        if self._jscr.numel() < need:
            self._jscr = torch.empty(need, dtype=torch.float64, device=self.dev)
        nb = max(1, min(-(-B // self.wpb["jvp"]), self.max_blocks["jvp"]))
        self.count.zero_()
        stream = torch.cuda.current_stream(self.dev).cuda_stream
        args = lambda ms: [x, T, P, dt, h, self._ptr(lm), rc, *tan, self._ptr(dlm), self._ptr(drc), ctypes.c_int(D),
                           xo, ho, done, nok, dxo, dho, self.count, ctypes.c_int(B), ctypes.c_double(rtol),
                           ctypes.c_double(atol), ctypes.c_int(ms), ctypes.c_int(int(adapt)), *self._tabs, self._jscr,
                           *self._st, *self._rounds[:2]]  # noqa: E731
        KC._launch(self._fn["jvp"], nb, 32 * self.wpb["jvp"], self.smem["jvp"], stream, args(max_steps))
        redo, lst = self._guard(x, T, P, dt, xo, done, nok, atol, stream)
        KC._launch(self._fn["jvp_rec"], nb, 32 * self.wpb["jvp"], self.smem["jvp"], stream,
                   args(KC.REC_STEPS * max_steps) + [redo, lst, self._wscr, self._didx, self._atoms])
        self._finish(redo)
        return xo, ho, done.bool(), nok, dxo[0] if one else dxo, dho[0] if one else dho

    def vjp(self, x, T, P, dt, h, gx, gh=None, lm=None, rc=None, adapt=True, rtol=1e-3, atol=1e-18, max_steps=4000):
        """The call and the cotangents of its inputs for the cotangents gx [B, S] of x_out and gh [B] of h_out
        (optional): (x, h, done, accepted steps, dict x, T, P, dt, h, lm [B, R] (of each reaction's log multiplier),
        rc [R, 9])."""
        B, (x, T, P, dt, h), lm, rc = self._prep(x, T, P, dt, h, lm, rc, max_steps)
        gx = gx.contiguous()
        gh = None if gh is None else gh.contiguous()
        xo = torch.empty_like(x)
        ho, nok = (torch.empty(B, dtype=torch.float64, device=self.dev) for _ in range(2))
        done = torch.empty(B, dtype=torch.int32, device=self.dev)
        g = dict(x=torch.empty_like(x), T=torch.empty_like(T), P=torch.empty_like(T), dt=torch.empty_like(T),
                 h=torch.empty_like(T), lm=torch.empty(B, self.nr, dtype=torch.float64, device=self.dev),
                 rc=torch.empty(self.nr, 9, dtype=torch.float64, device=self.dev))
        nb = max(1, min(-(-B // self.wpb["vjp"]), self.max_blocks["vjp"]))
        self.count.zero_()
        stream = torch.cuda.current_stream(self.dev).cuda_stream
        args = lambda ms, rec: [x, T, P, dt, h, self._ptr(lm), rc, gx, self._ptr(gh), xo, ho, done, nok, g["x"],
                                g["T"], g["P"], g["dt"], g["h"], g["lm"], self.count, ctypes.c_int(B),
                                ctypes.c_double(rtol), ctypes.c_double(atol), ctypes.c_int(ms),
                                ctypes.c_int(int(adapt)), *self._tabs, self._scratch, rec, *self._st, *self._gt,
                                *self._rounds]  # noqa: E731
        KC._launch(self._fn["vjp"], nb, 32 * self.wpb["vjp"], self.smem["vjp"], stream, args(max_steps, self._rec))
        redo, lst = self._guard(x, T, P, dt, xo, done, nok, atol, stream)
        KC._launch(self._fn["vjp_rec"], nb, 32 * self.wpb["vjp"], self.smem["vjp"], stream,
                   args(KC.REC_STEPS * max_steps, self._rrec) + [redo, lst, self._wscr, self._didx, self._atoms])
        KC._launch(self._fn["arrh"], self.nr, 256, 0, stream, [g["lm"], T, P, rc, self._tabs[2], g["rc"],
                                                               ctypes.c_int(B)])
        self._finish(redo)
        return xo, ho, done.bool(), nok, g
