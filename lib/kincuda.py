"""The solver's whole call as one CUDA kernel: what kinvulcan.TorchRates + kinbatch.rodas3_step(pivot=False) +
kinnet.advance_compact(order=3) compute for every cell, in one launch, for any network the reader takes (rate laws in
the reaction dicts; at most 254 species, 3 species per side, |S| <= 2).

The cell's threads compute its rate coefficients from T and P (NASA-9 Gibbs energies; each reaction's law from the
network: one or two Arrhenius k0 terms, falloff, Troe factor, efficiency; reverse rates from the equilibrium constants
or an explicit reverse law; n_tot folded into the three-body coefficients), then take every Rodas3 step of the call with
advance_compact's step control: a step is accepted when its RMS error is within tolerance and no species is below -atol,
accepted states are clamped at 0, the next step is 0.9 err^(-1/3) times the last (within [0.2, 5], at most 0.5 after a
rejection), at most max_steps tries. The exact Jacobian J = Jn + w 1^T (Jn = d(S v)/dn, w the constant-P term) enters
through an LU factor without row exchanges of M0 = I / (gamma h) - Jn on its sparse pattern, in a minimum-degree order,
and the Sherman-Morrison formula for the rank-one term.

Layout. W warps per cell (warps(): 2, or 4 when the shared memory holds at most 3 cells per block). The last NT =
min(NTAIL, NS) rows and columns of the order (the trailing block, dense or nearly so after the fill; structural zeros
included) are held as a dense NT x NT array after the other slots and factored in registers of the cell's first warp:
lane l owns block row l, and rows l, l + 32, ... of the sparse leading part in the solves. While the block is in
registers its shared-memory slots hold the reaction rates and the S v pieces, and they pass the factors to the second
warp. The S v sums, the Jacobian terms (32-bit words; each lane's finished slots in a side table; slots longer than an
even share cut into pieces added in order) and the LU updates of the sparse pivots run as balanced streams over the
cell's 32 W lanes, with descriptor bit widths set by the network's sizes; the reciprocals of the sparse pivots are taken
in batches of consecutive independent pivots. The four stage solves run from one copy of the triangular solve: stage 1
solves K1 on the first warp and M0^-1 w on the second, each stage ends with the Sherman-Morrison update on the first
warp. Cells per block: as many as the shared memory (per-cell arrays plus one copy of the small tables per block), the
named barriers and the registers (without spilling) allow. Persistent cells take work from an atomic queue, so no cell
waits for the stiffest one. Results do not depend on the batch size or order.

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


NTAIL = 32      # rows of the trailing block factored in registers (at most 32)


def warps(fit):
    """Warps per cell for a network whose per-cell arrays fit fit times in a block's shared memory: 4 when fit <= 3 (4
    warps x 32 lanes x about 160 registers already cap a block at 3 cells, so the extra lanes cost no cells), else 2."""
    return 4 if fit <= 3 else 2


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


def _lanes(items, size, nl):
    """Items spread over nl lanes, longest first onto the least-loaded lane: (items per lane, rounds)."""
    lanes, load = [[] for _ in range(nl)], np.zeros(nl, int)
    for it in sorted(items, key=lambda q: -size(q)):
        l_ = int(np.argmin(load))
        lanes[l_].append(it)
        load[l_] += size(it)
    return lanes, int(load.max())


def _bits(n):
    """Bits that hold the values 0..n."""
    return max(1, int(n).bit_length())


def _law(r):
    """Kernel row of a rate law: ln A0, B0, C0, ln A1, B1, C1 (second k0 term), ln A_inf, B_inf, C_inf, Troe a, T3, T1,
    T2, efficiency; flags: 1 falloff, 2 second k0 term, 64 Troe, 128 efficiency."""
    ln = lambda a: np.log(a) if a > 0 else -np.inf
    if r["kind"] == 2:
        a0, a1, ai = (r["A"], r["B"], r["C"]), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    else:
        a0 = (r["A0"], r["B0"], r["C0"])
        a1 = (r["A1"], r["B1"], r["C1"]) if "A1" in r else (0.0, 0.0, 0.0)
        ai = (r["Ai"], r["Bi"], r["Ci"]) if r["kind"] in (3, 6) else (1.0, 0.0, 0.0)
    if min(a0[0], a1[0], ai[0]) < 0:
        raise ValueError(f"negative Arrhenius prefactor in {r.get('eq', r)}: the kernel takes ln A")
    row = [ln(a0[0]), a0[1], a0[2], ln(a1[0]), a1[1], a1[2], ln(ai[0]), ai[1], ai[2],
           *r.get("troe", (0.0, 1.0, 1.0, 0.0)), r.get("eff", 1.0)]
    fl = int(r["kind"] in (3, 6)) | int("A1" in r) << 1 | int("troe" in r) << 6 | int("eff" in r) << 7
    return row, fl


def tables(net, w):
    """The kernel's tables from a VulcanNetwork (rate laws in the reaction dicts) for w >= 2 warps per cell: species
    order, rate laws, S v and Jacobian term streams, LU pattern and update operations (species in the kernel's order:
    perm[i] = network index of row i), descriptor bit widths."""
    S = np.asarray(net.S)
    ns, nr = S.shape
    nl = 32 * w
    re = [list(r["re"]) for r in net.reac]
    pr = [list(r["pr"]) for r in net.reac]
    if ns > 254 or max(len(q) for q in re + pr) > 3 or np.abs(S).max() > 2:
        raise ValueError("the kernel holds at most 254 species, 3 species per side, |S| <= 2")
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
    nt = min(NTAIL, ns)
    k0 = ns - nt                              # rows < k0: sparse leading part; rows >= k0: the trailing block
    P |= {(i, j) for i in range(k0, ns) for j in range(k0, ns)}
    lead = sorted(e for e in P if not (e[0] >= k0 and e[1] >= k0))   # row-major, then the trailing block row-major
    slot = {e: n for n, e in enumerate(lead)}
    tb0 = len(lead)
    for i in range(k0, ns):
        for j in range(k0, ns):
            slot[(i, j)] = tb0 + (i - k0) * nt + j - k0
    nnz = len(slot)
    # reactions: species slots (padding ns: n[ns] = 1, g[ns] = 0), rate laws (rows nr + r: explicit reverse laws)
    rev = any("rev" in r for r in net.reac)
    rx = np.full((nr, 6), ns, np.uint8)
    rc = np.zeros((2 * nr if rev else nr, 14))
    rf = np.zeros(len(rc), np.int32)
    for j, r in enumerate(net.reac):
        rx[j, :len(re[j])] = [pos[q] for q in re[j]]
        rx[j, 3:3 + len(pr[j])] = [pos[q] for q in pr[j]]
        rc[j], rf[j] = _law(r)
        rf[j] |= (r["kind"] in (3, 5, 6)) << 2 | (len(pr[j]) - len(re[j]) + 4) << 3 | ("rev" in r) << 8
        if "rev" in r:
            rc[nr + j], rf[nr + j] = _law(r["rev"])
    law = sorted({"troe" for r in net.reac if "troe" in r or "troe" in r.get("rev", {})}
                 | {"eff" for r in net.reac if "eff" in r or "eff" in r.get("rev", {})} | ({"rev"} if rev else set()))
    if not {"troe", "eff"} & set(law):
        rc = rc[:, :9]
    # widest forward and reverse side of each round of nl reactions
    wid = [(rx[:, o:o + 3] != ns).sum(1) for o in (0, 3)]
    vs = [[int(q[c:c + nl].max()) for c in range(0, nr, nl)] for q in wid]
    # descriptor widths: reaction, rate (kf, kr, the zero rate 2 nr), species (padding ns), LU slot, S v piece
    w_ = dict(r=_bits(nr - 1), k=_bits(2 * nr), s=_bits(ns), slot=_bits(nnz - 1))
    # S v: each row's terms cut into pieces of <= ch; a row's first piece writes part[i], the others part[ns + e]
    rows = [[(r, S[perm[i], r]) for r in live if S[perm[i], r] != 0] for i in range(ns)]
    ch = -(-sum(len(t) for t in rows) // nl)
    pieces, xp, nx = [], [0], 0
    for i, t in enumerate(rows):
        for c in range(0, len(t), ch):
            pieces.append((i if c == 0 else ns + nx, t[c:c + ch]))
            nx += c > 0
        xp.append(nx)
    npart = ns + nx
    if nx > 255:
        raise ValueError("too many S v pieces")
    w_["part"] = _bits(npart - 1)
    if w_["r"] + w_["part"] + 4 > 32 or w_["k"] + 2 * w_["s"] + 3 > 32 or 3 * w_["slot"] > 64:
        raise ValueError(f"descriptor widths {w_} do not fit")
    lanes, sround = _lanes(range(len(pieces)), lambda q: len(pieces[q][1]), nl)
    sst = np.zeros((sround, nl), np.uint32)
    o_dst = w_["r"] + 2
    for l_, its in enumerate(lanes):
        q = 0
        for it in its:
            dst, terms = pieces[it]
            for e, (r, c) in enumerate(terms):
                sst[q, l_] = (r | int(c < 0) << w_["r"] | int(abs(c) == 2) << (w_["r"] + 1) | dst << o_dst
                              | int(e == len(terms) - 1) << (o_dst + w_["part"]) | 1 << 31)
                q += 1
    # Jacobian: the terms of each slot (one lane each): coefficient * k * n_a * n_b, as 32-bit words (rate, sign, x2,
    # a, b, end of slot); padding words take the zero rate at index 2 nr; each lane's finished slots in order in jsl
    terms = [[] for _ in range(nnz)]
    for (p, q), lst in jn.items():
        s = slot[(pos[p], pos[q])]
        for c, (r, q_) in lst:
            for sgn, k, rest in dv[(r, q_)]:
                ab = [pos[o] for o in rest] + [ns] * (2 - len(rest))
                terms[s].append((k, c * sgn, ab[0], ab[1]))
    # a slot longer than the even share is cut into pieces: the first writes the slot, the others Y[e] (free during the
    # Jacobian), added to the slot afterwards in order (fix: slot of each extra piece)
    cap = -(-sum(len(t) for t in terms) // nl)
    jp, fix = [], []                          # pieces: (destination, terms); destination >= nnz: Y[dest - nnz]
    for s in range(nnz):
        for c in range(0, len(terms[s]), cap):
            jp.append((s if c == 0 else nnz + len(fix), terms[s][c:c + cap]))
            if c:
                fix.append(s)
    if len(fix) > ns or nnz + len(fix) > 0xFFFF:
        raise ValueError("too many Jacobian pieces")
    lanes, jround = _lanes([q for q in range(len(jp)) if jp[q][1]], lambda q: len(jp[q][1]), nl)
    o_a = w_["k"] + 2
    o_end = o_a + 2 * w_["s"]
    jw = np.zeros((jround, nl), np.uint32)
    for l_, its in enumerate(lanes):
        q = 0
        for it in its:
            tl = jp[it][1]
            for e, (k, c, a, b) in enumerate(tl):
                jw[q, l_] = (int(k) | int(c < 0) << w_["k"] | int(abs(c) == 2) << (w_["k"] + 1) | a << o_a
                             | b << (o_a + w_["s"]) | int(e == len(tl) - 1) << o_end)
                q += 1
        jw[q:, l_] = 2 * nr | ns << o_a | ns << (o_a + w_["s"])
    lanes = [[jp[it][0] for it in its] for its in lanes]
    jsmax = max(len(its) for its in lanes)
    jsl = np.zeros((nl, jsmax), np.uint16)
    for l_, its in enumerate(lanes):
        jsl[l_, :len(its)] = its
    jb = jw.tobytes() + jsl.tobytes() + np.array(fix, np.uint16).tobytes()
    jt = np.frombuffer(jb + b"\0" * (-len(jb) % 8), np.uint64)
    # LU of the sparse leading part (pivots < k0; their updates reach the trailing block too); batches of consecutive
    # mutually independent pivots (blen[k]: length of the batch starting at pivot k, else 0)
    lu = [(slot[(i, j)] | slot[(i, k)] << w_["slot"] | slot[(k, j)] << 2 * w_["slot"]) for k in range(k0)
          for (i, j) in ops[k]]
    optr = np.cumsum([0] + [len(ops[k]) for k in range(k0)]).astype(np.int32)
    blen, cur = np.zeros(k0, np.uint8), None
    for k in range(k0):
        if cur is None or blen[cur] == 255 or any((j, k) in P or (k, j) in P for j in range(cur, k)):
            cur = k
        blen[cur] += 1
    smap = np.full((ns, ns), 0xFFFF, np.uint16)
    for (i, j), s in slot.items():
        smap[i, j] = s
    rowptr = np.searchsorted([i for (i, j) in lead], np.arange(ns + 1)).astype(np.uint16)
    colof = np.array([j for (i, j) in lead], np.uint8)
    diag = np.array([slot[(i, i)] for i in range(ns)], np.uint16)
    nasa = np.concatenate([np.asarray(net.lo)[perm], np.asarray(net.hi)[perm]], 0)
    return dict(ns=ns, nr=nr, nnz=nnz, k0=k0, tb0=tb0, w=w, P=P, perm=np.array(perm, np.int32), rx=rx, rc=rc, rf=rf,
                law=law, nasa=nasa, npart=npart, xp=np.array(xp, np.uint8), sst=sst, jt=jt, jround=jround,
                jsmax=jsmax, nfix=len(fix), lu=np.array(lu, np.uint64), optr=optr, blen=blen, vs=vs, smap=smap,
                rowptr=rowptr, colof=colof, diag=diag, atoms=np.asarray(net.atoms, float), wd=w_)


_SRC = r"""
#define NS $ns
#define NR $nr
#define NNZ $nnz
#define TB0 $tb0
#define NPART $npart
#define K0 $k0
#define NT $nt
#define RP $rp
#define RCW $rcw
#define CPB $cpb
#define W $w
#define NL (32 * W)
#define PERC $perc
#define VOFF $voff
#define TBYTES $tbytes
#define JSMAX $jsmax
#define NFIX $nfix
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
#define O_BLEN $o_blen
// column k of the sparse leading part has L entries in the leading rows (LPRE), L entries in the trailing rows
// (LTAIL), U entries (UPRE); trailing column K0 + j has U entries in the leading rows (UTAIL(j)); VSA(q), VSB(q): the
// widest forward and reverse side of round q of NL reactions
$masks
#define FULL 0xffffffffu
#define CSYNC(g) asm volatile("bar.sync %0, %1;" :: "r"((g) + 1), "r"(NL) : "memory")   // the cell's threads
typedef unsigned short u16;
typedef unsigned char u8;
typedef unsigned long long u64;

struct Tab {   // small tables, copied to shared memory once per block
  const u16 *smap, *rowptr, *diag;   // slot of (row, col) or 0xFFFF; first leading slot of each row; diagonal slots
  const u8 *colof, *rx, *xp, *blen;  // column of each leading slot; species of each reaction (3 + 3, pad NS); pieces;
};                                   // pivot batches

__device__ __forceinline__ double wsum(double v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
  return v;
}

// Rate coefficient of the law in row q (_law) at T, ln T, 1 / T, n_tot; three-body n_tot excluded.
__device__ __forceinline__ double law(const double* q, int fl, double Tc, double lT, double iT, double nd) {
  double k = exp(q[0] + q[1] * lT - q[2] * iT);
  if (fl & 2) k += exp(q[3] + q[4] * lT - q[5] * iT);
$falloff  return k;
}

// F = (n_tot / nd) S v(n) at the state Y (n = nd Y / n_tot, n_tot = sum Y); u0 = S v / nd unless null; returns n_tot.
// Every warp of the cell sums n_tot itself; the rest is spread over the cell's NL lanes (cl).
__device__ __noinline__ double rhs(const double* Y, double* F, double* u0, double* n, double* v, double* part,
                                      const double* kf, const double* kr, double nd, const Tab& T,
                                      const unsigned* __restrict__ sst, int sround, int lane, int cl, int grp) {
  double p = 0.0;
  for (int i = lane; i < NS; i += 32) p += Y[i];
  const double ntot = wsum(p), xf = nd / ntot, ixf = ntot / nd;
  for (int i = cl; i < NS; i += NL) n[i] = Y[i] * xf;
  for (int i = cl; i < NPART; i += NL) part[i] = 0.0;
  if (cl == 0) n[NS] = 1.0;
  CSYNC(grp);
  #pragma unroll
  for (int qq = 0; qq < (NR + NL - 1) / NL; qq++) {   // each round of NL reactions with its widest sides
    const int r = cl + NL * qq, na = VSA(qq), nb = VSB(qq);
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
  CSYNC(grp);
  {   // S v as lane streams: reaction, sign, x2, destination piece, end of piece
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
  CSYNC(grp);
  for (int i = cl; i < NS; i += NL) {
    double s = part[i];
    for (int e = T.xp[i]; e < T.xp[i + 1]; e++) s += part[NS + e];
    F[i] = s * ixf;
    if (u0) u0[i] = s / nd;
  }
  CSYNC(grp);
  return ntot;
}

// a = M0^-1 y on one warp: lane l returns rows l + 32 p (< K0) in a0[p] and K0 + l (< NS) in a1. L and U of the sparse
// leading columns come from shared memory, the trailing block from registers c.
__device__ __forceinline__ void msolve(const double* M, const double* dinv, const double* y, const u16* smap,
                                       const double (&c)[NT], int lane, int myt, double (&a0)[RP], double& a1r) {
  const bool ht = lane < NT;
  double d0[RP];
  #pragma unroll
  for (int p = 0; p < RP; p++) {
    const int i = lane + 32 * p;
    a0[p] = i < K0 ? y[i] : 0.0;
    d0[p] = i < K0 ? dinv[i] : 0.0;
  }
  double a1 = ht ? y[K0 + lane] : 0.0;
  const double d1 = ht ? dinv[K0 + lane] : 0.0;
  #pragma unroll
  for (int k = 0; k < K0; k++) {                        // unit L, sparse columns
    if (!(LPRE(k) | LTAIL(k))) continue;
    const double yk = __shfl_sync(FULL, a0[k >> 5], k & 31);
    if (LPRE(k)) {
      #pragma unroll
      for (int p = k >> 5; p < RP; p++) {
        const int i = lane + 32 * p;
        if (i > k && i < K0) {
          const unsigned s = smap[i * NS + k];
          if (s != 0xFFFFu) a0[p] -= M[s] * yk;
        }
      }
    }
    if (LTAIL(k) && ht) {
      const unsigned s = smap[(K0 + lane) * NS + k];
      if (s != 0xFFFFu) a1 -= M[s] * yk;
    }
  }
  #pragma unroll
  for (int k = 0; k < NT - 1; k++) {                    // unit L, trailing columns
    const double yk = __shfl_sync(FULL, a1, k);
    if (myt > k) a1 -= c[k] * yk;
  }
  #pragma unroll
  for (int k = NT - 1; k >= 0; k--) {                   // U, trailing columns
    if (k == 0 && !UTAIL(0)) continue;
    const double zk = __shfl_sync(FULL, a1 * d1, k);
    if (UTAIL(k)) {
      #pragma unroll
      for (int p = 0; p < RP; p++) {
        const int i = lane + 32 * p;
        if (i < K0) {
          const unsigned s = smap[i * NS + K0 + k];
          if (s != 0xFFFFu) a0[p] -= M[s] * zk;
        }
      }
    }
    if (k > 0 && myt >= 0 && myt < k) a1 -= c[k] * zk;
  }
  #pragma unroll
  for (int k = K0 - 1; k >= 0; k--) {                   // U, sparse columns
    if (!UPRE(k)) continue;
    const double zk = __shfl_sync(FULL, a0[k >> 5] * d0[k >> 5], k & 31);
    #pragma unroll
    for (int p = 0; p <= (k >> 5); p++) {
      const int i = lane + 32 * p;
      if (i < k) {
        const unsigned s = smap[i * NS + k];
        if (s != 0xFFFFu) a0[p] -= M[s] * zk;
      }
    }
  }
  #pragma unroll
  for (int p = 0; p < RP; p++) a0[p] *= d0[p];
  a1r = a1 * d1;
}

// one block per SM is enough: without that bound ptxas may cap the registers below what the block size allows and spill
extern "C" __global__ void __launch_bounds__(NL * CPB, 1) call(
    const double* __restrict__ x0, const double* __restrict__ Ta, const double* __restrict__ Pa,
    const double* __restrict__ dta, double* __restrict__ hio, double* __restrict__ xout, int* __restrict__ doneo,
    double* __restrict__ noko, int* __restrict__ count, int B, double rtol, double atol, int max_steps,
    const int* __restrict__ perm, const unsigned* __restrict__ blob, const double* __restrict__ rc,
    const int* __restrict__ rf, const double* __restrict__ nasa, double* scratch, const unsigned* __restrict__ sst,
    const u64* __restrict__ jt, const u64* __restrict__ lu, const int* __restrict__ optr, int sround, int jround)
{
  extern __shared__ double smem[];
  unsigned* tb32 = (unsigned*)(smem + CPB * PERC);
  for (int i = threadIdx.x; i < TBYTES / 4; i += blockDim.x) tb32[i] = blob[i];
  __syncthreads();
  const u8* tb = (const u8*)tb32;
  Tab T;
  T.smap = (const u16*)(tb + O_SMAP); T.rowptr = (const u16*)(tb + O_ROWPTR); T.diag = (const u16*)(tb + O_DIAG);
  T.colof = tb + O_COLOF; T.rx = tb + O_RX; T.xp = tb + O_XP; T.blen = tb + O_BLEN;
  const int lane = threadIdx.x & 31, cl = threadIdx.x % NL, grp = threadIdx.x / NL;
  const bool w0 = cl < 32;                    // the cell's first warp: trailing block, solves
  const int myt = lane < NT ? lane : -1;      // this lane's row K0 + myt of the dense trailing block
  double c[NT];                               // that row: M0, then its LU factors
  #pragma unroll
  for (int j = 0; j < NT; j++) c[j] = 0.0;
  double* M = smem + grp * PERC;              // M0 by slot, then its LU factors (leading slots, then the block)
  double *dinv = M + NNZ, *n = dinv + NS + 1, *N = n + NS + 1, *Y = N + NS, *F = Y + NS, *u = F + NS, *K1 = u + NS,
         *K2 = K1 + NS, *K3 = K2 + NS, *K4 = K3 + NS;
  double *v = M + VOFF, *part = v + NR;       // in the trailing block's slots while that block is in registers if
                                              // they fit there (VOFF = TB0), else after the other arrays
  int* nextc = (int*)(K4 + NS);
  double* kf = scratch + (size_t)(blockIdx.x * CPB + grp) * (2 * NR + 2);
  double* kr = kf + NR;
  if (cl == 0) kf[2 * NR] = 0.0;              // the zero rate of the Jacobian's padding words
  int cell = blockIdx.x * CPB + grp;
  while (cell < B) {
    const double Tc = Ta[cell], dt = dta[cell], nd = Pa[cell] / (KBOLTZ * Tc);
    for (int i = cl; i < NS; i += NL) N[i] = x0[(size_t)cell * NS + perm[i]];
    CSYNC(grp);
    double p = 0.0;
    for (int i = lane; i < NS; i += 32) p += N[i];
    const bool fin = isfinite(wsum(p)) && isfinite(Tc) && isfinite(nd) && isfinite(dt), cold = Tc < TMIN;
    if (fin && !cold) {   // rate coefficients; three-body ones times nd
      const double lT = log(Tc), iT = 1.0 / Tc, lKT = log(KBOLTZ * Tc / P0BAR);
      double* g = v;                          // G / RT of each species
      const double* cf = nasa + (Tc < 1000.0 ? 0 : NS * 9);
      for (int i = cl; i < NS; i += NL) {
        const double* a = cf + 9 * i;
        const double T2 = Tc * Tc, T3 = T2 * Tc, T4 = T3 * Tc;
        const double hh = -a[0] * iT * iT + a[1] * lT * iT + a[2] + a[3] * Tc / 2 + a[4] * T2 / 3 + a[5] * T3 / 4
                          + a[6] * T4 / 5 + a[7] * iT;
        const double ss = -a[0] * iT * iT / 2 - a[1] * iT + a[2] * lT + a[3] * Tc + a[4] * T2 / 2 + a[5] * T3 / 3
                          + a[6] * T4 / 4 + a[8];
        g[i] = hh - ss;
      }
      if (cl == 0) g[NS] = 0.0;
      CSYNC(grp);
      for (int r = cl; r < NR; r += NL) {
        const int fl = rf[r];                 // law flags (law()), 4 three-body, (dmu + 4) << 3, 256 explicit reverse
        const double k = law(rc + RCW * r, fl, Tc, lT, iT, nd);
        const u8* x = T.rx + 6 * r;
        const double dg = g[x[3]] + g[x[4]] + g[x[5]] - g[x[0]] - g[x[1]] - g[x[2]];
        const double c3 = (fl & 4) ? nd : 1.0;
        kf[r] = k * c3;
        kr[r] = $kr;
      }
    }
    CSYNC(grp);
    double h = fmax(fmin(hio[cell], dt), 1e-12), t = 0.0, nok = 0.0;
    int it = (dt > 0.0 && fin && !cold) ? 0 : max_steps;   // no step for non-finite inputs or cold gas
    while (it < max_steps && t < dt * (1.0 - 1e-14)) {
      const double ha = fmin(h, dt - t), igh = 1.0 / (0.5 * ha), ih = 1.0 / ha;
      const double ntot = rhs(N, F, u, n, v, part, kf, kr, nd, T, sst, sround, lane, cl, grp);
      for (int s = cl; s < NNZ; s += NL) M[s] = 0.0;
      CSYNC(grp);
      {   // M = -Jn: lane streams of 32-bit terms (rate index, sign, x2, species a, b, end of slot); slots from jsl
        const unsigned* jw = (const unsigned*)jt;
        const u16* jsl = (const u16*)(jw + jround * NL) + cl * JSMAX;
        double acc = 0.0;
        int ne = 0;
        #pragma unroll 4
        for (int q = 0; q < jround; q++) {
          const unsigned w = __ldg(jw + q * NL + cl);   // kr = kf + NR
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
      CSYNC(grp);
#if NFIX
      if (cl == 0) {                            // the extra pieces of long slots, in order
        const u16* fx = (const u16*)((const unsigned*)jt + jround * NL) + NL * JSMAX;
        for (int e = 0; e < NFIX; e++) M[__ldg(fx + e)] += Y[e];
      }
      CSYNC(grp);
#endif
      for (int i = cl; i < NS; i += NL) {     // u = w = S v / nd - Jn x; M = M0 = I / (gamma h) - Jn
        double acc = 0.0;
        const int s1 = T.rowptr[i + 1];
        for (int s = T.rowptr[i]; s < s1; s++) acc += M[s] * N[T.colof[s]];
        if (i >= K0)
          for (int j = 0; j < NT; j++) acc += M[TB0 + (i - K0) * NT + j] * N[K0 + j];
        u[i] = u[i] + acc / ntot;
        M[T.diag[i]] += igh;
      }
      CSYNC(grp);
      for (int k = 0; k < K0; k++) {           // LU of the sparse leading columns (updates reach the trailing block)
        const int len = T.blen[k];
        if (len) {                              // a batch of consecutive independent pivots: their 1 / diagonal
          for (int i = cl; i < len; i += NL) dinv[k + i] = 1.0 / M[T.diag[k + i]];
          CSYNC(grp);
        }
        if (!(LPRE(k) | LTAIL(k))) continue;
        const int oa = optr[k], ob = optr[k + 1];
        u64 op = (oa + cl < ob) ? __ldg(lu + oa + cl) : 0ull;
        const double dk = dinv[k];
        for (int i = cl; i < NS; i += NL) {
          const unsigned q = T.smap[i * NS + k];
          if (i > k && q != 0xFFFFu) M[q] *= dk;
        }
        CSYNC(grp);
        for (int e = oa + cl; e < ob; e += NL) {
          const u64 nx = (e + NL < ob) ? __ldg(lu + e + NL) : 0ull;
          M[op & ${slm}ull] -= M[(op >> $lu_b) & ${slm}ull] * M[(op >> $lu_c) & ${slm}ull];
          op = nx;
        }
        CSYNC(grp);
      }
      if (w0) {                                 // dense trailing block in registers; 1 / its diagonal into dinv
        if (myt >= 0) {
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
        if (myt >= 0) {                         // the factors back into the block's slots for the second warp
          #pragma unroll
          for (int j = 0; j < NT; j++) M[TB0 + myt * NT + j] = c[j];
        }
      }
      for (int i = cl; i < NS; i += NL) K1[i] = F[i];
      CSYNC(grp);
      if (cl >= 32 && cl < 64 && myt >= 0) {    // the second warp's copy of the trailing factors (two-vector solve)
        #pragma unroll
        for (int j = 0; j < NT; j++) c[j] = M[TB0 + myt * NT + j];
      }
      // Rodas3 stages (kinbatch.RODAS3, gamma = 1/2) as one loop with one copy of the triangular solve: stage 1 solves
      // K1 on the first warp and u on the second, then c0 = 1 / (1 - sum M0^-1 u); every stage ends with the
      // Sherman-Morrison update on the first warp
      double c0 = 0.0;
      #pragma unroll 1
      for (int st = 1; st <= 4; st++) {
        if (st > 2) {
          for (int i = cl; i < NS; i += NL) {
            double y = N[i] + 2.0 * K1[i];
            if (st == 4) y = y + K3[i];
            Y[i] = y;
          }
          CSYNC(grp);
          rhs(Y, F, nullptr, n, v, part, kf, kr, nd, T, sst, sround, lane, cl, grp);
        }
        double* Ks = st == 1 ? K1 : (st == 2 ? K2 : (st == 3 ? K3 : K4));
        if (st > 1) {
          for (int i = cl; i < NS; i += NL)
            Ks[i] = st == 2 ? F[i] + 4.0 * ih * K1[i]
                            : (st == 3 ? F[i] + ih * (K1[i] - K2[i])
                                       : F[i] + ih * (K1[i] - K2[i] - (8.0 / 3.0) * K3[i]));
          CSYNC(grp);
        }
        if (cl < (st == 1 ? 64 : 32)) {
          const bool uw = st == 1 && !w0, ht = lane < NT;    // uw: this warp solves u
          double* yv = uw ? u : Ks;
          double a0[RP], a1;
          msolve(M, dinv, yv, T.smap, c, lane, myt, a0, a1);
          if (uw) {
            double sb = lane < K0 ? a0[0] : 0.0;
            #pragma unroll
            for (int p = 1; p < RP; p++) sb += lane + 32 * p < K0 ? a0[p] : 0.0;
            const double cc = 1.0 / (1.0 - wsum(sb + (ht ? a1 : 0.0)));
            if (lane == 0) dinv[NS] = cc;
          }
          __syncwarp();
          #pragma unroll
          for (int p = 0; p < RP; p++) if (lane + 32 * p < K0) yv[lane + 32 * p] = a0[p];
          if (ht) yv[K0 + lane] = a1;
        }
        CSYNC(grp);
        c0 = dinv[NS];
        if (w0) {
          const bool ht = lane < NT;
          double a0[RP];
          #pragma unroll
          for (int p = 0; p < RP; p++) a0[p] = lane + 32 * p < K0 ? Ks[lane + 32 * p] : 0.0;
          const double a1 = ht ? Ks[K0 + lane] : 0.0;
          double sa = a0[0];
          #pragma unroll
          for (int p = 1; p < RP; p++) sa += a0[p];
          const double s = wsum(sa + a1) * c0;
          __syncwarp();
          #pragma unroll
          for (int p = 0; p < RP; p++) if (lane + 32 * p < K0) Ks[lane + 32 * p] = a0[p] + s * u[lane + 32 * p];
          if (ht) Ks[K0 + lane] = a1 + s * u[K0 + lane];
        }
        CSYNC(grp);
      }
      // step control (kinnet.advance_compact, order 3); every warp of the cell reaches the same decision
      double e2 = 0.0;
      bool nn = true;
      for (int i = lane; i < NS; i += 32) {
        const double y = N[i] + 2.0 * K1[i] + K3[i] + K4[i];
        if (w0) Y[i] = y;
        const double sc = atol + rtol * fmax(fabs(N[i]), fabs(y));
        const double q = K4[i] / sc;
        e2 += q * q;
        nn = nn && (y >= -atol);
      }
      const double en = sqrt(wsum(e2) / NS);
      const bool nonneg = __all_sync(FULL, nn), efin = isfinite(en), ok = (en <= 1.0) && nonneg && efin;
      CSYNC(grp);
      if (ok) {
        for (int i = cl; i < NS; i += NL) N[i] = fmax(Y[i], 0.0);
        t += ha;
        nok += 1.0;
      }
      CSYNC(grp);
      double fac = efin ? 0.9 / cbrt(fmax(en, 1e-10)) : 0.1;
      fac = fmin(fmax(fac, 0.2), 5.0);
      if (!ok) fac = fmin(fac, 0.5);
      h = ha * fac;
      it++;
    }
    p = 0.0;
    for (int i = lane; i < NS; i += 32) p += N[i];
    const double s = wsum(p);
    for (int i = cl; i < NS; i += NL) xout[(size_t)cell * NS + perm[i]] = N[i] / s;
    if (cl == 0) { hio[cell] = h; doneo[cell] = (t >= dt * (1.0 - 1e-14)) ? 1 : 0; noko[cell] = nok; }
    if (cl == 0) *nextc = gridDim.x * CPB + atomicAdd(count, 1);
    CSYNC(grp);
    cell = *nextc;
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
    for name in ("smap", "rowptr", "diag", "colof", "rx", "xp", "blen"):
        blob += b"\0" * (-len(blob) % 2)
        offs[name] = len(blob)
        blob += np.ascontiguousarray(tab[name]).tobytes()
    return blob + b"\0" * (-len(blob) % 8), offs


def _alias(tab):
    """True if the rates v (or the G / RT of the species) and the S v pieces fit in the trailing block's slots."""
    nt = tab["ns"] - tab["k0"]
    return max(tab["nr"], tab["ns"] + 1) + tab["npart"] <= nt * nt


def _perc(tab):
    """Shared-memory doubles per cell: M (leading slots and the trailing block), dinv (and c0), n, N, Y, F, u, K1-K4,
    one for the queue's next cell; v and the S v pieces when they do not fit in the trailing block."""
    ns = tab["ns"]
    return tab["nnz"] + ns + 1 + ns + 1 + 8 * ns + 1 + (0 if _alias(tab) else max(tab["nr"], ns + 1) + tab["npart"])


def _mask(name, ks, n):
    """#define name(k): bit k of the set ks (k < n), as 64-bit words."""
    ks = set(ks)
    words = [sum(1 << (k - 64 * q) for k in ks if 64 * q <= k < 64 * q + 64) for q in range(max(1, -(-n // 64)))]
    e = f"(({words[-1]:#x}ull >> ((k) & 63)) & 1ull)"
    for q in range(len(words) - 2, -1, -1):
        e = f"((k) < {64 * q + 64} ? (({words[q]:#x}ull >> ((k) & 63)) & 1ull) : {e})"
    return f"#define {name}(k) {e}"


def _field(name, vals):
    """#define name(q): the 2-bit value vals[q], as 64-bit words."""
    words = [sum(int(v) << (2 * (q - 32 * w)) for q, v in enumerate(vals) if 32 * w <= q < 32 * w + 32)
             for w in range(max(1, -(-len(vals) // 32)))]
    e = f"((int)(({words[-1]:#x}ull >> (2 * ((q) & 31))) & 3ull))"
    for w in range(len(words) - 2, -1, -1):
        e = f"((q) < {32 * w + 32} ? ((int)(({words[w]:#x}ull >> (2 * ((q) & 31))) & 3ull)) : {e})"
    return f"#define {name}(q) {e}"


_FALLOFF = "  if (fl & 1) k = k / (1.0 + k * nd / exp(q[6] + q[7] * lT - q[8] * iT));\n"
_TROE = """  if (fl & 1) {
    const double pr = k * nd / exp(q[6] + q[7] * lT - q[8] * iT);
    k = k / (1.0 + pr);
    if (fl & 64) {   // Troe factor (a, T3, T1, T2 = q[9..12])
      const double fc = log10((1.0 - q[9]) * exp(-Tc / q[10]) + q[9] * exp(-Tc / q[11]) + exp(-q[12] / Tc));
      const double x = log10(pr) - 0.4 - 0.67 * fc, y = x / (0.75 - 1.27 * fc - 0.14 * x);
      k *= exp10(fc / (1.0 + y * y));
    }
  }
"""
_EFF = "  if (fl & 128) k *= q[13];\n"
_KR = "(k * exp(dg + (double)(((fl >> 3) & 7) - 4) * lKT)) * c3"
_KREV = "(fl & 256) ? law(rc + RCW * (NR + r), rf[NR + r], Tc, lT, iT, nd) * c3 : " + _KR


def source(tab, cpb):
    """The kernel's CUDA source for the tables of tables() and cpb cells per block."""
    ns, k0, P, w = tab["ns"], tab["k0"], tab["P"], tab["wd"]
    nt = ns - k0
    blob, offs = _blob(tab)
    m = lambda b: (1 << b) - 1
    masks = "\n".join([
        _mask("LPRE", (k for k in range(k0) if any((i, k) in P for i in range(k + 1, k0))), k0),
        _mask("LTAIL", (k for k in range(k0) if any((i, k) in P for i in range(k0, ns))), k0),
        _mask("UPRE", (k for k in range(k0) if any((i, k) in P for i in range(k))), k0),
        _mask("UTAIL", (j for j in range(nt) if any((i, k0 + j) in P for i in range(k0))), nt),
        _field("VSA", tab["vs"][0]), _field("VSB", tab["vs"][1])])
    law = tab["law"]
    falloff = (_TROE if "troe" in law else _FALLOFF) + (_EFF if "eff" in law else "")
    j_a = w["k"] + 2
    return string.Template(_SRC).substitute(
        ns=ns, nr=tab["nr"], nnz=tab["nnz"], tb0=tab["tb0"], npart=tab["npart"], k0=k0, nt=nt,
        rp=max(1, -(-k0 // 32)), rcw=tab["rc"].shape[1], cpb=cpb, w=tab["w"], perc=_perc(tab),
        voff=tab["tb0"] if _alias(tab) else tab["nnz"] + 10 * ns + 3, tbytes=len(blob), jsmax=tab["jsmax"],
        nfix=tab["nfix"], masks=masks, falloff=falloff, kr=_KREV if "rev" in law else _KR,
        sv_r=m(w["r"]), sv_sg=w["r"], sv_x2=w["r"] + 1, sv_dst=w["r"] + 2, sv_pm=m(w["part"]),
        sv_end=w["r"] + 2 + w["part"],
        j_km=m(w["k"]), j_sg=w["k"], j_x2=w["k"] + 1, j_a=j_a, j_b=j_a + w["s"], j_sm=m(w["s"]),
        j_end=j_a + 2 * w["s"], slm=m(w["slot"]), lu_b=w["slot"], lu_c=2 * w["slot"],
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
        lib.cuModuleUnload.argtypes = [vp]
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


def _load(src, arch):
    """Compile and load the source: (module, call kernel, guard kernel)."""
    cu = _cuda()
    mod, fn, guard = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    _check(cu.cuModuleLoadData(ctypes.byref(mod), compile_cubin(src, arch)), "load")
    _check(cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"call"), "function")
    _check(cu.cuModuleGetFunction(ctypes.byref(guard), mod, b"guard"), "function")
    return mod, fn, guard


def _attr(fn, a):
    """cuFuncGetAttribute (3 local bytes per thread, 4 registers per thread)."""
    v = ctypes.c_int()
    _check(_cuda().cuFuncGetAttribute(ctypes.byref(v), a, fn), "attribute")
    return v.value


class CudaSolver:
    """CudaSolver(net, device)(x, T, P, dt, h) -> (x, h, done, accepted steps): the call
    advance_compact(rodas3_step(pivot=False), x, TorchRates(T, P), dt, h, order=3) for x [B, S] mole fractions in the
    network's species order, T [B] K, P [B] dyn cm^-2, dt [B] s and starting step sizes h [B] (float64 CUDA tensors).
    fallback: the counts of the last call's guard (cells with non-finite inputs; kernel cells with a non-finite state,
    an unfinished call, an element drift; cells recomputed by the library path); recomputed_total: cells recomputed
    since construction; info: the kernel's layout (sizes, warps per cell, leading / trailing split, registers and
    local bytes per thread, cells and warps per block, shared memory per block, resident cells per SM). Calls run one at
    a time on torch's current stream (the queue counter and the rate scratch are shared)."""

    def __init__(self, net, device):
        self.dev = torch.device(device)
        torch.zeros(1, device=self.dev)                       # torch's context is current from here on
        prop = torch.cuda.get_device_properties(self.dev)
        optin = prop.shared_memory_per_block_optin
        fit = lambda t: (optin - len(_blob(t)[0])) // (8 * _perc(t))     # cells per block the shared memory holds
        tab = tables(net, 2)
        if warps(fit(tab)) != 2:
            tab = tables(net, warps(fit(tab)))
        self.ns, self.nr, self.w = tab["ns"], tab["nr"], tab["w"]
        nl = 32 * self.w
        blob, _ = _blob(tab)
        if fit(tab) < 1:
            raise ValueError(f"one cell needs {8 * _perc(tab) + len(blob)} B of shared memory per block; the device "
                             f"has {optin}")
        # cells per block: from the registers of a one-cell compile, then more while the shared memory and the named
        # barriers (15) allow them and the compiler fits them in the registers without more local memory
        arch = f"{prop.major}{prop.minor}"
        top = min(1024 // nl, 15, fit(tab))
        mod = _load(source(tab, 1), arch)
        regs, local = -(-_attr(mod[1], 4) // 8) * 8, _attr(mod[1], 3)   # registers (allocation unit 8), local bytes
        self.cpb = max(1, min(top, prop.regs_per_multiprocessor // (nl * regs)))
        if self.cpb > 1:
            _check(_cuda().cuModuleUnload(mod[0]), "unload")
            mod = _load(source(tab, self.cpb), arch)
        while self.cpb < top:
            nxt = _load(source(tab, self.cpb + 1), arch)
            if _attr(nxt[1], 3) > local:
                _check(_cuda().cuModuleUnload(nxt[0]), "unload")
                break
            _check(_cuda().cuModuleUnload(mod[0]), "unload")
            mod, self.cpb = nxt, self.cpb + 1
        self._mod, self._fn, self._guard = mod
        self.wpb = self.w * self.cpb                          # warps per block
        self.smem = 8 * self.cpb * _perc(tab) + len(blob)
        self.src = source(tab, self.cpb)
        cu = _cuda()
        _check(cu.cuFuncSetAttribute(self._fn, 8, self.smem), "shared memory")   # MAX_DYNAMIC_SHARED_SIZE_BYTES
        occ = ctypes.c_int()
        _check(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(occ), self._fn, nl * self.cpb, self.smem),
               "occupancy")
        if occ.value < 1:
            raise RuntimeError("the kernel does not fit on this device")
        self.info = dict(ns=self.ns, nr=self.nr, nnz=tab["nnz"], k0=tab["k0"], nt=self.ns - tab["k0"], w=self.w,
                         sround=tab["sst"].shape[0], jround=tab["jround"], lu_ops=len(tab["lu"]), alias=_alias(tab),
                         regs=_attr(self._fn, 4), spill_bytes=_attr(self._fn, 3), cpb=self.cpb, wpb=self.wpb,
                         smem=self.smem, blob=len(blob), cells_per_sm=self.cpb * occ.value)
        self.max_blocks = prop.multi_processor_count * occ.value
        f = lambda a, view=None: torch.tensor(a if view is None else np.asarray(a).view(view), device=self.dev)
        self._tabs = [f(tab["perm"]), f(np.frombuffer(blob, np.int32).copy()), f(tab["rc"]), f(tab["rf"]),
                      f(tab["nasa"])]
        self._scratch = torch.empty(self.max_blocks * self.cpb * (2 * self.nr + 2), dtype=torch.float64,
                                    device=self.dev)
        self._streams = [f(tab["sst"].ravel(), np.int32), f(tab["jt"], np.int64), f(tab["lu"], np.int64),
                         f(tab["optr"]), ctypes.c_int(tab["sst"].shape[0]), ctypes.c_int(tab["jround"])]
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
        nb = max(1, min(-(-B // self.cpb), self.max_blocks))
        self.count.zero_()
        stream = torch.cuda.current_stream(self.dev).cuda_stream
        _launch(self._fn, nb, 32 * self.w * self.cpb, self.smem, stream,
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
