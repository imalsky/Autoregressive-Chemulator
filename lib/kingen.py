"""Code generation for the ROS2 step of a fixed mass-action network (in the manner of KPP): the right-hand side, the sparse
Jacobian, a sparse LU factorization without pivoting on the static fill pattern and the triangular solves are written out
as straight-line operations on per-species [B] tensors, so a compiler can fuse a whole step into few GPU kernels.

The step is the one of kinbatch.ros2_step (constant T, P; N = molecules per starting molecule): J = Jn + w 1^T with
Jn = d(S v)/dn (sparse) and w = S v / n_tot - Jn x (the constant-P coupling, rank one); (I - gamma h J) is solved by
Sherman-Morrison from the sparse factorization of M0 = I - gamma h Jn. Without pivoting a pivot can be small or zero; the
caller's step control rejects any step whose result is not finite (smaller h moves M0 toward I).
    src, fn = generate(mech_like_network); step = fn  # step(N, h, kf, kr, nd) -> (N_new, err), same as ros2_step
"""
import math

import numpy as np

GAMMA = 1.0 + 1.0 / math.sqrt(2.0)


def _ordering(pattern, n):
    """Greedy minimum-degree elimination order on the symmetrized pattern."""
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


def generate(net):
    S = np.asarray(net.S)
    n_sp, n_r = S.shape
    re = [list(r["re"]) for r in net.reac]
    pr = [list(r["pr"]) for r in net.reac]
    tb = [bool(t) for t in net._tb]
    L = []
    w = L.append
    w("def step(N, h, kf, kr, nd):")
    w("    import torch")
    w("    y = N.unbind(1)")
    w("    ntot = N.sum(1)")
    w("    xf = nd / ntot")
    for i in range(n_sp):
        w(f"    n{i} = y[{i}] * xf")
    w("    kfs = kf.unbind(1)")
    w("    krs = kr.unbind(1)")
    # fluxes and their derivatives d v_r / d n_q
    dv = {}
    for r in range(n_r):
        if all(abs(S[:, r]) == 0):
            continue
        c = " * nd" if tb[r] else ""
        f = " * ".join([f"kfs[{r}]"] + [f"n{q}" for q in re[r]])
        b = " * ".join([f"krs[{r}]"] + [f"n{q}" for q in pr[r]])
        w(f"    v{r} = ({f} - {b}){c}")
        for side, k, sgn in ((re[r], "kfs", 1.0), (pr[r], "krs", -1.0)):
            for m_ in range(len(side)):
                q = side[m_]
                rest = [f"n{side[o]}" for o in range(len(side)) if o != m_]
                term = " * ".join([f"{k}[{r}]"] + rest) + c
                dv.setdefault((r, q), []).append((sgn, term))
    for (r, q), terms in dv.items():
        expr = " ".join(("+ " if s > 0 else "- ") + t for s, t in terms)
        w(f"    d{r}_{q} = {expr}")
    # S v and f = (n_tot / nd) S v
    for i in range(n_sp):
        terms = [(S[i, r], f"v{r}") for r in range(n_r) if S[i, r] != 0]
        expr = " ".join(f"{'+' if c > 0 else '-'} {abs(c):g} * {t}" for c, t in terms) if terms else "0.0 * ntot"
        w(f"    sv{i} = {expr}")
        w(f"    f{i} = sv{i} / xf")
    # Jn entries
    jn = {}
    for (r, q) in dv:
        for p in np.flatnonzero(S[:, r]):
            jn.setdefault((int(p), q), []).append((S[p, r], f"d{r}_{q}"))
    for (p, q), terms in jn.items():
        expr = " ".join(f"{'+' if c > 0 else '-'} {abs(c):g} * {t}" for c, t in terms)
        w(f"    J{p}_{q} = {expr}")
    # w = sv / nd - Jn x  (x = y / ntot)
    for i in range(n_sp):
        terms = [f"J{i}_{q} * y[{q}]" for (p, q) in jn if p == i]
        w(f"    w{i} = sv{i} / nd - ({' + '.join(terms) if terms else '0.0'}) / ntot")
    w("    gh = 1.7071067811865475 * h")
    # M0 = I - gh Jn, sparse LU without pivoting in a fill-reducing order
    pat = set(jn) | {(i, i) for i in range(n_sp)}
    order = _ordering(pat, n_sp)
    pos = {s: k for k, s in enumerate(order)}
    P = {(pos[i], pos[j]) for i, j in pat}
    for (i, j) in sorted(P):
        oi, oj = order[i], order[j]
        base = "1.0" if i == j else "0.0"
        if (oi, oj) in jn:
            w(f"    M{i}_{j} = {base} - gh * J{oi}_{oj}")
        else:
            w(f"    M{i}_{j} = {base} + 0.0 * gh")
    for k in range(n_sp):
        rows = sorted(i for (i, j) in P if j == k and i > k)
        cols = sorted(j for (i, j) in P if i == k and j > k)
        if rows:
            w(f"    p{k} = 1.0 / M{k}_{k}")
        for i in rows:
            w(f"    M{i}_{k} = M{i}_{k} * p{k}")
            for j in cols:
                if (i, j) in P:
                    w(f"    M{i}_{j} = M{i}_{j} - M{i}_{k} * M{k}_{j}")
                else:
                    P.add((i, j))
                    w(f"    M{i}_{j} = - M{i}_{k} * M{k}_{j}")
    # solve M0 z = b (b given in species order), result in species order
    def solve(b, z):
        for k in range(n_sp):
            terms = [f"M{k}_{j} * {z}{j}" for j in range(k) if (k, j) in P]
            w(f"    {z}{k} = {b}{order[k]}" + (" - (" + " + ".join(terms) + ")" if terms else ""))
        for k in reversed(range(n_sp)):
            terms = [f"M{k}_{j} * {z}{j}" for j in range(k + 1, n_sp) if (k, j) in P]
            w(f"    {z}{k} = ({z}{k}" + (" - (" + " + ".join(terms) + ")" if terms else "") + f") / M{k}_{k}")
    # Sherman-Morrison for M = M0 - gh w 1^T: M^-1 b = z + gh u (1.z) / (1 - gh 1.u), u = M0^-1 w
    solve("w", "u")
    w("    su = " + " + ".join(f"u{k}" for k in range(n_sp)))
    w("    den = 1.0 - gh * su")
    for i in range(n_sp):
        w(f"    b{i} = h * f{i}")
    solve("b", "z")
    w("    sz = " + " + ".join(f"z{k}" for k in range(n_sp)))
    w("    c1 = gh * sz / den")
    for k in range(n_sp):
        w(f"    k1_{order[k]} = z{k} + c1 * u{k}")
    # second stage: f at N + k1 (same rate coefficients)
    w("    N1 = N + torch.stack([" + ", ".join(f"k1_{i}" for i in range(n_sp)) + "], 1)")
    w("    y1 = N1.unbind(1)")
    w("    ntot1 = N1.sum(1)")
    w("    xf1 = nd / ntot1")
    for i in range(n_sp):
        w(f"    m{i} = y1[{i}] * xf1")
    for r in range(n_r):
        if all(abs(S[:, r]) == 0):
            continue
        c = " * nd" if tb[r] else ""
        f = " * ".join([f"kfs[{r}]"] + [f"m{q}" for q in re[r]])
        b = " * ".join([f"krs[{r}]"] + [f"m{q}" for q in pr[r]])
        w(f"    u{r}_ = ({f} - {b}){c}")
    for i in range(n_sp):
        terms = [(S[i, r], f"u{r}_") for r in range(n_r) if S[i, r] != 0]
        expr = " ".join(f"{'+' if c > 0 else '-'} {abs(c):g} * {t}" for c, t in terms) if terms else "0.0 * ntot"
        w(f"    b{i} = h * ({expr}) / xf1 - 2.0 * k1_{i}")
    solve("b", "z")
    w("    sz = " + " + ".join(f"z{k}" for k in range(n_sp)))
    w("    c2 = gh * sz / den")
    for k in range(n_sp):
        w(f"    k2_{order[k]} = z{k} + c2 * u{k}")
    w("    K1 = torch.stack([" + ", ".join(f"k1_{i}" for i in range(n_sp)) + "], 1)")
    w("    K2 = torch.stack([" + ", ".join(f"k2_{i}" for i in range(n_sp)) + "], 1)")
    w("    return N + 1.5 * K1 + 0.5 * K2, 0.5 * (K1 + K2)")
    src = "\n".join(L)
    ns = {}
    exec(compile(src, "<kingen>", "exec"), ns)
    return src, ns["step"]
