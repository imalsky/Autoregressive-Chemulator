"""Batched stiff integration of a mass-action network at constant T and P (torch, float64), one cell per row.

State: N [B, S], the molecules of each species per starting molecule (sum N = 1 at the start of a call). The concentrations
are n = n_tot N / sum N with n_tot = P / kT fixed (the reactor of lib/kinetics.py and VULCAN), so
    dN/dt = (sum N / n_tot) S v(n),   v_r = k_f,r prod n_reactants - k_r,r prod n_products   (x n_tot for three-body),
and every element total E N is a linear invariant. The integrator is the two-stage, second-order, L-stable Rosenbrock
method ROS2 (gamma = 1 + 1/sqrt 2) with the exact Jacobian; its increments lie in the range of S, so E N is kept to
rounding. Step-size control per cell from the embedded first-order solution; a step that leaves a species below -atol is
rejected; accepted states are clamped at 0 (the clamped amount is returned).
"""
import math

import numpy as np
import torch

GAMMA = 1.0 + 1.0 / math.sqrt(2.0)


class Mech:
    """Stoichiometry and index lists of a kinetics.Network on a device."""

    def __init__(self, net, device):
        S = torch.tensor(net.S, dtype=torch.float64, device=device)
        self.S, self.n_sp, self.n_r = S, S.shape[0], S.shape[1]
        self.tb = torch.tensor(net._tb, device=device)
        self.idx = []  # (index list [R, slots], sign): reactants +1, products -1
        for lst, sgn in ((net._re, 1.0), (net._pr, -1.0)):
            self.idx.append((torch.tensor(lst, device=device), sgn))
        # one-hot of each slot's species (the padding slot n_sp maps to a dropped column)
        self.hot = [[torch.nn.functional.one_hot(i[:, m], self.n_sp + 1).to(torch.float64)[:, :self.n_sp]
                     for m in range(i.shape[1])] for i, _ in self.idx]
        # sparse Jacobian map: entry (p, q) of d(S v)/dn gets S[p, r] * dv_r/dn_q from every slot of reaction r holding q
        src, dst, coef = [], [], []
        Snp = net.S
        col = 0
        for lst, _ in ((net._re, 1.0), (net._pr, -1.0)):
            for m_ in range(lst.shape[1]):
                for r_ in range(self.n_r):
                    q = int(lst[r_, m_])
                    if q < self.n_sp:
                        for p_ in np.flatnonzero(Snp[:, r_]):
                            src.append(col * self.n_r + r_)
                            dst.append(int(p_) * self.n_sp + q)
                            coef.append(Snp[p_, r_])
                col += 1
        self.j_src = torch.tensor(src, device=device)
        self.j_dst = torch.tensor(dst, device=device)
        self.j_coef = torch.tensor(coef, dtype=torch.float64, device=device)


def _prods(ne, i):
    return ne[:, i].prod(-1)  # ne [B, S + 1], i [R, slots] -> [B, R]


def rhs_jac(m, N, kf, kr, nd, jac=True):
    """dN/dt [B, S] and its Jacobian [B, S, S]."""
    ntot = N.sum(1, keepdim=True)
    x = N / ntot
    ne = torch.cat([x * nd[:, None], torch.ones_like(ntot)], 1)
    c = torch.where(m.tb[None], nd[:, None], torch.ones_like(kf))
    (ri, _), (pi, _) = m.idx
    v = c * (kf * _prods(ne, ri) - kr * _prods(ne, pi))
    sv = v @ m.S.T                                            # S v [B, S]
    f = sv * (ntot / nd[:, None])
    if not jac:
        return f, None
    dv = []                                                   # dv_r / dn_(slot species), slot-major [B, slots * R]
    for (i, sgn), k in zip(m.idx, (kf, kr)):
        for s in range(i.shape[1]):
            keep = [q for q in range(i.shape[1]) if q != s]
            dv.append(sgn * c * k * ne[:, i[:, keep]].prod(-1))
    dv = torch.cat(dv, 1)
    Jn = torch.zeros(N.shape[0], m.n_sp * m.n_sp, dtype=N.dtype, device=N.device)
    Jn = Jn.index_add(1, m.j_dst, dv[:, m.j_src] * m.j_coef).view(-1, m.n_sp, m.n_sp)   # d(S v)/dn
    # J = (S v / n_tot) 1^T + Jn (I - x 1^T)
    J = (sv / nd[:, None])[:, :, None] + Jn - (Jn @ x[:, :, None])
    return f, J


def integrate(m, x0, kf, kr, nd, dt, rtol=1e-4, atol=1e-20, h0=1e-10, max_steps=20000):
    """Mixing ratios after dt [B] seconds from x0 [B, S] (rows summing to 1); kf, kr [B, R] rate coefficients (three-body
    without n_tot), nd [B] = P / kT. Returns (x [B, S], info dict: steps, rejected, clamped (largest clamped amount),
    done (reached dt))."""
    B = x0.shape[0]
    N = x0.clone()
    t = torch.zeros(B, dtype=N.dtype, device=N.device)
    h = torch.minimum(torch.full_like(t, h0), dt)
    eye = torch.eye(m.n_sp, dtype=N.dtype, device=N.device)
    steps = torch.zeros(B, dtype=torch.long, device=N.device)
    rej = torch.zeros_like(steps)
    clamped = torch.zeros_like(t)
    active = t < dt
    for _ in range(max_steps):
        a = active.nonzero().squeeze(1)
        if a.numel() == 0:
            break
        Na, ha = N[a], torch.minimum(h[a], dt[a] - t[a])
        f, J = rhs_jac(m, Na, kf[a], kr[a], nd[a])
        A = eye[None] - (GAMMA * ha)[:, None, None] * J
        LU, piv = torch.linalg.lu_factor(A)
        k1 = torch.linalg.lu_solve(LU, piv, (ha[:, None] * f)[:, :, None]).squeeze(2)
        f2, _ = rhs_jac(m, Na + k1, kf[a], kr[a], nd[a], jac=False)
        k2 = torch.linalg.lu_solve(LU, piv, (ha[:, None] * f2 - 2.0 * k1)[:, :, None]).squeeze(2)
        Nn = Na + 1.5 * k1 + 0.5 * k2
        err = 0.5 * (k1 + k2)
        sc = atol + rtol * torch.maximum(Na.abs(), Nn.abs())
        en = ((err / sc) ** 2).mean(1).sqrt()
        ok = (en <= 1.0) & (Nn.min(1).values >= -atol) & torch.isfinite(en)
        # accepted: advance, clamp
        cl = torch.clamp(-Nn, min=0.0).max(1).values
        Nn = torch.clamp(Nn, min=0.0)
        N[a] = torch.where(ok[:, None], Nn, Na)
        t[a] = torch.where(ok, t[a] + ha, t[a])
        clamped[a] = torch.where(ok, torch.maximum(clamped[a], cl), clamped[a])
        steps[a] += ok.long()
        rej[a] += (~ok).long()
        fac = torch.where(torch.isfinite(en), 0.9 * en.clamp(min=1e-10) ** -0.5, torch.full_like(en, 0.1))
        fac = fac.clamp(0.2, 5.0)
        fac = torch.where(ok, fac, fac.clamp(max=0.5))
        h[a] = ha * fac
        active = t < dt * (1.0 - 1e-14)
    x = N / N.sum(1, keepdim=True)
    return x, dict(steps=steps, rejected=rej, clamped=clamped, done=~active)


def ros2_step(m, N, kf, kr, nd, h):
    """One ROS2 step of size h [B] from N [B, S] with the exact Jacobian; returns (new N, error estimate)."""
    f, J = rhs_jac(m, N, kf, kr, nd)
    eye = torch.eye(m.n_sp, dtype=N.dtype, device=N.device)
    LU, piv = torch.linalg.lu_factor(eye[None] - (GAMMA * h)[:, None, None] * J)
    k1 = torch.linalg.lu_solve(LU, piv, (h[:, None] * f)[:, :, None]).squeeze(2)
    f2, _ = rhs_jac(m, N + k1, kf, kr, nd, jac=False)
    k2 = torch.linalg.lu_solve(LU, piv, (h[:, None] * f2 - 2.0 * k1)[:, :, None]).squeeze(2)
    return N + 1.5 * k1 + 0.5 * k2, 0.5 * (k1 + k2)


def integrate_fixed(m, x0, rates, nd, dt, n_steps=16, ratio=2.0):
    """Mixing ratios after dt [B] in n_steps geometric ROS2 steps (each ratio x the last; the same work for every cell).
    rates(x) -> (kf, kr) [B, R] is evaluated at the start of every step and held within it, so every step is a mass-action
    step with consistent equilibria whatever rates returns. Negative entries are set to 0 after each step."""
    w = ratio ** torch.arange(n_steps, dtype=x0.dtype, device=x0.device)
    w = w / w.sum()
    N = x0
    for i in range(n_steps):
        kf, kr = rates(N / N.sum(1, keepdim=True))
        N, _ = ros2_step(m, N, kf, kr, nd, dt * w[i])
        N = torch.clamp(N, min=0.0)
    return N / N.sum(1, keepdim=True)


def ros2_step_fn(m, N, rates, nd, h):
    """One ROS2 step with state-dependent rate coefficients: rates(N) -> (kf, kr) is evaluated at the step's start (with
    the Jacobian at those rates) and again at the second stage's state."""
    kf, kr = rates(N)
    f, J = rhs_jac(m, N, kf, kr, nd)
    eye = torch.eye(m.n_sp, dtype=N.dtype, device=N.device)
    LU, piv = torch.linalg.lu_factor(eye[None] - (GAMMA * h)[:, None, None] * J)
    k1 = torch.linalg.lu_solve(LU, piv, (h[:, None] * f)[:, :, None]).squeeze(2)
    N1 = N + k1
    kf2, kr2 = rates(torch.clamp(N1, min=0.0))
    f2, _ = rhs_jac(m, N1, kf2, kr2, nd, jac=False)
    k2 = torch.linalg.lu_solve(LU, piv, (h[:, None] * f2 - 2.0 * k1)[:, :, None]).squeeze(2)
    return N + 1.5 * k1 + 0.5 * k2, 0.5 * (k1 + k2)


# Rodas3 (4 stages, order 3, L-stable, stiffly accurate) in the W = I / (gamma h) - J form:
# K_i = W^-1 (f(Y_i) + sum_j C_ij / h K_j), Y_i = N + sum_j A_ij K_j, N' = N + sum_i M_i K_i, error = sum_i E_i K_i.
RODAS3 = dict(g=0.5, A=[[], [0.0], [2.0, 0.0], [2.0, 0.0, 1.0]], C=[[], [4.0], [1.0, -1.0], [1.0, -1.0, -8.0 / 3.0]],
              newf=[True, False, True, True], M=[2.0, 0.0, 1.0, 1.0], E=[0.0, 0.0, 0.0, 1.0])


def rodas3_step(m, N, kf, kr, nd, h, pivot=True):
    """One Rodas3 step of size h [B] with the exact Jacobian; returns (new N, error estimate of order 2). pivot=False
    factors without row exchanges (GPU only; the step control rejects a step whose result is not finite)."""
    T = RODAS3
    f0, J = rhs_jac(m, N, kf, kr, nd)
    eye = torch.eye(m.n_sp, dtype=N.dtype, device=N.device)
    W = eye[None] / (T["g"] * h)[:, None, None] - J
    if pivot:
        LU, piv, _ = torch.linalg.lu_factor_ex(W)
        solve = lambda b: torch.linalg.lu_solve(LU, piv, b[:, :, None]).squeeze(2)
    else:
        LU, _, _ = torch.linalg.lu_factor_ex(W, pivot=False)
        solve = lambda b: torch.linalg.solve_triangular(
            LU, torch.linalg.solve_triangular(LU, b[:, :, None], upper=False, unitriangular=True), upper=True).squeeze(2)
    K, F = [], f0
    for i in range(4):
        if i > 0 and T["newf"][i]:
            Y = N + sum(T["A"][i][j] * K[j] for j in range(i) if T["A"][i][j] != 0.0)
            F, _ = rhs_jac(m, Y, kf, kr, nd, jac=False)
        rhs = F + sum((T["C"][i][j] / h)[:, None] * K[j] for j in range(i)) if i > 0 else F
        K.append(solve(rhs))
    return N + 2.0 * K[0] + K[2] + K[3], K[3]
