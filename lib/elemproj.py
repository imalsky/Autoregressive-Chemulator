"""Element-ratio projection for complete entry sets (every network species or reservoir, He included), shared by the
closed head (models.RelaxClosedRHead) and the training labels (data.RawSplit.logy_at).

project_ln(ln_y, lr, na, nh): ln y [..., S] (natural log mixing ratios), lr [..., 4] target log ratios to H of
(C, N, O, He); returns ln y' = ln y + na lam - logsumexp(.), the multiplicative change y exp(na lam) that meets the
ratios (lam [..., 4] from damped Newton, unrolled, fp64), normalised so sum y' = 1 (over the entries in mol), plus the
final max |residual| in log-ratio units per state (to detect non-convergence). It makes the training labels: its
arithmetic must stay as it is. project_ln_fused: the same mathematics for compiled training code. project_lm: the same
solution family by Levenberg-Marquardt, the rescue for rows where Newton fails.
"""
import re

import torch

ELEMS = ("C", "N", "O", "He")


def stoich(names, dtype=torch.float64):
    """H atoms [S] and (C, N, O, He) atoms [S, 4] per molecule, from the formulas."""
    n = lambda s, e: sum(int(k or 1) for el, k in re.findall(r"([A-Z][a-z]?)(\d*)", s) if el == e)
    h = torch.tensor([n(s, "H") for s in names], dtype=dtype)
    a = torch.tensor([[n(s, e) for e in ELEMS] for s in names], dtype=dtype)
    return h, a


def log_ratios(ln_y, na, nh):
    la = torch.log(na)
    return torch.logsumexp(ln_y.unsqueeze(-1) + la, -2) - torch.logsumexp(ln_y + torch.log(nh), -1).unsqueeze(-1)


def molidx(names):
    """Indices of the entries counted in sum y = 1: every name but the atom reservoirs C_res, N_res, O_res, H_res
    (reservoir sets: atoms of each element in the untracked species; M_res, the untracked molecule count, is counted)."""
    return torch.tensor([k for k, s in enumerate(names) if s not in ("C_res", "N_res", "O_res", "H_res")])


def project_ln(ln_y, lr, na, nh, iters=12, clip=2.0, mol=None):
    """mol: optional long [M], indices of the entries normalised to sum 1 (default all; index_select: graph-safe)."""
    la, lh = torch.log(na), torch.log(nh)
    lam = torch.zeros(lr.shape, dtype=ln_y.dtype, device=ln_y.device)
    for _ in range(iters):
        w = ln_y + lam @ na.T
        le = torch.logsumexp(w.unsqueeze(-1) + la, -2)
        lhs = torch.logsumexp(w + lh, -1, keepdim=True)
        F = le - lhs - lr
        pe = torch.exp(w.unsqueeze(-1) + la - le.unsqueeze(-2))
        q = torch.exp(w + lh - lhs)
        J = torch.einsum("...se,sf->...ef", pe, na) - (q @ na).unsqueeze(-2)
        step = torch.linalg.solve_ex(J, -F.unsqueeze(-1))[0].squeeze(-1)  # no error check: no host sync (CUDA graphs)
        step = torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)     # a singular solve must not poison lam or grads
        lam = lam + torch.clamp(step, -clip, clip)
    w = ln_y + lam @ na.T
    res = (log_ratios(w, na, nh) - lr).abs().amax(-1)
    return w - torch.logsumexp(w if mol is None else torch.index_select(w, -1, mol), -1, keepdim=True), res


PIVOT_TINY = 1e-100   # _solve4 divisors at or below this magnitude flag the row


def _solve4(J, b):
    """J [..., n, n] x = b [..., n] (n = 4) by unrolled Gaussian elimination with partial pivoting, elementwise torch ops
    only (no cuBLAS / cuSOLVER call), so torch.compile can fuse the whole solve into one kernel. Pivot: before eliminating
    column k, row k is swapped with each later row whose |entry k| is larger (strict >: the first maximum wins, as
    LAPACK's getrf). A pivot with |p| <= PIVOT_TINY (zero, subnormal or NaN) is replaced by 1 before it divides and
    flags its row; a flagged row returns step 0 (what nan_to_num makes of an LU solve's inf / NaN), and its backward
    stays finite (a division by 0 would give 0 * inf = NaN)."""
    n = J.shape[-1]
    a = [[J[..., i, j] for j in range(n)] + [b[..., i]] for i in range(n)]
    piv, bad = [None] * n, None
    for k in range(n):
        for r in range(k + 1, n):
            sw = a[r][k].abs() > a[k][k].abs()
            rk = a[k][:k] + [torch.where(sw, a[r][j], a[k][j]) for j in range(k, n + 1)]
            rr = a[r][:k] + [torch.where(sw, a[k][j], a[r][j]) for j in range(k, n + 1)]
            a[k], a[r] = rk, rr
        ok = a[k][k].abs() > PIVOT_TINY                     # NaN -> False
        bad = torch.logical_not(ok) if bad is None else bad | torch.logical_not(ok)
        piv[k] = torch.where(ok, a[k][k], torch.ones_like(a[k][k]))
        for r in range(k + 1, n):
            m = a[r][k] / piv[k]
            a[r] = a[r][:k + 1] + [a[r][j] - m * a[k][j] for j in range(k + 1, n + 1)]
    x = [None] * n
    for i in reversed(range(n)):
        s = a[i][n]
        for j in range(i + 1, n):
            s = s - a[i][j] * x[j]
        x[i] = s / piv[i]
    x = torch.stack(x, -1)
    return torch.where(bad.unsqueeze(-1), torch.zeros_like(x), x)


def project_ln_fused(ln_y, lr, na, nh, iters=12, clip=2.0, mol=None):
    """project_ln's mathematics written for torch.compile fusion (models.RelaxClosedRHead fused=True, training only). The four element log-sums and the H log-sum are one logsumexp over [..., 5, S] (rows C, N, O, He, then H);
    the Jacobian's two products (inner dim S) are one broadcast multiply-sum over [..., 5, S, 4]; lam @ na.T is a broadcast
    multiply-sum (inner dim 4); the 4x4 Newton solve is _solve4 (unrolled elimination, partial pivoting, no cuBLAS);
    nan_to_num, the step clamp, the residual and the mol normalisation as project_ln. Same returns; equal to project_ln up
    to rounding (summation order, solve arithmetic). Slower than project_ln in eager mode (many small kernels): meant
    for compiled code."""
    E = na.shape[-1]
    lall = torch.cat([torch.log(na).T, torch.log(nh).unsqueeze(0)], 0)          # [E + 1, S]
    lam = torch.zeros(lr.shape, dtype=ln_y.dtype, device=ln_y.device)
    for _ in range(iters):
        w = ln_y + (lam.unsqueeze(-2) * na).sum(-1)                             # lam @ na.T
        X = w.unsqueeze(-2) + lall                                              # [..., E + 1, S]
        L = torch.logsumexp(X, -1)                                              # (le, lhs)
        F = L[..., :E] - L[..., E:] - lr
        P = torch.exp(X - L.unsqueeze(-1))                                      # (pe^T, q)
        Jf = (P.unsqueeze(-1) * na).sum(-2)                                     # [..., E + 1, E]
        J = Jf[..., :E, :] - Jf[..., E:, :]
        step = _solve4(J, -F)
        step = torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
        lam = lam + torch.clamp(step, -clip, clip)
    w = ln_y + (lam.unsqueeze(-2) * na).sum(-1)
    res = (log_ratios(w, na, nh) - lr).abs().amax(-1)
    return w - torch.logsumexp(w if mol is None else torch.index_select(w, -1, mol), -1, keepdim=True), res


def project_lm(ln_y, lr, na, nh, iters=32, clip=2.0, mol=None):
    """project_ln's solution family (ln y + na lam, normalised), found by Levenberg-Marquardt on
    |F|^2: regularised normal equations (J^T J + d I) step = -J^T F, a step is taken only if it lowers |F|^2 (damping d
    x0.3 after a success, x10 after a failure, d in [1e-12, 1e12]), fixed iterations, no host sync. project_ln's clipped
    Newton can cycle (a componentwise clip of an ill-conditioned step is not a descent direction) on very large moves
    (first calls that change some entries by 13-17 dex); this solve converges there. Same returns as project_ln."""
    la, lh = torch.log(na), torch.log(nh)
    eye = torch.eye(lr.shape[-1], dtype=ln_y.dtype, device=ln_y.device)

    def resid(lam):
        w = ln_y + lam @ na.T
        le = torch.logsumexp(w.unsqueeze(-1) + la, -2)
        lhs = torch.logsumexp(w + lh, -1, keepdim=True)
        return w, le, lhs, le - lhs - lr

    lam = torch.zeros(lr.shape, dtype=ln_y.dtype, device=ln_y.device)
    damp = torch.full(lr.shape[:-1], 1e-3, dtype=ln_y.dtype, device=ln_y.device)
    w, le, lhs, F = resid(lam)
    for _ in range(iters):
        pe = torch.exp(w.unsqueeze(-1) + la - le.unsqueeze(-2))
        q = torch.exp(w + lh - lhs)
        J = torch.einsum("...se,sf->...ef", pe, na) - (q @ na).unsqueeze(-2)
        Jt = J.transpose(-1, -2)
        A = Jt @ J + damp.unsqueeze(-1).unsqueeze(-1) * eye
        step = -torch.linalg.solve_ex(A, Jt @ F.unsqueeze(-1))[0].squeeze(-1)
        step = torch.clamp(torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0), -clip, clip)
        w2, le2, lhs2, F2 = resid(lam + step)
        better = (F2 * F2).sum(-1) < (F * F).sum(-1)
        b1 = better.unsqueeze(-1)
        lam = torch.where(b1, lam + step, lam)
        w, le, lhs, F = torch.where(b1, w2, w), torch.where(b1, le2, le), torch.where(b1, lhs2, lhs), torch.where(b1, F2, F)
        damp = torch.where(better, damp * 0.3, damp * 10.0).clamp(1e-12, 1e12)
    res = (log_ratios(w, na, nh) - lr).abs().amax(-1)
    return w - torch.logsumexp(w if mol is None else torch.index_select(w, -1, mol), -1, keepdim=True), res
