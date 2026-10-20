"""Re-match of hidden species to reservoir totals: x_i = h_i exp(W_i . lam) with W^T x = b, by Newton on the convex
dual F(lam) = sum_i h_i exp(W_i . lam) - b . lam (targets scaled to 1, Jacobi-scaled Hessian, Armijo backtracking).
Infeasible targets (outside the cone of the hidden compositions) make lam diverge: reported as a residual.
cols: subset of the 5 totals to match (default all); fixed: entries held at their value."""
import numpy as np


def match(h, Wh, b, fixed=None, iters=400, cols=None):
    h = np.maximum(np.asarray(h, np.float64), 1e-30)
    b = np.asarray(b, np.float64)
    fr = np.zeros(h.size, bool) if fixed is None else np.asarray(fixed, bool)
    free = np.logical_not(fr)
    cols = np.arange(Wh.shape[1]) if cols is None else np.asarray(cols)
    bb = b - (h[fr, None] * Wh[fr]).sum(0)
    if (bb[cols] <= 1e-14 * np.abs(b[cols])).any():
        return None, np.inf
    A = Wh[free][:, cols] / bb[cols]           # scaled: target A^T x = 1
    lh = np.log(h[free])
    lam = np.zeros(len(cols))

    def F(l):
        z = lh + A @ l
        if z.max() > 700:
            return np.inf, None
        x = np.exp(z)
        return x.sum() - l.sum(), x
    f, x = F(lam)
    for _ in range(iters):
        g = A.T @ x - 1.0
        if np.abs(g).max() < 1e-13:
            break
        H = (A.T * x) @ A
        d = np.sqrt(np.maximum(np.diag(H), 1e-300))
        Hs = H / d[:, None] / d[None, :]
        dl = -np.linalg.solve(Hs + 1e-12 * np.eye(len(cols)), g / d) / d
        t = 1.0
        while True:
            fn, xn = F(lam + t * dl)
            if (xn is not None and fn <= f + 1e-4 * t * (g @ dl)) or t < 1e-14:
                break
            t *= 0.5
        if xn is None:
            break
        lam, f, x = lam + t * dl, fn, xn
    out = h.copy()
    out[free] = x
    res = float(np.abs((Wh[:, cols].T @ out) / b[cols] - 1).max())
    return out, res
