"""(lib copy of fixes-phase checks/solver_speed/struct_lm.py, 2026-10-12; tests: checks/solver_speed/struct_test.txt, ctest*.txt)
Structured single solver for the closed head's element-ratio projection (fixes phase 2026-10-12; author: one accurate
solver for every row if it costs a few us; Codex astra 1012-solver-speed: eliminate He, group species, scalar-capped LM).
Same solution family and equations as elemproj.project_ln: ln y' = ln y + na lam, ratios C/H, N/H, O/H, He/H = the
input's, then normalised over molecules. Species with the same (C, N, O) atom counts share the factor
exp(c lam_C + n lam_N + o lam_O), so the element totals are log-sum-exps over about 10 groups:
  ln E_e(lam) = lse_k [ ln(e_k) + G_k + p_k . lam ],   ln H(lam) = lse_k [ Hg_k + p_k . lam ],
G_k = lse of ln y over group k, Hg_k = lse of ln y + ln h over group k (both once per call). He is exact:
lam_He = lr_He + ln H(lam) - ln y_He. Levenberg-Marquardt on the three C/N/O residuals, the step scaled as a whole
(direction kept) to at most `cap` in max norm, accepted only if |F|^2 falls; explicit 3x3 Cholesky; fixed iterations."""
import torch


class StructLM:
    def __init__(self, na, nh, mol, iters=32, cap=2.0, d0=1e-3, dec=0.3, inc=10.0):
        dev, dt = na.device, na.dtype
        self.iters, self.cap, self.mol, self.d0, self.dec, self.inc = iters, cap, mol, d0, dec, inc
        self.na, self.nh = na, nh
        he = torch.nonzero(na[:, 3] > 0).squeeze(1)
        assert he.numel() == 1 and float(na[he, 3]) == 1.0 and float(nh[he]) == 0.0 and float(na[he, :3].abs().sum()) == 0.0
        self.he = int(he)
        cno = na[:, :3]
        keys = sorted({tuple(int(v) for v in r) for i, r in enumerate(cno.tolist()) if i != self.he})
        self.P = torch.tensor(keys, dtype=dt, device=dev)                                  # [K, 3]
        K, S = len(keys), na.shape[0]
        memb = torch.zeros(K, S, dtype=torch.bool, device=dev)
        for i in range(S):
            if i != self.he:
                memb[keys.index(tuple(int(v) for v in cno[i].tolist())), i] = True
        self.lmemb = torch.where(memb, 0.0, float("-inf")).to(dt)                          # [K, S]
        self.lh = torch.log(nh).to(dt)                                                     # -inf where no H
        self.lce = torch.log(self.P.T.clone())                                             # [3, K] ln(atoms of e in group k)
        self.eye = torch.eye(3, dtype=dt, device=dev)

    def totals(self, G, Hg, lam):
        t = G.unsqueeze(1) + self.lce.unsqueeze(0) + (lam @ self.P.T).unsqueeze(1)        # [B, 3, K]
        u = Hg + lam @ self.P.T                                                           # [B, K]
        lE = torch.logsumexp(t, -1)                                                       # [B, 3]
        lH = torch.logsumexp(u, -1)                                                       # [B]
        return t, u, lE, lH

    def __call__(self, ln_y, lr):
        G = torch.logsumexp(ln_y.unsqueeze(1) + self.lmemb, -1)                           # [B, K]
        Hg = torch.logsumexp((ln_y + self.lh).unsqueeze(1) + self.lmemb, -1)              # [B, K]
        lr3 = lr[:, :3]
        lam = torch.zeros(ln_y.shape[0], 3, dtype=ln_y.dtype, device=ln_y.device)
        damp = torch.full(ln_y.shape[:1], self.d0, dtype=ln_y.dtype, device=ln_y.device)
        t, u, lE, lH = self.totals(G, Hg, lam)
        F = lE - lH.unsqueeze(-1) - lr3
        for _ in range(self.iters):
            pe = torch.softmax(t, -1)                                                    # [B, 3, K]
            ph = torch.softmax(u, -1)                                                    # [B, K]
            J = pe @ self.P - (ph @ self.P).unsqueeze(1)                                 # [B, 3, 3]
            Jt = J.transpose(-1, -2)
            A = Jt @ J + damp.reshape(-1, 1, 1) * self.eye
            g = (Jt @ F.unsqueeze(-1)).squeeze(-1)
            step = -chol3_solve(A, g)
            step = torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
            step = step * torch.clamp(self.cap / step.abs().amax(-1, keepdim=True).clamp(min=1e-300), max=1.0)
            t2, u2, lE2, lH2 = self.totals(G, Hg, lam + step)
            F2 = lE2 - lH2.unsqueeze(-1) - lr3
            better = (F2 * F2).sum(-1) < (F * F).sum(-1)
            b1 = better.unsqueeze(-1)
            lam = torch.where(b1, lam + step, lam)
            t, u = torch.where(b1.unsqueeze(-1), t2, t), torch.where(b1, u2, u)
            lH, F = torch.where(better, lH2, lH), torch.where(b1, F2, F)
            damp = torch.where(better, damp * self.dec, damp * self.inc).clamp(1e-15, 1e12)
        lam_he = lr[:, 3] + lH - ln_y[:, self.he]
        lam4 = torch.cat([lam, lam_he.unsqueeze(-1)], -1)
        w = ln_y + lam4 @ self.na.T
        import elemproj as EP
        res = (EP.log_ratios(w, self.na, self.nh) - lr).abs().amax(-1)
        return w - torch.logsumexp(torch.index_select(w, -1, self.mol), -1, keepdim=True), res


def chol3_solve(A, b):
    """A [..., 3, 3] SPD, b [..., 3] -> x; explicit unrolled Cholesky (elementwise ops)."""
    a = lambda i, j: A[..., i, j]
    tiny = 1e-300
    l00 = torch.sqrt(torch.clamp(a(0, 0), min=tiny))
    l10, l20 = a(1, 0) / l00, a(2, 0) / l00
    l11 = torch.sqrt(torch.clamp(a(1, 1) - l10 * l10, min=tiny))
    l21 = (a(2, 1) - l20 * l10) / l11
    l22 = torch.sqrt(torch.clamp(a(2, 2) - l20 * l20 - l21 * l21, min=tiny))
    y0 = b[..., 0] / l00
    y1 = (b[..., 1] - l10 * y0) / l11
    y2 = (b[..., 2] - l20 * y0 - l21 * y1) / l22
    x2 = y2 / l22
    x1 = (y1 - l21 * x2) / l11
    x0 = (y0 - l10 * x1 - l20 * x2) / l00
    return torch.stack([x0, x1, x2], -1)
