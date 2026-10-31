"""Learned reaction rates inside the kinetic solver.

A small network maps the local state (log T, log P, log mixing ratios) to one log factor per reaction, bounded by
B tanh(o / B); the factor multiplies the reaction's forward and reverse rate coefficients together, so every step of the
solver keeps the elements, positivity and the equilibrium of the stock thermodynamics whatever the network outputs.

Training goes through the adaptive solver: a forward pass without gradients fixes each cell's accepted step sizes, then
the same steps are replayed with gradients (a zero step leaves a cell unchanged), so the gradient is the exact one of the
discrete solution the solver returns.
"""
import math

import torch

import kinbatch as KB

LX = 30.0   # log10 mixing ratios are floored at -LX


class RateNet(torch.nn.Module):
    def __init__(self, n_sp, n_r, width=128, depth=3, bound_dex=4.0):
        super().__init__()
        layers, d = [], n_sp + 2
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.SiLU()]
            d = width
        self.body = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(d, n_r)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)
        self.B = bound_dex * math.log(10.0)

    def forward(self, x, lT, lP):
        """x [B, S] mixing ratios (float64), lT, lP [B] log10 T / 1000 K and log10 P / bar -> log factors [B, R] (float64)."""
        lx = torch.log10(x.clamp(min=10.0 ** -LX))
        z = torch.cat([(lx + LX / 2) / (LX / 2), lT[:, None], lP[:, None] / 3.5], 1).float()
        o = self.out(self.body(z)).double()
        return self.B * torch.tanh(o / self.B)


def make_step(m, net):
    """step(N, h, kf0, kr0, nd, lT, lP) -> (N after one ROS2 step of size h with the network's factors, error estimate);
    every argument is a tensor, so one compiled graph serves every batch of the same size."""
    def step(N, h, kf0, kr0, nd, lT, lP):
        g = torch.exp(net(N / N.sum(1, keepdim=True), lT, lP))
        return KB.ros2_step(m, N, kf0 * g, kr0 * g, nd, h)
    return step


def adaptive_steps(step, x0, args, dt, rtol=1e-3, atol=1e-18, h0=1e-10, max_steps=4000):
    """Forward pass without gradients: the accepted step sizes [B, n] (zero-padded, in order) of the adaptive solver,
    the end state and whether each cell finished. args = the step's tensors after (N, h)."""
    with torch.no_grad():
        N = x0.clone()
        t = torch.zeros_like(dt)
        h = torch.minimum(torch.full_like(t, h0), dt)
        hist = []
        for i in range(max_steps):
            act = t < dt * (1.0 - 1e-14)
            if i % 16 == 0 and not bool(act.any()):
                break
            ha = torch.where(act, torch.minimum(h, dt - t), torch.zeros_like(h))
            Nn, err = step(N, ha, *args)
            sc = atol + rtol * torch.maximum(N.abs(), Nn.abs())
            en = ((err / sc) ** 2).mean(1).sqrt()
            ok = act & (en <= 1.0) & (Nn.min(1).values >= -atol) & torch.isfinite(en)
            N = torch.where(ok[:, None], torch.clamp(Nn, min=0.0), N)
            t = torch.where(ok, t + ha, t)
            hist.append(torch.where(ok, ha, torch.zeros_like(ha)))
            fac = torch.where(torch.isfinite(en), 0.9 * en.clamp(min=1e-10) ** -0.5, torch.full_like(en, 0.1)).clamp(0.2, 5.0)
            fac = torch.where(ok, fac, fac.clamp(max=0.5))
            h = torch.where(act, ha * fac, h)
        done = t >= dt * (1.0 - 1e-14)
        H = torch.stack(hist, 1)
        order = torch.argsort((H == 0).to(torch.int8), dim=1, stable=True)   # accepted steps first, in order
        H = torch.gather(H, 1, order)
        H = H[:, (H > 0).any(0)]
        return H, N / N.sum(1, keepdim=True), done


def replay(step, x0, args, H):
    """The solver's result with gradients: the accepted steps H [B, n] replayed (zero steps leave a cell unchanged)."""
    N = x0
    for i in range(H.shape[1]):
        N, _ = step(N, H[:, i], *args)
        N = torch.clamp(N, min=0.0)
    return N / N.sum(1, keepdim=True)


def advance(step, x0, args, dt, h, rtol=1e-3, atol=1e-18, max_steps=4000):
    """One call: mixing ratios after dt [B] from x0 with the adaptive solver (no gradients), starting from the step sizes h
    [B] (e.g. the last call's); returns (x, next step sizes, finished [B], accepted steps [B])."""
    with torch.no_grad():
        N = x0.clone()
        t = torch.zeros_like(dt)
        h = torch.minimum(h, dt).clamp(min=1e-12)
        n_ok = torch.zeros_like(dt)
        for i in range(max_steps):
            act = t < dt * (1.0 - 1e-14)
            if not bool(act.any()):
                break
            ha = torch.where(act, torch.minimum(h, dt - t), torch.zeros_like(h))
            Nn, err = step(N, ha, *args)
            sc = atol + rtol * torch.maximum(N.abs(), Nn.abs())
            en = ((err / sc) ** 2).mean(1).sqrt()
            ok = act & (en <= 1.0) & (Nn.min(1).values >= -atol) & torch.isfinite(en)
            N = torch.where(ok[:, None], torch.clamp(Nn, min=0.0), N)
            t = torch.where(ok, t + ha, t)
            n_ok = n_ok + ok.to(n_ok.dtype)
            fac = torch.where(torch.isfinite(en), 0.9 * en.clamp(min=1e-10) ** -0.5, torch.full_like(en, 0.1)).clamp(0.2, 5.0)
            fac = torch.where(ok, fac, fac.clamp(max=0.5))
            h = torch.where(act, ha * fac, h)
        done = t >= dt * (1.0 - 1e-14)
        return N / N.sum(1, keepdim=True), h, done, n_ok


class RateNet2(torch.nn.Module):
    """RateNet on the features of features(): log mixing ratios, log T, log P, each reaction's log ratio of forward to
    reverse flux at the stock rates (ln Q/K up to sign) and the log element ratios C/H, N/H, O/H."""

    def __init__(self, n_in, n_r, width=128, depth=3, bound_dex=6.0):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.SiLU()]
            d = width
        self.body = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(d, n_r)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)
        self.B = bound_dex * math.log(10.0)

    def forward(self, z):
        o = self.out(self.body(z.float())).double()
        return self.B * torch.tanh(o / self.B)


def features(m, atoms, iH, x, kf0, kr0, nd, lT, lP):
    """[B, S + 2 + R + 3] network input from mixing ratios x [B, S] and the stock rates (atoms [S, E]: element counts)."""
    lx = torch.log10(x.clamp(min=10.0 ** -LX))
    ne = torch.cat([torch.log(x.clamp(min=1e-300) * nd[:, None]), torch.zeros_like(nd)[:, None]], 1)
    (ri, _), (pi, _) = m.idx
    aff = (torch.log(kf0) + ne[:, ri].sum(-1)) - (torch.log(kr0) + ne[:, pi].sum(-1))
    el = x @ atoms
    er = torch.log10(el.clamp(min=1e-300) / el[:, iH:iH + 1]) 
    er = torch.cat([er[:, :iH], er[:, iH + 1:]], 1)
    er = er[:, [i for i in range(er.shape[1]) if i != 1]]   # drop He/H (inert)
    return torch.cat([(lx + LX / 2) / (LX / 2), lT[:, None], lP[:, None] / 3.5, aff.clamp(-60.0, 60.0) / 20.0,
                      (er + 3.5) / 1.5], 1)


def make_step2(m, net, atoms, iH):
    """step(N, h, kf0, kr0, nd, lT, lP) with RateNet2; the learned rates are evaluated at both ROS2 stages."""
    def step(N, h, kf0, kr0, nd, lT, lP):
        def rates(Nx):
            x = Nx / Nx.sum(1, keepdim=True)
            g = torch.exp(net(features(m, atoms, iH, x, kf0, kr0, nd, lT, lP)))
            return kf0 * g, kr0 * g
        return KB.ros2_step_fn(m, N, rates, nd, h)
    return step


def adaptive_compact(step, x0, args, dt, rtol=1e-3, atol=1e-18, h0=1e-10, max_iter=1000, cap=1000):
    """Forward pass without gradients on the still-active cells only: accepted step sizes H [B, n] (in order, zero-padded),
    the end state, whether each cell finished, and the number of rejected steps (error test, negativity)."""
    with torch.no_grad():
        B = x0.shape[0]
        N = x0.clone()
        t = torch.zeros_like(dt)
        h = torch.minimum(torch.full_like(t, h0), dt)
        H = torch.zeros(B, cap, dtype=N.dtype, device=N.device)
        cnt = torch.zeros(B, dtype=torch.long, device=N.device)
        rej_err = torch.zeros(B, dtype=torch.long, device=N.device)
        rej_neg = torch.zeros_like(rej_err)
        for _ in range(max_iter):
            a = ((t < dt * (1.0 - 1e-14)) & (cnt < cap)).nonzero().squeeze(1)
            if a.numel() == 0:
                break
            Na, ha = N[a], torch.minimum(h[a], dt[a] - t[a])
            Nn, err = step(Na, ha, *(q[a] for q in args))
            sc = atol + rtol * torch.maximum(Na.abs(), Nn.abs())
            en = ((err / sc) ** 2).mean(1).sqrt()
            e_ok = (en <= 1.0) & torch.isfinite(en)
            n_ok = Nn.min(1).values >= -atol
            ok = e_ok & n_ok
            rej_err[a] += (e_ok == 0).long()
            rej_neg[a] += (e_ok & (n_ok == 0)).long()
            acc = a[ok]
            H[acc, cnt[acc]] = ha[ok]
            cnt[acc] += 1
            N[acc] = torch.clamp(Nn[ok], min=0.0)
            t[acc] = t[acc] + ha[ok]
            fac = torch.where(torch.isfinite(en), 0.9 * en.clamp(min=1e-10) ** -0.5, torch.full_like(en, 0.1)).clamp(0.2, 5.0)
            fac = torch.where(ok, fac, fac.clamp(max=0.5))
            h[a] = ha * fac
        done = t >= dt * (1.0 - 1e-14)
        return H[:, :int(cnt.max())], N / N.sum(1, keepdim=True), done, rej_err, rej_neg


def replay_compact(step, x0, args, H):
    """replay() on the active cells only: cells sorted by their number of accepted steps, column i runs the prefix of
    cells that still have a step. Returns the end states in the original order."""
    cnt = (H > 0).sum(1)
    order = torch.argsort(cnt, descending=True)
    back = torch.argsort(order)
    N = x0[order]
    a = [q[order] for q in args]
    Hs = H[order]
    ns = (cnt[order][None, :] > torch.arange(H.shape[1], device=H.device)[:, None]).sum(1).tolist()
    for i, n in enumerate(ns):
        if n == 0:
            break
        Nn, _ = step(N[:n], Hs[:n, i], *(q[:n] for q in a))
        N = torch.cat([torch.clamp(Nn, min=0.0), N[n:]], 0)
    x = N / N.sum(1, keepdim=True)
    return x[back]


def make_iter(step, rtol=1e-3, atol=1e-18):
    """One masked solver iteration over the whole batch (static shapes, so it compiles into few kernels): cells with time
    left take a step of min(h, next stop - t); a step is accepted when its RMS error is within tolerance and no species
    goes below -atol. Stops (Q [B, M], increasing) are hit exactly; on reaching stop q the number of accepted steps is
    written to at_stop[:, q]. H [B, cap] collects the accepted step sizes in order."""
    def it(N, t, h, cnt, q, H, at_stop, Q, kf0, kr0, nd, lT, lP):
        cap, M = H.shape[1], Q.shape[1]
        tstop = torch.gather(Q, 1, q.clamp(max=M - 1)[:, None]).squeeze(1)
        pre = (q < M) & (t >= tstop * (1.0 - 1e-14))          # a stop equal to the last one: reached without a step
        qc = q.clamp(max=M - 1)[:, None]
        at_stop = at_stop.scatter(1, qc, torch.where(pre[:, None], cnt[:, None], torch.gather(at_stop, 1, qc)))
        q = q + pre.long()
        tstop = torch.gather(Q, 1, q.clamp(max=M - 1)[:, None]).squeeze(1)
        act = (q < M) & (cnt < cap) & (pre == 0)
        ha = torch.where(act, torch.minimum(h, tstop - t), torch.zeros_like(h))
        Nn, err = step(N, ha, kf0, kr0, nd, lT, lP)
        sc = atol + rtol * torch.maximum(N.abs(), Nn.abs())
        en = ((err / sc) ** 2).mean(1).sqrt()
        e_ok = (en <= 1.0) & torch.isfinite(en)
        n_ok = Nn.min(1).values >= -atol
        ok = act & e_ok & n_ok
        col = cnt.clamp(max=cap - 1)[:, None]
        H = H.scatter(1, col, torch.where(ok[:, None], ha[:, None], torch.gather(H, 1, col)))
        cnt = cnt + ok.long()
        N = torch.where(ok[:, None], torch.clamp(Nn, min=0.0), N)
        t = torch.where(ok, t + ha, t)
        hit = ok & (t >= tstop * (1.0 - 1e-14))
        qc = q.clamp(max=M - 1)[:, None]
        at_stop = at_stop.scatter(1, qc, torch.where(hit[:, None], cnt[:, None], torch.gather(at_stop, 1, qc)))
        q = q + hit.long()
        fac = torch.where(torch.isfinite(en), 0.9 * en.clamp(min=1e-10) ** -0.5, torch.full_like(en, 0.1)).clamp(0.2, 5.0)
        fac = torch.where(ok, fac, fac.clamp(max=0.5))
        h = torch.where(act, ha * fac, h)
        return N, t, h, cnt, q, H, at_stop, act & (e_ok == 0), act & e_ok & (n_ok == 0)
    return it


def solve_stops(it, x0, args, Q, cap=1000, h0=1e-10, check=16):
    """Forward solve without gradients through the stops Q [B, M] (elapsed times; a cell's later stops may repeat its
    last one, which counts as reached). Returns (H [B, n], at_stop [B, M], states at the stops [B, M, S], finished [B],
    rejections by the error test, by negativity)."""
    with torch.no_grad():
        B, M = Q.shape
        N = x0.clone()
        t = torch.zeros(B, dtype=N.dtype, device=N.device)
        h = torch.minimum(torch.full_like(t, h0), Q[:, 0])
        cnt = torch.zeros(B, dtype=torch.long, device=N.device)
        q = torch.zeros_like(cnt)
        H = torch.zeros(B, cap, dtype=N.dtype, device=N.device)
        at_stop = torch.zeros(B, M, dtype=torch.long, device=N.device)
        rej_e = torch.zeros_like(cnt)
        rej_n = torch.zeros_like(cnt)
        for i in range(cap * 2):
            if i % check == 0 and not bool(((q < M) & (cnt < cap)).any()):
                break
            N, t, h, cnt, q, H, at_stop, re_, rn_ = it(N, t, h, cnt, q, H, at_stop, Q, *args)
            rej_e += re_.long()
            rej_n += rn_.long()
        n = int(cnt.max())
        return H[:, :n], at_stop, q >= M, rej_e, rej_n


def replay_stops(step, x0, args, H, at_stop):
    """The states at the stops [B, M, S] with gradients: the accepted steps replayed on the active cells only (cells
    sorted by step count), the state after the at_stop-th step gathered for every stop."""
    B, M = at_stop.shape
    cnt = (H > 0).sum(1)
    order = torch.argsort(cnt, descending=True)
    N = x0[order]
    a = [p[order] for p in args]
    Hs = H[order]
    ns = (cnt[order][None, :] > torch.arange(H.shape[1], device=H.device)[:, None]).sum(1).tolist()
    states = [N]
    for i, n in enumerate(ns):
        if n == 0:
            break
        Nn, _ = step(N[:n], Hs[:n, i], *(p[:n] for p in a))
        N = torch.cat([torch.clamp(Nn, min=0.0), N[n:]], 0)
        states.append(N)
    St = torch.stack(states)                                   # [n_cols + 1, B, S], sorted order
    k = at_stop[order].clamp(max=len(states) - 1)              # [B, M]
    out = St[k, torch.arange(B, device=H.device)[:, None]]     # [B, M, S]
    out = out / out.sum(2, keepdim=True)
    return out[torch.argsort(order)]


def advance_compact(step, x0, args, dt, h, rtol=1e-3, atol=1e-18, max_steps=4000, order=2):
    """advance() that works only on the cells still running: each iteration gathers the active cells, takes one step for
    them and scatters the result back, so a few stiff cells cost a few small steps, not full-batch steps. Same accept /
    reject rule as advance(); step sizes scale by 0.9 err^(-1/order), order = the order of the method's error estimate
    plus one (2 for ROS2, 3 for Rodas3); returns (x, next step sizes, finished, accepted steps)."""
    with torch.no_grad():
        N = x0.clone()
        t = torch.zeros_like(dt)
        h = torch.minimum(h, dt).clamp(min=1e-12)
        n_ok = torch.zeros_like(dt)
        a = (dt > 0).nonzero().squeeze(1)
        for _ in range(max_steps):
            if a.numel() == 0:
                break
            Na, ta, dta = N[a], t[a], dt[a]
            ha = torch.minimum(h[a], dta - ta)
            Nn, err = step(Na, ha, *(q[a] for q in args))
            sc = atol + rtol * torch.maximum(Na.abs(), Nn.abs())
            en = ((err / sc) ** 2).mean(1).sqrt()
            ok = (en <= 1.0) & (Nn.min(1).values >= -atol) & torch.isfinite(en)
            N[a] = torch.where(ok[:, None], torch.clamp(Nn, min=0.0), Na)
            ta = torch.where(ok, ta + ha, ta)
            t[a] = ta
            n_ok[a] += ok.to(n_ok.dtype)
            fac = torch.where(torch.isfinite(en), 0.9 * en.clamp(min=1e-10) ** (-1.0 / order), torch.full_like(en, 0.1)).clamp(0.2, 5.0)
            fac = torch.where(ok, fac, fac.clamp(max=0.5))
            h[a] = ha * fac
            a = a[ta < dta * (1.0 - 1e-14)]
        done = t >= dt * (1.0 - 1e-14)
        return N / N.sum(1, keepdim=True), h, done, n_ok
