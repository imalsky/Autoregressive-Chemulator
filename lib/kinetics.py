"""A reduced thermochemical kinetics solver with the structure of the published mini-chem scheme (Tsai et al. 2022, A&A
664 A82): mass-action reactions over a small species set, forward rates from Arrhenius / Lindemann laws or from tabulated
net-reaction rates k_f(T, P), reverse rates from equilibrium constants of NASA 9-coefficient polynomials, integrated by a
stiff implicit method. Written from the paper's equations; it reads the reaction list, species thermodynamics and rate
tables as data files (paths given by the caller).

Per reaction r with reactant list a and product list b (a species listed m times enters m times):
    v_r = k_f,r n_a1 n_a2 ... - k_r,r n_b1 n_b2 ...,   k_r,r = k_f,r / K_r (k_B T / P0)^(|b| - |a|),
    K_r = exp(-dG_r / RT) from the NASA polynomials (P0 = 1 bar); three-body reactions multiply both directions by n_tot;
    dn/dt = S v (S: products minus reactants); He is inert.
The reactor holds T and P: the total number density stays n_tot = P / kT (as VULCAN keeps it after every step), so the mixing
ratios x = n / n_tot evolve as dx/dt = (S v - x sum(S v)) / n_tot.
A positive factor f_r multiplies k_f,r and k_r,r together, so the equilibrium, the element ratios and the descent of the Gibbs
energy (at fixed T, P) hold for any f_r > 0.
"""
import re

import numpy as np

KB = 1.380649e-16          # erg K^-1
RGAS = 8.31446261815324e7  # erg mol^-1 K^-1
P0 = 1.0e6                 # dyn cm^-2 (1 bar)


def formula(name):
    """Element counts of a species name, summed over repeats (CH3OH: C 1, H 4, O 1)."""
    c = {}
    for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", name):
        c[e] = c.get(e, 0) + int(n or 1)
    return c


def _rows(path):
    return [ln.strip() for ln in open(path) if ln.strip() and not ln.lstrip().startswith("#")]


def _nums(s):
    return [float(v) for v in s.replace(",", " ").split()]


class Network:
    """Species thermodynamics, reactions and rate tables. sp_file / data_file: the scheme's species and reaction lists;
    table_dir + prefix: the net-reaction tables (prefix + table name)."""

    def __init__(self, sp_file, data_file, table_dir, prefix):
        rows = _rows(sp_file)
        n = int(rows[0])
        self.species, self.lo, self.hi = [], [], []
        for i in range(n):
            self.species.append(rows[1 + 3 * i].split(",")[0].strip())
            self.lo.append(_nums(rows[2 + 3 * i]))
            self.hi.append(_nums(rows[3 + 3 * i]))
        self.lo, self.hi = np.array(self.lo), np.array(self.hi)
        rows = _rows(data_file)
        nr = int(rows[0])
        self.reac = []
        k = 1
        for _ in range(nr):
            kind = int(_nums(rows[k])[0])
            head = rows[k + 1]
            stoi = rows[k + 2]
            names = [s.strip() for s in rows[k + 3].split(",")]
            k += 4
            vals = head.replace(",", " ").split()
            na, nb = int(vals[0]), int(vals[1])
            r = {"kind": kind, "re": [self.species.index(s) for s in names[:na]],
                 "pr": [self.species.index(s) for s in names[na:na + nb]]}
            if kind == 2:
                r["A"], r["B"], r["C"] = (float(v) for v in vals[2:5])
            elif kind == 3:
                r["A0"], r["B0"], r["C0"], r["Ai"], r["Bi"], r["Ci"] = (float(v) for v in vals[2:8])
            else:
                fname = head.split("'")[1]
                trow = open(f"{table_dir}/{prefix}{fname}").read().split("\n")
                nT, nP, _ = (int(v) for v in trow[1].split())
                T = np.array(_nums(trow[3]))
                P = np.array(_nums(trow[5])) * 1.0e6
                kf = np.array([float(v) for v in trow[7:7 + nT * nP]]).reshape(nT, nP)
                r["lT"], r["lP"], r["lkf"] = np.log10(T), np.log10(P), np.log10(np.maximum(kf, 1e-199))
            assert all(int(float(s)) == 1 for s in stoi.replace(",", " ").split()), "non-unit stoichiometric factors"
            self.reac.append(r)
        S = np.zeros((len(self.species), nr))
        for j, r in enumerate(self.reac):
            for i in r["re"]:
                S[i, j] -= 1
            for i in r["pr"]:
                S[i, j] += 1
        self.S = S
        self.he = self.species.index("He") if "He" in self.species else None
        n_sp = len(self.species)  # reactant / product index lists padded with n_sp, the slot of a constant 1
        self._re = np.array([r["re"] + [n_sp] * (3 - len(r["re"])) for r in self.reac])
        self._pr = np.array([r["pr"] + [n_sp] * (4 - len(r["pr"])) for r in self.reac])
        self._tb = np.array([r["kind"] == 3 for r in self.reac])
        counts = [formula(s) for s in self.species]
        self.elements = sorted({e for c in counts for e in c})
        self.atoms = np.array([[c.get(e, 0) for e in self.elements] for c in counts], float)  # [species, element]

    def g_rt(self, T):
        """Standard Gibbs energy / RT per species at T (NASA 9-coefficient form)."""
        a = self.lo if T <= 1000.0 else self.hi
        h = (-a[:, 0] / T ** 2 + a[:, 1] * np.log(T) / T + a[:, 2] + a[:, 3] * T / 2 + a[:, 4] * T ** 2 / 3
             + a[:, 5] * T ** 3 / 4 + a[:, 6] * T ** 4 / 5 + a[:, 7] / T)
        s = (-a[:, 0] / T ** 2 / 2 - a[:, 1] / T + a[:, 2] * np.log(T) + a[:, 3] * T + a[:, 4] * T ** 2 / 2
             + a[:, 5] * T ** 3 / 3 + a[:, 6] * T ** 4 / 4 + a[:, 8])
        return h - s

    def rates(self, T, P_cgs, factors=None):
        """Forward and reverse rate coefficients (cm^3 units of mass action, three-body n_tot excluded) and n_tot."""
        nd = P_cgs / (KB * T)
        g = self.g_rt(T)
        kf = np.empty(len(self.reac))
        kr = np.empty(len(self.reac))
        for j, r in enumerate(self.reac):
            if r["kind"] == 2:
                k = r["A"] * T ** r["B"] * np.exp(-r["C"] / T)
            elif r["kind"] == 3:
                k0 = r["A0"] * T ** r["B0"] * np.exp(-r["C0"] / T)
                ki = r["Ai"] * T ** r["Bi"] * np.exp(-r["Ci"] / T)
                k = k0 / (1.0 + k0 * nd / ki)
            else:
                k = 10.0 ** _table(r, T, P_cgs)
            dg = g[r["pr"]].sum() - g[r["re"]].sum()
            dmu = len(r["pr"]) - len(r["re"])
            kf[j] = k
            kr[j] = k / np.exp(-dg) * (KB * T / P0) ** dmu
        if factors is not None:
            kf, kr = kf * factors, kr * factors
        return kf, kr, nd

    def flux(self, n, kf, kr, nd):
        ne = np.append(n, 1.0)
        v = kf * ne[self._re].prod(1) - kr * ne[self._pr].prod(1)
        return np.where(self._tb, v * nd, v)

    def jac(self, n, kf, kr, nd):
        """d(S v)/dn."""
        ne = np.append(n, 1.0)
        c = np.where(self._tb, nd, 1.0)
        D = np.zeros((len(self.reac), len(n) + 1))
        rows = np.arange(len(self.reac))
        for idx, k_, sgn in ((self._re, kf, 1.0), (self._pr, kr, -1.0)):
            for m in range(idx.shape[1]):
                rest = ne[np.delete(idx, m, 1)].prod(1)
                np.add.at(D, (rows, idx[:, m]), sgn * c * k_ * rest)
        return self.S @ D[:, :-1]

    def gibbs(self, x, x0, T, P_bar):
        """Gibbs energy / RT of the gas that started as x0 (per starting molecule) at fixed T, P: the molecule count scales as
        H(x0) / H(x) (H atoms per molecule), times sum x_i (g_i + ln(x_i P / P0)); non-increasing along any solution."""
        eh = self.atoms[:, self.elements.index("H")]
        pos = x > 0
        return float(eh @ x0 / (eh @ x) * (x[pos] * (self.g_rt(T)[pos] + np.log(x[pos] * P_bar))).sum())

    def integrate(self, x0, T, P_bar, t_out, factors=None, rtol=1e-6, atol=1e-30, method="BDF"):
        """Mixing ratios at the times t_out (s) from x0 (mixing ratios over self.species, summing to 1) at constant T, P."""
        from scipy.integrate import solve_ivp  # the reference integrator; the rest of the module needs only numpy
        P = P_bar * 1.0e6
        kf, kr, nd = self.rates(T, P, factors)

        def f(t, x):
            r = self.S @ self.flux(x * nd, kf, kr, nd)
            return (r - x * r.sum()) / nd

        def jac(t, x):
            Jn = self.jac(x * nd, kf, kr, nd)
            r = self.S @ self.flux(x * nd, kf, kr, nd)
            return Jn - np.outer(x, Jn.sum(0)) - np.eye(len(x)) * (r.sum() / nd)
        sol = solve_ivp(f, (0.0, float(np.max(t_out))), np.asarray(x0, float), method=method, t_eval=np.sort(t_out),
                        rtol=rtol, atol=atol, jac=jac)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y.T


def _bezier(xi, yi, x):
    """Monotone-limited quadratic Bezier through three points (as in the published scheme's table interpolation)."""
    dx, dx1 = xi[1] - xi[0], xi[2] - xi[1]
    dy, dy1 = yi[1] - yi[0], yi[2] - yi[1]
    eps = 10.0 * np.finfo(float).eps
    stol = eps * max(1.0, abs(yi[0]), abs(yi[1]), abs(yi[2]))
    first = xi[0] < x < xi[1]
    a, b, d_ = (xi[0], yi[0], dx) if first else (xi[1], yi[1], dx1)
    if abs(dy) <= stol or abs(dy1) <= stol:
        t = (x - a) / d_
        return (1 - t) * b + t * (yi[1] if first else yi[2])
    den, den1 = 1 - (dy1 / dy) * (dx / dx1), 1 - (dy / dy1) * (dx1 / dx)
    if abs(den) <= eps or abs(den1) <= eps:
        t = (x - a) / d_
        return (1 - t) * b + t * (yi[1] if first else yi[2])
    if first:
        wh, wl, wl1 = dx1 / (dx + dx1), 1 + 1 / den, 1 / den1
    else:
        wh, wl, wl1 = dx / (dx + dx1), 1 / den, 1 + 1 / den1
    if wh <= min(wl, wl1) or wh >= max(wl, wl1):
        wh = 1.0
    if first:
        yc = yi[1] - dx / 2 * (wh * dy / dx + (1 - wh) * dy1 / dx1)
        t = (x - xi[0]) / dx
        return (1 - t) ** 2 * yi[0] + 2 * t * (1 - t) * yc + t ** 2 * yi[1]
    yc = yi[1] + dx1 / 2 * (wh * dy1 / dx1 + (1 - wh) * dy / dx)
    t = (x - xi[1]) / dx1
    return (1 - t) ** 2 * yi[1] + 2 * t * (1 - t) * yc + t ** 2 * yi[2]


def _three(axis, v):
    """Three table points around v: j with axis[j] <= v < axis[j + 1] and its neighbours, shifted inside the table."""
    j = int(np.searchsorted(axis, v, side="right")) - 1
    j = min(max(j, 1), len(axis) - 2)
    return j - 1, j, j + 1


def _table(r, T, P_cgs):
    """log10 k_f from the (T, P) table: clamped to its edges, Bezier in log P at three T rows, then in log T."""
    lT, lP = np.log10(np.clip(T, 10 ** r["lT"][0], 10 ** r["lT"][-1])), np.log10(np.clip(P_cgs, 10 ** r["lP"][0], 10 ** r["lP"][-1]))
    p = _three(r["lP"], lP)
    lPa = r["lP"][list(p)]

    def at(it):
        y = r["lkf"][it, list(p)]
        return y[1] if lP == lPa[1] else (y[0] if lP == lPa[0] else (y[2] if lP == lPa[2] else _bezier(lPa, y, lP)))
    if lT <= r["lT"][0]:
        return at(0)
    if lT >= r["lT"][-1]:
        return at(len(r["lT"]) - 1)
    q = _three(r["lT"], lT)
    ya = np.array([at(i) for i in q])
    lTa = r["lT"][list(q)]
    return ya[1] if lT == lTa[1] else (ya[0] if lT == lTa[0] else (ya[2] if lT == lTa[2] else _bezier(lTa, ya, lT)))
