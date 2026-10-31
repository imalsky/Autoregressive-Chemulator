"""VULCAN-0D's NCHO mechanism as a kinetics.Network-compatible object, read as data: the reaction list
(NCHO_thermo_network.txt: id, reaction, Arrhenius coefficients) and one NASA-9 file per species (two ranges of 10
coefficients, split at 1000 K). Written from the published rate laws:
    two-body          k = A T^B exp(-C / T)
    three-body        k = k0 / (1 + k0 n_tot / k_inf) (k0, k_inf Arrhenius), both directions times n_tot
    low-pressure only k = k0, both directions times n_tot
    special           OH + CH3 + M -> CH3OH + M: k0 = 1.932e3 T^-9.88 exp(-7544/T) + 5.109e-11 T^-6.25 exp(-1433/T),
                      k_inf = 1.031e-10 T^-0.018 exp(16.74/T), falloff as three-body
Reverse rates from the equilibrium constants of the NASA-9 data (kinetics.Network.rates). The object exposes the
attributes kinetics.Network uses (species, S, reac, _re, _pr, _tb, atoms, elements, g_rt, rates), so kinbatch.Mech and the
numpy reference integrator work unchanged.
"""
import numpy as np

import kinetics as K


def _arr(A, B, C, T):
    return A * T ** B * np.exp(-C / T)


class VulcanNetwork(K.Network):
    """network_file: VULCAN's reaction list; nasa_dir: folder of <species>.txt NASA-9 files; species: optional order
    (default: order of first appearance in the reaction list, He appended)."""

    def __init__(self, network_file, nasa_dir, species=None):
        rows = open(network_file).read().split("\n")
        section, reac = None, []
        for ln in rows:
            s = ln.strip()
            if s.startswith("#"):
                low = s.lower()
                if "two-body" in low:
                    section = "two"
                elif "without high-pressure" in low:
                    section = "low"
                elif "3-body" in low:
                    section = "three"
                elif "special" in low:
                    section = "special"
                continue
            if not s or "[" not in s or section is None:
                continue
            body = s.split("[", 1)[1].split("]", 1)
            eq, rest = body[0].strip(), body[1].split()
            lhs, rhs = (side.split("+") for side in eq.split("->"))
            lhs = [q.strip() for q in lhs if q.strip() != "M"]
            rhs = [q.strip() for q in rhs if q.strip() != "M"]
            r = {"eq": eq, "names": (lhs, rhs)}
            if section == "two":
                r["kind"] = 2
                r["A"], r["B"], r["C"] = (float(v) for v in rest[:3])
            elif section == "three":
                r["kind"] = 3
                r["A0"], r["B0"], r["C0"], r["Ai"], r["Bi"], r["Ci"] = (float(v) for v in rest[:6])
            elif section == "low":
                r["kind"] = 5
                r["A0"], r["B0"], r["C0"] = (float(v) for v in rest[:3])
            else:
                if eq.replace(" ", "") != "OH+CH3+M->CH3OH+M":
                    raise ValueError(f"unknown special reaction {eq}")
                r["kind"] = 6
            reac.append(r)
        if species is None:
            species = []
            for r in reac:
                for q in r["names"][0] + r["names"][1]:
                    if q not in species:
                        species.append(q)
            if "He" not in species:
                species.append("He")
        self.species = list(species)
        lo, hi = [], []
        for sp in self.species:
            a = np.loadtxt(f"{nasa_dir}/{sp}.txt").flatten()
            lo.append(np.concatenate([a[0:7], a[8:10]]))   # a1..a7, b1, b2 (the 8th entry of each range is unused)
            hi.append(np.concatenate([a[10:17], a[18:20]]))
        self.lo, self.hi = np.array(lo), np.array(hi)
        for r in reac:
            r["re"] = [self.species.index(q) for q in r["names"][0]]
            r["pr"] = [self.species.index(q) for q in r["names"][1]]
        self.reac = reac
        S = np.zeros((len(self.species), len(reac)))
        for j, r in enumerate(reac):
            for i in r["re"]:
                S[i, j] -= 1
            for i in r["pr"]:
                S[i, j] += 1
        self.S = S
        self.he = self.species.index("He")
        n_sp = len(self.species)
        mr = max(len(r["re"]) for r in reac)
        mp = max(len(r["pr"]) for r in reac)
        self._re = np.array([r["re"] + [n_sp] * (mr - len(r["re"])) for r in reac])
        self._pr = np.array([r["pr"] + [n_sp] * (mp - len(r["pr"])) for r in reac])
        self._tb = np.array([r["kind"] in (3, 5, 6) for r in reac])
        counts = [K.formula(s) for s in self.species]
        self.elements = sorted({e for c in counts for e in c})
        self.atoms = np.array([[c.get(e, 0) for e in self.elements] for c in counts], float)

    def g_rt(self, T):
        """Standard Gibbs energy / RT per species (NASA-9; low range below 1000 K, as VULCAN)."""
        a = self.lo if T < 1000.0 else self.hi
        h = (-a[:, 0] / T ** 2 + a[:, 1] * np.log(T) / T + a[:, 2] + a[:, 3] * T / 2 + a[:, 4] * T ** 2 / 3
             + a[:, 5] * T ** 3 / 4 + a[:, 6] * T ** 4 / 5 + a[:, 7] / T)
        s = (-a[:, 0] / T ** 2 / 2 - a[:, 1] / T + a[:, 2] * np.log(T) + a[:, 3] * T + a[:, 4] * T ** 2 / 2
             + a[:, 5] * T ** 3 / 3 + a[:, 6] * T ** 4 / 4 + a[:, 8])
        return h - s

    def rates(self, T, P_cgs, factors=None):
        nd = P_cgs / (K.KB * T)
        g = self.g_rt(T)
        kf = np.empty(len(self.reac))
        kr = np.empty(len(self.reac))
        for j, r in enumerate(self.reac):
            if r["kind"] == 2:
                k = _arr(r["A"], r["B"], r["C"], T)
            elif r["kind"] == 3:
                k0, ki = _arr(r["A0"], r["B0"], r["C0"], T), _arr(r["Ai"], r["Bi"], r["Ci"], T)
                k = k0 / (1.0 + k0 * nd / ki)
            elif r["kind"] == 5:
                k = _arr(r["A0"], r["B0"], r["C0"], T)
            else:
                k0 = _arr(1.932e3, -9.88, 7544.0, T) + _arr(5.109e-11, -6.25, 1433.0, T)
                ki = _arr(1.031e-10, -0.018, -16.74, T)
                k = k0 / (1.0 + k0 * nd / ki)
            dg = g[r["pr"]].sum() - g[r["re"]].sum()
            dmu = len(r["pr"]) - len(r["re"])
            kf[j] = k
            kr[j] = k / np.exp(-dg) * (K.KB * T / K.P0) ** dmu
        if factors is not None:
            kf, kr = kf * factors, kr * factors
        return kf, kr, nd


    def rates_vec(self, T, P_cgs):
        """rates() for arrays of T [n] and P_cgs [n] (no factors): kf, kr [n, R], nd [n]."""
        T, P_cgs = np.asarray(T, float), np.asarray(P_cgs, float)
        nd = P_cgs / (K.KB * T)
        lo, hi = self.lo, self.hi
        g = np.empty((len(T), len(self.species)))
        for sel, a in ((T < 1000.0, lo), (T >= 1000.0, hi)):
            t = T[sel][:, None]
            h = (-a[:, 0] / t ** 2 + a[:, 1] * np.log(t) / t + a[:, 2] + a[:, 3] * t / 2 + a[:, 4] * t ** 2 / 3
                 + a[:, 5] * t ** 3 / 4 + a[:, 6] * t ** 4 / 5 + a[:, 7] / t)
            s_ = (-a[:, 0] / t ** 2 / 2 - a[:, 1] / t + a[:, 2] * np.log(t) + a[:, 3] * t + a[:, 4] * t ** 2 / 2
                  + a[:, 5] * t ** 3 / 3 + a[:, 6] * t ** 4 / 4 + a[:, 8])
            g[sel] = h - s_
        kf = np.empty((len(T), len(self.reac)))
        for j, r in enumerate(self.reac):
            if r["kind"] == 2:
                kf[:, j] = _arr(r["A"], r["B"], r["C"], T)
            elif r["kind"] == 3:
                k0, ki = _arr(r["A0"], r["B0"], r["C0"], T), _arr(r["Ai"], r["Bi"], r["Ci"], T)
                kf[:, j] = k0 / (1.0 + k0 * nd / ki)
            elif r["kind"] == 5:
                kf[:, j] = _arr(r["A0"], r["B0"], r["C0"], T)
            else:
                k0 = _arr(1.932e3, -9.88, 7544.0, T) + _arr(5.109e-11, -6.25, 1433.0, T)
                ki = _arr(1.031e-10, -0.018, -16.74, T)
                kf[:, j] = k0 / (1.0 + k0 * nd / ki)
        dg = g @ self.S                                         # [n, R] products minus reactants
        dmu = self.S.sum(0)                                     # [R]
        kr = kf / np.exp(-dg) * (K.KB * T[:, None] / K.P0) ** dmu[None]
        return kf, kr, nd


class TorchRates:
    """rates_vec() on the GPU: kf, kr [B, R] and nd [B] from T [B], P_cgs [B] (float64 tensors)."""

    def __init__(self, net, device):
        import torch
        self.torch = torch
        f = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64, device=device)
        R = len(net.reac)
        A0, B0, C0 = np.zeros(R), np.zeros(R), np.zeros(R)
        Ai, Bi, Ci = np.ones(R), np.zeros(R), np.zeros(R)
        falloff = np.zeros(R, bool)
        special = np.zeros(R, bool)
        for j, r in enumerate(net.reac):
            if r["kind"] == 2:
                A0[j], B0[j], C0[j] = r["A"], r["B"], r["C"]
            elif r["kind"] in (3, 5):
                A0[j], B0[j], C0[j] = r["A0"], r["B0"], r["C0"]
                if r["kind"] == 3:
                    Ai[j], Bi[j], Ci[j] = r["Ai"], r["Bi"], r["Ci"]
                    falloff[j] = True
            else:
                special[j] = True
        self.A0, self.B0, self.C0, self.Ai, self.Bi, self.Ci = (f(a) for a in (A0, B0, C0, Ai, Bi, Ci))
        self.falloff = torch.tensor(falloff, device=device)
        self.special = torch.tensor(special, device=device)
        self.lo, self.hi, self.S = f(net.lo), f(net.hi), f(net.S)
        self.dmu = self.S.sum(0)

    def __call__(self, T, P_cgs, t_min=300.0):
        """t_min: refuse any cell colder than this (the solver's domain starts at 300 K; None turns the check off)."""
        torch = self.torch
        if t_min is not None and bool((T < t_min).any()):
            raise ValueError(f"{int((T < t_min).sum())} cells below {t_min:g} K (coldest {float(T.min()):.1f} K): "
                             "outside the chemistry solver's domain")
        nd = P_cgs / (K.KB * T)
        t = T[:, None]
        a = torch.where((T < 1000.0)[:, None, None], self.lo[None], self.hi[None])          # [B, S, 9]
        h = (-a[..., 0] / t ** 2 + a[..., 1] * torch.log(t) / t + a[..., 2] + a[..., 3] * t / 2 + a[..., 4] * t ** 2 / 3
             + a[..., 5] * t ** 3 / 4 + a[..., 6] * t ** 4 / 5 + a[..., 7] / t)
        s = (-a[..., 0] / t ** 2 / 2 - a[..., 1] / t + a[..., 2] * torch.log(t) + a[..., 3] * t + a[..., 4] * t ** 2 / 2
             + a[..., 5] * t ** 3 / 3 + a[..., 6] * t ** 4 / 4 + a[..., 8])
        g = h - s
        k0 = self.A0 * t ** self.B0 * torch.exp(-self.C0 / t)
        ki = self.Ai * t ** self.Bi * torch.exp(-self.Ci / t)
        sp0 = 1.932e3 * t ** -9.88 * torch.exp(-7544.0 / t) + 5.109e-11 * t ** -6.25 * torch.exp(-1433.0 / t)
        spi = 1.031e-10 * t ** -0.018 * torch.exp(16.74 / t)
        k0 = torch.where(self.special, sp0, k0)
        ki = torch.where(self.special, spi, ki)
        kf = torch.where(self.falloff | self.special, k0 / (1.0 + k0 * nd[:, None] / ki), k0)
        dg = g @ self.S
        kr = kf / torch.exp(-dg) * (K.KB * t / K.P0) ** self.dmu
        return kf, kr, nd
