"""VULCAN's reaction lists (NCHO_thermo_network.txt and the other VULCAN-format networks) as kinetics.Network-compatible
objects, read as data (the thermal gas-phase part): the reaction list (id, reaction, rate coefficients) and one NASA-9
file per species (two ranges of 10 coefficients, split at 1000 K). Written from the published rate laws, by section:
    two-body                k = A T^B exp(-C / T)
    3-body                  k = k0 / (1 + k0 n_tot / k_inf) * F (k0, k_inf Arrhenius), both directions times n_tot
    without high-pressure   k = k0, both directions times n_tot
    special                 the law of SPECIAL for the equation, falloff as 3-body; OH + CH3 + M -> CH3OH + M:
                            k0 = 1.932e3 T^-9.88 exp(-7544/T) + 5.109e-11 T^-6.25 exp(-1433/T),
                            k_inf = 1.031e-10 T^-0.018 exp(16.74/T)
    irreversible            3-body falloff; the reverse law is REVERSE's entry for the equation
F = 1, or the Troe factor of a following "Troe [a, T3, T1, T2]" line (Venot et al. 2012, eq. C.2-C.5) with
Pr = k0 n_tot / k_inf; a following "effi {...}" line multiplies k by the efficiency of the bulk gas. The other reverse
rates come from the equilibrium constants of the NASA-9 data (kinetics.Network.rates). Reading stops at the first
"# reverse stops", "# condensation", "# photo" or "# re_end" heading (photolysis and condensation lines are skipped and
counted in skipped; a reaction between "# reverse stops" and the next of those headings is refused). Element counts come
from the species name up to its first "_" (CH2_1 -> CH2, N_2D -> N).
Each reaction dict holds its law: kind (2 two-body, 3 falloff, 5 low-pressure only, 6 special), A, B, C (kind 2) or
A0, B0, C0, optional A1, B1, C1 (a second k0 term) and Ai, Bi, Ci (kinds 3, 6); optional troe (a, T3, T1, T2), eff and
rev (the explicit reverse law, same keys). The object exposes the attributes kinetics.Network uses (species, S, reac,
_re, _pr, _tb, atoms, elements, g_rt, rates), so kinbatch.Mech and the numpy reference integrator work unchanged.
"""
import ast

import numpy as np

import kinetics as K

# laws not in the reaction list (VULCAN's code holds them), by equation without spaces
SPECIAL = {"OH+CH3+M->CH3OH+M": dict(A0=1.932e3, B0=-9.88, C0=7544.0, A1=5.109e-11, B1=-6.25, C1=1433.0,
                                     Ai=1.031e-10, Bi=-0.018, Ci=-16.74)}
REVERSE = {"H+CH3+M->M+CH4": dict(kind=3, A0=1.40e-06, B0=0.0, C0=45700.0, Ai=2.40e+16, Bi=0.0, Ci=52790.0,
                                  troe=(0.69, 90.0, 2210.0, 1e18))}
STOP = ("# reverse stops", "# condensation", "# photo", "# re_end")


class Laws:
    """Rate coefficients of a list of laws (the reaction dicts' keys) as arrays: Laws(laws, xp, to)(T, nd) -> k
    [..., R] for T, nd broadcast against [R] (xp: numpy or torch; to: numpy array -> xp array), three-body n_tot
    excluded."""

    def __init__(self, laws, xp, to):
        col = lambda key, d: to(np.array([r.get(key, d) if r["kind"] != 2 else d for r in laws], float))
        self.xp = xp
        self.A0 = to(np.array([r["A"] if r["kind"] == 2 else r["A0"] for r in laws], float))
        self.B0 = to(np.array([r["B"] if r["kind"] == 2 else r["B0"] for r in laws], float))
        self.C0 = to(np.array([r["C"] if r["kind"] == 2 else r["C0"] for r in laws], float))
        self.A1, self.B1, self.C1 = col("A1", 0.0), col("B1", 0.0), col("C1", 0.0)
        self.Ai, self.Bi, self.Ci = col("Ai", 1.0), col("Bi", 0.0), col("Ci", 0.0)
        self.fall = to(np.array([r["kind"] in (3, 6) for r in laws]))
        self.eff = to(np.array([r.get("eff", 1.0) for r in laws], float))
        self.troe = [(j, r["troe"]) for j, r in enumerate(laws) if "troe" in r]

    def __call__(self, T, nd):
        xp = self.xp
        arr = lambda A, B, C: A * T ** B * xp.exp(-C / T)
        k0 = arr(self.A0, self.B0, self.C0) + arr(self.A1, self.B1, self.C1)
        pr = k0 * nd / arr(self.Ai, self.Bi, self.Ci)
        k = xp.where(self.fall, k0 / (1.0 + pr), k0)
        for j, (a, t3, t1, t2) in self.troe:
            Tj = T[..., 0] if np.ndim(T) else T
            fc = xp.log10((1.0 - a) * xp.exp(-Tj / t3) + a * xp.exp(-Tj / t1) + xp.exp(-t2 / Tj))
            x = xp.log10(pr[..., j]) - 0.4 - 0.67 * fc
            y = x / (0.75 - 1.27 * fc - 0.14 * x)
            k[..., j] = k[..., j] * 10.0 ** (fc / (1.0 + y * y))
        return k * self.eff


class VulcanNetwork(K.Network):
    """network_file: a VULCAN reaction list; nasa_dir: folder of <species>.txt NASA-9 files; species: optional order
    (default: order of first appearance in the reaction list, He appended); bulk: the gas whose efficiency an effi line
    applies (VULCAN's atm_base)."""

    def __init__(self, network_file, nasa_dir, species=None, bulk="H2"):
        section, reac, self.skipped, stopped = None, [], 0, None
        for ln in open(network_file).read().split("\n"):
            s = ln.strip()
            if s.startswith("#"):
                low = s.lower()
                if low.startswith(STOP):
                    stopped = stopped or low
                    if not low.startswith("# reverse stops"):
                        section = "skip"
                elif stopped:
                    continue
                elif low.startswith("# two-body"):
                    section = "two"
                elif low.startswith("# 3-body reactions without high-pressure"):
                    section = "low"
                elif low.startswith("# 3-body"):
                    section = "three"
                elif low.startswith("# special"):
                    section = "special"
                elif low.startswith("# irreversible"):
                    section = "irrev"
                continue
            if stopped:
                if section == "skip" and "[" in s:
                    self.skipped += 1
                elif "[" in s:
                    raise ValueError(f"reaction after '# reverse stops' outside photolysis / condensation: {s}")
                continue
            if s.startswith("Troe"):
                reac[-1]["troe"] = tuple(float(v) for v in ast.literal_eval(s[4:].strip()))
                continue
            if s.startswith("effi"):
                reac[-1]["eff"] = float(ast.literal_eval(s[4:].strip())[bulk])
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
            elif section in ("three", "irrev"):
                r["kind"] = 3
                r["A0"], r["B0"], r["C0"], r["Ai"], r["Bi"], r["Ci"] = (float(v) for v in rest[:6])
                if section == "irrev":
                    if eq.replace(" ", "") not in REVERSE:
                        raise ValueError(f"no reverse law for the irreversible reaction {eq}")
                    r["rev"] = dict(REVERSE[eq.replace(" ", "")])
            elif section == "low":
                r["kind"] = 5
                r["A0"], r["B0"], r["C0"] = (float(v) for v in rest[:3])
            else:
                if eq.replace(" ", "") not in SPECIAL:
                    raise ValueError(f"unknown special reaction {eq}")
                r["kind"] = 6
                r.update(SPECIAL[eq.replace(" ", "")])
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
        counts = [K.formula(s.split("_")[0]) for s in self.species]
        self.elements = sorted({e for c in counts for e in c})
        self.atoms = np.array([[c.get(e, 0) for e in self.elements] for c in counts], float)
        self._rev = [j for j, r in enumerate(reac) if "rev" in r]

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
        kf = Laws(self.reac, np, np.asarray)(T, nd)
        kr = kf / np.exp(-(self.g_rt(T) @ self.S)) * (K.KB * T / K.P0) ** self.S.sum(0)
        if self._rev:
            kr[self._rev] = Laws([self.reac[j]["rev"] for j in self._rev], np, np.asarray)(T, nd)
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
        kf = Laws(self.reac, np, np.asarray)(T[:, None], nd[:, None])
        dg = g @ self.S                                         # [n, R] products minus reactants
        dmu = self.S.sum(0)                                     # [R]
        kr = kf / np.exp(-dg) * (K.KB * T[:, None] / K.P0) ** dmu[None]
        if self._rev:
            kr[:, self._rev] = Laws([self.reac[j]["rev"] for j in self._rev], np, np.asarray)(T[:, None], nd[:, None])
        return kf, kr, nd


class TorchRates:
    """rates_vec() on the GPU: kf, kr [B, R] and nd [B] from T [B], P_cgs [B] (float64 tensors)."""

    def __init__(self, net, device):
        import torch
        self.torch = torch
        f = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64, device=device)
        to = lambda a: torch.tensor(a, device=device)
        self.laws = Laws(net.reac, torch, to)
        self.rev = [j for j, r in enumerate(net.reac) if "rev" in r]
        self.rlaws = Laws([net.reac[j]["rev"] for j in self.rev], torch, to) if self.rev else None
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
        kf = self.laws(t, nd[:, None])
        kr = kf / torch.exp(-(g @ self.S)) * (K.KB * t / K.P0) ** self.dmu
        if self.rev:
            kr[:, self.rev] = self.rlaws(t, nd[:, None])
        return kf, kr, nd
