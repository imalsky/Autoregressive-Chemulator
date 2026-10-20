"""Pass rule v2 (author 2026-10-19: "get rid of the rule that prioritizes avoiding rare large misses so much"; project
CLAUDE.md "Large misses are readouts, not vetoes"). Replaces l/judge.py's passes() and p/judge_cb.py's helps() for decisions
from 2026-10-19 on. Input: a rules_t5.py verdict json j (X = x, reference R = r).
  passes_v2(j, x, r): inside the envelope, lasting > 0.5 X <= 0.90 x R, late > 0.5 X <= R, lasting fixed >= 2 x broken, late
  fixed >= broken; T2 snap-back (P3); finite states (G5: no non-finite or kept-input rows, Newton failures bounded); cooler
  stratum lasting > 0.5 X <= R + 3 and late > 0.5 X <= 1.10 x R. INVALID / INCONCLUSIVE / missing counts never pass.
  Dropped as vetoes (now readouts, large_misses()): rows > 2 dex (late, lasting, first call; G1, G2's > 2 part, G4, G6),
  path late p99 and max (G3), invented switches (G7)."""


def passes_v2(j, x, r):
    if j.get("verdict") in (None, "INVALID", "INCONCLUSIVE") or "counts" not in j or "rules" not in j:
        return False
    ci, cc, g = j["counts"]["inside"], j["counts"]["cooler"], j["rules"]
    return bool(ci[x]["last05"] <= 0.90 * ci[r]["last05"] and ci[x]["late05"] <= ci[r]["late05"]
                and ci["last_fixed"] >= 2 * ci["last_broken"] and ci["late_fixed"] >= ci["late_broken"]
                and g["P3"]["pass"] and g["G5"]["pass"]
                and cc[x]["last05"] <= cc[r]["last05"] + 3 and cc[x]["late05"] <= 1.10 * cc[r]["late05"])


def large_misses(j, x, r):
    """Readout line of the dropped guards' quantities (never a gate)."""
    if "counts" not in j:
        return "large misses: n/a"
    ci, cc = j["counts"]["inside"], j["counts"]["cooler"]
    pa = j.get("paths", {})
    ps = "; ".join(f"{k} late p99 {v[x]['late_p99']:.3f} vs {v[r]['late_p99']:.3f} (max {v[x]['max_all']:.2f} vs "
                   f"{v[r]['max_all']:.2f})" for k, v in pa.items()
                   if isinstance(v, dict) and x in v and r in v and "late_p99" in v[x]) if isinstance(pa, dict) else ""
    ng = j.get("new_gt2", {})
    return (f"large misses (readout): inside late > 2 {ci[x]['late2']} vs {ci[r]['late2']}, lasting > 2 {ci[x]['last2']} vs "
            f"{ci[r]['last2']}, first > 2 {ci[x]['first2']} vs {ci[r]['first2']}; cooler late > 2 {cc[x]['late2']} vs "
            f"{cc[r]['late2']}; new late > 2 rows {ng.get('n', 'n/a')} (invented switches {ng.get('invented', 'n/a')})"
            + (f"; paths {ps}" if ps else ""))
