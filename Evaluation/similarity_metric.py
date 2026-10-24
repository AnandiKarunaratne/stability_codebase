"""
The five elementary similarity measures (Def. 5-9) and the aggregated
behavioral profile metric (Def. 10) from:

  Kunze, Weidlich, Weske: "Behavioral Similarity - A Proper Metric",
  BPM 2011.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Tuple

from behavioral_profile import BehavioralProfile, Pair

ELEMENTARY_KEYS = ("+", "~>", "||", "~>'", "||'")


def jaccard_sim(a: FrozenSet, b: FrozenSet) -> float:
    """sim(A,B) = |A n B| / |A u B|. Two empty sets are defined as identical
    (similarity 1) -- this matches 'no constraints observed on either side'
    rather than being undefined; make this explicit since the paper doesn't
    state the 0/0 case."""
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _elementary_sets(bp: BehavioralProfile) -> Dict[str, FrozenSet[Pair]]:
    rev = bp.reverse_strict_order()
    return {
        "+": bp.exclusive,
        "~>": bp.strict_order,
        "||": bp.interleaving,
        "~>'": bp.strict_order | rev,                       # extended strict order (Def. 8)
        "||'": bp.strict_order | rev | bp.interleaving,      # extended interleaving order (Def. 9)
    }


def elementary_similarities(profile_p: BehavioralProfile, profile_q: BehavioralProfile) -> Dict[str, float]:
    """Compute all five elementary similarities (Def. 5-9).

    Correction vs. an earlier draft (A3): the paper does NOT restrict the
    two profiles to their common activities before taking the Jaccard
    coefficient. Each relation set lives over its own model's full
    activity set (A_P x A_P resp. A_Q x A_Q); "correspondence" (Sec 4.1)
    is realized purely through label matching inside the Jaccard formula
    itself -- a pair naming an activity absent from the other model can
    never land in the intersection, but it still inflates the union. This
    is exactly what reproduces their worked example (activity 'b' present
    only in m1: it contributes (b,b) to +_m1 which is simply absent from
    +_m2's domain, giving sim+ = 14/15 rather than 1.0, which is what
    restricting to common activities first would have produced)."""
    sets_p = _elementary_sets(profile_p)
    sets_q = _elementary_sets(profile_q)
    return {k: jaccard_sim(sets_p[k], sets_q[k]) for k in ELEMENTARY_KEYS}


DEFAULT_WEIGHTS: Dict[str, float] = {"+": 0.2, "~>'": 0.8}
# The paper's own experiments (Sec. 5.2) found exclusiveness + extended
# strict order to be the strongest pair of elementary measures against
# their human-similarity-judgement benchmark (mean average precision:
# Ex=0.70, So=0.75, In=0.29, ES=0.75, EI=0.76 individually), and Fig. 3(b)
# specifically tested this pair weighted 1:4 towards extended strict
# order ("1ES4"), which is what's used here as the default.

ALL_FIVE_EQUAL_WEIGHTS: Dict[str, float] = {
    "+": 0.2, "~>": 0.2, "||": 0.2, "~>'": 0.2, "||'": 0.2,
}  # kept as an alternative -- pass this explicitly via `weights=` if wanted

# The original 3-relation weighting from the ZEUS'11 paper (kept here only
# for validation against that paper's worked example a/b/c).
ZEUS_WEIGHTS: Dict[str, float] = {"+": 0.5, "~>": 0.3, "||": 0.2}


def m3_metric(
    profile_p: BehavioralProfile,
    profile_q: BehavioralProfile,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Aggregated behavioral profile distance (Def. 10):
        d(P,Q) = 1 - sum_h w_h * sim_h(P,Q)
    Returns (distance, per-measure similarities actually used).
    `weights` may use any subset of {'+','~>','||',"~>'","||'"}; missing
    keys are simply not included. Weights need not be given for all five --
    Theorem 1 only requires each w_h in (0,1) and sum w_h = 1 over the
    chosen subset.
    """
    weights = weights or DEFAULT_WEIGHTS
    total_w = sum(weights.values())
    if not all(0 < w < 1 for w in weights.values()):
        raise ValueError("all weights must be in (0,1)")
    if abs(total_w - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got {total_w}")

    sims = elementary_similarities(profile_p, profile_q)
    used = {k: sims[k] for k in weights}
    distance = 1.0 - sum(weights[k] * used[k] for k in weights)
    return distance, used
