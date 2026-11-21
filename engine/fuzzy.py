"""Mamdani Fuzzy Inference System: one FIS per career.

Stages: fuzzification → GA weight scaling → rule evaluation (AND=min, OR=max)
→ min-implication → max-aggregation → centroid defuzzification.

v2 changes (DESIGN.md §"v2 scoring"):
- Input sets form a Ruspini partition (μ_Low + μ_Medium + μ_High = 1 for every x),
  so a rating of 80 is still partly Medium. With the v1 sets, every rating above 75
  was "only High" and the centroid stopped moving.
- Five output sets (Very Low … Very High) and an 11-rule base per career, so the
  centroid is an interpolation between graded consequents instead of the centre of
  one clipped set.

Every career is evaluated in one vectorised NumPy pass so the GA can run
thousands of evaluations quickly. The maths is the same as running 18
separate Mamdani systems.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .data import SKILL_LABELS, SKILLS, careers

# Input linguistic terms (every skill)
SET_NAMES: list[str] = ["Low", "Medium", "High"]
LOW, MED, HIGH = 0, 1, 2
MF_PARAMS: list[tuple[float, float, float]] = [(0, 0, 50), (0, 50, 100), (50, 100, 100)]

# Output linguistic terms (career suitability)
OUT_NAMES: list[str] = ["Very Low", "Low", "Medium", "High", "Very High"]
O_VL, O_L, O_M, O_H, O_VH = range(5)
OUT_PARAMS: list[tuple[float, float, float]] = [(0, 0, 25), (0, 25, 50), (25, 50, 75), (50, 75, 100), (75, 100, 100)]
UNIVERSE: np.ndarray = np.linspace(0.0, 100.0, 201)  # output universe, step 0.5


def trimf(x: np.ndarray | float, a: float, b: float, c: float) -> np.ndarray:
    """Triangular membership function. Handles shoulders where a == b or b == c."""
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if a == b else (x - a) / (b - a)
    right = np.ones_like(x) if b == c else (c - x) / (c - b)
    out = np.minimum(left, right)
    out = np.where(x == b, 1.0, out)  # a shoulder set is 1 at its flat end, including exactly at b
    return np.clip(out, 0.0, 1.0)


def fuzzify(X: np.ndarray) -> np.ndarray:
    """Crisp inputs (n, 6) in [0, 100] → memberships (n, 6, 3) in [0, 1]."""
    X = np.asarray(X, dtype=float)
    return np.stack([trimf(X, *p) for p in MF_PARAMS], axis=-1)


OUTPUT_MF: np.ndarray = np.stack([trimf(UNIVERSE, *p) for p in OUT_PARAMS])  # (5, U)


def apply_weights(mu: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """GA weight scaling μ'_i = min(1, 6·w_i·μ_i). Equal weights leave μ unchanged."""
    w = np.asarray(weights, dtype=float).reshape(1, len(SKILLS), 1)
    return np.minimum(1.0, len(SKILLS) * w * mu)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    career_index: int
    antecedents: tuple[tuple[int, int], ...]  # (skill index, input set index)
    op: str  # "and" | "or"
    consequent: int  # output set index
    role: str  # short description of why the rule exists

    def text(self, career_name: str) -> str:
        joiner = " AND " if self.op == "and" else " OR "
        cond = joiner.join(f"{SKILL_LABELS[SKILLS[s]]} is {SET_NAMES[m]}" for s, m in self.antecedents)
        return f"IF {cond} THEN {career_name} suitability is {OUT_NAMES[self.consequent]}"


# Rule template per career, using its primary pair (P1, P2) and secondary skill S.
# R1–R9 are the complete 3×3 grid over P1 × P2, so every input combination fires
# at least one rule. A Low primary skill caps the consequent at Low.
RULE_TEMPLATE: list[tuple[tuple[tuple[str, int], ...], str, int, str]] = [
    ((("P1", LOW), ("P2", LOW)), "and", O_VL, "both primary skills weak"),
    ((("P1", LOW), ("P2", MED)), "and", O_L, "a primary skill is weak"),
    ((("P1", MED), ("P2", LOW)), "and", O_L, "a primary skill is weak"),
    ((("P1", LOW), ("P2", HIGH)), "and", O_L, "a primary skill is weak, the other is strong"),
    ((("P1", HIGH), ("P2", LOW)), "and", O_L, "a primary skill is weak, the other is strong"),
    ((("P1", MED), ("P2", MED)), "and", O_M, "both primary skills moderate"),
    ((("P1", MED), ("P2", HIGH)), "and", O_H, "one primary strong, one moderate"),
    ((("P1", HIGH), ("P2", MED)), "and", O_H, "one primary strong, one moderate"),
    ((("P1", HIGH), ("P2", HIGH)), "and", O_VH, "both primary skills strong"),
    ((("S", LOW),), "and", O_L, "penalty: secondary skill missing"),
    ((("S", MED), ("P1", HIGH), ("P2", HIGH)), "and", O_H, "tempering: strong primaries, secondary only moderate"),
]
RULES_PER_CAREER = len(RULE_TEMPLATE)


@lru_cache(maxsize=1)
def build_rules() -> tuple[Rule, ...]:
    rules: list[Rule] = []
    for ci, c in enumerate(careers()):
        p1, p2 = (SKILLS.index(s) for s in c["rule_skills"]["primary"])
        slot = {"P1": p1, "P2": p2, "S": SKILLS.index(c["rule_skills"]["secondary"])}
        for k, (ants, op, cons, role) in enumerate(RULE_TEMPLATE, start=1):
            rules.append(Rule(f"{c['id']}.R{k}", ci, tuple((slot[a], m) for a, m in ants), op, cons, role))
    return tuple(rules)


@lru_cache(maxsize=1)
def _rule_tables() -> tuple[np.ndarray, np.ndarray]:
    """Career index and consequent set index of every rule."""
    rules = build_rules()
    return np.array([r.career_index for r in rules]), np.array([r.consequent for r in rules])


def clip_levels(strengths: np.ndarray) -> np.ndarray:
    """Rule strengths (n, R) → clip level of each output set per career (n, C, 5).

    Mamdani max–min: max_r min(α_r, μ_{c_r}(u)) = max_k min(max_{r: c_r=k} α_r, μ_k(u)),
    so the rules that share a consequent can be merged before clipping.
    """
    career_idx, consequents = _rule_tables()
    levels = np.zeros((strengths.shape[0], len(careers()), len(OUT_NAMES)))
    np.maximum.at(levels, (slice(None), career_idx, consequents), strengths)
    return levels


@lru_cache(maxsize=1)
def _antecedent_index() -> tuple[np.ndarray, np.ndarray]:
    """(R, 3) skill and set indices. Shorter rules repeat their first antecedent, which leaves min() unchanged."""
    rules = build_rules()
    assert all(r.op == "and" for r in rules), "vectorised path assumes AND rules"
    width = max(len(r.antecedents) for r in rules)
    pad = [list(r.antecedents) + [r.antecedents[0]] * (width - len(r.antecedents)) for r in rules]
    return np.array([[a[0] for a in p] for p in pad]), np.array([[a[1] for a in p] for p in pad])


def rule_strengths(mu_w: np.ndarray) -> np.ndarray:
    """Weighted memberships (n, 6, 3) → firing strength of every rule (n, R), AND = min."""
    si, mi = _antecedent_index()
    return mu_w[:, si, mi].min(axis=2)


def infer(X: np.ndarray, weights: np.ndarray, keep_curves: bool = False) -> dict[str, np.ndarray]:
    """Run every career FIS for a batch of students.

    Returns:
        s: suitability (n, C) in [0, 100]
        strengths: rule firing strengths (n, R)
        no_rule_fired: bool (n, C)
        aggregated, area, moment: (n, C, U), (n, C), (n, C), only if keep_curves
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    mu_w = apply_weights(fuzzify(X), weights)
    strengths = rule_strengths(mu_w)
    levels = clip_levels(strengths)
    clipped = np.minimum(levels[..., None], OUTPUT_MF[None, None, :, :])             # implication (min)
    aggregated = clipped.max(axis=2)                                                  # aggregation (max)
    area = aggregated.sum(axis=2)
    moment = (aggregated * UNIVERSE).sum(axis=2)
    no_fire = area <= 1e-12
    s = np.where(no_fire, 0.0, moment / np.where(no_fire, 1.0, area))               # centroid
    out = {"s": s, "strengths": strengths, "no_rule_fired": no_fire, "levels": levels}
    if keep_curves:
        out.update(aggregated=aggregated, area=area, moment=moment)
    return out
