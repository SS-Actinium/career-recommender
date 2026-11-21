"""Mamdani Fuzzy Inference System: one FIS per career.

Stages: fuzzification → GA weight scaling → rule evaluation (AND=min, OR=max)
→ min-implication → max-aggregation → centroid defuzzification.

Every career is evaluated in one vectorised NumPy pass so the GA can run
thousands of evaluations quickly. The maths is the same as running 18
separate Mamdani systems.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .data import SKILL_LABELS, SKILLS, careers

SET_NAMES: list[str] = ["Low", "Medium", "High"]
LOW, MED, HIGH = 0, 1, 2
# Triangular breakpoints (a, b, c), shared by inputs and output (DESIGN.md).
MF_PARAMS: list[tuple[float, float, float]] = [(0, 0, 50), (25, 50, 75), (50, 100, 100)]
UNIVERSE: np.ndarray = np.linspace(0.0, 100.0, 201)  # output universe, step 0.5


def trimf(x: np.ndarray | float, a: float, b: float, c: float) -> np.ndarray:
    """Triangular membership function. Handles shoulders where a == b or b == c."""
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if a == b else (x - a) / (b - a)
    right = np.ones_like(x) if b == c else (c - x) / (c - b)
    out = np.minimum(left, right)
    # A shoulder set is 1 at its flat end, including exactly at b.
    out = np.where(x == b, 1.0, out)
    return np.clip(out, 0.0, 1.0)


def fuzzify(X: np.ndarray) -> np.ndarray:
    """Crisp inputs (n, 6) in [0, 100] → memberships (n, 6, 3) in [0, 1]."""
    X = np.asarray(X, dtype=float)
    return np.stack([trimf(X, *p) for p in MF_PARAMS], axis=-1)


OUTPUT_MF: np.ndarray = np.stack([trimf(UNIVERSE, *p) for p in MF_PARAMS])  # (3, U)


def apply_weights(mu: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """GA weight scaling μ'_i = min(1, 6·w_i·μ_i). Equal weights leave μ unchanged."""
    w = np.asarray(weights, dtype=float).reshape(1, len(SKILLS), 1)
    return np.minimum(1.0, len(SKILLS) * w * mu)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    career_index: int
    antecedents: tuple[tuple[int, int], ...]  # (skill index, set index)
    op: str  # "and" | "or"
    consequent: int

    def text(self, career_name: str) -> str:
        joiner = " AND " if self.op == "and" else " OR "
        cond = joiner.join(f"{SKILL_LABELS[SKILLS[s]]} is {SET_NAMES[m]}" for s, m in self.antecedents)
        return f"IF {cond} THEN {career_name} suitability is {SET_NAMES[self.consequent]}"


@lru_cache(maxsize=1)
def build_rules() -> tuple[Rule, ...]:
    """Seven rules per career, built from its primary pair (P1, P2) and secondary skill S."""
    rules: list[Rule] = []
    for ci, c in enumerate(careers()):
        p1, p2 = (SKILLS.index(s) for s in c["rule_skills"]["primary"])
        s = SKILLS.index(c["rule_skills"]["secondary"])
        spec = [
            (((p1, HIGH), (p2, HIGH)), "and", HIGH),
            (((p1, HIGH), (p2, MED)), "and", MED),
            (((p1, MED), (p2, HIGH)), "and", MED),
            (((p1, MED), (p2, MED)), "and", MED),
            (((p1, LOW), (p2, LOW)), "or", LOW),
            (((s, HIGH), (p1, HIGH)), "and", HIGH),
            (((s, LOW),), "and", LOW),
        ]
        for k, (ants, op, cons) in enumerate(spec, start=1):
            rules.append(Rule(f"{c['id']}.R{k}", ci, ants, op, cons))
    return tuple(rules)


@lru_cache(maxsize=1)
def _rule_tables() -> tuple[np.ndarray, np.ndarray]:
    rules = build_rules()
    career_idx = np.array([r.career_index for r in rules])
    starts = np.flatnonzero(np.r_[True, career_idx[1:] != career_idx[:-1]])
    consequents = np.array([r.consequent for r in rules])
    return starts, consequents


def rule_strengths(mu_w: np.ndarray) -> np.ndarray:
    """Weighted memberships (n, 6, 3) → firing strength of every rule (n, R)."""
    cols = []
    for r in build_rules():
        vals = np.stack([mu_w[:, s, m] for s, m in r.antecedents], axis=1)
        cols.append(vals.min(axis=1) if r.op == "and" else vals.max(axis=1))
    return np.stack(cols, axis=1)


def infer(X: np.ndarray, weights: np.ndarray, keep_curves: bool = False) -> dict[str, np.ndarray]:
    """Run every career FIS for a batch of students.

    Returns:
        s: suitability (n, C) in [0, 100]
        strengths: rule firing strengths (n, R)
        no_rule_fired: bool (n, C)
        aggregated: (n, C, U) aggregated output sets, only if keep_curves
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    mu_w = apply_weights(fuzzify(X), weights)
    strengths = rule_strengths(mu_w)
    starts, consequents = _rule_tables()
    clipped = np.minimum(strengths[:, :, None], OUTPUT_MF[consequents][None, :, :])  # implication (min)
    aggregated = np.maximum.reduceat(clipped, starts, axis=1)                        # aggregation (max)
    area = aggregated.sum(axis=2)
    moment = (aggregated * UNIVERSE).sum(axis=2)
    no_fire = area <= 1e-12
    s = np.where(no_fire, 0.0, moment / np.where(no_fire, 1.0, area))               # centroid
    out = {"s": s, "strengths": strengths, "no_rule_fired": no_fire}
    if keep_curves:
        out["aggregated"] = aggregated
    return out
