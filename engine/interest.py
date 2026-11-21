"""Interest match I_c in [0, 1]. Interests are not fuzzy inputs (see DESIGN.md)."""
from __future__ import annotations

import numpy as np

from .data import careers


def rank_weights(selected: list[str]) -> dict[str, float]:
    """With k ranked interests, the one at position r (1-based) has weight k − r + 1."""
    k = len(selected)
    return {tag: float(k - r) for r, tag in enumerate(selected)}


def interest_match(selected: list[str]) -> np.ndarray:
    """Share of the student's rank-weighted interests that are linked to each career → (C,)."""
    weights = rank_weights(selected)
    total = sum(weights.values())
    if total == 0:
        return np.zeros(len(careers()))
    return np.array([sum(weights.get(t, 0.0) for t in c["interests"]) / total for c in careers()])


def interest_match_batch(selections: list[list[str]]) -> np.ndarray:
    return np.stack([interest_match(s) for s in selections])
