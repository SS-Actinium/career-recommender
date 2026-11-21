"""Work-preference match W_c in [0, 1] (v2, DESIGN.md §"Work preferences").

Each answered preference is compared with the career's work_profile in
data/careers.json and scores 1 (match), 0.5 (partial) or 0 (mismatch).
W_c is the mean over the answered preferences. If the student answers
none, W is undefined and its share of the final score is redistributed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .data import careers

MATH_LEVEL = {"low": 0, "medium": 1, "high": 2}


def _one(key: str, value: str, wp: dict[str, Any]) -> float:
    if key == "work_type":
        return 1.0 if value in wp["work_type"] else 0.5 if value == "mixed" else 0.0
    if key == "environment":
        if value == wp["environment"]:
            return 1.0
        return 0.5 if "mixed" in (value, wp["environment"]) else 0.0
    if key == "activity":
        return 1.0 if value in wp["activity"] else 0.0
    if key == "advanced_math":  # willingness to learn advanced maths vs the career's demand
        diff = MATH_LEVEL[value] - MATH_LEVEL[wp["math_demand"]]
        return 1.0 if diff >= 0 else 0.5 if diff == -1 else 0.0
    raise KeyError(key)


def preference_match(prefs: dict[str, str]) -> tuple[np.ndarray | None, list[list[dict[str, Any]]]]:
    """Return (W per career or None if no preference answered, per-career detail rows)."""
    answered = {k: v for k, v in (prefs or {}).items() if v}
    detail = [[{"preference": k, "value": v, "match": _one(k, v, c["work_profile"])} for k, v in answered.items()]
              for c in careers()]
    if not answered:
        return None, detail
    return np.array([np.mean([d["match"] for d in rows]) for rows in detail]), detail
