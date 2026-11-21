"""Hybrid recommendation: FIS suitability + interest match → ranked careers, explanations, skill gaps."""
from __future__ import annotations

from typing import Any

import numpy as np

from . import fuzzy
from .data import SKILL_LABELS, SKILLS, careers, interests, target_matrix
from .interest import interest_match, interest_match_batch

FIS_SHARE = 0.7
INTEREST_SHARE = 0.3
TOP_K = 5
DISCLAIMER = ("This is guidance for career exploration, not a guarantee of suitability or success. "
              "Career targets are model assumptions, not official industry standards.")


def final_scores(s: np.ndarray, interest: np.ndarray) -> np.ndarray:
    """Score_c = 100 · (0.7 · s_c/100 + 0.3 · I_c). The result is already on 0–100."""
    return 100.0 * (FIS_SHARE * s / 100.0 + INTEREST_SHARE * interest)


def rank(scores: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Career indices sorted by score, then FIS suitability, then career id (deterministic ties)."""
    ids = np.array([c["id"] for c in careers()])
    return np.lexsort((ids, -np.round(s, 6), -np.round(scores, 6)))


def top_k_batch(X: np.ndarray, selections: list[list[str]], weights: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """Top-k career indices for many students → (n, k). Used by the GA fitness."""
    s = fuzzy.infer(X, weights)["s"]
    scores = final_scores(s, interest_match_batch(selections))
    return np.stack([rank(scores[i], s[i])[:k] for i in range(len(X))])


def _contributions(x: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Explanation heuristic (not part of the score): (t/100) · (min(x,t)/100) · 6w."""
    return (targets / 100.0) * (np.minimum(x, targets) / 100.0) * (len(SKILLS) * weights)


def _skill_rows(x: np.ndarray, targets: np.ndarray) -> list[dict[str, Any]]:
    return [{"skill": SKILLS[i], "label": SKILL_LABELS[SKILLS[i]], "student": float(x[i]),
             "target": float(targets[i]), "gap": float(targets[i] - x[i])} for i in range(len(SKILLS))]


def _explanation(name: str, score: float, strengths: list[str], gaps: list[dict[str, Any]],
                 matched_interests: list[str]) -> str:
    level = "strong" if score >= 70 else "moderate" if score >= 50 else "partial"
    parts = [f"{name} is a {level} match ({score:.1f}%)."]
    if strengths:
        parts.append(f"Your {', '.join(strengths[:-1]) + ' and ' + strengths[-1] if len(strengths) > 1 else strengths[0]} "
                     f"ratings contribute most to this career profile.")
    if matched_interests:
        parts.append(f"It also matches your interest in {', '.join(matched_interests)}.")
    else:
        parts.append("None of your selected interests are linked to this career, so the score comes from your skills only.")
    if gaps:
        parts.append("You may benefit from improving " + ", ".join(f"{g['label']} (gap {g['gap']:.0f})" for g in gaps[:2]) + ".")
    else:
        parts.append("Your ratings meet or exceed every target level for this career.")
    parts.append("This is guidance, not a guarantee.")
    return " ".join(parts)


def recommend(student: dict[str, Any], weights: np.ndarray, top_k: int = TOP_K) -> dict[str, Any]:
    """Run the full pipeline for one validated student and build the API response."""
    x = np.array([student["skills"][s] for s in SKILLS], dtype=float)
    selected: list[str] = student.get("interests", [])
    res = fuzzy.infer(x[None, :], weights, keep_curves=True)
    s, strengths, aggregated = res["s"][0], res["strengths"][0], res["aggregated"][0]
    interest = interest_match(selected)
    scores = final_scores(s, interest)
    order = rank(scores, s)
    T = target_matrix()
    rules = fuzzy.build_rules()
    all_careers = careers()
    interest_labels = interests()

    results = []
    for position, ci in enumerate(order[:top_k], start=1):
        c = all_careers[ci]
        rows = _skill_rows(x, T[ci])
        contrib = _contributions(x, T[ci], weights)
        top_strengths = [SKILL_LABELS[SKILLS[i]] for i in np.argsort(-contrib)[:3] if contrib[i] > 0]
        gaps = sorted([r for r in rows if r["gap"] > 0], key=lambda r: -r["gap"])
        matched = [interest_labels[t] for t in selected if t in c["interests"]]
        fired = [{"id": r.rule_id, "text": r.text(c["name"]), "strength": round(float(strengths[k]), 4)}
                 for k, r in enumerate(rules) if r.career_index == ci and strengths[k] > 0]
        results.append({
            "rank": position, "id": c["id"], "name": c["name"], "description": c["description"],
            "score": round(float(scores[ci]), 1),
            "fis_suitability": round(float(s[ci]), 2),
            "interest_match": round(float(interest[ci]), 3),
            "no_rule_fired": bool(res["no_rule_fired"][0][ci]),
            "strengths": top_strengths,
            "gaps": gaps,
            "skills": rows,
            "matched_interests": matched,
            "key_skills": c["key_skills"],
            "learning": c["learning"],
            "fired_rules": sorted(fired, key=lambda f: -f["strength"]),
            "aggregated_curve": [round(float(v), 4) for v in aggregated[ci][::4]],  # every 2 points on 0–100
            "explanation": _explanation(c["name"], float(scores[ci]), top_strengths, gaps, matched),
        })

    mu = fuzzy.fuzzify(x[None, :])[0]
    mu_w = fuzzy.apply_weights(mu[None], weights)[0]
    return {
        "display_name": student.get("display_name") or None,
        "recommendations": results,
        "all_scores": [{"id": all_careers[ci]["id"], "name": all_careers[ci]["name"],
                        "score": round(float(scores[ci]), 1)} for ci in order],
        "work_preferences": student.get("work_preferences", {}),
        "analysis": {
            "inputs": {s_: float(x[i]) for i, s_ in enumerate(SKILLS)},
            "memberships": {SKILLS[i]: dict(zip(fuzzy.SET_NAMES, map(lambda v: round(float(v), 4), mu[i])))
                            for i in range(len(SKILLS))},
            "weighted_memberships": {SKILLS[i]: dict(zip(fuzzy.SET_NAMES, map(lambda v: round(float(v), 4), mu_w[i])))
                                     for i in range(len(SKILLS))},
            "weights_used": [round(float(w), 4) for w in weights],
            "curve_universe": [float(u) for u in fuzzy.UNIVERSE[::4]],
        },
        "disclaimer": DISCLAIMER,
    }
