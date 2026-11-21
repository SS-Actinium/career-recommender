"""Hybrid recommendation: FIS suitability + interest match + work-preference match
→ ranked careers, fit bands, explanations, skill gaps and a learning roadmap."""
from __future__ import annotations

from typing import Any

import numpy as np

from . import fuzzy
from .data import SKILL_LABELS, SKILLS, careers, interests, load_dataset, target_matrix
from .interest import interest_match, interest_match_batch
from .preferences import preference_match

FIS_SHARE = 0.65
INTEREST_SHARE = 0.25
PREF_SHARE = 0.10
TOP_K = 5
# Fit bands on the final 0–100 score (DESIGN.md §"v2 scoring").
BANDS: list[tuple[float, str, str]] = [
    (70.0, "Strong fit", "strong"),
    (55.0, "Good fit", "good"),
    (40.0, "Moderate fit", "moderate"),
    (0.0, "Weak fit", "weak"),
]
DISCLAIMER = ("This is guidance for career exploration, not a guarantee of suitability or success. "
              "Career targets are model assumptions, not official industry standards.")


def score_formula(with_prefs: bool) -> str:
    if with_prefs:
        return f"Score = s × ({FIS_SHARE} + {INTEREST_SHARE}·I + {PREF_SHARE}·W)"
    total = FIS_SHARE + INTEREST_SHARE
    return f"Score = s × ({FIS_SHARE} + {INTEREST_SHARE}·I) / {total:.2f}   (no work preferences given)"


def multipliers(interest: np.ndarray, prefs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Base, interest and preference parts of the multiplier m (they sum to m ≤ 1)."""
    if prefs is None:
        total = FIS_SHARE + INTEREST_SHARE
        return np.full_like(interest, FIS_SHARE / total), INTEREST_SHARE * interest / total, None
    return np.full_like(interest, FIS_SHARE), INTEREST_SHARE * interest, PREF_SHARE * prefs


def final_scores(s: np.ndarray, interest: np.ndarray, prefs: np.ndarray | None = None) -> np.ndarray:
    """Score = s · m with m = 0.65 + 0.25·I + 0.10·W ∈ [0.65, 1].

    Interest and preferences scale the skill-based suitability; they can lower it for a
    career you are not drawn to, but can never lift a weak skill profile above s.
    Without preferences, the 0.10 share is redistributed pro rata.
    """
    base, i_part, p_part = multipliers(interest, prefs)
    m = base + i_part + (0.0 if p_part is None else p_part)
    return s * m


def band(score: float) -> dict[str, str]:
    for lo, name, key in BANDS:
        if score >= lo:
            return {"label": name, "key": key}
    return {"label": BANDS[-1][1], "key": BANDS[-1][2]}


def rank(scores: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Career indices sorted by score, then FIS suitability, then career id (deterministic ties)."""
    ids = np.array([c["id"] for c in careers()])
    return np.lexsort((ids, -np.round(s, 6), -np.round(scores, 6)))


def top_k_batch(X: np.ndarray, selections: list[list[str]] | np.ndarray, weights: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """Top-k career indices for many students → (n, k). Used by the GA fitness (no preferences).
    `selections` may be a precomputed interest-match matrix (n, C) to save time in the GA."""
    s = fuzzy.infer(X, weights)["s"]
    interest = selections if isinstance(selections, np.ndarray) else interest_match_batch(selections)
    scores = final_scores(s, interest)
    rank_ids = np.broadcast_to(_id_rank(), s.shape)
    return np.lexsort((rank_ids, -np.round(s, 6), -np.round(scores, 6)), axis=-1)[:, :k]


def _id_rank() -> np.ndarray:
    """Position of each career id in alphabetical order (the final tie-break)."""
    ids = [c["id"] for c in careers()]
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    r = np.empty(len(ids), dtype=int)
    r[order] = np.arange(len(ids))
    return r


def _contributions(x: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Explanation heuristic (not part of the score): (t/100) · (min(x,t)/100) · 6w."""
    return (targets / 100.0) * (np.minimum(x, targets) / 100.0) * (len(SKILLS) * weights)


def _skill_rows(x: np.ndarray, targets: np.ndarray) -> list[dict[str, Any]]:
    return [{"skill": SKILLS[i], "label": SKILL_LABELS[SKILLS[i]], "student": float(x[i]),
             "target": float(targets[i]), "gap": float(targets[i] - x[i])} for i in range(len(SKILLS))]


def roadmap(gaps: list[dict[str, Any]], career: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-gap next steps. The stage depends on the student's current rating, the priority on the gap size."""
    table = load_dataset()["skill_roadmap"]
    out = []
    for g in gaps:
        stage = "foundation" if g["student"] < 40 else "intermediate" if g["student"] < 70 else "advanced"
        priority = "high" if g["gap"] >= 25 else "medium" if g["gap"] >= 10 else "low"
        out.append({"skill": g["skill"], "label": g["label"], "gap": g["gap"], "current": g["student"],
                    "target": g["target"], "stage": stage, "priority": priority, "steps": table[g["skill"]][stage]})
    return out


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _explanation(name: str, score: float, fit: str, strengths: list[str], gaps: list[dict[str, Any]],
                 matched_interests: list[str], pref_value: float | None) -> str:
    parts = [f"{name} is a {fit.lower()} for you ({score:.1f} / 100)."]
    if strengths:
        parts.append(f"Your {_join(strengths)} ratings contribute most to this career profile.")
    if matched_interests:
        parts.append(f"It matches your interest in {_join(matched_interests)}.")
    else:
        parts.append("None of your selected interests are linked to this career, so the score comes mainly from your skills.")
    if pref_value is not None:
        parts.append("Your work preferences fit it well." if pref_value >= 0.75 else
                     "Your work preferences partly fit it." if pref_value >= 0.4 else
                     "Your work preferences do not fit it well.")
    if gaps:
        parts.append("You may benefit from improving " + ", ".join(f"{g['label']} (gap {g['gap']:.0f})" for g in gaps[:2]) + ".")
    else:
        parts.append("Your ratings meet or exceed every target level for this career.")
    return " ".join(parts)


def template_summary(name: str | None, recs: list[dict[str, Any]]) -> str:
    """Deterministic overall summary, used when no LLM is configured or it fails."""
    who = f"{name}, your" if name else "Your"
    top = recs[0]
    others = [r["name"] for r in recs[1:3]]
    gaps = {}
    for r in recs[:3]:
        for g in r["gaps"]:
            gaps[g["label"]] = max(gaps.get(g["label"], 0), g["gap"])
    common = sorted(gaps, key=lambda k: -gaps[k])[:2]
    text = (f"{who} profile points most strongly to {top['name']} ({top['fit']['label'].lower()}, {top['score']:.0f}/100)"
            + (f", followed by {_join(others)}." if others else "."))
    if top["strengths"]:
        text += f" The ratings that carry these matches are {_join(top['strengths'])}."
    if common:
        text += f" Across your top three careers, the biggest skill gap is in {_join(common)}, so that is where focused practice will help most."
    else:
        text += " You already meet the target levels for your top careers, so depth and real projects are the next step."
    text += " Treat this as a starting point for exploration and talk it through with a mentor or counsellor."
    return text


def _worked_example(ci: int, x: np.ndarray, mu: np.ndarray, mu_w: np.ndarray, strengths: np.ndarray,
                    area: float, moment: float, s: float) -> dict[str, Any]:
    """Step-by-step numbers for the top career, shown on the Soft Computing page."""
    rules = fuzzy.build_rules()
    rows, clip = [], [0.0] * len(fuzzy.OUT_NAMES)
    for k, r in enumerate(rules):
        if r.career_index != ci:
            continue
        ants = [{"skill": SKILL_LABELS[SKILLS[si]], "set": fuzzy.SET_NAMES[m], "x": float(x[si]),
                 "mu": round(float(mu[si, m]), 4), "mu_w": round(float(mu_w[si, m]), 4)} for si, m in r.antecedents]
        clip[r.consequent] = max(clip[r.consequent], float(strengths[k]))
        rows.append({"id": r.rule_id, "op": r.op.upper(), "antecedents": ants, "consequent": fuzzy.OUT_NAMES[r.consequent],
                     "strength": round(float(strengths[k]), 4), "role": r.role})
    return {"career_index": ci, "rules": rows,
            "clip_levels": dict(zip(fuzzy.OUT_NAMES, (round(v, 4) for v in clip))),
            "area": round(area, 4), "moment": round(moment, 4), "centroid": round(s, 4),
            "step": float(fuzzy.UNIVERSE[1] - fuzzy.UNIVERSE[0])}


def recommend(student: dict[str, Any], weights: np.ndarray, top_k: int = TOP_K) -> dict[str, Any]:
    """Run the full pipeline for one validated student and build the API response."""
    x = np.array([student["skills"][s] for s in SKILLS], dtype=float)
    selected: list[str] = student.get("interests", [])
    prefs_in = student.get("work_preferences", {}) or {}
    res = fuzzy.infer(x[None, :], weights, keep_curves=True)
    s, strengths, aggregated = res["s"][0], res["strengths"][0], res["aggregated"][0]
    interest = interest_match(selected)
    W, pref_detail = preference_match(prefs_in)
    scores = final_scores(s, interest, W)
    order = rank(scores, s)
    T = target_matrix()
    rules = fuzzy.build_rules()
    all_careers = careers()
    interest_labels = interests()
    mu = fuzzy.fuzzify(x[None, :])[0]
    mu_w = fuzzy.apply_weights(mu[None], weights)[0]

    base_m, int_m, pref_m = multipliers(interest, W)

    def breakdown(ci: int) -> dict[str, float | None]:
        """Score = s·base + s·interest part + s·preference part (points that sum to the score)."""
        return {"fis_points": round(float(s[ci] * base_m[ci]), 2),
                "interest_points": round(float(s[ci] * int_m[ci]), 2),
                "preference_points": None if pref_m is None else round(float(s[ci] * pref_m[ci]), 2),
                "multiplier": round(float(base_m[ci] + int_m[ci] + (0 if pref_m is None else pref_m[ci])), 4)}

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
        fit = band(float(scores[ci]))
        pref_value = None if W is None else float(W[ci])
        results.append({
            "rank": position, "id": c["id"], "name": c["name"], "description": c["description"],
            "score": round(float(scores[ci]), 1),
            "fit": fit,
            "fis_suitability": round(float(s[ci]), 2),
            "interest_match": round(float(interest[ci]), 3),
            "preference_match": None if pref_value is None else round(pref_value, 3),
            "preference_detail": pref_detail[ci],
            "breakdown": breakdown(ci),
            "no_rule_fired": bool(res["no_rule_fired"][0][ci]),
            "strengths": top_strengths,
            "gaps": gaps,
            "roadmap": roadmap(gaps, c),
            "skills": rows,
            "matched_interests": matched,
            "key_skills": c["key_skills"],
            "learning": c["learning"],
            "fired_rules": sorted(fired, key=lambda f: -f["strength"]),
            "aggregated_curve": [round(float(v), 4) for v in aggregated[ci][::4]],  # every 2 points on 0–100
            "explanation": _explanation(c["name"], float(scores[ci]), fit["label"], top_strengths, gaps, matched, pref_value),
        })

    top_ci = int(order[0])
    return {
        "display_name": student.get("display_name") or None,
        "recommendations": results,
        "all_scores": [{"id": all_careers[ci]["id"], "name": all_careers[ci]["name"],
                        "score": round(float(scores[ci]), 1), "fit": band(float(scores[ci])),
                        "fis_suitability": round(float(s[ci]), 2), "interest_match": round(float(interest[ci]), 3),
                        "preference_match": None if W is None else round(float(W[ci]), 3)} for ci in order],
        "work_preferences": {k: v for k, v in prefs_in.items() if v},
        "interests": selected,
        "score_formula": score_formula(W is not None),
        "summary": {"text": template_summary(student.get("display_name") or None, results), "source": "template"},
        "analysis": {
            "inputs": {s_: float(x[i]) for i, s_ in enumerate(SKILLS)},
            "memberships": {SKILLS[i]: dict(zip(fuzzy.SET_NAMES, map(lambda v: round(float(v), 4), mu[i])))
                            for i in range(len(SKILLS))},
            "weighted_memberships": {SKILLS[i]: dict(zip(fuzzy.SET_NAMES, map(lambda v: round(float(v), 4), mu_w[i])))
                                     for i in range(len(SKILLS))},
            "weights_used": [round(float(w), 4) for w in weights],
            "curve_universe": [float(u) for u in fuzzy.UNIVERSE[::4]],
            "worked_example": _worked_example(top_ci, x, mu, mu_w, strengths, float(res["area"][0][top_ci]),
                                              float(res["moment"][0][top_ci]), float(s[top_ci])),
        },
        "disclaimer": DISCLAIMER,
    }
