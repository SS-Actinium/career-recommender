"""Run the five demonstration profiles through the full pipeline using the saved GA weights.

Usage:  python demo_profiles.py
"""
from __future__ import annotations

import json

import numpy as np

from engine.data import DATA_DIR, SKILLS

DEMO_PROFILES: dict[str, dict] = {
    "Technology-oriented": {
        "skills": [88, 60, 85, 45, 50, 92], "interests": ["software", "cloud"],
        "work_preferences": {"work_type": "technical", "activity": "building"}},
    "Mathematics and data-oriented": {
        "skills": [60, 90, 82, 40, 60, 70], "interests": ["data", "ai"],
        "work_preferences": {"work_type": "analytical", "advanced_math": "high"}},
    "Creative and design-oriented": {
        "skills": [35, 35, 70, 92, 75, 55], "interests": ["design"],
        "work_preferences": {"work_type": "creative", "environment": "collaborative"}},
    "Communication and business-oriented": {
        "skills": [25, 60, 72, 55, 90, 45], "interests": ["business", "marketing"]},
    "Balanced with mixed interests": {
        "skills": [60, 60, 65, 60, 60, 60], "interests": ["data", "design", "business"]},
}


def load_weights() -> np.ndarray:
    with open(DATA_DIR / "ga_results.json", encoding="utf-8") as fh:
        return np.array(json.load(fh)["optimized_weights"])


def main() -> None:
    from engine.recommender import recommend

    w = load_weights()
    for name, p in DEMO_PROFILES.items():
        student = {"skills": dict(zip(SKILLS, p["skills"])), "interests": p["interests"],
                   "work_preferences": p.get("work_preferences", {})}
        out = recommend(student, w)
        print(f"\n{name}  skills={p['skills']}  interests={p['interests']}")
        for r in out["recommendations"]:
            pm = "  -" if r["preference_match"] is None else f"{r['preference_match']:.2f}"
            print(f"  {r['rank']}. {r['name']:<30s} score={r['score']:5.1f} ({r['fit']['label']})  "
                  f"FIS={r['fis_suitability']:5.1f}  I={r['interest_match']:.2f}  W={pm}")


if __name__ == "__main__":
    main()
