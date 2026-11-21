"""Server-side validation of assessment payloads."""
from __future__ import annotations

from typing import Any

from .data import SKILLS, interests

WORK_PREF_OPTIONS: dict[str, set[str]] = {
    "work_type": {"technical", "creative", "analytical", "people", "mixed"},
    "environment": {"independent", "collaborative", "mixed"},
    "activity": {"building", "researching", "analysing", "designing", "supporting", "managing"},
    "advanced_math": {"low", "medium", "high"},
}


class ValidationError(ValueError):
    def __init__(self, errors: dict[str, str]):
        super().__init__("invalid input")
        self.errors = errors


def _skills(skills_in: Any, errors: dict[str, str]) -> dict[str, float]:
    skills: dict[str, float] = {}
    if not isinstance(skills_in, dict):
        errors["skills"] = "Skills are required."
        return skills
    for s in SKILLS:
        v = skills_in.get(s)
        if v is None:
            errors[f"skills.{s}"] = "This rating is required."
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            errors[f"skills.{s}"] = "Rating must be a number."
        elif not 0 <= v <= 100 or v != v:
            errors[f"skills.{s}"] = "Rating must be between 0 and 100."
        else:
            skills[s] = float(v)
    extra = set(skills_in) - set(SKILLS)
    if extra:
        errors["skills"] = f"Unknown skills: {', '.join(sorted(map(str, extra)))[:200]}."
    return skills


def validate_playground(payload: Any, career_ids: list[str]) -> dict[str, Any]:
    """Skills plus one career id and a weight mode, for the live playground."""
    if not isinstance(payload, dict):
        raise ValidationError({"body": "Request body must be a JSON object."})
    errors: dict[str, str] = {}
    skills = _skills(payload.get("skills"), errors)
    career = payload.get("career_id")
    if career not in career_ids:
        errors["career_id"] = "Unknown career."
    mode = payload.get("weights", "optimized")
    if mode not in ("optimized", "equal"):
        errors["weights"] = "Weights must be 'optimized' or 'equal'."
    if errors:
        raise ValidationError(errors)
    return {"skills": skills, "career_id": career, "weights": mode}


def validate_assessment(payload: Any) -> dict[str, Any]:
    """Return a clean student dict, or raise ValidationError with per-field messages."""
    if not isinstance(payload, dict):
        raise ValidationError({"body": "Request body must be a JSON object."})
    errors: dict[str, str] = {}

    skills = _skills(payload.get("skills"), errors)

    sel = payload.get("interests", [])
    valid_tags = set(interests())
    if not isinstance(sel, list) or not all(isinstance(t, str) for t in sel):
        errors["interests"] = "Interests must be a list of interest ids."
        sel = []
    elif len(set(sel)) != len(sel) or not set(sel) <= valid_tags:
        errors["interests"] = "Interests must be unique, known interest ids."
    elif not sel:
        errors["interests"] = "Select at least one interest."

    prefs_in = payload.get("work_preferences") or {}
    prefs: dict[str, str] = {}
    if not isinstance(prefs_in, dict):
        errors["work_preferences"] = "Work preferences must be an object."
    else:
        for key, value in prefs_in.items():
            if key not in WORK_PREF_OPTIONS or value not in WORK_PREF_OPTIONS[key]:
                errors[f"work_preferences.{str(key)[:40]}"] = "Unknown option."
            else:
                prefs[key] = value

    name = payload.get("display_name") or ""
    if not isinstance(name, str) or len(name) > 40:
        errors["display_name"] = "Display name must be text of at most 40 characters."
        name = ""

    if errors:
        raise ValidationError(errors)
    return {"skills": skills, "interests": sel, "work_preferences": prefs, "display_name": name.strip()}
