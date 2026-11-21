"""Career dataset loading and shared constants."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

SKILLS: list[str] = [
    "programming", "mathematics", "problem_solving",
    "creativity", "communication", "technical_interest",
]
SKILL_LABELS: dict[str, str] = {
    "programming": "Programming",
    "mathematics": "Mathematics",
    "problem_solving": "Problem-solving",
    "creativity": "Creativity",
    "communication": "Communication",
    "technical_interest": "Technical Interest",
}


@lru_cache(maxsize=1)
def load_dataset() -> dict[str, Any]:
    """Load data/careers.json once and cache it."""
    with open(DATA_DIR / "careers.json", encoding="utf-8") as fh:
        return json.load(fh)


def careers() -> list[dict[str, Any]]:
    return load_dataset()["careers"]


def interests() -> dict[str, str]:
    return load_dataset()["interests"]


def career_by_id(career_id: str) -> dict[str, Any] | None:
    return next((c for c in careers() if c["id"] == career_id), None)


def target_matrix() -> np.ndarray:
    """Return an array of shape (n_careers, 6) holding the target levels, in SKILLS order."""
    return np.array([[c["targets"][s] for s in SKILLS] for c in careers()], dtype=float)
