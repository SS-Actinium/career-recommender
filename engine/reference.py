"""Synthetic reference profiles and the separate expert labelling function.

SYNTHETIC DATA: these profiles are generated, not real student responses.
The expert function is independent of the FIS rules but still encodes the
team's own assumptions (DESIGN.md, "Reference data").
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .data import DATA_DIR, SKILLS, careers, interests, target_matrix
from .interest import interest_match

N_PROFILES = 60
N_TRAIN = 40
SEED = 7
NOISE_SD = 12.0
BLEND_SHARE = 0.2


def expert_scores(x: np.ndarray, selected: list[str]) -> np.ndarray:
    """expert_c = 0.65·fit_c + 0.35·I_c, with fit_c = 1 − Σ t·max(0, t−x) / Σ t²."""
    T = target_matrix()
    unmet = (T * np.maximum(0.0, T - x[None, :])).sum(axis=1)
    fit = 1.0 - unmet / (T * T).sum(axis=1)
    return 0.65 * fit + 0.35 * interest_match(selected)


def reference_top5(x: np.ndarray, selected: list[str]) -> list[int]:
    ids = np.array([c["id"] for c in careers()])
    e = expert_scores(x, selected)
    return [int(i) for i in np.lexsort((ids, -np.round(e, 9)))[:5]]


def generate_profiles(n: int = N_PROFILES, seed: int = SEED) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    T = target_matrix()
    tags = list(interests())
    cs = careers()
    profiles = []
    for k in range(n):
        a = int(rng.integers(len(cs)))
        base = T[a]
        archetypes = [cs[a]["id"]]
        if rng.random() < BLEND_SHARE:
            b = int(rng.integers(len(cs)))
            base = (T[a] + T[b]) / 2.0
            archetypes.append(cs[b]["id"])
        x = np.clip(np.round(base + rng.normal(0, NOISE_SD, len(SKILLS))), 0, 100)
        own = [t for aid in archetypes for t in next(c for c in cs if c["id"] == aid)["interests"]]
        own = list(dict.fromkeys(own))
        chosen = [t for t in own if rng.random() < 0.85]
        chosen += [t for t in tags if t not in own and rng.random() < 0.15]
        if not chosen:
            chosen = [own[0]]
        primary = [t for t in chosen if t in own]
        rest = [t for t in chosen if t not in own]
        rng.shuffle(primary)
        rng.shuffle(rest)
        selected = primary + rest  # the archetype's interests are usually ranked first
        profiles.append({
            "id": f"syn-{k + 1:03d}",
            "synthetic": True,
            "archetypes": archetypes,
            "skills": {s: int(v) for s, v in zip(SKILLS, x)},
            "interests": selected,
            "reference_top5": [cs[i]["id"] for i in reference_top5(x, selected)],
            "split": "train" if k < N_TRAIN else "holdout",
        })
    return profiles


def save_profiles(path=DATA_DIR / "reference_profiles.json") -> list[dict[str, Any]]:
    profiles = generate_profiles()
    payload = {"_note": "SYNTHETIC profiles for GA training and evaluation. Not real student data.",
               "seed": SEED, "n_train": N_TRAIN, "profiles": profiles}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    return profiles


def load_profiles(path=DATA_DIR / "reference_profiles.json") -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["profiles"]


def as_arrays(profiles: list[dict[str, Any]]) -> tuple[np.ndarray, list[list[str]], list[list[int]]]:
    idx = {c["id"]: i for i, c in enumerate(careers())}
    X = np.array([[p["skills"][s] for s in SKILLS] for p in profiles], dtype=float)
    sel = [p["interests"] for p in profiles]
    ref = [[idx[c] for c in p["reference_top5"]] for p in profiles]
    return X, sel, ref
