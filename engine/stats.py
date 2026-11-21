"""Bootstrap confidence intervals for the evaluation (percentile method)."""
from __future__ import annotations

import numpy as np

N_BOOT = 2000
BOOT_SEED = 2024


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED, level: float = 0.95) -> dict[str, float]:
    """Mean and percentile CI of per-profile scores, resampling profiles with replacement."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi)}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict[str, float]:
    """CI of mean(b − a) on the same profiles, and the share of resamples where b is not better."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    ci = bootstrap_ci(d, n_boot, seed)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return {**ci, "p_not_better": float(np.mean(means <= 0))}
