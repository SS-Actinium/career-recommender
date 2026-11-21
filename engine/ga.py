"""Custom NumPy Genetic Algorithm that tunes the six input weights.

Chromosome: six non-negative weights that sum to 1.
Operators: tournament selection (k=3), blend crossover BLX-α (α=0.5, p=0.8),
Gaussian mutation (σ=0.05, p=0.2 per child), elitism (1), and repair
(clip to ≥0, then normalise).
Fitness: mean NDCG@5 against the synthetic reference rankings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .recommender import top_k_batch

N_GENES = 6
_DISCOUNT = 1.0 / np.log2(np.arange(2, 7))  # positions 1..5
_IDEAL_DCG = float(np.dot([5, 4, 3, 2, 1], _DISCOUNT))


@dataclass
class GAConfig:
    population: int = 30
    generations: int = 40
    crossover_prob: float = 0.8
    blend_alpha: float = 0.5
    mutation_prob: float = 0.2
    mutation_sigma: float = 0.05
    tournament_size: int = 3
    elitism: int = 1
    seed: int = 42
    seed_equal_weights: bool = True


@dataclass
class GAResult:
    best_weights: list[float]
    best_fitness: float
    initial_best_weights: list[float]
    initial_best_fitness: float
    history: list[dict[str, float]] = field(default_factory=list)
    config: dict = field(default_factory=dict)


def normalize(w: np.ndarray) -> np.ndarray:
    """Repair a chromosome (or a batch of them): clip negatives to 0 and scale to sum 1. All-zero → uniform."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    total = w.sum(axis=-1, keepdims=True)
    uniform = np.full_like(w, 1.0 / w.shape[-1])
    return np.where(total > 0, w / np.where(total > 0, total, 1.0), uniform)


def ndcg_at_5(model_top: list[int] | np.ndarray, reference_top: list[int]) -> float:
    grade = {c: 5 - i for i, c in enumerate(reference_top)}
    dcg = sum(grade.get(int(c), 0) * d for c, d in zip(model_top[:5], _DISCOUNT))
    return dcg / _IDEAL_DCG


def precision_at_5(model_top: list[int] | np.ndarray, reference_top: list[int]) -> float:
    return len(set(map(int, model_top[:5])) & set(reference_top)) / 5.0


def evaluate(weights: np.ndarray, X: np.ndarray, selections: list[list[str]],
             references: list[list[int]]) -> dict[str, float]:
    tops = top_k_batch(X, selections, np.asarray(weights, dtype=float))
    return {
        "ndcg@5": float(np.mean([ndcg_at_5(t, r) for t, r in zip(tops, references)])),
        "precision@5": float(np.mean([precision_at_5(t, r) for t, r in zip(tops, references)])),
    }


def make_fitness(X: np.ndarray, selections: list[list[str]], references: list[list[int]]) -> Callable[[np.ndarray], float]:
    return lambda w: evaluate(w, X, selections, references)["ndcg@5"]


def tournament(rng: np.random.Generator, fitness: np.ndarray, k: int) -> int:
    contenders = rng.choice(len(fitness), size=k, replace=False)
    return int(contenders[np.argmax(fitness[contenders])])


def blend_crossover(rng: np.random.Generator, a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    span = hi - lo
    c1 = rng.uniform(lo - alpha * span, hi + alpha * span)
    c2 = rng.uniform(lo - alpha * span, hi + alpha * span)
    return normalize(c1), normalize(c2)


def gaussian_mutation(rng: np.random.Generator, w: np.ndarray, sigma: float) -> np.ndarray:
    return normalize(w + rng.normal(0.0, sigma, w.shape))


def run_ga(fitness_fn: Callable[[np.ndarray], float], cfg: GAConfig | None = None,
           on_generation: Callable[[int, float, float], None] | None = None) -> GAResult:
    cfg = cfg or GAConfig()
    rng = np.random.default_rng(cfg.seed)
    pop = normalize(rng.dirichlet(np.ones(N_GENES), size=cfg.population))
    if cfg.seed_equal_weights:
        pop[0] = np.full(N_GENES, 1.0 / N_GENES)
    fit = np.array([fitness_fn(w) for w in pop])
    init_best = int(np.argmax(fit))
    result = GAResult(best_weights=[], best_fitness=0.0,
                      initial_best_weights=pop[init_best].tolist(),
                      initial_best_fitness=float(fit[init_best]), config=asdict(cfg))
    result.history.append({"generation": 0, "best": float(fit.max()), "average": float(fit.mean())})

    for gen in range(1, cfg.generations + 1):
        elite_idx = np.argsort(-fit)[:cfg.elitism]
        children = [pop[i].copy() for i in elite_idx]
        while len(children) < cfg.population:
            p1 = pop[tournament(rng, fit, cfg.tournament_size)]
            p2 = pop[tournament(rng, fit, cfg.tournament_size)]
            if rng.random() < cfg.crossover_prob:
                c1, c2 = blend_crossover(rng, p1, p2, cfg.blend_alpha)
            else:
                c1, c2 = p1.copy(), p2.copy()
            for c in (c1, c2):
                if rng.random() < cfg.mutation_prob:
                    c = gaussian_mutation(rng, c, cfg.mutation_sigma)
                if len(children) < cfg.population:
                    children.append(c)
        pop = np.array(children)
        # Elites keep their known fitness; only the new children are evaluated.
        fit = np.array([fit[elite_idx[i]] if i < cfg.elitism else fitness_fn(w) for i, w in enumerate(pop)])
        result.history.append({"generation": gen, "best": float(fit.max()), "average": float(fit.mean())})
        if on_generation:
            on_generation(gen, float(fit.max()), float(fit.mean()))

    best = int(np.argmax(fit))
    result.best_weights = pop[best].tolist()
    result.best_fitness = float(fit[best])
    return result
