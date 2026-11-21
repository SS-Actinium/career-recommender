"""Offline GA experiment: regenerate the synthetic profiles, optimise the weights on the
training split, evaluate baseline vs optimised weights on the held-out split, and save
data/ga_results.json (the web app loads that file at startup).

Usage:  python run_ga.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np

from engine.data import DATA_DIR, SKILLS
from engine.ga import GAConfig, evaluate, make_fitness, run_ga
from engine.reference import N_TRAIN, SEED, as_arrays, save_profiles


def main() -> None:
    profiles = save_profiles()
    train = [p for p in profiles if p["split"] == "train"]
    hold = [p for p in profiles if p["split"] == "holdout"]
    Xtr, Str, Rtr = as_arrays(train)
    Xho, Sho, Rho = as_arrays(hold)

    cfg = GAConfig()
    t0 = time.perf_counter()
    res = run_ga(make_fitness(Xtr, Str, Rtr), cfg,
                 on_generation=lambda g, b, a: print(f"gen {g:2d}  best={b:.4f}  avg={a:.4f}") if g % 5 == 0 else None)
    runtime = time.perf_counter() - t0

    baseline = np.full(len(SKILLS), 1.0 / len(SKILLS))
    optimized = np.array(res.best_weights)
    evaluation = {
        "train": {"baseline": evaluate(baseline, Xtr, Str, Rtr), "optimized": evaluate(optimized, Xtr, Str, Rtr)},
        "holdout": {"baseline": evaluate(baseline, Xho, Sho, Rho), "optimized": evaluate(optimized, Xho, Sho, Rho)},
    }
    out = {
        "_note": "Produced by run_ga.py on SYNTHETIC reference profiles. The labels come from an expert function "
                 "(our own assumptions), so these numbers do not measure real-world career success.",
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "skills": SKILLS,
        "config": res.config,
        "fitness_metric": "mean NDCG@5 vs reference top-5 (grades 5..1), training split",
        "data": {"profiles": len(profiles), "train": N_TRAIN, "holdout": len(hold), "profile_seed": SEED},
        "baseline_weights": baseline.tolist(),
        "initial_best_weights": res.initial_best_weights,
        "initial_best_fitness": res.initial_best_fitness,
        "optimized_weights": res.best_weights,
        "optimized_fitness": res.best_fitness,
        "history": res.history,
        "evaluation": evaluation,
        "runtime_seconds": round(runtime, 2),
    }
    with open(DATA_DIR / "ga_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print(f"\nruntime: {runtime:.1f}s")
    print("optimized weights:", ", ".join(f"{s}={w:.3f}" for s, w in zip(SKILLS, res.best_weights)))
    for split in ("train", "holdout"):
        b, o = evaluation[split]["baseline"], evaluation[split]["optimized"]
        print(f"{split:8s} baseline NDCG@5={b['ndcg@5']:.4f} P@5={b['precision@5']:.3f} | "
              f"optimized NDCG@5={o['ndcg@5']:.4f} P@5={o['precision@5']:.3f}")


if __name__ == "__main__":
    main()
