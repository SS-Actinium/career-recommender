"""Offline GA experiment.

1. Regenerate the 500 synthetic profiles (350 train / 75 validation / 75 test).
2. Optimise the weights on train; keep the chromosome with the best validation
   NDCG@5 (early stopping, patience 15).
3. Evaluate baseline (equal weights) vs selected weights on every split with
   95% bootstrap confidence intervals, plus a paired bootstrap of the difference.
4. Repeat the GA with 5 seeds to show how stable the result is.
5. Save data/ga_results.json (the web app loads it at startup).

Usage:  python run_ga.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np

from engine.data import DATA_DIR, SKILLS
from engine.ga import GAConfig, make_fitness, per_profile, run_ga
from engine.reference import N_TRAIN, N_VAL, SEED, as_arrays, save_profiles, split
from engine.stats import N_BOOT, bootstrap_ci, paired_bootstrap

ROBUSTNESS_SEEDS = [42, 1, 2, 3, 4]


def main() -> None:
    profiles = save_profiles()
    data = {name: as_arrays(split(profiles, name)) for name in ("train", "val", "test")}
    fit_train = make_fitness(*data["train"])
    fit_val = make_fitness(*data["val"])
    baseline = np.full(len(SKILLS), 1.0 / len(SKILLS))

    cfg = GAConfig()
    t0 = time.perf_counter()
    res = run_ga(fit_train, cfg, val_fn=fit_val,
                 on_generation=lambda g, b, a: print(f"gen {g:2d}  best={b:.4f}  avg={a:.4f}") if g % 5 == 0 else None)
    runtime = time.perf_counter() - t0
    selected = np.array(res.selected_weights)

    evaluation, per_split = {}, {}
    for name, arrays in data.items():
        b, o = per_profile(baseline, *arrays), per_profile(selected, *arrays)
        evaluation[name] = {"baseline": {k: bootstrap_ci(v) for k, v in b.items()},
                            "optimized": {k: bootstrap_ci(v) for k, v in o.items()},
                            "difference": {k: paired_bootstrap(b[k], o[k]) for k in b}}
        per_split[name] = len(arrays[0])

    robustness = []
    for seed in ROBUSTNESS_SEEDS:
        r = res if seed == cfg.seed else run_ga(fit_train, GAConfig(seed=seed), val_fn=fit_val)
        w = np.array(r.selected_weights)
        test = per_profile(w, *data["test"])
        robustness.append({"seed": seed, "weights": r.selected_weights, "generation": r.selected_generation,
                           "val_ndcg@5": r.selected_val_fitness, "test_ndcg@5": float(test["ndcg@5"].mean()),
                           "test_precision@5": float(test["precision@5"].mean())})
        print(f"seed {seed:2d}: test NDCG@5={robustness[-1]['test_ndcg@5']:.4f}")
    t_nd = np.array([r["test_ndcg@5"] for r in robustness])

    out = {
        "_note": "Produced by run_ga.py on SYNTHETIC reference profiles. The labels come from an expert function "
                 "(our own assumptions), so these numbers do not measure real-world career success.",
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "skills": SKILLS,
        "config": res.config,
        "fitness_metric": "mean NDCG@5 vs reference top-5 (grades 5..1), training split",
        "selection": "chromosome with the best validation NDCG@5; early stopping after "
                     f"{cfg.patience} generations without validation improvement",
        "data": {"profiles": len(profiles), "train": N_TRAIN, "val": N_VAL, "test": per_split["test"],
                 "holdout": per_split["test"], "profile_seed": SEED},
        "bootstrap": {"resamples": N_BOOT, "level": 0.95, "method": "percentile, resampling profiles"},
        "baseline_weights": baseline.tolist(),
        "initial_best_weights": res.initial_best_weights,
        "initial_best_fitness": res.initial_best_fitness,
        "final_best_train_weights": res.best_weights,
        "optimized_weights": res.selected_weights,
        "optimized_fitness": float(fit_train(selected)),
        "selected_generation": res.selected_generation,
        "selected_val_fitness": res.selected_val_fitness,
        "stopped_early": res.stopped_early,
        "history": res.history,
        "evaluation": evaluation,
        "robustness": {"seeds": robustness, "test_ndcg@5_mean": float(t_nd.mean()), "test_ndcg@5_sd": float(t_nd.std(ddof=1))},
        "runtime_seconds": round(runtime, 2),
    }
    with open(DATA_DIR / "ga_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print(f"\nruntime (main run): {runtime:.1f}s, selected generation {res.selected_generation}, early stop={res.stopped_early}")
    print("selected weights:", ", ".join(f"{s}={w:.3f}" for s, w in zip(SKILLS, res.selected_weights)))
    for name in ("train", "val", "test"):
        e = evaluation[name]
        b, o, d = e["baseline"]["ndcg@5"], e["optimized"]["ndcg@5"], e["difference"]["ndcg@5"]
        print(f"{name:5s} NDCG@5 baseline {b['mean']:.4f} [{b['lo']:.4f}, {b['hi']:.4f}]  "
              f"optimised {o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}]  diff {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")
    print(f"robustness: test NDCG@5 {t_nd.mean():.4f} +/- {t_nd.std(ddof=1):.4f} over {len(t_nd)} seeds")


if __name__ == "__main__":
    main()
