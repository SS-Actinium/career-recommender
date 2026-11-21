"""Tests for the dataset, reference generator, Mamdani FIS, GA and recommender."""
import numpy as np
import pytest

from engine import fuzzy
from engine.data import SKILLS, careers, interests, target_matrix
from engine.ga import (GAConfig, blend_crossover, gaussian_mutation, ndcg_at_5, normalize,
                       precision_at_5, run_ga, tournament)
from engine.interest import interest_match
from engine.recommender import final_scores, rank, recommend
from engine.reference import expert_scores, generate_profiles

EQUAL = np.full(6, 1 / 6)


# ---------- dataset ----------
def test_dataset_has_18_valid_careers():
    cs = careers()
    assert len(cs) == 18
    assert len({c["id"] for c in cs}) == 18
    tags = set(interests())
    for c in cs:
        assert set(c["targets"]) == set(SKILLS)
        assert all(0 <= v <= 100 for v in c["targets"].values())
        assert set(c["interests"]) <= tags and c["interests"]
        assert len(c["rule_skills"]["primary"]) == 2
        assert c["rule_skills"]["secondary"] in SKILLS
        assert set(c["related"]) <= {x["id"] for x in cs}


# ---------- reference profiles ----------
def test_reference_profiles_are_reproducible_and_valid():
    a, b = generate_profiles(), generate_profiles()
    assert a == b
    assert len(a) == 60
    assert sum(p["split"] == "train" for p in a) == 40
    for p in a:
        assert p["synthetic"] is True
        assert all(0 <= v <= 100 for v in p["skills"].values())
        assert len(p["reference_top5"]) == 5 and p["interests"]


def test_expert_function_is_bounded():
    e = expert_scores(np.full(6, 100.0), list(interests()))
    assert np.all(e <= 1.0 + 1e-9) and np.all(e >= 0)


# ---------- fuzzy ----------
def test_memberships_in_unit_interval():
    mu = fuzzy.fuzzify(np.linspace(0, 100, 1001)[:, None].repeat(6, axis=1))
    assert mu.min() >= 0 and mu.max() <= 1


@pytest.mark.parametrize("x,expected", [
    (0, (1, 0, 0)), (25, (0.5, 0, 0)), (50, (0, 1, 0)), (75, (0, 0, 0.5)), (100, (0, 0, 1)),
    (37.5, (0.25, 0.5, 0)), (62.5, (0, 0.5, 0.25)),
])
def test_membership_boundaries(x, expected):
    mu = fuzzy.fuzzify(np.full((1, 6), float(x)))[0, 0]
    assert np.allclose(mu, expected)


def test_rule_base_size_and_text():
    rules = fuzzy.build_rules()
    assert len(rules) == 18 * 7
    assert rules[0].text("Software Developer").startswith("IF Programming is High AND Problem-solving is High")


def test_centroid_output_in_range_and_monotone():
    X = np.random.default_rng(0).uniform(0, 100, (200, 6))
    s = fuzzy.infer(X, EQUAL)["s"]
    assert s.min() >= 0 and s.max() <= 100
    low, high = fuzzy.infer(np.full((1, 6), 10.0), EQUAL)["s"], fuzzy.infer(np.full((1, 6), 95.0), EQUAL)["s"]
    assert np.all(high > low)


def test_equal_weights_do_not_change_memberships():
    mu = fuzzy.fuzzify(np.array([[10, 30, 50, 70, 90, 100]]))
    assert np.allclose(fuzzy.apply_weights(mu, EQUAL), mu)


def test_zero_weight_on_primary_gives_fallback():
    w = np.array([0, 0.2, 0.2, 0.2, 0.2, 0.2])  # programming weight 0
    out = fuzzy.infer(np.full((1, 6), 80.0), w)
    sd = [c["id"] for c in careers()].index("software_developer")
    # P1 = programming has weight 0, but R7 (S Low) can still fire. Here S = 80, so Low is 0 → no rule fires.
    assert out["no_rule_fired"][0, sd] and out["s"][0, sd] == 0


# ---------- GA ----------
def test_normalize_produces_valid_chromosomes():
    w = normalize(np.array([[-1, 2, 3, 0, 0, 1], [0, 0, 0, 0, 0, 0]]))
    assert np.allclose(w.sum(axis=1), 1) and w.min() >= 0
    assert np.allclose(w[1], 1 / 6)


def test_operators_keep_chromosomes_valid():
    rng = np.random.default_rng(1)
    for _ in range(500):
        a, b = rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6))
        for c in (*blend_crossover(rng, a, b, 0.5), gaussian_mutation(rng, a, 0.05)):
            assert c.shape == (6,) and c.min() >= 0 and np.isclose(c.sum(), 1)


def test_tournament_favours_fitter():
    rng = np.random.default_rng(2)
    fit = np.arange(30, dtype=float)
    picks = [tournament(rng, fit, 3) for _ in range(2000)]
    assert np.mean(picks) > 14.5 + 5  # a uniform pick would average 14.5


def test_metrics():
    assert ndcg_at_5([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1)
    assert ndcg_at_5([9, 10, 11, 12, 13], [1, 2, 3, 4, 5]) == 0
    assert precision_at_5([1, 2, 9, 10, 11], [1, 2, 3, 4, 5]) == pytest.approx(0.4)


def test_ga_seed_reproducible_and_records_history():
    fit = lambda w: -float(np.sum((w - np.array([0.3, 0.1, 0.2, 0.1, 0.1, 0.2])) ** 2))
    cfg = GAConfig(population=12, generations=8)
    r1, r2 = run_ga(fit, cfg), run_ga(fit, cfg)
    assert r1.best_weights == r2.best_weights
    assert len(r1.history) == 9
    best = [h["best"] for h in r1.history]
    assert all(b2 >= b1 for b1, b2 in zip(best, best[1:]))  # elitism → never decreases
    assert np.isclose(sum(r1.best_weights), 1) and min(r1.best_weights) >= 0


# ---------- interest + recommender ----------
def test_interest_match():
    assert np.all(interest_match([]) == 0)
    m = interest_match(["design", "software"])  # rank weights 2 and 1
    cs = [c["id"] for c in careers()]
    assert m[cs.index("uiux_designer")] == pytest.approx(2 / 3)
    assert m[cs.index("web_developer")] == pytest.approx(1.0)


def test_final_score_formula():
    assert final_scores(np.array([80.0]), np.array([0.5]))[0] == pytest.approx(71.0)


def test_rank_tie_break_is_deterministic():
    scores = np.zeros(18)
    order = rank(scores, np.zeros(18))
    assert [careers()[i]["id"] for i in order] == sorted(c["id"] for c in careers())


def _student(skills, interests_=()):
    return {"skills": dict(zip(SKILLS, skills)), "interests": list(interests_)}


def test_recommend_shape_and_gaps():
    out = recommend(_student([90, 60, 90, 40, 50, 90], ["software"]), EQUAL)
    recs = out["recommendations"]
    assert len(recs) == 5 and [r["rank"] for r in recs] == [1, 2, 3, 4, 5]
    assert all(0 <= r["score"] <= 100 for r in recs)
    assert all(g["gap"] > 0 for r in recs for g in r["gaps"])
    assert "guidance" in recs[0]["explanation"]


def test_very_low_ratings_still_return_results():
    out = recommend(_student([5] * 6), EQUAL)
    assert len(out["recommendations"]) == 5
    assert all(r["score"] < 40 for r in out["recommendations"])
