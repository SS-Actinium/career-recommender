"""Tests for the dataset, reference generator, Mamdani FIS, GA and recommender."""
import numpy as np
import pytest

from engine import fuzzy
from engine.data import SKILLS, careers, interests, target_matrix
from engine.ga import (GAConfig, blend_crossover, gaussian_mutation, ndcg_at_5, normalize,
                       precision_at_5, run_ga, tournament)
from engine.interest import interest_match
from engine.preferences import preference_match
from engine.recommender import band, final_scores, rank, recommend
from engine.stats import bootstrap_ci, paired_bootstrap
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
    assert len(a) == 500
    assert [sum(p["split"] == k for p in a) for k in ("train", "val", "test")] == [350, 75, 75]
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
    (0, (1, 0, 0)), (25, (0.5, 0.5, 0)), (50, (0, 1, 0)), (75, (0, 0.5, 0.5)), (100, (0, 0, 1)),
    (37.5, (0.25, 0.75, 0)), (80, (0, 0.4, 0.6)),
])
def test_membership_boundaries(x, expected):
    mu = fuzzy.fuzzify(np.full((1, 6), float(x)))[0, 0]
    assert np.allclose(mu, expected)


def test_input_sets_form_a_partition():
    mu = fuzzy.fuzzify(np.linspace(0, 100, 401)[:, None])[:, 0, :]
    assert np.allclose(mu.sum(axis=1), 1.0)


def test_rule_base_size_and_text():
    rules = fuzzy.build_rules()
    assert len(rules) == 18 * fuzzy.RULES_PER_CAREER == 198
    assert rules[8].text("Software Developer") == (
        "IF Programming is High AND Problem-solving is High THEN Software Developer suitability is Very High")


def test_merged_clipping_equals_per_rule_mamdani():
    """Merging rules that share a consequent before clipping must equal the textbook per-rule max-min."""
    X = np.random.default_rng(3).uniform(0, 100, (25, 6))
    w = np.array([0.3, 0.1, 0.2, 0.1, 0.1, 0.2])
    out = fuzzy.infer(X, w, keep_curves=True)
    rules = fuzzy.build_rules()
    for n in range(len(X)):
        for ci in range(18):
            curves = [np.minimum(out["strengths"][n, k], fuzzy.OUTPUT_MF[r.consequent])
                      for k, r in enumerate(rules) if r.career_index == ci]
            assert np.allclose(np.max(curves, axis=0), out["aggregated"][n, ci])


def test_score_rises_with_a_primary_skill():
    sd = [c["id"] for c in careers()].index("software_developer")
    xs = np.array([[v, 60, 70, 50, 50, 70] for v in range(0, 101, 5)], dtype=float)
    s = fuzzy.infer(xs, EQUAL)["s"][:, sd]
    assert np.all(np.diff(s) >= -1e-9) and s[-1] - s[0] > 25


def test_no_saturation_above_75():
    """v1 bug: every rating above 75 was 'only High', so 80 and 95 gave the same s."""
    s80 = fuzzy.infer(np.full((1, 6), 80.0), EQUAL)["s"]
    s95 = fuzzy.infer(np.full((1, 6), 95.0), EQUAL)["s"]
    assert np.all(s95 > s80 + 5)


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
    # Every rule except R10 needs P1 = programming (weight 0). R10 is "S is Low" and S = 80 → no rule fires.
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
    # no preferences: s · (0.65 + 0.25·I) / 0.9
    assert final_scores(np.array([80.0]), np.array([0.5]))[0] == pytest.approx(80 * 0.775 / 0.9)
    assert final_scores(np.array([80.0]), np.array([0.5]), np.array([1.0]))[0] == pytest.approx(80 * 0.875)
    # interest can never lift the score above the skill-based suitability
    assert final_scores(np.array([40.0]), np.array([1.0]), np.array([1.0]))[0] == pytest.approx(40.0)


@pytest.mark.parametrize("score,key", [(90, "strong"), (70, "strong"), (69.9, "good"), (55, "good"), (45, "moderate"), (10, "weak")])
def test_bands(score, key):
    assert band(score)["key"] == key


def test_preference_match():
    W, detail = preference_match({})
    assert W is None
    cs = [c["id"] for c in careers()]
    W, detail = preference_match({"work_type": "creative", "advanced_math": "low"})
    assert W[cs.index("uiux_designer")] == pytest.approx(1.0)       # creative + low maths demand
    assert W[cs.index("data_scientist")] == pytest.approx(0.0)      # not creative, high maths demand
    assert W[cs.index("data_analyst")] == pytest.approx(0.25)       # not creative, medium demand → 0.5
    assert len(detail[0]) == 2


def test_roadmap_stage_and_priority():
    out = recommend(_student([20, 60, 80, 40, 50, 60], ["software"]), EQUAL)
    rec = next(r for r in out["recommendations"] if any(g["skill"] == "programming" for g in r["gaps"]))
    prog = next(r for r in rec["roadmap"] if r["skill"] == "programming")
    assert prog["stage"] == "foundation" and prog["priority"] == "high" and prog["steps"]


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
    assert recs[0]["fit"]["label"].lower() in recs[0]["explanation"]
    assert "mentor" in out["summary"]["text"] and out["summary"]["source"] == "template"
    we = out["analysis"]["worked_example"]
    assert len(we["rules"]) == 11 and we["centroid"] == pytest.approx(we["moment"] / we["area"], abs=1e-3)


def test_very_low_ratings_still_return_results():
    out = recommend(_student([5] * 6), EQUAL)
    assert len(out["recommendations"]) == 5
    assert all(r["score"] < 40 for r in out["recommendations"])


# ---------- statistics + GA model selection ----------
def test_bootstrap_ci_brackets_mean():
    v = np.random.default_rng(0).normal(0.8, 0.1, 200)
    ci = bootstrap_ci(v)
    assert ci["lo"] < ci["mean"] < ci["hi"] and ci["hi"] - ci["lo"] < 0.05
    d = paired_bootstrap(v, v + 0.05)
    assert d["lo"] > 0 and d["p_not_better"] == 0


def test_ga_early_stopping_keeps_best_validation():
    target = np.array([0.3, 0.1, 0.2, 0.1, 0.1, 0.2])
    fit = lambda w: -float(np.sum((w - target) ** 2))
    val = lambda w: -abs(float(w[0]) - 0.25)  # a different optimum, so validation peaks and stalls
    r = run_ga(fit, GAConfig(population=12, generations=60, patience=5), val_fn=val)
    vals = [h["val"] for h in r.history]
    assert r.selected_val_fitness == pytest.approx(max(vals))
    assert r.history[r.selected_generation]["val"] == pytest.approx(max(vals))
    assert r.stopped_early and len(r.history) < 61
