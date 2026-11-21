"""Flask backend. It serves the static frontend and the JSON API.

Pre-optimised GA weights are loaded from data/ga_results.json at startup.
The GA never runs during a student request. Run `python run_ga.py` to re-optimise.

Optional: put GROQ_API_KEY=... in a .env file next to this script to enable
LLM-written summaries. Without it the template summary is used.

Usage:  python app.py   →  http://127.0.0.1:5000
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from demo_profiles import DEMO_PROFILES
from engine import fuzzy, summary
from engine.data import DATA_DIR, ROOT, SKILL_LABELS, SKILLS, career_by_id, careers, interests, load_dataset
from engine.recommender import BANDS, FIS_SHARE, INTEREST_SHARE, PREF_SHARE, recommend, score_formula
from engine.validation import WORK_PREF_OPTIONS, ValidationError, validate_assessment, validate_playground

log = logging.getLogger("career")


def load_env(path=ROOT / ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines). Existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_ga_results() -> dict[str, Any]:
    path = DATA_DIR / "ga_results.json"
    if not path.exists():
        log.warning("ga_results.json missing; falling back to equal weights. Run run_ga.py.")
        return {"optimized_weights": [1 / 6] * 6, "missing": True}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _r4(v: float) -> float:
    return round(float(v), 4)


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    load_env()
    ga = load_ga_results()
    weights = np.array(ga["optimized_weights"], dtype=float)
    equal = np.full(len(SKILLS), 1.0 / len(SKILLS))
    summary_cache: dict[str, dict[str, Any]] = {}
    career_ids = [c["id"] for c in careers()]

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.post("/api/assess")
    def assess():
        payload = request.get_json(silent=True)
        try:
            student = validate_assessment(payload)
        except ValidationError as exc:
            return jsonify({"error": "validation_failed", "fields": exc.errors}), 400
        result = recommend(student, weights)
        result["weights_source"] = ("equal weights (GA results file missing)" if ga.get("missing")
                                    else "pre-optimised GA weights loaded from file (GA is not run per request)")
        result["llm_configured"] = summary.configured()
        return jsonify(result)

    @app.post("/api/summary")
    def make_summary():
        """Recompute the result server-side (client scores are never trusted), then summarise it."""
        payload = request.get_json(silent=True)
        try:
            student = validate_assessment(payload)
        except ValidationError as exc:
            return jsonify({"error": "validation_failed", "fields": exc.errors}), 400
        key = hashlib.sha256(json.dumps({k: student[k] for k in ("skills", "interests", "work_preferences")},
                                        sort_keys=True).encode()).hexdigest()
        if key in summary_cache:
            return jsonify(summary_cache[key])
        out = summary.summarise(recommend({**student, "display_name": ""}, weights))
        if out["source"] == "groq":  # fallbacks are not cached, so a later retry can still reach the LLM
            if len(summary_cache) > 256:
                summary_cache.clear()
            summary_cache[key] = out
        return jsonify(out)

    @app.post("/api/playground")
    def playground():
        """One career FIS, step by step, for the live slider playground."""
        try:
            req = validate_playground(request.get_json(silent=True), career_ids)
        except ValidationError as exc:
            return jsonify({"error": "validation_failed", "fields": exc.errors}), 400
        w = weights if req["weights"] == "optimized" else equal
        x = np.array([req["skills"][s] for s in SKILLS], dtype=float)
        ci = career_ids.index(req["career_id"])
        res = fuzzy.infer(x[None], w, keep_curves=True)
        mu = fuzzy.fuzzify(x[None])[0]
        mu_w = fuzzy.apply_weights(mu[None], w)[0]
        name = careers()[ci]["name"]
        rules = [(k, r) for k, r in enumerate(fuzzy.build_rules()) if r.career_index == ci]
        return jsonify({
            "career": {"id": req["career_id"], "name": name},
            "weights": [_r4(v) for v in w],
            "memberships": [[_r4(v) for v in row] for row in mu],
            "weighted_memberships": [[_r4(v) for v in row] for row in mu_w],
            "rules": [{"id": r.rule_id, "text": r.text(name), "consequent": fuzzy.OUT_NAMES[r.consequent],
                       "strength": _r4(res["strengths"][0][k]), "role": r.role,
                       "antecedents": [{"skill": SKILL_LABELS[SKILLS[si]], "set": fuzzy.SET_NAMES[m], "mu_w": _r4(mu_w[si, m])}
                                       for si, m in r.antecedents]} for k, r in rules],
            "clip_levels": [_r4(v) for v in res["levels"][0][ci]],
            "aggregated": [_r4(v) for v in res["aggregated"][0][ci][::2]],
            "universe": [float(u) for u in fuzzy.UNIVERSE[::2]],
            "area": _r4(res["area"][0][ci]), "moment": _r4(res["moment"][0][ci]),
            "s": _r4(res["s"][0][ci]), "no_rule_fired": bool(res["no_rule_fired"][0][ci]),
        })

    @app.get("/api/demo-profiles")
    def demo_profiles():
        return jsonify({"profiles": [{"name": n, "skills": dict(zip(SKILLS, p["skills"])), "interests": p["interests"],
                                      "work_preferences": p.get("work_preferences", {})} for n, p in DEMO_PROFILES.items()]})

    @app.get("/api/meta")
    def meta():
        return jsonify({"llm_configured": summary.configured(),
                        "skills": [{"id": s, "label": SKILL_LABELS[s]} for s in SKILLS],
                        "interests": [{"id": k, "label": v} for k, v in interests().items()],
                        "work_preferences": {k: sorted(v) for k, v in WORK_PREF_OPTIONS.items()}})

    @app.get("/api/careers")
    def list_careers():
        return jsonify({"note": load_dataset()["_note"],
                        "careers": [{"id": c["id"], "name": c["name"], "description": c["description"]}
                                    for c in careers()]})

    @app.get("/api/careers/<career_id>")
    def get_career(career_id: str):
        c = career_by_id(career_id)
        if c is None:
            return jsonify({"error": "not_found", "message": "Unknown career."}), 404
        related = [{"id": r, "name": career_by_id(r)["name"]} for r in c["related"]]
        return jsonify({**c, "related": related, "note": load_dataset()["_note"]})

    @app.get("/api/analysis")
    def analysis():
        names = {i: c["name"] for i, c in enumerate(careers())}
        u = fuzzy.UNIVERSE[::4]
        return jsonify({
            "fuzzy": {
                "sets": fuzzy.SET_NAMES,
                "params": fuzzy.MF_PARAMS,
                "output_sets": fuzzy.OUT_NAMES,
                "output_params": fuzzy.OUT_PARAMS,
                "curves": {"x": u.tolist(),
                           "y": {n: fuzzy.trimf(u, *p).round(4).tolist() for n, p in zip(fuzzy.SET_NAMES, fuzzy.MF_PARAMS)},
                           "out": {n: fuzzy.trimf(u, *p).round(4).tolist() for n, p in zip(fuzzy.OUT_NAMES, fuzzy.OUT_PARAMS)}},
                "inference": "Mamdani: AND=min, implication=min, aggregation=max, centroid defuzzification",
                "weighting": "mu' = min(1, 6 * w_i * mu_i)",
                "template": [{"k": k, "antecedents": [f"{slot} is {fuzzy.SET_NAMES[m]}" for slot, m in ants],
                              "consequent": fuzzy.OUT_NAMES[cons], "role": role}
                             for k, (ants, _op, cons, role) in enumerate(fuzzy.RULE_TEMPLATE, start=1)],
                "rules_per_career": fuzzy.RULES_PER_CAREER,
                "rule_skills": [{"career": c["name"], "id": c["id"],
                                 "primary": [SKILL_LABELS[s] for s in c["rule_skills"]["primary"]],
                                 "secondary": SKILL_LABELS[c["rule_skills"]["secondary"]]} for c in careers()],
                "rules": [{"id": r.rule_id, "career": names[r.career_index], "text": r.text(names[r.career_index])}
                          for r in fuzzy.build_rules()],
            },
            "ga": {k: ga.get(k) for k in ("config", "fitness_metric", "selection", "data", "bootstrap", "baseline_weights",
                                          "initial_best_weights", "initial_best_fitness", "final_best_train_weights",
                                          "optimized_weights", "optimized_fitness", "selected_generation",
                                          "selected_val_fitness", "stopped_early", "history", "evaluation", "robustness",
                                          "runtime_seconds", "run_at", "_note", "missing")},
            "skills": [SKILL_LABELS[s] for s in SKILLS],
            "score_formula": score_formula(True),
            "score_formula_no_prefs": score_formula(False),
            "shares": {"fis": FIS_SHARE, "interest": INTEREST_SHARE, "preferences": PREF_SHARE},
            "bands": [{"min": lo, "label": name, "key": key} for lo, name, key in BANDS],
            "llm_configured": summary.configured(),
        })

    @app.errorhandler(404)
    def not_found(_e):
        return jsonify({"error": "not_found", "message": "Resource not found."}), 404

    @app.errorhandler(405)
    def bad_method(_e):
        return jsonify({"error": "method_not_allowed", "message": "Method not allowed."}), 405

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({"error": "too_large", "message": "Request body too large."}), 413

    @app.errorhandler(Exception)
    def server_error(exc):
        if isinstance(exc, HTTPException):
            return jsonify({"error": "http_error", "message": exc.name}), exc.code
        log.exception("unhandled error: %s", exc)
        return jsonify({"error": "server_error", "message": "Something went wrong. Please try again."}), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
