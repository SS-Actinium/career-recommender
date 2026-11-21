"""Flask backend. It serves the static frontend and the JSON API.

Pre-optimised GA weights are loaded from data/ga_results.json at startup.
The GA never runs during a student request. Run `python run_ga.py` to re-optimise.

Usage:  python app.py   →  http://127.0.0.1:5000
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from engine import fuzzy
from engine.data import DATA_DIR, ROOT, SKILL_LABELS, SKILLS, career_by_id, careers, interests, load_dataset
from engine.recommender import FIS_SHARE, INTEREST_SHARE, recommend
from engine.validation import WORK_PREF_OPTIONS, ValidationError, validate_assessment

log = logging.getLogger("career")


def load_ga_results() -> dict[str, Any]:
    path = DATA_DIR / "ga_results.json"
    if not path.exists():
        log.warning("ga_results.json missing; falling back to equal weights. Run run_ga.py.")
        return {"optimized_weights": [1 / 6] * 6, "missing": True}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    ga = load_ga_results()
    weights = np.array(ga["optimized_weights"], dtype=float)

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
        return jsonify(result)

    @app.get("/api/meta")
    def meta():
        return jsonify({"skills": [{"id": s, "label": SKILL_LABELS[s]} for s in SKILLS],
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
                "curves": {"x": u.tolist(),
                           "y": {n: fuzzy.trimf(u, *p).round(4).tolist() for n, p in zip(fuzzy.SET_NAMES, fuzzy.MF_PARAMS)}},
                "inference": "Mamdani: AND=min, OR=max, implication=min, aggregation=max, centroid defuzzification",
                "weighting": "mu' = min(1, 6 * w_i * mu_i)",
                "rules": [{"id": r.rule_id, "career": names[r.career_index], "text": r.text(names[r.career_index])}
                          for r in fuzzy.build_rules()],
            },
            "ga": {k: ga.get(k) for k in ("config", "fitness_metric", "data", "baseline_weights",
                                          "initial_best_weights", "initial_best_fitness", "optimized_weights",
                                          "optimized_fitness", "history", "evaluation", "runtime_seconds",
                                          "run_at", "_note", "missing")},
            "skills": [SKILL_LABELS[s] for s in SKILLS],
            "score_formula": f"Score = 100 * ({FIS_SHARE} * s/100 + {INTEREST_SHARE} * I)",
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
