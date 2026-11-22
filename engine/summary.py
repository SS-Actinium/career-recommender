"""Optional LLM-written summary (Groq), with the deterministic template as fallback.

The LLM never scores anything. It only rewrites numbers the soft-computing
pipeline already produced. The student's display name is never sent.
Configure with GROQ_API_KEY (and optionally GROQ_MODEL) in .env or the environment.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger("career.summary")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was retired by Groq
TIMEOUT_S = 8.0
MAX_CHARS = 1400

SYSTEM_PROMPT = (
    "You write short, encouraging career-exploration summaries for university students. "
    "Use ONLY the facts in the JSON you are given; never invent careers, numbers, salaries or job guarantees. "
    "Write 90-140 words in plain English, second person, one paragraph, no headings, no lists, no markdown. "
    "Mention the top career and its fit band, one or two alternatives, the strongest skills, and the most useful "
    "skill to improve. End by reminding them this is guidance to discuss with a mentor, not a guarantee."
)


def configured() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def _facts(result: dict[str, Any]) -> dict[str, Any]:
    """Only computed facts. No display name or other personal data."""
    return {"top_careers": [{"rank": r["rank"], "career": r["name"], "score_out_of_100": r["score"], "fit": r["fit"]["label"],
                             "strongest_skills": r["strengths"], "matched_interests": r["matched_interests"],
                             "skill_gaps": [{"skill": g["label"], "points_below_target": round(g["gap"])} for g in r["gaps"][:3]]}
                            for r in result["recommendations"]],
            "method": "Mamdani fuzzy inference + GA-tuned weights + interest and work-preference match"}


def _call_groq(facts: dict[str, Any]) -> tuple[str, str]:
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    payload: dict[str, Any] = {"model": model, "temperature": 0.4, "max_tokens": 600,
                               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": json.dumps(facts)}]}
    if model.startswith("openai/gpt-oss"):  # reasoning model: keep thinking short so the answer fits
        payload["reasoning_effort"] = "low"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(GROQ_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}", "Content-Type": "application/json",
        "User-Agent": "career-recommender/2.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # noqa: S310 (fixed https URL)
        data = json.loads(resp.read().decode("utf-8"))
    text = data["choices"][0]["message"]["content"].strip()
    if not text:
        raise ValueError("empty completion")
    return text[:MAX_CHARS], model


def summarise(result: dict[str, Any]) -> dict[str, Any]:
    """Return {"text", "source": "groq"|"template", "model"?, "reason"?}."""
    template = result["summary"]
    if not configured():
        return {**template, "reason": "No GROQ_API_KEY configured, so the built-in template summary is shown."}
    try:
        text, model = _call_groq(_facts(result))
        return {"text": text, "source": "groq", "model": model}
    except urllib.error.HTTPError as exc:
        log.warning("Groq summary failed (HTTP %s), using template: %s", exc.code, exc.read().decode("utf-8", "replace")[:300])
        return {**template, "reason": "The LLM service returned an error, so the built-in template summary is shown."}
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError, OSError) as exc:
        log.warning("Groq summary failed, using template: %s", type(exc).__name__)
        return {**template, "reason": "The LLM service could not be reached, so the built-in template summary is shown."}
