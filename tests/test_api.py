"""Flask API tests: validation, pipeline output, career endpoints, analysis, errors."""
import pytest

from app import create_app
from engine.data import SKILLS

VALID = {"skills": {s: 70 for s in SKILLS}, "interests": ["software", "data"],
         "work_preferences": {"environment": "mixed"}, "display_name": "Test"}


@pytest.fixture()
def client():
    return create_app().test_client()


def test_valid_assessment_returns_top5(client):
    r = client.post("/api/assess", json=VALID)
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["recommendations"]) == 5
    assert len(body["all_scores"]) == 18
    assert "pre-optimised" in body["weights_source"]
    assert set(body["analysis"]["memberships"]) == set(SKILLS)


def test_missing_rating_rejected(client):
    bad = {**VALID, "skills": {s: 50 for s in SKILLS[:-1]}}
    r = client.post("/api/assess", json=bad)
    assert r.status_code == 400 and "skills.technical_interest" in r.get_json()["fields"]


@pytest.mark.parametrize("value", [-1, 101, "50", True, None])
def test_out_of_range_or_wrong_type_rejected(client, value):
    bad = {**VALID, "skills": {**VALID["skills"], "programming": value}}
    assert client.post("/api/assess", json=bad).status_code == 400


def test_unknown_interest_and_empty_interests_rejected(client):
    assert client.post("/api/assess", json={**VALID, "interests": ["astrology"]}).status_code == 400
    assert client.post("/api/assess", json={**VALID, "interests": []}).status_code == 400


def test_non_json_body_rejected_without_trace(client):
    r = client.post("/api/assess", data="not json", content_type="text/plain")
    assert r.status_code == 400
    text = r.get_data(as_text=True)
    assert "Traceback" not in text and "\\" not in text


def test_bad_work_preference_rejected(client):
    assert client.post("/api/assess", json={**VALID, "work_preferences": {"environment": "moon"}}).status_code == 400


def test_careers_endpoints(client):
    lst = client.get("/api/careers").get_json()
    assert len(lst["careers"]) == 18 and "not official" in lst["note"].lower()
    one = client.get("/api/careers/data_scientist").get_json()
    assert one["name"] == "Data Scientist" and one["related"][0]["name"]
    assert client.get("/api/careers/nope").status_code == 404


def test_analysis_endpoint(client):
    a = client.get("/api/analysis").get_json()
    assert len(a["fuzzy"]["rules"]) == 198 and len(a["fuzzy"]["template"]) == 11
    assert len(a["fuzzy"]["output_sets"]) == 5
    assert len(a["ga"]["history"]) <= a["ga"]["config"]["generations"] + 1
    assert a["ga"]["data"]["profiles"] == 500
    t = a["ga"]["evaluation"]["test"]["optimized"]["ndcg@5"]
    assert t["lo"] <= t["mean"] <= t["hi"]
    assert abs(sum(a["ga"]["optimized_weights"]) - 1) < 1e-9
    assert set(a["ga"]["evaluation"]) == {"train", "val", "test"}
    assert len(a["ga"]["robustness"]["seeds"]) == 5


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and b"CareerCompass" in r.data


@pytest.mark.parametrize("skills,interests,expected_any", [
    ([90, 60, 88, 40, 50, 92], ["software"], {"software_developer", "qa_engineer", "devops_engineer", "web_developer", "mobile_developer"}),
    ([55, 92, 85, 40, 60, 70], ["data"], {"data_analyst", "data_scientist"}),
    ([30, 30, 70, 95, 75, 50], ["design"], {"uiux_designer", "product_designer"}),
    ([25, 60, 70, 55, 92, 45], ["business"], {"business_analyst", "project_manager"}),
])
def test_expected_tendencies(client, skills, interests, expected_any):
    body = client.post("/api/assess", json={"skills": dict(zip(SKILLS, skills)), "interests": interests}).get_json()
    top2 = {r["id"] for r in body["recommendations"][:2]}
    assert top2 & expected_any


def test_fit_band_and_breakdown_present(client):
    body = client.post("/api/assess", json=VALID).get_json()
    r = body["recommendations"][0]
    assert r["fit"]["key"] in {"strong", "good", "moderate", "weak"}
    b = r["breakdown"]
    assert b["fis_points"] + b["interest_points"] + b["preference_points"] == pytest.approx(r["score"], abs=0.1)


def test_scores_are_not_inflated(client):
    """A uniformly average student must not be told they are a strong fit."""
    body = client.post("/api/assess", json={"skills": {s: 50 for s in SKILLS}, "interests": ["software"]}).get_json()
    top = body["all_scores"][0]
    assert top["score"] <= 50.0 + 1e-6 and top["fit"]["key"] == "moderate"
    weak = client.post("/api/assess", json={"skills": {s: 30 for s in SKILLS}, "interests": ["software"]}).get_json()
    assert weak["all_scores"][0]["fit"]["key"] == "weak"


def test_summary_falls_back_to_template(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    r = client.post("/api/summary", json=VALID).get_json()
    assert r["source"] == "template" and r["text"] and "reason" in r


def test_summary_survives_llm_failure(client, monkeypatch):
    from engine import summary
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    def boom(_facts):
        raise TimeoutError
    monkeypatch.setattr(summary, "_call_groq", boom)
    r = client.post("/api/summary", json=VALID).get_json()
    assert r["source"] == "template"


def test_summary_uses_llm_text_without_sending_name(client, monkeypatch):
    from engine import summary
    seen = {}
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    def fake(facts):
        seen["facts"] = facts
        return "LLM text.", "fake-model"
    monkeypatch.setattr(summary, "_call_groq", fake)
    r = client.post("/api/summary", json={**VALID, "display_name": "Secret Name"}).get_json()
    assert r == {"text": "LLM text.", "source": "groq", "model": "fake-model"}
    assert "Secret Name" not in str(seen["facts"])


def test_playground(client):
    r = client.post("/api/playground", json={"skills": VALID["skills"], "career_id": "data_scientist", "weights": "equal"})
    body = r.get_json()
    assert r.status_code == 200 and len(body["rules"]) == 11 and len(body["clip_levels"]) == 5
    assert body["s"] == pytest.approx(body["moment"] / body["area"], abs=1e-3)
    assert client.post("/api/playground", json={"skills": VALID["skills"], "career_id": "nope"}).status_code == 400


def test_demo_profiles(client):
    ps = client.get("/api/demo-profiles").get_json()["profiles"]
    assert len(ps) == 5
    for p in ps:
        assert client.post("/api/assess", json=p).status_code == 200
