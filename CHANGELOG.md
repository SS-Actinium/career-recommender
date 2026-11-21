# Changelog

## 2.0.0 — 2026-11-21

### Scoring realism
- Input fuzzy sets are now a Ruspini partition. Five output sets (Very Low … Very High) and an 11-rule base per career (198 rules) replace the 3-set, 7-rule design, which fixes the centroid getting stuck near 78.
- The final score is now multiplicative: `s × (0.65 + 0.25·I + 0.10·W)`. Interest and preferences can no longer lift weak skills, and the score never exceeds the fuzzy suitability s.
- Fit bands: Strong ≥ 70, Good 55–70, Moderate 40–55, Weak < 40.
- Work preferences are now scored (10% of the multiplier) against a per-career `work_profile`.

### Data and evaluation
- 500 synthetic profiles (350 train / 75 validation / 75 test) replace 60 (40 / 20).
- The GA keeps the chromosome with the best validation NDCG@5 and stops early (patience 15).
- 95% bootstrap confidence intervals, a paired bootstrap of the GA − baseline difference, and a 5-seed robustness check.
- Vectorised rule evaluation and ranking (about 3.4× faster).

### Features
- History page (browser `localStorage`): view, delete, clear, export JSON, and load demo history.
- Compare-two-careers page with a radar chart and per-skill gaps.
- Print-optimised PDF report page.
- Learning roadmap for each skill gap (stage and priority).
- Optional Groq LLM summary with a template fallback. The display name is never sent.
- Soft Computing page rebuilt as a 15-section viva walkthrough, plus a live playground (`POST /api/playground`).
- New endpoints: `/api/summary`, `/api/playground`, `/api/demo-profiles`.

### Tests
- 42 → 63 tests.

## 1.0.0 — 2026-11-21
- Base version: Mamdani FIS (18 careers, 126 rules), a custom NumPy GA, Flask API and a vanilla JS frontend.
