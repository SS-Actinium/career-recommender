# Design Note — AI-Based Career Recommendation System (v2)

This note records the design decisions behind the prototype, every change from the original build prompt, and what changed in v2 and why.

## Pipeline

```
student input ──► validation ──► fuzzification (6 inputs × Low/Med/High, Ruspini partition)
                                   │
                    GA weights ──► weighted memberships μ'_i = min(1, 6·w_i·μ_i)
                                   │
                                   ▼
                 18 Mamdani FIS (one per career, 11 rules each): min-implication
                 → max-aggregation over 5 output sets → centroid  ⇒  s_c ∈ [0, 100]
                                   │
   interests ──► interest match I_c ∈ [0, 1]
   work prefs ─► preference match W_c ∈ [0, 1] (optional)
                                   ▼
               Score_c = s_c · (0.65 + 0.25·I_c + 0.10·W_c)  → rank → top 5 + fit band
                                   │
                                   ▼
               optional: LLM (Groq) rewrites the computed result as a paragraph
```

## v2 changes and the reasons for them

### 1. Scores were unrealistically high (v1 → v2 scoring)

Two separate causes, two fixes.

**a) FIS saturation.** v1 used input sets Low (0,0,50), Medium (25,50,75), High (50,100,100). Above 75 a rating was *only* High, so 80 and 95 fired the same rules, and with three output sets the centroid of a single clipped High set sat near 78 whatever its firing strength. Many careers shared s ≈ 78.

- Input sets are now a **Ruspini partition**: Low (0,0,50), Medium (0,50,100), High (50,100,100). μ_Low + μ_Medium + μ_High = 1 for every x, so a rating of 80 is Medium 0.4 / High 0.6 and still differs from 95.
- Output has **five sets**: Very Low (0,0,25), Low (0,25,50), Medium (25,50,75), High (50,75,100), Very High (75,100,100).
- Each career has **11 rules** (198 in total), built from its primary pair (P1, P2) and secondary skill S:

  | # | IF | THEN |
  |---|---|---|
  | R1 | P1 Low AND P2 Low | Very Low |
  | R2 | P1 Low AND P2 Medium | Low |
  | R3 | P1 Medium AND P2 Low | Low |
  | R4 | P1 Low AND P2 High | Low |
  | R5 | P1 High AND P2 Low | Low |
  | R6 | P1 Medium AND P2 Medium | Medium |
  | R7 | P1 Medium AND P2 High | High |
  | R8 | P1 High AND P2 Medium | High |
  | R9 | P1 High AND P2 High | Very High |
  | R10 | S Low | Low (penalty for a missing secondary skill) |
  | R11 | S Medium AND P1 High AND P2 High | High (tempers Very High) |

  R1–R9 are the complete 3 × 3 grid, so at least one rule fires whenever the primary weights are > 0. A weak primary caps the consequent at Low.

  Result with equal weights and every skill equal to x: x = 30 → s ≈ 37, 50 → 50, 70 → 63, 80 → 67.5, 90 → 74, 100 → 92. The FIS is monotone in each primary skill (tested).

**b) Additive interest bonus.** v1 used `Score = 0.7·s + 30·I`. A student who rated every skill 50 but ticked one interest got 65.0, and 30 of those 65 points came from the interest bonus alone. v2 makes interest and preferences **scale** the skill-based suitability instead of adding to it:

```
Score = s · m,   m = 0.65 + 0.25·I + 0.10·W ∈ [0.65, 1]
no preferences given: m = (0.65 + 0.25·I) / 0.90
```

The score can never exceed s, so interest can discount a career the student is not drawn to but can never inflate weak skills. Every skill at 30 → 36 (Weak), 50 → 50 (Moderate), 70 → 65 (Good), 90 → 73 (Strong).

**Fit bands** on the final score: Strong ≥ 70, Good 55–70, Moderate 40–55, Weak < 40. The bands were set from these calibration points, so that "Strong" needs strong primary skills *and* matching interests.

**Breakdown.** The API returns `s × m` split into points that add up to the score: base = s·0.65, interest = s·0.25·I, preferences = s·0.10·W (rescaled by 1/0.9 without preferences). The results chart stacks these three parts.

### 2. Work preferences are now scored (10%)

Each career has a `work_profile` in `data/careers.json` (work types, environment, activities, maths demand), which is a model assumption. Each answered preference scores 1 (match), 0.5 (partial) or 0 (mismatch):

| Preference | Match | Partial |
|---|---|---|
| work_type | in the career's work types | student answered "mixed" |
| environment | same as the career | either side is "mixed" |
| activity | in the career's activities | — |
| advanced_math | willingness ≥ career demand | one level below |

W is the mean over the answered preferences. If none are answered, W is undefined and its 0.10 share is redistributed pro rata. The GA training data has no preferences, so the GA optimises the same no-preference formula.

### 3. Larger dataset, proper split, confidence intervals

- **500 synthetic profiles** (seed 7): 350 train / 75 validation / 75 test. They are generated as before (Gaussian noise σ = 12 around a career's targets, 20% blends of two careers).
- **Model selection on validation.** After every generation, the best training chromosome is scored on validation. The chromosome with the best validation NDCG@5 is the one the app uses. **Early stopping**: the run stops after 15 generations without validation improvement (max 60).
- **95% bootstrap CIs** (2,000 percentile resamples of profiles) for every split and metric, plus a **paired bootstrap** of the GA − baseline difference.
- **Robustness**: the GA is repeated with 5 seeds, and the test NDCG@5 is reported as mean ± SD.
- The test split is touched only once, for the final report.

The honest result: the GA improves training fitness, but on the test split it is **not distinguishable from equal weights** (the CI of the difference contains 0). See README → GA results. With 60 profiles in v1, the reported +0.015 "improvement" was within noise.

### 4. Speed

The rule evaluation is vectorised (antecedent index arrays; shorter rules repeat their first antecedent, which leaves min unchanged). Rules that share a consequent are merged before clipping, using max_r min(α_r, μ_k(u)) = min(max_r α_r, μ_k(u)). A test checks this equals the textbook per-rule max–min. Ranking uses one batched `lexsort`. One fitness evaluation on 350 profiles fell from about 140 ms to about 40 ms, and the main GA run from about 130 s to about 30 s.

### 5. Optional LLM summary

`engine/summary.py` can ask Groq (`llama-3.3-70b-versatile` by default, set by `GROQ_MODEL`) to rewrite the computed top 5 as a 90–140 word paragraph.

- The LLM **never scores**. The `/api/summary` endpoint re-validates the input and recomputes the result on the server, so client-sent scores are never trusted, and only computed facts are sent.
- The **display name is never sent**.
- **Fallback:** no key, a timeout (8 s), or any error falls back to the deterministic template summary. The UI labels which one is shown. Only successful LLM answers are cached (in memory, keyed by a hash of the inputs).
- No new dependency: it uses `urllib` and a minimal `.env` reader.

### 6. History, compare, report, roadmap

- **History** is stored in the browser's `localStorage` only (last 30 results, export to JSON, delete or clear). Nothing is stored on the server. "Load demo history" runs the five demonstration profiles through the real engine, so the page is not empty on a fresh machine.
- **Compare** shows two careers side by side: radar chart of you vs both targets, score components, and the gap per skill.
- **Report**: a print-optimised page. Use the browser's "Save as PDF", so no PDF library is needed.
- **Learning roadmap**: for each gap, the stage comes from the current rating (foundation < 40, intermediate < 70, advanced ≥ 70), and the priority from the gap size (high ≥ 25, medium ≥ 10). Steps come from `skill_roadmap` in `data/careers.json`.

### 7. Soft Computing page and playground

The page is a 15-section viva walkthrough: fuzzy logic basics, membership formulas with a worked example on the student's own number, GA weighting, rule base, a rule-firing table for the top career, implication and aggregation with clip levels, the centroid as moment / area, the hybrid score, GA operators with diagrams, NDCG@5 with a worked example, training curves, results with CIs, data, and limitations. The **playground** calls `POST /api/playground`, which runs the real engine for one career and returns memberships, rule strengths, clip levels, the aggregated curve and the centroid. Requests are debounced, and out-of-order responses are dropped.

## Decisions kept from v1

| Item | Decision |
|---|---|
| Inputs | Six self-reported ratings, 0–100 |
| Inference | Mamdani: AND = min, implication = min, aggregation = max, centroid on 201 points |
| GA weights | Six non-negative weights summing to 1, μ'_i = min(1, 6·w_i·μ_i). Equal weights give μ' = μ (the baseline) |
| Interest match | Rank-weighted share: the interest ranked r of k has weight k − r + 1 |
| Ties | Sort by score, then s, then career id |
| No-firing fallback | s = 0 and `no_rule_fired` if every rule has strength 0 (only possible when a primary weight is 0) |
| GA operators | Population 30, tournament k = 3, BLX-α (α = 0.5, p = 0.8), Gaussian mutation (σ = 0.05, p = 0.2), elitism 1, repair (clip, normalise), equal-weight chromosome seeded |
| Fitness | Mean NDCG@5 vs the reference top 5, grades 5..1 |
| Expert labels | fit_c = 1 − Σ t·max(0, t − x) / Σ t², expert_c = 0.65·fit_c + 0.35·I_c. Separate from the FIS, but our own assumptions |

## Explanations

- **Top 3 contributing inputs:** (t_i/100)·(min(x_i, t_i)/100)·6w_i. This is an explanation heuristic and not part of the score.
- **Skill gaps:** gap = t_i − x_i. Only positive gaps are shown.
- **Fired rules:** every rule with non-zero strength, shown with its strength.
- All career targets and work profiles are **model assumptions, not official industry standards**.
