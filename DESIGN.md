# Design Note — AI-Based Career Recommendation System

This note records the design decisions behind the prototype and every change from the build prompt.

## Pipeline

```
student input ──► validation ──► fuzzification (6 inputs × Low/Med/High)
                                   │
                    GA weights ──► weighted memberships μ'_i = min(1, 6·w_i·μ_i)
                                   │
                                   ▼
                 18 Mamdani FIS (one per career): rules → min-implication
                 → max-aggregation → centroid  ⇒  s_c ∈ [0, 100]
                                   │
   interests ──► interest match I_c ∈ [0, 1]
                                   ▼
               Score_c = 100 · (0.7 · s_c/100 + 0.3 · I_c)  → rank → top 5
```

## Fixed decisions (from the build prompt, kept as specified)

| Item | Decision |
|---|---|
| Inputs | Programming, Mathematics, Problem-solving, Creativity, Communication, Technical Interest, each self-reported on 0–100 |
| Input fuzzy sets | Triangular: Low (0,0,50), Medium (25,50,75), High (50,100,100) |
| Output fuzzy sets | The same three triangles on the suitability universe 0–100, sampled at 201 points (step 0.5) |
| Inference | Mamdani: AND = min, OR = max, implication = min, aggregation = max, centroid defuzzification |
| GA weights | Six non-negative weights summing to 1, applied as μ'_i = min(1, 6·w_i·μ_i). Equal weights (1/6) give μ' = μ, so the baseline is the plain FIS |
| Interest match | I_c = (rank weight of the student's selected interests that are linked to career c) ÷ (total rank weight of all selected interests). The interest ranked r out of k has weight k − r + 1. No interests selected gives I_c = 0 for every career |
| Final score | Score_c = 100 · (0.7·s_c/100 + 0.3·I_c), labelled "suitability match". It is already on 0–100 and is not normalised again |
| Ties | Sort by Score, then by s_c, then by career id (deterministic) |

## Changes from the build prompt and the reasons for them

1. **Rule count: 126 rules (7 per career) instead of 15–25.** Each career has its own FIS. With 18 careers, 15–25 rules would leave most careers with one rule, so many inputs would fire no rule and the centroid would be undefined. Each career has two **primary** skills (P1, P2) and one **secondary** skill (S), chosen per career in `data/careers.json`:
   - R1 P1 High AND P2 High → High
   - R2 P1 High AND P2 Medium → Medium
   - R3 P1 Medium AND P2 High → Medium
   - R4 P1 Medium AND P2 Medium → Medium
   - R5 P1 Low OR P2 Low → Low
   - R6 S High AND P1 High → High (reinforcement from the secondary skill)
   - R7 S Low → Low (penalty for a missing secondary skill)

   R1–R5 cover every Low/Medium/High combination of P1 and P2, so at least one rule fires whenever w > 0. The rules have the same structure for every career but use different antecedent skills. Careers that share a primary pair (for example Cybersecurity and Network Engineer) are told apart by S, by the interest match, and by their targets in the explanations.
2. **No-firing fallback.** If every rule has firing strength 0 (only possible when a primary skill's weight is 0), the FIS returns s_c = 0 and flags `no_rule_fired`.
3. **Work preferences are not scored.** They are collected and shown as explanation context only. The specification forbids adding them to the algorithm without a documented method, and adding a third term would make the 0.7/0.3 split harder to defend.
4. **The GA seeds one equal-weight chromosome** into the initial population. Because of elitism, training fitness can never fall below the baseline. Held-out results are still reported separately.
5. **`/api/analysis` returns the static model data** (membership curves, the rule base, GA history, weights, and the evaluation). The per-student data (memberships, fired rules, aggregated output curves) is returned in the `analysis` field of `POST /api/assess`, because it depends on the submitted input.
6. **SQLite was not added.** It is optional. The last result is kept in the browser's `sessionStorage` only.

## Genetic Algorithm

Custom NumPy implementation (`engine/ga.py`). DEAP is not used.

- Population 30, generations 40, seed 42.
- Tournament selection with size 3. Elitism keeps the best 1.
- Blend crossover (BLX-α, α = 0.5) with probability 0.8.
- Gaussian mutation (σ = 0.05 on each gene) with probability 0.2 per child.
- After crossover and mutation, each chromosome is repaired: clip to ≥ 0, divide by the sum, and use uniform weights if the sum is 0.
- **Fitness:** mean NDCG@5 between the model's top 5 and the reference top 5 over the 40 training profiles. Reference grades are 5, 4, 3, 2, 1 for the reference ranks 1–5.

## Reference data (synthetic — not real students)

`engine/reference.py` generates 60 profiles with seed 7 and splits them 40 for training and 20 held out. Each profile is sampled around one archetype career's targets (Gaussian noise, σ = 12). 20% of the profiles blend two archetypes. Interests include the archetype's tags with p = 0.85 and each other tag with p = 0.15.

The labels come from a **separate expert function** that does not use the FIS rules:

- fit_c = 1 − Σ_i t_i·max(0, t_i − x_i) / Σ_i t_i², the target-weighted share of unmet requirement
- expert_c = 0.65·fit_c + 0.35·I_c

The expert function still encodes our own assumptions. It is a consistency check, not ground truth.

## Explanations

- **Top 3 contributing inputs:** contribution_i = (t_i/100) · (min(x_i, t_i)/100) · 6w_i. This is a transparent explanation heuristic and is not part of the score.
- **Skill gaps:** gap = t_i − x_i. Only positive gaps are shown as improvement areas.
- **Fired rules:** every card lists the career's rules with non-zero firing strength.
- All career target values are **model assumptions, not official industry standards**.
