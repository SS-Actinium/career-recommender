/* Soft Computing page: a viva-ready walkthrough of the fuzzy system and the GA, using the
   current student's numbers, plus a live playground backed by POST /api/playground.
   Loaded before app.js; uses the helpers app.js publishes on window.CC at route time. */
(() => {
  "use strict";
  window.CCPages = window.CCPages || {};

  const f2 = v => (v == null ? "—" : Number(v).toFixed(2));
  const f3 = v => (v == null ? "—" : Number(v).toFixed(3));
  const f4 = v => (v == null ? "—" : Number(v).toFixed(4));

  function trimf(x, [a, b, c]) {
    if (x === b) return 1;
    const left = a === b ? 1 : (x - a) / (b - a);
    const right = b === c ? 1 : (c - x) / (c - b);
    return Math.max(0, Math.min(1, Math.min(left, right)));
  }
  /** Human-readable working for μ of one set, e.g. "(100 − 72) / 50 = 0.56". */
  function trimfWorking(x, [a, b, c], name) {
    const v = trimf(x, [a, b, c]);
    if (x === b) return `${name}: x = ${b} (peak) → 1`;
    if (x <= a && a !== b) return `${name}: x ≤ ${a} → 0`;
    if (x >= c && b !== c) return `${name}: x ≥ ${c} → 0`;
    if (x < b) return a === b ? `${name}: left shoulder → 1` : `${name}: (${x} − ${a}) / ${b - a} = ${f2(v)}`;
    return b === c ? `${name}: right shoulder → 1` : `${name}: (${c} − ${x}) / ${c - b} = ${f2(v)}`;
  }

  const TOC = [
    ["sc-overview", "Overview"], ["sc-fuzzy", "1 · Fuzzy logic"], ["sc-mf", "2 · Membership functions"],
    ["sc-weights", "3 · GA weighting"], ["sc-rules", "4 · Rule base"], ["sc-firing", "5 · Rule firing"],
    ["sc-agg", "6 · Implication & aggregation"], ["sc-centroid", "7 · Centroid"], ["sc-score", "8 · Hybrid score"],
    ["sc-play", "9 · Playground"], ["sc-ga", "10 · Genetic algorithm"], ["sc-fitness", "11 · Fitness (NDCG@5)"],
    ["sc-train", "12 · Training"], ["sc-results", "13 · Results & CIs"], ["sc-data", "14 · Data"], ["sc-limits", "15 · Limitations"],
  ];

  window.CCPages.analysis = async function analysisPage() {
    const CC = window.CC;
    const { esc } = CC;
    await CC.ensureAnalysis();
    const a = CC.state.analysis, r = CC.state.result, ga = a.ga, fz = a.fuzzy;
    const skills = CC.state.meta.skills;
    const top = r?.recommendations[0];
    const we = r?.analysis.worked_example;
    const w = ga.optimized_weights;

    // ----- 2: fuzzification worked example for the top career's first primary skill -----
    const primaryLabel = fz.rule_skills.find(x => x.id === top?.id)?.primary[0];
    const pSkill = skills.find(s => s.label === primaryLabel) || skills[0];
    const px = r ? r.analysis.inputs[pSkill.id] : 72;
    const pIdx = skills.indexOf(pSkill);

    const fuzzTable = r ? `<table><thead><tr><th>Input</th><th class="num">x</th>${CC.SETS.map(s => `<th class="num">μ ${s}</th>`).join("")}<th class="num">Sum</th></tr></thead><tbody>
      ${skills.map(s => { const m = r.analysis.memberships[s.id]; return `<tr><td>${esc(s.label)}</td><td class="num">${r.analysis.inputs[s.id]}</td>
        ${CC.SETS.map(n => `<td class="num">${f2(m[n])}</td>`).join("")}<td class="num">${f2(CC.SETS.reduce((t, n) => t + m[n], 0))}</td></tr>`; }).join("")}
      </tbody></table>` : "";

    const weightTable = r ? `<table><thead><tr><th>Input</th><th class="num">w (GA)</th><th class="num">6·w</th>${CC.SETS.map(s => `<th class="num">μ' ${s}</th>`).join("")}</tr></thead><tbody>
      ${skills.map((s, i) => { const m = r.analysis.weighted_memberships[s.id]; return `<tr><td>${esc(s.label)}</td><td class="num">${f3(w[i])}</td><td class="num">${f2(6 * w[i])}</td>
        ${CC.SETS.map(n => `<td class="num">${f2(m[n])}</td>`).join("")}</tr>`; }).join("")}</tbody></table>` : "";

    const firingRows = we ? we.rules.map(x => `<tr class="${x.strength > 0 ? "fired" : "idle"}">
        <td>${esc(x.id.split(".").pop())}</td>
        <td>${x.antecedents.map(t => `${esc(t.skill)} is <b>${esc(t.set)}</b> <span class="muted">(μ'=${f2(t.mu_w)})</span>`).join(" <b>AND</b> ")}</td>
        <td class="num">${x.antecedents.length > 1 ? `min(${x.antecedents.map(t => f2(t.mu_w)).join(", ")})` : f2(x.antecedents[0].mu_w)}</td>
        <td class="num"><b>${f2(x.strength)}</b></td><td>${esc(x.consequent)}</td></tr>`).join("") : "";

    const ev = ga.evaluation || {};
    const ci = m => m ? `${f4(m.mean)} <span class="muted small">[${f4(m.lo)}, ${f4(m.hi)}]</span>` : "—";
    const dci = d => d ? `${d.mean >= 0 ? "+" : ""}${f4(d.mean)} <span class="muted small">[${d.lo >= 0 ? "+" : ""}${f4(d.lo)}, ${d.hi >= 0 ? "+" : ""}${f4(d.hi)}]</span>` : "—";
    const testDiff = ev.test?.difference?.["ndcg@5"];
    const significant = testDiff && (testDiff.lo > 0 || testDiff.hi < 0);
    const verdict = !testDiff ? "" : significant
      ? (testDiff.mean > 0 ? "On the untouched test split, the GA weights are better than equal weights, and the 95% interval of the difference excludes zero."
                           : "On the untouched test split, the GA weights are <b>worse</b> than equal weights, and the 95% interval of the difference excludes zero.")
      : `On the untouched test split the difference is ${dci(testDiff)}. The interval contains zero, so the GA weights are <b>not distinguishable</b> from equal weights.
         The GA improved the training fitness, but that gain did not carry over to unseen profiles. This is reported honestly rather than claimed as an accuracy gain.`;

    // NDCG worked example (illustrative): reference A..E, model B, A, F, C, E
    const ref = ["A", "B", "C", "D", "E"], model = ["B", "A", "F", "C", "E"];
    const grade = c => { const i = ref.indexOf(c); return i < 0 ? 0 : 5 - i; };
    const disc = i => 1 / Math.log2(i + 2);
    const dcg = model.reduce((t, c, i) => t + grade(c) * disc(i), 0);
    const idcg = ref.reduce((t, c, i) => t + (5 - i) * disc(i), 0);

    CC.render(`<div class="sc">
      <h1>Soft Computing analysis</h1>
      <p class="muted">A step-by-step walkthrough of how this system computes a recommendation.
        ${r ? `The worked examples use <b>your latest assessment</b>${r.display_name ? ` (${esc(r.display_name)})` : ""} and its top career, <b>${esc(top.name)}</b>.`
            : `<b>Complete an assessment</b> (or open one from History) to see the worked examples with your own numbers. The playground below works without one.`}</p>
      <nav class="toc no-print" aria-label="Sections">${TOC.map(([id, t]) => `<button class="toc-link" data-jump="${id}">${t}</button>`).join("")}</nav>

      <section id="sc-overview" class="card">
        <h2>Overview: the pipeline</h2>
        <div class="flow" role="list">
          ${["6 self-ratings (0–100)", "Fuzzification → μ Low/Med/High", "GA weights → μ' = min(1, 6wμ)", "18 Mamdani FIS · 11 rules each",
             "Centroid → s (0–100)", "Score = s × (0.65 + 0.25·I + 0.10·W)", "Rank → top 5 + fit band"].map((t, i) =>
             `<div class="flow-step" role="listitem"><span class="flow-n">${i + 1}</span>${esc(t)}</div>`).join("")}
        </div>
        <p class="small muted">The GA runs <b>offline</b> (<code>python run_ga.py</code>). Each request only runs steps 1–7 with the saved weights.
          ${a.llm_configured ? "An optional LLM then rewrites the computed result as a paragraph; it does not change any score." : ""}</p>
      </section>

      <section id="sc-fuzzy" class="card">
        <h2>1 · What fuzzy logic is</h2>
        <p>Classical (crisp) logic puts a value in exactly one set: a programming rating of 74 is either "high" or "not high".
          Fuzzy logic (Zadeh, 1965) lets a value belong to several sets <b>to a degree</b> between 0 and 1. A rating of 74 is
          Medium to degree 0.52 and High to degree 0.48. Self-ratings are imprecise by nature, so this graded view fits them better
          than hard cut-offs, and a small change in a rating causes a small change in the result instead of a jump.</p>
        <div class="grid grid-2">
          <div class="formula"><b>Crisp set</b><br>χ<sub>High</sub>(x) = 1 if x ≥ 75, else 0<br><span class="muted">74 → 0, 75 → 1 (a jump)</span></div>
          <div class="formula"><b>Fuzzy set</b><br>μ<sub>High</sub>(x) ∈ [0, 1]<br><span class="muted">74 → 0.48, 75 → 0.50 (smooth)</span></div>
        </div>
        <p class="small">A <b>Mamdani fuzzy inference system</b> (Mamdani &amp; Assilian, 1975) reasons with IF–THEN rules over such sets:
          fuzzify the inputs → evaluate the rules → clip and combine the output sets → defuzzify to one crisp number.</p>
      </section>

      <section id="sc-mf" class="card">
        <h2>2 · Membership functions</h2>
        <p>Every input uses three triangular sets. A triangle (a, b, c) is</p>
        <div class="formula">μ(x; a, b, c) = max( 0, min( (x − a)/(b − a), (c − x)/(c − b) ) )</div>
        <p class="small">with a "shoulder" (the side equals 1) when a = b or b = c. The input sets are
          ${fz.sets.map((n, i) => `<b>${n}</b> (${fz.params[i].join(", ")})`).join(", ")}. They form a <b>Ruspini partition</b>:
          μ<sub>Low</sub> + μ<sub>Medium</sub> + μ<sub>High</sub> = 1 for every x, so every rating is fully described and no rating "falls between" sets.</p>
        <div class="grid grid-2">
          <div><h3>Input sets (each skill)</h3><div class="chart-box"><canvas id="mf"></canvas></div></div>
          <div><h3>Output sets (career suitability)</h3><div class="chart-box"><canvas id="mfo"></canvas></div>
            <p class="small muted">${fz.output_sets.map((n, i) => `${n} (${fz.output_params[i].join(", ")})`).join(" · ")}</p></div>
        </div>
        <h3>Worked example: ${esc(pSkill.label)} = ${px}</h3>
        <div class="formula">${fz.sets.map((n, i) => esc(trimfWorking(px, fz.params[i], `μ ${n}`))).join("<br>")}</div>
        ${r ? `<h3>All your inputs</h3>${fuzzTable}` : ""}
      </section>

      <section id="sc-weights" class="card">
        <h2>3 · GA weighting of the memberships</h2>
        <p>The genetic algorithm learns six weights w<sub>i</sub> ≥ 0 with Σw<sub>i</sub> = 1. They scale each input's memberships before the rules see them:</p>
        <div class="formula">μ'<sub>i</sub> = min( 1, 6 · w<sub>i</sub> · μ<sub>i</sub> )</div>
        <p class="small">With equal weights (w = 1/6) the factor 6w is 1 and nothing changes, so equal weights are the plain FIS baseline.
          A weight above 1/6 strengthens that skill's rules, and a weight below 1/6 weakens them.</p>
        ${r ? `<div class="formula">${esc(pSkill.label)}: 6 × ${f3(w[pIdx])} = ${f2(6 * w[pIdx])} →
            ${CC.SETS.map(n => `μ'<sub>${n}</sub> = min(1, ${f2(6 * w[pIdx])} × ${f2(r.analysis.memberships[pSkill.id][n])}) = ${f2(r.analysis.weighted_memberships[pSkill.id][n])}`).join("; ")}</div>
          ${weightTable}` : ""}
      </section>

      <section id="sc-rules" class="card">
        <h2>4 · Rule base</h2>
        <p>Each career has its own FIS. It uses two <b>primary</b> skills (P1, P2) and one <b>secondary</b> skill (S), chosen per career in
          <code>data/careers.json</code>. Every career uses the same ${fz.rules_per_career}-rule template, so ${fz.rules.length} rules in total:</p>
        <table><thead><tr><th>#</th><th>IF</th><th>THEN suitability is</th><th>Purpose</th></tr></thead><tbody>
          ${fz.template.map(t => `<tr><td>R${t.k}</td><td>${t.antecedents.map(esc).join(" AND ")}</td><td><b>${esc(t.consequent)}</b></td><td class="small muted">${esc(t.role)}</td></tr>`).join("")}
        </tbody></table>
        <p class="small">R1–R9 cover the full 3 × 3 grid of P1 × P2, so at least one rule fires for every input. A Low primary skill caps the result at Low (R2–R5).
          R10 penalises a missing secondary skill, and R11 tempers "Very High" when the secondary skill is only moderate.</p>
        <details><summary>Primary and secondary skills for all 18 careers</summary>
          <table><thead><tr><th>Career</th><th>P1</th><th>P2</th><th>S</th></tr></thead><tbody>
          ${fz.rule_skills.map(x => `<tr><td>${esc(x.career)}</td><td>${esc(x.primary[0])}</td><td>${esc(x.primary[1])}</td><td>${esc(x.secondary)}</td></tr>`).join("")}
          </tbody></table></details>
        <details><summary>Full rule base (${fz.rules.length} rules)</summary>
          ${fz.rules.map(x => `<div class="rule">${esc(x.id)}: ${esc(x.text)}</div>`).join("")}</details>
      </section>

      <section id="sc-firing" class="card">
        <h2>5 · How a rule fires</h2>
        <p>The <b>firing strength</b> α of a rule is how true its IF part is. With AND = min, it is the smallest weighted membership in the antecedent:</p>
        <div class="formula">α<sub>r</sub> = min( μ'<sub>1</sub>, μ'<sub>2</sub>, … )</div>
        ${we ? `<h3>Worked example: ${esc(top.name)}</h3>
          <div class="table-wrap"><table class="firing"><thead><tr><th>Rule</th><th>Antecedent</th><th class="num">AND (min)</th><th class="num">α</th><th>Consequent</th></tr></thead>
          <tbody>${firingRows}</tbody></table></div>
          <p class="small muted">Highlighted rows fired (α &gt; 0). The others have a zero membership somewhere in their IF part.</p>`
          : `<p class="muted">Complete an assessment to see each rule's firing strength for your top career, or use the playground below.</p>`}
      </section>

      <section id="sc-agg" class="card">
        <h2>6 · Implication and aggregation</h2>
        <p><b>Implication (min):</b> each fired rule clips its output set at height α. <b>Aggregation (max):</b> the clipped sets are combined into one shape by taking the maximum at every point u.
          Rules with the same consequent can be merged first, because max<sub>r</sub> min(α<sub>r</sub>, μ<sub>k</sub>(u)) = min(max<sub>r</sub> α<sub>r</sub>, μ<sub>k</sub>(u)).</p>
        <div class="formula">μ<sub>agg</sub>(u) = max<sub>k</sub> min( λ<sub>k</sub>, μ<sub>k</sub>(u) ),&nbsp;&nbsp; λ<sub>k</sub> = max α over the rules whose consequent is set k</div>
        ${we ? `<div class="grid grid-2">
          <div><table><thead><tr><th>Output set k</th><th class="num">Clip level λ<sub>k</sub></th></tr></thead><tbody>
            ${fz.output_sets.map(n => `<tr><td>${esc(n)}</td><td class="num">${f2(we.clip_levels[n])}</td></tr>`).join("")}</tbody></table></div>
          <div class="chart-box"><canvas id="agg"></canvas></div></div>` : ""}
      </section>

      <section id="sc-centroid" class="card">
        <h2>7 · Defuzzification (centroid)</h2>
        <p>The aggregated shape is turned into one number, the <b>centre of gravity</b> of the area under it. The universe 0–100 is sampled at 201 points (step 0.5):</p>
        <div class="formula">s = Σ<sub>u</sub> μ<sub>agg</sub>(u) · u &nbsp;/&nbsp; Σ<sub>u</sub> μ<sub>agg</sub>(u)</div>
        ${we ? `<div class="formula">s<sub>${esc(top.name)}</sub> = ${f2(we.moment)} / ${f2(we.area)} = <b>${f2(we.centroid)}</b></div>
          <p class="small">Why five output sets? With one set, the centroid of a clipped triangle barely moves as α changes, so many careers ended up with the same s
            (the v1 "saturation" problem). With graded consequents, a stronger profile shifts weight from Medium to High to Very High, and the centroid moves with it.</p>` : ""}
      </section>

      <section id="sc-score" class="card">
        <h2>8 · Hybrid score and fit band</h2>
        <div class="formula">${esc(a.score_formula)}<br><span class="muted">${esc(a.score_formula_no_prefs)}</span></div>
        <p class="small">I is the rank-weighted share of your interests linked to the career (interest ranked r of k has weight k − r + 1).
          W is the average work-preference match (1 match, 0.5 partial, 0 mismatch). If no preference is given, its 0.10 share is redistributed pro rata.</p>
        <p class="small"><b>Why multiply instead of add?</b> In v1 the interest term was added, so a student who rated every skill 50 but ticked one interest
          still scored 65, with 30 of those points coming from the interest alone. Now the multiplier m lies in [0.65, 1]: interest and preferences can discount a career you
          are not drawn to, but the score can never exceed the skill-based suitability s. Every skill at 50 now gives 50 (Moderate), every skill at 30 gives 36 (Weak).</p>
        ${top ? `<div class="formula">${esc(top.name)}: s = ${f2(top.fis_suitability)}, I = ${f2(top.interest_match)}${top.preference_match != null ? `, W = ${f2(top.preference_match)}` : ""}<br>
          m = ${top.preference_match != null ? `0.65 + 0.25 × ${f2(top.interest_match)} + 0.10 × ${f2(top.preference_match)}` : `(0.65 + 0.25 × ${f2(top.interest_match)}) / 0.90`} = ${f3(top.breakdown.multiplier)}<br>
          Score = ${f2(top.fis_suitability)} × ${f3(top.breakdown.multiplier)} = <b>${top.score.toFixed(1)}</b> → ${CC.bandPill(top.fit)}</div>` : ""}
        <table><thead><tr><th>Fit band</th><th class="num">Score</th></tr></thead><tbody>
          ${a.bands.map((b, i) => `<tr><td>${CC.bandPill(b)}</td><td class="num">${i === 0 ? `≥ ${b.min}` : `${b.min} – ${a.bands[i - 1].min}`}</td></tr>`).join("")}</tbody></table>
      </section>

      <section id="sc-play" class="card">
        <h2>9 · Interactive playground</h2>
        <p class="small muted">Move a slider and watch the memberships, rule strengths, aggregated output and centroid update. Every update runs the real engine on the server.</p>
        <div class="play">
          <div class="play-controls">
            <label>Career <select id="pg-career">${fz.rule_skills.map(x => `<option value="${x.id}" ${x.id === (top?.id || "software_developer") ? "selected" : ""}>${esc(x.career)}</option>`).join("")}</select></label>
            <div class="row small"><label><input type="radio" name="pg-w" value="optimized" checked> GA weights</label>
              <label><input type="radio" name="pg-w" value="equal"> Equal weights</label></div>
            ${skills.map(s => { const v = r ? r.analysis.inputs[s.id] : 60; return `<div class="field compact"><label for="pg-${s.id}">${esc(s.label)} <span class="pg-role" id="pg-role-${s.id}"></span></label>
              <div class="slider-row"><input id="pg-${s.id}" type="range" min="0" max="100" step="1" value="${v}" data-pg="${s.id}"><output id="pg-out-${s.id}">${v}</output></div></div>`; }).join("")}
          </div>
          <div class="play-out">
            <div class="play-s"><span class="muted small">FIS suitability s</span><b id="pg-s">—</b><span class="small muted" id="pg-calc"></span></div>
            <div class="chart-box"><canvas id="pg-agg"></canvas></div>
            <h3>Rule strengths</h3><div id="pg-rules" class="pg-rules"></div>
            <h3>Weighted memberships μ'</h3><div class="chart-box short"><canvas id="pg-mu"></canvas></div>
          </div>
        </div>
      </section>

      <section id="sc-ga" class="card">
        <h2>10 · Genetic algorithm</h2>
        <p>A GA (Holland, 1975) searches by evolving a <b>population</b> of candidate solutions with selection, crossover and mutation.
          Here a candidate (chromosome) is the six input weights. The implementation is custom NumPy code in <code>engine/ga.py</code>.</p>
        <h3>Chromosome (selected solution)</h3>
        <div class="chromo">${a.skills.map((s, i) => `<div class="gene" style="--h:${Math.round(w[i] * 300)}%"><b>${f3(w[i])}</b><span>${esc(s)}</span></div>`).join("")}</div>
        <p class="small muted">Real-valued genes, each ≥ 0, summing to 1. Equal weights would be 0.167 each.</p>
        <h3>One generation</h3>
        <div class="flow loop">${["Population of 30", "Evaluate fitness (NDCG@5 on train)", "Tournament selection (k = 3)", "BLX-α crossover (p = 0.8)",
          "Gaussian mutation (p = 0.2)", "Repair: clip ≥ 0, normalise", "Elitism: keep best 1", "Check validation → early stop?"].map((t, i) =>
          `<div class="flow-step"><span class="flow-n">${i + 1}</span>${esc(t)}</div>`).join("")}</div>
        <div class="grid grid-2">
          <div><h3>Tournament selection</h3><p class="small">Pick ${ga.config.tournament_size} chromosomes at random and keep the fittest as a parent.
            Fitter chromosomes win more often, but weaker ones still have a chance, which keeps diversity.</p>
            <svg viewBox="0 0 320 90" class="diagram" role="img" aria-label="Tournament of three, the fittest wins">
              ${[0.78, 0.91, 0.84].map((f, i) => `<rect x="${10 + i * 75}" y="10" width="62" height="34" rx="6" class="${f === 0.91 ? "d-win" : "d-box"}"/><text x="${41 + i * 75}" y="32" text-anchor="middle">${f}</text>`).join("")}
              <path d="M240 27 H262" class="d-line" marker-end="url(#arr)"/><rect x="266" y="10" width="48" height="34" rx="6" class="d-win"/><text x="290" y="32" text-anchor="middle">0.91</text>
              <text x="160" y="72" text-anchor="middle" class="d-cap">fitness of 3 random contenders → winner becomes a parent</text>
              <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="d-head"/></marker></defs>
            </svg></div>
          <div><h3>Blend crossover (BLX-α)</h3><p class="small">For each gene, with parents p₁ &lt; p₂ and d = p₂ − p₁, the child is drawn uniformly from
            [p₁ − αd, p₂ + αd], with α = ${ga.config.blend_alpha}. Children can land slightly outside the parents, which helps exploration.</p>
            <svg viewBox="0 0 320 90" class="diagram" role="img" aria-label="BLX-alpha interval extends the parent interval by alpha times d on each side">
              <line x1="20" y1="40" x2="300" y2="40" class="d-axis"/>
              <rect x="60" y="30" width="200" height="20" class="d-range"/>
              <rect x="110" y="30" width="100" height="20" class="d-parents"/>
              <circle cx="110" cy="40" r="6" class="d-p"/><circle cx="210" cy="40" r="6" class="d-p"/>
              <text x="110" y="22" text-anchor="middle">p₁</text><text x="210" y="22" text-anchor="middle">p₂</text>
              <text x="85" y="68" text-anchor="middle" class="d-cap">αd</text><text x="160" y="68" text-anchor="middle" class="d-cap">d</text><text x="235" y="68" text-anchor="middle" class="d-cap">αd</text>
              <text x="160" y="85" text-anchor="middle" class="d-cap">child gene ~ Uniform over the shaded range</text>
            </svg></div>
          <div><h3>Gaussian mutation</h3><p class="small">With probability ${ga.config.mutation_prob}, every gene gets small noise:
            w<sub>i</sub> ← w<sub>i</sub> + N(0, σ²), σ = ${ga.config.mutation_sigma}. This keeps the search from getting stuck.</p></div>
          <div><h3>Repair and elitism</h3><p class="small">After crossover and mutation: w ← max(w, 0), then w ← w / Σw (all zero → uniform), so every chromosome is a valid weight vector.
            Elitism copies the best ${ga.config.elitism} chromosome unchanged, so the best training fitness never decreases.</p></div>
        </div>
        <table><tbody>
          <tr><td>Population · max generations</td><td class="num">${ga.config.population} · ${ga.config.generations}</td></tr>
          <tr><td>Early stopping</td><td class="num">patience ${ga.config.patience} generations on validation NDCG@5</td></tr>
          <tr><td>Selected chromosome</td><td class="num">generation ${ga.selected_generation} (best validation ${f4(ga.selected_val_fitness)})${ga.stopped_early ? ", stopped early" : ""}</td></tr>
          <tr><td>Random seed · runtime</td><td class="num">${ga.config.seed} · ${ga.runtime_seconds} s offline (${esc(ga.run_at)})</td></tr>
        </tbody></table>
      </section>

      <section id="sc-fitness" class="card">
        <h2>11 · Fitness function: NDCG@5</h2>
        <p>For each training profile the model ranks all 18 careers. Its top 5 is compared with a <b>reference top 5</b> from a separate expert function.
          The reference careers get grades 5, 4, 3, 2, 1 by position. <b>NDCG@5</b> (Järvelin &amp; Kekäläinen, 2002) rewards putting the right careers near the top:</p>
        <div class="formula">DCG@5 = Σ<sub>i=1..5</sub> grade(model<sub>i</sub>) / log₂(i + 1) &nbsp;&nbsp; NDCG@5 = DCG@5 / IDCG@5</div>
        <p class="small">IDCG is the DCG of the perfect order, so NDCG is 1 for a perfect match and 0 when none of the reference careers appear. Fitness = mean NDCG@5 over the training profiles.</p>
        <h3>Worked example</h3>
        <p class="small">Reference: ${ref.join(", ")} (grades 5…1). Model: ${model.join(", ")}.</p>
        <div class="formula">DCG = ${model.map((c, i) => `${grade(c)}/log₂(${i + 2})`).join(" + ")} = ${f3(dcg)}<br>
          IDCG = ${ref.map((c, i) => `${5 - i}/log₂(${i + 2})`).join(" + ")} = ${f3(idcg)}<br>
          NDCG@5 = ${f3(dcg)} / ${f3(idcg)} = <b>${f3(dcg / idcg)}</b> &nbsp;·&nbsp; Precision@5 = ${model.filter(c => ref.includes(c)).length}/5 = ${f2(model.filter(c => ref.includes(c)).length / 5)}</div>
      </section>

      <section id="sc-train" class="card">
        <h2>12 · Training</h2>
        <div class="grid grid-2">
          <div><h3>Fitness by generation</h3><div class="chart-box"><canvas id="fit"></canvas></div>
            <p class="small muted">The dot marks the generation whose best chromosome had the highest <b>validation</b> NDCG@5. That chromosome is the one the app uses.</p></div>
          <div><h3>Weights: baseline vs GA</h3><div class="chart-box"><canvas id="wts"></canvas></div></div>
        </div>
      </section>

      <section id="sc-results" class="card">
        <h2>13 · Results with 95% confidence intervals</h2>
        <p class="small">Mean over profiles, with a ${ga.bootstrap?.resamples || 2000}-resample percentile bootstrap 95% CI in brackets (Efron &amp; Tibshirani, 1993).
          The difference column is a paired bootstrap on the same profiles.</p>
        <div class="table-wrap"><table><thead><tr><th>Split</th><th class="num">n</th><th class="num">Baseline NDCG@5</th><th class="num">GA NDCG@5</th><th class="num">Difference</th><th class="num">Baseline P@5</th><th class="num">GA P@5</th></tr></thead><tbody>
          ${["train", "val", "test"].map(s => ev[s] ? `<tr class="${s === "test" ? "hl" : ""}"><td>${s === "test" ? "<b>Test (untouched)</b>" : s === "val" ? "Validation (model selection)" : "Training (optimised)"}</td>
            <td class="num">${ga.data[s]}</td><td class="num">${ci(ev[s].baseline["ndcg@5"])}</td><td class="num">${ci(ev[s].optimized["ndcg@5"])}</td>
            <td class="num">${dci(ev[s].difference["ndcg@5"])}</td><td class="num">${ci(ev[s].baseline["precision@5"])}</td><td class="num">${ci(ev[s].optimized["precision@5"])}</td></tr>` : "").join("")}
        </tbody></table></div>
        <p class="notice ${significant && testDiff.mean > 0 ? "info" : ""}" style="margin-top:12px">${verdict}</p>
        ${ga.robustness ? `<h3>Robustness across GA seeds</h3>
          <table><thead><tr><th class="num">Seed</th><th class="num">Selected gen.</th><th class="num">Val NDCG@5</th><th class="num">Test NDCG@5</th><th>Weights (P, M, PS, C, Co, TI)</th></tr></thead><tbody>
          ${ga.robustness.seeds.map(x => `<tr><td class="num">${x.seed}</td><td class="num">${x.generation}</td><td class="num">${f4(x["val_ndcg@5"])}</td><td class="num">${f4(x["test_ndcg@5"])}</td>
            <td class="small">${x.weights.map(f2).join(", ")}</td></tr>`).join("")}</tbody></table>
          <p class="small">Test NDCG@5 over ${ga.robustness.seeds.length} seeds: <b>${f4(ga.robustness["test_ndcg@5_mean"])} ± ${f4(ga.robustness["test_ndcg@5_sd"])}</b> (mean ± SD). The small spread shows the result does not depend on one lucky seed.</p>` : ""}
      </section>

      <section id="sc-data" class="card">
        <h2>14 · Data</h2>
        <p>${ga.data.profiles} <b>synthetic</b> profiles: ${ga.data.train} train, ${ga.data.val} validation, ${ga.data.test} test (seed ${ga.data.profile_seed}).
          Each is sampled around one career's target profile with Gaussian noise (σ = 12). 20% blend two careers. Interests include the career's tags with p = 0.85 and other tags with p = 0.15.</p>
        <p>The reference labels come from a <b>separate expert function</b> that does not use the fuzzy rules:</p>
        <div class="formula">fit<sub>c</sub> = 1 − Σ<sub>i</sub> t<sub>i</sub> · max(0, t<sub>i</sub> − x<sub>i</sub>) / Σ<sub>i</sub> t<sub>i</sub>² &nbsp;&nbsp; expert<sub>c</sub> = 0.65 · fit<sub>c</sub> + 0.35 · I<sub>c</sub></div>
        <p class="small muted">${esc(ga._note)}</p>
      </section>

      <section id="sc-limits" class="card">
        <h2>15 · Limitations</h2>
        <ul>
          <li><b>Synthetic evaluation.</b> The labels encode our own assumptions, so the numbers measure consistency with them, not real-world accuracy.</li>
          <li><b>Shared interest term.</b> The model and the expert labels both use the interest match, which raises every agreement score.</li>
          <li><b>GA gain is not significant on unseen data.</b> The GA clearly beats equal weights on training data, but on the untouched test split the 95% CI of the difference includes zero.</li>
          <li><b>Shared rule template.</b> Careers with the same primary pair (for example Cloud Engineer and Network Engineer) get similar s and are separated mainly by S, interests and preferences.</li>
          <li><b>Self-reported inputs.</b> Ratings are subjective. Career targets and work profiles are model assumptions, not industry standards.</li>
          <li><b>Single-node prototype.</b> History lives in the browser only. The optional LLM summary depends on an external service and falls back to a template.</li>
        </ul>
      </section>
    </div>`);

    // TOC: scroll instead of changing the hash (the hash drives the router)
    document.querySelectorAll("[data-jump]").forEach(b => b.onclick = () => document.getElementById(b.dataset.jump)?.scrollIntoView({ behavior: "smooth", block: "start" }));

    const C = CC.COLORS;
    const curveX = fz.curves.x;
    const lineOpts = (xt, yt) => ({ parsing: false, scales: { x: { type: "linear", min: 0, max: 100, title: { display: true, text: xt } },
                                                              y: { min: 0, max: 1.05, title: { display: true, text: yt } } } });
    const xy = ys => ys.map((y, i) => ({ x: curveX[i], y }));
    CC.chart("mf", { type: "line", data: { datasets: CC.SETS.map(s => ({ label: s, data: xy(fz.curves.y[s]), borderColor: C[s.toLowerCase()], pointRadius: 0, borderWidth: 2 })) },
      options: lineOpts("Input value x", "Membership μ") });
    CC.chart("mfo", { type: "line", data: { datasets: fz.output_sets.map((s, i) => ({ label: s, data: xy(fz.curves.out[s]), borderColor: C.out[i], pointRadius: 0, borderWidth: 2 })) },
      options: lineOpts("Suitability u", "Membership μ") });

    if (we) {
      const ds = fz.output_sets.map((s, i) => ({ label: `${s} (λ=${f2(we.clip_levels[s])})`, data: xy(fz.curves.out[s].map(v => Math.min(v, we.clip_levels[s]))),
        borderColor: C.out[i], borderDash: [4, 3], pointRadius: 0, borderWidth: 1.5 }));
      ds.push({ label: "Aggregated μ_agg", data: xy(top.aggregated_curve), borderColor: C.brand, backgroundColor: "rgba(61,78,216,.18)", fill: true, pointRadius: 0, borderWidth: 2 });
      ds.push({ label: `Centroid s = ${f2(we.centroid)}`, data: [{ x: we.centroid, y: 0 }, { x: we.centroid, y: 1 }], borderColor: C.rose, borderWidth: 2, pointRadius: 0 });
      CC.chart("agg", { type: "line", data: { datasets: ds }, options: lineOpts("Suitability u", "μ") });
    }

    if (!ga.missing) {
      const h = ga.history;
      const sel = h.find(x => x.generation === ga.selected_generation);
      CC.chart("fit", { type: "line", data: { datasets: [
          { label: "Best (train)", data: h.map(x => ({ x: x.generation, y: x.best })), borderColor: C.brand, pointRadius: 0 },
          { label: "Average (train)", data: h.map(x => ({ x: x.generation, y: x.average })), borderColor: C.accent, pointRadius: 0 },
          { label: "Validation (best chromosome)", data: h.filter(x => x.val != null).map(x => ({ x: x.generation, y: x.val })), borderColor: C.warn, pointRadius: 0, borderDash: [4, 3] },
          ...(sel ? [{ label: "Selected", data: [{ x: sel.generation, y: sel.val }], borderColor: C.rose, backgroundColor: C.rose, pointRadius: 6, showLine: false }] : [])] },
        options: { parsing: false, scales: { x: { type: "linear", title: { display: true, text: "Generation" } }, y: { title: { display: true, text: "NDCG@5" } } } } });
      CC.chart("wts", { type: "bar", data: { labels: a.skills, datasets: [
          { label: "Baseline (equal)", data: ga.baseline_weights, backgroundColor: C.grey },
          { label: "Best in initial population", data: ga.initial_best_weights, backgroundColor: "#b9c0ff" },
          { label: "GA-selected", data: ga.optimized_weights, backgroundColor: C.brand }] },
        options: { scales: { y: { min: 0, title: { display: true, text: "Weight" } } } } });
    }

    initPlayground(CC, fz, skills);
  };

  function initPlayground(CC, fz, skills) {
    const C = CC.COLORS;
    const aggChart = CC.chart("pg-agg", { type: "line", data: { datasets: [
        { label: "Aggregated μ_agg", data: [], borderColor: C.brand, backgroundColor: "rgba(61,78,216,.18)", fill: true, pointRadius: 0, borderWidth: 2 },
        { label: "Centroid", data: [], borderColor: C.rose, borderWidth: 2, pointRadius: 0 }] },
      options: { parsing: false, scales: { x: { type: "linear", min: 0, max: 100, title: { display: true, text: "Suitability u" } }, y: { min: 0, max: 1.05 } } } });
    const muChart = CC.chart("pg-mu", { type: "bar", data: { labels: skills.map(s => s.label), datasets: CC.SETS.map(s => ({ label: s, data: [], backgroundColor: C[s.toLowerCase()] })) },
      options: { scales: { y: { min: 0, max: 1 } } } });
    let timer = null, seq = 0;

    function roles() {
      const id = document.getElementById("pg-career").value;
      const rs = fz.rule_skills.find(x => x.id === id);
      skills.forEach(s => {
        const el = document.getElementById(`pg-role-${s.id}`);
        el.textContent = rs.primary.includes(s.label) ? "primary" : rs.secondary === s.label ? "secondary" : "";
        el.className = `pg-role ${el.textContent}`;
      });
    }
    async function update() {
      const my = ++seq;
      const body = { career_id: document.getElementById("pg-career").value,
                     weights: document.querySelector("input[name=pg-w]:checked").value,
                     skills: Object.fromEntries(skills.map(s => [s.id, Number(document.getElementById(`pg-${s.id}`).value)])) };
      try {
        const p = await CC.api("/api/playground", { method: "POST", body: JSON.stringify(body) });
        if (my !== seq || !document.getElementById("pg-s")) return;  // a newer request is in flight, or the page changed
        document.getElementById("pg-s").textContent = p.no_rule_fired ? "0 (no rule fired)" : p.s.toFixed(2);
        document.getElementById("pg-calc").textContent = p.no_rule_fired ? "" : `= ${p.moment.toFixed(1)} / ${p.area.toFixed(2)}`;
        if (aggChart) {
          aggChart.data.datasets[0].data = p.aggregated.map((y, i) => ({ x: p.universe[i], y }));
          aggChart.data.datasets[1].data = [{ x: p.s, y: 0 }, { x: p.s, y: 1 }];
          aggChart.data.datasets[1].label = `Centroid s = ${p.s.toFixed(1)}`;
          aggChart.update("none");
        }
        if (muChart) {
          CC.SETS.forEach((_, k) => { muChart.data.datasets[k].data = p.weighted_memberships.map(row => row[k]); });
          muChart.update("none");
        }
        document.getElementById("pg-rules").innerHTML = p.rules.map(r => `
          <div class="pg-rule ${r.strength > 0 ? "on" : ""}" title="${CC.esc(r.text)}">
            <span class="pg-id">${CC.esc(r.id.split(".").pop())}</span>
            <span class="pg-txt">${r.antecedents.map(t => `${CC.esc(t.skill)} ${CC.esc(t.set)}`).join(" ∧ ")} → <b>${CC.esc(r.consequent)}</b></span>
            <span class="pg-bar"><i style="width:${(r.strength * 100).toFixed(0)}%"></i></span><span class="pg-v">${r.strength.toFixed(2)}</span></div>`).join("");
      } catch { /* keep the last good state; the next slider move retries */ }
    }
    const schedule = () => { clearTimeout(timer); timer = setTimeout(update, 60); };
    document.querySelectorAll("[data-pg]").forEach(el => el.addEventListener("input", () => {
      document.getElementById(`pg-out-${el.dataset.pg}`).textContent = el.value; schedule();
    }));
    document.getElementById("pg-career").addEventListener("change", () => { roles(); schedule(); });
    document.querySelectorAll("input[name=pg-w]").forEach(el => el.addEventListener("change", schedule));
    roles();
    update();
  }
})();
