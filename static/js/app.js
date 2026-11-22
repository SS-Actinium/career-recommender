/* CareerCompass frontend: vanilla JS single-page app with hash routing.
   All scoring happens on the Flask backend. This file collects input, renders results,
   keeps a per-browser history (localStorage) and hosts the shared helpers used by sc.js. */
(() => {
  "use strict";

  const app = document.getElementById("app");
  const charts = [];
  const SETS = ["Low", "Medium", "High"];
  const OUT_SETS = ["Very Low", "Low", "Medium", "High", "Very High"];
  const COLORS = {
    brand: "#3d4ed8", accent: "#0f9d8a", warn: "#d97706", grey: "#9aa1b2", rose: "#c0396b",
    low: "#d97706", medium: "#3d4ed8", high: "#0f9d8a",
    out: ["#b91c1c", "#d97706", "#3d4ed8", "#0f9d8a", "#7c3aed"]
  };
  const HISTORY_KEY = "cc_history_v2";
  const HISTORY_MAX = 30;

  const state = {
    meta: null,
    analysis: null,
    form: { display_name: "", skills: {}, interests: [], work_preferences: {} },
    step: 0,
    result: loadResult(),
  };

  // ---------- helpers ----------
  function esc(v) {
    return String(v ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function loadResult() {
    try { return JSON.parse(sessionStorage.getItem("cc_result") || "null"); } catch { return null; }
  }
  function saveResult(r) {
    try { sessionStorage.setItem("cc_result", JSON.stringify(r)); } catch { /* storage unavailable: keep in memory */ }
  }
  async function api(path, opts = {}) {
    const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON */ }
    if (!res.ok) { const e = new Error(body?.message || `Request failed (${res.status})`); e.body = body; e.status = res.status; throw e; }
    return body;
  }
  function clearCharts() { while (charts.length) charts.pop().destroy(); }
  function chart(id, config) {
    const el = typeof id === "string" ? document.getElementById(id) : id;
    if (!el || typeof Chart === "undefined") return null;
    config.options = { responsive: true, maintainAspectRatio: false, animation: false, ...config.options };
    const c = new Chart(el, config);
    charts.push(c);
    return c;
  }
  function label(skillId) { return state.meta?.skills.find(s => s.id === skillId)?.label || skillId; }
  function interestLabel(id) { return state.meta?.interests.find(i => i.id === id)?.label || id; }
  async function ensureMeta() { if (!state.meta) state.meta = await api("/api/meta"); initDefaults(); }
  async function ensureAnalysis() { if (!state.analysis) state.analysis = await api("/api/analysis"); }
  function initDefaults() {
    for (const s of state.meta.skills) if (state.form.skills[s.id] === undefined) state.form.skills[s.id] = 50;
  }
  function render(html) { clearCharts(); app.innerHTML = html; app.focus({ preventScroll: true }); window.scrollTo(0, 0); }
  function noResult() {
    return `<div class="card stack"><h2>No results yet</h2><p class="muted">Complete the assessment, or open a past result from History.</p>
      <div class="row"><a class="btn" href="#/instructions">Start Career Assessment</a><a class="btn secondary" href="#/history">Open History</a></div></div>`;
  }
  function bandPill(fit) { return `<span class="pill band-${esc(fit.key)}">${esc(fit.label)}</span>`; }
  function fmtDate(iso) {
    try { return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }); } catch { return iso; }
  }
  function prefText(prefs) {
    return Object.entries(prefs || {}).filter(([, v]) => v).map(([k, v]) => `${esc(k.replace("_", " "))}: <b>${esc(v)}</b>`).join(" · ");
  }

  // ---------- history (this browser only) ----------
  function readHistory() {
    try { const h = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); return Array.isArray(h) ? h : []; } catch { return []; }
  }
  function writeHistory(list) {
    let items = list.slice(0, HISTORY_MAX);
    for (; ;) {  // shrink if the browser quota is hit
      try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items)); return true; } catch {
        if (items.length <= 1) return false;
        items = items.slice(0, -1);
      }
    }
  }
  function addHistory(result, tag) {
    const entry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, at: new Date().toISOString(),
      tag: tag || null, result
    };
    writeHistory([entry, ...readHistory()]);
    return entry;
  }
  function updateHistorySummary(result) {
    const list = readHistory();
    const e = list.find(x => x.result?.input && JSON.stringify(x.result.input) === JSON.stringify(result.input));
    if (e) { e.result.summary = result.summary; writeHistory(list); }
  }

  // ---------- pages ----------
  function home() {
    render(`
      <section class="hero">
        <div>
          <span class="pill">Soft Computing · MCA project prototype</span>
          <h1>Find career paths that fit how you think and what you enjoy.</h1>
          <p class="lead">Rate six skills, rank your interests, add work preferences, and get five ranked career suggestions with
            fit bands, clear reasons, skill gaps and a learning roadmap. The backend uses a Mamdani fuzzy inference system whose input
            weights were tuned by a genetic algorithm.</p>
          <div class="row"><a class="btn" href="#/instructions">Start Career Assessment</a>
            <a class="btn secondary" href="#/analysis">See the algorithms</a></div>
        </div>
        <div class="hero-visual card">
          <h3>How it works</h3>
          <ol class="steps">
            <li><b>Fuzzification.</b> Each 0–100 rating becomes degrees of Low, Medium and High.</li>
            <li><b>Fuzzy rules.</b> 18 Mamdani systems (one per career, 11 rules each) produce a suitability from Very Low to Very High.</li>
            <li><b>GA weights.</b> A genetic algorithm tuned how much each skill matters, on 500 synthetic profiles.</li>
            <li><b>Ranking.</b> Interest and work preferences scale the fuzzy suitability: Score = s × (0.65 + 0.25·I + 0.10·W). They can lower a career you are not drawn to, but never inflate weak skills.</li>
          </ol>
        </div>
      </section>
      <section class="grid grid-3">
        <div class="card"><h3>Fuzzy Logic</h3><p class="muted">A rating of 65 is not just "medium" or just "high". It is partly both, and the rules reason with those degrees.</p></div>
        <div class="card"><h3>Genetic Algorithm</h3><p class="muted">The GA evolved 30 candidate weight vectors with selection, crossover and mutation, and kept the one that did best on a separate validation set.</p></div>
        <div class="card"><h3>Explainable results</h3><p class="muted">Each suggestion shows the rules that fired, the score breakdown, your strongest factors and the skills to build next.</p></div>
      </section>
      <p class="notice" style="margin-top:24px">These recommendations are guidance for exploration, not a guarantee of career success.
        They do not replace advice from a qualified career counsellor.</p>`);
  }

  function instructions() {
    render(`
      <div class="card stack" style="max-width:760px">
        <h1>Before you start</h1>
        <p>The assessment has four short sections and takes about <b>3–5 minutes</b>.</p>
        <ol class="steps">
          <li><b>Profile (optional).</b> A display name, or leave it blank to stay anonymous.</li>
          <li><b>Skill ratings.</b> Rate six skills from 0 to 100. These are <b>self-reported</b>, not exam marks.</li>
          <li><b>Interests.</b> Pick the areas you enjoy and put them in order. Your first choice counts most.</li>
          <li><b>Work preferences (optional).</b> They make up 10% of the score multiplier. If you skip them, the score uses skills and interests only.</li>
        </ol>
        <h3>Using the 0–100 scale</h3>
        <table><tbody>
          <tr><td class="num">0–25</td><td>Little or no experience yet</td></tr>
          <tr><td class="num">25–50</td><td>Basic: you understand the fundamentals</td></tr>
          <tr><td class="num">50–75</td><td>Comfortable: you can work on your own with some help</td></tr>
          <tr><td class="num">75–100</td><td>Strong: you could teach or lead others</td></tr>
        </tbody></table>
        <p class="notice info"><b>Privacy:</b> we do not ask for your email, age, gender or any other personal details.
          Your answers go to the server only to compute results, and nothing is saved on the server.
          Your results are kept in <b>this browser's</b> History page until you delete them.
          ${state.meta?.llm_configured ? "If the AI summary is enabled, only your scores and skill names are sent to the LLM service, never your name." : ""}</p>
        <div class="row"><a class="btn" href="#/assess">Begin assessment</a></div>
      </div>`);
  }

  const STEP_TITLES = ["Profile", "Skill ratings", "Interests", "Work preferences", "Review"];
  const PREF_NAMES = {
    work_type: "Preferred work type", environment: "Preferred environment", activity: "Preferred activity",
    advanced_math: "Willingness to learn advanced mathematics"
  };

  function assess() {
    const f = state.form, m = state.meta, step = state.step;
    const progress = STEP_TITLES.map((_, i) => `<span class="${i <= step ? "done" : ""}"></span>`).join("");
    let body = "";
    if (step === 0) {
      body = `<div class="field"><label for="dn">Display name (optional)</label>
        <input id="dn" type="text" maxlength="40" value="${esc(f.display_name)}" placeholder="e.g. Student A">
        <p class="muted small">Leave this blank to stay anonymous. The name is only shown in this browser.</p></div>`;
    } else if (step === 1) {
      body = `<p class="muted">Self-reported ratings on a 0–100 scale.</p>` + m.skills.map(s => `
        <div class="field"><label for="sk-${s.id}">${esc(s.label)}</label>
          <div class="slider-row"><input id="sk-${s.id}" type="range" min="0" max="100" step="1" value="${f.skills[s.id]}"
            data-skill="${s.id}" aria-describedby="out-${s.id}"><output id="out-${s.id}">${f.skills[s.id]}</output></div></div>`).join("");
    } else if (step === 2) {
      body = `<p class="muted">Select at least one interest. The order you pick them in sets their rank, and you can reorder them below.</p>
        <div class="checks">${m.interests.map(i => `
          <label class="check ${f.interests.includes(i.id) ? "selected" : ""}"><input type="checkbox" data-interest="${i.id}"
            ${f.interests.includes(i.id) ? "checked" : ""}> ${esc(i.label)}</label>`).join("")}</div>
        <h3 style="margin-top:18px">Your ranking</h3>
        ${f.interests.length ? `<ol class="rank-list">${f.interests.map((id, k) => `
          <li><span class="n">${k + 1}</span><span class="spacer">${esc(interestLabel(id))}</span>
            <button class="btn ghost" data-move="${k}" data-dir="-1" ${k === 0 ? "disabled" : ""} aria-label="Move up">↑</button>
            <button class="btn ghost" data-move="${k}" data-dir="1" ${k === f.interests.length - 1 ? "disabled" : ""} aria-label="Move down">↓</button></li>`).join("")}</ol>`
          : `<p class="muted small">No interests selected yet.</p>`}
        <div id="err" class="field-error"></div>`;
    } else if (step === 3) {
      body = `<p class="notice info">These answers are optional. Each answered preference is compared with the career's work profile
          (match 1, partial 0.5, mismatch 0) and the average makes up <b>10% of the score multiplier</b>. Skip them all and the score uses skills and interests only.</p>` +
        Object.entries(m.work_preferences).map(([k, opts]) => `
          <div class="field"><label for="wp-${k}">${PREF_NAMES[k] || k}</label>
            <select id="wp-${k}" data-pref="${k}"><option value="">Prefer not to say</option>
              ${opts.map(o => `<option value="${o}" ${f.work_preferences[k] === o ? "selected" : ""}>${o[0].toUpperCase() + o.slice(1)}</option>`).join("")}
            </select></div>`).join("");
    } else {
      body = `<table><tbody>
          <tr><th>Display name</th><td>${esc(f.display_name || "Anonymous")}</td></tr>
          ${m.skills.map(s => `<tr><th>${esc(s.label)}</th><td class="num">${f.skills[s.id]}</td></tr>`).join("")}
          <tr><th>Interests (ranked)</th><td>${f.interests.map((i, k) => `${k + 1}. ${esc(interestLabel(i))}`).join("<br>")}</td></tr>
          <tr><th>Work preferences</th><td>${prefText(f.work_preferences) || "—"}</td></tr>
        </tbody></table><div id="err" class="field-error"></div>`;
    }
    render(`<div class="card" style="max-width:760px">
      <div class="progress" aria-hidden="true">${progress}</div>
      <p class="muted small">Step ${step + 1} of ${STEP_TITLES.length}</p>
      <h1>${STEP_TITLES[step]}</h1>
      <form id="aform" novalidate>${body}
        <div class="row" style="margin-top:20px">
          ${step > 0 ? `<button type="button" class="btn secondary" id="prev">Previous</button>` : ""}
          <span class="spacer"></span>
          <button type="submit" class="btn">${step === 4 ? "Generate My Career Recommendations" : "Next"}</button>
        </div></form></div>`);

    const form = document.getElementById("aform");
    form.addEventListener("input", e => {
      const t = e.target;
      if (t.dataset.skill) { f.skills[t.dataset.skill] = Number(t.value); document.getElementById(`out-${t.dataset.skill}`).textContent = t.value; }
      if (t.id === "dn") f.display_name = t.value;
      if (t.dataset.pref) f.work_preferences[t.dataset.pref] = t.value;
    });
    form.addEventListener("change", e => {
      const id = e.target.dataset.interest;
      if (!id) return;
      f.interests = e.target.checked ? [...f.interests, id] : f.interests.filter(x => x !== id);
      assess();
    });
    form.addEventListener("click", e => {
      const b = e.target.closest("[data-move]");
      if (!b) return;
      const k = Number(b.dataset.move), j = k + Number(b.dataset.dir);
      [f.interests[k], f.interests[j]] = [f.interests[j], f.interests[k]];
      assess();
    });
    document.getElementById("prev")?.addEventListener("click", () => { state.step--; assess(); });
    form.addEventListener("submit", e => {
      e.preventDefault();
      const err = validateStep(step);
      if (err) { const el = document.getElementById("err"); if (el) el.textContent = err; return; }
      if (step < 4) { state.step++; assess(); } else location.hash = "#/processing";
    });
  }

  function validateStep(step) {
    const f = state.form;
    if (step === 1 || step === 4) {
      for (const s of state.meta.skills) {
        const v = f.skills[s.id];
        if (!Number.isFinite(v) || v < 0 || v > 100) return `${s.label} must be between 0 and 100.`;
      }
    }
    if ((step === 2 || step === 4) && f.interests.length === 0) return "Please select at least one interest.";
    return null;
  }

  function payloadFromForm() {
    const f = state.form;
    return {
      display_name: f.display_name, skills: { ...f.skills }, interests: [...f.interests],
      work_preferences: Object.fromEntries(Object.entries(f.work_preferences).filter(([, v]) => v))
    };
  }

  async function runAssessment(payload) {
    const r = await api("/api/assess", { method: "POST", body: JSON.stringify(payload) });
    r.input = payload;
    r.created_at = new Date().toISOString();
    return r;
  }

  async function processing() {
    if (!state.meta) return go("#/instructions");
    render(`<div class="card" style="max-width:560px;margin:40px auto;text-align:center">
      <div class="spinner" role="status" aria-label="Processing"></div>
      <h2 id="pmsg">Evaluating your profile…</h2>
      <p class="muted small">The server fuzzifies your ratings and runs 18 Mamdani inference systems with the <b>saved</b> GA-optimised weights.
        No genetic algorithm runs during your request.</p>
      <div id="perr"></div></div>`);
    try {
      const r = await runAssessment(payloadFromForm());
      state.result = r; saveResult(r); addHistory(r);
      location.hash = "#/results";
    } catch (e) {
      const fields = e.body?.fields ? `<ul>${Object.entries(e.body.fields).map(([k, v]) => `<li>${esc(k)}: ${esc(v)}</li>`).join("")}</ul>` : "";
      document.querySelector(".spinner")?.remove();
      document.getElementById("pmsg").textContent = "We couldn't process your assessment";
      document.getElementById("perr").innerHTML = `<div class="error-box" style="text-align:left">
        ${e.status ? esc(e.message) : "The server could not be reached. Check that the backend is running (python app.py)."}${fields}</div>
        <div class="row" style="justify-content:center;margin-top:16px">
          <button class="btn" id="retry">Retry</button><a class="btn secondary" href="#/assess">Edit answers</a></div>`;
      document.getElementById("retry").onclick = processing;
    }
  }

  function summarySource(sm) {
    if (sm.source === "groq") return `<span class="pill accent">AI-written · Groq (${esc(sm.model)})</span>`;
    return `<span class="pill">Template summary</span>`;
  }

  async function loadSummary(r) {
    const box = document.getElementById("summary-box");
    if (!box || !r.input || r.summary?.source === "groq") return;
    if (!(r.llm_configured ?? state.meta?.llm_configured)) return;
    box.querySelector(".summary-src").innerHTML = `<span class="pill">Asking the LLM…</span>`;
    try {
      const sm = await api("/api/summary", { method: "POST", body: JSON.stringify(r.input) });
      r.summary = sm; saveResult(r); updateHistorySummary(r);
      if (!document.body.contains(box)) return;
      box.querySelector(".summary-text").textContent = sm.text;
      box.querySelector(".summary-src").innerHTML = summarySource(sm) + (sm.reason ? ` <span class="muted small">${esc(sm.reason)}</span>` : "");
    } catch {
      if (document.body.contains(box)) box.querySelector(".summary-src").innerHTML = `${summarySource(r.summary)} <span class="muted small">The LLM request failed, so the template is shown.</span>`;
    }
  }

  function results() {
    const r = state.result;
    if (!r) return render(noResult());
    const recs = r.recommendations;
    const prefs = prefText(r.work_preferences);
    const hasPrefs = recs[0].breakdown?.preference_points != null;
    render(`
      <div class="row no-print" style="margin-bottom:12px"><span class="pill accent">${esc(r.weights_source)}</span>
        ${r.created_at ? `<span class="muted small">${esc(fmtDate(r.created_at))}</span>` : ""}</div>
      <h1>${r.display_name ? `${esc(r.display_name)}, here are your top matches` : "Your top career matches"}</h1>
      <p class="muted">Ranked by suitability match. <code>${esc(r.score_formula)}</code>. The score can never exceed the fuzzy suitability s. Fit bands: Strong ≥ 70, Good 55–70, Moderate 40–55, Weak &lt; 40.
        This is not a probability of success.</p>
      <div class="card summary" id="summary-box">
        <div class="row"><h3 style="margin:0">Summary</h3><span class="spacer"></span><span class="summary-src">${summarySource(r.summary)}</span></div>
        <p class="summary-text">${esc(r.summary.text)}</p>
      </div>
      <div class="grid grid-2" style="margin:20px 0">
        <div class="card"><h3>Score breakdown, top 5</h3><div class="chart-box"><canvas id="cmp"></canvas></div></div>
        <div class="card"><h3>All 18 careers</h3><div class="chart-box"><canvas id="all"></canvas></div></div>
      </div>
      ${prefs ? `<p class="notice info"><b>Your work preferences</b> (10% of the score multiplier): ${prefs}</p>`
        : `<p class="notice info">No work preferences were given, so the score uses skills and interests only.</p>`}
      <div class="stack" style="margin-top:16px">${recs.map(card).join("")}</div>
      <div class="row no-print" style="margin-top:24px">
        <a class="btn" href="#/report">Download PDF report</a>
        <a class="btn secondary" href="#/compare/${recs.slice(0, 3).map(x => x.id).join("/")}">Compare top 3 careers</a>
        <a class="btn secondary" href="#/people/current">Compare with another person</a>
        <a class="btn secondary" href="#/analysis">Soft Computing analysis</a>
        <a class="btn secondary" href="#/history">History</a>
        <a class="btn ghost" href="#/assess">Retake assessment</a>
      </div>
      <p class="notice" style="margin-top:20px">${esc(r.disclaimer)}</p>`);
    const ds = [
      { label: "Base (s × 0.65)", data: recs.map(x => x.breakdown.fis_points), backgroundColor: COLORS.brand },
      { label: "Interest part", data: recs.map(x => x.breakdown.interest_points), backgroundColor: COLORS.accent }];
    if (hasPrefs) ds.push({ label: "Preference part", data: recs.map(x => x.breakdown.preference_points), backgroundColor: COLORS.warn });
    chart("cmp", {
      type: "bar", data: { labels: recs.map(x => x.name), datasets: ds },
      options: { indexAxis: "y", scales: { x: { stacked: true, min: 0, max: 100, title: { display: true, text: "Points (0–100)" } }, y: { stacked: true } } }
    });
    const bandColor = { strong: COLORS.accent, good: COLORS.brand, moderate: COLORS.warn, weak: COLORS.grey };
    chart("all", {
      type: "bar", data: {
        labels: r.all_scores.map(x => x.name), datasets: [{
          label: "Score",
          data: r.all_scores.map(x => x.score), backgroundColor: r.all_scores.map(x => bandColor[x.fit.key])
        }]
      },
      options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { min: 0, max: 100 }, y: { ticks: { font: { size: 10 } } } } }
    });
    loadSummary(r);
  }

  function card(x) {
    const b = x.breakdown;
    return `<article class="card rec">
      <div class="rec-head"><div class="rank-badge">${x.rank}</div>
        <div><h2 style="margin:0">${esc(x.name)}</h2><p class="muted" style="margin:.2rem 0 0">${esc(x.description)}</p></div>
        <div class="score"><b>${x.score.toFixed(1)}</b><span class="muted"> / 100</span><div>${bandPill(x.fit)}</div></div></div>
      <div class="bar" aria-hidden="true"><i style="width:${x.score}%"></i></div>
      <p>${esc(x.explanation)}</p>
      <p class="small muted">Breakdown: fuzzy suitability s = ${x.fis_suitability.toFixed(1)} × multiplier ${b.multiplier.toFixed(3)} = ${x.score.toFixed(1)}
        (base ${b.fis_points.toFixed(1)} + interest ${b.interest_points.toFixed(1)}${b.preference_points != null ? ` + preferences ${b.preference_points.toFixed(1)}` : ""} points)</p>
      <div class="grid grid-3">
        <div><h3>Strongest factors</h3><div class="tags">${x.strengths.map(s => `<span class="pill accent">${esc(s)}</span>`).join("") || "—"}</div></div>
        <div><h3>To improve</h3><div class="tags">${x.gaps.slice(0, 3).map(g => `<span class="pill warn">${esc(g.label)} +${g.gap.toFixed(0)}</span>`).join("") || `<span class="muted small">No gaps: you meet every target.</span>`}</div></div>
        <div><h3>Key skills</h3><p class="small muted" style="margin:0">${x.key_skills.map(esc).join(" · ")}</p></div>
      </div>
      <details><summary>Why? Fuzzy rules that fired (FIS suitability ${x.fis_suitability.toFixed(1)}, interest match ${(x.interest_match * 100).toFixed(0)}%${x.preference_match != null ? `, preference match ${(x.preference_match * 100).toFixed(0)}%` : ""})</summary>
        ${x.fired_rules.map(f => `<div class="rule">[${f.strength.toFixed(2)}] ${esc(f.text)}</div>`).join("") || `<p class="muted small">No rule fired for this career.</p>`}
        ${x.preference_detail?.length ? `<p class="small" style="margin-top:8px"><b>Preference check:</b> ${x.preference_detail.map(d => `${esc(d.preference.replace("_", " "))} = ${esc(d.value)} → ${d.match === 1 ? "match" : d.match === 0.5 ? "partial" : "mismatch"}`).join(" · ")}</p>` : ""}
      </details>
      <div class="row no-print"><a class="btn secondary" href="#/career/${x.id}">Career details</a>
        <a class="btn ghost" href="#/gap/${x.id}">Skill gaps and roadmap →</a></div>
    </article>`;
  }

  async function careerPage(id) {
    const c = await api(`/api/careers/${encodeURIComponent(id)}`);
    const rec = state.result?.recommendations.find(x => x.id === id);
    const any = state.result?.all_scores.find(x => x.id === id);
    const wp = c.work_profile;
    render(`<div class="stack">
      <a href="#/results" class="no-print">← Back to results</a>
      <div class="card stack">
        <h1>${esc(c.name)}</h1><p class="muted">${esc(c.description)}</p>
        ${any ? `<p><span class="pill">Your match: ${any.score.toFixed(1)} / 100</span> ${bandPill(any.fit)}</p>` : ""}
        <div class="grid grid-2">
          <div><h3>Typical responsibilities</h3><ul>${c.responsibilities.map(r => `<li>${esc(r)}</li>`).join("")}</ul></div>
          <div><h3>Required skills</h3><ul>${c.key_skills.map(r => `<li>${esc(r)}</li>`).join("")}</ul></div>
        </div>
        <div><h3>Work profile (model assumption)</h3><p class="small">Work type: <b>${wp.work_type.map(esc).join(", ")}</b> · Environment: <b>${esc(wp.environment)}</b>
          · Activities: <b>${wp.activity.map(esc).join(", ")}</b> · Maths demand: <b>${esc(wp.math_demand)}</b></p></div>
        ${rec ? `<div class="grid grid-2">
          <div><h3>Your matching strengths</h3><ul>${rec.skills.filter(s => s.gap <= 0).map(s => `<li>${esc(s.label)}: ${s.student} (target ${s.target})</li>`).join("") || "<li>None yet</li>"}</ul></div>
          <div><h3>Your skill gaps</h3><ul>${rec.gaps.map(g => `<li>${esc(g.label)}: +${g.gap.toFixed(0)} needed</li>`).join("") || "<li>No gaps</li>"}</ul></div></div>` : ""}
        <div><h3>Suggested learning path</h3><ol class="steps">${c.learning.map(l => `<li>${esc(l)}</li>`).join("")}</ol></div>
        <div><h3>Related career paths</h3><div class="tags">${c.related.map(r => `<a class="pill" href="#/career/${r.id}">${esc(r.name)}</a>`).join("")}</div></div>
        <h3>Model target profile</h3>
        <table><thead><tr><th>Skill</th><th class="num">Target</th></tr></thead><tbody>
          ${Object.entries(c.targets).map(([k, v]) => `<tr><td>${esc(label(k))}</td><td class="num">${v}</td></tr>`).join("")}</tbody></table>
        <p class="notice small">${esc(c.note)} Learning suggestions are starting points. No job placement is implied.</p>
        ${rec ? `<a class="btn" href="#/gap/${c.id}">Open skill gaps and roadmap</a>` : ""}
      </div></div>`);
  }

  function roadmapHtml(rec) {
    if (!rec.roadmap?.length) return `<p>You meet every target level for ${esc(rec.name)}. Focus on depth: real projects, a portfolio and mentoring others.</p>`;
    return `<div class="roadmap">${rec.roadmap.map((g, i) => `
      <div class="road-step prio-${g.priority}">
        <div class="road-head"><span class="road-n">${i + 1}</span><b>${esc(g.label)}</b>
          <span class="pill ${g.priority === "high" ? "warn" : ""}">${esc(g.priority)} priority</span>
          <span class="muted small">${g.current} → ${g.target} (gap ${g.gap.toFixed(0)}) · ${esc(g.stage)} stage</span></div>
        <ul>${g.steps.map(s => `<li>${esc(s)}</li>`).join("")}</ul>
      </div>`).join("")}</div>
      <p class="muted small">Priority comes from the gap size (high ≥ 25, medium ≥ 10). The stage comes from your current rating
        (foundation &lt; 40, intermediate &lt; 70, advanced ≥ 70). Steps are generic starting points, not a curriculum.</p>`;
  }

  function gapPage(id) {
    const rec = state.result?.recommendations.find(x => x.id === id);
    if (!rec) return render(noResult());
    const strengths = rec.skills.filter(s => s.gap <= 0);
    render(`<div class="stack">
      <a href="#/results" class="no-print">← Back to results</a>
      <div class="card stack">
        <h1>Skill gaps: ${esc(rec.name)}</h1>
        <p class="muted">Your self-reported level compared with the model's target profile. Target values are model assumptions.</p>
        <div class="chart-box tall"><canvas id="gap"></canvas></div>
        <table><thead><tr><th>Skill</th><th class="num">You</th><th class="num">Target</th><th class="num">Gap</th></tr></thead><tbody>
          ${rec.skills.map(s => `<tr><td>${esc(s.label)}</td><td class="num">${s.student}</td><td class="num">${s.target}</td>
            <td class="num">${s.gap > 0 ? `<span class="pill warn">+${s.gap.toFixed(0)}</span>` : `<span class="pill accent">met</span>`}</td></tr>`).join("")}
        </tbody></table>
        <div><h3>Your learning roadmap</h3>${roadmapHtml(rec)}</div>
        <div class="grid grid-2">
          <div><h3>Existing strengths</h3>${strengths.length ? `<ul>${strengths.map(s => `<li>${esc(s.label)}</li>`).join("")}</ul>` : "<p class='muted'>None of your ratings meet the targets yet.</p>"}</div>
          <div><h3>Career-specific topics</h3><ul>${rec.learning.map(l => `<li>${esc(l)}</li>`).join("")}</ul></div>
        </div>
      </div></div>`);
    chart("gap", {
      type: "radar", data: {
        labels: rec.skills.map(s => s.label), datasets: [
          { label: "You", data: rec.skills.map(s => s.student), borderColor: COLORS.brand, backgroundColor: "rgba(61,78,216,.15)" },
          { label: "Career target", data: rec.skills.map(s => s.target), borderColor: COLORS.warn, backgroundColor: "rgba(217,119,6,.10)", borderDash: [5, 4] }]
      },
      options: { scales: { r: { min: 0, max: 100, ticks: { stepSize: 25 } } } }
    });
  }

  async function comparePage(...picked) {
    const r = state.result;
    if (!r) return render(noResult());
    const list = await api("/api/careers");
    const ids = list.careers.map(c => c.id);
    // up to three distinct, valid careers; default is the top three
    let sel = [...new Set(picked.filter(id => ids.includes(id)))].slice(0, 3);
    if (!picked.some(Boolean)) sel = r.all_scores.slice(0, 3).map(x => x.id);
    for (const x of r.all_scores) { if (sel.length >= 2) break; if (!sel.includes(x.id)) sel.push(x.id); }
    const cs = await Promise.all(sel.map(id => api(`/api/careers/${encodeURIComponent(id)}`)));
    const ss = sel.map(id => r.all_scores.find(x => x.id === id));
    const you = state.meta.skills.map(s => r.analysis.inputs[s.id]);
    const palette = [COLORS.accent, COLORS.rose, COLORS.warn];
    const opt = (cur, allowNone) => (allowNone ? `<option value="" ${cur ? "" : "selected"}>None (compare two)</option>` : "") +
      list.careers.map(c => `<option value="${c.id}" ${c.id === cur ? "selected" : ""}>${esc(c.name)}</option>`).join("");
    const row = (k, vals) => `<tr><td>${k}</td>${vals.map(v => `<td class="num">${v}</td>`).join("")}</tr>`;
    const pct = v => v == null ? "—" : `${(v * 100).toFixed(0)}%`;
    const gapOf = (c, i) => c.targets[state.meta.skills[i].id] - you[i];
    const cols = first => `<thead><tr><th>${first}</th>${cs.map(c => `<th class="num">${esc(c.name)}</th>`).join("")}</tr></thead>`;
    render(`<div class="stack">
      <a href="#/results" class="no-print">← Back to results</a>
      <h1>Compare ${cs.length === 3 ? "three" : "two"} careers</h1>
      <div class="card row no-print">
        <label class="grow">Career A <select class="csel">${opt(sel[0])}</select></label>
        <label class="grow">Career B <select class="csel">${opt(sel[1])}</select></label>
        <label class="grow">Career C <select class="csel">${opt(sel[2], true)}</select></label>
      </div>
      <div class="grid grid-2">
        <div class="card"><h3>Your skills vs the targets</h3><div class="chart-box tall"><canvas id="radar"></canvas></div></div>
        <div class="card table-wrap"><h3>Score comparison</h3>
          <table>${cols("")}<tbody>
            ${row("Final score", ss.map(s => `<b>${s.score.toFixed(1)}</b>`))}
            ${row("Fit band", ss.map(s => bandPill(s.fit)))}
            ${row("Rank (of 18)", ss.map(s => r.all_scores.indexOf(s) + 1))}
            ${row("Fuzzy suitability s", ss.map(s => s.fis_suitability.toFixed(1)))}
            ${row("Interest match I", ss.map(s => pct(s.interest_match)))}
            ${row("Preference match W", ss.map(s => pct(s.preference_match)))}
          </tbody></table>
          <h3 style="margin-top:16px">Gap per skill (target − you)</h3>
          <table>${cols("Skill")}<tbody>
            ${state.meta.skills.map((s, i) => row(esc(s.label), cs.map(c => {
      const g = gapOf(c, i); return g > 0 ? `<span class="pill warn">+${g.toFixed(0)}</span>` : `<span class="pill accent">met</span>`;
    }))).join("")}
          </tbody></table>
          <p class="small muted">Total gap: ${cs.map(c => `${esc(c.name)} ${state.meta.skills.reduce((t, _, i) => t + Math.max(0, gapOf(c, i)), 0).toFixed(0)} points`).join(" · ")}.</p>
        </div>
      </div>
      <p class="notice small">Targets are model assumptions, not industry standards.</p>
    </div>`);
    app.querySelectorAll(".csel").forEach(s => s.onchange = () =>
      go(`#/compare/${[...app.querySelectorAll(".csel")].map(x => x.value).filter(Boolean).map(encodeURIComponent).join("/")}`));
    chart("radar", {
      type: "radar", data: {
        labels: state.meta.skills.map(s => s.label), datasets: [
          { label: "You", data: you, borderColor: COLORS.brand, backgroundColor: "rgba(61,78,216,.18)" },
          ...cs.map((c, k) => ({
            label: c.name, data: state.meta.skills.map(s => c.targets[s.id]), borderColor: palette[k],
            backgroundColor: "transparent", borderDash: [[6, 4], [2, 3], [10, 3]][k]
          }))]
      },
      options: { scales: { r: { min: 0, max: 100, ticks: { stepSize: 25 } } } }
    });
  }

  // ---------- compare two people (current result + this browser's history) ----------
  function peopleList() {
    const cur = state.result;
    const people = cur ? [{ id: "current", label: `Current result · ${cur.display_name || "Anonymous"}`, result: cur }] : [];
    for (const e of readHistory()) {
      if (cur && e.result.created_at && e.result.created_at === cur.created_at) continue;  // same assessment as "current"
      people.push({ id: e.id, label: `${e.result.display_name || "Anonymous"} · ${fmtDate(e.at)}${e.tag ? ` (${e.tag})` : ""}`, result: e.result });
    }
    return people;
  }

  function peoplePage(a, b) {
    const people = peopleList();
    if (people.length < 2) {
      return render(`<div class="stack">
        <h1>Compare with another person</h1>
        <div class="error-box"><b>No other data available to compare.</b>
          ${people.length === 0 ? "There is no current result and no saved history in this browser yet."
          : "Only one assessment is available, so there is nobody else to compare it with."}
          Take another assessment or load the demo history, then try again.</div>
        <div class="row"><a class="btn" href="#/instructions">Start Career Assessment</a>
          <a class="btn secondary" href="#/history">Open History (load demo profiles)</a></div></div>`);
    }
    const find = id => people.find(p => p.id === id);
    const A = find(a) || people[0];
    const B = (b !== A.id && find(b)) || people.find(p => p !== A);
    const ra = A.result, rb = B.result, skills = state.meta.skills;
    const rawA = ra.display_name || "Person A", rawB = rb.display_name || "Person B";
    const na = esc(rawA), nb = esc(rawB);
    const opt = cur => people.map(p => `<option value="${esc(p.id)}" ${p.id === cur ? "selected" : ""}>${esc(p.label)}</option>`).join("");
    const topIds = r => r.recommendations.map(x => x.id);
    const shared = topIds(ra).filter(id => topIds(rb).includes(id)).map(id => ra.recommendations.find(x => x.id === id).name);
    const interests = r => (r.interests || r.input?.interests || []).map(i => esc(interestLabel(i))).join(", ") || "—";
    const topList = r => `<ol>${r.recommendations.map(x => `<li>${esc(x.name)} <b>${x.score.toFixed(1)}</b> ${bandPill(x.fit)}</li>`).join("")}</ol>`;
    const person = (n, r) => `<div class="card"><h3>${n}</h3>${topList(r)}
      <p class="small"><b>Interests:</b> ${interests(r)}<br><b>Work preferences:</b> ${prefText(r.work_preferences) || "not given"}</p></div>`;
    const scoreB = Object.fromEntries(rb.all_scores.map(x => [x.id, x.score]));
    render(`<div class="stack">
      <a href="#/history" class="no-print">← Back to history</a>
      <h1>Compare with another person</h1>
      <p class="muted">Both people are scored by the same engine, so the numbers are directly comparable. Data comes from this browser's history only.</p>
      <div class="card row no-print">
        <label class="grow">Person A <select id="pa">${opt(A.id)}</select></label>
        <label class="grow">Person B <select id="pb">${opt(B.id)}</select></label>
      </div>
      <p class="notice info">${shared.length ? `<b>Shared top-5 careers:</b> ${shared.map(esc).join(", ")}.` : "These two people have no careers in common in their top 5."}
        Top match: ${na} → <b>${esc(ra.recommendations[0].name)}</b>, ${nb} → <b>${esc(rb.recommendations[0].name)}</b>.</p>
      <div class="grid grid-2">${person(na, ra)}${person(nb, rb)}</div>
      <div class="grid grid-2">
        <div class="card"><h3>Skill profiles</h3><div class="chart-box tall"><canvas id="pradar"></canvas></div></div>
        <div class="card table-wrap"><h3>Skill ratings</h3>
          <table><thead><tr><th>Skill</th><th class="num">${na}</th><th class="num">${nb}</th><th class="num">Difference</th></tr></thead><tbody>
          ${skills.map(s => {
      const x = ra.analysis.inputs[s.id], y = rb.analysis.inputs[s.id], d = x - y;
      return `<tr><td>${esc(s.label)}</td><td class="num">${x}</td><td class="num">${y}</td><td class="num">${d === 0 ? "=" : `${d > 0 ? "A" : "B"} +${Math.abs(d)}`}</td></tr>`;
    }).join("")}</tbody></table></div>
      </div>
      <div class="card"><h3>Score for every career</h3><div class="chart-box tall"><canvas id="pall"></canvas></div></div>
      <p class="notice small">For exploration only. A higher score means a closer match to the model's assumptions, not a better person or a guaranteed outcome.</p>
    </div>`);
    const pick = () => go(`#/people/${encodeURIComponent(document.getElementById("pa").value)}/${encodeURIComponent(document.getElementById("pb").value)}`);
    document.getElementById("pa").onchange = pick;
    document.getElementById("pb").onchange = pick;
    chart("pradar", {
      type: "radar", data: {
        labels: skills.map(s => s.label), datasets: [
          { label: rawA, data: skills.map(s => ra.analysis.inputs[s.id]), borderColor: COLORS.brand, backgroundColor: "rgba(61,78,216,.15)" },
          { label: rawB, data: skills.map(s => rb.analysis.inputs[s.id]), borderColor: COLORS.rose, backgroundColor: "rgba(192,57,107,.10)", borderDash: [5, 4] }]
      },
      options: { scales: { r: { min: 0, max: 100, ticks: { stepSize: 25 } } } }
    });
    chart("pall", {
      type: "bar", data: {
        labels: ra.all_scores.map(x => x.name), datasets: [
          { label: rawA, data: ra.all_scores.map(x => x.score), backgroundColor: COLORS.brand },
          { label: rawB, data: ra.all_scores.map(x => scoreB[x.id] ?? null), backgroundColor: COLORS.rose }]
      },
      options: { indexAxis: "y", scales: { x: { min: 0, max: 100 }, y: { ticks: { font: { size: 10 } } } } }
    });
  }

  function reportPage() {
    const r = state.result;
    if (!r) return render(noResult());
    const recs = r.recommendations;
    render(`<div class="report">
      <div class="row no-print" style="margin-bottom:16px">
        <a href="#/results">← Back to results</a><span class="spacer"></span>
        <button class="btn" id="print">Print / Save as PDF</button></div>
      <p class="muted small no-print">In the print dialog, choose "Save as PDF" as the destination.</p>
      <header class="report-head">
        <div><h1 style="margin:0">Career Recommendation Report</h1>
          <p class="muted" style="margin:.2rem 0 0">CareerCompass · Mamdani fuzzy inference + genetic algorithm</p></div>
        <div class="small" style="text-align:right">${r.display_name ? `<b>${esc(r.display_name)}</b><br>` : ""}${esc(fmtDate(r.created_at || new Date().toISOString()))}</div>
      </header>
      <section><h2>Summary</h2><p>${esc(r.summary.text)}</p><p class="small muted">${r.summary.source === "groq" ? `AI-written by Groq (${esc(r.summary.model)}) from the computed scores.` : "Template summary generated from the computed scores."}</p></section>
      <section><h2>Your inputs</h2>
        <table><tbody>
          ${state.meta.skills.map(s => `<tr><td>${esc(s.label)}</td><td class="num">${r.analysis.inputs[s.id]}</td></tr>`).join("")}
          <tr><td>Interests (ranked)</td><td class="num">${(r.interests || r.input?.interests || []).map((i, k) => `${k + 1}. ${esc(interestLabel(i))}`).join(", ")}</td></tr>
          <tr><td>Work preferences</td><td class="num">${prefText(r.work_preferences) || "not given"}</td></tr>
        </tbody></table></section>
      <section><h2>Top 5 careers</h2>
        <table><thead><tr><th>#</th><th>Career</th><th class="num">Score</th><th>Fit</th><th class="num">Fuzzy s</th><th class="num">Interest</th><th class="num">Pref.</th></tr></thead><tbody>
          ${recs.map(x => `<tr><td>${x.rank}</td><td>${esc(x.name)}</td><td class="num"><b>${x.score.toFixed(1)}</b></td><td>${bandPill(x.fit)}</td>
            <td class="num">${x.fis_suitability.toFixed(1)}</td><td class="num">${(x.interest_match * 100).toFixed(0)}%</td>
            <td class="num">${x.preference_match == null ? "—" : (x.preference_match * 100).toFixed(0) + "%"}</td></tr>`).join("")}
        </tbody></table>
        <p class="small muted">${esc(r.score_formula)}</p></section>
      ${recs.slice(0, 3).map(x => `<section class="report-career"><h2>${x.rank}. ${esc(x.name)} ${bandPill(x.fit)}</h2>
        <p>${esc(x.explanation)}</p>
        <h3>Learning roadmap</h3>${roadmapHtml(x)}</section>`).join("")}
      <section><h2>How this was computed</h2>
        <p class="small">Each rating is fuzzified into Low, Medium and High. 18 Mamdani fuzzy systems (11 rules each) combine the memberships,
          scaled by GA-optimised weights, into a suitability from Very Low to Very High, and centroid defuzzification gives s on 0–100.
          The final score multiplies s by 0.65 + 0.25 × interest match + 0.10 × work-preference match, so it never exceeds s.</p></section>
      <p class="notice small">${esc(r.disclaimer)}</p>
    </div>`);
    document.getElementById("print").onclick = () => window.print();
  }

  async function historyPage() {
    const list = readHistory();
    render(`<div class="stack">
      <h1>History</h1>
      <p class="muted">Past assessments saved in <b>this browser only</b> (latest ${HISTORY_MAX}). Nothing is stored on the server,
        so another computer starts with an empty history.</p>
      <div class="row no-print">
        <button class="btn secondary" id="demo">Load demo history (5 profiles)</button>
        <a class="btn secondary" href="#/people">Compare two people</a>
        <button class="btn secondary" id="export" ${list.length ? "" : "disabled"}>Export JSON</button>
        <span class="spacer"></span>
        <button class="btn ghost danger" id="clear" ${list.length ? "" : "disabled"}>Clear all</button>
      </div>
      <div id="hmsg"></div>
      ${list.length ? `<div class="card table-wrap"><table class="history"><thead><tr><th>When</th><th>Name</th><th>Top career</th><th class="num">Score</th><th>Fit</th><th>Next two</th><th></th></tr></thead><tbody>
        ${list.map(e => {
      const t = e.result.recommendations; return `<tr>
          <td>${esc(fmtDate(e.at))}${e.tag ? ` <span class="pill">${esc(e.tag)}</span>` : ""}</td>
          <td>${esc(e.result.display_name || "Anonymous")}</td>
          <td><b>${esc(t[0].name)}</b></td><td class="num">${t[0].score.toFixed(1)}</td><td>${bandPill(t[0].fit)}</td>
          <td class="small">${t.slice(1, 3).map(x => `${esc(x.name)} (${x.score.toFixed(0)})`).join(", ")}</td>
          <td class="row" style="flex-wrap:nowrap"><button class="btn secondary" data-view="${e.id}">View</button>
            <a class="btn secondary" href="#/people/${state.result ? "current" : ""}/${encodeURIComponent(e.id)}">Compare</a>
            <button class="btn ghost" data-del="${e.id}" aria-label="Delete">Delete</button></td></tr>`;
    }).join("")}
      </tbody></table></div>
      <div class="card"><h3>Top career across saved results</h3><div class="chart-box"><canvas id="hchart"></canvas></div></div>`
        : `<div class="card"><p>No saved results yet. Take the assessment, or load the demo history to see how this page works.</p>
          <a class="btn" href="#/instructions">Start Career Assessment</a></div>`}
    </div>`);
    document.getElementById("demo").onclick = loadDemoHistory;
    document.getElementById("clear").onclick = () => { if (confirm("Delete every saved result in this browser?")) { writeHistory([]); historyPage(); } };
    document.getElementById("export").onclick = () => {
      const blob = new Blob([JSON.stringify(list, null, 1)], { type: "application/json" });
      const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "careercompass-history.json"; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    };
    app.querySelectorAll("[data-view]").forEach(b => b.onclick = () => {
      const e = list.find(x => x.id === b.dataset.view);
      state.result = e.result; saveResult(e.result); go("#/results");
    });
    app.querySelectorAll("[data-del]").forEach(b => b.onclick = () => { writeHistory(list.filter(x => x.id !== b.dataset.del)); historyPage(); });
    if (list.length) {
      const counts = {};
      list.forEach(e => { const n = e.result.recommendations[0].name; counts[n] = (counts[n] || 0) + 1; });
      const names = Object.keys(counts).sort((x, y) => counts[y] - counts[x]);
      chart("hchart", {
        type: "bar", data: { labels: names, datasets: [{ label: "Times ranked #1", data: names.map(n => counts[n]), backgroundColor: COLORS.brand }] },
        options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { ticks: { stepSize: 1 }, title: { display: true, text: "Saved results" } } } }
      });
    }
  }

  async function loadDemoHistory() {
    const msg = document.getElementById("hmsg");
    msg.innerHTML = `<p class="notice info">Running the five demonstration profiles through the real engine…</p>`;
    try {
      const { profiles } = await api("/api/demo-profiles");
      for (const p of profiles.slice().reverse()) {
        const r = await runAssessment({ display_name: p.name, skills: p.skills, interests: p.interests, work_preferences: p.work_preferences });
        addHistory(r, "demo");
      }
      historyPage();
    } catch (e) {
      msg.innerHTML = `<div class="error-box">Could not load the demo profiles: ${esc(e.message)}</div>`;
    }
  }

  function about() {
    render(`<div class="card stack" style="max-width:860px">
      <h1>About this project</h1>
      <h3>Problem statement</h3>
      <p>Students often find it hard to identify careers that match their abilities and interests. Ability is also gradual, not simply "good" or "bad".
        This prototype uses fuzzy reasoning to handle imprecise self-ratings, and an evolutionary algorithm to tune how much each skill matters.</p>
      <h3>Objectives</h3>
      <ul><li>Represent gradual skill levels with fuzzy membership functions.</li><li>Evaluate career suitability with Mamdani IF–THEN rules.</li>
        <li>Optimise the input weights with a Genetic Algorithm against a documented fitness function, with a separate validation and test set.</li>
        <li>Give five ranked, explained recommendations with fit bands, skill gaps and a learning roadmap.</li></ul>
      <h3>Technologies</h3>
      <p>HTML5, CSS3, vanilla JavaScript and Chart.js for the frontend. Python, Flask and NumPy for the backend. The Fuzzy Logic and Genetic Algorithm are custom, documented implementations.
        An optional LLM (Groq) only rewrites the computed results as a readable paragraph; it never scores anything.</p>
      <h3>Team members and responsibilities</h3>
      <table><tbody>
        <tr><td>Chandraprakash Verma</td><td>Fuzzy Logic module, rules, membership graphs</td></tr>
        <tr><td>Smit Joshi</td><td>Genetic Algorithm, fitness function, experiment</td></tr>
        <tr><td>Shrey Patel</td><td>Career dataset, reference profiles, ranking and skill gaps</td></tr>
        <tr><td>Krunal Patil</td><td>Frontend and integration</td></tr></tbody></table>
      <h3>Limitations</h3>
      <ul><li>Skill ratings are self-reported and subjective.</li><li>Career target and work profiles are simplified model assumptions, not industry standards.</li>
        <li>The GA was trained and evaluated on 500 synthetic profiles labelled by our own expert function. The evaluation shows consistency with those assumptions, not real-world accuracy.</li>
        <li>On the untouched test split, the GA weights did not beat equal weights by a statistically meaningful margin (see Soft Computing → Results).</li>
        <li>History is stored only in this browser. No data is stored on the server.</li>
        <li>This tool does not replace professional career counselling.</li></ul>
      <h3>Future scope</h3>
      <p>Validate the career profiles with real counsellors, collect voluntary and anonymised survey data as a real test set, tune the membership functions with the GA as well, and add server-side history with consent.</p>
      <h3>References</h3>
      <ul class="small"><li>Zadeh, L. A. (1965). Fuzzy sets. <i>Information and Control</i>, 8(3), 338–353.</li>
        <li>Mamdani, E. H., &amp; Assilian, S. (1975). An experiment in linguistic synthesis with a fuzzy logic controller. <i>Int. J. Man-Machine Studies</i>, 7(1), 1–13.</li>
        <li>Ruspini, E. H. (1969). A new approach to clustering. <i>Information and Control</i>, 15(1), 22–32 (fuzzy partitions).</li>
        <li>Holland, J. H. (1975). <i>Adaptation in Natural and Artificial Systems</i>. University of Michigan Press.</li>
        <li>Eshelman, L. J., &amp; Schaffer, J. D. (1993). Real-coded genetic algorithms and interval-schemata (BLX-α). <i>Foundations of Genetic Algorithms 2</i>.</li>
        <li>Järvelin, K., &amp; Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques (NDCG). <i>ACM TOIS</i>, 20(4).</li>
        <li>Efron, B., &amp; Tibshirani, R. (1993). <i>An Introduction to the Bootstrap</i>. Chapman &amp; Hall.</li></ul>
    </div>`);
  }

  // ---------- shared with sc.js ----------
  window.CC = { state, api, esc, chart, render, label, COLORS, SETS, OUT_SETS, ensureMeta, ensureAnalysis, bandPill };

  // ---------- router ----------
  function go(h) { location.hash = h; }
  async function route() {
    const [, page = "", arg, arg2, arg3] = (location.hash || "#/").split("/");
    document.querySelectorAll(".nav a").forEach(a => a.classList.toggle("active", a.getAttribute("href") === `#/${page}`));
    document.getElementById("nav").classList.remove("open");
    try {
      await ensureMeta();
      switch (page) {
        case "": return home();
        case "instructions": return instructions();
        case "assess": return assess();
        case "processing": return processing();
        case "results": return results();
        case "career": return careerPage(decodeURIComponent(arg || ""));
        case "gap": return gapPage(decodeURIComponent(arg || ""));
        case "compare": return comparePage(...[arg, arg2, arg3].map(x => decodeURIComponent(x || "")));
        case "people": return peoplePage(decodeURIComponent(arg || ""), decodeURIComponent(arg2 || ""));
        case "report": return reportPage();
        case "history": return historyPage();
        case "analysis": return window.CCPages.analysis(arg);
        case "about": return about();
        default: return render(`<div class="card"><h2>Page not found</h2><a href="#/">Go home</a></div>`);
      }
    } catch (e) {
      render(`<div class="error-box"><b>Could not load this page.</b> ${e.status ? esc(e.message) : "The server could not be reached. Is <code>python app.py</code> running?"}
        <div class="row" style="margin-top:10px"><button class="btn" id="retry">Retry</button></div></div>`);
      document.getElementById("retry").onclick = route;
    }
  }
  document.querySelector(".nav-toggle").addEventListener("click", e => {
    const nav = document.getElementById("nav");
    const open = nav.classList.toggle("open");
    e.currentTarget.setAttribute("aria-expanded", String(open));
  });
  window.addEventListener("hashchange", route);
  route();
})();
