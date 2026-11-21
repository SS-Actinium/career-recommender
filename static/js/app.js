/* CareerCompass frontend: vanilla JS single-page app with hash routing.
   All scoring happens on the Flask backend. This file only collects input and renders results. */
(() => {
  "use strict";

  const app = document.getElementById("app");
  const charts = [];
  const SETS = ["Low", "Medium", "High"];
  const COLORS = { brand: "#3d4ed8", accent: "#0f9d8a", warn: "#d97706", grey: "#9aa1b2",
                   low: "#d97706", medium: "#3d4ed8", high: "#0f9d8a" };

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
    const el = document.getElementById(id);
    if (!el || typeof Chart === "undefined") return;
    config.options = { responsive: true, maintainAspectRatio: false, animation: false, ...config.options };
    charts.push(new Chart(el, config));
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
    return `<div class="card stack"><h2>No results yet</h2><p class="muted">Complete the assessment to see your recommendations.</p>
      <a class="btn" href="#/instructions">Start Career Assessment</a></div>`;
  }

  // ---------- pages ----------
  function home() {
    render(`
      <section class="hero">
        <div>
          <span class="pill">Soft Computing · MCA project prototype</span>
          <h1>Find career paths that fit how you think and what you enjoy.</h1>
          <p class="lead">Rate six skills, choose your interests, and get five ranked career suggestions with clear reasons and skill gaps.
            The backend uses a Mamdani fuzzy inference system, with input weights tuned by a genetic algorithm.</p>
          <div class="row"><a class="btn" href="#/instructions">Start Career Assessment</a>
            <a class="btn secondary" href="#/analysis">See the algorithms</a></div>
        </div>
        <div class="hero-visual card">
          <h3>How it works</h3>
          <ol class="steps">
            <li><b>Fuzzification.</b> Each 0–100 rating becomes degrees of Low, Medium and High.</li>
            <li><b>Fuzzy rules.</b> 18 Mamdani systems (one per career) evaluate IF–THEN rules.</li>
            <li><b>GA weights.</b> A genetic algorithm tuned how much each skill matters.</li>
            <li><b>Ranking.</b> 70% fuzzy suitability and 30% interest match give the final score.</li>
          </ol>
        </div>
      </section>
      <section class="grid grid-3">
        <div class="card"><h3>Fuzzy Logic</h3><p class="muted">A rating of 65 is not just "medium" or just "high". It is partly both, and the rules reason with those degrees.</p></div>
        <div class="card"><h3>Genetic Algorithm</h3><p class="muted">The GA evolved 30 candidate weight vectors over 40 generations, using selection, crossover and mutation.</p></div>
        <div class="card"><h3>Explainable results</h3><p class="muted">Each suggestion shows the rules that fired, your strongest factors, and the skills to build next.</p></div>
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
          <li><b>Work preferences (optional).</b> These are shown as context only and do not change the score.</li>
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
          The latest result stays in this browser tab until you close it.</p>
        <div class="row"><a class="btn" href="#/assess">Begin assessment</a></div>
      </div>`);
  }

  const STEP_TITLES = ["Profile", "Skill ratings", "Interests", "Work preferences", "Review"];

  function assess() {
    const f = state.form, m = state.meta, step = state.step;
    const progress = STEP_TITLES.map((_, i) => `<span class="${i <= step ? "done" : ""}"></span>`).join("");
    let body = "";
    if (step === 0) {
      body = `<div class="field"><label for="dn">Display name (optional)</label>
        <input id="dn" type="text" maxlength="40" value="${esc(f.display_name)}" placeholder="e.g. Student A">
        <p class="muted small">Leave this blank to stay anonymous.</p></div>`;
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
      const names = { work_type: "Preferred work type", environment: "Preferred environment", activity: "Preferred activity", advanced_math: "Willingness to learn advanced mathematics" };
      body = `<p class="notice info">These answers are optional. They appear in your results for context, but they do <b>not</b> change the score.</p>` +
        Object.entries(m.work_preferences).map(([k, opts]) => `
          <div class="field"><label for="wp-${k}">${names[k] || k}</label>
            <select id="wp-${k}" data-pref="${k}"><option value="">Prefer not to say</option>
              ${opts.map(o => `<option value="${o}" ${f.work_preferences[k] === o ? "selected" : ""}>${o[0].toUpperCase() + o.slice(1)}</option>`).join("")}
            </select></div>`).join("");
    } else {
      body = `<table><tbody>
          <tr><th>Display name</th><td>${esc(f.display_name || "Anonymous")}</td></tr>
          ${m.skills.map(s => `<tr><th>${esc(s.label)}</th><td class="num">${f.skills[s.id]}</td></tr>`).join("")}
          <tr><th>Interests (ranked)</th><td>${f.interests.map((i, k) => `${k + 1}. ${esc(interestLabel(i))}`).join("<br>")}</td></tr>
          <tr><th>Work preferences</th><td>${Object.entries(f.work_preferences).filter(([, v]) => v).map(([k, v]) => `${esc(k.replace("_", " "))}: ${esc(v)}`).join("<br>") || "—"}</td></tr>
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
      if (step < 4) { state.step++; assess(); } else submit();
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

  async function submit() {
    location.hash = "#/processing";
  }

  async function processing() {
    if (!state.meta) return go("#/instructions");
    const f = state.form;
    render(`<div class="card" style="max-width:560px;margin:40px auto;text-align:center">
      <div class="spinner" role="status" aria-label="Processing"></div>
      <h2 id="pmsg">Evaluating your profile…</h2>
      <p class="muted small">The server fuzzifies your ratings and runs 18 Mamdani inference systems with the <b>saved</b> GA-optimised weights.
        No genetic algorithm runs during your request.</p>
      <div id="perr"></div></div>`);
    const payload = { display_name: f.display_name, skills: f.skills, interests: f.interests,
      work_preferences: Object.fromEntries(Object.entries(f.work_preferences).filter(([, v]) => v)) };
    try {
      const r = await api("/api/assess", { method: "POST", body: JSON.stringify(payload) });
      const msg = document.getElementById("pmsg"); if (msg) msg.textContent = "Calculating career matches…";
      state.result = r; saveResult(r);
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

  function results() {
    const r = state.result;
    if (!r) return render(noResult());
    const recs = r.recommendations;
    const prefs = Object.entries(r.work_preferences || {});
    render(`
      <div class="row no-print" style="margin-bottom:12px"><span class="pill accent">${esc(r.weights_source)}</span></div>
      <h1>${r.display_name ? `${esc(r.display_name)}, here are your top matches` : "Your top career matches"}</h1>
      <p class="muted">Ranked by suitability match: 70% fuzzy suitability and 30% interest match. This is not a probability of success.</p>
      <div class="grid grid-2" style="margin:20px 0">
        <div class="card"><h3>Score composition, top 5</h3><div class="chart-box"><canvas id="cmp"></canvas></div></div>
        <div class="card"><h3>All 18 careers</h3><div class="chart-box"><canvas id="all"></canvas></div></div>
      </div>
      ${prefs.length ? `<p class="notice info"><b>Your stated work preferences</b> (context only, not scored): ${prefs.map(([k, v]) => `${esc(k.replace("_", " "))}: <b>${esc(v)}</b>`).join(" · ")}</p>` : ""}
      <div class="stack" style="margin-top:16px">${recs.map(card).join("")}</div>
      <div class="row no-print" style="margin-top:24px">
        <a class="btn" href="#/analysis">View Soft Computing analysis</a>
        <button class="btn secondary" id="print">Print / save report as PDF</button>
        <a class="btn secondary" href="#/assess">Retake assessment</a>
      </div>
      <p class="notice" style="margin-top:20px">${esc(r.disclaimer)}</p>`);
    document.getElementById("print").onclick = () => window.print();
    chart("cmp", { type: "bar", data: { labels: recs.map(x => x.name), datasets: [
        { label: "Fuzzy part (0.7 × s)", data: recs.map(x => +(0.7 * x.fis_suitability).toFixed(1)), backgroundColor: COLORS.brand },
        { label: "Interest part (30 × I)", data: recs.map(x => +(30 * x.interest_match).toFixed(1)), backgroundColor: COLORS.accent }] },
      options: { indexAxis: "y", scales: { x: { stacked: true, max: 100, title: { display: true, text: "Suitability match (0–100)" } }, y: { stacked: true } } } });
    chart("all", { type: "bar", data: { labels: r.all_scores.map(x => x.name), datasets: [{ label: "Score",
        data: r.all_scores.map(x => x.score), backgroundColor: r.all_scores.map((_, i) => i < 5 ? COLORS.brand : COLORS.grey) }] },
      options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { max: 100 } , y: { ticks: { font: { size: 10 } } } } } });
  }

  function card(x) {
    return `<article class="card rec">
      <div class="rec-head"><div class="rank-badge">${x.rank}</div>
        <div><h2 style="margin:0">${esc(x.name)}</h2><p class="muted" style="margin:.2rem 0 0">${esc(x.description)}</p></div>
        <div class="score"><b>${x.score.toFixed(1)}%</b><div class="muted small">suitability match</div></div></div>
      <div class="bar" aria-hidden="true"><i style="width:${x.score}%"></i></div>
      <p>${esc(x.explanation)}</p>
      <div class="grid grid-3">
        <div><h3>Strongest factors</h3><div class="tags">${x.strengths.map(s => `<span class="pill accent">${esc(s)}</span>`).join("") || "—"}</div></div>
        <div><h3>To improve</h3><div class="tags">${x.gaps.slice(0, 3).map(g => `<span class="pill warn">${esc(g.label)} +${g.gap.toFixed(0)}</span>`).join("") || `<span class="muted small">No gaps: you meet every target.</span>`}</div></div>
        <div><h3>Key skills</h3><p class="small muted" style="margin:0">${x.key_skills.map(esc).join(" · ")}</p></div>
      </div>
      <details><summary>Why? Fuzzy rules that fired (FIS suitability ${x.fis_suitability.toFixed(1)}, interest match ${(x.interest_match * 100).toFixed(0)}%)</summary>
        ${x.fired_rules.map(f => `<div class="rule">[${f.strength.toFixed(2)}] ${esc(f.text)}</div>`).join("") || `<p class="muted small">No rule fired for this career.</p>`}
      </details>
      <div class="row no-print"><a class="btn secondary" href="#/career/${x.id}">View career details</a>
        <a class="btn ghost" href="#/gap/${x.id}">Skill gap analysis →</a></div>
    </article>`;
  }

  async function careerPage(id) {
    const c = await api(`/api/careers/${encodeURIComponent(id)}`);
    const rec = state.result?.recommendations.find(x => x.id === id);
    render(`<div class="stack">
      <a href="#/results" class="no-print">← Back to results</a>
      <div class="card stack">
        <h1>${esc(c.name)}</h1><p class="muted">${esc(c.description)}</p>
        ${rec ? `<p><span class="pill">Your match: ${rec.score.toFixed(1)}%</span></p>` : ""}
        <div class="grid grid-2">
          <div><h3>Typical responsibilities</h3><ul>${c.responsibilities.map(r => `<li>${esc(r)}</li>`).join("")}</ul></div>
          <div><h3>Required skills</h3><ul>${c.key_skills.map(r => `<li>${esc(r)}</li>`).join("")}</ul></div>
        </div>
        ${rec ? `<div class="grid grid-2">
          <div><h3>Your matching strengths</h3><ul>${rec.skills.filter(s => s.gap <= 0).map(s => `<li>${esc(s.label)}: ${s.student} (target ${s.target})</li>`).join("") || "<li>None yet</li>"}</ul></div>
          <div><h3>Your skill gaps</h3><ul>${rec.gaps.map(g => `<li>${esc(g.label)}: +${g.gap.toFixed(0)} needed</li>`).join("") || "<li>No gaps</li>"}</ul></div></div>` : ""}
        <div><h3>Suggested learning roadmap</h3><ol class="steps">${c.learning.map(l => `<li>${esc(l)}</li>`).join("")}</ol></div>
        <div><h3>Related career paths</h3><div class="tags">${c.related.map(r => `<a class="pill" href="#/career/${r.id}">${esc(r.name)}</a>`).join("")}</div></div>
        <h3>Model target profile</h3>
        <table><thead><tr><th>Skill</th><th class="num">Target</th></tr></thead><tbody>
          ${Object.entries(c.targets).map(([k, v]) => `<tr><td>${esc(label(k))}</td><td class="num">${v}</td></tr>`).join("")}</tbody></table>
        <p class="notice small">${esc(c.note)} Learning suggestions are starting points. No job placement is implied.</p>
        ${rec ? `<a class="btn" href="#/gap/${c.id}">Open skill gap analysis</a>` : ""}
      </div></div>`);
  }

  function gapPage(id) {
    const rec = state.result?.recommendations.find(x => x.id === id);
    if (!rec) return render(noResult());
    const strengths = rec.skills.filter(s => s.gap <= 0);
    render(`<div class="stack">
      <a href="#/results" class="no-print">← Back to results</a>
      <div class="card stack">
        <h1>Skill gap: ${esc(rec.name)}</h1>
        <p class="muted">Your self-reported level compared with the model's target profile. Target values are model assumptions.</p>
        <div class="chart-box tall"><canvas id="gap"></canvas></div>
        <table><thead><tr><th>Skill</th><th class="num">You</th><th class="num">Target</th><th class="num">Gap</th></tr></thead><tbody>
          ${rec.skills.map(s => `<tr><td>${esc(s.label)}</td><td class="num">${s.student}</td><td class="num">${s.target}</td>
            <td class="num">${s.gap > 0 ? `<span class="pill warn">+${s.gap.toFixed(0)}</span>` : `<span class="pill accent">met</span>`}</td></tr>`).join("")}
        </tbody></table>
        <div class="grid grid-2">
          <div><h3>Priority order for learning</h3>${rec.gaps.length ? `<ol>${rec.gaps.map(g => `<li><b>${esc(g.label)}</b> (gap ${g.gap.toFixed(0)})</li>`).join("")}</ol>` : "<p>You meet every target level. Focus on depth and projects.</p>"}</div>
          <div><h3>Existing strengths</h3>${strengths.length ? `<ul>${strengths.map(s => `<li>${esc(s.label)}</li>`).join("")}</ul>` : "<p class='muted'>None of your ratings meet the targets yet.</p>"}</div>
        </div>
        <div><h3>Suggested improvement topics</h3><ul>${rec.learning.map(l => `<li>${esc(l)}</li>`).join("")}</ul></div>
      </div></div>`);
    chart("gap", { type: "radar", data: { labels: rec.skills.map(s => s.label), datasets: [
        { label: "You", data: rec.skills.map(s => s.student), borderColor: COLORS.brand, backgroundColor: "rgba(61,78,216,.15)" },
        { label: "Career target", data: rec.skills.map(s => s.target), borderColor: COLORS.warn, backgroundColor: "rgba(217,119,6,.10)", borderDash: [5, 4] }] },
      options: { scales: { r: { min: 0, max: 100, ticks: { stepSize: 25 } } } } });
  }

  async function analysisPage() {
    await ensureAnalysis();
    const a = state.analysis, r = state.result, ga = a.ga;
    const ev = ga.evaluation;
    const fmt = v => (v == null ? "—" : v.toFixed(4));
    const top = r?.recommendations[0];
    render(`<div class="stack">
      <h1>Soft Computing analysis</h1>
      <p class="muted">This page shows how the fuzzy logic and the genetic algorithm actually work in this system.
        ${r ? "The student-specific charts use your latest assessment." : "Complete an assessment to see your own membership values and fired rules."}</p>

      <h2>1. Fuzzy Logic (Mamdani FIS)</h2>
      <div class="grid grid-2">
        <div class="card"><h3>Membership functions (every input and the output)</h3><div class="chart-box"><canvas id="mf"></canvas></div>
          <p class="small muted">Triangles: ${a.fuzzy.sets.map((n, i) => `${n} (${a.fuzzy.params[i].join(", ")})`).join(" · ")}</p></div>
        ${r ? `<div class="card"><h3>Your membership degrees</h3><div class="chart-box"><canvas id="mu"></canvas></div>
          <p class="small muted">GA weighting: ${esc(a.fuzzy.weighting)}</p></div>` : ""}
      </div>
      ${r ? `<div class="card"><h3>Selected input: your memberships</h3>
        <table><thead><tr><th>Input</th><th class="num">Value</th>${SETS.map(s => `<th class="num">${s}</th>`).join("")}<th class="num">Weighted (L/M/H)</th></tr></thead><tbody>
        ${Object.entries(r.analysis.memberships).map(([k, m]) => `<tr><td>${esc(label(k))}</td><td class="num">${r.analysis.inputs[k]}</td>
          ${SETS.map(s => `<td class="num">${m[s].toFixed(2)}</td>`).join("")}
          <td class="num">${SETS.map(s => r.analysis.weighted_memberships[k][s].toFixed(2)).join(" / ")}</td></tr>`).join("")}
        </tbody></table></div>
      <div class="grid grid-2">
        <div class="card"><h3>Aggregated output and centroid: ${esc(top.name)}</h3><div class="chart-box"><canvas id="agg"></canvas></div>
          <p class="small muted">The fired rules clip the output sets (min), the clipped sets are combined (max), and the centroid of the shaded area gives s = ${top.fis_suitability.toFixed(2)}.</p></div>
        <div class="card"><h3>Activated rules (top 5 careers)</h3>
          ${r.recommendations.map(x => `<p style="margin:.6rem 0 .2rem"><b>${esc(x.name)}</b>: s = ${x.fis_suitability.toFixed(1)}</p>
            ${x.fired_rules.map(f => `<div class="rule">[${f.strength.toFixed(2)}] ${esc(f.text)}</div>`).join("")}`).join("")}</div>
      </div>` : ""}
      <details class="card"><summary>Full rule base (${a.fuzzy.rules.length} rules, 7 per career)</summary>
        ${a.fuzzy.rules.map(x => `<div class="rule">${esc(x.id)}: ${esc(x.text)}</div>`).join("")}</details>

      <h2>2. Genetic Algorithm</h2>
      ${ga.missing ? `<p class="error-box">GA results file missing. Equal weights are in use. Run <code>python run_ga.py</code>.</p>` : `
      <div class="grid grid-2">
        <div class="card"><h3>Fitness by generation</h3><div class="chart-box"><canvas id="fit"></canvas></div></div>
        <div class="card"><h3>Baseline vs optimised weights</h3><div class="chart-box"><canvas id="wts"></canvas></div></div>
      </div>
      <div class="grid grid-2">
        <div class="card"><h3>Settings</h3><table><tbody>
          <tr><td>Population</td><td class="num">${ga.config.population}</td></tr>
          <tr><td>Generations</td><td class="num">${ga.config.generations}</td></tr>
          <tr><td>Selection</td><td class="num">Tournament (k = ${ga.config.tournament_size})</td></tr>
          <tr><td>Crossover</td><td class="num">Blend BLX-α (α = ${ga.config.blend_alpha}), p = ${ga.config.crossover_prob}</td></tr>
          <tr><td>Mutation</td><td class="num">Gaussian σ = ${ga.config.mutation_sigma}, p = ${ga.config.mutation_prob}</td></tr>
          <tr><td>Elitism</td><td class="num">${ga.config.elitism}</td></tr>
          <tr><td>Random seed</td><td class="num">${ga.config.seed}</td></tr>
          <tr><td>Runtime</td><td class="num">${ga.runtime_seconds} s (offline, ${esc(ga.run_at)})</td></tr>
        </tbody></table></div>
        <div class="card"><h3>Fitness function</h3>
          <p>${esc(ga.fitness_metric)}.</p>
          <p class="small muted">Each chromosome holds six non-negative weights that sum to 1. For every training profile we rank all 18 careers and compare the model's top 5
            with a reference top 5. The reference comes from a separate expert scoring function that does not use the fuzzy rules. NDCG@5 rewards putting the right careers near the top.
            Data: ${ga.data.profiles} <b>synthetic</b> profiles (${ga.data.train} train / ${ga.data.holdout} held out).</p></div>
      </div>
      <div class="card"><h3>Baseline (equal weights) vs GA-optimised weights</h3>
        <table><thead><tr><th>Split</th><th class="num">Baseline NDCG@5</th><th class="num">Optimised NDCG@5</th><th class="num">Baseline P@5</th><th class="num">Optimised P@5</th></tr></thead><tbody>
          ${["train", "holdout"].map(s => `<tr><td>${s === "holdout" ? "<b>Held-out (independent)</b>" : "Training"}</td>
            <td class="num">${fmt(ev[s].baseline["ndcg@5"])}</td><td class="num">${fmt(ev[s].optimized["ndcg@5"])}</td>
            <td class="num">${fmt(ev[s].baseline["precision@5"])}</td><td class="num">${fmt(ev[s].optimized["precision@5"])}</td></tr>`).join("")}
        </tbody></table>
        <p class="notice small" style="margin-top:12px">${esc(ga._note)} Only the held-out row is an independent check. The training row was optimised directly.</p>
      </div>`}
      <div class="card"><h3>Hybrid score</h3><p><code>${esc(a.score_formula)}</code>, where s is the centroid output of the career's Mamdani FIS and I is the rank-weighted interest match.</p></div>
    </div>`);

    const mfX = a.fuzzy.curves.x;
    chart("mf", { type: "line", data: { labels: mfX, datasets: SETS.map(s => ({ label: s, data: a.fuzzy.curves.y[s],
        borderColor: COLORS[s.toLowerCase()], pointRadius: 0, borderWidth: 2 })) },
      options: { scales: { x: { title: { display: true, text: "Input value (0–100)" }, ticks: { maxTicksLimit: 11 } }, y: { min: 0, max: 1.05, title: { display: true, text: "Membership μ" } } } } });
    if (r) {
      const keys = Object.keys(r.analysis.memberships);
      chart("mu", { type: "bar", data: { labels: keys.map(label), datasets: SETS.map(s => ({ label: s,
          data: keys.map(k => r.analysis.memberships[k][s]), backgroundColor: COLORS[s.toLowerCase()] })) },
        options: { scales: { y: { min: 0, max: 1 } } } });
      chart("agg", { type: "line", data: { labels: r.analysis.curve_universe, datasets: [
          { label: "Aggregated output μ", data: top.aggregated_curve, borderColor: COLORS.brand, backgroundColor: "rgba(61,78,216,.2)", fill: true, pointRadius: 0 },
          { label: `Centroid = ${top.fis_suitability.toFixed(1)}`, data: r.analysis.curve_universe.map(u => Math.abs(u - top.fis_suitability) < 1 ? 1 : null),
            borderColor: COLORS.warn, backgroundColor: COLORS.warn, pointRadius: 4, showLine: false }] },
        options: { scales: { x: { title: { display: true, text: "Suitability universe (0–100)" }, ticks: { maxTicksLimit: 11 } }, y: { min: 0, max: 1.05 } } } });
    }
    if (!ga.missing) {
      chart("fit", { type: "line", data: { labels: ga.history.map(h => h.generation), datasets: [
          { label: "Best fitness", data: ga.history.map(h => h.best), borderColor: COLORS.brand, pointRadius: 0 },
          { label: "Average fitness", data: ga.history.map(h => h.average), borderColor: COLORS.accent, pointRadius: 0 }] },
        options: { scales: { x: { title: { display: true, text: "Generation" } }, y: { title: { display: true, text: "NDCG@5 (train)" } } } } });
      chart("wts", { type: "bar", data: { labels: a.skills, datasets: [
          { label: "Baseline (equal)", data: ga.baseline_weights, backgroundColor: COLORS.grey },
          { label: "Best in initial population", data: ga.initial_best_weights, backgroundColor: "#b9c0ff" },
          { label: "GA-optimised", data: ga.optimized_weights, backgroundColor: COLORS.brand }] },
        options: { scales: { y: { min: 0, title: { display: true, text: "Weight" } } } } });
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
        <li>Optimise the input weights with a Genetic Algorithm against a documented fitness function.</li><li>Give five ranked, explained recommendations with skill gaps.</li></ul>
      <h3>Technologies</h3>
      <p>HTML5, CSS3, vanilla JavaScript and Chart.js for the frontend. Python, Flask and NumPy for the backend. The Fuzzy Logic and Genetic Algorithm are custom, documented implementations.</p>
      <h3>Team members and responsibilities</h3>
      <table><tbody>
        <tr><td>Member 1</td><td>Fuzzy Logic module, rules, membership graphs</td></tr>
        <tr><td>Member 2</td><td>Genetic Algorithm, fitness function, experiment</td></tr>
        <tr><td>Member 3</td><td>Career dataset, reference profiles, ranking and skill gaps</td></tr>
        <tr><td>Member 4</td><td>Frontend and integration</td></tr></tbody></table>
      <p class="muted small">Replace these with your team's names before submission.</p>
      <h3>Limitations</h3>
      <ul><li>Skill ratings are self-reported and subjective.</li><li>Career target profiles are simplified model assumptions, not industry standards.</li>
        <li>The GA was trained and evaluated on synthetic profiles labelled by our own expert function. The evaluation shows consistency with those assumptions, not real-world accuracy.</li>
        <li>The interest match is used by both the model and the reference labels, which makes the agreement scores higher than they would otherwise be.</li>
        <li>Work preferences are not scored. No data is stored on the server.</li>
        <li>This tool does not replace professional career counselling.</li></ul>
      <h3>Future scope</h3>
      <p>Validate the career profiles with real counsellors, collect voluntary and anonymised survey data for evaluation, add trapezoidal or tuned membership functions, and add optional result history.</p>
      <h3>References</h3>
      <ul class="small"><li>Zadeh, L. A. (1965). Fuzzy sets. <i>Information and Control</i>, 8(3), 338–353.</li>
        <li>Mamdani, E. H., &amp; Assilian, S. (1975). An experiment in linguistic synthesis with a fuzzy logic controller. <i>Int. J. Man-Machine Studies</i>, 7(1), 1–13.</li>
        <li>Holland, J. H. (1975). <i>Adaptation in Natural and Artificial Systems</i>. University of Michigan Press.</li>
        <li>Eshelman, L. J., &amp; Schaffer, J. D. (1993). Real-coded genetic algorithms and interval-schemata (BLX-α). <i>Foundations of Genetic Algorithms 2</i>.</li>
        <li>Järvelin, K., &amp; Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques (NDCG). <i>ACM TOIS</i>, 20(4).</li></ul>
    </div>`);
  }

  // ---------- router ----------
  function go(h) { location.hash = h; }
  async function route() {
    const [, page = "", arg] = (location.hash || "#/").split("/");
    document.querySelectorAll(".nav a").forEach(a => a.classList.toggle("active", a.getAttribute("href") === `#/${page}`));
    document.getElementById("nav").classList.remove("open");
    try {
      if (["assess", "processing", "analysis", "career", "gap", "results"].includes(page)) await ensureMeta();
      switch (page) {
        case "": return home();
        case "instructions": return instructions();
        case "assess": return assess();
        case "processing": return processing();
        case "results": return results();
        case "career": return careerPage(decodeURIComponent(arg || ""));
        case "gap": return gapPage(decodeURIComponent(arg || ""));
        case "analysis": return analysisPage();
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
