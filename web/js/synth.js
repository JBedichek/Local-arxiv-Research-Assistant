/* Synthesize: a research question grown into a graph of sub-questions, live.
 *
 * The left column is the transcript of the graph as it grows -- a goal appears the moment it
 * starts and its answer the moment it lands -- and the right is the deliverable in three
 * lengths, with follow-ups. A run lives on the server, so closing the page stops nothing:
 * reopening a run repaints from the saved record and, if it is still going, follows the tail. */

import { $, escapeHtml } from "./dom.js";
import { renderMath } from "./tex.js";

/* The synthesizer endpoints report problems in an `error` field, so this panel talks to the
 * server through its own small wrapper instead of api.js. */
async function request(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  let parsed = null;
  try { parsed = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) throw new Error(parsed?.error || `${res.status} ${text.slice(0, 160)}`);
  return parsed;
}

const API = "/api/synthesizer/runs";
const VERSIONS = [["full", "Full", "deliverable"], ["medium", "Medium", "deliverable_medium"],
                  ["short", "Short", "deliverable_short"]];
const STATUS_GLYPH = { done: "✓", running: "●", failed: "✗", pending: "○", cancelled: "■",
                       interrupted: "…" };

const S = {
  open: false, runs: [], id: "", run: null, graph: null, source: null,
  version: "full", note: "", error: "", busy: "", answer: null, phase: "", live: "",
};
let paint = 0;

/* ── rendering text ───────────────────────────────────────────────────────────── */

/* Minimal markdown -- headings, bold, lists and [12345] chunk citations -- kept dependency-free
 * so it never mangles LaTeX. A citation that resolved to a paper links to it; one that did not
 * stays plain text, since a link would invent a source. */
function render(md, refs = {}) {
  const cite = (id) => {
    const r = refs[id];
    return r?.arxiv_url
      ? `<a class="cite" href="${escapeHtml(r.arxiv_url)}" target="_blank" rel="noopener" title="${
        escapeHtml(`${r.title || ""}${r.claim ? " — " + r.claim : ""}`)}">${id}</a>`
      : `<span class="cite unresolved" title="not resolved to a paper">${id}</span>`;
  };
  const html = renderMath(escapeHtml(md || ""))
    .replace(/^#{3,} (.*)$/gm, "<h4>$1</h4>")
    .replace(/^#{2} (.*)$/gm, "<h3>$1</h3>")
    .replace(/^# (.*)$/gm, "<h3>$1</h3>")
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/\[(\d{3,}(?:\s*,\s*\d{3,})*)\]/g, (_m, ids) =>
      "[" + ids.split(/\s*,\s*/).map(cite).join(", ") + "]");
  return html.split(/\n{2,}/).map((block) => {
    if (/^<h[34]>/.test(block)) return block;
    const lines = block.split("\n");
    if (lines.every((l) => /^\s*[-*] /.test(l))) {
      return `<ul>${lines.map((l) => `<li>${l.replace(/^\s*[-*] /, "")}</li>`).join("")}</ul>`;
    }
    return `<p>${block.replace(/\n/g, " ")}</p>`;
  }).join("");
}

function when(ts) {
  return ts ? new Date(ts * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "";
}

const tokens = (n) => (n || 0).toLocaleString();

/* ── the run list ─────────────────────────────────────────────────────────────── */

function renderHome() {
  const runs = S.runs.map((r) => `
    <li><a href="#" data-run="${r.id}" class="synth-goal">${escapeHtml(r.goal)}</a>
      <span class="badge ${escapeHtml(r.status)}">${STATUS_GLYPH[r.status] || ""} ${escapeHtml(r.status)}</span>
      ${r.verdict?.kind ? `<span class="hint">${escapeHtml(r.verdict.kind)}</span>` : ""}
      ${r.imported ? `<span class="hint">imported from ${escapeHtml(r.imported)}</span>` : ""}
      <span class="hint">${when(r.created)}</span></li>`).join("");
  return `
    <p class="lede">Name a question. It is split into sub-questions that are researched in the paper
      corpus, grown round by round as answers land, then written up as one deliverable with every
      claim cited.</p>
    <form id="synth-new">
      <textarea id="synth-goal" rows="3" placeholder="e.g. What is the most sample-efficient Muon-based optimizer in the literature, and why?"></textarea>
      <details><summary>Options</summary>
        <label class="row"><input type="checkbox" id="synth-sub" checked>
          Let a sub-question run a whole nested research graph when it is itself multi-part</label>
        <label class="row">Deliverable token bound
          <input type="number" id="synth-tokens" min="0" step="1000" value="0">
          <span class="hint">0 = no bound</span></label>
        <label class="row">After the report, also answer this from it
          <input type="text" id="synth-final" placeholder="e.g. what are the top 5 methods here that could be implemented"></label>
      </details>
      <button type="submit" id="synth-start">Start</button>
    </form>
    ${S.error ? `<p class="err">${escapeHtml(S.error)}</p>` : ""}
    <h3>Runs</h3>
    ${runs ? `<ul class="synth-runs">${runs}</ul>` : `<p class="empty">No runs yet.</p>`}`;
}

/* ── a run ────────────────────────────────────────────────────────────────────── */

const live = () => S.run?.status === "running";

function statsLine() {
  const goals = Object.values(S.graph?.goals || {});
  const parts = [];
  if (S.run?.rounds) parts.push(`round ${S.run.rounds}`);
  parts.push(`${goals.length} goal(s) in the graph`);
  const shown = goals.map(goalStatus);
  for (const s of ["running", "done", "failed", "pending", "stopped"]) {
    const n = shown.filter((x) => x === s).length;
    if (n) parts.push(`${n} ${s}`);
  }
  if ((S.graph?.compressed || []).length) parts.push(`${S.graph.compressed.length} fold(s)`);
  if (S.run?.tokens_in || S.run?.tokens_out) {
    parts.push(`${tokens(S.run.tokens_in)} tokens in / ${tokens(S.run.tokens_out)} out`);
  }
  return parts.join(" · ");
}

/* A goal saved as running or pending under a run that is no longer live was cut off. */
const goalStatus = (g) => (!live() && ["running", "pending"].includes(g.status) ? "stopped" : g.status);

function goalHtml(g) {
  const refs = g.citations || {};
  const status = goalStatus(g);
  return `<article class="synth-entry" data-entry="${escapeHtml(g.id)}">
    <div class="synth-top"><span class="badge ${escapeHtml(status)}">${STATUS_GLYPH[status] || ""} ${escapeHtml(status)}</span>
      <code>${escapeHtml(g.id)}</code>
      ${g.refines ? `<span class="hint">↳ refines <code>${escapeHtml(g.refines)}</code></span>` : ""}
      ${g.nested ? `<span class="hint">nested graph</span>` : ""}
      ${status === "running" ? `<span class="hint">researching now…</span>` : ""}</div>
    <p class="synth-q">${escapeHtml(g.text)}</p>
    ${g.status === "failed" ? `<p class="err">Failed: ${escapeHtml(g.error || "unknown error")}</p>` : ""}
    ${g.status === "done" ? `<details><summary>Answer</summary><div class="prose">${render(g.summary, refs)}</div></details>` : ""}
  </article>`;
}

function feedHtml() {
  const folds = (S.graph?.compressed || []).map((text, i) => `
    <article class="synth-entry fold"><div class="synth-top"><span class="badge">⊞ fold ${i + 1}</span></div>
      <p class="hint">Everything established so far, folded into one summary so later rounds reason from a digest.</p>
      <details><summary>Summary</summary><div class="prose">${render(text)}</div></details></article>`);
  return folds.join("") + Object.values(S.graph?.goals || {}).map(goalHtml).join("");
}

function versionTabs() {
  const have = VERSIONS.filter(([, , key]) => (S.run?.[key] || "").trim());
  if (!have.length) return "";
  if (!have.some(([id]) => id === S.version)) S.version = have[0][0];
  return have.map(([id, label]) =>
    `<button type="button" class="tab${id === S.version ? " current" : ""}" data-version="${id}">${label}</button>`).join("");
}

function deliverableHtml() {
  const run = S.run || {};
  const [, , key] = VERSIONS.find(([id]) => id === S.version) || VERSIONS[0];
  const refs = run[`${key}_references`] ?? run.references ?? {};
  const text = run[key] || "";
  const followups = (run.followups || []).map((f) =>
    `<li><button type="button" class="chip" data-follow="${escapeHtml(f)}">${escapeHtml(f)}</button></li>`).join("");
  const done = run.status && run.status !== "running";
  return `
    ${text ? `<nav class="tabs">${versionTabs()}</nav><div class="prose deliverable">${render(text, refs)}</div>`
           : `<p class="empty">${live() ? "The deliverable is written once the graph has run its course." : "This run produced no deliverable."}</p>`}
    ${S.phase && live() ? `<p class="hint">${escapeHtml(S.phase)}…</p>` : ""}
    ${(run.notes || []).map((n) => `<p class="err">${escapeHtml(n)}</p>`).join("")}
    ${S.note ? `<p class="hint">${escapeHtml(S.note)}</p>` : ""}
    ${done && text ? `
      <h4>Ask this report a question</h4>
      <form id="synth-ask"><input id="synth-ask-q" type="text" placeholder="e.g. what are the top 5 methods here that could be implemented"
        ${S.busy === "ask" ? "disabled" : ""}><button type="submit" ${S.busy === "ask" ? "disabled" : ""}>Answer</button></form>
      <p class="hint">Answers from this report only; the answer is saved above the full report.</p>
      ${S.error ? `<p class="err">${escapeHtml(S.error)}</p>` : ""}` : ""}
    ${followups || (done && text) ? `
      <h4>Follow-ups <button type="button" id="synth-refresh" ${S.busy === "refresh" ? "disabled" : ""}
        title="Five different suggestions">${S.busy === "refresh" ? "thinking…" : "Refresh"}</button></h4>
      <ul class="synth-follow">${followups}</ul>` : ""}`;
}

function renderRun() {
  const r = S.run;
  if (!r) return `<p class="empty">Loading…</p>`;
  const badge = `<span class="badge ${escapeHtml(r.status)}">${STATUS_GLYPH[r.status] || ""} ${escapeHtml(r.status)}</span>`;
  return `
    <p><a href="#" id="synth-back">← all runs</a></p>
    <h3 class="synth-title">${escapeHtml(r.goal)}</h3>
    <p class="synth-meta">${badge}
      ${r.verdict ? `<span class="hint" title="${escapeHtml(r.verdict.because || "")}">${escapeHtml(r.verdict.kind)}: ${escapeHtml(r.verdict.because || "")}</span>` : ""}
      ${r.imported ? `<span class="hint">imported from ${escapeHtml(r.imported)} (read-only history)</span>` : ""}
      ${S.live ? `<span class="hint">${escapeHtml(S.live)}</span>` : ""}
      ${live() ? `<button type="button" id="synth-cancel">Stop</button>`
               : `<button type="button" id="synth-delete">Delete</button>`}</p>
    <p class="hint" id="synth-stats">${escapeHtml(statsLine())}</p>
    ${r.error ? `<p class="err">${escapeHtml(r.error)}</p>` : ""}
    <div id="synth-cols">
      <section id="synth-feed"><h4>Sub-questions</h4><div id="synth-feed-body">${feedHtml()}</div></section>
      <section id="synth-report"><h4>Deliverable</h4><div id="synth-report-body">${deliverableHtml()}</div></section>
    </div>`;
}

/* Repaints only the two columns' bodies, so a scrolled feed stays put while events stream. */
function repaint() {
  cancelAnimationFrame(paint);
  paint = requestAnimationFrame(() => {
    if (!S.open || !S.id) return;
    const feed = $("#synth-feed-body"), report = $("#synth-report-body");
    if (!feed || !report) { draw(); return; }
    const openIds = new Set([...feed.querySelectorAll("details[open]")]
      .map((d) => d.closest("[data-entry]")?.dataset.entry));
    feed.innerHTML = feedHtml();
    feed.querySelectorAll("[data-entry]").forEach((n) => {
      if (openIds.has(n.dataset.entry)) n.querySelector("details")?.setAttribute("open", "");
    });
    const askText = $("#synth-ask-q")?.value;
    report.innerHTML = deliverableHtml();
    if (askText && $("#synth-ask-q")) $("#synth-ask-q").value = askText;
    $("#synth-stats").textContent = statsLine();
  });
}

function draw() {
  const main = $("#synth-main");
  if (!main) return;
  main.innerHTML = S.id ? renderRun() : renderHome();
}

/* ── talking to the server ────────────────────────────────────────────────────── */

function stopWatching() {
  S.source?.close();
  S.source = null;
}

function applyGoal(g) {
  S.graph = S.graph || { goals: {}, compressed: [] };
  S.graph.goals[g.id] = g;
}

function addNote(d) {
  S.run.notes = [...(S.run.notes || []), d.error];
}

function watch(id) {
  stopWatching();
  S.live = "connecting…";
  const es = S.source = new EventSource(`${API}/${encodeURIComponent(id)}/events`);
  const on = (name, fn) => es.addEventListener(name, (e) => {
    if (S.source !== es) return;
    fn(JSON.parse(e.data));
    repaint();
  });
  on("snapshot", (d) => { S.run = d.run; S.graph = d.graph; S.live = live() ? "live" : ""; });
  on("goal.new", applyGoal);
  on("goal.update", applyGoal);
  on("compression", (d) => { S.graph.compressed[d.index] = d.text; });
  on("round", (d) => { S.run.rounds = d.round; S.run.tokens_in = d.tokens_in; S.run.tokens_out = d.tokens_out; });
  on("phase", (d) => { S.phase = d.name; });
  on("deliverable", (d) => { S.run.deliverable = d.text; S.run.references = d.references; S.phase = ""; });
  for (const level of ["medium", "short"]) {
    on(`deliverable_${level}`, (d) => {
      S.run[`deliverable_${level}`] = d.text;
      S.run[`deliverable_${level}_references`] = d.references;
    });
    on(`deliverable_${level}.failed`, addNote);
    on(`deliverable_${level}.empty`, addNote);
  }
  on("followups", (d) => { S.run.followups = d.suggestions; });
  on("followups.failed", addNote);
  on("followups.empty", addNote);
  on("verdict", (d) => { S.run.verdict = d; });
  es.addEventListener("end", async () => {
    if (S.source !== es) return;
    stopWatching();
    S.live = "";
    S.phase = "";
    try {
      S.run = (await request("GET", `${API}/${encodeURIComponent(id)}`)).run;
    } catch (err) {
      S.note = `The run ended, but its final state could not be loaded (${err.message}). Reload to see it.`;
    }
    repaint();
  });
  es.onerror = () => { if (S.source === es) S.live = "reconnecting…"; repaint(); };
}

async function openRunView(id) {
  S.id = id; S.run = null; S.graph = null; S.note = ""; S.error = ""; S.phase = ""; S.version = "full";
  draw();
  watch(id);
}

async function loadRuns() {
  try {
    S.runs = (await request("GET", API)).runs;
    S.error = "";
  } catch (err) {
    S.error = err.message;
  }
  if (!S.id) draw();
}

export function openSynth() {
  const view = $("#synth");
  if (!view) return;
  view.hidden = false;
  document.body.classList.add("synth-open");
  S.open = true;
  draw();
  loadRuns();
}

export function closeSynth() {
  const view = $("#synth");
  if (!view) return;
  view.hidden = true;
  document.body.classList.remove("synth-open");
  S.open = false;
  stopWatching();
}

function backToRuns() {
  stopWatching();
  S.id = ""; S.run = null; S.graph = null; S.error = "";
  draw();
  loadRuns();
}

async function start(goal, extra = {}) {
  const res = await request("POST", API, { goal, ...extra });
  S.runs = [res, ...S.runs];
  await openRunView(res.id);
}

export function bindSynth() {
  $("#synth-btn")?.addEventListener("click", openSynth);
  $("#synth-close")?.addEventListener("click", closeSynth);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && S.open) closeSynth(); });

  document.addEventListener("submit", async (e) => {
    const f = e.target;
    if (f.id === "synth-new") {
      e.preventDefault();
      const goal = $("#synth-goal").value.trim();
      if (!goal) return;
      $("#synth-start").disabled = true;
      try {
        await start(goal, {
          allow_subsynthesis: $("#synth-sub").checked,
          deliverable_tokens: Number($("#synth-tokens").value) || 0,
          final_compression_prompt: $("#synth-final").value.trim(),
        });
      } catch (err) {
        S.error = err.message;
        draw();
      }
    } else if (f.id === "synth-ask") {
      e.preventDefault();
      const prompt = $("#synth-ask-q").value.trim();
      if (!prompt) return;
      S.busy = "ask"; S.error = ""; S.note = "answering…";
      repaint();
      try {
        const out = await request("POST", `${API}/${encodeURIComponent(S.id)}/compress`, { prompt });
        S.run.deliverable = out.deliverable;
        S.version = "full";
        S.note = out.degraded ? `Answered, but under budget pressure: ${out.degraded_because || "unspecified"}`
                              : "Answered — saved above the full report.";
      } catch (err) {
        S.error = err.message; S.note = "";
      }
      S.busy = "";
      repaint();
    }
  });

  document.addEventListener("click", async (e) => {
    if (!S.open) return;
    const t = e.target;
    const runLink = t.closest("[data-run]");
    if (runLink) { e.preventDefault(); openRunView(runLink.dataset.run); return; }
    if (t.closest("#synth-back")) { e.preventDefault(); backToRuns(); return; }
    const tab = t.closest("[data-version]");
    if (tab) { S.version = tab.dataset.version; repaint(); return; }
    const chip = t.closest("[data-follow]");
    if (chip) {
      try { await start(chip.dataset.follow, { parent: S.id }); } catch (err) { S.error = err.message; repaint(); }
      return;
    }
    if (t.closest("#synth-cancel")) {
      try { await request("POST", `${API}/${encodeURIComponent(S.id)}/cancel`); } catch (err) { S.error = err.message; repaint(); }
      return;
    }
    if (t.closest("#synth-delete")) {
      if (!confirm("Delete this run and its graph?")) return;
      try {
        await request("DELETE", `${API}/${encodeURIComponent(S.id)}`);
        backToRuns();
      } catch (err) { S.error = err.message; repaint(); }
      return;
    }
    if (t.closest("#synth-refresh")) {
      S.busy = "refresh"; S.error = ""; repaint();
      try {
        S.run.followups = (await request("POST", `${API}/${encodeURIComponent(S.id)}/followups/refresh`)).followups;
      } catch (err) { S.error = err.message; }
      S.busy = "";
      repaint();
    }
  });
}

bindSynth();
