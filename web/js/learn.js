/* Learn: a course built from the paper corpus for a goal you name.
 *
 * Every lesson sentence carries the claims it rests on, and each claim its source passage, so
 * "where does it say that?" is one click. The page never shows a quiz answer until you have
 * answered, and never a chart number or diagram edge the server did not check against a claim. */

import { $, escapeHtml } from "./dom.js";
import { renderMath } from "./tex.js";
import { chartSvg } from "./chart.js";
import * as VOICE from "./voice.js";

/* lara's api.js returns parsed JSON and throws the raw response text; the Learn endpoints
 * report what is wrong in an `error` field and some requests run for minutes, so this panel
 * talks to the server through its own small wrapper. */
async function request(method, path, body, opts = {}) {
  const ctl = new AbortController();
  const timeoutMs = opts.timeoutMs || 30000;
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(path, {
      method, signal: ctl.signal,
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (err) {
    if (err.name === "AbortError") throw new Error(`request timed out after ${timeoutMs / 1000}s`);
    throw err;
  } finally {
    clearTimeout(timer);
  }
  const text = await res.text();
  let parsed = null;
  try { parsed = text ? JSON.parse(text) : null; } catch { /* not JSON */ }
  if (!res.ok) throw new Error(parsed?.error || `${res.status} ${text.slice(0, 160)}`);
  return parsed;
}

const get = (path) => request("GET", path);
const send = (method, path, body, opts) => request(method, path, body, opts);

const L = {
  courses: [], id: "", course: null, view: "path", conceptId: "", concept: null,
  claim: "", item: null, graded: null, pretest: null, error: "", busy: "",
  open: false, requested: new Set(), autostart: false, poll: 0, critique: null, items: {},
  pending: null, ask: null, asking: false, askResult: "", variant: "standard",
  partialOpen: "",
  showProfile: false, trace: [], traceCursor: 0, traceConceptId: "", traceTimer: 0,
  traceOpen: new Set(),
  mapTrace: [], mapTraceCursor: 0, mapTraceId: "",
  showKnowledge: false, knowledge: null,
  diag: null, diagResult: null, checkResult: null,
};

/* The five levels of the learner profile (lara/learn/profile.py). */
const LEVELS = ["unknown", "heard of", "intuition", "use", "derive"];

/* Lesson/claim/quiz text comes from the model, reading real papers -- 66% of chunks
 * carry inline LaTeX (see tex.js), so it has to go through the same renderer the paper
 * reader and Deep Research use, or every $O(n^2)$ shows up as literal dollar signs. */
const md = (s) => renderMath(escapeHtml(s ?? ""));

const CERTAINTY = {
  established: "several papers agree",
  "single-source": "one paper says this",
  contested: "papers disagree",
  speculative: "a hypothesis, not a result",
  superseded: "replaced by later work",
};

export async function loadLearn() {
  try {
    L.courses = (await get("/api/learn/courses")).courses;
    if (L.id) L.course = await get(`/api/learn/courses/${encodeURIComponent(L.id)}`);
    if (L.id && L.course?.status === "ready") await loadDiagnostic();
    L.error = "";
  } catch (err) {
    L.error = err.message;
  }
  renderLearn();
  schedulePoll();
}

/* Cross-course, so it lives at the top level (alongside the course list), not inside any
 * one course's own view -- see lara.learn.profile. Read-only: the only way to change it
 * is to answer more quizzes. */
async function loadKnowledge() {
  try {
    L.knowledge = await get("/api/learner/profile");
    L.error = "";
  } catch (err) {
    L.error = err.message;
  }
  renderLearn();
}

/* Poll only while something is being built and the tab is open. */
function schedulePoll() {
  clearTimeout(L.poll);
  if (!L.open || !L.id) return;
  const c = L.course;
  const waiting = c?.next?.action === "build" && L.requested.has(c.next.concept);
  const building = c && (waiting || c.status === "mapping" || c.pretest === "building"
    || L.diag?.state === "preparing"
    || (c.concepts || []).some((k) => k.build?.stage && !["done", "error", ""].includes(k.build.stage)
      && L.requested.has(k.id)));
  if (building || (L.conceptId && L.concept && buildingNow(L.concept))) {
    L.poll = setTimeout(async () => {
      await refresh();
      schedulePoll();
    }, 2500);
  }
}

function buildingNow(concept) {
  const s = concept.build?.stage;
  const stageBuilding = L.requested.has(concept.id) && s && !["done", "error"].includes(s);
  return stageBuilding || (concept.topics || []).some((t) => t.doc_status === "building");
}

/* Mirrors lara.learn.pipeline.STAGES -- the order a concept actually builds in. */
const BUILD_STAGES = ["claims", "lesson", "topics", "quiz", "visuals"];

/* A real progress bar (which of the 5 pipeline stages is done/current/pending, plus -- while
 * on claims, the usually-longest one -- how many of the planned facets have finished their
 * first pass) and elapsed time, rather than a predicted ETA: round 2/3 of a facet only happen
 * if the model finds a reason, so a confident "~Ns left" would just be a guess dressed as a
 * fact. tokens_in/out are an estimate (see TokenMeter in lara/learn/llm.py), not vLLM's own
 * count -- flagged with "~" rather than presented as exact. Refreshed every poll (2.5s, see
 * schedulePoll), the same cadence every other "live" number in this page already updates on. */
function buildProgress(concept) {
  const b = concept.build || {};
  const idx = BUILD_STAGES.indexOf(b.stage);
  if (idx < 0) return "";
  const segs = BUILD_STAGES.map((name, i) =>
    `<span class="build-seg ${i < idx ? "done" : i === idx ? "current" : ""}" title="${name}"></span>`).join("");
  const elapsed = b.started ? Math.max(0, Math.round(Date.now() / 1000 - b.started)) : 0;
  const mins = Math.floor(elapsed / 60);
  const elapsedText = mins ? `${mins}m ${elapsed % 60}s` : `${elapsed}s`;
  const t = concept.trace;
  const facetNote = (b.stage === "claims" && t?.budget?.facets)
    ? ` · facet ${new Set((t.rounds || []).map((r) => r.facet)).size} of ${t.budget.facets} researched` : "";
  const tokens = (b.tokens_in || b.tokens_out) ? ` · ~${b.tokens_in || 0} tokens in, ~${b.tokens_out || 0} out` : "";
  return `<div class="build-progress"><div class="build-bar">${segs}</div>
    <p class="hint">Step ${idx + 1} of ${BUILD_STAGES.length} (${escapeHtml(b.stage)})${facetNote} · ${elapsedText} elapsed${tokens}</p></div>`;
}

async function refresh() {
  if (!L.id) return;
  try {
    L.course = await get(`/api/learn/courses/${encodeURIComponent(L.id)}`);
    if (L.conceptId) L.concept = await fetchConcept(L.conceptId);
    if (L.pretest?.state === "building" || L.course.pretest === "building") await loadPretest();
    if (L.course.status === "mapping") await loadMapTrace();
    if (L.course.status === "ready") await loadDiagnostic();
    await autoBuild();
  } catch (err) {
    L.error = err.message;
  }
  renderLearn();
}

const base = () => `/api/learn/courses/${encodeURIComponent(L.id)}`;

async function fetchConcept(cid) {
  return get(`${base()}/concepts/${encodeURIComponent(cid)}`);
}

async function loadDiagnostic() {
  L.diag = await get(`${base()}/diagnostic`);
}

async function loadPretest() {
  L.pretest = await get(`${base()}/pretest`);
}

/* Once the learner has started, each next concept is built as they reach it -- never before
 * they have asked for the first, so opening a course costs nothing. */
async function autoBuild() {
  const next = L.course?.next;
  if (L.autostart && next?.action === "build" && !L.requested.has(next.concept)) {
    L.requested.add(next.concept);
    await send("POST", `${base()}/concepts/${encodeURIComponent(next.concept)}/build`, {});
  }
}

async function act(label, fn) {
  L.busy = label;
  L.error = "";
  renderLearn();
  try {
    await fn();
  } catch (err) {
    L.error = err.message;
  }
  L.busy = "";
  renderLearn();
  schedulePoll();
}

/* ── rendering ────────────────────────────────────────────────────────────────── */

export function renderLearn() {
  const host = $("#learn-main");
  if (!host) return;
  const err = L.error ? `<p class="error">${escapeHtml(L.error)}</p>` : "";
  const busy = L.busy ? `<p class="hint">${escapeHtml(L.busy)}…</p>` : "";
  if (L.showKnowledge) {
    host.innerHTML = `${err}${busy}${knowledgeView(L.knowledge)}`;
    return;
  }
  host.innerHTML = L.id && L.course ? `${err}${busy}${courseView(L.course)}` : `${err}${busy}${listView()}`;
}

function listView() {
  const rows = L.courses.map((c) => `
    <li class="learn-course">
      <button type="button" class="link" data-learn="open" data-id="${escapeHtml(c.id)}">${escapeHtml(c.goal)}</button>
      <span class="hint">${escapeHtml(c.status)} · ${c.concepts} concepts</span>
      <button type="button" class="link danger" data-learn="delete" data-id="${escapeHtml(c.id)}">delete</button>
    </li>`).join("");
  return `
    <p class="lede">Name something you want to learn. The course is built only from papers in the
      corpus: every claim links to its source, papers that disagree are shown disagreeing, and
      what the sources cannot support is left out rather than filled in.</p>
    <form id="learn-new" class="learn-new">
      <textarea id="learn-goal" rows="3" placeholder="e.g. I want to create the best pretraining recipe I can for a from-scratch LLM"></textarea>
      <button type="submit">Start a course</button>
    </form>
    <button type="button" class="link" data-learn="knowledge-open">Your knowledge →</button>
    ${rows ? `<h4>Your courses</h4><ul class="learn-courses">${rows}</ul>` : ""}`;
}

/* Read-only, cross-course: what quizzes (in any course) have shown this learner
 * understands -- see lara.learn.profile. Lessons already read this (the digest) when
 * first written; this is the same information, for the learner themselves to see. */
function knowledgeView(k) {
  const head = `<div class="learn-head">
    <button type="button" class="link" data-learn="knowledge-close">← all courses</button>
    <h3>Your knowledge</h3></div>`;
  if (!k) return `${head}<p class="hint">Loading…</p>`;
  const known = k.concepts.filter((c) => c.bucket === "known");
  const shaky = k.concepts.filter((c) => c.bucket === "shaky");
  const row = (c) => `<li>${md(c.title)}
    <span class="hint">most likely: ${escapeHtml(LEVELS[c.level] || "?")} · ${c.attempts} piece${c.attempts === 1 ? "" : "s"} of evidence</span></li>`;
  const background = k.anchors?.length || k.use
    ? `<p class="hint">${k.use ? `You'll use this for: ${md(k.use)}. ` : ""}${k.anchors?.length
      ? `Your background: ${k.anchors.map((a) => `${escapeHtml(a.domain)} (${escapeHtml(a.depth)})`).join(", ")}.` : ""}</p>` : "";
  const empty = !known.length && !shaky.length
    ? `<p class="hint">Nothing assessed yet — answer some quiz questions in a course and they'll show up here.</p>` : "";
  return `${head}
    <p class="lede">What your quiz answers, diagnostics and highlights across courses show you
      understand. Every new lesson is planned against this: what you know is used without
      explaining it, and what you don't is researched and explained.</p>
    ${background}
    ${k.digest ? `<div class="learn-card"><p>${md(k.digest)}</p></div>` : ""}
    ${known.length ? `<h4>Solid</h4><ul class="learn-plan">${known.map(row).join("")}</ul>` : ""}
    ${shaky.length ? `<h4>Shaky</h4><ul class="learn-plan">${shaky.map(row).join("")}</ul>` : ""}
    ${empty}`;
}

function courseView(c) {
  const head = `<div class="learn-head">
    <button type="button" class="link" data-learn="back">← all courses</button>
    <h3>${escapeHtml(c.goal)}</h3></div>`;
  if (["scoping", "scoped", "failed", "mapping", "awaiting_approval"].includes(c.status) || !c.concepts.length) {
    return head + scopeView(c);
  }
  return head + competenciesView(c) + mapView(c)
    + (L.view === "concept" && L.concept ? conceptView(L.concept) : focusView(c));
}

/* Scope ------------------------------------------------------------------------- */

function scopeView(c) {
  const s = c.scope || {};
  const comps = (s.competencies || []).map((x) => `<li>${escapeHtml(x.text)}</li>`).join("");
  const last = (s.map_history || []).slice(-1)[0];
  const diff = last?.diff && (last.diff.added.length || last.diff.removed.length)
    ? `<p class="hint">Your answer changed the plan:
        ${last.diff.added.map((t) => `<span class="ok">+ ${escapeHtml(t)}</span>`).join(" ")}
        ${last.diff.removed.map((t) => `<span class="err">− ${escapeHtml(t)}</span>`).join(" ")}</p>` : "";
  let action = "";
  if (s.pending) {
    const q = s.pending;
    action = `<div class="learn-card">
      <p><b>${md(q.question)}</b></p>
      ${q.why ? `<p class="hint">${md(q.why)}</p>` : ""}
      <p>${(q.options || []).map((o) => `<button type="button" data-learn="answer" data-value="${escapeHtml(o)}">${md(o)}</button>`).join(" ")}</p>
      <form id="learn-answer" class="learn-new"><input id="learn-answer-text" type="text" placeholder="or in your own words">
        <button type="submit">Answer</button></form>
      <button type="button" class="link" data-learn="accept">Good enough — use this plan</button></div>`;
  } else if (c.status === "scoped") {
    action = `<button type="button" data-learn="map">Build the concept map</button>`;
  } else if (c.status === "mapping") {
    const revising = (c.plan_history || []).length > 0;
    const phase = revising ? "Revising the plan from your feedback…"
      : c.mapping === "decomposing" ? "Mapping the whole subject with one thorough deep-research run, from its foundations to its frontier… <span class=\"hint\">It reads hundreds of passages and takes several minutes; a subject mapped before is reused.</span>"
        : "Researching the course and drawing the map of what you need to learn…";
    action = `<p class="hint">${phase}</p>
      <div class="learn-card">${profileView(L.mapTrace)}</div>`;
  } else if (c.status === "awaiting_approval") {
    action = planReviewView(c);
  } else if (c.status === "failed" || c.status === "ready") {
    action = `<p class="error">${escapeHtml(c.error || "That did not work.")}</p>
      <button type="button" data-learn="map">Try mapping again</button>`;
  }
  return `<h4>What you will be able to do</h4><ul class="learn-comps">${comps}</ul>${diff}${action}`;
}

/* Phase 1's gate: the concept map is shown before any lesson is built from it, so a
 * learner can catch a bad plan before paying for 12+ lessons written against it. */
/* Lessons grouped under their subject (course["subjects"], see graph._concepts_from_
 * result) -- a course this broad used to render as one flat list of 20-40 lessons with
 * no relationship between them. A course mapped before subjects existed (or by the old
 * blind pipeline) has none: falls back to one untitled group holding every lesson, so
 * the same rendering code covers both without a separate flat-list branch. */
function planReviewView(c) {
  const byId = Object.fromEntries(c.concepts.map((k) => [k.id, k]));
  const groups = (c.subjects || []).length
    ? c.subjects
    : [{ id: "", title: "", summary: "", concept_ids: c.concepts.map((k) => k.id) }];
  const nodes = c.concepts.map((k) => ({ ...k, label: k.title }));
  const edges = c.concepts.flatMap((k) => k.prereqs.map((p) => ({ from: p, to: k.id })));
  const graph = svgGraph(nodes, edges,
    { nodeSub: (n) => `${n.sources} source${n.sources === 1 ? "" : "s"}` });
  const subjectBlocks = groups.map((s) => {
    const lessons = (s.concept_ids || []).map((cid) => byId[cid]).filter(Boolean);
    if (!lessons.length) return "";
    const rows = lessons.map((k) => `
      <li><b>${md(k.title)}</b>
        <p>${md(k.summary)}</p>
        ${k.prereqs.length ? `<p class="hint">After: ${k.prereqs.map((p) =>
          escapeHtml(byId[p]?.title || p)).join(", ")}</p>` : ""}
      </li>`).join("");
    return `<div class="learn-subject">
      ${s.title ? `<h5>${md(s.title)}</h5>` : ""}
      ${s.summary ? `<p class="hint">${md(s.summary)}</p>` : ""}
      <ul class="learn-plan">${rows}</ul>
    </div>`;
  }).join("");
  const gaps = (c.uncovered || []).length
    ? `<p class="hint warn">The corpus had nothing to build these from: ${c.uncovered.map(escapeHtml).join("; ")}</p>` : "";
  return `<h4>Review the course plan</h4>
    <p class="lede">Here is the course outline before any lessons are written, grouped into
      subjects. Approve it to start building lessons, or say what you would like changed and
      the plan will be revised.</p>
    ${graph}${subjectBlocks}${gaps}
    ${historyView(c.plan_history, "Earlier plan")}
    <div class="learn-card">
      <button type="button" data-learn="plan-approve">Approve this plan</button>
      <form id="learn-plan-revise" class="learn-new">
        <textarea id="learn-plan-feedback" rows="3" placeholder='What would you like changed? e.g. "Add a subject on reward hacking" or "Merge these two lessons" or "Put optimizer choice before scaling laws"'></textarea>
        <button type="submit">Request changes</button>
      </form>
    </div>`;
}

/* Shared by the plan review above and the lesson revision card below -- a version kept
 * (course["plan_history"] / concept["lesson_history"]) is shown as what changed and why,
 * not re-rendered whole: the current version is already on screen. */
function historyView(history, label) {
  if (!history?.length) return "";
  const rows = history.map((h, i) => `<li><b>${escapeHtml(label)} ${i + 1}</b> — ${md(h.feedback)}
    <span class="hint">${new Date(h.superseded * 1000).toLocaleString()}</span></li>`).join("");
  return `<details class="hint"><summary>${history.length} earlier version${history.length > 1 ? "s" : ""}</summary><ul>${rows}</ul></details>`;
}

/* Progress ---------------------------------------------------------------------- */

function competenciesView(c) {
  const rows = c.competencies.map((x) => `
    <li class="${x.mastered ? "done" : ""}">
      <span class="bar"><i style="width:${Math.round(x.progress * 100)}%"></i></span>
      ${x.mastered ? "<b>You can now:</b> " : ""}${md(x.text)}
    </li>`).join("");
  const gaps = c.uncovered?.length
    ? `<p class="hint warn">The corpus had nothing to build these from: ${c.uncovered.map(escapeHtml).join("; ")}</p>` : "";
  return `<h4>Progress</h4><ul class="learn-progress">${rows}</ul>${gaps}`;
}

/* Layered layout shared by the concept map and diagrams ------------------------- */

function layout(nodes, edges, GX = 54) {
  const parents = {};
  nodes.forEach((n) => (parents[n.id] = []));
  edges.forEach((e) => parents[e.to]?.push(e.from));
  const layer = {};
  const depth = (id, seen = new Set()) => {
    if (layer[id] !== undefined) return layer[id];
    if (seen.has(id)) return 0;
    seen.add(id);
    const ps = (parents[id] || []).filter((p) => p in parents);
    layer[id] = ps.length ? 1 + Math.max(...ps.map((p) => depth(p, seen))) : 0;
    return layer[id];
  };
  nodes.forEach((n) => depth(n.id));
  const columns = {};
  nodes.forEach((n) => (columns[layer[n.id]] ||= []).push(n));
  const W = 168, H = 46, GY = 22, pos = {};
  Object.entries(columns).forEach(([l, list]) => list.forEach((n, i) => {
    pos[n.id] = { x: 12 + l * (W + GX), y: 12 + i * (H + GY), w: W, h: H };
  }));
  const cols = Math.max(...Object.keys(columns).map(Number)) + 1;
  const rows = Math.max(...Object.values(columns).map((c2) => c2.length));
  return { pos, width: 24 + cols * W + (cols - 1) * GX, height: 24 + rows * H + (rows - 1) * GY };
}

function svgGraph(nodes, edges, { nodeClass, nodeSub, action } = {}) {
  if (!nodes.length) return "";
  const { pos, width, height } = layout(nodes, edges, edges.some((e) => e.label) ? 110 : 54);
  const lines = edges.filter((e) => pos[e.from] && pos[e.to]).map((e) => {
    const a = pos[e.from], b = pos[e.to];
    const x1 = a.x + a.w, y1 = a.y + a.h / 2, x2 = b.x, y2 = b.y + b.h / 2;
    const label = e.label ? `<text x="${(x1 + x2) / 2}" y="${(y1 + y2) / 2 - 4}" class="elabel" text-anchor="middle"><title>${escapeHtml(e.label)}</title>${escapeHtml(String(e.label).slice(0, 16))}${String(e.label).length > 16 ? "…" : ""}</text>` : "";
    return `<path d="M${x1},${y1} C${x1 + 26},${y1} ${x2 - 26},${y2} ${x2},${y2}" class="edge" marker-end="url(#arrow)"/>${label}`;
  }).join("");
  const boxes = nodes.map((n) => {
    const p = pos[n.id];
    const label = String(n.label).length > 26 ? `${String(n.label).slice(0, 25)}…` : n.label;
    const cls = nodeClass ? nodeClass(n) : "";
    const attrs = action ? ` data-learn="${action}" data-id="${escapeHtml(n.id)}" tabindex="0" role="button"` : "";
    return `<g class="gnode ${cls}"${attrs}><title>${escapeHtml(n.label)}</title>
      <rect x="${p.x}" y="${p.y}" width="${p.w}" height="${p.h}" rx="6"/>
      <text x="${p.x + 8}" y="${p.y + 19}">${escapeHtml(label)}</text>
      ${nodeSub ? `<text x="${p.x + 8}" y="${p.y + 36}" class="sub">${escapeHtml(nodeSub(n))}</text>` : ""}</g>`;
  }).join("");
  return `<div class="svg-wrap"><svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img">
    <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker></defs>${lines}${boxes}</svg></div>`;
}

function mapView(c) {
  const nodes = c.concepts.map((k) => ({ ...k, label: k.title }));
  const edges = c.concepts.flatMap((k) => k.prereqs.map((p) => ({ from: p, to: k.id })));
  const cls = (n) => (n.unavailable ? "unavailable" : n.passed || n.mastery >= 0.75 ? "passed"
    : n.unlocked ? "open" : "locked") + (n.id === L.conceptId ? " current" : "");
  const sub = (n) => (n.unavailable ? "no sources" : n.inferred ? "known, per diagnostic"
    : n.need != null ? `you: ${LEVELS[n.level] || "?"} · needs: ${LEVELS[n.need] || "?"}`
      : `${Math.round(n.mastery * 100)}% · ${n.built.length ? "built" : "not built"}`);
  return `<h4>Concept map <span class="hint">click a concept to open it</span></h4>
    ${svgGraph(nodes, edges, { nodeClass: cls, nodeSub: sub, action: "concept" })}`;
}

/* The focus: what to do next ---------------------------------------------------- */

function focusView(c) {
  if (L.graded) return gradedCard();
  const n = c.next || {};
  const d = L.diag;
  if (L.diagResult) return diagResultCard();
  if (d && ["conversation", "preparing", "probing"].includes(d.state)) return diagnosticView(d);
  if (d && d.state === "todo" && c.pretest === "todo" && !Object.values(c.concepts).some((k) => k.mastery > 0)) {
    return diagnosticOffer() + nextCard(c, n);
  }
  if (L.pretest?.state === "active" || c.pretest === "building" || c.pretest === "active") {
    return pretestView(c);
  }
  return nextCard(c, n);
}

function diagnosticOffer() {
  return `<div class="learn-card"><p><b>Lessons pitched at what you already know.</b> A short conversation
    about your background, then up to ten questions answered in your own words. Lessons then explain
    what you need and skip what you have; what you already know is marked known.</p>
    <button type="button" data-learn="diag-start">Start</button>
    <button type="button" class="link" data-learn="diag-skip">Skip — start from the beginning</button></div>`;
}

function diagnosticView(d) {
  if (d.state === "conversation" && d.pending) {
    return `<div class="learn-card"><p class="hint">Getting to know you · ${d.turns.length + 1} of up to 4</p>
      <p><b>${md(d.pending.question)}</b></p>
      <form id="learn-diag-reply" class="learn-new"><textarea id="learn-diag-text" rows="2"></textarea>
        <button type="submit">Answer</button></form>
      <button type="button" class="link" data-learn="diag-skip">Skip the diagnostic</button></div>`;
  }
  if (d.state === "preparing" || (d.state === "conversation" && !d.pending)) {
    return `<div class="learn-card"><p>Writing your questions from the papers…
      <span class="hint">${d.prepared} ready${d.preparing ? `, ${d.preparing} still being researched` : ""}.
      Each one is grounded by a short deep-research run, so this takes a few minutes.</span></p></div>`;
  }
  if (d.state === "probing" && d.current) {
    return `<form id="learn-diag-probe" class="learn-card">
      <p class="hint">Question ${d.asked + 1} of up to ${d.budget} · about “${md(d.current.title)}”</p>
      <p><b>${md(d.current.question)}</b></p>
      <textarea id="learn-diag-answer" rows="3" placeholder="A few sentences in your own words"></textarea>
      <button type="submit">Submit</button>
      <button type="button" class="link" data-learn="diag-noidea">I don't know</button></form>`;
  }
  return "";
}

function diagResultCard() {
  const r = L.diagResult;
  const level = r.self ? "Noted — we'll start from the beginning on this." : `Your answer shows: <b>${escapeHtml(LEVELS[r.level])}</b>.`;
  return `<div class="learn-card"><p>${level}</p>
    ${r.feedback ? `<p>${md(r.feedback)}</p>` : ""}
    ${r.reference ? `<p class="hint">A strong answer: ${md(r.reference)}</p>` : ""}
    <button type="button" data-learn="diag-continue">Continue</button></div>`;
}

function nextCard(c, n) {
  if (n.action === "done") {
    return `<div class="learn-card"><p><b>You have worked through every concept the sources could support.</b></p>
      ${c.due ? `<p>${c.due} item(s) are due for review.</p>` : "<p>Nothing is due for review right now.</p>"}</div>`;
  }
  if (n.action === "build") {
    const k = c.concepts.find((x) => x.id === n.concept);
    const stage = k?.build?.stage;
    if (stage === "error") {
      return `<div class="learn-card"><p class="error">Building “${escapeHtml(k.title)}” failed: ${escapeHtml(k.build.error)}</p>
        <button type="button" data-learn="build" data-id="${escapeHtml(n.concept)}">Try again</button></div>`;
    }
    if (!L.requested.has(n.concept) && !L.autostart) {
      return `<div class="learn-card"><p><b>Next: ${escapeHtml(k?.title || n.concept)}</b>
        <span class="hint">${escapeHtml(k?.summary || "")}</span></p>
        <button type="button" data-learn="build" data-id="${escapeHtml(n.concept)}">Build this lesson from the papers</button></div>`;
    }
    return `<div class="learn-card"><p>Preparing “${escapeHtml(k?.title || n.concept)}” from the papers…
      <span class="hint">${escapeHtml(stage ? `step: ${stage}` : "starting")}</span></p></div>`;
  }
  if (n.action === "lesson") {
    const k = c.concepts.find((x) => x.id === n.concept);
    return `<div class="learn-card"><p><b>Next: ${escapeHtml(k?.title || n.concept)}</b>
      <span class="hint">${escapeHtml(k?.summary || "")}</span></p>
      <button type="button" data-learn="concept" data-id="${escapeHtml(n.concept)}">Open the lesson</button></div>`;
  }
  if (n.item) return itemCard(n.item, n.action === "review" ? `Review — ${n.due} due` : "Practice");
  return "";
}

async function openConcept(cid) {
  if (L.view === "concept" && L.conceptId === cid) return;
  L.view = "concept";
  L.variant = "standard";
  L.conceptId = cid;
  L.claim = "";
  L.critique = null;
  L.partialOpen = "";
  L.showProfile = false;
  watchTrace(false);
  try {
    L.concept = await fetchConcept(cid);
  } catch (err) {
    L.error = err.message;
  }
  renderLearn();
}

/* The Profile tab: every prompt, retrieval and tool call behind a concept's build, polled
 * only while that tab is open, appending only what's new via a `seq` cursor. */
function watchTrace(on) {
  clearInterval(L.traceTimer);
  L.traceTimer = 0;
  if (on) L.traceTimer = setInterval(() => { if (L.showProfile) loadTrace(); }, 2500);
}

async function loadTrace() {
  if (!L.conceptId) return;
  if (L.traceConceptId !== L.conceptId) {
    L.traceConceptId = L.conceptId;
    L.trace = [];
    L.traceCursor = 0;
    L.traceOpen.clear();
  }
  const body = await get(`/api/learn/courses/${encodeURIComponent(L.id)}/concepts/${encodeURIComponent(L.conceptId)}/trace?since=${L.traceCursor}`);
  const rows = body.events || [];
  if (rows.length) {
    // A rebuild truncates the file server-side, so its first row is seq 1 again -- a cursor
    // built on the old build's higher numbers would otherwise see nothing new forever, and
    // an old row's "open" state would otherwise wrongly attach to a new row reusing its seq.
    if (rows[0].seq <= L.traceCursor) { L.trace = []; L.traceOpen.clear(); }
    L.trace.push(...rows);
    L.traceCursor = rows[rows.length - 1].seq;
  }
  renderLearn();
}

/* The same Profile view, for the course-level research a mapping course is running --
 * piggybacked on `refresh()`'s own 2.5s poll (schedulePoll already polls while
 * `status === "mapping"`) rather than a second timer, since it needs no independent
 * on/off switch the way the per-concept Profile tab does: it is the only thing shown
 * while a course is mapping, not one tab among several. */
async function loadMapTrace() {
  if (L.mapTraceId !== L.id) {
    L.mapTraceId = L.id;
    L.mapTrace = [];
    L.mapTraceCursor = 0;
  }
  const body = await get(`/api/learn/courses/${encodeURIComponent(L.id)}/trace?since=${L.mapTraceCursor}`);
  const rows = body.events || [];
  if (rows.length) {
    if (rows[0].seq <= L.mapTraceCursor) { L.mapTrace = []; }
    L.mapTrace.push(...rows);
    L.mapTraceCursor = rows[rows.length - 1].seq;
  }
}

/* Lesson ------------------------------------------------------------------------ */

function badge(certainty) {
  return `<span class="badge cert ${escapeHtml(certainty)}" title="${escapeHtml(CERTAINTY[certainty] || "")}">${escapeHtml(certainty)}</span>`;
}

function sourceLine(p) {
  const bits = [p.title, p.date, p.cited_by ? `cited ${p.cited_by}×` : "", p.journal_ref ? "published in a venue" : "preprint"];
  return bits.filter(Boolean).map(escapeHtml).join(" · ");
}

function allClaims(concept) {
  return concept.claims.concat((concept.expansions || []).flatMap((e) => e.claims || []));
}

/* The keys actually cited on the page right now -- the lesson being read plus whichever
 * expansions are shown under it -- in the order a reader meets them. Not every claim the
 * concept ever extracted: a paper the corpus turned up but that lost out to a stronger
 * source (or was cut for length) never made it into what the learner is reading. */
function citedKeys(concept) {
  const keys = [];
  const add = (k) => { if (!keys.includes(k)) keys.push(k); };
  for (const sec of activeLesson(concept)?.sections || []) for (const s of sec.sentences) s.claims.forEach(add);
  for (const e of shownExpansions(concept)) for (const s of expansionSentences(e)) s.claims.forEach(add);
  return keys;
}

/* One line per paper actually cited, arXiv-linked, in the order the lesson first cites it --
 * a reader's own "what did this rest on", not a claim-by-claim audit trail (that is what the
 * "All claims" list below is for). */
function bibliography(concept) {
  const by = new Map(allClaims(concept).map((c) => [c.key, c]));
  const seen = new Set();
  const papers = [];
  for (const key of citedKeys(concept)) {
    const p = by.get(key)?.passage;
    if (!p || seen.has(p.arxiv_id)) continue;
    seen.add(p.arxiv_id);
    papers.push(p);
  }
  if (!papers.length) return "";
  return `<h4>Sources</h4><ul class="refs">${papers.map((p) => `<li class="ref paper">
    <div class="ref-title"><a href="https://arxiv.org/abs/${escapeHtml(p.arxiv_id)}" target="_blank" rel="noopener">${escapeHtml(p.title)}</a></div>
    <div class="ref-where">${sourceLine(p)} · arXiv:${escapeHtml(p.arxiv_id)}</div></li>`).join("")}</ul>`;
}

/* Profile tab: every prompt, retrieval and tool call behind this concept's build ---------- */

function offsetLabel(ts, t0) {
  const s = Math.max(0, ts - t0);
  return s < 60 ? `+${s.toFixed(1)}s` : `+${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

function passageRefLi(p) {
  return `<li class="ref paper"><div class="ref-title">${escapeHtml(p.title || p.arxiv_id || "(untitled)")}</div>
    <div class="ref-body">${escapeHtml(p.preview || "")}</div>
    <div class="ref-where">arXiv:${escapeHtml(p.arxiv_id || "")}${p.section ? " · " + escapeHtml(p.section) : ""}</div></li>`;
}

/* (one-line summary shown always, HTML shown only once the row is expanded) per event type --
 * a type this file does not know about (e.g. the pipeline grows a new one) still renders,
 * as raw JSON, rather than disappearing silently. */
function traceDetail(r) {
  if (r.type === "llm_call") {
    return { summary: `${escapeHtml(r.purpose || "")} <span class="hint">${r.duration_ms ?? "?"}ms</span>`,
      detail: `<p class="hint">System prompt</p><pre>${escapeHtml(r.system || "")}</pre>
        <p class="hint">Prompt</p><pre>${escapeHtml(r.prompt || "")}</pre>
        <p class="hint">Response</p><pre>${escapeHtml(r.response || "(nothing survived verification)")}</pre>` };
  }
  if (r.type === "search") {
    return { summary: `search “${escapeHtml((r.query || "").slice(0, 90))}” <span class="hint">${(r.kept || []).length} of ${r.hits ?? "?"} kept</span>`,
      detail: `<ul class="refs">${(r.kept || []).map(passageRefLi).join("") || `<li class="hint">nothing kept from this search</li>`}</ul>` };
  }
  if (r.type === "coverage_probe") {
    const cov = r.coverage || {};
    return { summary: `coverage probe <span class="hint">${cov.papers ?? "?"} paper(s) / ${cov.chunks ?? "?"} chunk(s) — ${escapeHtml(r.tier || "")}</span>`,
      detail: `<p class="hint">Query: ${escapeHtml(r.query || "")}</p>` };
  }
  if (r.type === "facets") {
    return { summary: `facets chosen <span class="hint">${(r.queries || []).length}, budget "${escapeHtml(r.budget?.tier || "")}"</span>`,
      detail: `<ul class="refs">${(r.queries || []).map((q) => `<li class="ref">${escapeHtml(q)}</li>`).join("")}</ul>` };
  }
  if (r.type === "citation_walk") {
    return { summary: `citation-graph walk <span class="hint">${(r.candidates || []).length} paper(s) → ${(r.kept || []).length} passage(s)</span>`,
      detail: `<p class="hint">Seed paper(s): ${(r.seed_papers ? r.seed_papers : [r.seed_paper]).filter(Boolean).map(escapeHtml).join(", ") || "(none)"}</p>
        <p class="hint">Neighbour papers searched: ${(r.candidates || []).map(escapeHtml).join(", ") || "(none)"}</p>
        <ul class="refs">${(r.kept || []).map(passageRefLi).join("") || `<li class="hint">nothing kept</li>`}</ul>` };
  }
  if (r.type === "full_paper_read") {
    return { summary: `full-paper read <span class="hint">${escapeHtml(r.arxiv_id || "")} — ${r.chunks ?? "?"} chunk(s)</span>`, detail: "" };
  }
  if (r.type === "deliverable_section") {
    return { summary: `wrote “${escapeHtml((r.label || "").slice(0, 90))}” <span class="hint">${(r.text || "").length} char(s)</span>`,
      detail: `<pre>${escapeHtml(r.text || "")}</pre>` };
  }
  if (r.type === "topic_graph_written" || r.type === "lesson_written") {
    // `exit_reason` is why synthesizer.run's loop actually stopped -- see that
    // function's own docstring on its three conditions. Named plainly here rather than
    // echoing the raw "finished"/"idle"/"max_rounds" value, since that vocabulary is an
    // implementation detail this reader never needs to have learned.
    const why = { finished: "the research judged itself done",
                 idle: "no further research was proposed",
                 max_rounds: "hit its round limit — this may be an incomplete answer" }[r.exit_reason]
      || escapeHtml(r.exit_reason || "");
    return { summary: `research finished <span class="hint">${r.rounds ?? "?"} round(s) — ${why}</span>`,
      detail: `<p class="hint">${r.goals_done ?? 0} goal(s) done, ${r.goals_failed ?? 0} failed`
        + `${r.degraded ? ` · degraded: ${escapeHtml(r.degraded_because || "")}` : ""}</p>` };
  }
  return { summary: escapeHtml(r.type || "event"), detail: `<pre>${escapeHtml(JSON.stringify(r, null, 1))}</pre>` };
}

/* `renderLearn()` fully replaces this DOM subtree on every poll tick while a build is live
 * (every 2.5s -- watchTrace), which would otherwise re-collapse a row the moment someone
 * opened it to read a prompt. `L.traceOpen` (keyed by each event's own `seq`, unique within
 * a build) survives the re-render; the capture-phase `toggle` listener in bindLearn keeps it
 * in sync with what the reader actually has open. */
function traceRow(r, t0) {
  const { summary, detail } = traceDetail(r);
  return `<details class="trace-row" data-seq="${r.seq}" ${L.traceOpen.has(r.seq) ? "open" : ""}>
    <summary>${summary} <span class="hint">${offsetLabel(r.ts, t0)}</span></summary>
    ${detail}</details>`;
}

/* Grouped by phase, not left in raw arrival order: concurrent per-facet rounds (claims.build's
 * `one`) and per-section research/writing (depth.py's `research`/`write_section`, each its own
 * gathered task) interleave their events in the file by timestamp, which would otherwise
 * scatter one facet's or section's story across the page. Phases appear in the order the
 * build first reached them. */
/* `rows` is `L.trace` for a concept's own Profile tab, or `L.mapTrace` for the
 * course-level research view a mapping course shows (see loadMapTrace) -- the same
 * renderer either way, distinguishing only by which marker event (if any) opens the
 * log, since that is the one thing that differs between the two call sites. */
function profileView(rows) {
  const started = rows.find((r) => r.type === "build_start" || r.type === "map_start");
  const t0 = rows.find((r) => r.type !== "build_start" && r.type !== "map_start")?.ts ?? started?.ts ?? 0;
  const groups = [];
  const index = new Map();
  for (const r of rows) {
    if (r.type === "build_start" || r.type === "map_start") continue;
    if (!index.has(r.phase)) { index.set(r.phase, groups.length); groups.push({ phase: r.phase, rows: [] }); }
    groups[index.get(r.phase)].rows.push(r);
  }
  const marker = started?.type === "build_start"
    ? `Building the <b>${escapeHtml(started.variant)}</b> lesson${started.forced ? " (forced rebuild)" : ""} — `
    : started?.type === "map_start" ? "Researching the course's concept map — " : "";
  const header = `<p class="hint">${marker}${rows.length - (started ? 1 : 0)} event(s) so far.</p>`;
  // The deliverable itself, growing section by section as `_write_deliverable` actually
  // writes each one (see synthesizer.py's `on_section`) -- shown up front and unfolded,
  // not buried in a collapsed trace-row, since watching the topic graph / lesson prose
  // actually being written is the whole point of this view while a research-driven build
  // is live. Concatenated by `index` (the order sections are really written in), not
  // arrival order in `rows` -- both happen to match today since `_write_deliverable`
  // writes sequentially, but `index` is the one that is actually guaranteed to.
  const written = rows.filter((r) => r.type === "deliverable_section").sort((a, b) => a.index - b.index);
  const live = written.length ? `<section class="trace-live">
    <h4>What's being written</h4>
    <pre class="trace-live-text">${written.map((r) => escapeHtml(r.text || "")).join("\n\n")}</pre></section>` : "";
  const body = groups.length
    ? groups.map((g) => `<section class="trace-phase"><h4>${escapeHtml(g.phase)}</h4>${g.rows.map((r) => traceRow(r, t0)).join("")}</section>`).join("")
    : `<p class="hint">${rows.length ? "Nothing matches that filter." : "Nothing traced yet — build or refresh this concept to see its prompts, retrieval and tool calls here."}</p>`;
  return header + live + body;
}

/* Versions of the lesson ------------------------------------------------------------ */

const PRESET = { tldr: "TL;DR", standard: "Standard", thorough: "Thorough (about 5 pages)",
  "compress-low": "Shorter", "compress-med": "Much shorter", "compress-high": "Essentials only",
  reorganized: "Reorganized" };
/* Compressed versions are cut down from the standard lesson's own text, discarding more detail
 * at each level -- lara/learn/compress.py. */
const COMPRESSED = ["compress-low", "compress-med", "compress-high"];

function activeLesson(concept) {
  return concept.lessons?.[L.variant] || null;
}

function variantLabel(key) {
  return PRESET[key] || (key.startsWith("pages-") ? `${key.slice(6)} pages` : key);
}

function variantBar(concept) {
  const keys = ["tldr", "standard", "reorganized", ...COMPRESSED, "thorough", ...Object.keys(concept.lessons || {}).filter((k) => k.startsWith("pages-"))];
  const buttons = keys.map((k) => `<button type="button" class="length-tab ${k === L.variant ? "active" : ""}" data-learn="variant"
      data-variant="${escapeHtml(k)}" title="${concept.lessons?.[k] ? "written" : "not written yet"}">${escapeHtml(variantLabel(k))}${concept.lessons?.[k] ? "" : " ·"}</button>`).join("");
  return `<div class="length-bar variant-bar">${buttons}
    <span class="hint">or</span> <input id="learn-pages" type="number" min="1" max="20" value="${L.customPages || 8}" aria-label="pages">
    <button type="button" class="link" data-learn="write-pages">write a version this many pages long</button></div>`;
}

/* Deeper versions research the corpus again for each section, so they take minutes, not
 * seconds -- they are written on request, never automatically. */
function writeCard(concept) {
  const b = concept.build || {};
  const writing = buildingNow(concept) && b.variant === L.variant;
  if (writing) {
    return `<div class="learn-card"><p>Writing the ${escapeHtml(variantLabel(L.variant))} version…
      <span class="hint">${escapeHtml(b.detail || "starting")}</span></p></div>`;
  }
  const failed = b.stage === "error" && b.variant === L.variant;
  const compressed = COMPRESSED.includes(L.variant);
  const reorganized = L.variant === "reorganized";
  const eta = L.variant === "tldr" ? "about half a minute"
    : compressed ? "a minute or two"
      : reorganized ? "a few minutes: it plans an outline, then rewrites the lesson section by section"
        : "a few minutes: it searches the papers again for every section";
  const what = compressed
    ? "It is cut down from the standard lesson's own text, deliberately dropping detail; nothing new is added."
    : reorganized
      ? "It rewrites the standard lesson in the order a learner should meet it, merging repeats and trimming material that belongs to the course's other lessons. Nothing is researched again and nothing new is added."
      : "It only says what the sources support, so it may come out shorter than asked.";
  const spec = L.variant.startsWith("pages-") ? `data-variant="pages" data-pages="${escapeHtml(L.variant.slice(6))}"` : `data-variant="${escapeHtml(L.variant)}"`;
  return `<div class="learn-card"><p>The ${escapeHtml(variantLabel(L.variant))} version has not been written yet.
    <span class="hint">Takes ${eta}. ${what}</span></p>
    ${failed ? `<p class="error">${escapeHtml(b.error)}</p>` : ""}
    <button type="button" data-learn="write" ${spec}>Write it</button></div>`;
}

/* Optional, non-blocking (unlike the plan gate above): only offered on the research-
 * driven standard lesson, which alone has a persisted synthesis graph behind it to
 * resume (concept.research_driven -- see pipeline.build_concept's "claims" stage). */
function reviseLessonCard(concept) {
  const revising = buildingNow(concept) && concept.build?.stage === "revising_lesson";
  if (revising) {
    return `<div class="learn-card"><p>Revising the lesson from your feedback…</p></div>`;
  }
  const failed = concept.build?.stage === "error" && L.requested.has(concept.id);
  return `<div class="learn-card">
    <form id="learn-lesson-revise" class="learn-new" data-id="${escapeHtml(concept.id)}">
      <label class="hint">Not quite right? Say what you'd like changed and it will be rewritten.</label>
      <textarea id="learn-lesson-feedback" rows="2" placeholder='e.g. "Go deeper on how the KL penalty is computed"'></textarea>
      <button type="submit">Request changes</button>
    </form>
    ${failed ? `<p class="error">${escapeHtml(concept.build.error)}</p>` : ""}
    ${historyView(concept.lesson_history, "Earlier version")}</div>`;
}

function claimCard(concept) {
  const claim = allClaims(concept).find((x) => x.key === L.claim);
  if (!claim) return "";
  const p = claim.passage;
  const notes = claim.conflicts.map((x) => `<li>${escapeHtml(x.relation === "scope" ? "differs by setting" : "conflicts")} with ${escapeHtml(x.with)}: ${escapeHtml(x.note)}</li>`).join("");
  return `<div class="learn-card claim">
    <p>${badge(claim.certainty)} ${md(claim.text)}</p>
    ${claim.conditions ? `<p class="hint">Holds when: ${md(claim.conditions)}</p>` : ""}
    ${notes ? `<ul class="hint">${notes}</ul>` : ""}
    <p class="hint">${sourceLine(p)} · <a href="https://arxiv.org/abs/${escapeHtml(p.arxiv_id)}" target="_blank" rel="noopener">arXiv:${escapeHtml(p.arxiv_id)}</a></p>
    <blockquote>${md(p.text.slice(0, 700))}${p.text.length > 700 ? "…" : ""}</blockquote>
    ${claim.flags?.length ? `<p class="hint">Flagged ${claim.flags.length}× — re-checked against this passage.</p>` : ""}
    <button type="button" data-learn="flag" data-claim="${escapeHtml(claim.key)}">This looks wrong — re-check it</button></div>`;
}

/* A deep lesson can put thirty sentences in one section; they are read in paragraphs. */
const PARAGRAPH = 5;

function paragraphs(sentences, size) {
  const out = [];
  for (let i = 0; i < sentences.length; i += size) out.push(sentences.slice(i, i + size));
  return out;
}

/* Visuals are generated per lesson section server-side (see lara/learn/visuals.py) but
 * carry no section index of their own -- only the claim keys they rest on. So placement is
 * done here, by the same signal: the section whose sentences cite the most of a visual's
 * claims is where it belongs. A visual that overlaps no section in the lesson actually
 * showing (built against a different variant's section boundaries, or from before this
 * scheme existed) falls back to the end, same as every visual used to render. */
function bestSectionFor(v, sections) {
  let best = -1, bestScore = 0;
  sections.forEach((sec, i) => {
    const cited = new Set(sec.sentences.flatMap((x) => x.claims));
    const score = v.claims.filter((k) => cited.has(k)).length;
    if (score > bestScore) { best = i; bestScore = score; }
  });
  return best;
}

function visualsBySection(concept, sections) {
  const by = Array.from({ length: sections.length }, () => []);
  const leftover = [];
  (concept.visuals || []).forEach((v) => {
    const i = bestSectionFor(v, sections);
    (i < 0 ? leftover : by[i]).push(v);
  });
  return { by, leftover };
}

function visualCard(v) {
  if (v.kind === "chart") return chartSvg(v);
  if (v.kind === "diagram") {
    return `<div class="learn-card"><p><b>${md(v.title)}</b> <span class="hint">every relation is stated by a source claim: ${v.claims.map(escapeHtml).join(", ")}</span></p>
      ${svgGraph(v.nodes.map((n) => ({ ...n })), v.edges)}</div>`;
  }
  if (v.kind === "pseudocode") return pseudocodeCard(v);
  if (v.kind === "figure") return figureCard(v);
  return "";
}

/* Not synthesized like the other three kinds -- the image itself, from wherever the cited
 * claim's own passage came from (see figures_in in lara/learn/visuals.py). A broken/expired
 * hotlink degrades to a line of text (see the capturing "error" listener in bindLearn)
 * rather than a broken-image icon filling the card. */
function figureCard(v) {
  return `<div class="learn-card figure">
    <img src="${escapeHtml(v.src)}" alt="${escapeHtml(v.caption || v.title)}" loading="lazy">
    <p class="hint">${md(v.caption)} <span class="hint">— from
      <a href="https://arxiv.org/abs/${escapeHtml(v.arxiv_id)}" target="_blank" rel="noopener">arXiv:${escapeHtml(v.arxiv_id)}</a></span></p>
  </div>`;
}

function pseudocodeCard(v) {
  const lines = v.steps.map((s) => `<li style="--depth: ${s.depth}"><span class="pc-line">${md(s.text)}</span>
    <sup class="ck ${s.claim === L.claim ? "on" : ""}" data-learn="claim" data-claim="${escapeHtml(s.claim)}">${escapeHtml(s.claim)}</sup></li>`).join("");
  return `<div class="learn-card"><p><b>${md(v.title)}</b> <span class="hint">every step is stated by a source claim: ${v.claims.map(escapeHtml).join(", ")}</span></p>
    <ol class="pseudocode">${lines}</ol></div>`;
}

/* A jump nav is only worth showing once there is more than one heading to jump between --
 * a TL;DR or a short standard lesson is one section and needs no map of itself. */
function sectionNav(sections) {
  if (sections.length < 2) return "";
  return `<nav class="lesson-nav">${sections.map((sec, i) =>
    `<button type="button" data-learn="jump" data-section="${i}">${md(sec.heading || `Section ${i + 1}`)}</button>`).join("")}</nav>`;
}

/* The claim card opens under the paragraph whose chip was clicked, so the source is beside the
 * sentence it backs rather than a long scroll away. */
function lessonBody(concept) {
  const lesson = activeLesson(concept);
  if (!lesson) return concept.lesson ? writeCard(concept) : `<p class="hint">Not built yet.</p>`;
  if (lesson.insufficient) return `<div class="learn-card"><p class="warn">${md(lesson.message)}</p></div>`;
  const s = lesson.stats;
  // The standard lesson researches an outline just like a "thorough" or N-page one, but it
  // never asked for a page count -- a reader who didn't request a length should not be told
  // how it compares to one. `dropped_sections` stays visible either way: a section the corpus
  // could not support is worth knowing about regardless of how the lesson was requested.
  const isVariant = lesson.variant && lesson.variant !== "standard";
  const squeezed = lesson.compression
    ? `<p class="hint">${lesson.compression.words} words, ${Math.round(lesson.compression.ratio * 100)}% of the standard lesson's ${lesson.compression.source_words}.</p>`
    : lesson.reorganized
      ? `<p class="hint">Reorganized from ${lesson.reorganized.source_sections} sections (${lesson.reorganized.source_words} words) into ${lesson.reorganized.sections} (${lesson.reorganized.words} words).</p>` : "";
  const length = (isVariant && lesson.target_pages
    ? `<p class="hint">About ${lesson.achieved_pages} page(s), for the ${lesson.target_pages} asked for.
        ${lesson.shortfall ? `<span class="warn">${md(lesson.shortfall)}</span>` : ""}</p>` : "")
    + (lesson.dropped_sections?.length
      ? `<p class="hint">Not enough sources for: ${lesson.dropped_sections.map(escapeHtml).join("; ")}.</p>` : "");
  const trust = squeezed + length + `<p class="hint trust" title="Every sentence is re-checked against the claims it cites; ones that fail are rewritten once, then dropped.">
    ${s.grounded_pct}% of sentences verified on the first pass · ${s.repaired} rewritten · ${s.dropped} dropped
    ${lesson.reader ? ` · a simulated reader flagged ${lesson.reader.flagged}, ${lesson.reader.rewritten} rewritten` : ""}
    ${lesson.stale ? ' · <span class="warn">a source was withdrawn — rewrite this version to refresh</span>' : ""}</p>`;
  const { by: visualsFor, leftover } = visualsBySection(concept, lesson.sections);
  const sections = lesson.sections.map((sec, i) => `
    <div id="learn-sec-${i}">
    ${sec.heading ? `<h4>${md(sec.heading)}</h4>` : ""}
    ${paragraphs(sec.sentences, PARAGRAPH).map((group) => `<p class="lesson-p" data-section="${i}">${group.map((x) => `<span class="lsent" data-claims="${escapeHtml(x.claims.join(","))}" data-text="${escapeHtml(x.text)}">${md(x.text)}${x.claims.map((k) =>
      `<sup class="ck ${k === L.claim ? "on" : ""}" data-learn="claim" data-claim="${escapeHtml(k)}">${escapeHtml(k)}</sup>`).join("")}</span>`).join(" ")}</p>`).join("")}
    ${visualsFor[i].map(visualCard).join("")}
    </div>
    ${sec.sentences.some((x) => x.claims.includes(L.claim)) ? claimCard(concept) : ""}
    ${askPanel(i)}${expansionsFor(concept, i)}`).join("");
  return trust + readAloudBar() + sectionNav(lesson.sections) + sections + leftover.map(visualCard).join("");
}

/* Read aloud ------------------------------------------------------------------------ */
/* Reads every .lsent in the currently-rendered lesson, in order, highlighting each as it
 * plays. State lives outside L: a MediaRecorder-style handle and an Audio element are not
 * serializable and have no business surviving a JSON round trip, and re-rendering the
 * whole panel on every state change here would also wipe an in-progress mic recording
 * elsewhere on the page (see the mic handlers in bindLearn). `readToken` invalidates a
 * running loop on stop/navigate without needing to cancel an in-flight fetch or audio. */
let readState = "idle";     // idle | playing | paused
let readToken = 0;
let pauseWaiter = null;

/* An in-progress mic recording (the ask panel's voice input) -- also kept outside L for
 * the same reason: it is not serializable, and its handlers mutate their button directly
 * rather than calling renderLearn(). Null whenever nothing is being recorded. */
let micHandle = null;

function readAloudBar() {
  if (!VOICE.ttsAvailable()) return "";
  if (readState === "idle") {
    return `<p class="read-bar"><button type="button" data-learn="read-start">🔊 Read this lesson aloud</button></p>`;
  }
  return `<p class="read-bar">
    ${readState === "playing"
      ? `<button type="button" data-learn="read-pause">⏸ Pause</button>`
      : `<button type="button" data-learn="read-resume">▶ Resume</button>`}
    <button type="button" class="link" data-learn="read-stop">⏹ Stop</button></p>`;
}

function readSentences() {
  return [...document.querySelectorAll("#learn-main .lesson-p .lsent")];
}

function highlightReading(el) {
  document.querySelectorAll("#learn-main .lsent.reading-now")
    .forEach((n) => { if (n !== el) n.classList.remove("reading-now"); });
  el.classList.add("reading-now");
  el.scrollIntoView({ behavior: "smooth", block: "center" });
}

function clearReadingHighlight() {
  document.querySelectorAll("#learn-main .lsent.reading-now")
    .forEach((n) => n.classList.remove("reading-now"));
}

/* Only the read-bar itself needs to change, the same reasoning as the mic handlers below:
 * a full renderLearn() here would also blow away anything typed in an open ask panel. */
function refreshReadBar() {
  const bar = document.querySelector("#learn-main .read-bar");
  if (bar) bar.outerHTML = readAloudBar();
}

async function waitIfPaused() {
  if (readState !== "paused") return;
  await new Promise((resolve) => { pauseWaiter = resolve; });
}

async function startReading() {
  const sents = readSentences();
  if (!sents.length) return;
  const token = ++readToken;
  readState = "playing";
  refreshReadBar();

  const textOf = (el) => el.dataset.text || el.textContent;
  let next = VOICE.fetchSpeech(textOf(sents[0])).catch((err) => ({ __error: err }));
  for (let i = 0; i < sents.length; i++) {
    if (token !== readToken) return;
    await waitIfPaused();
    if (token !== readToken) return;

    const clip = await next;
    if (token !== readToken) return;
    if (clip && clip.__error) {
      L.error = `Could not read this aloud: ${clip.__error.message}`;
      readState = "idle";
      clearReadingHighlight();
      renderLearn();
      return;
    }
    next = i + 1 < sents.length
      ? VOICE.fetchSpeech(textOf(sents[i + 1])).catch((err) => ({ __error: err }))
      : Promise.resolve(null);

    highlightReading(sents[i]);
    const { done } = VOICE.playBlob(clip);
    await done;
  }
  if (token === readToken) {
    readState = "idle";
    clearReadingHighlight();
    refreshReadBar();
  }
}

function pauseReading() {
  readState = "paused";
  VOICE.stopPlayback();
  refreshReadBar();
}

function resumeReading() {
  readState = "playing";
  refreshReadBar();
  if (pauseWaiter) { const w = pauseWaiter; pauseWaiter = null; w(); }
}

function stopReading() {
  readToken++;                    // invalidates the running loop at its next check
  readState = "idle";
  VOICE.stopPlayback();
  if (pauseWaiter) { const w = pauseWaiter; pauseWaiter = null; w(); }
  clearReadingHighlight();
  refreshReadBar();
}

/* Deep lessons find many conflicts; the first few are shown and the rest folded away. */
const CONFLICTS_SHOWN = 4;

function conflictsView(concept) {
  if (!concept.conflicts.length) return "";
  const row = (x) => `<li><b>${x.relation === "scope" ? "Differs by setting" : "Papers disagree"}</b>
    ${x.note ? `— ${md(x.note)}` : ""}
    <ul>${x.sides.map((s) => `<li>${md(s.text)} <span class="hint">(${escapeHtml(s.date)}${s.conditions ? `; ${escapeHtml(s.conditions)}` : ""})</span></li>`).join("")}</ul></li>`;
  const first = concept.conflicts.slice(0, CONFLICTS_SHOWN).map(row).join("");
  const rest = concept.conflicts.slice(CONFLICTS_SHOWN);
  return `<h4>The disagreement, side by side</h4><ul class="conflicts">${first}</ul>
    ${rest.length ? `<details><summary>${rest.length} more</summary><ul class="conflicts">${rest.map(row).join("")}</ul></details>` : ""}`;
}

function planCard(concept) {
  const plan = concept.plan;
  if (!plan) return "";
  const by = (t) => plan.items.filter((i) => i.treatment === t && !i.uncovered).map((i) => md(i.title));
  const explained = [...by("section"), ...by("refresher"), ...by("intuition")];
  const assumed = by("use");
  const lines = [];
  if (explained.length) lines.push(`Explains, for you: ${explained.join(", ")}.`);
  if (assumed.length) lines.push(`Assumes you know: ${assumed.join(", ")}.`);
  const level = concept.level && concept.need != null
    ? `<span class="hint">You: ${escapeHtml(LEVELS[concept.level.level])} · this course needs: ${escapeHtml(LEVELS[concept.need])}</span>` : "";
  const uncovered = plan.uncovered?.length
    ? `<p class="warn">The papers in the corpus don't explain ${plan.uncovered.map(md).join(", ")}. This lesson
        relies on ${plan.uncovered.length > 1 ? "them" : "it"}, so it's worth looking up in an introductory source.</p>` : "";
  return lines.length || uncovered ? `<div class="learn-card plan">${level}${lines.map((l) => `<p class="hint">${l}</p>`).join("")}${uncovered}</div>` : "";
}

function checksCard(concept) {
  if (L.checkResult) {
    const r = L.checkResult;
    return `<div class="learn-card"><p>${r.self ? "Noted." : `Your answer shows: <b>${escapeHtml(LEVELS[r.level])}</b>.`}</p>
      ${r.feedback ? `<p>${md(r.feedback)}</p>` : ""}
      <button type="button" data-learn="check-continue">Continue</button></div>`;
  }
  const c = (concept.checks || [])[0];
  if (!c || concept.lesson) return "";
  return `<form id="learn-check" class="learn-card" data-id="${escapeHtml(c.concept)}">
    <p class="hint">Quick check before this lesson · “${md(c.title)}” — so the lesson isn't written on an old guess</p>
    <p><b>${md(c.question)}</b></p>
    <textarea id="learn-check-answer" rows="2"></textarea>
    <button type="submit">Submit</button>
    <button type="button" class="link" data-learn="check-noidea" data-id="${escapeHtml(c.concept)}">I don't know</button></form>`;
}

function conceptView(concept) {
  // concept.lesson alone, not build.stage === "done": the lesson stage is what actually sets
  // it (even an "insufficient" lesson is a real object), and it becomes true the moment that
  // one stage finishes -- while quiz/topics/visuals may still be building -- which is exactly
  // when a lesson should start showing. Trusting build.stage instead, as this used to, meant
  // a concept whose file failed to get written (see build_concept's shared-reuse fix) could
  // say "done" with nothing to show and no way to retry: notBuilt below never rendered because
  // built was already true, so the "build it" button had nowhere to appear.
  const built = Boolean(concept.lesson);
  const notBuilt = !built ? `<div class="learn-card">${buildingNow(concept)
    ? `<p>Building…</p>${buildProgress(concept)}`
    : `<p>This concept has not been built yet.</p><button type="button" data-learn="build" data-id="${escapeHtml(concept.id)}">Build it from the papers</button>`}</div>` : "";
  const unanswered = (concept.topics || []).filter((t) => !t.answer);
  const gated = built && unanswered.length > 0;
  const list = concept.claims.map((k) => `<li><span class="ck ${k.key === L.claim ? "on" : ""}" data-learn="claim" data-claim="${escapeHtml(k.key)}">${escapeHtml(k.key)}</span> ${badge(k.certainty)} ${md(k.text)}${k.withdrawn ? ' <span class="err">withdrawn</span>' : ""}</li>`).join("");
  const s = concept.stats || {};
  const lesson = gated ? "" : `${backgroundLinks(concept)}${concept.lesson ? variantBar(concept) : ""}${lessonBody(concept)}${claimShownInLesson(concept) ? "" : claimCard(concept)}${conflictsView(concept)}${bibliography(concept)}
    ${concept.lesson && !concept.lesson.insufficient ? critiqueBox(concept) : ""}
    ${concept.lesson ? `<p><button type="button" data-learn="read" data-id="${escapeHtml(concept.id)}">I've read this — start practice</button>
      <button type="button" class="link" data-learn="known" data-id="${escapeHtml(concept.id)}">I already know this</button>
      <button type="button" class="link" data-learn="build" data-id="${escapeHtml(concept.id)}" data-force="1">Refresh from the corpus</button></p>` : ""}
    ${concept.lesson && !concept.lesson.insufficient && concept.research_driven && L.variant === "standard" ? reviseLessonCard(concept) : ""}
    <details><summary>All ${concept.claims.length} claims and their sources${s.unfaithful_dropped ? ` · ${s.unfaithful_dropped} extractions dropped as unfaithful` : ""}</summary><ul class="claims">${list}</ul></details>`;
  // The Profile tab (full prompt/response and retrieval detail) is separate from the
  // existing "How this lesson was built" trace card above (traceView): that one is the
  // claims stage's own coarse, always-visible summary; this is the deep, opt-in view across
  // every stage. Shown whenever there is anything to build from -- a concept mid-build has
  // claims (or at least a claims-stage in progress) before it has a lesson.
  const tabs = built || concept.claims.length || buildingNow(concept) ? `<div class="learn-tabs">
    <button type="button" class="${L.showProfile ? "" : "on"}" data-learn="profile-off">Lesson</button>
    <button type="button" class="${L.showProfile ? "on" : ""}" data-learn="profile-on">Profile</button></div>` : "";
  if (L.showProfile) {
    return `<div class="learn-concept">
      <button type="button" class="link" data-learn="close-concept">← back to your path</button>
      <h3>${md(concept.title)}</h3>${tabs}${profileView(L.trace)}</div>`;
  }
  // Not gated behind `built`/`gated`: the trace is written round by round as the claims stage
  // runs (see pipeline.build_concept's on_event), so it is worth showing -- open by default --
  // the moment a poll picks up the first of it, well before the lesson itself exists.
  return `<div class="learn-concept">
    <button type="button" class="link" data-learn="close-concept">← back to your path</button>
    <h3>${md(concept.title)}</h3><p class="hint">${md(concept.summary)}
      ${concept.reused ? " · reused from an earlier course" : ""}</p>
    ${tabs}${checksCard(concept)}${planCard(concept)}${notBuilt}${traceView(concept)}${gated ? topicsGate(concept, unanswered) : ""}${lesson}</div>`;
}

/* The blocking gate: every topic the lesson leans on without teaching gets a yes/no/partial
 * answer before the lesson prose is shown at all, so a reader never meets an assumption they
 * cannot place without first saying so. */
function topicsGate(concept, unanswered) {
  const rows = unanswered.map((t) => {
    const open = L.partialOpen === t.id;
    return `<div class="topic-gate-row">
      <p><b>${md(t.title)}</b>${t.note ? ` <span class="hint">— ${md(t.note)}</span>` : ""}</p>
      <div class="topic-gate-buttons">
        <button type="button" data-learn="familiarity" data-id="${escapeHtml(concept.id)}" data-topic="${escapeHtml(t.id)}" data-answer="yes">I know this</button>
        <button type="button" data-learn="familiarity" data-id="${escapeHtml(concept.id)}" data-topic="${escapeHtml(t.id)}" data-answer="no">Not familiar</button>
        <button type="button" class="${open ? "on" : ""}" data-learn="familiarity-partial-open" data-topic="${escapeHtml(t.id)}">Partially…</button>
      </div>
      ${open ? `<div class="topic-gate-partial">
        <textarea id="topic-explain-${escapeHtml(t.id)}" rows="2" required placeholder="What do you already know, and what don't you? (required)"></textarea>
        <button type="button" data-learn="familiarity" data-id="${escapeHtml(concept.id)}" data-topic="${escapeHtml(t.id)}" data-answer="partial">Submit</button>
      </div>` : ""}
    </div>`;
  }).join("");
  return `<div class="learn-card topics-gate">
    <p><b>Before the lesson:</b> it leans on a few things without explaining them. Say what you already know about each, and we'll fill in just what's missing.</p>
    ${rows}</div>`;
}

/* Once past the gate, a "no" or "partial" topic's document is either ready to open (in its own
 * tab, so the lesson's own place is never lost) or still being written -- a "yes" is recorded
 * but nothing was generated for it, so it gets no link. */
function backgroundLinks(concept) {
  const shown = (concept.topics || []).filter((t) => t.answer === "no" || t.answer === "partial");
  if (!shown.length) return "";
  const row = (t) => {
    if (t.doc_status === "building") return `<li>${md(t.title)} <span class="hint">preparing…</span></li>`;
    const href = `/topic.html?course=${encodeURIComponent(L.id)}&concept=${encodeURIComponent(concept.id)}&topic=${encodeURIComponent(t.id)}`;
    return `<li><a href="${href}" target="_blank" rel="noopener">${md(t.title)}</a></li>`;
  };
  return `<div class="learn-card"><p><b>Background</b> <span class="hint">opens in a new tab, so you keep your place here</span></p>
    <ul class="background-links">${shown.map(row).join("")}</ul></div>`;
}

/* The build's own profiling view: what was searched for each facet, how much of it came from
 * plain similarity versus a citation-graph walk, and how the coverage probe sized the whole
 * effort. Reuses Deep Research's round/tag styling (.deep-round, .tag.cit, ...) -- the same
 * idea (how did the system research this) rendered the same way in both places. */
/* Written round by round as the claims stage runs (see pipeline.build_concept's on_event), so
 * this can show up while the concept is still building, not only once it is done -- the panel
 * opens itself and says "live" while that is happening, and a poll (already running every
 * 2.5s during a build, see schedulePoll) is what makes it grow without the learner doing
 * anything. `t.ms` (and the rest of the closing summary) only exists once the whole claims
 * stage has actually returned; its absence is exactly the signal that this is still live. */
function traceView(concept) {
  const t = concept.trace;
  if (!t || !t.coverage) return "";
  const cov = t.coverage, bud = t.budget || {};
  const rounds = t.rounds || [];
  const live = buildingNow(concept) && concept.build?.stage === "claims";
  const facetOrder = [];
  rounds.forEach((r) => { if (!facetOrder.includes(r.facet)) facetOrder.push(r.facet); });
  const roundCards = rounds.map((r) => {
    const tags = `${r.round > 1 ? `<span class="tag">round ${r.round}</span>` : ""}${
      r.citation_passages_kept ? '<span class="tag cit">citation graph</span>' : ""}${
      r.full_paper_read ? '<span class="tag cit">full paper</span>' : ""}`;
    const note = [`${r.dense_retrieved} passage${r.dense_retrieved === 1 ? "" : "s"} by similarity`,
      r.citation_papers_tried ? `${r.citation_passages_kept} of ${r.citation_papers_tried} citation neighbour(s) added a passage` : "",
      r.full_paper_read ? `read the whole of ${escapeHtml(r.full_paper_read)} -- one paper anchored this round` : ""]
      .filter(Boolean).join(" · ");
    return `<details class="deep-round"><summary><b>Facet ${facetOrder.indexOf(r.facet) + 1}</b>${tags}<span class="rq">${md(r.query)}</span>
      <span class="rstat" style="margin-left:auto;opacity:.7">${r.claims} claim${r.claims === 1 ? "" : "s"}</span></summary>
      <div class="deep-claims"><div class="deep-note">${note}</div></div></details>`;
  }).join("");
  const depth = [bud.citation_walk ? "citation walk on" : "citation walk off",
    bud.gap_round ? "follow-up round on" : "", bud.full_paper ? "full-paper read on" : ""].filter(Boolean).join(", ");
  const waiting = live && !rounds.length ? `<p class="hint">Searching…</p>` : "";
  const done = t.ms != null ? `<p class="hint">${t.claims} claim(s) from ${t.papers} paper(s) · ${t.comparisons} pairwise comparison(s) · ${t.ms}ms</p>` : "";
  return `<details class="learn-card" ${live ? "open" : ""}><summary><b>How this lesson was built</b>${live ? ' <span class="hint">· live</span>' : ""}</summary>
    <p class="hint">${cov.papers ?? 0} paper(s) / ${cov.chunks ?? 0} passage(s) touch this in the corpus →
      researched as <b>${escapeHtml(bud.tier || "")}</b> (${bud.facets || 0} facet(s) planned, ${depth})</p>
    ${waiting}${roundCards}${done}
  </details>`;
}

function claimShownInLesson(concept) {
  const inLesson = (activeLesson(concept)?.sections || []).some((sec) => sec.sentences.some((x) => x.claims.includes(L.claim)));
  return inLesson || shownExpansions(concept).some((e) => expansionSentences(e).some((x) => x.claims.includes(L.claim)));
}

/* Inline detail on highlighted text ------------------------------------------------- */

function expansionSentences(e) {
  return e.answer.sections.flatMap((sec) => sec.sentences);
}

/* An answer belongs under its paragraph only while the lesson is the version it was asked
 * about; after a rebuild the paragraph numbers no longer mean the same thing. */
function shownExpansions(concept) {
  return (concept.expansions || []).filter((e) => e.lesson_generated === activeLesson(concept)?.generated);
}

function expansionsFor(concept, section) {
  return shownExpansions(concept).filter((e) => e.section === section).map((e) => expansionView(e, concept)).join("");
}

function expansionView(e, concept) {
  const sents = expansionSentences(e).map((x) => `<span class="lsent">${md(x.text)}${x.claims.map((k) =>
    `<sup class="ck ${k === L.claim ? "on" : ""}" data-learn="claim" data-claim="${escapeHtml(k)}">${escapeHtml(k)}</sup>`).join("")}</span>`).join(" ");
  const st = e.answer.stats;
  const shown = expansionSentences(e).some((x) => x.claims.includes(L.claim));
  const quoted = `“${md(e.selection.slice(0, 70))}${e.selection.length > 70 ? "…" : ""}”`;
  const label = e.question ? `“${md(e.question)}”` : e.intent === "explain" ? `Explained: ${quoted}` : `More on ${quoted}`;
  return `<details class="expansion" ${L.claim && shown ? "open" : ""}>
    <summary>${label}</summary>
    <p class="hint">${e.searched ? "Found by searching the corpus for this" : "From the lesson's own sources"} ·
      ${st.kept_first_pass} of ${st.written} sentences verified first time${st.dropped ? ` · ${st.dropped} dropped` : ""}
      ${e.stale ? ' · <span class="warn">a source was withdrawn — treat with care</span>' : ""}</p>
    <p class="lesson-p">${sents}</p>${shown ? claimCard(concept) : ""}
    <button type="button" class="link danger" data-learn="del-expansion" data-id="${escapeHtml(e.id)}">remove</button></details>`;
}

function askPanel(section) {
  if (!L.ask || L.ask.section !== section) return "";
  const a = L.ask;
  return `<div class="learn-card ask-panel">
    <blockquote>${md(a.selection.slice(0, 400))}${a.selection.length > 400 ? "…" : ""}</blockquote>
    ${L.askResult ? `<p class="warn">${md(L.askResult)}</p>` : ""}
    ${L.asking ? `<p class="hint">Looking through the sources… <span class="hint">this can take up to half a minute</span></p>` : `
    <div class="ask-row"><textarea id="learn-ask-question" rows="2" placeholder="${a.intent === "explain" ? "What didn't make sense? Or leave blank" : "Ask something specific, or leave blank for more detail on this"}"></textarea>
    ${VOICE.sttAvailable() ? `<button type="button" class="mic" data-learn="mic-start" title="Ask by voice">🎙</button>` : ""}</div>
    <button type="button" data-learn="ask-go">${a.intent === "explain" ? "Explain it more simply" : "Get more detail"}</button>
    <button type="button" class="link" data-learn="ask-cancel">Cancel</button>`}</div>`;
}

/* A highlight inside one lesson paragraph puts an Explain button beside it. The selection is
 * captured now: opening the panel re-renders the page and the browser drops it. */
function popup() {
  let pop = document.getElementById("learn-pop");
  if (!pop) {
    pop = document.createElement("div");
    pop.id = "learn-pop";
    pop.innerHTML = `<button type="button" data-learn="ask-open" data-intent="explain">Explain this</button>
      <button type="button" data-learn="ask-open" data-intent="deeper">Go deeper</button>`;
    pop.style.display = "none";
    pop.addEventListener("mousedown", (e) => e.preventDefault());     // keep the selection
    document.body.appendChild(pop);
  }
  return pop;
}

function checkSelection() {
  const pop = popup();
  const sel = window.getSelection();
  const anchor = sel?.anchorNode;
  /* A triple-click anchors on the paragraph element itself and ends at the start of the next
   * block, so the paragraph comes from the anchor's element and the selection is clamped to it. */
  const start = anchor && (anchor.nodeType === 1 ? anchor : anchor.parentElement);
  const para = start?.closest?.(".lesson-p[data-section]");
  if (!L.open || L.view !== "concept" || !para || sel.isCollapsed) {
    L.pending = null;
    pop.style.display = "none";
    return;
  }
  const bounds = document.createRange();
  bounds.selectNodeContents(para);
  const range = sel.getRangeAt(0).cloneRange();
  if (range.compareBoundaryPoints(Range.START_TO_START, bounds) < 0) range.setStart(bounds.startContainer, bounds.startOffset);
  if (range.compareBoundaryPoints(Range.END_TO_END, bounds) > 0) range.setEnd(bounds.endContainer, bounds.endOffset);
  /* The citation chips are part of the text; the question is about the words. */
  const words = range.cloneContents();
  words.querySelectorAll("sup").forEach((n) => n.remove());
  const text = words.textContent.replace(/\s+/g, " ").trim();
  if (text.length < 4) {
    L.pending = null;
    pop.style.display = "none";
    return;
  }
  const claims = [...para.querySelectorAll(".lsent")].filter((el) => range.intersectsNode(el))
    .flatMap((el) => (el.dataset.claims || "").split(",").filter(Boolean));
  L.pending = { section: Number(para.dataset.section), selection: text.slice(0, 2000), claims: [...new Set(claims)] };
  const rect = range.getBoundingClientRect();
  pop.style.left = `${Math.max(8, Math.min(rect.left, window.innerWidth - 220))}px`;
  pop.style.top = `${Math.min(rect.bottom + 6, window.innerHeight - 40)}px`;
  pop.style.display = "flex";
}

async function askForDetail() {
  const a = L.ask;
  const question = (document.getElementById("learn-ask-question")?.value || "").trim();
  L.asking = true;
  L.askResult = "";
  renderLearn();
  try {
    const out = await send("POST", `${base()}/concepts/${encodeURIComponent(L.conceptId)}/expand`,
      { selection: a.selection, question, section: a.section, claims: a.claims, variant: L.variant, intent: a.intent || "deeper" }, { timeoutMs: 240000 });
    if (out.insufficient) {
      L.askResult = out.message;
    } else {
      L.ask = null;
      L.concept = await fetchConcept(L.conceptId);
    }
  } catch (err) {
    L.askResult = err.message;
  }
  L.asking = false;
  renderLearn();
}

function critiqueBox(concept) {
  const out = L.critique;
  const points = out ? (out.length ? out.map((p) => `<li class="${p.verdict}"><b>${escapeHtml(p.verdict)}</b> — “${md(p.statement)}” ${md(p.advice)}
    <span class="hint">${p.claims.map(escapeHtml).join(", ")}</span></li>`).join("")
    : `<li class="hint">The sources I have do not speak to anything you wrote — so I will not comment on it.</li>`) : "";
  return `<h4>Test your own thinking</h4><form id="learn-critique" class="learn-new" data-id="${escapeHtml(concept.id)}">
    <textarea id="learn-critique-text" rows="3" placeholder="Write how you would apply this, or a plan. I only comment where the sources support or contradict it."></textarea>
    <button type="submit">Check against the sources</button></form>${points ? `<ul class="critique">${points}</ul>` : ""}`;
}

/* Practice ---------------------------------------------------------------------- */

/* The server moves on to the next item the moment one is answered, so the item just answered
 * is remembered here and its result shown until the learner continues. */
function gradedCard() {
  const item = L.items[L.graded.id] || { question: "", id: L.graded.id };
  const graded = L.graded;
  return `<div class="learn-card"><p><b>${md(item.question)}</b></p>
    <p class="${graded.correct ? "ok" : "err"}"><b>${graded.correct ? "Correct." : "Not quite."}</b> Answer: ${md(graded.answer)}</p>
    ${graded.feedback ? `<p>${md(graded.feedback)}</p>` : ""}
    <p class="hint">Source: ${escapeHtml(graded.source?.title || "")} · arXiv:${escapeHtml(graded.source?.arxiv_id || "")}</p>
    <button type="button" data-learn="next-item">Continue</button></div>`;
}

function itemCard(item, title) {
  L.items[item.id] = item;
  const choices = item.type === "mcq" ? item.choices.map((c, i) => `<label class="choice"><input type="radio" name="learn-choice" value="${String.fromCharCode(65 + i)}"> ${String.fromCharCode(65 + i)}. ${md(c.replace(/^\(?[A-Da-d][).:]\s+/, ""))}</label>`).join("")
    : `<textarea id="learn-response" rows="2" placeholder="${item.type === "predict" ? "Predict it before you look" : "Your answer"}"></textarea>`;
  return `<form id="learn-item" class="learn-card" data-id="${escapeHtml(item.id)}">
    <p class="hint">${escapeHtml(title)}${item.type === "predict" ? " · predict, then see" : ""}</p><p><b>${md(item.question)}</b></p>
    ${choices}
    <p class="hint">How sure are you?
      <label><input type="radio" name="learn-conf" value="1"> guessing</label>
      <label><input type="radio" name="learn-conf" value="2" checked> think so</label>
      <label><input type="radio" name="learn-conf" value="3"> sure</label></p>
    <button type="submit">Submit</button></form>`;
}

function pretestView(c) {
  if (L.graded) return gradedCard();
  const p = L.pretest;
  if (!p || p.state === "building") {
    return `<div class="learn-card"><p>Building the diagnostic from the papers… <span class="hint">this reads each sampled concept's sources once</span></p></div>`;
  }
  const todo = p.items.find((i) => !p.answered.includes(i.id));
  if (!todo) return nextCard(c, c.next || {});
  return `<h4>Diagnostic <span class="hint">${p.answered.length} of ${p.items.length}</span></h4>${itemCard(todo, "Diagnostic")}`;
}

/* ── events ───────────────────────────────────────────────────────────────────── */

function diagAnswer(response) {
  act("Reading your answer", async () => {
    const out = await send("POST", `${base()}/diagnostic/answer`, { response });
    L.diag = out.diagnostic;
    L.diagResult = out.result;
  });
}

function checkAnswer(pid, response) {
  act("Reading your answer", async () => {
    const out = await send("POST", `${base()}/concepts/${encodeURIComponent(L.conceptId)}/checks/${encodeURIComponent(pid)}/answer`, { response });
    L.checkResult = out.result;
    L.concept = { ...L.concept, checks: out.checks };
  });
}

async function openCourse(id) {
  L.id = id;
  L.view = "path";
  L.conceptId = "";
  L.concept = null;
  L.graded = null;
  L.pretest = null;
  L.claim = "";
  L.diag = null;
  L.diagResult = null;
  L.checkResult = null;
  await loadLearn();
}

export function openLearn() {
  const view = document.getElementById("learn");
  if (!view) return;
  view.hidden = false;
  document.body.classList.add("learn-open");
  L.open = true;
  loadLearn();
}

export function closeLearn() {
  const view = document.getElementById("learn");
  if (!view) return;
  view.hidden = true;
  document.body.classList.remove("learn-open");
  L.open = false;
  L.showKnowledge = false;
  L.knowledge = null;
  clearTimeout(L.poll);
  const pop = document.getElementById("learn-pop");
  if (pop) pop.style.display = "none";
  stopReading();
  if (micHandle) { VOICE.abortRecording(micHandle); micHandle = null; }
}

export function bindLearn() {
  document.getElementById("learn-btn")?.addEventListener("click", openLearn);
  document.getElementById("learn-close")?.addEventListener("click", closeLearn);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && L.open && !L.ask) closeLearn(); });
  document.addEventListener("mouseup", (e) => {
    if (!e.target.closest?.("#learn-pop")) setTimeout(checkSelection, 0);
  });
  document.addEventListener("keyup", (e) => { if (e.key === "Shift" || e.key.startsWith("Arrow")) checkSelection(); });

  // `toggle` does not bubble, so this listens in the capture phase -- the only way to
  // delegate it from one handler instead of binding one per `<details>`, which a poll-driven
  // re-render would just have to redo every 2.5 seconds anyway. Keeps a Profile row's
  // expanded/collapsed state across the live poll -- see loadTrace/traceRow.
  document.addEventListener("toggle", (e) => {
    const seq = Number(e.target?.dataset?.seq);
    if (!e.target.classList?.contains("trace-row") || !seq) return;
    if (e.target.open) L.traceOpen.add(seq); else L.traceOpen.delete(seq);
  }, true);
  // "error" does not bubble, so this has to capture -- the only way to catch it from one
  // listener rather than one per <img>, which a re-render would leak more of every time.
  document.addEventListener("error", (e) => {
    const img = e.target;
    if (img.tagName !== "IMG" || !img.closest(".learn-card.figure")) return;
    img.replaceWith(Object.assign(document.createElement("p"),
      { className: "hint", textContent: "The figure could not be loaded from arXiv." }));
  }, true);

  document.addEventListener("submit", async (e) => {
    const f = e.target;
    if (f.id === "learn-new") {
      e.preventDefault();
      const goal = $("#learn-goal").value.trim();
      if (!goal) return;
      act("Reading your goal", async () => {
        const course = await send("POST", "/api/learn/courses", { goal });
        await openCourse(course.id);
      });
    } else if (f.id === "learn-answer") {
      e.preventDefault();
      const answer = $("#learn-answer-text").value.trim();
      if (answer) act("Re-planning", async () => { await send("POST", `${base()}/answer`, { answer }); await loadLearn(); });
    } else if (f.id === "learn-item") {
      e.preventDefault();
      const item = f.dataset.id;
      const choice = f.querySelector("input[name=learn-choice]:checked");
      const response = choice ? choice.value : ($("#learn-response")?.value || "");
      const conf = Number(f.querySelector("input[name=learn-conf]:checked")?.value || 2);
      act("Checking", async () => {
        const out = await send("POST", `${base()}/items/${encodeURIComponent(item)}/answer`, { response, confidence: conf });
        L.graded = { id: item, ...out.graded };
        L.course = { ...L.course, ...out.overview };
        if (L.pretest?.state === "active") await loadPretest();
      });
    } else if (f.id === "learn-diag-reply") {
      e.preventDefault();
      const answer = $("#learn-diag-text").value.trim();
      if (answer) act("Thinking", async () => { L.diag = await send("POST", `${base()}/diagnostic/reply`, { answer }); });
    } else if (f.id === "learn-diag-probe") {
      e.preventDefault();
      diagAnswer($("#learn-diag-answer").value);
    } else if (f.id === "learn-check") {
      e.preventDefault();
      checkAnswer(f.dataset.id, $("#learn-check-answer").value);
    } else if (f.id === "learn-critique") {
      e.preventDefault();
      const text = $("#learn-critique-text").value.trim();
      if (text) act("Checking against the sources", async () => {
        L.critique = (await send("POST", `${base()}/concepts/${encodeURIComponent(f.dataset.id)}/critique`, { text })).points;
        await refresh();
      });
    } else if (f.id === "learn-plan-revise") {
      e.preventDefault();
      const text = $("#learn-plan-feedback").value.trim();
      if (text) act("Requesting changes to the plan", async () => {
        await send("POST", `${base()}/plan/revise`, { text });
        await loadLearn();
      });
    } else if (f.id === "learn-lesson-revise") {
      e.preventDefault();
      const text = $("#learn-lesson-feedback").value.trim();
      const cid = f.dataset.id;
      if (text) {
        L.requested.add(cid);
        act("Requesting a revision", async () => {
          await send("POST", `${base()}/concepts/${encodeURIComponent(cid)}/lesson/revise`, { text });
          await refresh();
        });
      }
    }
  });

  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-learn]");
    if (!t) return;
    const a = t.dataset.learn;
    const id = t.dataset.id;
    if (a === "open") openCourse(id);
    else if (a === "back") { L.id = ""; L.course = null; L.view = "path"; loadLearn(); }
    else if (a === "knowledge-open") { L.showKnowledge = true; loadKnowledge(); }
    else if (a === "knowledge-close") { L.showKnowledge = false; L.knowledge = null; renderLearn(); }
    else if (a === "delete") act("Deleting", async () => { await send("DELETE", `/api/learn/courses/${encodeURIComponent(id)}`, {}); await loadLearn(); });
    else if (a === "answer") act("Re-planning", async () => { await send("POST", `${base()}/answer`, { answer: t.dataset.value }); await loadLearn(); });
    else if (a === "accept") act("Accepting", async () => { await send("POST", `${base()}/accept`, {}); await loadLearn(); });
    else if (a === "map") act("Starting", async () => { await send("POST", `${base()}/map`, {}); await loadLearn(); });
    else if (a === "plan-approve") act("Approving", async () => { await send("POST", `${base()}/plan/approve`, {}); await loadLearn(); });
    else if (a === "concept") { openConcept(id); }
    else if (a === "close-concept") { stopReading(); L.view = "path"; L.conceptId = ""; L.concept = null; L.graded = null; L.ask = null; L.variant = "standard"; L.partialOpen = ""; L.showProfile = false; watchTrace(false); renderLearn(); }
    else if (a === "profile-on") { L.showProfile = true; renderLearn(); watchTrace(true); loadTrace(); }
    else if (a === "profile-off") { L.showProfile = false; watchTrace(false); renderLearn(); }
    else if (a === "claim") { L.claim = L.claim === t.dataset.claim ? "" : t.dataset.claim; renderLearn(); }
    else if (a === "familiarity-partial-open") { L.partialOpen = L.partialOpen === t.dataset.topic ? "" : t.dataset.topic; renderLearn(); }
    else if (a === "familiarity") {
      const topicId = t.dataset.topic;
      const answer = t.dataset.answer;
      const explain = answer === "partial" ? (document.getElementById(`topic-explain-${topicId}`)?.value || "").trim() : "";
      if (answer === "partial" && !explain) return;      // required field; the button does nothing until it is filled
      act("Saving", async () => {
        await send("POST", `${base()}/concepts/${encodeURIComponent(id)}/familiarity`, { topic_id: topicId, answer, explain });
        L.partialOpen = "";
        L.concept = await fetchConcept(L.conceptId);
      });
    } else if (a === "build") {
      L.requested.add(id);
      L.autostart = true;
      act("Building", async () => {
        await send("POST", `${base()}/concepts/${encodeURIComponent(id)}/build`, { force: !!t.dataset.force });
        await refresh();
      });
    } else if (a === "read") {
      L.autostart = true;
      act("Saving", async () => {
        L.course = { ...L.course, ...(await send("POST", `${base()}/concepts/${encodeURIComponent(id)}/read`, {})) };
        L.view = "path";
        L.conceptId = "";
        L.concept = null;
      });
    } else if (a === "flag") {
      const key = t.dataset.claim;
      act("Re-checking against its source", async () => {
        const out = await send("POST", `${base()}/concepts/${encodeURIComponent(L.conceptId)}/claims/${encodeURIComponent(key)}/flag`, { note: "flagged by learner" });
        L.error = out.withdrawn ? `“${key}” was not supported by its passage and has been withdrawn.` : `“${key}” was re-checked: its source passage does support it.`;
        L.concept = await fetchConcept(L.conceptId);
      });
    } else if (a === "variant") {
      stopReading();
      L.variant = t.dataset.variant;
      L.ask = null;
      L.claim = "";
      renderLearn();
    } else if (a === "jump") {
      document.getElementById(`learn-sec-${t.dataset.section}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    } else if (a === "read-start") {
      startReading();
    } else if (a === "read-pause") {
      pauseReading();
    } else if (a === "read-resume") {
      resumeReading();
    } else if (a === "read-stop") {
      stopReading();
    } else if (a === "mic-start") {
      /* Deliberately no renderLearn() anywhere in the mic flow (see startReading's own
       * note above) -- it would also wipe whatever the learner already typed into this
       * same ask box. Every state change here is a direct mutation of this one button. */
      t.dataset.learn = "mic-stop";
      t.classList.add("recording");
      t.title = "Click to stop and transcribe";
      t.textContent = "⏹";
      (async () => {
        try {
          micHandle = await VOICE.startRecording();
        } catch (err) {
          t.dataset.learn = "mic-start";
          t.classList.remove("recording");
          t.textContent = "🎙";
          t.title = `Could not access the microphone: ${err.message}`;
        }
      })();
    } else if (a === "mic-stop") {
      const handle = micHandle;
      micHandle = null;
      t.dataset.learn = "";
      t.classList.remove("recording");
      t.classList.add("transcribing");
      t.textContent = "…";
      t.title = "Transcribing…";
      (async () => {
        try {
          const blob = await VOICE.stopRecording(handle);
          const text = await VOICE.transcribe(blob);
          const box = document.getElementById("learn-ask-question");
          if (box && text) box.value = box.value.trim() ? `${box.value.trim()} ${text}` : text;
          t.title = text ? "Ask by voice" : "Heard nothing — try again";
        } catch (err) {
          t.title = `Could not transcribe: ${err.message}`;
        }
        t.classList.remove("transcribing");
        t.dataset.learn = "mic-start";
        t.textContent = "🎙";
      })();
    } else if (a === "write" || a === "write-pages") {
      const pages = a === "write-pages" ? Number($("#learn-pages")?.value || 8) : Number(t.dataset.pages || 0) || null;
      const variant = a === "write-pages" ? "pages" : t.dataset.variant;
      if (a === "write-pages") L.customPages = pages;
      L.requested.add(L.conceptId);
      L.autostart = true;
      act("Starting", async () => {
        const out = await send("POST", `${base()}/concepts/${encodeURIComponent(L.conceptId)}/lesson`, { variant, pages });
        L.variant = out.variant;
        L.concept = await fetchConcept(L.conceptId);
      });
    } else if (a === "ask-open" && L.pending) {
      L.ask = { ...L.pending, intent: t.dataset.intent || "deeper" };
      L.askResult = "";
      L.pending = null;
      popup().style.display = "none";
      renderLearn();
    } else if (a === "ask-go") {
      askForDetail();
    } else if (a === "ask-cancel") {
      if (micHandle) { VOICE.abortRecording(micHandle); micHandle = null; }
      L.ask = null;
      L.askResult = "";
      renderLearn();
    } else if (a === "del-expansion") {
      act("Removing", async () => {
        await send("DELETE", `${base()}/concepts/${encodeURIComponent(L.conceptId)}/expansions/${encodeURIComponent(id)}`, {});
        L.concept = await fetchConcept(L.conceptId);
      });
    } else if (a === "next-item") { L.graded = null; act("Loading", async () => { await refresh(); }); }
    else if (a === "pretest") act("Building the diagnostic", async () => {
      await send("POST", `${base()}/pretest/start`, {});
      L.pretest = { state: "building", items: [], answered: [] };
      await refresh();
    });
    else if (a === "skip-pretest") act("Skipping", async () => { await send("POST", `${base()}/pretest/skip`, {}); await refresh(); });
    else if (a === "diag-start") act("Starting", async () => { L.diag = await send("POST", `${base()}/diagnostic/start`, {}); });
    else if (a === "diag-skip") act("Skipping", async () => { L.diag = await send("POST", `${base()}/diagnostic/skip`, {}); await refresh(); });
    else if (a === "diag-noidea") diagAnswer("I don't know");
    else if (a === "diag-continue") { L.diagResult = null; act("Loading", async () => { await refresh(); }); }
    else if (a === "check-noidea") checkAnswer(id, "I don't know");
    else if (a === "check-continue") { L.checkResult = null; act("Loading", async () => { L.concept = await fetchConcept(L.conceptId); }); }
    else if (a === "known") {
      L.autostart = true;
      act("Saving", async () => {
        L.course = { ...L.course, ...(await send("POST", `${base()}/concepts/${encodeURIComponent(id)}/known`, {})) };
        L.view = "path";
        L.conceptId = "";
        L.concept = null;
      });
    }
  });
}

bindLearn();
// Checked once, well before anyone opens Learn, so the mic/speaker buttons' first render
// already knows whether either is installed rather than showing then hiding them.
VOICE.checkAvailable();
