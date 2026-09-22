/* Learn: a course built from the paper corpus for a goal you name.
 *
 * Every lesson sentence carries the claims it rests on, and each claim its source passage, so
 * "where does it say that?" is one click. The page never shows a quiz answer until you have
 * answered, and never a chart number or diagram edge the server did not check against a claim. */

import { $, escapeHtml } from "./dom.js";
import { renderMath } from "./tex.js";
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
};

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
    L.error = "";
  } catch (err) {
    L.error = err.message;
  }
  renderLearn();
  schedulePoll();
}

/* Poll only while something is being built and the tab is open. */
function schedulePoll() {
  clearTimeout(L.poll);
  if (!L.open || !L.id) return;
  const c = L.course;
  const waiting = c?.next?.action === "build" && L.requested.has(c.next.concept);
  const building = c && (waiting || c.status === "mapping" || c.pretest === "building"
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
  return L.requested.has(concept.id) && s && !["done", "error"].includes(s);
}

async function refresh() {
  if (!L.id) return;
  try {
    L.course = await get(`/api/learn/courses/${encodeURIComponent(L.id)}`);
    if (L.conceptId) L.concept = await fetchConcept(L.conceptId);
    if (L.pretest?.state === "building" || L.course.pretest === "building") await loadPretest();
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
    ${rows ? `<h4>Your courses</h4><ul class="learn-courses">${rows}</ul>` : ""}`;
}

function courseView(c) {
  const head = `<div class="learn-head">
    <button type="button" class="link" data-learn="back">← all courses</button>
    <h3>${escapeHtml(c.goal)}</h3></div>`;
  if (["scoping", "scoped", "failed", "mapping"].includes(c.status) || !c.concepts.length) {
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
    action = `<p class="hint">Reading survey papers to draw the map of what you need to learn…</p>`;
  } else if (c.status === "failed" || c.status === "ready") {
    action = `<p class="error">${escapeHtml(c.error || "That did not work.")}</p>
      <button type="button" data-learn="map">Try mapping again</button>`;
  }
  return `<h4>What you will be able to do</h4><ul class="learn-comps">${comps}</ul>${diff}${action}`;
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
  const sub = (n) => (n.unavailable ? "no sources" : n.inferred ? "assumed from diagnostic"
    : `${Math.round(n.mastery * 100)}% · ${n.built.length ? "built" : "not built"}`);
  return `<h4>Concept map <span class="hint">click a concept to open it</span></h4>
    ${svgGraph(nodes, edges, { nodeClass: cls, nodeSub: sub, action: "concept" })}`;
}

/* The focus: what to do next ---------------------------------------------------- */

function focusView(c) {
  if (L.graded) return gradedCard();
  const n = c.next || {};
  if (c.pretest === "todo" && !Object.values(c.concepts).some((k) => k.mastery > 0)) {
    return diagnosticOffer() + nextCard(c, n);
  }
  if (L.pretest?.state === "active" || c.pretest === "building" || c.pretest === "active") {
    return pretestView(c);
  }
  return nextCard(c, n);
}

function diagnosticOffer() {
  return `<div class="learn-card"><p><b>Skip what you already know.</b> A short diagnostic samples
    concepts across the course; what you pass, and what it depends on, is marked known.</p>
    <button type="button" data-learn="pretest">Take the diagnostic</button>
    <button type="button" class="link" data-learn="skip-pretest">No thanks, start from the beginning</button></div>`;
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
  try {
    L.concept = await fetchConcept(cid);
  } catch (err) {
    L.error = err.message;
  }
  renderLearn();
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

/* Versions of the lesson ------------------------------------------------------------ */

const PRESET = { tldr: "TL;DR", standard: "Standard", thorough: "Thorough (about 5 pages)" };

function activeLesson(concept) {
  return concept.lessons?.[L.variant] || null;
}

function variantLabel(key) {
  return PRESET[key] || (key.startsWith("pages-") ? `${key.slice(6)} pages` : key);
}

function variantBar(concept) {
  const keys = ["tldr", "standard", "thorough", ...Object.keys(concept.lessons || {}).filter((k) => k.startsWith("pages-"))];
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
  const eta = L.variant === "tldr" ? "about half a minute" : "a few minutes: it searches the papers again for every section";
  const spec = L.variant.startsWith("pages-") ? `data-variant="pages" data-pages="${escapeHtml(L.variant.slice(6))}"` : `data-variant="${escapeHtml(L.variant)}"`;
  return `<div class="learn-card"><p>The ${escapeHtml(variantLabel(L.variant))} version has not been written yet.
    <span class="hint">Takes ${eta}. It only says what the sources support, so it may come out shorter than asked.</span></p>
    ${failed ? `<p class="error">${escapeHtml(b.error)}</p>` : ""}
    <button type="button" data-learn="write" ${spec}>Write it</button></div>`;
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
  const length = lesson.target_pages
    ? `<p class="hint">About ${lesson.achieved_pages} page(s), for the ${lesson.target_pages} asked for.
        ${lesson.shortfall ? `<span class="warn">${md(lesson.shortfall)}</span>` : ""}
        ${lesson.dropped_sections?.length ? `Not enough sources for: ${lesson.dropped_sections.map(escapeHtml).join("; ")}.` : ""}</p>` : "";
  const trust = length + `<p class="hint trust" title="Every sentence is re-checked against the claims it cites; ones that fail are rewritten once, then dropped.">
    ${s.grounded_pct}% of sentences verified on the first pass · ${s.repaired} rewritten · ${s.dropped} dropped
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

function chartSvg(v) {
  const W = 520, H = 220, pad = 40, max = Math.max(...v.points.map((p) => p.value), 0) || 1;
  const bw = (W - pad * 2) / v.points.length;
  const bars = v.points.map((p, i) => {
    const h = Math.max(2, (p.value / max) * (H - pad * 2));
    const x = pad + i * bw + bw * 0.15, y = H - pad - h;
    return v.chart === "line"
      ? `<circle cx="${x + bw * 0.35}" cy="${y}" r="4" class="pt"/>`
      : `<rect x="${x}" y="${y}" width="${bw * 0.7}" height="${h}" rx="3" class="bar"/>`
      + `<text x="${x + bw * 0.35}" y="${y - 4}" text-anchor="middle" class="val">${p.value}</text>`
      + `<text x="${x + bw * 0.35}" y="${H - pad + 14}" text-anchor="middle" class="lab">${escapeHtml(String(p.label).slice(0, 14))}</text>`;
  }).join("");
  const line = v.chart === "line" ? `<polyline class="ln" points="${v.points.map((p, i) => `${pad + i * bw + bw * 0.5},${H - pad - Math.max(2, (p.value / max) * (H - pad * 2))}`).join(" ")}"/>`
    + v.points.map((p, i) => `<text x="${pad + i * bw + bw * 0.5}" y="${H - pad + 14}" text-anchor="middle" class="lab">${escapeHtml(String(p.label).slice(0, 14))}</text>`).join("") : "";
  return `<div class="learn-card"><p><b>${md(v.title)}</b> <span class="hint">${md(v.y_label || "")} · every number is in its claim: ${v.claims.map(escapeHtml).join(", ")}</span></p>
    <div class="svg-wrap"><svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img">
    <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" class="axis"/>${line}${bars}</svg></div></div>`;
}

function conceptView(concept) {
  const built = concept.build?.stage === "done" || concept.lesson;
  const notBuilt = !built ? `<div class="learn-card"><p>${buildingNow(concept) ? `Building… <span class="hint">step: ${escapeHtml(concept.build.stage)}</span>` : "This concept has not been built yet."}</p>
    ${buildingNow(concept) ? "" : `<button type="button" data-learn="build" data-id="${escapeHtml(concept.id)}">Build it from the papers</button>`}</div>` : "";
  const list = concept.claims.map((k) => `<li><span class="ck ${k.key === L.claim ? "on" : ""}" data-learn="claim" data-claim="${escapeHtml(k.key)}">${escapeHtml(k.key)}</span> ${badge(k.certainty)} ${md(k.text)}${k.withdrawn ? ' <span class="err">withdrawn</span>' : ""}</li>`).join("");
  const s = concept.stats || {};
  return `<div class="learn-concept">
    <button type="button" class="link" data-learn="close-concept">← back to your path</button>
    <h3>${md(concept.title)}</h3><p class="hint">${md(concept.summary)}
      ${concept.reused ? " · reused from an earlier course" : ""}</p>
    ${notBuilt}${concept.lesson ? variantBar(concept) : ""}${lessonBody(concept)}${claimShownInLesson(concept) ? "" : claimCard(concept)}${conflictsView(concept)}
    ${concept.lesson && !concept.lesson.insufficient ? critiqueBox(concept) : ""}
    ${concept.lesson ? `<p><button type="button" data-learn="read" data-id="${escapeHtml(concept.id)}">I've read this — start practice</button>
      <button type="button" class="link" data-learn="build" data-id="${escapeHtml(concept.id)}" data-force="1">Refresh from the corpus</button></p>` : ""}
    <details><summary>All ${concept.claims.length} claims and their sources${s.unfaithful_dropped ? ` · ${s.unfaithful_dropped} extractions dropped as unfaithful` : ""}</summary><ul class="claims">${list}</ul></details></div>`;
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
  const label = e.question ? `“${md(e.question)}”` : `More on “${md(e.selection.slice(0, 70))}${e.selection.length > 70 ? "…" : ""}”`;
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
    <div class="ask-row"><textarea id="learn-ask-question" rows="2" placeholder="Ask something specific, or leave blank for more detail on this"></textarea>
    ${VOICE.sttAvailable() ? `<button type="button" class="mic" data-learn="mic-start" title="Ask by voice">🎙</button>` : ""}</div>
    <button type="button" data-learn="ask-go">Get more detail</button>
    <button type="button" class="link" data-learn="ask-cancel">Cancel</button>`}</div>`;
}

/* A highlight inside one lesson paragraph puts an Explain button beside it. The selection is
 * captured now: opening the panel re-renders the page and the browser drops it. */
function popup() {
  let pop = document.getElementById("learn-pop");
  if (!pop) {
    pop = document.createElement("button");
    pop.id = "learn-pop";
    pop.type = "button";
    pop.dataset.learn = "ask-open";
    pop.textContent = "Explain this";
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
  pop.style.left = `${Math.max(8, Math.min(rect.left, window.innerWidth - 120))}px`;
  pop.style.top = `${Math.min(rect.bottom + 6, window.innerHeight - 40)}px`;
  pop.style.display = "block";
}

async function askForDetail() {
  const a = L.ask;
  const question = (document.getElementById("learn-ask-question")?.value || "").trim();
  L.asking = true;
  L.askResult = "";
  renderLearn();
  try {
    const out = await send("POST", `${base()}/concepts/${encodeURIComponent(L.conceptId)}/expand`,
      { selection: a.selection, question, section: a.section, claims: a.claims, variant: L.variant }, { timeoutMs: 240000 });
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

async function openCourse(id) {
  L.id = id;
  L.view = "path";
  L.conceptId = "";
  L.concept = null;
  L.graded = null;
  L.pretest = null;
  L.claim = "";
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
    if (e.target.id !== "learn-pop") setTimeout(checkSelection, 0);
  });
  document.addEventListener("keyup", (e) => { if (e.key === "Shift" || e.key.startsWith("Arrow")) checkSelection(); });
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
    } else if (f.id === "learn-critique") {
      e.preventDefault();
      const text = $("#learn-critique-text").value.trim();
      if (text) act("Checking against the sources", async () => {
        L.critique = (await send("POST", `${base()}/concepts/${encodeURIComponent(f.dataset.id)}/critique`, { text })).points;
        await refresh();
      });
    }
  });

  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-learn]");
    if (!t) return;
    const a = t.dataset.learn;
    const id = t.dataset.id;
    if (a === "open") openCourse(id);
    else if (a === "back") { L.id = ""; L.course = null; L.view = "path"; loadLearn(); }
    else if (a === "delete") act("Deleting", async () => { await send("DELETE", `/api/learn/courses/${encodeURIComponent(id)}`, {}); await loadLearn(); });
    else if (a === "answer") act("Re-planning", async () => { await send("POST", `${base()}/answer`, { answer: t.dataset.value }); await loadLearn(); });
    else if (a === "accept") act("Accepting", async () => { await send("POST", `${base()}/accept`, {}); await loadLearn(); });
    else if (a === "map") act("Starting", async () => { await send("POST", `${base()}/map`, {}); await loadLearn(); });
    else if (a === "concept") { openConcept(id); }
    else if (a === "close-concept") { stopReading(); L.view = "path"; L.conceptId = ""; L.concept = null; L.graded = null; L.ask = null; L.variant = "standard"; renderLearn(); }
    else if (a === "claim") { L.claim = L.claim === t.dataset.claim ? "" : t.dataset.claim; renderLearn(); }
    else if (a === "build") {
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
      L.ask = L.pending;
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
  });
}

bindLearn();
// Checked once, well before anyone opens Learn, so the mic/speaker buttons' first render
// already knows whether either is installed rather than showing then hiding them.
VOICE.checkAvailable();
