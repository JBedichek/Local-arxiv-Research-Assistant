/* Reader UI.
 *
 * Latency notes, since perceived responsiveness is the whole point:
 *
 *  - Retrieval fires on `selectionchange`, before the user has typed anything. Selecting
 *    a passage and composing a question takes 3-5 seconds; that is 3-5 seconds of free
 *    retrieval time. By submit, `pendingHits` is usually already resolved and generation
 *    starts immediately.
 *  - Answers stream over SSE, so first token lands in ~300ms instead of waiting 5-10s
 *    for a complete response.
 *  - Papers are injected as HTML with their LaTeXML ids untouched, so scrolling to a
 *    citation is a getElementById away with no parsing or re-layout.
 *  - Highlighting uses the CSS Custom Highlight API where available, which paints ranges
 *    without touching the DOM. Wrapping text in <mark> would mutate the tree the anchors
 *    live in; the fallback does that only when it must.
 */

const $ = (s) => document.querySelector(s);
const state = {
  paper: null, version: 1, hits: [], pendingHits: null,
  selection: null, candidate: null, models: [], busy: false,
  breadth: 'balanced', abort: null, lastAsk: null, lastAnswerChunk: null,
  /* What the citation-graph heat is measured against: the most recent search or question,
   * whichever happened last. Opening a paper deliberately does NOT overwrite it — you
   * arrived at that paper *from* a query, and that query is what you still want the
   * neighbourhood shaded by. */
  heatRef: null,
};

/* ---------- helpers ---------- */

function setStatus(text, kind = "") {
  const el = $("#status");
  el.textContent = text;
  el.className = "status " + kind;
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status} ${(await res.text()).slice(0, 160)}`);
  return res.json();
}

/* ---------- paper ---------- */

async function openPaper(id, fragment, push = true) {
  setStatus("loading paper…");
  try {
    const data = await api(`/api/paper/${encodeURIComponent(id)}`);
    state.paper = data.arxiv_id;
    state.version = data.version;
    state.paperTitle = data.title || "";
    $("#paper-meta").innerHTML = `
      <h1>${escapeHtml(data.title || data.arxiv_id)}</h1>
      <p class="authors">${escapeHtml(data.authors || "")}</p>
      <p class="meta">
        <a href="https://arxiv.org/abs/${data.arxiv_id}" target="_blank" rel="noopener">${data.arxiv_id}v${data.version}</a>
        · ${escapeHtml(data.categories || "")}
        · ${data.n_chunks} chunks
        · source: ${data.fulltext_source || "none"}
      </p>`;

    if (data.html) {
      $("#paper").innerHTML = data.html;
    } else {
      // Most of the corpus has not been crawled yet, so opening such a paper used to
      // show a near-empty pane and look broken. Fetch it now and show the abstract in
      // the meantime.
      $("#paper").innerHTML =
        `<div class="fetching"><span class="spin"></span>
           Fetching full text from arXiv…</div>
         <h3>Abstract</h3><p>${escapeHtml(data.abstract || "(no abstract)")}</p>`;
      fetchFullText(data.arxiv_id);
    }
    // pushState, not replaceState: the previous version overwrote the current entry, so
    // Back never had anywhere to go. Papers and searches are both real destinations.
    const url = `/p/${data.arxiv_id}v${data.version}${fragment || ""}`;
    const entry = { view: "paper", id: data.arxiv_id, fragment: fragment || null };
    if (push && location.pathname + location.hash !== url) history.pushState(entry, "", url);
    else history.replaceState(entry, "", url);
    setStatus("");
    loadGraph();
    loadLibrary();          // the server recorded this visit while serving the paper
    if (fragment) scrollToAnchor(fragment);
    applyHeatmap();
  } catch (err) {
    setStatus(String(err.message || err), "error");
  }
}

async function fetchFullText(id) {
  try {
    const r = await fetch(`/api/fetch/${encodeURIComponent(id)}`, { method: "POST" });
    const d = await r.json();
    if (d.status === "ok") {
      const fresh = await api(`/api/paper/${encodeURIComponent(id)}`);
      if (state.paper === id && fresh.html) {
        $("#paper").innerHTML = fresh.html;
        applyHeatmap();
        setStatus(`fetched ${d.chunks || fresh.n_chunks} chunks via ${d.source || "cache"}`);
      }
    } else if (d.status === "in_progress") {
      setTimeout(() => fetchFullText(id), 2500);
    } else {
      const el = $("#paper").querySelector(".fetching");
      if (el) el.innerHTML =
        `Full text unavailable (${escapeHtml(d.status || "failed")}). ` +
        `<a href="https://arxiv.org/abs/${id}" target="_blank" rel="noopener">Open on arXiv</a>`;
    }
  } catch (err) {
    const el = $("#paper").querySelector(".fetching");
    if (el) el.textContent = "Could not fetch full text.";
  }
}

/* Resolve `#S3.p4:120-480` — scroll to the element and paint the char range. */
function scrollToAnchor(fragment) {
  const raw = fragment.replace(/^#/, "");
  const [anchor, range] = raw.split(":");
  const el = document.getElementById(anchor);
  if (!el) return false;
  scrollIntoPane(el, $("#paper"));
  if (range) {
    const [s, e] = range.split("-").map(Number);
    if (Number.isFinite(s) && Number.isFinite(e)) paintRange(el, s, e);
  } else {
    paintRange(el, 0, el.textContent.length);
  }
  return true;
}

/* Scroll ONE container, never the document.
 *
 * scrollIntoView walks up and scrolls every scrollable ancestor it finds. `body` is
 * overflow:hidden here, which stops a user scrolling but not a programmatic scroll — so
 * following a citation moved the paper pane AND slid the whole document up, taking the
 * top bar with it (measured: bar top 0 -> -124px). Computing the offset and setting
 * scrollTop on the pane alone has no such side effect. */
function stickToBottom() {
  // Same hazard as scrollIntoPane: scrollIntoView inside the chat pane would also scroll
  // the document and carry the top bar off screen.
  const m = $("#messages");
  m.scrollTop = m.scrollHeight;
  const se = document.scrollingElement;
  if (se && se.scrollTop) se.scrollTop = 0;
}

function scrollIntoPane(el, pane) {
  const p = pane.getBoundingClientRect();
  const e = el.getBoundingClientRect();
  const delta = (e.top - p.top) - (p.height / 2 - e.height / 2);
  pane.scrollTo({ top: pane.scrollTop + delta, behavior: "smooth" });
  // Undo any document scroll a previous action left behind.
  const se = document.scrollingElement;
  if (se && se.scrollTop) se.scrollTop = 0;
}

/* Map character offsets within an element onto a DOM Range across its text nodes. */
function rangeFromOffsets(el, start, end) {
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  let pos = 0, range = document.createRange(), started = false, node;
  while ((node = walker.nextNode())) {
    const len = node.textContent.length;
    if (!started && pos + len > start) {
      range.setStart(node, Math.max(0, start - pos));
      started = true;
    }
    if (started && pos + len >= end) {
      range.setEnd(node, Math.max(0, Math.min(len, end - pos)));
      return range;
    }
    pos += len;
  }
  if (started) { range.setEndAfter(el.lastChild || el); return range; }
  return null;
}

let highlightRegistry = null;
function paintRange(el, start, end) {
  const range = rangeFromOffsets(el, start, end);
  if (!range) return;
  if (window.CSS && CSS.highlights) {
    if (!highlightRegistry) {
      highlightRegistry = new Highlight();
      CSS.highlights.set("lara-cite", highlightRegistry);
    }
    highlightRegistry.clear();
    highlightRegistry.add(range);
  } else {
    document.querySelectorAll("mark.lara-cite").forEach((m) => {
      m.replaceWith(...m.childNodes);
    });
    const mark = document.createElement("mark");
    mark.className = "lara-cite";
    try { range.surroundContents(mark); } catch { /* spans elements; skip */ }
  }
  el.classList.add("lara-flash");
  setTimeout(() => el.classList.remove("lara-flash"), 1200);
}

/* ---------- selection -> speculative retrieval (R4) ---------- */

let selTimer = null;
document.addEventListener("selectionchange", () => {
  clearTimeout(selTimer);
  selTimer = setTimeout(handleSelection, 180);
});

function handleSelection() {
  const sel = document.getSelection();
  const text = sel && String(sel).trim();
  const float = $("#ask-float");
  if (!text || text.length < 12 || !$("#paper").contains(sel.anchorNode)) {
    float.hidden = true;
    return;
  }
  // Highlighting is not the same as asking. People select text to read it, to copy it, or
  // to keep their place; attaching it to the prompt automatically hijacks an ordinary
  // reading gesture. The passage is only staged as context once "Ask about this" is
  // clicked. Retrieval still fires now, invisibly, so the latency saving survives — it is
  // just thrown away if the selection was never meant as a question.
  state.candidate = text;
  const rect = sel.getRangeAt(0).getBoundingClientRect();
  float.style.top = `${window.scrollY + rect.bottom + 8}px`;
  float.style.left = `${window.scrollX + rect.left}px`;
  float.hidden = false;
  state.pendingHits = retrieve(text.slice(0, 600), text).catch(() => null);
}

function showSelectionChip(text) {
  const chip = $("#selection-chip");
  chip.hidden = false;
  chip.firstElementChild.textContent =
    text.length > 120 ? text.slice(0, 120) + "…" : text;
}

$("#clear-sel").addEventListener("click", (ev) => {
  ev.preventDefault();
  ev.stopPropagation();
  clearSelectionContext();
});

function clearSelectionContext() {
  state.selection = null;
  state.candidate = null;
  state.pendingHits = null;
  $("#selection-chip").hidden = true;
  $("#ask-float").hidden = true;
  // Collapse the document selection too. Without this the browser keeps the range, the
  // debounced selectionchange handler fires again, and the chip reappears — which is why
  // the dismiss button looked like it did nothing.
  const sel = document.getSelection();
  if (sel && !sel.isCollapsed) sel.removeAllRanges();
}

$("#ask-float").addEventListener("click", () => {
  if (!state.candidate) return;
  state.selection = state.candidate;      // promote only on explicit request
  showSelectionChip(state.selection);
  $("#ask-float").hidden = true;
  $("#question").focus();
});

function retrieve(query, selection) {
  return api("/api/retrieve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query, selection, paper: state.paper,
      scope: $("#scope").value, k: 8,
    }),
  });
}

/* ---------- ask ---------- */

$("#ask-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const q = $("#question").value.trim();
  if (!q || state.busy) return;
  state.busy = true;
  state.abort = new AbortController();
  $("#ask-btn").disabled = true;
  $("#question").value = "";
  // A multi-round search can run for 20s at the deep end. Anything that long must be
  // interruptible, or the only way out is reloading the page.
  if (!$("#cancel-btn")) {
    const c = document.createElement("button");
    c.type = "button"; c.id = "cancel-btn"; c.textContent = "Stop";
    c.addEventListener("click", () => state.abort?.abort());
    $("#ask-form").append(c);
  }

  addMessage("user", q, state.selection);
  const answerEl = addMessage("assistant", "");
  const t0 = performance.now();

  try {
    // Prefer the speculative result; only retrieve now if there wasn't one.
    let hits = null;
    if (state.pendingHits) {
      const spec = await state.pendingHits;
      if (spec && spec.hits) hits = spec.hits;
    }
    if (!hits) {
      const fresh = await retrieve(q, state.selection);
      hits = fresh.hits;
    }
    state.hits = hits;
    state.lastAsk = q;
    state.heatRef = { text: q, kind: "question" };
    state.lastAnswerChunk = hits[0]?.chunk_id ?? null;
    renderCitations(answerEl, hits);

    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: q, selection: state.selection, paper: state.paper,
        scope: $("#scope").value, hits,
        model: $("#model").value || null,
        temperature: parseFloat($("#temp").value),
        breadth: state.breadth,
      }),
      signal: state.abort.signal,
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "", body = "", first = true;
    const textEl = answerEl.querySelector(".text");

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const frames = buf.split("\n\n");
      buf = frames.pop();
      for (const frame of frames) {
        const ev = /event: (\w+)/.exec(frame);
        const dm = /data: ([\s\S]*)$/.exec(frame);
        if (!ev || !dm) continue;
        if (ev[1] === "step") {
          const st = JSON.parse(dm[1]);
          addStep(answerEl, st.kind, st.detail);
        } else if (ev[1] === "hits") {
          // Later rounds append material; keep the citation list in sync so the
          // markers the model emits always resolve.
          hits = JSON.parse(dm[1]);
          state.hits = hits;
          renderCitations(answerEl, hits);
        } else if (ev[1] === "coverage") {
          renderCoverage(answerEl, JSON.parse(dm[1]));
        } else if (ev[1] === "grounding") {
          renderGrounding(answerEl, JSON.parse(dm[1]));
        } else if (ev[1] === "clarify") {
          renderClarify(answerEl, JSON.parse(dm[1]));
        } else if (ev[1] === "token") {
          if (first) {
            finishSteps(answerEl);
            setStatus(`first token ${Math.round(performance.now() - t0)}ms`);
            first = false;
          }
          body += JSON.parse(dm[1]);
          textEl.innerHTML = linkCitations(body, hits);
        } else if (ev[1] === "error") {
          textEl.innerHTML = `<span class="error">${escapeHtml(JSON.parse(dm[1]))}</span>`;
        }
      }
    }
    if (!body) textEl.innerHTML ||= `<span class="dim">(no generation — is vLLM running?)</span>`;
    loadGraph();
    loadLibrary();          // the answer was written to the library as the stream closed
    // Repaint here too, not only on paper-open: asking while already reading a paper is
    // the common case, and it is exactly when the reference vector becomes available.
    applyHeatmap();
  } catch (err) {
    answerEl.querySelector(".text").innerHTML =
      `<span class="error">${escapeHtml(String(err.message || err))}</span>`;
  } finally {
    finishSteps(answerEl);
    state.busy = false;
    state.abort = null;
    $("#ask-btn").disabled = false;
    $("#ask-btn").textContent = "Ask";
    state.pendingHits = null;
    clearSelectionContext();
    const el = $("#cancel-btn");
    if (el) el.remove();
  }
});

function addMessage(role, text, selection) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML =
    (selection ? `<blockquote>${escapeHtml(selection.slice(0, 300))}</blockquote>` : "") +
    `<div class="text">${escapeHtml(text)}</div><div class="cites"></div>`;
  $("#messages").append(div);
  stickToBottom();
  return div;
}

/* Turn [1] markers into links that scroll the paper pane and paint the highlight. */
function linkCitations(text, hits) {
  return escapeHtml(text).replace(/\[(\d+)\]/g, (m, n) => {
    const hit = hits[Number(n) - 1];
    if (!hit) return m;
    return `<a class="cite" href="${hit.url}" data-i="${Number(n) - 1}" title="${escapeHtml(
      (hit.paper_title || "") + " › " + (hit.section || "")
    )}">[${n}]</a>`;
  });
}

function renderCitations(msgEl, hits) {
  msgEl.querySelector(".cites").innerHTML = hits
    .map(
      (h, i) =>
        `<a class="chip" href="${h.url}" data-i="${i}">[${i + 1}] ${escapeHtml(
          (h.paper_title || h.arxiv_id).slice(0, 44)
        )} › ${escapeHtml((h.section || "").slice(0, 26))}</a>`
    )
    .join("");
}

document.addEventListener("click", (ev) => {
  const a = ev.target.closest("a.cite, a.chip");
  if (!a) return;
  ev.preventDefault();
  const idx = Number(a.dataset.i);
  const hit = state.hits[idx];
  if (!hit) return;
  // A followed citation is the one relevance signal here that no model produced. Fire and
  // forget — training telemetry must never delay or break navigation.
  if (state.lastAsk) {
    fetch("/api/click", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: state.lastAsk, chunk_id: hit.chunk_id, rank: idx + 1 }),
    }).catch(() => {});
  }
  if (hit.arxiv_id !== state.paper) {
    openPaper(hit.arxiv_id, `#${hit.anchor}:${hit.char_start}-${hit.char_end}`, true);
  } else {
    scrollToAnchor(`#${hit.anchor}:${hit.char_start}-${hit.char_end}`);
  }
});

/* ---------- graph (R8/R9) ---------- */

let graphData = null;

/* The reference prompt, and how to describe it. Falls back to the open paper's own title so
 * the graph is a heatmap even for someone who typed an arXiv id and never searched —
 * "relevance to nothing" would just render every node the same dead blue. */
function heatReference() {
  if (state.heatRef && state.heatRef.text) return state.heatRef;
  if (state.paperTitle) return { text: state.paperTitle, kind: "paper" };
  return null;
}

const HEAT_REF_LABEL = {
  search:   "relative to your search",
  question: "relative to your question",
  paper:    "relative to this paper",
};

function renderHeatRef(ref) {
  const el = $("#graph-ref");
  if (!el) return;
  if (!ref) { el.textContent = ""; el.removeAttribute("title"); return; }
  el.innerHTML = `${HEAT_REF_LABEL[ref.kind] || "relative to"}: <b>${escapeHtml(ref.text)}</b>`;
  el.title = ref.text;          // the pane is narrow; the full prompt lives in the tooltip
}

async function loadGraph(query = null) {
  if (!state.paper) return;
  // An explicit query still wins, but the default is the standing reference rather than
  // the empty string — passing "" is what made every node heat 0 on paper-open.
  const ref = query ? { text: query, kind: "question" } : heatReference();
  renderHeatRef(ref);
  try {
    graphData = await api(
      `/api/graph/${encodeURIComponent(state.paper)}?query=${encodeURIComponent(ref ? ref.text : "")}`
    );
    drawGraph();
  } catch { /* graph is optional */ }
}

function drawGraph() {
  const canvas = $("#graph");
  if (!graphData || !canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const nodes = graphData.nodes.slice(0, 80);
  const cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 26;
  const placed = nodes.map((n, i) => {
    if (n.id === graphData.root) return { ...n, x: cx, y: cy };
    const a = (i / Math.max(nodes.length - 1, 1)) * Math.PI * 2;
    return { ...n, x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r };
  });
  const byId = Object.fromEntries(placed.map((n) => [n.id, n]));

  ctx.strokeStyle = "rgba(140,150,170,.28)";
  for (const e of graphData.edges) {
    const a = byId[e.src], b = byId[e.dst];
    if (!a || !b) continue;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }
  for (const n of placed) {
    const heat = Math.max(0, Math.min(1, n.heat));
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.id === graphData.root ? 9 : 5 + heat * 5, 0, Math.PI * 2);
    ctx.fillStyle = n.in_corpus ? heatColor(heat) : "rgba(150,150,150,.35)";
    ctx.fill();
    if (n.id === graphData.root) { ctx.strokeStyle = "#e8ecf4"; ctx.lineWidth = 2; ctx.stroke(); }
  }
  canvas._nodes = placed;
}

/* Perceptually ordered ramp: cool + dim for low relevance, warm + bright for high. */
function heatColor(t) {
  const stops = [
    [58, 74, 104], [52, 118, 160], [64, 166, 156],
    [188, 190, 96], [226, 138, 66], [222, 78, 62],
  ];
  const x = t * (stops.length - 1), i = Math.floor(x), f = x - i;
  const a = stops[Math.min(i, stops.length - 1)], b = stops[Math.min(i + 1, stops.length - 1)];
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

$("#graph").addEventListener("mousemove", (ev) => {
  const canvas = $("#graph"), tip = $("#graph-tip");
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  const hit = (canvas._nodes || []).find(
    (n) => (n.x - x) ** 2 + (n.y - y) ** 2 < 100
  );
  if (!hit) { tip.hidden = true; return; }
  tip.hidden = false;
  tip.style.left = `${x + 12}px`;
  tip.style.top = `${y + 12}px`;
  tip.innerHTML = `<b>${escapeHtml((hit.title || hit.id).slice(0, 70))}</b><br>
    <span class="dim">${hit.id} · relevance ${hit.heat.toFixed(3)}</span>`;
});

$("#graph").addEventListener("click", (ev) => {
  const canvas = $("#graph");
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  const hit = (canvas._nodes || []).find((n) => (n.x - x) ** 2 + (n.y - y) ** 2 < 100);
  if (hit && hit.in_corpus) openPaper(hit.id);
});

window.addEventListener("popstate", (ev) => {
  const st = ev.state;
  if (st?.view === "search") {
    $("#arxiv-input").value = st.query || "";
    searchPapers(st.query, false);
  } else if (st?.view === "paper") {
    hideResults();
    openPaper(st.id, st.fragment, false);
  } else {
    // Entry predates this handler (or a fresh load): fall back to reading the URL.
    const m = location.pathname.match(/^\/p\/(.+?)(?:v(\d+))?$/);
    const q = new URLSearchParams(location.search).get("q");
    if (m) { hideResults(); openPaper(m[1], location.hash, false); }
    else if (q) searchPapers(q, false);
  }
});

/* ---------- chrome ---------- */

/* The one box does both. An arXiv id is unambiguous — 4 digits, a dot, 4-5 digits, or
 * the old `hep-th/9901001` form — so anything else is treated as a search query rather
 * than making the user choose a mode first. */
const ARXIV_ID = /^(arxiv:)?\s*(\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?\/\d{7})(v\d+)?$/i;

$("#open-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const raw = $("#arxiv-input").value.trim();
  if (!raw) return;
  const m = raw.match(ARXIV_ID);
  if (m) {
    hideResults();
    openPaper(m[2]);
  } else {
    searchPapers(raw);
  }
});

async function searchPapers(query, push = true) {
  setStatus("searching…");
  const surl = `/?q=${encodeURIComponent(query)}`;
  const sentry = { view: "search", query };
  if (push && location.search !== `?q=${encodeURIComponent(query)}`) history.pushState(sentry, "", surl);
  else history.replaceState(sentry, "", surl);
  state.lastQuery = query;
  state.heatRef = { text: query, kind: "search" };
  const list = $("#results-list");
  $("#results").hidden = false;
  $("#paper").hidden = true;
  $("#paper-meta").innerHTML = "";
  $("#results-head").innerHTML = `<h2>Searching…</h2>`;
  try {
    const t0 = performance.now();
    const data = await api("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, limit: Number($("#topk").value) || 20 }),
    });
    const ms = Math.round(performance.now() - t0);
    searchData = data;
    const nEdges = (data.edges || []).length;
    $("#results-head").innerHTML =
      `<h2>${data.results.length} papers for “${escapeHtml(query)}”</h2>
       <p class="meta">${ms}ms · server ${Math.round(data.timings_ms.total)}ms
       · abstracts ${Math.round(data.timings_ms.abstracts)}ms
       · full text ${Math.round(data.timings_ms.fulltext)}ms
       · ${nEdges} citation${nEdges === 1 ? "" : "s"} among results</p>`;
    list.innerHTML = data.results.map((r, i) => {
      // Show which evidence carried the paper: its abstract, its body text, or both.
      const ev = Object.entries(r.evidence)
        .map(([k, v]) => `<span class="ev ${k}">${k} ${v.toFixed(3)}</span>`).join("");
      return `<li class="result" data-id="${r.arxiv_id}">
        <div class="rank">${i + 1}</div>
        <div class="body">
          <h3>${escapeHtml(r.title)}</h3>
          <p class="meta">${r.arxiv_id} · ${escapeHtml(r.submitted)} ·
            ${escapeHtml(r.categories.split(" ").slice(0, 3).join(" "))}
            ${r.fulltext ? `· ${r.n_chunks} chunks` : "· abstract only"}
            ${r.cited_by ? `· ${r.cited_by} citations` : ""}</p>
          <p class="authors">${escapeHtml(r.authors)}</p>
          <p class="abstract">${escapeHtml(r.abstract)}…</p>
          <p class="evidence">score ${r.score.toFixed(3)} ${ev}</p>
        </div></li>`;
    }).join("") || `<li class="placeholder">No matches.</li>`;
    $("#graph-wrap").hidden = searchView !== "graph";
    $("#results-list").hidden = searchView !== "list";
    $("#results-head").insertAdjacentHTML("beforeend", LEGEND_HTML);
    drawSearchGraph();
    setStatus("");
  } catch (err) {
    $("#results-head").innerHTML = `<h2 class="error">${escapeHtml(String(err.message || err))}</h2>`;
    setStatus("search failed", "error");
  }
}

function hideResults() {
  $("#results").hidden = true;
  $("#paper").hidden = false;
}

$("#results-list").addEventListener("click", (ev) => {
  const li = ev.target.closest("li.result");
  if (!li) return;
  hideResults();
  openPaper(li.dataset.id);
});

$("#temp").addEventListener("input", (e) => {
  $("#temp-val").textContent = Number(e.target.value).toFixed(2);
});

document.addEventListener("keydown", (ev) => {
  if ((ev.metaKey || ev.ctrlKey) && ev.key === "k") { ev.preventDefault(); $("#question").focus(); }
  if (ev.key === "/" && document.activeElement === document.body) {
    ev.preventDefault(); $("#question").focus();
  }
});

$("#question").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) $("#ask-form").requestSubmit();
});

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------- boot ---------- */



function syncQuant() {
  const m = state.models.find((x) => x.repo === $("#model").value);
  $("#quant").innerHTML = (m ? m.quant_options : []).map((q) => `<option>${q}</option>`).join("");
}

/* ================= layout, typography, and the depth dial ================= */

/* Pane widths, font and size all persist: they are reading preferences, and having to
 * re-set them on every visit is exactly the kind of small friction that makes a tool
 * feel unfinished. */
const prefs = {
  get(k, d) { try { return localStorage.getItem("lara." + k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem("lara." + k, v); } catch { /* private mode */ } },
};

const PANES = { lib: "#library-pane", left: "#graph-pane", right: "#chat-pane" };

function setPaneCollapsed(edge, collapsed) {
  const pane = $(PANES[edge]);
  if (pane) pane.classList.toggle("collapsed", collapsed);
}

function applyLayout() {
  const lib = prefs.get("colLib", "240px");
  const l = prefs.get("colLeft", "300px"), r = prefs.get("colRight", "400px");
  document.documentElement.style.setProperty("--col-lib", lib);
  document.documentElement.style.setProperty("--col-left", l);
  document.documentElement.style.setProperty("--col-right", r);
  setPaneCollapsed("lib", parseInt(lib, 10) === 0);
  setPaneCollapsed("left", parseInt(l, 10) === 0);
  setPaneCollapsed("right", parseInt(r, 10) === 0);
}

/* Keyboard: [ and ] toggle the sidebars, \ gives the paper the whole window. */
document.addEventListener("keydown", (ev) => {
  // ev.target is not always an Element (it is `document` for synthetic events and when
  // nothing has focus), and Document has no .matches — calling it threw and killed the
  // handler before any shortcut could run.
  const t = ev.target;
  if (t instanceof Element && t.matches("input, textarea, select")) return;
  const root = document.documentElement;
  const toggle = (v, dflt, edge) => {
    const cur = parseInt(getComputedStyle(root).getPropertyValue(v), 10) || 0;
    const next = cur === 0 ? dflt : "0px";
    root.style.setProperty(v, next);
    prefs.set({ "--col-lib": "colLib", "--col-left": "colLeft", "--col-right": "colRight" }[v], next);
    setPaneCollapsed(edge, next === "0px");
  };
  if (ev.key === ";") { toggle("--col-lib", "240px", "lib"); }
  else if (ev.key === "[") { toggle("--col-left", "300px", "left"); drawGraph(); }
  else if (ev.key === "]") { toggle("--col-right", "400px", "right"); }
  else if (ev.key === "\\" || ev.code === "Backslash") {
    const hidden = parseInt(getComputedStyle(root).getPropertyValue("--col-left"), 10) === 0;
    const lib = hidden ? "240px" : "0px";
    const l = hidden ? "300px" : "0px", r = hidden ? "400px" : "0px";
    root.style.setProperty("--col-lib", lib);
    root.style.setProperty("--col-left", l);
    root.style.setProperty("--col-right", r);
    prefs.set("colLib", lib); prefs.set("colLeft", l); prefs.set("colRight", r);
    setPaneCollapsed("lib", !hidden);
    setPaneCollapsed("left", !hidden); setPaneCollapsed("right", !hidden);
    drawGraph();
  } else return;
  setTimeout(drawSearchGraph, 0);
});

function makeSplitter(el, varName, prefKey, edge) {
  let startX = 0, startPx = 0;
  const root = document.documentElement;
  const current = () =>
    parseInt(getComputedStyle(root).getPropertyValue(varName), 10) || 300;

  el.addEventListener("pointerdown", (ev) => {
    ev.preventDefault();
    startX = ev.clientX;
    startPx = current();
    el.setPointerCapture(ev.pointerId);
    el.classList.add("dragging");
    document.body.classList.add("resizing");
  });
  el.addEventListener("pointermove", (ev) => {
    if (!el.hasPointerCapture?.(ev.pointerId)) return;
    const delta = edge === "right" ? startX - ev.clientX : ev.clientX - startX;
    // Free range: 0 (collapsed, so the paper gets the whole window) up to 85%, so the
    // graph or chat can take over the screen instead. Below 60px it snaps shut rather
    // than leaving a useless sliver.
    let px = Math.max(0, Math.min(startPx + delta, window.innerWidth * 0.85));
    if (px < 60) px = 0;
    root.style.setProperty(varName, px + "px");
    setPaneCollapsed(edge, px === 0);
  });
  const end = (ev) => {
    if (!el.classList.contains("dragging")) return;
    el.releasePointerCapture?.(ev.pointerId);
    el.classList.remove("dragging");
    document.body.classList.remove("resizing");
    prefs.set(prefKey, getComputedStyle(root).getPropertyValue(varName).trim());
    drawGraph();
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
  // Double-click toggles collapsed <-> default, which is the fast path for "give me the
  // whole screen for the paper" and back.
  el.addEventListener("dblclick", () => {
    const dflt = { lib: "240px", left: "300px", right: "400px" }[edge] || "300px";
    const now = current();
    const next = now === 0 ? dflt : "0px";
    root.style.setProperty(varName, next);
    prefs.set(prefKey, next);
    setPaneCollapsed(edge, next === "0px");
    drawGraph();
    drawSearchGraph();
  });
}

/* ================= library: reading history and folders ================= */

/* The server owns the data — it records visits and questions itself, so a client that
 * navigates away mid-answer still leaves a complete history. This side only renders the
 * tree and issues moves, which keeps the two from disagreeing about what happened. */

let library = { folders: [], entries: [] };

async function loadLibrary() {
  try {
    library = await api("/api/memory");
  } catch {
    library = { folders: [], entries: [] };
  }
  renderLibrary();
}

const libOpen = {
  get(id) { return prefs.get("libOpen." + id, "1") === "1"; },
  set(id, v) { prefs.set("libOpen." + id, v ? "1" : "0"); },
};

function libEntryLabel(e) {
  if (e.kind === "question") return e.question || "(question)";
  return e.title || e.arxiv_id || "(paper)";
}

function renderLibrary() {
  const tree = $("#lib-tree");
  if (!tree) return;
  if (!library.folders.length && !library.entries.length) {
    tree.innerHTML =
      `<p class="lib-empty">Papers you open and questions you ask show up here.</p>`;
    return;
  }
  const byParent = new Map();
  for (const f of library.folders) {
    if (!byParent.has(f.parent || "")) byParent.set(f.parent || "", []);
    byParent.get(f.parent || "").push(f);
  }
  const byFolder = new Map();
  for (const e of library.entries) {
    const k = e.folder || "";
    if (!byFolder.has(k)) byFolder.set(k, []);
    byFolder.get(k).push(e);
  }

  const node = (html) => {
    const d = document.createElement("div");
    d.innerHTML = html;
    return d.firstElementChild;
  };

  function buildFolder(f) {
    const wrap = document.createElement("div");
    const open = libOpen.get(f.id);
    const kids = (byFolder.get(f.id) || []).length + (byParent.get(f.id) || []).length;
    const row = node(
      `<div class="lib-row lib-folder" draggable="true" data-folder="${f.id}">
         <span class="twist">${open ? "▾" : "▸"}</span>
         <span class="lib-label">${escapeHtml(f.name)}</span>
         <span class="lib-count">${kids || ""}</span>
         <button class="lib-del" title="Delete folder (contents move up)">×</button>
       </div>`);
    wrap.append(row);
    if (open) {
      const kidsEl = document.createElement("div");
      kidsEl.className = "lib-children";
      for (const sub of (byParent.get(f.id) || []).sort((a, b) => a.name.localeCompare(b.name)))
        kidsEl.append(buildFolder(sub));
      for (const e of byFolder.get(f.id) || []) kidsEl.append(buildEntry(e));
      wrap.append(kidsEl);
    }
    return wrap;
  }

  function buildEntry(e) {
    return node(
      `<div class="lib-row lib-entry ${e.kind}" draggable="true" data-entry="${e.id}"
            title="${escapeHtml(libEntryLabel(e))}">
         <span class="twist">${e.kind === "question" ? "?" : "•"}</span>
         <span class="lib-label">${escapeHtml(libEntryLabel(e))}</span>
         <span class="lib-kind">${e.kind === "question" ? escapeHtml((e.title || "").slice(0, 14)) : ""}</span>
         <button class="lib-del" title="Remove from library">×</button>
       </div>`);
  }

  tree.innerHTML = "";
  for (const f of (byParent.get("") || []).sort((a, b) => a.name.localeCompare(b.name)))
    tree.append(buildFolder(f));
  for (const e of byFolder.get("") || []) tree.append(buildEntry(e));
}

/* Restoring a question puts the reader back where they were: the paper open, the question
 * in the box, and the answer they already paid for shown rather than regenerated. */
async function restoreEntry(e) {
  if (e.arxiv_id) await openPaper(e.arxiv_id);
  if (e.kind !== "question") return;
  $("#question").value = e.question || "";
  state.heatRef = { text: e.question, kind: "question" };
  loadGraph();
  if (e.answer) {
    addMessage("user", e.question, e.selection || null);
    const el = addMessage("assistant", "");
    el.classList.add("restored");
    el.querySelector(".text").innerHTML =
      `<span class="dim">from your library · ${escapeHtml((e.created_utc || "").slice(0, 16).replace("T", " "))}</span><br>`
      + escapeHtml(e.answer);
  }
}

function libFindEntry(id) { return library.entries.find((e) => e.id === id); }

$("#lib-tree")?.addEventListener("click", async (ev) => {
  const del = ev.target.closest(".lib-del");
  const row = ev.target.closest(".lib-row");
  if (!row) return;
  const entryId = row.dataset.entry, folderId = row.dataset.folder;
  if (del) {
    ev.stopPropagation();
    try {
      if (entryId) await api(`/api/memory/entry/${entryId}`, { method: "DELETE" });
      else if (folderId) await api(`/api/memory/folder/${folderId}`, { method: "DELETE" });
    } catch { /* fall through to the reload, which shows the real state either way */ }
    loadLibrary();
    return;
  }
  if (folderId) { libOpen.set(folderId, !libOpen.get(folderId)); renderLibrary(); return; }
  const e = libFindEntry(entryId);
  if (e) restoreEntry(e);
});

/* Rename in place rather than through `window.prompt`. A modal dialog blocks the whole
 * page, cannot be styled to match, and on a tree you are mid-drag through it is a jarring
 * interruption for what should be a two-second edit. */
function beginRename(row) {
  const id = row.dataset.folder;
  const label = row.querySelector(".lib-label");
  if (!id || !label || label.querySelector("input")) return;
  const cur = library.folders.find((f) => f.id === id);
  const input = document.createElement("input");
  input.className = "lib-rename";
  input.value = cur ? cur.name : "";
  label.textContent = "";
  label.append(input);
  input.focus();
  input.select();

  let done = false;
  const commit = async (save) => {
    if (done) return;                       // blur fires after Enter; commit exactly once
    done = true;
    const name = input.value;
    if (save && name.trim() && (!cur || name !== cur.name)) {
      await api(`/api/memory/folder/${id}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      }).catch(() => {});
    }
    loadLibrary();
  };
  input.addEventListener("keydown", (ev) => {
    ev.stopPropagation();                   // ; [ ] are pane shortcuts outside a field
    if (ev.key === "Enter") { ev.preventDefault(); commit(true); }
    else if (ev.key === "Escape") { ev.preventDefault(); commit(false); }
  });
  input.addEventListener("blur", () => commit(true));
  input.addEventListener("click", (ev) => ev.stopPropagation());
  input.addEventListener("dblclick", (ev) => ev.stopPropagation());
}

$("#lib-tree")?.addEventListener("dblclick", (ev) => {
  const row = ev.target.closest(".lib-row.lib-folder");
  if (row) beginRename(row);
});

/* Create first, name second: the folder exists immediately and the name is an edit on a
 * real object, so an abandoned rename leaves "New folder" rather than nothing. */
$("#lib-newfolder")?.addEventListener("click", async () => {
  let created = null;
  try {
    created = await api("/api/memory/folder", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "New folder" }),
    });
  } catch { return; }
  await loadLibrary();
  const row = document.querySelector(`.lib-row.lib-folder[data-folder="${created.id}"]`);
  if (row) beginRename(row);
});

/* Drag to file. `refile`/`reparent` are sent explicitly because a null folder means "the
 * root", and without the flag the server cannot tell that apart from "field omitted". */
let libDragged = null;

$("#lib-tree")?.addEventListener("dragstart", (ev) => {
  const row = ev.target.closest(".lib-row");
  if (!row) return;
  libDragged = { entry: row.dataset.entry || null, folder: row.dataset.folder || null };
  ev.dataTransfer.effectAllowed = "move";
  ev.dataTransfer.setData("text/plain", row.dataset.entry || row.dataset.folder || "");
});

$("#lib-tree")?.addEventListener("dragover", (ev) => {
  if (!libDragged) return;
  ev.preventDefault();
  ev.dataTransfer.dropEffect = "move";
  const row = ev.target.closest(".lib-row.lib-folder");
  for (const el of document.querySelectorAll(".lib-row.drop-into")) el.classList.remove("drop-into");
  if (row) { row.classList.add("drop-into"); $("#lib-tree").classList.remove("drop-root"); }
  else $("#lib-tree").classList.add("drop-root");
});

$("#lib-tree")?.addEventListener("dragleave", (ev) => {
  if (ev.target === $("#lib-tree")) $("#lib-tree").classList.remove("drop-root");
});

$("#lib-tree")?.addEventListener("drop", async (ev) => {
  if (!libDragged) return;
  ev.preventDefault();
  const row = ev.target.closest(".lib-row.lib-folder");
  const target = row ? row.dataset.folder : null;
  for (const el of document.querySelectorAll(".lib-row.drop-into")) el.classList.remove("drop-into");
  $("#lib-tree").classList.remove("drop-root");
  const moved = libDragged;
  libDragged = null;
  if (moved.folder && moved.folder === target) return;   // dropping a folder on itself
  try {
    if (moved.entry) {
      await api(`/api/memory/entry/${moved.entry}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder: target, refile: true }),
      });
    } else if (moved.folder) {
      await api(`/api/memory/folder/${moved.folder}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ parent: target, reparent: true }),
      });
    }
  } catch { /* reload below reveals whatever actually happened */ }
  loadLibrary();
});

$("#lib-tree")?.addEventListener("dragend", () => {
  libDragged = null;
  for (const el of document.querySelectorAll(".lib-row.drop-into")) el.classList.remove("drop-into");
  $("#lib-tree")?.classList.remove("drop-root");
});

/* ================= system prompt ================= */

let promptState = { default: "", custom: null };

async function loadPrompt() {
  try {
    promptState = await api("/api/settings/prompt");
  } catch {
    return;
  }
  const ta = $("#sysprompt");
  if (!ta) return;
  ta.value = promptState.active || "";
  setPromptState(promptState.is_custom ? "custom" : "default");
}

function setPromptState(kind, note) {
  const el = $("#sysprompt-state");
  if (!el) return;
  el.textContent = note || (kind === "custom" ? "using your prompt" : "using the default");
}

$("#sysprompt-save")?.addEventListener("click", async () => {
  const text = $("#sysprompt").value;
  try {
    promptState = await api("/api/settings/prompt", {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    $("#sysprompt").value = promptState.active || "";
    setPromptState(promptState.is_custom ? "custom" : "default", "saved");
  } catch (err) {
    setPromptState("", "save failed: " + (err.message || err));
  }
});

$("#sysprompt-reset")?.addEventListener("click", async () => {
  try {
    promptState = await api("/api/settings/prompt", {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "" }),
    });
    $("#sysprompt").value = promptState.active || "";
    setPromptState("default", "restored");
  } catch (err) {
    setPromptState("", "reset failed: " + (err.message || err));
  }
});

/* ---------- reading style ---------- */

/* Top of the `line width` range means "no cap — follow the pane" rather than 100 characters.
 * Kept as a slider position rather than a separate checkbox so there is one control for one
 * decision, and so dragging left from `full width` is a continuous gesture. */
const MEASURE_FULL = 100;

/* One-time reset. Filling the pane is the new default, but anyone who used a previous build
 * has a fixed measure in localStorage that would silently override it — the change would
 * look like it had not shipped. Runs once, and only touches this one key. */
(function migrateMeasureToFull() {
  if (prefs.get("measureFull", "") === "1") return;
  prefs.set("measure", String(MEASURE_FULL));
  prefs.set("measureFull", "1");
})();

/* Presets bundle theme, face, size, leading and measure, because those five interact:
 * a serif at 17px wants more leading and a wider column than a 14px sans, and setting one
 * without the others usually makes reading worse rather than better. Touching any
 * individual control switches the preset to "custom" instead of silently disagreeing with
 * the label. */

const FONTS = {
  system: "var(--font-system)", inter: "var(--font-inter)",
  charter: "var(--font-charter)", literata: "var(--font-literata)",
  times: "var(--font-times)", mono: "var(--font-mono)",
  wide: "var(--font-wide)", atkinson: "var(--font-atkinson)",
};

/* Every preset now fills the pane. A preset that pinned its own column width would undo
 * the pane-following behaviour the moment someone picked one, which reads as the setting
 * randomly reverting. Narrowing stays available on the slider, independent of preset. */
const PRESETS = {
  default:  { theme: "auto",     font: "system",   size: 16, leading: 1.60, measure: MEASURE_FULL, justify: false },
  // Warm page, serif — closest to a printed journal.
  paper:    { theme: "light",    font: "charter",  size: 17, leading: 1.65, measure: MEASURE_FULL, justify: true  },
  night:    { theme: "dark",     font: "system",   size: 16, leading: 1.72, measure: MEASURE_FULL, justify: false },
  sepia:    { theme: "sepia",    font: "literata", size: 17, leading: 1.70, measure: MEASURE_FULL, justify: true  },
  // More text per screen for skimming, at the cost of comfort over long sessions.
  compact:  { theme: "auto",     font: "system",   size: 14, leading: 1.45, measure: MEASURE_FULL, justify: false },
  // Accessibility: Atkinson Hyperlegible was designed to disambiguate similar glyphs.
  contrast: { theme: "contrast", font: "atkinson", size: 19, leading: 1.85, measure: MEASURE_FULL, justify: false },
};

function currentStyle() {
  return {
    theme:   prefs.get("theme", "auto"),
    font:    prefs.get("font", "system"),
    size:    Number(prefs.get("fontsize", "16")),
    leading: Number(prefs.get("leading", "1.6")),
    measure: Number(prefs.get("measure", String(MEASURE_FULL))),
    justify: prefs.get("justify", "0") === "1",
  };
}

function applyTypography() {
  const st = currentStyle();
  const root = document.documentElement;
  if (st.theme === "auto") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", st.theme);

  root.style.setProperty("--paper-font", FONTS[st.font] || FONTS.system);
  root.style.setProperty("--paper-size", st.size + "px");
  root.style.setProperty("--paper-leading", String(st.leading));
  // Measure is in characters, converted at roughly half the font size per character —
  // the conventional approximation for average glyph advance in running text. At the top
  // of the range it resolves to `none` instead, which is what lets the column follow the
  // pane rather than sitting at a fixed width with dead space beside it.
  root.style.setProperty(
    "--measure",
    st.measure >= MEASURE_FULL ? "none" : Math.round(st.size * st.measure * 0.5) + "px",
  );
  root.style.setProperty("--paper-align", st.justify ? "justify" : "start");
  root.style.setProperty("--paper-hyphens", st.justify ? "auto" : "manual");

  $("#theme").value = st.theme;
  $("#font").value = st.font;
  $("#fontsize").value = String(st.size);
  $("#fontsize-val").textContent = String(st.size);
  $("#leading").value = String(st.leading);
  $("#leading-val").textContent = st.leading.toFixed(2);
  $("#measure").value = String(st.measure);
  $("#measure-val").textContent =
    st.measure >= MEASURE_FULL ? "full width" : `${st.measure} chars`;
  $("#justify").checked = st.justify;
  $("#preset").value = prefs.get("preset", "default");
  drawSearchGraph();
}

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  prefs.set("preset", name);
  prefs.set("theme", p.theme);
  prefs.set("font", p.font);
  prefs.set("fontsize", String(p.size));
  prefs.set("leading", String(p.leading));
  prefs.set("measure", String(p.measure));
  prefs.set("justify", p.justify ? "1" : "0");
  applyTypography();
}

$("#preset").addEventListener("change", (e) => {
  if (e.target.value === "custom") { prefs.set("preset", "custom"); return; }
  applyPreset(e.target.value);
});

// Any manual change means the active preset no longer describes the state.
function markCustom() {
  prefs.set("preset", "custom");
  $("#preset").value = "custom";
}

$("#theme").addEventListener("change", (e) => { prefs.set("theme", e.target.value); markCustom(); applyTypography(); });
$("#font").addEventListener("change", (e) => { prefs.set("font", e.target.value); markCustom(); applyTypography(); });
$("#fontsize").addEventListener("input", (e) => { prefs.set("fontsize", e.target.value); markCustom(); applyTypography(); });
$("#leading").addEventListener("input", (e) => { prefs.set("leading", e.target.value); markCustom(); applyTypography(); });
$("#measure").addEventListener("input", (e) => { prefs.set("measure", e.target.value); markCustom(); applyTypography(); });
$("#justify").addEventListener("change", (e) => { prefs.set("justify", e.target.checked ? "1" : "0"); markCustom(); applyTypography(); });

/* depth dial */
let BREADTH = [];
function applyBreadth() {
  const i = Number($("#breadth").value);
  const b = BREADTH[i];
  if (!b) return;
  state.breadth = b.name;
  prefs.set("breadth", String(i));
  const bits = [`${b.max_rounds} search round${b.max_rounds > 1 ? "s" : ""}`, `${b.k} excerpts`];
  if (b.expand_context) bits.push("reads around excerpts");
  if (b.allow_clarify) bits.push("may ask you to narrow");
  $("#breadth-val").innerHTML =
    `<strong>${escapeHtml(b.label)}</strong> · ${escapeHtml(b.estimate)}<br>
     <em>${escapeHtml(bits.join(" · "))}</em>`;
  $("#breadth").title =
    `${b.label}: ${bits.join(", ")}. Typical total time ${b.estimate}. ` +
    `Deeper settings search more and take longer; generation itself is a fixed ~3s.`;
}
$("#breadth").addEventListener("input", applyBreadth);

/* ================= agent progress rendering ================= */

function stepsEl(msgEl) {
  let el = msgEl.querySelector(".steps");
  if (!el) {
    el = document.createElement("div");
    el.className = "steps";
    msgEl.prepend(el);
  }
  return el;
}

function addStep(msgEl, kind, detail) {
  const host = stepsEl(msgEl);
  host.querySelectorAll(".step.active").forEach((s) => s.classList.remove("active"));
  const row = document.createElement("div");
  row.className = `step ${kind} active`;
  row.innerHTML = `<span class="dot"></span><span>${escapeHtml(detail)}</span>`;
  host.append(row);
  stickToBottom();
}

function finishSteps(msgEl) {
  msgEl.querySelectorAll(".step.active").forEach((s) => s.classList.remove("active"));
}

function renderClarify(msgEl, payload) {
  const box = document.createElement("div");
  box.className = "clarify";
  box.innerHTML =
    `<h4>${escapeHtml(payload.question)}</h4>` +
    (payload.reason ? `<div class="dim">${escapeHtml(payload.reason)}</div>` : "") +
    `<div class="opts">${(payload.options || [])
      .map((o) => `<button type="button" data-q="${escapeHtml(o)}">${escapeHtml(o)}</button>`)
      .join("")}</div>`;
  msgEl.prepend(box);
  box.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => {
      $("#question").value = b.dataset.q;
      $("#ask-form").requestSubmit();
    }));
}

/* ================= search results as a citation graph ================= */

/* Layout: one row per paper, ordered by relevance rank (most relevant at top); x is the
 * submission date. Two variables, no collisions.
 *
 * The earlier version packed nodes into shared lanes to look compact, and labels drawn
 * beside them overlapped into mush. A dedicated row per paper costs vertical space and
 * buys guaranteed legibility, which is the right trade for 20-25 results.
 *
 * Labels are real <a> elements positioned over the canvas rather than canvas text, so
 * they are selectable, keyboard-focusable, and support middle-click and "copy link" like
 * any other link. Canvas draws only what HTML cannot: the edges. */

let searchData = null;
let searchView = "graph";

const ROW_H = 26, PAD_L = 54, PAD_T = 16, PAD_B = 26, LABEL_GUTTER = 22;

function graphHeight(n) {
  return Math.max(160, PAD_T + n * ROW_H + PAD_B);
}

function layoutSearchGraph(width) {
  const nodes = searchData.results;
  if (!nodes.length) return [];
  // Even spacing by chronological RANK, not proportional to elapsed time. Real corpora
  // bunch heavily — a topic can have fifteen papers in one month and one from four years
  // earlier — and a proportional axis collapses the cluster into an unreadable smear at
  // one edge. Rank spacing keeps order and adjacency truthful while guaranteeing every
  // node is separated. The year ticks still label absolute time.
  const order = [...nodes]
    .map((n, i) => ({ i, t: Date.parse(n.submitted || "") || 0 }))
    .sort((a, b) => a.t - b.t);
  const xRank = new Array(nodes.length);
  order.forEach((o, rank) => { xRank[o.i] = rank; });

  // Labels go in a fixed column so the timeline gets the rest of the width. Previously
  // each label began at its own node, which confined the plot to ~42% of the canvas —
  // 225px for 20 papers, under 12px apart. Spacing was already even; the band was just
  // far too narrow to read as a timeline. A fixed column also left-aligns the titles,
  // which makes them scannable as a list.
  const labelW = Math.min(Math.max(width * 0.42, 240), 460);
  const plotW = Math.max(160, width - PAD_L - labelW - LABEL_GUTTER);
  const denom = Math.max(nodes.length - 1, 1);
  return nodes.map((n, i) => ({
    ...n,
    x: PAD_L + (xRank[i] / denom) * plotW,
    labelX: PAD_L + plotW + LABEL_GUTTER,
    y: PAD_T + i * ROW_H + ROW_H / 2,
    radius: 4.5 + Math.min(n.in_degree, 6) * 1.5,
  }));
}

function drawSearchGraph() {
  const canvas = $("#results-graph");
  const overlay = $("#results-labels");
  if (!searchData || searchView !== "graph" || !canvas) return;

  const width = canvas.parentElement.clientWidth;
  const placed = layoutSearchGraph(width);
  const height = graphHeight(placed.length);
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = width + "px";
  canvas.style.height = height + "px";
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  overlay.style.height = height + "px";

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const byId = Object.fromEntries(placed.map((n) => [n.arxiv_id, n]));

  // year ticks along the bottom
  const years = [...new Set(placed.map((n) => (n.submitted || "").slice(0, 4)))]
    .filter(Boolean).sort();
  ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
  for (const y of years) {
    const same = placed.filter((n) => (n.submitted || "").startsWith(y));
    const x = same.reduce((a, n) => a + n.x, 0) / same.length;
    ctx.strokeStyle = "rgba(140,150,170,.12)";
    ctx.beginPath(); ctx.moveTo(x, PAD_T - 6); ctx.lineTo(x, height - PAD_B + 4); ctx.stroke();
    ctx.fillStyle = "rgba(140,150,170,.75)";
    ctx.fillText(y, x - 11, height - PAD_B + 16);
  }

  // edges: citing -> cited, so they point back in time (leftward)
  for (const e of searchData.edges || []) {
    const from = byId[e.src], to = byId[e.dst];
    if (!from || !to) continue;
    ctx.strokeStyle = "rgba(150,170,215,.55)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    const dx = Math.max(26, Math.abs(from.x - to.x) * 0.55);
    ctx.moveTo(from.x, from.y);
    ctx.bezierCurveTo(from.x - dx, from.y, to.x + dx, to.y, to.x, to.y);
    ctx.stroke();
    const ang = Math.atan2(to.y - from.y, -dx);
    ctx.fillStyle = "rgba(150,170,215,.85)";
    ctx.beginPath();
    ctx.moveTo(to.x + to.radius + 1, to.y);
    ctx.lineTo(to.x + to.radius + 8, to.y - 3.6);
    ctx.lineTo(to.x + to.radius + 8, to.y + 3.6);
    ctx.closePath(); ctx.fill();
    void ang;
  }

  const scores = placed.map((n) => n.score);
  const lo = Math.min(...scores), hi = Math.max(...scores);
  for (const n of placed) {
    // Faint leader from the dot to its label column, so the eye keeps the row together
    // now that the two are far apart.
    ctx.strokeStyle = "rgba(140,150,170,.22)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(n.x + n.radius + 3, n.y);
    ctx.lineTo(n.labelX - 6, n.y);
    ctx.stroke();
    ctx.setLineDash([]);

    const t = hi > lo ? (n.score - lo) / (hi - lo) : 1;
    ctx.beginPath(); ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
    ctx.fillStyle = heatColor(t); ctx.fill();
    if (n.in_degree > 0) {
      ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.lineWidth = 1.2; ctx.stroke();
    }
  }

  // Each row is one full-width link, indented past its own dot. Making the whole row the
  // target means clicking the node circle, the title, or the empty space between them all
  // open the paper — an earlier version put the link only on the text, so clicking the
  // dot silently did nothing, which is the most natural thing to click.
  overlay.innerHTML = placed.map((n) => {
    const pad = Math.round(n.labelX);
    const cites = n.in_degree
      ? `<span class="deg" title="cited by ${n.in_degree} of these results">↩${n.in_degree}</span>` : "";
    return `<a class="glabel" href="/p/${n.arxiv_id}v${n.version}" data-id="${n.arxiv_id}"
       style="top:${Math.round(n.y - ROW_H / 2)}px; height:${ROW_H}px; padding-left:${pad}px"
       title="${escapeHtml(n.title)} — click to open">
       <span class="rk">${n.rank}</span>${cites}
       <span class="ttl">${escapeHtml(n.title)}</span>
       <span class="yr">${escapeHtml((n.submitted || "").slice(0, 7))}</span></a>`;
  }).join("");
  canvas._nodes = placed;
}

const LEGEND_HTML = `
  <div class="legend">
    <span><i class="swatch grad"></i> colour = relevance to your query (dim → bright)</span>
    <span><i class="swatch size"></i> size = how many of these results cite it</span>
    <span><i class="swatch edge"></i> arrow points from citing paper → paper it cites</span>
    <span><i class="swatch axis"></i> left→right = submission date</span>
  </div>`;

function bindSearchGraph() {
  const overlay = $("#results-labels");
  const canvas = $("#results-graph");
  const tip = $("#results-tip");

  overlay.addEventListener("click", (ev) => {
    const a = ev.target.closest("a.glabel");
    // Let ctrl/cmd/middle-click behave like a normal link and open a new tab.
    if (!a || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.button !== 0) return;
    ev.preventDefault();
    hideResults();
    openPaper(a.dataset.id, null, true);
  });
  overlay.addEventListener("mousemove", (ev) => {
    const a = ev.target.closest("a.glabel");
    if (!a) { tip.hidden = true; return; }
    const n = (canvas._nodes || []).find((x) => x.arxiv_id === a.dataset.id);
    if (!n) return;
    const host = canvas.parentElement.getBoundingClientRect();
    tip.hidden = false;
    tip.style.left = `${Math.min(ev.clientX - host.left + 16, host.width - 350)}px`;
    tip.style.top = `${ev.clientY - host.top + 16}px`;
    tip.innerHTML =
      `<b>${escapeHtml(n.title)}</b>
       <span class="dim">${n.arxiv_id} · ${escapeHtml(n.submitted)} · score ${n.score.toFixed(3)}
       · cited by ${n.in_degree} of these results · cites ${n.out_degree} of them</span>
       <p>${escapeHtml((n.abstract || "").slice(0, 200))}…</p>`;
  });
  overlay.addEventListener("mouseleave", () => { tip.hidden = true; });

  $("#results-toggle").addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    searchView = b.dataset.view;
    $("#results-toggle").querySelectorAll("button")
      .forEach((x) => x.classList.toggle("on", x.dataset.view === searchView));
    $("#graph-wrap").hidden = searchView !== "graph";
    $("#results-list").hidden = searchView !== "list";
    if (searchView === "graph") drawSearchGraph();
  });

  window.addEventListener("resize", () => drawSearchGraph());
}

/* top-k: re-run the search so the induced subgraph is recomputed over the new set —
 * trimming client-side would show edges to papers no longer displayed. */
$("#topk").addEventListener("input", (e) => {
  $("#topk-val").textContent = e.target.value;
  prefs.set("topk", e.target.value);
});
$("#topk").addEventListener("change", () => {
  if (state.lastQuery) searchPapers(state.lastQuery, false);
});

/* persist the graph pane height set via the native resize grabber */
(function watchGraphHeight() {
  const wrap = $("#graph-wrap");
  if (!wrap || !window.ResizeObserver) return;
  let t = null;
  new ResizeObserver(() => {
    clearTimeout(t);
    t = setTimeout(() => {
      const h = Math.round(wrap.getBoundingClientRect().height);
      if (h > 100) prefs.set("graphH", h + "px");
    }, 250);
  }).observe(wrap);
})();

/* ---------- boot ----------
 * Must be last in the file. This ran mid-file once, before the `const prefs` and
 * `const BREADTH` declarations below it, and hit the temporal dead zone:
 * "Cannot access 'prefs' before initialization" threw out of boot() on every load,
 * so bindSearchGraph() never ran and nothing in the graph was clickable. Search still
 * worked, because its listener is registered at top level, which made the failure
 * look like a graph bug rather than a load-order bug. */


/* ---------- coverage + grounding badges ---------- */

/* Both of these exist so the reader can calibrate trust without clicking every citation.
 * A grounded answer and a confidently-worded guess look identical in prose; the only
 * honest fix is to surface what the system itself knows about its evidence. */

const COVERAGE_UI = {
  full:    { label: "Sources answer this", cls: "cov-full" },
  partial: { label: "Partially answered",  cls: "cov-partial" },
  none:    { label: "Not answered by sources", cls: "cov-none" },
};

function renderCoverage(msgEl, v) {
  const ui = COVERAGE_UI[v.coverage] || COVERAGE_UI.full;
  const bits = [];
  if (v.missing) bits.push(`<div class="cov-detail">Missing: ${escapeHtml(v.missing)}</div>`);
  if (v.conflict) bits.push(`<div class="cov-detail warn">Sources disagree: ${escapeHtml(v.conflict)}</div>`);
  const el = document.createElement("div");
  el.className = `coverage ${ui.cls}`;
  el.innerHTML = `<span class="dot"></span><b>${ui.label}</b>` +
    (v.via === "scores" ? `<span class="dim"> · from retrieval scores</span>` : "") +
    bits.join("");
  const anchor = msgEl.querySelector(".steps");
  anchor ? anchor.after(el) : msgEl.prepend(el);
}

function renderGrounding(msgEl, checks) {
  const weak = checks.filter((c) => !c.supported);
  const el = document.createElement("div");
  el.className = "grounding" + (weak.length ? " has-weak" : "");
  if (!weak.length) {
    el.innerHTML = `<span class="ok">✓</span> all ${checks.length} cited statement${
      checks.length === 1 ? "" : "s"} matched their source`;
  } else {
    el.innerHTML =
      `<b>${weak.length} of ${checks.length} cited statement${checks.length === 1 ? "" : "s"} ` +
      `may not be supported by the excerpt cited</b>` +
      weak.map((c) => `<div class="weak">[${c.marker}] “${escapeHtml(c.sentence)}”
         <span class="dim">score ${c.support.toFixed(3)}</span></div>`).join("");
  }
  msgEl.append(el);
}

/* ================= passage heatmap ================= */

/* When a citation is followed, shade the paper's most relevant passages so the reader can
 * see at a glance where else the answer is supported — not just the one chunk that was
 * cited. Two reference vectors, because they answer different questions:
 *
 *   answer  similarity to the top retrieved chunk. Surfaces the argument AROUND the
 *           answer — its setup, its caveat, the ablation that qualifies it. Usually what
 *           someone following a citation wants next, so it is the default.
 *   query   similarity to what was asked. Surfaces restatements of the question, which is
 *           better for "where else does this paper discuss this".
 *
 * Bands are assigned by RANK, not raw score. Scores within one paper often sit in a narrow
 * range, and normalising them would render five nearly identical shades; rank guarantees
 * the gradation is actually visible.
 */

const HEAT_BANDS = 5;
let heatRegistry = [];

function clearHeatmap() {
  if (!window.CSS || !CSS.highlights) return;
  for (let b = 1; b <= HEAT_BANDS; b++) CSS.highlights.delete(`lara-heat-${b}`);
  heatRegistry = [];
  const el = $("#heat-legend");
  if (el) el.remove();
}

async function applyHeatmap() {
  const mode = prefs.get("heatMode", "answer");
  clearHeatmap();
  if (mode === "off" || !state.paper || !window.CSS || !CSS.highlights) return;

  const q = state.lastAsk;
  const anchorChunk = state.lastAnswerChunk;
  if (mode === "query" && !q) return;
  if (mode === "answer" && !anchorChunk) return;

  try {
    const data = await api("/api/heatmap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        arxiv_id: state.paper, query: q, anchor_chunk_id: anchorChunk,
        mode, k: Number(prefs.get("heatK", "5")),
      }),
    });
    paintHeatmap(data.chunks || [], mode);
  } catch { /* heatmap is decoration; never block reading */ }
}

function paintHeatmap(chunks, mode) {
  if (!chunks.length) return;
  const perBand = Math.ceil(chunks.length / HEAT_BANDS);
  const groups = Array.from({ length: HEAT_BANDS }, () => []);
  let painted = 0;
  chunks.forEach((c, i) => {
    const el = document.getElementById(c.anchor);
    if (!el) return;
    const r = rangeFromOffsets(el, c.char_start, c.char_end);
    if (!r) return;
    groups[Math.min(HEAT_BANDS - 1, Math.floor(i / perBand))].push(r);
    painted++;
  });
  groups.forEach((ranges, b) => {
    if (!ranges.length) return;
    const h = new Highlight(...ranges);
    CSS.highlights.set(`lara-heat-${b + 1}`, h);
    heatRegistry.push(h);
  });
  if (!painted) return;

  const legend = document.createElement("p");
  legend.id = "heat-legend";
  legend.innerHTML =
    `<span class="ramp"></span> ${painted} passage${painted === 1 ? "" : "s"} shaded by
     relevance to ${mode === "answer" ? "the answer passage" : "your question"}
     <button type="button" id="heat-jump">jump to top passage</button>`;
  $("#paper-meta").append(legend);
  $("#heat-jump").addEventListener("click", () => {
    const c = chunks[0];
    scrollToAnchor(`#${c.anchor}:${c.char_start}-${c.char_end}`);
  });
}

/* ---------- advanced settings ---------- */

$("#adv-btn").addEventListener("click", () => {
  const panel = $("#advanced");
  panel.hidden = !panel.hidden;
  $("#adv-btn").classList.toggle("on", !panel.hidden);
});
document.addEventListener("click", (ev) => {
  if (ev.target.closest("#advanced") || ev.target.closest("#adv-btn")) return;
  $("#advanced").hidden = true;
  $("#adv-btn").classList.remove("on");
});

const HEAT_NOTES = {
  answer: "Shades passages similar to the excerpt that answered you — the surrounding "
        + "argument, caveats and ablations.",
  query:  "Shades passages similar to your question — other places the paper discusses it.",
  off:    "No passage shading.",
};

function applyHeatPrefs() {
  const mode = prefs.get("heatMode", "answer");
  const k = prefs.get("heatK", "5");
  $("#heat-mode").value = mode;
  $("#heat-k").value = k;
  $("#heat-k-val").textContent = k;
  $("#heat-note").textContent = HEAT_NOTES[mode] || "";
}

$("#heat-mode").addEventListener("change", (e) => {
  prefs.set("heatMode", e.target.value);
  applyHeatPrefs();
  applyHeatmap();
});
$("#heat-k").addEventListener("input", (e) => {
  prefs.set("heatK", e.target.value);
  $("#heat-k-val").textContent = e.target.value;
});
$("#heat-k").addEventListener("change", () => applyHeatmap());

/* ---------- boot ----------
 * Deferred with queueMicrotask so file ORDER stops being load-bearing. Twice now an
 * appended block landed after this IIFE and boot hit the temporal dead zone on a
 * `const` below it — 'Cannot access prefs/HEAT_NOTES before initialization' — which
 * aborts boot entirely and silently disables every listener it registers. Deferring
 * runs boot after the whole module has evaluated, so a later append cannot break it. */
async function boot() {
  applyLayout();
  applyTypography();
  applyHeatPrefs();
  document.documentElement.style.setProperty("--graph-h", prefs.get("graphH", "460px"));
  $("#topk").value = prefs.get("topk", "20");
  $("#topk-val").textContent = $("#topk").value;
  makeSplitter($("#split-lib"), "--col-lib", "colLib", "lib");
  makeSplitter($("#split-left"), "--col-left", "colLeft", "left");
  makeSplitter($("#split-right"), "--col-right", "colRight", "right");
  bindSearchGraph();
  // Neither blocks the reader: the library and the prompt editor are both side panels,
  // and awaiting them here would delay the paper for two requests that nothing needs yet.
  loadLibrary();
  loadPrompt();

  try {
    const b = await api("/api/breadth");
    BREADTH = b.options;
    $("#breadth").max = String(BREADTH.length - 1);
    const saved = prefs.get("breadth", String(BREADTH.findIndex((x) => x.name === b.default)));
    $("#breadth").value = saved;
    applyBreadth();
  } catch { /* fall back to the server default */ }

  await loadModels();

  const path = location.pathname.match(/^\/p\/(.+?)(?:v(\d+))?$/);
  const q0 = new URLSearchParams(location.search).get("q");
  if (path) openPaper(path[1], location.hash, false);
  else if (q0) { $("#arxiv-input").value = q0; searchPapers(q0, false); }

  const h = await api("/api/health").catch(() => null);
  if (h) setStatus(`${h.vectors.toLocaleString()} chunks indexed`);
}
queueMicrotask(() => boot().catch((e) => {
  console.error('boot failed', e);
  setStatus('startup error: ' + (e?.message || e), 'error');
}));

/* ================= download a model from Hugging Face ================= */

/* Resolve before downloading. A checkpoint is tens of gigabytes, so the dialog reports
 * what the repo actually is — parameters, architecture, quantisation, size, and whether it
 * fits this machine's memory — before a byte is written. Committing to a download because
 * the name looked plausible is an expensive way to discover it was a 70B model. */

let dlResolved = null;
let dlPoll = null;

function openDownloadModal() {
  $("#dl-modal").hidden = false;
  $("#dl-info").innerHTML = "";
  $("#dl-start").hidden = true;
  $("#dl-progress").hidden = true;
  $("#dl-repo").value = "";
  $("#dl-repo").focus();
  api("/api/device").then((d) => {
    const where = d.unified_memory ? "unified RAM" : "VRAM";
    $("#dl-device").textContent =
      `${d.system}/${d.machine} · ${d.accelerator.toUpperCase()} · ${d.budget_gb} GB ${where}`
      + ` · backend: ${d.backend}`;
  }).catch(() => {});
}

$("#dl-close").addEventListener("click", () => {
  $("#dl-modal").hidden = true;
  clearInterval(dlPoll);
});

$("#dl-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const q = $("#dl-repo").value.trim();
  if (!q) return;
  $("#dl-info").innerHTML = `<span class="dim">looking up…</span>`;
  $("#dl-start").hidden = true;
  try {
    const r = await api("/api/model/resolve", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q }),
    });
    dlResolved = r;
    if (!r.exists || r.error) {
      $("#dl-info").innerHTML = `<span class="bad">${escapeHtml(r.error || "not found")}</span>`;
      return;
    }
    const row = (k, v, cls = "") =>
      `<div class="row"><span class="k">${k}</span><span class="v ${cls}">${v}</span></div>`;
    const fit = r.fit;
    let html = row("repo", escapeHtml(r.repo));
    if (r.params) html += row("parameters", (r.params / 1e9).toFixed(1) + "B");
    if (r.arch) html += row("architecture", escapeHtml(r.arch));
    if (r.quantization) html += row("quantization", escapeHtml(r.quantization));
    html += row("download size",
      r.size_gb ? `~${r.size_gb} GB`
                : (r.n_gguf ? "varies by quantisation" : "unknown"));
    if (fit) {
      html += row("fits here",
        fit.fits ? `yes — needs ~${fit.needed_gb} GB of ${fit.budget_gb} GB ${fit.where}`
                 : `no — needs ~${fit.needed_gb} GB but only ${fit.budget_gb} GB ${fit.where}`,
        fit.fits ? "ok" : "bad");
    }
    if (r.already_cached) html += row("status", "already in your cache", "ok");
    if (r.warning) html += `<div class="row"><span class="k"></span><span class="v warn">${escapeHtml(r.warning)}</span></div>`;
    $("#dl-info").innerHTML = html;
    // Offered even when it does not fit: the estimate is conservative and the machine is
    // the user's to judge.
    $("#dl-start").hidden = r.already_cached;
  } catch (err) {
    $("#dl-info").innerHTML = `<span class="bad">${escapeHtml(String(err.message || err))}</span>`;
  }
});

$("#dl-start").addEventListener("click", async () => {
  if (!dlResolved) return;
  $("#dl-start").hidden = true;
  $("#dl-progress").hidden = false;
  try {
    await api("/api/model/download", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo: dlResolved.repo, size_gb: dlResolved.size_gb }),
    });
  } catch (err) {
    $("#dl-progress-text").innerHTML = `<span class="bad">${escapeHtml(String(err.message || err))}</span>`;
    return;
  }
  clearInterval(dlPoll);
  dlPoll = setInterval(async () => {
    try {
      const j = await api(`/api/model/download/${dlResolved.repo}`);
      $("#dl-progress .bar span").style.width = `${j.pct}%`;
      $("#dl-progress-text").textContent =
        `${j.status} · ${j.downloaded_gb} / ${j.total_gb || "?"} GB (${j.pct}%) · ${j.elapsed_s}s`;
      if (j.status === "done") {
        clearInterval(dlPoll);
        $("#dl-progress-text").innerHTML =
          `<span class="ok">downloaded.</span> Restart the generator to serve it: ` +
          `<code>lara serve-llm --model ${escapeHtml(j.repo)}</code>`;
        loadModels();
      } else if (j.status === "error") {
        clearInterval(dlPoll);
        $("#dl-progress-text").innerHTML = `<span class="bad">${escapeHtml(j.error || "failed")}</span>`;
      }
    } catch { clearInterval(dlPoll); }
  }, 1500);
});

/* The picker doubles as the entry point: a sentinel option opens the dialog, so there is
 * no separate button competing for space in the top bar. */
const DL_SENTINEL = "__download__";

async function loadModels() {
  try {
    const m = await api("/api/models");
    const live = new Set(m.loaded?.length ? m.loaded : [m.configured_default].filter(Boolean));
    const opts = m.models.map((x) =>
      `<option value="${x.repo}"${live.has(x.repo) ? " selected" : ""}>` +
      `${x.repo} (${x.size_gb}GB)${live.has(x.repo) ? " — loaded" : " — not loaded"}</option>`);
    opts.push(`<option value="${DL_SENTINEL}">＋ Download new model…</option>`);
    state.models = m.models;
    $("#model").innerHTML = opts.join("");
    syncQuant();
  } catch { /* picker is optional for browsing */ }
}

$("#model").addEventListener("change", (ev) => {
  if (ev.target.value === DL_SENTINEL) {
    // Restore the previous selection so the sentinel never becomes the active model.
    ev.target.value = state.models.find((x) => x.loaded)?.repo || state.models[0]?.repo || "";
    openDownloadModal();
  }
  syncQuant();
});
