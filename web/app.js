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
    renderTaste();          // "where should I jump" is per-paper
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
      if (state.paper !== id) return;            // reader moved on while we waited
      if (fresh.html) {
        $("#paper").innerHTML = fresh.html;
        applyHeatmap();
        setStatus(`fetched ${d.chunks || fresh.n_chunks} chunks via ${d.source || "cache"}`);
      } else {
        /* "ok" with nothing to render. The status tracks indexed chunks, not renderable
         * HTML, so this means the fetch could not put a document on disk. Say so instead
         * of leaving the spinner up forever, which is what it used to do. */
        const el = $("#paper").querySelector(".fetching");
        if (el) el.innerHTML =
          `Indexed, but no renderable copy could be retrieved. ` +
          `<a href="https://arxiv.org/abs/${id}" target="_blank" rel="noopener">Open on arXiv</a>`;
        setStatus("full text not renderable", "error");
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
  scrollIntoPane(el, $("#paper-scroll"));
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
  const float = $("#sel-float");
  if (!text || text.length < 12 || !$("#paper").contains(sel.anchorNode)) {
    float.hidden = true;
    return;
  }
  $("#taste-float").classList.remove("done");
  $("#taste-float").textContent = "\u2605 Interesting";
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
  $("#sel-float").hidden = true;
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
  $("#sel-float").hidden = true;
  $("#question").focus();
});

/* Marking is deliberately NOT dismissing: a reader marking a passage is often about to
 * ask about it too, and closing the popup would make the two actions mutually exclusive. */
$("#taste-float").addEventListener("click", async () => {
  if (!state.candidate || !state.paper) return;
  const btn = $("#taste-float");
  btn.textContent = "…";
  try {
    await api("/api/taste/mark", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arxiv_id: state.paper, text: state.candidate }),
    });
    btn.textContent = "\u2605 Saved";
    btn.classList.add("done");
    loadTaste();
    if (prefs.get("heatMode", "answer") === "taste") applyHeatmap();
  } catch (err) {
    btn.textContent = "failed";
    setStatus(String(err.message || err), "error");
  }
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
        const ev = /^event: (.+)$/m.exec(frame);
        const dm = /data: ([\s\S]*)$/.exec(frame);
        if (!ev || !dm) continue;
        if (ev[1] === "step") {
          const st = JSON.parse(dm[1]);
          // Show what was actually searched. A follow-up is rewritten into a standalone
          // query before retrieval, and silently searching for something other than what
          // the reader typed is the kind of helpfulness that erodes trust when noticed.
          if (st.kind === "rewrite") {
            addStep(answerEl, "rewrite", `searched for: "${st.detail}"`);
          } else {
            addStep(answerEl, st.kind, st.detail);
          }
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
        } else if (ev[1] === "context") {
          renderContext(answerEl, JSON.parse(dm[1]));
        } else if (ev[1] === "perf") {
          renderPerf(answerEl, JSON.parse(dm[1]));
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
  // Maths first: it runs on escaped text and emits its own tags, so doing it after the
  // citation pass would let a <sub> land inside an <a> title and vice versa.
  return renderMath(escapeHtml(text)).replace(/\[(\d+)\]/g, (m, n) => {
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
  $("#paper-scroll").hidden = true;
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
  $("#paper-scroll").hidden = false;
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
/* Clicking a question restores the whole THREAD, not the single exchange.
 *
 * Threads are keyed by paper, exactly as the server keys them, so what is shown here is
 * what the model will be given as history on the next question. Restoring one turn in
 * isolation would show the reader less context than the model has, which makes the next
 * answer look like it came from nowhere. */
async function restoreEntry(e) {
  if (e.arxiv_id) await openPaper(e.arxiv_id);
  if (e.kind !== "question") return;

  const key = e.arxiv_id || "";
  const thread = library.entries
    .filter((x) => x.kind === "question" && (x.arxiv_id || "") === key)
    .sort((a, b) => String(a.created_utc || "").localeCompare(String(b.created_utc || "")));

  $("#messages").innerHTML = "";
  let clicked = null;
  for (const t of thread) {
    addMessage("user", t.question, t.selection || null);
    const el = addMessage("assistant", "");
    el.classList.add("restored");
    if (t.id === e.id) { el.classList.add("focused"); clicked = el; }
    const when = escapeHtml((t.created_utc || "").slice(0, 16).replace("T", " "));
    el.querySelector(".text").innerHTML =
      `<span class="dim">from your library · ${when}</span><br>` +
      // Same rendering as a live answer: maths and citations, not escaped source.
      renderMath(escapeHtml(t.answer || "(no answer recorded)"));
  }

  $("#question").value = "";
  $("#question").placeholder = thread.length > 1
    ? `Continue this thread (${thread.length} questions)…`
    : "Ask a follow-up…";
  state.heatRef = { text: e.question, kind: "question" };
  loadGraph();
  clicked?.scrollIntoView({ block: "center" });
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

/* ================= taste profile ================= */

/* The reader's marked passages, as a set of positions in embedding space. Deliberately a
 * SET and not a centroid: summing cosine similarities is arithmetically identical to
 * searching once with the summed vector, so "average my interests" and "add up their
 * similarities" are the same operation, and both match the midpoint between optimisers and
 * retrieval rather than either. The reductions that keep interests distinct — max, and the
 * LogSumExp soft maximum the server defaults to — are the reason the marks are kept apart.
 * See TASTE_REDUCTIONS in lara/serve/app.py. */

let tasteMarks = 0;
let tasteScope = "paper";       // paper | corpus

async function loadTaste() {
  try {
    const d = await api("/api/taste");
    tasteMarks = d.n || 0;
  } catch {
    tasteMarks = 0;
  }
  renderTaste();
}

function tasteBar(score, best) {
  const w = Math.max(3, Math.round(46 * (best > 0 ? score / best : 0)));
  return `<span class="bar" style="width:${w}px"></span>`;
}

async function renderTaste() {
  const pane = $("#taste-pane"), list = $("#taste-list"), count = $("#taste-count");
  if (!pane) return;
  pane.hidden = tasteMarks === 0 && tasteScope === "paper";
  if (count) count.textContent = tasteMarks ? `${tasteMarks} marked` : "";
  if (!tasteMarks) {
    list.innerHTML = `<p class="taste-empty">Highlight a passage you found interesting and
      press <b>★ Interesting</b>. Once a few are saved, the passages worth jumping to
      appear here.</p>`;
    pane.hidden = false;
    return;
  }
  $("#taste-recommend").classList.toggle("on", tasteScope === "corpus");

  let data;
  try {
    data = tasteScope === "corpus"
      ? await api("/api/taste/recommend?k=12")
      : (state.paper ? await api(`/api/taste/paper/${encodeURIComponent(state.paper)}?k=10`)
                     : { chunks: [] });
  } catch {
    list.innerHTML = `<p class="taste-empty">could not score against your profile</p>`;
    return;
  }
  const chunks = data.chunks || [];
  if (!chunks.length) {
    list.innerHTML = `<p class="taste-empty">${
      tasteScope === "corpus" ? "no matches yet" : "open a paper to see where to jump"}</p>`;
    return;
  }
  const best = Math.max(...chunks.map((c) => c.score));
  list.innerHTML = chunks.map((c) => `
    <button class="taste-item" data-anchor="${escapeHtml(c.anchor || "")}"
            data-start="${c.char_start}" data-end="${c.char_end}"
            data-endanchor="${escapeHtml(c.anchor_end || c.anchor || "")}"
            data-paper="${escapeHtml(c.arxiv_id || "")}">
      ${tasteBar(c.score, best)}${escapeHtml((c.preview || "").slice(0, 96))}…
      <div class="src">${c.score.toFixed(3)}${
        c.title ? " · " + escapeHtml(c.title.slice(0, 40)) : ""}</div>
    </button>`).join("");
}

$("#taste-list")?.addEventListener("click", async (ev) => {
  const b = ev.target.closest(".taste-item");
  if (!b) return;
  const frag = `#${b.dataset.anchor}:${b.dataset.start}-${b.dataset.end}`;
  if (b.dataset.paper && b.dataset.paper !== state.paper) {
    await openPaper(b.dataset.paper, frag);
  } else {
    scrollToAnchor(frag);
  }
});

$("#taste-recommend")?.addEventListener("click", () => {
  tasteScope = tasteScope === "corpus" ? "paper" : "corpus";
  renderTaste();
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

  // Taste has no per-question reference vector — the profile IS the reference — so it uses
  // its own endpoint, which reduces a set of vectors rather than scoring against one.
  if (mode === "taste") {
    if (!tasteMarks) return;
    try {
      const d = await api(
        `/api/taste/paper/${encodeURIComponent(state.paper)}?k=${Number(prefs.get("heatK", "5"))}`);
      paintHeatmap(d.chunks || [], mode);
    } catch { /* decoration only */ }
    return;
  }

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
     relevance to ${ {answer: "the answer passage",
                      taste: "your taste profile"}[mode] || "your question" }
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
  taste:  "Shades passages matching what you have marked interesting. Needs no question, "
        + "so it works the moment a paper opens.",
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
  loadTaste();

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
/* /api/device, cached so the dialog can report what fits without refetching. */
let dlDevice = null;

/* Where to actually find each format. A model name alone is not enough: "Qwen3-8B" names
 * three different sets of files depending on who converted it, and only one of them will
 * load here. Keyed by /api/device's wants_format. */
const FORMAT_GUIDE = {
  gguf: {
    label: "GGUF",
    placeholder: "e.g. unsloth/Qwen3-8B-GGUF, or paste a Hugging Face URL",
    help: `Browse <a href="https://huggingface.co/models?library=gguf" target="_blank"
             rel="noopener">all GGUF models on Hugging Face</a> — or go straight to
           <a href="https://huggingface.co/bartowski" target="_blank" rel="noopener">bartowski</a>
           and <a href="https://huggingface.co/unsloth" target="_blank" rel="noopener">unsloth</a>,
           who publish GGUF builds of most popular models. Pick a <code>Q4_K_M</code> file
           for the 4-bit sizing above; a plain safetensors repo will not load.`,
  },
  mlx: {
    label: "MLX",
    placeholder: "e.g. mlx-community/Qwen3-8B-4bit, or paste a Hugging Face URL",
    help: `Browse <a href="https://huggingface.co/mlx-community" target="_blank"
             rel="noopener">mlx-community</a>, which hosts essentially every MLX
           conversion. Look for a <code>-4bit</code> suffix to match the sizing above.`,
  },
  safetensors: {
    label: "safetensors",
    placeholder: "e.g. Qwen/Qwen3-8B, or paste a Hugging Face URL",
    help: `Standard Hugging Face repos ship this format, so most model pages work as-is.
           GGUF builds will not load under vLLM.`,
  },
};

function openDownloadModal() {
  $("#dl-modal").hidden = false;
  $("#dl-info").innerHTML = "";
  $("#dl-start").hidden = true;
  $("#dl-progress").hidden = true;
  $("#dl-repo").value = "";
  $("#dl-repo").focus();
  api("/api/device").then((d) => {
    dlDevice = d;
    const where = d.unified_memory ? "unified RAM" : "VRAM";
    $("#dl-device").textContent =
      `${d.system}/${d.machine} · ${d.accelerator.toUpperCase()} · ${d.budget_gb} GB ${where}`
      + ` · backend: ${d.backend}`;
    /* Both null until `lara setup` has run — say so rather than showing a blank line,
     * since "no advice" and "your machine fits nothing" look identical otherwise. */
    const hint = $("#dl-fits");
    if (d.generator_max_params_4bit) {
      const b = d.generator_max_params_4bit / 1e9;
      hint.innerHTML =
        `With your index loaded, about <strong>${b >= 10 ? b.toFixed(0) : b.toFixed(1)}B `
        + `parameters at 4-bit</strong> fits in the ${d.generator_headroom_gb} GB left over. `
        + `Bigger models still download — they just will not load alongside the corpus.`;
    } else {
      hint.innerHTML =
        `Run <code>lara setup</code> to have this dialog tell you what size fits.`;
    }

    /* The three weight formats are not interchangeable, and picking the wrong one is a
     * multi-gigabyte mistake you only discover at load time. Say which one this machine
     * needs, and where to find it, before the box is typed into. */
    const fmt = $("#dl-format");
    const backend = escapeHtml(d.backend || "the configured backend");
    if (FORMAT_GUIDE[d.wants_format]) {
      const g = FORMAT_GUIDE[d.wants_format];
      fmt.innerHTML = `<strong>${backend} needs ${g.label} weights.</strong> ${g.help}`;
      $("#dl-repo").placeholder = g.placeholder;
    } else {
      fmt.innerHTML = "";
    }
  }).catch(() => {});
}

$("#dl-close").addEventListener("click", () => {
  $("#dl-modal").hidden = true;
  clearInterval(dlPoll);
});

/* KV cache and activations on top of the weights — same 1.35x the setup wizard uses. */
const KV_OVERHEAD = 1.35;

function renderResolved(r) {
  if (!r) return;
  const row = (k, v, cls = "") =>
    `<div class="row"><span class="k">${k}</span><span class="v ${cls}">${v}</span></div>`;
  let html = row("repo", escapeHtml(r.repo));
  if (r.params) html += row("parameters", (r.params / 1e9).toFixed(1) + "B");
  if (r.arch) html += row("architecture", escapeHtml(r.arch));
  if (r.quantization) html += row("quantization", escapeHtml(r.quantization));

  /* The quantisation is a property of the repo, not a choice: the server picks the
   * 4-bit build and this reports it. The repo still physically contains every other
   * quantisation, which is why the download names its files rather than the repo. */
  if (r.pick) html += row("quantisation", escapeHtml(r.pick));

  const size = r.size_gb;
  html += row("download size",
    size ? `~${size.toFixed(1)} GB` : (r.n_gguf ? "pick a quantisation above" : "unknown"));

  /* Recomputed here rather than reusing r.fit, which the server calculated for the
   * default quantisation and which goes stale the moment the selection changes. */
  if (size && dlDevice && dlDevice.budget_gb) {
    const need = size * KV_OVERHEAD;
    const budget = dlDevice.budget_gb;
    const where = dlDevice.unified_memory ? "unified RAM" : "VRAM";
    const ok = need <= budget;
    html += row("fits here",
      ok ? `yes — needs ~${need.toFixed(1)} GB of ${budget} GB ${where}`
         : `no — needs ~${need.toFixed(1)} GB but only ${budget} GB ${where}`,
      ok ? "ok" : "bad");
  } else if (r.fit) {
    html += row("fits here",
      r.fit.fits ? `yes — needs ~${r.fit.needed_gb} GB of ${r.fit.budget_gb} GB ${r.fit.where}`
                 : `no — needs ~${r.fit.needed_gb} GB but only ${r.fit.budget_gb} GB ${r.fit.where}`,
      r.fit.fits ? "ok" : "bad");
  }

  if (r.already_cached) html += row("status", "already in your cache", "ok");
  if (r.warning) {
    html += `<div class="row"><span class="k"></span>`
          + `<span class="v warn">${escapeHtml(r.warning)}</span></div>`;
  }
  $("#dl-info").innerHTML = html;
  // Offered even when it does not fit: the estimate is conservative and the machine is
  // the user's to judge.
  $("#dl-start").hidden = !!r.already_cached;
}

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
    renderResolved(r);
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
      body: JSON.stringify({
        repo: dlResolved.repo,
        size_gb: dlResolved.size_gb,
        // Only the picked quantisation's files. Null for safetensors repos, where the
        // whole snapshot is the model. Without this a GGUF repo fetches every
        // quantisation it ships.
        files: dlResolved.pick_files || null,
      }),
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

let dlAutoOpened = false;

async function loadModels() {
  try {
    const m = await api("/api/models");
    const live = new Set(m.loaded?.length ? m.loaded : [m.configured_default].filter(Boolean));
    /* A model whose backend is not installed stays listed but disabled: it is genuinely
     * unusable until the runtime exists, and silently dropping it is what made a model
     * the user had just downloaded appear nowhere at all. The label carries the fix. */
    const opts = m.models.map((x) => {
      const isLive = live.has(x.repo);
      const state = x.needs_install ? ` — ${x.hint}`
                  : isLive ? " — loaded"
                  : " — not loaded";
      return `<option value="${x.repo}"${isLive ? " selected" : ""}`
           + `${x.needs_install ? " disabled" : ""}>`
           + `${x.repo} (${x.size_gb}GB)${escapeHtml(state)}</option>`;
    });
    /* With an empty cache the sentinel would be the *only* option, which makes it the
     * selection already — so choosing it fires no `change` event and the dialog never
     * opened. A disabled placeholder keeps the sentinel something you can change *to*. */
    if (!m.models.length) {
      opts.unshift(`<option value="" disabled selected>no model in cache</option>`);
    }
    opts.push(`<option value="${DL_SENTINEL}">＋ Download new model…</option>`);
    state.models = m.models;
    $("#model").innerHTML = opts.join("");
    syncQuant();

    /* Nothing to generate with, so the download dialog is the only useful next step.
     * Once per page load: re-opening it after every poll would trap the user. */
    if (!m.models.length && !dlAutoOpened) {
      dlAutoOpened = true;
      openDownloadModal();
    }
    if (m.models.length) dlAutoOpened = false;
  } catch { /* picker is optional for browsing */ }
}

$("#model").addEventListener("change", (ev) => {
  if (ev.target.value === DL_SENTINEL) {
    // Restore the previous selection so the sentinel never becomes the active model.
    // Never fall back to one whose backend is missing — it cannot be served.
    const usable = state.models.filter((x) => !x.needs_install);
    ev.target.value = usable.find((x) => x.loaded)?.repo || usable[0]?.repo || "";
    openDownloadModal();
  }
  syncQuant();
});

/* ── Deep Automated Research ──────────────────────────────────────────────────
 *
 * A run takes minutes, so nothing waits for the end. Rounds appear as they start,
 * confirmed passages appear as the model names them, and both answers stream in.
 * A progress display that only resolves at completion is indistinguishable from a
 * hang, and asks the reader to wait on faith.
 *
 * Each round is a <details>: open while it is the newest, collapsed once the next
 * begins. A twenty-round run stays readable without hiding the part that is live.
 */
(function deepResearch() {
  const view = document.getElementById("deep");
  if (!view) return;
  const q = document.getElementById("deep-q");
  const form = document.getElementById("deep-form");
  const goBtn = document.getElementById("deep-go");
  const stopBtn = document.getElementById("deep-stop");
  const statusEl = document.getElementById("deep-status");
  const graph = document.getElementById("deep-graph");
  const roundsEl = document.getElementById("deep-rounds");
  const tldrEl = document.getElementById("deep-tldr");
  const thoroughEl = document.getElementById("deep-thorough");
  const history = document.getElementById("deep-history");

  let controller = null;
  let current = null;          // the open round's <details>

  const show = (on) => {
    view.hidden = !on;
    document.body.classList.toggle("deep-open", on);
    if (on) q.focus();
  };
  document.getElementById("deep-btn")?.addEventListener("click", () => {
    show(true);
    loadHistory();
  });
  document.getElementById("deep-close")?.addEventListener("click", () => {
    show(false);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !view.hidden && !controller) show(false);
  });

  function setStatusDeep(text) { statusEl.textContent = text || ""; }

  function roundEl(n, query, via) {
    const d = document.createElement("details");
    d.className = "deep-round";
    d.open = true;
    d.innerHTML =
      `<summary><b>Round ${n}</b>` +
      `<span class="tag${via === "citations" ? " cit" : ""}">` +
      `${via === "citations" ? "citation graph" : "similarity"}</span>` +
      `<span class="rq">${escapeHtml(query || "")}</span>` +
      `<span class="rstat" style="margin-left:auto;opacity:.7"></span></summary>` +
      `<div class="deep-claims"></div>`;
    graph.appendChild(d);
    d.scrollIntoView({ block: "nearest" });
    return d;
  }

  /* The model's name for a passage is the link text, and the link is the real citation
   * URL, so a claim in the graph opens the exact passage it came from. */
  function claimEl(c) {
    const div = document.createElement("div");
    div.className = "deep-claim";
    const url = `/p/${c.arxiv_id}#chunk-${c.chunk_id}`;
    const M = (t) => renderMath(escapeHtml(t || ""));
    const meas = c.value
      ? `<span class="cmeas">${M(c.metric || "measured")} = ${M(c.value)}</span>`
      : "";
    const cond = c.condition ? `<span class="ccond">${M(c.condition)}</span>` : "";
    div.innerHTML =
      `<a class="cname" href="${url}" title="${escapeHtml(c.paper_title || "")}">` +
      `${escapeHtml(c.name || "(unnamed)")}</a>` +
      `<span class="cpaper">${escapeHtml(c.arxiv_id)}${c.section ? " · " + escapeHtml(c.section) : ""}</span>` +
      `<span class="cclaim">${M(c.claim || "")}</span>${meas}${cond}`;
    return div;
  }

  /* Minimal markdown: headings, bold, and [12345] chunk citations turned into links.
   * Deliberately not a full renderer — the answers are prose with citations, and a
   * dependency-free subset that never mangles LaTeX beats a complete one that does. */
  function render(md) {
    const esc = renderMath(escapeHtml(md || ""));
    return esc
      .replace(/^#{3} (.*)$/gm, "<h3>$1</h3>")
      .replace(/^#{2} (.*)$/gm, "<h2>$1</h2>")
      .replace(/^# (.*)$/gm, "<h2>$1</h2>")
      .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
      // A citation may carry several ids: [7127673, 7127615, 7127613]. Linking only the
      // single-id form left the most heavily-evidenced claims — the ones with the most
      // support — as the only ones a reader could not click through.
      .replace(/\[(\d{3,}(?:\s*,\s*\d{3,})*)\]/g, (_m, ids) =>
        "[" + ids.split(/\s*,\s*/).map((id) =>
          `<a class="cite" href="#" data-chunk="${id}">${id}</a>`).join(", ") + "]")
      .split(/\n{2,}/).map((p) => (/^<h[23]>/.test(p) ? p : `<p>${p.replace(/\n/g, " ")}</p>`))
      .join("");
  }

  graph.addEventListener("click", (e) => {
    const a = e.target.closest("a.cname");
    if (a && !e.metaKey && !e.ctrlKey) { /* let normal navigation happen */ }
  });

  async function loadHistory() {
    try {
      const r = await fetch("/api/synthesis/runs?limit=50");
      const { runs } = await r.json();
      history.innerHTML = `<option value="">past runs (${runs.length})…</option>` +
        runs.map((x) =>
          `<option value="${x.run_id}">${escapeHtml(x.question.slice(0, 70))} · ` +
          `${x.n_claims} claims / ${x.n_papers} papers</option>`).join("");
    } catch { /* history is a convenience; its absence must not block a new run */ }
  }

  history?.addEventListener("change", async () => {
    if (!history.value) return;
    const r = await fetch(`/api/synthesis/run/${history.value}`);
    if (!r.ok) return;
    const run = await r.json();
    q.value = run.question;
    graph.innerHTML = "";
    const byRound = new Map();
    for (const c of run.claims) {
      if (!byRound.has(c.round_n)) byRound.set(c.round_n, []);
      byRound.get(c.round_n).push(c);
    }
    for (const rd of run.rounds) {
      const d = roundEl(rd.n, rd.query, rd.via);
      d.open = false;
      d.querySelector(".rstat").textContent =
        `${rd.relevant}/${rd.retrieved} relevant · +${rd.new_papers} papers`;
      const box = d.querySelector(".deep-claims");
      for (const c of byRound.get(rd.n) || []) box.appendChild(claimEl(c));
      if (rd.gap) {
        const g = document.createElement("div");
        g.className = "deep-gap";
        g.innerHTML = `<b>gap:</b> ${escapeHtml(rd.gap)}`;
        d.appendChild(g);
      }
    }
    roundsEl.textContent = `${run.n_rounds} rounds · ${run.n_claims} claims · ${run.n_papers} papers`;
    tldrEl.innerHTML = render(run.tldr);
    thoroughEl.innerHTML = render(run.thorough);
    setStatusDeep(`saved run · ${Math.round((run.ms || 0) / 1000)}s`);
  });

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = q.value.trim();
    if (!question || controller) return;

    graph.innerHTML = "";
    tldrEl.innerHTML = "";
    thoroughEl.innerHTML = "";
    roundsEl.textContent = "";
    current = null;
    controller = new AbortController();
    goBtn.disabled = true;
    stopBtn.hidden = false;
    const t0 = performance.now();
    let claims = 0, papers = new Set(), rounds = 0;

    try {
      const res = await fetch("/api/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal: controller.signal,
      });
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        // SSE frames are separated by a blank line; a frame can arrive split across
        // reads, so only whole frames are consumed and the remainder is kept.
        const frames = buf.split("\n\n");
        buf = frames.pop();
        for (const f of frames) {
          const ev = /^event: (.+)$/m.exec(f);
          const da = /^data: ([\s\S]*)$/m.exec(f);
          if (!ev || !da) continue;
          let payload;
          try { payload = JSON.parse(da[1]); } catch { continue; }
          handle(ev[1], payload);
        }
      }
    } catch (err) {
      if (err.name !== "AbortError") setStatusDeep(`error: ${err.message}`);
    } finally {
      controller = null;
      goBtn.disabled = false;
      stopBtn.hidden = true;
      loadHistory();
    }

    function handle(name, p) {
      if (name === "round" && p.phase === "retrieving") {
        if (current) current.open = false;      // collapse the previous round
        current = roundEl(p.n, p.query, p.via);
        rounds = p.n;
        setStatusDeep(`round ${p.n} · searching`);
      } else if (name === "round" && p.phase === "reading") {
        setStatusDeep(`round ${p.n} · reading ${p.n_chunks} passages`);
      } else if (name === "claims") {
        const box = current?.querySelector(".deep-claims");
        for (const c of p.claims) {
          claims += 1;
          papers.add(c.arxiv_id);
          box?.appendChild(claimEl(c));
        }
        roundsEl.textContent = `${rounds} rounds · ${claims} claims · ${papers.size} papers`;
      } else if (name === "round_done") {
        const st = current?.querySelector(".rstat");
        if (st) st.textContent =
          `${p.relevant}/${p.retrieved} relevant · +${p.new_papers} papers · ${Math.round(p.ms)}ms`;
        if (p.relevant === 0 && current) {
          const n = document.createElement("div");
          n.className = "deep-note";
          n.textContent = "nothing new in this round";
          current.appendChild(n);
        }
      } else if (name === "decision") {
        if (!current) return;
        const g = document.createElement("div");
        g.className = "deep-gap";
        const forced = p.forced_continue ? " (minimum depth not yet reached)" : "";
        const votes = p.decision === "stop" ? ` — stop vote ${p.stop_votes}/${p.needed}` : "";
        g.innerHTML = `<b>${escapeHtml(p.decision)}${escapeHtml(votes)}${escapeHtml(forced)}</b>` +
          (p.gap ? ` · gap: ${escapeHtml(p.gap)}` : "");
        current.appendChild(g);
      } else if (name === "consolidating") {
        if (current) current.open = false;
        setStatusDeep(`consolidating ${p.claims} claims from ${p.papers} papers…`);
      } else if (name === "token") {
        const el = p.target === "tldr" ? tldrEl : thoroughEl;
        el.dataset.raw = (el.dataset.raw || "") + p.text;
        el.innerHTML = render(el.dataset.raw);
      } else if (name === "done") {
        tldrEl.dataset.raw = p.tldr; tldrEl.innerHTML = render(p.tldr);
        thoroughEl.dataset.raw = p.thorough; thoroughEl.innerHTML = render(p.thorough);
        roundsEl.textContent =
          `${p.rounds} rounds · ${p.claims} claims · ${p.papers} papers`;
        setStatusDeep(`done in ${Math.round(p.ms / 1000)}s — ${p.stopped_because}`);
      } else if (name === "error") {
        setStatusDeep(`error: ${String(p).slice(0, 200)}`);
      }
    }
  });

  stopBtn?.addEventListener("click", () => {
    // Aborting closes the stream, which the server reads as a cancel: the run still
    // consolidates what it gathered rather than throwing away minutes of retrieval.
    controller?.abort();
    setStatusDeep("stopping — consolidating what was found so far…");
  });
})();

/* ── Inline LaTeX ─────────────────────────────────────────────────────────────
 *
 * 66% of corpus chunks contain $math$. Papers themselves render fine — LaTeXML
 * ships MathML and browsers handle that natively — but chunk *text* is stored as
 * each element's `alttext`, i.e. the original LaTeX. So every excerpt, answer and
 * extracted claim showed raw `$O(\epsilon^{-3})$` until now.
 *
 * This is deliberately not KaTeX. Vendoring it means ~1 MB of third-party code and
 * webfonts inside a proprietary repository, and the math that actually appears in
 * these answers is inline notation — subscripts, Greek, big-O, fractions — not
 * displayed multi-line equations. The command set was measured rather than guessed:
 * frac, mathcal, in, left/right, mathbf, leq, displaystyle, bm, sum, prime, text,
 * alpha, textbf, theta, hat, mathbb, tilde account for nearly all of it.
 *
 * Anything it cannot express degrades to the source text with the delimiters
 * removed, which is what was displayed before — never worse, usually much better.
 * If full fidelity is wanted later, KaTeX drops into `texToHtml` unchanged.
 */
const MATH_SYMBOLS = {
  alpha:"α",beta:"β",gamma:"γ",delta:"δ",epsilon:"ε",varepsilon:"ε",zeta:"ζ",eta:"η",
  theta:"θ",vartheta:"ϑ",iota:"ι",kappa:"κ",lambda:"λ",mu:"μ",nu:"ν",xi:"ξ",pi:"π",
  rho:"ρ",sigma:"σ",tau:"τ",upsilon:"υ",phi:"φ",varphi:"φ",chi:"χ",psi:"ψ",omega:"ω",
  Gamma:"Γ",Delta:"Δ",Theta:"Θ",Lambda:"Λ",Xi:"Ξ",Pi:"Π",Sigma:"Σ",Upsilon:"Υ",
  Phi:"Φ",Psi:"Ψ",Omega:"Ω",
  times:"×",cdot:"·",cdots:"⋯",ldots:"…",dots:"…",div:"÷",pm:"±",mp:"∓",
  leq:"≤",le:"≤",geq:"≥",ge:"≥",neq:"≠",ne:"≠",approx:"≈",sim:"∼",simeq:"≃",
  equiv:"≡",propto:"∝",ll:"≪",gg:"≫",
  in:"∈",notin:"∉",subset:"⊂",subseteq:"⊆",supset:"⊃",supseteq:"⊇",cup:"∪",cap:"∩",
  emptyset:"∅",forall:"∀",exists:"∃",neg:"¬",land:"∧",lor:"∨",
  to:"→",rightarrow:"→",Rightarrow:"⇒",leftarrow:"←",Leftarrow:"⇐",
  leftrightarrow:"↔",Leftrightarrow:"⇔",mapsto:"↦",implies:"⟹",
  infty:"∞",partial:"∂",nabla:"∇",sum:"∑",prod:"∏",int:"∫",oint:"∮",
  sqrt:"√",angle:"∠",perp:"⊥",parallel:"∥",star:"⋆",ast:"∗",circ:"∘",
  oplus:"⊕",otimes:"⊗",odot:"⊙",bullet:"•",dagger:"†",prime:"′",ell:"ℓ",hbar:"ℏ",
  Re:"ℜ",Im:"ℑ",aleph:"ℵ",top:"⊤",bot:"⊥",vdots:"⋮",ddots:"⋱",lVert:"‖",rVert:"‖",
  dotsc:"…",dotsb:"⋯",dotso:"…",natural:"♮",flat:"♭",sharp:"♯",
  langle:"⟨",rangle:"⟩",lfloor:"⌊",rfloor:"⌋",lceil:"⌈",rceil:"⌉",quad:" ",qquad:"  ",
};
const BLACKBOARD = {R:"ℝ",N:"ℕ",Z:"ℤ",Q:"ℚ",C:"ℂ",E:"𝔼",P:"ℙ",1:"𝟙"};
const CALLIGRAPHIC = {A:"𝒜",B:"ℬ",C:"𝒞",D:"𝒟",E:"ℰ",F:"ℱ",G:"𝒢",H:"ℋ",I:"ℐ",J:"𝒥",
  K:"𝒦",L:"ℒ",M:"ℳ",N:"𝒩",O:"𝒪",P:"𝒫",Q:"𝒬",R:"ℛ",S:"𝒮",T:"𝒯",U:"𝒰",V:"𝒱",W:"𝒲",
  X:"𝒳",Y:"𝒴",Z:"𝒵"};
const UPRIGHT_FNS = ["log","exp","min","max","argmin","argmax","sin","cos","tan","det",
  "dim","ker","deg","gcd","lim","sup","inf","ln","Pr","tr","diag","softmax","sign","rank"];

/* Read the balanced {...} beginning at `i`, or a single following character.
   LaTeX groups nest, so a regex cannot do this: \frac{\frac{a}{b}}{c} is common. */
function texGroup(s, i) {
  if (s[i] !== "{") return [s[i] ?? "", i + 1];
  let depth = 0;
  for (let j = i; j < s.length; j++) {
    if (s[j] === "{") depth++;
    else if (s[j] === "}" && --depth === 0) return [s.slice(i + 1, j), j + 1];
  }
  return [s.slice(i + 1), s.length];
}

function texToHtml(src) {
  let out = "";
  for (let i = 0; i < src.length; ) {
    const ch = src[i];
    if (ch === "\\") {
      const m = /^\\([a-zA-Z]+|.)/.exec(src.slice(i));
      if (!m) { i++; continue; }
      const cmd = m[1];
      i += m[0].length;
      if (cmd === "frac" || cmd === "dfrac" || cmd === "tfrac") {
        const [num, i1] = texGroup(src, i); const [den, i2] = texGroup(src, i1);
        i = i2;
        out += `<span class="mfrac"><span class="mnum">${texToHtml(num)}</span>` +
               `<span class="mden">${texToHtml(den)}</span></span>`;
      } else if (cmd === "sqrt") {
        const [a, i1] = texGroup(src, i); i = i1;
        out += `√<span class="msqrt">${texToHtml(a)}</span>`;
      } else if (cmd === "mathbb") {
        const [a, i1] = texGroup(src, i); i = i1;
        out += BLACKBOARD[a] || `<span class="mbb">${texToHtml(a)}</span>`;
      } else if (cmd === "mathcal" || cmd === "mathscr") {
        const [a, i1] = texGroup(src, i); i = i1;
        out += CALLIGRAPHIC[a] || `<i>${texToHtml(a)}</i>`;
      } else if (cmd === "mathbf" || cmd === "bm" || cmd === "boldsymbol" || cmd === "textbf") {
        const [a, i1] = texGroup(src, i); i = i1;
        out += `<b>${texToHtml(a)}</b>`;
      } else if (cmd === "text" || cmd === "mathrm" || cmd === "textrm" || cmd === "operatorname") {
        const [a, i1] = texGroup(src, i); i = i1;
        out += `<span class="mup">${texToHtml(a)}</span>`;
      } else if (cmd === "mathit" || cmd === "textit" || cmd === "emph") {
        const [a, i1] = texGroup(src, i); i = i1;
        out += `<i>${texToHtml(a)}</i>`;
      } else if (cmd === "hat" || cmd === "tilde" || cmd === "bar" || cmd === "vec" ||
                 cmd === "dot" || cmd === "widehat" || cmd === "widetilde") {
        const [a, i1] = texGroup(src, i); i = i1;
        const acc = {hat:"̂",widehat:"̂",tilde:"̃",widetilde:"̃",
                     bar:"̄",vec:"⃗",dot:"̇"}[cmd];
        out += texToHtml(a) + acc;             // combining mark renders over the glyph
      } else if (UPRIGHT_FNS.includes(cmd)) {
        out += `<span class="mup">${cmd}</span>`;
      } else if (cmd === "left" || cmd === "right" || cmd === "displaystyle" ||
                 cmd === "textstyle" || cmd === "limits" || cmd === "nolimits" ||
                 cmd === "!" || cmd === "," || cmd === ";" || cmd === ":") {
        // Sizing and spacing hints with no meaning outside a real typesetter.
        if (cmd === "," || cmd === ";" || cmd === ":") out += " ";
      } else if (MATH_SYMBOLS[cmd] !== undefined) {
        out += MATH_SYMBOLS[cmd];
      } else if (cmd === "\\") {
        out += "<br>";
      } else if (cmd === "lx" && src.slice(i, i + 12) === "@sectionsign") {
        out += "§"; i += 12;                        // LaTeXML internal, common in alttext
      } else if (/^[a-zA-Z]+$/.test(cmd)) {
        out += `<span class="mup">${cmd}</span>`;   // unknown command: show its name
      } else {
        out += cmd;                                 // escaped literal: \{ \% \_ \&
      }
    } else if (ch === "^" || ch === "_") {
      const [a, i1] = texGroup(src, i + 1); i = i1;
      out += ch === "^" ? `<sup>${texToHtml(a)}</sup>` : `<sub>${texToHtml(a)}</sub>`;
    } else if (ch === "{" || ch === "}") {
      i++;                                          // grouping braces are not content
    } else if (ch === "&") {
      // Already-escaped entity from the caller (&amp; &lt; &gt;): copy it whole.
      const ent = /^&[a-z]+;|^&#\d+;/.exec(src.slice(i));
      if (ent) { out += ent[0]; i += ent[0].length; } else { out += "&amp;"; i++; }
    } else {
      out += ch; i++;
    }
  }
  return out;
}

/* Replace $…$ and $$…$$ in an ALREADY HTML-ESCAPED string.
   Escaped input is the precondition: this emits tags, so running it on raw text
   would let a passage containing markup inject it. */
/* Prose, not maths. Chunking can split a formula, leaving a chunk with an odd number
   of `$`; pairing then runs from one formula's closing delimiter to the next one's
   opening one and swallows the sentence between them — the observed case was
   "$X$ -QAM scheme, where $Y$" rendering the prose as maths. Real inline maths is short
   and carries at least one LaTeX marker, so anything long, marker-free and full of
   ordinary words is left exactly as it was. */
function looksLikeProse(src) {
  if (/[\\^_{}]/.test(src)) return false;
  const words = src.trim().split(/\s+/);
  return words.length >= 4 || src.length > 60;
}

function renderMath(escaped) {
  if (!escaped || escaped.indexOf("$") < 0) return escaped;
  return escaped.replace(/\$\$([\s\S]+?)\$\$|(?<!\\)\$([^$\n]+?)(?<!\\)\$/g,
    (whole, display, inline) => {
      const src = display ?? inline;
      if (looksLikeProse(src)) return whole;
      try {
        const html = texToHtml(src);
        return display
          ? `<span class="math-display">${html}</span>`
          : `<span class="math">${html}</span>`;
      } catch {
        return src;                 // never worse than the raw source it replaced
      }
    });
}

/* ── Context viewer and throughput ────────────────────────────────────────────
 *
 * What went into the prompt, and how much room was left. Both numbers come from
 * the generator's own tokenizer and its usage report, not from a local estimate:
 * a viewer that presents guesses as facts is worse than one that says it cannot
 * tell, so an inexact count is labelled rather than quietly shown.
 *
 * The bar is proportional to the model's real context window, so "excerpts filled
 * most of it" is visible at a glance rather than something you work out from
 * numbers. Free space is drawn too — the point of the panel is as much what was
 * left unused as what was spent.
 */
const CTX_COLORS = { system: "#8a8", "few-shot": "#a99", history: "#89b",
                     excerpts: "#c96", question: "#b8a" };

function renderContext(msgEl, ctx) {
  if (!msgEl || !ctx) return;
  const host = msgEl.querySelector(".meta") || msgEl;
  let box = msgEl.querySelector(".ctx");
  if (!box) {
    box = document.createElement("details");
    box.className = "ctx";
    host.append(box);
  }
  const limit = ctx.limit || 32768;
  const used = ctx.prompt_tokens ??
    ctx.parts.reduce((a, p) => a + (p.tokens || 0), 0);
  const reserved = ctx.reserved || 0;
  const free = Math.max(0, limit - used - reserved);
  const pct = (n) => `${(100 * n / limit).toFixed(1)}%`;

  const segs = ctx.parts.filter((p) => p.tokens > 0).map((p) =>
    `<span class="cseg" style="width:${pct(p.tokens)};background:${CTX_COLORS[p.name] || "#999"}"
           title="${escapeHtml(p.name)}: ${p.tokens} tokens"></span>`).join("");

  const rows = ctx.parts.map((p) =>
    `<tr><td><span class="cdot" style="background:${CTX_COLORS[p.name] || "#999"}"></span>` +
    `${escapeHtml(p.name)}</td><td class="num">${p.tokens.toLocaleString()}</td>` +
    `<td class="num dim">${pct(p.tokens)}</td>` +
    `<td class="num dim">${p.chars.toLocaleString()} ch</td></tr>`).join("");

  const histTokens = (ctx.parts.find((p) => p.name === "history") || {}).tokens || 0;
  box.innerHTML =
    `<summary>Context — <b>${used.toLocaleString()}</b> of ${limit.toLocaleString()} tokens` +
    ` <span class="dim">(${pct(used)} used · ${free.toLocaleString()} free` +
    `${reserved ? ` · ${reserved.toLocaleString()} reserved for the answer` : ""})</span>` +
    `${ctx.exact ? "" : ' <span class="warn-inline">estimated</span>'}</summary>` +
    `<div class="cbar">${segs}` +
    `<span class="cseg cres" style="width:${pct(reserved)}" title="reserved for the answer"></span>` +
    `<span class="cseg cfree" style="width:${pct(free)}" title="unused"></span></div>` +
    `<table class="ctable"><tbody>${rows}` +
    `<tr class="ctot"><td>reserved for answer</td><td class="num">${reserved.toLocaleString()}</td>` +
    `<td class="num dim">${pct(reserved)}</td><td></td></tr>` +
    `<tr class="ctot"><td>free</td><td class="num">${free.toLocaleString()}</td>` +
    `<td class="num dim">${pct(free)}</td><td></td></tr></tbody></table>` +
    (ctx.exact ? "" :
      `<p class="ctx-note">The generator did not answer /tokenize, so these are ` +
      `character-based estimates rather than real token counts.</p>`) +
    // Offered where the cost is actually visible. A button in the toolbar would be a
    // feature nobody connects to the number that motivates it.
    (histTokens > 0
      ? `<button type="button" class="ctx-compress">Compress conversation ` +
        `<span class="dim">(history is ${histTokens.toLocaleString()} tokens)</span></button>`
      : "");
}

function renderPerf(msgEl, p) {
  if (!msgEl || !p) return;
  const host = msgEl.querySelector(".meta") || msgEl;
  let el = msgEl.querySelector(".perf");
  if (!el) {
    el = document.createElement("div");
    el.className = "perf";
    host.append(el);
  }
  const bits = [];
  if (p.ttft_ms != null) bits.push(`${p.ttft_ms} ms to first token`);
  if (p.tok_per_sec != null) bits.push(`<b>${p.tok_per_sec}</b> tok/s`);
  if (p.completion_tokens != null) bits.push(`${p.completion_tokens} out`);
  if (p.prompt_tokens != null) bits.push(`${p.prompt_tokens.toLocaleString()} in`);
  el.innerHTML = bits.join(" · ");
}

/* ── Library as a directed graph ──────────────────────────────────────────────
 *
 * A list answers "what did I ask?". The graph answers "why?" — which question led
 * to which, and where an investigation from last week already covers the one about
 * to start. Nodes are conversations; edges always point forward in time, which
 * makes the whole thing a DAG and lets it be read in one direction.
 *
 * Laid out like a git log: one node per row, oldest at the top, with edges drawn as
 * curves in a narrow left gutter. A force-directed blob needs width the sidebar
 * does not have, and this shape is already familiar from every version-control UI.
 */
const REL_COLOR = {
  "follows-up": "#6a9bd1", applies: "#7fb069", compares: "#d9a441",
  contradicts: "#d16a6a", "background-for": "#9b8ac4", "same-topic": "#8a8a8a",
  diverges: "#c47ab0",
};
let libGraph = null, libMode = localStorage.getItem("libMode") || "list";

async function loadLibGraph(refresh = false) {
  const host = $("#lib-graph");
  if (!host) return;
  host.innerHTML = `<p class="lib-empty">${refresh ? "Rebuilding" : "Reading"} the graph…</p>`;
  try {
    const r = await fetch(`/api/library/graph${refresh ? "?refresh=1" : ""}`);
    libGraph = await r.json();
  } catch (e) {
    host.innerHTML = `<p class="lib-empty">Could not build the graph.</p>`;
    return;
  }
  renderLibGraph();
}

function renderLibGraph() {
  const host = $("#lib-graph");
  if (!host || !libGraph) return;
  const nodes = [...(libGraph.nodes || [])];
  if (!nodes.length) {
    host.innerHTML = `<p class="lib-empty">Ask a few questions and they will appear here as a graph.</p>`;
    return;
  }
  // Depth first, then time: the row order then reads as the order the thinking went.
  nodes.sort((a, b) => (a.depth - b.depth) || String(a.first_utc).localeCompare(b.first_utc));
  const row = new Map(nodes.map((n, i) => [n.id, i]));
  const ROW = 46, GUT = 26, PAD = 8;
  const height = nodes.length * ROW + PAD * 2;

  const edges = (libGraph.edges || []).filter((e) => row.has(e.source) && row.has(e.target));
  // Lanes keep concurrent edges from overdrawing each other in a 26px gutter.
  const lanes = [];
  for (const e of edges) {
    const a = Math.min(row.get(e.source), row.get(e.target));
    const b = Math.max(row.get(e.source), row.get(e.target));
    let lane = lanes.findIndex((end) => end <= a);
    if (lane === -1) { lane = lanes.length; lanes.push(b); } else { lanes[lane] = b; }
    e._lane = Math.min(lane, 3);
  }

  const paths = edges.map((e) => {
    const y1 = PAD + row.get(e.source) * ROW + ROW / 2;
    const y2 = PAD + row.get(e.target) * ROW + ROW / 2;
    const x = 7 + e._lane * 5;
    const bow = x + 9;
    const c = REL_COLOR[e.relation] || "#888";
    return `<path d="M ${x} ${y1} C ${bow} ${y1 + 14}, ${bow} ${y2 - 14}, ${x} ${y2}"
              fill="none" stroke="${c}" stroke-width="1.4" marker-end="url(#arw)"
              opacity=".85"><title>${escapeHtml(e.relation)}: ${escapeHtml(e.why || "")}</title></path>`;
  }).join("");

  const dots = nodes.map((n, i) =>
    `<circle cx="7" cy="${PAD + i * ROW + ROW / 2}" r="3.4" fill="var(--fg)" opacity=".65"/>`
  ).join("");

  const rows = nodes.map((n) => {
    const topics = (n.topics || []).slice(0, 3)
      .map((t) => `<span class="ltag">${escapeHtml(t)}</span>`).join("");
    return `<div class="lgnode" data-thread="${escapeHtml(n.id)}" data-entry="${escapeHtml(n.entry_ids?.[0] || "")}"
                 title="${escapeHtml(n.summary || "")}">
        <div class="lgtop"><span class="lglabel">${escapeHtml(n.label)}</span>
          <span class="lgn">${n.n_questions}</span></div>
        <div class="lgmeta">${escapeHtml((n.first_utc || "").slice(0, 10))}${topics}</div>
      </div>`;
  }).join("");

  host.innerHTML =
    `<div class="lgwrap" style="--gut:${GUT}px">
       <svg class="lgsvg" width="${GUT}" height="${height}" aria-hidden="true">
         <defs><marker id="arw" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5"
                       markerHeight="5" orient="auto-start-reverse">
                 <path d="M0 0 L8 4 L0 8 z" fill="context-stroke"/></marker></defs>
         ${paths}${dots}
       </svg>
       <div class="lgrows" style="--row:${ROW}px;--pad:${PAD}px">${rows}</div>
     </div>
     <div class="lglegend">${Object.entries(REL_COLOR)
        .filter(([k]) => edges.some((e) => e.relation === k))
        .map(([k, c]) => `<span><i style="background:${c}"></i>${k}</span>`).join("")}</div>
     <p class="lgfoot">${nodes.length} conversations · ${edges.length} connections${
        libGraph.cached ? "" : " · just rebuilt"}</p>`;
}

$("#lib-graph")?.addEventListener("click", (ev) => {
  const el = ev.target.closest(".lgnode");
  if (!el) return;
  const e = libFindEntry(el.dataset.entry);
  if (e) restoreEntry(e);
});

function setLibMode(mode) {
  libMode = mode;
  localStorage.setItem("libMode", mode);
  const graph = mode === "graph";
  $("#lib-graph").hidden = !graph;
  $("#lib-tree").hidden = graph;
  $("#lib-rebuild").hidden = !graph;
  $("#lib-view").textContent = graph ? "list" : "graph";
  if (graph && !libGraph) loadLibGraph(false);
}
$("#lib-view")?.addEventListener("click", () => setLibMode(libMode === "graph" ? "list" : "graph"));
$("#lib-rebuild")?.addEventListener("click", () => loadLibGraph(true));
queueMicrotask(() => setLibMode(libMode));

/* Compress the conversation. The most recent turns stay verbatim — they are what
 * follow-ups point at — so this reclaims the older ones without breaking reference
 * resolution. */
document.addEventListener("click", async (ev) => {
  const btn = ev.target.closest(".ctx-compress");
  if (!btn) return;
  btn.disabled = true;
  const was = btn.innerHTML;
  btn.textContent = "compressing…";
  try {
    const r = await fetch("/api/thread/compress", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paper: state.paper || null, model: $("#model").value || null }),
    });
    const d = await r.json();
    if (!d.ok) { btn.textContent = d.reason || "nothing to compress"; return; }
    const delta = d.chars_after - d.chars_before;
    // What compression buys is coverage, not always fewer characters: the whole thread
    // becomes representable instead of a sliding window of the most recent turns.
    // Claiming a saving on a short thread would be a number the panel could disprove.
    const cost = d.saves_chars
      ? `${Math.abs(delta).toLocaleString()} characters smaller`
      : `${delta.toLocaleString()} characters larger, but covering the whole thread`;
    btn.outerHTML =
      `<p class="ctx-note">Compressed ${d.compressed_turns} earlier ` +
      `exchange${d.compressed_turns === 1 ? "" : "s"} into a summary, keeping the last ` +
      `${d.kept_verbatim} verbatim. The model now sees all ${d.turns_after} turns ` +
      `instead of the most recent ${d.turns_before} — ${cost}. ` +
      `<a href="#" class="ctx-uncompress">undo</a></p>` +
      `<details class="ctx-summary"><summary>summary the model will now see</summary>` +
      `<div>${renderMath(escapeHtml(d.summary))}</div></details>`;
  } catch (e) {
    btn.disabled = false;
    btn.innerHTML = was;
  }
});

document.addEventListener("click", async (ev) => {
  const a = ev.target.closest(".ctx-uncompress");
  if (!a) return;
  ev.preventDefault();
  await fetch("/api/thread/uncompress", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paper: state.paper || null }),
  }).catch(() => {});
  a.closest(".ctx-note").textContent = "Full history restored for the next question.";
});
