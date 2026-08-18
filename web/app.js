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
  selection: null, models: [], busy: false, breadth: 'balanced', abort: null,
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

async function openPaper(id, fragment) {
  setStatus("loading paper…");
  try {
    const data = await api(`/api/paper/${encodeURIComponent(id)}`);
    state.paper = data.arxiv_id;
    state.version = data.version;
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
      $("#paper").innerHTML =
        `<p class="placeholder">Full text not fetched yet (status: ${data.fulltext_status}).</p>
         <h3>Abstract</h3><p>${escapeHtml(data.abstract || "")}</p>`;
    }
    history.replaceState(null, "", `/p/${data.arxiv_id}v${data.version}`);
    setStatus("");
    loadGraph();
    if (fragment) scrollToAnchor(fragment);
  } catch (err) {
    setStatus(String(err.message || err), "error");
  }
}

/* Resolve `#S3.p4:120-480` — scroll to the element and paint the char range. */
function scrollToAnchor(fragment) {
  const raw = fragment.replace(/^#/, "");
  const [anchor, range] = raw.split(":");
  const el = document.getElementById(anchor);
  if (!el) return false;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  if (range) {
    const [s, e] = range.split("-").map(Number);
    if (Number.isFinite(s) && Number.isFinite(e)) paintRange(el, s, e);
  } else {
    paintRange(el, 0, el.textContent.length);
  }
  return true;
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
  state.selection = text;
  const rect = sel.getRangeAt(0).getBoundingClientRect();
  float.style.top = `${window.scrollY + rect.bottom + 8}px`;
  float.style.left = `${window.scrollX + rect.left}px`;
  float.hidden = false;

  showSelectionChip(text);
  // Fire retrieval now, while the user is still reading and typing.
  state.pendingHits = retrieve(text.slice(0, 600), text).catch(() => null);
}

function showSelectionChip(text) {
  const chip = $("#selection-chip");
  chip.hidden = false;
  chip.firstElementChild.textContent =
    text.length > 120 ? text.slice(0, 120) + "…" : text;
}

$("#clear-sel").addEventListener("click", () => {
  state.selection = null;
  state.pendingHits = null;
  $("#selection-chip").hidden = true;
});

$("#ask-float").addEventListener("click", () => {
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
    loadGraph(q);
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
  div.scrollIntoView({ block: "end" });
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
  const hit = state.hits[Number(a.dataset.i)];
  if (!hit) return;
  if (hit.arxiv_id !== state.paper) {
    openPaper(hit.arxiv_id, `#${hit.anchor}:${hit.char_start}-${hit.char_end}`);
  } else {
    scrollToAnchor(`#${hit.anchor}:${hit.char_start}-${hit.char_end}`);
  }
});

/* ---------- graph (R8/R9) ---------- */

let graphData = null;
async function loadGraph(query = "") {
  if (!state.paper) return;
  try {
    graphData = await api(
      `/api/graph/${encodeURIComponent(state.paper)}?query=${encodeURIComponent(query)}`
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

async function searchPapers(query) {
  setStatus("searching…");
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
      body: JSON.stringify({ query, limit: 25 }),
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
    $("#results-graph").hidden = searchView !== "graph";
    $("#results-list").hidden = searchView !== "list";
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

(async function boot() {
  applyLayout();
  applyTypography();
  makeSplitter($("#split-left"), "--col-left", "colLeft", "left");
  makeSplitter($("#split-right"), "--col-right", "colRight", "right");
  bindSearchGraph();

  try {
    const b = await api("/api/breadth");
    BREADTH = b.options;
    $("#breadth").max = String(BREADTH.length - 1);
    const saved = prefs.get("breadth", String(BREADTH.findIndex((x) => x.name === b.default)));
    $("#breadth").value = saved;
    applyBreadth();
  } catch { /* fall back to the server default */ }

  try {
    const m = await api("/api/models");
    state.models = m.models;
    $("#model").innerHTML =
      m.models.map((x) => `<option value="${x.repo}">${x.repo} (${x.size_gb}GB)</option>`).join("")
      || `<option value="">none servable</option>`;
    syncQuant();
    $("#model").addEventListener("change", syncQuant);
  } catch { /* models are optional for browsing */ }

  const path = location.pathname.match(/^\/p\/(.+?)(?:v(\d+))?$/);
  if (path) openPaper(path[1], location.hash);

  const h = await api("/api/health").catch(() => null);
  if (h) setStatus(`${h.vectors.toLocaleString()} chunks indexed`);
})();

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

function applyLayout() {
  document.documentElement.style.setProperty("--col-left", prefs.get("colLeft", "300px"));
  document.documentElement.style.setProperty("--col-right", prefs.get("colRight", "400px"));
}

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
    const delta = edge === "left" ? ev.clientX - startX : startX - ev.clientX;
    // Clamp so a pane can never be dragged to nothing or swallow the paper.
    const px = Math.max(180, Math.min(startPx + delta, window.innerWidth * 0.45));
    root.style.setProperty(varName, px + "px");
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
  el.addEventListener("dblclick", () => {
    const d = edge === "left" ? "300px" : "400px";
    root.style.setProperty(varName, d);
    prefs.set(prefKey, d);
    drawGraph();
  });
}

const FONTS = { serif: "var(--font-serif)", sans: "var(--font-sans)",
                mono: "var(--font-mono)", dyslexic: "var(--font-wide)" };

function applyTypography() {
  const fam = prefs.get("font", "sans");
  const size = prefs.get("fontsize", "16");
  document.documentElement.style.setProperty("--paper-font", FONTS[fam] || FONTS.sans);
  document.documentElement.style.setProperty("--paper-size", size + "px");
  // Hold the measure near 70 characters as size changes; long lines at large type are
  // what actually makes reading tiring, not the type size itself.
  document.documentElement.style.setProperty("--measure", Math.round(size * 54) + "px");
  $("#font").value = fam;
  $("#fontsize").value = size;
  $("#fontsize-val").textContent = size;
}

$("#font").addEventListener("change", (e) => { prefs.set("font", e.target.value); applyTypography(); });
$("#fontsize").addEventListener("input", (e) => { prefs.set("fontsize", e.target.value); applyTypography(); });

/* depth dial */
let BREADTH = [];
function applyBreadth() {
  const i = Number($("#breadth").value);
  const b = BREADTH[i];
  if (!b) return;
  state.breadth = b.name;
  prefs.set("breadth", String(i));
  $("#breadth-val").innerHTML =
    `${escapeHtml(b.label)} <em>${escapeHtml(b.estimate)}</em>`;
  const bits = [`${b.max_rounds} search round${b.max_rounds > 1 ? "s" : ""}`, `top ${b.k}`];
  if (b.expand_context) bits.push("context expansion");
  if (b.allow_clarify) bits.push("may ask to clarify");
  $("#breadth").title = bits.join(" · ") + ` — estimated ${b.estimate}`;
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
  row.scrollIntoView({ block: "nearest" });
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

/* Laid out chronologically, not force-directed. A citation graph is a DAG ordered by
 * time: a paper can only cite work that already existed. Putting the date on the x axis
 * makes every edge point leftward, so lineage is readable at a glance — which paper
 * started a line of work, which are extensions, which are contemporaries that never cite
 * each other. A force layout throws that information away and produces a hairball. */

let searchData = null;
let searchView = "graph";

function layoutSearchGraph(canvas) {
  const nodes = searchData.results;
  if (!nodes.length) return [];
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;

  const times = nodes.map((n) => Date.parse(n.submitted || "") || 0).filter(Boolean);
  const tMin = Math.min(...times), tMax = Math.max(...times);
  const padL = 46, padR = 150, padT = 22, padB = 30;
  const span = Math.max(tMax - tMin, 1);

  // Lay out by date, then push apart vertically within date-neighbourhoods so labels
  // do not collide. Rank breaks ties so the most relevant paper sits highest.
  const placed = nodes.map((n) => {
    const t = Date.parse(n.submitted || "") || tMin;
    return { ...n, x: padL + ((t - tMin) / span) * (w - padL - padR), y: 0 };
  });
  placed.sort((a, b) => a.x - b.x);
  const lanes = [];
  const rowH = Math.max(22, Math.min(40, (h - padT - padB) / Math.max(placed.length, 1) * 1.6));
  for (const n of placed) {
    let lane = 0;
    while (lanes[lane] !== undefined && n.x - lanes[lane] < 130) lane++;
    lanes[lane] = n.x;
    n.y = padT + lane * rowH + rowH / 2;
    if (n.y > h - padB) n.y = padT + ((lane % Math.max(1, Math.floor((h - padT - padB) / rowH))) * rowH) + rowH / 2;
  }
  return placed;
}

function drawSearchGraph() {
  const canvas = $("#results-graph");
  if (!searchData || searchView !== "graph") return;
  const placed = layoutSearchGraph(canvas);
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  const byId = Object.fromEntries(placed.map((n) => [n.arxiv_id, n]));

  // year gridlines give the time axis meaning
  const years = [...new Set(placed.map((n) => (n.submitted || "").slice(0, 4)))].filter(Boolean).sort();
  ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
  for (const y of years) {
    const same = placed.filter((n) => (n.submitted || "").startsWith(y));
    const x = same.reduce((a, n) => a + n.x, 0) / same.length;
    ctx.strokeStyle = "rgba(140,150,170,.13)";
    ctx.beginPath(); ctx.moveTo(x, 12); ctx.lineTo(x, h - 18); ctx.stroke();
    ctx.fillStyle = "rgba(140,150,170,.7)";
    ctx.fillText(y, x - 12, h - 6);
  }

  // edges: cited paper <- citing paper, so arrows point back in time
  for (const e of searchData.edges || []) {
    const a = byId[e.src], b = byId[e.dst];
    if (!a || !b) continue;
    ctx.strokeStyle = "rgba(140,160,200,.42)";
    ctx.lineWidth = 1.1;
    ctx.beginPath();
    const mx = (a.x + b.x) / 2;
    ctx.moveTo(a.x, a.y);
    ctx.bezierCurveTo(mx, a.y, mx, b.y, b.x, b.y);
    ctx.stroke();
    // arrowhead at the cited end
    const ang = Math.atan2(b.y - a.y, b.x - mx);
    ctx.fillStyle = "rgba(140,160,200,.62)";
    ctx.beginPath();
    ctx.moveTo(b.x, b.y);
    ctx.lineTo(b.x - 7 * Math.cos(ang - 0.4), b.y - 7 * Math.sin(ang - 0.4));
    ctx.lineTo(b.x - 7 * Math.cos(ang + 0.4), b.y - 7 * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();
  }

  const scores = placed.map((n) => n.score);
  const lo = Math.min(...scores), hi = Math.max(...scores);
  for (const n of placed) {
    const t = hi > lo ? (n.score - lo) / (hi - lo) : 1;
    // size by how much the *result set* cites it: the local hub is the foundational work
    const r = 5 + Math.min(n.in_degree, 8) * 1.6;
    ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
    ctx.fillStyle = heatColor(t); ctx.fill();
    if (n.in_degree > 0) { ctx.strokeStyle = "rgba(255,255,255,.5)"; ctx.lineWidth = 1; ctx.stroke(); }
    ctx.fillStyle = "rgba(190,200,215,.92)";
    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    const label = (n.title || n.arxiv_id).slice(0, 30) + ((n.title || "").length > 30 ? "…" : "");
    ctx.fillText(`${n.rank}. ${label}`, n.x + r + 5, n.y + 3.5);
  }
  canvas._nodes = placed;
}

function bindSearchGraph() {
  const canvas = $("#results-graph"), tip = $("#results-tip");
  const at = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
    return (canvas._nodes || []).find((n) => (n.x - x) ** 2 + (n.y - y) ** 2 < 160);
  };
  canvas.addEventListener("mousemove", (ev) => {
    const n = at(ev);
    if (!n) { tip.hidden = true; return; }
    const rect = canvas.getBoundingClientRect();
    tip.hidden = false;
    tip.style.left = `${ev.clientX - rect.left + 14}px`;
    tip.style.top = `${ev.clientY - rect.top + 14 + canvas.offsetTop}px`;
    tip.innerHTML =
      `<b>${escapeHtml(n.title)}</b>
       <span class="dim">${n.arxiv_id} · ${escapeHtml(n.submitted)} ·
       score ${n.score.toFixed(3)} · cited by ${n.in_degree} of these results</span>
       <p style="margin:5px 0 0">${escapeHtml((n.abstract || "").slice(0, 180))}…</p>`;
  });
  canvas.addEventListener("mouseleave", () => { tip.hidden = true; });
  canvas.addEventListener("click", (ev) => {
    const n = at(ev);
    if (n) { hideResults(); openPaper(n.arxiv_id); }
  });
  $("#results-toggle").addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    searchView = b.dataset.view;
    $("#results-toggle").querySelectorAll("button")
      .forEach((x) => x.classList.toggle("on", x.dataset.view === searchView));
    $("#results-graph").hidden = searchView !== "graph";
    $("#results-list").hidden = searchView !== "list";
    if (searchView === "graph") drawSearchGraph();
  });
  window.addEventListener("resize", () => drawSearchGraph());
}
