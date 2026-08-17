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
  selection: null, models: [], busy: false,
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
  $("#ask-btn").disabled = true;
  $("#question").value = "";

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
      }),
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
        if (ev[1] === "token") {
          if (first) {
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
    state.busy = false;
    $("#ask-btn").disabled = false;
    state.pendingHits = null;
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
    $("#results-head").innerHTML =
      `<h2>${data.results.length} papers for “${escapeHtml(query)}”</h2>
       <p class="meta">${ms}ms · server ${Math.round(data.timings_ms.total)}ms
       · abstracts ${Math.round(data.timings_ms.abstracts)}ms
       · full text ${Math.round(data.timings_ms.fulltext)}ms</p>`;
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
