/* The open paper: fetching it, rendering it, scrolling it, highlighting inside it, and
 * the ego network of citations around it.
 *
 * The graph belongs here rather than beside it because it is a view OF the open paper --
 * clicking a node opens that paper, which reloads the graph. */

import { api } from "./api.js";
import { $, escapeHtml, setStatus } from "./dom.js";
import { applyHeatmap } from "./heatmap.js";
import { loadLibrary } from "./library.js";
import { hideResults, searchPapers } from "./search.js";
import { state } from "./state.js";
import { renderTaste } from "./taste.js";

export async function openPaper(id, fragment, push = true) {
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
export function scrollToAnchor(fragment) {
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
export function stickToBottom() {
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
export function rangeFromOffsets(el, start, end) {
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

export async function loadGraph(query = null) {
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

export function drawGraph() {
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
export function heatColor(t) {
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
