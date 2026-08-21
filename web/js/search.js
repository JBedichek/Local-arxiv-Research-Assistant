/* Searching the corpus for papers, and the two ways of reading the result: a ranked list
 * and a citation graph over the results.
 *
 * One module because the two views share `searchData` and each can switch to the other. */

import { api } from "./api.js";
import { $, escapeHtml, setStatus } from "./dom.js";
import { heatColor, openPaper } from "./paper.js";
import { prefs } from "./prefs.js";
import { state } from "./state.js";

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

export async function searchPapers(query, push = true) {
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

export function hideResults() {
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

export function syncQuant() {
  const m = state.models.find((x) => x.repo === $("#model").value);
  $("#quant").innerHTML = (m ? m.quant_options : []).map((q) => `<option>${q}</option>`).join("");
}

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

export function drawSearchGraph() {
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

export function bindSearchGraph() {
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
