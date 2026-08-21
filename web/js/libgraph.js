/* The library as a directed graph of conversations. */

import { $, escapeHtml } from "./dom.js";
import { libFindEntry, restoreEntry } from "./library.js";
import { state } from "./state.js";
import { renderMath } from "./tex.js";

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
    `<circle cx="7" cy="${PAD + i * ROW + ROW / 2}" r="3.4" fill="var(--text)" opacity=".65"/>`
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
