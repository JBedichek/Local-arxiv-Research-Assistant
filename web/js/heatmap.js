/* Shading a paper's passages by relevance to the question or to the answer. */

import { api } from "./api.js";
import { $ } from "./dom.js";
import { rangeFromOffsets, scrollToAnchor } from "./paper.js";
import { prefs } from "./prefs.js";
import { state } from "./state.js";
import { tasteMarks } from "./taste.js";

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

export async function applyHeatmap() {
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

export function applyHeatPrefs() {
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
