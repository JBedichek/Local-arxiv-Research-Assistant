/* Marked passages, and what in the corpus resembles them. */

import { api } from "./api.js";
import { $, escapeHtml } from "./dom.js";
import { openPaper, scrollToAnchor } from "./paper.js";
import { state } from "./state.js";

/* The reader's marked passages, as a set of positions in embedding space. Deliberately a
 * SET and not a centroid: summing cosine similarities is arithmetically identical to
 * searching once with the summed vector, so "average my interests" and "add up their
 * similarities" are the same operation, and both match the midpoint between optimisers and
 * retrieval rather than either. The reductions that keep interests distinct — max, and the
 * LogSumExp soft maximum the server defaults to — are the reason the marks are kept apart.
 * See TASTE_REDUCTIONS in lara/serve/app.py. */

export let tasteMarks = 0;
let tasteScope = "paper";       // paper | corpus

export async function loadTaste() {
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

export async function renderTaste() {
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
