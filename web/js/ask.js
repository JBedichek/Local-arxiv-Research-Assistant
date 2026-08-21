/* Asking a question and receiving an answer: the selection that seeds it, speculative
 * retrieval, the SSE stream, and everything rendered from it -- steps, citations,
 * coverage, grounding, context accounting and throughput.
 *
 * Selection lives here because its only purpose is to feed the question box: selecting a
 * passage starts a retrieval before anything has been typed, which is where the latency
 * saving in the header comment comes from. */

import { api, send } from "./api.js";
import { $, escapeHtml, setStatus } from "./dom.js";
import { applyHeatmap } from "./heatmap.js";
import { loadLibrary } from "./library.js";
import { loadGraph, openPaper, scrollToAnchor, stickToBottom } from "./paper.js";
import { prefs } from "./prefs.js";
import { state } from "./state.js";
import { loadTaste } from "./taste.js";
import { renderMath } from "./tex.js";

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
    await send("POST", "/api/taste/mark", { arxiv_id: state.paper, text: state.candidate });
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
  return send("POST", "/api/retrieve", {
      query, selection, paper: state.paper,
      scope: $("#scope").value, k: 8,
    });
}

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

export function addMessage(role, text, selection) {
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

/* depth dial */
let BREADTH = [];

/* The spectrum comes from the server, so the slider is not a hardcoded copy of it. Loaded
 * here rather than in boot because BREADTH is this module's state: an ES module cannot
 * assign to a binding it imported, and threading a setter through boot to say so would be
 * ceremony around a fact the module already knows. */
export async function loadBreadth() {
  try {
    const b = await api("/api/breadth");
    BREADTH = b.options;
    $("#breadth").max = String(BREADTH.length - 1);
    $("#breadth").value = prefs.get(
      "breadth", String(BREADTH.findIndex((x) => x.name === b.default)));
    applyBreadth();
  } catch { /* fall back to the server default */ }
}

export function applyBreadth() {
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
