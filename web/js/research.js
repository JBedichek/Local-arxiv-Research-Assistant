/* Deep automated research: a run that takes minutes and streams its progress. */

import { escapeHtml } from "./dom.js";
import { renderMath } from "./tex.js";

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

  /* Render a stored run into the pane exactly as a live one leaves it, so a report
   * reopened a month later reads the same as it did the moment it finished. */
  async function openRun(runId) {
    const r = await fetch(`/api/synthesis/run/${runId}`);
    if (!r.ok) return false;
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
    return true;
  }

  history?.addEventListener("change", () => {
    if (history.value) openRun(history.value);
  });

  /* The library pane opens reports through an event rather than an import: this module is
   * an IIFE closed over its own DOM, and a custom event crosses that boundary without
   * unpicking it. A library that could not reach the renderer would have to grow a second
   * copy of it, and the two would drift. */
  document.addEventListener("lara:open-run", async (e) => {
    const runId = e.detail && e.detail.runId;
    if (!runId) return;
    show(true);
    loadHistory();
    if (await openRun(runId) && history) history.value = runId;
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
