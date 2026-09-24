/* The standalone tab a lesson's "Background" link opens: one grounded note on one topic the
 * lesson leaned on without explaining, read-only, polling while it is still being written. No
 * app state, no course navigation -- closing the tab is how you come back to the lesson. */

import { escapeHtml } from "./dom.js";
import { renderMath } from "./tex.js";
import { chartSvg } from "./chart.js";

const md = (s) => renderMath(escapeHtml(s ?? ""));

const params = new URLSearchParams(location.search);
const course = params.get("course") || "";
const concept = params.get("concept") || "";
const topic = params.get("topic") || "";
const host = document.getElementById("topic-main");

function url() {
  return `/api/learn/courses/${encodeURIComponent(course)}/concepts/${encodeURIComponent(concept)}/topics/${encodeURIComponent(topic)}`;
}

function sourcesList(claims) {
  if (!claims?.length) return "";
  const rows = claims.map((c) => `<li>${md(c.text)} <span class="hint">— <a href="https://arxiv.org/abs/${escapeHtml(c.passage.arxiv_id)}" target="_blank" rel="noopener">${escapeHtml(c.passage.arxiv_id)}</a></span></li>`).join("");
  return `<details><summary>Sources</summary><ul class="claims">${rows}</ul></details>`;
}

function render(data) {
  document.title = data.title || "Background";
  if (data.status !== "done") {
    host.innerHTML = `<h1>${md(data.title)}</h1>
      <p class="hint">Preparing this background note… <span class="hint">this can take up to half a minute</span></p>`;
    return;
  }
  const doc = data.doc;
  if (!doc || doc.insufficient) {
    host.innerHTML = `<h1>${md(data.title)}</h1><p class="warn">${md(doc?.message || "Nothing here could be verified against the corpus.")}</p>`;
    return;
  }
  const body = doc.sections.map((sec) => `<p class="lesson-p">${sec.sentences.map((s) =>
    `<span class="lsent">${md(s.text)}${s.claims.map((k) => `<sup class="ck">${escapeHtml(k)}</sup>`).join("")}</span>`).join(" ")}</p>`).join("");
  host.innerHTML = `<h1>${md(data.title)}</h1>
    <p class="hint">Background for “${md(data.concept_title)}”${data.note ? ` — ${md(data.note)}` : ""}</p>
    ${body}
    ${doc.chart ? chartSvg(doc.chart) : ""}
    ${sourcesList(doc.claims)}`;
}

async function tick() {
  let res, data;
  try {
    res = await fetch(url());
    data = await res.json();
  } catch {
    host.innerHTML = `<p class="error">Could not reach the server.</p>`;
    return;
  }
  if (!res.ok) {
    host.innerHTML = `<p class="error">${escapeHtml(data?.error || `${res.status}`)}</p>`;
    return;
  }
  render(data);
  if (data.status !== "done") setTimeout(tick, 2500);
}

if (!course || !concept || !topic) {
  host.innerHTML = `<p class="error">This link is missing its course, concept or topic.</p>`;
} else {
  tick();
}
