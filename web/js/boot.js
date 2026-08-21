/* Reader UI -- entry point.
 *
 * Loaded by index.html as the only <script>; every other module is reached from here.
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
 *    live in; the fallback does that only when it must. */

import { api } from "./api.js";
import { loadBreadth } from "./ask.js";

/* Imported for their side effects: both are self-contained panels that register their own
 * listeners and export nothing, so nothing else has a reason to name them. In the single
 * file they ran because they were IN the file; as modules they run only if something
 * imports them, and "nothing imports it" is silent -- the panel simply stops responding.
 * app.js's entry point is the honest place to say the app includes them. */
import "./libgraph.js";
import "./research.js";
import { $, setStatus } from "./dom.js";
import { applyHeatPrefs } from "./heatmap.js";
import { applyLayout, makeSplitter } from "./layout.js";
import { loadLibrary } from "./library.js";
import { loadModels } from "./models.js";
import { openPaper } from "./paper.js";
import { prefs } from "./prefs.js";
import { bindSearchGraph, searchPapers } from "./search.js";
import { loadPrompt } from "./sysprompt.js";
import { loadTaste } from "./taste.js";
import { applyTypography } from "./typography.js";

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

/* Deferred with queueMicrotask, and kept that way.
 *
 * This began as a workaround for load order: twice, an appended block landed after boot
 * and boot hit the temporal dead zone on a `const` below it -- 'Cannot access
 * prefs/HEAT_NOTES before initialization' -- which aborts boot entirely and silently
 * disables every listener it registers. Modules fix that properly: every import above is
 * fully evaluated before this file's body runs, so there is no "below" any more.
 *
 * The deferral still earns its place. It lets the whole graph finish evaluating -- and
 * every top-level listener register -- before boot starts awaiting the network, so a slow
 * /api/breadth cannot leave the page half-wired. */
async function boot() {
  applyLayout();
  applyTypography();
  applyHeatPrefs();
  document.documentElement.style.setProperty("--graph-h", prefs.get("graphH", "460px"));
  $("#topk").value = prefs.get("topk", "20");
  $("#topk-val").textContent = $("#topk").value;
  makeSplitter($("#split-lib"), "--col-lib", "colLib", "lib");
  makeSplitter($("#split-left"), "--col-left", "colLeft", "left");
  makeSplitter($("#split-right"), "--col-right", "colRight", "right");
  bindSearchGraph();
  // Neither blocks the reader: the library and the prompt editor are both side panels,
  // and awaiting them here would delay the paper for two requests that nothing needs yet.
  loadLibrary();
  loadPrompt();
  loadTaste();

  await loadBreadth();

  await loadModels();

  const path = location.pathname.match(/^\/p\/(.+?)(?:v(\d+))?$/);
  const q0 = new URLSearchParams(location.search).get("q");
  if (path) openPaper(path[1], location.hash, false);
  else if (q0) { $("#arxiv-input").value = q0; searchPapers(q0, false); }

  const h = await api("/api/health").catch(() => null);
  if (h) setStatus(`${h.vectors.toLocaleString()} chunks indexed`);
}
queueMicrotask(() => boot().catch((e) => {
  console.error('boot failed', e);
  setStatus('startup error: ' + (e?.message || e), 'error');
}));

/* Resolve before downloading. A checkpoint is tens of gigabytes, so the dialog reports
 * what the repo actually is — parameters, architecture, quantisation, size, and whether it
 * fits this machine's memory — before a byte is written. Committing to a download because
 * the name looked plausible is an expensive way to discover it was a 70B model. */
