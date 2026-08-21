/* The three-pane layout: collapsing panes and dragging the splitters between them. */

import { $ } from "./dom.js";
import { drawGraph } from "./paper.js";
import { prefs } from "./prefs.js";
import { drawSearchGraph } from "./search.js";

const PANES = { lib: "#library-pane", left: "#graph-pane", right: "#chat-pane" };

function setPaneCollapsed(edge, collapsed) {
  const pane = $(PANES[edge]);
  if (pane) pane.classList.toggle("collapsed", collapsed);
}

export function applyLayout() {
  const lib = prefs.get("colLib", "240px");
  const l = prefs.get("colLeft", "300px"), r = prefs.get("colRight", "400px");
  document.documentElement.style.setProperty("--col-lib", lib);
  document.documentElement.style.setProperty("--col-left", l);
  document.documentElement.style.setProperty("--col-right", r);
  setPaneCollapsed("lib", parseInt(lib, 10) === 0);
  setPaneCollapsed("left", parseInt(l, 10) === 0);
  setPaneCollapsed("right", parseInt(r, 10) === 0);
}

/* Keyboard: [ and ] toggle the sidebars, \ gives the paper the whole window. */
document.addEventListener("keydown", (ev) => {
  // ev.target is not always an Element (it is `document` for synthetic events and when
  // nothing has focus), and Document has no .matches — calling it threw and killed the
  // handler before any shortcut could run.
  const t = ev.target;
  if (t instanceof Element && t.matches("input, textarea, select")) return;
  const root = document.documentElement;
  const toggle = (v, dflt, edge) => {
    const cur = parseInt(getComputedStyle(root).getPropertyValue(v), 10) || 0;
    const next = cur === 0 ? dflt : "0px";
    root.style.setProperty(v, next);
    prefs.set({ "--col-lib": "colLib", "--col-left": "colLeft", "--col-right": "colRight" }[v], next);
    setPaneCollapsed(edge, next === "0px");
  };
  if (ev.key === ";") { toggle("--col-lib", "240px", "lib"); }
  else if (ev.key === "[") { toggle("--col-left", "300px", "left"); drawGraph(); }
  else if (ev.key === "]") { toggle("--col-right", "400px", "right"); }
  else if (ev.key === "\\" || ev.code === "Backslash") {
    const hidden = parseInt(getComputedStyle(root).getPropertyValue("--col-left"), 10) === 0;
    const lib = hidden ? "240px" : "0px";
    const l = hidden ? "300px" : "0px", r = hidden ? "400px" : "0px";
    root.style.setProperty("--col-lib", lib);
    root.style.setProperty("--col-left", l);
    root.style.setProperty("--col-right", r);
    prefs.set("colLib", lib); prefs.set("colLeft", l); prefs.set("colRight", r);
    setPaneCollapsed("lib", !hidden);
    setPaneCollapsed("left", !hidden); setPaneCollapsed("right", !hidden);
    drawGraph();
  } else return;
  setTimeout(drawSearchGraph, 0);
});

export function makeSplitter(el, varName, prefKey, edge) {
  let startX = 0, startPx = 0;
  const root = document.documentElement;
  const current = () =>
    parseInt(getComputedStyle(root).getPropertyValue(varName), 10) || 300;

  el.addEventListener("pointerdown", (ev) => {
    ev.preventDefault();
    startX = ev.clientX;
    startPx = current();
    el.setPointerCapture(ev.pointerId);
    el.classList.add("dragging");
    document.body.classList.add("resizing");
  });
  el.addEventListener("pointermove", (ev) => {
    if (!el.hasPointerCapture?.(ev.pointerId)) return;
    const delta = edge === "right" ? startX - ev.clientX : ev.clientX - startX;
    // Free range: 0 (collapsed, so the paper gets the whole window) up to 85%, so the
    // graph or chat can take over the screen instead. Below 60px it snaps shut rather
    // than leaving a useless sliver.
    let px = Math.max(0, Math.min(startPx + delta, window.innerWidth * 0.85));
    if (px < 60) px = 0;
    root.style.setProperty(varName, px + "px");
    setPaneCollapsed(edge, px === 0);
  });
  const end = (ev) => {
    if (!el.classList.contains("dragging")) return;
    el.releasePointerCapture?.(ev.pointerId);
    el.classList.remove("dragging");
    document.body.classList.remove("resizing");
    prefs.set(prefKey, getComputedStyle(root).getPropertyValue(varName).trim());
    drawGraph();
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
  // Double-click toggles collapsed <-> default, which is the fast path for "give me the
  // whole screen for the paper" and back.
  el.addEventListener("dblclick", () => {
    const dflt = { lib: "240px", left: "300px", right: "400px" }[edge] || "300px";
    const now = current();
    const next = now === 0 ? dflt : "0px";
    root.style.setProperty(varName, next);
    prefs.set(prefKey, next);
    setPaneCollapsed(edge, next === "0px");
    drawGraph();
    drawSearchGraph();
  });
}
