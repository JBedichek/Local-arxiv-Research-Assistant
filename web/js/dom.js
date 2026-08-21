/* The three things every module needs: an element, a status line, and safe escaping.
 *
 * A leaf on purpose. Several modules reference `$` while their own module body is still
 * evaluating (binding listeners), and a module with no imports of its own is always fully
 * initialised before anything that imports it -- which is what makes the cycles further
 * up the graph safe. Nothing that knows about a feature belongs here. */

export const $ = (s) => document.querySelector(s);

export function setStatus(text, kind = "") {
  const el = $("#status");
  el.textContent = text;
  el.className = "status " + kind;
}

export function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
