/* Persisted UI preferences.
 *
 * A leaf, like dom.js, because typography reads it at evaluation time. */

/* Pane widths, font and size all persist: they are reading preferences, and having to
 * re-set them on every visit is exactly the kind of small friction that makes a tool
 * feel unfinished. */
export const prefs = {
  get(k, d) { try { return localStorage.getItem("lara." + k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem("lara." + k, v); } catch { /* private mode */ } },
};
