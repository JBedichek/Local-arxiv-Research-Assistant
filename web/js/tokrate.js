/* Generation speed in the top bar -- polled from /api/generation/rate, which reads vLLM's
 * own /metrics the same way autoresearch's Context tab does (scrape a Prometheus counter,
 * report the delta since the last poll as a rate). Always on: unlike a per-tab panel, this
 * lives in the header, so there is no "the panel is closed" state to gate polling behind --
 * only the page being backgrounded, which visibilitychange below covers. */

import { api } from "./api.js";
import { $ } from "./dom.js";

const EVERY_MS = 2000;

async function poll() {
  const el = $("#tps");
  if (!el || document.visibilityState === "hidden") return;
  let data;
  try {
    data = await api("/api/generation/rate");
  } catch {
    el.textContent = "";
    return;
  }
  el.textContent = !data.reachable ? "" : data.tokens_per_sec > 0 ? `${data.tokens_per_sec} tok/s` : "idle";
}

export function bindTokRate() {
  poll();
  setInterval(poll, EVERY_MS);
  // A poll skipped while hidden leaves a stale number on the badge until the next tick;
  // catching the tab becoming visible again is what keeps that gap under EVERY_MS.
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") poll();
  });
}
