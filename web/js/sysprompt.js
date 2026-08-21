/* The system-prompt override editor. */

import { api, send } from "./api.js";
import { $ } from "./dom.js";

let promptState = { default: "", custom: null };

export async function loadPrompt() {
  try {
    promptState = await api("/api/settings/prompt");
  } catch {
    return;
  }
  const ta = $("#sysprompt");
  if (!ta) return;
  ta.value = promptState.active || "";
  setPromptState(promptState.is_custom ? "custom" : "default");
}

function setPromptState(kind, note) {
  const el = $("#sysprompt-state");
  if (!el) return;
  el.textContent = note || (kind === "custom" ? "using your prompt" : "using the default");
}

$("#sysprompt-save")?.addEventListener("click", async () => {
  const text = $("#sysprompt").value;
  try {
    promptState = await send("PUT", "/api/settings/prompt", { text });
    $("#sysprompt").value = promptState.active || "";
    setPromptState(promptState.is_custom ? "custom" : "default", "saved");
  } catch (err) {
    setPromptState("", "save failed: " + (err.message || err));
  }
});

$("#sysprompt-reset")?.addEventListener("click", async () => {
  try {
    promptState = await send("PUT", "/api/settings/prompt", { text: "" });
    $("#sysprompt").value = promptState.active || "";
    setPromptState("default", "restored");
  } catch (err) {
    setPromptState("", "reset failed: " + (err.message || err));
  }
});
