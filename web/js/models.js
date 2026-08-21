/* Choosing a generator, and downloading one from Hugging Face. */

import { api } from "./api.js";
import { $, escapeHtml } from "./dom.js";
import { syncQuant } from "./search.js";
import { state } from "./state.js";

let dlResolved = null;
let dlPoll = null;
/* /api/device, cached so the dialog can report what fits without refetching. */
let dlDevice = null;

/* Where to actually find each format. A model name alone is not enough: "Qwen3-8B" names
 * three different sets of files depending on who converted it, and only one of them will
 * load here. Keyed by /api/device's wants_format. */
const FORMAT_GUIDE = {
  gguf: {
    label: "GGUF",
    placeholder: "e.g. unsloth/Qwen3-8B-GGUF, or paste a Hugging Face URL",
    help: `Browse <a href="https://huggingface.co/models?library=gguf" target="_blank"
             rel="noopener">all GGUF models on Hugging Face</a> — or go straight to
           <a href="https://huggingface.co/bartowski" target="_blank" rel="noopener">bartowski</a>
           and <a href="https://huggingface.co/unsloth" target="_blank" rel="noopener">unsloth</a>,
           who publish GGUF builds of most popular models. Pick a <code>Q4_K_M</code> file
           for the 4-bit sizing above; a plain safetensors repo will not load.`,
  },
  mlx: {
    label: "MLX",
    placeholder: "e.g. mlx-community/Qwen3-8B-4bit, or paste a Hugging Face URL",
    help: `Browse <a href="https://huggingface.co/mlx-community" target="_blank"
             rel="noopener">mlx-community</a>, which hosts essentially every MLX
           conversion. Look for a <code>-4bit</code> suffix to match the sizing above.`,
  },
  safetensors: {
    label: "safetensors",
    placeholder: "e.g. Qwen/Qwen3-8B, or paste a Hugging Face URL",
    help: `Standard Hugging Face repos ship this format, so most model pages work as-is.
           GGUF builds will not load under vLLM.`,
  },
};

function openDownloadModal() {
  $("#dl-modal").hidden = false;
  $("#dl-info").innerHTML = "";
  $("#dl-start").hidden = true;
  $("#dl-progress").hidden = true;
  $("#dl-repo").value = "";
  $("#dl-repo").focus();
  api("/api/device").then((d) => {
    dlDevice = d;
    const where = d.unified_memory ? "unified RAM" : "VRAM";
    $("#dl-device").textContent =
      `${d.system}/${d.machine} · ${d.accelerator.toUpperCase()} · ${d.budget_gb} GB ${where}`
      + ` · backend: ${d.backend}`;
    /* Both null until `lara setup` has run — say so rather than showing a blank line,
     * since "no advice" and "your machine fits nothing" look identical otherwise. */
    const hint = $("#dl-fits");
    if (d.generator_max_params_4bit) {
      const b = d.generator_max_params_4bit / 1e9;
      hint.innerHTML =
        `With your index loaded, about <strong>${b >= 10 ? b.toFixed(0) : b.toFixed(1)}B `
        + `parameters at 4-bit</strong> fits in the ${d.generator_headroom_gb} GB left over. `
        + `Bigger models still download — they just will not load alongside the corpus.`;
    } else {
      hint.innerHTML =
        `Run <code>lara setup</code> to have this dialog tell you what size fits.`;
    }

    /* The three weight formats are not interchangeable, and picking the wrong one is a
     * multi-gigabyte mistake you only discover at load time. Say which one this machine
     * needs, and where to find it, before the box is typed into. */
    const fmt = $("#dl-format");
    const backend = escapeHtml(d.backend || "the configured backend");
    if (FORMAT_GUIDE[d.wants_format]) {
      const g = FORMAT_GUIDE[d.wants_format];
      fmt.innerHTML = `<strong>${backend} needs ${g.label} weights.</strong> ${g.help}`;
      $("#dl-repo").placeholder = g.placeholder;
    } else {
      fmt.innerHTML = "";
    }
  }).catch(() => {});
}

$("#dl-close").addEventListener("click", () => {
  $("#dl-modal").hidden = true;
  clearInterval(dlPoll);
});

/* KV cache and activations on top of the weights — same 1.35x the setup wizard uses. */
const KV_OVERHEAD = 1.35;

function renderResolved(r) {
  if (!r) return;
  const row = (k, v, cls = "") =>
    `<div class="row"><span class="k">${k}</span><span class="v ${cls}">${v}</span></div>`;
  let html = row("repo", escapeHtml(r.repo));
  if (r.params) html += row("parameters", (r.params / 1e9).toFixed(1) + "B");
  if (r.arch) html += row("architecture", escapeHtml(r.arch));
  if (r.quantization) html += row("quantization", escapeHtml(r.quantization));

  /* The quantisation is a property of the repo, not a choice: the server picks the
   * 4-bit build and this reports it. The repo still physically contains every other
   * quantisation, which is why the download names its files rather than the repo. */
  if (r.pick) html += row("quantisation", escapeHtml(r.pick));

  const size = r.size_gb;
  html += row("download size",
    size ? `~${size.toFixed(1)} GB` : (r.n_gguf ? "pick a quantisation above" : "unknown"));

  /* Recomputed here rather than reusing r.fit, which the server calculated for the
   * default quantisation and which goes stale the moment the selection changes. */
  if (size && dlDevice && dlDevice.budget_gb) {
    const need = size * KV_OVERHEAD;
    const budget = dlDevice.budget_gb;
    const where = dlDevice.unified_memory ? "unified RAM" : "VRAM";
    const ok = need <= budget;
    html += row("fits here",
      ok ? `yes — needs ~${need.toFixed(1)} GB of ${budget} GB ${where}`
         : `no — needs ~${need.toFixed(1)} GB but only ${budget} GB ${where}`,
      ok ? "ok" : "bad");
  } else if (r.fit) {
    html += row("fits here",
      r.fit.fits ? `yes — needs ~${r.fit.needed_gb} GB of ${r.fit.budget_gb} GB ${r.fit.where}`
                 : `no — needs ~${r.fit.needed_gb} GB but only ${r.fit.budget_gb} GB ${r.fit.where}`,
      r.fit.fits ? "ok" : "bad");
  }

  if (r.already_cached) html += row("status", "already in your cache", "ok");
  if (r.warning) {
    html += `<div class="row"><span class="k"></span>`
          + `<span class="v warn">${escapeHtml(r.warning)}</span></div>`;
  }
  $("#dl-info").innerHTML = html;
  // Offered even when it does not fit: the estimate is conservative and the machine is
  // the user's to judge.
  $("#dl-start").hidden = !!r.already_cached;
}

$("#dl-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const q = $("#dl-repo").value.trim();
  if (!q) return;
  $("#dl-info").innerHTML = `<span class="dim">looking up…</span>`;
  $("#dl-start").hidden = true;
  try {
    const r = await api("/api/model/resolve", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q }),
    });
    dlResolved = r;
    if (!r.exists || r.error) {
      $("#dl-info").innerHTML = `<span class="bad">${escapeHtml(r.error || "not found")}</span>`;
      return;
    }
    renderResolved(r);
  } catch (err) {
    $("#dl-info").innerHTML = `<span class="bad">${escapeHtml(String(err.message || err))}</span>`;
  }
});

$("#dl-start").addEventListener("click", async () => {
  if (!dlResolved) return;
  $("#dl-start").hidden = true;
  $("#dl-progress").hidden = false;
  try {
    await api("/api/model/download", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repo: dlResolved.repo,
        size_gb: dlResolved.size_gb,
        // Only the picked quantisation's files. Null for safetensors repos, where the
        // whole snapshot is the model. Without this a GGUF repo fetches every
        // quantisation it ships.
        files: dlResolved.pick_files || null,
      }),
    });
  } catch (err) {
    $("#dl-progress-text").innerHTML = `<span class="bad">${escapeHtml(String(err.message || err))}</span>`;
    return;
  }
  clearInterval(dlPoll);
  dlPoll = setInterval(async () => {
    try {
      const j = await api(`/api/model/download/${dlResolved.repo}`);
      $("#dl-progress .bar span").style.width = `${j.pct}%`;
      $("#dl-progress-text").textContent =
        `${j.status} · ${j.downloaded_gb} / ${j.total_gb || "?"} GB (${j.pct}%) · ${j.elapsed_s}s`;
      if (j.status === "done") {
        clearInterval(dlPoll);
        $("#dl-progress-text").innerHTML =
          `<span class="ok">downloaded.</span> Restart the generator to serve it: ` +
          `<code>lara serve-llm --model ${escapeHtml(j.repo)}</code>`;
        loadModels();
      } else if (j.status === "error") {
        clearInterval(dlPoll);
        $("#dl-progress-text").innerHTML = `<span class="bad">${escapeHtml(j.error || "failed")}</span>`;
      }
    } catch { clearInterval(dlPoll); }
  }, 1500);
});

/* The picker doubles as the entry point: a sentinel option opens the dialog, so there is
 * no separate button competing for space in the top bar. */
const DL_SENTINEL = "__download__";

let dlAutoOpened = false;

export async function loadModels() {
  try {
    const m = await api("/api/models");
    const live = new Set(m.loaded?.length ? m.loaded : [m.configured_default].filter(Boolean));
    /* A model whose backend is not installed stays listed but disabled: it is genuinely
     * unusable until the runtime exists, and silently dropping it is what made a model
     * the user had just downloaded appear nowhere at all. The label carries the fix. */
    const opts = m.models.map((x) => {
      const isLive = live.has(x.repo);
      const state = x.needs_install ? ` — ${x.hint}`
                  : isLive ? " — loaded"
                  : " — not loaded";
      return `<option value="${x.repo}"${isLive ? " selected" : ""}`
           + `${x.needs_install ? " disabled" : ""}>`
           + `${x.repo} (${x.size_gb}GB)${escapeHtml(state)}</option>`;
    });
    /* With an empty cache the sentinel would be the *only* option, which makes it the
     * selection already — so choosing it fires no `change` event and the dialog never
     * opened. A disabled placeholder keeps the sentinel something you can change *to*. */
    if (!m.models.length) {
      opts.unshift(`<option value="" disabled selected>no model in cache</option>`);
    }
    opts.push(`<option value="${DL_SENTINEL}">＋ Download new model…</option>`);
    state.models = m.models;
    $("#model").innerHTML = opts.join("");
    syncQuant();

    /* Nothing to generate with, so the download dialog is the only useful next step.
     * Once per page load: re-opening it after every poll would trap the user. */
    if (!m.models.length && !dlAutoOpened) {
      dlAutoOpened = true;
      openDownloadModal();
    }
    if (m.models.length) dlAutoOpened = false;
  } catch { /* picker is optional for browsing */ }
}

$("#model").addEventListener("change", (ev) => {
  if (ev.target.value === DL_SENTINEL) {
    // Restore the previous selection so the sentinel never becomes the active model.
    // Never fall back to one whose backend is missing — it cannot be served.
    const usable = state.models.filter((x) => !x.needs_install);
    ev.target.value = usable.find((x) => x.loaded)?.repo || usable[0]?.repo || "";
    openDownloadModal();
  }
  syncQuant();
});
