/* The reading library as a tree: folders, entries, renaming, and drag-and-drop. */

import { api, send } from "./api.js";
import { addMessage } from "./ask.js";
import { $, escapeHtml } from "./dom.js";
import { loadGraph, openPaper } from "./paper.js";
import { prefs } from "./prefs.js";
import { state } from "./state.js";
import { renderMath } from "./tex.js";

/* The server owns the data — it records visits and questions itself, so a client that
 * navigates away mid-answer still leaves a complete history. This side only renders the
 * tree and issues moves, which keeps the two from disagreeing about what happened. */

let library = { folders: [], entries: [] };

export async function loadLibrary() {
  try {
    library = await api("/api/memory");
  } catch {
    library = { folders: [], entries: [] };
  }
  renderLibrary();
}

const libOpen = {
  get(id) { return prefs.get("libOpen." + id, "1") === "1"; },
  set(id, v) { prefs.set("libOpen." + id, v ? "1" : "0"); },
};

function libEntryLabel(e) {
  if (e.kind === "question") return e.question || "(question)";
  return e.title || e.arxiv_id || "(paper)";
}

function renderLibrary() {
  const tree = $("#lib-tree");
  if (!tree) return;
  if (!library.folders.length && !library.entries.length) {
    tree.innerHTML =
      `<p class="lib-empty">Papers you open and questions you ask show up here.</p>`;
    return;
  }
  const byParent = new Map();
  for (const f of library.folders) {
    if (!byParent.has(f.parent || "")) byParent.set(f.parent || "", []);
    byParent.get(f.parent || "").push(f);
  }
  const byFolder = new Map();
  for (const e of library.entries) {
    const k = e.folder || "";
    if (!byFolder.has(k)) byFolder.set(k, []);
    byFolder.get(k).push(e);
  }

  const node = (html) => {
    const d = document.createElement("div");
    d.innerHTML = html;
    return d.firstElementChild;
  };

  function buildFolder(f) {
    const wrap = document.createElement("div");
    const open = libOpen.get(f.id);
    const kids = (byFolder.get(f.id) || []).length + (byParent.get(f.id) || []).length;
    const row = node(
      `<div class="lib-row lib-folder" draggable="true" data-folder="${f.id}">
         <span class="twist">${open ? "▾" : "▸"}</span>
         <span class="lib-label">${escapeHtml(f.name)}</span>
         <span class="lib-count">${kids || ""}</span>
         <button class="lib-del" title="Delete folder (contents move up)">×</button>
       </div>`);
    wrap.append(row);
    if (open) {
      const kidsEl = document.createElement("div");
      kidsEl.className = "lib-children";
      for (const sub of (byParent.get(f.id) || []).sort((a, b) => a.name.localeCompare(b.name)))
        kidsEl.append(buildFolder(sub));
      for (const e of byFolder.get(f.id) || []) kidsEl.append(buildEntry(e));
      wrap.append(kidsEl);
    }
    return wrap;
  }

  function buildEntry(e) {
    return node(
      `<div class="lib-row lib-entry ${e.kind}" draggable="true" data-entry="${e.id}"
            title="${escapeHtml(libEntryLabel(e))}">
         <span class="twist">${e.kind === "question" ? "?" : "•"}</span>
         <span class="lib-label">${escapeHtml(libEntryLabel(e))}</span>
         <span class="lib-kind">${e.kind === "question" ? escapeHtml((e.title || "").slice(0, 14)) : ""}</span>
         <button class="lib-del" title="Remove from library">×</button>
       </div>`);
  }

  tree.innerHTML = "";
  for (const f of (byParent.get("") || []).sort((a, b) => a.name.localeCompare(b.name)))
    tree.append(buildFolder(f));
  for (const e of byFolder.get("") || []) tree.append(buildEntry(e));
}

/* Restoring a question puts the reader back where they were: the paper open, the question
 * in the box, and the answer they already paid for shown rather than regenerated. */
/* Clicking a question restores the whole THREAD, not the single exchange.
 *
 * Threads are keyed by paper, exactly as the server keys them, so what is shown here is
 * what the model will be given as history on the next question. Restoring one turn in
 * isolation would show the reader less context than the model has, which makes the next
 * answer look like it came from nowhere. */
export async function restoreEntry(e) {
  if (e.arxiv_id) await openPaper(e.arxiv_id);
  if (e.kind !== "question") return;

  const key = e.arxiv_id || "";
  const thread = library.entries
    .filter((x) => x.kind === "question" && (x.arxiv_id || "") === key)
    .sort((a, b) => String(a.created_utc || "").localeCompare(String(b.created_utc || "")));

  $("#messages").innerHTML = "";
  let clicked = null;
  for (const t of thread) {
    addMessage("user", t.question, t.selection || null);
    const el = addMessage("assistant", "");
    el.classList.add("restored");
    if (t.id === e.id) { el.classList.add("focused"); clicked = el; }
    const when = escapeHtml((t.created_utc || "").slice(0, 16).replace("T", " "));
    el.querySelector(".text").innerHTML =
      `<span class="dim">from your library · ${when}</span><br>` +
      // Same rendering as a live answer: maths and citations, not escaped source.
      renderMath(escapeHtml(t.answer || "(no answer recorded)"));
  }

  $("#question").value = "";
  $("#question").placeholder = thread.length > 1
    ? `Continue this thread (${thread.length} questions)…`
    : "Ask a follow-up…";
  state.heatRef = { text: e.question, kind: "question" };
  loadGraph();
  clicked?.scrollIntoView({ block: "center" });
}

export function libFindEntry(id) { return library.entries.find((e) => e.id === id); }

$("#lib-tree")?.addEventListener("click", async (ev) => {
  const del = ev.target.closest(".lib-del");
  const row = ev.target.closest(".lib-row");
  if (!row) return;
  const entryId = row.dataset.entry, folderId = row.dataset.folder;
  if (del) {
    ev.stopPropagation();
    try {
      if (entryId) await api(`/api/memory/entry/${entryId}`, { method: "DELETE" });
      else if (folderId) await api(`/api/memory/folder/${folderId}`, { method: "DELETE" });
    } catch { /* fall through to the reload, which shows the real state either way */ }
    loadLibrary();
    return;
  }
  if (folderId) { libOpen.set(folderId, !libOpen.get(folderId)); renderLibrary(); return; }
  const e = libFindEntry(entryId);
  if (e) restoreEntry(e);
});

/* Rename in place rather than through `window.prompt`. A modal dialog blocks the whole
 * page, cannot be styled to match, and on a tree you are mid-drag through it is a jarring
 * interruption for what should be a two-second edit. */
function beginRename(row) {
  const id = row.dataset.folder;
  const label = row.querySelector(".lib-label");
  if (!id || !label || label.querySelector("input")) return;
  const cur = library.folders.find((f) => f.id === id);
  const input = document.createElement("input");
  input.className = "lib-rename";
  input.value = cur ? cur.name : "";
  label.textContent = "";
  label.append(input);
  input.focus();
  input.select();

  let done = false;
  const commit = async (save) => {
    if (done) return;                       // blur fires after Enter; commit exactly once
    done = true;
    const name = input.value;
    if (save && name.trim() && (!cur || name !== cur.name)) {
      await send("PATCH", `/api/memory/folder/${id}`, { name }).catch(() => {});
    }
    loadLibrary();
  };
  input.addEventListener("keydown", (ev) => {
    ev.stopPropagation();                   // ; [ ] are pane shortcuts outside a field
    if (ev.key === "Enter") { ev.preventDefault(); commit(true); }
    else if (ev.key === "Escape") { ev.preventDefault(); commit(false); }
  });
  input.addEventListener("blur", () => commit(true));
  input.addEventListener("click", (ev) => ev.stopPropagation());
  input.addEventListener("dblclick", (ev) => ev.stopPropagation());
}

$("#lib-tree")?.addEventListener("dblclick", (ev) => {
  const row = ev.target.closest(".lib-row.lib-folder");
  if (row) beginRename(row);
});

/* Create first, name second: the folder exists immediately and the name is an edit on a
 * real object, so an abandoned rename leaves "New folder" rather than nothing. */
$("#lib-newfolder")?.addEventListener("click", async () => {
  let created = null;
  try {
    created = await send("POST", "/api/memory/folder", { name: "New folder" });
  } catch { return; }
  await loadLibrary();
  const row = document.querySelector(`.lib-row.lib-folder[data-folder="${created.id}"]`);
  if (row) beginRename(row);
});

/* Drag to file. `refile`/`reparent` are sent explicitly because a null folder means "the
 * root", and without the flag the server cannot tell that apart from "field omitted". */
let libDragged = null;

$("#lib-tree")?.addEventListener("dragstart", (ev) => {
  const row = ev.target.closest(".lib-row");
  if (!row) return;
  libDragged = { entry: row.dataset.entry || null, folder: row.dataset.folder || null };
  ev.dataTransfer.effectAllowed = "move";
  ev.dataTransfer.setData("text/plain", row.dataset.entry || row.dataset.folder || "");
});

$("#lib-tree")?.addEventListener("dragover", (ev) => {
  if (!libDragged) return;
  ev.preventDefault();
  ev.dataTransfer.dropEffect = "move";
  const row = ev.target.closest(".lib-row.lib-folder");
  for (const el of document.querySelectorAll(".lib-row.drop-into")) el.classList.remove("drop-into");
  if (row) { row.classList.add("drop-into"); $("#lib-tree").classList.remove("drop-root"); }
  else $("#lib-tree").classList.add("drop-root");
});

$("#lib-tree")?.addEventListener("dragleave", (ev) => {
  if (ev.target === $("#lib-tree")) $("#lib-tree").classList.remove("drop-root");
});

$("#lib-tree")?.addEventListener("drop", async (ev) => {
  if (!libDragged) return;
  ev.preventDefault();
  const row = ev.target.closest(".lib-row.lib-folder");
  const target = row ? row.dataset.folder : null;
  for (const el of document.querySelectorAll(".lib-row.drop-into")) el.classList.remove("drop-into");
  $("#lib-tree").classList.remove("drop-root");
  const moved = libDragged;
  libDragged = null;
  if (moved.folder && moved.folder === target) return;   // dropping a folder on itself
  try {
    if (moved.entry) {
      await send("PATCH", `/api/memory/entry/${moved.entry}`, { folder: target, refile: true });
    } else if (moved.folder) {
      await send("PATCH", `/api/memory/folder/${moved.folder}`, { parent: target, reparent: true });
    }
  } catch { /* reload below reveals whatever actually happened */ }
  loadLibrary();
});

$("#lib-tree")?.addEventListener("dragend", () => {
  libDragged = null;
  for (const el of document.querySelectorAll(".lib-row.drop-into")) el.classList.remove("drop-into");
  $("#lib-tree")?.classList.remove("drop-root");
});
