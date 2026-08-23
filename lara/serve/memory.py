"""Persistent reading history: papers opened, questions asked, and the reader's folders.

One JSON file, ``memory/library.json``, holding a flat entry list plus a flat folder list
that carries parent pointers. Flat-with-parents rather than a nested tree because every
operation the UI performs — move, rename, delete, reparent — is a single field write on
one record, where a nested structure would mean splicing subtrees and re-serialising
around them. The tree is rebuilt for display in the client, which is the only place its
shape matters.

**Why a file and not SQLite.** The corpus database is 42 GB and rebuilt by tooling that
has no reason to know about the reader's bookmarks; putting a few thousand rows of
personal state inside it couples the two and makes the library impossible to copy, diff,
or hand-edit. This file is expected to stay small — a heavy user generates a few thousand
entries a year — and being plain JSON means a reader can fix it with a text editor when
something goes wrong.

**Writes are atomic.** Every save writes a sibling temp file, fsyncs it, and renames over
the target. A half-written library read back at startup is indistinguishable from a
corrupt one, and losing a reading history to a crash mid-write is a bad trade for the few
milliseconds the fsync costs. A process-wide lock serialises writers, because the server
answers requests from a threadpool and two concurrent visits would otherwise
read-modify-write over each other.
"""

from __future__ import annotations

import calendar
import json
import os
import threading
import time
import uuid
from pathlib import Path

LIBRARY_NAME = "library.json"
PROMPT_NAME = "system_prompt.txt"

#: Consecutive visits to the same paper collapse into one row if they land inside this
#: window. Without it, a reader flipping between two papers writes a new entry per click
#: and the history becomes unreadable within a session.
VISIT_COALESCE_SEC = 6 * 60 * 60

_LOCK = threading.RLock()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _empty() -> dict:
    return {"version": 1, "folders": [], "entries": []}


def _path(root: Path) -> Path:
    return Path(root) / LIBRARY_NAME


def _read(path: Path, empty) -> dict:
    """Read one of this module's JSON files, tolerating absence and corruption.

    A missing file is the normal first-run state. A corrupt one is renamed aside rather
    than deleted — it is the reader's data, and a truncated file is often still readable
    by hand — and the empty shape is returned so the UI comes up instead of failing.

    Missing top-level keys are filled from that same shape, so a file written by an older
    version does not need a migration to be readable.
    """
    if not path.exists():
        return empty()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        try:
            path.rename(path.with_suffix(f".corrupt-{int(time.time())}.json"))
        except OSError:
            pass
        return empty()
    if not isinstance(data, dict):
        return empty()
    for k, v in empty().items():
        data.setdefault(k, v)
    return data


def _write(path: Path, data: dict) -> None:
    """Write atomically: fsync then rename, so a crash mid-write cannot truncate the file
    the reader's whole library lives in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load(root: Path) -> dict:
    """The reading library: folders and entries."""
    return _read(_path(root), _empty)


def save(root: Path, data: dict) -> None:
    _write(_path(root), data)


def _folder_exists(data: dict, folder_id: str | None) -> bool:
    return folder_id is None or any(f["id"] == folder_id for f in data["folders"])


# ── entries ──────────────────────────────────────────────────────────────────────────

def record_visit(root: Path, *, arxiv_id: str, version: int = 1, title: str = "") -> dict:
    """Record that a paper was opened, coalescing repeat visits.

    Returns the entry, new or updated. Reopening a paper bumps its timestamp and counter
    rather than appending, so the history reads as "papers I have read" instead of a
    click log.
    """
    with _LOCK:
        data = load(root)
        now = time.time()
        # The MOST RECENT matching row decides, not the first one encountered. Entries are
        # in insertion order, so scanning forward hits the oldest visit first; treating
        # that one as the candidate meant that once it aged past the window every later
        # visit appended a fresh row and the history became the click log this coalescing
        # exists to prevent.
        recent = max(
            (e for e in data["entries"]
             if e.get("kind") == "paper" and e.get("arxiv_id") == arxiv_id),
            key=lambda e: e.get("_ts", 0),
            default=None,
        )
        if recent is not None and now - recent.get("_ts", 0) <= VISIT_COALESCE_SEC:
            recent["_ts"] = now
            recent["updated_utc"] = _now()
            recent["visits"] = int(recent.get("visits", 1)) + 1
            if title:
                recent["title"] = title
            save(root, data)
            return recent
        entry = {
            "id": uuid.uuid4().hex[:12],
            "kind": "paper",
            "folder": None,
            "arxiv_id": arxiv_id,
            "version": int(version or 1),
            "title": title or arxiv_id,
            "visits": 1,
            "created_utc": _now(),
            "updated_utc": _now(),
            "_ts": now,
        }
        data["entries"].append(entry)
        save(root, data)
        return entry


def record_report(root: Path, *, run_id: str, question: str, tldr: str = "",
                  n_rounds: int = 0, n_claims: int = 0, n_papers: int = 0) -> dict:
    """File a completed deep-research run in the library.

    The entry holds a ``run_id`` and a summary line, never the report itself. The run --
    its rounds, its claims, every judgement it made -- already lives in the synthesis
    tables, and copying that into a JSON file that is read whole on every page load would
    make the library slower in proportion to how much research had been done with it.

    Keyed on ``run_id``, so re-recording a run it already has updates that entry. A run
    saves once, but a backfill sweep may meet it again and must not duplicate it.
    """
    with _LOCK:
        data = load(root)
        existing = next((e for e in data["entries"]
                         if e.get("kind") == "report" and e.get("run_id") == run_id), None)
        fields = {
            "question": question,
            "title": (question or "research")[:120],
            "tldr": (tldr or "")[:600],
            "n_rounds": int(n_rounds), "n_claims": int(n_claims),
            "n_papers": int(n_papers), "updated_utc": _now(),
        }
        if existing is not None:
            existing.update(fields)
            save(root, data)
            return existing
        entry = {
            "id": uuid.uuid4().hex[:12],
            "kind": "report",
            "folder": None,
            "run_id": run_id,
            **fields,
            "created_utc": _now(),
            "_ts": time.time(),
        }
        data["entries"].append(entry)
        save(root, data)
        return entry


def sync_reports(root: Path, runs: list[dict]) -> int:
    """Give every saved run a library entry. Returns how many were added.

    This is what makes the feature retroactive: research done before the library existed
    is still research the reader did, and it should be there the first time they look.

    Only ever adds. Deleting a report deletes the run itself, so a run that is gone stays
    gone -- if deletion merely removed the library entry, the next sweep would resurrect it
    and the delete button would appear not to work.
    """
    with _LOCK:
        data = load(root)
        known = {e.get("run_id") for e in data["entries"] if e.get("kind") == "report"}
        added = 0
        for r in runs:
            rid = r.get("run_id")
            if not rid or rid in known:
                continue
            data["entries"].append({
                "id": uuid.uuid4().hex[:12], "kind": "report", "folder": None,
                "run_id": rid, "question": r.get("question", ""),
                "title": (r.get("question") or "research")[:120],
                "tldr": (r.get("tldr") or "")[:600],
                "n_rounds": int(r.get("n_rounds") or 0),
                "n_claims": int(r.get("n_claims") or 0),
                "n_papers": int(r.get("n_papers") or 0),
                # Normalised to this store's format, not SQLite's. The library is sorted
                # by comparing these as STRINGS, and `2026-08-23 03:42` vs
                # `2026-08-23T03:31Z` compares 'T' (0x54) against ' ' (0x20) -- so every
                # ISO entry outranks every SQLite one whatever the clock says, and the
                # newest report files below yesterday's questions.
                "created_utc": _iso(_ts_of(r.get("created_utc"))),
                "updated_utc": _iso(_ts_of(r.get("created_utc"))),
                # A backfilled run carries the time it actually ran; stamping "now" would
                # file weeks of research as today's.
                "_ts": _ts_of(r.get("created_utc")),
            })
            known.add(rid)
            added += 1
        if added:
            save(root, data)
        return added


def _iso(ts: float) -> str:
    """Epoch seconds in the one timestamp format this store sorts by."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _ts_of(created_utc: str | None) -> float:
    """`2026-08-23 03:42:12` -> epoch seconds, falling back to now."""
    if not created_utc:
        return time.time()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return calendar.timegm(time.strptime(created_utc, fmt))
        except ValueError:
            continue
    return time.time()


def record_question(
    root: Path,
    *,
    question: str,
    answer: str = "",
    arxiv_id: str | None = None,
    version: int = 1,
    title: str = "",
    scope: str = "corpus",
    selection: str | None = None,
) -> dict:
    """Record a question and the answer it produced.

    Questions are never coalesced: asking the same thing twice about the same paper is a
    real event with a different answer, and collapsing them would hide exactly the
    comparison a reader wants.
    """
    with _LOCK:
        data = load(root)
        entry = {
            "id": uuid.uuid4().hex[:12],
            "kind": "question",
            "folder": None,
            "arxiv_id": arxiv_id,
            "version": int(version or 1),
            "title": title or (arxiv_id or ""),
            "question": question,
            # Answers are kept so a restored entry shows what was said without paying for
            # generation again. Truncated because the library is loaded whole on every
            # page load and a few hundred full answers would dominate it.
            "answer": (answer or "")[:4000],
            "scope": scope,
            "selection": (selection or "")[:1000] or None,
            "created_utc": _now(),
            "updated_utc": _now(),
            "_ts": time.time(),
        }
        data["entries"].append(entry)
        save(root, data)
        return entry


def update_entry(root: Path, entry_id: str, **fields) -> dict | None:
    """Move, rename or re-note one entry. Unknown fields are ignored, not stored."""
    allowed = {"folder", "title", "note", "question"}
    with _LOCK:
        data = load(root)
        for e in data["entries"]:
            if e["id"] != entry_id:
                continue
            for k, v in fields.items():
                if k not in allowed:
                    continue
                if k == "folder" and not _folder_exists(data, v):
                    continue           # a stale folder id would strand the entry
                e[k] = v
            e["updated_utc"] = _now()
            save(root, data)
            return e
        return None


def delete_entry(root: Path, entry_id: str) -> bool:
    with _LOCK:
        data = load(root)
        before = len(data["entries"])
        data["entries"] = [e for e in data["entries"] if e["id"] != entry_id]
        if len(data["entries"]) == before:
            return False
        save(root, data)
        return True


# ── folders ──────────────────────────────────────────────────────────────────────────

def create_folder(root: Path, *, name: str, parent: str | None = None) -> dict:
    with _LOCK:
        data = load(root)
        if not _folder_exists(data, parent):
            parent = None
        folder = {
            "id": uuid.uuid4().hex[:12],
            "name": (name or "New folder").strip()[:80] or "New folder",
            "parent": parent,
            "created_utc": _now(),
        }
        data["folders"].append(folder)
        save(root, data)
        return folder


def _descendants(data: dict, folder_id: str) -> set[str]:
    out, frontier = {folder_id}, [folder_id]
    while frontier:
        cur = frontier.pop()
        for f in data["folders"]:
            if f.get("parent") == cur and f["id"] not in out:
                out.add(f["id"])
                frontier.append(f["id"])
    return out


def update_folder(root: Path, folder_id: str, **fields) -> dict | None:
    """Rename or reparent a folder, refusing moves that would create a cycle."""
    with _LOCK:
        data = load(root)
        target = next((f for f in data["folders"] if f["id"] == folder_id), None)
        if target is None:
            return None
        if "name" in fields:
            target["name"] = (fields["name"] or "").strip()[:80] or target["name"]
        if "parent" in fields:
            parent = fields["parent"]
            # Dropping a folder into its own subtree would detach that whole branch from
            # the root and it would vanish from the UI with no way to get it back.
            if parent in _descendants(data, folder_id):
                return target
            if _folder_exists(data, parent):
                target["parent"] = parent
        save(root, data)
        return target


def delete_folder(root: Path, folder_id: str) -> bool:
    """Delete a folder, lifting its contents to its parent rather than destroying them.

    Deleting a container should not silently delete work filed inside it; a reader
    tidying folders is not asking to lose the papers they filed.
    """
    with _LOCK:
        data = load(root)
        target = next((f for f in data["folders"] if f["id"] == folder_id), None)
        if target is None:
            return False
        parent = target.get("parent")
        for f in data["folders"]:
            if f.get("parent") == folder_id:
                f["parent"] = parent
        for e in data["entries"]:
            if e.get("folder") == folder_id:
                e["folder"] = parent
        data["folders"] = [f for f in data["folders"] if f["id"] != folder_id]
        save(root, data)
        return True


# ── system prompt override ───────────────────────────────────────────────────────────

def get_prompt(root: Path) -> str | None:
    """The reader's system prompt, or None when they have not overridden the default."""
    p = Path(root) / PROMPT_NAME
    try:
        text = p.read_text().strip()
    except OSError:
        return None
    return text or None


def set_prompt(root: Path, text: str | None) -> None:
    """Save an override, or clear it when handed empty text.

    Clearing deletes the file rather than writing an empty one, so "no override" has a
    single representation and `get_prompt` cannot return an empty system prompt — which
    the generator would happily accept and then answer with no grounding rules at all.
    """
    p = Path(root) / PROMPT_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    if not (text or "").strip():
        try:
            p.unlink()
        except OSError:
            pass
        return
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write(text.strip())
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


# ── taste profile ────────────────────────────────────────────────────────────────────

TASTE_NAME = "taste.json"


def _taste_path(root: Path) -> Path:
    return Path(root) / TASTE_NAME


def _empty_taste() -> dict:
    return {"version": 1, "marks": []}


def load_taste(root: Path) -> dict:
    """Marks the reader made on passages they found interesting.

    Kept in its own file rather than in ``library.json``. The library is fetched on every
    page load to draw the tree; the taste profile is read when scoring a paper and written
    when a passage is marked, and the two would otherwise contend on the same lock and the
    same serialisation for no reason.
    """
    return _read(_taste_path(root), _empty_taste)


def save_taste(root: Path, data: dict) -> None:
    _write(_taste_path(root), data)


def record_taste(root: Path, *, chunk_id: int, vector_row: int, arxiv_id: str = "",
                 title: str = "", text: str = "", note: str = "") -> dict:
    """Mark one passage as interesting. Marking the same chunk twice is not an error.

    Re-marking updates the note and timestamp instead of adding a duplicate: the profile
    is a set of positions in embedding space, and the same position twice would silently
    double that interest's weight in every reduction that sums.
    """
    with _LOCK:
        data = load_taste(root)
        for m in data["marks"]:
            if m.get("chunk_id") == chunk_id:
                m["updated_utc"] = _now()
                if note:
                    m["note"] = note
                save_taste(root, data)
                return m
        mark = {
            "id": uuid.uuid4().hex[:12],
            "chunk_id": int(chunk_id),
            "vector_row": int(vector_row),
            "arxiv_id": arxiv_id,
            "title": title,
            "text": (text or "")[:400],
            "note": note or "",
            "created_utc": _now(),
            "updated_utc": _now(),
        }
        data["marks"].append(mark)
        save_taste(root, data)
        return mark


def delete_taste(root: Path, mark_id: str) -> bool:
    with _LOCK:
        data = load_taste(root)
        before = len(data["marks"])
        data["marks"] = [m for m in data["marks"] if m["id"] != mark_id]
        if len(data["marks"]) == before:
            return False
        save_taste(root, data)
        return True


# ── thread summaries ──────────────────────────────────────────────────────────────
#
# A compressed conversation. Kept beside the entries rather than replacing them: the
# reader's library should still show every question they asked, even once the model has
# stopped being sent them verbatim.

def get_thread_summary(root: Path, tid: str) -> str:
    return (load(root).get("thread_summaries", {}).get(tid) or {}).get("summary", "")


def get_thread_summary_covered(root: Path, tid: str) -> list[str]:
    """Entry ids already folded into the summary, so they are not sent twice."""
    return (load(root).get("thread_summaries", {}).get(tid) or {}).get("covered", [])


def set_thread_summary(root: Path, tid: str, summary: str, covered: list[str]) -> dict:
    with _LOCK:
        data = load(root)
        store = data.setdefault("thread_summaries", {})
        prev = store.get(tid) or {}
        # Compression is cumulative: a second pass folds the first summary's coverage in
        # rather than orphaning it, or those turns would silently return to the prompt.
        merged = list(dict.fromkeys([*prev.get("covered", []), *covered]))
        store[tid] = {"summary": summary, "covered": merged, "updated_utc": _now()}
        save(root, data)
        return store[tid]


def clear_thread_summary(root: Path, tid: str) -> bool:
    with _LOCK:
        data = load(root)
        if (data.get("thread_summaries") or {}).pop(tid, None) is None:
            return False
        save(root, data)
        return True


# ── the library graph's cache ─────────────────────────────────────────────────────
#
# lara.serve.library_graph owns what a graph IS; this owns where it is kept. It used to
# reach in here for _LOCK and call load() twice per cache check — once for the graph and
# once more inside the fingerprint — which is three reads of the whole library to answer
# "is the cached one still good".


def graph_cache(root: Path) -> tuple[list[str], dict | None]:
    """The question ids a cached graph is keyed on, and the cache entry, from one read."""
    data = _read(_path(root), _empty)
    ids = sorted(e.get("id", "") for e in data.get("entries", [])
                 if e.get("kind") == "question")
    return ids, data.get("library_graph")


def store_graph(root: Path, fingerprint: str, graph: dict) -> None:
    """Replace the cached graph, under the same lock every other writer takes."""
    with _LOCK:
        data = _read(_path(root), _empty)
        data["library_graph"] = {"fingerprint": fingerprint, "graph": graph}
        _write(_path(root), data)
