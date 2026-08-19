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


def load(root: Path) -> dict:
    """Read the library, tolerating absence and corruption.

    A missing file is the normal first-run state. A corrupt one is renamed aside rather
    than deleted — it is the reader's data, and a truncated file is often still readable
    by hand — and an empty library is returned so the UI comes up instead of failing.
    """
    p = _path(root)
    if not p.exists():
        return _empty()
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        try:
            p.rename(p.with_suffix(f".corrupt-{int(time.time())}.json"))
        except OSError:
            pass
        return _empty()
    if not isinstance(data, dict):
        return _empty()
    data.setdefault("version", 1)
    data.setdefault("folders", [])
    data.setdefault("entries", [])
    return data


def save(root: Path, data: dict) -> None:
    p = _path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


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
