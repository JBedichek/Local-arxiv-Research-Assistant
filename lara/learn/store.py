"""Courses on disk: plain JSON under ~/.lara/courses, written atomically, no database."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

ROOT = Path.home() / ".lara" / "courses"
SHARED = "_shared"
#: A shared concept older than this is rebuilt, not reused: the corpus and the field move.
SHARED_MAX_AGE_DAYS = 30


def slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit] or "x"


def new_course_id(goal: str) -> str:
    return f"{slug(goal, 24)}-{uuid.uuid4().hex[:6]}"


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def _read(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def course_dir(course_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]+", course_id or ""):
        raise ValueError(f"bad course id {course_id!r}")
    return ROOT / course_id


def save_course(course: dict) -> None:
    course["updated"] = time.time()
    _write(course_dir(course["id"]) / "course.json", course)


def load_course(course_id: str) -> dict | None:
    return _read(course_dir(course_id) / "course.json")


def list_courses() -> list[dict]:
    out = []
    for path in sorted(ROOT.glob("*/course.json")) if ROOT.exists() else []:
        c = _read(path)
        if c and path.parent.name != SHARED:
            out.append({"id": c["id"], "goal": c.get("goal", ""), "status": c.get("status", ""),
                        "created": c.get("created", 0), "updated": c.get("updated", 0),
                        "concepts": len(c.get("concepts") or [])})
    return sorted(out, key=lambda c: c["updated"], reverse=True)


def delete_course(course_id: str) -> bool:
    import shutil

    path = course_dir(course_id)
    existed = path.exists()
    shutil.rmtree(path, ignore_errors=True)
    return existed


def save_concept(course_id: str, concept_id: str, content: dict) -> None:
    _write(course_dir(course_id) / "concepts" / f"{slug(concept_id)}.json", content)


def load_concept(course_id: str, concept_id: str) -> dict | None:
    return _read(course_dir(course_id) / "concepts" / f"{slug(concept_id)}.json")


def save_build(course_id: str, concept_id: str, build: dict) -> None:
    """Progress of one concept's build, in its own file: builds run concurrently, and a
    shared course.json rewritten whole by each of them lost the others' progress."""
    _write(course_dir(course_id) / "concepts" / f"{slug(concept_id)}.build.json", build)


def load_build(course_id: str, concept_id: str) -> dict:
    return _read(course_dir(course_id) / "concepts" / f"{slug(concept_id)}.build.json") or {}


def trace_path(course_id: str, concept_id: str) -> Path:
    """Where the Profile tab's event log for one concept's build lives -- see `trace.py`.
    Plain JSONL, not `_write`'s JSON-with-tmp-and-replace: a trace is appended to as the build
    runs, not rewritten whole."""
    return course_dir(course_id) / "concepts" / f"{slug(concept_id)}.trace.jsonl"


def save_learner(course_id: str, learner: dict) -> None:
    _write(course_dir(course_id) / "learner.json", learner)


def load_learner(course_id: str) -> dict:
    return _read(course_dir(course_id) / "learner.json") or {}


def shared_get(title: str, *, max_age_days: float = SHARED_MAX_AGE_DAYS) -> dict | None:
    """A concept some earlier course already built, if recent enough to trust."""
    found = _read(ROOT / SHARED / f"{slug(title, 60)}.json")
    if not found or time.time() - found.get("built", 0) > max_age_days * 86_400:
        return None
    return found


def shared_put(title: str, content: dict) -> None:
    _write(ROOT / SHARED / f"{slug(title, 60)}.json", {**content, "built": time.time()})
