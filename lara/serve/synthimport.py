"""Copy synthesis work done under autoresearch into lara's own stores.

Copy-only: the source is never modified, and every step skips what the destination already
has, so importing twice adds nothing. Two kinds of thing come across:

- the memory that later runs read back -- distilled facts, goal embeddings, the shown/clicked
  follow-up log and the standing interest profile (`import_memory`);
- the finished runs themselves, as read-only records with their graph (`import_runs`).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from lara.serve import facts as FA
from lara.serve import interest as IN
from lara.serve import synthesizer as SY
from lara.serve import synthruns as SR

SOURCE = Path.home() / ".autoresearch"

#: Statuses a copied run keeps; anything else was live when it was recorded.
_KEPT = {SR.DONE, SR.FAILED, SR.CANCELLED}


def _rows(path: Path) -> list[dict]:
    out = []
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return out
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _canon(row: dict) -> str:
    return json.dumps(row, sort_keys=True)


def _append_new(src: Path, dst: Path) -> dict:
    have = {_canon(r) for r in _rows(dst)}
    fresh = [r for r in _rows(src) if _canon(r) not in have]
    if fresh:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("a") as f:
            for r in fresh:
                f.write(json.dumps(r) + "\n")
    return {"copied": len(fresh), "skipped": len(_rows(src)) - len(fresh)}


def import_memory(source: Path | None = None, *, facts: Path | None = None,
                  goals: Path | None = None, suggestions: Path | None = None,
                  profile: Path | None = None) -> dict:
    """Append the rows the destination lacks; copy the profile only if there is none."""
    src = source or SOURCE
    out = {
        "facts": _append_new(src / "facts.jsonl", facts or FA.FACTS_STORE),
        "goal_embeddings": _append_new(src / "goal_embeddings.jsonl",
                                       goals or IN.GOAL_EMBEDDINGS_STORE),
        "suggestions": _append_new(src / "followup_suggestions.jsonl",
                                   suggestions or IN.SUGGESTIONS_STORE),
    }
    dst = profile or IN.PROFILE_PATH
    theirs = src / "interest_profile.json"
    if dst.exists():
        out["profile"] = "kept the existing profile"
    elif theirs.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(theirs, dst)
        out["profile"] = "copied"
    else:
        out["profile"] = "none to copy"
    return out


def _as_record(d: dict) -> dict | None:
    """A lara run record from an autoresearch run record, or None if it is not a finished
    synthesis run: synthesis runs carry their graph as their plan."""
    plan = d.get("plan")
    if not (isinstance(plan, dict) and "goals" in plan) or not (d.get("deliverable") or "").strip():
        return None
    rec = SR.new_record(str(d.get("goal") or plan.get("objective") or ""),
                        parent=str(d.get("parent") or ""))
    status = d.get("status")
    rec.update(id=str(d["id"]), status=status if status in _KEPT else SR.INTERRUPTED,
               created=d.get("started"), ended=d.get("finished"), verdict=d.get("verdict"),
               error=str(d.get("error") or ""), rounds=int(plan.get("round") or 0),
               tokens_in=int(d.get("tokens_in") or 0), tokens_out=int(d.get("tokens_out") or 0),
               imported="autoresearch")
    for key in ("deliverable", "references", "deliverable_medium",
                "deliverable_medium_references", "deliverable_short",
                "deliverable_short_references", "followups"):
        if d.get(key):
            rec[key] = d[key]
    return rec


def import_runs(source: Path | None = None, *, root: Path | None = None,
                states: Path | None = None) -> dict:
    """Copy each finished synthesis run and its graph. A run already present is left as is;
    one that could not be written is listed under `failed`."""
    runs_dir = (source or SOURCE) / "runs"
    copied, skipped, failed = [], [], []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            d = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rec = _as_record(d) if isinstance(d, dict) else None
        if rec is None:
            continue
        if (SR._path(rec["id"], root)).exists():
            skipped.append(rec["id"])
            continue
        graph = SY.save(rec["id"], SY.SynthesizerState.from_dict(d["plan"]), root=states)
        # The record goes last: a run whose graph failed to write is retried next time
        # rather than skipped forever as "already present".
        if graph.exists() and SR.save_record(rec, root=root):
            copied.append(rec["id"])
        else:
            failed.append(rec["id"])
    return {"copied": copied, "skipped": skipped, "failed": failed}
