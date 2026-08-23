"""The reading library: visits, questions, folders, and the system-prompt override."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from lara.serve.deps import memory_root, require_state

router = APIRouter()


class VisitRequest(BaseModel):
    arxiv_id: str
    version: int = 1
    title: str = ""


class RememberRequest(BaseModel):
    question: str
    answer: str = ""
    arxiv_id: str | None = None
    version: int = 1
    title: str = ""
    scope: str = "corpus"
    selection: str | None = None


class FolderRequest(BaseModel):
    name: str = "New folder"
    parent: str | None = None


class FolderPatch(BaseModel):
    name: str | None = None
    parent: str | None = None
    reparent: bool = False       # distinguishes "move to root" from "leave where it is"


class EntryPatch(BaseModel):
    folder: str | None = None
    title: str | None = None
    note: str | None = None
    refile: bool = False         # same distinction: null folder means root, not no-op


class PromptRequest(BaseModel):
    text: str | None = None


@router.get("/api/memory")
def memory_list() -> JSONResponse:
    """The whole library. Small enough to send at once; the client builds the tree."""
    from lara.serve import memory as MEM

    root = memory_root()
    # Deep-research runs are filed here rather than recorded at completion, so that the
    # library is derived from the synthesis tables instead of racing them. A run that
    # finished while the reader was elsewhere -- or before this feature existed -- is
    # still their research, and shows up the first time they open the library.
    try:
        from lara.serve import synthesis as SY

        MEM.sync_reports(root, SY.list_runs(require_state().conn(), limit=10_000))
    except Exception:                                  # noqa: BLE001
        # The library is the reader's own history; failing to enrich it with reports must
        # never stop it from rendering the papers and questions it already has.
        pass

    data = MEM.load(root)
    # `_ts` is an internal coalescing clock, not something the UI should depend on.
    entries = [{k: v for k, v in e.items() if k != "_ts"} for e in data["entries"]]
    entries.sort(key=lambda e: e.get("updated_utc", ""), reverse=True)
    return JSONResponse({"folders": data["folders"], "entries": entries})


@router.post("/api/memory/visit")
def memory_visit(req: VisitRequest) -> JSONResponse:
    from lara.serve import memory as MEM

    e = MEM.record_visit(memory_root(), arxiv_id=req.arxiv_id,
                         version=req.version, title=req.title)
    return JSONResponse({k: v for k, v in e.items() if k != "_ts"})


@router.post("/api/memory/question")
def memory_question(req: RememberRequest) -> JSONResponse:
    from lara.serve import memory as MEM

    e = MEM.record_question(
        memory_root(), question=req.question, answer=req.answer,
        arxiv_id=req.arxiv_id, version=req.version, title=req.title,
        scope=req.scope, selection=req.selection,
    )
    return JSONResponse({k: v for k, v in e.items() if k != "_ts"})


@router.patch("/api/memory/entry/{entry_id}")
def memory_entry_patch(entry_id: str, req: EntryPatch) -> JSONResponse:
    from lara.serve import memory as MEM

    fields = {}
    if req.refile:
        fields["folder"] = req.folder
    if req.title is not None:
        fields["title"] = req.title
    if req.note is not None:
        fields["note"] = req.note
    e = MEM.update_entry(memory_root(), entry_id, **fields)
    if e is None:
        raise HTTPException(404, f"no entry {entry_id}")
    return JSONResponse({k: v for k, v in e.items() if k != "_ts"})


@router.delete("/api/memory/entry/{entry_id}")
def memory_entry_delete(entry_id: str) -> JSONResponse:
    from lara.serve import memory as MEM

    root = memory_root()
    # A report entry is a pointer, so deleting only the pointer would leave the run in the
    # synthesis tables -- still listed in the research pane's history, and restored by the
    # next sync. Deleting a report means deleting the run.
    entry = next((e for e in MEM.load(root)["entries"] if e.get("id") == entry_id), None)
    if entry is not None and entry.get("kind") == "report" and entry.get("run_id"):
        from lara.serve import synthesis as SY

        SY.delete_run(require_state().db_path, entry["run_id"])

    if not MEM.delete_entry(root, entry_id):
        raise HTTPException(404, f"no entry {entry_id}")
    return JSONResponse({"deleted": entry_id})


@router.post("/api/memory/folder")
def memory_folder_create(req: FolderRequest) -> JSONResponse:
    from lara.serve import memory as MEM

    return JSONResponse(MEM.create_folder(memory_root(), name=req.name, parent=req.parent))


@router.patch("/api/memory/folder/{folder_id}")
def memory_folder_patch(folder_id: str, req: FolderPatch) -> JSONResponse:
    from lara.serve import memory as MEM

    fields = {}
    if req.name is not None:
        fields["name"] = req.name
    if req.reparent:
        fields["parent"] = req.parent
    f = MEM.update_folder(memory_root(), folder_id, **fields)
    if f is None:
        raise HTTPException(404, f"no folder {folder_id}")
    return JSONResponse(f)


@router.delete("/api/memory/folder/{folder_id}")
def memory_folder_delete(folder_id: str) -> JSONResponse:
    from lara.serve import memory as MEM

    if not MEM.delete_folder(memory_root(), folder_id):
        raise HTTPException(404, f"no folder {folder_id}")
    return JSONResponse({"deleted": folder_id})


@router.get("/api/settings/prompt")
def prompt_get() -> JSONResponse:
    """The active system prompt, the built-in default, and which one is in force."""
    from lara.serve import memory as MEM
    from lara.serve.generate import SYSTEM

    custom = MEM.get_prompt(memory_root())
    return JSONResponse({
        "default": SYSTEM,
        "custom": custom,
        "active": custom or SYSTEM,
        "is_custom": custom is not None,
    })


@router.put("/api/settings/prompt")
def prompt_put(req: PromptRequest) -> JSONResponse:
    """Save an override, or clear it by sending empty text."""
    from lara.serve import memory as MEM
    from lara.serve.generate import SYSTEM

    MEM.set_prompt(memory_root(), req.text)
    custom = MEM.get_prompt(memory_root())
    return JSONResponse({
        "default": SYSTEM,
        "custom": custom,
        "active": custom or SYSTEM,
        "is_custom": custom is not None,
    })
