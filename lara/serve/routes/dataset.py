"""Serving a published corpus to other nodes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from lara.serve.deps import require_state

router = APIRouter()


@router.get("/api/dataset/manifest")
def dataset_manifest() -> JSONResponse:
    """What this node is publishing, if anything."""
    from lara.serve import dataset as DS

    s = require_state()
    m = DS.load_manifest(s.cfg.get_path("disk.root"))
    if m is None:
        raise HTTPException(404, "nothing published — run `lara dataset publish` first")
    return JSONResponse(m)


@router.get("/api/dataset/file/{name:path}")
def dataset_file(name: str) -> FileResponse:
    """Serve one published artefact. Range requests are honoured, so fetches resume."""
    from lara.serve import dataset as DS

    s = require_state()
    root = s.cfg.get_path("disk.root")
    m = DS.load_manifest(root)
    if m is None:
        raise HTTPException(404, "nothing published")
    # Serve only what the manifest names. Without this the endpoint is an arbitrary file
    # read on a machine that has no authentication.
    if name not in {f["name"] for f in m["files"]}:
        raise HTTPException(404, f"{name} is not published")
    path = (root / name).resolve()
    if not str(path).startswith(str(root.resolve())) or not path.is_file():
        raise HTTPException(404, "not a published file")
    return FileResponse(path, filename=Path(name).name)
