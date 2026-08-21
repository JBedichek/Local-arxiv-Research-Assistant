"""The static single-page app: HTML, JavaScript, stylesheet, and deep links."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

WEB_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "web"

# no-cache, not no-store: the browser still revalidates cheaply with ETag, but never
# serves a stale bundle after an edit. Debugging a fix that "did not work" because the
# browser kept yesterday's JavaScript is a bad afternoon.
_NOCACHE = {"Cache-Control": "no-cache, must-revalidate"}


@router.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html", headers=_NOCACHE)


@router.get("/app.js")
def appjs() -> FileResponse:
    return FileResponse(WEB_ROOT / "app.js", media_type="application/javascript",
                        headers=_NOCACHE)


@router.get("/style.css")
def appcss() -> FileResponse:
    return FileResponse(WEB_ROOT / "style.css", media_type="text/css", headers=_NOCACHE)


@router.get("/p/{arxiv_id:path}")
def reader(arxiv_id: str) -> FileResponse:
    """Deep links land here; the client reads the id and fragment from the URL."""
    return FileResponse(WEB_ROOT / "index.html", headers=_NOCACHE)
