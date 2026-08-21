"""The static single-page app: HTML, JavaScript, stylesheet, and deep links."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
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


_JS_ROOT = (WEB_ROOT / "js").resolve()


@router.get("/js/{name}")
def module(name: str) -> FileResponse:
    """One ES module. `/js/boot.js` is the entry point; it imports the rest by name.

    Served by route rather than a StaticFiles mount so these keep the no-cache header
    above — a mount would let the browser hold a stale module while its neighbours
    reload, which is a worse afternoon than a stale bundle.
    """
    path = (_JS_ROOT / name).resolve()
    # `name` cannot contain a slash, but resolve() is what makes that a guarantee rather
    # than a property of Starlette's path matching.
    if path.parent != _JS_ROOT or path.suffix != ".js" or not path.is_file():
        raise HTTPException(404, f"no module {name}")
    return FileResponse(path, media_type="application/javascript", headers=_NOCACHE)


@router.get("/style.css")
def appcss() -> FileResponse:
    return FileResponse(WEB_ROOT / "style.css", media_type="text/css", headers=_NOCACHE)


@router.get("/p/{arxiv_id:path}")
def reader(arxiv_id: str) -> FileResponse:
    """Deep links land here; the client reads the id and fragment from the URL."""
    return FileResponse(WEB_ROOT / "index.html", headers=_NOCACHE)
