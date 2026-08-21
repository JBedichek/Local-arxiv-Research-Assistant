"""The one thing every route module needs: the loaded :class:`AppState`.

Kept apart from :mod:`lara.serve.app` so that the routers can depend on the state without
depending on the application object — importing the app from a router, while the app
imports the routers, is a cycle. Nothing else belongs here: a helper used by one router
lives in that router.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import HTTPException

from lara.serve.state import AppState

_state: AppState | None = None


def set_state(s: AppState | None) -> None:
    """Install the state built at startup. Called once, by the app's startup hook."""
    global _state
    _state = s


def current_state() -> AppState | None:
    """The state as it is, warming up or not — for endpoints that report readiness."""
    return _state


def require_state() -> AppState:
    """The state, or 503 while it is still loading."""
    if _state is None or not _state.ready:
        raise HTTPException(503, "still warming up")
    # Every data endpoint goes through here, which makes it the one place that knows the
    # server is not idle. The cache reaper reads this so it never releases device memory
    # out from under a burst of queries.
    _state.last_query_at = time.time()
    return _state


def memory_root() -> Path:
    """Where the library lives. Created on demand so a fresh install needs no setup step."""
    root = require_state().cfg.get_path("paths.memory")
    root.mkdir(parents=True, exist_ok=True)
    return root
