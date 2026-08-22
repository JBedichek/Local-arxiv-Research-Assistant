"""The reader application: construction only (R3, R4, R6, R8, R9).

The endpoints live in :mod:`lara.serve.routes`, one module per area of the app, and the
state they share lives in :mod:`lara.serve.deps`. This file builds the application,
installs the middleware, loads the state once at startup and mounts the routers — which is
all anyone should have to read to find out how the server is wired.

**Endpoints are deliberately `def`, not `async def`.** Retrieval touches the GPU and
SQLite, both synchronous and both holding the GIL for milliseconds at a time. Declared
`async def`, that work runs *on the event loop* and stalls every other in-flight request —
which is what makes an otherwise fast server feel unresponsive the moment two people use
it. Starlette runs sync endpoints in a threadpool instead, so a slow retrieval blocks only
its own request. The genuinely async endpoints are the SSE streams, which are I/O-bound
and yield between tokens.
"""

from __future__ import annotations

import json
import os

from fastapi import FastAPI

from lara.serve import deps
from lara.serve.routes import ROUTERS
from lara.serve.state import AppState

app = FastAPI(title="Local arXiv Research Assistant", docs_url="/api/docs")

# `lara serve` exports LARA_TOKENS as JSON when more than one person has a secret, and
# LARA_TOKEN for the single-secret case. Read from the environment rather than the parsed
# config because uvicorn imports this module in its own process, where the CLI's config
# does not travel. Anything that imports the app directly — a test, an embedded runner —
# gets the same protection by setting the same variables, and no protection if it sets
# nothing and serves loopback.
_auth_tokens: dict[str, str] = {}
_env_many = os.environ.get("LARA_TOKENS", "").strip()
if _env_many:
    try:
        _auth_tokens = {k: v for k, v in json.loads(_env_many).items() if v}
    except Exception:
        _auth_tokens = {}
_env_one = os.environ.get("LARA_TOKEN", "").strip()
if _env_one:
    _auth_tokens.setdefault("env", _env_one)
if _auth_tokens:
    from lara.serve.auth import TokenAuthMiddleware

    app.add_middleware(TokenAuthMiddleware, token=_auth_tokens)


@app.on_event("startup")
def _startup() -> None:
    deps.set_state(AppState(os.environ.get("LARA_CONFIG"),
                            load_models=os.environ.get("LARA_NO_MODELS") != "1"))


for _router in ROUTERS:
    app.include_router(_router)
