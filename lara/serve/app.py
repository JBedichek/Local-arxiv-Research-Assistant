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

import os

from fastapi import FastAPI

from lara.serve import deps
from lara.serve.routes import ROUTERS
from lara.serve.state import AppState

app = FastAPI(title="Local arXiv Research Assistant", docs_url="/api/docs")

# Installed at import time from the environment rather than from config, because uvicorn
# imports this module in its own process and the CLI's parsed config does not travel with
# it. `lara serve` exports LARA_TOKEN after deciding whether one is required; anything
# that imports the app directly — a test, an embedded runner — gets the same protection by
# setting the same variable, and no protection if it sets nothing and serves loopback.
_auth_token = os.environ.get("LARA_TOKEN", "").strip()
if _auth_token:
    from lara.serve.auth import TokenAuthMiddleware

    app.add_middleware(TokenAuthMiddleware, token=_auth_token)


@app.on_event("startup")
def _startup() -> None:
    deps.set_state(AppState(os.environ.get("LARA_CONFIG"),
                            load_models=os.environ.get("LARA_NO_MODELS") != "1"))


for _router in ROUTERS:
    app.include_router(_router)
