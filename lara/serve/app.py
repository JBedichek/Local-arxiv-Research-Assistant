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
from lara.serve.auth import tokens_for_app
from lara.serve.routes import ROUTERS
from lara.serve.state import AppState

app = FastAPI(title="Local arXiv Research Assistant", docs_url="/api/docs")


def _config_or_none():
    """The configuration this process was started against, or None if there is none.

    Never fatal at import: a reader that cannot be imported because a config file is
    missing or malformed is a worse failure than the one this exists to fix, and the
    startup hook below builds `AppState` from the same path and will say so properly.
    """
    try:
        from lara import config as config_mod

        return config_mod.load(os.environ.get("LARA_CONFIG"))
    except Exception:                                      # noqa: BLE001
        return None


# `lara serve` exports LARA_TOKENS as JSON when more than one person has a secret, and
# LARA_TOKEN for the single-secret case — and those still win, because by the time it
# exports them the policy has been applied and a caller who sets them by hand means them.
#
# The config is read underneath, and that half is new. This used to read the environment
# and nothing else, on the reasoning that uvicorn imports the module in its own process
# where the CLI's config does not travel. True, and it made the reader's protection
# depend on `lara serve` having been the thing that started it — which for a while it
# could not be, because `serve` had lost its `@app.command()` decorator. A reader started
# with plain uvicorn against a config saying `auth.mode: always` installed no middleware
# at all. `tokens_for_app` applies the same `require_token_for` rule the CLI does, so
# loopback under the default `auto` mode is unchanged.
_auth_tokens: dict[str, str] = tokens_for_app(dict(os.environ), _config_or_none())
if _auth_tokens:
    from lara.serve.auth import TokenAuthMiddleware

    app.add_middleware(TokenAuthMiddleware, token=_auth_tokens)


@app.on_event("startup")
def _startup() -> None:
    deps.set_state(AppState(os.environ.get("LARA_CONFIG"),
                            load_models=os.environ.get("LARA_NO_MODELS") != "1"))


for _router in ROUTERS:
    app.include_router(_router)
