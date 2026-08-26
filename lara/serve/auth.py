"""Authentication for the reader, and the rule that stops it being forgotten.

Every endpoint was reachable by anyone who could route to the host, and the surface is
not merely "read my papers": ``/api/model/download`` pulls arbitrary Hugging Face repos
onto the disk using the operator's ``HF_TOKEN``, ``/api/ask`` is free inference on their
GPU, ``/api/fetch`` makes their machine crawl arXiv, and the memory and taste endpoints
delete their library. On a home LAN that was a documented trade. On a café network, a
conference network, or a tunnel to the internet it is not a trade, it is a mistake.

**The design decision that matters is fail-closed.** A token is *required* whenever the
server binds to anything other than loopback, and `lara serve --host 0.0.0.0` without one
refuses to start rather than coming up unprotected. Security that depends on remembering
a flag is security that lasts until the first hurried evening.

Three ways to present the token, because three different callers need it:

``Authorization: Bearer``  scripts and ``curl``
``?token=``                the first click of a link you sent yourself
``lara_auth`` cookie       every request after that, so the token leaves the URL

The query parameter is accepted once and immediately traded for a cookie via a redirect,
so the secret does not linger in browser history, in the Referer header of any outbound
link, or in the server log of whatever the reader clicks next.

Comparison is constant-time. The margin is theoretical over a LAN and free to obtain.
"""

from __future__ import annotations

import hmac
import os
import secrets
from urllib.parse import urlencode, urlsplit, urlunsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

COOKIE = "lara_auth"

#: Reachable without a token: the health check, so a tunnel or a load balancer can prove
#: the process is alive without holding a credential, and the login page itself.
PUBLIC_PATHS = frozenset({"/healthz", "/login"})

LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost", ""})


def is_loopback(host: str | None) -> bool:
    return (host or "").strip() in LOOPBACK


def generate_token() -> str:
    """A token worth having: 32 bytes of urandom, URL-safe."""
    return secrets.token_urlsafe(32)


def resolve_tokens(cfg) -> dict[str, str]:
    """Every accepted secret, as {label: token}.

    One secret per person rather than one shared secret, because a shared one cannot be
    taken back from a single reader: revoking it locks everyone out and means telling
    everybody a new one. With a token each, removing a line removes exactly that person,
    and the access log says who was using the reader.

    `$LARA_TOKEN` still works and is labelled `env`, so a single-user setup needs no
    config at all and nothing needs writing to disk.
    """
    out: dict[str, str] = {}
    env = os.environ.get("LARA_TOKEN")
    if env and env.strip():
        out["env"] = env.strip()
    try:
        single = cfg.get_in("serving.auth.token")
    except Exception:
        single = None
    if isinstance(single, str) and single.strip():
        out.setdefault("default", single.strip())
    try:
        many = cfg.get_in("serving.auth.tokens")
    except Exception:
        many = None
    if isinstance(many, dict):
        for label, tok in many.items():
            if isinstance(tok, str) and tok.strip():
                out[str(label)] = tok.strip()
    return out


def resolve_token(cfg) -> str | None:
    """First configured token, for callers that only need to know whether one exists."""
    toks = resolve_tokens(cfg)
    return next(iter(toks.values()), None)


def require_token_for(host: str, cfg) -> bool:
    """Whether this bind address demands a token.

    ``serving.auth.mode`` is ``auto`` by default: off on loopback, mandatory everywhere
    else. ``always`` demands one even on loopback, for a shared machine. ``off`` disables
    the check entirely and exists so that someone who genuinely wants the old behaviour
    has to say so in writing.
    """
    raw = "auto"
    try:
        raw = cfg.get_in("serving.auth.mode")
    except Exception:
        raw = "auto"
    # YAML 1.1 turns a bare `off` into the boolean False (and `on`/`yes`/`no` likewise).
    # Reading that as "unset" and falling back to `auto` would leave authentication ON
    # while the config file plainly says off — failing *closed*, but silently and against
    # the operator's stated intent. Both spellings are honoured.
    if raw is False:
        return False
    if raw is True:
        return True
    mode = (raw or "auto").strip().lower() if isinstance(raw, str) else "auto"
    if mode == "off":
        return False
    if mode == "always":
        return True
    return not is_loopback(host)


def tokens_for_app(env: dict | None = None, cfg=None) -> dict[str, str]:
    """Every secret the served app should accept, from the environment or the config.

    **The app must not depend on the CLI having run.** `lara.serve.app` read `LARA_TOKEN`
    and `LARA_TOKENS` and nothing else, and `lara serve` was the only thing that put a
    config-declared token into either. For the whole time `serve` was missing its
    `@app.command()` decorator the sole way to start the reader was
    `uvicorn lara.serve.app:app`, which exported nothing — so a config that said
    `serving.auth.mode: always` with a token under `serving.auth.tokens` served every
    endpoint unauthenticated. Not "asked for a token it did not have": no middleware was
    installed at all, and the reader looked entirely normal.

    The environment wins when it says anything, because `lara serve` has already applied
    the policy by the time it exports these, and a caller that sets them by hand means
    them. Reading the config underneath would either duplicate that decision or
    contradict it.

    Falling back to the config applies the same rule `lara serve` does — `serving.host`
    is the bind this configuration describes, and `require_token_for` decides — so plain
    loopback under the default `auto` mode is unchanged and a token left in
    `config.local.yaml` by `lara setup` still does not start demanding a login on
    127.0.0.1.
    """
    env = os.environ if env is None else env
    out: dict[str, str] = {}
    many = (env.get("LARA_TOKENS") or "").strip()
    if many:
        try:
            import json

            out = {str(k): v for k, v in json.loads(many).items()
                   if isinstance(v, str) and v.strip()}
        except Exception:                                  # noqa: BLE001
            out = {}
    one = (env.get("LARA_TOKEN") or "").strip()
    if one:
        out.setdefault("env", one)
    if out or cfg is None:
        return out
    host = cfg.get_in("serving.host", "127.0.0.1") or "127.0.0.1"
    if not require_token_for(str(host), cfg):
        return {}
    return resolve_tokens(cfg)


class TokenAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token: str | dict[str, str]) -> None:
        super().__init__(app)
        self.tokens = {"default": token} if isinstance(token, str) else dict(token)

    def _who(self, presented: str | None) -> str | None:
        """The label whose token matches, or None.

        Every candidate is compared even after a match, so the time taken does not depend
        on which token was presented or how many are configured.
        """
        if not presented:
            return None
        got = presented.encode("utf-8")
        found = None
        for label, tok in self.tokens.items():
            if hmac.compare_digest(got, tok.encode("utf-8")):
                found = label
        return found

    def _ok(self, presented: str | None) -> bool:
        return self._who(presented) is not None

    async def dispatch(self, request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        if self._ok(request.cookies.get(COOKIE)):
            return await call_next(request)

        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer ") and self._ok(header[7:].strip()):
            return await call_next(request)

        q = request.query_params.get("token")
        who = self._who(q)
        if who is not None:
            # Trade the query parameter for a cookie and drop it from the URL, so the
            # token does not survive in history or leak through Referer.
            parts = urlsplit(str(request.url))
            kept = [(k, v) for k, v in request.query_params.multi_items() if k != "token"]
            clean = urlunsplit((parts.scheme, parts.netloc, parts.path,
                                urlencode(kept), parts.fragment))
            resp = RedirectResponse(clean, status_code=303)
            resp.set_cookie(COOKIE, self.tokens[who], httponly=True, samesite="lax",
                            max_age=60 * 60 * 24 * 365,
                            secure=request.url.scheme == "https")
            return resp

        if "text/html" in request.headers.get("accept", ""):
            return HTMLResponse(_LOGIN_HTML, status_code=401)
        return JSONResponse({"error": "authentication required"}, status_code=401)


_LOGIN_HTML = """<!doctype html><meta charset=utf-8>
<title>lara — sign in</title>
<style>
 body{font:15px/1.55 system-ui,sans-serif;background:#0f1216;color:#e8ecf4;
      display:grid;place-items:center;height:100vh;margin:0}
 form{background:#171b21;border:1px solid #262c35;border-radius:10px;padding:26px 28px;
      width:min(92vw,380px)}
 h1{font-size:15px;margin:0 0 4px;letter-spacing:.08em;text-transform:uppercase;color:#8b95a5}
 p{color:#8b95a5;font-size:13px;margin:0 0 16px}
 input{width:100%;box-sizing:border-box;padding:9px 11px;border-radius:7px;
       border:1px solid #262c35;background:#0f1216;color:#e8ecf4;font:inherit}
 button{margin-top:12px;width:100%;padding:9px;border:0;border-radius:7px;
        background:#4aa8ff;color:#08111f;font:inherit;font-weight:600;cursor:pointer}
</style>
<form method=get action="/">
  <h1>lara</h1>
  <p>This reader is password protected. Paste the access token to continue.</p>
  <input name=token type=password autofocus autocomplete=current-password placeholder="access token">
  <button type=submit>Sign in</button>
</form>"""
