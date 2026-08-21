# Authentication

The reader's bearer-token auth, the rule that stops it being forgotten, and the YAML trap
that nearly inverted it.

Implementation: [`lara/serve/auth.py`](../../lara/serve/auth.py). Installed in
[`lara/serve/app.py`](../../lara/serve/app.py); the fail-closed check lives in
`serve()` in [`lara/cli.py`](../../lara/cli.py).

---

## 1. What is actually exposed

Every endpoint used to be reachable by anyone who could route to the host, and the surface
is not merely "read my papers":

| endpoint | what it gives away |
|---|---|
| `POST /api/model/download` | pulls arbitrary Hugging Face repos onto the disk using the operator's `HF_TOKEN` |
| `POST /api/ask`, `/api/synthesize` | free inference on their GPU |
| `POST /api/fetch/{id}` | makes their machine crawl arXiv |
| `DELETE /api/memory/*`, `/api/taste/*` | deletes their library |

On a home LAN that was a documented trade. On a café network, a conference network, or a
tunnel to the internet it is not a trade, it is a mistake.

## 2. Fail-closed is the design decision

A token is **required** whenever the server binds to anything other than loopback, and
`lara serve --host 0.0.0.0` without one **refuses to start**:

```
refusing to serve on 0.0.0.0 without authentication.

Every endpoint is reachable by anyone who can route here, including model downloads that
use your HF_TOKEN, generation on your GPU, and the endpoints that delete your library.

Set a token and try again:

  export LARA_TOKEN=<32 fresh urandom bytes, printed for you>

or put it in config.local.yaml under serving.auth.token, then open
  http://0.0.0.0:8080/?token=$LARA_TOKEN

To bind loopback-only instead, drop --host. To disable this check deliberately, set
serving.auth.mode: off
```

> Security that depends on remembering a flag is security that lasts until the first
> hurried evening.

The token it offers is `secrets.token_urlsafe(32)` — 32 bytes of urandom, URL-safe —
generated fresh so there is no excuse to invent a weak one.

## 3. Three modes

`require_token_for(host, cfg)` decides whether *this bind address* demands a token.

| `serving.auth.mode` | behaviour |
|---|---|
| `auto` *(default)* | off on loopback, mandatory everywhere else |
| `always` | demanded even on loopback — for a shared machine |
| `off` | the check is disabled entirely |

> `off` exists so that someone who genuinely wants the old behaviour has to say so in
> writing.

Loopback is `127.0.0.1`, `::1`, `localhost`, or empty.

## 4. The YAML `off` trap

YAML 1.1 turns a bare `off` into the boolean `False` — and `on`/`yes`/`no` likewise. So
this:

```yaml
serving:
  auth:
    mode: off        # ← parses as False, not "off"
```

reaches `require_token_for` as `False`, not the string `"off"`.

Reading that as "unset" and falling back to `auto` would leave authentication **on** while
the config file plainly says off — failing *closed*, but silently and against the operator's
stated intent. Both spellings are honoured:

```python
if raw is False:  return False
if raw is True:   return True
mode = (raw or "auto").strip().lower() if isinstance(raw, str) else "auto"
```

The same trap is handled from the other direction in `lara config set`: `serving.auth.mode`
is in the enum table, and enum-valued keys **bypass YAML parsing and stay strings**. So
`lara config set serving.auth.mode off` writes `mode: 'off'` — quoted — rather than
`mode: false`. Write it by hand and you will get the unquoted form; both work, but only the
quoted one survives a round trip legibly.

## 5. Three ways to present the token

Because three different callers need it.

| method | for |
|---|---|
| `Authorization: Bearer <token>` | scripts and `curl` |
| `?token=<token>` | the first click of a link you sent yourself |
| `lara_auth` cookie | every request after that |

The query parameter is accepted **once and immediately traded for a cookie** via a 303
redirect to the same URL with `token` stripped from the query string. The secret therefore
does not linger in browser history, in the `Referer` header of any outbound link, or in the
server log of whatever the reader clicks next.

The cookie is `httponly`, `samesite=lax`, one year `max_age`, and `secure` only when the
request arrived over https.

Comparison is `hmac.compare_digest` on UTF-8 bytes — constant-time, and on bytes so a
unicode token cannot raise. The timing margin is theoretical over a LAN and free to obtain.

## 6. Where the token comes from

```python
resolve_token(cfg):
    LARA_TOKEN environment variable   # wins
    serving.auth.token in the config  # fallback
```

Environment first, so it need never be written to disk.

## 7. How the middleware is installed — and the trap in it

`TokenAuthMiddleware` is added in `lara/serve/app.py` **at import time, from the
environment**, not from the config:

```python
_auth_token = os.environ.get("LARA_TOKEN", "").strip()
if _auth_token:
    app.add_middleware(TokenAuthMiddleware, token=_auth_token)
```

The reason is that uvicorn imports the module in its own process and the CLI's parsed config
does not travel with it. Anything else that imports the app directly — a test, an embedded
runner — gets the same protection by setting the same variable.

**The consequence is worth stating plainly: running `uvicorn lara.serve.app:app` yourself,
without `LARA_TOKEN` in the environment, gives you no authentication no matter what
`serving.auth.mode` says.** Only `lara serve` performs the fail-closed check and exports the
variable. Use `lara serve` for anything reachable off loopback.

## 8. A token is a credential, not a switch

`lara serve` turns auth on only where the bind actually demands it:

```python
if token and AUTH.require_token_for(host, cfg):
    os.environ["LARA_TOKEN"] = token
else:
    os.environ.pop("LARA_TOKEN", None)      # a stale env var must not re-enable it
```

Keying off "a token exists" alone meant a token left in `config.local.yaml` — one that
`lara setup` carries forward — forced a login prompt on plain loopback, where the mode says
none is needed. The explicit `pop` matters too: a `LARA_TOKEN` left over in the shell must
not silently re-enable auth that the mode has turned off.

When auth is on, `lara serve` prints:

```
authentication on — open http://HOST:PORT/?token=<your token> once per browser
```

and the ready-announcer appends `/?token=<your token>` to the URL it prints. When binding
off loopback with `mode: off`, it prints a yellow *"serving without authentication"* instead
of saying nothing.

## 9. Public paths

```python
PUBLIC_PATHS = frozenset({"/healthz", "/login"})
```

`GET /healthz` returns `{"ok": true}` and is reachable without a token so a tunnel or a load
balancer can prove the process is alive without holding a credential.

`/login` is reserved but **has no route** — requesting it returns 404, verified against the
running server. The login form is not served from a path: any unauthenticated request whose
`Accept` header contains `text/html` gets the form inline as the body of a **401**, with a
password field posting `token` back to `/`. Anything else gets
`{"error": "authentication required"}`, also 401.

## 10. Configuration

```yaml
serving:
  auth:
    mode: auto        # auto | always | off
    token: null
```

```bash
export LARA_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
lara serve --host 0.0.0.0

# or persist it
lara config set serving.auth.token "$LARA_TOKEN"
lara config set serving.auth.mode always      # enum-validated
```

```bash
curl -H "Authorization: Bearer $LARA_TOKEN" http://host:8080/api/health
```

## 11. Things worth knowing

- **There is one token, and it is not per-user.** Anyone holding it has every capability
  the operator has, including model downloads on the operator's HF account.
- **There is no logout, no rotation and no expiry** beyond the cookie's one-year `max_age`.
  Rotating means changing the token and restarting.
- **Auth is all-or-nothing.** There is no read-only mode: a holder of the token can delete
  the library and start crawls.
- **`/api/docs` is FastAPI's interactive docs and is *not* public.** It sits behind the
  middleware like everything else.
- **The tracked default is `auto`; a local override can silently be `off`.** The checkout
  these docs were written in has `mode: "off"` in `config.local.yaml`, which is exactly
  where a per-machine decision belongs — but it also disables the fail-closed refusal.
  Check `lara config get serving.auth.mode` before exposing a host.
- **HTTPS is not provided.** Over plain http the token crosses the network in a header or a
  cookie. On a LAN that is the accepted trade; over anything wider, terminate TLS in front
  of it.
