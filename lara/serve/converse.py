"""A model call that keeps its messages and can call tools natively while it does.

`talk()` keeps the real `messages` list in the provider's shape, appends the assistant's
replies and the tool results to it, and sends the whole thing every turn, so a model that
called a tool sees the answer it got. Tool calls are native: the server must be started
with `--enable-auto-tool-choice --tool-call-parser <parser>` (lara's vLLM launcher adds
them; see `serving.vllm.tool_call_parser`) or every request comes back 400.

No JSON schema is imposed on the final answer -- a conversation ends in prose, and forcing
a JSON shape on every turn is what makes a tool-calling model describe a call instead of
making it.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from lara.serve.context import clipped

#: Turns one conversation may take; a reply that calls three tools is still one turn.
MAX_TURNS = 24

#: How much of one tool's output goes back into the conversation. It stays there for every
#: later turn, so `clipped` announces the cut and the model can ask for the rest.
TOOL_RESULT_CHARS = 24_000

#: Seconds one request may take. A long conversation is many of these, not one.
REQUEST_TIMEOUT = 900


@dataclass
class Reply:
    """What one conversation produced, and why it stopped."""

    text: str = ""
    #: The whole exchange, so a caller can continue it in the same conversation.
    messages: list[dict] = field(default_factory=list)
    turns: int = 0
    tool_calls: int = 0
    tools_used: list[str] = field(default_factory=list)
    stopped_because: str = ""
    error: str = ""
    tokens: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def ok(self) -> bool:
        return not self.error

    def to_dict(self) -> dict:
        return {"turns": self.turns, "tool_calls": self.tool_calls,
                "tools_used": list(self.tools_used), "tokens": self.tokens,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "stopped_because": self.stopped_because, "error": self.error}


def _post(base_url: str, payload: dict, *, timeout: float, api_key: str = "") -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", method="POST",
        data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def opening(system: str, user: str) -> list[dict]:
    """The two messages a conversation starts from."""
    out = []
    if system.strip():
        out.append({"role": "system", "content": system})
    out.append({"role": "user", "content": user})
    return out


async def _dispatch(dispatch, name: str, args: dict) -> str:
    got = dispatch(name, args)
    if inspect.isawaitable(got):
        got = await got
    return str(got)


def _tally(out: Reply, got: dict) -> None:
    usage = got.get("usage") or {}
    out.tokens += int(usage.get("total_tokens") or 0)
    out.tokens_in += int(usage.get("prompt_tokens") or 0)
    out.tokens_out += int(usage.get("completion_tokens") or 0)


async def talk(base_url: str, model: str, messages: list[dict], *,
               tools: list[dict] | None = None, dispatch=None,
               max_turns: int = MAX_TURNS, max_tokens: int = 8_000,
               temperature: float = 0.3, timeout: float = REQUEST_TIMEOUT,
               on_call=None, finally_say: str = "", api_key: str = "",
               tool_choice: str | dict = "auto",
               enable_thinking: bool | None = None) -> Reply:
    """Run a conversation to its end. Never raises.

    `messages` is not copied into the reply until the end, so a caller that wants to
    continue passes back `reply.messages`. `dispatch(name, args)` returns the string a tool
    call answers with; anything it raises becomes that string, because a tool that failed
    is a fact the model should see rather than an exception that ends the run.

    `tool_choice` "auto" lets the model answer in plain text instead of calling anything; a
    caller with one tool and no acceptable plain answer can force it. `enable_thinking`
    None leaves the server's default; True/False sends `chat_template_kwargs`, so a call
    that does not need visible reasoning need not pay to generate it.
    """
    out = Reply(messages=list(messages))
    for turn in range(max_turns):
        payload = {"model": model, "messages": out.messages,
                   "max_tokens": max_tokens, "temperature": temperature}
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        try:
            got = await asyncio.to_thread(_post, base_url, payload, timeout=timeout,
                                          api_key=api_key)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:400]
            # The one failure that is a server flag, not anything about the request.
            if "tool" in body and "choice" in body:
                out.error = ("this server was started without tool calling — it needs "
                             f"--enable-auto-tool-choice and --tool-call-parser. {body}")
            else:
                out.error = f"the model refused the request ({exc.code}): {body}"
            return out
        except Exception as exc:                               # noqa: BLE001
            out.error = f"{type(exc).__name__}: {exc}"
            return out

        out.turns = turn + 1
        _tally(out, got)
        choice = (got.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = message.get("tool_calls") or []

        # Appended whole, `tool_calls` included: the next request must carry the call the
        # tool results answer, or the provider rejects the pairing.
        out.messages.append({k: v for k, v in message.items() if v is not None})

        if not calls:
            out.text = str(message.get("content") or "").strip()
            out.stopped_because = str(choice.get("finish_reason") or "stop")
            return out

        if dispatch is None:
            out.error = "the model called a tool and no dispatcher was given"
            return out

        for call in calls:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            out.tool_calls += 1
            if name not in out.tools_used:
                out.tools_used.append(name)
            try:
                answer = await _dispatch(dispatch, name, args)
            except Exception as exc:                           # noqa: BLE001
                answer = f"{type(exc).__name__}: {exc}"
            if on_call is not None:
                with contextlib.suppress(Exception):
                    on_call({"tool": name,
                             "path": str(args.get("path") or args.get("query")
                                         or args.get("pattern") or ""),
                             "why": "", "turn": out.turns, "of": max_turns,
                             "outcome": "failed" if answer.startswith(f"{name} failed:")
                                        else "ok",
                             "detail": f"{len(answer):,} chars"})
            out.messages.append({
                "role": "tool", "tool_call_id": call.get("id") or name, "name": name,
                "content": clipped(answer, TOOL_RESULT_CHARS, what=f"the {name} result")})

    # A model with tools explores until stopped and then has said nothing, so a turn cap
    # without a closing turn throws away the work it just paid for. The last word is a turn
    # of its own with the tools taken away.
    out.stopped_because = f"reached the {max_turns}-turn limit"
    if not finally_say:
        return out
    out.messages.append({"role": "user", "content": finally_say})
    final = {"model": model, "messages": out.messages, "max_tokens": max_tokens,
             "temperature": temperature}
    if enable_thinking is not None:
        final["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    try:
        got = await asyncio.to_thread(_post, base_url, final, timeout=timeout,
                                      api_key=api_key)
    except Exception as exc:                                   # noqa: BLE001
        out.error = f"{type(exc).__name__}: {exc}"
        return out
    out.turns += 1
    _tally(out, got)
    message = (got.get("choices") or [{}])[0].get("message") or {}
    out.messages.append({k: v for k, v in message.items() if v is not None})
    out.text = str(message.get("content") or "").strip()
    out.stopped_because = f"reached the {max_turns}-turn limit, then asked to answer"
    return out
