"""A model call that remembers the last one.

Every other model call in this system is single-shot, and where a loop was needed it was
hand-rolled on top of that. Both `reading.browse` and `coding.write_program` accumulate a
*description* of what happened and re-render it every turn. This keeps the turns.
"""

from __future__ import annotations

import asyncio
import json

from lara.serve import converse as C


def _tools():
    return [{"type": "function", "function": {
        "name": "read", "description": "read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}}]


def _reply(content=None, calls=None, finish="stop", usage=None):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {"id": f"c{i}", "type": "function",
             "function": {"name": n, "arguments": json.dumps(a)}}
            for i, (n, a) in enumerate(calls)]
    return {"choices": [{"message": message, "finish_reason": finish}],
            "usage": usage if usage is not None else {"total_tokens": 10}}


def _server(monkeypatch, replies):
    """A provider that answers from a list, and records what it was sent."""
    sent = []

    def post(base_url, payload, *, timeout, api_key=""):
        # Snapshotted: `talk` sends the live `messages` list, so a recorded payload keeps
        # mutating as the conversation grows and cannot be inspected afterwards.
        sent.append({**payload, "messages": [dict(m) for m in payload["messages"]]})
        return replies[min(len(sent) - 1, len(replies) - 1)]

    monkeypatch.setattr(C, "_post", post)
    return sent


# ── _post itself: the Authorization header a replica requiring one needs ──────────

def test_an_api_key_becomes_a_bearer_header(monkeypatch):
    """A replica that requires one refuses every turn otherwise -- silently, from
    `talk`'s point of view, since it never raises and just reports no questions."""
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(_reply(content="ok")).encode()

    def fake_urlopen(req, timeout):
        captured["headers"] = dict(req.headers)
        return _Resp()

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)
    C._post("http://x/v1", {"model": "m", "messages": []}, timeout=1.0, api_key="secret")
    assert captured["headers"].get("Authorization") == "Bearer secret"


def test_no_api_key_means_no_authorization_header(monkeypatch):
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(_reply(content="ok")).encode()

    def fake_urlopen(req, timeout):
        captured["headers"] = dict(req.headers)
        return _Resp()

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)
    C._post("http://x/v1", {"model": "m", "messages": []}, timeout=1.0)
    assert "Authorization" not in captured["headers"]


def test_the_whole_exchange_is_sent_every_turn(monkeypatch):
    """The point of the module. A model that called a tool and got an answer must be able
    to see that it did, rather than be told about it in a re-rendered prompt."""
    sent = _server(monkeypatch, [
        _reply(calls=[("read", {"path": "a.py"})]),
        _reply(content="a.py defines f()."),
    ])
    got = asyncio.run(C.talk("http://x/v1", "m", C.opening("sys", "what is in a.py?"),
                             tools=_tools(),
                             dispatch=lambda n, a: "def f(): ..."))
    assert got.ok and got.text == "a.py defines f()."
    # Second request carries: system, user, the assistant's tool call, and its result.
    roles = [m["role"] for m in sent[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert sent[1]["messages"][-1]["content"] == "def f(): ..."
    # And the caller gets the transcript back, so a later turn can continue it.
    assert [m["role"] for m in got.messages] == ["system", "user", "assistant", "tool",
                                                 "assistant"]


def test_the_assistant_turn_keeps_its_tool_calls(monkeypatch):
    """The next request has to carry the call the tool results answer, or the provider
    rejects the pairing."""
    _server(monkeypatch, [_reply(calls=[("read", {"path": "a.py"})]), _reply(content="ok")])
    got = asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u"),
                             tools=_tools(), dispatch=lambda n, a: "x"))
    assistant = next(m for m in got.messages if m["role"] == "assistant" and m.get("tool_calls"))
    assert assistant["tool_calls"][0]["function"]["name"] == "read"
    tool = next(m for m in got.messages if m["role"] == "tool")
    assert tool["tool_call_id"] == assistant["tool_calls"][0]["id"]


# ── usage accounting: prompt_tokens/completion_tokens alongside total_tokens ──────

def test_usage_populates_tokens_in_and_tokens_out(monkeypatch):
    """`usage.prompt_tokens`/`completion_tokens` accumulate into `Reply.tokens_in`/
    `tokens_out`, alongside the existing `total_tokens` -> `tokens` accounting."""
    _server(monkeypatch, [
        _reply(calls=[("read", {"path": "a"})],
              usage={"total_tokens": 30, "prompt_tokens": 20, "completion_tokens": 10}),
        _reply(content="done",
              usage={"total_tokens": 15, "prompt_tokens": 9, "completion_tokens": 6}),
    ])
    got = asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u"),
                             tools=_tools(), dispatch=lambda n, a: "x"))
    assert got.tokens == 45
    assert got.tokens_in == 29
    assert got.tokens_out == 16
    assert got.to_dict()["tokens_in"] == 29 and got.to_dict()["tokens_out"] == 16


def test_missing_usage_fields_count_as_zero(monkeypatch):
    """A provider that omits `prompt_tokens`/`completion_tokens` must not raise -- same
    "or 0" tolerance the existing `total_tokens` accounting already has."""
    _server(monkeypatch, [_reply(content="ok", usage={})])
    got = asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u")))
    assert got.tokens_in == 0 and got.tokens_out == 0


def test_the_closing_turns_usage_also_accumulates(monkeypatch):
    """The `finally_say` request is a second, separate payload/response lower in `talk`
    -- its usage has to add in too, not just the per-turn loop above."""
    _server(monkeypatch, [
        _reply(calls=[("read", {"path": "a"})],
              usage={"total_tokens": 10, "prompt_tokens": 7, "completion_tokens": 3}),
        _reply(content="here is the answer",
              usage={"total_tokens": 8, "prompt_tokens": 5, "completion_tokens": 3}),
    ])
    got = asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u"),
                             tools=_tools(), dispatch=lambda n, a: "x",
                             max_turns=1, finally_say="Answer now."))
    assert got.tokens_in == 12 and got.tokens_out == 6


# ── enable_thinking: every existing caller must see today's exact payload ─────────

def test_by_default_no_chat_template_kwargs_is_sent(monkeypatch):
    """`enable_thinking` defaults to None, and `talk` must not add the key at all in
    that case -- every caller that predates this parameter has to see the identical
    payload it always sent, or a replica that rejects an unknown key breaks them."""
    sent = _server(monkeypatch, [_reply(content="ok")])
    asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u")))
    assert "chat_template_kwargs" not in sent[0]


def test_enable_thinking_false_is_sent_in_the_shape_stream_answer_uses(monkeypatch):
    """Same key and value shape as `lara.serve.generate.stream_answer`'s known-working
    call, since this is the same server/model being asked to skip the same thing."""
    sent = _server(monkeypatch, [_reply(content="ok")])
    asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u"), enable_thinking=False))
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_enable_thinking_true_is_also_sent_explicitly(monkeypatch):
    """True is a real, explicit choice too -- distinct from the default of not asking."""
    sent = _server(monkeypatch, [_reply(content="ok")])
    asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u"), enable_thinking=True))
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": True}


def test_a_tool_that_raises_becomes_something_the_model_can_read(monkeypatch):
    """A tool that failed is a fact, not an outage. The model can read it and try something
    else, which is the whole reason this loop exists."""
    _server(monkeypatch, [_reply(calls=[("read", {"path": "nope"})]),
                          _reply(content="that path does not exist")])

    def boom(name, args):
        raise FileNotFoundError("nope")

    got = asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u"),
                             tools=_tools(), dispatch=boom))
    assert got.ok
    assert "FileNotFoundError" in next(m for m in got.messages if m["role"] == "tool")["content"]


def test_a_long_tool_result_is_clipped_and_says_so(monkeypatch):
    """It stays in the conversation for every later turn, so this cost is paid repeatedly
    — unlike a prompt block, which is rendered once."""
    _server(monkeypatch, [_reply(calls=[("read", {"path": "big"})]), _reply(content="ok")])
    got = asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u"),
                             tools=_tools(),
                             dispatch=lambda n, a: "x" * (C.TOOL_RESULT_CHARS * 3)))
    body = next(m for m in got.messages if m["role"] == "tool")["content"]
    assert len(body) < C.TOOL_RESULT_CHARS * 2 and "cut here" in body


def test_a_replica_without_tool_calling_says_which_flags_it_needs(monkeypatch):
    """The one failure worth naming precisely, because it is a server flag and nothing
    about the request the caller wrote. Only one of this machine's three replicas has it."""
    import urllib.error

    def refuse(base_url, payload, *, timeout, api_key=""):
        raise urllib.error.HTTPError(
            base_url, 400, "Bad Request", {},
            __import__("io").BytesIO(
                b'"auto" tool choice requires --enable-auto-tool-choice and '
                b'--tool-call-parser to be set'))

    monkeypatch.setattr(C, "_post", refuse)
    got = asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u"),
                             tools=_tools(), dispatch=lambda n, a: ""))
    assert not got.ok
    assert "--enable-auto-tool-choice" in got.error and "--tool-call-parser" in got.error


def test_a_conversation_with_no_tools_is_still_a_conversation(monkeypatch):
    """The difference between a planner that can be argued with and one that must be
    re-prompted from scratch."""
    sent = _server(monkeypatch, [_reply(content="here is the plan")])
    got = asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u")))
    assert got.ok and got.text == "here is the plan"
    assert "tools" not in sent[0], "a tool-less conversation must not send a tool surface"


def test_it_stops_at_the_turn_limit_and_says_so(monkeypatch):
    _server(monkeypatch, [_reply(calls=[("read", {"path": "a"})])])
    got = asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u"),
                             tools=_tools(),
                             dispatch=lambda n, a: "x", max_turns=3))
    assert got.turns == 3 and "3-turn limit" in got.stopped_because






def test_a_tool_conversation_gets_a_last_word(monkeypatch):
    """A model with tools explores until it is stopped, and then has said nothing.

    Measured: the planner given the read-only surface spent all ten turns on `map`,
    `search`, `read`, `outline` and `grep` — it found the right files — and returned no
    plan at all, because nothing told it the exploring was over. A turn cap without a
    closing turn throws away the work it just paid for.
    """
    sent = _server(monkeypatch, [
        _reply(calls=[("read", {"path": "a"})]),   # every turn calls a tool
        _reply(calls=[("read", {"path": "b"})]),
        _reply(content="here is the answer"),      # the closing turn, tools withheld
    ])
    got = asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u"),
                             tools=_tools(), dispatch=lambda n, a: "x",
                             max_turns=2, finally_say="You are out of turns. Answer now."))
    assert got.text == "here is the answer"
    assert "then asked to answer" in got.stopped_because
    # Nothing left to call, so the only thing to do is answer.
    assert "tools" not in sent[-1], "the closing turn still offered tools"
    assert sent[-1]["messages"][-1]["content"].startswith("You are out of turns")


def test_the_closing_turn_also_honors_enable_thinking(monkeypatch):
    """The `finally_say` request is a second, separate payload built lower down in
    `talk` -- it has to carry `enable_thinking` too, not just the per-turn loop above."""
    sent = _server(monkeypatch, [
        _reply(calls=[("read", {"path": "a"})]),
        _reply(content="here is the answer"),
    ])
    asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u"),
                       tools=_tools(), dispatch=lambda n, a: "x",
                       max_turns=1, finally_say="Answer now.", enable_thinking=False))
    assert sent[-1]["chat_template_kwargs"] == {"enable_thinking": False}


def test_without_a_last_word_the_turn_limit_still_just_stops(monkeypatch):
    _server(monkeypatch, [_reply(calls=[("read", {"path": "a"})])])
    got = asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u"),
                             tools=_tools(),
                             dispatch=lambda n, a: "x", max_turns=2))
    assert got.text == "" and got.stopped_because == "reached the 2-turn limit"


def test_a_tool_call_is_reported_in_the_shape_the_activity_log_already_uses(monkeypatch):
    """It emitted `{tool, args, chars, turn}` and the log rendered `[5/undefined] read`.

    `tool.used` is an existing event with an existing shape — `tool`, `path`, `why`,
    `turn`, `of`, `outcome`, `detail` — and the renderer reads all of them. An event named
    after an existing event has to *be* one, or it is an impostor that looks right until
    somebody reads the log.
    """
    _server(monkeypatch, [_reply(calls=[("read", {"path": "a.py"})]), _reply(content="ok")])
    seen = []
    asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u"),
                       tools=_tools(), dispatch=lambda n, a: "body",
                       max_turns=9, on_call=seen.append))
    got = seen[0]
    # Every field the renderer touches.
    for field in ("tool", "path", "why", "turn", "of", "outcome", "detail"):
        assert field in got, f"`tool.used` rendered without {field}"
    assert got["of"] == 9 and got["turn"] == 1
    assert got["path"] == "a.py" and got["outcome"] == "ok"


def test_a_call_without_a_path_still_names_what_it_asked_for(monkeypatch):
    """`search` and `probe` take no path; the renderer falls back to the tool name, and an
    empty `path` there is correct rather than missing."""
    _server(monkeypatch, [_reply(calls=[("search", {"query": "where the gate waits"})]),
                          _reply(content="ok")])
    seen = []
    asyncio.run(C.talk("http://x/v1", "m", C.opening("s", "u"),
                       tools=_tools(), dispatch=lambda n, a: "hits",
                       on_call=seen.append))
    assert seen[0]["path"] == "where the gate waits"




# ── research: the literature tool on a conversation's surface ──────────────────











