"""Tests for the /api/generation/rate route -- wiring current_state's config into
lara.serve.generation_rate.rate, not the scraping logic itself (see test_generation_rate.py)."""
from __future__ import annotations

import asyncio
import json
import types

from lara.serve import generation_rate as GR
from lara.serve.routes import health as H


def run(c):
    return asyncio.run(c)


def body(resp):
    return json.loads(resp.body)


def _state(base_url="http://127.0.0.1:8000/v1"):
    cfg = types.SimpleNamespace(get_in=lambda key: {"base_url": base_url} if key == "serving.vllm" else {})
    return types.SimpleNamespace(cfg=cfg)


def test_no_server_state_yet_is_unreachable_not_an_error(monkeypatch):
    monkeypatch.setattr(H, "current_state", lambda: None)
    out = run(H.generation_rate())
    assert body(out) == {"tokens_per_sec": 0.0, "reachable": False}


def test_no_vllm_base_url_configured_is_unreachable(monkeypatch):
    monkeypatch.setattr(H, "current_state", lambda: _state(base_url=""))
    out = run(H.generation_rate())
    assert body(out) == {"tokens_per_sec": 0.0, "reachable": False}


def test_the_configured_base_url_is_passed_through_to_the_scraper(monkeypatch):
    seen = []

    async def fake_rate(base_url, **kw):
        seen.append(base_url)
        return {"tokens_per_sec": 42.0, "reachable": True}

    monkeypatch.setattr(H, "current_state", lambda: _state())
    monkeypatch.setattr(GR, "rate", fake_rate)
    out = run(H.generation_rate())
    assert seen == ["http://127.0.0.1:8000/v1"]
    assert body(out) == {"tokens_per_sec": 42.0, "reachable": True}
