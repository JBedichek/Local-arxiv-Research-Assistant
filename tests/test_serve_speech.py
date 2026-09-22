"""Tests for the /api/speech routes -- HTTP plumbing around lara.serve.speech, which is
mocked here (see test_speech.py for the module's own behavior)."""
from __future__ import annotations

import asyncio
import io
import json
import types

import pytest
from starlette.datastructures import UploadFile

from lara.serve import speech as SP
from lara.serve.routes import speech as RT


def run(c):
    return asyncio.run(c)


def body(resp):
    return json.loads(resp.body)


def _upload(data: bytes, filename: str = "clip.webm") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


@pytest.fixture(autouse=True)
def _state(monkeypatch):
    monkeypatch.setattr(RT, "require_state", lambda: types.SimpleNamespace(cfg=object()))


def test_status_reports_availability(monkeypatch):
    monkeypatch.setattr(SP, "available", lambda: {"stt": True, "tts": False})
    assert body(RT.status()) == {"stt": True, "tts": False}


def test_transcribe_rejects_empty_audio():
    out = run(RT.transcribe(audio=_upload(b"")))
    assert out.status_code == 400


def test_transcribe_rejects_an_oversized_clip(monkeypatch):
    monkeypatch.setattr(RT, "MAX_AUDIO_BYTES", 10)
    out = run(RT.transcribe(audio=_upload(b"x" * 20)))
    assert out.status_code == 413


def test_transcribe_returns_the_text(monkeypatch):
    async def fake_transcribe(audio, *, cfg):
        return "what is the sample efficiency of dpo"
    monkeypatch.setattr(SP, "transcribe", fake_transcribe)
    out = run(RT.transcribe(audio=_upload(b"real bytes")))
    assert body(out) == {"text": "what is the sample efficiency of dpo"}


def test_transcribe_when_speech_is_not_installed_is_503_not_500(monkeypatch):
    async def fake_transcribe(audio, *, cfg):
        raise ImportError("no module named faster_whisper")
    monkeypatch.setattr(SP, "transcribe", fake_transcribe)
    out = run(RT.transcribe(audio=_upload(b"real bytes")))
    assert out.status_code == 503
    assert "pip install" in body(out)["error"]


def test_transcribe_an_unreadable_clip_is_422_not_500(monkeypatch):
    from fastapi import HTTPException

    async def fake_transcribe(audio, *, cfg):
        raise RuntimeError("could not decode")
    monkeypatch.setattr(SP, "transcribe", fake_transcribe)
    with pytest.raises(HTTPException) as exc:
        run(RT.transcribe(audio=_upload(b"garbage")))
    assert exc.value.status_code == 422


def test_synthesize_rejects_empty_text():
    out = run(RT.synthesize(RT.SynthesizeRequest(text="   ")))
    assert out.status_code == 400


def test_synthesize_returns_audio_bytes(monkeypatch):
    async def fake_synthesize(text, *, cfg, voice=None):
        assert text == "hello" and voice == "bm_george"
        return b"RIFF....WAVEfmt "
    monkeypatch.setattr(SP, "synthesize", fake_synthesize)
    out = run(RT.synthesize(RT.SynthesizeRequest(text="hello", voice="bm_george")))
    assert out.media_type == "audio/wav" and out.body == b"RIFF....WAVEfmt "


def test_synthesize_when_speech_is_not_installed_is_503(monkeypatch):
    async def fake_synthesize(text, *, cfg, voice=None):
        raise ImportError("no module named kokoro")
    monkeypatch.setattr(SP, "synthesize", fake_synthesize)
    out = run(RT.synthesize(RT.SynthesizeRequest(text="hello")))
    assert out.status_code == 503


def test_synthesize_of_nothing_produced_is_422(monkeypatch):
    async def fake_synthesize(text, *, cfg, voice=None):
        return b""
    monkeypatch.setattr(SP, "synthesize", fake_synthesize)
    out = run(RT.synthesize(RT.SynthesizeRequest(text="hello")))
    assert out.status_code == 422
