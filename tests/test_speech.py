"""Tests for lara.serve.speech -- lazy-loaded, cached STT/TTS, called with fakes rather
than real models (see the manual profiling notes for real latency/VRAM numbers)."""
from __future__ import annotations

import asyncio
import io
import types

import pytest

from lara.serve import speech as SP


def run(c):
    return asyncio.run(c)


@pytest.fixture(autouse=True)
def _reset():
    SP._stt = SP._stt_lock = SP._tts = SP._tts_lock = None
    SP._cuda_preloaded = False
    yield
    SP._stt = SP._stt_lock = SP._tts = SP._tts_lock = None
    SP._cuda_preloaded = False


def _lookup(data, dotted):
    node = data
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node


def _cfg(**overrides):
    data = {"speech": {"stt": {}, "tts": {}}, **overrides}
    return types.SimpleNamespace(get_in=lambda key: _lookup(data, key))


# ── pure helpers ──────────────────────────────────────────────────────────────────


def test_ct2_device_splits_a_cuda_ordinal():
    assert SP._ct2_device("cuda:1") == ("cuda", 1)
    assert SP._ct2_device("cuda:0") == ("cuda", 0)


def test_ct2_device_has_no_ordinal_on_cpu_or_mps():
    assert SP._ct2_device("cpu") == ("cpu", 0)
    assert SP._ct2_device("mps") == ("mps", 0)


def test_available_reflects_whether_each_package_importable(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: object() if name == "faster_whisper" else None)
    assert SP.available() == {"stt": True, "tts": False}


# ── preloading ────────────────────────────────────────────────────────────────────


def test_preload_is_idempotent(monkeypatch):
    import ctypes

    calls = []
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **kw: calls.append(a) or object())
    SP._preload_cuda_libs()
    first_count = len(calls)
    SP._preload_cuda_libs()          # a second call must not load anything again
    assert len(calls) == first_count


def test_preload_skips_quietly_when_the_nvidia_packages_are_absent(monkeypatch):
    import importlib

    def fail(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(importlib, "import_module", fail)
    SP._preload_cuda_libs()  # must not raise


# ── lazy loading ──────────────────────────────────────────────────────────────────


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeWhisperModel:
    instances = 0

    def __init__(self, model, **kw):
        FakeWhisperModel.instances += 1
        self.model = model
        self.kw = kw

    def transcribe(self, buf, **kw):
        return [FakeSegment(" hello "), FakeSegment("world ")], object()


def _install_fake_whisper(monkeypatch):
    import faster_whisper

    FakeWhisperModel.instances = 0
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)


def test_get_stt_loads_once_and_caches(monkeypatch):
    _install_fake_whisper(monkeypatch)
    m1 = run(SP._get_stt(_cfg()))
    m2 = run(SP._get_stt(_cfg()))
    assert m1 is m2 and FakeWhisperModel.instances == 1


def test_get_stt_loads_once_under_concurrent_callers(monkeypatch):
    _install_fake_whisper(monkeypatch)
    cfg = _cfg()

    async def go():
        return await asyncio.gather(*(SP._get_stt(cfg) for _ in range(8)))
    results = run(go())
    assert len(set(id(r) for r in results)) == 1
    assert FakeWhisperModel.instances == 1


def test_get_stt_passes_the_configured_model_and_compute_type(monkeypatch):
    _install_fake_whisper(monkeypatch)
    cfg = _cfg(speech={"stt": {"model": "small", "compute_type": "int8", "device": "cpu"}})
    m = run(SP._get_stt(cfg))
    assert m.model == "small" and m.kw["compute_type"] == "int8" and m.kw["device"] == "cpu"


def test_transcribe_joins_segment_text(monkeypatch):
    _install_fake_whisper(monkeypatch)
    text = run(SP.transcribe(b"fake audio bytes", cfg=_cfg()))
    assert text == "hello world"


def test_transcribe_raises_importerror_when_not_installed(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "faster_whisper":
            raise ImportError("no module named faster_whisper")
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError):
        run(SP.transcribe(b"x", cfg=_cfg()))


class FakeKPipeline:
    instances = 0

    def __init__(self, lang_code, device=None):
        FakeKPipeline.instances += 1
        self.lang_code = lang_code
        self.device = device

    def __call__(self, text, voice=None):
        import numpy as np

        self.last_voice = voice
        yield "g", "p", np.zeros(2400, dtype="float32")  # 0.1s of silence at 24kHz


def _install_fake_kokoro(monkeypatch):
    import kokoro

    FakeKPipeline.instances = 0
    monkeypatch.setattr(kokoro, "KPipeline", FakeKPipeline)


def test_get_tts_loads_once_and_caches(monkeypatch):
    _install_fake_kokoro(monkeypatch)
    p1 = run(SP._get_tts(_cfg()))
    p2 = run(SP._get_tts(_cfg()))
    assert p1 is p2 and FakeKPipeline.instances == 1


def test_synthesize_returns_playable_wav_bytes(monkeypatch):
    _install_fake_kokoro(monkeypatch)
    wav = run(SP.synthesize("hello there", cfg=_cfg()))
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"


def test_synthesize_of_empty_text_is_a_noop_no_model_load(monkeypatch):
    _install_fake_kokoro(monkeypatch)
    assert run(SP.synthesize("   ", cfg=_cfg())) == b""
    assert FakeKPipeline.instances == 0


def test_synthesize_uses_the_request_voice_over_the_configured_default(monkeypatch):
    _install_fake_kokoro(monkeypatch)
    run(SP.synthesize("hi", cfg=_cfg(speech={"tts": {"voice": "af_heart"}}), voice="bm_george"))
    assert SP._tts.last_voice == "bm_george"


def test_synthesize_falls_back_to_the_configured_voice(monkeypatch):
    _install_fake_kokoro(monkeypatch)
    run(SP.synthesize("hi", cfg=_cfg(speech={"tts": {"voice": "bm_george"}})))
    assert SP._tts.last_voice == "bm_george"
