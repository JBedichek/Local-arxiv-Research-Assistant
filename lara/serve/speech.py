"""Reading a lesson aloud, and asking a question by mic.

Two small, independent models, each loaded once and held warm the same way the embedder
and reranker already are (see `AppState._load`) — except lazily, on the first request that
actually needs one, since most installs never touch a microphone and there is no reason to
pay the load cost (or the VRAM) for a capability nobody asked for this run.

**STT** is `faster-whisper` (CTranslate2, not torch): ~2 GB VRAM for `distil-large-v3`,
~120ms to transcribe a several-second question once warm. **TTS** is Kokoro-82M (torch):
~1 GB VRAM, ~300x realtime -- a lesson paragraph is audio in well under a second. Neither
is close to being the bottleneck in "ask a question and get an answer"; the generator is.

Both are optional (`pip install -e '.[speech]'`): importing them lives inside the lazy
getters, not at module scope, so a base install that never calls into this module never
needs them installed at all. `available()` is what the UI checks before showing a mic or a
speaker button.
"""
from __future__ import annotations

import asyncio
import io

from lara import device as dev

#: faster-whisper's default; distil-large-v3 measured at the same latency as `small` with
#: better accuracy, so it is the one worth the extra ~1 GB of VRAM over `small`.
DEFAULT_STT_MODEL = "distil-large-v3"
DEFAULT_TTS_VOICE = "af_heart"

_stt = None
_stt_lock: asyncio.Lock | None = None
_tts = None
_tts_lock: asyncio.Lock | None = None
#: Set once cuBLAS/cuDNN have been preloaded into the process -- see `_preload_cuda_libs`.
_cuda_preloaded = False


def _preload_cuda_libs() -> None:
    """CTranslate2 (faster-whisper's backend) `dlopen`s cuBLAS/cuDNN by soname at inference
    time, and finds nothing: torch's own pip wheels ship both under `nvidia/*/lib`, not on
    the system linker path, and setting `$LD_LIBRARY_PATH` from inside an already-running
    process does not reach a *later* `dlopen` call -- measured, not assumed: it loads the
    model fine either way and then fails on the first real transcription with `Library
    libcublas.so.12 is not found`.

    Loading each `.so` once here with `RTLD_GLOBAL`, before CTranslate2 ever asks for it,
    works because the dynamic linker resolves a `dlopen()` of an already-loaded soname to
    the existing handle rather than searching again. Only meaningful with a CUDA torch
    install; on CPU-only or Mac there is nothing to preload and this is a silent no-op.
    """
    global _cuda_preloaded
    if _cuda_preloaded:
        return
    _cuda_preloaded = True                    # set first: never worth retrying per call
    import ctypes
    import glob
    import importlib

    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            mod = importlib.import_module(pkg)
            lib_dir = list(mod.__path__)[0] + "/lib"
        except Exception:                                      # noqa: BLE001
            continue
        for so in sorted(glob.glob(f"{lib_dir}/*.so*")):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue                       # not every file here is directly loadable


def _ct2_device(device: str) -> tuple[str, int]:
    """`lara.device.resolve`'s `"cuda:1"` -> CTranslate2's own split (device, device_index)."""
    if ":" in device:
        kind, idx = device.split(":", 1)
        return kind, int(idx)
    return device, 0


def available() -> dict:
    """{"stt": bool, "tts": bool} -- whether each package is installed, without loading
    either model. What the UI asks before showing a mic or a speaker button."""
    def can_import(name: str) -> bool:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    return {"stt": can_import("faster_whisper"), "tts": can_import("kokoro")}


async def _get_stt(cfg):
    global _stt, _stt_lock
    if _stt is not None:
        return _stt
    # Created here, not at module import time: an asyncio.Lock binds to whatever event
    # loop is running when it is first awaited, and this module is imported long before
    # uvicorn's loop exists. In production there is only ever one loop for the process's
    # whole life, so this only ever runs once regardless.
    if _stt_lock is None:
        _stt_lock = asyncio.Lock()
    async with _stt_lock:
        if _stt is not None:                   # re-check: lost the race while waiting
            return _stt
        from faster_whisper import WhisperModel

        scfg = cfg.get_in("speech.stt") or {}
        device = dev.resolve(scfg.get("device", "auto"))
        kind, idx = _ct2_device(device)
        if kind == "cuda":
            _preload_cuda_libs()
        else:
            idx = 0                            # ct2 has no cpu/mps ordinals
        compute_type = scfg.get("compute_type") or ("float16" if kind == "cuda" else "int8")

        def load():
            return WhisperModel(scfg.get("model", DEFAULT_STT_MODEL), device=kind,
                               device_index=idx, compute_type=compute_type)
        _stt = await asyncio.to_thread(load)
        return _stt


async def _get_tts(cfg):
    global _tts, _tts_lock
    if _tts is not None:
        return _tts
    if _tts_lock is None:
        _tts_lock = asyncio.Lock()
    async with _tts_lock:
        if _tts is not None:
            return _tts
        from kokoro import KPipeline

        tcfg = cfg.get_in("speech.tts") or {}
        device = dev.resolve(tcfg.get("device", "auto"))

        def load():
            return KPipeline(lang_code="a", device=device)
        _tts = await asyncio.to_thread(load)
        return _tts


def _transcribe_sync(model, audio: bytes) -> str:
    segments, _info = model.transcribe(io.BytesIO(audio), beam_size=5)
    return " ".join(s.text.strip() for s in segments).strip()


async def transcribe(audio: bytes, *, cfg) -> str:
    """The spoken words in `audio` (any container ffmpeg/PyAV reads -- webm/opus straight
    off a phone's `MediaRecorder`, wav, mp3, ...), or "" for silence/near-silence. Raises
    only if `faster-whisper` is not installed; a bad or unreadable clip transcribes to a
    faster-whisper-reported empty result rather than an exception, same as silence does."""
    model = await _get_stt(cfg)
    return await asyncio.to_thread(_transcribe_sync, model, audio)


def _synthesize_sync(pipeline, text: str, voice: str) -> bytes:
    import numpy as np
    import soundfile as sf

    chunks = [audio for _graphemes, _phonemes, audio in pipeline(text, voice=voice)]
    if not chunks:
        return b""
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(chunks), 24000, format="WAV")
    return buf.getvalue()


async def synthesize(text: str, *, cfg, voice: str | None = None) -> bytes:
    """`text` read aloud, as WAV bytes at 24kHz -- b"" for empty/whitespace-only text.
    Raises only if `kokoro` is not installed."""
    if not text.strip():
        return b""
    pipeline = await _get_tts(cfg)
    tcfg = cfg.get_in("speech.tts") or {}
    return await asyncio.to_thread(_synthesize_sync, pipeline, text,
                                   voice or tcfg.get("voice", DEFAULT_TTS_VOICE))
