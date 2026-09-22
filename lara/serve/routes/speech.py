"""`/api/speech` -- transcribe a mic recording, and read text aloud.

Both handlers are thin: the actual models are lazy-loaded, cached, and called from
`lara.serve.speech`. This module only turns HTTP into bytes in and bytes/JSON out.
"""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from lara.serve import speech as SP
from lara.serve.deps import require_state

router = APIRouter()

#: A phone clip of a spoken question is a few seconds; this is generous headroom against
#: someone leaving the mic open, not a bound on a legitimate recording.
MAX_AUDIO_BYTES = 25 << 20


class SynthesizeRequest(BaseModel):
    text: str
    voice: str | None = None


def _unavailable(exc: ImportError) -> JSONResponse:
    return JSONResponse(
        {"error": f"speech support is not installed ({exc}); run "
                  "pip install -e '.[speech]' in lara's venv and restart the server"},
        status_code=503)


@router.get("/api/speech/status")
def status() -> JSONResponse:
    """{"stt", "tts"} -- whether each is installed, so the UI knows whether to offer a
    mic or a speaker button at all. Says nothing about whether a model has *loaded* yet;
    that happens lazily on first real use and just costs that one request more time."""
    return JSONResponse(SP.available())


@router.post("/api/speech/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> JSONResponse:
    state = require_state()
    body = await audio.read()
    if not body:
        return JSONResponse({"error": "no audio received"}, status_code=400)
    if len(body) > MAX_AUDIO_BYTES:
        return JSONResponse({"error": f"clip too large ({len(body):,} bytes, "
                                      f"max {MAX_AUDIO_BYTES:,})"}, status_code=413)
    try:
        text = await SP.transcribe(body, cfg=state.cfg)
    except ImportError as exc:
        return _unavailable(exc)
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(422, f"could not read that clip: {type(exc).__name__}: {exc}") from exc
    return JSONResponse({"text": text})


@router.post("/api/speech/synthesize")
async def synthesize(req: SynthesizeRequest) -> Response:
    state = require_state()
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "no text to read"}, status_code=400)
    try:
        wav = await SP.synthesize(text, cfg=state.cfg, voice=req.voice)
    except ImportError as exc:
        return _unavailable(exc)
    if not wav:
        return JSONResponse({"error": "nothing was synthesized"}, status_code=422)
    return Response(content=wav, media_type="audio/wav")
