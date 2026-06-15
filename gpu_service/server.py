"""FastAPI app wrapping :mod:`gpu_service.voxwrap`.

Run with::

    PEEKY_REFERENCES_DIR=/srv/peeky/enrollment \
    VOXCPM_MODEL=openbmb/VoxCPM2 \
    uvicorn gpu_service.server:app --host 0.0.0.0 --port 8080

Endpoints (see ``standup.md`` for the contract the T10 client builds against):

- ``GET  /healthz``                  liveness + model status
- ``GET  /references``               list enrolled reference ids
- ``POST /references``               register a new reference (id + wav + transcript)
- ``POST /synthesize``               text + reference_id -> WAV bytes
- ``GET  /``                         tiny index page so a browser hit isn't 404
"""

from __future__ import annotations

import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from .voxwrap import SynthRequest, VoxCPMUnavailable, VoxCPMWrapper, make_wrapper_from_env

logging.basicConfig(
    level=os.environ.get("PEEKY_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("peeky.gpu.server")


class HealthResponse(BaseModel):
    ok: bool
    model_loaded: bool
    model: str


class ReferenceUpload(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    audio_wav_b64: str = Field(description="Base64-encoded WAV file")
    transcript: str = Field(default="", description="Optional transcript for ultimate cloning")
    overwrite: bool = Field(default=False)


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    reference_id: Optional[str] = Field(default=None, max_length=128)
    language: str = Field(default="en", max_length=8)
    sample_rate: int = Field(default=48000, ge=8000, le=48000)


_wrapper: Optional[VoxCPMWrapper] = None


def get_wrapper() -> VoxCPMWrapper:
    global _wrapper
    if _wrapper is None:
        _wrapper = make_wrapper_from_env()
    return _wrapper


@asynccontextmanager
async def lifespan(_: FastAPI):
    w = get_wrapper()
    if os.environ.get("PEEKY_EAGER_LOAD") == "1":
        try:
            w.ensure_loaded()
        except VoxCPMUnavailable:
            log.warning("eager load failed; service still up — /synthesize will retry")
    yield


app = FastAPI(
    title="Peeky VoxCPM2 service",
    version="0.1.0",
    description="Zero-shot caregiver voice clone for the Peeky baby/pet monitor.",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def index() -> JSONResponse:
    return JSONResponse({
        "service": "peeky-voxcpm2",
        "endpoints": ["/healthz", "/references", "/synthesize"],
    })


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    w = get_wrapper()
    return HealthResponse(ok=True, model_loaded=w.is_loaded, model=w.model_id)


@app.get("/references")
def list_references() -> dict:
    w = get_wrapper()
    return {"references": w.list_references()}


@app.post("/references")
def add_reference(upload: ReferenceUpload) -> dict:
    w = get_wrapper()
    if w.references_dir is None:
        raise HTTPException(500, detail="PEEKY_REFERENCES_DIR not configured on server")
    target_wav = w.references_dir / f"{upload.id}.wav"
    target_txt = w.references_dir / f"{upload.id}.txt"
    if target_wav.exists() and not upload.overwrite:
        raise HTTPException(409, detail=f"reference {upload.id!r} already exists")
    try:
        wav_bytes = base64.b64decode(upload.audio_wav_b64, validate=True)
    except Exception as exc:
        raise HTTPException(400, detail=f"invalid base64 audio: {exc}") from exc
    w.references_dir.mkdir(parents=True, exist_ok=True)
    target_wav.write_bytes(wav_bytes)
    if upload.transcript:
        target_txt.write_text(upload.transcript)
    elif target_txt.exists():
        target_txt.unlink()
    return {"ok": True, "id": upload.id, "has_transcript": bool(upload.transcript)}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest) -> Response:
    w = get_wrapper()
    try:
        samples = w.synth(SynthRequest(
            text=req.text,
            reference_id=req.reference_id,
            language=req.language,
            sample_rate=req.sample_rate,
        ))
    except FileNotFoundError as exc:
        raise HTTPException(404, detail=str(exc)) from exc
    except VoxCPMUnavailable as exc:
        log.warning("model unavailable: %s", exc)
        raise HTTPException(503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("synthesis failed")
        raise HTTPException(500, detail=f"synthesis failed: {exc!r}") from exc

    wav_bytes = _samples_to_wav_bytes(samples, req.sample_rate)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Peeky-Sample-Rate": str(req.sample_rate),
            "X-Peeky-Samples": str(len(samples)),
        },
    )


def _samples_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono as 16-bit PCM WAV using stdlib only (no soundfile
    dep required for the GPU service's tiny edge)."""
    import struct
    import wave

    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())
    return buf.getvalue()
