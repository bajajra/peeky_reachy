"""FastAPI baby-cry classification service (runs on turing).

Run with::

    PEEKY_CRY_PREFER_ML=1 \
    uvicorn cry_service.server:app --host 0.0.0.0 --port 8080

Endpoints:
- ``GET  /healthz``    liveness + model status
- ``POST /classify``   {audio_wav_b64} -> {event, score}
- ``GET  /``           tiny index page
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .classifier_wrap import ClassifierWrapper, make_wrapper_from_env, wav_b64_to_samples

logging.basicConfig(
    level=os.environ.get("PEEKY_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("peeky.cry.server")


class HealthResponse(BaseModel):
    ok: bool
    model_loaded: bool
    model: str


class ClassifyRequest(BaseModel):
    audio_wav_b64: str = Field(description="Base64-encoded mono WAV (PCM16)")


class ClassifyResponse(BaseModel):
    event: str
    score: float


_wrapper: Optional[ClassifierWrapper] = None


def get_wrapper() -> ClassifierWrapper:
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
        except Exception as exc:  # noqa: BLE001
            log.warning("eager load failed; service up, will retry on first request: %s", exc)
    yield


app = FastAPI(
    title="Peeky cry-classification service",
    version="0.1.0",
    description="Baby/pet cry detection for the Peeky monitor (YAMNet, numpy fallback).",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def index() -> JSONResponse:
    return JSONResponse({"service": "peeky-cry", "endpoints": ["/healthz", "/classify"]})


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    w = get_wrapper()
    # Touch the model so model_loaded reflects reality after first hit.
    try:
        w.ensure_loaded()
    except Exception as exc:  # noqa: BLE001
        log.warning("model load on healthz failed: %s", exc)
    return HealthResponse(ok=True, model_loaded=w.is_loaded, model=w.model_id)


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    w = get_wrapper()
    try:
        samples, sr = wav_b64_to_samples(req.audio_wav_b64)
    except Exception as exc:
        raise HTTPException(400, detail=f"invalid base64 wav: {exc}") from exc
    try:
        event, score = w.classify(samples, sr)
    except Exception as exc:  # noqa: BLE001
        log.exception("classify failed")
        raise HTTPException(500, detail=f"classify failed: {exc!r}") from exc
    return ClassifyResponse(event=event, score=score)
