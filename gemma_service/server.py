"""FastAPI app wrapping :mod:`gemma_service.gemmawrap`.

Run with::

    uvicorn gemma_service.server:app --host 0.0.0.0 --port 8082

Endpoints (frozen contract; mirrored in ``README.md`` and sent to
ai-engineer-3 on 2026-06-15):

- ``GET  /healthz``    liveness + model status (target + drafter)
- ``POST /reason``     {audio_wav_b64, sample_rate?, prompt?}
                      -> {event, reason, confidence, transcription, raw_text}
- ``GET  /``           tiny index page so a browser hit isn't 404

Server-side guards (all enforced here, never trust the client):

* **30 s audio cap** — gemma-4-E4B-it's audio encoder max. We trim the
  first 30 s if longer and *also* reject anything > 35 s with 400 (we
  don't want callers to silently get truncated audio that they think
  is the whole clip).
* **256 max output tokens** — enough for the JSON object + a short
  transcription; well below the model's context so a runaway doesn't
  melt the box.
* **30 s request timeout** — concurrent requests serialise through the
  model's ``generate`` call, so we cap individual latencies.
* **Parse fallback** — if the model returns malformed JSON, the server
  returns 200 with ``event="other"``, ``reason="unknown"``,
  ``confidence=0.0``, and the raw text in ``raw_text``. 5xx is reserved
  for actual model errors.
* **No third-party audio libs** — stdlib ``wave`` only (matches the
  ``cry_service`` / ``gpu_service`` pattern).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import wave
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .gemmawrap import (DEFAULT_PROMPT, GemmaReasonWrapper, GemmaUnavailable,
                        MAX_AUDIO_SECONDS, ReasonRequest, make_wrapper_from_env)

logging.basicConfig(
    level=os.environ.get("PEEKY_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("peeky.gemma.server")

# Cap is MAX_AUDIO_SECONDS; we reject (not silently truncate) anything
# over MAX+5 to surface "I sent 60s of audio" bugs in the caller.
HARD_REJECT_SECONDS = MAX_AUDIO_SECONDS + 5.0
DEFAULT_TIMEOUT_S = float(os.environ.get("PEEKY_GEMMA_TIMEOUT_S", "30"))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("PEEKY_GEMMA_MAX_NEW_TOKENS", "256"))


# ---- Audio I/O (stdlib only, matches cry_service pattern) ----

def wav_b64_to_samples(data_b64: str) -> tuple["np.ndarray", int]:
    """Decode base64 -> mono float32 + sample_rate. Stdlib ``wave`` only.

    Mirrors ``cry_service.classifier_wrap.wav_b64_to_samples`` so the
    wire format is identical between the cry and gemma services.
    """
    import numpy as np

    raw = base64.b64decode(data_b64, validate=True)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        pcm = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    return pcm, sr


# ---- Pydantic request/response models ----

class HealthResponse(BaseModel):
    ok: bool
    model_loaded: bool
    target: str
    drafter: str


class ReasonRequestModel(BaseModel):
    audio_wav_b64: str = Field(description="Base64-encoded mono PCM16 WAV, ≤30s")
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    prompt: Optional[str] = Field(
        default=None,
        description="Override the default JSON-only prompt. The default is hard-coded.",
    )


class ReasonResponseModel(BaseModel):
    event: str = Field(description="One of: silence|speech|baby_cry|dog|other")
    reason: str = Field(description="One of: hungry|tired|discomfort|pain|burping|unknown")
    confidence: float = Field(ge=0.0, le=1.0)
    transcription: str
    raw_text: str


# ---- Wrapper holder ----

_wrapper: Optional[GemmaReasonWrapper] = None


def get_wrapper() -> GemmaReasonWrapper:
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
        except GemmaUnavailable as exc:
            log.warning("eager load failed; service up, will retry on first request: %s", exc)
    yield


app = FastAPI(
    title="Peeky gemma-4 reason service",
    version="0.1.0",
    description=(
        "Reasoning + cry-reason hints for the Peeky baby/pet monitor. "
        "Wraps google/gemma-4-E4B-it (target) + gemma-4-E4B-it-assistant (MTP drafter)."
    ),
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def index() -> JSONResponse:
    return JSONResponse({
        "service": "peeky-gemma-reason",
        "endpoints": ["/healthz", "/reason"],
        "target": "google/gemma-4-E4B-it",
        "drafter": "google/gemma-4-E4B-it-assistant",
    })


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    w = get_wrapper()
    # Touch the model so model_loaded reflects reality after first hit
    # (mirrors cry_service behaviour).
    try:
        w.ensure_loaded()
    except GemmaUnavailable as exc:
        log.warning("model load on healthz failed: %s", exc)
    return HealthResponse(
        ok=True,
        model_loaded=w.is_loaded,
        target=w.target_id,
        drafter=w.drafter_id,
    )


@app.post("/reason", response_model=ReasonResponseModel)
def reason(req: ReasonRequestModel) -> ReasonResponseModel:
    w = get_wrapper()
    # 1. Decode the WAV + enforce the 30 s cap.
    try:
        samples, sr = wav_b64_to_samples(req.audio_wav_b64)
    except Exception as exc:
        raise HTTPException(400, detail=f"invalid base64 wav: {exc}") from exc

    duration_s = len(samples) / float(sr) if sr > 0 else 0.0
    if duration_s > HARD_REJECT_SECONDS:
        raise HTTPException(
            400,
            detail=(
                f"audio is {duration_s:.1f}s, exceeds hard cap of "
                f"{HARD_REJECT_SECONDS:.0f}s (gemma-4 max is {MAX_AUDIO_SECONDS:.0f}s)"
            ),
        )

    # 2. Run the model. 5xx on real model errors (no /reason result at all).
    try:
        result = w.reason(ReasonRequest(
            audio_wav_b64=req.audio_wav_b64,  # we re-decode inside; cheap
            sample_rate=req.sample_rate,
            prompt=req.prompt,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            timeout_s=DEFAULT_TIMEOUT_S,
        ))
    except GemmaUnavailable as exc:
        # Model isn't loaded / load failed — tell the client it can retry.
        log.warning("gemma unavailable: %s", exc)
        raise HTTPException(503, detail=f"gemma unavailable: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("reason failed")
        raise HTTPException(500, detail=f"reason failed: {exc!r}") from exc

    # 3. Parse the JSON. **Never raise on parse miss** — that's a model
    #    formatting bug, not a service fault. Return 200 with the
    #    safe-default shape so the pipeline can fall back to local
    #    reasoning in `RemoteEventClassifier`-style flow.
    parsed = _parse_model_json(result.text)
    if parsed is None:
        log.warning("model returned unparseable JSON (truncated to 200 chars): %.200s",
                    result.text)
        return ReasonResponseModel(
            event="other",
            reason="unknown",
            confidence=0.0,
            transcription="",
            raw_text=result.text,
        )

    return ReasonResponseModel(
        event=_coerce_event(parsed.get("event")),
        reason=_coerce_reason(parsed.get("reason")),
        confidence=_coerce_confidence(parsed.get("confidence")),
        transcription=str(parsed.get("transcription", "") or "")[:1024],
        raw_text=result.text,
    )


# ---- JSON parsing helpers (defensive, never raise) ----

# Match a JSON object inside a model's response. gemma-4 with the
# default prompt should emit it bare, but some drift is expected
# (markdown fences, leading "Sure, here is the JSON:" prose, etc.).
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_model_json(text: str) -> Optional[dict]:
    """Try hard to extract a JSON object from ``text``. Returns None on miss."""
    if not text:
        return None
    # 1. Try the whole string.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # 2. Try the first { ... } block (greedy across newlines).
    m = _JSON_OBJECT_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


_EVENT_VALUES = {"silence", "speech", "baby_cry", "dog", "other"}
_REASON_VALUES = {"hungry", "tired", "discomfort", "pain", "burping", "unknown"}


def _coerce_event(v) -> str:
    if isinstance(v, str) and v in _EVENT_VALUES:
        return v
    return "other"


def _coerce_reason(v) -> str:
    if isinstance(v, str) and v in _REASON_VALUES:
        return v
    return "unknown"


def _coerce_confidence(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if f != f:  # NaN
        return 0.0
    return max(0.0, min(1.0, f))
