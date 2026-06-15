"""Client for the remote gemma-4-E4B-it reason service (ml-engineer T27 pt 2).

Frozen contract (see ml-engineer's note in standup.md, 2026-06-15):
  Host: turing (192.168.1.220), port 8082
  GET  /healthz   -> {ok, model_loaded, target, drafter}
  POST /reason    -> {event, reason, confidence, transcription, raw_text}

The model returns structured JSON describing both the *event* class and, when
the event is ``baby_cry``, a weak *reason* hint. The server has already parsed
the model output; a parse miss is signalled as a 200 with ``event="other"`` /
``reason="unknown"`` / ``confidence=0.0`` rather than as a 5xx, so the client
never has to handle JSON parse errors. 4xx/5xx/connect errors DO propagate so
the pipeline can fall back to the local heuristic.

The hardened WAV encoder (int16 passthrough, stereo downmix, clamp, empty-safe)
is reused from :mod:`peeky_reachy.detect.remote_classifier`.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .events import CryReason, SoundEvent
from .remote_classifier import _wav_b64

log = logging.getLogger("peeky.gemma_remote")

# Server caps input at 30s (gemma-4 audio-encoder max). Refuse client-side so
# the user gets a clean error instead of a wasted 30s round-trip + server 400.
_MAX_INPUT_SECONDS = 30.0


class GemmaReasonError(RuntimeError):
    """Raised on any client-side or HTTP failure. The pipeline catches this
    and falls back to the local heuristic, mirroring RemoteEventClassifier."""


class GemmaReasonClient:
    def __init__(self, base_url: str = "http://192.168.1.220:8082",
                 timeout_s: float = 10.0, health_timeout_s: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.health_timeout_s = health_timeout_s

    def _client(self, timeout: Optional[float] = None):
        import httpx

        return httpx.Client(
            base_url=self.base_url,
            timeout=timeout if timeout is not None else self.timeout_s,
        )

    def available(self) -> bool:
        """Fast /healthz with a short timeout. Returns True only if the
        service reports ``model_loaded``."""
        try:
            with self._client(self.health_timeout_s) as c:
                r = c.get("/healthz")
                if r.status_code != 200:
                    return False
                body = r.json()
                return bool(body.get("ok")) and bool(body.get("model_loaded"))
        except Exception as exc:
            log.info("Gemma reason service unavailable: %s", exc)
            return False

    def reason(self, samples: np.ndarray, sample_rate: int,
               prompt: Optional[str] = None) -> dict:
        """POST /reason, return the parsed response dict.

        The returned dict is normalised to the schema the pipeline expects::

            {
              "event": SoundEvent,                # parsed via the enum
              "reason": CryReason | None,         # None if not a cry
              "confidence": float in [0, 1],
              "transcription": str,
              "raw_text": str,
              "target": str,                      # server-side model id
              "drafter": str,                     # server-side drafter id
            }

        Raises :class:`GemmaReasonError` on any failure (network, 4xx, 5xx,
        schema mismatch, oversize input). A 200 with parse-fallback is NOT
        an error — the dict is returned with the safe defaults.
        """
        # Client-side cap: server would 400 on >30s anyway, so fail fast and
        # give the pipeline a clear exception to catch.
        n = int(np.asarray(samples).size)
        if sample_rate <= 0:
            raise GemmaReasonError(f"invalid sample_rate: {sample_rate}")
        if n / sample_rate > _MAX_INPUT_SECONDS + 1e-3:
            raise GemmaReasonError(
                f"input too long: {n / sample_rate:.2f}s > {_MAX_INPUT_SECONDS}s cap"
            )

        body: dict = {"audio_wav_b64": _wav_b64(samples, sample_rate)}
        if prompt is not None:
            body["prompt"] = prompt
        # `sample_rate` is optional per the contract (server default = 16000);
        # we always send it so the server can sanity-check the WAV header.

        try:
            with self._client() as c:
                r = c.post("/reason", json=body)
                r.raise_for_status()
                data = r.json()
        except Exception as exc:
            log.warning("Gemma reason failed: %s", exc)
            raise GemmaReasonError(str(exc)) from exc

        try:
            event = SoundEvent(data.get("event", "other"))
        except ValueError:
            # Unknown event class from a future server version — keep the
            # text but treat it as OTHER for the pipeline's enums.
            log.warning("Unknown gemma event class %r, mapping to OTHER",
                        data.get("event"))
            event = SoundEvent.OTHER

        reason_str = data.get("reason")
        reason: Optional[CryReason]
        if reason_str in (None, "", "unknown"):
            reason = None  # server's parse-fallback sentinel
        else:
            try:
                reason = CryReason(reason_str)
            except ValueError:
                log.warning("Unknown gemma reason %r, ignoring", reason_str)
                reason = None

        # confidence: server may return 0.0 on parse-fallback; clamp into [0,1]
        try:
            conf = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        return {
            "event": event,
            "reason": reason,
            "confidence": conf,
            "transcription": str(data.get("transcription", "")),
            "raw_text": str(data.get("raw_text", "")),
            "target": str(data.get("target", "")),
            "drafter": str(data.get("drafter", "")),
        }
