"""Tests for the gemma-4 reason client (ai-engineer-3 scope).

Frozen contract with ml-engineer's ``gemma_service`` (turing :8082, 2026-06-15):
- ``GET  /healthz``   -> ``{ok, model_loaded, target, drafter}``
- ``POST /reason``    -> ``{event, reason, confidence, transcription, raw_text}``
- Server caps input at 30s. >30s -> 400.
- 4xx/5xx/connect errors must propagate as :class:`GemmaReasonError` so the
  pipeline can fall back to the local heuristic.
- 200 with parse-fallback (event="other", reason="unknown", confidence=0.0)
  is NOT an error — return the dict.

All tests are offline; the network is mocked with ``httpx.MockTransport``.
"""

from __future__ import annotations

import base64
import io
import json
import wave

import httpx
import numpy as np
import pytest

from peeky_reachy.detect.events import CryReason, SoundEvent
from peeky_reachy.detect.gemma_reason import (
    GemmaReasonClient,
    GemmaReasonError,
)
from tests.conftest import SR, baby_cry


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _patched_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)

    def _client(self, timeout=None):
        return httpx.Client(base_url=self.base_url, timeout=timeout or self.timeout_s,
                            transport=transport)

    monkeypatch.setattr(GemmaReasonClient, "_client", _client)


def _assert_wav_mono_pcm16(b64: str, expected_frames: int):
    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getnframes() == expected_frames


# ---------------------------------------------------------------------------
# available()
# ---------------------------------------------------------------------------


def test_available_true_when_healthz_model_loaded(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={
                "ok": True, "model_loaded": True,
                "target": "google/gemma-4-E4B-it",
                "drafter": "google/gemma-4-E4B-it-assistant",
            })
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    assert GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0).available() is True


def test_available_false_when_model_not_loaded(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "ok": True, "model_loaded": False,
            "target": "google/gemma-4-E4B-it", "drafter": "...",
        })

    _patched_client(monkeypatch, handler)
    assert GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0).available() is False


def test_available_false_when_ok_false(monkeypatch):
    """The server may report ok=false during startup; treat as unavailable."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "ok": False, "model_loaded": True,
            "target": "...", "drafter": "...",
        })

    _patched_client(monkeypatch, handler)
    assert GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0).available() is False


def test_available_false_on_5xx(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="crash")

    _patched_client(monkeypatch, handler)
    assert GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0).available() is False


def test_available_uses_short_timeout(monkeypatch):
    """`available()` should not block on the full request timeout."""
    from peeky_reachy.detect import gemma_reason

    seen_timeouts: list[float] = []

    def _client(self, timeout=None):
        seen_timeouts.append(timeout)
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json={
            "ok": True, "model_loaded": True, "target": "...", "drafter": "...",
        }))
        return httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    monkeypatch.setattr(GemmaReasonClient, "_client", _client)
    GemmaReasonClient("http://fake-turing:8082", timeout_s=10.0,
                      health_timeout_s=1.5).available()
    assert seen_timeouts == [1.5]  # only the healthz timeout, not 10s


# ---------------------------------------------------------------------------
# reason() — happy paths
# ---------------------------------------------------------------------------


def test_reason_happy_path_baby_cry(monkeypatch):
    received: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/reason":
            body = json.loads(request.content)
            received.update(body)
            _assert_wav_mono_pcm16(body["audio_wav_b64"], 16000)
            return httpx.Response(200, json={
                "event": "baby_cry",
                "reason": "hungry",
                "confidence": 0.62,
                "transcription": "wah wah",
                "raw_text": '{"event": "baby_cry", "reason": "hungry", "confidence": 0.62}',
                "target": "google/gemma-4-E4B-it",
                "drafter": "google/gemma-4-E4B-it-assistant",
            })
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    client = GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0)
    out = client.reason(baby_cry(1.0), SR)
    assert out["event"] == SoundEvent.BABY_CRY
    assert out["reason"] == CryReason.HUNGRY
    assert out["confidence"] == pytest.approx(0.62)
    assert out["transcription"] == "wah wah"
    # Default body must include the audio + a hint about sample_rate. The
    # contract makes sample_rate optional; we always send it.
    assert "audio_wav_b64" in received
    # prompt was not supplied -> not present in body
    assert "prompt" not in received


def test_reason_silence_with_unknown_reason_is_normalised(monkeypatch):
    """For non-cry events the server's reason field may be 'unknown' or
    empty; client must return reason=None so the pipeline skips the hint."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "event": "silence",
            "reason": "unknown",
            "confidence": 0.95,
            "transcription": "",
            "raw_text": "...",
            "target": "...", "drafter": "...",
        })

    _patched_client(monkeypatch, handler)
    out = GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0).reason(
        np.zeros(8000, dtype=np.float32), SR
    )
    assert out["event"] == SoundEvent.SILENCE
    assert out["reason"] is None
    assert out["confidence"] == pytest.approx(0.95)


def test_reason_parse_fallback_returns_safe_defaults(monkeypatch):
    """200 with event=other / reason=unknown / confidence=0.0 is the
    server's parse-fallback signal. Client must NOT raise; just return."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "event": "other",
            "reason": "unknown",
            "confidence": 0.0,
            "transcription": "",
            "raw_text": "garbage from the model: not even close to json",
        })

    _patched_client(monkeypatch, handler)
    out = GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0).reason(
        np.zeros(1000, dtype=np.float32), SR
    )
    assert out["event"] == SoundEvent.OTHER
    assert out["reason"] is None
    assert out["confidence"] == 0.0
    # raw_text preserved so the pipeline / log can show what the model said
    assert "garbage" in out["raw_text"]


def test_reason_optional_prompt_is_forwarded(monkeypatch):
    received: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/reason":
            received.update(json.loads(request.content))
            return httpx.Response(200, json={
                "event": "speech", "reason": "unknown", "confidence": 0.8,
                "transcription": "...", "raw_text": "...",
                "target": "...", "drafter": "...",
            })
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0).reason(
        np.zeros(100, dtype=np.float32), SR, prompt="explain in one word"
    )
    assert received["prompt"] == "explain in one word"


# ---------------------------------------------------------------------------
# reason() — failure modes (all raise GemmaReasonError)
# ---------------------------------------------------------------------------


def test_reason_400_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "invalid base64 wav"})

    _patched_client(monkeypatch, handler)
    client = GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0)
    with pytest.raises(GemmaReasonError):
        client.reason(np.zeros(100, dtype=np.float32), SR)


def test_reason_500_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "model OOM"})

    _patched_client(monkeypatch, handler)
    client = GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0)
    with pytest.raises(GemmaReasonError):
        client.reason(np.zeros(100, dtype=np.float32), SR)


def test_reason_503_raises(monkeypatch):
    """503 (model not loaded yet) is distinct from parse-fallback; the
    server explicitly returns 503 for load errors so the client can
    back off, and we must propagate that as GemmaReasonError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "loading"})

    _patched_client(monkeypatch, handler)
    with pytest.raises(GemmaReasonError):
        GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0).reason(
            np.zeros(100, dtype=np.float32), SR
        )


def test_reason_connection_refused_raises():
    """Real network failure (closed port) must raise, not return."""
    client = GemmaReasonClient("http://127.0.0.1:9", timeout_s=0.2)
    with pytest.raises(GemmaReasonError):
        client.reason(np.zeros(100, dtype=np.float32), SR)


def test_reason_invalid_sample_rate_raises_client_side():
    """Bad sample_rate should fail at the client, not as a server 400."""
    client = GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0)
    with pytest.raises(GemmaReasonError, match="invalid sample_rate"):
        client.reason(np.zeros(100, dtype=np.float32), sample_rate=0)


def test_reason_oversize_input_fails_client_side(monkeypatch):
    """>30s input must be rejected locally with a clear error instead of
    being sent (server would 400 anyway)."""
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})  # would be unused

    _patched_client(monkeypatch, handler)
    client = GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0)
    big = np.zeros(31 * SR, dtype=np.float32)  # 31 seconds
    with pytest.raises(GemmaReasonError, match="too long"):
        client.reason(big, SR)
    # Critically: must NOT have made the HTTP call.
    assert calls == []


# ---------------------------------------------------------------------------
# reason() — defensive schema handling
# ---------------------------------------------------------------------------


def test_reason_unknown_event_class_maps_to_other(monkeypatch):
    """A future server version may add a new event class; the client
    should map it to OTHER rather than raising KeyError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "event": "foghorn",  # not in our enum
            "reason": "unknown", "confidence": 0.5,
            "transcription": "", "raw_text": "...",
        })

    _patched_client(monkeypatch, handler)
    out = GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0).reason(
        np.zeros(100, dtype=np.float32), SR
    )
    assert out["event"] == SoundEvent.OTHER


def test_reason_unknown_reason_class_is_ignored(monkeypatch):
    """A future server version may add a new reason; client should drop
    the hint, not crash."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "event": "baby_cry", "reason": "teleportation",
            "confidence": 0.5, "transcription": "", "raw_text": "...",
        })

    _patched_client(monkeypatch, handler)
    out = GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0).reason(
        np.zeros(100, dtype=np.float32), SR
    )
    assert out["event"] == SoundEvent.BABY_CRY
    assert out["reason"] is None


def test_reason_clamps_out_of_range_confidence(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "event": "silence", "reason": "unknown",
            "confidence": 1.7,  # bug on the server
            "transcription": "", "raw_text": "...",
        })

    _patched_client(monkeypatch, handler)
    out = GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0).reason(
        np.zeros(100, dtype=np.float32), SR
    )
    assert 0.0 <= out["confidence"] <= 1.0


def test_reason_accepts_stereo_input(monkeypatch):
    """The hardened _wav_b64 must downmix to mono before sending, otherwise
    the server would see a 2x-long clip."""
    captured_frames: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/reason":
            body = json.loads(request.content)
            raw = base64.b64decode(body["audio_wav_b64"])
            with wave.open(io.BytesIO(raw), "rb") as wf:
                captured_frames.append(wf.getnframes())
            return httpx.Response(200, json={
                "event": "silence", "reason": "unknown", "confidence": 0.9,
                "transcription": "", "raw_text": "...",
            })
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    stereo = np.zeros((1600, 2), dtype=np.float32)
    GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0).reason(stereo, SR)
    # 1600 frames in mono, NOT 3200 (broadcast-flattened)
    assert captured_frames == [1600]


def test_reason_accepts_int16_pcm(monkeypatch):
    """int16 input must be passed through bit-exact (covered for the
    shared encoder in test_remote_classifier; here we just confirm the
    client module reuses it)."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/reason":
            body = json.loads(request.content)
            _assert_wav_mono_pcm16(body["audio_wav_b64"], 4)
            return httpx.Response(200, json={
                "event": "silence", "reason": "unknown", "confidence": 0.0,
                "transcription": "", "raw_text": "...",
            })
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    GemmaReasonClient("http://fake-turing:8082", timeout_s=1.0).reason(
        np.array([100, -200, 30000, -30000], dtype=np.int16), SR
    )


# ---------------------------------------------------------------------------
# config keys
# ---------------------------------------------------------------------------


def test_config_keys_default_to_disabled_and_turing_8082():
    from peeky_reachy.config import Config

    cfg = Config()
    assert cfg.use_remote_gemma is False
    assert cfg.gemma_reason_url == "http://192.168.1.220:8082"
    assert cfg.gemma_timeout_s == pytest.approx(10.0)


def test_config_env_overrides(monkeypatch):
    from peeky_reachy.config import Config

    monkeypatch.setenv("PEEKY_USE_REMOTE_GEMMA", "true")
    monkeypatch.setenv("PEEKY_GEMMA_REASON_URL", "http://other-host:9999")
    monkeypatch.setenv("PEEKY_GEMMA_TIMEOUT_S", "5.5")
    cfg = Config.from_env()
    assert cfg.use_remote_gemma is True
    assert cfg.gemma_reason_url == "http://other-host:9999"
    assert cfg.gemma_timeout_s == pytest.approx(5.5)
