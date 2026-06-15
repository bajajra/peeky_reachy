"""Contract tests for ``gemma_service.server``.

The real gemma-4 model is not loaded in tests — we monkey-patch the
server's wrapper with a ``_FakeGemmaWrapper`` that returns canned text
on ``reason()`` and reports a fixed ``is_loaded`` state. This mirrors
the pattern in ``tests/test_cry_service.py``.

What we assert:

* ``/healthz`` shape and content (incl. target/drafter ids).
* ``/reason`` happy path: model returns clean JSON -> we echo it.
* ``/reason`` parse-fallback path: model returns prose -> 200 with
  ``event="other"``, ``reason="unknown"``, ``raw_text`` populated.
* ``/reason`` rejects > 35 s of audio (hard cap) with 400.
* ``/reason`` rejects bad base64 with 400.
* ``/reason`` coerces out-of-enum values to the safe default.
* ``/reason`` clamps ``confidence`` to [0, 1].
* ``/reason`` model unavailable -> 503.
* ``/reason`` model exception -> 500.
* ``/reason`` accepts a prompt override without breaking.
* The ``/`` index page returns 200.

All offline, deterministic, no real network, no real model.
"""

from __future__ import annotations

import base64
import io
import wave

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from gemma_service import server as gemma_server  # noqa: E402
from gemma_service.gemmawrap import GemmaUnavailable, GemmaReasonWrapper  # noqa: E402
from tests.conftest import SR, baby_cry, silence  # noqa: E402


# ---- fakes ----

class _FakeGemmaWrapper(GemmaReasonWrapper):
    """Subclass that never imports transformers / never loads a model."""

    def __init__(self, *, is_loaded: bool = True, raise_on_reason=None,
                 return_text: str = "{}"):
        # Skip super().__init__ — we don't need the HF state.
        self.target_id = "google/gemma-4-E4B-it"
        self.drafter_id = "google/gemma-4-E4B-it-assistant"
        self._is_loaded = is_loaded
        self._raise_on_reason = raise_on_reason
        self._return_text = return_text
        self._calls: list[dict] = []

    @property
    def is_loaded(self) -> bool:  # type: ignore[override]
        return self._is_loaded

    def ensure_loaded(self):  # type: ignore[override]
        if not self._is_loaded:
            raise GemmaUnavailable("fake: not loaded")
        return self

    def reason(self, req):  # type: ignore[override]
        self._calls.append({
            "sample_rate": req.sample_rate,
            "max_new_tokens": req.max_new_tokens,
            "timeout_s": req.timeout_s,
            "prompt_override": req.prompt,
        })
        if self._raise_on_reason is not None:
            raise self._raise_on_reason
        from gemma_service.gemmawrap import ReasonResult
        return ReasonResult(text=self._return_text)


# ---- fixtures ----

@pytest.fixture
def fake_wrapper():
    return _FakeGemmaWrapper()


@pytest.fixture
def client(fake_wrapper):
    gemma_server._wrapper = fake_wrapper
    return TestClient(gemma_server.app)


def _wav_b64(samples: np.ndarray, sr: int = SR) -> str:
    """Encode float32 mono as PCM16 WAV -> base64 (same shape as the client)."""
    pcm_i16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_i16.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


# ---- /healthz ----

def test_healthz_reports_target_and_drafter(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "ok": True,
        "model_loaded": True,
        "target": "google/gemma-4-E4B-it",
        "drafter": "google/gemma-4-E4B-it-assistant",
    }


def test_healthz_reports_unloaded():
    gemma_server._wrapper = _FakeGemmaWrapper(is_loaded=False)
    c = TestClient(gemma_server.app)
    r = c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is False
    assert body["target"] == "google/gemma-4-E4B-it"
    assert body["drafter"] == "google/gemma-4-E4B-it-assistant"


def test_healthz_unavailable_does_not_500():
    """If the model is unloadable, healthz must still be 200 with model_loaded=false."""

    class _Broken(_FakeGemmaWrapper):
        def __init__(self):
            # Start in the "I never managed to load" state — is_loaded=False
            # is the *real* wrapper's behavior after a load failure (it sets
            # self._load_error and never assigns self._target).
            super().__init__(is_loaded=False)

        def ensure_loaded(self):
            # Simulate the real wrapper: it tries to load, fails, and the
            # property still reports unloaded. We re-raise so /healthz
            # catches it (matching the production code path).
            raise GemmaUnavailable("disk full")

    gemma_server._wrapper = _Broken()
    c = TestClient(gemma_server.app)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is False


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "peeky-gemma-reason"
    assert "/healthz" in body["endpoints"]
    assert "/reason" in body["endpoints"]


# ---- /reason: happy path ----

def test_reason_clean_json_echoed(client, fake_wrapper):
    fake_wrapper._return_text = (
        '{"event": "baby_cry", "reason": "hungry", "confidence": 0.73, '
        '"transcription": "wah wah wah"}'
    )
    r = client.post("/reason", json={"audio_wav_b64": _wav_b64(baby_cry(1.0))})
    assert r.status_code == 200
    body = r.json()
    assert body["event"] == "baby_cry"
    assert body["reason"] == "hungry"
    assert body["confidence"] == pytest.approx(0.73)
    assert body["transcription"] == "wah wah wah"
    assert "baby_cry" in body["raw_text"]


def test_reason_silence_class(client, fake_wrapper):
    fake_wrapper._return_text = '{"event": "silence", "reason": "unknown", "confidence": 0.99}'
    r = client.post("/reason", json={"audio_wav_b64": _wav_b64(silence(2.0))})
    assert r.status_code == 200
    body = r.json()
    assert body["event"] == "silence"
    assert body["reason"] == "unknown"  # silence -> unknown is the rule


def test_reason_passes_sample_rate_through(client, fake_wrapper):
    fake_wrapper._return_text = '{"event": "other", "reason": "unknown", "confidence": 0.5}'
    r = client.post("/reason", json={
        "audio_wav_b64": _wav_b64(baby_cry(1.0)),
        "sample_rate": 16000,
    })
    assert r.status_code == 200
    assert fake_wrapper._calls[-1]["sample_rate"] == 16000
    assert fake_wrapper._calls[-1]["max_new_tokens"] == 256
    assert fake_wrapper._calls[-1]["timeout_s"] == 30.0


def test_reason_accepts_prompt_override(client, fake_wrapper):
    fake_wrapper._return_text = '{"event": "speech", "reason": "unknown", "confidence": 0.4}'
    r = client.post("/reason", json={
        "audio_wav_b64": _wav_b64(baby_cry(0.5)),
        "prompt": "you are a strict classifier, no prose, JSON only please",
    })
    assert r.status_code == 200
    assert fake_wrapper._calls[-1]["prompt_override"] == "you are a strict classifier, no prose, JSON only please"


# ---- /reason: parse-fallback path (the contract says: NEVER 5xx for bad JSON) ----

def test_reason_unparseable_model_text_falls_back_to_other(client, fake_wrapper):
    """Model returned prose / markdown / nothing — 200 with safe defaults."""
    fake_wrapper._return_text = "Sure! Here is the JSON:\n```json\n{not json at all}\n```"
    r = client.post("/reason", json={"audio_wav_b64": _wav_b64(baby_cry(0.5))})
    assert r.status_code == 200
    body = r.json()
    assert body["event"] == "other"
    assert body["reason"] == "unknown"
    assert body["confidence"] == 0.0
    assert body["transcription"] == ""
    assert "Sure!" in body["raw_text"]


def test_reason_empty_model_text_falls_back_to_other(client, fake_wrapper):
    fake_wrapper._return_text = ""
    r = client.post("/reason", json={"audio_wav_b64": _wav_b64(baby_cry(0.5))})
    assert r.status_code == 200
    body = r.json()
    assert body["event"] == "other"
    assert body["reason"] == "unknown"
    assert body["confidence"] == 0.0


def test_reason_partial_json_is_extracted_from_surrounding_text(client, fake_wrapper):
    fake_wrapper._return_text = (
        'Let me analyze...\n'
        '{"event": "dog", "reason": "unknown", "confidence": 0.82, "transcription": "woof"}\n'
        "Hope that helps!"
    )
    r = client.post("/reason", json={"audio_wav_b64": _wav_b64(baby_cry(0.5))})
    assert r.status_code == 200
    body = r.json()
    assert body["event"] == "dog"
    assert body["confidence"] == pytest.approx(0.82)
    assert body["transcription"] == "woof"


# ---- /reason: coercion / safety ----

def test_reason_coerces_unknown_event_to_other(client, fake_wrapper):
    fake_wrapper._return_text = '{"event": "spaceship", "reason": "unknown", "confidence": 0.5}'
    r = client.post("/reason", json={"audio_wav_b64": _wav_b64(baby_cry(0.5))})
    assert r.status_code == 200
    assert r.json()["event"] == "other"


def test_reason_coerces_unknown_reason_to_unknown(client, fake_wrapper):
    fake_wrapper._return_text = '{"event": "baby_cry", "reason": "alien_abduction", "confidence": 0.5}'
    r = client.post("/reason", json={"audio_wav_b64": _wav_b64(baby_cry(0.5))})
    assert r.status_code == 200
    assert r.json()["reason"] == "unknown"


def test_reason_clamps_confidence_above_1(client, fake_wrapper):
    fake_wrapper._return_text = '{"event": "speech", "reason": "unknown", "confidence": 1.7}'
    r = client.post("/reason", json={"audio_wav_b64": _wav_b64(baby_cry(0.5))})
    assert r.status_code == 200
    assert r.json()["confidence"] == 1.0


def test_reason_clamps_confidence_below_0(client, fake_wrapper):
    fake_wrapper._return_text = '{"event": "speech", "reason": "unknown", "confidence": -0.4}'
    r = client.post("/reason", json={"audio_wav_b64": _wav_b64(baby_cry(0.5))})
    assert r.status_code == 200
    assert r.json()["confidence"] == 0.0


def test_reason_handles_missing_confidence_key(client, fake_wrapper):
    fake_wrapper._return_text = '{"event": "speech", "reason": "unknown"}'
    r = client.post("/reason", json={"audio_wav_b64": _wav_b64(baby_cry(0.5))})
    assert r.status_code == 200
    assert r.json()["confidence"] == 0.0


def test_reason_caps_long_transcription(client, fake_wrapper):
    long = "x" * 5000
    fake_wrapper._return_text = (
        f'{{"event": "speech", "reason": "unknown", "confidence": 0.5, '
        f'"transcription": "{long}"}}'
    )
    r = client.post("/reason", json={"audio_wav_b64": _wav_b64(baby_cry(0.5))})
    assert r.status_code == 200
    assert len(r.json()["transcription"]) <= 1024


# ---- /reason: error paths ----

def test_reason_rejects_bad_base64(client):
    r = client.post("/reason", json={"audio_wav_b64": "not-base64!!"})
    assert r.status_code == 400
    assert "invalid base64 wav" in r.json()["detail"]


def test_reason_rejects_audio_above_hard_cap(client):
    """35 s of audio exceeds the hard cap (gemma-4 max is 30 s)."""
    r = client.post("/reason", json={"audio_wav_b64": _wav_b64(silence(40.0))})
    assert r.status_code == 400
    assert "hard cap" in r.json()["detail"]


def test_reason_accepts_audio_just_under_cap(client, fake_wrapper):
    fake_wrapper._return_text = '{"event": "speech", "reason": "unknown", "confidence": 0.5}'
    r = client.post("/reason", json={"audio_wav_b64": _wav_b64(silence(30.0))})
    assert r.status_code == 200
    assert r.json()["event"] == "speech"


def test_reason_rejects_non_wav_payload(client):
    """Random base64 that isn't a valid WAV header."""
    junk = base64.b64encode(b"this is not a wav file at all").decode()
    r = client.post("/reason", json={"audio_wav_b64": junk})
    assert r.status_code == 400


def test_reason_model_unavailable_returns_503():
    gemma_server._wrapper = _FakeGemmaWrapper(
        is_loaded=True, raise_on_reason=GemmaUnavailable("disk full"),
    )
    c = TestClient(gemma_server.app)
    r = c.post("/reason", json={"audio_wav_b64": _wav_b64(baby_cry(0.5))})
    assert r.status_code == 503
    assert "gemma unavailable" in r.json()["detail"]


def test_reason_model_exception_returns_500():
    gemma_server._wrapper = _FakeGemmaWrapper(
        is_loaded=True, raise_on_reason=RuntimeError("cuda OOM"),
    )
    c = TestClient(gemma_server.app)
    r = c.post("/reason", json={"audio_wav_b64": _wav_b64(baby_cry(0.5))})
    assert r.status_code == 500
    assert "reason failed" in r.json()["detail"]


# ---- /reason: stereo (downmix contract — same as cry_service) ----

def test_reason_accepts_stereo_wav(client, fake_wrapper):
    """Stereo WAVs are downmixed to mono server-side, just like cry_service."""
    fake_wrapper._return_text = '{"event": "speech", "reason": "unknown", "confidence": 0.5}'
    sr = SR
    pcm_mono = (np.clip(baby_cry(0.5), -1.0, 1.0) * 32767.0).astype("<i2")
    stereo = np.stack([pcm_mono, pcm_mono], axis=1)  # (N, 2)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())
    r = client.post("/reason", json={"audio_wav_b64": base64.b64encode(buf.getvalue()).decode()})
    assert r.status_code == 200
    assert r.json()["event"] == "speech"
