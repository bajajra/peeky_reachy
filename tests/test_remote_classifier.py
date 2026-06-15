"""Tests for the remote cry-classification client (ai-engineer-3 scope).

Cover the JSON / audio contract with the on-turing ``cry_service``:
- ``GET  /healthz``   -> ``{ok, model_loaded, model}``
- ``POST /classify``  -> ``{event, score}``

All tests are offline: the network is mocked via ``httpx.MockTransport`` and
exceptions are simulated directly. The fallback path is exercised by pointing
the client at a closed port (``127.0.0.1:9`` is the IANA discard service).
"""

from __future__ import annotations

import base64
import io
import json
import wave

import httpx
import numpy as np
import pytest

from peeky_reachy.detect.classifier import HeuristicClassifier
from peeky_reachy.detect.events import SoundEvent
from peeky_reachy.detect.remote_classifier import RemoteEventClassifier, _wav_b64
from tests.conftest import SR, baby_cry


# ---------------------------------------------------------------------------
# Audio encoder (_wav_b64) — must accept whatever the pipeline might hand it
# ---------------------------------------------------------------------------


def _decode_wav(b64: str) -> tuple[int, int, int, bytes]:
    """Round-trip: decode the b64 string and read the WAV header."""
    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.readframes(wf.getnframes())


def test_wav_b64_float32_mono_short_clip():
    b64 = _wav_b64(np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32), SR)
    ch, sw, rate, frames = _decode_wav(b64)
    assert (ch, sw, rate) == (1, 2, SR)
    pcm = np.frombuffer(frames, dtype=np.int16)
    assert len(pcm) == 4
    # 0.1 * 32767 ≈ 3277
    assert pcm[0] == pytest.approx(3277, abs=2)


def test_wav_b64_int16_passthrough():
    """int16 input should not be silently re-scaled to noise."""
    src = np.array([100, -200, 30000, -30000], dtype=np.int16)
    b64 = _wav_b64(src, SR)
    _, _, _, frames = _decode_wav(b64)
    pcm = np.frombuffer(frames, dtype=np.int16)
    assert pcm.tolist() == src.tolist()


def test_wav_b64_stereo_downmixed_to_mono():
    """A 2-channel input must collapse to 1 channel, not stretch the timeline."""
    stereo = np.array([[0.4, 0.0], [0.0, 0.8], [-0.4, 0.0], [0.0, -0.8]], dtype=np.float32)
    b64 = _wav_b64(stereo, SR)
    ch, _, _, frames = _decode_wav(b64)
    assert ch == 1
    pcm = np.frombuffer(frames, dtype=np.int16)
    # 4 frames (downmixed), not 8 (broadcast-flattened)
    assert len(pcm) == 4
    # Mean of first channel pair: (0.4 + 0.0) / 2 = 0.2 -> ~6553
    assert pcm[0] == pytest.approx(6553, abs=5)


def test_wav_b64_empty_input_is_a_valid_zero_frame_wav():
    """Empty windows still need to encode (server may 400 but the client must not crash)."""
    b64 = _wav_b64(np.zeros(0, dtype=np.float32), SR)
    ch, sw, rate, frames = _decode_wav(b64)
    assert (ch, sw, rate) == (1, 2, SR)
    assert frames == b""


def test_wav_b64_clamps_out_of_range_float():
    """Values outside [-1, 1] must clamp, not wrap or overflow."""
    b64 = _wav_b64(np.array([5.0, -5.0, 0.0], dtype=np.float32), SR)
    _, _, _, frames = _decode_wav(b64)
    pcm = np.frombuffer(frames, dtype=np.int16)
    assert pcm[0] == 32767   # clamp
    assert pcm[1] == -32767  # clamp (asymmetric to avoid 2's-complement wrap)
    assert pcm[2] == 0


# ---------------------------------------------------------------------------
# HTTP / JSON contract (mocked transport)
# ---------------------------------------------------------------------------


def _patched_client(monkeypatch, handler):
    """Replace RemoteEventClassifier._client with a factory that uses MockTransport."""
    transport = httpx.MockTransport(handler)

    def _client(self):
        return httpx.Client(base_url=self.base_url, timeout=self.timeout_s, transport=transport)

    monkeypatch.setattr(RemoteEventClassifier, "_client", _client)


def test_available_true_when_healthz_says_model_loaded(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True, "model_loaded": True, "model": "yamnet"})
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    clf = RemoteEventClassifier("http://fake-turing:8080", timeout_s=1.0)
    assert clf.available() is True


def test_available_false_when_model_not_loaded(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "model_loaded": False})

    _patched_client(monkeypatch, handler)
    assert RemoteEventClassifier("http://x", timeout_s=1.0).available() is False


def test_available_false_on_5xx(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="down")

    _patched_client(monkeypatch, handler)
    assert RemoteEventClassifier("http://x", timeout_s=1.0).available() is False


def test_classify_happy_path_uses_remote_score(monkeypatch):
    """Confident remote result must propagate; local fallback MUST NOT fire."""
    fallback_called = {"n": 0}

    class _Counting(HeuristicClassifier):
        def classify(self, window, sample_rate):
            fallback_called["n"] += 1
            return super().classify(window, sample_rate)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True, "model_loaded": True})
        if request.url.path == "/classify":
            # Confirm the wire shape: a base64 wav, nothing else.
            body = json.loads(request.content)
            assert set(body.keys()) == {"audio_wav_b64"}
            base64.b64decode(body["audio_wav_b64"], validate=True)
            return httpx.Response(200, json={"event": "baby_cry", "score": 0.93})
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    clf = RemoteEventClassifier("http://fake-turing:8080", timeout_s=1.0,
                                fallback=_Counting())
    event, score = clf.classify(baby_cry(1.0), SR)
    assert event == SoundEvent.BABY_CRY
    assert score == pytest.approx(0.93)
    assert fallback_called["n"] == 0, "fallback must not fire when remote returns a score"


def test_classify_5xx_falls_back_silently(monkeypatch):
    """A 500 from the service must become a local classify(), never an exception."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/classify":
            return httpx.Response(500, text="kaboom")
        return httpx.Response(200, json={"ok": True, "model_loaded": True})

    _patched_client(monkeypatch, handler)
    clf = RemoteEventClassifier("http://fake-turing:8080", timeout_s=1.0)
    event, score = clf.classify(baby_cry(1.0), SR)
    assert isinstance(event, SoundEvent)
    assert 0.0 <= score <= 1.0
    assert event == SoundEvent.BABY_CRY  # heuristic picks up the harmonic signal


def test_classify_4xx_falls_back(monkeypatch):
    """Bad-payload 4xx (e.g. server can't decode) must also fall back, not raise."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/classify":
            return httpx.Response(400, json={"detail": "invalid base64 wav"})
        return httpx.Response(200, json={"ok": True, "model_loaded": True})

    _patched_client(monkeypatch, handler)
    clf = RemoteEventClassifier("http://fake-turing:8080", timeout_s=1.0)
    event, _ = clf.classify(baby_cry(1.0), SR)
    assert event == SoundEvent.BABY_CRY


def test_classify_timeout_falls_back():
    """Real network timeout (closed port) must fall back, not raise."""
    clf = RemoteEventClassifier("http://127.0.0.1:9", timeout_s=0.2)  # discard
    assert clf.available() is False
    event, _ = clf.classify(baby_cry(1.0), SR)
    assert event == SoundEvent.BABY_CRY


def test_classify_connection_refused_falls_back():
    """ConnectionRefused (port bound, nothing listening) must also fall back."""
    clf = RemoteEventClassifier("http://127.0.0.1:1", timeout_s=0.2)
    assert clf.available() is False
    # Should not raise
    clf.classify(baby_cry(0.5), SR)


def test_classify_handles_one_sample_window(monkeypatch):
    """A single-sample window must not crash the encoder or the transport."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/classify":
            body = json.loads(request.content)
            raw = base64.b64decode(body["audio_wav_b64"])
            with wave.open(io.BytesIO(raw), "rb") as wf:
                assert wf.getnframes() == 1
            return httpx.Response(200, json={"event": "silence", "score": 0.1})
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    clf = RemoteEventClassifier("http://fake-turing:8080", timeout_s=1.0)
    event, score = clf.classify(np.array([0.01], dtype=np.float32), SR)
    assert event == SoundEvent.SILENCE
    assert score == pytest.approx(0.1)


def test_classify_does_not_re_check_healthz(monkeypatch):
    """``classify()`` must be a single round-trip; it must not also call /healthz."""
    hits = {"healthz": 0, "classify": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            hits["healthz"] += 1
            return httpx.Response(200, json={"ok": True, "model_loaded": True})
        if request.url.path == "/classify":
            hits["classify"] += 1
            return httpx.Response(200, json={"event": "silence", "score": 0.0})
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    clf = RemoteEventClassifier("http://fake-turing:8080", timeout_s=1.0)
    clf.classify(np.zeros(100, dtype=np.float32), SR)
    assert hits == {"healthz": 0, "classify": 1}
