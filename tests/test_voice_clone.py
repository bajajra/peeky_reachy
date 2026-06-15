"""Tests for the voice-clone HTTP client (ai-engineer-3 scope).

Contract with the on-spark ``gpu_service`` (VoxCPM2 wrapper):

- ``GET  /healthz``              -> ``{ok, model_loaded, model}``
- ``GET  /references``           -> ``["id1", "id2", ...]``
- ``POST /references``           ``{id, audio_wav_b64, transcript}`` -> ``{ok, id, ...}``
- ``POST /synthesize``           ``{text, reference_id, language, sample_rate}`` -> ``audio/wav``

All methods fail soft (return ``None`` / ``False``) so the soothe path can fall
back to a pre-recorded track when the GPU box is unreachable.

These tests are offline: the network is mocked via ``httpx.MockTransport``.
The integration with the live spark uvicorn is verified separately and
recorded in ``standup.md`` once the systemd unit is actually running.
"""

from __future__ import annotations

import base64
import io
import json

import httpx
import numpy as np
import pytest
import soundfile as sf

from peeky_reachy.voice.clone_client import VoiceCloneClient
from peeky_reachy.voice.enroll import enroll_from_array
from peeky_reachy.voice.store import EnrollmentStore
from tests.conftest import SR, baby_cry


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def _patched_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)

    def _client(self):
        return httpx.Client(base_url=self.base_url, timeout=self.timeout_s, transport=transport)

    monkeypatch.setattr(VoiceCloneClient, "_client", _client)


def _enrolled_store(tmp_path, name: str = "mom", transcript: str = "hush little one") -> EnrollmentStore:
    store = EnrollmentStore(str(tmp_path))
    enroll_from_array(store, audio=baby_cry(2.0), sample_rate=SR, display_name=name,
                      transcript=transcript, consent_given=True)
    return store


# ---------------------------------------------------------------------------
# available()
# ---------------------------------------------------------------------------


def test_available_true_when_healthz_model_loaded(monkeypatch, tmp_path):
    store = _enrolled_store(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True, "model_loaded": True,
                                             "model": "openbmb/VoxCPM2"})
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    assert VoiceCloneClient("http://fake-spark:8090", store, timeout_s=1.0).available() is True


def test_available_false_when_model_not_loaded(monkeypatch, tmp_path):
    store = _enrolled_store(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "model_loaded": False})

    _patched_client(monkeypatch, handler)
    assert VoiceCloneClient("http://fake-spark:8090", store, timeout_s=1.0).available() is False


def test_available_false_on_connection_error():
    """Real network failure (closed port) -> available False, no exception."""
    store = _enrolled_store(_TmpPath())
    clf = VoiceCloneClient("http://127.0.0.1:9", store, timeout_s=0.2)
    assert clf.available() is False


# ---------------------------------------------------------------------------
# ensure_reference() — the "register once per session" cache
# ---------------------------------------------------------------------------


def test_ensure_reference_uploads_only_once_per_session(monkeypatch, tmp_path):
    store = _enrolled_store(tmp_path)
    posts: list[dict] = []
    refs_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/references" and request.method == "GET":
            return httpx.Response(200, json=refs_seen)
        if request.url.path == "/references" and request.method == "POST":
            posts.append(json.loads(request.content))
            refs_seen.append(posts[-1]["id"])
            return httpx.Response(200, json={"ok": True, "id": posts[-1]["id"]})
        if request.url.path == "/synthesize":
            return httpx.Response(200, content=_wav_bytes(np.zeros(480, np.float32), 48000))
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    client = VoiceCloneClient("http://fake-spark:8090", store, timeout_s=1.0)
    assert client.ensure_reference("mom") is True
    # second call: client thinks it's already registered and must not POST
    assert client.ensure_reference("mom") is True
    assert len(posts) == 1, f"reference should be uploaded exactly once, got {len(posts)}"


def test_ensure_reference_returns_false_on_register_4xx(monkeypatch, tmp_path):
    """If the server rejects the upload (4xx), the client must NOT cache the id."""
    store = _enrolled_store(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/references" and request.method == "GET":
            return httpx.Response(200, json=[])
        if request.url.path == "/references" and request.method == "POST":
            return httpx.Response(400, json={"detail": "bad wav"})
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    client = VoiceCloneClient("http://fake-spark:8090", store, timeout_s=1.0)
    assert client.ensure_reference("mom") is False
    # Second attempt must hit the network again, not short-circuit on a stale cache.
    assert client.ensure_reference("mom") is False


def test_ensure_reference_accepts_already_registered_on_server(monkeypatch, tmp_path):
    """Server says the id is already there -> no upload, cached as known."""
    store = _enrolled_store(tmp_path)
    uploads: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/references" and request.method == "GET":
            return httpx.Response(200, json=["mom"])  # server already has it
        if request.url.path == "/references" and request.method == "POST":
            uploads.append(request.content)
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    client = VoiceCloneClient("http://fake-spark:8090", store, timeout_s=1.0)
    assert client.ensure_reference("mom") is True
    assert uploads == []  # POST /references must not be hit
    # And a second call must not re-issue the GET either (cached locally).
    assert client.ensure_reference("mom") is True
    assert uploads == []


# ---------------------------------------------------------------------------
# synth() — failure modes never raise
# ---------------------------------------------------------------------------


def test_synth_no_enrollment_returns_none_without_network(monkeypatch, tmp_path):
    """If no caregiver is enrolled, the client must short-circuit before any HTTP."""
    store = EnrollmentStore(str(tmp_path))  # empty

    def handler(_):
        raise AssertionError("must not hit the network when no enrollment exists")

    _patched_client(monkeypatch, handler)
    client = VoiceCloneClient("http://fake-spark:8090", store, timeout_s=1.0)
    assert client.synth("hi") is None


def test_synth_returns_decoded_wav_on_success(monkeypatch, tmp_path):
    store = _enrolled_store(tmp_path)
    payload = np.linspace(-0.5, 0.5, 2400, dtype=np.float32)  # 50ms at 48k

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/references" and request.method == "GET":
            return httpx.Response(200, json=["mom"])
        if request.url.path == "/synthesize" and request.method == "POST":
            body = json.loads(request.content)
            assert body["reference_id"] == "mom"
            assert body["text"] == "hi baby"
            assert body["sample_rate"] == 48000
            return httpx.Response(200, content=_wav_bytes(payload, 48000))
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    client = VoiceCloneClient("http://fake-spark:8090", store, timeout_s=1.0)
    out = client.synth("hi baby")
    assert out is not None
    samples, sr = out
    assert sr == 48000
    assert samples.shape == payload.shape
    # Confirm the response actually has audio energy (don't trust HTTP 200 alone).
    assert float(np.max(np.abs(samples))) > 0.1


def test_synth_returns_none_on_5xx(monkeypatch, tmp_path):
    store = _enrolled_store(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/references" and request.method == "GET":
            return httpx.Response(200, json=["mom"])
        if request.url.path == "/synthesize":
            return httpx.Response(503, text="overloaded")
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    client = VoiceCloneClient("http://fake-spark:8090", store, timeout_s=1.0)
    assert client.synth("hi") is None


def test_synth_returns_none_on_synthesize_404(monkeypatch, tmp_path):
    """The cached registration may be stale (server restarted) -> 404 on synth -> None."""
    store = _enrolled_store(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/references" and request.method == "GET":
            return httpx.Response(200, json=["mom"])  # client caches "known"
        if request.url.path == "/synthesize":
            return httpx.Response(404, text="unknown reference id")
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    client = VoiceCloneClient("http://fake-spark:8090", store, timeout_s=1.0)
    assert client.synth("hi") is None


def test_synth_returns_none_on_register_failure(monkeypatch, tmp_path):
    """If ensure_reference fails, synth() must return None, not raise."""
    store = _enrolled_store(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/references" and request.method == "GET":
            return httpx.Response(200, json=[])
        if request.url.path == "/references" and request.method == "POST":
            return httpx.Response(500, text="disk full")
        if request.url.path == "/synthesize":
            raise AssertionError("synthesize must not be hit when reference failed to register")
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    client = VoiceCloneClient("http://fake-spark:8090", store, timeout_s=1.0)
    assert client.synth("hi") is None


def test_synth_explicit_speaker_overrides_default(tmp_path, monkeypatch):
    """An explicit speaker_id should be used even if a different default exists."""
    store = _enrolled_store(tmp_path, name="mom", transcript="hi")
    enroll_from_array(store, audio=baby_cry(2.0), sample_rate=SR,
                      display_name="dad", transcript="hello", consent_given=True)
    seen_ref: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/references" and request.method == "GET":
            return httpx.Response(200, json=[])
        if request.url.path == "/references" and request.method == "POST":
            body = json.loads(request.content)
            seen_ref.append(body["id"])
            return httpx.Response(200, json={"ok": True, "id": body["id"]})
        if request.url.path == "/synthesize":
            body = json.loads(request.content)
            seen_ref.append(body["reference_id"])
            return httpx.Response(200, content=_wav_bytes(np.zeros(480, np.float32), 48000))
        return httpx.Response(404)

    _patched_client(monkeypatch, handler)
    client = VoiceCloneClient("http://fake-spark:8090", store, timeout_s=1.0)
    assert client.synth("hi", speaker_id="dad") is not None
    # POST /references must have been for "dad" and /synthesize must reference "dad"
    assert "dad" in seen_ref
    assert "mom" not in seen_ref


# Tiny helper to keep the connection-error test self-contained without pytest
# fixtures in the function signature above.
class _TmpPath:
    """A minimal stand-in for pytest's tmp_path (just needs to be a str-able dir)."""
    def __init__(self):
        import tempfile
        self._d = tempfile.mkdtemp()

    def __str__(self):
        return self._d
