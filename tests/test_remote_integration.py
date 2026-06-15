"""Integration tests for the client <-> remote-service HTTP paths.

These tests mock the network transport with `httpx.MockTransport`, so they run
fully offline and exercise the real `RemoteEventClassifier` / `VoiceCloneClient`
code paths (URL shape, JSON contract, error fallbacks).
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
from peeky_reachy.detect.remote_classifier import RemoteEventClassifier
from peeky_reachy.voice.clone_client import VoiceCloneClient
from peeky_reachy.voice.enroll import enroll_from_array
from peeky_reachy.voice.store import EnrollmentStore
from tests.conftest import SR, baby_cry, silence


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def _patch_clients(monkeypatch, *targets, transport):
    """Patch each target module's `_client` factory to use `transport`."""
    base_url_seen = {}

    def make_factory(orig_self_ref):
        def _client(self):
            base_url_seen.setdefault("url", self.base_url)
            return httpx.Client(base_url=self.base_url,
                                timeout=self.timeout_s,
                                transport=transport)
        return _client

    for cls in targets:
        monkeypatch.setattr(cls, "_client", make_factory(cls))
    return base_url_seen


# -------------------- RemoteEventClassifier --------------------


def test_remote_classifier_happy_path_uses_service_score(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True, "model_loaded": True,
                                             "model": "yamnet"})
        if request.url.path == "/classify":
            body = json.loads(request.content)
            # service contract: audio_wav_b64 is required and base64-decodable
            assert "audio_wav_b64" in body and base64.b64decode(body["audio_wav_b64"])
            return httpx.Response(200, json={"event": "baby_cry", "score": 0.91})
        return httpx.Response(404)

    _patch_clients(monkeypatch, RemoteEventClassifier,
                   transport=_mock_transport(handler))

    clf = RemoteEventClassifier("http://fake-turing:8080", timeout_s=1.0,
                                fallback=HeuristicClassifier())
    assert clf.available() is True
    event, score = clf.classify(baby_cry(1.0), SR)
    assert event == SoundEvent.BABY_CRY
    assert score == pytest.approx(0.91)
    assert ("GET", "/healthz") in calls
    assert ("POST", "/classify") in calls


def test_remote_classifier_falls_back_when_service_500s(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/classify":
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"ok": True, "model_loaded": True})

    _patch_clients(monkeypatch, RemoteEventClassifier,
                   transport=_mock_transport(handler))

    fallback = HeuristicClassifier()
    clf = RemoteEventClassifier("http://fake-turing:8080", timeout_s=1.0, fallback=fallback)
    event, _ = clf.classify(baby_cry(1.0), SR)
    # The heuristic still recognises the cry-shaped signal.
    assert event == SoundEvent.BABY_CRY


def test_remote_classifier_available_false_when_model_not_loaded(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "model_loaded": False})

    _patch_clients(monkeypatch, RemoteEventClassifier,
                   transport=_mock_transport(handler))
    clf = RemoteEventClassifier("http://fake-turing:8080", timeout_s=1.0)
    assert clf.available() is False


def test_remote_classifier_handles_short_window(monkeypatch):
    """A very short audio window should still encode + post cleanly."""
    received_bytes = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        raw = base64.b64decode(body["audio_wav_b64"])
        with wave.open(io.BytesIO(raw), "rb") as wf:
            received_bytes["nframes"] = wf.getnframes()
            received_bytes["rate"] = wf.getframerate()
        return httpx.Response(200, json={"event": "silence", "score": 0.05})

    _patch_clients(monkeypatch, RemoteEventClassifier,
                   transport=_mock_transport(handler))
    clf = RemoteEventClassifier("http://fake-turing:8080", timeout_s=1.0)
    event, score = clf.classify(np.zeros(64, dtype=np.float32), SR)
    assert event == SoundEvent.SILENCE
    assert score == pytest.approx(0.05)
    assert received_bytes["nframes"] == 64
    assert received_bytes["rate"] == SR


# -------------------- VoiceCloneClient --------------------


def _wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, samples.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def _enrolled_store(tmp_path, name="Mom") -> EnrollmentStore:
    store = EnrollmentStore(str(tmp_path))
    enroll_from_array(store, audio=baby_cry(2.0), sample_rate=SR, display_name=name,
                      transcript="hush little one", consent_given=True)
    return store


def test_voice_clone_registers_then_synthesizes(monkeypatch, tmp_path):
    store = _enrolled_store(tmp_path)
    registered = {"refs": []}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/healthz":
            return httpx.Response(200, json={"ok": True, "model_loaded": True,
                                             "model": "openbmb/VoxCPM2"})
        if path == "/references" and request.method == "GET":
            return httpx.Response(200, json=registered["refs"])
        if path == "/references" and request.method == "POST":
            body = json.loads(request.content)
            assert body["id"] == "mom"
            assert body["transcript"] == "hush little one"
            # base64 must decode to a real WAV
            base64.b64decode(body["audio_wav_b64"])
            registered["refs"].append(body["id"])
            return httpx.Response(200, json={"ok": True})
        if path == "/synthesize" and request.method == "POST":
            body = json.loads(request.content)
            assert body["reference_id"] == "mom"
            assert body["language"] == "en"
            assert body["sample_rate"] == 48000
            assert body["text"] == "Shhh, you're safe."
            return httpx.Response(200, content=_wav_bytes(np.zeros(4800, dtype=np.float32), 48000))
        return httpx.Response(404)

    _patch_clients(monkeypatch, VoiceCloneClient, transport=_mock_transport(handler))

    client = VoiceCloneClient("http://fake-spark:8090", store, timeout_s=1.0)
    assert client.available() is True
    out = client.synth("Shhh, you're safe.", language="en")
    assert out is not None
    samples, sr = out
    assert sr == 48000
    assert isinstance(samples, np.ndarray) and samples.dtype == np.float32
    # Second synth shouldn't re-register the reference.
    client.synth("again", language="en")
    assert registered["refs"] == ["mom"]


def test_voice_clone_skips_register_if_already_known(monkeypatch, tmp_path):
    store = _enrolled_store(tmp_path)
    posts = {"refs": 0, "synth": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/references" and request.method == "GET":
            return httpx.Response(200, json=["mom"])
        if request.url.path == "/references" and request.method == "POST":
            posts["refs"] += 1
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/synthesize":
            posts["synth"] += 1
            return httpx.Response(200, content=_wav_bytes(np.zeros(480, dtype=np.float32), 48000))
        return httpx.Response(404)

    _patch_clients(monkeypatch, VoiceCloneClient, transport=_mock_transport(handler))
    client = VoiceCloneClient("http://fake-spark:8090", store, timeout_s=1.0)
    out = client.synth("hello")
    assert out is not None
    assert posts == {"refs": 0, "synth": 1}


def test_voice_clone_returns_none_when_service_500s(monkeypatch, tmp_path):
    store = _enrolled_store(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/references" and request.method == "GET":
            return httpx.Response(200, json=[])
        if request.url.path == "/references" and request.method == "POST":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/synthesize":
            return httpx.Response(503, text="overloaded")
        return httpx.Response(404)

    _patch_clients(monkeypatch, VoiceCloneClient, transport=_mock_transport(handler))
    client = VoiceCloneClient("http://fake-spark:8090", store, timeout_s=1.0)
    assert client.synth("hello") is None


def test_voice_clone_no_enrollment_returns_none(monkeypatch, tmp_path):
    store = EnrollmentStore(str(tmp_path))  # no caregiver enrolled

    def handler(_):  # should never be called
        raise AssertionError("HTTP must not be hit when no enrollment exists")

    _patch_clients(monkeypatch, VoiceCloneClient, transport=_mock_transport(handler))
    client = VoiceCloneClient("http://fake-spark:8090", store, timeout_s=1.0)
    assert client.synth("hello") is None


def test_voice_clone_available_false_on_network_error(monkeypatch, tmp_path):
    store = _enrolled_store(tmp_path)

    def handler(_):
        raise httpx.ConnectError("simulated network error")

    _patch_clients(monkeypatch, VoiceCloneClient, transport=_mock_transport(handler))
    client = VoiceCloneClient("http://fake-spark:8090", store, timeout_s=1.0)
    assert client.available() is False


# -------------------- Pipeline glued to the (mocked) remote services --------------------


def test_pipeline_uses_remote_classifier_when_configured(monkeypatch):
    """End-to-end: with use_remote_cry=true and a mocked healthy service,
    the pipeline's classifier should be the RemoteEventClassifier instance."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True, "model_loaded": True})
        if request.url.path == "/classify":
            return httpx.Response(200, json={"event": "baby_cry", "score": 0.97})
        return httpx.Response(404)

    _patch_clients(monkeypatch, RemoteEventClassifier,
                   transport=_mock_transport(handler))

    from peeky_reachy.audio.io import ArrayAudioIO
    from peeky_reachy.config import Config
    from peeky_reachy.pipeline import Pipeline

    cfg = Config.from_env()
    cfg.use_remote_cry = True
    cfg.cry_service_url = "http://fake-turing:8080"
    cfg.assets_dir = "/nonexistent"  # force motion-only soothing
    io = ArrayAudioIO(np.concatenate([silence(2.5), baby_cry(5.0)]), SR,
                      cfg.sample_rate, cfg.frame_size)
    events = []
    pipe = Pipeline(cfg, io, voice_client=None, on_soothe=events.append)
    assert isinstance(pipe.classifier, RemoteEventClassifier)
    pipe.run()
    assert len(events) >= 1
    assert events[0].decision.event == SoundEvent.BABY_CRY


def test_pipeline_falls_back_to_local_when_remote_unhealthy(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    _patch_clients(monkeypatch, RemoteEventClassifier,
                   transport=_mock_transport(handler))

    from peeky_reachy.audio.io import ArrayAudioIO
    from peeky_reachy.config import Config
    from peeky_reachy.pipeline import Pipeline

    cfg = Config.from_env()
    cfg.use_remote_cry = True
    cfg.cry_service_url = "http://fake-turing:8080"
    io = ArrayAudioIO(silence(0.1), SR, cfg.sample_rate, cfg.frame_size)
    pipe = Pipeline(cfg, io, voice_client=None)
    # availability check failed -> classifier is local, not remote
    assert not isinstance(pipe.classifier, RemoteEventClassifier)
