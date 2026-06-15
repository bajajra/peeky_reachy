import numpy as np
import pytest

from peeky_reachy.detect.remote_classifier import _wav_b64
from tests.conftest import SR, baby_cry

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cry_service import server as cry_server  # noqa: E402
from cry_service.classifier_wrap import ClassifierWrapper  # noqa: E402


class _FakeWrapper(ClassifierWrapper):
    def __init__(self):
        super().__init__()
        self.model_id = "fake-yamnet"

    @property
    def is_loaded(self) -> bool:
        return True

    def ensure_loaded(self):
        return self

    def classify(self, samples, sample_rate):
        return "baby_cry", 0.9


@pytest.fixture
def client():
    cry_server._wrapper = _FakeWrapper()
    return TestClient(cry_server.app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["model_loaded"] is True
    assert body["model"] == "fake-yamnet"


def test_classify_returns_event_and_score(client):
    b64 = _wav_b64(baby_cry(1.0), SR)
    r = client.post("/classify", json={"audio_wav_b64": b64})
    assert r.status_code == 200
    body = r.json()
    assert body["event"] == "baby_cry"
    assert body["score"] == pytest.approx(0.9)


def test_classify_rejects_bad_base64(client):
    r = client.post("/classify", json={"audio_wav_b64": "not-base64!!"})
    assert r.status_code == 400


def test_index(client):
    assert client.get("/").status_code == 200
