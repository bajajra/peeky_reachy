"""HTTP-layer tests for the VoxCPM2 GPU service. The model itself is mocked
so this runs on the dev machine (no GPU, no voxcpm dep).

Contract under test — see also standup.md:
  GET  /healthz                  -> {ok, model_loaded, model}
  GET  /references               -> {"references": [id, ...]}
  POST /references               -> 200 / 400 / 409
  POST /synthesize               -> audio/wav / 404 / 503 / 500
"""

from __future__ import annotations

import base64
import io
import wave
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gpu_service import server
from gpu_service.voxwrap import VoxCPMUnavailable, VoxCPMWrapper


# --- helpers --------------------------------------------------------------

def _sine_wav_b64(seconds: float = 0.4, sr: int = 16000, freq: float = 220.0) -> tuple[str, bytes]:
    t = np.arange(int(seconds * sr)) / sr
    samples = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((samples * 32767).astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii"), buf.getvalue()


def _make_fake_wav_on_disk(path: Path, seconds: float = 0.5, sr: int = 16000) -> None:
    t = np.arange(int(seconds * sr)) / sr
    samples = (0.1 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((samples * 32767).astype(np.int16).tobytes())


class _FakeVoxCPM(VoxCPMWrapper):
    """Stub wrapper: never imports voxcpm, never touches the GPU."""

    def __init__(self, references_dir: Path | None = None) -> None:
        super().__init__(
            model_id="fake/voxcpm",
            references_dir=str(references_dir) if references_dir else None,
        )
        # Force is_loaded=True so /healthz reports model_loaded without us
        # having to call ensure_loaded() (which would import voxcpm).
        self._model = object()

    def ensure_loaded(self) -> None:  # type: ignore[override]
        return None

    def synth(self, req):  # type: ignore[override]
        # Mirror the real wrapper's reference-resolve-and-raise behavior so
        # the "missing reference -> 404" path is exercised by the server.
        if req.reference_id:
            ref_wav, _ = self._resolve_reference(req.reference_id)
            if ref_wav is None:
                raise FileNotFoundError(
                    f"reference {req.reference_id!r} not found in {self.references_dir}"
                )
        sr = req.sample_rate or 48000
        n = int(sr * 0.5)
        t = np.arange(n) / sr
        return (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


@pytest.fixture
def refs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "refs"
    d.mkdir()
    return d


@pytest.fixture
def fake_wrapper(refs_dir: Path) -> _FakeVoxCPM:
    w = _FakeVoxCPM(references_dir=refs_dir)
    return w


@pytest.fixture
def client(fake_wrapper: _FakeVoxCPM) -> Iterator[TestClient]:
    server._wrapper = fake_wrapper  # type: ignore[attr-defined]
    server.app.dependency_overrides = {}  # ensure clean
    with TestClient(server.app) as c:
        yield c
    server._wrapper = None  # type: ignore[attr-defined]


# --- /healthz -------------------------------------------------------------

def test_healthz_reports_model_loaded(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["model_loaded"] is True
    assert body["model"] == "fake/voxcpm"


def test_healthz_reports_not_loaded(client: TestClient) -> None:
    real = server._wrapper  # type: ignore[attr-defined]
    real._model = None  # type: ignore[attr-defined]
    try:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is False
    finally:
        real._model = object()  # type: ignore[attr-defined]


# --- /references ----------------------------------------------------------

def test_list_references_empty(client: TestClient) -> None:
    r = client.get("/references")
    assert r.status_code == 200
    assert r.json() == {"references": []}


def test_list_references_picks_up_wav_files(refs_dir: Path, client: TestClient) -> None:
    _make_fake_wav_on_disk(refs_dir / "dad.wav")
    _make_fake_wav_on_disk(refs_dir / "mom.wav")
    _make_fake_wav_on_disk(refs_dir / "ignored.txt")  # not a wav
    r = client.get("/references")
    assert r.status_code == 200
    assert sorted(r.json()["references"]) == ["dad", "mom"]


def test_add_reference_writes_wav_and_transcript(
    refs_dir: Path, client: TestClient
) -> None:
    b64, _raw = _sine_wav_b64()
    r = client.post(
        "/references",
        json={"id": "dad", "audio_wav_b64": b64, "transcript": "hi baby"},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "id": "dad", "has_transcript": True}
    assert (refs_dir / "dad.wav").exists()
    assert (refs_dir / "dad.txt").read_text() == "hi baby"


def test_add_reference_duplicate_rejected(refs_dir: Path, client: TestClient) -> None:
    b64, _ = _sine_wav_b64()
    client.post("/references", json={"id": "dad", "audio_wav_b64": b64})
    r = client.post("/references", json={"id": "dad", "audio_wav_b64": b64})
    assert r.status_code == 409
    assert "already exists" in r.json()["detail"]


def test_add_reference_overwrite_replaces(refs_dir: Path, client: TestClient) -> None:
    b64_a, _ = _sine_wav_b64(freq=220.0)
    b64_b, _ = _sine_wav_b64(freq=440.0)
    client.post("/references", json={"id": "dad", "audio_wav_b64": b64_a})
    r = client.post(
        "/references",
        json={"id": "dad", "audio_wav_b64": b64_b, "overwrite": True},
    )
    assert r.status_code == 200
    # file was replaced
    raw = (refs_dir / "dad.wav").read_bytes()
    assert raw == base64.b64decode(b64_b)


def test_add_reference_invalid_base64(client: TestClient) -> None:
    r = client.post(
        "/references",
        json={"id": "bad", "audio_wav_b64": "!!!not-base64!!!"},
    )
    assert r.status_code == 400


# --- /synthesize ----------------------------------------------------------

def test_synthesize_returns_wav(client: TestClient) -> None:
    r = client.post(
        "/synthesize",
        json={"text": "hi baby", "sample_rate": 16000},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.headers["x-peeky-sample-rate"] == "16000"
    # body should be parseable as a 16-bit PCM mono wav
    with wave.open(io.BytesIO(r.content), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        frames = wf.readframes(wf.getnframes())
        assert len(frames) > 0


def test_synthesize_with_reference_id(refs_dir: Path, client: TestClient) -> None:
    _make_fake_wav_on_disk(refs_dir / "dad.wav")
    (refs_dir / "dad.txt").write_text("hi baby")
    r = client.post(
        "/synthesize",
        json={"text": "shh it's ok", "reference_id": "dad", "sample_rate": 16000},
    )
    assert r.status_code == 200, r.text


def test_synthesize_missing_reference_404(refs_dir: Path, client: TestClient) -> None:
    r = client.post(
        "/synthesize",
        json={"text": "hi", "reference_id": "ghost", "sample_rate": 16000},
    )
    assert r.status_code == 404
    assert "ghost" in r.json()["detail"]


def test_synthesize_model_unavailable_503(
    refs_dir: Path, client: TestClient, fake_wrapper: _FakeVoxCPM
) -> None:
    def boom(_req):  # type: ignore[no-untyped-def]
        raise VoxCPMUnavailable("CUDA OOM")

    fake_wrapper.synth = boom  # type: ignore[method-assign]
    r = client.post("/synthesize", json={"text": "hi", "sample_rate": 16000})
    assert r.status_code == 503
    assert "OOM" in r.json()["detail"]


def test_synthesize_rejects_empty_text(client: TestClient) -> None:
    r = client.post("/synthesize", json={"text": "", "sample_rate": 16000})
    assert r.status_code == 422  # pydantic validation


def test_synthesize_rejects_out_of_range_sample_rate(client: TestClient) -> None:
    r = client.post("/synthesize", json={"text": "hi", "sample_rate": 1000})
    assert r.status_code == 422


# --- / --------------------------------------------------------------------

def test_root_lists_endpoints(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "peeky-voxcpm2"
    assert "/synthesize" in body["endpoints"]
