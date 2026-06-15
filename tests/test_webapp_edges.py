"""Edge-case tests for the Gradio webapp glue (`peeky_reachy.webapp`).

The webapp module is pure-Python glue over the pipeline; these tests call its
top-level functions directly without launching a Gradio server.
"""

from __future__ import annotations

import io

import httpx
import numpy as np
import pytest
import soundfile as sf

from peeky_reachy.voice.clone_client import VoiceCloneClient
from peeky_reachy.webapp import (
    _to_float_mono,
    _to_gradio_audio,
    analyze,
    enroll,
    preview,
)
from tests.conftest import SR, baby_cry, silence


def _audio(samples: np.ndarray, sr: int = SR):
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    return (sr, pcm)


# -------------------- _to_float_mono / _to_gradio_audio --------------------


def test_to_float_mono_handles_int16_stereo():
    stereo = np.stack([np.full(100, 16000, dtype=np.int16),
                       np.full(100, -16000, dtype=np.int16)], axis=1)
    out, sr = _to_float_mono((48000, stereo))
    assert sr == 48000
    assert out.shape == (100,)
    assert out.dtype == np.float32
    # mean of +16000 and -16000 -> ~0
    assert abs(out.mean()) < 1e-3


def test_to_float_mono_none():
    out, sr = _to_float_mono(None)
    assert out is None and sr == 16000


def test_to_gradio_audio_clips_and_int16():
    sr, pcm = _to_gradio_audio(np.array([2.0, -2.0, 0.5], dtype=np.float32), 22050)
    assert sr == 22050
    assert pcm.dtype == np.int16
    assert pcm[0] == 32767 and pcm[1] == -32767


# -------------------- _to_float_mono: every shape Gradio 6 may hand us --------------------


def test_to_float_mono_handles_bare_ndarray():
    """Gradio 6 sometimes hands us a bare ndarray; assume 16 kHz default."""
    arr = np.zeros(8000, dtype=np.float32)
    samples, sr = _to_float_mono(arr)
    assert sr == 16000
    assert samples.shape == (8000,)
    assert samples.dtype == np.float32


def test_to_float_mono_handles_file_dict(tmp_path):
    """A Gradio 6 FileData-shaped dict must round-trip through the loader."""
    import soundfile as sf
    wav = tmp_path / "x.wav"
    sf.write(str(wav), np.zeros(8000, dtype=np.float32), 16000)
    payload = {"path": str(wav), "data": None, "orig_name": "x.wav",
               "mime_type": "audio/wav"}
    samples, sr = _to_float_mono(payload)
    assert sr == 16000
    assert samples.shape == (8000,)


def test_to_float_mono_handles_path_string(tmp_path):
    """A path string (e.g. from a typed-filepath component) must load."""
    import soundfile as sf
    wav = tmp_path / "x.wav"
    sf.write(str(wav), np.zeros(8000, dtype=np.float32), 16000)
    samples, sr = _to_float_mono(str(wav))
    assert sr == 16000
    assert samples.shape == (8000,)


def test_to_float_mono_unreadable_path_returns_none():
    samples, sr = _to_float_mono("/no/such/file.wav")
    assert samples is None and sr == 16000


def test_enroll_handles_file_dict_payload(tmp_path):
    """End-to-end: enroll must succeed with a Gradio 6 FileData-shaped dict.

    Regression for the user-reported 'uploading the voice for cloning gave
    an error' bug — Gradio 6.18 may pass a dict for the upload, and the
    pre-fix _to_float_mono crashed on `sr, data = audio`.
    """
    import soundfile as sf
    wav = tmp_path / "voice.wav"
    sf.write(str(wav), 0.05 * np.random.default_rng(0).standard_normal(16000 * 3).astype(np.float32), 16000)
    payload = {"path": str(wav), "data": None, "orig_name": "voice.wav",
               "mime_type": "audio/wav"}
    status, rows = enroll(payload, name="Bug Mom", transcript="hush now little one",
                          language="en", consent=True)
    assert "✅" in status, status
    # enrolled row appears in the table
    assert any(r[0] == "bug-mom" for r in rows), rows


def test_enroll_handles_bare_ndarray_payload():
    arr = 0.05 * np.random.default_rng(1).standard_normal(16000 * 3).astype(np.float32)
    status, _ = enroll(arr, name="BareNumpy", transcript="hush now little one",
                       language="en", consent=True)
    assert "✅" in status, status


def test_enroll_no_audio_returns_clear_message():
    status, _ = enroll(None, name="NoAudio", transcript="hush now little one",
                       language="en", consent=True)
    assert "record or upload" in status.lower(), status


# -------------------- analyze edge cases --------------------


def test_analyze_empty_audio_returns_friendly_message():
    summary, out_audio, rows, plot, state = analyze(None, 0.55, 3.0, 30.0, 3.0, 5, 0.5,
                                                     False, False, "")
    assert "Please record or upload" in summary
    assert out_audio is None
    assert rows == []
    assert plot is None
    assert state == "idle"


def test_analyze_zero_length_array_returns_friendly_message():
    summary, _, _, _, _ = analyze((SR, np.zeros(0, dtype=np.int16)),
                                   0.55, 3.0, 30.0, 3.0, 5, 0.5, False, False, "")
    assert "Please record or upload" in summary


def test_analyze_very_short_clip_does_not_crash():
    # Way too short to ever calibrate or sustain.
    short = (0.1 * np.random.randn(int(0.2 * SR))).astype(np.float32)
    summary, _, rows, _, _ = analyze(_audio(short), 0.55, 3.0, 30.0, 3.0, 5, 0.5,
                                      False, False, "")
    # 0 windows is fine; just don't crash and report "no soothe".
    assert "No soothing triggered" in summary or "Soothing triggered" in summary
    assert isinstance(rows, list)


# -------------------- enroll consent + validation --------------------


def test_enroll_requires_consent():
    msg, rows = enroll(_audio(baby_cry(1.0)), "Dad", "hush now", "en", consent=False)
    assert "Consent required" in msg
    assert isinstance(rows, list)


def test_enroll_requires_name_and_transcript():
    msg, _ = enroll(_audio(baby_cry(1.0)), "", "hush", "en", consent=True)
    assert "Name and an exact transcript" in msg
    msg, _ = enroll(_audio(baby_cry(1.0)), "Dad", "   ", "en", consent=True)
    assert "Name and an exact transcript" in msg


def test_enroll_happy_path_lists_record(tmp_path, monkeypatch):
    monkeypatch.setenv("PEEKY_ENROLLMENT_DIR", str(tmp_path))
    msg, rows = enroll(_audio(baby_cry(2.0)), "Dad", "hush little one", "en",
                       consent=True)
    assert "Enrolled" in msg and "dad" in msg.lower()
    assert any(r[0] == "dad" for r in rows)


# -------------------- preview with mocked voice service --------------------


def _patch_voice_transport(monkeypatch, handler):
    transport = httpx.MockTransport(handler)

    def _client(self):
        return httpx.Client(base_url=self.base_url, timeout=self.timeout_s,
                            transport=transport)

    monkeypatch.setattr(VoiceCloneClient, "_client", _client)


def _wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def test_preview_uses_cloned_voice_when_service_up(monkeypatch, tmp_path):
    from peeky_reachy.voice.enroll import enroll_from_array
    from peeky_reachy.voice.store import EnrollmentStore

    monkeypatch.setenv("PEEKY_ENROLLMENT_DIR", str(tmp_path))
    store = EnrollmentStore(str(tmp_path))
    enroll_from_array(store, audio=baby_cry(2.0), sample_rate=SR,
                      display_name="Mom", transcript="hush", consent_given=True)

    cloned_audio = np.linspace(-0.5, 0.5, 48000, dtype=np.float32)

    def handler(request):
        if request.url.path == "/references" and request.method == "GET":
            return httpx.Response(200, json=["mom"])
        if request.url.path == "/synthesize":
            return httpx.Response(200, content=_wav_bytes(cloned_audio, 48000))
        return httpx.Response(404)

    _patch_voice_transport(monkeypatch, handler)
    status, audio = preview("Shhh, little one.", "")
    assert "cloned voice" in status
    sr, pcm = audio
    assert sr == 48000
    assert pcm.dtype == np.int16
    assert pcm.shape == (48000,)


def test_preview_falls_back_to_track_when_service_down(monkeypatch, tmp_path):
    monkeypatch.setenv("PEEKY_ENROLLMENT_DIR", str(tmp_path))
    monkeypatch.setenv("PEEKY_ASSETS_DIR", "assets/soothing")

    def handler(_):
        raise httpx.ConnectError("nope")

    _patch_voice_transport(monkeypatch, handler)
    status, audio = preview("Shhh, little one.", "")
    # No enrollment -> synth returns None immediately; fallback track present.
    assert "fallback" in status.lower() or "no" in status.lower()
    # If assets dir has the default hum, audio is delivered.
    if audio is not None:
        sr, pcm = audio
        assert sr > 0 and pcm.dtype == np.int16


def test_preview_no_voice_no_assets(monkeypatch, tmp_path):
    monkeypatch.setenv("PEEKY_ENROLLMENT_DIR", str(tmp_path))
    monkeypatch.setenv("PEEKY_ASSETS_DIR", str(tmp_path / "missing"))

    def handler(_):
        raise httpx.ConnectError("nope")

    _patch_voice_transport(monkeypatch, handler)
    status, audio = preview("Shhh.", "")
    assert "No GPU voice" in status or "no fallback" in status.lower()
    assert audio is None


# -------------------- analyze with use_voice=True (mocked GPU) --------------------


def test_analyze_with_use_voice_uses_clone(monkeypatch, tmp_path):
    """When use_voice=True and the GPU is reachable, analyze() should report
    a cloned voice soothing."""
    from peeky_reachy.voice.enroll import enroll_from_array
    from peeky_reachy.voice.store import EnrollmentStore

    monkeypatch.setenv("PEEKY_ENROLLMENT_DIR", str(tmp_path))
    store = EnrollmentStore(str(tmp_path))
    enroll_from_array(store, audio=baby_cry(2.0), sample_rate=SR,
                      display_name="Mom", transcript="hush", consent_given=True)

    def handler(request):
        if request.url.path == "/references" and request.method == "GET":
            return httpx.Response(200, json=["mom"])
        if request.url.path == "/synthesize":
            return httpx.Response(200,
                                  content=_wav_bytes(np.zeros(4800, dtype=np.float32), 48000))
        return httpx.Response(404)

    _patch_voice_transport(monkeypatch, handler)
    clip = np.concatenate([silence(2.5), baby_cry(5.0)])
    summary, soothe_audio, _, _, _ = analyze(_audio(clip), 0.55, 3.0, 30.0, 3.0, 5, 0.5,
                                              False, True, "http://fake-spark:8090")
    assert "Soothing triggered" in summary
    assert "caregiver clone" in summary
    assert soothe_audio is not None
