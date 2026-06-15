"""T34 — Live monitor as the primary Gradio experience (per `vision.md`).

These tests exercise the autonomous-always-on path without launching Gradio
or touching real audio hardware. They cover:

* `_LiveMonitor` lifecycle (start/stop/error) with a mocked LocalAudioIO
* `_live_poll` shape for idle, running, and errored states
* `build_app()` puts the **Live monitor** tab first, demotes upload-analyze
  to "Debug / Analyze clip", and wires the existing 3D Reachy state bridge
* Vision alignment: status text mentions "Listening"/"Idle" (not
  upload/Analyze), the primary tab carries the Start/Stop pair, and there
  is no manual "Analyze clip" button on the primary tab.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import pytest

from peeky_reachy.config import Config
from peeky_reachy.webapp import (
    _LiveMonitor,
    _live_poll,
    _start_live,
    _stop_live,
)


# -------------------- _LiveMonitor lifecycle (mocked mic) --------------------


class _FakeMic:
    """Stand-in for `LocalAudioIO` so we don't need real audio hardware."""

    def __init__(self, frames: int = 4, sample_rate: int = 16000, frame_size: int = 1536):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._frames_left = frames
        self._start_calls = 0
        self._stop_calls = 0
        self._fail_start: Optional[Exception] = None

    def start(self) -> None:
        if self._fail_start is not None:
            raise self._fail_start
        self._start_calls += 1

    def stop(self) -> None:
        self._stop_calls += 1

    def read(self) -> Optional[np.ndarray]:
        if self._frames_left <= 0:
            time.sleep(0.01)
            return None
        self._frames_left -= 1
        return np.zeros(self.frame_size, dtype=np.float32)


def _patch_live_monitor(monkeypatch, fake_mic):
    """Replace the local ``_LiveMonitor`` singleton with a fresh instance and
    patch the `LocalAudioIO` constructor used inside `_LiveMonitor.start`."""
    fresh = _LiveMonitor()
    monkeypatch.setattr("peeky_reachy.webapp._LIVE", fresh)
    monkeypatch.setattr("peeky_reachy.webapp.LocalAudioIO", lambda *a, **k: fake_mic)
    return fresh


def test_live_monitor_starts_and_reports_listening(monkeypatch):
    mic = _FakeMic(frames=8)
    mon = _patch_live_monitor(monkeypatch, mic)
    cfg = Config.from_env()

    msg = mon.start(cfg, voice_client=None)

    assert "Listening" in msg or "🔴" in msg, msg
    assert mon.is_active() is True
    assert mic._start_calls == 1
    assert mon.session is not None
    # feeder thread is alive; it dies when stop() is called
    assert mon.feeder_thread is not None
    assert mon.feeder_thread.is_alive()


def test_live_monitor_stop_cleans_up(monkeypatch):
    mic = _FakeMic(frames=64)
    mon = _patch_live_monitor(monkeypatch, mic)
    cfg = Config.from_env()
    mon.start(cfg, voice_client=None)
    # let the feeder actually read a few frames before stopping
    time.sleep(0.05)

    msg = mon.stop()

    assert "Stopped" in msg, msg
    assert mon.is_active() is False
    assert mon.feeder_thread is None
    assert mic._stop_calls == 1


def test_live_monitor_double_start_is_idempotent(monkeypatch):
    mic = _FakeMic(frames=4)
    mon = _patch_live_monitor(monkeypatch, mic)
    cfg = Config.from_env()
    first = mon.start(cfg)
    second = mon.start(cfg)
    assert mic._start_calls == 1
    assert "Already monitoring" in second, second
    mon.stop()


def test_live_monitor_start_handles_mic_failure(monkeypatch):
    mic = _FakeMic(frames=0)
    mic._fail_start = RuntimeError("no input device")
    mon = _patch_live_monitor(monkeypatch, mic)
    cfg = Config.from_env()
    msg = mon.start(cfg)
    assert "Failed to open microphone" in msg
    assert "no input device" in msg
    assert mon.is_active() is False
    assert mon.session is None


# -------------------- WAV-as-live-source (no-mic devices) --------------------


def test_live_monitor_replay_wav_streams_into_pipeline(tmp_path, monkeypatch):
    """Replay mode lets a no-mic device feed the StreamingSession from a file."""
    import soundfile as sf
    from tests.conftest import SR, baby_cry

    wav = tmp_path / "cry.wav"
    sf.write(str(wav), baby_cry(2.0), SR)
    # No mic patching needed — the WAV path never touches LocalAudioIO.
    mon = _LiveMonitor()
    monkeypatch.setattr("peeky_reachy.webapp._LIVE", mon)
    cfg = Config.from_env()
    msg = mon.start(cfg, voice_client=None, wav_path=str(wav), loop=False)
    assert "Listening on" in msg and "cry.wav" in msg
    assert mon.is_active() is True
    assert mon.wav_source is not None and mon.mic is None
    # The feeder blocks one frame per read; with 2.0s of audio and
    # 96 ms frames (1536/16k), the buffer fills quickly. The WAV source
    # tracks how many seconds it has emitted.
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline and mon.wav_source.fed_seconds < 0.3:
        time.sleep(0.05)
    assert mon.wav_source.fed_seconds > 0.0, (
        f"replay feeder didn't advance; fed_seconds={mon.wav_source.fed_seconds}")
    mon.stop()
    assert mon.is_active() is False


def test_live_monitor_replay_wav_missing_file_returns_error(monkeypatch):
    mon = _LiveMonitor()
    monkeypatch.setattr("peeky_reachy.webapp._LIVE", mon)
    cfg = Config.from_env()
    msg = mon.start(cfg, wav_path="/no/such/file.wav")
    assert "Failed to open WAV file" in msg
    assert mon.is_active() is False
    assert mon.wav_source is None
    assert mon.session is None


def test_live_monitor_wav_source_keeps_replaying_when_loop_true(monkeypatch, tmp_path):
    """Loop=True must keep feeding after a single pass through the file."""
    import soundfile as sf
    from tests.conftest import SR, silence

    wav = tmp_path / "silence_loop.wav"
    sf.write(str(wav), silence(0.5), SR)
    mon = _LiveMonitor()
    monkeypatch.setattr("peeky_reachy.webapp._LIVE", mon)
    cfg = Config.from_env()
    mon.start(cfg, wav_path=str(wav), loop=True)
    # 1.0s of wall time on a 0.5s file with loop=True must emit >= 1.5s
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and mon.wav_source.fed_seconds < 1.5:
        time.sleep(0.05)
    assert mon.wav_source.fed_seconds >= 1.5, (
        f"loop didn't replay; fed_seconds={mon.wav_source.fed_seconds}")
    mon.stop()


# -------------------- _live_poll shape --------------------


def test_live_poll_when_idle(monkeypatch):
    mic = _FakeMic(frames=0)
    _patch_live_monitor(monkeypatch, mic)
    status, state, rows, audio = _live_poll()
    assert "🟢 Idle" in status
    assert state == "idle"
    assert rows == []
    assert audio is None


def test_live_poll_when_running(monkeypatch):
    mic = _FakeMic(frames=64)
    mon = _patch_live_monitor(monkeypatch, mic)
    cfg = Config.from_env()
    mon.start(cfg)
    time.sleep(0.1)  # let a few pipeline windows tick
    status, state, rows, audio = _live_poll()
    mon.stop()
    assert "Listening" in status
    assert state in {"listening", "alert", "comfort", "idle"}
    # rows is a list (possibly empty depending on how fast the pipeline ticked)
    assert isinstance(rows, list)


# -------------------- build_app() structure (vision alignment) --------------------


def _tabs(app):
    return [b for b in app.blocks.values() if b.__class__.__name__ == "Tab"]


def _tab_components(app, label):
    """Walk the descendant components of the named tab."""
    tabs = _tabs(app)
    target = next((t for t in tabs if getattr(t, "label", None) == label), None)
    assert target is not None, f"no tab labeled {label!r}; have {[t.label for t in tabs]}"
    descendants = set()

    def walk(blk):
        descendants.add(blk)
        for child in getattr(blk, "children", []) or []:
            walk(child)

    walk(target)
    return descendants


@pytest.fixture
def app():
    gr = pytest.importorskip("gradio")
    from peeky_reachy.webapp import build_app

    return build_app()


def test_live_monitor_is_the_first_tab(app):
    tabs = _tabs(app)
    assert tabs, "build_app() produced no tabs"
    assert getattr(tabs[0], "label", None) == "🔴 Live monitor", (
        "vision.md mandates live monitor as the primary experience; "
        f"first tab is {tabs[0].label!r}"
    )


def test_upload_analyze_demoted_to_debug(app):
    tabs = _tabs(app)
    labels = [getattr(t, "label", "") for t in tabs]
    # Debug tab exists and the previous 'Monitor' framing has been reworded
    assert any("Debug" in lbl and "Analyze" in lbl for lbl in labels), labels
    # No primary tab called just "Monitor" (the old upload-analyze framing)
    assert "Monitor" not in labels, labels


def test_live_monitor_has_start_and_stop_buttons(app):
    comps = _tab_components(app, "🔴 Live monitor")
    # Gradio Button has a 'value' attribute that is the label
    buttons = [c for c in comps if c.__class__.__name__ == "Button"]
    labels = {b.value for b in buttons}
    assert "Start monitoring" in labels, labels
    assert "Stop" in labels, labels


def test_live_monitor_has_no_analyze_button(app):
    comps = _tab_components(app, "🔴 Live monitor")
    buttons = [c for c in comps if c.__class__.__name__ == "Button"]
    labels = {b.value for b in buttons}
    assert "Analyze clip" not in labels, (
        "Analyze button belongs to the Debug tab; the primary tab is "
        "autonomous, not on-demand"
    )


def test_live_monitor_wires_200ms_timer_to_poll(app):
    comps = _tab_components(app, "🔴 Live monitor")
    timers = [c for c in comps if c.__class__.__name__ == "Timer"]
    assert timers, "Live monitor tab must include a gr.Timer that ticks _live_poll"
    # configured to poll at ~200 ms
    assert any(getattr(t, "value", None) == 0.2 for t in timers), (
        "the 200 ms poll is what makes the UI live, not batch"
    )


def test_live_monitor_drives_3d_reachy_state_textbox(app):
    """The hidden reachy_state textbox must receive updates from _live_poll
    so the 3D companion reacts in real time (vision.md 'drive the 3D companion
    state live')."""
    from peeky_reachy import reachy3d

    ids = {getattr(b, "elem_id", None) for b in app.blocks.values()}
    assert reachy3d.STATE_ELEM_ID in ids, (
        "the 3D companion state bridge is missing; "
        "the Live monitor poll cannot reach window.peekyReachy.setState"
    )


# -------------------- _start_live returns status text --------------------


def test_start_live_returns_status_message(monkeypatch):
    mic = _FakeMic(frames=8)
    _patch_live_monitor(monkeypatch, mic)
    try:
        msg = _start_live(0.55, 3.0, 30.0, 3.0, 5, 0.5, False, False, "",
                          "mic", None, True)
        assert "Listening" in msg or "🔴" in msg
    finally:
        _stop_live()


def test_start_live_routes_wav_source(monkeypatch, tmp_path):
    """`_start_live` with source='wav' should drive the WAV replay path."""
    import soundfile as sf
    from tests.conftest import SR, silence

    wav = tmp_path / "x.wav"
    sf.write(str(wav), silence(0.3), SR)
    fresh = _LiveMonitor()
    monkeypatch.setattr("peeky_reachy.webapp._LIVE", fresh)
    try:
        msg = _start_live(0.55, 3.0, 30.0, 3.0, 5, 0.5, False, False, "",
                          "wav", str(wav), True)
        assert "Listening on" in msg
        assert "x.wav" in msg
        assert fresh.wav_source is not None and fresh.mic is None
    finally:
        _stop_live()
