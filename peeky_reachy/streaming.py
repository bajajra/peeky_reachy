"""Always-on streaming soothing mode (T34).

`StreamingSession` is the autonomous-agent harness around the existing
`pipeline.Pipeline` + `soothe.controller.SootheController`. It:

- runs the pipeline in a worker thread driven by a `stop_event`,
- accepts audio in arbitrary chunks from anywhere (`feed(samples, sr)`):
  browser mic stream, headless `LocalAudioIO`, tests with synthetic buffers,
- surfaces a thread-safe `status()` snapshot the UI can poll (current
  companion state — idle/listening/alert/comfort — last sound event, last
  score, last soothe details),
- maps **sound type → action** via an explicit, testable function
  (`action_for_event`) so the type-to-action contract is unit-testable.

We deliberately do NOT reimplement detection or the soothe decision logic —
the pipeline already does that and is covered by tests. This file is glue.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable, Deque, Optional

import numpy as np

from .audio.io import AudioIO, resample_linear, to_mono
from .config import Config
from .detect.events import SoundEvent
from .pipeline import Pipeline, SootheEvent

log = logging.getLogger("peeky.streaming")


# --------------------------------------------------------------------------
# Sound-type → action mapping
# --------------------------------------------------------------------------

# How the autonomous agent responds to each sound type. The full soothe
# decision (sustain, cooldown, hysteresis) is enforced by the pipeline's
# `SootheController` — this map encodes *intent* (which the controller
# enforces for distress and we surface to the UI for everything else).
SOUND_ACTION = {
    SoundEvent.BABY_CRY: "soothe",   # distress: full soothe (cloned voice / fallback track + comfort motion)
    SoundEvent.DOG: "soothe",        # pet distress: same path
    SoundEvent.SPEECH: "listen",     # caregiver / sibling talking: stay aware, no intervention
    SoundEvent.OTHER: "listen",      # positive / unknown valence: don't soothe at a happy/quiet baby
    SoundEvent.SILENCE: "idle",      # nothing to do
}

# Mapping into the 3D Reachy companion's four animation states.
STATE_FOR_EVENT = {
    SoundEvent.BABY_CRY: "alert",
    SoundEvent.DOG: "alert",
    SoundEvent.SPEECH: "listening",
    SoundEvent.OTHER: "listening",
    SoundEvent.SILENCE: "idle",
}


def action_for_event(event: SoundEvent) -> str:
    """Pure, testable mapping from a detected sound type to an action label."""
    return SOUND_ACTION.get(event, "idle")


def state_for_event(event: SoundEvent, *, voiced: bool) -> str:
    """Map an event + voicing flag to a 3D-Reachy companion state."""
    if event in (SoundEvent.BABY_CRY, SoundEvent.DOG):
        return "alert"
    if voiced:
        return "listening"
    return STATE_FOR_EVENT.get(event, "idle")


# --------------------------------------------------------------------------
# Queue-backed AudioIO (the streaming buffer)
# --------------------------------------------------------------------------


class _SilentFrame:
    """Sentinel returned when no data is available within a short timeout —
    keeps the pipeline loop ticking so calibration and clock advance even if
    the browser briefly stops sending chunks."""


@dataclass
class _QueueAudioIO(AudioIO):
    """Streams whatever is `feed()`-ed in, frame-by-frame at `frame_size`.

    Behavior:
    - `read()` returns the next frame if buffered, otherwise blocks up to
      `read_timeout_s` and emits a silent frame so the pipeline doesn't stall.
    - Resampling is per-feed (linear), keeping the contract that frames
      out of the IO are mono at `sample_rate`.
    - `stop()` ends the source (subsequent `read()` returns `None`).
    """

    sample_rate: int = 16000
    frame_size: int = 1536
    read_timeout_s: float = 0.25
    # If True, the IO never returns None — the pipeline.run loop exits only
    # on its `stop_event`. The webapp's "Start/Stop" lifecycle relies on this.
    streaming: bool = True

    def __post_init__(self) -> None:
        self._q: "queue.Queue[object]" = queue.Queue()
        self._buf = np.zeros(0, dtype=np.float32)
        self._closed = False
        self._lock = threading.Lock()
        self.fed_seconds = 0.0
        self.played: list[tuple[np.ndarray, int]] = []

    # --- AudioIO ---
    def start(self) -> None:
        self._closed = False
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)
        # Drain queue.
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        self._closed = True
        # Wake any blocked reader.
        self._q.put(None)

    def read(self) -> Optional[np.ndarray]:
        if self._closed and self._q.empty() and len(self._buf) < self.frame_size:
            return None if not self.streaming else np.zeros(self.frame_size, dtype=np.float32)

        # Pull buffered samples until we have a full frame or run dry.
        deadline = time.monotonic() + self.read_timeout_s
        while len(self._buf) < self.frame_size:
            timeout = max(0.0, deadline - time.monotonic())
            try:
                item = self._q.get(timeout=timeout if timeout > 0 else 0.001)
            except queue.Empty:
                # No data within the window — yield a silent frame so the
                # pipeline keeps its clock; UI sees `silence`.
                return np.zeros(self.frame_size, dtype=np.float32)
            if item is None:
                # Stop sentinel.
                if not self.streaming:
                    return None
                # In streaming mode, pad to a silent frame and continue.
                pad = self.frame_size - len(self._buf)
                self._buf = np.concatenate([self._buf, np.zeros(pad, dtype=np.float32)])
                break
            chunk = np.asarray(item, dtype=np.float32)
            self._buf = np.concatenate([self._buf, chunk])

        with self._lock:
            frame = self._buf[:self.frame_size].astype(np.float32)
            self._buf = self._buf[self.frame_size:]
        return frame

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        # Streaming mode is headless from the server's perspective: we record
        # what would have been played so the UI can surface it. The browser
        # (or robot) handles real playback.
        self.played.append((to_mono(samples), int(sample_rate)))

    # --- feeders ---
    def feed(self, samples: np.ndarray, src_sample_rate: int) -> None:
        if self._closed:
            return
        if samples is None:
            return
        mono = to_mono(samples)
        if mono.size == 0:
            return
        resampled = resample_linear(mono, int(src_sample_rate), self.sample_rate)
        if resampled.size == 0:
            return
        self.fed_seconds += resampled.size / float(self.sample_rate)
        self._q.put(resampled.astype(np.float32, copy=False))


# --------------------------------------------------------------------------
# Status snapshot
# --------------------------------------------------------------------------


@dataclass
class StreamingStatus:
    running: bool = False
    state: str = "idle"
    last_event: str = SoundEvent.SILENCE.value
    last_score: float = 0.0
    last_snr_db: float = 0.0
    voiced: bool = False
    clock: float = 0.0
    soothe_count: int = 0
    last_soothe_at: Optional[float] = None
    last_soothe_event: Optional[str] = None
    last_phrase: Optional[str] = None
    used_clone: bool = False
    cooldown_remaining: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# StreamingSession
# --------------------------------------------------------------------------


class StreamingSession:
    """Always-on streaming wrapper around `Pipeline`.

    Typical use (UI):

        sess = StreamingSession(Config.from_env())
        sess.start()
        # ... browser mic chunks arrive
        sess.feed(samples, sr)
        sess.status()  # poll for UI
        sess.stop()
    """

    def __init__(self, config: Config, *, voice_client=None, mini=None,
                 audio_io: Optional[_QueueAudioIO] = None,
                 on_soothe=None, on_window=None,
                 window_history: int = 50) -> None:
        self.cfg = config
        self._extra_on_soothe = on_soothe
        self._extra_on_window = on_window
        self._lock = threading.Lock()
        self._status = StreamingStatus()
        self._last_soothe: Optional[SootheEvent] = None
        self._soothe_events: list[SootheEvent] = []
        self._windows: Deque[list] = deque(maxlen=window_history)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.audio = audio_io or _QueueAudioIO(
            sample_rate=config.sample_rate, frame_size=config.frame_size)
        self.pipeline = Pipeline(
            config, self.audio, mini=mini, voice_client=voice_client,
            on_window=self._on_window, on_soothe=self._on_soothe,
        )

    # --- callbacks from the pipeline ---
    def _on_window(self, t: float, event: SoundEvent, score: float,
                   snr: float, voiced: bool) -> None:
        row = [round(float(t), 2), event.value, round(float(score), 2),
               round(float(snr), 1), bool(voiced)]
        with self._lock:
            self._windows.append(row)
            ctrl = self.pipeline.controller
            cooldown = max(0.0, float(ctrl.cooldown_remaining(t)))
            self._status.clock = float(t)
            self._status.last_event = event.value
            self._status.last_score = float(score)
            self._status.last_snr_db = float(snr)
            self._status.voiced = bool(voiced)
            self._status.cooldown_remaining = float(cooldown)
            self._status.state = state_for_event(event, voiced=bool(voiced))
        if self._extra_on_window is not None:
            try:
                self._extra_on_window(t, event, score, snr, voiced)
            except Exception:
                log.exception("on_window callback failed")

    def _on_soothe(self, ev: SootheEvent) -> None:
        with self._lock:
            self._last_soothe = ev
            self._soothe_events.append(ev)
            self._status.soothe_count = len(self._soothe_events)
            self._status.last_soothe_at = float(ev.at_seconds)
            self._status.last_soothe_event = ev.decision.event.value
            self._status.last_phrase = ev.phrase
            self._status.used_clone = bool(ev.used_clone)
            self._status.state = "comfort"
        if self._extra_on_soothe:
            try:
                self._extra_on_soothe(ev)
            except Exception:  # callback isolation
                log.exception("on_soothe callback failed")

    # --- lifecycle ---
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        with self._lock:
            self._status = StreamingStatus(running=True)
            self._last_soothe = None
            self._soothe_events.clear()
        self._thread = threading.Thread(
            target=self._run, name="peeky-stream", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        # Wake the audio reader if it's blocked.
        try:
            self.audio.stop()
        except Exception:
            pass
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        self._thread = None
        with self._lock:
            self._status.running = False

    def _run(self) -> None:
        try:
            self.pipeline.run(self._stop)
        except Exception:
            log.exception("streaming pipeline crashed")
        finally:
            with self._lock:
                self._status.running = False

    # --- inputs ---
    def feed(self, samples, sample_rate: int) -> None:
        """Push a chunk of audio (any sample rate / channel count)."""
        self.audio.feed(np.asarray(samples), int(sample_rate))

    # --- outputs ---
    def status(self) -> StreamingStatus:
        with self._lock:
            return StreamingStatus(**asdict(self._status))

    def last_soothe(self) -> Optional[SootheEvent]:
        with self._lock:
            return self._last_soothe

    def soothe_events(self) -> list[SootheEvent]:
        with self._lock:
            return list(self._soothe_events)

    def recent_windows(self) -> list[list]:
        """Snapshot of the last `window_history` per-window rows (read-only)."""
        with self._lock:
            return [list(r) for r in self._windows]
