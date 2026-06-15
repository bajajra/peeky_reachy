"""Audio capture/playback behind one interface so dev (laptop/file) == robot."""

from __future__ import annotations

import logging
import queue
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("peeky.audio")


def to_mono(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return samples.reshape(-1)


def resample_linear(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or samples.size == 0:
        return samples.astype(np.float32)
    duration = samples.shape[0] / src_sr
    dst_n = int(round(duration * dst_sr))
    src_t = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
    dst_t = np.linspace(0.0, duration, num=dst_n, endpoint=False)
    return np.interp(dst_t, src_t, samples).astype(np.float32)


class AudioIO(ABC):
    """Block-oriented mono audio capture + playback at ``sample_rate``."""

    sample_rate: int
    frame_size: int

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return the next mono float32 frame, or None when the source ends."""

    @abstractmethod
    def play(self, samples: np.ndarray, sample_rate: int) -> None: ...

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


class NullAudioIO(AudioIO):
    """Emits silence forever and discards playback. Useful for sim/headless."""

    def __init__(self, sample_rate: int = 16000, frame_size: int = 1536):
        self.sample_rate = sample_rate
        self.frame_size = frame_size

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self) -> Optional[np.ndarray]:
        return np.zeros(self.frame_size, dtype=np.float32)

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        log.info("[null-audio] would play %.2fs of audio", len(samples) / max(sample_rate, 1))


class FileAudioIO(AudioIO):
    """Streams a wav file frame-by-frame; records playback for inspection."""

    def __init__(self, wav_path: str, sample_rate: int = 16000, frame_size: int = 1536,
                 output_dir: Optional[str] = None):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._path = wav_path
        self._cursor = 0
        self._data: np.ndarray = np.zeros(0, dtype=np.float32)
        self._output_dir = output_dir
        self.played: list[np.ndarray] = []

    def start(self) -> None:
        import soundfile as sf

        data, sr = sf.read(self._path, dtype="float32", always_2d=False)
        self._data = resample_linear(to_mono(data), sr, self.sample_rate)
        self._cursor = 0

    def stop(self) -> None:
        self._cursor = len(self._data)

    def read(self) -> Optional[np.ndarray]:
        if self._cursor >= len(self._data):
            return None
        chunk = self._data[self._cursor:self._cursor + self.frame_size]
        self._cursor += self.frame_size
        if len(chunk) < self.frame_size:
            chunk = np.pad(chunk, (0, self.frame_size - len(chunk)))
        return chunk.astype(np.float32)

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        samples = to_mono(samples)
        self.played.append(samples)
        log.info("[file-audio] captured %.2fs of soothing playback", len(samples) / max(sample_rate, 1))
        if self._output_dir:
            import soundfile as sf

            out = Path(self._output_dir)
            out.mkdir(parents=True, exist_ok=True)
            sf.write(out / f"played_{len(self.played):03d}.wav", samples, sample_rate)


class ArrayAudioIO(AudioIO):
    """Streams an in-memory numpy array frame-by-frame; records playback.

    Used by the web app to push an uploaded/recorded clip through the pipeline
    without touching disk.
    """

    def __init__(self, samples: np.ndarray, src_sample_rate: int,
                 sample_rate: int = 16000, frame_size: int = 1536):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._data = resample_linear(to_mono(samples), src_sample_rate, sample_rate)
        self._cursor = 0
        self.played: list[tuple[np.ndarray, int]] = []

    def start(self) -> None:
        self._cursor = 0

    def stop(self) -> None:
        self._cursor = len(self._data)

    def read(self) -> Optional[np.ndarray]:
        if self._cursor >= len(self._data):
            return None
        chunk = self._data[self._cursor:self._cursor + self.frame_size]
        self._cursor += self.frame_size
        if len(chunk) < self.frame_size:
            chunk = np.pad(chunk, (0, self.frame_size - len(chunk)))
        return chunk.astype(np.float32)

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        self.played.append((to_mono(samples), sample_rate))


class LocalAudioIO(AudioIO):
    """Laptop mic/speaker via sounddevice (dev path; robot uses ReachyAudioIO)."""

    def __init__(self, sample_rate: int = 16000, frame_size: int = 1536, channels: int = 1):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.channels = channels
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        def _callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                log.debug("sounddevice status: %s", status)
            try:
                self._q.put_nowait(to_mono(indata.copy()))
            except queue.Full:
                pass

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            channels=self.channels,
            dtype="float32",
            callback=_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=1.0)
        except queue.Empty:
            return np.zeros(self.frame_size, dtype=np.float32)

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        import sounddevice as sd

        sd.play(to_mono(samples), sample_rate)
        sd.wait()


class ReachyAudioIO(AudioIO):
    """Robot 4-mic array + speaker via ``mini.media.*`` (hardware path)."""

    def __init__(self, mini, sample_rate: int = 16000, frame_size: int = 1536):
        self.mini = mini
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._buf = np.zeros(0, dtype=np.float32)

    def start(self) -> None:
        self.mini.media.start_recording()

    def stop(self) -> None:
        try:
            self.mini.media.stop_recording()
        except Exception as exc:  # hardware teardown is best-effort
            log.debug("stop_recording failed: %s", exc)

    def read(self) -> Optional[np.ndarray]:
        sample = self.mini.media.get_audio_sample()
        if sample is None:
            return np.zeros(self.frame_size, dtype=np.float32)
        self._buf = np.concatenate([self._buf, to_mono(sample)])
        if len(self._buf) < self.frame_size:
            return np.zeros(self.frame_size, dtype=np.float32)
        frame, self._buf = self._buf[:self.frame_size], self._buf[self.frame_size:]
        return frame.astype(np.float32)

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        samples = resample_linear(to_mono(samples), sample_rate, self.sample_rate)
        self.mini.media.start_playing()
        self.mini.media.push_audio_sample(samples.reshape(-1, 1))
        threading.Event().wait(len(samples) / self.sample_rate)
        self.mini.media.stop_playing()
