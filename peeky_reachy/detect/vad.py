"""Always-on voice/sound activity gate: Silero when available, energy fallback."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger("peeky.vad")


class VAD(ABC):
    @abstractmethod
    def is_active(self, frame: np.ndarray) -> tuple[bool, float]:
        """Return (active, probability) for a mono float32 frame."""


class EnergyVAD(VAD):
    """RMS gate with an adaptive noise floor. No ML deps."""

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5,
                 abs_floor: float = 1e-3):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.abs_floor = abs_floor
        self._noise = abs_floor

    def is_active(self, frame: np.ndarray) -> tuple[bool, float]:
        rms = float(np.sqrt(np.mean(np.square(frame))) + 1e-9)
        ratio = rms / max(self._noise, self.abs_floor)
        prob = float(np.clip((ratio - 2.0) / 8.0, 0.0, 1.0))
        active = rms > self.abs_floor and prob >= self.threshold
        # Only adapt the floor on quiet frames so a sustained loud cry can't
        # raise it and gate itself out; let it fall faster than it rises.
        if not active:
            rate = 0.05 if rms < self._noise else 0.01
            self._noise = (1 - rate) * self._noise + rate * rms
        return active, prob


class SileroVAD(VAD):
    """Wraps snakers4/silero-vad (operates on 512-sample 16 kHz chunks)."""

    CHUNK = 512

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        from silero_vad import load_silero_vad  # raises if not installed
        import torch  # noqa: F401

        if sample_rate != 16000:
            raise ValueError("SileroVAD requires 16 kHz audio")
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._model = load_silero_vad()
        self._torch = __import__("torch")

    def is_active(self, frame: np.ndarray) -> tuple[bool, float]:
        prob = 0.0
        for start in range(0, len(frame) - self.CHUNK + 1, self.CHUNK):
            chunk = frame[start:start + self.CHUNK]
            t = self._torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))
            prob = max(prob, float(self._model(t, self.sample_rate).item()))
        return prob >= self.threshold, prob


def make_vad(sample_rate: int, threshold: float, prefer_ml: bool = True) -> VAD:
    if prefer_ml:
        try:
            vad = SileroVAD(sample_rate, threshold)
            log.info("VAD: using Silero")
            return vad
        except Exception as exc:
            log.info("VAD: Silero unavailable (%s); using EnergyVAD fallback", exc)
    return EnergyVAD(sample_rate, threshold)
