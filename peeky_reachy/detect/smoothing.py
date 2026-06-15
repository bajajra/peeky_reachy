"""Stabilize a noisy per-window classification stream.

Per-window predictions flicker (a cry briefly reads as speech, a cough spikes a
single frame). We vote over a short sliding window and apply hysteresis so a
state only flips on sustained evidence — fewer false alarms, fewer missed cries.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .events import SoundEvent


class TemporalSmoother:
    def __init__(self, window: int = 5):
        self._buf: deque[tuple[SoundEvent, float]] = deque(maxlen=window)

    def update(self, event: SoundEvent, score: float) -> tuple[SoundEvent, float]:
        self._buf.append((event, score))
        agg: dict[SoundEvent, float] = {}
        for e, s in self._buf:
            agg[e] = agg.get(e, 0.0) + s
        best = max(agg, key=agg.get)
        matching = [s for e, s in self._buf if e == best]
        return best, float(np.mean(matching))

    def reset(self) -> None:
        self._buf.clear()


class Hysteresis:
    """Two-threshold latch: turn on at ``enter``, off at ``exit`` (< enter)."""

    def __init__(self, enter: float, exit: float):
        if exit > enter:
            raise ValueError("exit threshold must be <= enter threshold")
        self.enter = enter
        self.exit = exit
        self.active = False

    def update(self, value: float) -> bool:
        if self.active:
            if value < self.exit:
                self.active = False
        elif value >= self.enter:
            self.active = True
        return self.active

    def reset(self) -> None:
        self.active = False
