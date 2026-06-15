"""Front-end signal conditioning so detection generalizes across rooms.

A home is not a lab: distance, volume, HVAC rumble and TV chatter all vary. We
make the classifier input volume- and rumble-invariant (DC/low-cut + loudness
normalization) and expose a live SNR estimate the pipeline uses to gate weak,
far-off sounds. Pure numpy — no extra deps.
"""

from __future__ import annotations

import numpy as np


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)


def dc_block(x: np.ndarray, sample_rate: int, cutoff_hz: float = 80.0) -> np.ndarray:
    """High-pass by subtracting a moving average; kills DC + low rumble."""
    win = max(3, int(sample_rate / max(cutoff_hz, 1.0)))
    if win >= len(x):
        return (x - x.mean()).astype(np.float32)
    kernel = np.ones(win, dtype=np.float32) / win
    low = np.convolve(x, kernel, mode="same")
    return (x - low).astype(np.float32)


def pre_emphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    return np.append(x[0], x[1:] - coeff * x[:-1]).astype(np.float32)


def normalize_rms(x: np.ndarray, target_rms: float = 0.1, silence_rms: float = 1e-3) -> np.ndarray:
    cur = rms(x)
    if cur < silence_rms:
        return x.astype(np.float32)
    gain = min(target_rms / cur, 30.0)  # cap gain so silence isn't blown up
    return np.clip(x * gain, -1.0, 1.0).astype(np.float32)


def snr_db(signal_rms: float, noise_rms: float) -> float:
    return float(20.0 * np.log10((signal_rms + 1e-9) / (noise_rms + 1e-9)))


class NoiseFloor:
    """EMA estimate of the ambient noise floor, updated only on quiet frames."""

    def __init__(self, init_rms: float = 1e-3, alpha: float = 0.99):
        self.noise_rms = init_rms
        self.alpha = alpha

    def update(self, frame_rms: float, is_active: bool) -> None:
        if not is_active:
            self.noise_rms = self.alpha * self.noise_rms + (1 - self.alpha) * frame_rms

    def calibrate(self, frames: list[np.ndarray]) -> None:
        if frames:
            self.noise_rms = float(np.median([rms(f) for f in frames]))


class WindowPreprocessor:
    """Conditions a classification window and reports its SNR vs the noise floor."""

    def __init__(self, sample_rate: int, target_rms: float = 0.1):
        self.sample_rate = sample_rate
        self.target_rms = target_rms

    def prepare(self, window: np.ndarray, noise_rms: float) -> tuple[np.ndarray, float, float]:
        cleaned = dc_block(window, self.sample_rate)
        sig_rms = rms(cleaned)
        snr = snr_db(sig_rms, noise_rms)
        normalized = normalize_rms(cleaned, self.target_rms)
        return normalized, snr, sig_rms
