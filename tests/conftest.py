"""Synthetic audio fixtures so the suite needs no real recordings or hardware."""

from __future__ import annotations

import numpy as np
import pytest

SR = 16000
_RNG = np.random.default_rng(1234)


def _t(dur: float, sr: int = SR) -> np.ndarray:
    return np.arange(int(dur * sr)) / sr


def silence(dur: float = 2.0, sr: int = SR) -> np.ndarray:
    return (0.0005 * _RNG.standard_normal(int(dur * sr))).astype(np.float32)


def baby_cry(dur: float = 5.0, sr: int = SR) -> np.ndarray:
    t = _t(dur, sr)
    f0 = 450 + 30 * np.sin(2 * np.pi * 5 * t)  # vibrato
    sig = np.zeros_like(t)
    for n in range(1, 11):
        amp = np.exp(-((n - 3) ** 2) / 6.0)  # harmonic energy peaked mid-band
        sig += amp * np.sin(2 * np.pi * f0 * n * t)
    sig /= np.max(np.abs(sig)) + 1e-9
    env = 0.7 + 0.3 * np.sin(2 * np.pi * 0.8 * t)
    return (0.6 * sig * env).astype(np.float32)


def speech(dur: float = 5.0, sr: int = SR) -> np.ndarray:
    t = _t(dur, sr)
    f0 = 130
    sig = np.zeros_like(t)
    for n in range(1, 14):
        amp = np.exp(-((n * f0 - 1000) ** 2) / (2 * 500 ** 2))  # formant ~1 kHz
        sig += amp * np.sin(2 * np.pi * f0 * n * t)
    sig /= np.max(np.abs(sig)) + 1e-9
    env = np.abs(np.sin(2 * np.pi * 3 * t)) + 0.05      # ~3 Hz syllables
    sig = sig * env + 0.02 * _RNG.standard_normal(len(t))
    return (0.25 * sig / (np.max(np.abs(sig)) + 1e-9)).astype(np.float32)


def dog(dur: float = 5.0, sr: int = SR) -> np.ndarray:
    t = _t(dur, sr)
    base = 0.4 * np.sin(2 * np.pi * 250 * t) + 0.3 * np.sin(2 * np.pi * 180 * t)
    noise = _RNG.standard_normal(len(t))
    noise = np.convolve(noise, np.ones(20) / 20, mode="same")  # low-pass
    burst = (np.sin(2 * np.pi * 2 * t) > 0).astype(np.float32)  # ~2 Hz barks
    sig = (base + 0.3 * noise) * burst
    return (0.5 * sig / (np.max(np.abs(sig)) + 1e-9)).astype(np.float32)


def _write(path, data, sr=SR):
    import soundfile as sf

    sf.write(str(path), np.asarray(data, dtype=np.float32), sr)
    return str(path)


@pytest.fixture
def cry_wav(tmp_path):
    return _write(tmp_path / "cry.wav", np.concatenate([silence(2.5), baby_cry(5.0)]))


@pytest.fixture
def silence_wav(tmp_path):
    return _write(tmp_path / "silence.wav", silence(7.0))


@pytest.fixture
def speech_wav(tmp_path):
    return _write(tmp_path / "speech.wav", np.concatenate([silence(2.5), speech(5.0)]))
