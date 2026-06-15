"""Classify a short audio window into {silence, speech, baby_cry, dog}.

YamnetClassifier is the production path. HeuristicClassifier is a dependency-free
numpy fallback that keys on coarse spectral shape; it is good enough to exercise
the full pipeline and tests, and is clearly a placeholder for YAMNet.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from .events import SoundEvent

log = logging.getLogger("peeky.classifier")


class EventClassifier(ABC):
    @abstractmethod
    def classify(self, window: np.ndarray, sample_rate: int) -> tuple[SoundEvent, float]:
        """Return the dominant (event, score in [0,1]) for a mono window."""


def _spectral_features(window: np.ndarray, sr: int) -> dict:
    w = window * np.hanning(len(window))
    spec = np.abs(np.fft.rfft(w)) + 1e-9
    freqs = np.fft.rfftfreq(len(window), 1.0 / sr)
    total = float(spec.sum())
    centroid = float((freqs * spec).sum() / total)
    dom_freq = float(freqs[int(np.argmax(spec))])
    flatness = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
    hi_ratio = float(spec[freqs >= 1500].sum() / total)
    low_ratio = float(spec[freqs < 500].sum() / total)
    rms = float(np.sqrt(np.mean(np.square(window))) + 1e-9)
    return dict(centroid=centroid, dom_freq=dom_freq, flatness=flatness,
                hi_ratio=hi_ratio, low_ratio=low_ratio, rms=rms)


class HeuristicClassifier(EventClassifier):
    def __init__(self, silence_rms: float = 0.02):
        self.silence_rms = silence_rms

    def classify(self, window: np.ndarray, sample_rate: int) -> tuple[SoundEvent, float]:
        f = _spectral_features(window, sample_rate)
        if f["rms"] < self.silence_rms:
            return SoundEvent.SILENCE, float(np.clip(1.0 - f["rms"] / self.silence_rms, 0.5, 1.0))

        low, hi, cen, flat = f["low_ratio"], f["hi_ratio"], f["centroid"], f["flatness"]

        # Rules use only scale-invariant spectral shape (loudness varies with
        # distance and is normalized upstream). This is a placeholder for YAMNet.
        if low > 0.45:  # low-band dominant + broadband -> dog/growl
            score = 0.4 + 0.3 * float(np.clip(low, 0, 1)) + 0.2 * float(np.clip(flat / 0.5, 0, 1))
            return SoundEvent.DOG, float(np.clip(score, 0, 1))

        if hi >= 0.5 and low < 0.4:  # bright, high-band-rich + not low -> cry
            score = (0.5
                     + 0.3 * float(np.clip((hi - 0.45) / 0.4, 0, 1))
                     + 0.2 * float(np.clip((cen - 1500) / 2500, 0, 1)))
            return SoundEvent.BABY_CRY, float(np.clip(score, 0, 1))

        score = 0.3 + 0.15 * float(np.clip((cen - 300) / 1500, 0, 1))
        return SoundEvent.SPEECH, float(np.clip(score, 0.15, 0.5))


class YamnetClassifier(EventClassifier):
    """Google YAMNet via tensorflow_hub, mapped to our SoundEvent set."""

    _MAP = {
        "Baby cry, infant cry": SoundEvent.BABY_CRY,
        "Crying, sobbing": SoundEvent.BABY_CRY,
        "Dog": SoundEvent.DOG,
        "Bark": SoundEvent.DOG,
        "Bow-wow": SoundEvent.DOG,
        "Whimper (dog)": SoundEvent.DOG,
        "Speech": SoundEvent.SPEECH,
        "Conversation": SoundEvent.SPEECH,
        "Silence": SoundEvent.SILENCE,
    }

    def __init__(self):
        import csv
        import tensorflow_hub as hub  # raises if not installed

        self._model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        with open(class_map_path) as fh:
            self._labels = [row["display_name"] for row in csv.DictReader(fh)]

    def classify(self, window: np.ndarray, sample_rate: int) -> tuple[SoundEvent, float]:
        if sample_rate != 16000:
            from ..audio.io import resample_linear
            window = resample_linear(window, sample_rate, 16000)
        scores, _, _ = self._model(window.astype(np.float32))
        mean = scores.numpy().mean(axis=0)
        agg: dict[SoundEvent, float] = {}
        for idx, label in enumerate(self._labels):
            event = self._MAP.get(label)
            if event is not None:
                agg[event] = max(agg.get(event, 0.0), float(mean[idx]))
        if not agg:
            return SoundEvent.OTHER, 0.0
        event = max(agg, key=agg.get)
        return event, agg[event]


def make_classifier(prefer_ml: bool = True) -> EventClassifier:
    if prefer_ml:
        try:
            clf = YamnetClassifier()
            log.info("Classifier: using YAMNet")
            return clf
        except Exception as exc:
            log.info("Classifier: YAMNet unavailable (%s); using heuristic fallback", exc)
    return HeuristicClassifier()
