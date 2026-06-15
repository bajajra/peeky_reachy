"""Lazy classifier holder for the cry service (reuses the package classifier).

Single source of truth: this wraps ``peeky_reachy.detect.classifier`` so the
remote service and the local fallback share identical logic. Install the package
on the GPU host (``pip install 'peeky-reachy[ml]'`` for the YAMNet path).
"""

from __future__ import annotations

import io
import logging
import wave

import numpy as np

from peeky_reachy.detect.classifier import (HeuristicClassifier, YamnetClassifier,
                                            make_classifier)

log = logging.getLogger("peeky.cry.wrap")


def wav_b64_to_samples(data_b64: str) -> tuple[np.ndarray, int]:
    import base64

    raw = base64.b64decode(data_b64, validate=True)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        pcm = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    return pcm, sr


class ClassifierWrapper:
    def __init__(self, prefer_ml: bool = True):
        self._prefer_ml = prefer_ml
        self._clf = None
        self.model_id = "uninitialized"

    @property
    def is_loaded(self) -> bool:
        return isinstance(self._clf, YamnetClassifier)

    def ensure_loaded(self):
        if self._clf is None:
            self._clf = make_classifier(prefer_ml=self._prefer_ml)
            self.model_id = ("google/yamnet" if isinstance(self._clf, YamnetClassifier)
                             else "numpy-heuristic")
            log.info("Cry classifier ready: %s", self.model_id)
        return self._clf

    def classify(self, samples: np.ndarray, sample_rate: int) -> tuple[str, float]:
        event, score = self.ensure_loaded().classify(samples, sample_rate)
        return event.value, float(score)


def make_wrapper_from_env() -> ClassifierWrapper:
    import os

    prefer = os.environ.get("PEEKY_CRY_PREFER_ML", "1") != "0"
    return ClassifierWrapper(prefer_ml=prefer)
