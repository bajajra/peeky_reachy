"""Combine several classifiers into a more robust, abstaining decision.

No single model is reliable across all homes. A weighted soft-vote over diverse
members (numpy heuristic + YAMNet + optional CLAP) is steadier than any one, and
an abstain rule (return OTHER when confidence/agreement is low) trades a few
missed detections for far fewer false soothes — the right trade for a baby.
"""

from __future__ import annotations

import logging

import numpy as np

from .classifier import EventClassifier
from .events import SoundEvent

log = logging.getLogger("peeky.ensemble")


class EnsembleClassifier(EventClassifier):
    def __init__(self, members: list[tuple[EventClassifier, float]],
                 min_score: float = 0.4):
        if not members:
            raise ValueError("EnsembleClassifier needs at least one member")
        self.members = members
        self.min_score = min_score
        self._total_w = sum(w for _, w in members) or 1.0

    def classify(self, window: np.ndarray, sample_rate: int) -> tuple[SoundEvent, float]:
        agg: dict[SoundEvent, float] = {}
        for clf, weight in self.members:
            try:
                event, score = clf.classify(window, sample_rate)
            except Exception as exc:
                log.debug("ensemble member failed: %s", exc)
                continue
            agg[event] = agg.get(event, 0.0) + weight * score
        if not agg:
            return SoundEvent.OTHER, 0.0
        best = max(agg, key=agg.get)
        score = agg[best] / self._total_w
        if score < self.min_score:
            return SoundEvent.OTHER, float(score)
        return best, float(score)
