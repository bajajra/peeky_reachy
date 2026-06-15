"""Optional, low-confidence cry-reason hint. Off by default.

WARNING: inferring *why* a baby cries from audio alone is scientifically weak
(trained nurses score ~33%). This is surfaced as a hint only, never as fact, and
the caregiver is always kept in the loop. Disabled unless reason_hint_enabled.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from .events import CryReason

log = logging.getLogger("peeky.reason")

# Confidence is intentionally capped low; this signal is advisory only.
MAX_CONFIDENCE = 0.4


class ReasonHinter(ABC):
    @abstractmethod
    def hint(self, window: np.ndarray, sample_rate: int) -> tuple[CryReason, float]:
        """Return a (reason, confidence) hint; confidence is deliberately low."""


class HeuristicReasonHinter(ReasonHinter):
    """Maps coarse rhythm/pitch cues to Dunstan-style guesses. Placeholder."""

    def hint(self, window: np.ndarray, sample_rate: int) -> tuple[CryReason, float]:
        env = np.abs(window)
        # Rough "rhythmicity": autocorrelation peak of the amplitude envelope.
        env = env - env.mean()
        if np.allclose(env, 0):
            return CryReason.UNKNOWN, 0.0
        ac = np.correlate(env, env, mode="full")[len(env) - 1:]
        ac = ac / (ac[0] + 1e-9)
        rhythmic = float(ac[sample_rate // 10: sample_rate // 2].max()) if len(ac) > sample_rate // 2 else 0.0
        spec = np.abs(np.fft.rfft(window * np.hanning(len(window)))) + 1e-9
        freqs = np.fft.rfftfreq(len(window), 1.0 / sample_rate)
        centroid = float((freqs * spec).sum() / spec.sum())

        if rhythmic > 0.6:
            reason = CryReason.HUNGRY      # rhythmic "neh"-like bursts
        elif centroid > 1800:
            reason = CryReason.PAIN        # sharp, high, sudden
        elif centroid < 700:
            reason = CryReason.TIRED       # low, whiny
        else:
            reason = CryReason.DISCOMFORT
        confidence = min(MAX_CONFIDENCE, 0.2 + 0.2 * rhythmic)
        return reason, float(confidence)


class HFReasonHinter(ReasonHinter):
    """Hook for a Hugging Face donateacry/Dunstan model (loaded lazily)."""

    def __init__(self, model_id: str = "foduucom/baby-cry-classification"):
        from transformers import pipeline  # raises if not installed

        self._pipe = pipeline("audio-classification", model=model_id)
        self._map = {
            "hungry": CryReason.HUNGRY,
            "tired": CryReason.TIRED,
            "discomfort": CryReason.DISCOMFORT,
            "belly_pain": CryReason.PAIN,
            "burping": CryReason.BURPING,
        }

    def hint(self, window: np.ndarray, sample_rate: int) -> tuple[CryReason, float]:
        out = self._pipe({"array": window.astype("float32"), "sampling_rate": sample_rate})
        top = out[0]
        reason = self._map.get(top["label"].lower(), CryReason.UNKNOWN)
        return reason, float(min(MAX_CONFIDENCE, top["score"]))


def make_reason_hinter(enabled: bool, prefer_ml: bool = False) -> ReasonHinter | None:
    if not enabled:
        return None
    if prefer_ml:
        try:
            return HFReasonHinter()
        except Exception as exc:
            log.info("Reason hinter: HF model unavailable (%s); using heuristic", exc)
    return HeuristicReasonHinter()


class EpisodeReasonAggregator:
    """Aggregate per-frame reason hints across a whole cry episode.

    A single frame's reason guess is noise. We accumulate confidence-weighted
    votes for the duration of one cry, require enough votes and clear agreement,
    optionally apply a *weak* time-of-day prior, and abstain to UNKNOWN when the
    evidence is thin. Confidence stays capped — this is always advisory.
    """

    def __init__(self, min_votes: int = 3, min_agreement: float = 0.5,
                 max_confidence: float = MAX_CONFIDENCE):
        self.min_votes = min_votes
        self.min_agreement = min_agreement
        self.max_confidence = max_confidence
        self._votes: dict[CryReason, float] = {}
        self._count = 0

    def add(self, reason: CryReason, confidence: float) -> None:
        if reason == CryReason.UNKNOWN or confidence <= 0:
            return
        self._votes[reason] = self._votes.get(reason, 0.0) + confidence
        self._count += 1

    def result(self, hour_of_day: int | None = None) -> tuple[CryReason, float]:
        if self._count < self.min_votes or not self._votes:
            return CryReason.UNKNOWN, 0.0
        votes = dict(self._votes)
        # Weak, explicit context prior: late night nudges TIRED a little.
        if hour_of_day is not None and (hour_of_day >= 22 or hour_of_day <= 5):
            votes[CryReason.TIRED] = votes.get(CryReason.TIRED, 0.0) * 1.15 + 0.05
        total = sum(votes.values())
        best = max(votes, key=votes.get)
        agreement = votes[best] / total
        if agreement < self.min_agreement:
            return CryReason.UNKNOWN, 0.0
        confidence = min(self.max_confidence, agreement * (votes[best] / self._count))
        return best, float(confidence)

    def reset(self) -> None:
        self._votes.clear()
        self._count = 0
