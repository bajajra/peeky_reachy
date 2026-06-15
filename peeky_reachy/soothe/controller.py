"""Decide WHEN and WHAT to soothe from a stream of detections.

Debounces brief noises (a single yelp, a door slam) by requiring a cry to be
sustained, and rate-limits actions with a cooldown so Peeky doesn't badger.
Time is passed in (``now``) to keep this pure and testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ..detect.events import CryReason, DetectionResult, SoundEvent

log = logging.getLogger("peeky.controller")

# How long a cry may dip below threshold before we consider it interrupted.
_HANGOVER_S = 1.0


@dataclass
class SootheDecision:
    event: SoundEvent
    reason: Optional[CryReason]
    reason_confidence: float
    score: float
    notification: str


class SootheController:
    def __init__(self, cry_score_threshold: float, sustain_seconds: float,
                 cooldown_seconds: float):
        self.cry_score_threshold = cry_score_threshold
        self.sustain_seconds = sustain_seconds
        self.cooldown_seconds = cooldown_seconds
        self._cry_since: Optional[float] = None
        self._last_cry_at: Optional[float] = None
        self._last_action_at: Optional[float] = None

    @property
    def in_cooldown(self) -> bool:
        return self._last_action_at is not None

    def cooldown_remaining(self, now: float) -> float:
        if self._last_action_at is None:
            return 0.0
        return max(0.0, self.cooldown_seconds - (now - self._last_action_at))

    def observe(self, result: DetectionResult, now: float) -> Optional[SootheDecision]:
        cry = (result.is_cry and result.is_voiced
               and result.score >= self.cry_score_threshold)

        if not cry:
            if self._cry_since is not None and self._last_cry_at is not None \
                    and (now - self._last_cry_at) > _HANGOVER_S:
                self._cry_since = None
            return None

        self._last_cry_at = now
        if self._cry_since is None:
            self._cry_since = now

        if self.cooldown_remaining(now) > 0:
            return None

        if (now - self._cry_since) < self.sustain_seconds:
            return None

        self._last_action_at = now
        self._cry_since = None
        return self._decide(result)

    def _decide(self, result: DetectionResult) -> SootheDecision:
        who = "pet" if result.event == SoundEvent.DOG else "baby"
        note = f"[notify caregiver] Sustained {who} {result.event.value} (score {result.score:.2f})"
        if result.reason and result.reason != CryReason.UNKNOWN:
            note += f"; possible reason: {result.reason.value} (~{result.reason_confidence:.0%}, low confidence)"
        note += "; Peeky is soothing now."
        log.info(note)
        return SootheDecision(
            event=result.event,
            reason=result.reason,
            reason_confidence=result.reason_confidence,
            score=result.score,
            notification=note,
        )

    def reset(self) -> None:
        self._cry_since = None
        self._last_cry_at = None
        self._last_action_at = None
