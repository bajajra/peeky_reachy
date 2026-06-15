"""Shared detection types used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SoundEvent(str, Enum):
    SILENCE = "silence"
    SPEECH = "speech"
    BABY_CRY = "baby_cry"
    DOG = "dog"
    OTHER = "other"


class CryReason(str, Enum):
    """Dunstan-style categories. Treated as a weak hint, never as fact."""

    HUNGRY = "hungry"
    TIRED = "tired"
    DISCOMFORT = "discomfort"
    PAIN = "pain"
    BURPING = "burping"
    UNKNOWN = "unknown"


@dataclass
class DetectionResult:
    event: SoundEvent
    score: float
    is_voiced: bool
    reason: Optional[CryReason] = None
    reason_confidence: float = 0.0

    @property
    def is_cry(self) -> bool:
        return self.event in (SoundEvent.BABY_CRY, SoundEvent.DOG)
