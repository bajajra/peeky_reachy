"""Soothing phrase selection (for voice clone) + fallback track picking."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from ..detect.events import CryReason, SoundEvent

_GENERIC = [
    "Shhh, it's okay. Mama's here, little one.",
    "There there, you're safe. I've got you.",
    "It's alright, sweetheart. Take a deep breath with me.",
    "Hush now, my love. Everything is okay.",
]

_BY_REASON = {
    CryReason.HUNGRY: ["I know, you're hungry. Someone's coming with your milk soon."],
    CryReason.TIRED: ["You're so sleepy, aren't you? Close your eyes, I'm right here."],
    CryReason.DISCOMFORT: ["Let's get you comfy. It's okay, I'm here with you."],
    CryReason.PAIN: ["I'm here, I'm here. You're not alone, sweet one."],
    CryReason.BURPING: ["A little tummy bubble, that's all. It'll pass, my love."],
}

_PET = ["Easy now, good boy. Settle down, it's okay."]


def pick_phrase(event: SoundEvent, reason: Optional[CryReason] = None,
                seed: Optional[int] = None) -> str:
    rng = random.Random(seed)
    if event == SoundEvent.DOG:
        return rng.choice(_PET)
    if reason and reason in _BY_REASON:
        return rng.choice(_BY_REASON[reason] + _GENERIC)
    return rng.choice(_GENERIC)


def pick_fallback_track(assets_dir: str) -> Optional[Path]:
    base = Path(assets_dir)
    if not base.exists():
        return None
    tracks = sorted(p for p in base.glob("*.wav"))
    return tracks[0] if tracks else None
