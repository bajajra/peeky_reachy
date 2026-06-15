"""Soft content guards for bedtime text.

We are generating audio for a child — the bar isn't safety-critical filtering,
it's avoiding obvious mood-killers (monsters, dying, violence) that an LLM
might produce on a quirky prompt. Two guards, both used post-generation:

- ``contains_unsafe(text)``: True if any blocked word appears (case-insensitive,
  whole-word). On True the caller should discard the LLM output and fall back
  to the template.
- ``cap_words(text, max_words)``: hard length cap so a runaway model can't
  produce a 20-minute monologue when the user asked for 2 minutes.

Word list is intentionally small and conservative; it is not a moderation
system. The deterministic template never trips these.
"""

from __future__ import annotations

import re

_BLOCKED = (
    "kill", "killed", "killing", "die", "died", "dying", "dead",
    "blood", "bloody", "murder", "weapon", "gun", "knife",
    "monster", "monsters", "demon", "demons", "ghost", "ghosts",
    "nightmare", "nightmares", "scary", "scared", "scream", "screaming",
    "hate", "war", "fight", "fighting", "hurt", "pain",
)
_PATTERN = re.compile(r"\b(" + "|".join(_BLOCKED) + r")\b", re.IGNORECASE)


def contains_unsafe(text: str) -> bool:
    return bool(_PATTERN.search(text or ""))


def cap_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return (text or "").strip()
    return " ".join(words[:max_words]).rstrip(",;:-") + "."
