"""Generate a calming, age-appropriate bedtime story.

``StoryGenerator.generate(age_months, theme, minutes)`` returns plain text
suitable for TTS. It tries the configured LLM backend first; on failure,
empty output, or a tripped safety guard, it returns a deterministic template
story so the feature works offline.

Tone rules baked into both paths:
- short sentences, soft repetition, present tense
- gentle ending that fades to sleep ("close your eyes ... sleep tight")
- no scary elements (see ``safety.contains_unsafe``)

Length is paced at ~110 spoken words/minute and hard-capped.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from .backends import LLMBackend, make_backend
from .safety import cap_words, contains_unsafe

log = logging.getLogger("peeky.generate.story")

WORDS_PER_MINUTE = 110
MAX_MINUTES = 10

_SYSTEM = (
    "You write very gentle, calming bedtime stories for young children. "
    "Use short sentences, soft repetition, present tense, and a slow rhythm. "
    "Never include anything scary, sad, violent, or upsetting (no monsters, "
    "no fighting, no death). End with a peaceful image and a soft cue to "
    "sleep. Output only the story text — no titles, preamble, or commentary."
)

_AGE_HINTS = (
    (12, "very simple words a one-year-old can follow, lots of repetition"),
    (36, "simple words a toddler can follow, gentle rhyme welcome"),
    (72, "short, vivid sentences a preschooler can picture"),
    (144, "calm, descriptive prose suitable for a young child"),
)


def _age_hint(age_months: int) -> str:
    for limit, hint in _AGE_HINTS:
        if age_months <= limit:
            return hint
    return _AGE_HINTS[-1][1]


class StoryGenerator:
    def __init__(self, backend: Optional[LLMBackend] = None, seed: Optional[int] = None):
        self.backend = backend or make_backend()
        self._seed = seed

    def generate(self, age_months: int = 24, theme: str = "the moon",
                 minutes: float = 2.0) -> str:
        age_months = max(0, int(age_months))
        minutes = max(0.5, min(MAX_MINUTES, float(minutes)))
        target_words = int(minutes * WORDS_PER_MINUTE)
        max_words = int(target_words * 1.4)

        user = (
            f"Write a bedtime story about {theme.strip() or 'the moon'} "
            f"for a child about {age_months} months old "
            f"({_age_hint(age_months)}). Aim for roughly {target_words} words "
            f"(~{minutes:.1f} minutes when read slowly). End softly with the "
            f"child drifting off to sleep."
        )

        text = self.backend.complete(_SYSTEM, user, max_tokens=max(400, target_words * 3))
        if text and not contains_unsafe(text):
            return cap_words(text, max_words)
        if text:
            log.info("LLM story tripped safety guard; using template fallback.")
        return _template_story(age_months, theme, target_words, seed=self._seed)


# ---------------------------------------------------------------------------
# Deterministic template — used when no LLM is available or output is rejected.
# Builds a story from a small library of safe, pre-vetted lines, paced to hit
# the requested word target so a "5 minute" story really is ~5 minutes.

_OPENERS = [
    "Once upon a quiet night, when the world was warm and still,",
    "In a soft, sleepy little room, where the curtains barely moved,",
    "Far, far away, in a calm and gentle valley,",
    "Under a sky full of slow, twinkling stars,",
]

_MIDDLES = [
    "the {theme} smiled a slow, sleepy smile.",
    "everything around {theme} began to breathe slowly, in and out.",
    "a little breeze whispered hello to {theme} and tiptoed away.",
    "{theme} listened to the hush of the world and felt very safe.",
    "soft clouds drifted by, waving gently to {theme}.",
    "the trees swayed just a little, like they were rocking {theme} to sleep.",
    "a tiny star blinked at {theme}, as if to say goodnight.",
    "the moon peeked out and shone a soft silver light on {theme}.",
]

_REPEATS = [
    "Breathe in, slow and soft. Breathe out, slow and soft.",
    "Everything is calm. Everything is warm. Everything is safe.",
    "Soft, soft, soft. Slow, slow, slow.",
    "Little eyes, getting heavy. Little hands, getting still.",
]

_ENDINGS = [
    "Now close your eyes, little one. The night is here to hold you. "
    "Sleep tight. Sleep tight. Sleep tight.",
    "Snuggle in, my dear. The stars are watching over you. "
    "Goodnight. Goodnight. Goodnight.",
    "Let your eyes grow heavy. Let your breath grow slow. "
    "It is time to sleep. Sleep, sleep, sleep.",
]


def _template_story(age_months: int, theme: str, target_words: int,
                    seed: Optional[int] = None) -> str:
    rng = random.Random(seed if seed is not None else (age_months, theme, target_words))
    theme = theme.strip() or "the little star"

    parts: list[str] = [rng.choice(_OPENERS)]
    parts.append(rng.choice(_MIDDLES).format(theme=theme))

    while sum(len(p.split()) for p in parts) < target_words - 25:
        parts.append(rng.choice(_MIDDLES).format(theme=theme))
        if rng.random() < 0.4:
            parts.append(rng.choice(_REPEATS))

    parts.append(rng.choice(_ENDINGS))
    return " ".join(parts)
