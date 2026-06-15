"""Generate a short rhyming lullaby on a theme.

Like ``StoryGenerator``, tries the LLM backend first and falls back to a
deterministic stanza template on any failure or unsafe output. The template
produces an AABB-rhymed two-stanza lullaby — short and sing-songy enough that
the cloned-voice TTS reads it gently.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from .backends import LLMBackend, make_backend
from .safety import cap_words, contains_unsafe

log = logging.getLogger("peeky.generate.lullaby")

MAX_WORDS = 120

_SYSTEM = (
    "You write very short, gentle lullabies for young children. "
    "Two or three short stanzas, simple AABB or ABAB rhymes, soft imagery "
    "(stars, moon, sleepy animals). Never include anything scary, sad, or "
    "violent. End with a peaceful goodnight. Output only the lullaby text — "
    "no titles, preamble, or commentary."
)


class LullabyGenerator:
    def __init__(self, backend: Optional[LLMBackend] = None, seed: Optional[int] = None):
        self.backend = backend or make_backend()
        self._seed = seed

    def generate(self, theme: str = "the moon") -> str:
        theme = theme.strip() or "the moon"
        user = (f"Write a short bedtime lullaby about {theme}. "
                f"Two or three soft, rhyming stanzas. End with goodnight.")
        text = self.backend.complete(_SYSTEM, user, max_tokens=400)
        if text and not contains_unsafe(text):
            return cap_words(text, MAX_WORDS)
        if text:
            log.info("LLM lullaby tripped safety guard; using template fallback.")
        return _template_lullaby(theme, seed=self._seed)


# ---------------------------------------------------------------------------
# Deterministic AABB-rhymed template. Each (a, b) pair already rhymes; we mix
# pairs to vary the output without ever producing a non-rhyme.

_PAIRS_A = [
    ("The {theme} is soft and bright,", "watching over you tonight."),
    ("Little one, it's time to rest,", "in the place that you love best."),
    ("Stars are peeking through the sky,", "singing you a lullaby."),
    ("Sleepy clouds are drifting by,", "humming sweetly, by and by."),
]

_PAIRS_B = [
    ("Close your eyes and softly sigh,", "drift away on dreams that fly."),
    ("Hush now, dear, the world is still,", "warm and safe, and always will."),
    ("Hold my hand and feel the breeze,", "calm as moonlight on the trees."),
    ("Slow your breath and let it go,", "soft and gentle, soft and slow."),
]

_CLOSERS = [
    "Goodnight, little one. Goodnight, goodnight.",
    "Sleep tight, my dear. The stars will keep you bright.",
    "Sweet dreams, my love. Goodnight, goodnight, goodnight.",
]


def _template_lullaby(theme: str, seed: Optional[int] = None) -> str:
    rng = random.Random(seed if seed is not None else theme)
    a1, a2 = rng.choice(_PAIRS_A)
    b1, b2 = rng.choice(_PAIRS_B)
    c1, c2 = rng.choice(_PAIRS_A)
    closer = rng.choice(_CLOSERS)
    return "\n".join([
        a1.format(theme=theme), a2,
        "",
        b1, b2,
        "",
        c1.format(theme=theme), c2,
        "",
        closer,
    ])
