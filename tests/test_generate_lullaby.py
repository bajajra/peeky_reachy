"""Tests for the lullaby generator's offline (template) path."""

from __future__ import annotations

from peeky_reachy.generate.backends import LLMBackend, TemplateBackend
from peeky_reachy.generate.lullaby import MAX_WORDS, LullabyGenerator
from peeky_reachy.generate.safety import contains_unsafe


class _UnsafeBackend(LLMBackend):
    name = "unsafe"

    def complete(self, system, user, *, max_tokens=400):
        return "Monsters and ghosts will scare the moon to death."


class _GoodBackend(LLMBackend):
    name = "good"

    def complete(self, system, user, *, max_tokens=400):
        return "Twinkle twinkle little star.\nHow I wonder what you are.\nGoodnight."


def test_template_lullaby_deterministic_with_seed():
    gen = LullabyGenerator(backend=TemplateBackend(), seed=11)
    a = gen.generate(theme="the moon")
    b = gen.generate(theme="the moon")
    assert a == b


def test_template_lullaby_mentions_theme_and_goodnight():
    gen = LullabyGenerator(backend=TemplateBackend(), seed=5)
    text = gen.generate(theme="the moon").lower()
    assert "moon" in text
    assert "goodnight" in text or "sleep" in text


def test_template_lullaby_has_multiple_stanzas():
    gen = LullabyGenerator(backend=TemplateBackend(), seed=8)
    text = gen.generate(theme="the moon")
    stanzas = [s for s in text.split("\n\n") if s.strip()]
    assert len(stanzas) >= 3


def test_template_lullaby_is_safe():
    gen = LullabyGenerator(backend=TemplateBackend(), seed=0)
    for theme in ["the moon", "a sleepy fox", "the night"]:
        text = gen.generate(theme=theme)
        assert not contains_unsafe(text)


def test_unsafe_llm_output_falls_back_to_template():
    gen = LullabyGenerator(backend=_UnsafeBackend(), seed=3)
    text = gen.generate(theme="the moon")
    assert not contains_unsafe(text)


def test_good_llm_output_is_used():
    gen = LullabyGenerator(backend=_GoodBackend())
    text = gen.generate(theme="a star")
    assert "twinkle" in text.lower()
    assert len(text.split()) <= MAX_WORDS


def test_empty_theme_defaults_safely():
    gen = LullabyGenerator(backend=TemplateBackend(), seed=4)
    text = gen.generate(theme="   ")
    assert "moon" in text.lower()
