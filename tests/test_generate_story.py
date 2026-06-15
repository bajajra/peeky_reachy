"""Tests for the bedtime-story generator's offline (template) path.

We never hit the network here: every test forces ``TemplateBackend`` (or a
fake backend that returns ``None``) so the deterministic fallback runs.
"""

from __future__ import annotations

from peeky_reachy.generate.backends import LLMBackend, TemplateBackend, make_backend
from peeky_reachy.generate.safety import cap_words, contains_unsafe
from peeky_reachy.generate.story import WORDS_PER_MINUTE, StoryGenerator


class _UnsafeBackend(LLMBackend):
    name = "unsafe"

    def complete(self, system, user, *, max_tokens=800):
        return "The scary monster came and the moon began to die."


class _GoodBackend(LLMBackend):
    name = "good"

    def __init__(self, text: str):
        self._text = text

    def complete(self, system, user, *, max_tokens=800):
        return self._text


def test_template_story_is_deterministic_with_seed():
    gen = StoryGenerator(backend=TemplateBackend(), seed=42)
    a = gen.generate(age_months=18, theme="the moon", minutes=2.0)
    b = gen.generate(age_months=18, theme="the moon", minutes=2.0)
    assert a == b
    assert len(a) > 100


def test_template_story_hits_target_length():
    gen = StoryGenerator(backend=TemplateBackend(), seed=7)
    text = gen.generate(age_months=24, theme="a sleepy bunny", minutes=3.0)
    target = int(3.0 * WORDS_PER_MINUTE)
    words = len(text.split())
    # Template pads until target - 25 then appends an ending; allow ample slack.
    assert words >= target - 60
    assert words <= int(target * 1.6)


def test_template_story_contains_theme_and_calming_ending():
    gen = StoryGenerator(backend=TemplateBackend(), seed=1)
    text = gen.generate(age_months=12, theme="a little owl", minutes=1.0).lower()
    assert "little owl" in text
    assert any(cue in text for cue in ("sleep", "goodnight"))


def test_template_story_is_safe():
    gen = StoryGenerator(backend=TemplateBackend(), seed=0)
    for theme in ["the moon", "a sleepy bear", "the river"]:
        text = gen.generate(age_months=24, theme=theme, minutes=2.0)
        assert not contains_unsafe(text), f"unsafe word in template story: {text!r}"


def test_unsafe_llm_output_falls_back_to_template():
    gen = StoryGenerator(backend=_UnsafeBackend(), seed=3)
    text = gen.generate(age_months=18, theme="the moon", minutes=1.0)
    assert not contains_unsafe(text)
    # Template includes a calming ending — quick proxy that the fallback ran.
    assert any(cue in text.lower() for cue in ("sleep", "goodnight"))


def test_good_llm_output_is_used_and_capped():
    long_text = "The moon is round and bright. " * 500  # ~3000 words
    gen = StoryGenerator(backend=_GoodBackend(long_text))
    text = gen.generate(age_months=24, theme="the moon", minutes=2.0)
    target_max = int(2.0 * WORDS_PER_MINUTE * 1.4)
    assert len(text.split()) <= target_max


def test_minutes_clamped_to_valid_range():
    gen = StoryGenerator(backend=TemplateBackend(), seed=2)
    short = gen.generate(age_months=12, theme="the star", minutes=0.0)
    long = gen.generate(age_months=12, theme="the star", minutes=999)
    assert len(short.split()) > 0
    assert len(long.split()) > 0


def test_make_backend_defaults_to_template_with_no_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PEEKY_STORY_BACKEND", raising=False)
    monkeypatch.setenv("PEEKY_OLLAMA_URL", "http://127.0.0.1:1")  # unreachable
    b = make_backend()
    assert isinstance(b, TemplateBackend)


def test_make_backend_explicit_template(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")  # would otherwise prefer anthropic
    b = make_backend("template")
    assert isinstance(b, TemplateBackend)


def test_cap_words_helper():
    assert cap_words("a b c d e", 3) == "a b c."
    assert cap_words("hello world", 10) == "hello world"
