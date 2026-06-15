"""LLM-powered bedtime stories & lullabies for Peeky.

Heavy LLM dependencies (Anthropic, Ollama) are optional. Every entry point
gracefully falls back to a deterministic template so the full feature works
offline, on a laptop, with no API key and no GPU — mirroring how
``voice/clone_client.py`` and ``detect/classifier.py`` degrade.

Public surface:
    StoryGenerator   - bedtime story for a given age / theme / length
    LullabyGenerator - short rhyming lullaby on a theme
    make_backend     - select an LLM backend from env / overrides
    SpeakCache       - text -> cloned-voice WAV with on-disk caching
    build_story_tab  - Gradio tab factory (the webapp owner wires this in)
"""

from .backends import (
    AnthropicBackend,
    LLMBackend,
    OllamaBackend,
    TemplateBackend,
    make_backend,
)
from .lullaby import LullabyGenerator
from .speak import SpeakCache
from .story import StoryGenerator
from .webtab import build_story_tab

__all__ = [
    "AnthropicBackend",
    "LLMBackend",
    "LullabyGenerator",
    "OllamaBackend",
    "SpeakCache",
    "StoryGenerator",
    "TemplateBackend",
    "build_story_tab",
    "make_backend",
]
