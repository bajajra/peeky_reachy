"""Gradio tab factory for the bedtime-story & lullaby generator.

This module deliberately does NOT import or modify ``peeky_reachy/webapp.py``
— that file is owned by another agent. Instead it exposes
``build_story_tab(...)`` which the webapp owner can call inside their
``gr.Blocks(...)`` context to add a "Bedtime story" tab.

Wiring (one line in ``webapp.build_app``)::

    from .generate import build_story_tab
    ...
    with gr.Blocks(title="Peeky") as app:
        ...
        build_story_tab()

The tab is fully self-contained: it instantiates its own LLM backend,
generator, voice client, and speak cache; it never touches the monitor /
enroll tabs' state.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from ..config import Config
from ..voice.clone_client import VoiceCloneClient
from ..voice.store import EnrollmentStore
from .backends import make_backend
from .lullaby import LullabyGenerator
from .speak import SpeakCache
from .story import StoryGenerator

log = logging.getLogger("peeky.generate.webtab")


def _to_gradio_audio(samples: np.ndarray, sr: int):
    pcm = np.clip(samples, -1.0, 1.0)
    return sr, (pcm * 32767).astype(np.int16)


def _make_voice_client(voice_url: str) -> VoiceCloneClient:
    cfg = Config.from_env()
    if voice_url:
        cfg.voice_clone_url = voice_url
    store = EnrollmentStore(cfg.enrollment_dir)
    return VoiceCloneClient(cfg.voice_clone_url, store, cfg.voice_clone_timeout_s)


def generate_story_action(age_months: int, theme: str, minutes: float,
                          backend_name: str, speak: bool, voice_url: str):
    backend = make_backend(backend_name if backend_name != "auto" else None)
    gen = StoryGenerator(backend=backend)
    text = gen.generate(age_months=int(age_months), theme=theme, minutes=float(minutes))
    status = f"Generated story via **{backend.name}** backend ({len(text.split())} words)."
    audio = None
    if speak:
        audio, status = _speak_with_status(text, voice_url, status)
    return text, status, audio


def generate_lullaby_action(theme: str, backend_name: str, speak: bool, voice_url: str):
    backend = make_backend(backend_name if backend_name != "auto" else None)
    gen = LullabyGenerator(backend=backend)
    text = gen.generate(theme=theme)
    status = f"Generated lullaby via **{backend.name}** backend."
    audio = None
    if speak:
        audio, status = _speak_with_status(text, voice_url, status)
    return text, status, audio


def _speak_with_status(text: str, voice_url: str, status_prefix: str):
    try:
        voice = _make_voice_client(voice_url)
        cache = SpeakCache(voice_client=voice,
                           cache_dir=os.environ.get("PEEKY_STORY_CACHE_DIR",
                                                    "cache/stories"))
        result = cache.speak(text)
    except Exception as exc:
        log.warning("Speak failed: %s", exc)
        return None, f"{status_prefix}\n\nVoice synthesis failed: {exc}"
    if result is None:
        return None, (f"{status_prefix}\n\nVoice service unreachable and no "
                      "cached audio — text-only.")
    return _to_gradio_audio(*result), f"{status_prefix}\n\nSpoken in cloned voice."


def build_story_tab(default_voice_url: Optional[str] = None) -> None:
    """Add a "Bedtime story" tab to the surrounding ``gr.Blocks(...)``.

    Must be called inside an active ``gr.Blocks`` context. Returns nothing —
    Gradio components attach themselves to the open context as a side effect.
    """
    import gradio as gr

    cfg = Config.from_env()
    voice_url_default = default_voice_url or cfg.voice_clone_url

    with gr.Tab("Bedtime story"):
        gr.Markdown(
            "Generate a calming bedtime story or a short lullaby. Uses your "
            "configured LLM backend (Anthropic / Ollama) when available; "
            "otherwise falls back to a deterministic safe template. Optional "
            "playback uses the enrolled caregiver's cloned voice (cached on "
            "disk so re-plays are instant)."
        )
        with gr.Row():
            backend_name = gr.Dropdown(
                choices=["auto", "anthropic", "ollama", "template"],
                value="auto", label="LLM backend",
            )
            voice_url = gr.Textbox(value=voice_url_default,
                                   label="Voice service URL (for cloned playback)")

        with gr.Tab("Story"):
            with gr.Row():
                age = gr.Slider(0, 144, value=24, step=1, label="Child age (months)")
                minutes = gr.Slider(0.5, 10.0, value=2.0, step=0.5, label="Length (minutes)")
            theme = gr.Textbox(value="the moon", label="Story theme")
            speak_story = gr.Checkbox(value=False, label="Speak in cloned caregiver voice")
            gen_story_btn = gr.Button("Generate story", variant="primary")
            story_text = gr.Textbox(label="Story text", lines=12)
            story_status = gr.Markdown()
            story_audio = gr.Audio(label="Spoken story", type="numpy")
            gen_story_btn.click(
                generate_story_action,
                inputs=[age, theme, minutes, backend_name, speak_story, voice_url],
                outputs=[story_text, story_status, story_audio],
            )

        with gr.Tab("Lullaby"):
            l_theme = gr.Textbox(value="the moon", label="Lullaby theme")
            speak_lullaby = gr.Checkbox(value=False, label="Speak in cloned caregiver voice")
            gen_lullaby_btn = gr.Button("Generate lullaby", variant="primary")
            lullaby_text = gr.Textbox(label="Lullaby text", lines=10)
            lullaby_status = gr.Markdown()
            lullaby_audio = gr.Audio(label="Spoken lullaby", type="numpy")
            gen_lullaby_btn.click(
                generate_lullaby_action,
                inputs=[l_theme, backend_name, speak_lullaby, voice_url],
                outputs=[lullaby_text, lullaby_status, lullaby_audio],
            )
