"""Gradio v6 web UI for Peeky: monitor a clip, enroll a voice, preview soothing.

This is a thin front-end over the existing pipeline (`pipeline.Pipeline`) and
voice stack — all detection/soothing logic lives in the package, not here.
Launch with ``peeky-web`` or ``python -m peeky_reachy.webapp``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .audio.io import ArrayAudioIO
from .config import Config
from .pipeline import Pipeline
from .soothe.responses import pick_fallback_track, pick_phrase
from .voice.clone_client import VoiceCloneClient
from .voice.enroll import CONSENT_TEXT, enroll_from_array
from .voice.store import EnrollmentStore

log = logging.getLogger("peeky.webapp")

SAFETY = (
    "**Peeky is a soothing companion, not a safety/medical/SIDS monitor.** "
    "Never rely on it to keep a child safe. Cry-*reason* guesses are advisory and "
    "low-confidence — keep a caregiver in the loop."
)


def _to_float_mono(audio) -> tuple[Optional[np.ndarray], int]:
    if audio is None:
        return None, 16000
    sr, data = audio
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, int(sr)


def _to_gradio_audio(samples: np.ndarray, sr: int) -> tuple[int, np.ndarray]:
    pcm = np.clip(samples, -1.0, 1.0)
    return sr, (pcm * 32767).astype(np.int16)


def _config_from_ui(cry_thr, sustain, cooldown, min_snr, smoothing, vad_thr,
                    reason, voice_url) -> Config:
    cfg = Config.from_env()
    cfg.cry_score_threshold = float(cry_thr)
    cfg.sustain_seconds = float(sustain)
    cfg.cooldown_seconds = float(cooldown)
    cfg.min_snr_db = float(min_snr)
    cfg.smoothing_window = int(smoothing)
    cfg.vad_threshold = float(vad_thr)
    cfg.reason_hint_enabled = bool(reason)
    if voice_url:
        cfg.voice_clone_url = voice_url
    return cfg


def analyze(audio, cry_thr, sustain, cooldown, min_snr, smoothing, vad_thr,
            reason, use_voice, voice_url):
    samples, sr = _to_float_mono(audio)
    if samples is None or len(samples) == 0:
        return "Please record or upload an audio clip first.", None, [], None

    cfg = _config_from_ui(cry_thr, sustain, cooldown, min_snr, smoothing, vad_thr,
                          reason, voice_url)
    io = ArrayAudioIO(samples, sr, cfg.sample_rate, cfg.frame_size)
    timeline: list[list] = []
    events = []
    voice = None
    if use_voice:
        store = EnrollmentStore(cfg.enrollment_dir)
        voice = VoiceCloneClient(cfg.voice_clone_url, store, cfg.voice_clone_timeout_s)

    pipe = Pipeline(
        cfg, io, voice_client=voice,
        on_window=lambda t, e, s, snr, v: timeline.append(
            [round(t, 2), e.value, round(float(s), 2), round(float(snr), 1), bool(v)]),
        on_soothe=events.append,
    )
    pipe.run()

    rows = timeline
    plot_df = _plot_df(timeline)

    if not events:
        summary = (f"### No soothing triggered\n"
                   f"Analyzed {pipe.clock:.1f}s. No sustained cry crossed the "
                   f"threshold (needs ≥ {cfg.sustain_seconds:.0f}s above "
                   f"{cfg.cry_score_threshold:.2f} and SNR ≥ {cfg.min_snr_db:.0f} dB).")
        return summary, None, rows, plot_df

    ev = events[0]
    d = ev.decision
    reason_txt = (f"{d.reason.value} (~{d.reason_confidence:.0%}, low confidence)"
                  if d.reason and d.reason.value != "unknown" else "not inferred")
    summary = (
        f"### 🍼 Soothing triggered at {ev.at_seconds:.1f}s\n"
        f"- **Detected:** {d.event.value} (score {d.score:.2f})\n"
        f"- **Possible reason:** {reason_txt}\n"
        f"- **Voice:** {'caregiver clone' if ev.used_clone else 'fallback track / motion-only'}\n"
        f"- **Spoken phrase:** “{ev.phrase}”\n"
        f"- Total soothe events in clip: {len(events)}"
    )
    out_audio = _to_gradio_audio(*ev.audio) if ev.audio else None
    return summary, out_audio, rows, plot_df


def _plot_df(timeline):
    try:
        import pandas as pd
    except Exception:
        return None
    if not timeline:
        return pd.DataFrame({"t": [], "score": []})
    return pd.DataFrame({"t": [r[0] for r in timeline], "score": [r[2] for r in timeline]})


def enroll(audio, name, transcript, language, consent):
    if not consent:
        return f"Consent required to enroll a voice.\n\n{CONSENT_TEXT}", _enrolled_rows()
    samples, sr = _to_float_mono(audio)
    if samples is None or len(samples) == 0:
        return "Please record or upload a voice sample.", _enrolled_rows()
    if not (name and transcript.strip()):
        return "Name and an exact transcript of the sample are both required.", _enrolled_rows()
    cfg = Config.from_env()
    store = EnrollmentStore(cfg.enrollment_dir)
    try:
        rec = enroll_from_array(store, audio=samples, sample_rate=sr, display_name=name,
                                transcript=transcript, language=language, consent_given=True)
    except Exception as exc:
        return f"Enrollment failed: {exc}", _enrolled_rows()
    return f"✅ Enrolled **{rec.display_name}** as `{rec.speaker_id}`.", _enrolled_rows()


def _enrolled_rows():
    cfg = Config.from_env()
    store = EnrollmentStore(cfg.enrollment_dir)
    rows = []
    for sid in store.list_ids():
        r = store.load_record(sid)
        rows.append([r.speaker_id, r.display_name, r.language, r.transcript[:40]])
    return rows


def preview(phrase, voice_url):
    cfg = Config.from_env()
    if voice_url:
        cfg.voice_clone_url = voice_url
    store = EnrollmentStore(cfg.enrollment_dir)
    voice = VoiceCloneClient(cfg.voice_clone_url, store, cfg.voice_clone_timeout_s)
    text = phrase.strip() or pick_phrase.__doc__ or "Shhh, it's okay."
    result = voice.synth(text, language=cfg.language)
    if result is not None:
        return f"Spoken in cloned voice via {cfg.voice_clone_url}.", _to_gradio_audio(*result)
    track = pick_fallback_track(cfg.assets_dir)
    if track is None:
        return ("No GPU voice service reachable and no fallback track in "
                f"`{cfg.assets_dir}`.", None)
    import soundfile as sf

    data, sr = sf.read(str(track), dtype="float32", always_2d=False)
    return f"GPU voice unavailable — playing fallback track `{track.name}`.", _to_gradio_audio(data, sr)


def build_app():
    import gradio as gr

    with gr.Blocks(title="Peeky — AI Baby/Pet Monitor") as app:
        gr.Markdown("# 🐣 Peeky — AI Baby/Pet Monitor (Reachy Mini)")
        gr.Markdown(SAFETY)

        with gr.Tab("Monitor"):
            gr.Markdown("Upload or record room audio; Peeky detects a sustained "
                        "cry and soothes (cloned voice or fallback track + motion).")
            audio_in = gr.Audio(sources=["upload", "microphone"], type="numpy",
                                label="Room audio clip")
            with gr.Accordion("Detection settings", open=False):
                cry_thr = gr.Slider(0.1, 0.95, value=0.55, step=0.05, label="Cry score threshold")
                sustain = gr.Slider(0.5, 10.0, value=3.0, step=0.5, label="Sustain seconds")
                cooldown = gr.Slider(0.0, 120.0, value=30.0, step=5.0, label="Cooldown seconds")
                min_snr = gr.Slider(0.0, 20.0, value=3.0, step=1.0, label="Min SNR (dB)")
                smoothing = gr.Slider(1, 15, value=5, step=1, label="Smoothing window (frames)")
                vad_thr = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="VAD threshold")
                reason = gr.Checkbox(value=False, label="Enable weak cry-reason hint (advisory)")
                use_voice = gr.Checkbox(value=False, label="Attempt caregiver voice clone (GPU)")
                voice_url = gr.Textbox(value="", label="Voice service URL (blank = config default)")
            analyze_btn = gr.Button("Analyze clip", variant="primary")
            summary = gr.Markdown()
            soothe_audio = gr.Audio(label="Soothing playback", type="numpy")
            score_plot = gr.LinePlot(x="t", y="score", title="Cry score over time", height=220)
            timeline = gr.Dataframe(
                headers=["t (s)", "event", "score", "snr (dB)", "voiced"],
                label="Per-window timeline", wrap=True)
            analyze_btn.click(
                analyze,
                inputs=[audio_in, cry_thr, sustain, cooldown, min_snr, smoothing,
                        vad_thr, reason, use_voice, voice_url],
                outputs=[summary, soothe_audio, timeline, score_plot])

        with gr.Tab("Enroll caregiver voice"):
            gr.Markdown("Clone a caregiver's voice to soothe in their absence. "
                        "**Only enroll a consenting caregiver's own voice.**")
            gr.Markdown(f"> {CONSENT_TEXT}")
            e_audio = gr.Audio(sources=["upload", "microphone"], type="numpy",
                               label="Voice sample")
            e_name = gr.Textbox(label="Caregiver name", placeholder="Mom")
            e_transcript = gr.Textbox(label="Exact words spoken in the sample",
                                      placeholder="hush now little one, everything is okay")
            e_lang = gr.Dropdown(choices=["en", "es", "fr", "de", "hi", "zh"],
                                 value="en", label="Language")
            e_consent = gr.Checkbox(value=False, label="I confirm consent (required)")
            e_btn = gr.Button("Enroll voice", variant="primary")
            e_status = gr.Markdown()
            e_list = gr.Dataframe(headers=["id", "name", "lang", "transcript"],
                                  label="Enrolled voices", value=_enrolled_rows())
            e_btn.click(enroll, inputs=[e_audio, e_name, e_transcript, e_lang, e_consent],
                        outputs=[e_status, e_list])

        with gr.Tab("Soothe preview"):
            gr.Markdown("Hear a phrase in the enrolled caregiver voice (falls back "
                        "to a soothing track if the GPU service is unreachable).")
            p_phrase = gr.Textbox(label="Phrase", value="Shhh, it's okay. Mama's here, little one.")
            p_url = gr.Textbox(value="", label="Voice service URL (blank = config default)")
            p_btn = gr.Button("Speak", variant="primary")
            p_status = gr.Markdown()
            p_audio = gr.Audio(label="Output", type="numpy")
            p_btn.click(preview, inputs=[p_phrase, p_url], outputs=[p_status, p_audio])

        with gr.Tab("About & Safety"):
            gr.Markdown(SAFETY)
            gr.Markdown(
                "Peeky runs a layered, precision-first pipeline (VAD → ensemble "
                "classifier → temporal smoothing → SNR/sustain gates) and clones a "
                "consenting caregiver's voice on a local GPU (VoxCPM2). See "
                "`PLAN.md` and `ROBUSTNESS.md`. Heavy models are optional; without "
                "them Peeky uses numpy fallbacks so this UI runs anywhere.")

    return app


def main():
    logging.basicConfig(level=logging.INFO)
    import os

    app = build_app()
    app.launch(server_name=os.environ.get("PEEKY_WEB_HOST", "127.0.0.1"),
               server_port=int(os.environ.get("PEEKY_WEB_PORT", "7860")))


if __name__ == "__main__":
    main()
