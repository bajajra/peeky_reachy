"""Gradio v6 web UI for Peeky.

Per `vision.md`, the primary experience is **autonomous, always-on listening**:
you turn Peeky on, it listens, understands, and soothes. The upload-and-analyze
flow lives in a "Debug" tab. Detection, soothing, and the live 3D Reachy
companion all run over the same `Pipeline` / `StreamingSession` seams.

Launch with ``peeky-web`` or ``python -m peeky_reachy.webapp``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from . import reachy3d
from .audio.io import ArrayAudioIO, LocalAudioIO
from .config import Config
from .detect.events import SoundEvent
from .pipeline import Pipeline
from .soothe.responses import pick_fallback_track, pick_phrase
from .streaming import StreamingSession, _QueueAudioIO
from .voice.clone_client import VoiceCloneClient
from .voice.enroll import CONSENT_TEXT, enroll_from_array
from .voice.store import EnrollmentStore

log = logging.getLogger("peeky.webapp")

SAFETY = (
    "**Peeky is a soothing companion, not a safety/medical/SIDS monitor.** "
    "Never rely on it to keep a child safe. Cry-*reason* guesses are advisory and "
    "low-confidence — keep a caregiver in the loop."
)


# --------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Live monitor session (singleton, lives for the Gradio process)
# --------------------------------------------------------------------------


class _LiveMonitor:
    """Owns the StreamingSession + LocalAudioIO mic for the Live monitor tab.

    Server-side mic is used (laptop, via sounddevice). On macOS, the user must
    grant Microphone permission to the terminal/Gradio process on first run.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.session: Optional[StreamingSession] = None
        self.mic: Optional[LocalAudioIO] = None
        self.feeder_thread: Optional[threading.Thread] = None
        self.feeder_stop: Optional[threading.Event] = None
        self.error: Optional[str] = None

    def is_active(self) -> bool:
        return self.session is not None and self.session.running

    def start(self, cfg: Config, voice_client=None) -> str:
        if self.is_active():
            return "🔴 Already monitoring (click Stop first)."
        self._reset_state()
        try:
            self.mic = LocalAudioIO(cfg.sample_rate, cfg.frame_size)
            self.mic.start()
        except Exception as exc:
            self.mic = None
            return (
                f"❌ Failed to open microphone: {exc}\n\n"
                "On macOS, grant Microphone permission to your terminal/Gradio "
                "process (System Settings → Privacy & Security → Microphone). "
                "Then click Start again."
            )
        qio = _QueueAudioIO(sample_rate=cfg.sample_rate, frame_size=cfg.frame_size)
        self.session = StreamingSession(cfg, voice_client=voice_client, audio_io=qio)
        self.session.start()
        self.feeder_stop = threading.Event()
        self.feeder_thread = threading.Thread(
            target=self._feed_loop, name="peeky-mic-feeder", daemon=True)
        self.feeder_thread.start()
        return "🔴 Listening — speak or play a cry clip. Peeky soothes on a sustained cry."

    def _feed_loop(self) -> None:
        assert self.mic is not None
        try:
            while self.feeder_stop is not None and not self.feeder_stop.is_set():
                frame = self.mic.read()
                if frame is None or frame.size == 0:
                    continue
                if self.session is not None:
                    self.session.feed(frame, self.mic.sample_rate)
        except Exception as exc:
            with self._lock:
                self.error = str(exc)
            log.exception("mic feeder crashed")

    def stop(self) -> str:
        if not self.is_active() and self.session is None:
            return "🟢 Idle (not monitoring)."
        if self.feeder_stop is not None:
            self.feeder_stop.set()
        if self.feeder_thread is not None:
            self.feeder_thread.join(timeout=2.0)
            self.feeder_thread = None
        if self.session is not None:
            self.session.stop()
            count = self.session.status().soothe_count
        else:
            count = 0
        if self.mic is not None:
            try:
                self.mic.stop()
            except Exception:
                pass
            self.mic = None
        return f"🟢 Stopped. {count} soothe event(s) this session."

    def snapshot(self):
        if self.session is None:
            return None
        return self.session.status()

    def recent_windows(self) -> list[list]:
        if self.session is None:
            return []
        return self.session.recent_windows()

    def last_soothe(self):
        if self.session is None:
            return None
        return self.session.last_soothe()

    def _reset_state(self) -> None:
        with self._lock:
            self.error = None
            self.feeder_stop = None
            self.feeder_thread = None
            self.session = None
            self.mic = None


_LIVE = _LiveMonitor()


# --------------------------------------------------------------------------
# Live monitor UI callbacks
# --------------------------------------------------------------------------


def _start_live(cry_thr, sustain, cooldown, min_snr, smoothing, vad_thr,
                reason, use_voice, voice_url):
    cfg = _config_from_ui(cry_thr, sustain, cooldown, min_snr, smoothing, vad_thr,
                          reason, voice_url)
    voice = None
    if use_voice:
        store = EnrollmentStore(cfg.enrollment_dir)
        voice = VoiceCloneClient(cfg.voice_clone_url, store, cfg.voice_clone_timeout_s)
    return _LIVE.start(cfg, voice_client=voice)


def _stop_live():
    return _LIVE.stop()


def _live_poll():
    """Polled by the gr.Timer; updates the Live monitor UI from session state."""
    if _LIVE.session is None:
        return ("🟢 Idle — click **Start monitoring** to begin live listening.",
                reachy3d.DEFAULT_STATE, [], None)
    s = _LIVE.snapshot()
    if s is None:
        return ("🟢 Idle", reachy3d.DEFAULT_STATE, [], None)
    if _LIVE.error:
        return (f"❌ {_LIVE.error}", reachy3d.DEFAULT_STATE,
                _LIVE.recent_windows(), None)
    if s.running:
        status = (
            f"🔴 **Listening** — `{s.last_event}` (score {s.last_score:.2f}, "
            f"SNR {s.last_snr_db:.1f} dB) · clock {s.clock:.1f}s · "
            f"{s.soothe_count} soothe(s) · cooldown {s.cooldown_remaining:.0f}s"
        )
    else:
        status = (f"🟢 Stopped — `{s.last_event}` last · "
                  f"{s.soothe_count} soothe(s) this session")
    last = _LIVE.last_soothe()
    audio = _to_gradio_audio(*last.audio) if (last is not None and last.audio) else None
    return (status, s.state, _LIVE.recent_windows(), audio)


# --------------------------------------------------------------------------
# Debug / Analyze clip (the old upload flow, demoted)
# --------------------------------------------------------------------------


def analyze(audio, cry_thr, sustain, cooldown, min_snr, smoothing, vad_thr,
            reason, use_voice, voice_url):
    samples, sr = _to_float_mono(audio)
    if samples is None or len(samples) == 0:
        return "Please record or upload an audio clip first.", None, [], None, "idle"

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
    reachy_state = reachy3d.reachy_state_from_run(timeline, events)

    if not events:
        summary = (f"### No soothing triggered\n"
                   f"Analyzed {pipe.clock:.1f}s. No sustained cry crossed the "
                   f"threshold (needs ≥ {cfg.sustain_seconds:.0f}s above "
                   f"{cfg.cry_score_threshold:.2f} and SNR ≥ {cfg.min_snr_db:.0f} dB).")
        return summary, None, rows, plot_df, reachy_state

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
    return summary, out_audio, rows, plot_df, reachy_state


def _plot_df(timeline):
    try:
        import pandas as pd
    except Exception:
        return None
    if not timeline:
        return pd.DataFrame({"t": [], "score": []})
    return pd.DataFrame({"t": [r[0] for r in timeline], "score": [r[2] for r in timeline]})


# --------------------------------------------------------------------------
# Enroll + preview (unchanged)
# --------------------------------------------------------------------------


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
    text = phrase.strip() or pick_phrase(SoundEvent.BABY_CRY)
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


# --------------------------------------------------------------------------
# Settings widgets (reused between Live and Debug tabs)
# --------------------------------------------------------------------------


def _settings_block():
    """Returns a tuple of (cry_thr, sustain, cooldown, min_snr, smoothing,
    vad_thr, reason, use_voice, voice_url) — same widgets the Debug tab uses."""
    import gradio as gr

    cry_thr = gr.Slider(0.1, 0.95, value=0.55, step=0.05, label="Cry score threshold")
    sustain = gr.Slider(0.5, 10.0, value=3.0, step=0.5, label="Sustain seconds")
    cooldown = gr.Slider(0.0, 120.0, value=30.0, step=5.0, label="Cooldown seconds")
    min_snr = gr.Slider(0.0, 20.0, value=3.0, step=1.0, label="Min SNR (dB)")
    smoothing = gr.Slider(1, 15, value=5, step=1, label="Smoothing window (frames)")
    vad_thr = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="VAD threshold")
    reason = gr.Checkbox(value=False, label="Enable weak cry-reason hint (advisory)")
    use_voice = gr.Checkbox(value=False, label="Attempt caregiver voice clone (GPU)")
    voice_url = gr.Textbox(value="", label="Voice service URL (blank = config default)")
    return cry_thr, sustain, cooldown, min_snr, smoothing, vad_thr, reason, use_voice, voice_url


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


def build_app():
    import gradio as gr

    with gr.Blocks(title="Peeky — AI Baby/Pet Monitor") as app:
        with gr.Row():
            with gr.Column(elem_id=reachy3d.SIDEBAR_ELEM_ID, min_width=250):
                gr.HTML(reachy3d.sidebar_html())
                # Hidden bridge: _live_poll() writes the companion state here;
                # a 200 ms JS poll feeds it into window.peekyReachy.setState
                # (see reachy3d.py). analyze() in the Debug tab also writes here.
                reachy_state = gr.Textbox(value=reachy3d.DEFAULT_STATE,
                                          visible=False,
                                          elem_id=reachy3d.STATE_ELEM_ID)
            with gr.Column(scale=3):
                gr.Markdown("# 🐣 Peeky — AI Baby/Pet Monitor (Reachy Mini)")
                gr.Markdown(SAFETY)
                gr.Markdown(
                    "**How it works:** turn it on once, Peeky listens continuously, "
                    "classifies each sound (cry / pet / speech / silence), and "
                    "**auto-soothes on a sustained cry** in the caregiver's cloned "
                    "voice (or a fallback track) with comfort motion. The 3D Reachy "
                    "companion on the left reacts in real time."
                )

                # ---------------- Live monitor (PRIMARY) ----------------
                with gr.Tab("🔴 Live monitor"):
                    live_status = gr.Markdown(
                        "🟢 Idle — click **Start monitoring** to begin live listening."
                    )
                    with gr.Row():
                        start_btn = gr.Button("Start monitoring", variant="primary")
                        stop_btn = gr.Button("Stop")
                    with gr.Accordion("Detection settings", open=False):
                        (cry_thr_l, sustain_l, cooldown_l, min_snr_l, smoothing_l,
                         vad_thr_l, reason_l, use_voice_l, voice_url_l) = _settings_block()
                    live_timeline = gr.Dataframe(
                        headers=["t (s)", "event", "score", "snr (dB)", "voiced"],
                        label="Live per-window timeline (last 50)", wrap=True)
                    live_soothe = gr.Audio(label="Last soothing playback (auto)", type="numpy")
                    # gr.Timer polls _live_poll at 200 ms; updates status, the
                    # 3D Reachy state, the live timeline, and the soothe audio.
                    timer = gr.Timer(value=0.2)
                    timer.tick(_live_poll, outputs=[live_status, reachy_state,
                                                    live_timeline, live_soothe])
                    start_btn.click(
                        _start_live,
                        inputs=[cry_thr_l, sustain_l, cooldown_l, min_snr_l,
                                smoothing_l, vad_thr_l, reason_l, use_voice_l,
                                voice_url_l],
                        outputs=[live_status])
                    stop_btn.click(_stop_live, outputs=[live_status])

                # ---------------- Debug / Analyze clip (DEMOTED) ----------------
                with gr.Tab("🛠 Debug / Analyze clip"):
                    gr.Markdown(
                        "**Debug / demo path** — upload or record a clip and run the "
                        "same pipeline against it. The headline experience is the "
                        "**Live monitor** tab above; this one is for inspecting a "
                        "specific clip's timeline and per-window score."
                    )
                    audio_in = gr.Audio(sources=["upload", "microphone"],
                                        type="numpy", label="Room audio clip")
                    with gr.Accordion("Detection settings", open=False):
                        (cry_thr_d, sustain_d, cooldown_d, min_snr_d, smoothing_d,
                         vad_thr_d, reason_d, use_voice_d, voice_url_d) = _settings_block()
                    analyze_btn = gr.Button("Analyze clip", variant="primary")
                    summary = gr.Markdown()
                    soothe_audio = gr.Audio(label="Soothing playback", type="numpy")
                    score_plot = gr.LinePlot(x="t", y="score", title="Cry score over time", height=220)
                    timeline = gr.Dataframe(
                        headers=["t (s)", "event", "score", "snr (dB)", "voiced"],
                        label="Per-window timeline", wrap=True)
                    analyze_btn.click(
                        analyze,
                        inputs=[audio_in, cry_thr_d, sustain_d, cooldown_d, min_snr_d,
                                smoothing_d, vad_thr_d, reason_d, use_voice_d, voice_url_d],
                        outputs=[summary, soothe_audio, timeline, score_plot,
                                 reachy_state])

                with gr.Tab("Enroll caregiver voice"):
                    gr.Markdown("Clone a caregiver's voice to soothe in their "
                                "absence. **Only enroll a consenting caregiver's "
                                "own voice.**")
                    gr.Markdown(f"> {CONSENT_TEXT}")
                    e_audio = gr.Audio(sources=["upload", "microphone"],
                                       type="numpy", label="Voice sample")
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
                    e_btn.click(enroll,
                                inputs=[e_audio, e_name, e_transcript, e_lang, e_consent],
                                outputs=[e_status, e_list])

                with gr.Tab("Soothe preview"):
                    gr.Markdown("Hear a phrase in the enrolled caregiver voice "
                                "(falls back to a soothing track if the GPU "
                                "service is unreachable).")
                    p_phrase = gr.Textbox(label="Phrase", value="Shhh, it's okay. Mama's here, little one.")
                    p_url = gr.Textbox(value="", label="Voice service URL (blank = config default)")
                    p_btn = gr.Button("Speak", variant="primary")
                    p_status = gr.Markdown()
                    p_audio = gr.Audio(label="Output", type="numpy")
                    p_btn.click(preview, inputs=[p_phrase, p_url], outputs=[p_status, p_audio])

                with gr.Tab("About & Safety"):
                    gr.Markdown(SAFETY)
                    gr.Markdown(
                        "Peeky runs a layered, precision-first pipeline (VAD → "
                        "ensemble classifier → temporal smoothing → SNR/sustain "
                        "gates) and clones a consenting caregiver's voice on a "
                        "local GPU (VoxCPM2). See `PLAN.md`, `ROBUSTNESS.md`, and "
                        "`vision.md`. Heavy models are optional; without them "
                        "Peeky uses numpy fallbacks so this UI runs anywhere."
                    )

        # Re-boot the 3D scene once Gradio has mounted the DOM (idempotent).
        app.load(None, None, None, js=reachy3d.BOOT_JS)

    return app


def main():
    logging.basicConfig(level=logging.INFO)
    import os

    app = build_app()
    app.launch(server_name=os.environ.get("PEEKY_WEB_HOST", "127.0.0.1"),
               server_port=int(os.environ.get("PEEKY_WEB_PORT", "7860")),
               css=reachy3d.SIDEBAR_CSS, head=reachy3d.head_html())


if __name__ == "__main__":
    main()
