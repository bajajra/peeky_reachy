"""Reachy Mini app entry point (discovered via the reachy_mini_apps group).

The daemon constructs the app and calls ``run(reachy_mini, stop_event)``. On the
robot, audio flows through the 4-mic array/speaker (ReachyAudioIO) and motion
through the same handle. The base class is guarded so this module stays
importable on a dev machine without the SDK installed.
"""

from __future__ import annotations

import logging

from .config import Config
from .pipeline import Pipeline, SootheEvent
from .voice.clone_client import VoiceCloneClient
from .voice.store import EnrollmentStore

log = logging.getLogger("peeky.app")

try:
    from reachy_mini import ReachyMiniApp  # type: ignore
except Exception:  # pragma: no cover - SDK absent on dev machine
    ReachyMiniApp = object  # type: ignore


def _log_soothe(event: SootheEvent) -> None:
    log.info("Soothed: %s | clone=%s played=%s | \"%s\"",
             event.decision.event.value, event.used_clone, event.played, event.phrase)


class PeekyApp(ReachyMiniApp):
    def run(self, reachy_mini, stop_event):  # noqa: ANN001
        from .audio.io import ReachyAudioIO

        cfg = Config.from_env()
        audio = ReachyAudioIO(reachy_mini, cfg.sample_rate, cfg.frame_size)
        store = EnrollmentStore(cfg.enrollment_dir)
        voice = VoiceCloneClient(cfg.voice_clone_url, store, cfg.voice_clone_timeout_s)
        pipeline = Pipeline(cfg, audio, mini=reachy_mini, voice_client=voice,
                            on_soothe=_log_soothe)
        log.info("Peeky is listening. (Companion only — not a safety/medical monitor.)")
        pipeline.run(stop_event)
