"""The Peeky perception+soothe loop, wired from the building blocks.

Flow per audio frame (audio-derived clock keeps it deterministic):

    frame -> VAD -> [noise-floor update] -> window buffer
          -> preprocess (DC-block, normalize, SNR) -> classifier/ensemble
          -> temporal smoothing -> SNR/hysteresis gate -> DetectionResult
          -> (reason hint -> episode aggregator)
          -> SootheController (debounce/sustain/cooldown) -> SootheDecision
          -> execute: speak (cloned voice or fallback track) + comfort motion
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Config
from .detect.classifier import HeuristicClassifier, make_classifier
from .detect.ensemble import EnsembleClassifier
from .detect.events import CryReason, DetectionResult, SoundEvent
from .detect.preprocess import NoiseFloor, WindowPreprocessor, rms
from .detect.reason import EpisodeReasonAggregator, make_reason_hinter
from .detect.smoothing import Hysteresis, TemporalSmoother
from .detect.vad import make_vad
from .soothe.controller import SootheController, SootheDecision
from .soothe.motion import ComfortMotion
from .soothe.responses import pick_fallback_track, pick_phrase

log = logging.getLogger("peeky.pipeline")


@dataclass
class SootheEvent:
    decision: SootheDecision
    phrase: str
    used_clone: bool
    played: bool
    audio: Optional[tuple[np.ndarray, int]] = None  # rendered soothing audio, if any
    at_seconds: float = 0.0


class Pipeline:
    def __init__(self, config: Config, audio_io, *, mini=None, voice_client=None,
                 on_soothe=None, on_window=None):
        self.cfg = config
        self.audio = audio_io
        self.voice = voice_client
        self.on_soothe = on_soothe
        self.on_window = on_window
        self.frame_seconds = config.frame_ms / 1000.0

        self.vad = make_vad(config.sample_rate, config.vad_threshold,
                            prefer_ml=config.prefer_ml_vad)
        self.classifier = self._build_classifier(config)

        self.noise = NoiseFloor()
        self.prep = WindowPreprocessor(config.sample_rate)
        self.smoother = TemporalSmoother(window=config.smoothing_window)
        self.hyst = Hysteresis(enter=config.cry_score_threshold,
                               exit=config.cry_score_threshold * 0.6)
        self.reason_hinter = make_reason_hinter(config.reason_hint_enabled)
        self.reason_agg = EpisodeReasonAggregator()
        self.controller = SootheController(
            config.cry_score_threshold, config.sustain_seconds, config.cooldown_seconds)
        self.motion = ComfortMotion(mini)

        win_frames = max(1, round(config.classify_window_seconds / self.frame_seconds))
        self._win: deque[np.ndarray] = deque(maxlen=win_frames)
        self._t = 0.0

    @staticmethod
    def _local_ensemble(base, config: Config):
        if config.use_ensemble and not isinstance(base, HeuristicClassifier):
            return EnsembleClassifier(
                [(base, 0.7), (HeuristicClassifier(), 0.3)],
                min_score=config.cry_score_threshold * 0.7)
        return base

    def _build_classifier(self, config: Config):
        base = make_classifier(prefer_ml=config.prefer_ml_classifier)
        if config.use_remote_cry:
            from .detect.remote_classifier import RemoteEventClassifier

            remote = RemoteEventClassifier(config.cry_service_url,
                                           config.cry_service_timeout_s, fallback=base)
            if remote.available():
                log.info("Classifier: remote cry service @ %s", config.cry_service_url)
                return remote
            log.info("Classifier: remote cry service down; using local")
        return self._local_ensemble(base, config)

    @property
    def clock(self) -> float:
        return self._t

    def calibrate(self, frames: Optional[list[np.ndarray]] = None) -> None:
        if frames is None:
            frames = []
            n = max(1, round(self.cfg.calibration_seconds / self.frame_seconds))
            for _ in range(n):
                f = self.audio.read()
                if f is None:
                    break
                frames.append(f)
        self.noise.calibrate(frames)
        log.info("Calibrated ambient noise floor: rms=%.5f", self.noise.noise_rms)

    def process_frame(self, frame: np.ndarray) -> Optional[SootheEvent]:
        self._t += self.frame_seconds
        active, _ = self.vad.is_active(frame)
        self.noise.update(rms(frame), active)
        self._win.append(frame)
        if len(self._win) < self._win.maxlen:
            return None

        window = np.concatenate(list(self._win))
        norm, snr, _ = self.prep.prepare(window, self.noise.noise_rms)
        event, score = self.classifier.classify(norm, self.cfg.sample_rate)
        event, score = self.smoother.update(event, score)

        # Gate: a cry must be voiced AND loud enough above the room floor.
        snr_ok = snr >= self.cfg.min_snr_db
        cry_signal = score if (event in (SoundEvent.BABY_CRY, SoundEvent.DOG) and snr_ok) else 0.0
        cry_stable = self.hyst.update(cry_signal)

        reason = CryReason.UNKNOWN
        reason_conf = 0.0
        if cry_stable and event == SoundEvent.BABY_CRY and self.reason_hinter:
            r, c = self.reason_hinter.hint(window, self.cfg.sample_rate)
            self.reason_agg.add(r, c)

        result = DetectionResult(
            event=event,
            score=score,
            is_voiced=active and snr_ok and cry_stable,
            reason=reason,
            reason_confidence=reason_conf,
        )
        if self.on_window:
            self.on_window(self._t, event, score, snr, result.is_voiced)
        decision = self.controller.observe(result, self._t)
        if decision is None:
            return None

        if self.reason_hinter:
            r, c = self.reason_agg.result()
            decision.reason, decision.reason_confidence = r, c
        self.reason_agg.reset()
        self.smoother.reset()
        self.hyst.reset()
        return self._execute(decision)

    def _execute(self, decision: SootheDecision) -> SootheEvent:
        phrase = pick_phrase(decision.event, decision.reason)
        audio_sr = self._synthesize(phrase)
        used_clone = audio_sr is not None
        if audio_sr is None:
            audio_sr = self._fallback_track()

        played = False

        def _play():
            nonlocal played
            if audio_sr is not None:
                self.audio.play(audio_sr[0], audio_sr[1])
                played = True

        player = threading.Thread(target=_play, daemon=True)
        player.start()
        self.motion.comfort(cycles=2)
        player.join(timeout=30.0)

        event = SootheEvent(decision=decision, phrase=phrase,
                            used_clone=used_clone, played=played,
                            audio=audio_sr, at_seconds=self._t)
        if self.on_soothe:
            self.on_soothe(event)
        return event

    def _synthesize(self, phrase: str) -> Optional[tuple[np.ndarray, int]]:
        if self.voice is None:
            return None
        return self.voice.synth(phrase, language=self.cfg.language)

    def _fallback_track(self) -> Optional[tuple[np.ndarray, int]]:
        track = pick_fallback_track(self.cfg.assets_dir)
        if track is None:
            log.warning("No cloned voice and no fallback track; soothing with motion only.")
            return None
        import soundfile as sf

        from .audio.io import to_mono

        data, sr = sf.read(str(track), dtype="float32", always_2d=False)
        return to_mono(data), int(sr)

    def run(self, stop_event: Optional[threading.Event] = None) -> None:
        self.audio.start()
        try:
            self.calibrate()
            while stop_event is None or not stop_event.is_set():
                frame = self.audio.read()
                if frame is None:
                    break
                self.process_frame(frame)
        finally:
            self.audio.stop()
            self.motion.rest()
