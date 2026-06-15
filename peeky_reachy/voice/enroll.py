"""Capture/import a caregiver voice sample behind an explicit consent gate."""

from __future__ import annotations

import logging
import re

import numpy as np

from .store import EnrollmentRecord, EnrollmentStore

log = logging.getLogger("peeky.enroll")

CONSENT_TEXT = (
    "I confirm I am this caregiver (or their authorized guardian) and I consent "
    "to Peeky cloning this voice solely to soothe my child on this device. The "
    "sample stays on my local network and is never uploaded to the cloud."
)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "caregiver"


def enroll_from_array(store: EnrollmentStore, *, audio: np.ndarray, sample_rate: int,
                      display_name: str, transcript: str, language: str = "en",
                      consent_given: bool = False) -> EnrollmentRecord:
    if not consent_given:
        raise PermissionError(
            "Voice enrollment requires explicit consent. Re-run with consent "
            "confirmed. Consent statement:\n" + CONSENT_TEXT)
    if not transcript.strip():
        raise ValueError("A transcript of the sample is required (VoxCPM2 ultimate cloning).")
    return store.save(
        speaker_id=_slugify(display_name),
        display_name=display_name,
        transcript=transcript,
        language=language,
        consent_given=consent_given,
        consent_text=CONSENT_TEXT,
        audio=audio,
        sample_rate=sample_rate,
    )


def enroll_from_wav(store: EnrollmentStore, *, wav_path: str, display_name: str,
                    transcript: str, language: str = "en",
                    consent_given: bool = False) -> EnrollmentRecord:
    import soundfile as sf

    from ..audio.io import to_mono

    data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    return enroll_from_array(store, audio=to_mono(data), sample_rate=sr,
                             display_name=display_name, transcript=transcript,
                             language=language, consent_given=consent_given)


def record_and_enroll(store: EnrollmentStore, audio_io, *, seconds: float,
                      display_name: str, transcript: str, language: str = "en",
                      consent_given: bool = False) -> EnrollmentRecord:
    """Record ``seconds`` of audio from an AudioIO source, then enroll it."""
    if not consent_given:
        raise PermissionError("Voice enrollment requires explicit consent.\n" + CONSENT_TEXT)
    audio_io.start()
    try:
        needed = int(seconds * audio_io.sample_rate)
        chunks: list[np.ndarray] = []
        have = 0
        while have < needed:
            frame = audio_io.read()
            if frame is None:
                break
            chunks.append(frame)
            have += len(frame)
    finally:
        audio_io.stop()
    audio = np.concatenate(chunks)[:needed] if chunks else np.zeros(0, dtype=np.float32)
    return enroll_from_array(store, audio=audio, sample_rate=audio_io.sample_rate,
                             display_name=display_name, transcript=transcript,
                             language=language, consent_given=consent_given)
