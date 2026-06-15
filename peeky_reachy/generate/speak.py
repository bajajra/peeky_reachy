"""Render generated text in the cloned caregiver voice, with on-disk caching.

A bedtime story is ~5 minutes of TTS; doing that on every "play" call burns
GPU time and battery for no reason — stories don't change between plays. We
cache by SHA-256 of ``(text, speaker_id, language)`` under ``cache/stories/``
(gitignored) so the second play is instant and offline.

Falls back to ``None`` if the voice service is unreachable AND nothing is
cached, so callers can choose to play a pre-recorded soothing track instead.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("peeky.generate.speak")

CACHE_DIR_DEFAULT = "cache/stories"


def _cache_key(text: str, speaker_id: str, language: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(b"\0")
    h.update(speaker_id.encode("utf-8"))
    h.update(b"\0")
    h.update(language.encode("utf-8"))
    return h.hexdigest()[:24]


class SpeakCache:
    def __init__(self, voice_client=None, cache_dir: str = CACHE_DIR_DEFAULT):
        self.voice_client = voice_client
        self.cache_dir = Path(cache_dir)

    def speak(self, text: str, speaker_id: Optional[str] = None,
              language: str = "en") -> Optional[tuple[np.ndarray, int]]:
        sid = speaker_id or self._default_speaker()
        if not sid:
            log.info("No enrolled speaker; cannot synthesize.")
            return None

        key = _cache_key(text, sid, language)
        cached = self._load(key)
        if cached is not None:
            return cached

        if self.voice_client is None:
            log.info("No voice client and nothing cached; cannot speak.")
            return None

        result = self.voice_client.synth(text, speaker_id=sid, language=language)
        if result is None:
            return None
        samples, sr = result
        self._save(key, samples, sr)
        return samples, sr

    def cached_paths(self) -> list[Path]:
        if not self.cache_dir.exists():
            return []
        return sorted(self.cache_dir.glob("*.wav"))

    def _default_speaker(self) -> Optional[str]:
        store = getattr(self.voice_client, "store", None)
        if store is None:
            return None
        try:
            return store.default_id()
        except Exception:
            return None

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.wav"

    def _load(self, key: str) -> Optional[tuple[np.ndarray, int]]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
            return np.asarray(data, dtype=np.float32), int(sr)
        except Exception as exc:
            log.warning("Cache read failed for %s: %s", path, exc)
            return None

    def _save(self, key: str, samples: np.ndarray, sr: int) -> None:
        try:
            import soundfile as sf

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            sf.write(str(self._path(key)), samples, sr, subtype="PCM_16")
        except Exception as exc:
            log.info("Cache write skipped (%s); continuing without caching.", exc)
