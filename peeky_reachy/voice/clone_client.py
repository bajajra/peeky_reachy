"""HTTP client to the VoxCPM2 GPU service (agent-2's T11 contract).

Contract (see standup.md):
  GET  /healthz        -> {"ok", "model_loaded", "model"}
  GET  /references     -> ["id", ...]
  POST /references     {"id", "audio_wav_b64", "transcript"} -> registers ref
  POST /synthesize     {"text", "reference_id", "language", "sample_rate"} -> audio/wav

Every method fails soft (returns None / False) so the soothe path can fall back
to a pre-recorded track when the GPU box is unreachable.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

import numpy as np

from .store import EnrollmentStore

log = logging.getLogger("peeky.voice")


class VoiceCloneClient:
    def __init__(self, base_url: str, store: EnrollmentStore, timeout_s: float = 20.0,
                 out_sample_rate: int = 48000):
        self.base_url = base_url.rstrip("/")
        self.store = store
        self.timeout_s = timeout_s
        self.out_sample_rate = out_sample_rate
        self._registered: set[str] = set()

    def _client(self):
        import httpx

        return httpx.Client(base_url=self.base_url, timeout=self.timeout_s)

    def available(self) -> bool:
        try:
            with self._client() as c:
                r = c.get("/healthz")
                return r.status_code == 200 and bool(r.json().get("model_loaded"))
        except Exception as exc:
            log.info("Voice service unavailable: %s", exc)
            return False

    def ensure_reference(self, speaker_id: str) -> bool:
        if speaker_id in self._registered:
            return True
        try:
            with self._client() as c:
                existing = c.get("/references").json()
                if speaker_id in existing:
                    self._registered.add(speaker_id)
                    return True
                rec = self.store.load_record(speaker_id)
                wav_b64 = base64.b64encode(self.store.load_audio_bytes(speaker_id)).decode()
                r = c.post("/references", json={
                    "id": speaker_id,
                    "audio_wav_b64": wav_b64,
                    "transcript": rec.transcript,
                })
                r.raise_for_status()
                self._registered.add(speaker_id)
                return True
        except Exception as exc:
            log.warning("Could not register reference '%s': %s", speaker_id, exc)
            return False

    def synth(self, text: str, speaker_id: Optional[str] = None,
              language: str = "en") -> Optional[tuple[np.ndarray, int]]:
        speaker_id = speaker_id or self.store.default_id()
        if not speaker_id:
            log.info("No enrolled caregiver voice; cannot clone.")
            return None
        if not self.ensure_reference(speaker_id):
            return None
        try:
            with self._client() as c:
                r = c.post("/synthesize", json={
                    "text": text,
                    "reference_id": speaker_id,
                    "language": language,
                    "sample_rate": self.out_sample_rate,
                })
                r.raise_for_status()
                import soundfile as sf

                audio, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=False)
                return np.asarray(audio, dtype=np.float32), int(sr)
        except Exception as exc:
            log.warning("Voice synthesis failed: %s", exc)
            return None
