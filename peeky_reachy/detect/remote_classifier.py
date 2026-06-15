"""Client for the remote baby-cry classification service on turing.

Keeps the heavy YAMNet model off the robot/dev box and on the GPU host, while
*never* breaking the loop: any network/service error falls back to a local
classifier so detection keeps working if turing is down.
"""

from __future__ import annotations

import base64
import io
import logging

import numpy as np

from .classifier import EventClassifier, HeuristicClassifier
from .events import SoundEvent

log = logging.getLogger("peeky.cry_remote")


def _wav_b64(samples: np.ndarray, sample_rate: int) -> str:
    """Encode a window as a base64 mono 16-bit PCM WAV.

    Accepts float32 in [-1, 1] (typical from the pipeline) and int16 directly
    (typical from raw soundfile reads), and downmixes multi-channel input to
    mono so the server doesn't see a 2x-long clip from a stereo array. Empty
    input still produces a valid (zero-frame) WAV.
    """
    import wave

    arr = np.asarray(samples)
    if arr.size == 0:
        pcm = np.zeros(0, dtype=np.int16)
    elif arr.ndim > 1:
        # Downmix to mono (mean across channels). The server would do this
        # too, but doing it on the client means the WAV matches what the user
        # actually heard, not 2x the duration.
        arr = arr.mean(axis=1)
        if arr.dtype == np.int16:
            pcm = arr.astype(np.int16)
        elif np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            scale = float(abs(info.min))
            pcm = np.round(arr.astype(np.float64) / scale * 32767.0).astype(np.int16)
        else:
            pcm = np.round(np.clip(arr.astype(np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
    elif arr.dtype == np.int16:
        # Already int16 PCM — pass through bit-exact, don't re-quantize.
        pcm = arr.astype(np.int16)
    elif np.issubdtype(arr.dtype, np.integer):
        # int32 (or other) input: scale into int16 range. Use the full signed
        # magnitude (|min|) so a value of +/-|min|/2+1 round-trips through the
        # representation range symmetrically.
        info = np.iinfo(arr.dtype)
        scale = float(abs(info.min))
        pcm = np.round(arr.astype(np.float64) / scale * 32767.0).astype(np.int16)
    else:
        pcm = np.round(np.clip(arr.astype(np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


class RemoteEventClassifier(EventClassifier):
    def __init__(self, base_url: str, timeout_s: float = 5.0,
                 fallback: EventClassifier | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.fallback = fallback or HeuristicClassifier()

    def _client(self):
        import httpx

        return httpx.Client(base_url=self.base_url, timeout=self.timeout_s)

    def available(self) -> bool:
        try:
            with self._client() as c:
                r = c.get("/healthz")
                return r.status_code == 200 and bool(r.json().get("model_loaded"))
        except Exception as exc:
            log.info("Cry service unavailable: %s", exc)
            return False

    def classify(self, window: np.ndarray, sample_rate: int) -> tuple[SoundEvent, float]:
        try:
            with self._client() as c:
                r = c.post("/classify", json={
                    "audio_wav_b64": _wav_b64(window, sample_rate),
                })
                r.raise_for_status()
                data = r.json()
                return SoundEvent(data["event"]), float(data["score"])
        except Exception as exc:
            log.warning("Remote classify failed (%s); using local fallback", exc)
            return self.fallback.classify(window, sample_rate)
