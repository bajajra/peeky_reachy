"""Local, consented caregiver voice enrollment store (encrypted-at-rest).

Reference audio never leaves the LAN: it is stored encrypted on disk and only
decrypted in-memory when synthesizing on the local GPU box. Encryption uses
Fernet when ``cryptography`` is installed; otherwise it stores plaintext and
loudly warns (never silently degrade a privacy promise).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("peeky.enroll")

_KEY_FILE = "key.fernet"


@dataclass
class EnrollmentRecord:
    speaker_id: str
    display_name: str
    transcript: str
    language: str
    consent_given: bool
    consent_text: str
    created_at: str
    sample_rate: int


class _Cipher:
    def __init__(self, base_dir: Path):
        self._fernet = None
        try:
            from cryptography.fernet import Fernet

            key_path = base_dir / _KEY_FILE
            if key_path.exists():
                key = key_path.read_bytes()
            else:
                key = Fernet.generate_key()
                base_dir.mkdir(parents=True, exist_ok=True)
                key_path.write_bytes(key)
                key_path.chmod(0o600)
            self._fernet = Fernet(key)
        except Exception as exc:
            log.warning("cryptography unavailable (%s); enrollment audio stored "
                        "UNENCRYPTED. Install 'cryptography' for encryption-at-rest.", exc)

    @property
    def enabled(self) -> bool:
        return self._fernet is not None

    def encrypt(self, data: bytes) -> bytes:
        return self._fernet.encrypt(data) if self._fernet else data

    def decrypt(self, data: bytes) -> bytes:
        return self._fernet.decrypt(data) if self._fernet else data


class EnrollmentStore:
    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self._cipher = _Cipher(self.base)

    def _meta_path(self, speaker_id: str) -> Path:
        return self.base / f"{speaker_id}.json"

    def _audio_path(self, speaker_id: str) -> Path:
        suffix = "wav.enc" if self._cipher.enabled else "wav"
        return self.base / f"{speaker_id}.{suffix}"

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.base.glob("*.json"))

    def default_id(self) -> Optional[str]:
        ids = self.list_ids()
        return ids[0] if ids else None

    def save(self, *, speaker_id: str, display_name: str, transcript: str,
             language: str, consent_given: bool, consent_text: str,
             audio: np.ndarray, sample_rate: int) -> EnrollmentRecord:
        if not consent_given:
            raise PermissionError("Refusing to enroll a voice without explicit consent.")
        import soundfile as sf
        import io

        rec = EnrollmentRecord(
            speaker_id=speaker_id,
            display_name=display_name,
            transcript=transcript,
            language=language,
            consent_given=consent_given,
            consent_text=consent_text,
            created_at=datetime.now(timezone.utc).isoformat(),
            sample_rate=sample_rate,
        )
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio, dtype=np.float32), sample_rate, format="WAV")
        self._audio_path(speaker_id).write_bytes(self._cipher.encrypt(buf.getvalue()))
        self._meta_path(speaker_id).write_text(json.dumps(asdict(rec), indent=2))
        log.info("Enrolled '%s' (%s), encrypted=%s", display_name, speaker_id,
                 self._cipher.enabled)
        return rec

    def load_record(self, speaker_id: str) -> EnrollmentRecord:
        data = json.loads(self._meta_path(speaker_id).read_text())
        return EnrollmentRecord(**data)

    def load_audio_bytes(self, speaker_id: str) -> bytes:
        return self._cipher.decrypt(self._audio_path(speaker_id).read_bytes())

    def delete(self, speaker_id: str) -> None:
        self._meta_path(speaker_id).unlink(missing_ok=True)
        self._audio_path(speaker_id).unlink(missing_ok=True)
