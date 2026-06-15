import numpy as np
import pytest

from peeky_reachy.voice.enroll import enroll_from_array
from peeky_reachy.voice.store import EnrollmentStore
from tests.conftest import SR, baby_cry


def test_consent_gate_blocks_enrollment(tmp_path):
    store = EnrollmentStore(str(tmp_path))
    with pytest.raises(PermissionError):
        enroll_from_array(store, audio=np.zeros(SR, dtype=np.float32), sample_rate=SR,
                          display_name="Mom", transcript="hello", consent_given=False)


def test_transcript_required(tmp_path):
    store = EnrollmentStore(str(tmp_path))
    with pytest.raises(ValueError):
        enroll_from_array(store, audio=np.zeros(SR, dtype=np.float32), sample_rate=SR,
                          display_name="Mom", transcript="   ", consent_given=True)


def test_enroll_roundtrip(tmp_path):
    store = EnrollmentStore(str(tmp_path))
    audio = baby_cry(2.0)  # any voiced sample stands in for a caregiver clip
    rec = enroll_from_array(store, audio=audio, sample_rate=SR, display_name="Mom",
                            transcript="hush now little one", consent_given=True)
    assert rec.speaker_id == "mom"
    assert store.default_id() == "mom"
    loaded = store.load_record("mom")
    assert loaded.transcript == "hush now little one"
    assert loaded.consent_given is True
    # audio decodes back to roughly the same duration
    import io
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(store.load_audio_bytes("mom")), dtype="float32")
    assert sr == SR
    assert abs(len(data) - len(audio)) < SR * 0.1
