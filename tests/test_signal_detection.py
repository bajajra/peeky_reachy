import numpy as np

from peeky_reachy.detect.classifier import HeuristicClassifier
from peeky_reachy.detect.events import SoundEvent
from peeky_reachy.detect.preprocess import (NoiseFloor, WindowPreprocessor,
                                            dc_block, normalize_rms, rms, snr_db)
from peeky_reachy.detect.vad import EnergyVAD
from tests.conftest import SR, baby_cry, dog, silence, speech


def _window(sig, seconds=1.0):
    n = int(seconds * SR)
    return sig[:n] if len(sig) >= n else np.pad(sig, (0, n - len(sig)))


def test_energy_vad_distinguishes_loud_from_quiet():
    vad = EnergyVAD(SR, threshold=0.5)
    quiet = 0
    for f in np.array_split(silence(2.0), 20):
        active, _ = vad.is_active(f.astype(np.float32))
        quiet += active
    loud = 0
    for f in np.array_split(baby_cry(2.0), 20):
        active, _ = vad.is_active(f.astype(np.float32))
        loud += active
    assert quiet == 0
    assert loud >= 10


def test_heuristic_classifier_cry_and_silence():
    clf = HeuristicClassifier()
    ev, score = clf.classify(_window(baby_cry()), SR)
    assert ev == SoundEvent.BABY_CRY
    assert score >= 0.55
    ev, _ = clf.classify(_window(silence()), SR)
    assert ev == SoundEvent.SILENCE


def test_heuristic_classifier_speech_is_not_a_false_cry():
    clf = HeuristicClassifier()
    ev, _ = clf.classify(_window(speech()), SR)
    assert ev != SoundEvent.BABY_CRY


def test_heuristic_classifier_dog_is_not_a_cry():
    clf = HeuristicClassifier()
    ev, _ = clf.classify(_window(dog()), SR)
    assert ev != SoundEvent.BABY_CRY


def test_dc_block_removes_offset():
    x = np.ones(SR, dtype=np.float32) * 0.5 + 0.1 * np.sin(2 * np.pi * 500 * np.arange(SR) / SR)
    y = dc_block(x, SR)
    assert abs(float(np.mean(y))) < 0.01


def test_normalize_rms_targets_level_but_leaves_silence():
    loud = normalize_rms(baby_cry(1.0), target_rms=0.1)
    assert abs(rms(loud) - 0.1) < 0.03
    quiet = silence(1.0)
    assert np.allclose(normalize_rms(quiet), quiet)


def test_snr_and_noise_floor():
    nf = NoiseFloor()
    nf.calibrate(list(np.array_split(silence(2.0), 20)))
    prep = WindowPreprocessor(SR)
    _, snr, _ = prep.prepare(_window(baby_cry()), nf.noise_rms)
    assert snr > 10
    assert snr_db(0.1, 0.001) > snr_db(0.01, 0.001)
