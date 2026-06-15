from peeky_reachy.detect.events import SoundEvent
from peeky_reachy.detect.smoothing import Hysteresis, TemporalSmoother


def test_temporal_smoother_ignores_single_spike():
    s = TemporalSmoother(window=5)
    for _ in range(4):
        s.update(SoundEvent.SILENCE, 0.9)
    event, _ = s.update(SoundEvent.BABY_CRY, 0.6)  # one-off spike
    assert event == SoundEvent.SILENCE


def test_temporal_smoother_follows_sustained_change():
    s = TemporalSmoother(window=5)
    last = None
    for _ in range(6):
        last, _ = s.update(SoundEvent.BABY_CRY, 0.8)
    assert last == SoundEvent.BABY_CRY


def test_hysteresis_latches():
    h = Hysteresis(enter=0.6, exit=0.36)
    assert h.update(0.5) is False     # below enter
    assert h.update(0.65) is True     # crosses enter
    assert h.update(0.4) is True      # between exit and enter -> stays on
    assert h.update(0.3) is False     # below exit -> off
