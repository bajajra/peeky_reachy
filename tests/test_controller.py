from peeky_reachy.detect.events import DetectionResult, SoundEvent
from peeky_reachy.soothe.controller import SootheController


def _cry(score=0.8):
    return DetectionResult(SoundEvent.BABY_CRY, score, is_voiced=True)


def _silence():
    return DetectionResult(SoundEvent.SILENCE, 0.9, is_voiced=False)


def test_requires_sustained_cry_before_acting():
    c = SootheController(cry_score_threshold=0.55, sustain_seconds=3.0, cooldown_seconds=30.0)
    assert c.observe(_cry(), now=0.0) is None       # just started
    assert c.observe(_cry(), now=2.9) is None       # not long enough
    decision = c.observe(_cry(), now=3.1)
    assert decision is not None
    assert decision.event == SoundEvent.BABY_CRY


def test_brief_cry_does_not_trigger():
    c = SootheController(0.55, 3.0, 30.0)
    assert c.observe(_cry(), now=0.0) is None
    assert c.observe(_silence(), now=0.5) is None
    # hangover passed -> cry timer resets; a later short cry shouldn't fire
    assert c.observe(_silence(), now=2.0) is None
    assert c.observe(_cry(), now=2.2) is None


def test_cooldown_blocks_repeated_soothing():
    c = SootheController(0.55, 1.0, 30.0)
    assert c.observe(_cry(), now=0.0) is None
    first = c.observe(_cry(), now=1.5)
    assert first is not None
    # within cooldown -> no new action even if cry persists
    assert c.observe(_cry(), now=5.0) is None
    assert c.observe(_cry(), now=20.0) is None
    # once cooldown elapses and the cry is still ongoing -> act again
    assert c.observe(_cry(), now=33.0) is not None


def test_below_threshold_score_is_ignored():
    c = SootheController(0.55, 1.0, 30.0)
    assert c.observe(_cry(score=0.4), now=0.0) is None
    assert c.observe(_cry(score=0.4), now=2.0) is None
