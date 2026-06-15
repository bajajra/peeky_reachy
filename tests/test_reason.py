from peeky_reachy.detect.events import CryReason
from peeky_reachy.detect.reason import EpisodeReasonAggregator


def test_aggregator_abstains_without_enough_votes():
    agg = EpisodeReasonAggregator(min_votes=3)
    agg.add(CryReason.HUNGRY, 0.3)
    reason, conf = agg.result()
    assert reason == CryReason.UNKNOWN
    assert conf == 0.0


def test_aggregator_picks_majority_reason():
    agg = EpisodeReasonAggregator(min_votes=3)
    for _ in range(4):
        agg.add(CryReason.TIRED, 0.3)
    agg.add(CryReason.HUNGRY, 0.2)
    reason, conf = agg.result()
    assert reason == CryReason.TIRED
    assert 0 < conf <= 0.4  # confidence stays capped (advisory only)


def test_aggregator_abstains_on_split():
    agg = EpisodeReasonAggregator(min_votes=3, min_agreement=0.6)
    agg.add(CryReason.TIRED, 0.3)
    agg.add(CryReason.HUNGRY, 0.3)
    agg.add(CryReason.PAIN, 0.3)
    reason, _ = agg.result()
    assert reason == CryReason.UNKNOWN


def test_night_prior_nudges_tired():
    agg = EpisodeReasonAggregator(min_votes=2, min_agreement=0.4)
    agg.add(CryReason.TIRED, 0.25)
    agg.add(CryReason.DISCOMFORT, 0.25)
    assert agg.result(hour_of_day=2)[0] == CryReason.TIRED
