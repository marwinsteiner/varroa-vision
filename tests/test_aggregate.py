import pytest

from varroa_vision.pipeline.aggregate import (
    AggregationRule,
    FrameObs,
    ScanEstimate,
    aggregate,
    decide,
    group_by_track,
    judge_track,
    wilson_interval,
)

RULE = AggregationRule(conf=0.35, conf_high=0.8, min_hits=2, min_frames=3, min_bee_px=60)


def obs(n, px=100, confs=()):
    return [FrameObs(i, px, tuple(confs[i]) if i < len(confs) else ()) for i in range(n)]


def test_short_or_small_tracks_are_not_counted():
    assert not judge_track(1, obs(2), RULE).counted
    assert not judge_track(2, obs(5, px=40), RULE).counted
    assert judge_track(3, obs(3), RULE).counted


def test_single_weak_hit_is_not_infested():
    v = judge_track(1, obs(5, confs=[(0.5,)]), RULE)
    assert v.counted and not v.infested and v.mites == 0 and v.hits == 1


def test_two_hits_or_one_strong_hit_is_infested():
    assert judge_track(1, obs(5, confs=[(0.4,), (), (0.5,)]), RULE).infested
    assert judge_track(2, obs(5, confs=[(), (0.9,)]), RULE).infested
    assert not judge_track(3, obs(5, confs=[(0.3,), (0.3,), (0.3,)]), RULE).infested


def test_mite_count_is_median_over_hit_frames():
    v = judge_track(1, obs(6, confs=[(0.5, 0.6), (0.5,), (0.7, 0.9), (0.4, 0.5)]), RULE)
    assert v.infested and v.mites == 2  # hits per frame 2,1,2,2 -> median 2
    v1 = judge_track(2, obs(4, confs=[(0.5,), (0.1, 0.6)]), RULE)
    assert v1.mites == 1


def test_wilson_interval():
    lo, hi = wilson_interval(9, 300)
    assert lo == pytest.approx(0.0159, abs=5e-4)
    assert hi == pytest.approx(0.0560, abs=5e-4)
    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert wilson_interval(0, 10)[0] == 0.0


def test_aggregate_and_decide():
    tracks = {
        1: obs(5),
        2: obs(5, confs=[(0.5,), (0.6,)]),
        3: obs(2),  # too short, ignored
        4: obs(5, confs=[(0.9, 0.9), (0.9, 0.9)]),  # two mites
        5: obs(5),
    }
    est = aggregate(tracks, RULE)
    assert (est.bees, est.infested_bees, est.mites) == (4, 2, 3)
    assert est.rate_per100 == pytest.approx(75.0)
    assert est.ci95_per100[0] < 75.0 < est.ci95_per100[1]
    assert decide(est, 3.0) == "above"
    assert decide(ScanEstimate(0, 0, 0, 0.0, (0.0, 0.0)), 3.0) == "undecided"
    clean = aggregate({i: obs(4) for i in range(400)}, RULE)
    assert decide(clean, 3.0) == "below"  # 0/400 -> upper bound ~0.95%
    few = aggregate({i: obs(4) for i in range(20)}, RULE)
    assert decide(few, 3.0) == "undecided"  # 0/20 -> upper bound ~16%


def test_group_by_track():
    recs = [(1, FrameObs(0, 90)), (2, FrameObs(0, 90)), (1, FrameObs(1, 95))]
    g = group_by_track(recs)
    assert sorted(g) == [1, 2] and len(g[1]) == 2
