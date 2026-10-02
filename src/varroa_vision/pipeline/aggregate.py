"""Pool per-frame detections into per-bee verdicts and a colony-level estimate.

Nothing here touches a model; the inputs are plain records so the rules can be unit
tested and ported to the app unchanged.

Terminology: a *track* is one bee followed across frames by the tracker. Each frame in
which the bee was seen yields a ``FrameObs``: the bee box size (a proxy for crop
quality) and the confidences of any mite detections on that crop.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FrameObs:
    frame: int
    bee_px: float  # shorter side of the bee box in pixels
    mite_confs: tuple[float, ...] = ()


@dataclass(frozen=True)
class AggregationRule:
    """Decide whether a track is an infested bee and how many mites it carries.

    A track counts as a bee if it was seen in at least ``min_frames`` frames and its
    median box is at least ``min_bee_px`` wide; shorter or smaller tracks are ignored
    (false bee detections, bees at the frame edge, bees too far away).

    A bee counts as infested if a mite was detected at or above ``conf`` in at least
    ``min_hits`` frames, or once at or above ``conf_high``. The mite count for an
    infested bee is the median number of detections at or above ``conf`` over the frames
    where at least one was found, so a single noisy frame does not add mites.
    """

    conf: float = 0.35
    conf_high: float = 0.80
    min_hits: int = 2
    min_frames: int = 3
    min_bee_px: float = 60.0


@dataclass
class TrackVerdict:
    track_id: int
    frames: int
    median_bee_px: float
    counted: bool
    infested: bool
    mites: int
    hits: int
    max_conf: float


@dataclass
class ScanEstimate:
    bees: int
    infested_bees: int
    mites: int
    rate_per100: float
    ci95_per100: tuple[float, float]
    verdicts: list[TrackVerdict] = field(default_factory=list)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def judge_track(track_id: int, obs: list[FrameObs], rule: AggregationRule) -> TrackVerdict:
    frames = len(obs)
    med_px = _median([o.bee_px for o in obs])
    counted = frames >= rule.min_frames and med_px >= rule.min_bee_px
    hits_per_frame = [sum(c >= rule.conf for c in o.mite_confs) for o in obs]
    hits = sum(h > 0 for h in hits_per_frame)
    max_conf = max((c for o in obs for c in o.mite_confs), default=0.0)
    infested = counted and (hits >= rule.min_hits or max_conf >= rule.conf_high)
    mites = int(round(_median([h for h in hits_per_frame if h > 0]))) if infested else 0
    return TrackVerdict(track_id, frames, med_px, counted, infested, max(mites, int(infested)), hits, max_conf)


def wilson_interval(successes: int, trials: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; (0, 0) when there are no trials."""
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def aggregate(tracks: dict[int, list[FrameObs]], rule: AggregationRule = AggregationRule()) -> ScanEstimate:
    """Turn per-track observations into a colony estimate with a Wilson interval on the
    infested-bee proportion (mites per 100 bees uses the mite count; its interval is
    approximated by scaling the infested-bee interval by mites per infested bee)."""
    verdicts = [judge_track(tid, obs, rule) for tid, obs in sorted(tracks.items())]
    counted = [v for v in verdicts if v.counted]
    bees = len(counted)
    infested = sum(v.infested for v in counted)
    mites = sum(v.mites for v in counted)
    lo, hi = wilson_interval(infested, bees)
    per_infested = mites / infested if infested else 1.0
    rate = 100 * mites / bees if bees else 0.0
    return ScanEstimate(bees, infested, mites, rate, (100 * lo * per_infested, 100 * hi * per_infested), verdicts)


def decide(estimate: ScanEstimate, threshold_per100: float) -> str:
    """Sequential stopping rule: 'below', 'above' or 'undecided' versus an action threshold."""
    lo, hi = estimate.ci95_per100
    if estimate.bees == 0:
        return "undecided"
    if hi < threshold_per100:
        return "below"
    if lo > threshold_per100:
        return "above"
    return "undecided"


def group_by_track(records) -> dict[int, list[FrameObs]]:
    """Helper for the video driver: records are (track_id, FrameObs) pairs."""
    out: dict[int, list[FrameObs]] = defaultdict(list)
    for tid, obs in records:
        out[tid].append(obs)
    return dict(out)
