# Sampling methodology

How a beekeeper should scan a hive so that the mite count means something. This is the
main open question of the project; what follows is the current proposal and the
experiments that would settle it.

## What the alcohol wash establishes

- Standard practice (Lee et al. 2010; Honey Bee Health Coalition): shake about 300 adult
  bees (half a cup) from one brood-nest frame into alcohol, count the mites, report mites
  per 100 bees. Action thresholds are 2 to 3 per 100 depending on season.
- One 300-bee sample is enough for a management decision; three samples for research
  precision; eight colonies per apiary for an apiary-level estimate.
- Frame choice matters. Counts from different frames of the same colony ranged 8 to 28
  around a mean of 19 (Scientific Beekeeping). Brood frames run higher and vary more;
  the recommended source is drawn or honey comb adjacent to the brood nest.

## What 300 bees actually buys

A 300-bee sample is a coarse instrument. If the true rate is 3 per 100, the 95% Wilson
interval on 9 mites in 300 bees is roughly 1.6 to 5.6 per 100. The wash cannot tell
"below threshold" from "well above" at the threshold itself; it works because most
colonies are far from it.

The camera changes the economics: bees are free to count, so the sample size does not
have to be fixed at 300. The app should keep scanning until the interval on the rate
lies on one side of the action threshold, or until a cap is hit (say 1,000 bees), and
report "undecided, treat as borderline" otherwise. This is a sequential test, and it is
the main methodological advantage of counting with a phone instead of a cup.

## Why bees on one frame are not 300 independent samples

Mites cluster. Bees on one comb share age cohort and task, and infestation differs
between combs. Counting 300 bees from one side of one frame therefore has a smaller
effective sample size than 300. The standard correction is a design effect
DEFF = 1 + (m - 1) * ICC, where m is bees per frame and ICC the intra-class correlation
of infestation between frames. ICC for phone-scanned frames is unknown; it is the single
number that decides how many frames a beekeeper must scan, and it has to be measured.

## Proposed scanning protocol, version 0 (to be validated)

1. Open the brood box. Pick three frames: the two frames adjacent to the brood nest on
   either side and one brood-nest frame. Do not shake bees off.
2. Hold the frame vertically in good daylight with the sun behind the phone, no flash.
   Film each side slowly in landscape for 10 to 15 seconds at 15 to 25 cm, so that a
   bee spans at least 80 px. Keep the whole comb face in frame; the app tracks bees
   and counts each once per clip.
3. Repeat for the other side and the other frames. The app accumulates unique bees and
   visible mites per clip and per frame.
4. Stop when the sequential rule fires or all six sides are done. Report visible mites
   per 100 bees with its interval, per frame and pooled, and the seasonal threshold.

Bees that walk between frames while the beekeeper works are counted twice across clips.
That inflates n by a small amount and does not bias the rate; the within-clip tracker
prevents the large double counts.

## Visible mites are not washed mites

The wash dislodges every phoretic mite. A phone sees only mites on the dorsal side or
visible between tergites; phoretic mites prefer the underside of the abdomen. The
visible fraction in phone video is unknown and probably depends on lighting, distance
and bee density. Until it is measured, the app must report visible mites per 100 bees
and not claim equivalence with a wash result.

## Experiments that would settle the open questions

| Question | Experiment | Output |
|---|---|---|
| Visible fraction | On 20 colonies: film 300 bees from one frame, then wash the same bees | ratio camera / wash with its spread; whether it depends on rate |
| Frames needed | On 10 colonies: film every frame of the brood box, both sides | per-frame rates, ICC, DEFF, frames needed for a given interval width |
| Minimum image quality | Downscale and blur EV2 crops; measure stage 2 recall vs bee size in px | minimum bee size and motion-blur budget for the app to enforce |
| Stopping rule | Simulate the sequential test on the pooled per-frame data | expected bees scanned and error rates at the threshold |
| Disturbance | Compare counts in the first and last 5 s of each clip | whether bees react to handling in a way that changes visibility |

The first two experiments need a beekeeper, a camera and an afternoon in an apiary; the
other three can be done from the data already collected.
