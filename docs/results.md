# Results log

Every run on lynx (20-core CPU while its GPU driver is broken; YOLO11n). Box metrics
from ultralytics `val` at conf 0.001; bee-level metrics from `varroa_vision.evaluate`
(a bee is predicted infested if any mite detection reaches the threshold).

## Stage 2: mite detector on VarroaDataset, official splits

### Run 1: `configs/mite_yolo11n.yaml` (320 px, mosaic on), 2026-10-03

Stopped by early stopping at epoch 27 (patience 15, best epoch 12), 1.6 h on CPU.
The validation curve was erratic: mAP50 0.76 at epoch 7, 0.46 at epoch 22, 0.80 at
epoch 24. Mosaic is the prime suspect: it stitches four crops and rescales, so a 32 px
mite trains at about 16 px while validation sees it at 37 px, and `close_mosaic` would
only have switched it off at epoch 50. Runs 2 and 3 drop mosaic (320 and 416 px).

Box level (best.pt from epoch 12):

| split | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|
| val | 0.819 | 0.283 | 0.842 | 0.743 |
| test | 0.801 | 0.281 | 0.824 | 0.742 |

Bee level across thresholds. `rate` is mites per 100 bees, the number the app reports.
True rate: val 34.9, test 31.3.

| conf | val sens | val spec | val F1 | val rate | test sens | test spec | test F1 | test rate |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.953 | 0.928 | 0.874 | 65.7 | 0.921 | 0.918 | 0.863 | 53.9 |
| 0.20 | 0.929 | 0.954 | 0.896 | 48.5 | 0.876 | 0.946 | 0.868 | 40.4 |
| 0.25 | 0.882 | 0.971 | 0.894 | 36.6 | 0.830 | 0.968 | 0.868 | 32.1 |
| 0.30 | 0.845 | 0.983 | 0.890 | 30.7 | 0.773 | 0.979 | 0.845 | 26.7 |
| 0.35 | 0.783 | 0.991 | 0.864 | 26.5 | 0.707 | 0.987 | 0.812 | 23.2 |
| 0.40 | 0.670 | 0.997 | 0.798 | 20.3 | 0.574 | 0.993 | 0.721 | 18.3 |

Reading: the F1-optimal threshold (0.20) over-counts mites by about 30% because
infested bees collect extra boxes. The rate-matched threshold is 0.25 (val +5%, test
+2.5%), with sensitivity 0.83 and specificity 0.97 on test. Pick thresholds on the rate,
not on F1. Per-bee count error at 0.25 to 0.30 is about 0.1 mites per bee.

The low mAP50-95 (0.28) says boxes are found but loosely placed; for counting that is
irrelevant, for the app's overlay it only affects where the marker is drawn.

### Runs 2 and 3: no mosaic, 320 px and 416 px, 2026-10-03

Mosaic was not the cause. Both runs swing just as hard between neighbouring epochs
(320 px: mAP50 0.79 at epoch 29, 0.50 at 33, 0.78 at 37; 416 px: 0.57 at epoch 5, 0.24
at 13) and only settle once the cosine schedule has decayed the learning rate. The
416 px run was stopped at epoch 30 (worse and 2.5x the cost). Run 2 reached mAP50 0.76
at epoch 52 and is reported below once finished.

The common factor is the optimiser step: `optimizer: auto` selects AdamW at lr 0.002 for
this run length. Run 4 uses AdamW at 0.0005 with everything else as in run 2.

### Run 4: no mosaic, 320 px, AdamW lr 0.0005 (GPU, after the lynx reboot)

Not the learning rate either, at least not lr0. The curve is the clearest yet: mAP50
0.73 after epoch 1, 0.40 by epoch 5, 0.42 at epoch 9, then a slow climb back to 0.73
by epoch 49. best.pt: val mAP50 0.753, test 0.708; bee level on test at the
rate-matched threshold (0.25): sensitivity 0.75, specificity 0.99, rate 29.1 vs 31.3.
Worse than run 1.

Reading: the COCO-pretrained features already localise mites after one epoch, and the
early epochs damage them. The one schedule element that lr0 does not control is
`warmup_bias_lr`, which ultralytics defaults to 0.1 and ramps down to lr0 over the
warmup; with AdamW on a tiny single-class head that is a large step. Runs 5 and 6 set
it to lr0 (`warmfix`), run 6 also freezes the backbone. The GPU makes each of these a
five-minute experiment; `scripts/experiments/stage2_gpu_sweep1.sh` runs the set.

### GPU sweep 1 (runs 2, 1, 5, 6 re-run or run fresh on the RTX 4060, ~5 min each)

Validation mAP50 by epoch:

| run | 1 | 6 | 11 | 16 | 21 | 26 | 31 | 41 | 51 | note |
|---|---|---|---|---|---|---|---|---|---|---|
| no mosaic (run 2 config) | 0.70 | 0.38 | 0.28 | 0.28 | 0.68 | 0.42 | | | | early stop, best epoch 1 |
| mosaic (run 1 config, patience 25) | 0.35 | 0.50 | 0.40 | 0.75 | 0.80 | 0.84 | 0.85 | 0.80 | 0.80 | best run so far |
| warmfix (no mosaic, lr 0.001, bias warmup fixed) | 0.45 | 0.32 | 0.42 | 0.52 | 0.52 | 0.53 | | | | flat and low |
| warmfix + frozen backbone | | | | | | | | | | see comparison table |

This settles it, and reverses the earlier reading: mosaic is not the problem, its
absence is. Without it the 8k crops are memorised within a few epochs and validation
decays from the epoch-1 level that the pretrained features already give; with it the
model keeps improving to 0.85. The warmup and learning-rate hypotheses are dropped.
Sweep 2 keeps mosaic and adds regularisation (scale 0.5, mixup 0.1, 80 epochs) at 320
and 416 px, plus yolo11s.

A note on resuming: ultralytics cannot resume a CPU checkpoint on the GPU (optimiser
state stays on the CPU: "params, grads, exp_avgs ... must have same device"). The
interrupted CPU runs were restarted from scratch on the GPU instead.

### EV2 cross-domain check (different camera, bee crops 300-700 px)

Run 1 weights on all 5,170 EV2 crops, bee level only (EV2 has no mite boxes). 3,183
crops have a visible mite.

| conf | sensitivity | specificity | precision | F1 | predicted infested |
|---|---|---|---|---|---|
| 0.10 | 0.489 | 0.981 | 0.976 | 0.651 | 1,593 |
| 0.20 | 0.384 | 0.993 | 0.989 | 0.553 | 1,236 |
| 0.25 | 0.324 | 0.998 | 0.996 | 0.489 | 1,035 |
| 0.30 | 0.225 | 0.998 | 0.996 | 0.367 | 719 |

Reading: a clear domain gap. The detector almost never fires on a mite-free crop, but
it finds only a third of the visible mites under EV2's camera at the threshold that was
calibrated on the tunnel data. The high precision is the lever: EV2 crops that the
detector does fire on are reliable pseudo-labels, and `data/ev2_pseudo.py` turns them
plus all mite-free EV2 crops into a second training domain with a video-grouped split.
A detector that works on a beekeeper's phone must survive this kind of shift, so the
EV2 held-out videos become a standing test alongside VarroaDataset test.

## Stage 1: bee detector on BEEHIVE (frame + bottom merged)

### Run 1: `configs/bee_yolo11n.yaml` (640 px), 2026-10-03

50 epochs, 3.9 h on CPU.

| split | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|
| val | 0.961 | 0.654 | 0.925 | 0.898 |
| test | 0.950 | 0.646 | 0.907 | 0.902 |

Good on BEEHIVE's own cameras. Says nothing yet about phone footage of a comb, which is
the domain that matters; that needs own recordings (roadmap item 2).
