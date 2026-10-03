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

Comparison of all runs so far (`python -m varroa_vision.compare`). `conf` is the
rate-matched threshold chosen on val; test and EV2 metrics are at that threshold. The
`warmfix_freeze` row was taken mid-run (epoch 10 of 60) and is refreshed below.

| run | best epoch | val mAP50 | test mAP50 | conf | test sens | test spec | test rate pred / true | EV2 sens | EV2 spec |
|---|---|---|---|---|---|---|---|---|---|
| run 1, mosaic (CPU) | 12 | 0.819 | 0.801 | 0.20 | 0.876 | 0.946 | 40.4 / 31.3 | 0.384 | 0.993 |
| run 4, low lr, no mosaic | 24 | 0.753 | 0.708 | 0.30 | 0.725 | 0.991 | 25.5 / 31.3 | 0.500 | 0.999 |
| no mosaic (GPU) | 1 | 0.703 | 0.640 | 0.20 | 0.734 | 0.957 | 30.7 / 31.3 | 0.308 | 0.999 |
| mosaic (GPU, patience 25) | 26 | 0.840 | 0.806 | 0.25 | 0.817 | 0.975 | 28.6 / 31.3 | 0.220 | 0.999 |
| warmfix, no mosaic | 43 | 0.745 | 0.679 | 0.30 | 0.785 | 0.985 | 26.7 / 31.3 | 0.411 | 0.998 |
| warmfix + frozen backbone (partial) | 7 | 0.822 | 0.758 | 0.25 | 0.831 | 0.962 | 31.7 / 31.3 | 0.409 | 0.999 |

Two readings. In-domain, the mosaic recipe wins (test mAP50 0.81, bee-level
sensitivity 0.82 at specificity 0.98). Out-of-domain, every run is poor and the best
in-domain run is the worst on EV2 (sensitivity 0.22): the more the model fits the tunnel
camera, the less it transfers. Specificity stays at 0.999 everywhere, so the EV2
detections are trustworthy as pseudo-labels. Run 9 (`combined`) trains the mosaic
recipe on VarroaDataset plus the pseudo-labelled EV2 crops, with EV2's held-out videos
as the out-of-domain test.

### GPU sweeps 2 and 3, and the final comparison

Sweep 2 kept mosaic and added augmentation (scale 0.5, mixup 0.1, 80 epochs) at 320 px,
at 416 px, and with yolo11s. Sweep 3 trained the plain mosaic recipe on VarroaDataset
plus the pseudo-labelled EV2 train videos (`combined`), and on the date-grouped split.

Final comparison (`runs/compare/final.md`). `conf` is chosen on val to match the true
mites-per-100-bees rate; test columns are at that threshold. `EV2 held-out` is bee-level
sensitivity on the 12 EV2 videos that no model saw (898 crops, 217 with a visible mite);
specificity there is 1.000 for every run.

| run | best epoch | val mAP50 | test mAP50 | conf | test sens | test spec | test rate pred / true | EV2 held-out sens |
|---|---|---|---|---|---|---|---|---|
| mosaic, CPU (run 1) | 12 | 0.819 | 0.801 | 0.25 | 0.830 | 0.968 | 32.1 / 31.3 | 0.005 |
| mosaic, GPU, patience 25 | 26 | 0.840 | 0.806 | 0.25 | 0.817 | 0.975 | 28.6 / 31.3 | 0.000 |
| no mosaic | 1 | 0.703 | 0.640 | 0.20 | 0.734 | 0.957 | 30.7 / 31.3 | 0.046 |
| low lr, no mosaic | 24 | 0.753 | 0.708 | 0.30 | 0.725 | 0.991 | 25.5 / 31.3 | 0.000 |
| warmfix, no mosaic | 43 | 0.745 | 0.679 | 0.30 | 0.785 | 0.985 | 26.7 / 31.3 | 0.065 |
| warmfix + frozen backbone | 7 | 0.822 | 0.758 | 0.25 | 0.831 | 0.962 | 31.7 / 31.3 | 0.014 |
| strong aug, 320 | 43 | 0.804 | 0.761 | 0.30 | 0.837 | 0.972 | 29.1 / 31.3 | 0.000 |
| strong aug, 416 | 62 | 0.757 | 0.692 | 0.25 | 0.893 | 0.955 | 32.9 / 31.3 | 0.000 |
| strong aug, yolo11s | 54 | 0.827 | 0.817 | 0.30 | 0.827 | 0.972 | 29.6 / 31.3 | 0.000 |
| **combined (mosaic + EV2 pseudo)** | 26 | 0.844 | 0.801 | 0.30 | 0.842 | 0.964 | 29.9 / 31.3 | **0.143** |

Date-grouped split (no recording day shared between train, val and test), same recipe:

| run | best epoch | val mAP50 | test mAP50 | conf | test sens | test spec | test rate pred / true | EV2 held-out sens |
|---|---|---|---|---|---|---|---|---|
| mosaic, date split | 19 | 0.817 | 0.701 | 0.35 | 0.874 | 0.966 | 33.7 / 35.3 | 0.014 |

What this says:

1. In-domain, the recipe is settled and the variants are within noise of each other:
   test mAP50 0.80 to 0.82, bee-level sensitivity 0.82 to 0.84 at specificity about
   0.97, predicted rate within 10% of the true rate. Extra augmentation, 416 px and
   yolo11s buy nothing. The date split puts the honest in-domain box mAP at 0.70, about
   0.1 below the official split, as predicted from the session overlap; bee-level
   sensitivity holds up (0.87) because missed boxes are mostly second mites on bees that
   are already flagged.
2. Out-of-domain is the real result. On EV2 videos no model has seen, every
   VarroaDataset-only model finds essentially nothing (0 to 6% of visible mites) while
   never producing a false positive. The earlier 22 to 50% figures on "all of EV2" came
   from a handful of videos that resemble the tunnel footage; those videos are now in
   the pseudo-label train split. Training on VarroaDataset plus 700 pseudo-labelled EV2
   crops lifts held-out sensitivity to 14% at unchanged in-domain accuracy, so the
   mechanism works, but the domain gap between two cameras is far larger than any
   recipe change within one dataset. A phone app needs mite boxes from its own camera
   and viewpoint; nothing public provides them.
3. Threshold choice: 0.25 to 0.35 depending on the run, always chosen on val by rate
   matching. The best-F1 threshold over-counts by 30 to 40% everywhere.

Inference size matters out of domain: EV2 crops are bee-filling, so after letterboxing
to 320 px the mite is smaller than in VarroaDataset crops. On the held-out videos the
`combined` model's sensitivity at threshold 0.25 is 0.143 at 320 px, 0.161 at 480 px
and 0.074 at 640 px (at threshold 0.05: 0.235 / 0.267 / 0.226); the VarroaDataset-only
model stays at zero at every size. The app should run stage 2 on crops scaled so that
the bee is about the size it has in VarroaDataset (bee about 200 px long at 320 px
input), not on the raw crop.

Current model: `combined` (yolo11n, 320 px, mosaic, VarroaDataset + EV2 pseudo),
threshold 0.30. It is the only run with any out-of-domain signal and it loses nothing
in-domain. Exports: `best.onnx`, `best.tflite` (float32), `best_int8.tflite` (2.9 MB).

### Export and parity

`saved_model` (onnx2tf) and LiteRT (litert-torch, separate env) both export. The
exported graphs take batch 1, so evaluate them with `--batch 1`; ultralytics' box-level
`val` returns zeros for them (a backend issue; predictions are fine), so use bee level.
On val, `mosaic_gpu` weights at threshold 0.25:

| model | sens | spec | rate pred / true | size |
|---|---|---|---|---|
| PyTorch .pt | 0.882 | 0.971 | 36.6 / 34.9 (run 1 numbers) | 5.4 MB |
| TFLite float32 (onnx2tf) | 0.894 | 0.982 | 33.8 / 34.9 | 10.1 MB |
| TFLite INT8 (LiteRT, calibrated on train) | 0.882 | 0.985 | 34.1 / 34.9 | 2.9 MB |

INT8 loses nothing measurable and is the Android candidate.

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
