# varroa-vision

Fine-tuned object detection of *Varroa destructor* mites on honeybees, built towards an
Android app that lets a beekeeper film a comb and get a mite count per 100 bees.

## Problem

Varroa treatment decisions are made on a per-100-bee infestation rate, conventionally
measured by washing roughly 300 adult bees (alcohol wash or sugar roll) and counting the
dislodged mites. The wash is destructive and tedious, so most hobby beekeepers skip it.
This project replaces the wash with a camera: film the bees on a comb until about 300
distinct bees have been seen, detect the mites riding on them, and report the rate with
a confidence interval.

Two things make this harder than "run a detector on a photo":

1. Bees move. A video of a comb shows the same bee many times, so bees must be tracked
   across frames and counted once, and mite evidence must be pooled per bee.
2. A mite is 1-2 mm on a 12 mm bee. On a frame of a whole comb it is a few pixels wide,
   so mites are detected on per-bee crops, not on the full frame.

## Approach

Two-stage pipeline, each stage a small YOLO model that can run on a phone:

| Stage | Input | Output | Training data |
|---|---|---|---|
| 1. Bee detector + tracker | video frame of a comb | bee boxes with persistent track IDs | EV2, BEEHIVE (bee boxes on frames) |
| 2. Mite detector | crop of one bee | mite boxes on that bee | VarroaDataset (13,509 single-bee crops, 4,628 mite boxes) |

The app then aggregates: unique bees seen, bees with at least one confirmed mite,
mites per 100 bees, and a binomial confidence interval on that rate.

Stage 2 is the first priority because it is the hard, data-rich part and it can be
validated cleanly on held-out recording sessions. See `docs/` for dataset notes, related
work and the open sampling-methodology question.

## Status

- [x] Repo scaffold
- [x] VarroaDataset download and conversion to YOLO format (official and date-grouped splits)
- [x] EV2 (bee-level cross-domain test) and BEEHIVE (bee boxes) conversion
- [x] Stage 2 run 1: mAP50 0.80 on test, bee-level sensitivity 0.83 / specificity 0.97 at the rate-matched threshold; see `docs/results.md`
- [ ] Stage 2: stable recipe (no-mosaic runs), date-grouped split, EV2 pseudo-label domain
- [x] Stage 1 run 1 on BEEHIVE: mAP50 0.95 on test (in-hive cameras, not phone footage)
- [x] Tracking + per-bee aggregation + stopping rule (unit-tested with fake models; no real comb video yet)
- [ ] Export to TFLite and build the Android app
- [ ] Field calibration of camera counts against alcohol-wash counts

## Layout

```
src/varroa_vision/   library code (data download/conversion, training, evaluation, export)
configs/             training configs (one YAML per experiment)
scripts/             thin CLI entry points
docs/                datasets, related work, sampling methodology, roadmap
tests/               unit tests on synthetic data (no dataset download needed)
data/                raw downloads (gitignored)
datasets/            converted YOLO datasets (gitignored)
runs/                training outputs (gitignored)
```

## Quickstart

```
uv sync
uv run python -m varroa_vision.data.zenodo varroa_dataset      # 1.2 GB
uv run python -m varroa_vision.data.varroa_dataset             # -> datasets/varroa_mite
uv run python -m varroa_vision.train --config configs/mite_yolo11n.yaml --device 0
uv run python -m varroa_vision.evaluate --weights runs/mite/yolo11n_320_official/weights/best.pt --split val
uv run python -m varroa_vision.evaluate --weights ... --split test --conf <chosen on val>

# cross-domain check on EV2 crops (bee-level only, no mite boxes there)
uv run python -m varroa_vision.data.zenodo ev2                 # 1.1 GB
uv run python -m varroa_vision.data.ev2                        # -> datasets/ev2_bees
uv run python -m varroa_vision.evaluate --weights ... --data datasets/ev2_bees/data.yaml --split all --bee-only

# mobile export
uv sync --extra export-tflite
uv run python -m varroa_vision.export --weights ... --formats tflite onnx
```

Training on a machine without git: `scripts/deploy_lynx.sh user@host` pushes HEAD over ssh.

## License

AGPL-3.0. The detectors are built with Ultralytics YOLO, which is AGPL-3.0 licensed, and
weights fine-tuned from Ultralytics pretrained checkpoints inherit that license. The app
and all code here are open source, so this is compatible with the project's purpose.
Datasets keep their own licenses (CC BY 4.0 for VarroaDataset, EV2 and BEEHIVE); see
`docs/datasets.md` for citations.
