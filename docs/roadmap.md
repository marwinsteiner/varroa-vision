# Roadmap

Ordered by what unblocks the most.

## 1. Stage 2 mite detector (in progress)

- Fine-tune YOLO11n at 320 px on VarroaDataset, official splits, 60 epochs.
- Report box mAP and bee-level sensitivity/specificity on val and test, choose the
  operating threshold on val.
- Repeat on the date-grouped split to get an honest held-out-day number.
- Ablations worth one run each: imgsz 416, YOLO11s, no vertical flip, no mosaic.
- Teacher test: train yolo11m or yolo11x on the same domains and compare held-out EV2
  and internet-photo sensitivity with the nano. Only if the larger model is clearly
  better out of domain does it earn a role as pseudo-labeller for unlabelled footage.
- Export TFLite float16 and check that the exported model reproduces the PyTorch
  bee-level numbers on the test split.

## 2. Stage 1 bee detector on comb footage

- Inspect EV2 and BEEHIVE formats, convert to YOLO with a single `bee` class
  (BEEHIVE's blurred/occluded variants merged, kept in meta for analysis).
- Fine-tune YOLO11n at 640 px. Metric: bee mAP50 and, more importantly, crop quality
  for stage 2: run the stage 2 model on EV2 crops and compare the mite-visible flag
  with the detections.
- Phone footage of real combs is not in any public dataset. Record some, annotate a few
  hundred frames, and fine-tune again; this is the domain the app lives in.

## 3. Video pipeline

- ByteTrack (built into ultralytics) over stage 1 detections for persistent bee IDs.
- Per track: run stage 2 on the best few crops (largest, sharpest), pool mite evidence
  with a rule such as "mite seen in at least 2 frames or once above a high threshold".
- Output per clip: unique bees, infested bees, mites, with a binomial interval.
- Offline evaluation on EV2 sequences if they are ordered frames; otherwise on our own
  recordings.

## 4. Android app

- Kotlin app with CameraX; stage 1 and stage 2 as TFLite models on the GPU delegate.
- Live overlay of tracked bees and running count; the sequential stopping rule from
  `docs/sampling-methodology.md`; a session summary per hive.
- Offline only. No account, no upload, except an opt-in to contribute annotated clips.
- Runtime constants (input size, thresholds, aggregation rule) come from the
  `model_card.json` written at export, and the app checks a versioned manifest so a
  retrained model can be delivered without a store release.
- Photo mode for stills: stage 1 on overlapping tiles (`pipeline/tiled.py`) so a
  whole-comb photo keeps bees at training scale.
- Later, with consent: low-confidence frames uploaded for review and retraining, the
  improvement loop the field data needs.

## 5. Field calibration

- Camera count against alcohol wash on the same bees across 20 colonies.
- Frame-to-frame variance study to fix the number of frames the protocol requires.
- Only after this can the app show a treatment recommendation instead of a count.
