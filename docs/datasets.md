# Datasets

All datasets below are CC BY 4.0 unless stated. Raw downloads live in `data/raw/<key>/`,
converted YOLO datasets in `datasets/<name>/`. Neither directory is committed.

## VarroaDataset (stage 2: mite detector)

Schurischuster, S. and Kampel, M. (2020). VarroaDataset. Zenodo.
https://doi.org/10.5281/zenodo.4085043 (record 4085044). GitHub: schurist/VarroaDataset.

- 13,509 PNG crops of single bees, 160 x 280 px (w x h), extracted from videos recorded
  with a hive-entrance tunnel device under constant lighting (Schurischuster et al. 2016).
  13 recording sessions between 2017-08-28 and 2017-10-17.
- Files: `train.zip` 703 MB, `test.zip` 292 MB, `val.zip` 163 MB, `gt.csv` 1.2 MB. The
  zips unpack to `<split>/videos/<session>/<file>.png`, matching the paths in `gt.csv`.
- `gt.csv` rows are space separated: `path label x1 y1 x2 y2 [x1 y1 x2 y2 ...]`, with
  pixel coordinates, top-left then bottom-right. Labels: 0 = healthy (9,562 rows),
  1 = infected (3,083), 3 = infected (864). Label 3 is not documented in the README, but
  every label-3 row carries at least one mite box and the README's positive counts
  (train 2,554 / test 942 / val 451) equal label-1 plus label-3. We treat {1, 3} as
  infected and keep the raw label in `meta.csv`.
- 4,628 mite boxes in total, mean size 33 x 32 px. Infected bees carry 1 box (3,332),
  2 boxes (549) or 3 boxes (66). One box has zero width and is dropped.
- Two train images (`bee_id_10000` and `bee_id_10001` of session 2017-08-30_15-42-59)
  are listed twice, once with slightly different boxes. The converter keeps the first
  row, so the converted dataset has 13,507 images.
- Official splits: train 8,225 / test 3,408 / val 1,876, assigned per recording session.

Caveat on the official splits: sessions recorded on the same day appear in different
splits (2017-09-01 in test and val, 2017-09-25 in train and val, 2017-10-17 in train and
test). Same-day sessions share colony, mite population and camera setup, so the official
val/test scores are an optimistic estimate of generalisation. The converter therefore
supports two split modes:

| mode | train | val | test | use |
|---|---|---|---|---|
| `official` | 8,225 | 1,876 | 3,408 | comparability with published numbers |
| `date` | 6,956 (08-28, 08-30, 09-20) | 2,001 (09-25, 09-29) | 4,552 (09-01, 10-17) | honest held-out-day estimate |

Domain note: these are top-down crops of bees walking through a tunnel, not bees on a
comb. The mite detector trained here must be re-validated on crops produced by the
stage 1 bee detector on comb footage (EV2 provides exactly that kind of frame).

## EV2 (cross-domain bee-level test set for stage 2)

EV2 Dataset (2024). Zenodo record 13771384, https://doi.org/10.5281/zenodo.13771384.
Project "EdgeVision against Varroa", Italian MUR PRIN2022. One file, `dataset.zip`,
1.08 GB, MD5 c626a1f198cf7d0f41eae9c2660b0985.

Despite the Zenodo description, EV2 does not contain full frames. Every PNG is a crop of
one bee, 180 to 740 px wide (typically about 470 x 390), cut from a 1920 x 1080 camcorder
frame; consecutive frames of the same bee are consecutive crops.

- 5,170 crops from 32 videos (9 to 874 crops per video), in `dataset_free/` (1,987) and
  `dataset_infested/` (3,183). Files are named `<video>_frame<n>.png`.
- `labels.txt` has one JSON object per line (with blank separator lines): `video`
  (`varroa_infested/<id>.MTS` or `varroa_free/<id>.MTS`), `id` (`frame_<n>`),
  `varroa_visible` (`yes`/`no`), `coord_1` and `coord_2` (the crop's top-left and
  bottom-right corner in the source frame). Exactly one object per crop.
- The folder encodes visibility, not health: `dataset_free` holds 1,288 crops from
  mite-free videos plus 699 crops from infested videos in which the mite is hidden.
- No mite boxes, so EV2 cannot train the detector. It is the right test of domain shift
  at bee level (different camera, lighting, viewpoint and bee size from VarroaDataset):
  does the stage 2 model fire on crops where a mite is visible and stay quiet otherwise.
  `python -m varroa_vision.data.ev2` builds `datasets/ev2_bees` for
  `python -m varroa_vision.evaluate --data datasets/ev2_bees/data.yaml --split all --bee-only`.
- Group any split by `video`; frames within a video are near-duplicates.

## BEEHIVE (stage 1: bee detector inside the hive)

BEEHIVE: a public dataset of Apis mellifera images to empower honeybee monitoring
research. Data in Brief, 2024. Mendeley Data https://data.mendeley.com/datasets/5yz78xxpmy.

- "Frame" subset: camera inside a hive frame, close-range bees, classes `bee` and
  `blurred_bee`; 1,440 train / 411 val / 206 test.
- "Bottom" subset: camera at the hive bottom behind a metal grid, classes `bee` and
  `occluded_bee`; 1,044 train / 303 val / 147 test.
- Annotated in Roboflow for object detection. Must be downloaded manually from Mendeley
  ("Download All"); no stable direct URL.

## Varroa mite fall on sticky boards (later: bottom-board mode)

Divason et al. (2024). Dataset for varroa mite detection on sticky boards. Zenodo
record 10231845. 64 images of 8064 x 6048 px with mite annotations, plus trained models
and code (github.com/jodivaso/varroa_detector). Different modality (natural mite fall
over 24-72 h), not needed for the on-bee counter, but a cheap second mode for the app.

## Roboflow Universe

Over 100 community datasets match "varroa". Quality and licensing vary and downloads
need an API key. Candidates to inspect once the first model exists:
abc-seffg/varroa-mite-detec (972 images), dip-project-bvluv/varroa-detection-demo (984),
beesap2025panthers/varroa-mite-detector, honeybee/honeybee_varroamite,
beproj/varroa-mites-detection--train-set.

## BeeDataset (TensorFlow Datasets `bee_dataset`)

About 7,500 entrance-camera images with multi-label tags (varroa, pollen, wasp, cooling).
Classification only, no boxes. Possible extra positives/negatives for a bee-level
infested/healthy classifier; not used for detection.
