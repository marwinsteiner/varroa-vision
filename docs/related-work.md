# Related work

What others have done, what they got, and what we take from it.

## Detection of mites on bees

| Work | Data | Model | Result | Takeaway |
|---|---|---|---|---|
| Schurischuster and Kampel 2020, VarroaDataset (TU Wien) | 13.5k tunnel crops, 4.6k mite boxes | dataset release | n/a | The cleanest public mite-box data. Our stage 2 training set. |
| Bilik et al. 2021, Sensors 21(8):2764 (arXiv 2103.03133) | 600 custom images, various scenes | YOLOv5, SSD, Deep SVDD | F1 0.874 for infected bee, 0.727 for the mite itself | Detecting the mite directly on a full scene is much harder than flagging the bee. Detect on per-bee crops. |
| Korean YOLOv8 two-stage framework 2024 (JKSCI) | 5k honeycomb images for bees, 10k bee crops for infestation | YOLOv8n detector + YOLOv8-cls | bee mAP50 0.70, infested/uninfested 91% (99% / 87%) | Two-stage works on comb images; the weak spot is false positives on healthy bees. We use a detector, not a classifier, in stage 2 so each positive is localised and countable. |
| tim-field/bee-mite-detector (GitHub) | 15k+ images, mostly Roboflow | YOLOv8n at 640 | mAP50 0.898 on test; 26-31 FPS on Pi 5 + Hailo-8L | Nano models are enough for edge deployment. |
| Enhancing bee mite detection with YOLO, 2025 (ResearchGate 392390813) | augmented, stratified sampling | YOLO11n, 2.6M params | P 0.925, R 0.921, mAP50 0.956 on val | Stratified sampling and augmentation matter more than model size. |
| Bjerge et al. 2019, Computers and Electronics in Agriculture | tunnel at the hive entrance, multispectral | classical CV + CNN | video-rate monitoring | Fixed-camera tunnel designs trade installation effort for image consistency. Our phone-video setting has neither. |

## Mite fall on sticky boards

| Work | Data | Model | Result |
|---|---|---|---|
| Divason et al. 2024, open software for mite fall analysis (PMC11207890) | 64 phone photos of sticky boards, 8064 x 6048 | YOLO-family detector on tiles | field-usable counts from phone photos |
| Hybrid ML vs DL comparison 2025 (PMC12390549) | 12 hyperspectral images | PCA + kNN + SVM vs Faster R-CNN | ML F1 0.99 vs DL 0.89 on 38 mites; tiny-data, hyperspectral, not transferable to phones |

## Commercial and existing apps

BeeScanning (Sweden), Apisfero, Bee Varroa Scanner, beemapping. All photo-based; none
publish their models or their calibration against alcohol-wash counts.

## Sampling bees for an infestation estimate

- Lee, Moon, Burkness, Hutchison and Spivak 2010, J. Econ. Entomol. 103(4):1039-1050.
  Practical sampling plans for Varroa destructor. 31 commercial apiaries. A 300-bee
  sample from one brood-nest frame gives a usable colony estimate; three 300-bee samples
  for research precision; eight colonies per apiary regardless of apiary size. Mites on
  adults times two approximates adults plus brood.
- Scientific Beekeeping, Re-evaluating Varroa monitoring part 3: frame-to-frame counts
  from one colony ranged 8 to 28 (mean 19). Brood-frame samples run higher and vary more
  than samples from drawn or honey comb next to the brood nest, which is the recommended
  sampling location.
- Honey Bee Health Coalition, Tools for Varroa Management. Action thresholds of 2-3 mites
  per 100 bees depending on season.

See `docs/sampling-methodology.md` for how these translate into a scanning protocol and
the open questions.

## Caveat on visual counting

An alcohol wash counts every phoretic mite on the sampled bees. A camera only counts the
mites it can see, and phoretic mites prefer the underside of the abdomen, between the
sternites. Visible-mite rates will therefore underestimate wash rates, and the ratio is
not known a priori for a phone-video setting. Field calibration (camera count vs wash on
the same bees) is a required step before the app can give treatment advice, and is on
the roadmap.
