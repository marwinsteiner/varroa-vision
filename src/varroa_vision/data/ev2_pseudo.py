"""Pseudo-label EV2 crops with a trained mite detector to add a second camera domain.

The first detector, trained on VarroaDataset's tunnel crops, transfers to EV2 with
precision above 0.99 but sensitivity around 0.3: when it fires it is right, it just
misses most mites under the new camera. High precision is what makes pseudo-labelling
safe. This module:

1. runs the detector on every EV2 crop;
2. keeps crops flagged ``varroa_visible == yes`` that received at least one detection,
   with those boxes as labels (crops with a visible mite but no detection are dropped,
   not written as background, because that would teach the wrong thing);
3. keeps every ``varroa_visible == no`` crop as a background image;
4. splits by video with a deterministic hash, so no video leaks between train, val
   and test, and writes ``data.yaml`` and ``meta.csv``.

With ``--combine-with datasets/varroa_mite`` it also writes a dataset whose train list
is VarroaDataset train plus EV2 pseudo train, with VarroaDataset val/test unchanged, so
the official numbers stay comparable and EV2 test is a held-out camera domain.

    python -m varroa_vision.data.ev2_pseudo --weights runs/mite/<run>/weights/best.pt --conf 0.25
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from varroa_vision.data.varroa_dataset import CLASS_NAMES, _place

SPLIT_FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}


def video_split(video: str, seed: int = 0) -> str:
    """Deterministic train/val/test assignment per video from a hash of its name."""
    h = hashlib.sha1(f"{seed}:{video}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    acc = 0.0
    for split, frac in SPLIT_FRACTIONS.items():
        acc += frac
        if u < acc:
            return split
    return "test"


def predict_boxes(model, img_dir: Path, conf: float, imgsz: int, device, batch: int) -> dict[str, list[tuple[float, float, float, float, float]]]:
    """Map image name -> [(cx, cy, w, h, conf), ...] normalised, for detections >= conf."""
    out: dict[str, list] = {}
    for r in model.predict(source=str(img_dir), imgsz=imgsz, conf=conf, device=device, batch=batch, stream=True, verbose=False):
        name = Path(r.path).name
        boxes = []
        if r.boxes is not None and len(r.boxes.conf):
            for (cx, cy, w, h), c in zip(r.boxes.xywhn.tolist(), r.boxes.conf.tolist()):
                boxes.append((cx, cy, w, h, c))
        out[name] = boxes
    return out


def build(ev2_dir: Path, detections: dict[str, list], out_dir: Path, link: bool = True, seed: int = 0) -> Path:
    ev2_dir, out_dir = Path(ev2_dir), Path(out_dir)
    meta = pd.read_csv(ev2_dir / "meta.csv")
    img_src = ev2_dir / "images" / "all"
    for split in SPLIT_FRACTIONS:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    stats = {s: Counter() for s in SPLIT_FRACTIONS}
    rows = []
    for rec in meta.itertuples(index=False):
        boxes = detections.get(rec.image, [])
        split = video_split(rec.video, seed)
        st = stats[split]
        if rec.infected and not boxes:
            st["dropped_positive"] += 1
            continue
        lines = "".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h, _ in boxes) if rec.infected else ""
        _place(img_src / rec.image, out_dir / "images" / split / rec.image, link)
        (out_dir / "labels" / split / f"{Path(rec.image).stem}.txt").write_text(lines)
        st["images"] += 1
        st["positives"] += int(rec.infected)
        st["boxes"] += len(boxes) if rec.infected else 0
        st["unlabelled_detections_on_negatives"] += len(boxes) if not rec.infected else 0
        rows.append({"image": rec.image, "split": split, "video": rec.video, "video_class": rec.video_class, "infected": int(rec.infected), "n_boxes": len(boxes) if rec.infected else 0, "pseudo": 1})
    with (out_dir / "meta.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "stats.json").write_text(json.dumps({s: dict(c) for s, c in stats.items()}, indent=2))
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": CLASS_NAMES}, sort_keys=False))
    for split, st in stats.items():
        logger.info("{:5s} {}", split, dict(st))
    return data_yaml


def write_combined(base_dir: Path, pseudo_dir: Path, out_dir: Path) -> Path:
    """data.yaml whose train set is base train + pseudo train; val/test stay the base's."""
    base_dir, pseudo_dir, out_dir = Path(base_dir).resolve(), Path(pseudo_dir).resolve(), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": [str(base_dir / "images" / "train"), str(pseudo_dir / "images" / "train")],
                "val": str(base_dir / "images" / "val"),
                "test": str(base_dir / "images" / "test"),
                "names": CLASS_NAMES,
            },
            sort_keys=False,
        )
    )
    # evaluate.py reads meta.csv next to data.yaml for bee-level metrics on val/test
    (out_dir / "meta.csv").write_bytes((base_dir / "meta.csv").read_bytes())
    return data_yaml


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--ev2", type=Path, default=Path("datasets/ev2_bees"))
    parser.add_argument("--out", type=Path, default=Path("datasets/ev2_pseudo"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--combine-with", type=Path, default=None, help="base YOLO dataset dir, e.g. datasets/varroa_mite")
    parser.add_argument("--combined-out", type=Path, default=Path("datasets/mite_combined"))
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    dets = predict_boxes(YOLO(str(args.weights)), args.ev2 / "images" / "all", args.conf, args.imgsz, args.device, args.batch)
    data_yaml = build(args.ev2, dets, args.out, link=not args.copy, seed=args.seed)
    logger.info("wrote {}", data_yaml)
    if args.combine_with:
        logger.info("wrote {}", write_combined(args.combine_with, args.out, args.combined_out))


if __name__ == "__main__":
    main()
