"""Single-class bee dataset from a Roboflow export, for stage 1 (phone photos of combs).

Keeps one copy per source image, keeps only images with at least one bee or queen box,
maps those to ``0 = bee`` and drops every other class. Split by source image hash.

    python -m varroa_vision.data.roboflow_bees --project hofer_bees
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import yaml
from loguru import logger

from varroa_vision.data.beehive import BEE_CLASS
from varroa_vision.data.internet_mites import (
    BEE_CLASSES,
    read_labels,
    source_stem,
    split_for,
    unique_images,
)
from varroa_vision.data.varroa_dataset import _place


def build(project_dir: Path, out_dir: Path, test_fraction: float = 0.2, val_fraction: float = 0.1, seed: int = 0, link: bool = True) -> Path:
    out_dir = Path(out_dir)
    for s in ("train", "val", "test"):
        (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)
    rows, stats = [], {s: Counter() for s in ("train", "val", "test")}
    for img_path, lbl_path, names in unique_images(Path(project_dir)):
        labels = read_labels(lbl_path, names)
        bees = [b for b in labels if b[0].lower() in BEE_CLASSES]
        if not bees:
            continue
        stem = source_stem(img_path.name)
        split = split_for(stem, test_fraction, seed)
        if split == "train" and split_for("val:" + stem, val_fraction / (1 - test_fraction), seed) == "test":
            split = "val"
        name = f"{stem}{img_path.suffix.lower()}"
        _place(img_path, out_dir / "images" / split / name, link)
        (out_dir / "labels" / split / f"{stem}.txt").write_text("".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for _, cx, cy, w, h in bees))
        stats[split]["images"] += 1
        stats[split]["boxes"] += len(bees)
        rows.append({"image": name, "split": split, "stem": stem, "n_boxes": len(bees)})
    with (out_dir / "meta.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "stats.json").write_text(json.dumps({s: dict(c) for s, c in stats.items()}, indent=2))
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": BEE_CLASS}, sort_keys=False))
    for s, c in stats.items():
        logger.info("{:5s} {}", s, dict(c))
    return data_yaml


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="hofer_bees")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/roboflow"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args(argv)
    out = args.out or Path("datasets") / f"{args.project}_bees"
    logger.info("wrote {}", build(args.raw / args.project, out, link=not args.copy))


if __name__ == "__main__":
    main()
