"""Merge the BEEHIVE Roboflow exports into one single-class bee detection dataset.

BEEHIVE (Mendeley Data 5yz78xxpmy) ships two YOLOv8-format exports, every image stretched
to 640 x 640::

    <raw>/frame_dataset/{train,valid,test}/{images,labels}    classes bee, blurred_bee
    <raw>/bottom_dataset/{train,valid,test}/{images,labels}   classes bee, occluded_bee

Stage 1 only needs "where is a bee", so both subsets are merged and every class is
mapped to ``0 = bee``. The original sub-class is kept per image in ``meta.csv``
(``n_clear`` and ``n_degraded``) so the effect of blur/occlusion can still be analysed.

Output::

    <out>/images/<split>/<subset>__<file>.jpg
    <out>/labels/<split>/<subset>__<file>.txt
    <out>/data.yaml, <out>/meta.csv, <out>/stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import yaml
from loguru import logger
from tqdm import tqdm

from varroa_vision.data.varroa_dataset import _place

SUBSETS = ("frame", "bottom")
SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}
BEE_CLASS = {0: "bee"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def remap_label(text: str) -> tuple[str, int, int]:
    """Map every class to 0. Returns (new_text, n_class0, n_other)."""
    out, clear, degraded = [], 0, 0
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"expected 'cls cx cy w h', got {line!r}")
        if parts[0] == "0":
            clear += 1
        else:
            degraded += 1
        out.append(" ".join(["0", *parts[1:]]))
    return "".join(l + "\n" for l in out), clear, degraded


def convert(raw_dir: Path, out_dir: Path, subsets: tuple[str, ...] = SUBSETS, link: bool = True) -> Path:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    rows, stats = [], {s: Counter() for s in SPLIT_DIRS}
    for subset in subsets:
        for split, src_split in SPLIT_DIRS.items():
            img_dir = raw_dir / f"{subset}_dataset" / src_split / "images"
            lbl_dir = raw_dir / f"{subset}_dataset" / src_split / "labels"
            if not img_dir.is_dir():
                raise FileNotFoundError(img_dir)
            (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            for img in tqdm(images, desc=f"{subset}/{split}", unit="img"):
                lbl = lbl_dir / f"{img.stem}.txt"
                text, clear, degraded = remap_label(lbl.read_text()) if lbl.exists() else ("", 0, 0)
                name = f"{subset}__{img.name}"
                _place(img, out_dir / "images" / split / name, link)
                (out_dir / "labels" / split / f"{subset}__{img.stem}.txt").write_text(text)
                st = stats[split]
                st["images"] += 1
                st["boxes"] += clear + degraded
                st[f"{subset}_images"] += 1
                rows.append({"image": name, "split": split, "subset": subset, "n_boxes": clear + degraded, "n_clear": clear, "n_degraded": degraded})

    with (out_dir / "meta.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "stats.json").write_text(json.dumps({s: dict(c) for s, c in stats.items()}, indent=2))
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": BEE_CLASS},
            sort_keys=False,
        )
    )
    for split, st in stats.items():
        logger.info("{:5s} {}", split, dict(st))
    return data_yaml


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/beehive"))
    parser.add_argument("--out", type=Path, default=Path("datasets/beehive_bees"))
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args(argv)
    logger.info("wrote {}", convert(args.raw, args.out, tuple(args.subsets), link=not args.copy))


if __name__ == "__main__":
    main()
