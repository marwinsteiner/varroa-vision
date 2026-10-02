"""Convert VarroaDataset (Zenodo 4085044) to Ultralytics YOLO detection format.

Input layout, as produced by ``python -m varroa_vision.data.zenodo varroa_dataset`` once the
three zips are extracted next to ``gt.csv`` (this module extracts them if needed)::

    <raw>/gt.csv
    <raw>/<split>/videos/<session>/<file>.png

Output layout::

    <out>/images/<split>/<file>.png     hard-linked (or copied) from the raw dir
    <out>/labels/<split>/<file>.txt     YOLO lines "0 cx cy w h"; empty for healthy bees
    <out>/data.yaml                     Ultralytics dataset description
    <out>/meta.csv                      one row per image with split, session, raw label, counts
    <out>/stats.json                    per-split counts

Healthy bees are kept as background images (empty label files). They are the majority of
what the app will see, and a detector trained without them learns a high false-positive
rate. See docs/datasets.md for the label semantics and the split caveat.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger
from PIL import Image
from tqdm import tqdm

CLASS_NAMES = {0: "varroa"}
INFECTED_LABELS = frozenset({1, 3})
SPLITS = ("train", "val", "test")

# Date-grouped split: no recording day is shared between splits. Chosen so that the
# largest session (2017-09-20, 3,839 crops) trains and both test days contain a large
# session recorded weeks apart.
DATE_SPLIT = {
    "2017-08-28": "train",
    "2017-08-30": "train",
    "2017-09-20": "train",
    "2017-09-25": "val",
    "2017-09-29": "val",
    "2017-09-01": "test",
    "2017-10-17": "test",
}

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Sample:
    rel_path: str
    official_split: str
    session: str
    label_raw: int
    boxes: tuple[Box, ...]

    @property
    def date(self) -> str:
        return self.session[:10]

    @property
    def infected(self) -> bool:
        return self.label_raw in INFECTED_LABELS

    @property
    def file_name(self) -> str:
        return self.rel_path.rsplit("/", 1)[-1]


def parse_gt(gt_csv: Path) -> list[Sample]:
    """Parse ``gt.csv``: ``path label [x1 y1 x2 y2]*`` with whitespace separators."""
    samples = []
    with gt_csv.open() as fh:
        for lineno, line in enumerate(fh, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 2 or (len(fields) - 2) % 4:
                raise ValueError(f"{gt_csv}:{lineno}: expected 'path label [x1 y1 x2 y2]*', got {line!r}")
            rel_path, label = fields[0], int(fields[1])
            parts = rel_path.split("/")
            if len(parts) != 4 or parts[0] not in SPLITS or parts[1] != "videos":
                raise ValueError(f"{gt_csv}:{lineno}: unexpected path layout {rel_path!r}")
            coords = list(map(int, fields[2:]))
            boxes = tuple(tuple(coords[i : i + 4]) for i in range(0, len(coords), 4))
            samples.append(Sample(rel_path, parts[0], parts[2], label, boxes))  # type: ignore[arg-type]
    return samples


def dedupe(samples: list[Sample]) -> list[Sample]:
    """Keep the first row per image path. gt.csv lists two train images twice (once with
    slightly different boxes), so the dataset has 13,507 unique images, not 13,509."""
    seen: dict[str, Sample] = {}
    for s in samples:
        if s.rel_path in seen:
            logger.warning("duplicate row for {} (keeping first)", s.rel_path)
            continue
        seen[s.rel_path] = s
    return list(seen.values())


def clean_boxes(boxes: tuple[Box, ...], width: int, height: int) -> tuple[list[Box], int]:
    """Clip boxes to the image and drop degenerate ones. Returns (kept, n_dropped)."""
    kept: list[Box] = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        if x2 > x1 and y2 > y1:
            kept.append((x1, y1, x2, y2))
    return kept, len(boxes) - len(kept)


def to_yolo_line(box: Box, width: int, height: int, cls: int = 0) -> str:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    return f"{cls} {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def ensure_extracted(raw_dir: Path) -> None:
    """Unzip ``<split>.zip`` into ``raw_dir`` for every split directory that is missing."""
    for split in SPLITS:
        if (raw_dir / split / "videos").is_dir():
            continue
        archive = raw_dir / f"{split}.zip"
        if not archive.exists():
            raise FileNotFoundError(f"neither {raw_dir / split} nor {archive} exists")
        logger.info("extracting {}", archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)


def _place(src: Path, dst: Path, link: bool) -> None:
    if dst.exists():
        return
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def assign_split(sample: Sample, mode: str) -> str:
    if mode == "official":
        return sample.official_split
    if mode == "date":
        try:
            return DATE_SPLIT[sample.date]
        except KeyError as exc:
            raise KeyError(f"no date split for session {sample.session}") from exc
    raise ValueError(f"unknown split mode {mode!r}")


def convert(raw_dir: Path, out_dir: Path, split_mode: str = "official", link: bool = True) -> Path:
    """Build the YOLO dataset and return the path of its ``data.yaml``."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    ensure_extracted(raw_dir)
    samples = dedupe(parse_gt(raw_dir / "gt.csv"))
    names = Counter(s.file_name for s in samples)
    dupes = [n for n, c in names.items() if c > 1]
    if dupes:
        raise ValueError(f"{len(dupes)} duplicate file names, cannot flatten: {dupes[:3]}")

    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {s: Counter() for s in SPLITS}
    meta_rows = []
    for sample in tqdm(samples, desc=f"convert ({split_mode})", unit="img"):
        split = assign_split(sample, split_mode)
        src = raw_dir / sample.rel_path
        with Image.open(src) as im:
            width, height = im.size
        boxes, dropped = clean_boxes(sample.boxes, width, height)
        if sample.infected and not boxes:
            logger.warning("{}: infected but no valid box", sample.rel_path)
        stem = Path(sample.file_name).stem
        _place(src, out_dir / "images" / split / sample.file_name, link)
        (out_dir / "labels" / split / f"{stem}.txt").write_text(
            "".join(to_yolo_line(b, width, height) + "\n" for b in boxes)
        )
        st = stats[split]
        st["images"] += 1
        st["infected"] += sample.infected
        st["boxes"] += len(boxes)
        st["boxes_dropped"] += dropped
        meta_rows.append(
            {
                "image": sample.file_name,
                "split": split,
                "official_split": sample.official_split,
                "session": sample.session,
                "date": sample.date,
                "label_raw": sample.label_raw,
                "infected": int(sample.infected),
                "n_boxes": len(boxes),
                "width": width,
                "height": height,
            }
        )

    with (out_dir / "meta.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(meta_rows[0]))
        writer.writeheader()
        writer.writerows(meta_rows)
    (out_dir / "stats.json").write_text(
        json.dumps({"split_mode": split_mode, **{s: dict(c) for s, c in stats.items()}}, indent=2)
    )
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": CLASS_NAMES,
            },
            sort_keys=False,
        )
    )
    for split in SPLITS:
        logger.info("{:5s} {}", split, dict(stats[split]))
    return data_yaml


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/varroa_dataset"))
    parser.add_argument("--out", type=Path, default=None, help="default datasets/varroa_mite[_date]")
    parser.add_argument("--split", choices=("official", "date"), default="official")
    parser.add_argument("--copy", action="store_true", help="copy images instead of hard-linking")
    args = parser.parse_args(argv)
    out = args.out or Path("datasets") / ("varroa_mite" if args.split == "official" else "varroa_mite_date")
    data_yaml = convert(args.raw, out, args.split, link=not args.copy)
    logger.info("wrote {}", data_yaml)


if __name__ == "__main__":
    main()
